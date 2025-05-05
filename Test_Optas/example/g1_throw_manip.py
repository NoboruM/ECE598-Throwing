# Python standard lib
import os
import sys
import pathlib

# OpTaS
import optas
from optas.templates import Manager
import pybullet as p

# PyBullet
import pybullet_api
import casadi as cs
from casadi import SX, transpose, trace
g1_base_position = [0.0, 0.0, 0.5]

x_targ = 0.50
y_targ = 0.0
z_targ = 0.50
g = 9.81

class G1JntIdx:
    waist_y = 0
    waist_r = 1
    waist_p = 2
    l_shoulder_p = 3
    l_shoulder_r = 4
    l_shoulder_y = 5
    l_elbow = 6
    l_wrist_r = 7
    l_wrist_p = 8
    l_wrist_y = 9
    r_shoulder_p = 10
    r_shoulder_r = 11
    r_shoulder_y = 12
    r_elbow = 13
    r_wrist_r = 14
    r_wrist_p = 15
    r_wrist_y = 16

class DualG1Planner(Manager):
    def setup_solver(self):
        # Parameters
        self.T = 50
        Tmax = 0.5
        link_ee_r = "right_wrist_yaw_link"
        link_ee_l = "left_wrist_yaw_link"
        link_head = "head_link"
        t = optas.linspace(0, Tmax, self.T)
        dt = float((t[1] - t[0]).toarray()[0, 0])
        # setup throwing parameters
        x_targ = 1.0
        y_targ = 0.0
        z_targ = 1.0

        theta = 0.0
        speed_release = 0.0
        
        # Setup robot models
        g1 = self._setup_g1_model("g1_dual_arm")
        # Get robot names
        self.g1_name = g1.get_name()

        # Setup optimization builder
        builder = optas.OptimizationBuilder(T=self.T, robots=[g1])

        # Setup parameters
        qc = builder.add_parameter("qc", g1.ndof)

        # Constraint: initial configuration
        builder.fix_configuration(self.g1_name, qc)

        # Constraint: dynamics
        builder.integrate_model_states(
            self.g1_name,
            time_deriv=1,  # i.e. integrate velocities to positions
            dt=dt)
        builder.enforce_model_limits(self.g1_name, time_deriv=0)
        builder.enforce_model_limits(self.g1_name, time_deriv=1)
        q0 = builder.get_model_state(self.g1_name, t=0, time_deriv=0)
        ##########################################################################


        ######################## Setup Obstacle Avoidance ########################
        # place obstacle at the torso
        torso_position = g1.get_global_link_position('torso_link', optas.np.deg2rad([0.0]*17))
        obs = optas.DM.zeros(3, 3)
        obs[:,0] = torso_position
        obs[:,1] = torso_position
        obs[:,2] = torso_position
        obs[2,0] += 0.20
        obs[2,1] += 0.05
        obs[2,2] += 0.125
        linkrad = 0.04
        obsrad = [0.09]*obs.shape[1]
        link_names = [
            'right_shoulder_yaw_link',
            'right_elbow_link',
            'right_wrist_roll_link',
            'right_wrist_pitch_link',
            'right_wrist_yaw_link']
        obstacle_names = []
        for i in range(len(obsrad)):
            obstacle_names.append("torso{}".format(i))
        # builder.sphere_collision_avoidance_constraints(self.g1_name, obstacle_names, link_names=link_names)
        # params = {}
        # for link_name in link_names:
        #     params[link_name + "_radii"] = linkrad

        # for i, obstacle_name in enumerate(obstacle_names):
        #     params[obstacle_name + "_position"] = obs[:, i]
        #     params[obstacle_name + "_radii"] = obsrad[i]
        ##############################################################


        ######################## Robot States ########################
        # Get end effector position FK function
        posr_ee = g1.get_global_link_position_function(link_ee_r, n=self.T)
        posl_ee = g1.get_global_link_position_function(link_ee_l, n=self.T)
        
        # Get Torso position FK function
        pos_head = g1.get_link_position_function(link_head, "pelvis", n=self.T)
        rot_head = g1.get_link_rotation_function(link_head, "pelvis", n=self.T)

        # Get joint trajectory: ndof-by-T sym array for robot trajectory
        Q = builder.get_model_states(self.g1_name) 
        ddQ = builder.get_model_states(self.g1_name, time_deriv=2)
        # Get end-effector position trajectories
        ee_pos_path_r = posr_ee(Q)
        ee_pos_path_l = posl_ee(Q)

        # Get joint velocity trajectory
        dQr = builder.get_model_states(self.g1_name, time_deriv=1)
        dQl = builder.get_model_states(self.g1_name, time_deriv=1)

        ee_jacobian = g1.get_global_link_geometric_jacobian_function(link_ee_r, n=self.T)
        # Compute Cartesian velocity: J(q) * dq
        cart_vel_r = optas.SX.zeros(6, self.T-1)  # 6D twist (linear + angular)
        J = ee_jacobian(Q)
        ee_vel = optas.SX.zeros(3, self.T)
        for i in range(self.T-1):
            cart_vel_r[:, i] = J[i] @ dQr[:, i]  # J * dq
            ee_vel[:, i] = optas.norm_2(cart_vel_r[:3, i])
        #############################################################



        ######################## Constraints ########################
        # add equality constraint to keep head upright
        pg = builder.add_parameter("position_goal", 3)
        pF = g1.get_global_link_position(link_head, Q)
        builder.add_equality_constraint("head_upright", pF, pg)

        # goal_eff_vel = optas.DM.zeros(3, T-1) # [vx, vy, vz]
        # goal_eff_vel[0, -1] = 2.0
        goal_eff_vel = optas.DM([1.5, 0.0, 1.5]) # [vx, vy, vz]
        goal_eff_ang = optas.DM([0.0, 0.0, 0.0])
        # builder.add_equality_constraint("eff_vel", cart_vel_r[:3,-1], goal_eff_vel, reduce_constraint=True)
        # builder.add_equality_constraint("eff_ang_vel", cart_vel_r[3:, -1], goal_eff_vel, reduce_constraint=True)
        
        #######################################################


        ######################## Costs ########################

        ee_jacobian = g1.get_global_link_linear_jacobian(link_ee_r, Q)
        # Compute Cartesian velocity: J(q) * dq
        cart_vel_r = optas.SX.zeros(6, self.T-1)  # 6D twist (linear + angular)
        J = ee_jacobian

        r_targ = optas.SX([x_targ, y_targ, z_targ])
        r_ee = posr_ee(Q)
        mu = optas.SX.zeros(3, self.T)
        mu_des = optas.SX.zeros(1, self.T)
        mu_hat = optas.SX.zeros(3, self.T)
        mu_2 = optas.SX.zeros(3, self.T)
        for i in range(self.T):
            tmp = r_targ - r_ee[:3,i]
            tmp[2] = optas.norm_2(tmp[0:2])
            Z = tmp[2]
            tmp_hat = tmp/optas.norm_2(tmp)
            v_0 = optas.sqrt(g*optas.norm_2(tmp[:2])/(2*(tmp_hat[2]*optas.norm_2(tmp[:2]) - Z*optas.norm_2(tmp_hat[:2]))*optas.norm_2(tmp_hat[:2])))
            mu[:,i] = tmp_hat*v_0
            mu_des[i] = optas.norm_2(mu[:,i])
            mu_hat[:,i] = mu[:,i]/mu_des[i]
            mu_2[:,i] = mu_hat[:,i]/optas.sqrt(transpose(mu_hat[:,i])@optas.inv(J[i]@transpose(J[i]))@mu_hat[:,i])
        builder.add_cost_term("manipulability1", optas.sumsqr(mu - mu_2))
        # builder.add_cost_term("ee_vel", optas.sumsqr(mu - ee_vel))

        # Setup solver
        solver = optas.CasADiSolver(builder.build()).setup("ipopt")

        # solver.reset_parameters(params)
        # Save variables for later
        self.g1_ndof = g1.ndof
        self.Tmax = Tmax
        return solver

    def _setup_g1_model(self, name):
        cwd = pathlib.Path( __file__).parent.resolve()  # path to current working directory
        urdf_filename = os.path.join(cwd, "robots", "g1", "g1_dual_arm.urdf")
        print("urdf_filename: ", urdf_filename)
        model = optas.RobotModel(
            urdf_filename=urdf_filename,
            name=name,
            time_derivs=[0, 1, 2],  # i.e. joint position/velocity trajectory
        )
        return model

    def is_ready(self):
        return True

    def reset(self, qc, qcr):
        # Set parameters
        self.solver.reset_parameters({"qc": optas.DM(qc)}
        )
        Q0 = optas.diag(qc) @ optas.DM.ones(self.g1_ndof, self.T)
        self.solver.reset_initial_seed({f"{self.g1_name}/q/x": Q0})

    def get_target(self):
        return self.solution

    def plan(self):
        self.solve()
        solution = self.get_target()

        # Interpolate
        planl = self.solver.interpolate(solution[f"{self.g1_name}/q"], self.Tmax)

        return planl#, planr


def main(gui=True):
    dual_g1_planner = DualG1Planner()

    hz = 50
    dt = 1.0 / float(hz)
    pb = pybullet_api.PyBullet(dt, gui=gui)

    # box = pybullet_api.DynamicBox(
    #     base_position=[0.25, 0, 0.15], half_extents=[0.15, 0.15, 0.15]
    # )

    g1_dual_arm = pybullet_api.G1DualArm(base_position=g1_base_position)

    # in order of: 
    waist_y_jnt = -90.0
    waist_r_jnt = 0.0
    waist_p_jnt = 0.0
    l_shoulder_p_jnt = 0.0
    l_shoulder_r_jnt = 0.0
    l_shoulder_y_jnt = 0.0
    l_elbow_jnt = 0.0
    l_wrist_r_jnt = 0.0
    l_wrist_p_jnt = 0.0
    l_wrist_y_jnt = 0.0
    r_shoulder_p_jnt = 0#45
    r_shoulder_r_jnt = 0.0
    r_shoulder_y_jnt = 0.0
    r_elbow_jnt = 45#120
    r_wrist_r_jnt = 90
    r_wrist_p_jnt = 0.0
    r_wrist_y_jnt = 0.0#-90.0
    qc = optas.np.deg2rad([waist_y_jnt,
        waist_r_jnt,
        waist_p_jnt,
        l_shoulder_p_jnt,
        l_shoulder_r_jnt,
        l_shoulder_y_jnt,
        l_elbow_jnt,
        l_wrist_r_jnt,
        l_wrist_p_jnt,
        l_wrist_y_jnt,
        r_shoulder_p_jnt,
        r_shoulder_r_jnt,
        r_shoulder_y_jnt,
        r_elbow_jnt,
        r_wrist_r_jnt,
        r_wrist_p_jnt,
        r_wrist_y_jnt])
    g1_dual_arm.reset(qc)
    dual_g1_planner.reset(qc, qc)
    planl = dual_g1_planner.plan()

    pb.start()
    pybullet_api.time.sleep(2.0)
    start_time = pybullet_api.time.time()
    traj_done = False
    rtf = 1.0
    while True:
        t = pybullet_api.time.time() - start_time
        if not traj_done and t > dual_g1_planner.Tmax*rtf:
            print("done!")
            traj_done = True
        elif not traj_done:
            g1_dual_arm.cmd(planl(t/rtf))
            wrist_link_state = g1_dual_arm.GetLinkState(20, True, True) # get wrist yaw joint state

            linear_vel = wrist_link_state[6]  # Extract linear velocity vector
            link_pos = wrist_link_state[0]    # Extract link position
            
            print("Linear vel: ", linear_vel)
            
            # Calculate arrow endpoint using velocity vector
            arrow_end = [link_pos[i] + linear_vel[i] for i in range(3)]
            
            # Draw red arrow showing velocity direction/magnitude
            p.addUserDebugLine(
                link_pos,
                arrow_end,
                lineColorRGB=[1, 0, 0],  # Red
                lineWidth=2,
                lifeTime=100
            )
            print("Angular_val: ", wrist_link_state[7])
        pybullet_api.time.sleep(rtf*dt*float(gui))

    pybullet_api.time.sleep(10.0*float(gui))

    pb.stop()
    pb.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
