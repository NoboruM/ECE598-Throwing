# Python standard lib
import os
import sys
import pathlib

# OpTaS
import optas
from optas.templates import Manager

# PyBullet
import pybullet_api
import casadi as cs
from casadi import SX, transpose, trace
g1_base_position = [0.0, 0.0, 0.5]

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
        Tmax = 0.25
        link_ee_r = "right_wrist_yaw_link"
        link_ee_l = "left_wrist_yaw_link"
        link_head = "head_link"
        t = optas.linspace(0, Tmax, self.T)
        dt = float((t[1] - t[0]).toarray()[0, 0])

        # Setup robot models
        g1 = self._setup_g1_model("g1_dual_arm", g1_base_position)
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
        for i in range(self.T-1):
            cart_vel_r[:, i] = J[i] @ dQr[:, i]  # J * dq
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
        # Cost: minimize joint velocity
        w_dq = 0.01
        w_q = 10
        w = 1000.0
        # builder.add_cost_term("g1_min_join_vel_r", w_dq * optas.sumsqr(dQr))
        builder.add_cost_term("joint_accel_cost", w * optas.sumsqr(ddQ)) # minimize joint acceleration
        # builder.add_cost_term("joint_pos_cost", w_q * optas.sumsqr(Q)) # Avoid negative joints
        # builder.add_cost_term("head_pos_cost", w_q * optas.sumsqr(rot_head(Q) - optas.SX.zeros(g1.ndof, T))) # Avoid negative joints
        
        # Get start position for each arm
        pos0r = g1.get_global_link_position(link_ee_r, qc)
        qf = optas.SX.zeros(17, 1)
        ctrl_joints = [G1JntIdx.waist_y, G1JntIdx.r_elbow, G1JntIdx.r_shoulder_p]
        qf[G1JntIdx.waist_y] = 0.0
        qf[G1JntIdx.r_elbow] = optas.np.deg2rad(-10.0)
        qf[G1JntIdx.r_shoulder_p] = optas.np.deg2rad(-45.0)
        Q_lim = g1.get_limits(time_deriv=0)
        print("type: ", Q_lim)
        # Find approximate joint path
        path_rq = optas.SX.zeros(len(ctrl_joints), self.T)
        for i in range(self.T):
            alpha = float(i)/float(self.T - 1)
            for j, jnt in enumerate(ctrl_joints):
                path_rq[j, i] = (qf[jnt] - q0[jnt])*optas.sin(alpha*optas.pi/2.0 - optas.pi/2.0) + qf[jnt]
                # path_rq[j, i] = (1.0 - alpha)*q0[jnt] + alpha*qf[jnt]
                # path_rq[j, i] = q0[jnt]
                # print("i: ", i)
                # print("\tpath: ", path_rq[j,i])
        w_jnt = 0.001
        builder.add_cost_term("jnt_path_r", w_jnt*optas.sumsqr(Q[ctrl_joints, :] - path_rq))
        # builder.add_cost_term("jnt_val", 10*optas.sumsqr(Q_lim[0] - Q[:,-1]))

        # Setup solver
        solver = optas.CasADiSolver(builder.build()).setup("ipopt")

        # solver.reset_parameters(params)
        # Save variables for later
        self.g1_ndof = g1.ndof
        self.Tmax = Tmax
        return solver

    def _setup_g1_model(self, name, base_position):
        cwd = pathlib.Path( __file__).parent.resolve()  # path to current working directory
        urdf_filename = os.path.join(cwd, "robots", "g1", "g1_dual_arm.urdf")
        print("urdf_filename: ", urdf_filename)
        model = optas.RobotModel(
            urdf_filename=urdf_filename,
            name=name,
            time_derivs=[0, 1, 2],  # i.e. joint position/velocity trajectory
        )
        # model.add_base_frame("pelvis", xyz=base_position)
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
    r_shoulder_p_jnt = 45
    r_shoulder_r_jnt = 0.0
    r_shoulder_y_jnt = 0.0
    r_elbow_jnt = 120
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
    while True:
        t = pybullet_api.time.time() - start_time
        if not traj_done and t > dual_g1_planner.Tmax:
            print("done!")
            traj_done = True
        elif not traj_done:
            g1_dual_arm.cmd(planl(t))
            wrist_link_state = g1_dual_arm.GetLinkState(20, True, True) # get wrist yaw joint state
            print("Linear val: ", wrist_link_state[6])
            # print("Angular_val: ", wrist_link_state[7])
        pybullet_api.time.sleep(dt*float(gui))

    pybullet_api.time.sleep(10.0*float(gui))

    pb.stop()
    pb.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
