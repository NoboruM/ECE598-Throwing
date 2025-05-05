# Python standard lib
import os
import sys
import pathlib

# OpTaS
import optas
from optas.templates import Manager

# PyBullet
import pybullet_api
from g1_manip_search_v2 import G1ThrowSearch, CalcTrajParams
from optas.visualize import Visualizer
import numpy as np

g1_base_position = [0.0, 0.0, 0.5]

class DualG1Planner(Manager):
    def setup_solver(self):
        self.T_traj = 0.0
        # Parameters
        self.T = 100
        self.T_half = 50
        link_ee_r = "right_wrist_yaw_link"
        link_head = "head_link"


        # Setup robot models
        g1 = self._setup_g1_model("g1_dual_arm", g1_base_position)
        # Get robot names
        self.g1_name = g1.get_name()

        # Setup optimization builder
        builder = optas.OptimizationBuilder(T=self.T, robots=[g1])

        Tmax = builder.add_decision_variables("Tmax", 1, 1)  # [1x1]
        
        dt = Tmax/(self.T - 1)

        # Setup parameters
        q0 = builder.add_parameter("q0", g1.ndof)
        v_ee = builder.add_parameter("v_ee", 3)
        q_f = builder.add_parameter("q_f", g1.ndof)
        dq_release = builder.add_parameter("dq_release", g1.ndof)

        # Constraint: initial configuration
        builder.fix_configuration(self.g1_name, q0, t=0)

        # Constraint: dynamics
        builder.integrate_model_states(
            self.g1_name,
            time_deriv=1,  # i.e. integrate velocities to positions
            dt=dt)
        builder.enforce_model_limits(self.g1_name, time_deriv=0)
        builder.enforce_model_limits(self.g1_name, time_deriv=1)
        # q0 = builder.get_model_state(self.g1_name, t=0, time_deriv=0)
        ##########################################################################


        ######################## Setup Obstacle Avoidance ########################
        # place obstacle at the torso
        torso_position = g1.get_global_link_position('torso_link', optas.np.deg2rad([0.0]*g1.ndof))
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
        # posl_ee = g1.get_global_link_position_function(link_ee_l, n=self.T)
        
        # Get Torso position FK function
        pos_head = g1.get_link_position_function(link_head, "pelvis", n=self.T)
        rot_head = g1.get_link_rotation_function(link_head, "pelvis", n=self.T)

        # Get joint trajectory: ndof-by-T sym array for robot trajectory
        Q = builder.get_model_states(self.g1_name) 
        ddQ = builder.get_model_states(self.g1_name, time_deriv=2)
        # Get end-effector position trajectories
        ee_pos_path_r = posr_ee(Q)
        # ee_pos_path_l = posl_ee(Q)

        # Get joint velocity trajectory
        dQr = builder.get_model_states(self.g1_name, time_deriv=1)
        dQf = builder.get_model_state(self.g1_name, t=-1, time_deriv=1)
        # dQl = builder.get_model_states(self.g1_name, time_deriv=1)

        ee_jacobian = g1.get_global_link_geometric_jacobian_function(link_ee_r, n=self.T)
        # Compute Cartesian velocity: J(q) * dq
        cart_vel_r = optas.SX.zeros(6, self.T-1)  # 6D twist (linear + angular)
        J = ee_jacobian(Q)
        for i in range(self.T-1):
            cart_vel_r[:, i] = J[i] @ dQr[:, i]  # J * dq
        vel_ee = J[-1]@dQf
        #############################################################



        ######################## Constraints ########################
        # add equality constraint to keep head upright
        pg = builder.add_parameter("position_goal", 3)
        pF = g1.get_global_link_position(link_head, Q)
        Qf = builder.get_model_state(self.g1_name, t=-1, time_deriv=0)
        # builder.add_equality_constraint("head_upright", pF, pg)
        builder.add_equality_constraint("end_pose", Qf, q_f) 

        goal_eff_ang = optas.DM([0.0, 0.0, 0.0])
        # builder.add_equality_constraint("eff_vel_release", cart_vel_r[:3,-1], 2*v_ee)
        builder.add_equality_constraint("eff_vel_release", dQf, dq_release)
        # builder.add_equality_constraint("eff_ang_vel_release", cart_vel_r[3:, -1], goal_eff_ang)

        # builder.add_equality_constraint("dq_end", dQr[:,-1], optas.DM.zeros(17,1))
        # builder.add_equality_constraint("q_end", Q[:,-1], optas.DM.zeros(17,1))
        
        # builder.add_cost_term("velocity", -100*optas.sumsqr(cart_vel_r))
        # builder.add_leq_inequality_constraint("eff_vel_upper", cart_vel_r[:3,-1], v_ee)
        builder.add_leq_inequality_constraint("Tmax_lower", Tmax, 0.0)
        builder.add_leq_inequality_constraint("Tmax_upper", 1.0, Tmax)
        #######################################################


        ######################## Costs ########################
        # Cost: minimize joint velocity
        w_dq = 0.01
        w_q = 10
        w = 1.0
        # builder.add_cost_term("g1_min_join_vel_r", w_dq * optas.sumsqr(dQr))
        builder.add_cost_term("joint_accel_cost", w * optas.sumsqr(ddQ)) # minimize joint acceleration
        # builder.add_cost_term('traj_length', 100*optas.sumsqr(Tmax))
        # builder.add_cost_term("joint_pos_cost", w_q * optas.sumsqr(Q)) # Avoid negative joints
        # builder.add_cost_term("head_pos_cost", w_q * optas.sumsqr(rot_head(Q) - optas.SX.zeros(g1.ndof, T))) # Avoid negative joints
        # builder.add_cost_term("ee_velocity", 100*optas.sumsqr(cart_vel_r[:3,-1] - v_ee))

        # Setup solver
        solver = optas.CasADiSolver(builder.build()).setup("ipopt")

        # solver.reset_parameters(params)
        # Save variables for later
        self.g1_ndof = g1.ndof
        self.Tmax = Tmax
        return solver

    def _setup_g1_model(self, name, base_position):
        cwd = pathlib.Path( __file__).parent.resolve()  # path to current working directory
        cwd = os.path.split(cwd)[0]
        urdf_filename = os.path.join(cwd, "robot", "g1", "g1_dual_arm.urdf")
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

    def reset(self, q0, v_ee, q_f, dqf):
        # Set parameters
        print("TYPE: ", optas.DM(dqf))
        self.solver.reset_parameters({"q0": optas.DM(q0), "v_ee": optas.DM(v_ee), "q_f": q_f, 'dq_release': optas.DM(dqf)})
        # self.solver.reset_parameters({"q0": optas.DM(q0), "v_ee": optas.DM(v_ee), "q_f": q_f})
        Q0 = optas.diag(q0) @ optas.DM.ones(self.g1_ndof, self.T)
        self.solver.reset_initial_seed({f"{self.g1_name}/q/x": Q0})

    def get_target(self):
        return self.solution

    def plan(self):
        self.solve()
        solution = self.get_target()
        # Interpolate
        self.T_traj = float(solution[f"Tmax"][0]) 
        plan_q = self.solver.interpolate(solution[f"{self.g1_name}/q"], self.T_traj)
        plan_dq = self.solver.interpolate(solution[f"{self.g1_name}/dq"], self.T_traj)
        return plan_q, plan_dq
    
    def Compute1PolyTraj(self, t, q0, dq0, qf, dqf, t0, tf):
        A = np.array([
            [1, t0, t0**2, t0**3],
            [0, 1, 2*t0, 3*t0**2],
            [1, tf, tf**2, tf**3],
            [0, 1, 2*tf, 3*tf**2]
        ])
        B = np.array([q0, dq0, qf, dqf])
        a0, a1, a2, a3  = np.linalg.solve(A, B)
        return a0 + a1*t + a2*t**2 + a3*t**3

    def ComputePolyThrowTraj(self, q0, dq0, qf, dqf, t0, tf):
        t = np.arange(0, 2.0, 0.02)
        self.plan_q = np.zeros((15, len(t)))
        for i in range(15):
            self.plan_q[i,:] = self.Compute1PolyTraj(t, q0[i], dq0[i], qf[i], dqf[i], t0, tf)
        return self.plan_q
def main(gui=True):
    dual_g1_planner = DualG1Planner()
    hz = 50
    dt = 1.0 / float(hz)
    pb = pybullet_api.PyBullet(dt, gui=gui)
    g1_dual_arm = pybullet_api.G1DualArm(base_position=g1_base_position)
    cwd = pathlib.Path( __file__).parent.resolve()  # path to current working directory
    cwd = os.path.split(cwd)[0]
    urdf_filename = os.path.join(cwd, "robot", "g1", "g1_dual_arm.urdf")
    # Setup robot
    robot = optas.RobotModel(urdf_filename)
    link_ee = "right_wrist_yaw_link"
    ik_solver = G1ThrowSearch(robot, link_ee)

    # in order of: 
    waist_y_jnt = -90.0
    # waist_r_jnt = 0.0
    # waist_p_jnt = 0.0
    l_shoulder_p_jnt = 0.0
    l_shoulder_r_jnt = 0.0
    l_shoulder_y_jnt = 0.0
    l_elbow_jnt = 0.0
    l_wrist_r_jnt = 0.0
    l_wrist_p_jnt = 0.0
    l_wrist_y_jnt = 0.0
    r_shoulder_p_jnt = 45
    r_shoulder_r_jnt = -30.0
    r_shoulder_y_jnt = 0.0
    r_elbow_jnt = 120
    r_wrist_r_jnt = 90
    r_wrist_p_jnt = 0.0
    r_wrist_y_jnt = 0.0#-90.0
    q_0 = optas.np.deg2rad([waist_y_jnt, l_shoulder_p_jnt, l_shoulder_r_jnt, l_shoulder_y_jnt, l_elbow_jnt, l_wrist_r_jnt, l_wrist_p_jnt, l_wrist_y_jnt, r_shoulder_p_jnt, r_shoulder_r_jnt, r_shoulder_y_jnt, r_elbow_jnt, r_wrist_r_jnt, r_wrist_p_jnt, r_wrist_y_jnt])
    # q_0 = optas.np.deg2rad([waist_y_jnt, waist_r_jnt, waist_p_jnt, l_shoulder_p_jnt, l_shoulder_r_jnt, l_shoulder_y_jnt, l_elbow_jnt, l_wrist_r_jnt, l_wrist_p_jnt, l_wrist_y_jnt, r_shoulder_p_jnt, r_shoulder_r_jnt, r_shoulder_y_jnt, r_elbow_jnt, r_wrist_r_jnt, r_wrist_p_jnt, r_wrist_y_jnt])
    print("q_0: ", q_0.shape)
    # r_T = [1, 0, 1]
    r_T = [2.0, 0, 0]
    
    soln, soln_ee, soln_manip, dqf = ik_solver.SolveIK(optas.np.zeros(robot.ndof), r_T)
    # soln, soln_ee, soln_manip = ik_solver.SolveIK(optas.np.zeros(17), r_T)
    soln_ee = ((soln_ee.full()).T)[0] # convert to numpy array 
    ## Get end configuration, end velocity
    quat_T, mu_hat, v_0 = CalcTrajParams(r_T, soln_ee)
    # 2.51106, 0, 0, 0, 0, 0, 0, 0, -4.87836, 0.730323, 0.0289881, -2.38707, -0.00207667, -0.167986, 02.51106, 0, 0, 0, 0, 0, 0, 0, -4.87836, 0.730323, 0.0289881, -2.38707, -0.00207667, -0.167986, 0
    # q_0 = soln
    print("dqf: ", dqf*v_0)
    v_ee = mu_hat*v_0
    # soln = optas.DM.zeros(17, 1)
    ## plan trajectory with those as constraints
    ## Contraints: end pose, linear velocity, zero angular velocity, 
    
    # vis = Visualizer()
    # vis.robot(robot, soln, alpha=1.0, show_links = True)
    # vis.grid_floor()
    # vis.start()
    # dqf = optas.SX.zeros(17)
    g1_dual_arm.reset(q_0)
    dual_g1_planner.reset(q_0, v_ee, soln, dqf*v_0)
    # dual_g1_planner.reset(q_0, v_ee, soln)
    # plan_q, plan_dq = dual_g1_planner.plan()
    soln = ((soln.full()).T)[0] # convert to numpy array 
    dqf = ((dqf.full()).T)[0] # convert to numpy array 
    plan_q = dual_g1_planner.ComputePolyThrowTraj(q_0, np.zeros(len(q_0)), soln, dqf, 0, 1.0)
    
    pb.start()
    pybullet_api.time.sleep(2.0)
    start_time = pybullet_api.time.time()
    traj_done = False
    i = 0
    while True:
        t = pybullet_api.time.time() - start_time
        if not traj_done and t > dual_g1_planner.T_traj:
            print("done!")
            traj_done = True
        elif not traj_done:
            g1_dual_arm.cmd(plan_q[:,i])
            wrist_link_state = g1_dual_arm.GetLinkState(20, True, True) # get wrist yaw joint state
            print("Linear val: ", wrist_link_state[6])
            # print("Angular_val: ", wrist_link_state[7])
            i += 1
        pybullet_api.time.sleep(dt*float(gui))

    pybullet_api.time.sleep(10.0*float(gui))

    pb.stop()
    pb.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
