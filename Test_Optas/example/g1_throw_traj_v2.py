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
from g1_ik import G1IK, CalcTrajParams
from optas.visualize import Visualizer

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
        self.T_traj = 0.0
        # Parameters
        self.T = 50
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
        r_targ = builder.add_parameter("r_targ", 3)  # ball target position in world coordinates

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

        ######################## Robot States ########################
        # Get end effector position FK function
        posr_ee = g1.get_global_link_position_function(link_ee_r, n=self.T)
        # posl_ee = g1.get_global_link_position_function(link_ee_l, n=self.T)
        
        # Get Torso position FK function
        pos_head = g1.get_link_position_function(link_head, "pelvis", n=self.T)
        rot_head = g1.get_link_rotation_function(link_head, "pelvis", n=self.T)
        torso_yaw = g1.get_global_link_rpy_function(link="torso_link")

        # Get joint trajectory: ndof-by-T sym array for robot trajectory
        Q = builder.get_model_states(self.g1_name) 
        ddQ = builder.get_model_states(self.g1_name, time_deriv=2)
        # Get end-effector position trajectories
        ee_pos_path_r = posr_ee(Q)
        # ee_pos_path_l = posl_ee(Q)
        eff_y_axis = g1.get_global_link_axis(link_ee_r, Q, 'y')

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

        
        ############################ Constrain End Effector #################################

        mu_hat = optas.SX.zeros(3, 1)
        tmp = optas.SX.zeros(3, 1)
        tmp = r_targ - ee_pos_path_r[:3, -1] # vector from end effector to target
        Z = tmp[2]
        tmp[2] = optas.norm_2(tmp) #modify z component to make the 45 deg angle 
        mu_hat = tmp/optas.norm_2(tmp) # unit launch direction vector
        v_0 = optas.sqrt(9.81*optas.norm_2(tmp)/(2*(mu_hat[2]*optas.norm_2(tmp) - Z*optas.norm_2(mu_hat))*optas.norm_2(mu_hat)))

        ######################## Constraints ########################
        # add equality constraint to keep head upright
        pg = builder.add_parameter("position_goal", 3)
        pF = g1.get_global_link_position(link_head, Q)
        Qf = builder.get_model_state(self.g1_name, t=-1, time_deriv=0)
        # builder.add_equality_constraint("head_upright", pF, pg)
        # builder.add_equality_constraint("end_pose", Qf, q_f) 

        goal_eff_ang = optas.DM([0.0, 0.0, 0.0])
        temp_vee = optas.DM([2.83203217, 0.77524903, 2.93622501])
        # builder.add_equality_constraint("eff_vel", vel_ee, temp_vee, reduce_constraint=True)
        builder.add_equality_constraint("eff_orientation", eff_y_axis[:,-1], mu_hat)
        # builder.add_equality_constraint("eff_vel", cart_vel_r[:3,-1], mu_hat*v_0, reduce_constraint=True)
        builder.add_equality_constraint("eff_ang_vel", cart_vel_r[3:, -1], goal_eff_ang, reduce_constraint=True)
        builder.add_bound_inequality_constraint("torso_yaw", -1.57, torso_yaw(Q), 1.57)

        builder.add_leq_inequality_constraint("Tmax_lower", Tmax, 0.0)
        builder.add_leq_inequality_constraint("Tmax_upper", 1.0, Tmax)
        #######################################################


        ######################## Costs ########################
        # Cost: minimize joint velocity
        w_dq = 0.01
        w_q = 10
        w = 0.0
        # builder.add_cost_term("g1_min_join_vel_r", w_dq * optas.sumsqr(dQr))
        builder.add_cost_term("joint_accel_cost", w * optas.sumsqr(ddQ)) # minimize joint acceleration
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

    def reset(self, q0, r_T):
        # Set parameters
        self.solver.reset_parameters({"q0": optas.DM(q0), "r_targ": optas.DM(r_T)})
        Q0 = optas.diag(q0) @ optas.DM.ones(self.g1_ndof, self.T)
        self.solver.reset_initial_seed({f"{self.g1_name}/q/x": Q0})

    def get_target(self):
        return self.solution

    def plan(self):
        self.solve()
        solution = self.get_target()
        print('solution: ', float(solution[f"Tmax"][0]))
        # Interpolate
        self.T_traj = float(solution[f"Tmax"][0]) 
        plan = self.solver.interpolate(solution[f"{self.g1_name}/q"], self.T_traj)

        return plan


def main(gui=True):
    dual_g1_planner = DualG1Planner()
    hz = 50
    dt = 1.0 / float(hz)
    pb = pybullet_api.PyBullet(dt, gui=gui)
    g1_dual_arm = pybullet_api.G1DualArm(base_position=g1_base_position)
    cwd = pathlib.Path(__file__).parent.resolve()  # path to current working directory

    urdf_filename = os.path.join(cwd, "robots", "g1", "g1_dual_arm.urdf")
    # Setup robot
    robot = optas.RobotModel(urdf_filename)
    link_ee = "right_wrist_yaw_link"
    ik_solver = G1IK(robot, link_ee)

    # in order of: 
    waist_y_jnt = -45.0
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
    r_shoulder_r_jnt = -45.0
    r_shoulder_y_jnt = 0.0
    r_elbow_jnt = 120
    r_wrist_r_jnt = 90
    r_wrist_p_jnt = 0.0
    r_wrist_y_jnt = 0.0#-90.0
    q_0 = optas.np.deg2rad([waist_y_jnt, waist_r_jnt, waist_p_jnt, l_shoulder_p_jnt, l_shoulder_r_jnt, l_shoulder_y_jnt, l_elbow_jnt, l_wrist_r_jnt, l_wrist_p_jnt, l_wrist_y_jnt, r_shoulder_p_jnt, r_shoulder_r_jnt, r_shoulder_y_jnt, r_elbow_jnt, r_wrist_r_jnt, r_wrist_p_jnt, r_wrist_y_jnt])
    # q_0 = optas.np.zeros(17)
    ## get target position
    x_T = [0.21, -0.49, 0.2]
    x_T = [0.0, -0.55, 0.25]
    r_T = [1, 0, 1]
    r_T = [0.5, 0, 0.5]
    
    ## Get end configuration, end velocity
    # quat_T, mu_hat, v_0 = CalcTrajParams(r_T, x_T)
    # print("release velocity: ", v_0)
    # soln, soln_ee, soln_manip = ik_solver.SolveIK(x_T, quat_T, optas.np.zeros(17), r_T)
    
    # q_0 = soln
    # v_ee = mu_hat*v_0
    # print('type: :', type(soln))
    # print("release vector: ", v_ee)
    # soln = optas.DM.zeros(17, 1)
    ## plan trajectory with those as constraints
    ## Contraints: end pose, linear velocity, zero angular velocity, 
    
    # vis = Visualizer()
    # vis.robot(robot, soln, alpha=1.0, show_links = True)
    # vis.grid_floor()
    # vis.start()

    g1_dual_arm.reset(q_0)
    dual_g1_planner.reset(q_0, r_T)
    plan = dual_g1_planner.plan()

    pb.start()
    pybullet_api.time.sleep(2.0)
    start_time = pybullet_api.time.time()
    traj_done = False
    while True:
        t = pybullet_api.time.time() - start_time
        if not traj_done and t > dual_g1_planner.T_traj:
            print("done!")
            traj_done = True
        elif not traj_done:
            g1_dual_arm.cmd(plan(t))
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
