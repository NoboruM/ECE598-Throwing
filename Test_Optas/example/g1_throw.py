# Python standard lib
import os
import sys
import pathlib
import time
# OpTaS
import optas
from optas.templates import Manager

# PyBullet
import pybullet_api
###### Only if using Casadi
import casadi as cs

g1_base_position = [0.0, 0.0, 0.5]


class DualG1Planner(Manager):
    def setup_solver(self):
        # Parameters
        self.T = 50
        Tmax = 10.0
        link_ee_r = "right_wrist_yaw_link"
        link_head = "head_link"
        link_torso = "torso_link"
        t = optas.linspace(0, Tmax, self.T)
        dt = float((t[1] - t[0]).toarray()[0, 0])
        # setup throwing parameters
        x_targ = 0.0
        y_targ = 0.0
        z_targ = 0.0

        theta = 0.0
        speed_release = 0.0
        v_release = optas.SX.zeros(3, self.T)
        v_release[:,-1] = optas.SX.ones(3,1)*1

        ######################## Initial setup ########################
        # Setup robot models
        self.g1 = self._setup_g1_model("g1_dual_arm", g1_base_position)
        # Get robot names
        g1_name = self.g1.get_name()
        # Setup optimization builder
        builder = optas.OptimizationBuilder(T=self.T, robots=[self.g1], derivs_align=True)
        # Setup parameters
        qc = builder.add_parameter("qc", self.g1.ndof)
        head_orient = builder.add_parameter("head_orientation", 3)

        ##################################################################
        
        ######################## Robot State Info ########################
        # Get end effector position FK function
        posr_ee = self.g1.get_global_link_position_function(link_ee_r, n=self.T)
        # Get joint trajectory: ndof-by-T sym array for robot trajectory
        Q = builder.get_model_states(g1_name)
        # Get joint velocity trajectory
        dQ = builder.get_model_states(g1_name, time_deriv=1)
        ddQ = builder.get_model_states(g1_name, time_deriv=2)

        # Get start position for each arm
        pos0r = self.g1.get_global_link_position(link_ee_r, qc)

        ee_jacobian = self.g1.get_global_link_geometric_jacobian_function(link_ee_r, n=self.T)
        # Compute Cartesian velocity: J(q) * dq
        cart_vel_r = optas.SX.zeros(6, self.T-1)  # 6D twist (linear + angular)
        J = ee_jacobian(Q)
        for i in range(self.T-1):
            cart_vel_r[:, i] = J[i] @ dQ[:, i]  # J * dq
        #############################################################


        ######################## Constraints ########################
        # Constraint: initial configuration
        builder.fix_configuration(g1_name, qc)
        # builder.fix_configuration(g1_name, qc, time_deriv=1)  # initial joint vel is zero
        builder.initial_configuration(g1_name, time_deriv=1) # set initial joint velocity zero
        # Constrain dynamics
        builder.enforce_model_limits(g1_name, time_deriv=0)
        builder.enforce_model_limits(g1_name, time_deriv=1)

        builder.integrate_model_states(g1_name, time_deriv=1,  dt=dt) # i.e. integrate velocities to positions 

        # goal_eff_position = optas.DM([0.825, 0.35, 0.2])
        # goal_eff_vel = optas.DM([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # [wx, wy, wz, vx, vy, vz]
        goal_eff_vel = optas.SX([0.2, 0.0, 0.0]) # [vx, vy, vz]
        qf = builder.get_model_state(g1_name, t=-1, time_deriv=0)
        dqf = builder.get_model_state(g1_name, t=-1, time_deriv=1)
        # cart_vel_r = optas.SX.zeros(6, self.T-1)  # 6D twist (linear + angular)
        # J_t = self.g1.get_global_link_geometric_jacobian(link_ee_r, qf)
        # J_t = self.g1.get_global_link_analytical_jacobian_function(link_ee_r, n=self.T)
        J_t = self.g1.get_global_link_linear_jacobian(link_ee_r, qf)
        J_test = self.g1.get_link_linear_jacobian(link_ee_r, qf, link_torso)
        # goal_ee_vel = optas.SX.zeros(6, self.T-1)
        # for i in range(self.T-1):

        # for i in range(self.T):
        #     vf[:,i] = J_t(qc)
        print("J_t: ", J_t.shape)
        print("J_test: ", J_test.shape)
        print("dqf: ", dqf.shape)
        vf = J_t@dqf
        print("vf: ", vf.shape)
        print("goal_eff_vel: ", goal_eff_vel.shape)
        builder.add_equality_constraint("eff_vel", vf, goal_eff_vel, reduce_constraint=True)

        ## attempting end effector velocity trajectry constraint

        # eff_vel_traj = optas.SX.zeros(3, self.T-1)
        # eff_vel_traj[-1,0] = 0.4
        # builder.add_equality_constraint("eff_vel_traj", cart_vel_r[:3,:], eff_vel_traj)
        # add equality constraint to keep head upright
        # pF = g1.get_global_link_position(link_head, Q)
        # builder.add_equality_constraint("head_upright", pF, head_orient)
        #######################################################


        ######################## Costs ########################
        w_dq = 0.1
        # vel_traj = optas.SX.ones(17, self.T-1)*(-1)
        # builder.add_cost_term("g1_min_join_vel_r", w_dq * optas.sumsqr(vel_traj - dQ))

        # Cost: minimize Cartesian velocity magnitude
        w_vel = 0.01
        w_terminal = 10.0  # Weight for terminal velocity
        w_vel_costs = optas.SX.zeros(1, self.T)
        w_vel_costs[-1] = w_terminal
        w_broadcasted = optas.repmat(w_terminal, 3, 1)  # Shape: 3xT
        vel_traj = optas.SX.ones(6, self.T-1)
        builder.add_cost_term("joint_accel_cost", w_vel * optas.sumsqr(ddQ))
        # builder.add_cost_term("end_vel", w_terminal*optas.sumsqr(goal_eff_vel - vf))
        # Terminal cost: penalize final velocity
        # terminal_vel_cost = w_terminal * optas.sumsqr((v_release[:,-1] - cart_vel_r[:3, -1]))
        # builder.add_cost_term("terminal_vel_cost", terminal_vel_cost)
        w = 100
        # manipulability_cost = -w * optas.log(optas.det(J @ J.T))  # Maximize manipulability
        # builder.add_cost_term("singularity_avoidance", manipulability_cost)

        #######################################################

        # Setup solver
        optimization = builder.build()

        solver = optas.CasADiSolver(optimization).setup("ipopt")
        # solver.reset_initial_seed()
        # Save variables for later
        self.g1_name = g1_name
        self.Tmax = Tmax

        return solver

    def _setup_g1_model(self, name, base_position):
        cwd = pathlib.Path(
            __file__
        ).parent.resolve()  # path to current working directory
        urdf_filename = os.path.join(cwd, "robots", "g1", "g1_dual_arm.urdf")
        print("urdf_filename: ", urdf_filename)
        model = optas.RobotModel(
            urdf_filename=urdf_filename,
            name=name,
            time_derivs=[0, 1, 2],  # i.e. joint position/velocity trajectory
        )
        # model.add_base_frame("base", xyz=base_position)
        return model

    def is_ready(self):
        return True

    def reset(self, qc, qcr):
        # Set parameters
        self.solver.reset_parameters({"qc": optas.DM(qc)})
        # Set initial seed, note joint velocity will be set to zero
        Q0 = optas.diag(qc) @ optas.DM.ones(self.g1.ndof, self.T)
        self.solver.reset_initial_seed({f"{self.g1_name}/q/x": Q0})

    def get_target(self):
        return self.solution

    def plan(self):
        self.solve()
        
        solution = self.get_target()
        # print("solution: ", solution)
        # print("constraint violation: ", solution["eff_vel"])
        # time.sleep(10)
        # Interpolate
        plan = self.solver.interpolate(solution[f"{self.g1_name}/q"], self.Tmax)

        return plan


def main(gui=True):
    dual_g1_planner = DualG1Planner()

    hz = 50
    dt = 1.0 / float(hz)
    pb = pybullet_api.PyBullet(dt, gui=gui)

    g1_dual_arm = pybullet_api.G1DualArm(base_position=g1_base_position)

    # in order of: 
    waist_y_jnt = 0.0
    waist_r_jnt = 0.0
    waist_p_jnt = 0.0
    l_shoulder_p_jnt = 0.0
    l_shoulder_r_jnt = 0.0
    l_shoulder_y_jnt = 0.0
    l_elbow_jnt = 0.0
    l_wrist_r_jnt = 0.0
    l_wrist_p_jnt = 0.0
    l_wrist_y_jnt = 0.0
    r_shoulder_p_jnt = 0.0#90
    r_shoulder_r_jnt = 0.0
    r_shoulder_y_jnt = 0.0
    r_elbow_jnt = 0#120
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
    plan = dual_g1_planner.plan()

    pb.start()
    pybullet_api.time.sleep(2.0)
    start_time = pybullet_api.time.time()
    traj_done = False
    while True:
        t = pybullet_api.time.time() - start_time
        if t > dual_g1_planner.Tmax and not traj_done:
            print('done!!!')
            traj_done = True
            # break
        elif not traj_done:
            g1_dual_arm.cmd(plan(t))
            wrist_link_state = g1_dual_arm.GetLinkState(20, True, True) # get wrist yaw joint state
            print("Linear val: ", wrist_link_state[6])
            print("Angular_val: ", wrist_link_state[7])

        # print(t/float(dual_g1_planner.Tmax))
        pybullet_api.time.sleep(dt*float(gui))

    pybullet_api.time.sleep(10.0*float(gui))

    pb.stop()
    pb.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())


