# Python standard lib
import os
import sys
import pathlib

# OpTaS
import optas
from optas.templates import Manager

# PyBullet
import pybullet_api

g1_base_position = [0.0, 0.0, 0.5]


class DualG1Planner(Manager):
    def setup_solver(self):
        # Parameters
        T = 50
        Tmax = 10.0
        link_ee_r = "right_wrist_yaw_link"
        link_ee_l = "left_wrist_yaw_link"
        link_head = "head_link"
        t = optas.linspace(0, Tmax, T)
        dt = float((t[1] - t[0]).toarray()[0, 0])

        # Setup robot models
        g1 = self._setup_g1_model("g1_dual_arm", g1_base_position)
        # Get robot names
        g1_name = g1.get_name()

        # Setup optimization builder
        builder = optas.OptimizationBuilder(T=T, robots=[g1])

        # Setup parameters
        qc = builder.add_parameter("qc", g1.ndof)

        # Constraint: initial configuration
        builder.fix_configuration(g1_name, qc)

        # Constraint: dynamics
        builder.integrate_model_states(
            g1_name,
            time_deriv=1,  # i.e. integrate velocities to positions
            dt=dt,
        )

        # Get end effector position FK function
        posr_ee = g1.get_global_link_position_function(link_ee_r, n=T)
        posl_ee = g1.get_global_link_position_function(link_ee_l, n=T)
        
        # Get Torso position FK function
        pos_head = g1.get_link_position_function(link_head, "torso_link", n=T)

        # Get joint trajectory: ndof-by-T sym array for robot trajectory
        Ql = builder.get_model_states(
            g1_name
        ) 
        # add equality constraint to keep head upright
        pg = builder.add_parameter("position_goal", 3)
        pF = g1.get_global_link_position(link_head, Ql)
        builder.add_equality_constraint("head_upright", pF, pg)

        # Get end-effector position trajectories
        ee_pos_path_r = posr_ee(Ql)
        ee_pos_path_l = posl_ee(Ql)

        # Get joint velocity trajectory
        dQr = builder.get_model_states(g1_name, time_deriv=1)
        dQl = builder.get_model_states(g1_name, time_deriv=1)

        # Cost: minimize joint velocity
        w_dq = 0.01
        builder.add_cost_term("g1_min_join_vel_r", w_dq * optas.sumsqr(dQr))
        builder.add_cost_term("g1_min_join_vel_l", w_dq * optas.sumsqr(dQl))

        # Get start position for each arm
        pos0r = g1.get_global_link_position(link_ee_r, qc)
        pos0l = g1.get_global_link_position(link_ee_l, qc)

        # Get start position for head
        pos0_head = g1.get_link_position(link_head, qc, "torso_link")

        # Find first goal positions
        pos1r = pos0r + optas.DM([0.0, -0.015, -0.2])
        pos1l = pos0l + optas.DM([0.0, 0.015, -0.2])

        # Find second goal positions
        pos2r = pos1r + optas.DM([0.0, -0.015, 0.3])
        pos2l = pos1l + optas.DM([0.0, 0.015, 0.3])

        # Find ee path
        path_ee_r = optas.SX.zeros(3, T)
        path_ee_l = optas.SX.zeros(3, T)
        for i in range(T):
            alpha_ = float(i) / float(T - 1)
            if alpha_ < 0.4:
                alpha = alpha_ / 0.4

                path_ee_r[:, i] = alpha * pos1r + (1.0 - alpha) * pos0r
                path_ee_l[:, i] = alpha * pos1l + (1.0 - alpha) * pos0l

            elif 0.4 <= alpha_ < 0.5:
                path_ee_r[:, i] = pos1r
                path_ee_l[:, i] = pos1l

            else:
                alpha = (alpha_ - 0.5) / 0.5

                path_ee_r[:, i] = alpha * pos2r + (1.0 - alpha) * pos1r
                path_ee_l[:, i] = alpha * pos2l + (1.0 - alpha) * pos1l
        # cost term to track the end effector paths
        builder.add_cost_term("ee_pos_path_r", optas.sumsqr(ee_pos_path_r - path_ee_r))
        builder.add_cost_term("ee_pos_path_l", optas.sumsqr(ee_pos_path_l - path_ee_l))

        # Setup solver
        optimization = builder.build()
        solver = optas.CasADiSolver(optimization).setup("ipopt")

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
            time_derivs=[0, 1],  # i.e. joint position/velocity trajectory
        )
        return model

    def is_ready(self):
        return True

    def reset(self, qc, qcr):
        # Set parameters
        self.solver.reset_parameters(
            {
                "qc": optas.DM(qc)
            }
        )

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

    g1_dual_arm = pybullet_api.G1DualArm(base_position=g1_base_position)
    print("ndof = ", g1_dual_arm.ndof)
    qc = optas.np.deg2rad([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    g1_dual_arm.reset(qc)

    dual_g1_planner.reset(qc, qc)
    planl = dual_g1_planner.plan()

    pb.start()
    pybullet_api.time.sleep(2.0)
    start_time = pybullet_api.time.time()

    while True:
        t = pybullet_api.time.time() - start_time
        if t > dual_g1_planner.Tmax:
            break

        g1_dual_arm.cmd(planl(t))

        pybullet_api.time.sleep(dt*float(gui))

    pybullet_api.time.sleep(10.0*float(gui))

    pb.stop()
    pb.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())