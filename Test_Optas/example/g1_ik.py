# Example of an Inverse Kinematic (IK) solver applied to a 3 dof planar robot

# Python standard lib
import os
import pathlib

# OpTaS
import optas
from optas.visualize import Visualizer

class G1IK:
    def __init__(self, robot, eff_link, base_link=None):
        self.robot_name = robot.get_name()
        # Setup optimization builder
        self.builder = optas.OptimizationBuilder(T=1, robots=[robot])
        self.fk = robot.get_global_link_position_function(link=eff_link)
        self.quat = robot.get_global_link_quaternion_function(link=eff_link)
        limits = robot.get_limits(time_deriv=0)
        q_min = limits[0]
        q_max = limits[1]

        # get robot state variables
        self.q_T = self.builder.get_model_states(self.robot_name)
        self.builder.add_bound_inequality_constraint("joint", q_min, self.q_T, q_max)

        q_0 = self.builder.add_parameter("q_0", robot.ndof)  # initial robot joint configuration
        x_T = self.builder.add_parameter("x_T", 3)  # target position
        theta_T = self.builder.add_parameter("theta_T", 4)  # target ee orientation

        self.builder.add_cost_term("cost", optas.sumsqr(self.q_T - q_0))

        self.builder.add_equality_constraint("FK", self.fk(self.q_T), x_T)
        self.builder.add_equality_constraint("FK_orientation", self.quat(self.q_T), theta_T)
        # setup solver
        self.solver_casadi = optas.CasADiSolver(self.builder.build()).setup("ipopt")

    def SolveIK(self, x_T, theta_T, q_0):

        # Set parameters
        self.solver_casadi.reset_parameters({
            "q_0": optas.DM(q_0),
            "x_T": x_T,
            "theta_T": theta_T})

        # set initial seed
        self.solver_casadi.reset_initial_seed({f"{self.robot_name}/q/x": q_0})
        # solve problem
        solution_casadi = self.solver_casadi.solve()
        print("***********************************")
        print("Casadi solution:")
        print(solution_casadi[f"{self.robot_name}/q"] * (180.0 / optas.pi))
        print(self.fk(solution_casadi[f"{self.robot_name}/q"]))
        return solution_casadi[f"{self.robot_name}/q"]
        
# def old_main():
#     cwd = pathlib.Path(__file__).parent.resolve()  # path to current working directory

#     urdf_filename = os.path.join(cwd, "robots", "g1", "g1_dual_arm.urdf")
#     # Setup robot
#     robot = optas.RobotModel(urdf_filename)
#     robot_name = robot.get_name()
#     link_ee = "right_wrist_yaw_link"  # end-effector link name

#     # Setup optimization builder
#     builder = optas.OptimizationBuilder(T=1, robots=[robot])

#     # get robot state variables
#     q_T = builder.get_model_states(robot_name)

#     x_T = [0.2, -0.2, 0.2]  # target end-effector position
#     theta_T = [0, 0, 0, 1] # target end-effector orientation
#     waist_y_jnt = 0.0
#     waist_r_jnt = 0.0
#     waist_p_jnt = 0.0
#     l_shoulder_p_jnt = 0.0
#     l_shoulder_r_jnt = 0.0
#     l_shoulder_y_jnt = 0.0
#     l_elbow_jnt = 0.0
#     l_wrist_r_jnt = 0.0
#     l_wrist_p_jnt = 0.0
#     l_wrist_y_jnt = 0.0
#     r_shoulder_p_jnt = 0
#     r_shoulder_r_jnt = 0.0
#     r_shoulder_y_jnt = 0.0
#     r_elbow_jnt = 0
#     r_wrist_r_jnt = 0
#     r_wrist_p_jnt = 0.0
#     r_wrist_y_jnt = 0.0#-90.0
#     q_0 = optas.np.deg2rad([waist_y_jnt,
#         waist_r_jnt,
#         waist_p_jnt,
#         l_shoulder_p_jnt,
#         l_shoulder_r_jnt,
#         l_shoulder_y_jnt,
#         l_elbow_jnt,
#         l_wrist_r_jnt,
#         l_wrist_p_jnt,
#         l_wrist_y_jnt,
#         r_shoulder_p_jnt,
#         r_shoulder_r_jnt,
#         r_shoulder_y_jnt,
#         r_elbow_jnt,
#         r_wrist_r_jnt,
#         r_wrist_p_jnt,
#         r_wrist_y_jnt])
#     limits = robot.get_limits(time_deriv=0)
#     q_min = limits[0]
#     q_max = limits[1]

#     # forward kinematics
#     fk = robot.get_global_link_position_function(link=link_ee)
#     quat = robot.get_global_link_quaternion_function(link=link_ee)

#     # Setting optimization - cost term and constraints
#     builder.add_cost_term("cost", optas.sumsqr(q_T - q_0))
#     print("shape; ", fk(q_T).shape)
#     builder.add_equality_constraint("FK", fk(q_T), x_T)
#     builder.add_equality_constraint("FK_orientation", quat(q_T), theta_T)

#     builder.add_bound_inequality_constraint("joint", q_min, q_T, q_max)

#     # setup solver
#     solver_casadi = optas.CasADiSolver(builder.build()).setup("ipopt")
#     # set initial seed
#     solver_casadi.reset_initial_seed({f"{robot_name}/q/x": q_0})
#     # solve problem
#     solution_casadi = solver_casadi.solve()
#     print("***********************************")
#     print("Casadi solution:")
#     print(solution_casadi[f"{robot_name}/q"] * (180.0 / optas.pi))
#     print(fk(solution_casadi[f"{robot_name}/q"]))

#     solver_slsqp = optas.ScipyMinimizeSolver(builder.build()).setup("SLSQP")
#     solver_slsqp.reset_initial_seed(solution_casadi)
#     # solve problem
#     solution_slsqp = solver_slsqp.solve()
#     print("***********************************")
#     print("SLSQP solution:")
#     print(solution_slsqp[f"{robot_name}/q"] * (180.0 / optas.pi))
#     print(fk(solution_slsqp[f"{robot_name}/q"]))
    
#     # print("soln_slsqp type: ", type(solution_slsqp[]))
#     #setup visualizer
#     vis = Visualizer()
#     vis.robot(robot, solution_casadi[f"{robot_name}/q"], alpha=0.5)
#     vis.grid_floor()
#     vis.start()

#     vis = Visualizer()
#     vis.robot(robot, solution_slsqp[f"{robot_name}/q"], alpha=0.5)
#     vis.grid_floor()
#     vis.start()
    
#     return 0

def main():
    cwd = pathlib.Path(__file__).parent.resolve()  # path to current working directory

    urdf_filename = os.path.join(cwd, "robots", "g1", "g1_dual_arm.urdf")
    # Setup robot
    robot = optas.RobotModel(urdf_filename)
    robot_name = robot.get_name()
    link_ee = "right_wrist_yaw_link"  # end-effector link name

    ik_solver = G1IK(robot, link_ee)

    x_T = [0.2, -0.2, 0.2]  # target end-effector position
    theta_T = [0, 0, 0, 1] # target end-effector orientation
    theta_T = [0, 0, 0.3826834, 0.9238795] # target end-effector orientation
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
    r_shoulder_p_jnt = 0
    r_shoulder_r_jnt = 0.0
    r_shoulder_y_jnt = 0.0
    r_elbow_jnt = 0
    r_wrist_r_jnt = 0
    r_wrist_p_jnt = 0.0
    r_wrist_y_jnt = 0.0#-90.0
    q_0 = optas.np.deg2rad([waist_y_jnt,
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
    
    vis = Visualizer()
    ### Test basic trajectory with previous soln as initial guess. Speeds up a significant amount if have good init guess
    # for i in range(10):
    #     soln = ik_solver.SolveIK([0.1 + i/20.0, 0, 0.2], theta_T, q_0)
    #     vis.robot(robot, soln, alpha=0.5)
    #     q_0 = soln

    soln1 = ik_solver.SolveIK(x_T, theta_T, q_0)

    soln2 = ik_solver.SolveIK([0.5, 0, 0.2], theta_T, soln1)
    vis.robot(robot, soln1, alpha=0.75)
    vis.robot(robot, soln2, alpha=0.75)
    vis.grid_floor()
    vis.start()

if __name__ == "__main__":
    main()
