# Example of an Inverse Kinematic (IK) solver applied to a 3 dof planar robot

# Python standard lib
import os
import pathlib
from casadi import SX, transpose
import numpy as np
from scipy.spatial.transform import Rotation
# OpTaS
import optas
from optas.visualize import Visualizer

class G1ThrowSearch:
    def __init__(self, robot, eff_link, base_link=None):
        self.robot = robot
        self.robot_name = robot.get_name()
        self.eff_link = eff_link
        # Setup optimization builder
        self.builder = optas.OptimizationBuilder(T=1, robots=[robot])
        self.fk = robot.get_global_link_position_function(link=eff_link)
        self.quat = robot.get_global_link_quaternion_function(link=eff_link)
        self.torso_quat = robot.get_global_link_quaternion_function(link="torso_link")
        self.torso_rpy = robot.get_global_link_rpy_function(link="torso_link")
        limits = robot.get_limits(time_deriv=0)
        q_min = limits[0]
        q_max = limits[1]
        self.g = 9.81
        # get robot state variables
        self.q_T = self.builder.get_model_states(self.robot_name)
        self.J = robot.get_global_link_linear_jacobian(eff_link, self.q_T)
        self.eff_y_axis = robot.get_global_link_axis(eff_link, self.q_T, 'y')

        self.builder.add_bound_inequality_constraint("joint", q_min, self.q_T, q_max)
        self.builder.add_bound_inequality_constraint("torso_yaw", -0.5, self.torso_rpy(self.q_T)[2,:], 0.5)
        # self.builder.add_bound_inequality_constraint("torso_roll", -0.174533, self.torso_rpy(self.q_T)[0,:], 0.174533)
        self.builder.add_equality_constraint("torso_roll", 0, self.torso_rpy(self.q_T)[0,:])
        # self.builder.add_equality_constraint("torso_pitch", 0, self.torso_rpy(self.q_T)[1,:])
        self.builder.add_bound_inequality_constraint("torso_pitch", -0.174533, self.torso_rpy(self.q_T)[1,:], 0.174533)

        q_0 = self.builder.add_parameter("q_0", robot.ndof)  # initial robot joint configuration
        r_targ = self.builder.add_parameter("r_targ", 3)  # ball target position in world coordinates


        r_ee = self.fk(self.q_T)
        mu_hat = optas.SX.zeros(3, 1)
        mu_2 = optas.SX.zeros(3, 1)

        tmp = r_targ - r_ee # vector from end effector to target
        Z = optas.norm_2(tmp[0:2]) #modify z component to make the 45 deg angle 
        tmp[2] = Z
        mu_hat = tmp/optas.norm_2(tmp) # unit launch direction vector


        # self.builder.add_cost_term("cost", optas.sumsqr(self.q_T - q_0))
        # self.builder.add_cost_term("torso_tilt", 10*optas.sumsqr(optas.DM([0, 0, 0, 1]) - self.torso_quat(self.q_T)))
        self.builder.add_equality_constraint("eff_orientation", self.eff_y_axis, mu_hat)
        self.builder.add_leq_inequality_constraint("eff_in_front", 0.0, r_ee[0])
        A = self.J@transpose(self.J)
        mu_2 = 1.0/optas.sqrt(transpose(mu_hat)@optas.inv(A)@mu_hat) # for position q_t, this measures the max achievable velocity in direction mu_hat 
        print("mu_2; ", mu_2.shape)
        self.builder.add_cost_term("manipulability", -1000*optas.sumsqr(mu_2)) # maximize mu2

        # setup solver
        self.solver_casadi = optas.CasADiSolver(self.builder.build()).setup("ipopt", {"ipopt.print_level": 0})

    def SolveIK(self, q_0, r_targ):

        # Set parameters
        self.solver_casadi.reset_parameters({
            "q_0": optas.DM(q_0),
            "r_targ": r_targ})

        # set initial seed
        self.solver_casadi.reset_initial_seed({f"{self.robot_name}/q/x": q_0})
        # solve problem
        solution_casadi = self.solver_casadi.solve()

        ########################## calculate the manipulability of result: ##########################
        r_ee = self.fk(solution_casadi[f"{self.robot_name}/q"])
        mu_hat = optas.SX.zeros(3, 1)
        tmp = r_targ - r_ee # vector from end effector to target
        Z = optas.norm_2(tmp[0:2]) #modify z component to make the 45 deg angle 
        tmp[2] = Z
        mu_hat = tmp/optas.norm_2(tmp) # unit launch direction vector

        J = self.robot.get_global_link_linear_jacobian(self.eff_link, solution_casadi[f"{self.robot_name}/q"])
        A = J@transpose(J)
        mu_2 = 1.0/optas.sqrt(transpose(mu_hat)@optas.inv(A)@mu_hat) # for position q_t, this measures the max achievable velocity in direction mu_hat 


        print("***********************************")
        print("Casadi solution:")
        print("\t",solution_casadi[f"{self.robot_name}/q"] * (180.0 / optas.pi))
        print("end effector position: ", r_ee)
        print("manipulability score: ", mu_2)
        return solution_casadi[f"{self.robot_name}/q"], r_ee, mu_2
      
def main():
    cwd = pathlib.Path(__file__).parent.resolve()  # path to current working directory

    urdf_filename = os.path.join(cwd, "robots", "g1", "g1_dual_arm.urdf")
    # Setup robot
    robot = optas.RobotModel(urdf_filename)
    robot_name = robot.get_name()
    link_ee = "right_wrist_yaw_link"  # end-effector link name

    throw_pose_finder = G1ThrowSearch(robot, link_ee)

    # theta_T = [0, 0, 0.3826834, 0.9238795] # target end-effector orientation
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
    r_targ = [1.0, 0, 0.5]
    soln1, soln1_ee, soln1_manip = throw_pose_finder.SolveIK(q_0, r_targ)
    # r_targ = [0.5, -0.5, 0.5]
    # soln2, soln2_ee, soln2_manip = ik_solver.SolveIK(q_0, r_targ)
    vis.robot(robot, soln1, alpha=1.0, show_links = True, link_axis_scale=1.0)
    vis.sphere(position=r_targ, radius=0.05, alpha=0.8, rgb=[1, 0, 0])
    # vis.robot(robot, soln2, alpha=0.75, )
    vis.grid_floor()
    vis.start()

if __name__ == "__main__":
    main()
