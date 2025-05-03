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

class G1IK:
    def __init__(self, robot, eff_link, base_link=None):
        self.robot = robot
        self.robot_name = robot.get_name()
        self.eff_link = eff_link
        # Setup optimization builder
        self.builder = optas.OptimizationBuilder(T=1, robots=[robot])
        self.fk = robot.get_global_link_position_function(link=eff_link)
        self.quat = robot.get_global_link_quaternion_function(link=eff_link)
        limits = robot.get_limits(time_deriv=0)
        q_min = limits[0]
        q_max = limits[1]
        self.g = 9.81
        # get robot state variables
        self.q_T = self.builder.get_model_states(self.robot_name)
        self.J = robot.get_global_link_linear_jacobian(eff_link, self.q_T)

        self.builder.add_bound_inequality_constraint("joint", q_min, self.q_T, q_max)

        q_0 = self.builder.add_parameter("q_0", robot.ndof)  # initial robot joint configuration
        x_T = self.builder.add_parameter("x_T", 3)  # target position
        theta_T = self.builder.add_parameter("theta_T", 4)  # target ee orientation
        r_targ = self.builder.add_parameter("r_targ", 3)  # ball target position in world coordinates

        self.builder.add_cost_term("cost", optas.sumsqr(self.q_T - q_0))

        self.builder.add_equality_constraint("FK", self.fk(self.q_T), x_T)
        self.builder.add_equality_constraint("FK_orientation", self.quat(self.q_T), theta_T)

        r_ee = self.fk(self.q_T)
        mu_hat = optas.SX.zeros(3, 1)
        mu_2 = optas.SX.zeros(3, 1)

        tmp = r_targ - r_ee # vector from end effector to target
        Z = optas.norm_2(tmp[0:2]) #modify z component to make the 45 deg angle 
        tmp[2] = Z
        mu_hat = tmp/optas.norm_2(tmp) # unit launch direction vector

        A = self.J@transpose(self.J)
        mu_2 = 1.0/optas.sqrt(transpose(mu_hat)@optas.inv(A)@mu_hat) # for position q_t, this measures the max achievable velocity in direction mu_hat 
        print("mu_2; ", mu_2.shape)
        self.builder.add_cost_term("manipulability", -100*optas.sumsqr(mu_2)) # maximize mu2

        # setup solver
        self.solver_casadi = optas.CasADiSolver(self.builder.build()).setup("ipopt")

    def SolveIK(self, x_T, theta_T, q_0, r_targ):

        # Set parameters
        self.solver_casadi.reset_parameters({
            "q_0": optas.DM(q_0),
            "x_T": x_T,
            "theta_T": theta_T,
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
      
def CalcEEQuat(r_targ, x_T):
    r_target = np.array(r_targ)
    r_ee = np.array(x_T)
    tmp = r_target - r_ee # vector from end effector to target
    Z = np.linalg.norm(tmp[0:2]) #modify z component to make the 45 deg angle 
    tmp[2] = Z
    mu_hat = tmp/np.linalg.norm(tmp) # unit launch direction vector
    original_dir = np.array([0, 1, 0]) # align y axis with the direction
    # compute rotation axis and angle
    cross = np.cross(original_dir, mu_hat)
    axis = cross / np.linalg.norm(cross) if np.linalg.norm(cross) > 1e-6 else np.array([1, 0, 0])
    angle = np.arccos(np.dot(original_dir, mu_hat))
    
    # Handle edge case (opposite vectors)
    if angle < 1e-6:
        return Rotation.identity()  # No rotation needed
    if np.isclose(angle, np.pi):  # 180° rotation
        axis = np.array([1, 0, 0]) if original_dir[0] != 0 else np.array([0, 1, 0])
        
    rot = Rotation.from_quat([*(axis * np.sin(angle/2)), np.cos(angle/2)])
    return rot.as_quat() 

def main():
    cwd = pathlib.Path(__file__).parent.resolve()  # path to current working directory

    urdf_filename = os.path.join(cwd, "robots", "g1", "g1_dual_arm.urdf")
    # Setup robot
    robot = optas.RobotModel(urdf_filename)
    robot_name = robot.get_name()
    link_ee = "right_wrist_yaw_link"  # end-effector link name

    ik_solver = G1IK(robot, link_ee)

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

    x_T = [0.15, -0.2, 0.2]  # target end-effector position in global frame
    r_targ = [1, 1, 0]

    soln1, soln1_ee, soln1_manip = ik_solver.SolveIK(x_T, CalcEEQuat(r_targ, x_T), q_0, r_targ)
    r_targ = [0.5, -0.5, 0.5]
    soln2, soln2_ee, soln2_manip = ik_solver.SolveIK(x_T, CalcEEQuat(r_targ, x_T), q_0, r_targ)
    vis.robot(robot, soln1, alpha=0.15, show_links = True)
    # vis.robot(robot, soln2, alpha=0.75, )
    vis.grid_floor()
    vis.start()

if __name__ == "__main__":
    main()
