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

def Compute1PolyTraj(t, q0, dq0, qf, dqf, t0, tf):
    A = np.array([
        [1, t0, t0**2, t0**3],
        [0, 1, 2*t0, 3*t0**2],
        [1, tf, tf**2, tf**3],
        [0, 1, 2*tf, 3*tf**2]
    ])
    B = np.array([q0, dq0, qf, dqf])
    a0, a1, a2, a3  = np.linalg.solve(A, B)
    return a0 + a1*t + a2*t**2 + a3*t**3


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
        self.builder.add_bound_inequality_constraint("torso_yaw", -1.57, self.torso_rpy(self.q_T)[2,:], 1.57)
        # self.builder.add_bound_inequality_constraint("torso_roll", -0.174533, self.torso_roll(self.q_T)[0,:], 0.174533)
        # self.builder.add_equality_constraint("torso_roll", 0, self.torso_rpy(self.q_T)[0,:])
        # self.builder.add_equality_constraint("torso_pitch", 0, self.torso_rpy(self.q_T)[1,:])
        # self.builder.add_bound_inequality_constraint("torso_pitch", -0.174533, self.torso_rpy(self.q_T)[1,:], 0.174533)

        q_0 = self.builder.add_parameter("q_0", robot.ndof)  # initial robot joint configuration
        r_T = self.builder.add_parameter("r_T", 3)  # ball target position in world coordinates


        r_ee = self.fk(self.q_T)
        mu_hat = optas.SX.zeros(3, 1)
        mu_2 = optas.SX.zeros(3, 1)

        tmp = r_T - r_ee # vector from end effector to target
        Z = optas.norm_2(tmp[0:2]) #modify z component to make the 45 deg angle 
        tmp[2] = Z
        mu_hat = tmp/optas.norm_2(tmp) # unit launch direction vector


        # self.builder.add_cost_term("cost", optas.sumsqr(self.q_T - q_0))
        # self.builder.add_cost_term("torso_tilt", 10*optas.sumsqr(optas.DM([0, 0, 0, 1]) - self.torso_quat(self.q_T)))
        self.builder.add_equality_constraint("eff_orientation", self.eff_y_axis, mu_hat)
        self.builder.add_geq_inequality_constraint("release_position_x", r_ee[0], 0)
        self.builder.add_leq_inequality_constraint("release_position_y", r_ee[1], 0)

        A = self.J@transpose(self.J)
        mu_2 = 1.0/optas.sqrt(transpose(mu_hat)@optas.inv(A)@mu_hat) # for position q_t, this measures the max achievable velocity in direction mu_hat 
        print("mu_2; ", mu_2.shape)
        self.builder.add_cost_term("manipulability", -1000*optas.sumsqr(mu_2)) # maximize mu2

        # setup solver
        self.solver_casadi = optas.CasADiSolver(self.builder.build()).setup("ipopt", {"ipopt.print_level": 0})

    def SolveIK(self, q_0, r_T):

        # Set parameters
        self.solver_casadi.reset_parameters({
            "q_0": optas.DM(q_0),
            "r_T": r_T})

        # set initial seed
        self.solver_casadi.reset_initial_seed({f"{self.robot_name}/q/x": q_0})
        # solve problem
        solution_casadi = self.solver_casadi.solve()

        ########################## calculate the manipulability of result: ##########################
        r_ee = self.fk(solution_casadi[f"{self.robot_name}/q"])
        mu_hat = optas.SX.zeros(3, 1)
        tmp = r_T - r_ee # vector from end effector to target
        Z = optas.norm_2(tmp[0:2]) #modify z component to make the 45 deg angle 
        tmp[2] = Z
        mu_hat = tmp/optas.norm_2(tmp) # unit launch direction vector

        J = self.robot.get_global_link_geometric_jacobian(self.eff_link, solution_casadi[f"{self.robot_name}/q"])
        print("J shape; ", J[:3].shape)
        A = J[:3,:]@transpose(J[:3,:])
        mu_2 = 1.0/optas.sqrt(transpose(mu_hat)@optas.inv(A)@mu_hat) # for position q_t, this measures the max achievable velocity in direction mu_hat 
        lambda_ = 1e-3  # Damping coefficient
        # J_pinv_damped = transpose(J) @ optas.inv(J @ transpose(J) + lambda_ * SX.eye(J.shape[0]))
        # v_eff = optas.DM([mu_hat[0], mu_hat[1], mu_hat[2], 0, 0, 0])
        # unit_dq_f = J_pinv_damped @ v_eff 


        J_pinv_damped = transpose(J[:3,:]) @ optas.inv(J[:3,:] @ transpose(J[:3,:]) + lambda_ * SX.eye(J[:3,:].shape[0]))
        v_eff = optas.DM([mu_hat[0], mu_hat[1], mu_hat[2]])
        unit_dq_f = J_pinv_damped @ v_eff 


        unit_dq_f = optas.DM(unit_dq_f)
        print("unitdq: ", unit_dq_f)

        print("***********************************")
        print("Casadi solution:")
        print("\t",solution_casadi[f"{self.robot_name}/q"] * (180.0 / optas.pi))
        print("end effector position: ", r_ee)
        print("manipulability score: ", mu_2)
        return solution_casadi[f"{self.robot_name}/q"], r_ee, mu_2, unit_dq_f
      
def CalcTrajParams(r_T, x_T):
    r_Tet = np.array(r_T)
    r_ee = np.array([x_T[0], x_T[1], x_T[2]])
    r_e2T = r_Tet - r_ee
    tmp = r_Tet - r_ee # vector from end effector to target
    tmp[2] = np.linalg.norm(tmp[0:2]) #modify z component to make the 45 deg angle 
    Z = r_e2T[2]
    X = tmp[0]
    Y = tmp[1]
    mu_hat = tmp/np.linalg.norm(tmp) # unit launch direction vector
    original_dir = np.array([0, 1, 0]) # align y axis with the direction
    v_0 = np.sqrt(9.81*np.linalg.norm(tmp[:2])/(2*(mu_hat[2]*np.linalg.norm(tmp[:2]) - Z*np.linalg.norm(mu_hat[:2]))*np.linalg.norm(mu_hat[:2])))
    # v_0 = np.sqrt(9.81*(X**2 + Y**2)/(np.linalg.norm(tmp[:2]) - Z))
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
    return rot.as_quat(), mu_hat, v_0

def main():
    cwd = pathlib.Path( __file__).parent.resolve()  # path to current working directory
    cwd = os.path.split(cwd)[0]
    urdf_filename = os.path.join(cwd, "robot", "g1", "g1_dual_arm.urdf")
    # Setup robot
    robot = optas.RobotModel(urdf_filename)
    joints = robot.urdf.joints
    joint_names = [jnt.name for jnt in joints]
    for name in joint_names:
        print(name)
    robot_name = robot.get_name()
    link_ee = "center_palm"  # end-effector link name

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
        r_wrist_y_jnt,
        ])
    vis = Visualizer()
    ### Test basic trajectory with previous soln as initial guess. Speeds up a significant amount if have good init guess
    # for i in range(10):
    #     soln = ik_solver.SolveIK([0.1 + i/20.0, 0, 0.2], theta_T, q_0)
    #     vis.robot(robot, soln, alpha=1.0)
    #     q_0 = soln
    r_T = [3.0, 0, 1]
    # soln1, soln1_ee, soln1_manip, dqf = throw_pose_finder.SolveIK(q_0, r_T)
    # calculate trajectory: 
    # soln1_ee = ((soln1_ee.full()).T)[0] # convert to numpy array 

    pelvis_z = 0.793
    z_offset = pelvis_z - 0.5
    r_T[2] = r_T[2] - z_offset
    soln, soln_ee, soln_manip, dqf = throw_pose_finder.SolveIK(optas.np.zeros(robot.ndof), r_T)
    soln_ee = ((soln_ee.full()).T)[0] # convert to numpy array 
    ## Get end configuration, end velocity
    quat_T, mu_hat, v_0 = CalcTrajParams(r_T, soln_ee)
    print("desired vel: ", mu_hat*v_0)
    qf = ((soln.full()).T)[0] # convert to numpy array 
    dqf = ((dqf.full()).T)[0] # convert to numpy array 
    dqf = dqf*v_0
    q_0 = qf - dqf/2.0*1.0
    dq0 = np.zeros(len(q_0))
    t = np.arange(0, 2.0, 0.02)
    plan_q = np.zeros((len(q_0), len(t)))
    for i in range(len(q_0)):
        plan_q[i,:] = Compute1PolyTraj(t, q_0[i], dq0[i], qf[i], dqf[i], 0, 1.0)

    soln = plan_q[:,0]
    vis.robot(robot, soln, alpha=0.2, show_links = False, link_axis_scale=0.2)
    soln = plan_q[:,10]
    vis.robot(robot, soln, alpha=0.4, show_links = False, link_axis_scale=0.2)
    soln = plan_q[:,20]
    vis.robot(robot, soln, alpha=0.6, show_links = False, link_axis_scale=0.2)
    soln = plan_q[:,30]
    vis.robot(robot, soln, alpha=0.8, show_links = False, link_axis_scale=0.2)
    soln = plan_q[:,40]
    vis.robot(robot, soln, alpha=1.0, show_links = False, link_axis_scale=0.2)
    vis.sphere(position=r_T, radius=0.05, alpha=1.0, rgb=[1, 0, 0])
    
    # ee_pos = robot.get_global_link_position(link_ee, soln1)
    # ee_rot = robot.get_global_link_rotation(link_ee, soln1)
    # y_axis = ee_rot@[0, 1, 0]
    # orientation = robot.get_global_link_rpy(link_ee, soln1)
    # vis.cylinder(
    #     radius=0.01,
    #     height=0.3,
    #     rgb=[0, 1, 0],
    #     alpha=1.0,
    #     position=ee_pos + 0.15*y_axis,
    #     orientation=orientation
    # )
    # vis.cylinder(
    #     radius=0.005,
    #     height=r_T[2],
    #     rgb=[1, 0, 0],
    #     alpha=0.8,
    #     position=[3, 0, r_T[2]/2.0],
    #     orientation=[1.57, 0, 0]
    # )
    # vis.text(msg="target", position=[3, 0.03, r_T[2]], scale=[0.008, 0.008, 0.008], rgb=[1, 1, 1])
    vis.grid_floor(rgb=[0.5, 0.5, 0.5])
    vis.start()

if __name__ == "__main__":
    main()


