# Python standard lib
import os
import sys
import pathlib

# OpTaS
import optas

from g1_ik import G1IK, CalcTrajParams
from optas.visualize import Visualizer
import numpy as np
import matplotlib.pyplot as plt


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

def main(gui=True):
    cwd = pathlib.Path(__file__).parent.resolve()  # path to current working directory

    urdf_filename = os.path.join(cwd, "robots", "g1", "g1_dual_arm.urdf")
    # Setup robot
    robot = optas.RobotModel(urdf_filename)
    robot_name = robot.get_name()
    link_ee = "right_wrist_yaw_link"  # end-effector link name

    ik_solver = G1IK(robot, link_ee)
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
    best_position = np.array([0, 0, 0])
    best_manip = 0.0
    best_q = np.zeros(17)
    prev_soln = q_0
    solve_num = 0
    x = 0.0
    box_dims = (0.0, 0.6)
    increment = 0.05
    width = int(np.rint((box_dims[1] - box_dims[0])/increment))
    height = int(np.rint((box_dims[1] - box_dims[0])/increment))
    print("width: ", type(width))
    print("height: ", type(height))
    solns = np.zeros((width, height))
    # for x in np.arange(0.2, 0.5, 0.01):
    for i, y in enumerate(np.arange(box_dims[0], box_dims[1], increment)):
        for j, z in enumerate(np.arange(box_dims[0], box_dims[1], increment)):
            x_T = [x, -y, z]
            r_targ = [2, 0, 1]
            quat_T, mu_hat, v_0 = CalcTrajParams(r_targ, x_T)
            soln, soln_ee, soln_manip = ik_solver.SolveIK(x_T, quat_T, prev_soln, r_targ)
            prev_soln = soln
            solns[i, j] = soln_manip
            if (soln_manip > best_manip):
                best_position = soln_ee
                best_q = soln
                best_manip = soln_manip
            solve_num += 1
            print("solve_num: ", solve_num)

    vis.robot(robot, best_q, alpha=0.75, show_links = True)
    print("\nbest position: ", best_position)
    print("best manip: ", best_manip)
    np.savetxt('solns.csv', solns, delimiter=',')
    plt.imshow(solns, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Value')
    plt.title('2D Heatmap')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    # using box from 0.2 to 0.5 in x, y, and z direction (negative y). r_targ = [2, 0, 1]
    # best end effector position is [0.21, -0.49, 0.2]
    # with manipulability metric of 0.621343
    # x_T = [0.15, -0.2, 0.2]  # target end-effector position in global frame
    # r_targ = [1, 1, 0]

    # quat_T, mu_hat, v_0 = CalcTrajParams(r_targ, x_T)
    # soln1, soln1_ee, soln1_manip = ik_solver.SolveIK(x_T, quat_T, q_0, r_targ)
    # r_targ = [0.5, -0.5, 0.5]
    # quat_T, mu_hat, v_0 = CalcTrajParams(r_targ, x_T)
    # soln2, soln2_ee, soln2_manip = ik_solver.SolveIK(x_T, quat_T, q_0, r_targ)
    # vis.robot(robot, soln1, alpha=0.15, show_links = True)
    # vis.robot(robot, soln2, alpha=0.75, )
    vis.grid_floor()
    vis.start()
if __name__ == "__main__":
    sys.exit(main())
