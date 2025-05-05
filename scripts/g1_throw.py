from joint_write import *
import os
import pathlib
import time
import copy
from g1_manip_search_v2 import G1ThrowSearch, CalcTrajParams
import optas
from g1_throw_traj import DualG1Planner
import matplotlib.pyplot as plt

class ThrowController:
    def __init__(self):
        self.time_ = 0.0
        self.control_dt_ = 0.02
        self.MujocoInterface = MujocoInterface()
        self.MujocoInterface.Init()
        self.MujocoInterface.Start()
        self.joint_pos_cmd = [0.0 for _ in range(41)]
        self.joint_vel_cmd = [0.0 for _ in range(41)]
        self.joint_init_pos = []
        self.init_time = 3.0
        self.back_time = self.init_time + 3.0
        self.throw_time = 1.0
        self.hand_release_t = 0.22
        self.traj_idx = 0
        self.traj_calculated = False
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
        self.q_0 = optas.np.deg2rad([waist_y_jnt, l_shoulder_p_jnt, l_shoulder_r_jnt, l_shoulder_y_jnt, l_elbow_jnt, l_wrist_r_jnt, l_wrist_p_jnt, l_wrist_y_jnt, r_shoulder_p_jnt, r_shoulder_r_jnt, r_shoulder_y_jnt, r_elbow_jnt, r_wrist_r_jnt, r_wrist_p_jnt, r_wrist_y_jnt])


        self.dual_g1_planner = DualG1Planner()
        cwd = pathlib.Path( __file__).parent.resolve()  # path to current working directory
        cwd = os.path.split(cwd)[0]
        urdf_filename = os.path.join(cwd, "robot", "g1", "g1_dual_arm.urdf")
        # Setup robot
        self.robot = optas.RobotModel(urdf_filename)
        link_ee = "right_wrist_yaw_link"
        self.ik_solver = G1ThrowSearch(self.robot, link_ee)

    def CalculateTrajectory(self):
        ##########################################################################
        r_T = [2.0, 0, 0]
        soln, soln_ee, soln_manip, dqf = self.ik_solver.SolveIK(optas.np.zeros(self.robot.ndof), r_T)
        soln_ee = ((soln_ee.full()).T)[0] # convert to numpy array 
        ## Get end configuration, end velocity
        quat_T, mu_hat, v_0 = CalcTrajParams(r_T, soln_ee)
        print("calculated the trajectory")
        print("v_0: ", v_0)
        print("dqf: ", dqf)

        v_ee = mu_hat*v_0
        self.dual_g1_planner.reset(self.q_0,  v_ee, soln, v_0*dqf)
        self.plan_q, self.plan_dq = self.dual_g1_planner.plan()
        self.throw_time = self.dual_g1_planner.T_traj
        self.traj_calculated = True
        self.start_throw_t = self.time_
        ##########################################################################

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

    def ComputePolyThrowTraj(self):
        pelvis_z = 0.793
        z_offset = pelvis_z - 0.5
        r_T = [3.0, 0, 0.945 - z_offset]
        soln, soln_ee, soln_manip, dqf = self.ik_solver.SolveIK(optas.np.zeros(self.robot.ndof), r_T)
        soln_ee = ((soln_ee.full()).T)[0] # convert to numpy array 
        ## Get end configuration, end velocity
        quat_T, mu_hat, v_0 = CalcTrajParams(r_T, soln_ee)

        qf = ((soln.full()).T)[0] # convert to numpy array 
        dqf = ((dqf.full()).T)[0] # convert to numpy array 
        dqf = dqf*v_0
        dq0 = np.zeros(len(self.q_0))
        t = np.arange(0, 2.0, self.control_dt_)
        self.plan_q = np.zeros((len(self.q_0), len(t)))
        for i in range(len(self.q_0)):
            self.plan_q[i,:] = self.Compute1PolyTraj(t, self.q_0[i], dq0[i], qf[i], dqf[i], 0, self.throw_time)
        self.start_throw_t = self.time_
        self.traj_calculated = True
        return self.plan_q
    
    def Init(self):
        self.joint_pos_cmd = [joint.q for joint in self.MujocoInterface.target_pos]
        self.joint_init_pos = copy.copy(self.joint_pos_cmd)

    def Start(self):
        self.ControllerThreadPtr = RecurrentThread(
            interval=self.control_dt_, target=self.Controller, name="control"
        )
        self.time_ = 0.0
        self.ControllerThreadPtr.Start()

    def Controller(self):
        self.time_ += self.control_dt_
        # calculate the joint commands
        if (not self.traj_calculated or (self.time_ < self.init_time)):
            ratio = np.clip(self.time_ / self.init_time, 0.0, 1.0)
            # move to throw init pose
            for i in range(len(self.joint_pos_cmd)):
                if (i >= 12 and i < (12 + len(self.q_0) + 2)) and (i != 13) and (i != 14):
                    if (i > 14):
                        a = self.joint_init_pos[i]
                        b = self.q_0[i-14]
                        self.joint_pos_cmd[i] = (b - a)/2.0*np.sin(np.pi*ratio - np.pi/2) + (a + b)/2.0
                    else:
                        a = self.joint_init_pos[i]
                        b = self.q_0[i-12]
                        self.joint_pos_cmd[i] = (b - a)/2.0*np.sin(np.pi*ratio - np.pi/2) + (a + b)/2.0
                else: 
                    self.joint_pos_cmd[i] = self.joint_init_pos[i]

            self.start_throw_t = self.time_
        elif (self.time_ < (self.start_throw_t + self.throw_time)):
            # q = self.plan_q((self.time_ - self.start_throw_t)/1.0)
            q = self.plan_q[:,self.traj_idx]
            print(q)
            # dq = self.plan_dq((self.time_ - self.start_throw_t)/1.0)
            for i in range(len(self.joint_pos_cmd)):
                if (i >= 12 and i < (12 + len(self.q_0) + 2)) and (i != 13) and (i != 14):
                    if (i > 14):
                        self.joint_pos_cmd[i] = q[i-14]
                        # self.joint_vel_cmd[i] = dq[i-14]
                    else:
                        self.joint_pos_cmd[i] = q[i-12]
                        # self.joint_vel_cmd[i] = dq[i-12]
                elif (i >= 29 and i < 35):
                    if (self.time_ > (self.start_throw_t + self.throw_time - self.hand_release_t)):
                        self.joint_pos_cmd[i] = 0.0
                    else:
                        self.joint_pos_cmd[i] = self.joint_init_pos[i]
                else:
                    self.joint_pos_cmd[i] = self.joint_init_pos[i]
                    self.joint_vel_cmd[i] = 0.0
            self.traj_idx += 1
        else:
            controller.MujocoInterface.done = False
        # store the target joint positions
        for i in range(41):
            self.MujocoInterface.target_pos[i].q = self.joint_pos_cmd[i]
            self.MujocoInterface.target_pos[i].dq = self.joint_vel_cmd[i]


if __name__ == '__main__':

    # input("Press Enter to continue...")

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(1, "lo")

    controller = ThrowController()
    controller.Init()
    controller.Start()
    controller.ComputePolyThrowTraj()
    # controller.CalculateTrajectory()

    while True:        
        time.sleep(1)
        if controller.MujocoInterface.done: 
           print("Done!")
           sys.exit(-1)    