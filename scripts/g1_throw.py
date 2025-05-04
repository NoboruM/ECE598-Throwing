from joint_write import *
import os
import pathlib
import time
import copy
from g1_manip_search_v2 import G1ThrowSearch
from g1_ik import CalcTrajParams
import optas
from g1_throw_traj import DualG1Planner

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
        self.init_time = 5.0
        self.back_time = self.init_time + 3.0
        self.throw_time = 1.0
        self.traj_calculated = False
        # in order of: 
        waist_y_jnt = -90.0
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
        r_shoulder_r_jnt = -30.0
        r_shoulder_y_jnt = 0.0
        r_elbow_jnt = 120
        r_wrist_r_jnt = 90
        r_wrist_p_jnt = 0.0
        r_wrist_y_jnt = 0.0#-90.0
        self.q_0 = optas.np.deg2rad([waist_y_jnt, waist_r_jnt, waist_p_jnt, l_shoulder_p_jnt, l_shoulder_r_jnt, l_shoulder_y_jnt, l_elbow_jnt, l_wrist_r_jnt, l_wrist_p_jnt, l_wrist_y_jnt, r_shoulder_p_jnt, r_shoulder_r_jnt, r_shoulder_y_jnt, r_elbow_jnt, r_wrist_r_jnt, r_wrist_p_jnt, r_wrist_y_jnt])


        self.dual_g1_planner = DualG1Planner()
        cwd = pathlib.Path( __file__).parent.resolve()  # path to current working directory
        cwd = os.path.split(cwd)[0]
        urdf_filename = os.path.join(cwd, "robot", "g1", "g1_dual_arm.urdf")
        # Setup robot
        robot = optas.RobotModel(urdf_filename)
        link_ee = "right_wrist_yaw_link"
        self.ik_solver = G1ThrowSearch(robot, link_ee)

    def CalculateTrajectory(self):
        ##########################################################################
        r_T = [0.5, 0, 0]
        soln, soln_ee, soln_manip = self.ik_solver.SolveIK(optas.np.zeros(17), r_T)
        soln_ee = ((soln_ee.full()).T)[0] # convert to numpy array 
        ## Get end configuration, end velocity
        quat_T, mu_hat, v_0 = CalcTrajParams(r_T, soln_ee)
        print("calculated the trajectory")

        self.dual_g1_planner.reset(self.q_0, mu_hat*v_0, soln)
        self.plan_q, self.plan_dq = self.dual_g1_planner.plan()
        self.throw_time = self.dual_g1_planner.T_traj
        self.traj_calculated = True
        self.start_throw_t = self.time_
        ##########################################################################
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
                if (i >= 12 and i < 29):
                    a = self.joint_init_pos[i]
                    b = self.q_0[i-12]
                    self.joint_pos_cmd[i] = (b - a)/2.0*np.sin(np.pi*ratio - np.pi/2) + (a + b)/2.0
                else: 
                    self.joint_pos_cmd[i] = self.joint_init_pos[i]

            self.start_throw_t = self.time_
        elif (self.time_ < (self.start_throw_t + self.throw_time)):
            q = self.plan_q((self.time_ - self.start_throw_t)/1.0)
            dq = self.plan_dq((self.time_ - self.start_throw_t)/1.0)
            for i in range(len(self.joint_pos_cmd)):
                if (i >= 12 and i < 29):
                    self.joint_pos_cmd[i] = q[i-12]
                    self.joint_vel_cmd[i] = dq[i-12]
                else:
                    self.joint_pos_cmd[i] = self.joint_init_pos[i]
                    self.joint_vel_cmd[i] = 0.0

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
    controller.CalculateTrajectory()

    while True:        
        time.sleep(1)
        if controller.MujocoInterface.done: 
           print("Done!")
           sys.exit(-1)    