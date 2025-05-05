import time
import sys

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import MotorCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

import numpy as np

kPi = 3.141592654
kPi_2 = 1.57079632

class G1JntIdx:
    WaistYaw = 12
    WaistRoll = 13        # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistPitch = 14       # NOTE: INVALID for g1 23dof/29dof with waist locked

    # Left arm
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20   # NOTE: INVALID for g1 23dof
    LeftWristYaw = 21     # NOTE: INVALID for g1 23dof

    # Right arm
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27  # NOTE: INVALID for g1 23dof
    RightWristYaw = 28    # NOTE: INVALID for g1 23dof

    # Right hand
    RightHandIndex = 29
    RightHandMiddle = 30
    RightHandRing = 31
    RightHandPinky = 32
    RightHandThumb1 = 33
    RightHandThumb2 = 34

    # Left hand
    LeftHandIndex = 35
    LeftHandMiddle = 36
    LeftHandRing = 37
    LeftHandPinky = 38
    LeftHandThumb1 = 39
    LeftHandThumb2 = 40

class MujocoInterface:
    def __init__(self):
        self.time_ = 0.0
        self.control_dt_ = 0.02  
        self.duration_ = 3.0   
        self.counter_ = 0
        self.weight = 0.
        self.weight_rate = 0.2
        self.arm_kp = 40 # 50
        self.arm_kd = 1.0# 1.5
        self.hand_kp = 40.0
        self.hand_kd = 0.4
        self.leg_kp = 40 
        self.leg_kd = 0.5
        self.torso_kp = 40 
        self.torso_kd = 1.0
        self.dq = 0.
        self.tau_ff = 0.
        self.mode_machine_ = 0
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()  
        self.low_state = None 
        self.first_update_low_state = False
        self.crc = CRC()
        self.done = False

        self.target_pos = [
            MotorCmd_(G1JntIdx.WaistYaw, 0.0, 0.0, 0.0, 60, self.torso_kd, 0),
            MotorCmd_(G1JntIdx.WaistRoll, 0.0, 0.0, 0.0, 40, self.torso_kd, 0),
            MotorCmd_(G1JntIdx.WaistPitch, 0.0, 0.0, 0.0, 40, self.torso_kd, 0),
            MotorCmd_(G1JntIdx.LeftShoulderPitch, 0.0, 0.0, 0.0, self.arm_kp, self.arm_kd, 0),
            MotorCmd_(G1JntIdx.LeftShoulderRoll, 0.0, 0.0, 0.0, self.arm_kp, self.arm_kd, 0),
            MotorCmd_(G1JntIdx.LeftShoulderYaw, 0.0, 0.0, 0.0, self.arm_kp, self.arm_kd, 0),
            MotorCmd_(G1JntIdx.LeftElbow, 0.0, 0.0, 0.0, self.arm_kp, self.arm_kd, 0),
            MotorCmd_(G1JntIdx.LeftWristRoll, 0.0, 0.0, 0.0, self.arm_kp, self.arm_kd, 0),
            MotorCmd_(G1JntIdx.LeftWristPitch, 0.0, 0.0, 0.0, self.arm_kp, self.arm_kd, 0), 
            MotorCmd_(G1JntIdx.LeftWristYaw, 0.0, 0.0, 0.0, self.arm_kp, self.arm_kd, 0),
            MotorCmd_(G1JntIdx.RightShoulderPitch, 0.0, 0.0, 0.0, self.arm_kp, self.arm_kd, 0),
            MotorCmd_(G1JntIdx.RightShoulderRoll, 0.0, 0.0, 0.0, self.arm_kp, self.arm_kd, 0),
            MotorCmd_(G1JntIdx.RightShoulderYaw, 0.0, 0.0, 0.0, self.arm_kp, self.arm_kd, 0),
            MotorCmd_(G1JntIdx.RightElbow, 0.0, 0.0, 0.0, self.arm_kp, self.arm_kd, 0),
            MotorCmd_(G1JntIdx.RightWristRoll, 0.0, 0.0, 0.0, self.arm_kp, self.arm_kd, 0),
            MotorCmd_(G1JntIdx.RightWristPitch, 0.0, 0.0, 0.0, self.arm_kp, self.arm_kd, 0),
            MotorCmd_(G1JntIdx.RightWristYaw, 0.0, 0.0, 0.0, self.arm_kp, self.arm_kd, 0),
            MotorCmd_(G1JntIdx.RightHandIndex, 0.0, 0.0, 0.0, self.hand_kp, self.hand_kd, 0),
            MotorCmd_(G1JntIdx.RightHandMiddle, 0.0, 0.0, 0.0, self.hand_kp, self.hand_kd, 0),
            MotorCmd_(G1JntIdx.RightHandRing, 0.0, 0.0, 0.0, self.hand_kp, self.hand_kd, 0),
            MotorCmd_(G1JntIdx.RightHandPinky, 0.0, 0.0, 0.0, self.hand_kp, self.hand_kd, 0),
            MotorCmd_(G1JntIdx.RightHandThumb1, 0.0, 0.0, 0.0, self.hand_kp, self.hand_kd, 0),
            MotorCmd_(G1JntIdx.RightHandThumb2, 0.0, 0.0, 0.0, self.hand_kp, self.hand_kd, 0),
            MotorCmd_(G1JntIdx.LeftHandIndex, 0.0, 0.0, 0.0, self.hand_kp, self.hand_kd, 0),
            MotorCmd_(G1JntIdx.LeftHandMiddle, 0.0, 0.0, 0.0, self.hand_kp, self.hand_kd, 0),
            MotorCmd_(G1JntIdx.LeftHandRing, 0.0, 0.0, 0.0, self.hand_kp, self.hand_kd, 0),
            MotorCmd_(G1JntIdx.LeftHandPinky, 0.0, 0.0, 0.0, self.hand_kp, self.hand_kd, 0),
            MotorCmd_(G1JntIdx.LeftHandThumb1, 0.0, 0.0, 0.0, self.hand_kp, self.hand_kd, 0),
            MotorCmd_(G1JntIdx.LeftHandThumb2, 0.0, 0.0, 0.0, self.hand_kp, self.hand_kd, 0)
            ]
        print("target_pose length: ", len(self.target_pos))
    def Init(self):
        # create publisher #
        self.arm_sdk_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.arm_sdk_publisher.Init()

        # create subscriber # 
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

    def Start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=self.control_dt_, target=self.LowCmdWrite, name="control"
        )
        while self.first_update_low_state == False:
            time.sleep(1)

        if self.first_update_low_state == True:
            self.lowCmdWriteThreadPtr.Start()

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg
        if self.first_update_low_state == False:
            for i in range(len(self.target_pos)):
                self.target_pos[i].q = msg.motor_state[i].q
            self.first_update_low_state = True
        
    def LowCmdWrite(self):
        self.time_ += self.control_dt_
        for i, joint in enumerate(self.target_pos):
            self.low_cmd.motor_cmd[joint.mode].tau = self.target_pos[i].tau
            self.low_cmd.motor_cmd[joint.mode].q = self.target_pos[i].q
            self.low_cmd.motor_cmd[joint.mode].dq = self.target_pos[i].dq
            if (self.target_pos[i].kP != )
            self.low_cmd.motor_cmd[joint.mode].kp = self.target_pos[i].kp
            self.low_cmd.motor_cmd[joint.mode].kd = self.target_pos[i].kd

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.arm_sdk_publisher.Write(self.low_cmd)

if __name__ == '__main__':

    input("Press Enter to continue...")

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(1, "lo")

    MujocoInterface = MujocoInterface()
    MujocoInterface.Init()
    MujocoInterface.Start()

    while True:        
        time.sleep(1)
        if MujocoInterface.done: 
           print("Done!")
           sys.exit(-1)    