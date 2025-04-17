import time
import sys

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

import numpy as np

kPi = 3.141592654
kPi_2 = 1.57079632

class G1JointIndex:
    # Left leg
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5

    # Right leg
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11

    WaistYaw = 12
    WaistRoll = 13        # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistA = 13           # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistPitch = 14       # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistB = 14           # NOTE: INVALID for g1 23dof/29dof with waist locked

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

    kNotUsedJoint = 29 # NOTE: Weight

    # Left hand
    LeftHandIndex = 35
    LeftHandMiddle = 36
    LeftHandRing = 37
    LeftHandPinky = 38
    LeftHandThumb1 = 39
    LeftHandThumb2 = 40
    # Right hand
    RightHandIndex = 29
    RightHandMiddle = 30
    RightHandRing = 31
    RightHandPinky = 32
    RightHandThumb1 = 33
    RightHandThumb2 = 34

class Custom:
    def __init__(self):
        self.time_ = 0.0
        self.control_dt_ = 0.02  
        self.duration_ = 3.0   
        self.counter_ = 0
        self.weight = 0.
        self.weight_rate = 0.2
        self.arm_kp = 40 # 50
        self.arm_kd = 0.5# 1.5
        self.hand_kp = 0.3
        self.hand_kd = 0.001
        self.leg_kp = 40 
        self.leg_kd = 0.5
        self.torso_kp = 40 
        self.torso_kd = 0.5
        self.dq = 0.
        self.tau_ff = 0.
        self.mode_machine_ = 0
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()  
        self.low_state = None 
        self.first_update_low_state = False
        self.crc = CRC()
        self.done = False


        self.arm_joints = [
          G1JointIndex.LeftShoulderPitch,  G1JointIndex.LeftShoulderRoll,
          G1JointIndex.LeftShoulderYaw,    G1JointIndex.LeftElbow,
          G1JointIndex.LeftWristRoll,      G1JointIndex.LeftWristPitch,
          G1JointIndex.LeftWristYaw,
          G1JointIndex.RightShoulderPitch, G1JointIndex.RightShoulderRoll,
          G1JointIndex.RightShoulderYaw,   G1JointIndex.RightElbow,
          G1JointIndex.RightWristRoll,     G1JointIndex.RightWristPitch,
          G1JointIndex.RightWristYaw,
          G1JointIndex.WaistYaw,
          G1JointIndex.WaistRoll,
          G1JointIndex.WaistPitch
        ]
        self.hand_joints = [
            G1JointIndex.LeftHandIndex, 
            G1JointIndex.LeftHandMiddle, 
            G1JointIndex.LeftHandRing, 
            G1JointIndex.LeftHandPinky, 
            G1JointIndex.LeftHandThumb1, 
            G1JointIndex.LeftHandThumb2, 
            G1JointIndex.RightHandIndex, 
            G1JointIndex.RightHandMiddle, 
            G1JointIndex.RightHandRing, 
            G1JointIndex.RightHandPinky, 
            G1JointIndex.RightHandThumb1, 
            G1JointIndex.RightHandThumb2
        ]
        self.leg_joints = [
            G1JointIndex.LeftHipPitch,
            G1JointIndex.LeftHipRoll,
            G1JointIndex.LeftHipYaw,
            G1JointIndex.LeftKnee,
            G1JointIndex.LeftAnklePitch,
            G1JointIndex.LeftAnkleRoll,
            G1JointIndex.RightHipPitch,
            G1JointIndex.RightHipRoll,
            G1JointIndex.RightHipYaw,
            G1JointIndex.RightKnee,
            G1JointIndex.RightAnklePitch,
            G1JointIndex.RightAnkleRoll
        ]
        self.torso_joints = [
            G1JointIndex.WaistYaw,
            G1JointIndex.WaistRoll,
            G1JointIndex.WaistPitch
        ]

        self.arm_target_pos = [0 for _ in range(len(self.arm_joints))]
        self.leg_target_pos = [0 for _ in range(len(self.leg_joints))]
        self.hand_target_pos = [0 for _ in range(len(self.hand_joints))]
        self.torso_target_pos = [0 for _ in range(len(self.torso_joints))]

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
            # TODO: need to set the initial target pose to the actual position instead of zero
            self.arm_target_pos = [0 for _ in range(len(self.arm_joints))]
            self.leg_target_pos = [0 for _ in range(len(self.leg_joints))]
            self.hand_target_pos = [0 for _ in range(len(self.hand_joints))]
            self.torso_target_pos = [0 for _ in range(len(self.torso_joints))]
            self.first_update_low_state = True
        
    def LowCmdWrite(self):
        self.time_ += self.control_dt_
        for i,joint in enumerate(self.arm_joints):
            self.low_cmd.motor_cmd[joint].tau = 0. 
            self.low_cmd.motor_cmd[joint].q = self.arm_target_pos[i]
            self.low_cmd.motor_cmd[joint].dq = 0. 
            self.low_cmd.motor_cmd[joint].kp = self.arm_kp 
            self.low_cmd.motor_cmd[joint].kd = self.arm_kd
        for i, joint in enumerate(self.hand_joints):
            self.low_cmd.motor_cmd[joint].tau = 0. 
            self.low_cmd.motor_cmd[joint].q = self.hand_target_pos[i]
            self.low_cmd.motor_cmd[joint].dq = 0. 
            self.low_cmd.motor_cmd[joint].kp = self.hand_kp
            self.low_cmd.motor_cmd[joint].kd = self.hand_kd
        for i, joint in enumerate(self.leg_joints):
            self.low_cmd.motor_cmd[joint].tau = 0. 
            self.low_cmd.motor_cmd[joint].q = self.leg_target_pos[i]
            self.low_cmd.motor_cmd[joint].dq = 0. 
            self.low_cmd.motor_cmd[joint].kp = self.leg_kp 
            self.low_cmd.motor_cmd[joint].kd = self.leg_kd

        for i, joint in enumerate(self.torso_joints):
            self.low_cmd.motor_cmd[joint].tau = 0. 
            self.low_cmd.motor_cmd[joint].q = self.torso_target_pos[i]
            self.low_cmd.motor_cmd[joint].dq = 0. 
            self.low_cmd.motor_cmd[joint].kp = self.torso_kp 
            self.low_cmd.motor_cmd[joint].kd = self.torso_kd

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.arm_sdk_publisher.Write(self.low_cmd)

if __name__ == '__main__':

    input("Press Enter to continue...")

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(1, "lo")

    custom = Custom()
    custom.Init()
    custom.Start()

    while True:        
        time.sleep(1)
        if custom.done: 
           print("Done!")
           sys.exit(-1)    