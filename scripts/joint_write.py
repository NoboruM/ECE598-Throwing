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
import time
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
        self.start_time = time.time()
        self.time_ = 0.0
        self.control_dt_ = 0.02  
        self.duration_ = 3.0   
        self.counter_ = 0
        self.weight = 0.
        self.weight_rate = 0.2
        self.shoulder_kp = 125 
        self.shoulder_kd = 2.0
        self.hand_kp = 40.0
        self.hand_kd = 0.4
        self.leg_kp = 100
        self.leg_kd = 0.5
        self.torso_kp = 300 
        self.torso_kd = 5.0
        self.dq = 0.
        self.tau_ff = 0.
        self.mode_machine_ = 0
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()  
        self.low_state = None 
        self.first_update_low_state = False
        self.crc = CRC()
        self.done = False
        self.measured_file = open("measured_joints.csv", 'w')
        self.commanded_file = open("commanded_joints.csv", 'w')
        header_read = ','.join(['time_'] + [f'read/joint_{i+1}' for i in range(41)])
        header_cmd = ','.join(['time_'] + [f'cmd/joint_{i+1}' for i in range(41)])
        self.measured_file.write(f"{header_read}\n")
        self.commanded_file.write(f"{header_cmd}\n")
        self.data_meas = np.zeros(41)
        self.data_cmd = np.zeros(41)
        self.target_pos = [
            MotorCmd_(G1JointIndex.LeftHipPitch, 0.0, 0.0, 0.0, 60, 1.0, 0),
            MotorCmd_(G1JointIndex.LeftHipRoll, 0.0, 0.0, 0.0, 60, 1.0, 0),
            MotorCmd_(G1JointIndex.LeftHipYaw, 0.0, 0.0, 0.0, 60, 1.0, 0),
            MotorCmd_(G1JointIndex.LeftKnee, 0.0, 0.0, 0.0, 100, 2.0, 0),
            MotorCmd_(G1JointIndex.LeftAnklePitch, 0.0, 0.0, 0.0, 40, 1.0, 0),
            MotorCmd_(G1JointIndex.LeftAnkleRoll, 0.0, 0.0, 0.0, 40, 1.0, 0),
            MotorCmd_(G1JointIndex.RightHipPitch, 0.0, 0.0, 0.0, 60, 1.0, 0),
            MotorCmd_(G1JointIndex.RightHipRoll, 0.0, 0.0, 0.0, 60, 1.0, 0),
            MotorCmd_(G1JointIndex.RightHipYaw, 0.0, 0.0, 0.0, 60, 1.0, 0),
            MotorCmd_(G1JointIndex.RightKnee, 0.0, 0.0, 0.0, 100, 2.0, 0),
            MotorCmd_(G1JointIndex.RightAnklePitch, 0.0, 0.0, 0.0, 40, 1.0, 0),
            MotorCmd_(G1JointIndex.RightAnkleRoll, 0.0, 0.0, 0.0, 40, 1.0, 0),
            MotorCmd_(G1JointIndex.WaistYaw, 0.0, 0.0, 0.0, self.torso_kp, self.torso_kd, 0),
            MotorCmd_(G1JointIndex.WaistRoll, 0.0, 0.0, 0.0, self.torso_kp, self.torso_kd, 0),
            MotorCmd_(G1JointIndex.WaistPitch, 0.0, 0.0, 0.0, self.torso_kp, self.torso_kd, 0),
            MotorCmd_(G1JointIndex.LeftShoulderPitch, 0.0, 0.0, 0.0, self.shoulder_kp, self.shoulder_kd, 0),
            MotorCmd_(G1JointIndex.LeftShoulderRoll, 0.0, 0.0, 0.0, self.shoulder_kp, self.shoulder_kd, 0),
            MotorCmd_(G1JointIndex.LeftShoulderYaw, 0.0, 0.0, 0.0, self.shoulder_kp, self.shoulder_kd, 0),
            MotorCmd_(G1JointIndex.LeftElbow, 0.0, 0.0, 0.0, 100, 1.0, 0),
            MotorCmd_(G1JointIndex.LeftWristRoll, 0.0, 0.0, 0.0, 10, 0.5, 0),
            MotorCmd_(G1JointIndex.LeftWristPitch, 0.0, 0.0, 0.0, 10, 0.5, 0), 
            MotorCmd_(G1JointIndex.LeftWristYaw, 0.0, 0.0, 0.0, 5, 0.5, 0),
            MotorCmd_(G1JointIndex.RightShoulderPitch, 0.0, 0.0, 0.0, self.shoulder_kp, self.shoulder_kd, 0),
            MotorCmd_(G1JointIndex.RightShoulderRoll, 0.0, 0.0, 0.0, self.shoulder_kp, self.shoulder_kd, 0),
            MotorCmd_(G1JointIndex.RightShoulderYaw, 0.0, 0.0, 0.0, self.shoulder_kp, self.shoulder_kd, 0),
            MotorCmd_(G1JointIndex.RightElbow, 0.0, 0.0, 0.0, 100, 1.0, 0),
            MotorCmd_(G1JointIndex.RightWristRoll, 0.0, 0.0, 0.0, 10, 0.5, 0),
            MotorCmd_(G1JointIndex.RightWristPitch, 0.0, 0.0, 0.0, 10, 0.5, 0),
            MotorCmd_(G1JointIndex.RightWristYaw, 0.0, 0.0, 0.0, 5, 0.5, 0),
            MotorCmd_(G1JointIndex.RightHandIndex, 0.0, 0.0, 0.0, self.hand_kp, self.hand_kd, 0),
            MotorCmd_(G1JointIndex.RightHandMiddle, 0.0, 0.0, 0.0, self.hand_kp, self.hand_kd, 0),
            MotorCmd_(G1JointIndex.RightHandRing, 0.0, 0.0, 0.0, self.hand_kp, self.hand_kd, 0),
            MotorCmd_(G1JointIndex.RightHandPinky, 0.0, 0.0, 0.0, self.hand_kp, self.hand_kd, 0),
            MotorCmd_(G1JointIndex.RightHandThumb1, 0.0, 0.0, 0.0, self.hand_kp, self.hand_kd, 0),
            MotorCmd_(G1JointIndex.RightHandThumb2, 0.0, 0.0, 0.0, self.hand_kp, self.hand_kd, 0),
            MotorCmd_(G1JointIndex.LeftHandIndex, 0.0, 0.0, 0.0, self.hand_kp, self.hand_kd, 0),
            MotorCmd_(G1JointIndex.LeftHandMiddle, 0.0, 0.0, 0.0, self.hand_kp, self.hand_kd, 0),
            MotorCmd_(G1JointIndex.LeftHandRing, 0.0, 0.0, 0.0, self.hand_kp, self.hand_kd, 0),
            MotorCmd_(G1JointIndex.LeftHandPinky, 0.0, 0.0, 0.0, self.hand_kp, self.hand_kd, 0),
            MotorCmd_(G1JointIndex.LeftHandThumb1, 0.0, 0.0, 0.0, self.hand_kp, self.hand_kd, 0),
            MotorCmd_(G1JointIndex.LeftHandThumb2, 0.0, 0.0, 0.0, self.hand_kp, self.hand_kd, 0)
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
            # TODO: need to set the initial target pose to the actual position instead of zero
            for i in range(len(self.target_pos)):
                self.target_pos[i].q = msg.motor_state[i].q
            self.first_update_low_state = True
        
        for i in range(len(self.target_pos)):
            self.data_meas[i] = msg.motor_state[i].q
        self.write_row(self.measured_file, time.time()- self.start_time, self.data_meas)
        
    def write_row(self, file, timestamp, positions):
        row = ','.join([f"{timestamp:.6f}"] + [f"{pos:.6f}" for pos in positions])
        file.write(f"{row}\n")

    def LowCmdWrite(self):
        self.time_ += self.control_dt_
        for i, joint in enumerate(self.target_pos):
            self.low_cmd.motor_cmd[joint.mode].tau = self.target_pos[i].tau
            self.low_cmd.motor_cmd[joint.mode].q = self.target_pos[i].q
            self.low_cmd.motor_cmd[joint.mode].dq = self.target_pos[i].dq
            if (self.target_pos[i].kp != self.hand_kp):
                self.low_cmd.motor_cmd[joint.mode].kp = self.target_pos[i].kp
                self.low_cmd.motor_cmd[joint.mode].kd = self.target_pos[i].kd

            self.data_cmd[i] = self.target_pos[i].q
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.arm_sdk_publisher.Write(self.low_cmd)
        self.write_row(self.commanded_file, time.time()- self.start_time, self.data_cmd)
        

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