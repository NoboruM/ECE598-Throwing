from joint_write import *
import time
import copy

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
        self.init_time = 2.0
        self.back_time = self.init_time + 3.0
        self.throw_time = 1.0

    def Init(self):
        self.joint_pos_cmd = [joint.q for joint in self.MujocoInterface.target_pos]
        self.joint_init_pos = copy.copy(self.joint_pos_cmd)

    def Start(self):
        self.ControllerThreadPtr = RecurrentThread(
            interval=self.control_dt_, target=self.Controller, name="control"
        )
        self.ControllerThreadPtr.Start()
    def Controller(self):
        self.time_ += self.control_dt_
        # calculate the joint commands
        if (self.time_ < self.init_time):
            # move to initial pose
            ratio = np.clip(self.time_ / self.init_time, 0.0, 1.0)
            self.joint_pos_cmd[G1JointIndex.RightWristRoll] = self.joint_init_pos[G1JointIndex.RightWristRoll] + np.pi/2.0*ratio
            self.joint_pos_cmd[G1JointIndex.RightShoulderRoll] = self.joint_init_pos[G1JointIndex.RightShoulderRoll] - np.pi/8.0*ratio
            self.joint_pos_cmd[G1JointIndex.RightElbow] = self.joint_init_pos[G1JointIndex.RightElbow] + (np.pi/1.9)*ratio
            self.joint_pos_cmd[G1JointIndex.RightHandIndex] = self.joint_init_pos[G1JointIndex.RightHandIndex]
            self.joint_pos_cmd[G1JointIndex.RightHandMiddle] = self.joint_init_pos[G1JointIndex.RightHandMiddle]
            self.joint_pos_cmd[G1JointIndex.RightHandRing] = self.joint_init_pos[G1JointIndex.RightHandRing]
            self.joint_pos_cmd[G1JointIndex.RightHandPinky] = self.joint_init_pos[G1JointIndex.RightHandPinky]
            self.joint_pos_cmd[G1JointIndex.RightHandThumb1] = self.joint_init_pos[G1JointIndex.RightHandThumb1]
            self.joint_pos_cmd[G1JointIndex.RightHandThumb2] = self.joint_init_pos[G1JointIndex.RightHandThumb2]

        elif (self.time_ < 2*self.init_time):
            ratio = np.clip((self.time_-self.init_time) / (self.init_time), 0.0, 1.0)
            self.joint_pos_cmd[G1JointIndex.RightShoulderPitch] = self.joint_init_pos[G1JointIndex.RightShoulderPitch] + np.pi/4.0*(np.sin(ratio*np.pi - np.pi/2.0) + 1)/2.0
            self.joint_vel_cmd[G1JointIndex.RightShoulderPitch] = np.pi/8.0*np.cos(ratio*np.pi - np.pi/2.0)
            self.joint_pos_cmd[G1JointIndex.RightHandIndex] = self.joint_init_pos[G1JointIndex.RightHandIndex]
            self.joint_pos_cmd[G1JointIndex.RightHandMiddle] = self.joint_init_pos[G1JointIndex.RightHandMiddle]
            self.joint_pos_cmd[G1JointIndex.RightHandRing] = self.joint_init_pos[G1JointIndex.RightHandRing]
            self.joint_pos_cmd[G1JointIndex.RightHandPinky] = self.joint_init_pos[G1JointIndex.RightHandPinky]
            self.joint_pos_cmd[G1JointIndex.RightHandThumb1] = self.joint_init_pos[G1JointIndex.RightHandThumb1]
            self.joint_pos_cmd[G1JointIndex.RightHandThumb2] = self.joint_init_pos[G1JointIndex.RightHandThumb2]
        elif (self.time_ < (2*self.init_time + self.throw_time)):
            ratio = np.clip((self.time_- 2*self.init_time) / (self.throw_time), 0.0, 1.0)
            self.joint_pos_cmd[G1JointIndex.RightShoulderPitch] = self.joint_init_pos[G1JointIndex.RightShoulderPitch] + np.pi/4.0 - np.pi*3/4*(np.sin(ratio*np.pi - np.pi/2.0) + 1)/2.0
            self.joint_vel_cmd[G1JointIndex.RightShoulderPitch] = -np.pi*3/2.0*np.cos(ratio*np.pi - np.pi/2.0)
            self.joint_pos_cmd[G1JointIndex.RightHandThumb1] = self.joint_init_pos[G1JointIndex.RightHandThumb1]
            self.joint_pos_cmd[G1JointIndex.RightHandThumb2] = self.joint_init_pos[G1JointIndex.RightHandThumb2]
            if (self.time_ > (2*self.init_time + self.throw_time*8.0/10.0)):
                self.joint_pos_cmd[G1JointIndex.RightHandIndex] = 0.0
                self.joint_pos_cmd[G1JointIndex.RightHandMiddle] = 0.0
                self.joint_pos_cmd[G1JointIndex.RightHandRing] = 0.0
                self.joint_pos_cmd[G1JointIndex.RightHandPinky] = 0.0
            else:
                self.joint_pos_cmd[G1JointIndex.RightHandIndex] = self.joint_init_pos[G1JointIndex.RightHandIndex]
                self.joint_pos_cmd[G1JointIndex.RightHandMiddle] = self.joint_init_pos[G1JointIndex.RightHandMiddle]
                self.joint_pos_cmd[G1JointIndex.RightHandRing] = self.joint_init_pos[G1JointIndex.RightHandRing]
                self.joint_pos_cmd[G1JointIndex.RightHandPinky] = self.joint_init_pos[G1JointIndex.RightHandPinky]
        else:
            controller.MujocoInterface.done = False
        # store the target joint positions
        for i in range(41):
            self.MujocoInterface.target_pos[i].q = self.joint_pos_cmd[i]
            self.MujocoInterface.target_pos[i].dq = self.joint_vel_cmd[i]


if __name__ == '__main__':

    input("Press Enter to continue...")

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(1, "lo")
    controller = ThrowController()
    controller.Init()
    controller.Start()

    while True:        
        time.sleep(1)
        if controller.MujocoInterface.done: 
           print("Done!")
           sys.exit(-1)    