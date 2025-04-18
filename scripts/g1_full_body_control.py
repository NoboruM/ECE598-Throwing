from joint_write import *

class ThrowController:
    def __init__(self):
        self.time_ = 0.0
        self.control_dt_ = 0.02
        self.MujocoInterface = MujocoInterface()
        self.MujocoInterface.Init()
        self.MujocoInterface.Start()
        self.joint_pos_cmd = [0.0 for _ in range(41)]
        self.joint_vel_cmd = [0.0 for _ in range(41)]
    def Init(self):
        self.joint_pos_cmd = [joint.q for joint in self.MujocoInterface.target_pos]

    def Start(self):
        self.ControllerThreadPtr = RecurrentThread(
            interval=self.control_dt_, target=self.Controller, name="control"
        )
        self.ControllerThreadPtr.Start()
    def Controller(self):
        self.time_ += self.control_dt_
        # calculate the joint commands

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