import time
import mujoco
import mujoco.viewer
import cv2
from threading import Thread, Lock
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand
import config

# Add EGL configuration for headless rendering
import os
os.environ["MUJOCO_GL"] = "egl"  # Force EGL backend

locker = Lock()
mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)
desired_qpos_values = np.zeros(56)
init_finger_angle = 0.5
desired_hand_pos = np.array([
    1.4*init_finger_angle,
    1.4*1.05851325*init_finger_angle +0.72349796,
    1.4*init_finger_angle,
    1.4*1.05851325*init_finger_angle +0.72349796,
    1.4*init_finger_angle,
    1.4*1.05851325*init_finger_angle +0.72349796,
    1.5*init_finger_angle,
    1.5*1.05851325*init_finger_angle +0.72349796,
    -1.8,
    0.0
])

desired_qpos_values[49:] = np.array([0.335, -0.105, 0.9, 0, 0, 0, 1])# ball pose
desired_qpos_values[39:49] = desired_hand_pos
joint_names = [mj_model.joint(i).name for i in range(mj_model.njnt)]
for i, jnt in enumerate(joint_names):
    if (i < 49):
        print("jnt: {}, {}".format(jnt, i))
    else:
        print("jnt: {}, {}-{}".format(jnt, i, i+7))
print('\n'*10)
print("qpos LENGTH: ", len(mj_data.qpos))
print("ctrl LENGTH: ", len(mj_data.ctrl))
print('\n'*10)
for i in range(0, 56):
    mj_data.qpos[i] = desired_qpos_values[i]
mujoco.mj_forward(mj_model, mj_data)    # Propagates initial state

# set initial joint torques on right arm to keep zero pose:
mj_data.ctrl[22:29] = [-2.5, -0.209, 0.0, -2.9, -0.0128, -1.2, -0.00198]
# set initial joint torques on the right hand to grasp the ball
for i in range(29, 35):
    mj_data.ctrl[i] = 0.1
mj_data.ctrl[33] = -0.1

if config.ENABLE_ELASTIC_BAND:
    elastic_band = ElasticBand()
# Initialize renderer in main thread
if config.USE_CAMERA:
    renderer = mujoco.Renderer(mj_model, height=480, width=640)
band_attached_link = mj_model.body("torso_link").id
# Viewer must be created in main thread
viewer = mujoco.viewer.launch_passive(
    mj_model, mj_data,
    key_callback=elastic_band.MujuocoKeyCallback
)

def SimulationThread():
    global mj_data
    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)
    
    while viewer.is_running():
        step_start = time.perf_counter()
        with locker:  # Use context manager for safer locking
            if elastic_band.enable:
                mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3]
                )
            mujoco.mj_step(mj_model, mj_data)
        time_until_next_step = mj_model.opt.timestep - (time.perf_counter() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

def PhysicsViewerThread():
    while viewer.is_running():
        with locker:
            viewer.sync()
            if config.USE_CAMERA:
                renderer.update_scene(mj_data, camera="rgb_cam")
                rgb_image = renderer.render()
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # Convert to BGR
        if config.USE_CAMERA:
            # OpenCV operations in main thread
            cv2.imshow('RGB Camera View', rgb_image)
            cv2.waitKey(1)
        
        time.sleep(config.VIEWER_DT)

if __name__ == "__main__":
    # Run viewer in main thread
    sim_thread = Thread(target=SimulationThread)
    sim_thread.start()
    
    # Handle rendering in main thread
    PhysicsViewerThread()
    sim_thread.join()
    if config.USE_CAMERA:
        cv2.destroyAllWindows()
