import time
import mujoco
import mujoco.viewer
import cv2
from threading import Thread, Lock
import numpy as np

# Add EGL configuration for headless rendering (critical fix)
import os
os.environ["MUJOCO_GL"] = "egl"  # Force EGL backend

# Rest of your imports...
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand
import config

locker = Lock()
mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)

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
