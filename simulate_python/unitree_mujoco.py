import time
import mujoco
import mujoco.viewer
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
for i in range(7, 56):
    mj_data.qpos[i] = 0
mj_data.qpos[34] = -np.pi/2
mj_data.qpos[23] = np.pi/2
mujoco.mj_forward(mj_model, mj_data)    # Propagates initial state

elastic_band = ElasticBand()
if not config.ENABLE_ELASTIC_BAND:
    elastic_band.damping = 0
    elastic_band.stiffness = 0
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
        time.sleep(config.VIEWER_DT)

if __name__ == "__main__":
    # Run viewer in main thread
    sim_thread = Thread(target=SimulationThread)
    sim_thread.start()
    
    # Handle rendering in main thread
    PhysicsViewerThread()
    sim_thread.join()