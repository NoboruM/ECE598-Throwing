import time
import mujoco
import mujoco.viewer
import cv2
from threading import Thread, Lock
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand
import config
import math 

# Add EGL configuration for headless rendering
import os
os.environ["MUJOCO_GL"] = "egl"  # Force EGL backend

locker = Lock()
mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)
fovy = mj_model.cam_fovy
desired_qpos_values = np.zeros(56)
init_finger_angle = 0.5
desired_hand_pos = np.array([
    3.0*init_finger_angle,
    3.0*1.05851325*init_finger_angle +0.72349796,
    2.0*init_finger_angle,
    2.0*1.05851325*init_finger_angle +0.72349796,
    2.0*init_finger_angle,
    2.0*1.05851325*init_finger_angle +0.72349796,
    3.0*init_finger_angle,
    3.0*1.05851325*init_finger_angle +0.72349796,
    -1.9,
    0.5
])

desired_qpos_values[49:] = np.array([0.330, -0.123, 0.886, 0, 0, 0, 1])# ball pose
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
mj_data.ctrl[22:29] = [-2.4, -0.209, 0.0, -2.6, -0.0128, -0.8, -0.00198]
# set initial joint torques on the right hand to grasp the ball
hand_ctrl = np.array([
    desired_hand_pos[0], 
    desired_hand_pos[2], 
    desired_hand_pos[4], 
    desired_hand_pos[6], 
    desired_hand_pos[8], 
    desired_hand_pos[9]])
for i in range(29, 35):
    mj_data.ctrl[i] = hand_ctrl[i-29]


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
body_id = mj_model.body("right_wrist_yaw_link").id
velocity = np.zeros(6)  # [linear, rotational]
# mujoco.mj_objectVelocity(mj_model, mj_data, mujoco.mjtObj.mjOBJ_SITE, body_id, velocity, 0)
# linear_velocity = velocity[:3]
# print("linear veloctiy: ", linear_velocity)

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
            mujoco.mj_objectVelocity(mj_model, mj_data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0)
            linear_velocity = velocity[:3]
            print("linear veloctiy: ", linear_velocity)
            print("magnitude of velocity: ", np.linalg.norm(linear_velocity))
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

                #Detecting Clowns 
                hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
                lower_yellow = np.array([20,100,100])
                upper_yellow = np.array([30,255,255])
                mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area >100: 
                        x,y,w,h = cv2.boundingRect(cnt) 
                        cx, cy = x+w //2, y+h //2
                        width = 640 #parameters known
                        height = 480 #parameters known 
                        depth = 3 #3 meters known value 
                        
                        fovy_rad = math.radians(fovy)
                        fy = 0.5*height/math.tan(fovy_rad/2)
                        fx = fy*(width/height)
                        cx0 = width/2
                        cy0 = height/2
                        #cy2 = -1*(cy-height) #rotating cy frame to  


                        X = ((cx - cx0)*depth)/fx
                        Y= ((cy - cy0))*depth/fy
                        Z= depth
                        
                        print(f"World Coordinates: X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}")

                        cv2.rectangle(rgb_image, (x,y), (x+w, y+h), (0, 255,0),2) #Bounding box highlighting the clowns
                        cv2.circle(rgb_image, (cx,cy),5,(255,0,0),-1) #Circular point of the midpoint (direct center of clowns)
                        cv2.putText(rgb_image, f"({X:.2f},{Y:.2f},{Z:.2f})", (cx+10,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)



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
