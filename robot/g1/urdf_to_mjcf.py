import mujoco

# Load the URDF file
model = mujoco.MjModel.from_xml_path('/home/noboru/KIMLAB_WS/G1_ws/src/unitree_mujoco/unitree_robots/g1/ability_hand_right_small.urdf')

# Save as MJCF
mujoco.mj_saveLastXML('/home/noboru/KIMLAB_WS/G1_ws/src/unitree_mujoco/unitree_robots/g1/ability_hand_right_small.xml', model)

print("Conversion complete. MJCF file saved.")
