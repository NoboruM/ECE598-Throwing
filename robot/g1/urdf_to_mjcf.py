import mujoco

# Load the URDF file
model = mujoco.MjModel.from_xml_path('/home/noboru/KIMLAB_WS/G1_ws/src/ECE598-Throwing/robot/g1/ability_hand_left.urdf', )

# Save as MJCF
mujoco.mj_saveLastXML('/home/noboru/KIMLAB_WS/G1_ws/src/ECE598-Throwing/robot/g1/ability_hand_left.xml', model)

print("Conversion complete. MJCF file saved.")
