import os
import pathlib

import optas
from optas.visualize import Visualizer

def main(vis=True):

    # Setup robot model
    cwd = pathlib.Path(__file__).parent.resolve()  # path to current working directory
    urdf_filename = os.path.join(cwd, "robots", "g1", "g1_dual_arm.urdf")
    g1 = optas.RobotModel(urdf_filename=urdf_filename, time_derivs=[0, 1, 2])
    # in order of: 
    waist_y_jnt = -90.0
    waist_r_jnt = 0.0
    waist_p_jnt = 0.0
    l_shoulder_p_jnt = 0.0
    l_shoulder_r_jnt = 0.0
    l_shoulder_y_jnt = 0.0
    l_elbow_jnt = 0.0
    l_wrist_r_jnt = 0.0
    l_wrist_p_jnt = 0.0
    l_wrist_y_jnt = 0.0
    r_shoulder_p_jnt = 45
    r_shoulder_r_jnt = 0.0
    r_shoulder_y_jnt = 0.0
    r_elbow_jnt = 120
    r_wrist_r_jnt = 90
    r_wrist_p_jnt = 0.0
    r_wrist_y_jnt = 0.0#-90.0
    q0 = optas.np.deg2rad([waist_y_jnt,
        waist_r_jnt,
        waist_p_jnt,
        l_shoulder_p_jnt,
        l_shoulder_r_jnt,
        l_shoulder_y_jnt,
        l_elbow_jnt,
        l_wrist_r_jnt,
        l_wrist_p_jnt,
        l_wrist_y_jnt,
        r_shoulder_p_jnt,
        r_shoulder_r_jnt,
        r_shoulder_y_jnt,
        r_elbow_jnt,
        r_wrist_r_jnt,
        r_wrist_p_jnt,
        r_wrist_y_jnt])
    
    torso_position = g1.get_global_link_position('torso_link', optas.np.deg2rad([0.0]*17))
    obs = optas.DM.zeros(3, 3)
    obs[:,0] = torso_position
    obs[:,1] = torso_position
    obs[:,2] = torso_position
    obs[2,0] += 0.20
    obs[2,1] += 0.05
    obs[2,2] += 0.125
    linkrad = 0.04
    obsrad = [0.09]*obs.shape[1]
    link_names = [
        'right_shoulder_yaw_link',
        'right_elbow_link',
        'right_wrist_roll_link',
        'right_wrist_pitch_link',
        'right_wrist_yaw_link']

    vis = Visualizer()
    vis.robot(g1, q0, alpha=0.5)
    vis.grid_floor()
    for i in range(len(obsrad)):
        vis.sphere(radius=obsrad[i], position=obs[:, i], rgb=[1.0, 0.0, 0.0])

    for link_name in link_names:
        pos = optas.DM.zeros(3, 1)
        pos[:] = g1.get_global_link_position(link_name, q0)

        vis.sphere(radius=linkrad, 
                   position=pos, 
                   rgb=[1, 0.64705882352, 0.0])

    vis.start()

if __name__ == '__main__':
    main()
