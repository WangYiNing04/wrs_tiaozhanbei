'''
Author: wang yining
Date: 2025-10-21 12:41:27
LastEditTime: 2025-10-21 16:02:32
FilePath: /wrs_tiaozhanbei/my_project/tiaozhanbei/grasp/visualize_graspcollection.py
Description: 
e-mail: wangyining0408@outlook.com
'''
from wrs import wd, rm, mgm, mcm, ppp, rrtc, gg, gpa
import wrs.robot_sim.end_effectors.grippers.piper_gripper.piper_gripper as pg
from wrs.grasping.grasp import *

base = wd.World(cam_pos=[1.2, .7, 1], lookat_pos=[.0, 0, .15])
grasp_collection = GraspCollection()
grasp_collection = grasp_collection.load_from_disk(r'/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/grasp/manual_grasps.pickle')
gripper = pg.PiperGripper()
for grasp in grasp_collection:
    gripper.grip_at_by_pose(grasp.ac_pos, grasp.ac_rotmat, grasp.ee_values)
    gripper.gen_meshmodel(alpha=1).attach_to(base)

base.run()