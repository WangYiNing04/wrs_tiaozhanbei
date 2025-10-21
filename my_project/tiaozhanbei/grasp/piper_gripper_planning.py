'''
Author: wang yining
Date: 2025-10-20 20:49:36
LastEditTime: 2025-10-21 16:13:58
FilePath: /wrs_tiaozhanbei/my_project/tiaozhanbei/grasp/piper_gripper_planning.py
Description: 
e-mail: wangyining0408@outlook.com
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/10/19 11:01
# @Author : ZhangXi
from wrs import wd, rm, gpa, mcm,mgm
import wrs.robot_sim.end_effectors.grippers.piper_gripper.piper_gripper as pg

base = wd.World(cam_pos=rm.vec(.5, .5, .5), lookat_pos=rm.vec(0, 0, 0))
mgm.gen_frame().attach_to(base)

obj_cmodel = mcm.CollisionModel(r"F:\wrs_tiaozhanbei\0000_examples\objects\tiaozhanbei\cup.stl")
obj_cmodel.attach_to(base)

# 实例化 PiperGripper
gripper = pg.PiperGripper()
grasp_collection = gpa.plan_gripper_grasps(gripper,
                                           obj_cmodel,
                                           angle_between_contact_normals=rm.radians(175),
                                           rotation_interval=rm.radians(45),
                                           max_samples=50,
                                           min_dist_between_sampled_contact_points=.03,
                                           contact_offset=.02,
                                           toggle_dbg=False)

print(grasp_collection)
grasp_collection.save_to_disk(file_name="piper_gripper_grasps.pickle")
for grasp in grasp_collection:
    gripper.grip_at_by_pose(grasp.ac_pos, grasp.ac_rotmat, grasp.ee_values)
    gripper.gen_meshmodel(alpha=1).attach_to(base)

base.run()