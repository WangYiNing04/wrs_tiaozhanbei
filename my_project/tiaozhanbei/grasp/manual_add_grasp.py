'''
Author: wang yining
Date: 2025-10-21 12:17:23
LastEditTime: 2025-10-21 13:48:16
FilePath: /wrs_tiaozhanbei/my_project/tiaozhanbei/grasp/manual_add_grasp.py
Description: 
e-mail: wangyining0408@outlook.com
'''
# from wrs import wd, rm, gpa, mcm,mgm
# import wrs.robot_sim.end_effectors.grippers.piper_gripper.piper_gripper as pg
# from wrs.grasping.grasp import GraspCollection
# base = wd.World(cam_pos=rm.vec(.5, .5, .5), lookat_pos=rm.vec(0, 0, 0))
# mgm.gen_frame().attach_to(base)

# obj_cmodel = mcm.CollisionModel(r"/home/wyn/PycharmProjects/wrs_tiaozhanbei/0000_examples/objects/tiaozhanbei/cup.stl")
# obj_cmodel.attach_to(base)

# # 实例化 PiperGripper
# gripper = pg.PiperGripper()
# grasp_collection = gpa.plan_gripper_grasps(gripper,
#                                            obj_cmodel,
#                                            angle_between_contact_normals=rm.radians(175),
#                                            rotation_interval=rm.radians(15),
#                                            max_samples=50,
#                                            min_dist_between_sampled_contact_points=.1,
#                                            contact_offset=.02,
#                                            toggle_dbg=False)


# print(grasp_collection)
# grasp_collection.save_to_disk(file_name="piper_gripper_grasps.pickle")
# for grasp in grasp_collection:
#     gripper.grip_at_by_pose(grasp.ac_pos, grasp.ac_rotmat, grasp.ee_values)
#     gripper.gen_meshmodel(alpha=1).attach_to(base)

# base.run()


from wrs import wd, rm, gpa, mcm, mgm
import wrs.robot_sim.end_effectors.grippers.piper_gripper.piper_gripper as pg
from wrs.grasping.grasp import GraspCollection
from panda3d.core import *
from direct.showbase.DirectObject import DirectObject
import numpy as np

class GripperController(DirectObject):
    def __init__(self, gripper, base):
        self.gripper = gripper
        self.base = base
        self.pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.rotmat = np.eye(3)
        self.ee_values = gripper.jaw_range[1]  # 默认张开
        
        self.gripper_model = None
        self.update_gripper()
        self.setup_keyboard_controls()
        self.grasp_collection = GraspCollection(end_effector=gripper)

        self.accept('window-event-close', self.on_window_close)
        
    def on_window_close(self):
        self.save_grasps("manual_grasps.pickle")
        base.userExit()  # 确保程序退出

    def setup_keyboard_controls(self):
        # 平移控制 (注意这里传递的是列表，会被打包成一个参数)
        self.accept('w', self.move, [[0, 0.01, 0]])
        self.accept('s', self.move, [[0, -0.01, 0]])
        self.accept('a', self.move, [[-0.01, 0, 0]])
        self.accept('d', self.move, [[0.01, 0, 0]])
        self.accept('q', self.move, [[0, 0, 0.01]])
        self.accept('e', self.move, [[0, 0, -0.01]])
        
        # 旋转控制
        self.accept('z', self.rotate, [[0.1, 0, 0]])
        self.accept('x', self.rotate, [[-0.1, 0, 0]])
        self.accept('c', self.rotate, [[0, 0.1, 0]])
        self.accept('v', self.rotate, [[0, -0.1, 0]])
        self.accept('b', self.rotate, [[0, 0, 0.1]])
        self.accept('n', self.rotate, [[0, 0, -0.1]])
        
        # 夹爪控制
        self.accept('f', self.adjust_gripper, [0.01])
        self.accept('g', self.adjust_gripper, [-0.01])
        
        # 记录抓取姿势
        self.accept('enter', self.record_grasp)
        self.accept('h', self.toggle_gripper_visibility)

        self.accept('p', self.save_grasps, ["manual_grasps.pickle"])  # 按p键保存
    
    def move(self, delta, *args):  # 修改为接受额外参数
        self.pos += np.array(delta)
        self.update_gripper()
    
    def rotate(self, angles, *args):  # 修改为接受额外参数
        rotmat_x = rm.rotmat_from_axangle([1, 0, 0], angles[0])
        rotmat_y = rm.rotmat_from_axangle([0, 1, 0], angles[1])
        rotmat_z = rm.rotmat_from_axangle([0, 0, 1], angles[2])
        self.rotmat = self.rotmat @ rotmat_x @ rotmat_y @ rotmat_z
        self.update_gripper()
    
    def adjust_gripper(self, delta, *args):  # 修改为接受额外参数
        self.ee_values = np.clip(self.ee_values + delta, 
                               self.gripper.jaw_range[0], 
                               self.gripper.jaw_range[1])
        self.update_gripper()
    
    def update_gripper(self):
        if self.gripper_model is not None:
            self.gripper_model.detach()
        
        self.gripper.grip_at_by_pose(self.pos, self.rotmat, self.ee_values)
        self.gripper_model = self.gripper.gen_meshmodel(alpha=0.7)
        self.gripper_model.attach_to(self.base)
    
    def record_grasp(self):
        try:
            grasp = self.gripper.get_grasp(ac_pos=self.pos, ac_rotmat=self.rotmat)
            self.grasp_collection.append(grasp)
            print(f"记录抓取姿势 #{len(self.grasp_collection)}:")
            print(f"位置: {self.pos}")
            print(f"旋转矩阵:\n{self.rotmat}")
            print(f"夹爪宽度: {self.ee_values}")
            
        except Exception as e:
            print(f"记录抓取姿势失败: {str(e)}")
    
    def toggle_gripper_visibility(self):
        if self.gripper_model is not None:
            if self.gripper_model.isHidden():
                self.gripper_model.show()
            else:
                self.gripper_model.hide()
    
    def save_grasps(self, filename):
        self.grasp_collection.save_to_disk(file_name=filename)
        print(f"已保存 {len(self.grasp_collection)} 个抓取姿势到 {filename}")

# 主程序
base = wd.World(cam_pos=rm.vec(.5, .5, .5), lookat_pos=rm.vec(0, 0, 0))
mgm.gen_frame().attach_to(base)

# 加载物体模型
obj_cmodel = mcm.CollisionModel(r"/home/wyn/PycharmProjects/wrs_tiaozhanbei/0000_examples/objects/tiaozhanbei/cup.stl")
obj_cmodel.attach_to(base)

# 实例化PiperGripper
gripper = pg.PiperGripper()

# 创建交互控制器
controller = GripperController(gripper, base)

# 自动生成的抓取姿势
# auto_grasp_collection = gpa.plan_gripper_grasps(
#     gripper,
#     obj_cmodel,
#     angle_between_contact_normals=rm.radians(175),
#     rotation_interval=rm.radians(15),
#     max_samples=50,
#     min_dist_between_sampled_contact_points=.1,
#     contact_offset=.02,
#     toggle_dbg=False
# )

# print(f"自动生成了 {len(auto_grasp_collection)} 个抓取姿势")

# 运行可视化界面
base.run()

# 保存手动记录的抓取姿势
controller.save_grasps("manual_grasps.pickle")