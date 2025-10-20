#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import wrs.robot_sim.robots.dual_arm_robot_interface as dari
import wrs.motion.probabilistic.rrt_connect as rrtc
from wrs.robot_sim.robots.piper.piper_single_arm import PiperSglArm
import wrs.modeling.model_collection as mmc
import wrs.modeling.collision_model as mcm
import wrs.basis.robot_math as rm  # 确保导入了 rm


class DualPiperNoBody(dari.DualArmRobotInterface):
    """
    双臂系统（没有身体），只有左右两只 Piper 机械臂。
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name='dual_piper', enable_cc=enable_cc)

        lft_pos = np.array([0, 0, 0]) + pos
        self._lft_arm = PiperSglArm(pos=lft_pos, rotmat=rotmat.copy(), enable_cc=True)
        rgt_pos = np.array([0, -0.597, 0]) + pos
        self._rgt_arm = PiperSglArm(pos=rgt_pos, rotmat=rotmat.copy(), enable_cc=True)

        if self.cc is not None:
            self.setup_cc()
        # 默认使用左臂
        self.use_lft()
        # 默认回家姿态
        self.goto_home_conf()

    @property
    def lft_arm(self):
        return self._lft_arm

    @property
    def rgt_arm(self):
        return self._rgt_arm

    def use_lft(self):
        self._delegator = self._lft_arm
        self.cc = self._delegator.cc

    def use_rgt(self):
        self._delegator = self._rgt_arm
        self.cc = self._delegator.cc

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=1,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      toggle_flange_frame=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False):
        """
        生成双臂（无身体）的可视化模型。
        """
        m_col = mmc.ModelCollection(name=self.name + "_meshmodel")

        # 左臂模型
        self._lft_arm.gen_meshmodel(
            rgb=rgb,
            alpha=alpha,
            toggle_tcp_frame=toggle_tcp_frame,
            toggle_jnt_frames=toggle_jnt_frames,
            toggle_flange_frame=toggle_flange_frame,
            toggle_cdprim=toggle_cdprim,
            toggle_cdmesh=toggle_cdmesh
        ).attach_to(m_col)

        # 右臂模型
        self._rgt_arm.gen_meshmodel(
            rgb=rgb,
            alpha=alpha,
            toggle_tcp_frame=toggle_tcp_frame,
            toggle_jnt_frames=toggle_jnt_frames,
            toggle_flange_frame=toggle_flange_frame,
            toggle_cdprim=toggle_cdprim,
            toggle_cdmesh=toggle_cdmesh
        ).attach_to(m_col)

        return m_col

    def setup_cc(self):
        """如果需要可以在这里添加碰撞检测"""
        # self._lft_arm.cc = None
        # self._rgt_arm.cc = None
        if self.cc is not None:
            # 可以加左右臂互碰检测
            pass


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd
    import wrs.modeling.geometric_model as mgm
    import wrs.basis.robot_math as rm
    import math

    base = wd.World(cam_pos=[2, 2, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    box1 = mcm.gen_box(xyz_lengths=[0.8, 1.4, 1], pos=np.array([0.34, -0.2985, -0.5]))
    box1.attach_to(base)
    box2 = mcm.gen_box(xyz_lengths=[0.03, 0.03, 0.555], pos=np.array([-0.05, -0.2985, 0.2775]))
    box2.attach_to(base)
    box3 = mcm.gen_box(xyz_lengths=[0.05, 0.05, 0.05], pos=np.array([0.2761,-0.2985, 0.0522]))
    box3.attach_to(base)
    obs_list = [box1, box2]
    robot = DualPiperNoBody(enable_cc=True)
    robot.gen_meshmodel(toggle_cdprim=True).attach_to(base)

    robot.use_lft()
    tgt_pos = np.array([0.2761, -0.2985, 0.0722])

    # 下方 RRTConnect 和 IK 测试代码保持注释状态
    goal_conf = robot.ik(tgt_pos=np.array([0.2747, -0.2986, 0.0743]),tgt_rotmat = rm.rotmat_from_euler(3.0369, -0.0483, 2.7970))
    rrtc_planner = rrtc.RRTConnect(robot)
    start_conf = robot.get_jnt_values()
    mot_data = rrtc_planner.plan(start_conf=start_conf,
                                 goal_conf=goal_conf,
                                 obstacle_list=obs_list,
                                 ext_dist=.1,
                                 max_time=300)
    if mot_data is not None:
        n_step = len(mot_data.mesh_list)
        for i, model in enumerate(mot_data.mesh_list):
            model.rgb = rm.const.winter_map(i / n_step)
            model.alpha = .3
            model.attach_to(base)
    else:
        print("No available motion found.")

    # # IK 测试右臂
    # tgt_pos_r = np.array([0.3397, -0.2887, 0.2201])
    # tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    # mgm.gen_frame(pos=tgt_pos_r, rotmat=tgt_rotmat).attach_to(base)
    # jnt_values_r = robot.lft_arm.ik(tgt_pos_r, tgt_rotmat)
    # print(repr(jnt_values_r))
    # if jnt_values_r is not None:
    #     robot.rgt_arm.goto_given_conf(jnt_values=jnt_values_r)
    #     robot.rgt_arm.gen_meshmodel(alpha=1, toggle_tcp_frame=True).attach_to(base)

    base.run()
