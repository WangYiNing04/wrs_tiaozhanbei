#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/10/10 18:25
# @Author : ZhangXi
import math
import numpy as np
import os
import wrs.basis.robot_math as rm
import wrs.robot_sim.robots.single_arm_robot_interface as sari
from wrs.robot_sim.manipulators.piper.piper import Piper
from wrs.robot_sim.end_effectors.grippers.piper_gripper.piper_gripper import PiperGripper


class PiperSglArm(sari.SglArmRobotInterface):
    """
    Piper 机械臂整合类：基于 Piper 本体与 PiperGripper 夹爪。
    模仿 RealmanR 结构，提供高层接口（例如 fk、ik、fix_to、goto_given_conf）。
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3),
                 name="piper_arm", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        home_conf = np.zeros(6)
        # home_conf[1] = -math.pi / 3
        # home_conf[2] = math.pi / 2
        # home_conf[4] = math.pi / 6
        # 初始化机械臂
        self.manipulator = Piper(pos=self.pos,
                                 rotmat=self.rotmat,
                                 name=name + "_arm",
                                 enable_cc=False)
        self.manipulator.home_conf = home_conf
        self.manipulator._ik_solver = None
        self.manipulator.is_trac_ik = False

        self.end_effector = PiperGripper(
            pos=self.manipulator.gl_flange_pos,
            rotmat=self.manipulator.gl_flange_rotmat @ rm.rotmat_from_axangle([0, 0, 1], -math.pi / 2),
            name=name + "_piper_gripper")

        # 设置工具中心点（TCP）
        self.manipulator.loc_tcp_pos = self.end_effector.loc_acting_center_pos
        self.manipulator.loc_tcp_rotmat = self.end_effector.loc_acting_center_rotmat

        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self):
        """设置自碰撞检测，参考 RealmanR"""
        mlb = self.cc.add_cce(self.manipulator.jlc.anchor.lnk_list[0])
        ml0 = self.cc.add_cce(self.manipulator.jlc.jnts[0].lnk)
        ml1 = self.cc.add_cce(self.manipulator.jlc.jnts[1].lnk)
        ml2 = self.cc.add_cce(self.manipulator.jlc.jnts[2].lnk)
        ml3 = self.cc.add_cce(self.manipulator.jlc.jnts[3].lnk)
        ml4 = self.cc.add_cce(self.manipulator.jlc.jnts[4].lnk)
        mlee = self.cc.add_cce(self.end_effector.jlc.anchor.lnk_list[0])

        from_list = [ml3, ml4, mlee]
        into_list = [mlb, ml0, ml1]
        self.cc.set_cdpair_by_ids(from_list, into_list)
        self.cc.dynamic_into_list = [mlb, ml0, ml1, ml2]
        self.cc.dynamic_ext_list = []

    def fk(self, jnt_values, toggle_jacobian=False, update=False):
        """前向运动学"""
        results = self.manipulator.fk(jnt_values=jnt_values,
                                      toggle_jacobian=toggle_jacobian,
                                      update=update)
        if update:
            self.update_end_effector()
        return results

    def fix_to(self, pos, rotmat):
        """固定机械臂基座到指定位置"""
        self._pos = pos
        self._rotmat = rotmat
        self.manipulator.fix_to(pos=pos, rotmat=rotmat)
        self.update_end_effector()

    def get_jaw_width(self):
        return self.end_effector.get_jaw_width()

    def change_jaw_width(self, jaw_width):
        self.end_effector.change_jaw_width(jaw_width=jaw_width)


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd
    import wrs.modeling.geometric_model as mgm
    import wrs.basis.robot_math as rm

    base = wd.World(cam_pos=[1.5, 1.5, 1.0], lookat_pos=[0, 0, 0.3])
    mgm.gen_frame().attach_to(base)
    robot = PiperSglArm(enable_cc=True)
    robot.change_jaw_width(0)
    robot.gen_meshmodel(alpha=.5,toggle_jnt_frames=True).attach_to(base)
    robot.end_effector.gen_meshmodel(alpha=1,toggle_tcp_frame=True).attach_to(base)

    tgt_pos =np.array([0.378, -0.099417, 0.047612])
    tgt_rotmat = rm.rotmat_from_euler(3.0369, -0.0483, 2.7970)
    mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    jnt_values = robot.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, toggle_dbg=False)
    print(jnt_values)
    if jnt_values is not None:
        robot.goto_given_conf(jnt_values=jnt_values)
        robot.gen_meshmodel(alpha=1, toggle_tcp_frame=True).attach_to(base)
    else:
        print(1111)
    # robot.show_cdprim()
    robot.gen_meshmodel(alpha=.5, toggle_tcp_frame=True,toggle_jnt_frames = True).attach_to(base)
    base.run()
