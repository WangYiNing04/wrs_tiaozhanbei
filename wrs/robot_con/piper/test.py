"""
Created on 2025/10/5 
Author: Hao Chen (chen960216@gmail.com)
"""
import time

from wrs.robot_con.piper.piper import PiperArmController

arm1 = PiperArmController(can_name="can0", has_gripper=True, )
arm2 = PiperArmController(can_name="can1", has_gripper=True, )

print("arm1 joint values:", arm1.get_joint_values())
print("arm2 joint values:", arm2.get_joint_values())

arm1.move_j([0, 0, 0, 0, 0, 1.3], block=False)
arm2.move_j([0, 0, 0, 0, 0, -1.3], block=False)
time.sleep(1)
arm1.move_j([0, 0, 0, 0, 0, 0], block=False)
arm2.move_j([0, 0, 0, 0, 0, 0], block=False)
