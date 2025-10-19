'''
该文件用于部署Piper的SEM策略
Obs:
- 包含彩色图像、深度图像、相机内参、相机外参、关节位置

Action:
- 包含关节位置、夹爪位置

Piper API:
- get_color_image(cam) -> np.ndarray       获得真实世界彩色图像
- get_depth_image(cam) -> np.ndarray       获得真实世界深度图像
- get_camera_intrinsics(cam) -> np.ndarray       获得相机内参
- get_camera_extrinsics(cam) -> np.ndarray       获得相机内参
- get_joint_positions(cam) -> np.ndarray       获得关节位置
- get_gripper_position(cam) -> np.ndarray       获得夹爪位置
- execute_action(action) -> bool       执行动作

PYTHONPATH:
/home/wyn/RoboOrchardLab-master/projects/sem/robotwin/onnx_scripts:
:/home/wyn/wrs-main:/home/wyn/RoboOrchardLab-master/projects/sem/robotwin:
/~/PycharmProjects/wrs_tiaozhanbei:/home/wyn/PycharmProjects/wrs_tiaozhanbei


使用前测试，确保相机和机械臂能正常工作
'''

import copy
import importlib
import os
import time
import math
import yaml
from typing import Dict, List, Optional, Tuple, Any


import copy
import importlib
import os
import time
import math
import yaml
from typing import Dict, List, Optional, Tuple, Any
import onnxruntime as ort
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
#import cv2



from wrs.robot_con.piper.MultiRobotInterface import *



def pose_to_homogeneous(xyz, quat):
    """
    将 (x, y, z, qx, qy, qz, qw) 转换为 4x4 齐次变换矩阵

    参数:
        xyz: [x, y, z]
        quat: [qx, qy, qz, qw]  # 注意顺序，w 在最后

    返回:
        4x4 numpy.ndarray
    """
    # 1. 计算旋转矩阵
    rotation = R.from_quat(quat).as_matrix()  # 3x3

    # 2. 构造 4x4 齐次矩阵
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = xyz
    return T

# 工具函数
def rotmat_to_euler_xyz(R):
    """旋转矩阵转欧拉角 (XYZ顺序)"""
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    
    return np.array([x, y, z])


def euler_to_rotmat(euler):
    """欧拉角转旋转矩阵 (XYZ顺序)"""
    x, y, z = euler
    
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(x), -math.sin(x)],
        [0, math.sin(x), math.cos(x)]
    ])
    
    Ry = np.array([
        [math.cos(y), 0, math.sin(y)],
        [0, 1, 0],
        [-math.sin(y), 0, math.cos(y)]
    ])
    
    Rz = np.array([
        [math.cos(z), -math.sin(z), 0],
        [math.sin(z), math.cos(z), 0],
        [0, 0, 1]
    ])
    
    return Rz @ Ry @ Rx

# 获取彩色图像
def get_color_image(cam):
    """获取彩色图像"""
    try:
        return cam.get_color_image()
    except Exception as e:
        print(f"Error getting color image: {e}")
        return None

# 获取深度图像
def get_depth_image(cam):
    """获取深度图像"""
    try:
        return cam.get_depth_image()
    except Exception as e:
        print(f"Error getting depth image: {e}")
        return None

# 获取相机内参
def get_camera_intrinsics(cam):
    """获取相机内参"""
    try:
        return cam.get_intrinsics()
    except Exception as e:
        print(f"Error getting camera intrinsics: {e}")
        return np.eye(3, dtype=np.float32)

# 获取相机外参
def get_camera_extrinsics(cam):
    """获取相机外参"""
    try:
        return cam.get_extrinsics()
    except Exception as e:
        print(f"Error getting camera extrinsics: {e}")
        return np.eye(4, dtype=np.float32)



import sys

# 添加onnx_scripts到Python路径 - 必须在导入其他模块之前
onnx_scripts_path = os.path.join(os.path.dirname(__file__), "..", "onnx_scripts")
sys.path.insert(0, os.path.abspath(onnx_scripts_path))

import sys
sys.path.append('/home/wyn/RoboOrchardLab-master/projects/sem/robotwin/onnx_scripts')


# 内联函数定义，替代外部依赖
import json
from contextlib import contextmanager
from typing import Any, Dict, Type, Union, Literal
from accelerate import Accelerator
from safetensors.torch import load_model as safetensors_load_model, save_model as safetensors_save_model

# 内联 in_cwd 函数
@contextmanager
def in_cwd(destination: str):
    """Context manager to temporarily change the current working directory."""
    try:
        original_path = os.getcwd()
        os.chdir(destination)
        yield destination
    finally:
        os.chdir(original_path)

# 内联 load_config_class 函数
def load_config_class(config_str: str) -> Type:
    """Load a class from configuration string."""
    try:
        config_dict = json.loads(config_str)
        if 'class_name' in config_dict:
            # 动态导入类
            module_path = config_dict['class_name']
            module_name, class_name = module_path.rsplit('.', 1)
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            return cls
        else:
            # 如果没有class_name，返回一个简单的配置类
            class SimpleConfig:
                def __init__(self, **kwargs):
                    for key, value in kwargs.items():
                        setattr(self, key, value)
            return SimpleConfig
    except Exception as e:
        print(f"Error loading config class: {e}")
        # 返回一个默认的配置类
        class DefaultConfig:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        return DefaultConfig

# 内联 ModelMixin 类
class ModelMixin:
    """A simplified ModelMixin for loading models."""
    
    @staticmethod
    def load_model(
        directory: str,
        load_weight: bool = True,
        strict: bool = True,
        device: str | None = "cpu",
        device_map: str | dict[str, int | str | torch.device] | None = None,
        model_prefix: str = "model",
        load_impl: Literal["native", "accelerate"] = "accelerate",
    ):
        """Loads a model from a local directory."""
        
        directory = os.path.abspath(directory)
        
        if not os.path.exists(directory):
            raise FileNotFoundError(f"checkpoint {directory} does not exists!")
        
        config_file = os.path.join(directory, f"{model_prefix}.config.json")
        
        with open(config_file, "r") as f:
            cfg = load_config_class(f.read())
        
        with in_cwd(directory):
            model = cfg()
        
        if load_weight:
            model.load_weights(
                directory=directory,
                strict=strict,
                device=device,
                device_map=device_map,
                model_prefix=model_prefix,
                load_impl=load_impl,
            )
        return model
    
    def load_weights(
        self,
        directory: str,
        device: str | None = "cpu",
        device_map: str | dict[str, int | str | torch.device] | None = None,
        strict: bool = True,
        model_prefix: str = "model",
        load_impl: Literal["native", "accelerate"] = "accelerate",
    ):
        """Loads pretrained weights from a directory."""
        
        if load_impl == "native":
            if device_map is not None:
                raise ValueError(
                    "device_map is not supported when load_impl is 'native'."
                )
            ckpt_path = os.path.join(directory, f"{model_prefix}.safetensors")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"{ckpt_path} does not exists!")
            missing, unexpected = safetensors_load_model(
                self, filename=ckpt_path, strict=strict
            )
            if len(missing) > 0:
                print(f"Warning: Some weights are missing: {missing}")
            if len(unexpected) > 0:
                print(f"Warning: Some unexpected weights found: {unexpected}")
        elif load_impl == "accelerate":
            from accelerate import load_checkpoint_and_dispatch
            
            if device is not None and device_map is not None:
                raise ValueError(
                    "Only one of device or device_map can be specified."
                )
            ckpt_path = os.path.join(directory, f"{model_prefix}.safetensors")
            if os.path.exists(ckpt_path):
                load_checkpoint_and_dispatch(
                    self, ckpt_path, strict=strict, device_map=device_map
                )
            else:
                load_checkpoint_and_dispatch(
                    self, directory, strict=strict, device_map=device_map
                )
        else:
            raise ValueError(
                f"Invalid load_impl: {load_impl}, "
                f"expected 'native' or 'accelerate'."
            )
        
        if device is not None:
            self.to(device=device)

from data_builder import MultiArmDataBuilder

def main():
    """
    主函数 - 演示如何使用多机械臂SEM策略

    开始前通过插拔can口确认左臂为can0,右臂为can1
    
    """

     # 检查GPU可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # 配置参数
    ckpt_path = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/wrs/robot_con/piper/policy/sem/model/stack_blocks_three/model.safetensors"  # 替换为实际的模型路径
    config_file = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/wrs/robot_con/piper/policy/sem/config_sem_robotwin.py"  # SEM配置文件路径
    instruction = "Place red block, green block, and blue block in the center, then arrange green block on red block and blue block on green block"  # 任务指令
    camera_config_path = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/wrs/robot_con/piper/collect_data/config/camera_correspondence.yaml"  # 相机配置文件路径
    home_state_path = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/wrs/robot_con/piper/collect_data/config/home_state1.yaml"  # 零位状态配置文件路径
    
    # 多机械臂配置
    robot_config = {
        'arms': {
            'left_arm': {
                'can_name': 'can0',
                'has_gripper': True,
                'auto_enable': True
            },
            'right_arm': {
                'can_name': 'can1',
                'has_gripper': True,
                'auto_enable': True
            }
        }
    }
    
    # 相机名称列表（使用所有3个相机）
    camera_names = ["left", "middle", "right"]
    
    # 控制参数
    step_hz = 20.0  # 控制频率
    dt = 1.0 / step_hz
    max_joint_step = 0.1  # 最大关节位置步长
    
    
    try:
        # 初始化多机器人接口
        robot_interface = MultiRobotInterface(robot_config, camera_config_path, home_state_path)
        
        # 连接机器人
        if not robot_interface.connect():
            print("Failed to connect to robots.")
            return
        
        # 移动到零位
        print("Moving robots to home position before inference...")
        if not robot_interface.move_to_home_position():
            print("Failed to move to home position.")
            return
        
        # # 等待到达零位
        # if not robot_interface.wait_for_home_position(timeout=30.0):
        #     print("Timeout waiting for home position.")
        #     return
        
        print("All robots are now at home position. Ready for inference.")

        print("Initializing data builder...")
        # 初始化数据构建器
        builder = MultiArmDataBuilder(
            interface=robot_interface,
            image_height=320,
            image_width=256,
            num_joints_per_arm=7,
            hist_steps=1
        )

        data = builder.build_multi_arm_data(
            instruction="Your task instruction",
            camera_data_dict=None,
            robot_arm_data=None,
            t_robot2world=None,
            step_index=None
        )
        
        # 初始化多机械臂SEM策略
        print("Initializing Multi-Robot SEM policy...")
        
        
        # 主循环
        print("Starting inference loop...")
        print(f"Control frequency: {step_hz} Hz")
        print(f"Max joint step: {max_joint_step} radians")
        print("Press Ctrl+C to stop...")
        
    
    
        # 定义模型路径 - 指向workspace目录
        output_path = os.path.join(os.path.dirname(__file__), "workspace")

        

        print("Loading model...")
        model = ModelMixin.load_model(
            output_path, 
            load_weight=False,
            device=device,         # ✅ 指定 GPU
            load_impl="accelerate",   # ✅ 使用 accelerate 后端（默认即可）
            )
   
        model.eval()
        #print(f"✅ Model loaded on device: {next(model.parameters()).device}")
       
        import onnxruntime as ort
        print(ort.get_device()) 

        #model = torch.compile(model, mode="reduce-overhead", fullgraph=True)



        # 加载处理器（只执行一次）
        processor_cfg = load_config_class(
            open(f"{output_path}/processor.json").read()
        )
        with in_cwd(output_path):
            processor = processor_cfg()

       
    
        # 预热模型（第一次推理通常较慢）
        print("Warming up model...")
        dummy_data = builder.build_multi_arm_data(
            instruction="dummy",
            camera_data_dict=None,
            robot_arm_data=None,
            t_robot2world=None,
            step_index=0
        )
        dummy_data = processor.pre_process(dummy_data)
        
        # 将数据移至GPU
        dummy_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in dummy_data.items()}
        
        with torch.no_grad():

             _ = model(dummy_data)
    
        print("Model warmup completed.")

        step_count = 0
        max_steps = 1000  # 最大执行步数
        
        while step_count < max_steps:
            try:
                print(f"\nStep {step_count + 1}/{max_steps}")
                time.sleep(3) #保证相机稳定
                data_start = time.time()
                data = builder.build_multi_arm_data(
                    instruction="Your task instruction",
                    camera_data_dict=None,
                    robot_arm_data=None,
                    t_robot2world=None,
                    step_index=step_count
                )
                data_build_time = time.time() - data_start

                #输出data
                #print(data)

                # 预测动作
                start_time = time.time()
                

                # 定义模型路径 - 指向workspace目录
                # output_path = os.path.join(os.path.dirname(__file__), "workspace")

                # model = ModelMixin.load_model(output_path, load_weight=False)
                # model = model.to(device)
                # model.eval()

                # processor_cfg = load_config_class(
                #     open(f"{output_path}/processor.json").read()
                # )
                # with in_cwd(output_path):
                #     processor = processor_cfg()

                # # 清空GPU缓存
                # torch.cuda.empty_cache()

                # init data dict with imgs, depths, text, intrinsic, joint_state

                preprocess_start = time.time()
                data = processor.pre_process(data)
                preprocess_time = time.time() - preprocess_start

                
                # GPU推理
                inference_start = time.time()
                with torch.no_grad():
                    model_outs = model(data)
                inference_time = time.time() - inference_start


                postprocess_start = time.time()
    
                #print(f"model_outs content: {model_outs}")

      
                #model_outs = model(data)

                
                #64步
                
                actions = processor.post_process(batch = 1,model_outputs = model_outs).action

                postprocess_time = time.time() - postprocess_start
                #print(actions)
                # 定期清理缓存
             
                inference_time = time.time() - start_time
                
                # 打印时间统计
                print(f"Data build: {data_build_time:.4f}s | "
                      f"Preprocess: {preprocess_time:.4f}s | "
                      f"Inference: {inference_time:.4f}s | "
                      f"Postprocess: {postprocess_time:.4f}s")
                
                print(f"Inference time: {inference_time:.3f}s")
                print(f"Predicted actions shape: {actions.shape}")
                
                print("First action command:")
                #print(actions[0])

                # 执行完整的64步动作序列

                #continue
                #actions.shape[0]
                for i in range(64):
                    if step_count >= max_steps:
                        break
                        
                    action = actions[i].tolist()
                    
                    # 检查动作长度
                    if len(action) != 14:
                        print(f"Warning: Expected 14 action elements, got {len(action)}")
                        # 如果动作长度不正确，使用零动作
                        action = [0] * 14
                    
                    # # 限幅处理（避免大步跳变）
                    # # 前6个是左臂关节位置
                    # action[:6] = np.clip(action[:6], -max_joint_step, max_joint_step)
                    # # 第7个是左臂夹爪开合度
                    # action[6] = np.clip(action[6], 0, 1)  # 假设夹爪开合度在0-1之间
                    
                    # # 后6个是右臂关节位置
                    # action[7:13] = np.clip(action[7:13], -max_joint_step, max_joint_step)
                    # # 第14个是右臂夹爪开合度
                    # action[13] = np.clip(action[13], 0, 1)  # 假设夹爪开合度在0-1之间

                    # 获取当前关节位置
                    current_joints_left = robot_interface.arms['left_arm'].get_joint_values()
                    current_joints_right = robot_interface.arms['right_arm'].get_joint_values()

                    # 计算目标位置与当前位置的差值
                    delta_left = np.array(action[:6]) - current_joints_left[:6]
                    delta_right = np.array(action[7:13]) - current_joints_right[:6]

                    # if (abs(delta_left) > 0.1).any():  # 任一元素 > 0.1 则返回 True
                    #     print("大步限位工作中")

                    # if (abs(delta_left) > 0.1).any():  # 任一元素 > 0.1 则返回 True
                    #     print("大步限位工作中")
                    # 限制变化量大小
                    delta_left = np.clip(delta_left, -max_joint_step, max_joint_step)
                    delta_right = np.clip(delta_right, -max_joint_step, max_joint_step)

                    # 计算平滑后的目标位置
                    target_left = current_joints_left[:6] + delta_left
                    target_right = current_joints_right[:6] + delta_right

                    # 夹爪处理
                    gripper_left = np.clip(action[6], 0, 1)
                    gripper_right = np.clip(action[13], 0, 1)

                    # 将动作分配给不同的机械臂
                    # 组合成最终动作
                    arm_actions = {
                        'left_arm': np.concatenate([target_left, [gripper_left]]),
                        'right_arm': np.concatenate([target_right, [gripper_right]])
                    }
        
                    # 执行动作
                    success = robot_interface.execute_action(arm_actions)
                    
                    if not success:
                        print(f"Failed to execute action at step {step_count + 1}")
                        break
                    
                    print(f"Executed action {i+1}/64: {arm_actions}")
                    
                    # 控制频率
                    time.sleep(dt)
                    step_count += 1
                
            except KeyboardInterrupt:
                print("\nStopping inference loop...")
                break
            except Exception as e:
                print(f"Error in step {step_count + 1}: {e}")
                import traceback
                traceback.print_exc()
                # 继续执行，不中断循环
                time.sleep(1.0)
                step_count += 1
        
        print("Inference loop completed.")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        if 'robot_interface' in locals():
            #robot_interface.disconnect()
            pass

        print("Cleanup completed.")

        print("Moving robots to home position before inference...")
        if not robot_interface.move_to_home_position():
            print("Failed to move to home position.")
            return

if __name__ == "__main__":
    main()