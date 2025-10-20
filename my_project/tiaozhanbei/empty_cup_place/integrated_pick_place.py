# 集成抓取规划和Pick-and-Place任务的完整解决方案
# 1. 创建抓取姿态集合
# 2. 执行Pick-and-Place任务
# 3. 导出关节角轨迹供真实机械臂使用
# 4. 执行关节角轨迹
# 5. 可视化运动轨迹
# 6. 导出关节角轨迹到JSON文件
# 7. 从JSON文件读取关节角轨迹并逐点执行 move_j。
# 8. 要求每个轨迹点为6个关节角。
# 9. 得到杯子位置后判断使用哪只手抓取
# 10. 执行抓取

#任务流程:检测杯子位置和杯垫位置->规划抓取->执行抓取和放置
from wrs import wd, rm, mgm, mcm, ppp, rrtc, gg, gpa
import wrs.robot_sim.robots.piper.piper_single_arm as psa
import wrs.robot_sim.end_effectors.grippers.piper_gripper.piper_gripper as pg
import numpy as np
from direct.task.TaskManagerGlobal import taskMgr
import os
# 兼容作为脚本直接运行或作为包运行的导入方式
try:
    from my_project.tiaozhanbei.empty_cup_place.detect_mini import PointCloudProcessor  # 绝对导入（包方式）
except Exception:
    try:
        from .detect_mini import PointCloudProcessor  # 相对导入（包方式）
    except Exception:
        import os as _os, sys as _sys
        _sys.path.append(_os.path.dirname(_os.path.abspath(__file__)))
        from detect_mini import PointCloudProcessor  # 脚本方式
import json
from datetime import datetime
from pathlib import Path
from wrs.robot_con.piper.piper import PiperArmController
def create_grasp_collection(obj_path, save_path, gripper=None, base=None):
    """
    为指定物体创建抓取姿态集合
    
    Args:
        obj_path: 物体STL文件路径
        save_path: 保存抓取姿态的pickle文件路径
        gripper: 夹爪对象，如果为None则创建新的PiperGripper
        base: 3D世界对象，如果为None则创建新的
    
    Returns:
        GraspCollection: 抓取姿态集合
    """
    print("正在生成抓取姿态集合...")
    
    # 如果没有提供base，则创建新的3D环境
    if base is None:
        base = wd.World(cam_pos=rm.vec(.5, .5, .5), lookat_pos=rm.vec(0, 0, 0))
        mgm.gen_frame().attach_to(base)
    
    # 加载目标物体
    obj_cmodel = mcm.CollisionModel(obj_path)
    obj_cmodel.attach_to(base)
    
    # 实例化夹爪
    if gripper is None:
        gripper = pg.PiperGripper()
    
    # 生成抓取姿态
    grasp_collection = gpa.plan_gripper_grasps(
        gripper,
        obj_cmodel,
        angle_between_contact_normals=rm.radians(175),
        rotation_interval=rm.radians(15),
        max_samples=50,
        min_dist_between_sampled_contact_points=.1,
        contact_offset=.02,
        toggle_dbg=False
    )
    
    print(f"生成了 {len(grasp_collection)} 个抓取姿态")
    
    # 保存抓取姿态
    grasp_collection.save_to_disk(file_name=save_path)
    print(f"抓取姿态已保存到: {save_path}")
    
    # 可视化抓取姿态（可选）
    # for grasp in grasp_collection:
    #     gripper.grip_at_by_pose(grasp.ac_pos, grasp.ac_rotmat, grasp.ee_values)
    #     gripper.gen_meshmodel(alpha=1).attach_to(base)
    
    return grasp_collection

def run_pick_place_task(obj_path, grasp_collection_path, start_pos, goal_pos, 
                       start_rot=None, goal_rot=None, obstacle_list=None, base=None):
    """
    执行Pick-and-Place任务
    
    Args:
        obj_path: 物体STL文件路径
        grasp_collection_path: 抓取姿态集合文件路径
        start_pos: 起始位置 [x, y, z]
        goal_pos: 目标位置 [x, y, z]
        start_rot: 起始旋转矩阵，如果为None则使用单位矩阵
        goal_rot: 目标旋转矩阵，如果为None则使用单位矩阵
        obstacle_list: 障碍物列表
        base: 3D世界对象，如果为None则创建新的
    
    Returns:
        MotionData: 运动轨迹数据
    """
    print("正在执行Pick-and-Place任务...")
    
    # 如果没有提供base，则创建新的3D环境
    if base is None:
        base = wd.World(cam_pos=[1.2, .7, 1], lookat_pos=[.0, 0, .15])
        mgm.gen_frame().attach_to(base)
    
    # 创建地面
    # ground = mcm.gen_box(xyz_lengths=rm.vec(5, 5, 1), rgb=rm.vec(.7, .7, .7), alpha=1)
    # ground.pos = rm.np.array([0, 0, -.5])
    # ground.attach_to(base)
    # ground.show_cdprim()
    
    # 设置旋转矩阵
    if start_rot is None:
        start_rot = rm.eye(3)
    if goal_rot is None:
        goal_rot = rm.eye(3)
    
    # 创建起始位置的物体
    holder_start = mcm.CollisionModel(obj_path)
    holder_start.rgba = rm.np.array([.5, .5, .5, 1])
    holder_start.pos = np.array(start_pos)
    holder_start.rotmat = start_rot
    mgm.gen_frame().attach_to(holder_start)
    
    # 可视化起始位置
    h1_copy = holder_start.copy()
    h1_copy.attach_to(base)
    h1_copy.show_cdprim()
    
    # 创建目标位置的物体（半透明）
    holder_goal = mcm.CollisionModel(obj_path)
    holder_goal.pos = np.array(goal_pos)
    holder_goal.rotmat = goal_rot
    h2_copy = holder_goal.copy()
    h2_copy.rgb = rm.const.tab20_list[0]
    h2_copy.alpha = .3
    h2_copy.attach_to(base)
    h2_copy.show_cdprim()
    
    # 创建机器人
    robot = psa.PiperSglArm()
    robot.gen_meshmodel().attach_to(base)
    
    # 实例化规划器
    rrtc_planner = rrtc.RRTConnect(robot)
    ppp_planner = ppp.PickPlacePlanner(robot)
    
    # 加载抓取姿态集合
    grasp_collection = gg.GraspCollection.load_from_disk(file_name=grasp_collection_path)
    start_conf = robot.get_jnt_values()
    goal_pose_list = [(goal_pos, goal_rot)]
    
    # 设置障碍物
    if obstacle_list is None:
        obstacle_list = []
    
    # 生成Pick and Place运动
    mot_data = ppp_planner.gen_pick_and_place(
        obj_cmodel=holder_start,
        end_jnt_values=start_conf,
        grasp_collection=grasp_collection,
        goal_pose_list=goal_pose_list,
        place_approach_distance_list=[.05] * len(goal_pose_list),
        place_depart_distance_list=[.05] * len(goal_pose_list),
        pick_approach_distance=.05,
        pick_depart_distance=.05,
        pick_depart_direction=rm.const.z_ax,
        obstacle_list=obstacle_list,
        use_rrt=True
    )
    
    if mot_data is None:
        print("错误：无法生成Pick-and-Place运动轨迹！")
        return None
    
    print("Pick-and-Place运动轨迹生成成功！")
    return mot_data

def animate_motion(mot_data, base):
    """
    动画显示运动轨迹
    
    Args:
        mot_data: 运动轨迹数据
        base: 3D世界对象
    """
    class Data(object):
        def __init__(self, mot_data):
            self.counter = 0
            self.mot_data = mot_data

    anime_data = Data(mot_data)

    def update(anime_data, task):
        if anime_data.counter > 0:
            anime_data.mot_data.mesh_list[anime_data.counter - 1].detach()
        if anime_data.counter >= len(anime_data.mot_data):
            anime_data.counter = 0

        mesh_model = anime_data.mot_data.mesh_list[anime_data.counter]
        mesh_model.attach_to(base)
        mesh_model.show_cdprim()

        if base.inputmgr.keymap['space']:
            anime_data.counter += 1
        return task.again

    taskMgr.doMethodLater(0.01, update, "update",
                          extraArgs=[anime_data],
                          appendTask=True)

def export_joint_trajectory(mot_data, save_dir=None, filename=None):
    """
    导出关节角轨迹到JSON文件，便于真实机械臂部署
    返回: (trajectory_list, saved_path)
    """
    trajectory = [jnt_values.tolist() for jnt_values in mot_data.jv_list]
    if save_dir is None:
        save_dir = r"F:\wrs_tiaozhanbei\my_project\tiaozhanbei\empty_cup_place\exported"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"joint_trajectory_empty_cup_place.json"
    saved_path = os.path.join(save_dir, filename)
    with open(saved_path, 'w', encoding='utf-8') as f:
        json.dump({"joint_trajectory": trajectory}, f, ensure_ascii=False, indent=2)
    print(f"关节角列表已保存: {saved_path} (共 {len(trajectory)} 个点)")
    return trajectory, saved_path

def excute_move_j(arm: PiperArmController, json_path: str):
    """
    从JSON文件读取关节角轨迹并逐点执行 move_j。
    要求每个轨迹点为6个关节角。
    """
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"未找到轨迹文件: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, dict) or 'joint_trajectory' not in data:
        raise ValueError("轨迹文件格式不正确，缺少 'joint_trajectory' 字段")

    trajectory_raw = data['joint_trajectory']
    if not isinstance(trajectory_raw, list):
        raise ValueError("'joint_trajectory' 应为列表")

    # 规整为6关节角的列表
    trajectory: list[list[float]] = []
    for idx, point in enumerate(trajectory_raw):
        if not isinstance(point, (list, tuple)):
            raise ValueError(f"第 {idx+1} 个轨迹点格式错误，应为列表/元组")
        if len(point) < 6:
            raise ValueError(f"第 {idx+1} 个轨迹点关节数不足: {len(point)} < 6")
        jv6 = [float(point[i]) for i in range(6)]
        trajectory.append(jv6)

    print(f"读取到 {len(trajectory)} 个关节轨迹点，将依次执行 move_j（6轴）...")

    for i, jv in enumerate(trajectory):
        print(f"执行第 {i+1}/{len(trajectory)} 个点: {jv}")
        arm.move_j(jv)


def main():
    """
    主函数：完整的抓取规划和Pick-and-Place任务流程
    """
    #初始化
    left_arm_con = PiperArmController(can_name='can0', has_gripper=True)
    right_arm_con = PiperArmController(can_name='can1', has_gripper=True)
    # 文件路径配置
    obj_path = r"F:\wrs_tiaozhanbei\0000_examples\objects\tiaozhanbei\cup.stl"
    grasp_save_path = r"F:\wrs_tiaozhanbei\my_project\tiaozhanbei\empty_cup_place\piper_gripper_grasps.pickle"

    #######################################################################
    #检测杯子位置
    #######################################################################
    #初始化点云检测
  
    #创建处理器（会自动初始化相机）
    processor = PointCloudProcessor()

    # 启动相机流 返回杯口中心点
    x,y,z = processor.start_camera_stream()
    
    #处理杯口坐标
    z = z - 7.5

    cup_x,cup_y,cup_z = 0.3397, -0.2887, 0
    #######################################################################
    

    #######################################################################
    #检测杯垫
    #######################################################################

    coaster_x,coaster_y,coaster_z = 0.378, -0.099417, 0
    #######################################################################


    #设定杯子位置和杯垫位置
    start_pos = [cup_x, cup_y, cup_z]  # 杯子位置
    goal_pos = [coaster_x, coaster_y, coaster_z]  # 杯垫位置
    start_rot = rm.rotmat_from_euler(0, 0, 0)  # 杯子旋转
    goal_rot = rm.rotmat_from_euler(0, 0, 0)   # 杯垫旋转

    # 创建统一的3D环境
    print("正在初始化3D环境...")
    base = wd.World(cam_pos=[1.2, .7, 1], lookat_pos=[.0, 0, .15])
    mgm.gen_frame().attach_to(base)
    
    # 检查抓取姿态文件是否存在
    if not os.path.exists(grasp_save_path):
        print("抓取姿态文件不存在，正在生成...")
        grasp_collection = create_grasp_collection(obj_path, grasp_save_path, base=base)
    else:
        print("使用已存在的抓取姿态文件")
    
    # 定义任务参数

    #start_pos and start_rot 物体的现实位置
    #goal_pos and goal_rot   物体的目标放置位置
    # start_pos = [0.3397, -0.2887, 0]  
    # goal_pos = [0.378, -0.099417, 0]  
    # start_rot = rm.rotmat_from_euler(0, 0, 0)  
    # goal_rot = rm.rotmat_from_euler(0, 0, 0)   

    box1 = mcm.gen_box(xyz_lengths=[0.8, 1.4, 1], pos=np.array([0.34, -0.2985, -0.5]))
    box1.attach_to(base)
    box2 = mcm.gen_box(xyz_lengths=[0.03, 0.03, 0.555], pos=np.array([-0.05, -0.2985, 0.2775]))
    box2.attach_to(base)

    # 定义障碍物
    # obstacle_list = [
    #     mcm.gen_box(xyz_lengths=[0.8, 1.4, 1], pos=np.array([0.34, -0.2985, -0.5])),
    #     mcm.gen_box(xyz_lengths=[0.03, 0.03, 0.555], pos=np.array([-0.05, -0.2985, 0.2775]))
    # ]
    
    obstacle_list = [box1, box2]
    try:
        # 执行Pick-and-Place任务
        result = run_pick_place_task(
            obj_path=obj_path,
            grasp_collection_path=grasp_save_path,
            start_pos=start_pos,
            goal_pos=goal_pos,
            start_rot=start_rot,
            goal_rot=goal_rot,
            obstacle_list=obstacle_list,
            base=base
        )
        
        if result is not None:
            mot_data = result
            # 导出关节角轨迹供真实机械臂使用
            trajectory, traj_path = export_joint_trajectory(mot_data)
            # 示例：打印前3个关节角
            for idx, jv in enumerate(trajectory[:3]):
                print(f"轨迹点 {idx+1}: {jv}")
            print("开始动画演示...")
            print("按空格键逐步播放动画")
            animate_motion(mot_data, base)
            base.run()
        else:
            print("任务执行失败！")
            print("启动基础3D环境...")
            base.run()
    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        print("启动基础3D环境...")
        base.run()


    #得到杯子位置后判断使用哪只手抓取
    if y > -0.3:
        arm = left_arm_con
    else:
        arm = right_arm_con
        
    traj_path = r"F:\wrs_tiaozhanbei\my_project\tiaozhanbei\empty_cup_place\exported\joint_trajectory_empty_cup_place.json"
    excute_move_j(arm,traj_path)

if __name__ == '__main__':
    main()
