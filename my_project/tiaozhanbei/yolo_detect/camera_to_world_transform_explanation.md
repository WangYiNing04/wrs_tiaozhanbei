# 相机坐标系到世界坐标系转换详解

## 🎯 转换原理

在 `manual_calib_piper_fixed_eye_camera.py` 中，相机坐标系到世界坐标系的转换是通过**齐次变换矩阵**实现的。

## 🔧 核心转换函数

### 1. 主要转换函数：`align_pcd()`

```python
def align_pcd(self, pcd):
    """
    For fixed eye camera, the calibration matrix directly transforms camera points to world frame
    :param pcd: point cloud in camera frame
    :return: point cloud in world frame
    """
    # The calibration matrix is camera-to-world transformation
    c2w_mat = self._init_calib_mat
    return rm.transform_points_by_homomat(c2w_mat, points=pcd)
```

### 2. 底层变换函数：`transform_points_by_homomat()`

```python
def transform_points_by_homomat(homomat: np.ndarray, points: np.ndarray):
    """
    do homotransform on a point or an array of points using pos
    :param homomat: 4x4齐次变换矩阵
    :param points: 1x3 或 nx3 点云数组
    :return: 变换后的点云
    """
    if points.ndim == 1:
        # 单个点的情况
        homo_point = np.insert(points, 3, 1)  # 添加齐次坐标 [x, y, z, 1]
        return np.dot(homomat, homo_point)[:3]  # 矩阵乘法后取前3维
    else:
        # 多个点的情况
        homo_points = np.ones((4, points.shape[0]))  # 创建齐次坐标矩阵
        homo_points[:3, :] = points.T[:3, :]  # 填充点云数据
        transformed_points = np.dot(homomat, homo_points).T  # 矩阵乘法
        return transformed_points[:, :3]  # 返回变换后的前3维坐标
```

## 📐 齐次变换矩阵结构

### 变换矩阵格式
```
T_c2w = [R3x3  t3x1]  = [r11  r12  r13  tx]
        [0     1   ]    [r21  r22  r23  ty]
                        [r31  r32  r33  tz]
                        [0    0    0    1 ]
```

其中：
- **R3x3**: 3×3旋转矩阵，表示相机坐标系相对于世界坐标系的姿态
- **t3x1**: 3×1平移向量，表示相机坐标系原点在世界坐标系中的位置

### 初始变换矩阵
```python
# 默认初始变换矩阵（相机在机器人前方1米，高度1米，向下看）
init_mat = np.array([
    [1, 0, 0, 0],    # X轴：世界坐标系X轴方向
    [0, 1, 0, 0],    # Y轴：世界坐标系Y轴方向  
    [0, 0, 1, 1],    # Z轴：世界坐标系Z轴方向，高度1米
    [0, 0, 0, 1]     # 齐次坐标
])
```

## 🔄 转换过程详解

### 数学公式
对于相机坐标系中的点 `P_camera = [x_c, y_c, z_c]`，转换到世界坐标系的公式为：

```
P_world = R * P_camera + t
```

在齐次坐标系中：
```
[P_world]   [R  t] [P_camera]
[   1   ] = [0  1] [   1    ]
```

### 具体实现步骤

1. **添加齐次坐标**：
   ```python
   # 将3D点转换为齐次坐标
   P_homo = [x_c, y_c, z_c, 1]
   ```

2. **矩阵乘法**：
   ```python
   # 应用变换矩阵
   P_world_homo = T_c2w * P_homo
   ```

3. **提取3D坐标**：
   ```python
   # 取前3维作为世界坐标系中的点
   P_world = P_world_homo[:3]
   ```

## 🎮 手动标定过程

### 平移调整
```python
def move_adjust(self, dir, dir_global, key_name=None):
    # 直接调整变换矩阵的平移部分
    self._init_calib_mat[:3, 3] = self._init_calib_mat[:3, 3] + dir_global * self.move_resolution
```

### 旋转调整
```python
def rotate_adjust(self, dir, dir_global, key_name=None):
    # 通过轴角旋转更新旋转矩阵
    self._init_calib_mat[:3, :3] = np.dot(
        rm.rotmat_from_axangle(dir_global, self.rotation_resolution),
        self._init_calib_mat[:3, :3]
    )
```

## 🎯 键盘控制映射

| 按键 | 功能 | 调整方向 |
|------|------|----------|
| W/S | X轴平移 | 相机前后移动 |
| A/D | Y轴平移 | 相机左右移动 |
| Q/E | Z轴平移 | 相机上下移动 |
| Z/X | 绕X轴旋转 | 相机俯仰 |
| C/V | 绕Y轴旋转 | 相机偏航 |
| B/N | 绕Z轴旋转 | 相机滚转 |

## 📊 实际应用示例

### 在detect.py中的应用
```python
# 1. 获取相机点云（相机坐标系）
pcd, pcd_color, depth_img, color_img = pipeline.get_pcd_texture_depth()

# 2. 转换到世界坐标系
pcd = self.align_pcd(pcd)  # 调用相机到世界坐标系的转换

# 3. 在世界坐标系中进行后续处理
cropped_pcd, original_count, cropped_count = self.crop_pointcloud_world(
    pcd, 
    x_range=(0, 0.6), 
    y_range=(0, -0.6), 
    z_range=(0.07, 0.08)
)
```

## 🔍 关键特点

### 1. 固定眼相机配置
- 相机固定在世界坐标系中
- 变换矩阵直接表示相机到世界的变换
- 不需要考虑机器人运动

### 2. 实时标定
- 通过键盘实时调整变换矩阵
- 可视化反馈帮助标定
- 自动保存标定结果

### 3. 齐次坐标变换
- 使用4×4齐次变换矩阵
- 支持旋转和平移的复合变换
- 高效的向量化计算

## 🎉 总结

相机坐标系到世界坐标系的转换通过以下步骤实现：

1. **获取标定矩阵**：通过手动标定获得相机到世界的变换矩阵
2. **齐次坐标转换**：将3D点转换为齐次坐标
3. **矩阵乘法**：应用变换矩阵进行坐标变换
4. **提取结果**：从齐次坐标中提取3D世界坐标

这种方法的优势是：
- **直观**：变换矩阵直接表示相机与世界坐标系的关系
- **高效**：使用向量化计算处理大量点云数据
- **灵活**：支持实时调整和标定
- **准确**：通过手动标定可以获得高精度的变换关系




