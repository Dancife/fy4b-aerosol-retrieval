# FY4B AGRI 气溶胶反演完整工作流程

## 概述
本文档描述了基于FY4B AGRI卫星数据的完整气溶胶光学厚度反演工作流程。

## 系统架构

### 1. 数据预处理模块
- **输入**: FY4B AGRI L1B数据 (HDF5格式)
- **输出**: 几何校正后的表观反射率、几何角度、有效掩膜
- **文件**: `fy4b_dt_preprocessor.ipynb`

### 2. 查找表生成模块
- **输入**: 大气参数配置
- **输出**: HDF5格式的查找表
- **文件**: `generate_lut_server.py`
- **依赖**: Py6S辐射传输模型

### 3. 气溶胶反演核心模块
- **输入**: 预处理数据 + 查找表
- **输出**: 气溶胶光学厚度、地表反射率等
- **文件**: `aerosol_retrieval.py`

### 4. 质量控制模块
- **输入**: 反演结果
- **输出**: 质量控制的最终产品
- **文件**: `aerosol_retrieval.py` (QualityControl类)

### 5. 可视化与输出模块
- **输入**: 质量控制后的结果
- **输出**: 图像、NetCDF、GeoTIFF文件
- **文件**: `aerosol_retrieval.py` (Visualization, OutputHandler类)

## 完整工作流程

### 步骤1: 环境准备
```bash
# 安装依赖
pip install numpy pandas matplotlib scipy h5py xarray rasterio py6s
# 安装Py6S (用于生成LUT)
# 注意: Py6S安装较复杂，可能需要从源码编译
```

### 步骤2: 生成查找表
```python
# 找个服务器运行查找表生成程序
python generate_lut_server.py
# 输出: FY4B_AGRI_LUT_Server.h5
```

### 步骤3: 数据预处理（主要是筛像元）
```python
# 运行 fy4b_dt_preprocessor.ipynb
# 或导入相关函数
from fy4b_dt_preprocessor import (
    load_nav_and_mask,
    calculate_fy4b_angles,
    read_and_filter_fy4b_data
)

# 1. 加载地理信息
lats, lons, land_mask = load_nav_and_mask(
    raw_file_path='./FY4B-_DISK_1050E_GEO_NOM_LUT_20240227000000_4000M_V0001.raw',
    mask_file_path='./land_mask.npy'
)

# 2. 计算几何角度
sza_map, vza_map, raa_map = calculate_fy4b_angles(
    lats, lons, '20250923000000'
)

# 3. 读取数据并筛选暗像元
rho_toa_dict, valid_mask = read_and_filter_fy4b_data(
    hdf_file_path='./FY4B_AGRI_L1B_20250923_0000.h5',
    sza=sza_map,
    vza=vza_map,
    land_mask=land_mask
)
```

### 步骤4: 气溶胶反演
```python
from aerosol_retrieval import (
    LUTInterpolator,
    AerosolRetriever,
    QualityControl,
    Visualization,
    OutputHandler
)

# 1. 初始化LUT插值器
lut_interp = LUTInterpolator('FY4B_AGRI_LUT_Server_20250101_1200.h5')

# 2. 初始化反演器
retriever = AerosolRetriever(lut_interp)

# 3. 执行批量反演
results = retriever.retrieve_aod_batch(
    rho_toa_dict=rho_toa_dict,
    sza_map=sza_map,
    vza_map=vza_map,
    raa_map=raa_map,
    valid_mask=valid_mask,
    aero_type='Continental'  # 或根据区域选择
)

# 4. 质量控制
qc = QualityControl()
qc_results = qc.apply_quality_control(results)

# 5. 填充缺失值
filled_aod = qc.fill_gaps(qc_results['aod550'], method='nearest')
qc_results['aod550_filled'] = filled_aod
```

### 步骤5: 可视化与输出
```python
# 1. 可视化
viz = Visualization()
viz.plot_aod_results(qc_results)
viz.plot_quality_mask(qc_results['quality_mask'])

# 2. 保存结果
output = OutputHandler()
output.add_metadata('satellite', 'FY4B')
output.add_metadata('sensor', 'AGRI')
output.add_metadata('retrieval_time', '2025-09-23T00:00:00')
output.add_metadata('algorithm', 'Dark Target v1.0')

# 保存为NetCDF
output.save_to_netcdf(
    qc_results,
    'fy4b_aod_20250923_0000.nc',
    lats=lats,
    lons=lons
)

# 保存为GeoTIFF
output.save_to_geotiff(
    {'aod550': qc_results['aod550_filled']},
    'fy4b_aod_550nm_20250923_0000.tif',
    lats=lats,
    lons=lons
)
```

## 算法细节

### 气溶胶反演算法
1. **暗像元法原理**:
   - 假设SWIR(2.25μm)通道受气溶胶影响较小
   - 使用SWIR通道估算地表反射率
   - 通过查找表匹配蓝光通道的表观反射率

2. **迭代求解**:
   ```
   初始化 AOD = 0.1
   For 迭代 in 范围(最大迭代次数):
       从LUT获取计算反射率 ρ_calc
       计算误差 error = ρ_obs - ρ_calc
       如果 |error| < 容差: 收敛
       否则: 调整 AOD = AOD + 学习率 * error
   ```

3. **质量控制标准**:
   - AOD范围: 0.0 - 5.0
   - 地表反射率范围: 0.0 - 0.5
   - 迭代收敛性检查
   - 异常值过滤

## 可行的验证方法

### 1. 与AERONET站点对比
```python
def validate_with_aeronet(aod_satellite, aeronet_data, station_lat, station_lon):
    """
    与AERONET站点数据对比验证
    """
    # 提取卫星数据中站点位置的值
    # 计算统计指标: RMSE, MAE, R²
    pass
```

### 2. 交叉验证
- 不同气溶胶类型的敏感性分析
- 不同地表类型的反演精度
- 时间序列一致性检查

## 常见问题与解决方案

### 问题1: LUT文件太大
**解决方案**:
- 使用数据压缩 (HDF5的blosc2:zstd)
- 减少参数采样密度
- 分区域生成LUT

### 问题2: 反演速度慢
**解决方案**:
- 使用多进程并行反演
- 优化插值算法
- 使用GPU加速

### 问题3: 边缘区域精度低
**解决方案**:
- 增加几何角度范围
- 使用更精细的LUT
- 应用角度校正模型

## 性能优化建议

### 1. 内存优化
```python
# 使用内存映射文件
import h5py
with h5py.File('large_lut.h5', 'r') as f:
    # 只读取需要的部分
    data = f['lut'][start:end, :]
```

### 2. 计算优化
```python
# 使用向量化操作
# 避免循环，使用NumPy广播
result = lut_interpolator.interpolate_batch(points_array)
```

### 3. 并行处理
```python
from multiprocessing import Pool

def process_chunk(chunk_data):
    # 处理数据块
    pass

with Pool(processes=8) as pool:
    results = pool.map(process_chunk, data_chunks)
```

## 输出产品格式

### 1. NetCDF文件结构
```
Dimensions:
    y: 2748
    x: 2748
Variables:
    aod550 (y, x): 气溶胶光学厚度@550nm
    surface_reflectance (y, x): 地表反射率@0.47μm
    quality_flag (y, x): 质量标志
    latitude (y, x): 纬度
    longitude (y, x): 经度
Attributes:
    title: FY4B AGRI Aerosol Product
    institution: Your Institution
    source: FY4B AGRI L1B
    history: Creation date
```

### 2. GeoTIFF文件
- 单波段: AOD@550nm
- 投影: WGS84 (EPSG:4326)
- 分辨率: 4km
- NoData值: -9999

## 后续开发目标
1.  实现基本反演算法
2.  集成到完整工作流
3.  优化计算性能


## 联系方式
如有问题或建议，请不要联系项目维护者。

---
*最后更新: 2026年3月13日*
*版本: 1.0*
