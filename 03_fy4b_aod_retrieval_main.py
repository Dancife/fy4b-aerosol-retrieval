import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pyorbital import astronomy
from scipy.interpolate import RegularGridInterpolator

# ==========================================
# 1. 基础地理与静态掩膜加载
# ==========================================
def load_nav_and_mask(raw_file_path, mask_file_path):
    print("=== 1. 加载地理信息与静态海陆掩膜 ===")
    GRID_SIZE = 2748
    
    raw_data = np.fromfile(raw_file_path, dtype='<f8')
    lat_flat = raw_data[0::2]
    lon_flat = raw_data[1::2]
    lats = lat_flat.reshape((GRID_SIZE, GRID_SIZE), order='F').T
    lons = lon_flat.reshape((GRID_SIZE, GRID_SIZE), order='F').T 
    
    land_mask = np.load(mask_file_path)
    print("✅ 经纬度与海陆掩膜加载成功！")
    return lats, lons, land_mask

# ==========================================
# 2. 动态几何角度计算
# ==========================================
def calculate_fy4b_angles(lats, lons, utc_time_str):
    print(f"\n=== 2. 计算动态几何角度 ({utc_time_str}) ===")
    utc_time = datetime.strptime(utc_time_str, '%Y%m%d%H%M%S')
    valid_mask = (lats >= -90) & (lats <= 90) & (lons >= -180) & (lons <= 180)
    
    sun_alt_rad, sun_azi_rad = astronomy.get_alt_az(utc_time, lons, lats)
    SZA = 90.0 - np.degrees(sun_alt_rad)
    SAA = np.degrees(sun_azi_rad)
    
    SAT_LON = 105.0 
    Re = 6378.137
    H = 42164.0
    rs = H / Re

    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)
    sat_lon_rad = np.radians(SAT_LON)
    
    lon_diff = sat_lon_rad - lon_rad
    cos_beta = np.cos(lat_rad) * np.cos(lon_diff)
    sin_beta = np.sqrt(1 - cos_beta**2)
    
    vza_rad = np.arctan2(sin_beta, cos_beta - (1 / rs))
    VZA = np.degrees(vza_rad)
    VZA = np.where(cos_beta < (1 / rs), np.nan, VZA)

    vaa_rad = np.arctan2(np.sin(lon_diff), -np.sin(lat_rad) * np.cos(lon_diff))
    VAA = np.degrees(vaa_rad)
    VAA = (VAA + 360) % 360 

    RAA = np.abs(SAA - VAA) % 360
    RAA = np.where(RAA > 180, 360 - RAA, RAA)

    SZA[~valid_mask] = np.nan
    VZA[~valid_mask] = np.nan
    RAA[~valid_mask] = np.nan
    
    print("✅ 角度计算完成！")
    return SZA, VZA, RAA

# ==========================================
# 3. 读取观测数据 & 生成当次有效动态掩膜
# ==========================================
def read_and_filter_fy4b_data(hdf_file_path, sza, vza, land_mask):
    print("\n=== 3. 读取表观反射率 & 生成动态暗像元掩膜 ===")
    channels_needed = {'01': 'rho_047', '02': 'rho_065', '03': 'rho_0825', '06': 'rho_225'}
    toa_dict = {}
    
    with h5py.File(hdf_file_path, 'r') as f:
        for ch_str, name in channels_needed.items():
            data_name = f"NOMChannel{ch_str}"
            cal_name = f"CALChannel{ch_str}"
            
            dn_data = f['Data'][data_name][()] if 'Data' in f else f[data_name][()]
            cal_table = f['Calibration'][cal_name][()] if 'Calibration' in f else f[cal_name][()]
            
            physical_data = np.full(dn_data.shape, np.nan, dtype=np.float32)
            valid_dn_mask = dn_data < 65000
            physical_data[valid_dn_mask] = cal_table[dn_data[valid_dn_mask]]
            toa_dict[name] = physical_data
            
    r047, r065, r225 = toa_dict['rho_047'], toa_dict['rho_065'], toa_dict['rho_225']
    
    cond_valid = (r047 > 0) & (r065 > 0) & (r225 > 0)
    # 此处的 0 和 80 对应了你生成 LUT 时 np.arange(0, 81, 5) 的边界
    cond_geom = (sza >= 0) & (sza <= 80) & (vza >= 0) & (vza <= 80)
    cond_land = (land_mask == 1)
    cond_cloud = (r047 < 0.4)
    cond_dark = (r225 < 0.25)
    
    final_mask = cond_valid & cond_geom & cond_land & cond_cloud & cond_dark
    
    print(f"✅ 动态掩膜生成完毕！有效暗像元数量: {np.sum(final_mask)}")
    return toa_dict, final_mask

# ==========================================
# 4. 估算地表反射率 (GOES ATBD 算法)
# ==========================================
def estimate_surface_reflectance(toa_dict, sza_matrix, final_mask):
    print("\n=== 4. 估算地表反射率 (基于 GOES ATBD) ===")
    
    r047, r065 = toa_dict['rho_047'], toa_dict['rho_065']
    r0825, r225 = toa_dict['rho_0825'], toa_dict['rho_225']
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (r0825 - r065) / (r0825 + r065)
    
    rho_047_surf = np.full_like(r047, np.nan)
    rho_065_surf = np.full_like(r065, np.nan)
    
    m1 = final_mask & (ndvi >= 0.55)
    m2 = final_mask & (ndvi >= 0.3) & (ndvi < 0.55)
    m3 = final_mask & (ndvi >= 0.2) & (ndvi < 0.3)
    m4 = final_mask & (ndvi < 0.2)
    
    def apply_formula(mask, c1, c2, c3, c4, target_surf):
        if np.any(mask):
            target_surf[mask] = (c1 + c2 * sza_matrix[mask]) + (c3 + c4 * sza_matrix[mask]) * r225[mask]
            
    apply_formula(m1,  1.436330E-02,  2.060893E-04,  1.749239E-01, -2.859502E-03, rho_047_surf)
    apply_formula(m1,  1.374160E-02, -5.128175E-05,  2.761044E-01,  1.034823E-03, rho_065_surf)
    
    apply_formula(m2,  4.163894E-02, -2.147513E-04,  1.598440E-01,  7.401292E-04, rho_047_surf)
    apply_formula(m2,  2.990101E-02, -1.873911E-04,  4.602174E-01,  9.658934E-04, rho_065_surf)
    
    apply_formula(m3,  5.154307E-02,  5.679386E-05,  2.048702E-01, -7.064656E-04, rho_047_surf)
    apply_formula(m3,  5.179930E-02, -1.043257E-04,  4.937035E-01,  4.310074E-04, rho_065_surf)
    
    apply_formula(m4, -4.990575E-02,  2.138207E-03,  8.498076E-01, -1.179596E-02, rho_047_surf)
    apply_formula(m4, -3.397737E-02,  1.640336E-03,  1.087497E+00, -9.538776E-03, rho_065_surf)
    
    rho_047_surf = np.clip(rho_047_surf, 0.001, 1.0)
    rho_065_surf = np.clip(rho_065_surf, 0.001, 1.0)
    
    print("✅ 地表反射率估算完成！")
    return rho_047_surf, rho_065_surf

# ==========================================
# 5. 全矩阵代价函数寻优 (AOD 核心求解)
# ==========================================
def retrieve_aod_vectorized(lut_path, toa_dict, rho_047_surf, sza, vza, raa, final_mask, 
                            target_aero='Continental', target_alt=0.0, target_wave=0.470):
    print(f"\n=== 5. 气溶胶反演 (AOD) - {target_aero} ===")
    
    print("   -> 读取 LUT 字典并构建多维插值引擎...")
    df = pd.read_hdf(lut_path, key='lut')
    
    # 根据你生成 LUT 时的参数进行严格匹配
    sub_df = df[(df['aero_type'] == target_aero) & 
                (df['altitude_km'] == target_alt) & 
                (df['wavelength_um'] == target_wave)].copy()
    
    if sub_df.empty:
        raise ValueError(f"LUT 字典中未找到条件: {target_aero}, {target_alt}km, {target_wave}µm 的数据！")
        
    sub_df = sub_df.sort_values(by=['sza', 'oza', 'raa', 'aod550'])
    
    sza_nodes = np.sort(sub_df['sza'].unique())
    oza_nodes = np.sort(sub_df['oza'].unique())
    raa_nodes = np.sort(sub_df['raa'].unique())
    aod_nodes = np.sort(sub_df['aod550'].unique())
    
    grid_shape = (len(sza_nodes), len(oza_nodes), len(raa_nodes), len(aod_nodes))
    points = (sza_nodes, oza_nodes, raa_nodes, aod_nodes)
    
    interp_rho0 = RegularGridInterpolator(points, sub_df['path_radiance'].values.reshape(grid_shape), bounds_error=False, fill_value=None)
    interp_T = RegularGridInterpolator(points, sub_df['transmittance'].values.reshape(grid_shape), bounds_error=False, fill_value=None)
    interp_S = RegularGridInterpolator(points, sub_df['spherical_albedo'].values.reshape(grid_shape), bounds_error=False, fill_value=None)
    
    valid_count = np.sum(final_mask)
    print(f"   -> 正在为 {valid_count} 个有效像素寻找最优 AOD...")
    
    if valid_count == 0:
        print("   ⚠️ 警告: 当前图像没有有效的暗像元像素，跳过反演。")
        return np.full_like(sza, np.nan)

    # 提取一维有效像素
    sza_1d = sza[final_mask]
    vza_1d = vza[final_mask]
    raa_1d = raa[final_mask]
    surf_1d = rho_047_surf[final_mask]
    obs_toa_1d = toa_dict['rho_047'][final_mask]
    
    errors_matrix = []
    
    # 核心 Cost Function 循环：遍历所有 AOD 节点
    for aod in aod_nodes:
        pts = np.vstack((sza_1d, vza_1d, raa_1d, np.full_like(sza_1d, aod))).T
        
        rho0 = interp_rho0(pts)
        T = interp_T(pts)
        S = interp_S(pts)
        
        # 公式推导：模拟表观反射率
        sim_toa = rho0 + (surf_1d * T) / (1 - surf_1d * S)
        
        # 计算观测与模拟之间的绝对误差
        error = np.abs(sim_toa - obs_toa_1d)
        errors_matrix.append(error)
    
    # 找出每个像素误差最小的那一层对应的 AOD 值
    errors_matrix = np.array(errors_matrix)
    best_aod_indices = np.argmin(errors_matrix, axis=0)
    best_aod_1d = aod_nodes[best_aod_indices]
    
    # 组装回 2D 圆盘矩阵
    final_aod_2d = np.full_like(sza, np.nan)
    final_aod_2d[final_mask] = best_aod_1d
    
    print("✅ AOD 反演大功告成！")
    return final_aod_2d

# ==========================================
# 6. 主控制节点 (Main 流水线)
# ==========================================
if __name__ == "__main__":
    # ================= ⚠️ 用户配置区 ⚠️ =================
    RAW_FILE = "FY4B-_DISK_1050E_GEO_NOM_LUT_20240227000000_4000M_V0001.raw"
    MASK_FILE = "FY4B_LandMask_4km.npy"
    # 根据你提供的生成脚本，假设你的 LUT 命名是下面这个：
    LUT_FILE = "FY4B_AGRI_LUT_Server.h5" 
    
    # 你每次想处理的新卫星影像及其对应时间
    HDF_FILE = "FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20250923000000_20250923001459_4000M_V0001.HDF"
    OBS_TIME = "20250923000000" 
    # ====================================================

    # 1-4. 数据预处理
    lats, lons, land_mask = load_nav_and_mask(RAW_FILE, MASK_FILE)
    sza, vza, raa = calculate_fy4b_angles(lats, lons, OBS_TIME)
    toa_dict, final_mask = read_and_filter_fy4b_data(HDF_FILE, sza, vza, land_mask)
    rho_047_surf, rho_065_surf = estimate_surface_reflectance(toa_dict, sza, final_mask)

    # 5. 反演求解 (注意：默认使用 Continental 大陆型气溶胶)
    aod_result = retrieve_aod_vectorized(
        lut_path=LUT_FILE, 
        toa_dict=toa_dict, 
        rho_047_surf=rho_047_surf, 
        sza=sza, vza=vza, raa=raa, 
        final_mask=final_mask,
        target_aero='Continental' # 与你生成代码里 aero_name_map 的键值保持一致
    )

    # 6. 可视化战果
    print("\n=== 正在生成 AOD 最终产品图 ===")
    plt.figure(figsize=(10, 10))
    # AOD 通常色彩映射使用 jet，高污染区显示为红黄，洁净区显示为蓝
    im = plt.imshow(aod_result, cmap='jet', vmin=0.0, vmax=2.0)
    plt.colorbar(im, fraction=0.046, pad=0.04, label='Aerosol Optical Depth (AOD @ 0.47 µm)')
    plt.title(f"FY-4B Retrieved AOD (Dark Target) - {OBS_TIME}", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()