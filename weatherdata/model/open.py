import os
import numpy as np
import xarray as xr
from datetime import datetime

def update_dims_grib(ds, level_type='surface'):
    ds = ds.expand_dims(["valid_time", "time"])
    name_dict = {
        "lead_time": "step",
        "prediction_timedelta": "step",
        "isobaricInhPa": "level",
        "latitude": "lat",
        "longitude": "lon",
        "number": "member",
        "time":"init_time",
        "valid_time":"time"
    }
    if isinstance(ds, xr.DataArray):
        dims = {k: name_dict.get(k, k) for k in ds.dims}
    else:
        dims = {k: name_dict.get(k, k) for k in ds.sizes}
    ds = ds.rename(dims)
    ds = ds.assign_coords({'lat': ds.lat.values.astype(np.float32)})
    ds = ds.assign_coords({'lon': ds.lon.values.astype(np.float32)})
    if level_type == "surface":
        if 'u100' in ds:
            ds = ds.rename({'u100':'u100m'})
        if 'v100' in ds:
            ds = ds.rename({'v100': 'v100m'})
        if 'u10' in ds:
            ds = ds.rename({'u10':'u10m'})
        if 'v10' in ds:
            ds = ds.rename({'v10': 'v10m'})
    return ds

def process_cmc_grib_data(file_path):
    """
    Processes GRIB data for East China and global pressure levels.

    Parameters:
    file_path (str): The path to the GRIB file to be processed.

    Returns:
    tuple: A tuple containing two xarray.Dataset objects. The first dataset represents the surface data for East China, and the second dataset represents the pressure level data for global coverage.
    """
    # Process surface data for East China
    ds_surface = xr.open_dataset(file_path, engine="cfgrib", backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface', 'edition': 1}})
    ds_surface_ptype = xr.open_dataset(file_path, engine="cfgrib", backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface', 'edition': 2}})
    ds_surface = update_dims_grib(ds_surface, level_type='surface')  # east china
    ds_surface_ptype = update_dims_grib(ds_surface_ptype, level_type='pl')  # east china
    ds_surface = ds_surface.to_array().to_dataset(name="data").rename(variable="channel")
    ds_surface_ptype = ds_surface_ptype.to_array().to_dataset(name="data").rename(variable="channel")
    ds_surface = xr.concat([ds_surface, ds_surface_ptype], dim="channel").drop_vars(["surface","step","number"])

    # Process pressure level data for global coverage
    ds_pl = xr.open_dataset(file_path, engine="cfgrib", backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'edition': 1}})
    ds_pl = update_dims_grib(ds_pl, level_type='pl')
    ds_pl = ds_pl.to_array().to_dataset(name="data").rename(variable="channel")
    levels = ds_pl.to_array().level.values
    print(levels)
    ds_ps_res = []
    for var_pl_cur in ds_pl.to_array().channel.values:
        ds_pl_var_cur = ds_pl.sel(channel=[var_pl_cur])
        for level_cur in levels:
            level_cur = int(level_cur)
            ds_var_cur_level_cur = ds_pl_var_cur.sel(level=level_cur).assign_coords(channel=[f"{var_pl_cur}{level_cur}"])
            ds_ps_res.append(ds_var_cur_level_cur)
    ds_pl = xr.concat(ds_ps_res, dim="channel").drop_vars(["level","step","number"])  # global
    return ds_surface, ds_pl


def process_ec_grib_data(file_path):
    """
    Processes GRIB data for surface and pressure levels.

    Parameters:
    file_path (str): The path to the GRIB file to be processed.

    Returns:
    xarray.Dataset: A concatenated dataset containing both surface and pressure level data.
    """
    # Process surface data
    ds_surface = xr.open_dataset(file_path, engine="cfgrib", backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})
    ds_surface = update_dims_grib(ds_surface, level_type='surface').drop_vars(["surface","step"])
    ds_surface = ds_surface.to_array().to_dataset(name="data").rename(variable="channel")

    # Process pressure level data
    ds_pl = xr.open_dataset(file_path, engine="cfgrib", backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}})
    ds_pl = update_dims_grib(ds_pl,level_type='surface')
    ds_pl = ds_pl.to_array().to_dataset(name="data").rename(variable="channel")
    levels = ds_pl.to_array().level.values
    print(levels)
    ds_ps_res = []
    for var_pl_cur in ds_pl.to_array().channel.values:
        ds_pl_var_cur = ds_pl.sel(channel=[var_pl_cur])
        for level_cur in levels:
            level_cur = int(level_cur)
            ds_var_cur_level_cur = ds_pl_var_cur.sel(level=level_cur).assign_coords(channel=[f"{var_pl_cur}{level_cur}"])
            ds_ps_res.append(ds_var_cur_level_cur)
    ds_pl = xr.concat(ds_ps_res, dim="channel").drop_vars(["level","step"])
    # Update dimensions and drop unnecessary variables
    # Concatenate datasets
    ds = xr.concat([ds_surface,ds_pl], dim="channel")  # east china
    return ds

var_needed_dic = dict(
    t2m='2m_temperature',
    u10m='10m_u_component_of_wind',
    v10m='10m_v_component_of_wind',
    msl='mean_sea_level_pressure',
    tp='total_precipitation',
    d2m='2m_dewpoint_temperature',
    sst='sea_surface_temperature',
    u100m='100m_u_component_of_wind',
    v100m='100m_v_component_of_wind',
    lcc='low_cloud_cover',
    mcc='medium_cloud_cover',
    hcc='high_cloud_cover',
    tcc='total_cloud_cover',
    ssr='surface_net_solar_radiation',
    ssrd='surface_solar_radiation_downwards',
    fdir='total_sky_direct_solar_radiation_at_surface',
    ttr='top_net_thermal_radiation',
    mdts='mean_direction_of_total_swell',
    mdww='mean_direction_of_wind_waves',
    mpts='mean_period_of_total_swell',
    mpww='mean_period_of_wind_waves',
    shts='significant_height_of_total_swell',
    shww='significant_height_of_wind_waves'
)

var_needed_pl_dic = dict(
    z='geopotential',
    t='temperature',
    u='u_component_of_wind',
    v='v_component_of_wind',
    q='specific_humidity',
)

levels=[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


import os
import xarray as xr
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from dask.diagnostics import ProgressBar


def chunk(ds, channel):
    '''
    chunk 可以控制数据方式，将数据分块进行延迟加载和并行计算
    :param ds:
    :return:
    '''
    dims = {k: v for k, v in ds.dims.items()}
    dims['time'] = 1
    if channel in ds.dims:
        dims[channel] = 1
    ds = ds.chunk(dims)
    return ds


def update_dims_era5(ds):
    if 'valid_time' in ds.dims:
        ds = ds.rename({'valid_time': 'time'})
    if 'pressure_level' in ds.dims:
        ds = ds.rename({'pressure_level': 'level'})
    return ds

def transform_concat(var_ds, var_name):
    var_ds = chunk(var_ds, 'level')
    var_ds = var_ds.to_array()
    var_ds = var_ds.expand_dims({'level': [1]}, axis=1)
    var_ds = var_ds.rename({'level': 'channel', 'latitude': 'lat', 'longitude': 'lon'})
    var_ds = var_ds.assign_coords(channel=[var_name], variable=['data'])
    var_ds = var_ds.to_dataset(name='data')
    return chunk(var_ds, 'channel').data

def generate_era5(basename="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/database/era5/from_official",init_time="2024-11-09 00:00:00",fcst_days=2):
    '''
    init_time:2024-11-09 00:00:00'
    basename:/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/database/era5/from_official
    '''
    init_time = pd.to_datetime(init_time)
    
    def get_filenames(basename, fcst_days=fcst_days):
        file_names = os.listdir(basename)
        file_needed = [f'{(init_time - pd.Timedelta("1D")).strftime("%Y%m%d.nc")}']
        # 读取未来
        for fcst_day in range(fcst_days+1):
            file_needed.append(f'{(init_time + pd.Timedelta(f"{fcst_day}D")).strftime("%Y%m%d.nc")}')
        return [os.path.join(basename, file_name) for file_name in file_needed if file_name in file_names]

    # 去重
    years = [(init_time - pd.Timedelta("1D")).strftime("%Y")]
    for fcst_day in range(fcst_days):
        years.append((init_time + pd.Timedelta(f"{fcst_day}D")).strftime("%Y"))

    years = list(set(years))
    # sfc
    for idx, var in enumerate(var_needed_dic):
        var_need_path = os.path.join(basename, 'surface_level')
        var_ds = []
        for year_idx, year in enumerate(years):
            var_path = os.path.join(var_need_path, var_needed_dic.get(var), f'{year}')
            print(var, var_path)
            var_year_idx = xr.open_mfdataset(get_filenames(var_path))
            var_year_idx = update_dims_era5(var_year_idx)
            if ('waves' in var_needed_dic.get(var) or 'swell' in var_needed_dic.get(var)):
                # print(var_needed_dic.get(var))
                var_year_idx = var_year_idx.interp(longitude=np.arange(0, 360, 0.25),
                                                   latitude=np.arange(90, -90.1, -0.25))
            var_ds.append(var_year_idx)
        var_ds = xr.concat(var_ds, dim='time')
        var_ds = transform_concat(var_ds, var)
        var_ds = var_ds.drop_vars("expver")
        if idx == 0:
            data = var_ds
        else:
            data = xr.concat([data, var_ds], dim='channel')
    # pl
    var_need_path = os.path.join(basename, 'pressure_level')
    for idx, var in enumerate(var_needed_pl_dic):
        var_ds = []
        for year_idx, year in enumerate(years):
            var_path = os.path.join(var_need_path, var_needed_pl_dic.get(var), f'{year}')
            print(var, var_path)
            var_year_idx = xr.open_mfdataset(get_filenames(var_path))
            var_year_idx = update_dims_era5(var_year_idx)
            var_ds.append(var_year_idx)
        var_ds_all_levels = xr.concat(var_ds, dim='time')
        for level in levels:
            var_ds = var_ds_all_levels.sel(level=level).drop_vars(['expver', 'number'])
            var_ds = transform_concat(var_ds, str(var) + str(level))
            data = xr.concat([data, var_ds], dim='channel')

    try:
        data = data.mean(dim=['expver']).transpose('variable', 'time', 'channel', 'lat', 'lon')
    except:
        data = data.transpose('variable', 'time', 'channel', 'lat', 'lon')
    data  = data.isel(variable=0).drop_vars("variable").to_dataset(name="data")
    return data


# ec fc0
import pygrib
def extract_and_save_ec_time_slice(grib_file_path, output_file_path, date_index, time_index):
    # date_index :20240101
    # time_index : {'00': 0, '06': 600, '12': 1200, '18': 1800}.values
    
    # 打开 GRIB 文件
    with pygrib.open(grib_file_path) as grbs:
        # 获取指定时间片的消息
        grb_messages = [grb for grb in grbs if (grb['dataDate'] == int(date_index) and grb['time'] == time_index)]

    if not grb_messages:
        raise ValueError(f"No data found for time index {date_index}")

    # 创建一个新的 GRIB 文件并写入提取的消息
    with open(output_file_path, 'wb') as output_file:
        for grb in grb_messages:
            output_file.write(grb.tostring())


def get_save_name_pattern(flag, date_index):
    # date_index :20240101
    # flags = ['00', '06', '12', '18']

    # save_name_pattern
    if flag in ['00', '12']:
        init_time_path = '00_12'
        save_name_pattern = {
            'pl': f'A1D{date_index[-4:]}{flag}00{date_index[-4:]}{flag}011',
            'sfc': f'A2D{date_index[-4:]}{flag}00{date_index[-4:]}{flag}011',
            'ocean': f'T3P{date_index[-4:]}{flag}00{date_index[-4:]}{flag}011',
        }
    elif flag in ['06', '18']:
        init_time_path = '06_18'
        save_name_pattern = {
            'pl': f'A1S{date_index[-4:]}{flag}00{date_index[-4:]}{flag}011',
            'sfc': f'A2S{date_index[-4:]}{flag}00{date_index[-4:]}{flag}011',
            'ocean': f'T3Q{date_index[-4:]}{flag}00{date_index[-4:]}{flag}011',
        }
    else:
        raise ValueError("Invalid flag")
    return save_name_pattern, init_time_path


# fuxi-ens
def merge_output(output_dir, init_time, save_type='nc', member_max_num=2, step_min_num=0, step_max_num=8):
    save_name = os.path.join(output_dir, f'{init_time.strftime("%Y%m%d-%H")}', f"output.nc")

    if os.path.exists(save_name):
        # os.remove(save_name)
        return
    else:        
        # 使用 glob 匹配所有文件，并筛选出需要的文件
        file_names = sorted(
            glob.glob(os.path.join(output_dir, init_time.strftime("%Y%m%d-%H"), "member_[0-9][0-9][0-9]/[0-9][0-9][0-9].nc"), recursive=True)
        )
        # 过滤出 member_000 到 member_{member_max} 的文件夹，并匹配步数范围内的文件
        file_names = [
            file for file in file_names
            if 0 <= int(os.path.basename(os.path.dirname(file)).split('_')[1]) <= member_max_num + 1
            and step_min_num <= int(os.path.basename(file).split('.')[0]) <= step_max_num
        ]
        assert len(file_names) > 0, f"No files found in {output_dir}"

        ds = xr.open_mfdataset(
            file_names, engine="zarr" if save_type == "zarr" else "netcdf4",
        ).chunk({'time': 1, 'step': 1})
        # ds = ensemble_mean(ds)
        # save_with_progress(ds, save_name)
        ds = ds.chunk(dict(member=-1))
        return ds

def step_to_time(ds):# 提取 time 和 step
    time_value = ds['time'].values[0]  # 单一时间值
    time_value = ds['time'].values[0] + np.timedelta64(8, 'h')
    step_values = ds['step'].values  # 多个 step 值

    # 计算新的 time 值: time + step * 6 小时
    new_time_values = time_value + np.timedelta64(6, 'h') * step_values 
    # 将 step 维度修改为新的 time
    ds = ds.rename({"time":"init_time"}).assign_coords(step=new_time_values).rename({"step":"time"})
    return ds



# hres
def read_hres_fcst(file_path):
    ds = xr.open_dataset(file_path, engine='cfgrib',backend_kwargs={'filter_by_keys':{'typeOfLevel':'surface', 'edition': 1}})[["u10","v10"]]
    ds = update_dims_grib(ds, level_type='surface')
    ds = ds.to_array().to_dataset(name="data").rename(variable="channel").drop_vars(["surface","step","number"])
    ds = ds.assign_coords(lon=((ds.lon + 360) % 360))
    ds = ds.sortby('lon', ascending=True)
    return ds

def get_hres_file_ls(init_time, data_path, file_type):
    # file_type : 数据类型 A1 A2 T1 T2 T3 
    fn_path = os.path.join(data_path, init_time.strftime("%Y%m%d%H"))
    try:
        os.chmod(fn_path, 0o777)  # 八进制表示法
    except PermissionError:
        print(f"Permission denied: unable to change permissions for {fn_path}")
    except FileNotFoundError:
        print(f"Folder not found: {fn_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
    fn_ls = [[init_time.strftime("%Y")+ifn[11:-3], os.path.join(fn_path, ifn)] for ifn in os.listdir(fn_path) if ifn.startswith(file_type) and ifn.endswith("001")]
    timeStamp_dict = {}
    timeStamp_ls = []
    fn_ls.sort()
    for ifn_ in fn_ls:
        if ifn_[0].endswith("."):
            continue
        elif int(ifn_[0][-2:]) % 1 == 0:
            itstamp = np.datetime64(datetime.strptime(ifn_[0], '%Y%m%d%H'))
        else:
            continue
        timeStamp_dict[itstamp] = ifn_[1]
        timeStamp_ls.append(itstamp)
    file_ls = []
    for ikey, ival in timeStamp_dict.items():
        file_ls.append(ival)
    timeStamp_ls.sort()
    return file_ls

def hres_data_read(fn_ls, N_CORE=10):
    from multiprocessing import Pool
    from tqdm import tqdm
    pool = Pool(N_CORE)
    pbar = tqdm(total=len(fn_ls))
    pbar.set_description(f'ec data reading')
    update = lambda *args: pbar.update()
    processes = [pool.apply_async(read_hres_fcst, args=(filename,),  callback=update) for filename in fn_ls]
    ds_all = []
    for values in processes:
        ds_all.append(values.get())
    ds_all = xr.concat(ds_all, dim='time')
    ds_all = ds_all.sortby('time')
    return ds_all


def hres_data_processing(data_path, init_time):
    fn_ls = get_hres_file_ls(init_time=init_time, data_path=data_path, file_type='T2')
    ds = hres_data_read(fn_ls)
    ds = chunk(ds, channel="channel")
    return ds

# ens

def get_ens_file_ls(init_time, ens_path):
    fn_path = os.path.join(ens_path, init_time.strftime("%Y%m%d%H"))
    try:
        os.chmod(fn_path, 0o777)  # 八进制表示法
    except PermissionError:
        print(f"Permission denied: unable to change permissions for {fn_path}")
    except FileNotFoundError:
        print(f"Folder not found: {fn_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
    fn_ls = [[init_time.strftime("%Y")+ifn[11:-3], os.path.join(fn_path, ifn)] for ifn in os.listdir(fn_path)]
    timeStamp_dict = {}
    timeStamp_ls = []
    fn_ls.sort()
    for ifn_ in fn_ls:
        if ifn_[0].endswith("."):
            continue
        elif int(ifn_[0][-2:])%6==0:
            itstamp = np.datetime64(datetime.strptime(ifn_[0], '%Y%m%d%H'))
        else:
            continue
        timeStamp_dict[itstamp] = ifn_[1]
        timeStamp_ls.append(itstamp)
    file_ls = []
    for ikey, ival in timeStamp_dict.items():
        file_ls.append(ival)
    timeStamp_ls.sort()
    return file_ls

def ens_read(ifn):
    ds_0 = xr.open_dataset(ifn, engine='cfgrib',backend_kwargs={'filter_by_keys':{'typeOfLevel':'surface', 'dataType':'pf', 'edition':1}})
    ds_1 = xr.open_dataset(ifn, engine='cfgrib',backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface', 'dataType': 'cf', 'edition':1}})
    ds = xr.concat([ds_0, ds_1], dim='number')
    ds = update_dims(ds)
    return ds

# # simgle processing
# def ens_data_read(fn_ls,):
#     ds_all = None
#     for idx,ifn_ in enumerate(fn_ls):
#         ids = ens_read(ifn_)
#         if ds_all is None:
#             ds_all = ids
#         else:
#             ds_all = xr.concat([ds_all, ids], dim='time')
#             # break
#     ds_all = ds_all.sortby('time')
#     return ds_all

# multiprocessing
def ens_data_read(fn_ls, N_CORE=10):
    from multiprocessing import Pool
    from tqdm import tqdm
    pool = Pool(N_CORE)
    pbar = tqdm(total=len(fn_ls))
    pbar.set_description(f'ec data reading')
    update = lambda *args: pbar.update()
    processes = [pool.apply_async(ens_read, args=(filename,),  callback=update) for filename in fn_ls]
    ds_all = []
    for values in processes:
        ds_all.append(values.get())
    ds_all = xr.concat(ds_all, dim='time')
    ds_all = ds_all.sortby('time')
    ds_all = ds_all.drop_vars(['step','surface']).to_array().rename({"variable":"channel"}).to_dataset(name='data')
    return ds_all


def ens_data_processing(init_time, ens_path, N_CORE=10, member_max_num=51,step_min_num=0,step_max_num=2, zone=None):
    fn_ls = get_ens_file_ls(init_time, ens_path)[step_min_num:step_max_num+1] # step_min_num, step_max_num+2,right=True 去掉一个初始场
    ds = ens_data_read(fn_ls, N_CORE).isel(member=slice(0,member_max_num))
    ds = deal_tp(ds).isel(time=slice(1, None)) # 降水diff之后去掉第一个时间片
    ds = deal_10m(ds)
    # ds = append_ws(ds)# 服务代码中去增加风速
    ds = ds.chunk({'time': 1, 'init_time': 1})
    ds = ds.chunk(dict(member=-1))
    lon_min, lon_max, lat_min, lat_max = zone
    new_lat = np.arange(lat_min, lat_max+0.25, 0.25)[::-1]  # 0.25° 分辨率的纬度
    new_lon = np.arange(lon_min, lon_max+0.25, 0.25)  # 0.25° 分辨率的经度
    ds = ds.interp(lat=new_lat, lon=new_lon, method="nearest")
    return ds