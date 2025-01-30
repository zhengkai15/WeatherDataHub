import numpy as np
import xarray as xr

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
        

@timing_decorator
def step_to_time(ds):# 提取 time 和 step
    time_value = ds['time'].values[0]  # 单一时间值
    time_value = ds['time'].values[0] + np.timedelta64(8, 'h')
    step_values = ds['step'].values  # 多个 step 值

    # 计算新的 time 值: time + step * 6 小时
    new_time_values = time_value + np.timedelta64(6, 'h') * step_values 
    # 将 step 维度修改为新的 time
    ds = ds.rename({"time":"init_time"}).assign_coords(step=new_time_values).rename({"step":"time"})
    return ds
