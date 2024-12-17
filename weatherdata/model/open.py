import numpy as np
import xarray as xr

def update_dims(ds, level_type='surface'):
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
    ds_surface = update_dims(ds_surface, level_type='surface')  # east china
    ds_surface_ptype = update_dims(ds_surface_ptype, level_type='pl')  # east china
    ds_surface = ds_surface.to_array().to_dataset(name="data").rename(variable="channel")
    ds_surface_ptype = ds_surface_ptype.to_array().to_dataset(name="data").rename(variable="channel")
    ds_surface = xr.concat([ds_surface, ds_surface_ptype], dim="channel").drop_vars(["surface","step","number"])

    # Process pressure level data for global coverage
    ds_pl = xr.open_dataset(file_path, engine="cfgrib", backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'edition': 1}})
    ds_pl = update_dims(ds_pl, level_type='pl')
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
    ds_surface = update_dims(ds_surface, level_type='surface').drop_vars(["surface","step"])
    ds_surface = ds_surface.to_array().to_dataset(name="data").rename(variable="channel")

    # Process pressure level data
    ds_pl = xr.open_dataset(file_path, engine="cfgrib", backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}})
    ds_pl = update_dims(ds_pl,level_type='surface')
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