import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pyinterp


dir_path = os.path.dirname(os.path.abspath(__file__))
dir_data = os.path.join(dir_path,"data")

path_obs = [
    f"{dir_data}/dc_obs/2020a_SSH_mapping_NATL60_envisat.nc",
    f"{dir_data}/dc_obs/2020a_SSH_mapping_NATL60_geosat2.nc",
    f"{dir_data}/dc_obs/2020a_SSH_mapping_NATL60_jason1.nc",
    f"{dir_data}/dc_obs/2020a_SSH_mapping_NATL60_topex-poseidon_interleaved.nc",
]

ds_obs = [xr.open_dataset(path) for path in path_obs]
ds_ref = xr.open_mfdataset(f"{dir_data}/dc_ref/*.nc")
ds_ref = ds_ref.assign_coords({ 'lon':('lon', ds_ref.lon.data % 360) }) # convert longitudes to [0,360]
# Create a regular grid for lon and lat
lon_min, lon_max = ds_ref.lon.min().values,ds_ref.lon.max().values
lat_min, lat_max = ds_ref.lat.min().values,ds_ref.lat.max().values
time_min, time_max = ds_ref.time.min().values,ds_ref.time.max().values
dlon = dlat = 0.1 # °
dt = 1 # days
lon_regular = np.arange(lon_min, lon_max, dlon)
lat_regular = np.arange(lat_min, lat_max, dlat)
# lon_regular = np.linspace(lon_min, lon_max, 224)
# lat_regular = np.linspace(lat_min, lat_max, 224)
lon_grid, lat_grid = np.meshgrid(lon_regular, lat_regular)
time_regular = np.arange(time_min, time_max, np.timedelta64(dt, 'D'))
# Prepare a container for the new grid data
ssh_grid = np.empty((len(time_regular), lat_regular.size, lon_regular.size))

# Initialize binning object
binning = pyinterp.Binning2D(
    pyinterp.Axis(lon_regular, is_circle=True),
    pyinterp.Axis(lat_regular),
    )

# Interpolate data for each time step
for t_idx, time in enumerate(time_regular):

    binning.clear()

    # Loop on datasets
    lon1d = np.array([])
    lat1d = np.array([])
    ssh1d = np.array([])
    for _ds_obs in ds_obs:
        # Mask for the current time step within a tolerance
        mask = np.abs(_ds_obs.time - time) <= np.timedelta64(int(5*24*dt/2), 'h')
        _ds_obs_mask = _ds_obs.where(mask, drop=True)
        ssh1d = np.concatenate((ssh1d,_ds_obs_mask.ssh_model.values))
        lon1d = np.concatenate((lon1d,_ds_obs_mask.lon.values))
        lat1d = np.concatenate((lat1d,_ds_obs_mask.lat.values))

    # Binning data
    binning.push(lon1d,lat1d,ssh1d, True)

    ssh_grid[t_idx, :, :] = binning.variable('mean').T


# Create a new xarray dataset
ds_inputs = xr.Dataset(
    {
        "ssh": (["time", "lat", "lon"], ssh_grid)
    },
    coords={
        "time": time_regular,
        "lat": lat_regular,
        "lon": lon_regular
    }
)

# Save the new dataset
ds_inputs.to_netcdf(f"{dir_data}/data_inputs_5days.nc")


ds_inputs.close()
# Define source grid
x_source_axis = pyinterp.Axis(ds_ref.lon.values, is_circle=True)
y_source_axis = pyinterp.Axis(ds_ref.lat.values)
z_source_axis = pyinterp.TemporalAxis(ds_ref.time.values)
grid_source = pyinterp.Grid3D(x_source_axis, y_source_axis, z_source_axis, ds_ref.sossheig.T)

# Define target grid
time_target = z_source_axis.safe_cast(np.ascontiguousarray(time_regular))
z_target = np.tile(time_target,(lon_regular.size,lat_regular.size,1))
nt = len(time_regular)
x_target = np.repeat(lon_grid.transpose()[:,:,np.newaxis],nt,axis=2)
y_target = np.repeat(lat_grid.transpose()[:,:,np.newaxis],nt,axis=2)

# Interpolation
ssh_interp = pyinterp.trivariate(grid_source,
                                x_target.flatten(),
                                y_target.flatten(),
                                z_target.flatten(),
                                bounds_error=False).reshape(x_target.shape).T

# Create a new xarray dataset
ds_target = xr.Dataset(
    {
        "ssh": (["time", "lat", "lon"], ssh_interp)
    },
    coords={
        "time": time_regular,
        "lat": lat_regular,
        "lon": lon_regular
    }
)

# Save the new dataset
ds_target.to_netcdf(f"{dir_data}/data_target.nc")
ds_target.close()
