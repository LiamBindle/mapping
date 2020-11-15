


import numpy as np
import xarray as xr
import shapely.geometry
import maps



grid = xr.open_dataset('/home/liam/Downloads/scratch/C900e-CA/comparison_grid.nc')
xc = grid['grid_boxes_centers'].isel(XY=0).values
yc = grid['grid_boxes_centers'].isel(XY=1).values


bad_area = shapely.geometry.Polygon([
    (-118.7759399, 34.7799717),
    (-118.1442261, 34.2810491),
    (-116.9796753, 34.2118022),
    (-116.8945313, 34.9354820),
    (-118.3090210, 35.0457382),
    (-118.7759399, 34.7799717),
])

bad_mask = maps.mask_outside(xc, yc, bad_area)

data = xr.open_dataset('/extra-space/foo2.nc')

NO2_mean = np.stack([data.SPC_NO2.mean(dim=['nf', 'Ydim', 'Xdim']).values for _ in range(85)], axis=0)
NO_mean = np.stack([data.SPC_NO.mean(dim=['nf', 'Ydim', 'Xdim']).values for _ in range(85)], axis=0)

data.SPC_NO2.values[~bad_mask] = NO_mean
data.SpeciesConc_NO2.values[~bad_mask] = NO2_mean

data.to_netcdf('foo3.nc')
