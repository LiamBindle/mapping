

import numpy as np
import xarray as xr
grids = {
    'C96': xr.open_dataset('/extra-space/sg-stats/June/C96/grid_box_outlines_and_centers.nc'),
    'C98': xr.open_dataset('/extra-space/sg-stats/June/C98/grid_box_outlines_and_centers.nc'),
    'C100': xr.open_dataset('/extra-space/sg-stats/June/C100/grid_box_outlines_and_centers.nc'),
    'S48': xr.open_dataset('/extra-space/sg-stats/June/S48/grid_box_outlines_and_centers.nc'),
    'C90': xr.open_dataset('/extra-space/sg-stats/June/C90/grid_box_outlines_and_centers.nc'),
}

data = {
    'C96': xr.open_dataset('/extra-space/sg-stats/June/C96/species.june.nc'),
    'C98': xr.open_dataset('/extra-space/sg-stats/June/C98/species.june.C96.nc'),
    'C100': xr.open_dataset('/extra-space/sg-stats/June/C100/species.june.C96.nc'),
    'S48': xr.open_dataset('/extra-space/sg-stats/June/S48/species.june.C96.nc'),
    'C90': xr.open_dataset('/extra-space/sg-stats/June/C90/species.june.C96.nc'),
}


import matplotlib.pyplot as plt
import sklearn.metrics


x = data['C96'].SpeciesConc_O3.isel(time=0, lev=slice(0, 20)).values #.where(data['S48'].max_intersect > 0.5).values
y = data['S48'].SpeciesConc_O3.isel(time=0, lev=slice(0, 20)).values #.where(data['S48'].max_intersect > 0.5).values
isfinite = np.isfinite(y)
y = y[isfinite].flatten()
x = x[isfinite].flatten()

#plt.scatter(np.sort(x), np.sort(y))
plt.scatter(x, y)

plt.text(0.05, 0.95,
    f"RMSE={np.sqrt(sklearn.metrics.mean_squared_error(x, y))}",
    transform=plt.gca().transAxes, horizontalalignment='left', verticalalignment='top')
plt.show()