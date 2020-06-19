
import xarray as xr
grids = {
    'C96': xr.open_dataset('/extra-space/sg-stats/June/C96/grid_box_outlines_and_centers.nc'),
    'C98': xr.open_dataset('/extra-space/sg-stats/June/C98/grid_box_outlines_and_centers.nc'),
    'C100': xr.open_dataset('/extra-space/sg-stats/June/C100/grid_box_outlines_and_centers.nc'),
    'S48': xr.open_dataset('/extra-space/sg-stats/June/S48/grid_box_outlines_and_centers.nc'),
}

# data = xr.open_dataset('/extra-space/sg-stats/June/C98/species.june.C96.nc')
# data = xr.open_dataset('/extra-space/sg-stats/June/C98/species.june.nc')
# data = xr.open_dataset('/extra-space/sg-stats/June/C100/species.june.C96.nc')
data = xr.open_dataset('/extra-space/sg-stats/June/S48/species.june.C96.nc') - xr.open_dataset('/extra-space/sg-stats/June/C96/species.june.nc')
# data = xr.open_dataset('/extra-space/sg-stats/June/C90/species.june.C96.nc') - xr.open_dataset('/extra-space/sg-stats/June/C96/species.june.nc')

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
# ax.set_global()
for i in range(6):
    xe = grids['C96'].xe.isel(nf=i)
    ye = grids['C96'].ye.isel(nf=i)
    v = data.SpeciesConc_O3.isel(time=0, nf=i, lev=0)
    v = (v - v.mean())**2
    v = np.log10(v)
    ax.pcolormesh(xe.values, ye.values, v, vmin=-19, vmax=-17, transform=ccrs.PlateCarree())
plt.show()
