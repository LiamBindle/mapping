
import xarray as xr
grids = {
    'C96': xr.open_dataset('/extra-space/sg-stats/June/C96/grid_box_outlines_and_centers.nc'),
    'C98': xr.open_dataset('/extra-space/sg-stats/June/C98/grid_box_outlines_and_centers.nc'),
    'C100': xr.open_dataset('/extra-space/sg-stats/June/C100/grid_box_outlines_and_centers.nc'),
    'S48': xr.open_dataset('/extra-space/sg-stats/June/S48/grid_box_outlines_and_centers.nc'),
}

import stats_stuff.open_ds

import matplotlib.pyplot as plt
import matplotlib.cm
import cartopy.crs as ccrs
import numpy as np
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()

data = stats_stuff.open_ds.open_ds('C96', 'C96')
data2 = stats_stuff.open_ds.open_ds('S48', 'C96')

# ax.set_global()

# for i in range(6):
#     xe = grids['C96'].xe.isel(nf=i)
#     ye = grids['C96'].ye.isel(nf=i)
#     v = data.SpeciesConc_O3.isel(time=0, nf=i, lev=20)
#     v = (v - v.mean())**2
#     v = np.log10(v)
#     ax.pcolormesh(xe.values, ye.values, v, vmin=-19, vmax=-17, transform=ccrs.PlateCarree())
for i in range(6):
    xe = grids['C96'].xe.isel(nf=i)
    ye = grids['C96'].ye.isel(nf=i)
    v = data.SpeciesConc_O3.isel(time=0, nf=i, lev=20)
    # v = np.log10(v)
    vmin=0.4e-7
    vmax=0.8e-7
    ax.pcolormesh(xe.values, ye.values, v, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())

cv = (data.SpeciesConc_O3.isel(time=0, lev=20).std()/data.SpeciesConc_O3.isel(time=0, lev=20).mean()).item()
rmse = np.nanstd(data2.SpeciesConc_O3.isel(time=0, lev=20) - data.SpeciesConc_O3.isel(time=0, lev=20))/(0.5*(data.SpeciesConc_O3.isel(time=0, lev=20)+data2.SpeciesConc_O3.isel(time=0, lev=20))).mean()
mb = np.nanmean(data2.SpeciesConc_O3.isel(time=0, lev=20) - data.SpeciesConc_O3.isel(time=0, lev=20))/(0.5*(data.SpeciesConc_O3.isel(time=0, lev=20)+data2.SpeciesConc_O3.isel(time=0, lev=20))).mean()

print('CV: ', cv)
print('RMSE/mu: ', rmse)
print('MB/mu: ', mb)

plt.colorbar(matplotlib.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax)))

plt.show()
