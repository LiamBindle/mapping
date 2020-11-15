
import xarray as xr
grids = {
    'C96': xr.open_dataset('/extra-space/sg-stats/Sept/S48/grid_box_outlines_and_centers.nc'),
}

import stats_stuff.open_ds

import maps

import matplotlib.pyplot as plt
import matplotlib.cm
import cartopy.crs as ccrs
import numpy as np
ax = plt.axes(projection=ccrs.epsg(2163))
ax.coastlines()

data = stats_stuff.open_ds.open_ds('S48')
data2 = stats_stuff.open_ds.open_ds('C94', 'C96')

ax.set_extent([-91, -84, 40.7, 46])
maps.outlines(
    ax, coastlines=False, borders=False, states=True, lakes=False,
    linewidth=0.4, edgecolor='white'  #edgecolor=matplotlib.colors.to_rgba('snow', 0.9)
)
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
    v = data.SpeciesConc_NO2.isel(time=0, nf=i, lev=20)
    # v = np.log10(v)
    vmin=3e-11
    vmax=7e-11
    ax.pcolormesh(xe.values, ye.values, v, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())

plt.scatter([-87.6298], [41.8781], zorder=400, transform=ccrs.PlateCarree(), c='white')
ax.annotate('Chicago', xy=(-87.8,42), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
            ha='left', va='bottom', color='white')

# cv = (data.SpeciesConc_O3.isel(time=0, lev=20).std()/data.SpeciesConc_O3.isel(time=0, lev=20).mean()).item()
# rmse = np.nanstd(data2.SpeciesConc_O3.isel(time=0, lev=20) - data.SpeciesConc_O3.isel(time=0, lev=20))/(0.5*(data.SpeciesConc_O3.isel(time=0, lev=20)+data2.SpeciesConc_O3.isel(time=0, lev=20))).mean()
# mb = np.nanmean(data2.SpeciesConc_O3.isel(time=0, lev=20) - data.SpeciesConc_O3.isel(time=0, lev=20))/(0.5*(data.SpeciesConc_O3.isel(time=0, lev=20)+data2.SpeciesConc_O3.isel(time=0, lev=20))).mean()

# print('CV: ', cv)
# print('RMSE/mu: ', rmse)
# print('MB/mu: ', mb)

plt.colorbar(matplotlib.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax)))

plt.show()
