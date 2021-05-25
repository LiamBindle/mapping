
import xarray as xr
print('1')
grids = {
    'C96': xr.open_dataset('/extra-space/sg-stats/Sept/C96/grid_box_outlines_and_centers.nc'),
}
print('2')
import stats_stuff.open_ds

import maps

import matplotlib.pyplot as plt
import matplotlib.cm
import cartopy.crs as ccrs
import numpy as np
from tqdm import tqdm
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
print('3')
#data = stats_stuff.open_ds.open_ds('S48')
#data2 = stats_stuff.open_ds.open_ds('C94', 'C96')
#data = stats_stuff.open_ds.open_percent_diff('S48', 'C96')
data1 = stats_stuff.open_ds.open_ds('S48', 'C96')
data2 = stats_stuff.open_ds.open_ds('C96', 'C96')
print('4')
#ax.set_extent([-91, -84, 40.7, 46])
ax.set_extent([-130, -65, 20, 50])
maps.outlines(
    ax, coastlines=False, borders=False, states=True, lakes=False,
    linewidth=0.4, edgecolor='white'  #edgecolor=matplotlib.colors.to_rgba('snow', 0.9)
)
print('5')
# ax.set_global()

# for i in range(6):
#     xe = grids['C96'].xe.isel(nf=i)
#     ye = grids['C96'].ye.isel(nf=i)
#     v = data.SpeciesConc_O3.isel(time=0, nf=i, lev=20)
#     v = (v - v.mean())**2
#     v = np.log10(v)
#     ax.pcolormesh(xe.values, ye.values, v, vmin=-19, vmax=-17, transform=ccrs.PlateCarree())
for i in tqdm(range(6)):
    xe = grids['C96'].xe.isel(nf=i)
    ye = grids['C96'].ye.isel(nf=i)
    v1 = data1.SpeciesConc_NO.isel(time=0, nf=i, lev=1)
    v2 = data2.SpeciesConc_NO.isel(time=0, nf=i, lev=1)

    v = (v1 - v2)/v2*100
    vmin=-200
    vmax=200
    #v = data.SpeciesConc_NO2.isel(time=0, nf=i, lev=20)
    # v = np.log10(v)
    # vmin=3e-11
    # vmax=7e-11
    ax.pcolormesh(xe.values, ye.values, v, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), cmap='RdBu_r')

# plt.scatter([-87.6298], [41.8781], zorder=400, transform=ccrs.PlateCarree(), c='white')
# ax.annotate('Chicago', xy=(-87.8,42), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
#             ha='left', va='bottom', color='white')

# cv = (data.SpeciesConc_O3.isel(time=0, lev=20).std()/data.SpeciesConc_O3.isel(time=0, lev=20).mean()).item()
# rmse = np.nanstd(data2.SpeciesConc_O3.isel(time=0, lev=20) - data.SpeciesConc_O3.isel(time=0, lev=20))/(0.5*(data.SpeciesConc_O3.isel(time=0, lev=20)+data2.SpeciesConc_O3.isel(time=0, lev=20))).mean()
# mb = np.nanmean(data2.SpeciesConc_O3.isel(time=0, lev=20) - data.SpeciesConc_O3.isel(time=0, lev=20))/(0.5*(data.SpeciesConc_O3.isel(time=0, lev=20)+data2.SpeciesConc_O3.isel(time=0, lev=20))).mean()

# print('CV: ', cv)
# print('RMSE/mu: ', rmse)
# print('MB/mu: ', mb)

plt.colorbar(matplotlib.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap='RdBu_r'))

plt.show()
