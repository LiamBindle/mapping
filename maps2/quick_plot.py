import matplotlib.pyplot as plt
import cartopy.crs as ccrs # cartopy > 0.18
import xarray as xr

# Set up GeoAxes
ax = plt.axes(projection=ccrs.EqualEarth())
ax.set_global()
ax.coastlines()

# Read data and get vmin, vmax

# ds2 = xr.open_dataset('/media/liam/external1/gchp-37/4_node/GCHP.TestCollectionNativeConservativeOtherOrder.20160701_0030z.nc4')
ds = xr.open_dataset('/extra-space/scratch/mini2/OutputDir/GCHP.SpeciesConc.20190101_1200z.nc4')
da = ds['SpeciesConc_Rn222'].isel(lev=0).squeeze()
vmin = da.quantile(0.02).item()
vmax = da.quantile(0.98).item()

ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.set_global()

# Plot data
for nf in range(6):
    x = ds['lons'].isel(nf=nf).values
    y = ds['lats'].isel(nf=nf).values
    v = da.isel(nf=nf).values
    plt.pcolormesh(x, y, v, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)

# x = ds['lon'].values
# y = ds['lat'].values
# v = da.values
# plt.pcolormesh(x, y, v, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)

plt.show()