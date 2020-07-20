

import xarray as xr
import numpy as np
import sys
import datetime
date = sys.argv[1] # "20180629_0530"


time = datetime.datetime(int(date[0:4]), int(date[4:6]), int(date[6:8]), int(date[9:11]), int(date[11:13])) - datetime.timedelta(0, 7*60*60)

ds = xr.open_dataset(f'diags_{date}.nc')
# ds = xr.open_dataset(f'/extra-space/diags_{date}.nc')
grid = xr.open_dataset('grid.nc')
# grid = xr.open_dataset('/extra-space/comparison_grid.nc')
xc = grid['grid_boxes_centers'].isel(XY=0).values
yc = grid['grid_boxes_centers'].isel(XY=1).values

lon_slice = -117.93
lat_slice = (33.61, 35.8)

def central_angle(x0, y0, x1, y1):
    RAD2DEG = 180 / np.pi
    DEG2RAD = np.pi / 180
    x0 = x0 * DEG2RAD
    x1 = x1 * DEG2RAD
    y0 = y0 * DEG2RAD
    y1 = y1 * DEG2RAD
    return np.arccos(np.sin(y0) * np.sin(y1) + np.cos(y0) * np.cos(y1) * np.cos(np.abs(x0-x1))) * RAD2DEG

# find closest coordinate
gp1 = np.unravel_index(np.argmin(central_angle(lon_slice, lat_slice[0], xc, yc)), (6, 90, 90))
gp2 = np.unravel_index(np.argmin(central_angle(lon_slice, lat_slice[1], xc, yc)), (6, 90, 90))

nf = 5
Ydim_slice = slice(57, 82)
Xdim=32

ds = ds.isel(nf=nf, Ydim=Ydim_slice, Xdim=Xdim)

# compute_no2_column(ds.isel(lev=slice(0,72)), ds.isel(lev=slice(0,72)), ds)

Md = 28.9647e-3 / 6.0221409e+23  # [kg molec-1]
no2_area_density = ds['Met_AIRDEN'] * ds['Met_BXHEIGHT'] * ds['SpeciesConc_NO2'] / Md / 100**2

u_winds = ds.Met_U.values[0, :-1, :]
v_winds = ds.Met_V.values[0, :-1, :]
values = (ds.SpeciesConc_NO2 + ds.SpeciesConc_NO).values[0, :-1, :]
ye = np.cumsum(ds.Met_BXHEIGHT.values[0,:-1,:], axis=0)
ye = np.concatenate([np.zeros((1,25,)), ye])

pbl = ds.Met_PBLH.values[0, :]
pbl2 = np.ndarray((pbl.shape[0]+1))
pbl2[0] = pbl[0]
pbl2[-1] = pbl[-1]
pbl2[1:-1] = np.diff(pbl)/2 + pbl[:-1]

ye2 = np.ndarray((ye.shape[0], ye.shape[1]+1))
ye2[:,0] = ye[:,0]
ye2[:,-1] = ye[:,-1]
ye2[:,1:-1] = np.diff(ye, axis=1)/2 + ye[:,:-1]

xe = grid['grid_boxes_centers'].isel(XY=1, nf=nf, Ydim=Ydim_slice, Xdim=Xdim).values

xe2 = np.ndarray((xe.size + 1))
xe2[0] = xe[0]
xe2[-1] = xe[-1]
xe2[1:-1] = np.diff(xe)/2 + xe[:-1]


# import requests
# import urllib
# import pandas as pd
#
# # USGS Elevation Point Query Service
# url = r'https://nationalmap.gov/epqs/pqs.php?'
#
# # coordinates with known elevation
# lat = xe2.tolist()
# lon = [lon_slice]*len(lat)
#
# # create data frame
# df = pd.DataFrame({
#     'lat': lat,
#     'lon': lon
# })
#
# def elevation_function(df, lat_column, lon_column):
#     """Query service using lat, lon. add the elevation values as a new column."""
#     elevations = []
#     for lat, lon in zip(df[lat_column], df[lon_column]):
#
#         # define rest query params
#         params = {
#             'output': 'json',
#             'x': lon,
#             'y': lat,
#             'units': 'Meters'
#         }
#
#         # format query string and return query value
#         result = requests.get((url + urllib.parse.urlencode(params)))
#         elevations.append(result.json()['USGS_Elevation_Point_Query_Service']['Elevation_Query']['Elevation'])
#
#     df['elev_meters'] = elevations
#
# elevation_function(df, 'lat', 'lon')
# df.head()

elev_meters = np.array([1602.73, 1044.06, 1417.29, 964.84, 997.79, 604.53, 690.99, 714.13,
 744.82, 716.81, 701.82, 739.54, 802.29, 962.91, 1902.15, 1399.72,
 603.78, 130.87, 207.15, 80.48, 30.68, 16.31, 0, 0,
 0, 0])
ye2 += elev_meters


import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
for i in range(xe.size):
    #v = values[:,i]
    v = no2_area_density.values[0,:-1,i]
    x = xe2[i:i+2]
    y = ye2[:,i:i+2]

    x = np.column_stack([x]*73).transpose()

    # xx, yy = np.meshgrid(x, y, indexing='ij')
    pcm = plt.pcolormesh(x, y, v[:, np.newaxis], norm=plt.Normalize(0, 2e15), cmap='cividis')
    plt.plot(x[0], pbl2[i:i+2]+y[0], color='k')
    y_mean = np.mean(y, axis=1)
    plt.quiver(np.mean(x, axis=1)[:-1], (y_mean[:-1]+y_mean[1:])/2, v_winds[:,i], np.zeros_like(v_winds[:,i]), scale=100, width=0.002)

plt.colorbar(pcm, label='NO$_2$ column density (molec cm$^{-2}$)')

plt.xticks(
    [33.61, 33.833, 34.027, 34.151, 34.308, 34.685, 35.109],
    ['Newport Beach', 'Anaheim', 'LA', 'Pasadena', 'Angeles National\nForest', 'Lancaster', 'Mojave Desert'],
    rotation=90
)
plt.subplots_adjust(0.1, 0.3, 0.95, 0.95)

if time.hour == 13 and time.minute == 30:
    plt.text(0.9, 0.9, str(time), transform=plt.gca().transAxes, horizontalalignment='right', verticalalignment='top', color='green', fontsize=20)
else:
    plt.text(0.9, 0.9, str(time), transform=plt.gca().transAxes, horizontalalignment='right', verticalalignment='top', color='white', fontsize=20)

plt.ylim((0, 4000))
# plt.show()
plt.savefig(f'frame-{date}.png')


# print('foo')