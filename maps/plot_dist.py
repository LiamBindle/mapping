import numpy as np
import cartopy.crs as ccrs
import pyproj
import matplotlib.pyplot as plt
import shapely.geometry
import argparse
import os.path
import ast
import matplotlib.colors
import shapely.ops
import xarray as xr

import maps

from tqdm import tqdm

import sklearn.metrics

import scipy.stats
import statsmodels.graphics.gofplots

if __name__ == '__main__':

    ds1 = xr.open_dataset('/home/liam/Downloads/foobar/C180e-US/GCHP_NO2_July2018.nc')
    ds2 = xr.open_dataset('/home/liam/Downloads/foobar/C180-global/GCHP_NO2_July2018.nc')
    grid = xr.open_dataset('/home/liam/Downloads/scratch/comparison_grid.nc')

    project = pyproj.Transformer.from_crs('epsg:4326', 'epsg:2163', always_xy=True).transform

    xc = grid['grid_boxes_centers'].isel(XY=0).values
    yc = grid['grid_boxes_centers'].isel(XY=1).values
    xc, yc = project(xc, yc)

    US = maps.get_countries('/home/liam/Downloads').loc['United States of America'].geometry.convex_hull
    US = shapely.ops.transform(project, US)

    mask = maps.mask_outside(xc, yc, US)

    da1 = ds1.GCHP_NO2.values[~mask]
    da2 = ds2.GCHP_NO2.values[~mask]

    keep = np.isfinite(da1) & np.isfinite(da2)

    da1 = da1[keep]
    da2 = da2[keep]

    def plot_lognorm(dist, **kwargs):
        dist = np.log(dist)
        np.random.choice(dist.size, 50000, replace=False)
        kde = scipy.stats.gaussian_kde(dist)
        x = np.linspace(dist.min(), dist.max(), 1000)
        y = kde.evaluate(x)

        plt.plot(x, y, **kwargs)

    def plot_lognorm_hist(dist, **kwargs):
        dist = np.log(dist)
        plt.hist(dist, bins=20, **kwargs)


    def plot_qq(dist1, dist2, **kwargs):
        x = np.linspace(0.005, 0.995, 100)
        q1 = np.quantile(dist1, x, interpolation='linear')
        q2 = np.quantile(dist2, x, interpolation='linear')
        plt.plot(q1, q2, marker='.')
        plt.xlim(0, 1.25e16)
        plt.ylim(0, 1.25e16)
        plt.gca().set_aspect('equal', adjustable='box')


    print('hi')