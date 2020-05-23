
import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import geopandas
import xarray as xr
import sklearn.metrics
import shapely.geometry
from scipy.stats import gaussian_kde
import pyproj
from shapely.ops import transform
from tqdm import tqdm
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xfile',
                        type=str,
                        required=True)
    parser.add_argument('--xvar',
                        type=str,
                        required=True)
    parser.add_argument('--yfile',
                        type=str,
                        required=True)
    parser.add_argument('--yvar',
                        type=str,
                        required=True)
    parser.add_argument('--grid_def',
                        type=str,
                        required=True)
    parser.add_argument('--shapefiles',
                        type=str,
                        default='/home/liam/Downloads')
    args = vars(parser.parse_args())

    x_da = xr.open_dataset(args['xfile'])[args['xvar']].squeeze()
    y_da = xr.open_dataset(args['yfile'])[args['yvar']].squeeze()
    grid = xr.open_dataset(args['grid_def'])

    import maps
    # ca = maps.get_california_counties(args['shapefiles'])

    regions = maps.get_california_air_basins(args['shapefiles'])

    project = pyproj.Transformer.from_crs('epsg:4326', 'epsg:2163', always_xy=True).transform

    xc = grid['grid_boxes_centers'].isel(XY=0).values
    yc = grid['grid_boxes_centers'].isel(XY=1).values

    xc, yc = project(xc, yc)

    df = pd.DataFrame(index=regions.index, columns=['MB', 'MAE', 'RMSE', 'R2', 'STD_X', 'STD_Y', 'MEAN_X', 'MEAN_Y'])

    regions = regions.to_crs('epsg:2163')

    for name, region in zip(regions.index, regions.geometry):
        print(name)
        mask = maps.mask_outside(xc, yc, region)

        x = x_da.values[~mask]
        y = y_da.values[~mask]

        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]

        df.at[name, 'MB'] = y.mean() - x.mean()
        df.at[name, 'MAE'] = sklearn.metrics.mean_absolute_error(x, y)
        df.at[name, 'RMSE'] = np.sqrt(sklearn.metrics.mean_squared_error(x, y))
        df.at[name, 'R2']= sklearn.metrics.r2_score(x, y)
        df.at[name, 'STD_X'] = np.std(x)
        df.at[name, 'STD_Y'] = np.std(y)
        df.at[name, 'MEAN_X'] = np.mean(x)
        df.at[name, 'MEAN_Y'] = np.mean(y)

    print(df)
    df.to_csv('region_stats.csv')
