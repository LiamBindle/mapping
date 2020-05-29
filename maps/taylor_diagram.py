
import argparse
import numpy as np
import xarray as xr
import sklearn.metrics
import pyproj
import pandas as pd

import matplotlib.pyplot as plt

import maps
import cartopy.crs as ccrs

if __name__ == '__main__':

    ax = plt.axes(projection='polar')
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_ylim([0, 2])
    # ax.set_ylim([0, 1e15])

    # fnames = ['c180.csv', 'c900.csv'] #['/home/liam/Downloads/bar/C180-global/masked_stats.csv', '/home/liam/Downloads/bar/C900e-CA/masked_stats.csv']
    fnames = ['/home/liam/Downloads/foobar/C180-global/masked_stats.csv', '/home/liam/Downloads/foobar/C900e-CA/masked_stats.csv'] #['/home/liam/Downloads/bar/C180-global/masked_stats.csv', '/home/liam/Downloads/bar/C900e-CA/masked_stats.csv']

    # fnames = ['/home/liam/Downloads/bar/C180-global/masked_stats.csv', '/home/liam/Downloads/bar/C900e-CA/masked_stats.csv']

    dfs = [pd.read_csv(f) for f in fnames]

    names = dfs[0]['NAME']
    r = np.array([df['R'] for df in dfs]).transpose()
    std_x = np.array([df['STD_X'] for df in dfs]).transpose()
    std_y = np.array([df['STD_Y'] for df in dfs]).transpose()
    nmb = np.array([df['MB']/df['MEAN_X'] for df in dfs]).transpose()

    print('r:', r)
    print('nmb:', nmb)


    # grid = xr.open_dataset('/home/liam/Downloads/scratch/comparison_grid.nc')
    # ctl = xr.open_dataset('/home/liam/Downloads/foo/foo/TROPOMI_NO2_July2018.nc')
    # c180 = xr.open_dataset('/home/liam/Downloads/bar/C180-global/GCHP_NO2_July2018.nc')
    # c900 = xr.open_dataset('/home/liam/Downloads/bar/C900e-CA/GCHP_NO2_July2018.nc')
    #
    # region = maps.get_tiger_states('/home/liam/Downloads').loc['California'].geometry
    # # region = region.to_crs('epsg:4326').geometry
    #
    # mask = maps.mask_outside(grid.grid_boxes_centers.isel(XY=0).values, grid.grid_boxes_centers.isel(XY=1).values, region)
    #
    # dat_c180 = c180.GCHP_NO2.values[~mask].flatten()
    # dat_c900 = c900.GCHP_NO2.values[~mask].flatten()
    # dat_ctl = ctl.TROPOMI_NO2_molec_per_cm2.values[~mask].flatten()
    #
    # valid = np.isfinite(dat_ctl) & (dat_ctl > 0.2e15)
    # dat_ctl = dat_ctl[valid]
    # dat_c180 = dat_c180[valid]
    # dat_c900 = dat_c900[valid]
    #
    # import scipy.stats
    # r = [None, None]
    # r[0], _ = scipy.stats.pearsonr(dat_ctl, dat_c180)
    # r[1], _ = scipy.stats.pearsonr(dat_ctl, dat_c900)
    #
    # std_x = [np.nanstd(dat_ctl), np.nanstd(dat_ctl)]
    # std_y = [np.nanstd(dat_c180), np.nanstd(dat_c900)]
    #
    # names = ['C180 vs C900e']
    #
    # r = np.array(r)[np.newaxis,:]
    # std_x = np.array(std_x)[np.newaxis,:]
    # std_y = np.array(std_y)[np.newaxis,:]

    for i, (corr, std, target, n, mb) in enumerate(zip(r, std_y/std_x, std_x[:,0], names, nmb)):
        # if n in ['Mojave Desert', 'San Diego County', 'South Central Coast', 'South Coast']: continue
        c = plt.get_cmap('tab20').colors[i]
        # ax.arrow(np.arccos(corr[0]), std[0], np.arccos(corr[1]) - np.arccos(corr[0]), std[1] - std[0], fc=c, ec=c, head_width=0.1, head_length=0.1)
        ax.scatter(np.arccos(corr[0]), std[0], c=c, marker='D', label=n)
        ax.scatter(np.arccos(corr[1]), std[1], c=c, marker='x')
        ax.scatter(0, target, c=c, marker='o')

    plt.legend()
    plt.show()

    # basins = maps.get_california_air_basins('/home/liam/Downloads')
    # basins = basins.to_crs('epsg:2163')
    #
    # region = maps.get_provinces_and_states('/home/liam/Downloads').loc['California'].geometry
    # crs = ccrs.epsg(2163)
    # plt.figure(figsize=maps.figsize_fitting_polygon(region, crs, width=4.72))
    # ax = plt.axes(projection=crs)
    # maps.set_extent(ax, region)
    # maps.features.format_page(ax, linewidth_axis_spines=0)
    # maps.features.add_polygons(ax, region, exterior=True, zorder=100, facecolor='white')
    # # maps.add_roads(ax,'/home/liam/Downloads', **road_params)
    # # maps.add_hills(ax, '/home/liam/Downloads')
    #
    # df_ctl = dfs[0].set_index('NAME')
    # df_exp = dfs[1].set_index('NAME')
    #
    # basins['E_CTL'] = np.sqrt(df_ctl['STD_X']**2 + df_ctl['STD_Y']**2 - 2*df_ctl['STD_X']*df_ctl['STD_Y']*df_ctl['R'])
    # basins['E_EXP'] = np.sqrt(df_exp['STD_X']**2 + df_exp['STD_Y']**2 - 2*df_exp['STD_X']*df_exp['STD_Y']*df_exp['R'])
    #
    #
    # for geo, v in zip(basins.geometry, basins['E_EXP']/basins['E_CTL']):
    #     norm = plt.Normalize(0.5, 1.5)
    #     c = plt.get_cmap('RdYlGn_r')(norm(v))
    #     maps.add_polygons(ax, geo, crs, facecolor=c)
    # plt.show()
