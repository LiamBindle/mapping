
import argparse
import numpy as np
import xarray as xr
import sklearn.metrics
import pyproj
import pandas as pd
import shapely.geometry
import scipy.stats

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import maps
import cartopy.crs as ccrs

if __name__ == '__main__':

    grids = {
        'C90-global': xr.open_dataset('/home/liam/Downloads/scratch/C90-global/comparison_grid.nc'),
        'C900e-CA': xr.open_dataset('/home/liam/Downloads/scratch/C900e-CA/comparison_grid.nc'),
    }
    masks = {}
    bad_area = shapely.geometry.Polygon([
        (-118.7759399, 34.7799717),
        (-118.1442261, 34.2810491),
        (-116.9796753, 34.2118022),
        (-116.8945313, 34.9354820),
        (-118.3090210, 35.0457382),
        (-118.7759399, 34.7799717),
    ])
    for grid_name, grid in grids.items():
        xc = grid['grid_boxes_centers'].isel(XY=0).values
        yc = grid['grid_boxes_centers'].isel(XY=1).values

        region = maps.get_tiger_states('/home/liam/Downloads').loc['California'].geometry.buffer(0.5)
        mask = maps.mask_outside(xc, yc, region)

        bad_mask = maps.mask_outside(xc, yc, bad_area)
        mask |= ~bad_mask
        masks[grid_name] = mask


    x_files = [
        '/home/liam/Downloads/scratch/C90-global/TROPOMI_NO2_July2018.nc',
        '/home/liam/Downloads/scratch/C90-global/TROPOMI_NO2_July2018.nc',
        '/home/liam/Downloads/scratch/C900e-CA/TROPOMI_NO2_July2018.nc',
        '/home/liam/Downloads/scratch/C900e-CA/TROPOMI_NO2_July2018.nc',
    ]

    y_files = [
        '/home/liam/Downloads/scratch/C90-global/GCHP_NO2_July2018.nc',
        '/home/liam/Downloads/scratch/C900e-CA/C900e_on_C90.nc',
        '/home/liam/Downloads/scratch/C90-global/C90_on_C900e.nc',
        '/home/liam/Downloads/scratch/C900e-CA/GCHP_NO2_July2018.nc',
    ]

    which_grid = [
        'C90-global',
        'C90-global',
        'C900e-CA',
        'C900e-CA',
    ]

    names = [
        'C90-global',
        'C900e-CA (upscaled to C90-global)',
        'C90-global (downscaled to C900e-CA)',
        'C900e-CA',
    ]

    colors = [
        plt.get_cmap('Dark2').colors[0],
        plt.get_cmap('Dark2').colors[0],
        plt.get_cmap('Dark2').colors[1],
        plt.get_cmap('Dark2').colors[1],
    ]

    markers = [
        'x',
        '^',
        'x',
        '^',
    ]
    annotate_specs = [
        dict(xytext=(0.4, -0.05), horizontalalignment='left', verticalalignment='bottom',),
        dict(xytext=(0.25, -0.05), horizontalalignment='right', verticalalignment='top',),
        dict(xytext=(0.25, -0.05), horizontalalignment='center', verticalalignment='bottom',),
        dict(xytext=(0.25, -0.05), horizontalalignment='right', verticalalignment='bottom',),
    ]

    plt.figure(figsize=(3.26772,5))
    ax = plt.axes(projection='polar')
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    plt.ylim([0, 0.25])
    ax.set_xticks(
        np.arccos(np.linspace(1, 0, 11)),
    )
    ax.set_xticklabels(['1', *[f'0.{v}' for v in range(10)][::-1]])

    plotted_trop = {}
    plotted_trop_colors = {}

    for xfile, yfile, grid_name, name, c, m, anno_spec in zip(x_files, y_files, which_grid, names, colors, markers, annotate_specs):
        xvar = 'TROPOMI_NO2_molec_per_cm2'
        yvar = 'GCHP_NO2'
        mask = masks[grid_name]
        x = xr.open_dataset(xfile)[xvar].squeeze().values #[~mask].flatten()/1e15
        x = x[~np.broadcast_to(mask, x.shape)].flatten()
        y = xr.open_dataset(yfile)[yvar].squeeze().values #[~mask].flatten()/1e15
        y = y[~np.broadcast_to(mask, y.shape)].flatten()

        isfinite = np.isfinite(x) & np.isfinite(y)
        x = np.log10(x[isfinite])  # residuals are heteroscedastic and NO2 is log-normally distributed
        y = np.log10(y[isfinite])

        std_x = np.std(x)
        std_y = np.std(y)
        r, _ = scipy.stats.pearsonr(x, y)

        if grid_name not in plotted_trop:
            ax.scatter(0, std_x, marker=f'o', c=c, s=50, label=f'TROPOMI (upscaled to {grid_name})')
            plotted_trop[grid_name] = (std_x, c)

        ax.scatter(np.arccos(r), std_y, marker=m, s=50, c=c, label=name)


    for name, (std, c) in plotted_trop.items():
        for rmsd in [0.5, 1, 1.5, 2, 2.5]:
            theta = np.linspace(0, np.pi, 100)
            x = std + rmsd * std * np.cos(theta)
            y = rmsd * std * np.sin(theta)

            r = np.sqrt(x**2+y**2)
            theta = np.arctan2(y, x)

            ax.plot(theta, r, linestyle='--', linewidth=0.4, color=c)

    plt.ylabel('Correlation, R (unitless)', fontsize=8)
    plt.xlabel('Log-normal standard deviation (unitless)', labelpad=20, fontsize=8)
    plt.legend(loc='upper center',  framealpha=1, bbox_to_anchor=(0.48, 1.8), fontsize=8)
    # plt.legend(loc='upper center',  framealpha=1, fontsize=8)

    ax.tick_params(axis='both', which='major', labelsize=8)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height*0.8])
    plt.show()
    # plt.savefig('/home/liam/gmd-sg-manuscript-2020/figures/CA_taylor_diagram.png', dpi=300)

    # for xfile, yfile, grid_name, name, c, m in zip(x_files, y_files, which_grid, names, colors, markers):
    #     xvar = 'TROPOMI_NO2_molec_per_cm2'
    #     yvar = 'GCHP_NO2'
    #     mask = masks[grid_name]
    #     x = xr.open_dataset(xfile)[xvar].squeeze().values #[~mask].flatten()/1e15
    #     x = x[~np.broadcast_to(mask, x.shape)].flatten()
    #     y = xr.open_dataset(yfile)[yvar].squeeze().values #[~mask].flatten()/1e15
    #     y = y[~np.broadcast_to(mask, y.shape)].flatten()
    #
    #     isfinite = np.isfinite(x) & np.isfinite(y)
    #     x = x[isfinite]
    #     y = y[isfinite]
    #
    #     plt.subplot(2,1,1)
    #     plt.hist(y)
    #     plt.hist(x)
    #
    #     plt.subplot(2,1,2)
    #     plt.hist(np.log10(y))
    #     plt.hist(np.log10(x))
    #     plt.show()