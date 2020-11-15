
import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import geopandas
import xarray as xr
import sklearn.metrics
import shapely.geometry
from scipy.stats import gaussian_kde

from tqdm import tqdm
import maps


if __name__ == '__main__':
    xvar = 'TROPOMI_NO2_molec_per_cm2'
    yvar = 'GCHP_NO2'

    xfiles = [
        # '/home/liam/Downloads/scratch/C90-global/TROPOMI_NO2_July2018.nc',
        # '/home/liam/Downloads/scratch/C90-global/C90_on_C900e-TROP.nc',
        '/home/liam/Downloads/scratch/C900e-CA/TROPOMI_NO2_July2018.nc',
        '/home/liam/Downloads/scratch/C900e-CA/TROPOMI_NO2_July2018.nc',
    ]

    yfiles = [
        # '/home/liam/Downloads/scratch/C90-global/GCHP_NO2_July2018.nc',
        '/home/liam/Downloads/scratch/C90-global/C90_on_C900e.nc',
        '/home/liam/Downloads/scratch/C900e-CA/GCHP_NO2_July2018.nc',
    ]

    names = [
        'C90-global',
        'C900e-CA',
    ]

    grid_files = [
        # '/home/liam/Downloads/scratch/C90-global/comparison_grid.nc',
        '/home/liam/Downloads/scratch/C900e-CA/comparison_grid.nc',
        '/home/liam/Downloads/scratch/C900e-CA/comparison_grid.nc',
    ]

    fig = plt.figure(figsize=(4.724, 2))

    # gs = plt.GridSpec(2, 3, fig, width_ratios=[2, 2, 1.2], height_ratios=[20,1], top=1, bottom=0.15)
    gs = plt.GridSpec(1, 2, fig)


    bad_area = shapely.geometry.Polygon([
        (-118.7759399, 34.7799717),
        (-118.1442261, 34.2810491),
        (-116.9796753, 34.2118022),
        (-116.8945313, 34.9354820),
        (-118.3090210, 35.0457382),
        (-118.7759399, 34.7799717),
    ])



    # pdf_norm = plt.Normalize(0, 0.8)
    pdf_norm = plt.Normalize(0, 3)

    qs = np.linspace(0, 1, 101)

    for i, (xfile, yfile, name, gridfile) in enumerate(zip(xfiles, yfiles, names, grid_files)):

        grid = xr.open_dataset(gridfile)
        xc = grid['grid_boxes_centers'].isel(XY=0).values
        yc = grid['grid_boxes_centers'].isel(XY=1).values

        region = maps.get_tiger_states('/home/liam/Downloads').loc['California'].geometry.buffer(0.5)
        mask = maps.mask_outside(xc, yc, region)

        bad_mask = maps.mask_outside(xc, yc, bad_area)
        mask |= ~bad_mask


        fig.add_subplot(gs[0, i])
        ax = plt.gca()
        x = xr.open_dataset(xfile)[xvar].squeeze().values[~mask].flatten()/1e15
        y = xr.open_dataset(yfile)[yvar].squeeze().values[~mask].flatten()/1e15

        isfinite = np.isfinite(x) & np.isfinite(y)
        x = np.log10(x[isfinite])
        y = np.log10(y[isfinite])


        qx = np.quantile(x, qs)
        qy = np.quantile(y, qs)

        # mb = y.mean() - x.mean()
        # mae = sklearn.metrics.mean_absolute_error(x, y)
        # rmse = np.sqrt(sklearn.metrics.mean_squared_error(x, y))
        # r2 = sklearn.metrics.r2_score(x, y)

        # xy = np.vstack([x, y])
        # z = gaussian_kde(xy)(xy)
        # print(z.max())

        # ax.scatter(x, y, c=z, s=7, edgecolor='', cmap='jet', marker='.', norm=pdf_norm)
        ax.scatter(qx, qy)#, s=20, edgecolor='')

        ax.margins(0.05)
        limits = [*ax.get_xlim(), *ax.get_ylim()]
        lower_limit = -0.5
        upper_limit = 1.5
        # lower_limit = 0.5
        # upper_limit = 4.5

        ax.set_xlim(lower_limit, upper_limit)
        ax.set_ylim(lower_limit, upper_limit)
        ax.set_aspect('equal', adjustable='box')

        ticker = matplotlib.ticker.MaxNLocator(nbins=3, min_n_ticks=3)
        ax.yaxis.set_major_locator(ticker)
        ax.xaxis.set_major_locator(ticker)

        plt.plot(
            [lower_limit*1.1, upper_limit*10],
            [lower_limit*1.1, upper_limit*10],
            linewidth=0.5, linestyle='--', color='k'
        )

        if i == 0:
            ax.set_ylabel(f'Simulated', fontsize='small')

        if i == 1:
            plt.setp(ax.get_yticklabels(), visible=False)


        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(0.5)
            ax.tick_params(width=0.5)

    # ax = fig.add_subplot(gs[1, 0:2])
    # ax.axis('off')
    # ax.text(
    #     0.5, 0.5,
    #     "Observed column density, [10$^{15}$ molec cm-2]",
    #     transform=ax.transAxes,
    #     horizontalalignment='center',
    #     verticalalignment='center',
    #     fontsize='small',
    # )

    # from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    # ax = fig.add_subplot(gs[1, 2])
    # ax.axis('off')
    # ax.text(
    #     0.5, 0.5,
    #     "PDF density:",
    #     transform=ax.transAxes,
    #     horizontalalignment='right',
    #     verticalalignment='center',
    #     fontsize='xx-small',
    # )
    # ax = inset_axes(ax,
    #                 width="70%",
    #                 height="100%",
    #                 # loc='upper right',
    #                 bbox_to_anchor=(0.3, 0., 0.7, 1),
    #                 bbox_transform=ax.transAxes,
    #                 borderpad=0,
    # )
    #
    # cb = matplotlib.colorbar.ColorbarBase(ax, cmap=plt.get_cmap('jet'),
    #                                       norm=pdf_norm,
    #                                       orientation='horizontal', ticks=[pdf_norm.vmin, pdf_norm.vmax])
    # cb.set_label('PDF density', fontsize='xx-small', labelpad=-100, x=0)
    #
    # cb.outline.set_linewidth(0.3)
    # for axis in ['top','bottom','left','right']:
    #     ax.tick_params(width=0.3, length=2)
    # cb.ax.tick_params(labelsize='xx-small', pad=2)
    #
    # import cartopy.crs as ccrs
    # ax = fig.add_subplot(gs[0, 2], projection=ccrs.epsg(2163))
    # ax.set_facecolor('white')
    #
    # california = maps.get_provinces_and_states('/home/liam/Downloads/').loc['California'].geometry
    # maps.set_extent(ax, california)
    # maps.features.format_page(ax, linewidth_axis_spines=0)
    # maps.features.add_polygons(ax, california, outline=True, edgecolor='dimgray')
    # maps.add_hills(ax, '/home/liam/Downloads/')
    # maps.add_roads(ax, '/home/liam/Downloads/', edgecolor='k', linewidth=0.1)
    #
    # maps.add_polygons(ax, region, outline=True, crs=ccrs.PlateCarree(), edgecolor='tab:blue', linewidth=0.1, facecolor='tab:blue', alpha=0.6)
    #
    # maps.features.add_polygons(ax, california, exterior=True, facecolor='white', zorder=200)
    # plt.savefig('/home/liam/gmd-sg-manuscript-2020/figures/central-valley-scatter.png',  dpi=300, bbox_inches='tight')

    # plt.savefig('/home/liam/gmd-sg-manuscript-2020/figures/central-valley-scatter.png',  dpi=300, bbox_inches='tight')
    plt.show()