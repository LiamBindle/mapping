
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
    parser.add_argument('--density',
                        action='store_true')
    parser.add_argument('--grid_def',
                        type=str,
                        required=True)
    parser.add_argument('--region',
                        type=str,
                        choices=['US', 'California', 'global'],
                        default='California')
    parser.add_argument('--shapefiles',
                        type=str,
                        default='/home/liam/Downloads')
    parser.add_argument('--name',
                        type=str,
                        required=False,
                        default='')
    parser.add_argument('-s',
                        type=float,
                        default=80)
    parser.add_argument('-o',
                        type=str,
                        required=True
                        )
    args = vars(parser.parse_args())

    x_da = xr.open_dataset(args['xfile'])[args['xvar']].squeeze()
    y_da = xr.open_dataset(args['yfile'])[args['yvar']].squeeze()
    grid = xr.open_dataset(args['grid_def'])


    import maps
    if args['region'] == 'California':
        region = maps.get_provinces_and_states(args['shapefiles']).loc['California'].geometry
    elif args['region'] == 'US':
        region = maps.get_countries(args['shapefiles']).loc['United States of America'].geometry

    xc = grid['grid_boxes_centers'].isel(XY=0).values
    yc = grid['grid_boxes_centers'].isel(XY=1).values
    region_mask = maps.mask_outside(xc, yc, region.buffer(0.2).simplify(0.1))  # buffer of 0.2 deg
    x_da.values[region_mask] = np.nan
    y_da.values[region_mask] = np.nan

    plt.figure()
    ax = plt.gca()
    # ax.axis('equal')

    x = x_da.values[~region_mask].flatten()
    y = y_da.values[~region_mask].flatten()

    mask = np.isfinite(x)
    x = x[mask]
    y = y[mask]

    if args['density']:
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        # z = [1 if abs(xx-yy)/yy > 0.5 and yy > 25 else 0 for xx, yy, in zip(x, y)]
        # q50_z = np.quantile(z, 0.7)
        # z = [1 if zz > q50_z and abs(xx-yy)/yy > 0.5 and yy > 13 else 0 for xx, yy, zz in zip(x, y, z)]

        ax.scatter(x, y, c=z, s=args['s'], edgecolor='', cmap='jet', marker='.')
    else:
        ax.scatter(x, y, s=args['s'], edgecolor='', cmap='jet', marker='.')

    ax.margins(0.05)
    limits = [*ax.get_xlim(), *ax.get_ylim()]
    lower_limit = max(min(limits), 0)
    upper_limit = max(x.max(), y.max())  #max(np.quantile(x, 0.999), np.quantile(y, 0.999))  # max(limits)

    ax.set_xlim(lower_limit, upper_limit)
    ax.set_ylim(lower_limit, upper_limit)
    ax.set_aspect('equal', adjustable='box')

    ticker = matplotlib.ticker.MaxNLocator(nbins=3, min_n_ticks=3)
    ax.yaxis.set_major_locator(ticker)
    ax.xaxis.set_major_locator(ticker)

    plt.plot(
        [lower_limit*0.1, upper_limit*10],
        [lower_limit*0.1, upper_limit*10],
        linewidth=0.9, linestyle='--', color='k'
    )

    ax.set_xlabel('Observed')
    ax.set_ylabel(f'Simulated')
    ax.text(
        0.05, 0.95,
        args['name'],
        transform=ax.transAxes,
        horizontalalignment='left',
        verticalalignment='top',
    )

    mb = y.mean() - x.mean()
    mae = sklearn.metrics.mean_absolute_error(x, y)
    rmse = np.sqrt(sklearn.metrics.mean_squared_error(x, y))
    r2 = sklearn.metrics.r2_score(x, y)

    std_x = np.std(x)
    std_y = np.std(y)

    summary_string = f"MB:   {mb:4.2e}\nMAE:  {mae:4.2e}\nRMSE: {rmse:4.2e}\nR2:   {r2:4.2f}\nSTD: {std_x:4.2e}, {std_y:4.2e}"

    ax.text(
        0.98, 0.02,
        summary_string,
        transform=ax.transAxes,
        horizontalalignment='right',
        verticalalignment='bottom',
    )

    # gs.tight_layout(fig)
    # plt.tight_layout()
    plt.savefig(args['o'],  dpi=300, bbox_inches='tight')
    plt.show()