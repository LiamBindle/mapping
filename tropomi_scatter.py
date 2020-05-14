
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
    parser.add_argument('--tropomi',
                        type=str,
                        required=True)
    parser.add_argument('--gchp',
                        type=str,
                        required=True)
    parser.add_argument('--shapefile',
                        type=str,
                        # default='/home/liam/Downloads/cb_2018_us_nation_20m.shp'
                        default='/home/liam/Downloads/tl_2017_us_state.shp'
                        )
    args = parser.parse_args()

    ds_tropomi = xr.open_dataset(args.tropomi)
    ds_gchp = xr.open_dataset(args.gchp)

    lats = ds_gchp.lats.values.flatten()
    lons = ds_gchp.lons.values.flatten()

    shapefile = geopandas.read_file(args.shapefile)
    region = shapefile.loc[shapefile['NAME'] == 'California'].iloc[0].geometry
    # region = shapefile.iloc[0].geometry  # US

    plt.figure()
    ax = plt.gca()
    # ax.axis('equal')

    x = ds_tropomi['TROPOMI_NO2'].values.flatten()
    y = ds_gchp['GCHP_NO2'].values.flatten()

    mask = np.isfinite(x)
    x = x[mask]
    y = y[mask]

    lats = lats[mask]
    lons = lons[mask]

    import maps.shapefiles
    foo = maps.shapefiles.mask_outside(lons, lats, region)

    inside_region = [region.contains(shapely.geometry.Point(xp, yp)) for xp, yp in tqdm(zip(lons, lats), total=lats.size)]

    x = x[inside_region]
    y = y[inside_region]

    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    # z = [1 if abs(xx-yy)/yy > 0.5 and yy > 25 else 0 for xx, yy, in zip(x, y)]
    # q50_z = np.quantile(z, 0.7)
    # z = [1 if zz > q50_z and abs(xx-yy)/yy > 0.5 and yy > 13 else 0 for xx, yy, zz in zip(x, y, z)]

    ax.scatter(x, y, c=z, s=50, edgecolor='', cmap='jet', marker='.')

    ax.margins(0.05)
    limits = [*ax.get_xlim(), *ax.get_ylim()]
    lower_limit = 0  # max(min(limits), 0)
    upper_limit = max(np.quantile(x, 0.95), np.quantile(y, 0.95))  # max(limits)

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
        'GCHP (updated NEI11) reduced by 40% vs TROPOMI',
        transform=ax.transAxes,
        horizontalalignment='left',
        verticalalignment='top',
    )

    mb = y.mean() - x.mean()
    mae = sklearn.metrics.mean_absolute_error(x, y)
    rmse = np.sqrt(sklearn.metrics.mean_squared_error(x, y))
    r2 = sklearn.metrics.r2_score(x, y)

    summary_string = f"MB:   {mb:4.2e}\nMAE:  {mae:4.2e}\nRMSE: {rmse:4.2e}\nR2:   {r2:4.2f}"

    ax.text(
        0.98, 0.02,
        summary_string,
        transform=ax.transAxes,
        horizontalalignment='right',
        verticalalignment='bottom',
    )

    # gs.tight_layout(fig)
    plt.tight_layout()
    plt.show()