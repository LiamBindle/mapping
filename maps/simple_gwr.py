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


def setup_axes():
    region = maps.get_provinces_and_states(args['shapefiles']).loc['California'].geometry

    crs = ccrs.epsg(2163)
    plt.figure(figsize=maps.figsize_fitting_polygon(region, crs, width=1.63386))
    ax = plt.axes(projection=crs)
    maps.set_extent(ax, region)
    maps.features.format_page(ax, linewidth_axis_spines=0)
    maps.features.add_polygons(ax, region, exterior=True, zorder=100, facecolor='white')
    maps.features.add_polygons(ax, region, outline=True, zorder=99, edgecolor='black', linewidth=0.2)
    maps.add_roads(ax, args['shapefiles'], linewidth=0.3,  edgecolor=matplotlib.colors.to_rgba('snow', 0.5))
    maps.add_hills(ax, args['shapefiles'])
    return ax


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
    parser.add_argument('-o',
                        type=str,
                        required=True
                        )
    args = vars(parser.parse_args())

    x_da = xr.open_dataset(args['xfile'])[args['xvar']].squeeze()
    y_da = xr.open_dataset(args['yfile'])[args['yvar']].squeeze()
    grid = xr.open_dataset(args['grid_def'])

    region = maps.get_provinces_and_states(args['shapefiles']).loc['California'].geometry


    xc = grid['grid_boxes_centers'].isel(XY=0).values
    yc = grid['grid_boxes_centers'].isel(XY=1).values

    project = pyproj.Transformer.from_crs('epsg:4326', 'epsg:2163', always_xy=True).transform
    xc, yc = project(xc, yc)
    region = shapely.ops.transform(project, region)
    xmin, ymin, xmax, ymax = region.bounds

    imin = np.argmin(abs(xc[:, 0] - xmin))-1
    imax = np.argmin(abs(xc[:, 0] - xmax))+2
    jmin = np.argmin(abs(yc[0, :] - ymin))-1
    jmax = np.argmin(abs(yc[0, :] - ymax))+2


    dx = 4e3
    dy = 4e3
    bw = 50e3*np.sqrt(2)
    di = np.ceil(bw/dx).astype(int)
    dj = np.ceil(bw/dy).astype(int)

    r = np.ones_like(xc) * np.nan
    p = np.ones_like(xc) * np.nan
    std = np.ones_like(xc) * np.nan
    mean_x = np.ones_like(xc) * np.nan
    mean_y = np.ones_like(yc) * np.nan
    std_x = np.ones_like(xc) * np.nan
    std_y = np.ones_like(yc) * np.nan
    # e = np.ones_like(yc) * np.nan

    r2 = np.ones_like(yc) * np.nan



    def cov(x, y, w):
        return np.sum(w * (x - np.average(x, weights=w)) * (y - np.average(y, weights=w))) / np.sum(w)

    def corr(x, y, w):
        return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))


    for i in tqdm(range(imin, imax)):
        for j in range(jmin, jmax):
            ic_slice = slice(max(0, i-di), min(grid.dims['i'], i+di+1))
            jc_slice = slice(max(0, j-dj), min(grid.dims['j'], j+dj+1))
            x = x_da.values[ic_slice, jc_slice].flatten()
            y = y_da.values[ic_slice, jc_slice].flatten()

            dist = np.sqrt((xc[ic_slice, jc_slice] - xc[i, j])**2 + (yc[ic_slice, jc_slice] - yc[i, j])**2)

            # w = ((1-(dist**2/bw**2))**2).flatten()
            # w[dist.flatten() >= bw] = 0
            # w = (dist.flatten() < bw) & np.isfinite(x)

            weights = ((1-(dist**2/bw**2))**2).flatten()
            weights[dist.flatten() >= bw] = 0

            valid=np.isfinite(x)
            r2[i, j] = sklearn.metrics.r2_score(x[valid], y[valid], sample_weight=weights[valid])

            mean_x[i, j] = np.average(x[valid], weights=weights[valid])
            mean_y[i, j] = np.average(y[valid], weights=weights[valid])
            std_x[i, j] = np.sqrt(np.cov(x[valid], aweights=weights[valid]))
            std_y[i, j] = np.sqrt(np.cov(y[valid], aweights=weights[valid]))

    ic_slice = slice(max(0, imin), min(grid.dims['i'], imax))
    jc_slice = slice(max(0, jmin), min(grid.dims['j'], jmax))
    ie_slice = slice(max(0, imin), min(grid.dims['i'], imax+1))
    je_slice = slice(max(0, jmin), min(grid.dims['j'], jmax+1))

    xe, ye = grid.xe[ie_slice, je_slice].values, grid.ye[ie_slice, je_slice].values #pyproj.Transformer.from_crs('epsg:4326', 'epsg:2163', always_xy=True).transform(grid.xe, grid.ye)

    xmin, xmax = xe.min().item(), xe.max().item()
    ymin, ymax = ye.min().item(), ye.max().item()

    std = std_x[ic_slice, jc_slice].transpose()[::-1,:]

    cv = std_x[ic_slice, jc_slice].transpose()[::-1,:]/mean_x[ic_slice, jc_slice].transpose()[::-1,:]

    # dat = r[ic_slice, jc_slice].transpose()[::-1,:]


    data_r2 = r2[ic_slice, jc_slice].transpose()[::-1,:]
    data_r2[std < 0.15e15] = np.nan

    data_mb = (mean_y[ic_slice, jc_slice].transpose()[::-1,:] - mean_x[ic_slice, jc_slice].transpose()[::-1,:]) / mean_x[ic_slice, jc_slice].transpose()[::-1,:]

    setup_axes()
    plt.imshow(data_r2, extent=[xmin, xmax, ymin, ymax], norm=plt.Normalize(-1, 1), cmap='RdYlGn')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                        hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.savefig(f"{args['o']}/R2.png", dpi=300)

    setup_axes()
    plt.imshow(data_mb, extent=[xmin, xmax, ymin, ymax], norm=plt.Normalize(-0.3, 0.3), cmap='RdBu_r')   # <-- this one
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                        hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.savefig(f"{args['o']}/NMB.png", dpi=300)

    plt.figure(figsize=(3.26772, 0.8))
    ax = plt.axes()
    cb = matplotlib.colorbar.ColorbarBase(ax, cmap=plt.get_cmap('RdYlGn'),
                                          norm=plt.Normalize(-1, 1),
                                          orientation='horizontal')
    cb.set_label('R$^{2}$ score')
    plt.subplots_adjust(top=0.95, bottom=0.7, right=0.95, left=0.05,
                        hspace=0, wspace=0)
    plt.margins(0.1,0.1)
    plt.savefig(f"{args['o']}/R2-cbar.png", dpi=300, pad_inches=0)

    plt.figure(figsize=(3.26772, 0.8))
    ax = plt.axes()
    cb = matplotlib.colorbar.ColorbarBase(ax, cmap=plt.get_cmap('RdBu_r'),
                                          norm=plt.Normalize(-0.3, 0.3),
                                          orientation='horizontal')
    cb.set_label('Normalized mean bias')
    plt.subplots_adjust(top=0.95, bottom=0.7, right=0.95, left=0.05,
                        hspace=0, wspace=0)
    plt.margins(0.1,0.1)
    plt.savefig(f"{args['o']}/NMB-cbar.png", dpi=300, pad_inches=0)

    plt.figure(figsize=(3.26772, 0.05))
    ax = plt.axes()
    plt.text(
        x=1/4-0.5/4, y=0.5, s="CTL",
        horizontalalignment='center',
        verticalalignment='center',
    )
    plt.text(
        x=3/4-0.5/4, y=0.5, s="C900e-CA",
        horizontalalignment='center',
        verticalalignment='center',
    )
    plt.axis('off')
    plt.savefig(f"{args['o']}/titles.png", dpi=300, bbox_inches='tight', pad_inches=0.05)

