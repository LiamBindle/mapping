import numpy as np
import cartopy.crs as ccrs
import pyproj
import matplotlib.pyplot as plt
import shapely.geometry
import argparse
import os.path
import ast
import matplotlib.colors


if __name__ == '__main__':
    import xarray as xr
    import yaml
    import  matplotlib.cm
    parser = argparse.ArgumentParser(description='Make a map')
    parser.add_argument('filein',
                        type=str,)
    parser.add_argument('-v', '--var',
                        nargs='+',
                        type=str,
                        required=True,
                        help='path to the control\'s output directory')
    parser.add_argument('--isel',
                        type=str,
                        nargs='+',
                        default=[],)
    parser.add_argument('-o',
                        type=str,
                        default='output.png',)
    parser.add_argument('-n', '--norm',
                        type=float,
                        nargs=2,
                        default=None,)
    parser.add_argument('--cmap',
                        type=str,
                        default='cividis',)
    parser.add_argument('--grid_def',
                        type=str,
                        required=True)
    parser.add_argument('--width',
                        type=float,
                        default=8)
    parser.add_argument('--cbar_only',
                        nargs=2,
                        metavar='XY XY',
                        type=float,
                        default=None,)
    parser.add_argument('--cbar_label',
                        type=str,
                        default='',)
    parser.add_argument('--stats',
                        nargs='+',
                        type=str,
                        choices=['mean', 'sum', 'std', 'median'],
                        default=[],)
    parser.add_argument('--name',
                        type=str,
                        default='',)
    parser.add_argument('--region',
                        type=str,
                        choices=['US', 'California'],
                        default='California',)
    parser.add_argument('--shapefiles',
                        type=str,
                        default='/home/liam/Downloads')
    args = vars(parser.parse_args())
    plt.rc('text', usetex=False)

    ds = xr.open_dataset(args['filein'])

    for isel_k, isel_v in zip(args['isel'][::2], args['isel'][1::2]):
        ds = ds.isel(**{isel_k: int(isel_v)})

    da = sum([ds[varname].squeeze() for varname in args['var']])

    # Open grid def file
    grid = xr.open_dataset(args['grid_def'])

    import maps
    if args['region'] == 'California':
        region = maps.get_provinces_and_states(args['shapefiles']).loc['California'].geometry
        stats_text_pos = dict(
            x=0.5,
            y=0.75,
            horizontalalignment='left',
            verticalalignment='top',
        )
        road_params = dict(linewidth=0.5,  edgecolor=matplotlib.colors.to_rgba('snow', 0.5))
    elif args['region'] == 'US':
        region = maps.get_countries(args['shapefiles']).loc['United States of America'].geometry
        stats_text_pos = dict(
            x=0.03,
            y=0.05,
            horizontalalignment='left',
            verticalalignment='bottom',
        )
        road_params = dict(linewidth=0.6,  edgecolor=matplotlib.colors.to_rgba('snow', 0.9))

    crs = ccrs.epsg(2163)
    plt.figure(figsize=maps.figsize_fitting_polygon(region, crs, width=args['width']))
    ax = plt.axes(projection=crs)
    maps.set_extent(ax, region)
    maps.features.format_page(ax, linewidth_axis_spines=0)
    maps.features.add_polygons(ax, region, exterior=True, zorder=100, facecolor='white')
    # maps.features.add_polygons(ax, region, outline=True, zorder=100, edgecolor='black', linewidth=0.8)
    maps.add_roads(ax, args['shapefiles'], **road_params)
    maps.add_hills(ax, args['shapefiles'])

    # xc = grid['grid_boxes_centers'].isel(XY=0).values
    # yc = grid['grid_boxes_centers'].isel(XY=1).values
    # region_mask = maps.mask_outside(xc, yc, region.buffer(0.2).simplify(0.1))  # buffer of 0.2 deg
    data_for_stats = da.values
    # data_for_stats[region_mask] = np.nan

    if os.path.exists(args['cmap']):
        import xml.etree.ElementTree as ET
        root = ET.parse(args['cmap']).getroot()

        x = []
        rgb = []
        for pt in root[0]:
            if pt.tag != 'Point':
                continue
            items = dict(pt.items())
            x.append(float(items['x']))
            rgb.append((float(items['r']), float(items['g']), float(items['b'])))

        x = np.array(x)
        rgb = np.array(rgb)
        x = x[1:]
        rgb = rgb[1:,:]

        n = 1000
        new_x = np.linspace(0, 1, n)
        r = np.interp(new_x, x, rgb[:, 0])
        g = np.interp(new_x, x, rgb[:, 1])
        b = np.interp(new_x, x, rgb[:, 2])
        x = new_x
        rgb = np.moveaxis([r, g, b], 0, -1)
        cmap = matplotlib.colors.ListedColormap(rgb)
    else:
        cmap = plt.get_cmap(args['cmap'])


    # Compute stats
    stats = {}
    for stat in args['stats']:
        if stat == 'mean':
            stats['mean'] = np.nanmean(data_for_stats)
        elif stat == 'sum':
            stats['sum'] = np.nansum(data_for_stats)
        elif stat == 'std':
            stats['std'] = np.nanstd(data_for_stats)
        elif stat == 'median':
            stats['median'] = np.nanmedian(data_for_stats)

    if args['norm'] is None:
        norm = 0, da.quantile(0.995)
    else:
        norm = plt.Normalize(args['norm'][0], args['norm'][1])

    if args['cbar_only']:
        plt.figure(figsize=(args['cbar_only']))
        ax = plt.axes()
        cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap,
                                              norm=norm,
                                              orientation='horizontal')
        cb.set_label(args['cbar_label'])
        plt.savefig(args['o'], dpi=300, bbox_inches='tight')
        exit(0)

    xe, ye = grid.xe.values, grid.ye.values #pyproj.Transformer.from_crs('epsg:4326', 'epsg:2163', always_xy=True).transform(grid.xe, grid.ye)

    xmin, xmax = xe.min().item(), xe.max().item()
    ymin, ymax = ye.min().item(), ye.max().item()

    plt.imshow(da.transpose()[::-1,:], norm=norm, cmap=cmap, extent=[xmin, xmax, ymin, ymax])
    # plt.pcolormesh(xe, ye, da, norm=norm, cmap=args['cmap'])

    # if len(stats) > 0:
    #     plt.text(
    #         s="\n".join([args['name'], *[f"{stat_name}: {stat_value:4.2e}" for stat_name, stat_value in stats.items()]]),
    #         transform=ax.transAxes,
    #         zorder=101,
    #         **stats_text_pos
    #     )

    # plt.colorbar(matplotlib.cm.ScalarMappable(norm, args['cmap']), orientation='horizontal')
    # plt.tight_layout()
    # plt.savefig(args['o'], dpi=300, bbox_inches='tight')
    plt.show()
