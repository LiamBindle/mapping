import numpy as np
import cartopy.crs as ccrs
import pyproj
import matplotlib.pyplot as plt
import shapely.geometry
import argparse
import os.path
import ast


if __name__ == '__main__':
    import xarray as xr
    import yaml
    import  matplotlib.cm
    parser = argparse.ArgumentParser(description='Make a map')
    parser.add_argument('filein',
                        metavar='FILEIN',
                        type=str,
                        help="input file")
    parser.add_argument('-v', '--var',
                        nargs='+',
                        metavar='NAME',
                        type=str,
                        required=True,
                        help='path to the control\'s output directory')
    parser.add_argument('--sel',
                        type=str,
                        nargs='+',
                        default=[],
                        help='selectors')
    parser.add_argument('--isel',
                        metavar='NAME',
                        type=str,
                        nargs='+',
                        default=[],
                        help='index selectors')
    parser.add_argument('-o',
                        metavar='O',
                        type=str,
                        default='output.png',
                        help='path to output')
    parser.add_argument('-n', '--norm',
                        metavar='N',
                        type=float,
                        nargs=2,
                        default=None,
                        help='norm')
    parser.add_argument('--cmap',
                        metavar='CMAP',
                        type=str,
                        default='cividis',
                        help='color map')
    parser.add_argument('--extent',
                        nargs=4,
                        metavar='x0x1y0y1',
                        type=float,
                        default=None,
                        help='map extent')
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
                        metavar='x0x1y0y1',
                        type=str,
                        choices=['US', 'California', 'global'],
                        default='global',
                        help='map extent')
    parser.add_argument('--norm1-quantile',
                        metavar='Q',
                        type=float,
                        default=0.0,
                        help='lower norm quantile')
    parser.add_argument('--norm2-quantile',
                    metavar='Q',
                    type=float,
                    default=0.98,
                    help='upper norm quantile')
    parser.add_argument('--shapefiles',
                        type=str,
                        default='/home/liam/Downloads')
    parser.add_argument('--crs',
                        metavar='CRS',
                        type=str,
                        default='ccrs.EqualEarth()',
                        help='cartopy CRS eval')
    args = vars(parser.parse_args())
    plt.rc('text', usetex=False)

    ds = xr.open_dataset(args['filein'])

    for sel_k, sel_v_str in zip(args['sel'][::2], args['sel'][1::2]):
        try:
            sel_v = ast.literal_eval(sel_v_str)
        except ValueError:
            sel_v = sel_v_str
        ds = ds.sel(**{sel_k: sel_v})

    for isel_k, isel_v in zip(args['isel'][::2], args['isel'][1::2]):
        ds = ds.isel(**{isel_k: int(isel_v)})

    da = sum([ds[varname].squeeze() for varname in args['var']])

    # Open grid def file
    grid = xr.open_dataset(args['grid_def'])


    if args['region'] == 'global':
        plt.figure(figsize=(args['width'], args['width']*0.75))
        ax = plt.axes(projection=eval(args['crs']))
        if args['extent'] is None:
            ax.set_global()
        else:
            ax.set_extent(args['extent'], ccrs.PlateCarree())
        ax.coastlines(linewidth=0.5)
        nf_range = range(6)
        data_for_stats = da.values
        stats_text_pos = dict(
            x=0.03,
            y=0.05,
            horizontalalignment='left',
            verticalalignment='bottom',
        )
    else:
        import maps
        if args['region'] == 'California':
            region = maps.get_provinces_and_states(args['shapefiles']).loc['California'].geometry
            stats_text_pos = dict(
                x=0.7,
                y=0.8,
                horizontalalignment='left',
                verticalalignment='top',
            )
        elif args['region'] == 'US':
            region = maps.get_countries(args['shapefiles']).loc['United States of America'].geometry
            stats_text_pos = dict(
                x=0.03,
                y=0.05,
                horizontalalignment='left',
                verticalalignment='bottom',
            )

        crs = ccrs.epsg(2163)
        plt.figure(figsize=maps.figsize_fitting_polygon(region, crs, width=args['width']))
        ax = plt.axes(projection=crs)
        maps.set_extent(ax, region)
        maps.features.format_page(ax, linewidth_axis_spines=0)
        # ax.set_facecolor('white')
        maps.features.add_polygons(ax, region, exterior=True, zorder=100, facecolor='black')
        # maps.features.add_polygons(ax, region, outline=True)
        # nf_range = [0, 1, 3, 4, 5]
        #
        # xc = grid['grid_boxes_centers'].isel(XY=0).values
        # yc = grid['grid_boxes_centers'].isel(XY=1).values
        # region_mask = maps.mask_outside(xc, yc, region.buffer(0.2).simplify(0.1))  # buffer of 0.2 deg
        data_for_stats = da.values
        # data_for_stats[region_mask] = np.nan


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
        norm = plt.Normalize(da.quantile(args['norm1_quantile']), da.quantile(args['norm2_quantile']))
    else:
        norm = plt.Normalize(args['norm'][0], args['norm'][1])

    if args['cbar_only']:
        plt.figure(figsize=(args['cbar_only']))
        ax = plt.axes()
        cb = matplotlib.colorbar.ColorbarBase(ax, cmap=plt.get_cmap(args['cmap']),
                                              norm=norm,
                                              orientation='horizontal')
        cb.set_label(args['cbar_label'])
        plt.savefig(args['o'], dpi=300, bbox_inches='tight')
        exit(0)

    xe = grid['xe']
    ye = grid['ye']

    plt.pcolormesh(xe, ye, da, norm=norm, cmap=args['cmap'])

    if len(stats) > 0:
        plt.text(
            s="\n".join([args['name'], *[f"{stat_name}: {stat_value:4.2e}" for stat_name, stat_value in stats.items()]]),
            transform=ax.transAxes,
            zorder=101,
            **stats_text_pos
        )

    # plt.colorbar(matplotlib.cm.ScalarMappable(norm, args['cmap']), orientation='horizontal')
    # plt.tight_layout()
    plt.savefig(args['o'], dpi=300, bbox_inches='tight')
