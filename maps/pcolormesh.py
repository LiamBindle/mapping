import numpy as np
import cartopy.crs as ccrs
import pyproj
import matplotlib.pyplot as plt
import shapely.geometry
import argparse
import os.path
import ast


def get_minor_xy(xe, ye):
    p0 = slice(0, -1)
    p1 = slice(1, None)
    boxes_x = np.moveaxis(np.array([xe[p0, p0], xe[p1, p0], xe[p1, p1], xe[p0, p1], xe[p0, p0]]), 0, -1)
    boxes_y = np.moveaxis(np.array([ye[p0, p0], ye[p1, p0], ye[p1, p1], ye[p0, p1], ye[p0, p0]]), 0, -1)
    return np.moveaxis(np.array([boxes_x, boxes_y]), 0, -1)


def transform_xy(xy_in, in_proj: pyproj.Proj, out_proj: pyproj.Proj):
    xy_out = pyproj.transform(in_proj, out_proj, xy_in[..., 0], xy_in[..., 1])
    xy_out = np.moveaxis(xy_out, 0, -1)
    return xy_out


def xy_to_polygons(xy, transform=None, error_on_bad_polygon=True, only_valid=False):
    if len(xy.shape) == 2:
        xy = np.expand_dims(xy, 0)

    output_shape = xy.shape[:-2]

    indexes = np.moveaxis(np.meshgrid(*[range(i) for i in xy.shape[:-2]], indexing='ij'), 0, -1)
    stacked = np.product(xy.shape[:-2])
    xy = np.reshape(xy, (stacked, *xy.shape[-2:]))
    indexes = np.reshape(indexes, (stacked, len(xy.shape[-2:])))
    polygons = []
    bad = []
    zero_area = []
    index_lut = {}
    for i, (polygon_xy, index) in enumerate(zip(xy, indexes)):
        polygon = shapely.geometry.Polygon(polygon_xy)

        is_bad = False
        if np.count_nonzero(np.isnan(polygon_xy)) > 0:
            bad.append(i)
            is_bad = True
        elif not polygon.is_valid:
            bad.append(i)
            is_bad = True
        elif polygon.area <= 0:
            zero_area.append(i)
            is_bad = True

        if not only_valid:
            polygons.append(polygon)
            index_lut[id(polygons[-1])] = tuple(index)
        elif only_valid and not is_bad:
            polygons.append(polygon)
            index_lut[id(polygons[-1])] = tuple(index)

    if error_on_bad_polygon and (len(bad) > 0 or len(zero_area) > 0):
        raise RuntimeError('A bad polygon was detected')
    elif not only_valid and (len(bad) > 0 or len(zero_area) > 0):
        for bad_index in [*bad, *zero_area]:
            polygons[bad_index] = shapely.geometry.Polygon([(0,0), (0,0), (0,0)]) # zero area

    if only_valid:
        return index_lut, polygons
    else:
        return np.reshape(polygons, output_shape)


def determine_blocksize(xy, xc, yc):
    latlon = pyproj.Proj('+init=epsg:4326')
    N = xc.shape[0]
    factors = [i for i in range(1, N+1) if N % i == 0]

    for f in factors:
        blocksize = N // f
        nblocks = f

        blocked_shape = (nblocks, nblocks, blocksize, blocksize)

        try:
            for bi in range(nblocks):
                for bj in range(nblocks):

                    block = np.ix_(range(bi*blocksize, (bi+1)*blocksize), range(bj*blocksize, (bj+1)*blocksize))

                    xc_block = xc[block]
                    yc_block = yc[block]
                    xy_block = xy[block]

                    center_x = xc_block[blocksize//2, blocksize//2]
                    center_y = yc_block[blocksize//2, blocksize//2]

                    local_gno = pyproj.Proj(ccrs.Gnomonic(center_y, center_x).proj4_init)

                    block_xy_gno = transform_xy(xy_block, latlon, local_gno)
                    _ = xy_to_polygons(block_xy_gno, error_on_bad_polygon=True)
            return blocksize
        except RuntimeError:
            pass
    raise RuntimeError('Failed to determine the appropriate blocksize')


def get_am_and_pm_masks_and_polygons_outline(xe, ye, far_from_pm=80):
    if np.any(xe >= 180):
        raise ValueError('xe must be in [-180, 180)')
    # xe must be [-180 to 180]
    p0 = slice(0, -1)
    p1 = slice(1, None)

    # Mask where bounding box crosses the prime meridian or antimeridian
    cross_pm_or_am_line1 = np.not_equal(np.sign(xe[p0, p0]), np.sign(xe[p1, p0]))
    cross_pm_or_am_line2 = np.not_equal(np.sign(xe[p1, p0]), np.sign(xe[p1, p1]))
    cross_pm_or_am_line3 = np.not_equal(np.sign(xe[p1, p1]), np.sign(xe[p0, p1]))
    cross_pm_or_am_line4 = np.not_equal(np.sign(xe[p0, p1]), np.sign(xe[p0, p0]))
    cross_pm_or_am = cross_pm_or_am_line1 | cross_pm_or_am_line2 | cross_pm_or_am_line3 | cross_pm_or_am_line4

    # Make xy polygons for each gridbox
    boxes_x = np.moveaxis(np.array([xe[p0, p0], xe[p1, p0], xe[p1, p1], xe[p0, p1]]), 0, -1)
    boxes_y = np.moveaxis(np.array([ye[p0, p0], ye[p1, p0], ye[p1, p1], ye[p0, p1]]), 0, -1)
    polygon_outlines = np.moveaxis(np.array([boxes_x, boxes_y]), 0, -1)

    pm = np.ones((xe.shape[0]-1, xe.shape[1]-1), dtype=bool)
    am = np.ones((xe.shape[0]-1, xe.shape[1]-1), dtype=bool)

    # Figure out which polygon_outlines cross the prime meridian and antimeridian
    crossing_indexes = np.argwhere(cross_pm_or_am)
    for idx in crossing_indexes:
        box = shapely.geometry.LinearRing(polygon_outlines[tuple(idx)])
        far_from_the_prime_meridian = shapely.geometry.LineString([(far_from_pm, -90), (80, far_from_pm)])
        if box.crosses(far_from_the_prime_meridian):
            am[tuple(idx)] = False
        else:
            pm[tuple(idx)] = False

    return am, pm, polygon_outlines


def _pcolormesh2_internal(ax, X, Y, C, cmap, norm):
    X[X >= 180] -= 360

    am, pm, boxes_xy_pc = get_am_and_pm_masks_and_polygons_outline(X, Y)

    center_i = int(X.shape[0] / 2)
    center_j = int(X.shape[1] / 2)
    cX = X[center_i, center_j]
    cY = Y[center_i, center_j]

    gnomonic_crs = ccrs.Gnomonic(cY, cX)
    gnomonic_proj = pyproj.Proj(gnomonic_crs.proj4_init)

    X_gno, Y_gno = gnomonic_proj(X, Y)
    boxes_xy_gno = np.moveaxis(gnomonic_proj(boxes_xy_pc[..., 0], boxes_xy_pc[..., 1]), 0, -1)

    if np.any(np.isnan(X_gno)) or np.any(np.isnan(Y_gno)):
        raise ValueError('Block size is too big!')
    else:
        plt.pcolormesh(X_gno, Y_gno, np.ma.masked_array(C, ~am), transform=gnomonic_crs, cmap=cmap, norm=norm)

        for idx in np.argwhere(~am):
            c = cmap(norm(C[idx[0], idx[1]]))
            ax.add_geometries(
                [shapely.geometry.Polygon(boxes_xy_gno[idx[0], idx[1],...])],
                gnomonic_crs, edgecolor=c, facecolor=c
            )


def pcolormesh2(X, Y, C, blocksize, norm, **kwargs):
    kwargs.setdefault('cmap', 'viridis')
    cmap = plt.get_cmap(kwargs['cmap'])

    ax = plt.gca()

    for si, ei in [(s * blocksize, (s + 1) * blocksize + 1) for s in range(X.shape[0] // blocksize)]:
        for sj, ej in [(s * blocksize, (s + 1) * blocksize + 1) for s in range(X.shape[1] // blocksize)]:
            _pcolormesh2_internal(ax,
                                  X[si:ei, sj:ej],
                                  Y[si:ei, sj:ej],
                                  C[si:ei - 1, sj:ej - 1],
                                  cmap, norm
                                  )


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
    parser.add_argument('--grid_def',
                        type=str,
                        required=True)
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
    parser.add_argument('--norm-from',
                        metavar='PATH',
                        type=str,
                        default=None,
                        help='other file to take norm from')
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
    parser.add_argument('--cbar_only',
                        nargs=2,
                        metavar='XY XY',
                        type=float,
                        default=None,)
    parser.add_argument('--cbar_label',
                        type=str,
                        default='',)
    parser.add_argument('--scale_factor',
                        type=float,
                        default=1.0,)
    parser.add_argument('--stats',
                        nargs='+',
                        type=str,
                        choices=['mean', 'sum', 'std'],
                        default=[],)
    parser.add_argument('--region',
                        metavar='x0x1y0y1',
                        type=str,
                        choices=['US', 'California', 'global'],
                        default='global',
                        help='map extent')
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
    ds_norm = xr.open_dataset(args['norm_from']) if args['norm_from'] is not None else None

    for sel_k, sel_v_str in zip(args['sel'][::2], args['sel'][1::2]):
        try:
            sel_v = ast.literal_eval(sel_v_str)
        except ValueError:
            sel_v = sel_v_str
        ds = ds.sel(**{sel_k: sel_v})
        if ds_norm is not None:
            ds_norm = ds_norm.sel(**{sel_k: sel_v})

    for isel_k, isel_v in zip(args['isel'][::2], args['isel'][1::2]):
        ds = ds.isel(**{isel_k: int(isel_v)})
        if ds_norm is not None:
            ds_norm = ds_norm.isel(**{isel_k: int(isel_v)})

    da = sum([ds[varname].squeeze().transpose('nf', 'Ydim', 'Xdim') for varname in args['var']])
    if ds_norm is not None:
        da_norm = ds_norm[args['var']].squeeze().transpose('nf', 'Ydim', 'Xdim')

    if ds_norm is None:
        da_norm = da

    if args['norm'] is None:
        norm = plt.Normalize(da_norm.quantile(args['norm1_quantile']), da_norm.quantile(args['norm2_quantile']))
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

    # Open grid def file
    grid = xr.open_dataset(args['grid_def'])


    if args['region'] == 'global':
        plt.figure(figsize=(8,6))
        ax = plt.axes(projection=eval(args['crs']))
        if args['extent'] is None:
            ax.set_global()
        else:
            ax.set_extent(args['extent'], ccrs.PlateCarree())
        ax.coastlines(linewidth=0.5)
        nf_range = range(6)
        data_for_stats = da.values
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
        plt.figure(figsize=maps.figsize_fitting_polygon(region, crs, width=8))
        ax = plt.axes(projection=crs)
        maps.set_extent(ax, region)
        maps.features.format_page(ax, linewidth_axis_spines=0)
        # ax.set_facecolor('white')
        maps.features.add_polygons(ax, region, exterior=True, zorder=100, facecolor='white')
        # maps.features.add_polygons(ax, region, outline=True)
        nf_range = [0, 1, 3, 4, 5]

        xc = grid['grid_boxes_centers'].isel(XY=0).values
        yc = grid['grid_boxes_centers'].isel(XY=1).values
        region_mask = maps.mask_outside(xc, yc, region.buffer(0.2).simplify(0.1))  # buffer of 0.2 deg
        data_for_stats = da.values
        data_for_stats[region_mask] = np.nan


    # Compute stats
    stats = {}
    for stat in args['stats']:
        if stat == 'mean':
            stats['mean'] = np.nanmean(data_for_stats)
        elif stat == 'sum':
            stats['sum'] = np.nansum(data_for_stats)
        elif stat == 'std':
            stats['std'] = np.nanstd(data_for_stats)


    for nf in nf_range:
        xe = grid['xe'].isel(nf=nf).values % 360
        ye = grid['ye'].isel(nf=nf).values
        xc = grid['grid_boxes_centers'].isel(nf=nf, XY=0).values % 360
        yc = grid['grid_boxes_centers'].isel(nf=nf, XY=1).values
        face_xy = get_minor_xy(xe % 360, ye)
        blocksize = determine_blocksize(face_xy, xc % 360, yc)
        # print(f'Block size for face {nf+1}: {blocksize}')
        pcolormesh2(xe, ye, da.isel(nf=nf) * args['scale_factor'], blocksize, norm, cmap=args['cmap'])

    if len(stats) > 0:
        plt.text(
            s="\n".join([f"{stat_name}: {stat_value:4.2e}" for stat_name, stat_value in stats.items()]),
            transform=ax.transAxes,
            zorder=101,
            **stats_text_pos
        )

    # plt.colorbar(matplotlib.cm.ScalarMappable(norm, args['cmap']), orientation='horizontal')
    # plt.tight_layout()
    plt.savefig(args['o'], dpi=300, bbox_inches='tight')
