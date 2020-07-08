import numpy as np
import cartopy.crs as ccrs
import pyproj
import matplotlib.pyplot as plt
import shapely.geometry
import argparse
import os.path
import ast
import matplotlib.colors

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
    #gnomonic_proj = pyproj.Proj(gnomonic_crs.proj4_init)
    gnomonic_proj = pyproj.Transformer.from_crs('epsg:4326', gnomonic_crs.proj4_init, always_xy=True).transform

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
                        default=[0.2e15, 3.7e15],)
    parser.add_argument('--cmap',
                        type=str,
                        default='Oranges',)
    parser.add_argument('--grid_def',
                        type=str,
                        required=True)
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
    parser.add_argument('--titles',
                        action='store_true')
    parser.add_argument('--region',
                        type=str,
                        choices=['US', 'California'],
                        default='California',)
    parser.add_argument('--shapefiles',
                        type=str,
                        default='/home/liam/Downloads')
    args = vars(parser.parse_args())
    # plt.rc('text', usetex=False)

    def save_fig():
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.savefig(args['o'], dpi=300)

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
        road_params = dict(linewidth=0.25,  edgecolor=matplotlib.colors.to_rgba('snow', 0.5))
        width=1.575
    elif args['region'] == 'US':
        region = maps.get_countries(args['shapefiles']).loc['United States of America'].geometry
        stats_text_pos = dict(
            x=0.03,
            y=0.05,
            horizontalalignment='left',
            verticalalignment='bottom',
        )
        road_params = dict(linewidth=0.2,  edgecolor=matplotlib.colors.to_rgba('snow', 0.9))
        width=3.27

    crs = ccrs.epsg(2163)
    plt.figure(figsize=maps.figsize_fitting_polygon(region, crs, width=width))
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

    if args['cbar_only']: # --cbar_only 4.724 0.2 --cbar_label "NO$_2$ column density, [molec cm-2]"
        plt.figure(figsize=(args['cbar_only']))
        ax = plt.axes()
        cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap,
                                              norm=norm,
                                              orientation='horizontal')
        cb.set_label(args['cbar_label'])
        plt.subplots_adjust(top=0.95, bottom=0.7, right=0.95, left=0.05,
                            hspace=0, wspace=0)
        plt.margins(0.1,0.1)
        plt.savefig(args['o'], dpi=300, pad_inches=0)
        exit(0)

    if args['titles']: # --cbar_only 4.724 0.2 --cbar_label "NO$_2$ column density, [molec cm-2]"
        plt.figure(figsize=(4.724, 0.05))
        ax = plt.axes()
        plt.text(
            x=1/6-0.8/6, y=0.5, s="TROPOMI",
            horizontalalignment='center',
            verticalalignment='center',
        )
        plt.text(
            x=3/6-0.8/6, y=0.5, s="CTL",
            horizontalalignment='center',
            verticalalignment='center',
        )
        plt.text(
            x=5/6-0.8/6, y=0.5, s="C900e-CA",
            horizontalalignment='center',
            verticalalignment='center',
        )
        plt.axis('off')
        save_fig()
        #plt.savefig(args['o'], dpi=300, bbox_inches='tight')
        exit(0)

    if 'nf' in da.dims:
        # -128 -65 23 50
        for nf in range(6):
            xe = grid.xe.isel(nf=nf).values % 360
            ye = grid.ye.isel(nf=nf).values
            xc = grid['grid_boxes_centers'].isel(nf=nf, XY=0).values % 360
            yc = grid['grid_boxes_centers'].isel(nf=nf, XY=1).values
            face_xy = get_minor_xy(xe % 360, ye)
            blocksize = determine_blocksize(face_xy, xc % 360, yc)
            # print(f'Block size for face {nf+1}: {blocksize}')
            pcolormesh2(xe, ye, da.isel(nf=nf), blocksize, norm, cmap=cmap)

    else:
        xe, ye = grid.xe.values, grid.ye.values #pyproj.Transformer.from_crs('epsg:4326', 'epsg:2163', always_xy=True).transform(grid.xe, grid.ye)

        xmin, xmax = xe.min().item(), xe.max().item()
        ymin, ymax = ye.min().item(), ye.max().item()

        plt.imshow(da.transpose()[::-1,:], norm=norm, cmap=cmap, extent=[xmin, xmax, ymin, ymax])

        save_fig()
        # plt.show()
