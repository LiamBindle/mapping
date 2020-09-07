import xarray as xr
import numpy as np
import pyproj
import scipy.stats
from tqdm import tqdm

import shapely.geometry
import shapely.ops
import shapely.strtree


def mask(xyc, shape):
    xyc = xyc.stack(boxes=['nf', 'Ydim', 'Xdim']).transpose('boxes', 'XY').values
    contained = []
    for b in tqdm(range(xyc.shape[0]), desc='Masking boxes'):
        if shape.contains(shapely.geometry.Point(xyc[b,:])):
            contained.append(b)

    return np.array(contained)


def tessellate_a_to_b(a_xy, b_xy):

    # rows correspond to boxes in grid_out, columns correspond to boxes in grid_in
    latlon = pyproj.Proj('+init=epsg:4326')

    M_data = []
    M_i = []
    M_j = []

    laea = pyproj.Proj(
        f'+proj=laea +lat_0={b_xy[..., 1].mean().item()} +lon_0={b_xy[..., 0].mean().item()}  +x_0=0 +y_0=0 +a=6370997 +b=6370997 +units=m +no_defs'
    )

    a_xy = shapely.geometry.MultiPolygon([shapely.geometry.Polygon(xy) for xy in a_xy.values])
    b_xy = shapely.geometry.MultiPolygon([shapely.geometry.Polygon(xy) for xy in b_xy.values])

    transform = pyproj.Transformer.from_proj(latlon, laea, always_xy=True).transform

    a_xy = list(shapely.ops.transform(transform, a_xy))
    b_xy = list(shapely.ops.transform(transform, b_xy))
    b_ids = dict((id(pt), i) for i, pt in enumerate(b_xy))

    rtree = shapely.strtree.STRtree(b_xy)

    for a in tqdm(range(len(a_xy)), desc='Calculating tessellations'):
        matches = rtree.query(a_xy[a])
        matching_indexes = [b_ids[id(p)] for p in matches]

        for b in matching_indexes:
            box_a = a_xy[a]
            box_b = b_xy[b]
            weight = box_a.intersection(box_b).area / box_b.area
            if weight > 0:
                M_data.append(weight)
                M_i.append(b)
                M_j.append(a)

    M = scipy.sparse.coo_matrix((M_data, (M_i, M_j)), shape=(len(b_xy), len(a_xy)))

    row_sums = np.zeros((M.shape[0],))
    for i in range(M.shape[0]):
        row_i = M.getrow(i)

        row_sums[i] = np.sum(row_i)

    for i, s in enumerate(row_sums):
        if s < 0.95:
            M.data[np.argwhere(M.row == i)] = np.nan
        else:
            M.data[np.argwhere(M.row == i)] /= s

    return M


def full_tessellation_matrix(M, a_indexes, b_indexes, size_a, size_b):
    new_i = [b_indexes[i] for i in M.row]
    new_j = [a_indexes[j] for j in M.col]

    M = scipy.sparse.coo_matrix((M.data, (new_i, new_j)), shape=(size_b, size_a))
    return M


def ufunc_multiply(x, M):
    y = M @ x
    return y


def apply_tessellation(M, ds_in, ds_out_coords, keep_only=[]):

    droppers = ['cubed_sphere', 'contacts', 'orientation', 'anchor']
    for d in droppers:
        if d in ds_in:
            ds_in = ds_in.drop(d)

    if len(keep_only) > 0:
        droppers = set(ds_in.data_vars.keys()) - set(keep_only)
        ds_in = ds_in.drop(droppers)

    droppers = ['time', 'lats', 'lons', 'ncontacts']
    for d in droppers:
        if d in ds_out_coords:
            del ds_out_coords[d]

    ds_in = ds_in.rename({'nf': 'nf_in', 'Xdim': 'Xdim_in', 'Ydim': 'Ydim_in'})
    ds_in.coords.update(ds_out_coords)

    ds_in = ds_in.stack(iboxes=['nf_in', 'Ydim_in', 'Xdim_in'])
    ds = ds_in.stack(oboxes=['nf', 'Ydim', 'Xdim'])

    ds = xr.apply_ufunc(
        ufunc_multiply,
        ds, M,
        input_core_dims=[['iboxes'], []],
        output_core_dims=[['oboxes']],
        vectorize=True
    )

    ds['max_intersect'] = xr.DataArray(
        np.array(M.max(axis=1).todense()).squeeze(),
        dims='oboxes'
    )

    ds = ds.unstack('oboxes')

    for v in ds.data_vars.keys():
        ds[v] = ds[v].where(ds[v] != 0)

    return ds


# if __name__ == '__main__':
#     import maps
#     grid = xr.open_dataset('/extra-space/sg-stats/foo/grid_box_outlines_and_centers.nc')
#     stacked_grid = grid.drop(['xe', 'ye', 'XdimE', 'YdimE']) \
#         .stack(boxes=['nf', 'Ydim', 'Xdim']) \
#         .transpose('boxes', 'POLYGON_PTS', 'XY')
#     US = maps.get_countries('/home/liam/Downloads').loc['United States of America'].geometry.buffer(1).simplify(0.5)
#     idx = mask(grid.grid_boxes_centers, US)
#     inside = stacked_grid.grid_boxes.isel(boxes=idx)
#     p = [shapely.geometry.Polygon(p) for p in inside.values]
#     m = shapely.geometry.MultiPolygon(p)
#     laea = pyproj.Proj(
#         f'+proj=laea +lat_0={39.48} +lon_0={-98.38}  +x_0=0 +y_0=0 +a=6370997 +b=6370997 +units=m +no_defs'
#     )
#     latlon = pyproj.Proj('+init=epsg:4326')
#     transform = pyproj.Transformer.from_proj(latlon, laea, always_xy=True).transform
#     m2 = shapely.ops.transform(transform, m)
#     p2 = list(m2)
#     area = [f.area for f in p2]
#     print(np.mean(area))
#     exit(0)

if __name__ == '__main__':
    import maps

    # Tessellates a->b
    A_GRID='S48'
    B_GRID='C96'

    fname_ds_a_post = f'/extra-space/sg-stats/Sept-2/{A_GRID}/GCHP.SpeciesConc.Sept.{B_GRID}.nc'
    ds_a = xr.open_dataset(f'/extra-space/sg-stats/Sept-2/{A_GRID}/GCHP.SpeciesConc.Sept.nc', decode_times=False)
    ds_b = xr.open_dataset(f'/extra-space/sg-stats/Sept-2/{B_GRID}/GCHP.SpeciesConc.Sept.nc', decode_times=False)

    grid_a = xr.open_dataset(f'/extra-space/sg-stats/Sept/{A_GRID}/grid_box_outlines_and_centers.nc')
    grid_b = xr.open_dataset(f'/extra-space/sg-stats/Sept/{B_GRID}/grid_box_outlines_and_centers.nc')

    stacked_grid_a = grid_a.drop(['xe', 'ye', 'XdimE', 'YdimE'])\
        .stack(boxes=['nf', 'Ydim', 'Xdim'])\
        .transpose('boxes', 'POLYGON_PTS', 'XY')
    stacked_grid_b = grid_b.drop(['xe', 'ye', 'XdimE', 'YdimE']) \
        .stack(boxes=['nf', 'Ydim', 'Xdim']) \
        .transpose('boxes', 'POLYGON_PTS', 'XY')

    US = maps.get_countries('/home/liam/Downloads').loc['United States of America'].geometry.buffer(1).simplify(0.5)

    masked_stacked_indexes_a = mask(grid_a.grid_boxes_centers, US)
    masked_stacked_indexes_b = mask(grid_b.grid_boxes_centers, US)

    M = tessellate_a_to_b(
        stacked_grid_a.grid_boxes.isel(boxes=masked_stacked_indexes_a),
        stacked_grid_b.grid_boxes.isel(boxes=masked_stacked_indexes_b),
    )

    M = full_tessellation_matrix(
        M,
        masked_stacked_indexes_a,
        masked_stacked_indexes_b,
        stacked_grid_a.sizes['boxes'],
        stacked_grid_b.sizes['boxes'],
    )

    odata = apply_tessellation(
        M,
        ds_a,
        ds_b.coords
    )

    odata.to_netcdf(fname_ds_a_post)

    print(odata)
    exit(0)

if __name__ == '__main__':
    import maps
    CS_species = {
        'C96': xr.open_dataset('/extra-space/sg-stats/June/C96/species.june.nc'),
        'C98': xr.open_dataset('/extra-space/sg-stats/June/C98/species.june.nc'),
        'C100': xr.open_dataset('/extra-space/sg-stats/June/C100/species.june.nc'),
        'S48': xr.open_dataset('/extra-space/sg-stats/June/S48/species.june.nc'),
        'C90': xr.open_dataset('/extra-space/sg-stats/June/C90/species.june.nc'),
    }

    grids = {
        'C96': xr.open_dataset('/extra-space/sg-stats/June/C96/grid_box_outlines_and_centers.nc'),
        'C98': xr.open_dataset('/extra-space/sg-stats/June/C98/grid_box_outlines_and_centers.nc'),
        'C100': xr.open_dataset('/extra-space/sg-stats/June/C100/grid_box_outlines_and_centers.nc'),
        'S48': xr.open_dataset('/extra-space/sg-stats/June/S48/grid_box_outlines_and_centers.nc'),
        'C90': xr.open_dataset('/extra-space/sg-stats/June/C90/grid_box_outlines_and_centers.nc'),
    }

    stacked_grids = {
        k: v.drop(['xe', 'ye', 'XdimE', 'YdimE'])
            .stack(boxes=['nf', 'Ydim', 'Xdim'])
            .transpose('boxes', 'POLYGON_PTS', 'XY')
        for k, v in grids.items()
    }

    US = maps.get_countries('/home/liam/Downloads').loc['United States of America'].geometry.buffer(1).simplify(0.5)

    masked_stacked_indexes = {
        'C96': mask(grids['C96'].grid_boxes_centers, US),
        'C98': mask(grids['C98'].grid_boxes_centers, US),
        'C100': mask(grids['C100'].grid_boxes_centers, US),
        'S48': mask(grids['S48'].grid_boxes_centers, US),
        'C90': mask(grids['C90'].grid_boxes_centers, US),
    }

    M = {
        'C98->C96': tessellate_a_to_b(
            stacked_grids['C98'].grid_boxes.isel(boxes=masked_stacked_indexes['C98']),
            stacked_grids['C96'].grid_boxes.isel(boxes=masked_stacked_indexes['C96']),
        ),
        'C100->C96': tessellate_a_to_b(
            stacked_grids['C100'].grid_boxes.isel(boxes=masked_stacked_indexes['C100']),
            stacked_grids['C96'].grid_boxes.isel(boxes=masked_stacked_indexes['C96']),
        ),
        'S48->C96': tessellate_a_to_b(
            stacked_grids['S48'].grid_boxes.isel(boxes=masked_stacked_indexes['S48']),
            stacked_grids['C96'].grid_boxes.isel(boxes=masked_stacked_indexes['C96']),
        ),
        'C90->C96': tessellate_a_to_b(
            stacked_grids['C90'].grid_boxes.isel(boxes=masked_stacked_indexes['C90']),
            stacked_grids['C96'].grid_boxes.isel(boxes=masked_stacked_indexes['C96']),
        )
    }

    M = {
        'C98->C96': full_tessellation_matrix(
            M['C98->C96'],
            masked_stacked_indexes['C98'],
            masked_stacked_indexes['C96'],
            stacked_grids['C98'].sizes['boxes'],
            stacked_grids['C96'].sizes['boxes'],
        ),
        'C100->C96': full_tessellation_matrix(
            M['C100->C96'],
            masked_stacked_indexes['C100'],
            masked_stacked_indexes['C96'],
            stacked_grids['C100'].sizes['boxes'],
            stacked_grids['C96'].sizes['boxes'],
        ),
        'S48->C96': full_tessellation_matrix(
            M['S48->C96'],
            masked_stacked_indexes['S48'],
            masked_stacked_indexes['C96'],
            stacked_grids['S48'].sizes['boxes'],
            stacked_grids['C96'].sizes['boxes'],
        ),
        'C90->C96': full_tessellation_matrix(
            M['C90->C96'],
            masked_stacked_indexes['C90'],
            masked_stacked_indexes['C96'],
            stacked_grids['C90'].sizes['boxes'],
            stacked_grids['C96'].sizes['boxes'],
        ),
    }

    data = {
        'C98': apply_tessellation(
            M['C98->C96'],
            CS_species['C98'],
            CS_species['C96'].coords,
        ),
        'C100': apply_tessellation(
            M['C100->C96'],
            CS_species['C100'],
            CS_species['C96'].coords,
        ),
        'S48': apply_tessellation(
            M['S48->C96'],
            CS_species['S48'],
            CS_species['C96'].coords,
        ),
        'C90': apply_tessellation(
            M['C90->C96'],
            CS_species['C90'],
            CS_species['C96'].coords,
        ),
    }

    data['C98'].to_netcdf('/extra-space/sg-stats/June/C98/species.june.C96.nc')
    data['C100'].to_netcdf('/extra-space/sg-stats/June/C100/species.june.C96.nc')
    data['S48'].to_netcdf('/extra-space/sg-stats/June/S48/species.june.C96.nc')
    data['C90'].to_netcdf('/extra-space/sg-stats/June/C90/species.june.C96.nc')


    print(data['C98'])



