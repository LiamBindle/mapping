import numpy as np


import cartopy.crs as ccrs
import matplotlib.pyplot as plt

def central_angle(x0, y0, x1, y1):
    RAD2DEG = 180 / np.pi
    DEG2RAD = np.pi / 180

    x0 = x0 * DEG2RAD
    x1 = x1 * DEG2RAD
    y0 = y0 * DEG2RAD
    y1 = y1 * DEG2RAD

    return np.arccos(np.sin(y0) * np.sin(y1) + np.cos(y0) * np.cos(y1) * np.cos(np.abs(x0-x1))) * RAD2DEG

def euclidean_distance(x0, y0, x1, y1):
    return np.sqrt((x0-x1)**2 + (y0-y1)**2)



def get_area(xy_ll):
    laea = pyproj.Proj(f'+proj=laea +lon_0={xy_ll[0,0]} +lat_0={xy_ll[0,1]}')
    ll = laea.to_latlong()
    transform = pyproj.Transformer.from_proj(ll, laea, always_xy=True).transform
    x, y = transform(xy_ll[:,0], xy_ll[:,1])
    xy_laea = np.moveaxis([x, y], 0, -1)
    area = np.sqrt(shapely.geometry.Polygon(xy_laea).area/1e6)
    print('Area: ', area)
    return area

def draw_minor_grid_boxes(xx, yy):

    xy = get_minor_xy(xx, yy)

    def draw_line(x0, y0, x1, y1, linewidth=None, **kwargs):
        kwargs.setdefault('color', 'k')
        dist = central_angle(x0, y0, x1, y1)
        eucl_dist = euclidean_distance(x0, y0, x1, y1)

        dist_lower=0.1
        dist_upper=6
        lw_lower=0.02
        lw_upper=0.2

        if linewidth is None:
            linewidth = np.interp(np.log(dist), np.log([dist_lower, dist_upper]), [lw_lower, lw_upper])
        if eucl_dist > 180:
            x0 = x0 if x0 < 180 else x0-360
            x1 = x1 if x1 < 180 else x1-360
            ax.plot([x0, x1], [y0, y1], linewidth=linewidth, transform=ccrs.PlateCarree(), **kwargs)
        else:
            ax.plot([x0, x1], [y0, y1], linewidth=linewidth, transform=ccrs.PlateCarree(), **kwargs)

    for i in range(xy.shape[0]):
        for j in range(xy.shape[1]):
            draw_line(*xy[i, j, 0], *xy[i, j, 1])
            draw_line(*xy[i, j, 1], *xy[i, j, 2])
    for k in range(xy.shape[0]):
        draw_line(*xy[k, -1, 2], *xy[k, -1, 3])
        draw_line(*xy[0, k, 3], *xy[0, k, 4])
    major_xy = get_major_xy(xe, ye)
    get_area(major_xy)
    for k in range(major_xy.shape[0]-1):
        draw_line(major_xy[k, 0], major_xy[k, 1], major_xy[k+1, 0], major_xy[k+1, 1], linewidth=1, color='k', antialiased=True)

import pyproj

def get_minor_xy(xe, ye):
    p0 = slice(0, -1)
    p1 = slice(1, None)
    boxes_x = np.moveaxis(np.array([xe[p0, p0], xe[p1, p0], xe[p1, p1], xe[p0, p1], xe[p0, p0]]), 0, -1)
    boxes_y = np.moveaxis(np.array([ye[p0, p0], ye[p1, p0], ye[p1, p1], ye[p0, p1], ye[p0, p0]]), 0, -1)
    return np.moveaxis(np.array([boxes_x, boxes_y]), 0, -1)


def get_major_xy(xe, ye):
    boxes_x = np.moveaxis(np.array([*xe[:-1, 0], *xe[-1, :-1], *xe[1:, -1][::-1], *xe[0, :][::-1]]), 0, -1)
    boxes_y = np.moveaxis(np.array([*ye[:-1, 0], *ye[-1, :-1], *ye[1:, -1][::-1], *ye[0, :][::-1]]), 0, -1)
    return np.moveaxis(np.array([boxes_x, boxes_y]), 0, -1)

def draw_major_grid_boxes_naive(ax, xx, yy, **kwargs):
    kwargs.setdefault('color', 'k')
    kwargs.setdefault('linewidth', 0.8)
    xx_majors = [xx[0, :], xx[-1, :], xx[:, 0], xx[:, -1]]
    yy_majors = [yy[0, :], yy[-1, :], yy[:, 0], yy[:, -1]]
    for xm, ym in zip(xx_majors, yy_majors):
        ax.plot(xm, ym, transform=ccrs.PlateCarree(), **kwargs)


from tqdm import tqdm
import shapely.geometry.polygon

def draw_minor_boxes_slow(ax, grid):
    grid_xe = grid['lon_b']
    grid_ye = grid['lat_b']

    lw = lambda res: np.interp(res, [80, 120], [0.5, 2])

    for nf in tqdm(range(6)):
        xe = grid_xe[nf,...]
        ye = grid_ye[nf,...]
        n = xe.shape[0]
        logical_x = xe[n//2, n//2]
        logical_y = ye[n//2, n//2]

        laea = pyproj.Proj(f'+proj=laea +lon_0={logical_x} +lat_0={logical_y}')
        ll = laea.to_latlong()
        transform = pyproj.Transformer.from_proj(ll, laea, always_xy=True).transform

        xe_laea, ye_laea = transform(xe, ye)

        xy_ll = get_minor_xy(xe, ye)
        xy_laea = get_minor_xy(xe_laea, ye_laea)

        for i in range(n-1):
            for j in range(n-1):
                res = np.sqrt(shapely.geometry.Polygon(xy_laea[i, j, ...]).area/1e6)
                ax.add_geometries([shapely.geometry.Polygon(xy_ll[i, j, ...])], facecolor='none', linewidth=lw(res), crs=ccrs.PlateCarree())

import gcpy.grid
fig = plt.Figure(figsize=(4.72441, 2.5), dpi=300)
ax = fig.add_axes([0.02, 0, 0.96, 1], projection=ccrs.EqualEarth())

import cartopy.feature as cfeature
ax.set_global()
ax.add_feature(cfeature.OCEAN, linewidth=0, color='w')
ax.add_feature(cfeature.LAND, facecolor='none', linewidth=0, color='#bdbdbd')
ax.add_feature(cfeature.LAKES, linewidth=0, color='w')

ax.outline_patch.set_linewidth(0.4)
ax.outline_patch.set_edgecolor('gray')


fname='C180e-US.png'
grid, _ = gcpy.grid.make_grid_CS(24) #gcpy.grid.make_grid_SG(90, 10, 240.5, 37.2)
for nf in [1]: #tqdm(range(6)):
    xe = grid['lon_b'][nf, ...] % 360
    ye = grid['lat_b'][nf, ...]
    draw_minor_grid_boxes(xe, ye)

    # r_earth = 6378.1
    # plt.plot((xe[0,:-1]+xe[0,1:])/2, np.diff(xe[0,:])*np.pi/180 * r_earth / (10e3/24))


# plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
#                     hspace = 0, wspace = 0)
# plt.margins(0,0)

#fig.savefig(f'/home/liam/gmd-sg-manuscript-2020/figures/{fname}', dpi=300)
fig.savefig(f'foo.png', dpi=300)
# plt.show()


#