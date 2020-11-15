import numpy as np

import maps

import cartopy.mpl.geoaxes
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors
import pyproj
import shapely.geometry
from tqdm import tqdm
import gcpy.grid
import scipy.interpolate
import matplotlib.patheffects as PathEffects
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.colorbar
from mpl_toolkits.axes_grid1 import AxesGrid


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


def compare_grid(grid):
    grid_xe = grid['lon_b']
    grid_ye = grid['lat_b']
    grid_xc = grid['lon']
    grid_yc = grid['lat']

    resolution = np.ones_like(grid_xc) * np.nan


    xx, yy = np.meshgrid(np.linspace(-180, 180,1000), np.linspace(-90, 90, 500), indexing='ij')

    pts = []
    data = []

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

        xy_laea = get_minor_xy(xe_laea, ye_laea)

        for i in range(n-1):
            for j in range(n-1):
                resolution[nf, i, j] = np.sqrt(shapely.geometry.Polygon(xy_laea[i, j, ...]).area/1e6)

        pts.extend(np.moveaxis((grid_xc[nf,...].flatten(), grid_yc[nf,...].flatten()), 0, -1))
        data.extend(resolution[nf,...].flatten())

        # pcm = ax.pcolormesh(xe, ye, resolution, transform=ccrs.PlateCarree(), norm=norm, cmap=cmap)
        # ax.contourf(grid_xc[nf,...], grid_yc[nf,...], resolution,  levels=[25, 50, 100, 200, 400, 800, 1600],transform=ccrs.PlateCarree(), norm=norm, cmap=cmap)

    data = np.array(data)
    pts = np.array(pts)
    interp = scipy.interpolate.LinearNDInterpolator(pts, data)
    xi = np.moveaxis((xx.flatten(), yy.flatten()), 0, -1)

    d = interp(xi)
    d = d.reshape(xx.shape)

    return grid_xe, grid_ye,  resolution, grid_xc, grid_yc, xx, yy, d



# grid1, _ = gcpy.grid.make_grid_SG(48, 2.3567, 264.0, 35.0)
# grid2, _ = gcpy.grid.make_grid_SG(96, 1, 170, -90)


def setup_axes(ax, plot_bbox):
    ax.coastlines(linewidth=0.4, color='#222222')
    for spine in ax.spines.values():
        spine.set_edgecolor('#222222')
        spine.set_linewidth(0.4)
    region = maps.get_countries('/home/liam/Downloads').loc['United States of America'].geometry
    maps.features.add_polygons(ax, region, outline=True, zorder=100, linewidth=1, edgecolor='k')
    ax.set_extent(plot_bbox)
    # ax.set_global()

def draw_major_grid_boxes(ax, xx, yy, x_split=180, **kwargs):
    kwargs.setdefault('linewidth', 1.8)
    kwargs.setdefault('color', 'black')

    xx_majors = [xx[0, :], xx[-1, :], xx[:, 0], xx[:, -1]]
    yy_majors = [yy[0, :], yy[-1, :], yy[:, 0], yy[:, -1]]
    for x, y in zip(xx_majors, yy_majors):
        if x_split:
            idx = np.argwhere(np.diff(np.sign(x % 360 - x_split))).flatten()
            x360 = x % 360
            idx = idx[(x360[idx] > 10) & (x360[idx] < 350)]
            start = [0, *(idx + 1)]
            end = [*(idx + 1), len(x)]
        else:
            start = [0]
            end = [len(x)]
        for s, e in zip(start, end):
            ax.plot(x[s:e], y[s:e], **kwargs)

def plot_grid_data(xe, ye, res, x, y, c, norm, cmap, bbox=None):
    for nf in range(6):
        if bbox is not None:
            x_mask = np.logical_or(x[nf,...] < bbox[0], x[nf,...] > bbox[1])
            y_mask = np.logical_or(y[nf,...] < bbox[2], y[nf,...] > bbox[3])
            mask = np.logical_or(x_mask, y_mask)
            pcm = ax.pcolormesh(xe[nf,...], ye[nf,...], np.ma.masked_array(res[nf,...], mask), transform=ccrs.PlateCarree(), norm=norm, cmap=cmap)
        else:
            pcm = ax.pcolormesh(xe[nf,...], ye[nf,...], res[nf,...], transform=ccrs.PlateCarree(), norm=norm, cmap=cmap)
        # xy = get_major_xy(xe[nf,...], ye[nf,...])
        # ax.plot(xy[...,0], xy[...,1], transform=ccrs.PlateCarree(), color='k')
        # draw_major_grid_boxes(ax, xe[nf,...], ye[nf,...], transform=ccrs.PlateCarree(), linewidth=1)
    return pcm





norm = matplotlib.colors.Normalize(vmin=40, vmax=62)
cmap = 'RdYlGn_r'

plot_bbox = [-125, -67, 2, 70]

fig = plt.figure(figsize=(4.72441, 3.25))



import cartopy.feature as cfeature
ax = plt.axes(projection=ccrs.EqualEarth())
ax.set_global()
ax.coastlines(linewidth=0.4, color='#222222')
ax.outline_patch.set_linewidth(0.5)
ax.outline_patch.set_edgecolor('#222222')

#setup_axes(ax, plot_bbox)
grid, _ = gcpy.make_grid_CS(180)
grid_xe, grid_ye, resolution, grid_xc, grid_yc, xx, yy, res_c = compare_grid(grid)
print(f'res: {resolution.min()}//{resolution.max()}')
pcm = plot_grid_data(grid_xe, grid_ye, resolution, grid_xc, grid_yc, res_c, norm, cmap)
plt.colorbar(pcm, ticks=[40, 51, 62], label='Resolution (km)', orientation='horizontal', pad=0.1, aspect=30)

plt.tight_layout()
plt.savefig('/home/liam/gmd-sg-manuscript-2020/figures/cs_diff.png', dpi=300, pad_inches=0.01)
plt.show()
