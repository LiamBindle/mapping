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

    region = maps.get_provinces_and_states('/home/liam/Downloads').loc['California'].geometry # maps.get_countries('/home/liam/Downloads/').loc['United States of America'].geometry

    in_region = []
    face5 = []
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
                if region.contains(shapely.geometry.Point(grid_xc[nf, i, j], grid_yc[nf, i, j])):
                    in_region.append(resolution[nf, i, j])
                if nf == 5:
                    face5.append(resolution[nf, i, j])

        pts.extend(np.moveaxis((grid_xc[nf,...].flatten(), grid_yc[nf,...].flatten()), 0, -1))
        data.extend(resolution[nf,...].flatten())

    print(f'region stats:\n-- min: {np.min(in_region)}\n-- max: {np.max(in_region)}\n-- avg: {np.mean(in_region)}\n-- count: {len(in_region)}')
    print(f'face5 stats:\n-- min: {np.min(face5)}\n-- max: {np.max(face5)}\n-- avg: {np.mean(face5)}\n-- count: {len(face5)}')

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

grid1, _ = gcpy.grid.make_grid_SG(90, 10, 240.5, 37.2)
grid2, _ = gcpy.grid.make_grid_SG(90, 1, 170, -90)


def setup_axes(ax, plot_bbox):
    ax.coastlines(linewidth=0.4, color='#222222')
    for spine in ax.spines.values():
        spine.set_edgecolor('#222222')
        spine.set_linewidth(0.4)
    region = maps.get_provinces_and_states('/home/liam/Downloads').loc['California'].geometry
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





# norm = matplotlib.colors.LogNorm(vmin=80, vmax=500)
#norm = matplotlib.colors.Normalize(vmin=75, vmax=125)
norm = matplotlib.colors.Normalize(vmin=10, vmax=125)
cmap = 'RdYlGn_r'

projection=ccrs.LambertAzimuthalEqualArea(central_latitude=39, central_longitude=-96)
mask_box = [-170, -10, -5, 85]
plot_bbox = [-125, -67, 2, 70]

fig = plt.figure(figsize=(4.72441, 2.5*2-0.5))

axes_class = (cartopy.mpl.geoaxes.GeoAxes,
              dict(map_projection=projection))

# grid = AxesGrid(fig, 111, axes_class=axes_class,
#                 nrows_ncols=(1, 2),
#                 axes_pad=0.05,
#                 cbar_mode='single',
#                 cbar_location='right',
#                 cbar_pad=0.2,
#                 cbar_size='10%',
#                 label_mode=''
#                 )



# -------------
# ax = fig.add_subplot(1,2,1, projection=projection)
ax = fig.add_axes([0.02,0.05+0.5, 0.37, 0.8/2], projection=projection)
# ax = grid.axes_all[0]
setup_axes(ax, plot_bbox)
grid_xe, grid_ye, resolution, grid_xc, grid_yc, xx, yy, res_c = compare_grid(grid2)
print(f'res: {resolution.min()}//{resolution.max()}')
plot_grid_data(grid_xe, grid_ye, resolution, grid_xc, grid_yc, res_c, norm, cmap, mask_box)
plt.title('C90-global', fontsize=10)

grid1_xe = grid_xe
grid1_ye = grid_ye

xx1 = xx
yy1 = yy
cc1 = res_c

# -------------
#ax = fig.add_subplot(1,2,2, projection=projection)
# ax = grid.axes_all[1]
ax = fig.add_axes([0.41, 0.05+0.5, 0.37, 0.8/2], projection=projection)
setup_axes(ax, plot_bbox)
grid_xe, grid_ye, resolution, grid_xc, grid_yc, xx, yy, res_c = compare_grid(grid1)

print(f'res: {resolution.min()}//{resolution.max()}')

pcm = plot_grid_data(grid_xe, grid_ye, resolution, grid_xc, grid_yc, res_c, norm, cmap, mask_box)
plt.title('C900e-CA', fontsize=10)

grid2_xe = grid_xe
grid2_ye = grid_ye

# -------------
# ax = fig.add_subplot(gs[0, 2])
ax = fig.add_axes([0.8, 0.05+0.5, 0.03, 0.8/2])
# plt.colorbar(pcm, cax=ax, ticks=[40, 75, 100], label='Resolution (km)', extend='max')
plt.colorbar(pcm, cax=ax, ticks=[10, 25, 50, 75, 100, 125], label='Resolution (km)', extend='max')
# cbar = grid.cbar_axes[0].colorbar(pcm, ticks=[75, 100, 125])
# cbar.ax.set_ylabel('Resolution (km)', rotation=270)

# ax = fig.add_subplot(gs[1, :], projection=ccrs.EqualEarth())
import cartopy.feature as cfeature
ax = fig.add_axes([0.02, 0.02, 0.96, 0.48], projection=ccrs.EqualEarth())
ax.set_global()
region = maps.get_provinces_and_states('/home/liam/Downloads').loc['California'].geometry
maps.features.add_polygons(ax, region, outline=True, zorder=100, linewidth=1, edgecolor='k')
ax.add_feature(cfeature.OCEAN, linewidth=0, color='#f0f0f0')
ax.add_feature(cfeature.LAND, facecolor='none', linewidth=0, color='#bdbdbd')
ax.add_feature(cfeature.LAKES, linewidth=0, color='#f0f0f0')
ax.outline_patch.set_linewidth(0.5)
ax.outline_patch.set_edgecolor('gray')
for nf in range(6):
    cmap = plt.get_cmap('Dark2')
    draw_major_grid_boxes(ax, grid1_xe[nf,...], grid1_ye[nf,...], transform=ccrs.PlateCarree(), color=cmap(6), zorder=101)
    draw_major_grid_boxes(ax, grid2_xe[nf,...], grid2_ye[nf,...], transform=ccrs.PlateCarree(), color=cmap(4), zorder=102)

# ax.contour(xx1, yy1, cc1, transform=ccrs.PlateCarree(), levels=[100, 200, 400], colors=cmap(3), linestyles='dashed')
ax.contour(xx, yy, res_c, transform=ccrs.PlateCarree(), levels=[100, 500, 1000], colors=cmap(4), linestyles=['dashed', 'dashdot', 'dotted'], linewidths=[0.9, 0.9, 0.9], inlines=True)

from matplotlib.lines import Line2D
custom_lines = [
    Line2D([0], [0], color=cmap(6), lw=2),
    Line2D([0], [0], color=cmap(4), lw=2)
    ]


ax.annotate('100 km', xy=ccrs.EqualEarth().transform_point(-120.07, 1.21, ccrs.PlateCarree()), xycoords='data', color=cmap(4),
            xytext=(0.20, 0.35), textcoords='axes fraction', arrowprops=dict(arrowstyle="->", edgecolor=cmap(4)), horizontalalignment='center', verticalalignment='center')

ax.annotate('500 km', xy=ccrs.EqualEarth().transform_point(-80.27, -33.65, ccrs.PlateCarree()), xycoords='data', color=cmap(4),
            xytext=(0.3, 0.05), textcoords='axes fraction', arrowprops=dict(arrowstyle="->", edgecolor=cmap(4)), horizontalalignment='center', verticalalignment='center')

ax.annotate('1000 km', xy=ccrs.EqualEarth().transform_point(82.11, 7.45, ccrs.PlateCarree()), xycoords='data', color=cmap(4),
            xytext=(0.7, 0.4), textcoords='axes fraction', arrowprops=dict(arrowstyle="->", edgecolor=cmap(4)), horizontalalignment='center', verticalalignment='center')



legend = ax.legend(custom_lines, ['C90-global', 'C900e-CA'], loc='upper right')
legend.set_zorder(200)

plt.tight_layout()
# plt.savefig('/home/liam/gmd-sg-manuscript-2020/figures/new3.png', dpi=300, pad_inches=0.01)
plt.show()
