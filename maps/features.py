import rasterio
import cartopy.crs as ccrs
import pyproj
import matplotlib.colors

import cartopy.io.shapereader
import cartopy.feature
import shapely.geometry
import shapely.ops
import geopandas

import maps.defaults


def shaded_hills(ax, sr_path, azdeg=315, altdeg=45, vert_exag=10000, fraction=1.0):
    # sr_path is 1:10m Shaded Relief from Natural Earth
    #   https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/raster/SR_LR.zip
    (x1, x2, y1, y2) = ax.get_extent(ccrs.PlateCarree())

    img = rasterio.open(sr_path)
    img_top_left = img.index(x1, y2)
    img_bot_right = img.index(x2, y1)
    img = img.read(1)[img_top_left[0]:img_bot_right[0], img_top_left[1]:img_bot_right[1]]

    ls = matplotlib.colors.LightSource(azdeg=azdeg, altdeg=altdeg)
    ls.hillshade(img, vert_exag=vert_exag, fraction=fraction)

    # opacity colormap
    cm = matplotlib.colors.LinearSegmentedColormap.from_list('opacity_cmap', [(0, 0, 0, 0.4), (0, 0, 0, 0)])

    ax.imshow(img, origin='upper', extent=[x1, x2, y1, y2], transform=ccrs.PlateCarree(), cmap=cm, zorder=10, vmax=206)


def tiger_roads(ax, tiger_path, rttyp=('U', 'I'), **add_feature_kwargs):
    # TIGER/Line shapefiles:
    #   https://catalog.data.gov/dataset/tiger-line-shapefile-2016-nation-u-s-primary-roads-national-shapefile
    reader = cartopy.io.shapereader.Reader(tiger_path)
    geometries = []

    (x1, x2, y1, y2) = ax.get_extent(ccrs.PlateCarree())
    extent = shapely.geometry.Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

    for record in reader.records():
        if record.attributes['RTTYP'] in rttyp: #and extent.intersects(record.geometry) or extent.contains(record.geometry):
            geometries.append(record.geometry)

    add_feature_kwargs.setdefault('facecolor', 'none')
    add_feature_kwargs.setdefault('edgecolor', maps.defaults.edgecolor_roads)
    add_feature_kwargs.setdefault('linewidth', maps.defaults.linewidth_roads)
    ax.add_feature(
        cartopy.feature.ShapelyFeature(geometries, crs=ccrs.PlateCarree()),
        **add_feature_kwargs
    )

def add_roads(ax, shapefile_paths, **kwargs):
    roads = maps.find_shapefile(shapefile_paths, "tiger_roads")
    tiger_roads(ax, roads, **kwargs)


def add_hills(ax, shapefile_paths, **kwargs):
    sr = maps.find_shapefile(shapefile_paths, "naturalearth_sr")
    shaded_hills(ax, sr)


def add_polygons(ax, polygons: list, crs=ccrs.PlateCarree(), outline=False, exterior=False, biggest_n_shapes=1, **add_feature_kwargs):
    if isinstance(polygons, shapely.geometry.base.BaseGeometry):
        polygons = [polygons]
    if outline:
        add_feature_kwargs.setdefault('facecolor', 'none')
        add_feature_kwargs.setdefault('edgecolor', 'blue')
        add_feature_kwargs.setdefault('linewidth', 0.5)
    if exterior:
        new_polygons = []
        for polygon in polygons:
            if isinstance(polygon, shapely.geometry.MultiPolygon):

                holes = [p for p in polygon]
                box_shape = polygon.bounds
                polygon = shapely.geometry.box(box_shape[0] - 5, box_shape[1] - 5, box_shape[2] + 5, box_shape[3] + 5)

                holes_sorted_size = sorted(holes, key=lambda x: x.area, reverse=True)
                #polygon = polygon.difference(shapely.geometry.MultiPolygon(holes_sorted_size[0:2]))
                polygon = shapely.geometry.Polygon(polygon.exterior.coords, [h.exterior.coords for h in holes_sorted_size[:biggest_n_shapes]])
                # polygon = polygon.difference(holes_sorted_size[1])

                # for i, p in enumerate(holes):
                #     if p.area < exterior_area_thresh:
                #         continue
                #     polygon = polygon.difference(p)

                new_polygons.append(polygon)
            else:
                new_polygons.append(polygon.envelope.buffer(100).difference(polygon))
        polygons = new_polygons
    ax.add_feature(
        cartopy.feature.ShapelyFeature(polygons, crs=crs),
        **add_feature_kwargs
    )


def set_extent(ax, polygon):
    projection = pyproj.Proj(ax.projection.proj4_init)
    transform = pyproj.Transformer.from_proj('+proj=latlon', projection).transform
    polygon = shapely.ops.transform(transform, polygon)
    xmin, ymin, xmax, ymax = polygon.bounds
    ax.set_extent([xmin, xmax, ymin, ymax], crs=ax.projection)


def figsize_fitting_polygon(polygon, projection: ccrs.Projection, width=8):
    projection = pyproj.Proj(projection.proj4_init)
    transform = pyproj.Transformer.from_proj('+proj=latlon', projection).transform

    polygon = shapely.ops.transform(transform, polygon)
    xmin, ymin, xmax, ymax = polygon.bounds
    return width, width * (ymax-ymin)/(xmax-xmin)


def outlines(ax, coastlines=True, borders=True, states=False, lakes=True, **kwargs):
    if coastlines:
        temp_kwargs = kwargs.copy()
        temp_kwargs.setdefault('linewidth', maps.defaults.linewidth_coastlines)
        temp_kwargs.setdefault('edgecolor', maps.defaults.edgecolor_coastlines)
        ax.add_feature(cartopy.feature.COASTLINE, **temp_kwargs)
    if borders:
        temp_kwargs = kwargs.copy()
        temp_kwargs.setdefault('linewidth', maps.defaults.linewidth_borders)
        temp_kwargs.setdefault('edgecolor', maps.defaults.edgecolor_borders)
        ax.add_feature(cartopy.feature.BORDERS, **temp_kwargs)
    if states:
        temp_kwargs = kwargs.copy()
        temp_kwargs.setdefault('linewidth', maps.defaults.linewidth_states)
        temp_kwargs.setdefault('edgecolor', maps.defaults.edgecolor_states)
        ax.add_feature(cartopy.feature.STATES, **temp_kwargs)
    if lakes:
        temp_kwargs = kwargs.copy()
        temp_kwargs.setdefault('linewidth', maps.defaults.linewidth_lakes)
        temp_kwargs.setdefault('edgecolor', maps.defaults.edgecolor_lakes)
        temp_kwargs.setdefault('facecolor', maps.defaults.facecolor_lakes)
        ax.add_feature(cartopy.feature.LAKES, **temp_kwargs)


def format_page(ax, linewidth_axis_spines=None):
    ax.outline_patch.set_visible(True)
    if linewidth_axis_spines is None:
        linewidth_axis_spines = maps.defaults.linewidth_axis_spines
    ax.outline_patch.set_linewidth(linewidth_axis_spines)
