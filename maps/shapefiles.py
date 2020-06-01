import os.path
import numpy as np
import geopandas
import shapely.geometry
import pyproj

from tqdm import tqdm

def contiguous_states():
    return [
        "Alabama", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
        "Delaware", "Florida", "Georgia", "Idaho", "Illinois", "Indiana", "Iowa",
        "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts",
        "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska",
        "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York",
        "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
        "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah",
        "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"
    ]


filename_lookup = {
    'tiger_states': ['tl_2017_us_state.shp', 'tl_2017_us_state/tl_2017_us_state.shp'],
    'tiger_roads': ['tl_2016_us_primaryroads.shp', 'tl_2016_us_primaryroads/tl_2016_us_primaryroads.shp'],
    #'tiger_counties': ['tl_2016_06_cousub.shp', 'tl_2016_06_cousub/tl_2016_06_cousub.shp'],
    'tiger_counties': ['CA_Counties_TIGER2016.shp', 'CA_Counties/CA_Counties_TIGER2016.shp'],
    'naturalearth_sr': ['SR_LR.tif', 'SR_LR/SR_LR.tif'],
    'naturalearth_countries': ['ne_10m_admin_0_map_subunits.shp', 'ne_10m_admin_0_map_subunits/ne_10m_admin_0_map_subunits.shp'],
    'naturalearth_states': ['ne_10m_admin_1_states_provinces.shp', 'ne_10m_admin_1_states_provinces/ne_10m_admin_1_states_provinces.shp'],
    'ca_air_basins': ['CaAirBasin.shp', 'CaAirBasin/CaAirBasin.shp'],
    'central_valley': ['cvhm_texture_regions.shp', 'cvhm_texture_regions/cvhm_texture_regions.shp']
}

def find_shapefile(shapefile_paths: list, item_name):
    if isinstance(shapefile_paths, str):
        shapefile_paths = [shapefile_paths]
    filenames = filename_lookup[item_name]
    for directory in shapefile_paths:
        for filename in filenames:
            if os.path.exists(os.path.join(directory, filename)):
                return os.path.join(directory, filename)
    raise ValueError(f"Can't find \"{item_name}\" in shapefile paths")


def get_tiger_states(shapefile_paths: list):
    # TIGER/Line shapefiles:
    #   https://www2.census.gov/geo/tiger/TIGER2017//STATE/tl_2017_us_state.zip
    shapefile = find_shapefile(shapefile_paths, "tiger_states")
    df = geopandas.read_file(shapefile).set_index('NAME')
    return df


def get_countries(shapefile_paths: list):
    # Natural Earth Admin 0 Subunits:
    #   https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_0_map_subunits.zip
    shapefile = find_shapefile(shapefile_paths, "naturalearth_countries")
    df = geopandas.read_file(shapefile).set_index('NAME')
    return df


def get_provinces_and_states(shapefile_paths: list):
    # Natural Earth Admin 1:
    #   https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_1_states_provinces.zip
    shapefile = find_shapefile(shapefile_paths, "naturalearth_states")
    df = geopandas.read_file(shapefile).set_index('name')
    return df

def tiger_states_to_contiguous_us(tiger_states, simplify_tolerance=0.01):
    tiger_states = tiger_states.loc[contiguous_states()]
    contiguous_us = shapely.geometry.MultiPolygon(tiger_states['geometry'].to_list()).convex_hull
    if simplify_tolerance > 0:
        return contiguous_us.simplify(simplify_tolerance)
    else:
        return contiguous_us


def central_valley(shapefile_paths):
    shapefile = find_shapefile(shapefile_paths, "central_valley")
    df = geopandas.read_file(shapefile)
    proj = pyproj.Transformer.from_proj(pyproj.Proj('+proj=aea +lat_0=23 +lon_0=-120 +lat_1=29.5 +lat_2=45.5 +units=m'), pyproj.Proj('epsg:4326'), always_xy=True).transform
    shape = shapely.ops.unary_union(df.geometry)
    shape = shapely.geometry.Polygon(shapely.ops.transform(proj, shape).buffer(0.01).exterior) #.convex_hull
    return shape


def central_basins(shapefile_paths):
    df = get_california_air_basins(shapefile_paths)
    df = df[df.index.isin(['San Francisco Bay', 'Mountain Counties', 'San Joaquin Valley', 'Sacramento Valley'])].to_crs('epsg:2163')
    return shapely.ops.unary_union(df.geometry)


def central_valley_and_bay_area(shapefile_paths):
    valley = central_valley(shapefile_paths)
    valley = shapely.ops.transform(pyproj.Transformer.from_crs('epsg:4326', 'epsg:2163', always_xy=True).transform, valley)
    bay_area = get_california_air_basins(shapefile_paths).to_crs('epsg:2163').loc['San Francisco Bay'].geometry

    connectors = get_california_counties(shapefile_paths).loc[['Yolo', 'Sacramento', 'San Joaquin', 'Stanislaus', 'Merced']].to_crs('epsg:2163').geometry

    area = shapely.geometry.Polygon(shapely.ops.unary_union([bay_area, valley, *connectors]).buffer(1000).exterior)
    return area #[bay_area, valley, *connectors]

def get_california_counties(shapefile_paths: list, north_central_sout=False):
    shapefile = find_shapefile(shapefile_paths, "tiger_counties")
    df = geopandas.read_file(shapefile).set_index('NAME')

    if north_central_sout:
        northern = [
            "Butte", "Colusa", "Del Norte", "Glenn", "Humboldt", "Lake", "Lassen", "Mendocino", "Modoc", "Plumas", "Shasta", "Siskiyou", "Tehama", "Trinity",
            "Amador", "El Dorado", "Marin", "Napa", "Nevada", "Placer", "Sacramento", "Sierra", "Solano", "Sonoma", "Sutter", "Yolo", "Yuba"
        ] # 14 + 13
        central = [
            "Alameda", "Contra Costa", "Monterey", "San Benito", "San Francisco", "San Mateo", "Santa Clara", "Santa Cruz",
            "Alpine", "Calaveras", "Fresno", "Inyo", "Kings",  "Kern", "Madera", "Mariposa", "Merced", "Mono", "San Joaquin", "Stanislaus", "Tulare", "Tuolumne",
            "San Luis Obispo", "Santa Barbara"
        ] # 8 + 14 + 2
        southern = [
            "Los Angeles", "Ventura",
            "Imperial", "Orange", "Riverside", "San Bernardino", "San Diego"
        ] # 2 + 5

        df = df.to_crs('epsg:4326')

        # US = get_countries(shapefile_paths).to_crs('epsg:4326').loc['United States of America'].geometry
        import shapely.ops

        def list_to_polygon(counties, req_len):
            subdf = df[df.index.isin(counties)]
            assert len(subdf) == req_len
            return shapely.ops.unary_union(subdf.geometry)
            # geometries = []
            # for p in subdf.geometry.to_list():
            #     if isinstance(p, shapely.geometry.MultiPolygon):
            #         for subp in p:
            #             if US.intersects(subp):
            #                 geometries.append(subp)
            #     else:
            #         geometries.append(p)
            # return shapely.geometry.MultiPolygon(geometries).convex_hull
        norther_counties = list_to_polygon(northern, 27)
        central_counties = list_to_polygon(central, 24)
        southern_counties = list_to_polygon(southern, 7)
        return norther_counties, central_counties, southern_counties
    else:
        return df


def get_california_air_basins(shapefile_paths: list):
    shapefile = find_shapefile(shapefile_paths, "ca_air_basins")
    df = geopandas.read_file(shapefile).set_index('NAME')
    return df



def mask_outside(x, y, polygon: shapely.geometry.MultiPolygon):
    x_flat = x.flatten()
    y_flat = y.flatten()
    points = [shapely.geometry.Point(xp, yp) for xp, yp in zip(x_flat, y_flat)]
    envelope = polygon.envelope
    mask = np.array([envelope.contains(pt) for pt in points])
    convex_hull = polygon.convex_hull
    for i in tqdm(np.argwhere(mask), desc='Region mask (1/2)'):
        mask[i.item()] = convex_hull.contains(points[i.item()])
    for i in tqdm(np.argwhere(mask), desc='Region mask (2/2)'):
        mask[i.item()] = polygon.contains(points[i.item()])
    return ~mask.reshape(x.shape)


if __name__ == '__main__':
    # shp = get_tiger_states('/home/liam/Downloads')
    # cus = tiger_states_to_contiguous_us(shp)

    import maps.features
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    shp = get_countries('/home/liam/Downloads/')
    # region = shp.loc['United States of America'].geometry
    shp = get_provinces_and_states('/home/liam/Downloads/')
    region = shp.loc['California'].geometry

    fig = plt.figure(figsize=maps.figsize_fitting_polygon(region, projection=ccrs.epsg(2163)))
    ax = plt.axes(projection=ccrs.epsg(2163))
    ax.set_facecolor('black')
    maps.set_extent(ax, region)
    maps.features.format_page(ax, linewidth_axis_spines=0)
    # maps.features.add_polygons(ax, shp.loc[contiguous_states()]['geometry'])

    # poly = [p for p in region]
    #
    # region = region.envelope.buffer(100)
    # for p in poly:
    #     region = region.difference(p)

    # valley = central_valley('/home/liam/Downloads/')
    # foo = central_valley_and_bay_area('/home/liam/Downloads')
    # foo = central_valley('/home/liam/Downloads')
    foo = maps.get_california_air_basins('/home/liam/Downloads').to_crs('epsg:2163').to_crs('epsg:4326').loc['San Francisco Bay'].geometry
    # maps.add_polygons(ax, valley, outline=True)
    maps.add_polygons(ax, foo, outline=True, crs=ccrs.PlateCarree())

    maps.features.add_polygons(ax, region, exterior=True)
    plt.tight_layout()
    # plt.show()


    # ax.set_extent([-110, -85, 25, 50])
    # maps.outlines(ax, states=True)
    plt.show()

    print('Done')
