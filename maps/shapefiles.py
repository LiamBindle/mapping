import os.path
import numpy as np
import geopandas
import shapely.geometry

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
    'tiger_counties': ['tl_2016_06_cousub.shp', 'tl_2016_06_cousub/tl_2016_06_cousub.shp'],
    'naturalearth_sr': ['SR_LR.tif', 'SR_LR/SR_LR.tif'],
    'naturalearth_countries': ['ne_10m_admin_0_map_subunits.shp', 'ne_10m_admin_0_map_subunits/ne_10m_admin_0_map_subunits.shp'],
    'naturalearth_states': ['ne_10m_admin_1_states_provinces.shp', 'ne_10m_admin_1_states_provinces/ne_10m_admin_1_states_provinces.shp'],
    'ca_air_basins': ['CaAirBasin.shp', 'CaAirBasin/CaAirBasin.shp']
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


def get_california_counties(shapefile_paths: list):
    shapefile = find_shapefile(shapefile_paths, "tiger_counties")
    df = geopandas.read_file(shapefile).set_index('NAME')
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
    region = shp.loc['United States of America'].geometry
    # shp = get_provinces_and_states('/home/liam/Downloads/')
    # region = shp.loc['California'].geometry

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

    maps.features.add_polygons(ax, region, exterior=True)
    plt.tight_layout()
    # plt.show()


    # ax.set_extent([-110, -85, 25, 50])
    # maps.outlines(ax, states=True)
    plt.show()

    print('Done')
