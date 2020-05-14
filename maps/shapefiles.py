import os.path
import geopandas
import shapely.geometry

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
    'naturalearth_sr': ['SR_LR.tif', 'SR_LR/SR_LR.tif'],
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


def tiger_states_to_contiguous_us(tiger_states, simplify_tolerance=0.01):
    tiger_states = tiger_states.loc[contiguous_states()]
    contiguous_us = shapely.geometry.MultiPolygon(tiger_states['geometry'].to_list()).convex_hull
    if simplify_tolerance > 0:
        return contiguous_us.simplify(simplify_tolerance)
    else:
        return contiguous_us


if __name__ == '__main__':
    shp = get_tiger_states('/home/liam/Downloads')
    cus = tiger_states_to_contiguous_us(shp)

    import maps.features
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    CA = shp.loc['California'].geometry

    fig = plt.figure(figsize=maps.figsize_fitting_polygon(CA, 5))
    ax = plt.axes(projection=ccrs.epsg(2163))
    # maps.set_extent(ax, CA)
    maps.features.format_page(ax, linewidth_axis_spines=0)
    maps.features.add_polygons(ax, shp.loc[contiguous_states()]['geometry'])
    plt.tight_layout()
    # plt.show()


    # ax.set_extent([-110, -85, 25, 50])
    # maps.outlines(ax, states=True)
    plt.show()

    print('Done')
