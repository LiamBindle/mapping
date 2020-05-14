import cartopy.feature as cfeature

import maps

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs


    plt.figure()
    ax = plt.axes(projection=ccrs.EqualEarth())
    ax.set_extent([-110, -85, 25, 50])

    # maps.shaded_hills(
    #     ax,

    #     '/home/liam/Downloads/SR_LR/SR_LR.tif',
    # )
    maps.tiger_roads(
        ax,
        '/home/liam/Downloads/tl_2016_us_primaryroads.shp',
    )
    maps.outlines(ax, states=True)
    maps.format_page(ax)
    plt.show()