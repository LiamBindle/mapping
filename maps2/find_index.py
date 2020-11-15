import gcpy
import numpy as np
import shapely.geometry
import shapely.ops
import pyproj

def corners_to_xy(xc, yc):
    """ Creates xy coordinates for each grid-box. The shape is (n, n, 5) where n is the cubed-sphere size.

    :param xc: grid-box corner longitudes; shape (n+1, n+1)
    :param yc: grid-box corner latitudes; shape (n+1, n+1)
    :return: grid-box xy coordinates
    """
    p0 = slice(0, -1)
    p1 = slice(1, None)
    boxes_x = np.moveaxis(np.array([xc[p0, p0], xc[p1, p0], xc[p1, p1], xc[p0, p1], xc[p0, p0]]), 0, -1)
    boxes_y = np.moveaxis(np.array([yc[p0, p0], yc[p1, p0], yc[p1, p1], yc[p0, p1], yc[p0, p0]]), 0, -1)
    return np.moveaxis(np.array([boxes_x, boxes_y]), 0, -1)


def central_angle(x0, y0, x1, y1):
    """ Returns the distance (central angle) between coordinates (x0, y0) and (x1, y1). This is vectorizable.

    :param x0: pt0's longitude (degrees)
    :param y0: pt0's latitude  (degrees)
    :param x1: pt1's longitude (degrees)
    :param y1: pt1's latitude  (degrees)
    :return: Distance          (degrees)
    """
    RAD2DEG = 180 / np.pi
    DEG2RAD = np.pi / 180
    x0 = x0 * DEG2RAD
    x1 = x1 * DEG2RAD
    y0 = y0 * DEG2RAD
    y1 = y1 * DEG2RAD
    return np.arccos(np.sin(y0) * np.sin(y1) + np.cos(y0) * np.cos(y1) * np.cos(np.abs(x0-x1))) * RAD2DEG


if __name__ == '__main__':
    X_FIND = -110
    Y_FIND = 40

    cs_size = 24
    grid, _ = gcpy.make_grid_CS(cs_size)
    x_corners = grid['lon_b']
    y_corners = grid['lat_b']
    x_centers = grid['lon']
    y_centers = grid['lat']
    x_centers_flat = x_centers.flatten()
    y_centers_flat = y_centers.flatten()

    # Get XY polygon definitions for grid boxes
    xy = np.zeros((6, cs_size, cs_size, 5, 2))  # 5 (x,y) points defining polygon corners (first and last are same)
    for nf in range(6):
        xy[nf, ...] = corners_to_xy(xc=x_corners[nf, :, :], yc=y_corners[nf, :, :])

    # Find 4 shortest distances to (x_find, y_find)
    distances = central_angle(X_FIND, Y_FIND, x_centers_flat, y_centers_flat)
    four_nearest_indexes = np.argpartition(distances, 4)[:4]

    # Unravel 4 smallest indexes
    four_nearest_indexes = np.unravel_index(four_nearest_indexes, (6, cs_size, cs_size))
    four_nearest_xy = xy[four_nearest_indexes]
    four_nearest_polygons = [shapely.geometry.Polygon(polygon_xy) for polygon_xy in four_nearest_xy]

    # Transform to gnomonic projection
    gnomonic_crs = pyproj.Proj(f'+proj=gnom +lat_0={Y_FIND} +lon_0={X_FIND}')  # centered on (X_FIND. Y_FIND)
    latlon_crs = pyproj.Proj("+proj=latlon")
    gno_transform = pyproj.Transformer.from_proj(latlon_crs, gnomonic_crs, always_xy=True).transform
    four_nearest_polygons_gno = [shapely.ops.transform(gno_transform, polygon) for polygon in four_nearest_polygons]

    # Figure out which polygon contains te point
    XY_FIND = shapely.geometry.Point(X_FIND, Y_FIND)
    XY_FIND_GNO = shapely.ops.transform(gno_transform, XY_FIND)
    polygon_contains_point = [polygon.contains(XY_FIND_GNO) for polygon in four_nearest_polygons_gno]

    assert np.count_nonzero(polygon_contains_point) == 1
    polygon_with_point = np.argmax(polygon_contains_point)

    # Get original index
    nf = four_nearest_indexes[0][polygon_with_point]
    YDim= four_nearest_indexes[1][polygon_with_point]
    XDim= four_nearest_indexes[2][polygon_with_point]

    # Print result
    print(f"Searched for grid-box containing {X_FIND} E, {Y_FIND} N")
    print(f"    Grid-box index:   ({nf}, {YDim}, {XDim})")
    print(f"    Grid-box center:  {x_centers[nf, YDim, XDim]} E, {y_centers[nf, YDim, XDim]} N")