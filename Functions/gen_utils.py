import numpy as np
from datetime import datetime, timedelta
from matplotlib import cm, colors
from mpl_toolkits.basemap import Basemap

# Define a normal colormap function.
def NormColorMap(cmap_select, my_bounds):
    """Creates a normalized color map.

    Parameters
    ---------
    cmap_select: String, the cmap to normalize.
    bounds: Numpy array, the array of levels in the colormap. Has to be diverging.

    Returns
    ---------
    my_cmap: The normalized colormap.
    norm: The normalization array.
    """
    cmap = cm.get_cmap(cmap_select)
    bounds = my_bounds
    numColors = bounds.size - 1
    my_colors = []
    for i in range(1,numColors+1):
        if (i != (numColors / 2.)):
            my_colors.append(cmap(i/float(numColors)))
        else:
            my_colors.extend(['white']*2)

    my_cmap = colors.ListedColormap(my_colors)
    my_cmap.set_under(cmap(0.0))
    my_cmap.set_over(cmap(1.0))
    norm = colors.BoundaryNorm(bounds, my_cmap.N)

    return my_cmap, norm

# Define a npstere map function.
def NPStere_Map(lat_bound = 21, cen_lon = -100):
    """Creates a normalized color map.

    Parameters
    ---------
    lat_bound: Integer or float, latitude boundary to plot with.
    cen_lon: integer or float, longitude to center the map on.

    Returns
    ---------
    map: The map projection.
    """
    map = Basemap(projection = 'npstere', resolution = 'l', boundinglat = lat_bound, lon_0 = cen_lon, round = True)
    map.drawcoastlines(linewidth=0.25)
    map.drawcountries(linewidth=0.25)

    return map

# Define a npstere map function.
def LambConfMap(width = 7000000, height = 6100000, area_thresh = 1000., lat_range = [35., 55.], center_range = [45., -100.]):
    """Creates a normalized color map. DEFAULT IS FOR N AMERICA.

    Parameters
    ---------
    width: Width of map region.
    height: Height of map region.
    area_thresh: area threshold parameter, default 1000.
    lat_range: list of two latitudes to go between.
    center_range = list of two values of [center_lat, center_lon] to center map on.


    Returns
    ---------
    map: The map projection.
    """
    map = Basemap(width=width ,height=height , resolution='l',area_thresh=area_thresh,projection='lcc',lat_1=lat_range[0],lat_2=lat_range[1],lat_0=center_range[0],lon_0=center_range[1]) # Replace with 6000000, 5000000 for N America
    map.drawcoastlines(linewidth=0.25)
    map.drawcountries(linewidth=0.25)
    map.drawstates(linewidth=0.25)

    return map

# Draw box on basemap object.
def DrawPolygon(map, lat_range = [48, 30], lon_range = [256.5, 268.5], grid_space = 1.5, lw = 3, color = 'black'):
    """Draws a box on a basemap map.

    Parameters
    ---------
    lat1: highest latitude point in degrees north.
    lat2: lowest latitude point in degrees north.
    lon1: furthest west longitude in degrees east.
    lon2: furthest east longitude in degrees west.
    grid_space: grid spacing in degrees.
    map_type: string identifier.

    Returns
    ---------
    p1: Polygon line 1.
    p2: Polygon line2.
    p3 Polygon line3.
    p4: Polygon line4.
    """

    # Draw polygon 1.
    bottomLon1 = np.arange(lon_range[0], lon_range[1]+grid_space, grid_space)
    bottomLat1 = np.repeat(lat_range[1], bottomLon1.size)
    topLon1 = np.arange(lon_range[0], lon_range[1]+grid_space, grid_space)
    topLat1 = np.repeat(lat_range[0], topLon1.size)
    leftLat1 = np.arange(lat_range[0], lat_range[1]-grid_space, -grid_space)
    leftLon1 = np.repeat(lon_range[0], leftLat1.size)
    rightLat1 = np.arange(lat_range[0], lat_range[1]-grid_space, -grid_space)
    rightLon1 = np.repeat(lon_range[1],rightLat1.size)

    p1 = map.plot(bottomLon1, bottomLat1, latlon=True, lw = lw, color=color)
    p2 = map.plot(topLon1, topLat1, latlon=True, lw = lw, color=color)
    p3 = map.plot(leftLon1, leftLat1, latlon=True, lw = lw, color=color)
    p4 = map.plot(rightLon1, rightLat1, latlon=True, lw = lw, color=color)

    return p1, p2, p3, p4
