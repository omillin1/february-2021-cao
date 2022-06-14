import numpy as np
from netCDF4 import Dataset, num2date
import os
from datetime import datetime, timedelta
from glob import glob
import re
from matplotlib import cm, colors
from mpl_toolkits.basemap import Basemap


# Define the linear detrend function.
def LinearDetrend(y):
    """Linearly detrends an array of data with time in the first dimension.

    Parameters
    ---------
    y: The array to be detrended.

    Returns
    ---------
    detrendedData: The detrended data.
    longitude: The slope of the trend line.
    """
    time = np.arange(y.shape[0])

    E = np.ones((y.shape[0],2))
    E[:,0] = time

    invETE = np.linalg.inv(np.dot(E.T,E))
    xhat = invETE.dot(E.T).dot(y)
    trend = np.dot(E,xhat)
    detrendedData = y - trend

    return detrendedData,trend

# Define a normal colormap function.
def NormColorMap(cmap_select, my_bounds):
    """Creates a normalized color map.

    Parameters
    ---------
    cmap_select: String, the cmap to normalize.
    bounds: Numpy array, the array of levels in the colormap. Has to be diverging
            and equal on each side.

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
    lat_bound: Latitude boundary to plot with.
    cen_lon: longitude to center the map on.

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

# Define function to take consecutive array indices meeting a certain criteria.
def consec(arr, stepsize=1):
    """Splits an array where it meets a condition consecutively.

    Parameters
    ---------
    arr: numpy array 1d.
    stepsize: how many steps to split by.

    Returns
    ---------
    List of split arrays of indices in original array that meet certain criteria consecutively.
    """
    return np.split(arr, np.where(np.diff(arr) != stepsize)[0]+1)

# Define function to go through consec lists and get consec indexes with no more than one day falling below threshold.
def consec_relax(lst, length, sep):
    """Splits an array where it meets a condition consecutively.

    Parameters
    ---------
    lst: lst of consecutive arrays with indices where consecutive condition is met.
    length: minimum number of days that consecutively have to meet a threhold.
    sep_days: separation to not allow a relaxation to the consecutive rule.
    If you want one day in between consecutive conditions hits to be allowed, you set this to 2.

    Returns
    ---------
    final: New list of arrays containing consecutive indices for condition after relaxation.
    """
    final = []
    lstcopy = lst[:]
    while len(lst)>0:
        if (len(lst) == 1) & (lst[0].size >= length):
            final.append(lst[0])
            del lst[0]
        elif (len(lst) == 1) & (lst[0].size < length):
            del lst[0]
        elif ((lst[1][0] - lst[0][-1]) == sep):
            tmp = np.concatenate([lst[0],[lst[0][-1]+1],lst[1]])
            del lst[0:2]
            if tmp.size >= length:
                final.append(tmp)
        elif ((lst[1][0] - lst[0][-1]) != sep) & (lst[0].size >= length):
            final.append(lst[0])
            del lst[0]
        else:
            del lst[0]
    return final

# Calculate CAO start dates from ERA5 for Millin et al (2022).
def CAODates2022(t2m_data, t2m_time, day_no = 5, sep_days = 2, perc = 10, ind_sep = 4):
    """Finds CAO start dates for Millin et al. (2022).

    Parameters
    ---------
    t2m_data: 1D array of domain averaged daily temp anomalies.
    t2m_time: 1D array of datetimes corresponding to domain averaged temp anomalies.
    day_no: number of consecutive days to define CAO.
    sep_days: no of days that don't meet consecutive threshold for cao. If you want one day in between consecutive conditions hits to be allowed, you set this to 2.
    perc: percentile level for cold events.

    Returns
    ---------
    ind_time: Array of start times for independent cold air outbreaks in Millin et al. (2022).
    """

    # Get percentile threshold.
    percentile_temp = np.percentile(t2m_data, perc)

    # Get consecutive days' indexes below threshold.
    lst = consec(np.where(t2m_data <= percentile_temp)[0])

    # We don't want a CAO that starts on say 27th Feb and meets with the following December to count, so filter it out if there are any:
    lst_filter = []
    for i in range(len(lst)):
        if (t2m_time[lst[i][0]].month == 2) & (t2m_time[lst[i][-1]].month==12):
            pass
        else:
            lst_filter.append(lst[i])

    # Apply the algorithm for CAO events.
    updated_list = consec_relax(lst_filter, day_no, sep_days)

    # Define the array for the list of CAO start dates.
    CAO_start = []

    # For each list of indexes where the CAO is more than 5 days below 10th percetile, append the index of the first day this happens in the time array:
    for i in range(len(updated_list)):
        CAO_start.append(t2m_time[updated_list[i][0]])

    # Make dates into array.
    CAO_start = np.asarray(CAO_start)

    # This is the list to append the start dates of the independent CAOs.
    ind_time = []

    for i in range(len(CAO_start)):
        # Append first data point.
        if i == 0:
            ind_time.append(CAO_start[0])
            # If the next start date is less then the previous start date + separation + length of previous CAO then pass, else append to ind_time.
        elif CAO_start[i] < (CAO_start[i-1]+timedelta(days = ind_sep)+timedelta(days=len(updated_list[i-1]))):
            pass
        else:
            ind_time.append(CAO_start[i])

    # Make the list of independent CAO start dates an array:

    ind_time = np.asarray(ind_time)

    return ind_time


# Calculate CAO start dates from ERA5 for Millin et al (2022).
def WinterConsecIndEvent(data, time, day_no = 5, sep_days = 2, perc = 10, ind_sep = 4, thresh_type = 'less'):
    """Finds CAO start dates for Millin et al. (2022).

    Parameters
    ---------
    data: 1D array of data.
    time: 1D array of datetimes.
    day_no: number of consecutive days to define events.
    sep_days: no of days that don't meet consecutive threshold. If you want one day in between consecutive conditions hits to be allowed, you set this to 2.
    perc: percentile level for data.

    Returns
    ---------
    ind_time: Array of independent start times.
    """

    # Get percentile threshold.
    percentile = np.percentile(data, perc)
    if thresh_type == 'less':
        # Get consecutive days' indexes below threshold.
        lst = consec(np.where(data <= percentile)[0])
    elif thresh_type == 'more':
        lst = consec(np.where(data >= percentile)[0])


    # We don't want a consec that starts on say 27th Feb and meets with the following December to count, so filter it out if there are any:
    lst_filter = []
    for i in range(len(lst)):
        if (time[lst[i][0]].month == 2) & (time[lst[i][-1]].month==12):
            pass
        else:
            lst_filter.append(lst[i])

    # Apply the algorithm for consec events.
    updated_list = consec_relax(lst_filter, day_no, sep_days)

    # Define the array for the list of start dates.
    start = []

    # For each list of indexes where the consec is more than 5 days below 10th percetile, append the index of the first day this happens in the time array:
    for i in range(len(updated_list)):
        start.append(time[updated_list[i][0]])

    # Make dates into array.
    start = np.asarray(start)

    # This is the list to append the start dates of the independent consec events.
    ind_time = []

    for i in range(len(start)):
        # Append first data point.
        if i == 0:
            ind_time.append(start[0])
            # If the next start date is less then the previous start date + separation + length of previous consec event then pass, else append to ind_time.
        elif start[i] < (start[i-1]+timedelta(days = ind_sep)+timedelta(days=len(updated_list[i-1]))):
            pass
        else:
            ind_time.append(start[i])

    # Make the list of independent start dates an array:

    ind_time = np.asarray(ind_time)

    return ind_time

# Filter field function

def filter_field3D(data, wavenumber1, wavenumber2):
    """Filters field for wavenumber.

    Parameters
    ---------
    data: 3D data with longitude in last dimension.
    wavenumber1: beginning of wavenumber range you want.
    wavenumber2: end of wavenumber

    Returns
    ---------
    irfft: Array of filtered field.
    """
    rft_data = np.fft.rfft(data, axis = -1)
    rft_data[:, :, 0:wavenumber1]= 0
    rft_data[:, :, wavenumber2+1:] = 0
    irfft = np.fft.irfft(rft_data, axis =-1)

    return irfft

# Filter field function for wavenumbers with no end.

def filter_field3D_open(data, wavenumber):
    """Filters field for wavenumber.

    Parameters
    ---------
    data: 3D data with longitude in last dimension.
    wavenumber: beginning of wavenumber range you want.

    Returns
    ---------
    irfft: Array of filtered field.
    """
    rft_data = np.fft.rfft(data, axis = -1)
    rft_data[:, :, 0:wavenumber]= 0
    irfft = np.fft.irfft(rft_data, axis =-1)

    return irfft

# Define the linear detrend function.
def LinearRegression(x,var, inc):
    """Performs a linear regression between x and y data.

    Parameters
    ---------
    x: The x data to perform linear regression on.
    var: The y data to perform linear regression on.
    inc: The increment of the x data.

    Returns
    ---------
    y_line: The y-values of the trend line.
    x_graph: The x-values of the trend line.
    a: The slope of the linear regression line.
    b: The y-intercept of the linear regression line.
    """
    # Set up the matrix E.
    E = np.zeros((x.shape[0], 2))

    # input y variable
    y = var.copy()

    # We know y = Ex + n. x contains the regression coefficient.
    # Set first column of E to be the dates, and second column as ones.
    E[:, 0] = x
    E[:, 1] = 1

    # Now calculate xhat to get slope and y-intercept.
    inv_term = np.linalg.inv(np.dot(E.T, E))
    xhat = np.dot(np.dot(inv_term, E.T), y)

    # For equation y=ax+b:
    a = xhat[0]
    b = xhat[1]

    # Now createa straight line covering all the data you just had.
    minx = np.nanmin(x)
    maxx = np.nanmax(x)
    x_graph = np.arange(minx, maxx+inc, inc)
    y_line = (a*x_graph)+b

    return y_line, x_graph, a, b


# Test for github.
