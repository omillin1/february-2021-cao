import numpy as np
from netCDF4 import Dataset, num2date
import os
from datetime import datetime
from glob import glob
import re
from scipy import stats

def load_latlon(nc):
    """Loads in the latitudes and longitudes for S2S models.

    Parameters
    ---------
    nc: The netCDF4.Dataset read object for the nc file.

    Returns
    ---------
    latitude: The array of latitudes.
    longitude: The array of longitudes
    """
    latitude = nc.variables['latitude'][:]
    latitude = np.where(latitude.mask, np.nan, latitude.data)
    longitude = nc.variables['longitude'][:]
    longitude = np.where(longitude.mask, np.nan, longitude.data)
    return latitude, longitude

def load_modelparam(nc):
    """Loads in day number, ensemble member numbers and the start time for S2S models.

    Parameters
    ---------
    nc: The netCDF4.Dataset read object for the nc file.

    Returns
    ---------
    day_init: Array of integers for days in the forecast.
    number: Array of integers representing ensemble member.
    start_time: Datetime object representing the first day in the forecast.
    """
    day_init = num2date(nc.variables['time'][:],nc.variables['time'].units,nc.variables['time'].calendar,only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    number = nc.variables['number'][:]
    start_time = day_init[0]
    return day_init, number, start_time

def load_model_ltm_2D(model = 'ECMWF', param = 'surfT', lat_bounds = [90, 9], lon_bounds = [0, 358.5]):
    """Loads in ltm S2S model data for given lat, lon bounds.

    Parameters
    ---------
    nc: The netCDF4.Dataset read object for the nc file.

    Returns
    ---------
    ltm_latitude: Array of floats for latitude.
    ltm_longitude: Array of floats for longitude.
    ltm_dates: Array of python datetimes representing the initialization dates of the hindcast climo.
    ltm_days: Array of python integers representing the days in the forecast.
    ltm_region: Array of the 2D variable restricted to lat and lon bounds.
    """
    # Open the data file for the climo.
    nc = Dataset(f'climo_{model}_{param}_hindcast.nc', 'r')
    # Read in lat, lon.
    ltm_latitude, ltm_longitude = load_latlon(nc)
    # Read in hindcast dates, days array.
    ltm_dates = num2date(nc.variables['hdates'][:], nc.variables['hdates'].units, calendar = 'gregorian', only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    ltm_days = nc.variables['days'][:]
    # Read in the ltm data.
    ltm_var = nc.variables[param][:] # Shape (init date, lead time, lat, lon)
    # Close file.
    nc.close()
    # Find lat, lon inds for ltm.
    ltm_lat_ind1, ltm_lat_ind2 = np.where(ltm_latitude == lat_bounds[0])[0][0], np.where(ltm_latitude == lat_bounds[1])[0][0]
    ltm_lon_ind1, ltm_lon_ind2 = np.where(ltm_longitude == lon_bounds[0])[0][0], np.where(ltm_longitude == lon_bounds[1])[0][0]
    # Now restrict ltm spatially to the lat, lon regions.
    ltm_region = ltm_var[:, :, ltm_lat_ind1:ltm_lat_ind2+1, ltm_lon_ind1:ltm_lon_ind2+1]
    return ltm_latitude, ltm_longitude, ltm_dates, ltm_days, ltm_region

def load_model_ltm_3D(model = 'ECMWF', param = 'hgt', lat_bounds = [90, 9], lon_bounds = [0, 358.5], level = 500):
    """Loads in ltm latitudes, longitudes, hindcast dates, days array and the ltm variable for S2S models.

    Parameters
    ---------
    nc: The netCDF4.Dataset read object for the nc file.

    Returns
    ---------
    ltm_latitude: Array of floats for latitude.
    ltm_longitude: Array of floats for longitude.
    ltm_levels: Array of floats for pressure levels.
    ltm_dates: Array of python datetimes representing the initialization dates of the hindcast climo.
    ltm_days: Array of python integers representing the days in the forecast.
    ltm_var: Array of the 3D variable restricted to lat, lon, and level bounds.
    """
    # Open the data file for the climo.
    nc = Dataset(f'climo_{model}_{param}_hindcast.nc', 'r')
    # Read in lat, lon.
    ltm_latitude, ltm_longitude = load_latlon(nc)
    ltm_levels = nc.variables['level'][:]
    # Read in hindcast dates, days array.
    ltm_dates = num2date(nc.variables['hdates'][:], nc.variables['hdates'].units, calendar = 'gregorian', only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    ltm_days = nc.variables['days'][:]
    # Read in the ltm data.
    ltm_var = nc.variables[param][:] # Shape (init date, lead time, lat, lon)
    # Close file.
    nc.close()
    # Find lat, lon inds for ltm.
    ltm_lat_ind1, ltm_lat_ind2 = np.where(ltm_latitude == lat_bounds[0])[0][0], np.where(ltm_latitude == lat_bounds[1])[0][0]
    ltm_lon_ind1, ltm_lon_ind2 = np.where(ltm_longitude == lon_bounds[0])[0][0], np.where(ltm_longitude == lon_bounds[1])[0][0]
    # Find level index.
    level_ind = np.where(ltm_levels == level)[0][0]
    # Restrict ltm to the region desired.
    ltm_region = ltm_var[:, :, level_ind, ltm_lat_ind1:ltm_lat_ind2+1, ltm_lon_ind1:ltm_lon_ind2+1]
    return ltm_latitude, ltm_longitude, ltm_levels, ltm_dates, ltm_days, ltm_region

# Write a function for this.
def Extreme_T2M_Thresholds_HC(model = 'ECWMF', init = '01-04', lat_range = [90, 0], lon_range = [0, 358.5], perc = 5):
    """Calculates lead-dependent extreme event thresholds in the T2M field from a hindcast climatology at a given initialisation.

    Parameters
    ----------
    model: A string containing the name of the model.
    init: A string containing the MM-DD format for the chosen initialisation.
    lat_range: A length 2 list containing the latitude range from highest to lowest latitude in degrees.
    lon_range: A length 2 list containing the longitude range from west to east.
    perc: An integer for the percentile to use to calculate the extreme event thresholds.

    Returns
    ----------
    thresh_array: An array of extreme event thresholds of shape (lead time, lat, lon) for a given initialisation.
    start_time: A datetime object for the initialisation date.
    """
    # Go to directory with all s2s hindcast data required.
    path = f'/share/data1/Students/ollie/CAOs/Data/Feb_2021_CAO/Model_Data/{model}/surfT'
    dir = os.chdir(path)

    # Import a test file to get the lat, lon, level etc for reference. Perturbed file is (days, ens, level, lat, lon)
    filename_pert = f'{model}_surfT_2001-{init}_perturbed.nc'
    filename_cont = f'{model}_surfT_2001-{init}_control.nc'
    nc = Dataset(filename_pert, 'r')
    latitude, longitude = load_latlon(nc)
    day_init, number, start_time = load_modelparam(nc)
    nc.close()

    # Get lat and lons of selected domain.
    lat_ind1, lat_ind2 = np.where(latitude == lat_range[0])[0][0], np.where(latitude == lat_range[1])[0][0]
    lon_ind1, lon_ind2 = np.where(longitude == lon_range[0])[0][0], np.where(longitude == lon_range[1])[0][0]
    lat_region = latitude[lat_ind1:lat_ind2+1]
    lon_region = longitude[lon_ind1:lon_ind2+1]

    # Now get a list of filenames for your given initialisation for control and perturbed files.
    date_string = init
    pert_string = f'*{date_string}_perturbed.nc'
    con_string = f'*{date_string}_control.nc'
    pert_files = glob(pert_string)
    con_files = glob(con_string)
    pert_files.sort()
    con_files.sort()

    # Define an array to store the model data in.
    data_store = np.zeros((len(pert_files), len(day_init), len(number)+1, len(lat_region), len(lon_region)))

    # Loop through each perturbation and control file, concatenate together for all ens members and then save in array.
    for f in range(len(pert_files)):
        nc_pert = Dataset(pert_files[f], 'r')
        pert_data = nc_pert.variables['t2m'][:, :, lat_ind1:lat_ind2+1, lon_ind1:lon_ind2+1]
        nc_pert.close()
        nc_con = Dataset(con_files[f], 'r')
        con_data = nc_con.variables['t2m'][:, lat_ind1:lat_ind2+1, lon_ind1:lon_ind2+1]
        nc_con.close()
        data_store[f] = np.concatenate((con_data[:, None, :, :], pert_data), axis = 1)

    # Now calculate the thresholds from anomalies.
    # Calculate mean over year and member.
    mean_yr_mem = np.nanmean(data_store, axis = (0, 2))
    # Now calculate anomalies from hindcasts.
    anom_hc = data_store - mean_yr_mem[None, :, None, :, :]
    # Now calculate percentiles of anoms over years and member space.
    thresh_array = np.percentile(anom_hc, q = perc, axis = (0, 2))

    return thresh_array, start_time

def sort_file_realtime(inits = ['2021-01-14', '2021-02-11'], model = 'ECMWF', var = 'surfT'):
    """Calculates positional indices for realtime initializations in a list of filenames and outputs the sorted filenames

    Parameters
    ----------
    inits: list containing two strings for the first and last initialization dates required.
    model: string containing the name of the model.
    var: string containing the name of the variable.

    Returns
    ----------
    files_pert: list of perturbed realtime filenames.
    files_con: list of control realtime filenames.
    pert_ind1: index of first initialization in files_pert.
    pert_ind2: index of second initialization in files_pert.
    con_ind1: index of first initialization in files_con.
    con_ind2: index of second initialization in files_con.
    model_dates_pert: list of all perturbed model run dates.
    model_dates_con: list of all control model run dates.
    """
    # Do perturbation indexing.
    # Get all filenames that feature "perturbed.nc" and sort them.
    model_dir_pert = f'/data/deluge/models/S2S/realtime/{model}/{var}/*perturbed.nc'
    files_pert = glob(model_dir_pert)
    files_pert.sort()
    # Now split the names into strings of just the init date form "YYYY-MM-DD".
    model_dates_pert = [i.split("_")[2] for i in files_pert]
    # Find indices where our selected init date limit lie within the strings.
    pert_ind1 = model_dates_pert.index(inits[0])
    pert_ind2 = model_dates_pert.index(inits[1])

    # Do control indexing for ECMWF.
    # Get all filenames that feature "control.nc" and sort them.
    model_dir_con = f'/data/deluge/models/S2S/realtime/{model}/{var}/*control.nc'
    files_con = glob(model_dir_con)
    files_con.sort()
    # Now split the names into strings of just the init date form "YYYY-MM-DD".
    model_dates_con = [i.split("_")[2] for i in files_con]
    # Find indices where our selected init date limit lie within the strings.
    con_ind1 = model_dates_con.index(inits[0])
    con_ind2 = model_dates_con.index(inits[1])

    return files_pert, files_con, pert_ind1, pert_ind2, con_ind1, con_ind2, model_dates_pert, model_dates_con

def calc_prob_extreme(data, x, thresholds):
    """Calculates probability of an extreme cold T2M anomaly event based on fitting a cumulative normal distribution and
    finding where the predicted T2M anomaly crosses the extreme event threshold.

    Parameters
    ----------
    data: 2D array, shape (time init, ens no.), contains the area-averaged forecasted T2M anomaly for a target period at each initialization.
    x: 1D array, contains the temperature range to fit the distribution to to 1dp.
    thresholds: 1D array, shape (timie init,), contains the area-averaged extreme event threshold for the period in question.

    Returns
    ----------
    prob_arr: 1D array, shape (time init,), contains the probability of the extreme cold event at a given initialization.
    """
    # Define the probability array.
    prob_arr = np.zeros(data.shape[0])
    # Loop through each init time, fit a normal distribution and find where cumulative function hits the threshold.
    for i in range(data.shape[0]):
        # Get the mean and standard deviation of each run ensemble spread.
        mean_member = np.nanmean(data[i, :])
        std_member = np.nanstd(data[i, :])
        # Fit normal distribution (cumulative) to data.
        norm_dist = stats.norm.cdf(x, loc =  mean_member, scale= std_member)
        # Find the point where your temp array for the distribition meets the extreme threshold for that run.
        cond_extreme = np.where(x == np.round(thresholds[i], 1))[0][0]
        # Use the condition above to find the cumulative probability of <=10% extreme threshold! Store.
        prob_arr[i] = norm_dist[cond_extreme]

    return prob_arr
