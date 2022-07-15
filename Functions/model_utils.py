import numpy as np
from netCDF4 import Dataset, num2date
import os
from datetime import datetime
from glob import glob
import re

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
        pert_data = np.where(pert_data.mask, np.nan, pert_data.data)
        nc_pert.close()
        nc_con = Dataset(con_files[f], 'r')
        con_data = nc_con.variables['t2m'][:, lat_ind1:lat_ind2+1, lon_ind1:lon_ind2+1]
        con_data = np.where(con_data.mask, np.nan, con_data.data)
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
