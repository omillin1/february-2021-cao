import numpy as np
from netCDF4 import Dataset, num2date
import os
from datetime import datetime
from mpl_toolkits.basemap import Basemap, addcyclic
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import glob
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter, DayLocator
from scipy import stats
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import re

dir = '/share/data1/Students/ollie/CAOs/project-2021-cao/Functions'
path = os.chdir(dir)
from model_utils import Extreme_T2M_Thresholds_HC

# Get the ltm data for ECMWF.
dir = '/share/data1/Students/ollie/CAOs/Data/Feb_2021_CAO/Model_Data/ECMWF/surfT'
path = os.chdir(dir)

nc = Dataset('climo_ECMWF_surfT_hindcast.nc', 'r')
ltm_latitude = nc.variables['latitude'][:]
ltm_longitude = nc.variables['longitude'][:]
ltm_dates_ecmwf = num2date(nc.variables['hdates'][:], nc.variables['hdates'].units, calendar = 'gregorian', only_use_cftime_datetimes=False, only_use_python_datetimes=True)
ltm_days_ecmwf = nc.variables['days'][:]
ltm_t2m_ecmwf = nc.variables['surfT'][:]
nc.close()

# Get the ltm data for NCEP.
dir = '/share/data1/Students/ollie/CAOs/Data/Feb_2021_CAO/Model_Data/NCEP/surfT'
path = os.chdir(dir)

nc = Dataset('climo_NCEP_surfT_hindcast.nc', 'r')
ltm_dates_ncep = num2date(nc.variables['hdates'][:], nc.variables['hdates'].units, calendar = 'gregorian', only_use_cftime_datetimes=False, only_use_python_datetimes=True)
ltm_days_ncep = nc.variables['days'][1:] # Omit first day because surfT NCEP data does not include init day like ECMWF.
ltm_t2m_ncep = nc.variables['surfT'][:, 1:] # Omit first day because surfT NCEP data does not include init day like ECMWF.
nc.close()

# Select lat/lon for the Great Plains
lat1, lat2 = 48, 30
lon1, lon2 = 256.5, 268.5


# Now get latitude indices for ECMWF and NCEP.
ltm_lat_ind1, ltm_lat_ind2 = np.where(ltm_latitude == lat1)[0][0], np.where(ltm_latitude == lat2)[0][0]
ltm_lon_ind1, ltm_lon_ind2 = np.where(ltm_longitude == lon1)[0][0], np.where(ltm_longitude == lon2)[0][0]

# Restrict the ltm in ECMWF and NCEP to the Great Plains region.
t2m_ltm_region_ecmwf = ltm_t2m_ecmwf[:, :, ltm_lat_ind1:ltm_lat_ind2+1, ltm_lon_ind1:ltm_lon_ind2+1]
t2m_ltm_region_ncep = ltm_t2m_ncep[:, :, ltm_lat_ind1:ltm_lat_ind2+1, ltm_lon_ind1:ltm_lon_ind2+1]


# Go to directory for ensemble of ECMWF T2M data.
dir = '/data/deluge/scratch/S2S/realtime/ECMWF/surfT'
path = os.chdir(dir)

# First get ECMWF model parameters via opening model test data.
# Open data.
filename = 'ECMWF_surfT_2021-01-04_perturbed.nc'
nc = Dataset(filename, 'r')
time_ecmwf = num2date(nc.variables['time'][:], nc.variables['time'].units, calendar = 'gregorian', only_use_cftime_datetimes=False, only_use_python_datetimes=True)
number_ecmwf = nc.variables['number'][:]
mod_latitude = nc.variables['latitude'][:]
mod_longitude = nc.variables['longitude'][:]
nc.close()

# Set model lat and lon.
mod_lat_ind1, mod_lat_ind2 = np.where(mod_latitude == lat1)[0][0], np.where(mod_latitude == lat2)[0][0]
mod_lon_ind1, mod_lon_ind2 = np.where(mod_longitude == lon1)[0][0], np.where(mod_longitude == lon2)[0][0]

# Restrict model lat and lon.
mod_region_lat = mod_latitude[mod_lat_ind1:mod_lat_ind2+1]
mod_region_lon = mod_longitude[mod_lon_ind1:mod_lon_ind2+1]

# Set initial start date and end date strings for model initializations.
date_init = '2021-01-14'
date_end = '2021-02-11'

# Set start and end date to average over.
cold_start = datetime(2021, 2, 12, 0, 0)
cold_end = datetime(2021, 2, 18, 0, 0)

# Do perturbation indexing for ECMWF.
model_dir_pert = '/data/deluge/scratch/S2S/realtime/ECMWF/surfT/*perturbed.nc'
files_pert = glob.glob(model_dir_pert)
files_pert.sort()
model_dates_pert = [i.split("_")[2] for i in files_pert]
pert_ind1 = model_dates_pert.index(date_init)
pert_ind2 = model_dates_pert.index(date_end)

# Do control indexing for ECMWF.
model_dir_con = '/data/deluge/scratch/S2S/realtime/ECMWF/surfT/*control.nc'
files_con = glob.glob(model_dir_con)
files_con.sort()
model_dates_con = [i.split("_")[2] for i in files_con]
con_ind1 = model_dates_con.index(date_init)
con_ind2 = model_dates_con.index(date_end)

# Now get the ECMWF perturbed and control arrays to put data into. Perturbed will be shaped INIT NO. x time x number x lat x lon.
# Control will be shaped INIT NO. x time x lat x lon.
perturbed_t2m_ecmwf = np.zeros((pert_ind2-pert_ind1+1, len(ltm_days_ecmwf), len(number_ecmwf), len(mod_region_lat), len(mod_region_lon)))
control_t2m_ecmwf = np.zeros((con_ind2-con_ind1+1, len(ltm_days_ecmwf), len(mod_region_lat), len(mod_region_lon)))
time_arr_ecmwf = []
time_init_ecmwf = []

# Now loop through files and store the data! First for perturbed.
for i in range(pert_ind1, pert_ind2+1):
    pert_filename = files_pert[i]
    nc = Dataset(pert_filename, 'r')
    time_pert_ecmwf = num2date(nc.variables['time'][:], nc.variables['time'].units, calendar = 'gregorian', only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    perturbed_t2m_ecmwf[i-pert_ind1, :, :, :, :] = nc.variables['t2m'][:, :, mod_lat_ind1:mod_lat_ind2+1, mod_lon_ind1:mod_lon_ind2+1]
    nc.close()
    time_arr_ecmwf.append(time_pert_ecmwf)
    time_init_ecmwf.append(time_pert_ecmwf[0])

# Now for control.
for i in range(con_ind1, con_ind2+1):
    con_filename = files_con[i]
    nc = Dataset(con_filename, 'r')
    time_con_ecmwf = num2date(nc.variables['time'][:], nc.variables['time'].units, calendar = 'gregorian', only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    control_t2m_ecmwf[i-con_ind1, :, :, :] = nc.variables['t2m'][:, mod_lat_ind1:mod_lat_ind2+1, mod_lon_ind1:mod_lon_ind2+1]
    nc.close()

# Go to directory for ensemble of NCEP T2M data.
dir = '/data/deluge/scratch/S2S/realtime/NCEP/surfT'
path = os.chdir(dir)

# Now get NCEP model parameters via opening model test data.
# Open data.
filename = 'NCEP_surfT_2021-01-04_perturbed.nc'
nc = Dataset(filename, 'r')
time_ncep = num2date(nc.variables['time'][:], nc.variables['time'].units, calendar = 'gregorian', only_use_cftime_datetimes=False, only_use_python_datetimes=True)
number_ncep = nc.variables['number'][:]
nc.close()

# Now get the NCEP perturbed and control arrays to put data into. Perturbed will be shaped INIT NO. x time x number x lat x lon.
# Control will be shaped INIT NO. x time x lat x lon.
perturbed_t2m_ncep = np.zeros((pert_ind2-pert_ind1+1, len(ltm_days_ncep), len(number_ncep), len(mod_region_lat), len(mod_region_lon)))
control_t2m_ncep = np.zeros((con_ind2-con_ind1+1, len(ltm_days_ncep), len(mod_region_lat), len(mod_region_lon)))
time_arr_ncep = []
time_init_ncep = []

# Now select init dates for NCEP.
files_pert_select = model_dates_pert[pert_ind1:pert_ind2+1]
files_con_select = model_dates_con[con_ind1:con_ind2+1]

# Now loop through files and store the data! First for perturbed.
for i in range(len(files_pert_select)):
    pert_filename = f'NCEP_surfT_{files_pert_select[i]}_perturbed.nc'
    nc = Dataset(pert_filename, 'r')
    time_pert_ncep = num2date(nc.variables['time'][:], nc.variables['time'].units, calendar = 'gregorian', only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    perturbed_t2m_ncep[i, :, :, :, :] = nc.variables['t2m'][:, :, mod_lat_ind1:mod_lat_ind2+1, mod_lon_ind1:mod_lon_ind2+1]
    nc.close()
    time_arr_ncep.append(time_pert_ncep)
    time_init_ncep.append(time_pert_ncep[0]-timedelta(days = 1))

# Now for control.
for i in range(len(files_con_select)):
    con_filename = f'NCEP_surfT_{files_con_select[i]}_control.nc'
    nc = Dataset(con_filename, 'r')
    time_con_ncep = num2date(nc.variables['time'][:], nc.variables['time'].units, calendar = 'gregorian', only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    control_t2m_ncep[i, :, :, :] = nc.variables['t2m'][:, mod_lat_ind1:mod_lat_ind2+1, mod_lon_ind1:mod_lon_ind2+1]
    nc.close()

# Make time_init lists into arrays.
time_init_ncep, time_init_ecmwf = np.asarray(time_init_ncep), np.asarray(time_init_ecmwf)
time_arr_ncep, time_arr_ecmwf = np.stack(time_arr_ncep), np.stack(time_arr_ecmwf)

# Now join all members with control.
all_members_t2m_ncep = np.concatenate((control_t2m_ncep[:, :, None, :, :], perturbed_t2m_ncep), axis = 2)
all_members_t2m_ecmwf = np.concatenate((control_t2m_ecmwf[:, :, None, :, :], perturbed_t2m_ecmwf), axis = 2)

# Now go through each of the hindcast dates that you have chosen and take away the corresponding climo.
# Set a month and day locator for this.
months_ltm_ncep = np.array([d.month for d in ltm_dates_ncep])
days_ltm_ncep = np.array([d.day for d in ltm_dates_ncep])

months_ltm_ecmwf = np.array([d.month for d in ltm_dates_ecmwf])
days_ltm_ecmwf = np.array([d.day for d in ltm_dates_ecmwf])

t2m_anom_ncep = np.zeros((all_members_t2m_ncep.shape))
t2m_anom_ecmwf = np.zeros((all_members_t2m_ecmwf.shape))

# Get anomalies for ECMWF.
for i in range(len(time_init_ecmwf)):
    t2m_anom_ecmwf[i] = (all_members_t2m_ecmwf[i] - t2m_ltm_region_ecmwf[np.where((months_ltm_ecmwf == time_init_ecmwf[i].month)&(days_ltm_ecmwf == time_init_ecmwf[i].day))[0][0], :, None, :, :])

# Get anomalies for NCEP, use time init of ECMWF since these show the dates we want to investigate.
for i in range(len(time_init_ecmwf)):
    t2m_anom_ncep[i] = (all_members_t2m_ncep[i] - t2m_ltm_region_ncep[np.where((months_ltm_ncep == time_init_ecmwf[i].month)&(days_ltm_ncep == time_init_ecmwf[i].day))[0][0], :, None, :, :])


# Now find extreme event thresholds.
# Begin by getting a list of initializations in 'MM-DD' form.
init_dates = [re.split("[-_.]", i)[-2]+'-'+re.split("[-_.]", i)[-1] for i in files_pert_select]

# Get a (lead, lat, lon) array of zeros for both ECMWF and NCEP.
threshold_region_ecmwf = np.zeros((len(init_dates),len(time_ecmwf), len(mod_region_lat), len(mod_region_lon)))
threshold_region_ncep = np.zeros((len(init_dates),len(time_ncep)+1, len(mod_region_lat), len(mod_region_lon)))
for i in range(len(init_dates)):
    threshold_region_ecmwf[i] = Extreme_T2M_Thresholds_HC('ECMWF', init_dates[i], [lat1, lat2], [lon1, lon2], perc = 10)[0]
    threshold_region_ncep[i] = Extreme_T2M_Thresholds_HC('NCEP', init_dates[i], [lat1, lat2], [lon1, lon2], perc = 10)[0]

# Now adjust the ncep thresholds to get rid of the first day as in real-time.
threshold_region_ncep = threshold_region_ncep[:, 1:]

# Area average thresholds.
weights = np.cos(np.radians(mod_region_lat))
extreme_thresh_lat_ecmwf, extreme_thresh_lat_ncep = np.average(threshold_region_ecmwf, weights = weights, axis = 2), np.average(threshold_region_ncep, weights = weights, axis = 2)
extreme_thresh_ecmwf, extreme_thresh_ncep = np.nanmean(extreme_thresh_lat_ecmwf, axis = -1),np.nanmean(extreme_thresh_lat_ncep, axis = -1)

# Now we want to lead-time average the t2m anoms for 12-18 Feb 2021.
all_members_ecmwf = np.zeros((t2m_anom_ecmwf.shape[0], t2m_anom_ecmwf.shape[2], t2m_anom_ecmwf.shape[3], t2m_anom_ecmwf.shape[4]))
extreme_thresh_all_ecmwf = np.zeros(len(time_init_ecmwf))
all_members_ncep = np.zeros((t2m_anom_ncep.shape[0], t2m_anom_ncep.shape[2], t2m_anom_ncep.shape[3], t2m_anom_ncep.shape[4]))
extreme_thresh_all_ncep = np.zeros(len(time_init_ncep))
for i in range(len(time_init_ecmwf)):
    all_members_ecmwf[i] = np.nanmean(t2m_anom_ecmwf[i, np.where(time_arr_ecmwf[i]==cold_start)[0][0]:np.where(time_arr_ecmwf[i]==cold_end)[0][0]+1], axis = 0)
    extreme_thresh_all_ecmwf[i] = np.nanmean(extreme_thresh_ecmwf[i, np.where(time_arr_ecmwf[i]==cold_start)[0][0]:np.where(time_arr_ecmwf[i]==cold_end)[0][0]+1])
    all_members_ncep[i] = np.nanmean(t2m_anom_ncep[i, np.where(time_arr_ncep[i]==cold_start)[0][0]:np.where(time_arr_ncep[i]==cold_end)[0][0]+1], axis = 0)
    extreme_thresh_all_ncep[i] = np.nanmean(extreme_thresh_ncep[i, np.where(time_arr_ncep[i]==cold_start)[0][0]:np.where(time_arr_ncep[i]==cold_end)[0][0]+1])

# Now area-average the anomalies.
all_members_lat_ecmwf, all_members_lat_ncep = np.average(all_members_ecmwf, weights = weights, axis = 2), np.average(all_members_ncep, weights = weights, axis = 2)
all_members_final_ecmwf, all_members_final_ncep = np.nanmean(all_members_lat_ecmwf, axis = -1), np.nanmean(all_members_lat_ncep, axis = -1)

# Go through each init and find the prob of extreme event.
# Set temp array to find probs.
dist_temps = np.round(np.arange(-20, 20.1, 0.1),3)
prob_arr_ncep = np.zeros(len(time_init_ncep))
prob_arr_ecmwf = np.zeros(len(time_init_ecmwf))

# ECMWF
for i in range(len(time_init_ecmwf)):
    mean_member_ecmwf = np.nanmean(all_members_final_ecmwf[i, :])
    std_member_ecmwf = np.nanstd(all_members_final_ecmwf[i, :])
    # Fit normal dist.
    norm_dist_ecmwf = stats.norm.cdf(dist_temps, loc =  mean_member_ecmwf, scale= std_member_ecmwf)
    norm_pdf_ecmwf = stats.norm.pdf(dist_temps, loc =  mean_member_ecmwf, scale= std_member_ecmwf)
    # Set conditions to get correct probabilities.
    cond_extreme_ecmwf = np.where(dist_temps == np.round(extreme_thresh_all_ecmwf[i], 1))[0][0]
    prob_arr_ecmwf[i] = norm_dist_ecmwf[cond_extreme_ecmwf]

# NCEP
for i in range(len(time_init_ncep)):
    mean_member_ncep = np.nanmean(all_members_final_ncep[i, :])
    std_member_ncep = np.nanstd(all_members_final_ncep[i, :])
    # Fit normal dist.
    norm_dist_ncep = stats.norm.cdf(dist_temps, loc =  mean_member_ncep, scale= std_member_ncep)
    norm_pdf_ncep = stats.norm.pdf(dist_temps, loc =  mean_member_ncep, scale= std_member_ncep)
    # Set conditions to get correct probabilities.
    cond_extreme_ncep = np.where(dist_temps == np.round(extreme_thresh_all_ncep[i], 1))[0][0]
    prob_arr_ncep[i] = norm_dist_ncep[cond_extreme_ncep]

date_ticks = [f'{i.month}/{i.day}' for i in time_init_ecmwf]

fig, ax = plt.subplots(figsize = (8, 3))
ax.plot(np.arange(1, len(date_ticks)+1), prob_arr_ecmwf*100, lw = 2, color = 'darkblue', marker = 'o', markersize = 5, label = 'ECMWF')
ax.plot(np.arange(1, len(date_ticks)+1), prob_arr_ncep*100, lw = 2, color = 'darkred', marker = 'o', markersize = 5, label = 'NCEP')
plt.title(f'a) Extreme Event Probability')
plt.xticks(np.arange(1, len(date_ticks)+1), date_ticks, rotation = 30)
plt.xlabel('Initialization Date (M/DD)', fontsize = 10, weight = 'bold')
plt.ylabel('Probability of Extreme Event (%)', fontsize = 9, weight = 'bold')
plt.yticks(np.arange(0, 120, 20))
ax.yaxis.set_minor_locator(MultipleLocator(5))
plt.ylim([0, 105])
plt.xlim([1, len(time_init_ecmwf)])
#plt.title('Extreme Event Probability Great Plains 2/12/21 - 2/18/21')
plt.legend()
plt.savefig("/share/data1/Students/ollie/CAOs/project-2021-cao/main/S2S_Models/Prob_Extreme_Event/ECMWF_NCEP/extreme_event_prob.png", bbox_inches = 'tight', dpi = 500)
