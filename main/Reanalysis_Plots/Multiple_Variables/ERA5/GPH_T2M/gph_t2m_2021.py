###### IMPORT MODULES ######
import sys
sys.path.insert(5, '../')
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset, num2date
import os
from datetime import datetime, timedelta
from mpl_toolkits.basemap import Basemap, addcyclic
from Functions import gen_utils, model_utils

# Get region descriptors to import data.
region = 'nhemi'
period = 'ndjfm'

###### Now read in the gph data. ######
directory = '/share/data1/Students/ollie/CAOs/Data/GPH/Non_Detrended'
path = os.chdir(directory)

# Choose level for analysis.
chosen_level = 500

# Choose start and end dates to analyze.
start_date1 = datetime(2021, 2, 1, 0, 0)
start_date2 = datetime(2021, 2, 18, 0, 0)

# Open the nc file
nc = Dataset(f'era5.hgt.500.{region}.{period}.all.daily.nc', 'r')
# Import time.
time_first = num2date(nc.variables['time'][:],nc.variables['time'].units,nc.variables['time'].calendar,only_use_cftime_datetimes=False, only_use_python_datetimes=True)
# Get time indices to restrict data to.
time_ind1 = np.where(time_first == start_date1)
time_ind2 = np.where(time_first == start_date2)
# Import geopotential height data and remove mask.
hgt = nc.variables['hgt'][time_ind1[0][0], time_ind1[1][0]:time_ind2[1][0]+1]
hgt = np.where(hgt.mask, np.nan, hgt.data)
# Get the restricted time array
time = time_first[time_ind1[0][0], time_ind1[1][0]:time_ind2[1][0]+1]
# Import latitude and longitude arrays.
latitude = nc.variables['latitude'][:]
longitude = nc.variables['longitude'][:]
nc.close()

###### Read in hgt ltm data. ######
# Choose directory where ltm data is stored.
directory = '/data/deluge/reanalysis/REANALYSIS/ERA5/3D/4xdaily/hgt'
path = os.chdir(directory)

# Read in 1981-2010 ltm hgt array.
# Open the file.
ltmFile = 'hgt.day.ltm.nc'
nc = Dataset(ltmFile, 'r')
# Find max and min latitude bounds from inital reanalysis file to restrict to the Northern Hemisphere later.
lat_1 = np.max(latitude) # Pole
lat_2 = np.min(latitude) # Equator
# Read in the latitude and level, longitude not required since it the same as the initial file and for the whole globe.
latitude_ltm = nc.variables['latitude'][:]
level_ltm = nc.variables['level'][:]
# Find latitude and level indices.
lat_index1 = np.where(latitude_ltm == lat_1)[0][0] # Pole
lat_index2 = np.where(latitude_ltm == lat_2)[0][0] # Equator
level_index = np.where(level_ltm == chosen_level)[0][0] # Choose pressure in hPa
# Import the datetimes from the ltm file.
date_ltm = num2date(nc.variables['time'][:], nc.variables['time'].units,nc.variables['time'].calendar,only_use_cftime_datetimes=False, only_use_python_datetimes=True)
# Get a month and day tracker for the ltm.
months_ltm = np.array([d.month for d in date_ltm])
days_ltm = np.array([d.day for d in date_ltm])
# Find the ltm time indices corresponding to the days and months for the start times.
ltm_time_ind1 = np.where((months_ltm == start_date1.month)&(days_ltm == start_date1.day))[0][0]
ltm_time_ind2 = np.where((months_ltm == start_date2.month)&(days_ltm == start_date2.day))[0][0]
# Now read in and restrict the geopotential height data to the correct times, level, and latitudes. Remove the mask too.
hgt_ltm = nc.variables['hgt'][ltm_time_ind1:ltm_time_ind2+1, level_index, lat_index1:lat_index2+1, :]
hgt_ltm = np.where(hgt_ltm.mask, np.nan, hgt_ltm.data)
nc.close()

###### Now read in the T2M data ######
# Go to the correct directory.
directory = '/share/data1/Students/ollie/CAOs/Data/Temps/T2M/Non_Detrended'
path = os.chdir(directory)
# Open the file.
nc = Dataset(f'era5.t2m.{region}.{period}.all.daily.nc', 'r')
# Read the times in.
time_first_t2m = num2date(nc.variables['time'][:],nc.variables['time'].units,nc.variables['time'].calendar,only_use_cftime_datetimes=False, only_use_python_datetimes=True)
# Restrict times to the start dates required.
t2m_time_ind1 = np.where(time_first_t2m == start_date1)
t2m_time_ind2 = np.where(time_first_t2m == start_date2)
# Read in the T2M data  with the correct time restriction and remove the mask.
t2m = nc.variables['t2m'][t2m_time_ind1[0][0], t2m_time_ind1[1][0]:t2m_time_ind2[1][0]+1]
t2m = np.where(t2m.mask, np.nan, t2m.data)
nc.close()

###### Now read in T2M ltm data ######
# Go to directory where ltm data is stored.
directory = '/data/deluge/reanalysis/REANALYSIS/ERA5/2D/4xdaily/t2m'
path = os.chdir(directory)

# Read in 1981-2010 ltm t2m array.
# Open the file.
ltmFile = 't2m.day.ltm.nc'
nc = Dataset(ltmFile, 'r')
# Get the datetimes corresponding to each day in the ltm.
date_ltm_t2m = num2date(nc.variables['time'][:], nc.variables['time'].units,nc.variables['time'].calendar,only_use_cftime_datetimes=False, only_use_python_datetimes=True)
# Create a month and day tracker for the ltm dates.
months_ltm_t2m = np.array([d.month for d in date_ltm_t2m])
days_ltm_t2m = np.array([d.day for d in date_ltm_t2m])
# Find the ltm time indices corresponding to the days and months for the start times.
t2m_ltm_time_ind1 = np.where((months_ltm_t2m == start_date1.month)&(days_ltm_t2m == start_date1.day))[0][0]
t2m_ltm_time_ind2 = np.where((months_ltm_t2m == start_date2.month)&(days_ltm_t2m == start_date2.day))[0][0]
# Read in the 2 m temperature data and remove mask, close the file.
t2m_ltm = nc.variables['t2m'][t2m_ltm_time_ind1:t2m_ltm_time_ind2+1, lat_index1:lat_index2+1, :]
t2m_ltm = np.where(t2m_ltm.mask, np.nan, t2m_ltm.data)
nc.close()

###### Anomalies and period separation ######
# Now find anomalies of both geopotential height and 2 m temperature.
t2m_anom = t2m-t2m_ltm
hgt_anom = (hgt-hgt_ltm)/10 # Divide by ten to get units of dam.
# Now define the three periods for each of the figure panels.
period1 = [datetime(2021, 2, 1), datetime(2021, 2, 4)]
period2 = [datetime(2021, 2, 5), datetime(2021, 2, 11)]
period3 = [datetime(2021, 2, 12), datetime(2021, 2, 18)]
# Stack the periods together.
periods_all = np.stack([period1, period2, period3])
# Fine the indices corresponding to the times for the start and end of each of the three averaging periods.
ind1_period1, ind2_period1 = np.where(time == period1[0])[0][0], np.where(time == period1[1])[0][0]
ind1_period2, ind2_period2 = np.where(time == period2[0])[0][0], np.where(time == period2[1])[0][0]
ind1_period3, ind2_period3 = np.where(time == period3[0])[0][0], np.where(time == period3[1])[0][0]
# Now split hgt and 2 m temperature data into the three periods and take the mean across time.
hgt_period1, t2m_period1 = np.nanmean(hgt_anom[ind1_period1:ind2_period1+1], axis = 0), np.nanmean(t2m_anom[ind1_period1:ind2_period1+1], axis = 0)
hgt_period2, t2m_period2 = np.nanmean(hgt_anom[ind1_period2:ind2_period2+1], axis = 0), np.nanmean(t2m_anom[ind1_period2:ind2_period2+1], axis = 0)
hgt_period3, t2m_period3 = np.nanmean(hgt_anom[ind1_period3:ind2_period3+1], axis = 0), np.nanmean(t2m_anom[ind1_period3:ind2_period3+1], axis = 0)
# Now stack the hgt and 2 m temperature arrays into a 3xlatxlon array.
hgt_all = np.stack([hgt_period1, hgt_period2, hgt_period3])
t2m_all = np.stack([t2m_period1, t2m_period2, t2m_period3])

###### PLOTTING ######
##### Set custom normalized colormap for t2m anomalies. #####
clevs = np.arange(-20, 21, 1)
my_cmap, norm = gen_utils.NormColorMap('RdBu_r', clevs)

# Now loop through each period and plot!
# Set rows and columns for subplotting.
nrows, ncols = 1, 3
# Set figure numbers for subplotting.
fig_no = np.arange(1, 4, 1)
# Set labels for each figure.
labels = ['a)', 'b)', 'c)']
# Set figure size and create figure variable.
fig = plt.figure(figsize = (15,5.5))
for i in range(3):
    # Shift the data to add a cyclic point.
    hgts_shifted, t2m_shifted, lons_shifted = addcyclic(hgt_all[i], t2m_all[i], longitude)
    # Meshgrid the latitude and longitudes.
    lons, lats = np.meshgrid(lons_shifted, latitude)
    # Add subplot at given position.
    ax = fig.add_subplot(nrows, ncols, fig_no[i])
    # Create map.
    map = gen_utils.NPStere_Map()
    # Create x and y coordinates.
    x,y = map(lons, lats)
    # Contourf the t2m anomalies.
    cs = map.contourf(x, y, t2m_shifted, clevs, norm=norm, extend='both', cmap = my_cmap)
    # Contour the geopotential height anomalies and label.
    lines2 = map.contour(x, y, hgts_shifted, levels = [-30, -25, -20, -15, -10,-5, 5, 10, 15, 20, 25, 30], colors = 'black')
    plt.clabel(lines2)
    # Draw Great Plains polygon.
    p1, p2, p3, p4 = gen_utils.DrawPolygon(map, lat_range = [48, 30], lon_range = [256.5, 268.5], grid_space = 0.5, lw = 2, color = 'purple')
    # Set figure title.
    ax.set_title(f"{labels[i]} {periods_all[i][0].strftime('%Y-%m-%d')} - {periods_all[i][1].strftime('%Y-%m-%d')}", fontsize = 14, weight = 'bold')
    plt.tight_layout()

# Add colorbar.
cb_ax = fig.add_axes([0.05, -0.02, 0.91, 0.04])
cbar = fig.colorbar(cs, cax=cb_ax,orientation="horizontal",ticks= np.arange(-20, 25, 5),extend="both")
cbar.set_label("T2M Anomaly ($^\circ$C)", fontsize = 16)
cbar.ax.tick_params(labelsize=13)
# Save figure.
plt.savefig("/share/data1/Students/ollie/CAOs/project-2021-cao/main/Reanalysis_Plots/Multiple_Variables/ERA5/GPH_T2M/gph_t2m_2021.png", bbox_inches = 'tight', dpi = 500)
