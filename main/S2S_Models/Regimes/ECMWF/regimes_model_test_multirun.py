end_date###### Import modules ######
import numpy as np
from netCDF4 import Dataset, num2date
import os
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import glob
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter, MonthLocator
import pickle
dir = '/share/data1/Students/ollie/CAOs/project-2021-cao/Functions'
path = os.chdir(dir)
from model_utils import load_latlon, load_modelparam
from gen_utils import NormColorMap, NPStere_Map, DrawPolygon

# Select lat, lon, level bounds.
lat1, lat2 = 79.5, 19.5
lon1, lon2 = 180, 330
level_select = 500

###### Get the ltm data for 500 hPa GPH. ######
# Go to directory for data.
dir = '/share/data1/Students/ollie/CAOs/Data/Feb_2021_CAO/Model_Data/ECMWF/hgt'
path = os.chdir(dir)
# Open the file.
nc = Dataset('climo_ECMWF_hgt_hindcast.nc', 'r')
# Read in lat, lon, levels.
ltm_latitude, ltm_longitude = load_latlon(nc)
ltm_levels = nc.variables['level'][:]
# Load model hindcast dates, days array and the ltm hgt array.
ltm_dates = num2date(nc.variables['hdates'][:], nc.variables['hdates'].units, calendar = 'gregorian', only_use_cftime_datetimes=False, only_use_python_datetimes=True)
ltm_days = nc.variables['days'][:]
ltm_hgt = nc.variables['hgt'][:]
# Close the file
nc.close()

# Find the lat, lon, and level indices according to the bounds.
ltm_lat_ind1, ltm_lat_ind2 = np.where(ltm_latitude == lat1)[0][0], np.where(ltm_latitude == lat2)[0][0]
ltm_lon_ind1, ltm_lon_ind2 = np.where(ltm_longitude == lon1)[0][0], np.where(ltm_longitude == lon2)[0][0]
level_ind = np.where(ltm_levels == level_select)[0][0]
# Restrict the ltm GPH to the region bounds you want.
hgt_ltm_region = ltm_hgt[:, :, level_ind, ltm_lat_ind1:ltm_lat_ind2+1, ltm_lon_ind1:ltm_lon_ind2+1]

###### Read in the realtime model data. ######
# Choose model.
model = 'ECMWF'

# Go to directory for ensemble of hgt data.
dir = f'/data/deluge/models/S2S/realtime/{model}/hgt'
path = os.chdir(dir)

# Open a data file to get the parameters.
filename =f'{model}_hgt_2021-01-04_perturbed.nc'
nc = Dataset(filename, 'r')
# Load the latitude, longitude, level, time, and ensemble number test data for later.
mod_latitude, mod_longitude = load_latlon(nc)
level = nc.variables['level'][:]
time, number = load_modelparam(nc)[0], load_modelparam(nc)[1]
# Close the nc file.
nc.close()

# Set model lat and lon indices according to bounds.
mod_lat_ind1, mod_lat_ind2 = np.where(mod_latitude == lat1)[0][0], np.where(mod_latitude == lat2)[0][0]
mod_lon_ind1, mod_lon_ind2 = np.where(mod_longitude == lon1)[0][0], np.where(mod_longitude == lon2)[0][0]
# Restrict the model latitudes and longitudes to the indices above.
mod_region_lat = mod_latitude[mod_lat_ind1:mod_lat_ind2+1]
mod_region_lon = mod_longitude[mod_lon_ind1:mod_lon_ind2+1]

# Set initial start date and end date strings for model initializations.
date_init = '2021-01-04'
date_end = '2021-02-04'

# Set start and end date to average over.
start_date = datetime(2021, 2, 8, 0, 0)
end_date = datetime(2021, 2, 10, 0, 0)

# Do perturbation indexing.
# Go to directory and get all files ending with "perturbed.nc".
model_dir_pert = f'/data/deluge/models/S2S/realtime/{model}/hgt/*perturbed.nc'
files_pert = glob.glob(model_dir_pert)
# Sort the files.
files_pert.sort()
# Get the date string in YYYY-MM-DD format.
model_dates_pert = [i.split("_")[2] for i in files_pert]
# Find where our chosen dates are in this data.
pert_ind1 = model_dates_pert.index(date_init)
pert_ind2 = model_dates_pert.index(date_end)

# Do control indexing.
# Go to directory and get all files ending with "control.nc".
model_dir_con = f'/data/deluge/models/S2S/realtime/{model}/hgt/*control.nc'
files_con = glob.glob(model_dir_con)
# Sort the files.
files_con.sort()
# Get the date string in YYYY-MM-DD format.
model_dates_con = [i.split("_")[2] for i in files_con]
# Find where our chosen dates are in this data.
con_ind1 = model_dates_con.index(date_init)
con_ind2 = model_dates_con.index(date_end)

# Now get the perturbed and control arrays to put data into using np.zeros. Perturbed will be shaped INIT NO. x time x number x lat x lon.
# Control will be shaped INIT NO. x time x lat x lon.
perturbed_hgt = np.zeros((pert_ind2-pert_ind1+1, len(ltm_days), len(number), len(mod_region_lat), len(mod_region_lon)))
control_hgt = np.zeros((con_ind2-con_ind1+1, len(ltm_days), len(mod_region_lat), len(mod_region_lon)))
# Get associated time and init time lists to append to.
time_arr = []
time_init = []

# Now loop through files and store the data! First for perturbed.
# Loop through the perturbation indexes.
for i in range(pert_ind1, pert_ind2+1):
    # Get filename.
    pert_filename = files_pert[i]
    # Open file.
    nc = Dataset(pert_filename, 'r')
    # Get the time.
    time_pert = num2date(nc.variables['time'][:], nc.variables['time'].units, calendar = 'gregorian', only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    # Read in the data.
    perturbed_hgt[i-pert_ind1, :, :, :, :] = nc.variables['gh'][:, :, np.where(level == level_select)[0][0], mod_lat_ind1:mod_lat_ind2+1, mod_lon_ind1:mod_lon_ind2+1]
    # Close file.
    nc.close()
    # Append time arrays and init times to the lists.
    time_arr.append(time_pert)
    time_init.append(time_pert[0])

# Now for control.
for i in range(con_ind1, con_ind2+1):
    # Get filename.
    con_filename = files_con[i]
    # Open file.
    nc = Dataset(con_filename, 'r')
    # Read in the data.
    control_hgt[i-con_ind1, :, :, :] = nc.variables['gh'][:, np.where(level == level_select)[0][0], mod_lat_ind1:mod_lat_ind2+1, mod_lon_ind1:mod_lon_ind2+1]
    # Close the file.
    nc.close()

# Make time_init an array.
time_init = np.asarray(time_init)
time_arr = np.stack(time_arr)

# Now join all members with control for the whole ensemble suite.
all_members_hgt = np.concatenate((control_hgt[:, :, None, :, :], perturbed_hgt), axis = 2)

###### Find anomalies. ######
# Set a month and day locator for ltm dates.
months_ltm = np.array([d.month for d in ltm_dates])
days_ltm = np.array([d.day for d in ltm_dates])
# Create an empty array to store the hgt anomalies.
hgt_anom = np.zeros((all_members_hgt.shape))
# Loop through each initialization start time and take away the climo where the month/day trackers equal the init time month/day.
for i in range(len(time_init)):
    hgt_anom[i] = (all_members_hgt[i] - hgt_ltm_region[np.where((months_ltm == time_init[i].month)&(days_ltm == time_init[i].day))[0][0], :, None, :, :])

###### Now lead time average the anomalies between the selected dates! ######
# Define the empty array for the lead time averaged data.
all_members_final = np.zeros((hgt_anom.shape[0], hgt_anom.shape[2], hgt_anom.shape[3], hgt_anom.shape[4]))
# Loop through each initialization and average between the start date and end date.
for i in range(len(time_init)):
    all_members_final[i] = np.nanmean(hgt_anom[i, np.where(time_arr[i]==start_date)[0][0]:np.where(time_arr[i]==end_date)[0][0]+1], axis = 0)

###### Regime projections ######

# Now read in k-means and EOF objects from Millin et al. (2022).
dir = '/share/data1/Students/ollie/CAOs/Data/Regimes/Cluster_Info_S2S'
path = os.chdir(dir)
kmeans = pickle.load(open("kmeans_object.pkl","rb"))
solver = pickle.load(open("eof_object.pkl","rb"))

## Get the PC loadings for each ensemble member and time
n_eof = 12 ## the number of EOFs to return, cannot exceed the number previously computed
# Define array of shape init no. x ens number x n_eof for the PCs.
pc_ens = np.zeros((all_members_final.shape[0],all_members_final.shape[1],n_eof))
# Loop through each initialization and project the EOF onto the hgt array.
for i in range(all_members_final.shape[0]):
	## project onto EOF & obtain psuedo PC
	## this automatically uses the same weights as when the EOF was constructed
	pc_ens[i,:,:] = solver.projectField(all_members_final[i,:,:,:],neofs=n_eof)

## now attribute each to a cluster based on the PC loadings
# Define array of shape init no. x ens member to store clustering numbers.
clusters = np.zeros((pc_ens.shape[0],pc_ens.shape[1]))
# Loop through each init and essentially use k-means to assign the cluster to each ensemble member.
for i in range(clusters.shape[0]):
	clusters[i,:] = kmeans.predict(pc_ens[i,:])


###### Make a plot of a certain regime number. ######
# This section is just to visually inspect the outcomes of the clustering to assign the correct numbers to each regime visually!
# Can be commented out otherwise.
'''# Select cluster number.
select_no = 4
# Find run init index.
run_ind = np.where(time_init == datetime(2021, 1, 18))[0][0]
# Isolate clusters and hgt pattern by the run in question.
select_clust = clusters[run_ind]
hgt_select = all_members_final[run_ind]
# Find where the selected run clusters equal the selected cluster number.
cluster_ind = np.where(select_clust == select_no)[0]

# Now composite the pattern to check this is correct!
comp_select = np.nanmean(hgt_select[cluster_ind], axis = 0)

# Plot the regimes.
# Make the colormap.
clevs = np.arange(-30, 32, 2)
my_cmap, norm = NormColorMap('RdBu_r', clevs)
# Plot the figure.
fig = plt.figure(figsize=(6, 6))
m = Basemap(width=14000000,height=8000000,resolution='l',projection='laea',lat_ts=60,lat_0=55,lon_0=-105.)
lonsmesh, latsmesh = np.meshgrid(mod_region_lon, mod_region_lat)
x,y = m(lonsmesh,latsmesh)
cf = m.contourf(x,y,comp_select/10,cmap=my_cmap,extend='both',levels=clevs)
m.drawcoastlines(linewidth=0.5)
plt.title(f"TEST for {select_no}")
plt.show(block = False)'''

###### Count each regime for each ensemble member ######
# From the above analysis, the regime numbers are:
# 0 = PT
# 1 = WCR
# 2 = AkR
# 3 = ArH
# 4 = ArL

# Go through each init and count how many PT, WCR, AkR, ArH, and ArL.
# Change the array to integers to use bincount.
clusters = clusters.astype(int)
# Define the empty arrays to hold the count of each regime.
pt_count = np.zeros(len(time_init))
wcr_count = np.zeros(len(time_init))
akr_count = np.zeros(len(time_init))
arh_count = np.zeros(len(time_init))
arl_count = np.zeros(len(time_init))
# Loop through each init and count how many are in each regime type.
for i in range(len(time_init)):
	pt_count[i] = np.count_nonzero(clusters[i] == 0)
	wcr_count[i] = np.count_nonzero(clusters[i] == 1)
	akr_count[i] = np.count_nonzero(clusters[i] == 2)
	arh_count[i] = np.count_nonzero(clusters[i] == 3)
	arl_count[i] = np.count_nonzero(clusters[i] == 4)


# Create color list for each regime.
colors = ['darkgreen', 'darkblue', 'darkorange', 'darkred', 'darkpurple']
# Get labels for plot in correct form "M-DD".
labels = [f'{i.month}/{i.day}' for i in time_init]
# Make the plot!
fig, ax = plt.subplots(figsize = (8,3))
ax.bar(labels, pt_count, color = 'darkgreen', label = 'PT', edgecolor = 'black')
ax.bar(labels, wcr_count, label = 'WCR', color = 'darkblue', bottom = pt_count, edgecolor = 'black')
ax.bar(labels, akr_count, label = 'AkR', color = 'orange', bottom = pt_count+wcr_count, edgecolor = 'black')
ax.bar(labels, arh_count, label = 'ArH', color = 'darkred', bottom = pt_count+wcr_count+akr_count, edgecolor = 'black')
ax.bar(labels, arl_count, label = 'ArL', color = 'purple', bottom = pt_count+wcr_count+akr_count+arh_count, edgecolor = 'black')
plt.xlabel('Initialization Date (Month-Day)', fontsize = 10, weight = 'bold')
plt.xticks(labels, rotation = 30)
plt.ylabel('Ensemble Member Count', fontsize = 10, weight = 'bold')
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.legend(loc = 'lower left', ncol = 5)
plt.title(f'b) ECMWF Regimes {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}')
plt.savefig("/share/data1/Students/ollie/CAOs/project-2021-cao/main/S2S_Models/Regimes/ECMWF/bar_init.png", bbox_inches = 'tight', dpi = 500)
