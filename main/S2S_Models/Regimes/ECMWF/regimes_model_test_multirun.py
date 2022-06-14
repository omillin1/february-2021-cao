import numpy as np
from netCDF4 import Dataset, num2date
import os
from datetime import datetime
from mpl_toolkits.basemap import Basemap, addcyclic
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import glob
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter, MonthLocator
from scipy.stats import pearsonr
dir = '/share/data1/Students/ollie/CAOs/project-2021-cao/Functions'
path = os.chdir(dir)
from model_utils import load_latlon, load_modelparam
from gen_utils import NormColorMap, NPStere_Map, DrawPolygon
import pickle

# Select lat/lon you want.
lat1, lat2 = 79.5, 19.5
lon1, lon2 = 180, 330
level_select = 500

# Get the ltm data.
dir = '/share/data1/Students/ollie/CAOs/Data/Feb_2021_CAO/Model_Data/ECMWF/hgt'
path = os.chdir(dir)

nc = Dataset('climo_ECMWF_hgt_hindcast.nc', 'r')
ltm_latitude = nc.variables['latitude'][:]
ltm_longitude = nc.variables['longitude'][:]
ltm_levels = nc.variables['level'][:]
ltm_dates = num2date(nc.variables['hdates'][:], nc.variables['hdates'].units, calendar = 'gregorian', only_use_cftime_datetimes=False, only_use_python_datetimes=True)
ltm_days = nc.variables['days'][:]
ltm_hgt = nc.variables['hgt'][:]
nc.close()

ltm_lat_ind1, ltm_lat_ind2 = np.where(ltm_latitude == lat1)[0][0], np.where(ltm_latitude == lat2)[0][0]
ltm_lon_ind1, ltm_lon_ind2 = np.where(ltm_longitude == lon1)[0][0], np.where(ltm_longitude == lon2)[0][0]
level_ind = np.where(ltm_levels == level_select)[0][0]
hgt_ltm_region = ltm_hgt[:, :, level_ind, ltm_lat_ind1:ltm_lat_ind2+1, ltm_lon_ind1:ltm_lon_ind2+1]

# Choose model.

model = 'ECMWF'

# Go to directory for ensemble of hgt data.
dir = '/data/deluge/scratch/S2S/realtime/'+model+'/hgt'
path = os.chdir(dir)

# Go through different initializations and get the hgt array for each dataset. Data is time, number, level, lat, lon.
# First get model parameters via opening model test data.
# Open data.
filename = model +'_hgt_2021-01-04_perturbed.nc'
nc = Dataset(filename, 'r')
time = num2date(nc.variables['time'][:], nc.variables['time'].units, calendar = 'gregorian', only_use_cftime_datetimes=False, only_use_python_datetimes=True)
number = nc.variables['number'][:]
level = nc.variables['level'][:]
mod_latitude = nc.variables['latitude'][:]
mod_longitude = nc.variables['longitude'][:]
nc.close()

# Set model lat and lon.
mod_lat_ind1, mod_lat_ind2 = np.where(mod_latitude == lat1)[0][0], np.where(mod_latitude == lat2)[0][0]
mod_lon_ind1, mod_lon_ind2 = np.where(mod_longitude == lon1)[0][0], np.where(mod_longitude == lon2)[0][0]

mod_region_lat = mod_latitude[mod_lat_ind1:mod_lat_ind2+1]
mod_region_lon = mod_longitude[mod_lon_ind1:mod_lon_ind2+1]

# Set initial start date and end date strings for model initializations.
date_init = '2021-01-04'
date_end = '2021-02-04'

# Set start and end date to average over.
length = 3
cold_start = datetime(2021, 2, 8, 0, 0)
cold_end = cold_start+timedelta(days = length-1)

# Do perturbation indexing.
model_dir_pert = '/data/deluge/scratch/S2S/realtime/'+model+'/hgt/*perturbed.nc'
files_pert = glob.glob(model_dir_pert)
files_pert.sort()
model_dates_pert = [i.split("_")[2] for i in files_pert]
pert_ind1 = model_dates_pert.index(date_init)
pert_ind2 = model_dates_pert.index(date_end)

# Do control indexing.
model_dir_con = '/data/deluge/scratch/S2S/realtime/'+model+'/hgt/*control.nc'
files_con = glob.glob(model_dir_con)
files_con.sort()
model_dates_con = [i.split("_")[2] for i in files_con]
con_ind1 = model_dates_con.index(date_init)
con_ind2 = model_dates_con.index(date_end)


# Now get the perturbed and control arrays to put data into. Perturbed will be shaped INIT NO. x time x number x lat x lon.
# Control will be shaped INIT NO. x time x lat x lon.
perturbed_hgt = np.zeros((pert_ind2-pert_ind1+1, len(ltm_days), len(number), len(mod_region_lat), len(mod_region_lon)))
control_hgt = np.zeros((con_ind2-con_ind1+1, len(ltm_days), len(mod_region_lat), len(mod_region_lon)))
time_arr = []
time_init = []

# Now loop through files and store the data! First for perturbed.
for i in range(pert_ind1, pert_ind2+1):
    pert_filename = files_pert[i]
    nc = Dataset(pert_filename, 'r')
    time = num2date(nc.variables['time'][:], nc.variables['time'].units, calendar = 'gregorian', only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    perturbed_hgt[i-pert_ind1, :, :, :, :] = nc.variables['gh'][:, :, np.where(level == level_select)[0][0], mod_lat_ind1:mod_lat_ind2+1, mod_lon_ind1:mod_lon_ind2+1]
    nc.close()
    time_arr.append(time)
    time_init.append(time[0])

# Now for control.
for i in range(con_ind1, con_ind2+1):
    con_filename = files_con[i]
    nc = Dataset(con_filename, 'r')
    time = num2date(nc.variables['time'][:], nc.variables['time'].units, calendar = 'gregorian', only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    control_hgt[i-con_ind1, :, :, :] = nc.variables['gh'][:, np.where(level == level_select)[0][0], mod_lat_ind1:mod_lat_ind2+1, mod_lon_ind1:mod_lon_ind2+1]
    nc.close()

# Make time_init an array.

time_init = np.asarray(time_init)
time_arr = np.stack(time_arr)

# Now join all members with control.
all_members_hgt = np.concatenate((control_hgt[:, :, None, :, :], perturbed_hgt), axis = 2)

# Now go through each of the hindcast dates that you have chosen and take away the corresponding climo.
# Set a month and day locator for this.
months_ltm = np.array([d.month for d in ltm_dates])
days_ltm = np.array([d.day for d in ltm_dates])

hgt_anom = np.zeros((all_members_hgt.shape))

for i in range(len(time_init)):
    hgt_anom[i] = (all_members_hgt[i] - hgt_ltm_region[np.where((months_ltm == time_init[i].month)&(days_ltm == time_init[i].day))[0][0], :, None, :, :])

all_members_final = np.zeros((hgt_anom.shape[0], hgt_anom.shape[2], hgt_anom.shape[3], hgt_anom.shape[4]))
for i in range(len(time_init)):
    all_members_final[i] = np.nanmean(hgt_anom[i, np.where(time_arr[i]==cold_start)[0][0]:np.where(time_arr[i]==cold_end)[0][0]+1], axis = 0)

# Now read in k-means objects.
dir = '/share/data1/Students/ollie/CAOs/Data/Regimes/Cluster_Info_S2S'
path = os.chdir(dir)
kmeans = pickle.load(open("kmeans_object.pkl","rb"))
solver = pickle.load(open("eof_object.pkl","rb"))

## Get the PC loadings for each ensemble member and time
## output array is shape time, member, PC number
n_eof = 12 ## the number of EOFs to return, cannot exceed the number previously computed
pc_ens = np.zeros((all_members_final.shape[0],all_members_final.shape[1],n_eof))
for i in range(all_members_final.shape[0]):
	## project onto EOF & obtain psuedo PC
	## this automatically uses the same weights as when the EOF was constructed
	pc_ens[i,:] = solver.projectField(all_members_final[i,:],neofs=n_eof)

## now attribute each to a cluster based on the PC loadings
clusters = np.zeros((pc_ens.shape[0],pc_ens.shape[1]))
for i in range(clusters.shape[0]):
	clusters[i,:] = kmeans.predict(pc_ens[i,:])


# Make a plot of a certain regime number.
select_no = 4
run_ind = np.where(time_init == datetime(2021, 1, 28))[0][0]
select_clust = clusters[run_ind]
hgt_select = all_members_final[run_ind]
cluster_ind = np.where(select_clust == select_no)[0]

# Now composite to check this is correct!
comp_select = np.nanmean(hgt_select[cluster_ind], axis = 0)

# Plot the regimes.
clevs = np.arange(-30, 32, 2)
my_cmap, norm = NormColorMap('RdBu_r', clevs)

fig = plt.figure(figsize=(6, 6))
m = Basemap(width=14000000,height=8000000,resolution='l',projection='laea',lat_ts=60,lat_0=55,lon_0=-105.)
lonsmesh, latsmesh = np.meshgrid(mod_region_lon, mod_region_lat)
x,y = m(lonsmesh,latsmesh)
cf = m.contourf(x,y,comp_select/10,cmap=my_cmap,extend='both',levels=clevs)
m.drawcoastlines(linewidth=0.5)
plt.title("TEST")

# 0 = PT
# 1 = WCR
# 2 = AkR
# 3 = ArH
# 4 = ArL

# Go through each init and count how many PT, WCR, AkR, ArH, and ArL.

# Change the array to integers to use bincount.

clusters = clusters.astype(int)

pt_count = np.zeros(len(time_init))
wcr_count = np.zeros(len(time_init))
akr_count = np.zeros(len(time_init))
arh_count = np.zeros(len(time_init))
arl_count = np.zeros(len(time_init))

for i in range(len(time_init)):
	pt_count[i] = np.count_nonzero(clusters[i] == 0)
	wcr_count[i] = np.count_nonzero(clusters[i] == 1)
	akr_count[i] = np.count_nonzero(clusters[i] == 2)
	arh_count[i] = np.count_nonzero(clusters[i] == 3)
	arl_count[i] = np.count_nonzero(clusters[i] == 4)


# Create color list

colors = ['darkgreen', 'darkblue', 'darkorange', 'darkred', 'darkpurple']
# Now create a stacked bar graph.
labels = [i.strftime("%m-%d") for i in time_init]
fig, ax = plt.subplots(figsize = (8,3))
ax.bar(labels, pt_count, color = 'darkgreen', label = 'PT', edgecolor = 'black')
ax.bar(labels, wcr_count, label = 'WCR', color = 'darkblue', bottom = pt_count, edgecolor = 'black')
ax.bar(labels, akr_count, label = 'AkR', color = 'orange', bottom = pt_count+wcr_count, edgecolor = 'black')
ax.bar(labels, arh_count, label = 'ArH', color = 'darkred', bottom = pt_count+wcr_count+akr_count, edgecolor = 'black')
ax.bar(labels, arl_count, label = 'ArL', color = 'purple', bottom = pt_count+wcr_count+akr_count+arh_count, edgecolor = 'black')
plt.legend(loc = 'lower left', ncol = 5)
plt.xlabel('Initialization Date (MM-DD)', fontsize = 10, weight = 'bold')
plt.xticks(labels, rotation = 30)
plt.ylabel('Ensemble Member Count', fontsize = 10, weight = 'bold')
plt.title(f'b) ECMWF Regimes {cold_start.strftime("%Y-%m-%d")} to {cold_end.strftime("%Y-%m-%d")}')
plt.savefig("/share/data1/Students/ollie/CAOs/project-2021-cao/main/S2S_Models/Regimes/ECMWF/bar_init.png", bbox_inches = 'tight', dpi = 500)
