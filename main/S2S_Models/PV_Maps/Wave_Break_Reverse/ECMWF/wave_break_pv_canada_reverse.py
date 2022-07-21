import numpy as np
from netCDF4 import Dataset, num2date
from datetime import datetime, timedelta
import os
from mpl_toolkits.basemap import Basemap, addcyclic
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import glob
from scipy.stats import mode
from matplotlib.dates import DateFormatter, MonthLocator
import matplotlib.dates as mdates
from scipy.stats import percentileofscore
from tqdm import tqdm
dir = '/share/data1/Students/ollie/CAOs/project-2021-cao/Functions'
path = os.chdir(dir)
from gen_utils import DrawPolygon, LambConfMap, NormColorMap, NPStere_Map
from model_utils import load_latlon, load_modelparam

# Pick date to read in.
date_str = '2021-01-28'

# Pick lat, lon, you want to select.
lat1, lat2 = 90, 9
lon1, lon2 = 0, 358.5

# Now read in the PV data for the run.
dir = '/data/deluge/models/S2S/realtime/ECMWF/pv'
path = os.chdir(dir)

# Data is time, number, level, lat, lon.
# Open perturbed data.
filename = f'ECMWF_pv_{date_str}_perturbed.nc'
nc = Dataset(filename, 'r')
mod_latitude = nc.variables['latitude'][:]
mod_longitude = nc.variables['longitude'][:]

# Now find indices to restrict data.
lat_ind1, lat_ind2 = np.where(mod_latitude == lat1)[0][0], np.where(mod_latitude == lat2)[0][0]
lon_ind1, lon_ind2 = np.where(mod_longitude == lon1)[0][0], np.where(mod_longitude == lon2)[0][0]
time = num2date(nc.variables['time'][:], nc.variables['time'].units, calendar = 'gregorian', only_use_cftime_datetimes=False, only_use_python_datetimes=True)
number = nc.variables['number'][:]
pv_pert = nc.variables['pv'][:, :, lat_ind1:lat_ind2+1, lon_ind1:lon_ind2+1]
init_time = time[0]
nc.close()

# Now get the control PV data.
filename = f'ECMWF_pv_{date_str}_control.nc'
nc = Dataset(filename, 'r')
pv_con = nc.variables['pv'][:, lat_ind1:lat_ind2+1, lon_ind1:lon_ind2+1]
nc.close()

# Now we want to get the concatenated PV array of control and perturbed.
all_pv = np.concatenate((pv_con[:, None, :, :], pv_pert), axis = 1)

# Outline lat and lon regions.
lat_region, lon_region = mod_latitude[lat_ind1:lat_ind2+1], mod_longitude[lon_ind1:lon_ind2+1]

# Now we want to select the dates to average PV over.
lead_select1 = datetime(2021, 2, 2)
lead_select2 = datetime(2021, 2, 4)
pv_lead = np.nanmean(all_pv[np.where(time == lead_select1)[0][0]:np.where(time == lead_select2)[0][0]+1],axis = 0)

# Restrict PV to wavebreak region.
lat_ave1, lat_ave2 = 85.5, 66.0
lon_ave1, lon_ave2 = 111, 187.5
lat_ave_ind1, lat_ave_ind2 = np.where(lat_region == lat_ave1)[0][0], np.where(lat_region == lat_ave2)[0][0]
lon_ave_ind1, lon_ave_ind2 = np.where(lon_region == lon_ave1)[0][0], np.where(lon_region == lon_ave2)[0][0]

pv_wb = pv_lead[:, lat_ave_ind1:lat_ave_ind2+1, lon_ave_ind1:lon_ave_ind2+1]
lat_ave, lon_ave = lat_region[lat_ave_ind1:lat_ave_ind2+1], lon_region[lon_ave_ind1:lon_ave_ind2+1]

# Now average PV across wavebreak region.
weights = np.cos(np.radians(lat_ave))
lat_ave_pv = np.average(pv_wb, weights = weights, axis = 1)
wb_ave_pv = np.nanmean(lat_ave_pv, axis = -1)/(1e-6)

# Now take the 10% lowest and the 10% highest PV.
ind_sort = np.argsort(wb_ave_pv)
no_members = 5
lowest_members, highest_members = ind_sort[:no_members], ind_sort[-no_members:]

# Now read in the surfT data for this run.
dir = '/data/deluge/models/S2S/realtime/ECMWF/surfT'
path = os.chdir(dir)

# Data is time, number, lat, lon.
# Open perturbed data.
filename = f'ECMWF_surfT_{date_str}_perturbed.nc'
nc = Dataset(filename, 'r')
time_surfT = num2date(nc.variables['time'][:], nc.variables['time'].units, calendar = 'gregorian', only_use_cftime_datetimes=False, only_use_python_datetimes=True)
surfT_pert = nc.variables['t2m'][:, :, lat_ind1:lat_ind2+1, lon_ind1:lon_ind2+1]
init_time_surfT = time_surfT[0]
nc.close()

# Now get the control data.
filename = f'ECMWF_surfT_{date_str}_control.nc'
nc = Dataset(filename, 'r')
surfT_con = nc.variables['t2m'][:, lat_ind1:lat_ind2+1, lon_ind1:lon_ind2+1]
nc.close()

# Now we want to get the concatenated surfT array of control and perturbed.
all_surfT = np.concatenate((surfT_con[:, None, :, :], surfT_pert), axis = 1)

# Now read in the climo.
# Get the ltm data.
dir = '/share/data1/Students/ollie/CAOs/Data/Feb_2021_CAO/Model_Data/ECMWF/surfT'
path = os.chdir(dir)

nc = Dataset('climo_ECMWF_surfT_hindcast.nc', 'r')
ltm_latitude = nc.variables['latitude'][:]
ltm_longitude = nc.variables['longitude'][:]
ltm_dates = num2date(nc.variables['hdates'][:], nc.variables['hdates'].units, calendar = 'gregorian', only_use_cftime_datetimes=False, only_use_python_datetimes=True)
ltm_days = nc.variables['days'][:]
ltm_surfT = nc.variables['surfT'][:]
nc.close()

# Now restrict to the same regions are your real-time array.
ltm_lat_ind1, ltm_lat_ind2 = np.where(ltm_latitude == lat1)[0][0], np.where(ltm_latitude == lat2)[0][0]
ltm_lon_ind1, ltm_lon_ind2 = np.where(ltm_longitude == lon1)[0][0], np.where(ltm_longitude == lon2)[0][0]
surfT_ltm_region = ltm_surfT[:, :, ltm_lat_ind1:ltm_lat_ind2+1, ltm_lon_ind1:ltm_lon_ind2+1]

# Now set a day and month tracker for the ltm dates.
months_ltm = np.array([d.month for d in ltm_dates])
days_ltm = np.array([d.day for d in ltm_dates])

# Now get the anomalies.
surfT_anom = (all_surfT - surfT_ltm_region[np.where((months_ltm == init_time.month)&(days_ltm == init_time.day))[0][0], :, None, :, :])

# Now import hgt data.
level = 500

# Get the ltm data.
dir = '/share/data1/Students/ollie/CAOs/Data/Feb_2021_CAO/Model_Data/ECMWF/hgt'
path = os.chdir(dir)

nc = Dataset('climo_ECMWF_hgt_hindcast.nc', 'r')
ltm_latitude_hgt = nc.variables['latitude'][:]
ltm_longitude_hgt = nc.variables['longitude'][:]
ltm_levels_hgt = nc.variables['level'][:]
ltm_dates_hgt = num2date(nc.variables['hdates'][:], nc.variables['hdates'].units, calendar = 'gregorian', only_use_cftime_datetimes=False, only_use_python_datetimes=True)
ltm_days_hgt = nc.variables['days'][:]
ltm_hgt = nc.variables['hgt'][:]
nc.close()

ltm_lat_ind1_hgt, ltm_lat_ind2_hgt = np.where(ltm_latitude_hgt == lat1)[0][0], np.where(ltm_latitude_hgt == lat2)[0][0]
ltm_lon_ind1_hgt, ltm_lon_ind2_hgt = np.where(ltm_longitude_hgt == lon1)[0][0], np.where(ltm_longitude_hgt == lon2)[0][0]
level_ind = np.where(ltm_levels_hgt == level)[0][0]
hgt_ltm_region = ltm_hgt[:, :, level_ind, ltm_lat_ind1_hgt:ltm_lat_ind2_hgt+1, ltm_lon_ind1_hgt:ltm_lon_ind2_hgt+1]

# Now read in the hgt data for this run.
dir = '/data/deluge/models/S2S/realtime/ECMWF/hgt'
path = os.chdir(dir)

# Data is time, number, lat, lon.
# Open perturbed data.
filename = f'ECMWF_hgt_{date_str}_perturbed.nc'
nc = Dataset(filename, 'r')
time_hgt = num2date(nc.variables['time'][:], nc.variables['time'].units, calendar = 'gregorian', only_use_cftime_datetimes=False, only_use_python_datetimes=True)
hgt_pert = nc.variables['gh'][:, :, level_ind, lat_ind1:lat_ind2+1, lon_ind1:lon_ind2+1]
init_time_hgt = time_hgt[0]
nc.close()

# Now get the control data.
filename = f'ECMWF_hgt_{date_str}_control.nc'
nc = Dataset(filename, 'r')
hgt_con = nc.variables['gh'][:, level_ind, lat_ind1:lat_ind2+1, lon_ind1:lon_ind2+1]
nc.close()

# Now we want to get the concatenated hgt array of control and perturbed.
all_hgt = np.concatenate((hgt_con[:, None, :, :], hgt_pert), axis = 1)

# Now set a day and month tracker for the ltm dates.
months_ltm_hgt = np.array([d.month for d in ltm_dates_hgt])
days_ltm_hgt = np.array([d.day for d in ltm_dates_hgt])

# Now get the anomalies.
hgt_anom = (all_hgt - hgt_ltm_region[np.where((months_ltm_hgt == init_time_hgt.month)&(days_ltm_hgt == init_time_hgt.day))[0][0], :, None, :, :])/10


# Now restrict between the dates you want.
lead_1_surfT = datetime(2021, 2, 14)
lead_2_surfT = datetime(2021, 2, 16)
dates_select = time_surfT[np.where(time_surfT == lead_1_surfT)[0][0]:np.where(time_surfT == lead_2_surfT)[0][0]+1]
surfT_lead = surfT_anom[np.where(time_surfT == lead_1_surfT)[0][0]:np.where(time_surfT == lead_2_surfT)[0][0]+1]
hgt_lead = hgt_anom[np.where(time_hgt == lead_1_surfT)[0][0]:np.where(time_hgt == lead_2_surfT)[0][0]+1]

# Separate PV by category.
lowpv_surfT = np.nanmean(surfT_lead[:, lowest_members], axis = 1)
highpv_surfT = np.nanmean(surfT_lead[:, highest_members], axis = 1)
diff_surfT = lowpv_surfT - highpv_surfT
lowpv_hgt = np.nanmean(hgt_lead[:, lowest_members], axis = 1)
highpv_hgt = np.nanmean(hgt_lead[:, highest_members], axis = 1)

# Do significance testing for difference between low and high composites of surfT.
# Do some bootstrap for the significance on the difference.
no_samples = 5000
ens_ind_arr = np.arange(0, 51, 1)
comp_boot_surfT = np.zeros((no_samples, len(dates_select), len(lat_region), len(lon_region)))
for i in tqdm(range(no_samples)):
    rand_inds = np.random.choice(ens_ind_arr, size = (len(lowest_members)+len(highest_members)), replace  = False) # Selects random ensemble number indexes.
    split_indexes = np.split(rand_inds, [len(lowest_members)])
    boot_low_ind, boot_high_ind = split_indexes[0], split_indexes[1]
    comp_boot_low, comp_boot_high = np.nanmean(surfT_lead[:, boot_low_ind], axis = 1), np.nanmean(surfT_lead[:, boot_high_ind], axis = 1)
    comp_boot_surfT[i] = comp_boot_low - comp_boot_high

# Now find the where your composite difference lies in the bootstrapped distribution at each grid point.
percent_arr_surfT = np.zeros((len(dates_select), len(lat_region), len(lon_region)))
for i in tqdm(range(len(dates_select))):
    for j in range(len(lat_region)):
        for k in range(len(lon_region)):
            percent_arr_surfT[i, j, k] = percentileofscore(comp_boot_surfT[:, i, j, k], diff_surfT[i, j, k])

clevs = np.arange(-10, 10.5, 0.5)
my_cmap, norm = NormColorMap('RdBu_r', clevs)
perc1, perc2 = 5, 95

# Make figure.
# Fig dimensions:
nrows = 2
ncols = 3
fig_no = np.arange(1, 7, 1)
labels = ['d)', 'e)', 'f)', 'g)', 'h)', 'i)']
fig = plt.figure(figsize=(12, 6))
for i in tqdm(range(len(dates_select))):
    if i == 0:
        # Now plot low pv surfT.
        lowpv_surfT_shifted, lowpv_hgt_shifted, sig_shifted, lons_shifted = addcyclic(lowpv_surfT[i], lowpv_hgt[i],  percent_arr_surfT[i], lon_region)
        lons, lats = np.meshgrid(lons_shifted, lat_region)
        surfT_masked = np.ma.masked_where((sig_shifted >= perc1)&(sig_shifted <= perc2), lowpv_surfT_shifted)

        ax = fig.add_subplot(nrows, ncols, fig_no[i])

        # Basemap plot.
        map = LambConfMap(9000000, 8100000, 1000.0, [35, 75], [55, -100])
        x, y = map(lons, lats)

        # Contourf.
        cs = map.contourf(x, y, lowpv_surfT_shifted, clevs, norm = norm, extend='both', cmap = my_cmap)
        lines = map.contour(x, y, lowpv_hgt_shifted, levels = [-30, -25, -20, -15, -10,-5, 5, 10, 15, 20, 25, 30], colors = 'black')
        plt.clabel(lines)
        pcol = plt.pcolor(x, y, surfT_masked, hatch = '////', alpha = 0)
        #p1, p2, p3, p4 = DrawPolygon(map, lat_range = [48, 30], lon_range = [256.5, 268.5], grid_space = 1.5, lw = 2, color = 'purple')
        ax.set_title(f"{labels[i]} {dates_select[i].strftime('%Y/%m/%d')}", fontsize = 14, weight = 'bold')
        text = ax.text(-700000,3500000,"Low PV Members", size=13, verticalalignment='center', rotation=90., weight = 'bold')
        plt.tight_layout()
    else:
        # Now plot low pv surfT.
        lowpv_surfT_shifted, lowpv_hgt_shifted, sig_shifted, lons_shifted = addcyclic(lowpv_surfT[i], lowpv_hgt[i], percent_arr_surfT[i], lon_region)
        lons, lats = np.meshgrid(lons_shifted, lat_region)
        surfT_masked = np.ma.masked_where((sig_shifted >= perc1)&(sig_shifted <= perc2), lowpv_surfT_shifted)

        ax = fig.add_subplot(nrows, ncols, fig_no[i])

        # Basemap plot.
        map = LambConfMap(9000000, 8100000, 1000.0, [35, 75], [55, -100])
        x, y = map(lons, lats)

        # Contourf.
        cs = map.contourf(x, y, lowpv_surfT_shifted, clevs, norm = norm, extend='both', cmap = my_cmap)
        lines = map.contour(x, y, lowpv_hgt_shifted, levels = [-30, -25, -20, -15, -10,-5, 5, 10, 15, 20, 25, 30], colors = 'black')
        plt.clabel(lines)
        pcol = plt.pcolor(x, y, surfT_masked, hatch = '////', alpha = 0)
        #p1, p2, p3, p4 = DrawPolygon(map, lat_range = [48, 30], lon_range = [256.5, 268.5], grid_space = 1.5, lw = 2, color = 'purple')
        ax.set_title(f"{labels[i]} {dates_select[i].strftime('%Y/%m/%d')}", fontsize = 14, weight = 'bold')
        plt.tight_layout()

for i in range(len(dates_select)):
    if i == 0:
        # Now plot high pv surfT.
        highpv_surfT_shifted, highpv_hgt_shifted, lons_shifted = addcyclic(highpv_surfT[i], highpv_hgt[i], lon_region)
        lons, lats = np.meshgrid(lons_shifted, lat_region)

        ax = fig.add_subplot(nrows, ncols, fig_no[i+3])

        # Basemap plot.
        map = LambConfMap(9000000, 8100000, 1000.0, [35, 75], [55, -100])
        x, y = map(lons, lats)

        # Contourf.
        cs = map.contourf(x, y, highpv_surfT_shifted, clevs, norm = norm, extend='both', cmap = my_cmap)
        lines = map.contour(x, y, highpv_hgt_shifted, levels = [-30, -25, -20, -15, -10,-5, 5, 10, 15, 20, 25, 30], colors = 'black')
        plt.clabel(lines)
        #p1, p2, p3, p4 = DrawPolygon(map, lat_range = [48, 30], lon_range = [256.5, 268.5], grid_space = 1.5, lw = 2, color = 'purple')
        text = ax.text(-700000,3500000,"High PV Members", size=13, verticalalignment='center', rotation=90., weight = 'bold')
        ax.set_title(f"{labels[i+3]} {dates_select[i].strftime('%Y/%m/%d')}", fontsize = 14, weight = 'bold')
        plt.tight_layout()
    else:
        # Now plot high pv surfT.
        highpv_surfT_shifted, highpv_hgt_shifted, lons_shifted = addcyclic(highpv_surfT[i], highpv_hgt[i], lon_region)
        lons, lats = np.meshgrid(lons_shifted, lat_region)

        ax = fig.add_subplot(nrows, ncols, fig_no[i+3])

        # Basemap plot.
        map = LambConfMap(9000000, 8100000, 1000.0, [35, 75], [55, -100])
        x, y = map(lons, lats)

        # Contourf.
        cs = map.contourf(x, y, highpv_surfT_shifted, clevs, norm = norm, extend='both', cmap = my_cmap)
        lines = map.contour(x, y, highpv_hgt_shifted, levels = [-30, -25, -20, -15, -10,-5, 5, 10, 15, 20, 25, 30], colors = 'black')
        plt.clabel(lines)
        #p1, p2, p3, p4 = DrawPolygon(map, lat_range = [48, 30], lon_range = [256.5, 268.5], grid_space = 1.5, lw = 2, color = 'purple')
        ax.set_title(f"{labels[i+3]} {dates_select[i].strftime('%Y/%m/%d')}", fontsize = 14, weight = 'bold')
        plt.tight_layout()

cb_ax = fig.add_axes([0.05, -0.03, 0.91, 0.04])
cbar = fig.colorbar(cs, cax=cb_ax,orientation="horizontal",ticks= np.arange(-10, 12, 2),extend="both")
cbar.set_label("T2M Anomaly ($^\circ$C)", fontsize = 16)
cbar.ax.tick_params(labelsize=13)
plt.savefig(f'/share/data1/Students/ollie/CAOs/project-2021-cao/main/S2S_Models/PV_Maps/Wave_Break_Reverse/ECMWF/test_{date_str}_canada_t2m_ecmwf.png', bbox_inches = 'tight', dpi = 500)



# Plot the PV patterns out too!
pv_low = np.nanmean(pv_lead[lowest_members], axis = 0)
pv_high= np.nanmean(pv_lead[highest_members], axis = 0)
pv_diff = pv_low - pv_high

# Do some bootstrap for the significance on the difference.
no_samples = 5000
ens_ind_arr = np.arange(0, 51, 1)
comp_boot_pv = np.zeros((no_samples, len(lat_region), len(lon_region)))
for i in tqdm(range(no_samples)):
    rand_inds = np.random.choice(ens_ind_arr, size = (len(lowest_members)+len(highest_members)), replace  = False) # Selects random ensemble number indexes.
    split_indexes = np.split(rand_inds, [len(lowest_members)])
    boot_low_ind, boot_high_ind = split_indexes[0], split_indexes[1]
    comp_boot_low, comp_boot_high = np.nanmean(pv_lead[boot_low_ind], axis = 0), np.nanmean(pv_lead[boot_high_ind], axis = 0)
    comp_boot_pv[i] = comp_boot_low - comp_boot_high

# Now find the where your composite difference lies in the bootstrapped distribution at each grid point.
percent_arr_pv = np.zeros((len(lat_region), len(lon_region)))
for i in range(len(lat_region)):
    for j in range(len(lon_region)):
        percent_arr_pv[i, j] = percentileofscore(comp_boot_pv[:, i, j], pv_diff[i, j])


# Cmap for the differences
clevs_pv = np.arange(-5, 5.5, 0.5)
my_cmap, norm = NormColorMap('bwr', clevs_pv)


# Make figure.
fig = plt.figure(figsize = (12, 6))
ax = fig.add_subplot(1, 3, 1)

pv_low_shifted, lons_shifted = addcyclic(pv_low, lon_region)
lons, lats = np.meshgrid(lons_shifted, lat_region)
map = map = NPStere_Map(9, -100)
x, y = map(lons, lats)

#Contourf.
cs = map.contourf(x, y, pv_low_shifted/1e-6, np.arange(0, 7.2, 0.2), extend='max', cmap = 'Spectral_r')
line = map.contour(x, y, pv_low_shifted/(1e-6), [2], colors = 'black')
p1, p2, p3, p4 = DrawPolygon(map, lat_range = [lat_ave1, lat_ave2], lon_range = [lon_ave1, lon_ave2], grid_space = 1.5, lw = 3, color = 'black')
cbar = map.colorbar(cs, location = 'bottom', pad = "5%", ticks = np.arange(0, 8, 1))
cbar.ax.set_xlabel('PV (PVU)', fontsize = 12)
cbar.ax.tick_params(labelsize=11)
ax.set_title(f"a) LOW PV", fontsize = 15, weight = 'bold')
plt.tight_layout()

ax = fig.add_subplot(1, 3, 2)

pv_high_shifted, lons_shifted = addcyclic(pv_high, lon_region)
lons, lats = np.meshgrid(lons_shifted, lat_region)
map = map = NPStere_Map(9, -100)
x, y = map(lons, lats)

#Contourf.
cs = map.contourf(x, y, pv_high_shifted/1e-6, np.arange(0, 7.2, 0.2), extend='max', cmap = 'Spectral_r')
line = map.contour(x, y, pv_high_shifted/(1e-6), [2], colors = 'black')
p1, p2, p3, p4 = DrawPolygon(map, lat_range = [lat_ave1, lat_ave2], lon_range = [lon_ave1, lon_ave2], grid_space = 1.5, lw = 3, color = 'black')
cbar = map.colorbar(cs, location = 'bottom', pad = "5%", ticks = np.arange(0, 8, 1))
cbar.ax.set_xlabel('PV (PVU)', fontsize = 12)
cbar.ax.tick_params(labelsize=11)
ax.set_title(f"b) HIGH PV", fontsize = 15, weight = 'bold')
plt.tight_layout()

ax = fig.add_subplot(1, 3, 3)

pv_diff_shifted, sig_shifted, lons_shifted = addcyclic(pv_diff, percent_arr_pv, lon_region)
lons, lats = np.meshgrid(lons_shifted, lat_region)
pv_masked = np.ma.masked_where((sig_shifted >= perc1)&(sig_shifted <= perc2), pv_diff_shifted)
map = map = NPStere_Map(9, -100)
x, y = map(lons, lats)

#Contourf.
cs = map.contourf(x, y, pv_diff_shifted/1e-6, clevs_pv, norm = norm, extend='both', cmap = my_cmap)
pcol = plt.pcolor(x, y, pv_masked, hatch = '////', alpha = 0)
p1, p2, p3, p4 = DrawPolygon(map, lat_range = [lat_ave1, lat_ave2], lon_range = [lon_ave1, lon_ave2], grid_space = 1.5, lw = 3, color = 'black')
cbar = map.colorbar(cs, location = 'bottom', pad = "5%", ticks = np.arange(-5, 6, 1))
cbar.ax.set_xlabel('PV (PVU)', fontsize = 12)
cbar.ax.tick_params(labelsize=11)
ax.set_title(f"c) LOW - HIGH PV", fontsize = 15, weight = 'bold')
plt.tight_layout()
plt.savefig(f'/share/data1/Students/ollie/CAOs/project-2021-cao/main/S2S_Models/PV_Maps/Wave_Break_Reverse/ECMWF/test_{date_str}_canada_pv_ecmwf.png', bbox_inches = 'tight', dpi = 500)
