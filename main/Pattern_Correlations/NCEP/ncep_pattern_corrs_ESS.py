###### Import Modules ######
import numpy as np
from netCDF4 import Dataset, num2date
from scipy.stats import pearsonr
import os
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic
dir = '/share/data1/Students/ollie/CAOs/project-2021-cao/Functions'
path = os.chdir(dir)
from gen_utils import DrawPolygon, LambConfMap, NormColorMap, NPStere_Map
from model_utils import load_latlon, load_modelparam, load_model_ltm_2D, load_model_ltm_3D

###### Set spatial bounds ######
level = 500 # Level bound.
lat1, lat2 = 90, 30 # Latitude bounds.
lon1, lon2 = 0, 358.5 # Longitude bounds.

###### Read in the reanalysis ######
# Go to the data directory.
path = "/share/data1/Students/ollie/CAOs/Data/GPH/Non_Detrended/Regrid/1.5x1.5"
dir = os.chdir(path)

# Open the ERA5 500 hPa GPH daily averaged dataset.
filename = "era5.hgt.500.nhemi.ndjfm.all.daily.regrid.nc"
nc = Dataset(filename, 'r')
# Read in the data, lats, lons, levels, times etc.
latitude, longitude = nc.variables['lat'][:], nc.variables['lon'][:]
time_era5 = num2date(nc.variables['time'][:], nc.variables['time'].units, nc.variables['time'].calendar, only_use_cftime_datetimes = False, only_use_python_datetimes = True)
# Set lat/lon/time indices.
lat_ind1, lat_ind2 = np.where(latitude == lat1)[0][0], np.where(latitude == lat2)[0][0]
lon_ind1, lon_ind2 = np.where(longitude == lon1)[0][0], np.where(longitude == lon2)[0][0]
select_date = datetime(2021, 2, 2)
time_ind = np.where(time_era5 == select_date)[0][0]
hgt_era5 = nc.variables['hgt'][:, lat_ind1:lat_ind2+1, lon_ind1:lon_ind2+1] # Reads the hgt data in.
nc.close()

# Reshape into years, days, lat, lon.
start_year, end_year = 1950, 2021 # Set year limits for the era5 data.
years_arr = np.arange(start_year, end_year+1, 1) # Set year array.
hgt_era5_reshape = hgt_era5.reshape(years_arr.shape[0], 151, hgt_era5.shape[1], hgt_era5.shape[2])

# Calculate the ltm.
hgt_era5_ltm = np.nanmean(hgt_era5_reshape[np.where(years_arr == 1981)[0][0]:np.where(years_arr == 2010)[0][0]+1], axis = 0)

# Calculate ERA5 anomalies, reshape, and restrict to the dates required.
hgt_anom_era5 = hgt_era5_reshape - hgt_era5_ltm
hgt_anom_reshape_era5 = hgt_anom_era5.reshape(hgt_anom_era5.shape[0]*hgt_anom_era5.shape[1], hgt_anom_era5.shape[2], hgt_anom_era5.shape[3])

# Restrict to the correct times.
hgt_anom_period = hgt_anom_reshape_era5[time_ind]
date = time_era5[time_ind]

###### Read in ECMWF hgt data for high and low pv members ######

# Pick init date to read in.
date_str = '2021-01-28'

# Now go to directory for the data.
dir = '/data/deluge/models/S2S/realtime/NCEP/pv'
path = os.chdir(dir)

# Data is time, number, level, lat, lon.
# Open perturbed file.
filename = f'NCEP_pv_{date_str}_perturbed.nc'
nc = Dataset(filename, 'r')
# Read lat/lon.
mod_latitude, mod_longitude = load_latlon(nc)
# Now find lat/lon indices to restrict data to.
lat_ind1, lat_ind2 = np.where(mod_latitude == lat1)[0][0], np.where(mod_latitude == lat2)[0][0]
lon_ind1, lon_ind2 = np.where(mod_longitude == lon1)[0][0], np.where(mod_longitude == lon2)[0][0]
# Read in lead time, ensemble number array, and the pv data restricted to spatial domains.
time, number, init_time = load_modelparam(nc)
pv_pert = nc.variables['pv'][:, :, lat_ind1:lat_ind2+1, lon_ind1:lon_ind2+1]
# Close file.
nc.close()

# Now open control PV file.
filename = f'NCEP_pv_{date_str}_control.nc'
nc = Dataset(filename, 'r')
# Read in the pv control data to restricted spatial domains.
pv_con = nc.variables['pv'][:, lat_ind1:lat_ind2+1, lon_ind1:lon_ind2+1]
# Close file.
nc.close()

# Now concatenate together control and perturbed for whole ensemble suite (time, number, lat, lon).
all_pv = np.concatenate((pv_con[:, None, :, :], pv_pert), axis = 1)

# Restrict lat and lon regions.
lat_region, lon_region = mod_latitude[lat_ind1:lat_ind2+1], mod_longitude[lon_ind1:lon_ind2+1]

###### RESTRICT DATA TO WAVE BREAK TIME/REGION ######
# Now select the dates to average PV over for the Labrador Sea wave break.
lead_select1 = datetime(2021, 2, 2)
lead_select2 = datetime(2021, 2, 4)
# Restrict the pv data to the lead time in which this wave break happens.
pv_lead = np.nanmean(all_pv[np.where(time == lead_select1)[0][0]:np.where(time == lead_select2)[0][0]+1],axis = 0)

# Select lat and lon bounds for the East Siberian Sea wave break region.
lat_ave1, lat_ave2 = 85.5, 66.0
lon_ave1, lon_ave2 = 111, 187.5
# Find indices of these chosen bounds
lat_ave_ind1, lat_ave_ind2 = np.where(lat_region == lat_ave1)[0][0], np.where(lat_region == lat_ave2)[0][0]
lon_ave_ind1, lon_ave_ind2 = np.where(lon_region == lon_ave1)[0][0], np.where(lon_region == lon_ave2)[0][0]

# Now restrict the pv data, lats, and lons to the lat and lons of the wave break region.
pv_wb = pv_lead[:, lat_ave_ind1:lat_ave_ind2+1, lon_ave_ind1:lon_ave_ind2+1]
lat_ave, lon_ave = lat_region[lat_ave_ind1:lat_ave_ind2+1], lon_region[lon_ave_ind1:lon_ave_ind2+1]

# Now find weights from cosine of latitude of wave breaking region.
weights = np.cos(np.radians(lat_ave))
# Weighted average the pv data across latitude in the wave break region.
lat_ave_pv = np.average(pv_wb, weights = weights, axis = 1)
# Take zonal mean to complete the area-average.
wb_ave_pv = np.nanmean(lat_ave_pv, axis = -1)/(1e-6) # Unit conversion to look at PVU!

###### SEPARATE WAVE BREAK ENSEMBLE MEMBERS ######
# Sort the wave break region averaged PV from low to high and get indices.
ind_sort = np.argsort(wb_ave_pv)
# Take 5 members, approx 10% of 51.
no_members = 3
# Find the member indexes with lowest and highest PV values.
lowest_members, highest_members = ind_sort[:no_members], ind_sort[-no_members:]

###### READ IN HGT CLIMATOLOGY S2S ######
# Go to hgt ltm directory.
dir = '/share/data1/Students/ollie/CAOs/Data/Feb_2021_CAO/Model_Data/NCEP/hgt'
path = os.chdir(dir)
# Read in T2M ltm data and parameters
ltm_latitude_hgt, ltm_longitude_hgt, ltm_levels_hgt, ltm_dates_hgt, ltm_days_hgt, hgt_ltm_region = load_model_ltm_3D('NCEP', 'hgt', [lat1, lat2], [lon1, lon2], level = level)
# Set level index.
level_ind = np.where(ltm_levels_hgt == level)[0][0]

###### READ IN HGT DATA ######
# Go to real-time directory for hgt.
dir = '/data/deluge/models/S2S/realtime/NCEP/hgt'
path = os.chdir(dir)

# Open perturbed hgt data file.
filename = f'NCEP_hgt_{date_str}_perturbed.nc'
nc = Dataset(filename, 'r')
# Read in the lead time, init time.
time_hgt, init_time_hgt = load_modelparam(nc)[0], load_modelparam(nc)[-1]
# Now read in the hgt perturbed data restricted to the level and domain.
hgt_pert = nc.variables['gh'][:, :, level_ind, lat_ind1:lat_ind2+1, lon_ind1:lon_ind2+1]
# Close file
nc.close()

# Now open the control hgt file.
filename = f'NCEP_hgt_{date_str}_control.nc'
nc = Dataset(filename, 'r')
# Read in control hgt data with restricted domain.
hgt_con = nc.variables['gh'][:, level_ind, lat_ind1:lat_ind2+1, lon_ind1:lon_ind2+1]
# Close file.
nc.close()

# Now concatenate together all the control and perturbed data for full ensemble suite.
all_hgt = np.concatenate((hgt_con[:, None, :, :], hgt_pert), axis = 1)

###### GET HGT ANOMALIES ######
# Now set a day and month tracker for the ltm dates for hgt.
months_ltm_hgt = np.array([d.month for d in ltm_dates_hgt])
days_ltm_hgt = np.array([d.day for d in ltm_dates_hgt])

# Now get the anomalies by finding the indices where the hindcast dates months/days are the same as the init time
hgt_anom = (all_hgt - hgt_ltm_region[np.where((months_ltm_hgt == init_time_hgt.month)&(days_ltm_hgt == init_time_hgt.day))[0][0], :, None, :, :])

###### Isolate model hgt for wave break day and compare high vs low pv member correlation. ######
# Find where model time is the selected date.
model_time_ind = np.where(time_hgt == select_date)[0][0]

# Restrict model hgt to this day.
hgt_select_day = hgt_anom[model_time_ind]

# Restrict to low and high pv members.
lowpv_hgt, highpv_hgt = hgt_select_day[lowest_members], hgt_select_day[highest_members]

# Flatten arrays in spatial dimension.
lowpv_hgt_flat, highpv_hgt_flat = lowpv_hgt.reshape(lowpv_hgt.shape[0], lowpv_hgt.shape[1]*lowpv_hgt.shape[2]), highpv_hgt.reshape(highpv_hgt.shape[0], highpv_hgt.shape[1]*highpv_hgt.shape[2])

# Now set correlation arrays to store!
corr_low = np.zeros(len(lowest_members))
corr_high = np.zeros(len(highest_members))

# Now loop through and correlate the ERA5 with each low and high pv member correlations.
for i in range(len(lowest_members)):
    corr_low[i] = pearsonr(hgt_anom_period.reshape(hgt_anom_period.shape[0]*hgt_anom_period.shape[1]), lowpv_hgt_flat[i])[0]
    corr_high[i] = pearsonr(hgt_anom_period.reshape(hgt_anom_period.shape[0]*hgt_anom_period.shape[1]), highpv_hgt_flat[i])[0]

###### Plot out individual patterns ######
# Set colormap levels for hgt anoms.
clevs = np.arange(-50, 52, 2)
# Set normalized colormap for hgt anoms.
my_cmap, norm = NormColorMap('RdBu_r', clevs)

###### PLOTTING ######
# Fig dimensions: rows and columns
nrows = 2
ncols = 2
# Array for figure number to plot.
fig_no = np.arange(1, 4, 1)
# Set labels for figures.
labels = ['a) LOW PV 1', 'LOW PV 2', 'LOW PV 3']
# Generate figure amnd set figure size.
fig = plt.figure(figsize=(6, 6))

for i in range(len(corr_low)):
    # Add cyclic point to data.
    lowpv_hgt_shifted, lons_shifted = addcyclic(lowpv_hgt[i], lon_region)
    # Mesh the grid.
    lons, lats = np.meshgrid(lons_shifted, lat_region)
    # Add subplot at given position.
    ax = fig.add_subplot(nrows, ncols, fig_no[i])
    # Set up lambert conformal projection.
    map = NPStere_Map(30, -100)
    # Map the lons and lats to x and y coordinates.
    x, y = map(lons, lats)
    # Contourf the T2M
    cs = map.contourf(x, y, lowpv_hgt_shifted/10, clevs, norm = norm, extend='both', cmap = my_cmap)
    # Set the title for the plot.
    ax.set_title(f"{labels[i]} (r = {np.round(corr_low[i], 2)})", fontsize = 14, weight = 'bold')
    plt.tight_layout()
# Plot reanalysis.
# Add cyclic point to data.
era5_hgt_shifted, lons_shifted = addcyclic(hgt_anom_period, lon_region)
# Mesh the grid.
lons, lats = np.meshgrid(lons_shifted, lat_region)
# Add subplot at given position.
ax = fig.add_subplot(nrows, ncols, fig_no[-1]+1)
# Set up lambert conformal projection.
map = NPStere_Map(30, -100)
# Map the lons and lats to x and y coordinates.
x, y = map(lons, lats)
# Contourf the T2M
cs = map.contourf(x, y, era5_hgt_shifted/10, clevs, norm = norm, extend='both', cmap = my_cmap)
# Set the title for the plot.
ax.set_title(f"ERA5", fontsize = 14, weight = 'bold')
plt.tight_layout()

# Create colorbar.
# Add axes to place the colorbar.
cb_ax = fig.add_axes([0.05, -0.03, 0.91, 0.04])
# Insert the colorbar.
cbar = fig.colorbar(cs, cax=cb_ax,orientation="horizontal",ticks= np.arange(-50, 60, 10),extend="both")
# Set colorbar label.
cbar.set_label("500 hPa Geopotential Height Anomaly (dam)", fontsize = 16)
# Set colorbar tick size.
cbar.ax.tick_params(labelsize=13)
plt.savefig("/share/data1/Students/ollie/CAOs/project-2021-cao/main/Pattern_Correlations/NCEP/Corr_Plots/lowpv_corr_ESS_NCEP.png", bbox_inches = 'tight', dpi = 500)


# Set labels for figures.
labels = ['a) HIGH PV 1', 'HIGH PV 2', 'HIGH PV 3']
# Generate figure amnd set figure size.
fig = plt.figure(figsize=(6, 6))

for i in range(len(corr_high)):
    # Add cyclic point to data.
    highpv_hgt_shifted, lons_shifted = addcyclic(highpv_hgt[i], lon_region)
    # Mesh the grid.
    lons, lats = np.meshgrid(lons_shifted, lat_region)
    # Add subplot at given position.
    ax = fig.add_subplot(nrows, ncols, fig_no[i])
    # Set up lambert conformal projection.
    map = NPStere_Map(30, -100)
    # Map the lons and lats to x and y coordinates.
    x, y = map(lons, lats)
    # Contourf the T2M
    cs = map.contourf(x, y, highpv_hgt_shifted/10, clevs, norm = norm, extend='both', cmap = my_cmap)
    # Set the title for the plot.
    ax.set_title(f"{labels[i]} (r = {np.round(corr_high[i], 2)})", fontsize = 14, weight = 'bold')
    plt.tight_layout()
# Plot reanalysis.
# Add cyclic point to data.
era5_hgt_shifted, lons_shifted = addcyclic(hgt_anom_period, lon_region)
# Mesh the grid.
lons, lats = np.meshgrid(lons_shifted, lat_region)
# Add subplot at given position.
ax = fig.add_subplot(nrows, ncols, fig_no[-1]+1)
# Set up lambert conformal projection.
map = NPStere_Map(30, -100)
# Map the lons and lats to x and y coordinates.
x, y = map(lons, lats)
# Contourf the T2M
cs = map.contourf(x, y, era5_hgt_shifted/10, clevs, norm = norm, extend='both', cmap = my_cmap)
# Set the title for the plot.
ax.set_title(f"ERA5", fontsize = 14, weight = 'bold')
plt.tight_layout()

# Create colorbar.
# Add axes to place the colorbar.
cb_ax = fig.add_axes([0.05, -0.03, 0.91, 0.04])
# Insert the colorbar.
cbar = fig.colorbar(cs, cax=cb_ax,orientation="horizontal",ticks= np.arange(-50, 60, 10),extend="both")
# Set colorbar label.
cbar.set_label("500 hPa Geopotential Height Anomaly (dam)", fontsize = 16)
# Set colorbar tick size.
cbar.ax.tick_params(labelsize=13)
plt.savefig("/share/data1/Students/ollie/CAOs/project-2021-cao/main/Pattern_Correlations/NCEP/Corr_Plots/highpv_corr_ESS_NCEP.png", bbox_inches = 'tight', dpi = 500)

# Take average correlations and print them.
ave_corr_low = np.nanmean(corr_low)
ave_corr_high = np.nanmean(corr_high)
print("Average low PV correlation on 2nd Feb is:", ave_corr_low) # 0.776
print("Average high PV correlation on 2nd Feb is:", ave_corr_high) # 0.589

# Now find the differences between the low/high PV maps and ERA5 and plot the composite difference.
diff_low = lowpv_hgt - hgt_anom_period[None, :, :]
diff_high = highpv_hgt - hgt_anom_period[None, :, :]
comp_low_diff = np.nanmean(diff_low, axis = 0)/10 # For dam.
comp_high_diff = np.nanmean(diff_high, axis = 0)/10 # For dam.

# Set colormap levels for hgt anoms.
clevs = np.arange(-30, 32, 2)
# Set normalized colormap for hgt anoms.
my_cmap, norm = NormColorMap('bwr', clevs)

# Now plot the difference.
# Generate figure and set figure size.
fig = plt.figure(figsize=(12, 6))
# Shift the grid.
diff_low_shifted, lons_shifted = addcyclic(comp_low_diff, lon_region)
# Mesh the grid.
lons, lats = np.meshgrid(lons_shifted, lat_region)
# Add subplot at given position.
ax = fig.add_subplot(1, 2, 1)
# Set up lambert conformal projection.
map = NPStere_Map(30, -100)
# Map the lons and lats to x and y coordinates.
x, y = map(lons, lats)
# Contourf the T2M
cs = map.contourf(x, y, diff_low_shifted, clevs, norm = norm, extend='both', cmap = my_cmap)
# Set the title for the plot.
ax.set_title(f"LOW PV - ERA5", fontsize = 14, weight = 'bold')
plt.tight_layout()

# Shift the grid.
diff_high_shifted, lons_shifted = addcyclic(comp_high_diff, lon_region)
# Mesh the grid.
lons, lats = np.meshgrid(lons_shifted, lat_region)
# Add subplot at given position.
ax = fig.add_subplot(1, 2, 2)
# Set up lambert conformal projection.
map = NPStere_Map(30, -100)
# Map the lons and lats to x and y coordinates.
x, y = map(lons, lats)
# Contourf the T2M
cs = map.contourf(x, y, diff_high_shifted, clevs, norm = norm, extend='both', cmap = my_cmap)
# Set the title for the plot.
ax.set_title(f"HIGH PV - ERA5", fontsize = 14, weight = 'bold')
plt.tight_layout()

# Create colorbar.
# Add axes to place the colorbar.
cb_ax = fig.add_axes([0.05, -0.03, 0.91, 0.04])
# Insert the colorbar.
cbar = fig.colorbar(cs, cax=cb_ax,orientation="horizontal",ticks= np.arange(-30, 40, 10),extend="both")
# Set colorbar label.
cbar.set_label("500 hPa Geopotential Height Anomaly Difference (dam)", fontsize = 16)
# Set colorbar tick size.
cbar.ax.tick_params(labelsize=13)
plt.savefig("/share/data1/Students/ollie/CAOs/project-2021-cao/main/Pattern_Correlations/NCEP/Diff_Plots/diff_ESS_NCEP.png", bbox_inches = 'tight', dpi = 500)
