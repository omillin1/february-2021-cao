###### IMPORT MODULES ######
import sys
sys.path.insert(5, '../')
import numpy as np
from netCDF4 import Dataset, num2date
from datetime import datetime, timedelta
import os
from mpl_toolkits.basemap import Basemap, addcyclic
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import glob
from matplotlib.dates import DateFormatter, MonthLocator
import matplotlib.dates as mdates
from scipy.stats import percentileofscore
from tqdm import tqdm
from Functions import gen_utils, model_utils

###### READ IN PV DATA ######
# Pick init date to read in.
date_str = '2021-01-28'
# Pick lat, lon, you want to select.
lat1, lat2 = 90, 9
lon1, lon2 = 0, 358.5

# Now go to directory for the data.
dir = '/data/deluge/models/S2S/realtime/ECMWF/pv'
path = os.chdir(dir)

# Data is time, number, level, lat, lon.
# Open perturbed file.
filename = f'ECMWF_pv_{date_str}_perturbed.nc'
nc = Dataset(filename, 'r')
# Read lat/lon.
mod_latitude, mod_longitude = model_utils.load_latlon(nc)
# Now find lat/lon indices to restrict data to.
lat_ind1, lat_ind2 = np.where(mod_latitude == lat1)[0][0], np.where(mod_latitude == lat2)[0][0]
lon_ind1, lon_ind2 = np.where(mod_longitude == lon1)[0][0], np.where(mod_longitude == lon2)[0][0]
# Read in lead time, ensemble number array, and the pv data restricted to spatial domains.
time, number, init_time = model_utils.load_modelparam(nc)
pv_pert = nc.variables['pv'][:, :, lat_ind1:lat_ind2+1, lon_ind1:lon_ind2+1]
# Close file.
nc.close()

# Now open control PV file.
filename = f'ECMWF_pv_{date_str}_control.nc'
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
# Now select the dates to average PV over for the East Siberian Sea wave break.
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
no_members = 5
# Find the member indexes with lowest and highest PV values.
lowest_members, highest_members = ind_sort[:no_members], ind_sort[-no_members:]

###### READ IN T2M DATA ######
# Go to the data directory for T2M.
dir = '/data/deluge/models/S2S/realtime/ECMWF/surfT'
path = os.chdir(dir)

# Data is time, number, lat, lon.
# Open perturbed data file for T2M.
filename = f'ECMWF_surfT_{date_str}_perturbed.nc'
nc = Dataset(filename, 'r')
# Read in time, init time, and T2M data restricted to N Hemi domain.
time_surfT, init_time_surfT = model_utils.load_modelparam(nc)[0], model_utils.load_modelparam(nc)[-1]
surfT_pert = nc.variables['t2m'][:, :, lat_ind1:lat_ind2+1, lon_ind1:lon_ind2+1]
# Close file.
nc.close()

# Now open control file for T2M.
filename = f'ECMWF_surfT_{date_str}_control.nc'
nc = Dataset(filename, 'r')
# Read in T2M data in restricted domain.
surfT_con = nc.variables['t2m'][:, lat_ind1:lat_ind2+1, lon_ind1:lon_ind2+1]
# Close file.
nc.close()

# Now concatenate the perturbed and control T2M to get the full ensemble suite (time, number, lat, lon).
all_surfT = np.concatenate((surfT_con[:, None, :, :], surfT_pert), axis = 1)

###### READ IN T2M CLIMATOLOGY S2S ######
# Go to T2M climo directory.
dir = '/share/data1/Students/ollie/CAOs/Data/Feb_2021_CAO/Model_Data/ECMWF/surfT'
path = os.chdir(dir)
# Read in T2M ltm data and parameters
ltm_latitude, ltm_longitude, ltm_dates, ltm_days, surfT_ltm_region = model_utils.load_model_ltm_2D('ECMWF', 'surfT', [lat1, lat2], [lon1, lon2])

###### CALCULATE T2M ANOMALIES ######
# Now set a day and month tracker for the ltm dates.
months_ltm = np.array([d.month for d in ltm_dates])
days_ltm = np.array([d.day for d in ltm_dates])

# Now get the anomalies by finding the indices where the hindcast dates months/days are the same as the init time
surfT_anom = (all_surfT - surfT_ltm_region[np.where((months_ltm == init_time_surfT.month)&(days_ltm == init_time_surfT.day))[0][0], :, None, :, :])

###### READ IN HGT CLIMATOLOGY S2S ######
# Specify level in atmosphere
level = 500
# Go to hgt ltm directory.
dir = '/share/data1/Students/ollie/CAOs/Data/Feb_2021_CAO/Model_Data/ECMWF/hgt'
path = os.chdir(dir)
# Read in T2M ltm data and parameters
ltm_latitude_hgt, ltm_longitude_hgt, ltm_levels_hgt, ltm_dates_hgt, ltm_days_hgt, hgt_ltm_region = model_utils.load_model_ltm_3D('ECMWF', 'hgt', [lat1, lat2], [lon1, lon2], level = level)
# Set level index.
level_ind = np.where(ltm_levels_hgt == level)[0][0]

###### READ IN HGT DATA ######
# Go to real-time directory for hgt.
dir = '/data/deluge/models/S2S/realtime/ECMWF/hgt'
path = os.chdir(dir)

# Open perturbed hgt data file.
filename = f'ECMWF_hgt_{date_str}_perturbed.nc'
nc = Dataset(filename, 'r')
# Read in the lead time, init time.
time_hgt, init_time_hgt = model_utils.load_modelparam(nc)[0], model_utils.load_modelparam(nc)[-1]
# Now read in the hgt perturbed data restricted to the level and domain.
hgt_pert = nc.variables['gh'][:, :, level_ind, lat_ind1:lat_ind2+1, lon_ind1:lon_ind2+1]
# Close file
nc.close()

# Now open the control hgt file.
filename = f'ECMWF_hgt_{date_str}_control.nc'
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
hgt_anom = (all_hgt - hgt_ltm_region[np.where((months_ltm_hgt == init_time_hgt.month)&(days_ltm_hgt == init_time_hgt.day))[0][0], :, None, :, :])/10

###### SEPARATE FIELDS INTO LOW AND HIGH PV ######
# Now restrict t2m, hgt, and date arr between the dates you want to study.
# Select dates for mid feb.
lead_1_surfT = datetime(2021, 2, 14)
lead_2_surfT = datetime(2021, 2, 16)
# Find the dates array for these.
dates_select = time_surfT[np.where(time_surfT == lead_1_surfT)[0][0]:np.where(time_surfT == lead_2_surfT)[0][0]+1]
# Find restricted T2M and hgt arrays for these days in the run.
surfT_lead = surfT_anom[np.where(time_surfT == lead_1_surfT)[0][0]:np.where(time_surfT == lead_2_surfT)[0][0]+1]
hgt_lead = hgt_anom[np.where(time_hgt == lead_1_surfT)[0][0]:np.where(time_hgt == lead_2_surfT)[0][0]+1]

# Now restrict T2M low pv members and average along ensemble member for composite.
lowpv_surfT = np.nanmean(surfT_lead[:, lowest_members], axis = 1)
# Now restrict T2M high pv members and average along ensemble member for composite.
highpv_surfT = np.nanmean(surfT_lead[:, highest_members], axis = 1)
# Now difference between low and high pv temperature composites.
diff_surfT = lowpv_surfT - highpv_surfT
# Now restrict hgt low pv members and average along ensemble member for composite.
lowpv_hgt = np.nanmean(hgt_lead[:, lowest_members], axis = 1)
# Now restrict hgt high pv members and average along ensemble member for composite.
highpv_hgt = np.nanmean(hgt_lead[:, highest_members], axis = 1)

###### SIGNIFICANCE TESTING ON THE LOW-HIGH PV MEMBERS T2M COMPOSITE DIFFERENCE ######.
# Set number of bootstrap resamples.
no_samples = 5000
# Set ensemble number array.
ens_ind_arr = np.arange(0, 51, 1)
# Set the array of shape (5000, 3 days, lat, lon) to store bootstrap data in.
comp_boot_surfT = np.zeros((no_samples, len(dates_select), len(lat_region), len(lon_region)))
# Now loop through number of samples to take.
for i in tqdm(range(no_samples)):
    rand_inds = np.random.choice(ens_ind_arr, size = (len(lowest_members)+len(highest_members)), replace  = False) # Selects random ensemble number indexes without replacement.
    split_indexes = np.split(rand_inds, [len(lowest_members)]) # Split chosen ensemble members into two groups.
    boot_low_ind, boot_high_ind = split_indexes[0], split_indexes[1] # Select the random low and high groups.
    comp_boot_low, comp_boot_high = np.nanmean(surfT_lead[:, boot_low_ind], axis = 1), np.nanmean(surfT_lead[:, boot_high_ind], axis = 1) # Composite the random two sets.
    comp_boot_surfT[i] = comp_boot_low - comp_boot_high # Store T2M composite difference.

# Set array to store percentages of where your composite lies in the random distribution at each grid point.
percent_arr_surfT = np.zeros((len(dates_select), len(lat_region), len(lon_region)))
# Loop through 3 chosen dates, lat, and lon and find the percentile of your composite in the bootstrap distribution.
for i in tqdm(range(len(dates_select))):
    for j in range(len(lat_region)):
        for k in range(len(lon_region)):
            percent_arr_surfT[i, j, k] = percentileofscore(comp_boot_surfT[:, i, j, k], diff_surfT[i, j, k])

# Set colorbar levels.
clevs = np.arange(-10, 10.5, 0.5)
# Create the normalized colormap.
my_cmap, norm = gen_utils.NormColorMap('RdBu_r', clevs)
# Set significant levels (two-sided)
perc1, perc2 = 5, 95

###### PLOTTING ######
# Fig dimensions: rows and columns
nrows = 2
ncols = 3
# Array for figure number to plot.
fig_no = np.arange(1, 7, 1)
# Set labels for figures.
labels = ['d)', 'e)', 'f)', 'g)', 'h)', 'i)']
# Generate figure amnd set figure size.
fig = plt.figure(figsize=(12, 6))
# Loop through the three dates and plot the LOW PV hgt and T2M composites.
for i in tqdm(range(len(dates_select))):
    # This if statement adds the low PV label to the plot on LHS.
    if i == 0:
        # Add cyclic point to data (not necessary for this type of plot, but useful if you want to plot N Hemi instead).
        lowpv_surfT_shifted, lowpv_hgt_shifted, sig_shifted, lons_shifted = addcyclic(lowpv_surfT[i], lowpv_hgt[i],  percent_arr_surfT[i], lon_region)
        # Mesh the grid.
        lons, lats = np.meshgrid(lons_shifted, lat_region)
        # Mask where the T2M where the significance is >5% or <95%.
        surfT_masked = np.ma.masked_where((sig_shifted >= perc1)&(sig_shifted <= perc2), lowpv_surfT_shifted)
        # Add subplot at given position.
        ax = fig.add_subplot(nrows, ncols, fig_no[i])
        # Set up lambert conformal projection.
        map = gen_utils.LambConfMap(9000000, 8100000, 1000.0, [35, 75], [55, -100])
        # Map the lons and lats to x and y coordinates.
        x, y = map(lons, lats)
        # Contourf the T2M
        cs = map.contourf(x, y, lowpv_surfT_shifted, clevs, norm = norm, extend='both', cmap = my_cmap)
        # Contour the hgt, and add labels.
        lines = map.contour(x, y, lowpv_hgt_shifted, levels = [-30, -25, -20, -15, -10,-5, 5, 10, 15, 20, 25, 30], colors = 'black')
        plt.clabel(lines)
        # Place hatching over the significant area using the mask.
        pcol = plt.pcolor(x, y, surfT_masked, hatch = '////', alpha = 0)
        # Set the title for the plot.
        ax.set_title(f"{labels[i]} {dates_select[i].strftime('%Y/%m/%d')}", fontsize = 14, weight = 'bold')
        # Place text for the Low PV members.
        text = ax.text(-700000,3500000,"Low PV Members", size=13, verticalalignment='center', rotation=90., weight = 'bold')
        plt.tight_layout()
    else:
        # Go through the rest of the data and plot without the text.
        # Add cyclic point.
        lowpv_surfT_shifted, lowpv_hgt_shifted, sig_shifted, lons_shifted = addcyclic(lowpv_surfT[i], lowpv_hgt[i], percent_arr_surfT[i], lon_region)
        # Mesh the grid.
        lons, lats = np.meshgrid(lons_shifted, lat_region)
        # Mask where the T2M where the significance is >5% or <95%.
        surfT_masked = np.ma.masked_where((sig_shifted >= perc1)&(sig_shifted <= perc2), lowpv_surfT_shifted)
        # Add subplot at given position.
        ax = fig.add_subplot(nrows, ncols, fig_no[i])
        # Set up lambert conformal projection.
        map = gen_utils.LambConfMap(9000000, 8100000, 1000.0, [35, 75], [55, -100])
        # Map the lons and lats to x and y coordinates.
        x, y = map(lons, lats)
        # Contourf the T2M
        cs = map.contourf(x, y, lowpv_surfT_shifted, clevs, norm = norm, extend='both', cmap = my_cmap)
        # Contour the hgt, and add labels.
        lines = map.contour(x, y, lowpv_hgt_shifted, levels = [-30, -25, -20, -15, -10,-5, 5, 10, 15, 20, 25, 30], colors = 'black')
        plt.clabel(lines)
        # Place hatching over the significant area using the mask.
        pcol = plt.pcolor(x, y, surfT_masked, hatch = '////', alpha = 0)
        # Set the title for the plot.
        ax.set_title(f"{labels[i]} {dates_select[i].strftime('%Y/%m/%d')}", fontsize = 14, weight = 'bold')
        plt.tight_layout()

# Loop through the three dates and plot the HIGH PV hgt and T2M composites. Here i+3 for correct placement!
for i in range(len(dates_select)):
    # This if statement adds the HIGH PV label to the plot on LHS.
    if i == 0:
        # Add cyclic point to data (not necessary for this type of plot, but useful if you want to plot N Hemi instead).
        highpv_surfT_shifted, highpv_hgt_shifted, lons_shifted = addcyclic(highpv_surfT[i], highpv_hgt[i], lon_region)
        # Mesh the grid.
        lons, lats = np.meshgrid(lons_shifted, lat_region)
        # Add subplot at given position.
        ax = fig.add_subplot(nrows, ncols, fig_no[i+3])
        # Set up lambert conformal projection.
        map = gen_utils.LambConfMap(9000000, 8100000, 1000.0, [35, 75], [55, -100])
        # Map the lons and lats to x and y coordinates.
        x, y = map(lons, lats)
        # Contourf the T2M
        cs = map.contourf(x, y, highpv_surfT_shifted, clevs, norm = norm, extend='both', cmap = my_cmap)
        # Contour the hgt, and add labels.
        lines = map.contour(x, y, highpv_hgt_shifted, levels = [-30, -25, -20, -15, -10,-5, 5, 10, 15, 20, 25, 30], colors = 'black')
        plt.clabel(lines)
        # Place text for the high PV members.
        text = ax.text(-700000,3500000,"High PV Members", size=13, verticalalignment='center', rotation=90., weight = 'bold')
        # Set the title for the plot.
        ax.set_title(f"{labels[i+3]} {dates_select[i].strftime('%Y/%m/%d')}", fontsize = 14, weight = 'bold')
        plt.tight_layout()
    else:
        # Now plot the rest of the data!
        # Add cyclic point to data.
        highpv_surfT_shifted, highpv_hgt_shifted, lons_shifted = addcyclic(highpv_surfT[i], highpv_hgt[i], lon_region)
        # Mesh the grid.
        lons, lats = np.meshgrid(lons_shifted, lat_region)
        # Add subplot at given position.
        ax = fig.add_subplot(nrows, ncols, fig_no[i+3])
        # Set up lambert conformal projection.
        map = gen_utils.LambConfMap(9000000, 8100000, 1000.0, [35, 75], [55, -100])
        # Map the lons and lats to x and y coordinates.
        x, y = map(lons, lats)
        # Contourf the T2M
        cs = map.contourf(x, y, highpv_surfT_shifted, clevs, norm = norm, extend='both', cmap = my_cmap)
        # Contour the hgt, and add labels.
        lines = map.contour(x, y, highpv_hgt_shifted, levels = [-30, -25, -20, -15, -10,-5, 5, 10, 15, 20, 25, 30], colors = 'black')
        plt.clabel(lines)
        # Set the title for the plot.
        ax.set_title(f"{labels[i+3]} {dates_select[i].strftime('%Y/%m/%d')}", fontsize = 14, weight = 'bold')
        plt.tight_layout()
# Create colorbar.
# Add axes to place the colorbar.
cb_ax = fig.add_axes([0.05, -0.03, 0.91, 0.04])
# Insert the colorbar.
cbar = fig.colorbar(cs, cax=cb_ax,orientation="horizontal",ticks= np.arange(-10, 12, 2),extend="both")
# Set colorbar label.
cbar.set_label("T2M Anomaly ($^\circ$C)", fontsize = 16)
# Set colorbar tick size.
cbar.ax.tick_params(labelsize=13)
# Save the figure.
plt.savefig(f'/share/data1/Students/ollie/CAOs/project-2021-cao/main/S2S_Models/PV_Maps/Wave_Break_Reverse/ECMWF/ess_wb_ecmwf_t2m_{date_str}.png', bbox_inches = 'tight', dpi = 500)

###### COMPOSITE PV PATTERNS ######
# Composite mean of low pv maps.
pv_low = np.nanmean(pv_lead[lowest_members], axis = 0)
# Composite mean of high maps.
pv_high= np.nanmean(pv_lead[highest_members], axis = 0)
# Composite difference.
pv_diff = pv_low - pv_high

###### PV SIGNIFICANCE ######
# Set number of bootstrap resamples.
no_samples = 5000
# Set ensemble number array.
ens_ind_arr = np.arange(0, 51, 1)
# Set array to store randomly drawn bootstrap PV sample, shape (5000, lat, lon).
comp_boot_pv = np.zeros((no_samples, len(lat_region), len(lon_region)))
# Loop through no. samples.
for i in tqdm(range(no_samples)):
    # Choose random ensemble member indexes, without replacement.
    rand_inds = np.random.choice(ens_ind_arr, size = (len(lowest_members)+len(highest_members)), replace  = False) # Selects random ensemble number indexes.
    # Split the indexes and get the "low" and "high" random groups.
    split_indexes = np.split(rand_inds, [len(lowest_members)])
    boot_low_ind, boot_high_ind = split_indexes[0], split_indexes[1]
    # Composite the maps along ensemble member direction for the two groups.
    comp_boot_low, comp_boot_high = np.nanmean(pv_lead[boot_low_ind], axis = 0), np.nanmean(pv_lead[boot_high_ind], axis = 0)
    # Now take the composite difference of low-high.
    comp_boot_pv[i] = comp_boot_low - comp_boot_high

# Set array to store percentages of where your composite lies in the random distribution at each grid point.
percent_arr_pv = np.zeros((len(lat_region), len(lon_region)))
# Loop through each lat and lon to get where your composite lies in the distribution as a percentile.
for i in range(len(lat_region)):
    for j in range(len(lon_region)):
        percent_arr_pv[i, j] = percentileofscore(comp_boot_pv[:, i, j], pv_diff[i, j])


# Set colormap levels for PV difference.
clevs_pv = np.arange(-5, 5.5, 0.5)
# Set normalized colormap for PV difference.
my_cmap, norm = gen_utils.NormColorMap('bwr', clevs_pv)

###### PLOTTING PV ######
# Set figure size.
fig = plt.figure(figsize = (12, 6))
#Add subplot at correct first position for LOW PV.
ax = fig.add_subplot(1, 3, 1)
# Add cyclic point.
pv_low_shifted, lons_shifted = addcyclic(pv_low, lon_region)
# Mesh the grid.
lons, lats = np.meshgrid(lons_shifted, lat_region)
# Generate the N Hemi map.
map = gen_utils.NPStere_Map(9, -100)
# Map the lons and lats to x, y coords.
x, y = map(lons, lats)
# Contourf  the low PV.
cs = map.contourf(x, y, pv_low_shifted/1e-6, np.arange(0, 7.2, 0.2), extend='max', cmap = 'Spectral_r')
# Contour the 2 PVU (dynamic tropopause) surface.
line = map.contour(x, y, pv_low_shifted/(1e-6), [2], colors = 'black')
# Add polygon for wave break region.
p1, p2, p3, p4 = gen_utils.DrawPolygon(map, lat_range = [lat_ave1, lat_ave2], lon_range = [lon_ave1, lon_ave2], grid_space = 1.5, lw = 3, color = 'black')
# Create sequential colorbar and labels.
cbar = map.colorbar(cs, location = 'bottom', pad = "5%", ticks = np.arange(0, 8, 1))
cbar.ax.set_xlabel('PV (PVU)', fontsize = 12)
cbar.ax.tick_params(labelsize=11)
# Set title.
ax.set_title(f"a) LOW PV", fontsize = 15, weight = 'bold')
plt.tight_layout()

# Add subplot at correct second position for HIGH PV.
ax = fig.add_subplot(1, 3, 2)
# Add cyclic point.
pv_high_shifted, lons_shifted = addcyclic(pv_high, lon_region)
# Mesh the grid.
lons, lats = np.meshgrid(lons_shifted, lat_region)
# Generate N Hemi map.
map = gen_utils.NPStere_Map(9, -100)
# Map lons and lats to x,y coords.
x, y = map(lons, lats)
# Contourf high PV composite map.
cs = map.contourf(x, y, pv_high_shifted/1e-6, np.arange(0, 7.2, 0.2), extend='max', cmap = 'Spectral_r')
# Contour the 2 PVU.
line = map.contour(x, y, pv_high_shifted/(1e-6), [2], colors = 'black')
# Generate polygon for wave break region.
p1, p2, p3, p4 = gen_utils.DrawPolygon(map, lat_range = [lat_ave1, lat_ave2], lon_range = [lon_ave1, lon_ave2], grid_space = 1.5, lw = 3, color = 'black')
# Create sequential colorbar and labels.
cbar = map.colorbar(cs, location = 'bottom', pad = "5%", ticks = np.arange(0, 8, 1))
cbar.ax.set_xlabel('PV (PVU)', fontsize = 12)
cbar.ax.tick_params(labelsize=11)
# Set the title.
ax.set_title(f"b) HIGH PV", fontsize = 15, weight = 'bold')
plt.tight_layout()

# Add subplot at correct second position for DIFF PV.
ax = fig.add_subplot(1, 3, 3)
# Add cyclic point.
pv_diff_shifted, sig_shifted, lons_shifted = addcyclic(pv_diff, percent_arr_pv, lon_region)
# Mesh the grid.
lons, lats = np.meshgrid(lons_shifted, lat_region)
# Mask the field where significance >5% and <95%.
pv_masked = np.ma.masked_where((sig_shifted >= perc1)&(sig_shifted <= perc2), pv_diff_shifted)
# Generate N Hemi map.
map = map = gen_utils.NPStere_Map(9, -100)
# Map lons and lats to x,y coords.
x, y = map(lons, lats)
# Contourf diff PV composite map.
cs = map.contourf(x, y, pv_diff_shifted/1e-6, clevs_pv, norm = norm, extend='both', cmap = my_cmap)
# Hatch significance.
pcol = plt.pcolor(x, y, pv_masked, hatch = '////', alpha = 0)
# Generate polygon for wave break region.
p1, p2, p3, p4 = gen_utils.DrawPolygon(map, lat_range = [lat_ave1, lat_ave2], lon_range = [lon_ave1, lon_ave2], grid_space = 1.5, lw = 3, color = 'black')
# Create diverging colorbar and labels.
cbar = map.colorbar(cs, location = 'bottom', pad = "5%", ticks = np.arange(-5, 6, 1))
cbar.ax.set_xlabel('PV (PVU)', fontsize = 12)
cbar.ax.tick_params(labelsize=11)
# Set title.
ax.set_title(f"c) LOW - HIGH PV", fontsize = 15, weight = 'bold')
plt.tight_layout()
# Save figure.
plt.savefig(f'/share/data1/Students/ollie/CAOs/project-2021-cao/main/S2S_Models/PV_Maps/Wave_Break_Reverse/ECMWF/ess_wb_ecmwf_pv_{date_str}.png', bbox_inches = 'tight', dpi = 500)
