import numpy as np
from netCDF4 import Dataset, num2date
from datetime import datetime, timedelta
import os
from mpl_toolkits.basemap import Basemap, addcyclic
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import glob
from tqdm import tqdm
from scipy.stats import mode, percentileofscore
dir = '/share/data1/Students/ollie/Functions'
path = os.chdir(dir)
from gen_utils import DrawPolygon, LambConfMap, NormColorMap, NPStere_Map
from model_utils import load_latlon, load_modelparam

# Go to directory with the data in.
path = '/share/data1/Students/ollie/CAOs/Data/Feb_2021_CAO/ERA5/theta'
dir = os.chdir(path)

# Now import the data for Jan and Feb 2021.
# Jan first.
nc = Dataset('theta_2PVU.202101.nc', 'r')
time_jan = num2date(nc.variables['time'][:], nc.variables['time'].units, nc.variables['time'].calendar, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
latitude = nc.variables['latitude'][:]
longitude = nc.variables['longitude'][:]

# Find indexes for latitude you want.
lat1, lat2 = 90, 0
lat_ind1, lat_ind2 = np.where(latitude == lat1)[0][0], np.where(latitude == lat2)[0][0]

# Now import pt data.
pt_jan = nc.variables['pt'][:, lat_ind1:lat_ind2+1, :]
nc.close()

# Now import Feb.
nc = Dataset('theta_2PVU.202102.nc', 'r')
time_feb = num2date(nc.variables['time'][:], nc.variables['time'].units, nc.variables['time'].calendar, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
pt_feb = nc.variables['pt'][:, lat_ind1:lat_ind2+1, :]
nc.close()

# Set longitude and latitude regional arrays.
lat_region, lon_region = latitude[lat_ind1:lat_ind2+1], longitude

# Now concatenate them along time dimension.
all_time = np.concatenate((time_jan, time_feb))
all_pt = np.concatenate((pt_jan, pt_feb), axis = 0)

# Now reshape the arrays and isolate to daily averaged data.
time_daily = all_time[::4]

pt_reshape = all_pt.reshape(int(all_pt.shape[0]/4), 4, all_pt.shape[1], all_pt.shape[2])
pt_daily = np.nanmean(pt_reshape, axis = 1)

# Now select date ranges to plot through.
date1, date2, date3, date4 = datetime(2021, 2, 2), datetime(2021, 2, 4), datetime(2021, 2, 9), datetime(2021, 2, 11)
time_ind1, time_ind2, time_ind3, time_ind4 = np.where(time_daily == date1)[0][0], np.where(time_daily == date2)[0][0], np.where(time_daily == date3)[0][0], np.where(time_daily == date4)[0][0]

pt_period1 = pt_daily[time_ind1:time_ind2+1]
pt_period2 = pt_daily[time_ind3:time_ind4+1]
time_period1 = time_daily[time_ind1:time_ind2+1]
time_period2 = time_daily[time_ind3:time_ind4+1]

# Now loop through and plot.

# Make figure.
# Fig dimensions:
nrows = 2
ncols = 3
fig_no = np.arange(1, 9, 1)
labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']
fig = plt.figure(figsize=(12, 9))
for i in range(len(time_period1)):
    if i == 0:
        pt_shifted, lons_shifted = addcyclic(pt_period1[i], lon_region)
        lons, lats = np.meshgrid(lons_shifted, lat_region)

        ax = fig.add_subplot(nrows, ncols, fig_no[i])
        # Basemap plot.
        map = LambConfMap(width = 8000000, height = 7100000, area_thresh = 1000.0, lat_range = [89.5, 60], center_range = [65.0, 200])

        x, y = map(lons, lats)

        # Contourf.
        cs = map.contourf(x, y, pt_shifted, np.arange(270, 352, 2), extend='max', cmap = 'Spectral_r')
        p1, p2, p3, p4 = DrawPolygon(map, lat_range = [85.5, 66], lon_range = [111, 187.5], grid_space = 0.5, lw = 3, color = 'black')
        #text = ax.text(-1500000,9000000,"Wave Break 1", size=15, verticalalignment='center', rotation=90., weight = 'bold')
        ax.set_title(f"{labels[i]} {time_period1[i].strftime('%Y/%m/%d')}", fontsize = 16, weight = 'bold')
        plt.tight_layout()
    else:
        pt_shifted, lons_shifted = addcyclic(pt_period1[i], lon_region)
        lons, lats = np.meshgrid(lons_shifted, lat_region)

        ax = fig.add_subplot(nrows, ncols, fig_no[i])
        # Basemap plot.
        map = LambConfMap(width = 8000000, height = 7100000, area_thresh = 1000.0, lat_range = [89.5, 60], center_range = [65.0, 200])

        x, y = map(lons, lats)

        # Contourf.
        cs = map.contourf(x, y, pt_shifted, np.arange(270, 352, 2), extend='max', cmap = 'Spectral_r')
        p1, p2, p3, p4 = DrawPolygon(map, lat_range = [85.5, 66], lon_range = [111, 187.5], grid_space = 0.5, lw = 3, color = 'black')
        ax.set_title(f"{labels[i]} {time_period1[i].strftime('%Y/%m/%d')}", fontsize = 16, weight = 'bold')
        plt.tight_layout()

for i in range(len(time_period2)):
    if i == 0:
        pt_shifted, lons_shifted = addcyclic(pt_period2[i], lon_region)
        lons, lats = np.meshgrid(lons_shifted, lat_region)

        ax = fig.add_subplot(nrows, ncols, fig_no[i+3])
        # Basemap plot.
        map = LambConfMap(width = 7000000, height = 6100000, area_thresh = 1000.0, lat_range = [70, 40], center_range = [55.0, 310])

        x, y = map(lons, lats)

        # Contourf.
        cs = map.contourf(x, y, pt_shifted, np.arange(270, 352, 2), extend='max', cmap = 'Spectral_r')
        p1, p2, p3, p4 = DrawPolygon(map, lat_range = [60, 48], lon_range = [291, 324], grid_space = 0.5, lw = 3, color = 'black')
        ax.set_title(f"{labels[i+3]} {time_period2[i].strftime('%Y/%m/%d')}", fontsize = 16, weight = 'bold')
        #text = ax.text(-1500000,9000000,"Wave Break 2", size=15, verticalalignment='center', rotation=90., weight = 'bold')
        plt.tight_layout()
    else:
        pt_shifted, lons_shifted = addcyclic(pt_period2[i], lon_region)
        lons, lats = np.meshgrid(lons_shifted, lat_region)

        ax = fig.add_subplot(nrows, ncols, fig_no[i+3])
        # Basemap plot.
        map = LambConfMap(width = 7000000, height = 6100000, area_thresh = 1000.0, lat_range = [70, 40], center_range = [55.0, 310])

        x, y = map(lons, lats)

        # Contourf.
        cs = map.contourf(x, y, pt_shifted, np.arange(270, 352, 2), extend='max', cmap = 'Spectral_r')
        p1, p2, p3, p4 = DrawPolygon(map, lat_range = [60, 48], lon_range = [291, 324], grid_space = 0.5, lw = 3, color = 'black')
        ax.set_title(f"{labels[i+3]} {time_period2[i].strftime('%Y/%m/%d')}", fontsize = 16, weight = 'bold')
        plt.tight_layout()

cb_ax = fig.add_axes([0.05, -0.01, 0.91, 0.04])
cbar = fig.colorbar(cs, cax=cb_ax,orientation="horizontal",ticks= np.arange(270, 360, 10),extend="max",spacing='proportional')
cbar.set_label("Theta (C)", fontsize = 17)
cbar.ax.tick_params(labelsize=15)
plt.savefig("/share/data1/Students/ollie/CAOs/project-2021-cao/main/Reanalysis_Plots/Theta_2PVU/ERA5/pt_2021.png", bbox_inches = 'tight', dpi = 500)
