import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset, num2date
from tqdm import tqdm
import os
from datetime import datetime, timedelta
from mpl_toolkits.basemap import Basemap, addcyclic
from matplotlib import cm, colors
path = '/share/data1/Students/ollie/CAOs/project-2021-cao/Functions'
dir = os.chdir(path)
from gen_utils import DrawPolygon, NormColorMap, NPStere_Map

region = 'nhemi'
period = 'ndjfm'
# Now read in the gph data.
directory = '/share/data1/Students/ollie/CAOs/Data/GPH/Non_Detrended'
path = os.chdir(directory)

chosen_level = 500
cold_date1 = datetime(2021, 2, 1, 0, 0)
cold_date2 = datetime(2021, 2, 18, 0, 0)

nc = Dataset(f'era5.hgt.500.{region}.{period}.all.daily.nc', 'r')
time_first = num2date(nc.variables['time'][:],nc.variables['time'].units,nc.variables['time'].calendar,only_use_cftime_datetimes=False, only_use_python_datetimes=True)
time_ind1 = np.where(time_first == cold_date1)
time_ind2 = np.where(time_first == cold_date2)

hgt = nc.variables['hgt'][time_ind1[0][0], time_ind1[1][0]:time_ind2[1][0]+1]
hgt = np.where(hgt.mask, np.nan, hgt.data)
time = time_first[time_ind1[0][0], time_ind1[1][0]:time_ind2[1][0]+1]
latitude = nc.variables['latitude'][:]
longitude = nc.variables['longitude'][:]
nc.close()

# Read in hgt ltm data.
# Choose directory where ltm data is stored.

directory = '/data/deluge/scratch/ERA5/3D/4xdaily/hgt'
path = os.chdir(directory)

# Read in 1981-2010 ltm hgt array.

ltmFile = 'hgt.day.ltm.nc'
nc = Dataset(ltmFile, 'r')
lat_1 = np.max(latitude) # Pole
lat_2 = np.min(latitude) # Equator

# Set lat, lon, level and read, starting with a file to track.
latitude_ltm = nc.variables['latitude'][:]
level_ltm = nc.variables['level'][:]

lat_index1 = np.where(latitude_ltm == lat_1)[0][0] # Pole
lat_index2 = np.where(latitude_ltm == lat_2)[0][0] # Equator
level_index = np.where(level_ltm == chosen_level)[0][0] # Choose pressure in hPa

date_ltm = num2date(nc.variables['time'][:], nc.variables['time'].units,nc.variables['time'].calendar,only_use_cftime_datetimes=False, only_use_python_datetimes=True)
months_ltm = np.array([d.month for d in date_ltm])
days_ltm = np.array([d.day for d in date_ltm])

ltm_time_ind1 = np.where((months_ltm == cold_date1.month)&(days_ltm == cold_date1.day))[0][0]
ltm_time_ind2 = np.where((months_ltm == cold_date2.month)&(days_ltm == cold_date2.day))[0][0]

hgt_ltm = nc.variables['hgt'][ltm_time_ind1:ltm_time_ind2+1, level_index, lat_index1:lat_index2+1, :]

hgt_ltm = np.where(hgt_ltm.mask, np.nan, hgt_ltm.data)

nc.close()

###### Now T2M data ######

region = 'nhemi'
period = 'ndjfm'

directory = '/share/data1/Students/ollie/CAOs/Data/Temps/T2M/Non_Detrended'
path = os.chdir(directory)

nc = Dataset(f'era5.t2m.{region}.{period}.all.daily.nc', 'r')
time_first_t2m = num2date(nc.variables['time'][:],nc.variables['time'].units,nc.variables['time'].calendar,only_use_cftime_datetimes=False, only_use_python_datetimes=True)
t2m_time_ind1 = np.where(time_first_t2m == cold_date1)
t2m_time_ind2 = np.where(time_first_t2m == cold_date2)
t2m = nc.variables['t2m'][t2m_time_ind1[0][0], t2m_time_ind1[1][0]:t2m_time_ind2[1][0]+1]
t2m = np.where(t2m.mask, np.nan, t2m.data)
nc.close()

# Choose directory where ltm data is stored.

directory = '/data/deluge/scratch/ERA5/2D/4xdaily/t2m'
path = os.chdir(directory)

# Read in 1981-2010 ltm t2m array.
ltmFile = 't2m.day.ltm.nc'

nc = Dataset(ltmFile, 'r')

date_ltm_t2m = num2date(nc.variables['time'][:], nc.variables['time'].units,nc.variables['time'].calendar,only_use_cftime_datetimes=False, only_use_python_datetimes=True)
months_ltm_t2m = np.array([d.month for d in date_ltm_t2m])
days_ltm_t2m = np.array([d.day for d in date_ltm_t2m])

t2m_ltm_time_ind1 = np.where((months_ltm_t2m == cold_date1.month)&(days_ltm_t2m == cold_date1.day))[0][0]
t2m_ltm_time_ind2 = np.where((months_ltm_t2m == cold_date2.month)&(days_ltm_t2m == cold_date2.day))[0][0]

t2m_ltm = nc.variables['t2m'][t2m_ltm_time_ind1:t2m_ltm_time_ind2+1, lat_index1:lat_index2+1, :]

t2m_ltm = np.where(t2m_ltm.mask, np.nan, t2m_ltm.data)

nc.close()

# Now find anomalies.
t2m_anom = t2m-t2m_ltm
hgt_anom = (hgt-hgt_ltm)/10

# Now define the three periods and get their indexes.
period1 = [datetime(2021, 2, 1), datetime(2021, 2, 4)]
period2 = [datetime(2021, 2, 5), datetime(2021, 2, 11)]
period3 = [datetime(2021, 2, 12), datetime(2021, 2, 18)]

periods_all = np.stack([period1, period2, period3])

ind1_period1, ind2_period1 = np.where(time == period1[0])[0][0], np.where(time == period1[1])[0][0]
ind1_period2, ind2_period2 = np.where(time == period2[0])[0][0], np.where(time == period2[1])[0][0]
ind1_period3, ind2_period3 = np.where(time == period3[0])[0][0], np.where(time == period3[1])[0][0]

# Now split data into the three periods and take the mean across time.
hgt_period1, t2m_period1 = np.nanmean(hgt_anom[ind1_period1:ind2_period1+1], axis = 0), np.nanmean(t2m_anom[ind1_period1:ind2_period1+1], axis = 0)
hgt_period2, t2m_period2 = np.nanmean(hgt_anom[ind1_period2:ind2_period2+1], axis = 0), np.nanmean(t2m_anom[ind1_period2:ind2_period2+1], axis = 0)
hgt_period3, t2m_period3 = np.nanmean(hgt_anom[ind1_period3:ind2_period3+1], axis = 0), np.nanmean(t2m_anom[ind1_period3:ind2_period3+1], axis = 0)

hgt_all = np.stack([hgt_period1, hgt_period2, hgt_period3])
t2m_all = np.stack([t2m_period1, t2m_period2, t2m_period3])

##### Set custom normalized colormap for t2m anomalies. #####
clevs = np.arange(-20, 21, 1)
my_cmap, norm = NormColorMap('RdBu_r', clevs)

# Now loop through and plot!
# Set rows and columns
nrows, ncols = 1, 3
fig_no = np.arange(1, 4, 1)
labels = ['a)', 'b)', 'c)']
fig = plt.figure(figsize = (15,5.5))
for i in range(3):
    ##### Prepare data for plotting #####
    hgts_shifted, t2m_shifted, lons_shifted = addcyclic(hgt_all[i], t2m_all[i], longitude)
    lons, lats = np.meshgrid(lons_shifted, latitude)
    ax = fig.add_subplot(nrows, ncols, fig_no[i])

    # Create map
    map = NPStere_Map()
    x,y = map(lons, lats)

    # Contourf.
    cs = map.contourf(x, y, t2m_shifted, clevs, norm=norm, extend='both', cmap = my_cmap)
    lines2 = map.contour(x, y, hgts_shifted, levels = [-30, -25, -20, -15, -10,-5, 5, 10, 15, 20, 25, 30], colors = 'black')
    plt.clabel(lines2)
    ax.set_title(f"{labels[i]} {periods_all[i][0].strftime('%Y-%m-%d')} - {periods_all[i][1].strftime('%Y-%m-%d')}", fontsize = 14, weight = 'bold')
    plt.tight_layout()

# Add colorbar.
cb_ax = fig.add_axes([0.05, -0.02, 0.91, 0.04])
cbar = fig.colorbar(cs, cax=cb_ax,orientation="horizontal",ticks= np.arange(-20, 25, 5),extend="both")
cbar.set_label("T2M Anomaly ($^\circ$C)", fontsize = 16)
cbar.ax.tick_params(labelsize=13)
plt.savefig("/share/data1/Students/ollie/CAOs/project-2021-cao/main/Reanalysis_Plots/Multiple_Variables/ERA5/GPH_T2M/gph_t2m_2021.png", bbox_inches = 'tight', dpi = 500)
