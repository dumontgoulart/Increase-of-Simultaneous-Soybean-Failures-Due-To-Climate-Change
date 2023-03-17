# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 17:58:11 2022

@author: morenodu
"""

import os
os.chdir('D:/paper_3/code')
import xarray as xr 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import cartopy
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib as mpl

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['figure.dpi'] = 144
mpl.rcParams.update({'font.size': 14})


# Import data
'''
here we explore the storm surge & tide levels for the Xynthia (2010) and Xaver (2013) stomrs 
'''

DS_storm_surge = xr.open_dataset("../data/surge_tides/reanalysis_waterlevel_hourly_2013_12_v1.nc")
DS_storm_surge_xaver = DS_storm_surge.sel(time = slice('2013-12-05','2013-12-10'))

DS_storm_surge_xaver['waterlevel'].sel(stations = 2).plot(x='station_x_coordinate', y='station_y_coordinate',transform=ccrs.PlateCarree(), robust=True)

rm = {'station_x_coordinate':'lon', 'station_y_coordinate':'lat'}

DS_storm_surge_xaver_test = DS_storm_surge_xaver.rename(rm)
DS_storm_surge_xaver_test['waterlevel'].isel(time = 0, stations = 1).plot(x='lon', y='lat')

## CODEC
bbox =  44.182204,-6.899414,54.188155,9.887695
bbox_hamburg =  4.987793,52.133488,13.820801,57.326521

rm = {'station_x_coordinate':'lon', 'station_y_coordinate':'lat'}

ds_gtsm0 = xr.open_dataset('../data/surge_tides/reanalysis_waterlevel_hourly_2013_12_v1.nc').rename(rm)
# Create MultiIndex coordinate
ds_gtsm0_2 = ds_gtsm0.set_index(station=["lat", "lon"]).sel(lat=slice(bbox[1],bbox[3]), lon=slice(bbox[0],bbox[2]))
# Unstack the MultiIndex
ds_gtsm0_2 = ds_gtsm0_2.unstack()



ds_gtsm0.waterlevel.isel(time=0).to_series().plot(x='lon', y='lat')



# Now try with gridded data:
ds_gtsm_gridded = xr.open_dataset('../data/surge_tides/era5-water_level-2013-m12-v0.0.nc').rename(rm)

# Create MultiIndex coordinate
stations_2 = ds_gtsm_gridded.stations.values
ds_gtsm_gridded = ds_gtsm_gridded.assign_coords(stations = ('stations',stations_2))
ds_multiindex = ds_gtsm_gridded.set_index(station=["lat", "lon"]).sel(lat=slice(bbox[1],bbox[3]), lon=slice(bbox[0],bbox[2]))
# Unstack the MultiIndex
ds_gtsm_gridded_2 = ds_multiindex.unstack()

ds_gtsm_gridded_2['waterlevel'].isel(time = 0).plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)


# construct a full grid
def expand_grid(lat,lon):
    '''list all combinations of lats and lons using expand_grid(lat,lon)'''
    test = [(A,B) for A in lat for B in lon]
    test = np.array(test)
    test_lat = test[:,0]
    test_lon = test[:,1]
    full_grid = pd.DataFrame({'lat': test_lat, 'lon': test_lon})
    full_grid = full_grid.sort_values(by=['lat','lon'])
    full_grid = full_grid.reset_index(drop=True)
    return full_grid


data_onto_full_grid = pd.merge(full_grid, your_actual_data,how='left")
                               

                               

                               
# compare two datasets:
ds_gtsm0.sel(stations = 3985) 
ds_gtsm_gridded.sel(stations = 3985) 


plt.plot(ds_gtsm0['waterlevel'].sel(stations = 3985), label = '2022_dataset')
plt.plot(ds_gtsm_gridded['waterlevel'].sel(stations = 3990), color = 'red', label = '2020_dataset')
plt.legend()
plt.show()

plt.figure(figsize = (10,10))
plt.title('Xaver')
plt.plot(ds_gtsm0['waterlevel'].sel(stations = 3985, time = slice('2013-12-05','2013-12-12')).time,ds_gtsm0['waterlevel'].sel(stations = 3985, time = slice('2013-12-05','2013-12-12')), label = '2022_dataset' )
plt.plot(ds_gtsm_gridded['waterlevel'].sel(stations = 3990, time = slice('2013-12-05','2013-12-12')).time,ds_gtsm_gridded['waterlevel'].sel(stations = 3990, time = slice('2013-12-05','2013-12-12')) , label = '2020_dataset')
plt.legend()
plt.show()

