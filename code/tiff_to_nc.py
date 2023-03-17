# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 16:05:19 2021

@author: morenodu
"""
import os
os.chdir('C:/Users/morenodu/OneDrive - Stichting Deltares/Documents/PhD/paper_hybrid_agri/data')
import xarray as xr 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import cartopy
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from  scipy import signal 
import seaborn as sns
import matplotlib as mpl
from osgeo import gdal
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

#Change the following variables to the file you want to convert (inputfile) and
#what you want to name your output file (outputfile).
inputfile = 'spam2010v1r0_global_harvested-area_soyb_r.tif'
outputfile = 'spam2010v1r0_global_harvested-area_soyb_r.nc'
#Do not change this line, the following command will convert the geoTIFF to a netCDF
ds = gdal.Translate(outputfile, inputfile, format='NetCDF')

DS_y_obs = xr.open_dataset("spam2010v1r0_global_harvested-area_soyb_r.nc", decode_times=True)

def plot_2d_map(dataarray_2d):
    # Plot 2D map of DataArray, remember to average along time or select one temporal interval
    plt.figure(figsize=(12,5)) #plot clusters
    ax=plt.axes(projection=ccrs.Mercator())
    dataarray_2d.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
    # ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
    ax.set_extent([-80.73,-34,-35,6], ccrs.PlateCarree())
    plt.show()
    
plot_2d_map(DS_y_obs.Band1)



DS_y_epic = xr.open_dataset("epic-iiasa_gswp3-w5e5_obsclim_2015soc_default_yield-soy-noirr_global_annual_1901_2016.nc", decode_times=False)
DS_y_epic_mean = DS_y_epic.mean('time')  
DS_y_epic_mean.to_netcdf("epic_grid_05x05.nc")

#%% TRANSFORM HARVEST AREA

# Load Geotiff
os.chdir('C:/Users/morenodu/OneDrive - Stichting Deltares/Documents/PhD/paper_hybrid_agri/data')

# US
rio = xr.open_rasterio("soy_harvest_area_us_1980_2016_05x05_density_02.tif")
rio = rio.rename({'y': 'lat', 'x': 'lon', 'band':'time'})
rio['time'] = list(map(int, rio.attrs['descriptions']))
rio.name = 'harvest_area'
# rio = rio/1000

ds_rio = rio.to_dataset()
ds_rio['time'] = pd.date_range(start='1980', periods=ds_rio.sizes['time'], freq='YS').year
# ds_rio = ds_rio.mean('time')
ds_rio['time'].attrs = {'units':'years since 1980-01-01'}
ds_rio['lat'].attrs = {'units':'degrees_north'}
ds_rio['lon'].attrs = {'units':'degrees_east'}
# ds_rio = ds_rio.where(ds_rio['harvest_area'] > -1000)

# ds_rio = ds_rio.mean('time')
# ds_rio_2 = ds_rio.sel(time = slice('1980', '2016'))
ds_rio['harvest_area'].sum(['lat','lon']).plot(label = 'US')
ds_rio['harvest_area'].sum(['lat','lon']).sel(time = 2010)

ds_rio.to_netcdf("soy_harvest_area_US_all_1980_2020_05x05_density.nc")

plot_2d_am_map(ds_rio['harvest_area'].mean('time'))

df_rio = rio.to_dataframe().dropna()
df_rio_mean = df_rio.groupby('time').mean(...)

####
# ARG
rio_2 = xr.open_rasterio("soy_harvest_area_arg_1978_2000_05x05_density_02.tif")
rio_2 = rio_2.rename({'y': 'lat', 'x': 'lon', 'band':'time'})
rio['time'] = list(map(int, rio.attrs['descriptions']))
rio_2.name = 'harvest_area'
# rio = rio/1000

ds_rio = rio_2.to_dataset()
ds_rio['time'] = pd.date_range(start='1978', periods=ds_rio.sizes['time'], freq='YS').year
ds_rio['time'].attrs = {'units':'years since 1978-01-01'}
# ds_rio = ds_rio.mean('time')
ds_rio['lat'].attrs = {'units':'degrees_north'}
ds_rio['lon'].attrs = {'units':'degrees_east'}

# ds_rio = ds_rio.mean('time')
# ds_rio_2 = ds_rio.sel(time = slice('1980', '2016'))
ds_rio.to_netcdf("soy_harvest_area_arg_1978_2019_05x05_density.nc")

plot_2d_am_map(ds_rio['harvest_area'].sel(time=1990))

df_rio = rio.to_dataframe().dropna()
df_rio_mean = df_rio.groupby('time').mean(...)


# Brazil
rio_2 = xr.open_rasterio("soy_harvest_area_br_1980_2016_05x05_density_03.tif")
rio_2 = rio_2.rename({'y': 'lat', 'x': 'lon', 'band':'time'})
rio_2['time'] = list(map(int, rio_2.attrs['descriptions']))
rio_2.name = 'harvest_area'
# rio = rio/1000

ds_rio = rio_2.to_dataset()
ds_rio['time'] = pd.date_range(start='1980', periods=ds_rio.sizes['time'], freq='YS').year
# ds_rio = ds_rio.mean('time')
ds_rio['lat'].attrs = {'units':'degrees_north'}
ds_rio['lon'].attrs = {'units':'degrees_east'}
ds_rio['time'].attrs = {'units':'years since 1980-01-01'}

# ds_rio = ds_rio.mean('time')
# ds_rio_2 = ds_rio.sel(time = slice('1980', '2016'))
ds_rio.to_netcdf("soy_harvest_area_br_1980_2016_05x05_density.nc")

plot_2d_am_map(ds_rio['harvest_area'].sel(time=2016))

df_rio = rio.to_dataframe().dropna()
df_rio_mean = df_rio.groupby('time').mean(...)

# SPAM

rio_2 = xr.open_rasterio("soy_harvest_spam_agg_resamp.tif")
rio_2 = rio_2.rename({'y': 'lat', 'x': 'lon', 'band':'time'})

rio_2.name = 'harvest_area'
rio_2 = rio_2.mean('time')
ds_rio = rio_2.to_dataset()
plot_2d_am_map(ds_rio['harvest_area'].where(ds_rio['harvest_area']>0))

ds_rio.to_netcdf("soy_harvest_spam_agg_resamp.nc")

# SPAM 2
rio_spam = xr.open_rasterio("soy_harvest_spam_native.tif")
rio_spam = rio_spam.rename({'y': 'lat', 'x': 'lon', 'band':'time'})

rio_spam.name = 'harvest_area'
rio_spam = rio_spam.mean('time')
ds_rio_spam = rio_spam.to_dataset()
ds_rio_spam['lat'].attrs = {'units':'degrees_north'}
ds_rio_spam['lon'].attrs = {'units':'degrees_east'}
plot_2d_am_map(ds_rio_spam['harvest_area'].where(ds_rio_spam['harvest_area']>10))

ds_rio_spam.to_netcdf("soy_harvest_spam_native.nc")

#%% TRANSFORM YIELD

# Load Geotiff
os.chdir('C:/Users/morenodu/OneDrive - Stichting Deltares/Documents/PhD/paper_hybrid_agri/data')

rio = xr.open_rasterio("soy_yields_US_all_1975_2020_1prc_002.tif")
rio = rio.rename({'y': 'lat', 'x': 'lon', 'band':'time'})
# rio['time'] = list(map(int, rio.attrs['descriptions']))
rio.name = 'Yield'
# rio = rio/1000

ds_rio = rio.to_dataset()
# ds_rio = ds_rio.where(ds_rio['harvest_area'] > -1000)
ds_rio['time'] = pd.date_range(start='1975', periods=ds_rio.sizes['time'], freq='YS').year

ds_rio['time'].attrs = {'units':'years since 1975-01-01'}
ds_rio['lat'].attrs = {'units':'degrees_north'}
ds_rio['lon'].attrs = {'units':'degrees_east'}
ds_rio = ds_rio.sortby('lat')
ds_rio = ds_rio.sortby('lon')
# ds_rio = ds_rio.mean('time')
# ds_rio_2 = ds_rio.sel(time = slice('1980', '2016'))
ds_rio.to_netcdf("soy_yields_US_all_1975_2020_1prc_002.nc")

plot_2d_am_map(ds_rio['Yield'].sel(time = 1987))
plot_2d_am_map(ds_rio['Yield'].sel(time = 1990))
plot_2d_am_map(ds_rio['Yield'].mean('time'))


# ARGE
rio_arg = xr.open_rasterio("soy_yield_arg_1978_2019_005.tif")
rio_arg = rio_arg.rename({'y': 'lat', 'x': 'lon', 'band':'time'})
rio_arg['time'] = list(map(int, rio_arg.attrs['descriptions']))
rio_arg.name = 'Yield'

ds_rio_arg = rio_arg.to_dataset()
# ds_rio = ds_rio.where(ds_rio['harvest_area'] > -1000)
ds_rio_arg = ds_rio_arg.sel(lat = slice(-15,-55), lon = slice(-90,-40)) # SLICE IT TO BECOME FASTER TO CHANGE RESOLUTION LATER ON CDO
ds_rio_arg['Yield'] = ds_rio_arg['Yield']/1000

ds_rio_arg['lat'].attrs = {'units':'degrees_north'}
ds_rio_arg['lon'].attrs = {'units':'degrees_east'}
ds_rio_arg['time'].attrs = {'units':f'years since {rio_arg.time[0]-1}-01-01'}
ds_rio_arg = ds_rio_arg.sortby('lat')
ds_rio_arg = ds_rio_arg.sortby('lon')
# ds_rio = ds_rio.mean('time')
# ds_rio_2 = ds_rio.sel(time = slice('1980', '2016'))
ds_rio_arg.to_netcdf("soy_yield_arg_1974_2019_005.nc")

plot_2d_am_map(ds_rio_arg['Yield'].mean('time'))



# BRASIL
rio_arg = xr.open_rasterio("soy_yield_1975_2016_005_1prc.tif")
rio_arg = rio_arg.rename({'y': 'lat', 'x': 'lon', 'band':'time'})
rio_arg['time'] = list(map(int, rio_arg.attrs['descriptions']))
rio_arg.name = 'Yield'

ds_rio_arg = rio_arg.to_dataset()
ds_rio_arg = ds_rio_arg.sel(lat = slice(0,-35), lon = slice(-73.98,-34.72)) # SLICE IT TO BECOME FASTER TO CHANGE RESOLUTION LATER ON CDO
# ds_rio = ds_rio.where(ds_rio['harvest_area'] > -1000)

ds_rio_arg['lat'].attrs = {'units':'degrees_north'}
ds_rio_arg['lon'].attrs = {'units':'degrees_east'}
ds_rio_arg['time'].attrs = {'units':f'years since {rio_arg.time[0]-1}-01-01'}
ds_rio_arg = ds_rio_arg.sortby('lat')
ds_rio_arg = ds_rio_arg.sortby('lon')
# ds_rio = ds_rio.mean('time')
# ds_rio_2 = ds_rio.sel(time = slice('1980', '2016'))
ds_rio_arg.to_netcdf("soy_yield_1975_2016_005_1prc.nc")

plot_2d_am_map(ds_rio_arg['Yield'].mean('time'))
plot_2d_am_map(ds_rio_arg['Yield'].sel(time = 1979))

#%% TRANSFORM usda

# Load Geotiff
os.chdir('C:/Users/morenodu/OneDrive - Stichting Deltares/Documents/PhD/Paper_drought/data')

rio = xr.open_rasterio("soy_yields_US_all_1980_2020_05x05.tif")
rio = rio.rename({'y': 'lat', 'x': 'lon', 'band':'time'})
rio['time'] = list(map(int, rio.attrs['descriptions']))
rio.name = 'usda_yield'

ds_rio = rio.to_dataset()
# ds_rio = ds_rio.where(ds_rio['harvest_area'] > -1000)
ds_rio['time'].attrs = {'units':'years since 1980-01-01'}
ds_rio['lat'].attrs = {'units':'degrees_north'}
ds_rio['lon'].attrs = {'units':'degrees_east'}
# ds_rio = ds_rio.mean('time')
# ds_rio_2 = ds_rio.sel(time = slice('1980', '2016'))
ds_rio.to_netcdf("soy_yields_US_all_1980_2020_05x05_02.nc")

plot_2d_us_map(ds_rio['usda_yield'].mean('time'))

df_rio = rio.to_dataframe().dropna()
df_rio_mean = df_rio.groupby('time').mean(...)

#%%

# TEST

# cal_test = pd.read_csv('calendar_soybeans/mirca/CELL_SPECIFIC_CROPPING_CALENDARS_30MN.txt', sep = "\t")
# cal_test_sub = cal_test[(cal_test['crop'] == 34) & (cal_test['subcrop'] == 1)]
# cal_test_sub = cal_test_sub[['lat', 'lon', 'start', 'end']]
# cal_test_sub.set_index(['lat', 'lon'], inplace=True)

# DS_cal_test_sub = cal_test_sub.to_xarray()
# DS_cal_test_sub.to_netcdf('mirca2000_soy_calendar.nc')

# # Check for duplicates
# cal_test_sub[cal_test_sub.index.duplicated()]
# cal_test.query('lat==-39.25 & lon==-63.25 & crop==34')

# plot_2d_us_map(DS_cal_test_sub['start'])
# plot_2d_us_map(DS_cal_test_sub['end'])

#%% Load Geotiff
os.chdir('C:/Users/morenodu/OneDrive - Stichting Deltares/Documents/PhD/paper_hybrid_agri/data')
# TEST
DS_y_planting_flach = xr.open_dataset("soy_planting_area_1981_2019.nc", decode_times=True).sel(time = slice('1980', '2016'))
DS_y_harvested_flach = xr.open_dataset("soy_harvest_area_1981_2019.nc", decode_times=True).sel(time = slice('1980', '2016'))
# plot_2d_map(DS_y_planting_flach.Planted_area.mean('time'))
# plot_2d_map(DS_y_harvested_flach.Harvested_area.mean('time'))

test = DS_y_planting_flach.Planted_area.sel(lon=slice(-55.12,-55.1), lat=slice(-13.63,-13.65), time=slice(1980,2014))
test_b = test.sel(time=slice(2013,2014))
test_c = test.sel(lon=slice(-55.11,-55.1), lat=slice(-13.64,-13.65))
# Create a mock test where two cells are np.nan
counter_test = test_b.copy()
counter_test.values[0,0] = np.nan

xr.where( (test_b > 0 ) & (np.isnan(counter_test) == True ), 1, 0 )

test_2=DS_y_planting_flach.Planted_area.sel(lon=slice(-74.25, -74.23), lat=slice(6.246, 6.237), time=slice(2014,2014))


def earth_radius(lat):
    '''
    calculate radius of Earth assuming oblate spheroid defined by WGS84
    
    Input
    ---------
    lat: vector or latitudes in degrees  
    
    Output
    ----------
    r: vector of radius in meters
    
    Notes
    -----------
    WGS84: https://earth-info.nga.mil/GandG/publications/tr8350.2/tr8350.2-a/Chapter%203.pdf
    '''
    from numpy import deg2rad, sin, cos

    # define oblate spheroid from WGS84
    a = 6378137
    b = 6356752.3142
    e2 = 1 - (b**2/a**2)
    
    # convert from geodecic to geocentric
    # see equation 3-110 in WGS84
    lat = deg2rad(lat)
    lat_gc = np.arctan( (1-e2)*np.tan(lat) )

    # radius equation
    # see equation 3-107 in WGS84
    r = ( (a * (1 - e2)**0.5) / (1 - (e2 * np.cos(lat_gc)**2))**0.5 )

    return r


def area_grid(lat, lon):
    """
    Calculate the area of each grid cell
    Area is in square meters
    
    Input
    -----------
    lat: vector of latitude in degrees
    lon: vector of longitude in degrees
    
    Output
    -----------
    area: grid-cell area in square-meters with dimensions, [lat,lon]
    
    Notes
    -----------
    Based on the function in
    https://github.com/chadagreene/CDT/blob/master/cdt/cdtarea.m
    """
    from numpy import meshgrid, deg2rad, gradient, cos
    from xarray import DataArray

    xlon, ylat = meshgrid(lon, lat)
    R = earth_radius(ylat)

    dlat = deg2rad(gradient(ylat, axis=0))
    dlon = deg2rad(gradient(xlon, axis=1))

    dy = abs(dlat) * R
    dx = dlon * R * cos(deg2rad(ylat))

    area = dy * dx

    xda = DataArray(
        area,
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
        attrs={
            "long_name": "area_per_pixel",
            "description": "area per pixel",
            "units": "m^2",
        },
    )
    return xda


def fraction_mask(dataArray, percentage_limit):
    
    da_area = area_grid(dataArray['lat'], dataArray['lon'])
    time_da = dataArray.time
    
    # Expand the mask to the entire time period
    da_area_time = da_area.expand_dims(time=time_da)
    
    #Check if all values along time are equal -> True expected
    print(np.all(np.diff(da_area_time, axis=0) == 0))
    
    # Final mask with the fraction of harvested area by total grid area
    area_frac = dataArray/da_area_time
    
    # Final mask selecting only grid cells above 5% of harvested area
    da_clip = dataArray.where(area_frac > percentage_limit)
    return da_clip


da_area = area_grid(DS_y_harvested_flach.Harvested_area['lat'], DS_y_harvested_flach.Harvested_area['lon'])
time_da = DS_y_harvested_flach.Harvested_area.time

# Expand the mask to the entire time period
da_area_time = da_area.expand_dims(time=time_da)

#Check if all values along time are equal -> True expected
print(np.all(np.diff(da_area_time, axis=0) == 0))

# Final mask with the fraction of harvested area by total grid area
area_frac = DS_y_harvested_flach.Harvested_area/da_area_time

area_frac_perc = area_frac * 100
area_frac_perc_nozero = area_frac_perc.where(area_frac_perc > 0)
area_frac_perc_nozero.plot.hist( bins = 100)
# sns.histplot(data=area_frac_perc)

area_frac_perc_nozero_count_0 = xr.where(area_frac_perc_nozero > 0, 1, 0)
area_frac_perc_nozero_count_1 = xr.where(area_frac_perc_nozero > 1, 1, 0)

#%% Different approaches for masks

DS_y_harvested_flach_clip = fraction_mask(DS_y_harvested_flach.Harvested_area, 0.01)
DS_y_planting_flach_clip = fraction_mask(DS_y_planting_flach.Planted_area, 0.01)

test_3 = DS_y_obs['Yield'].where((DS_y_harvested_flach_clip > 0) )

test_3.to_netcdf('soy_ibge_yield_1981_2019_1percent.nc')

plot_2d_map(DS_y_harvested_flach_clip.mean('time'))
plot_2d_map(DS_y_planting_flach_clip.mean('time'))


DS_plant_count = xr.where(DS_y_planting_flach['Planted_area'] > 0 , 1, 0)
DS_harvest_count = xr.where(DS_y_harvested_flach['Harvested_area'] > 0 , 1, 0)
DS_yield_count = xr.where(DS_y_obs['Yield'] > 0 , 1, 0)

# If planted is > 0 and harvest is 0, then area = 0
plant_harvest_zeros = xr.where( (DS_plant_count == 1) & (DS_harvest_count==0) , 1, 0 )

# IF planted is 0 and harvest is > 0, raise error
plant_harvest_error = xr.where( (DS_plant_count == 0) & (DS_harvest_count==1) , 1, 0 )


DS_yield_plant_clip = DS_y_obs['Yield'].where( DS_plant_count == 1 )
DS_yield_harvest_clip = DS_y_obs['Yield'].where( DS_harvest_count == 1)

DS_yield_error_harv = xr.where( (DS_y_obs['Yield'] > 0) & (DS_harvest_count==0), 1, 0) # This should be zero
DS_yield_error_plant = xr.where( (DS_y_obs['Yield'] > 0) & (DS_plant_count==0), 1, 0) # This should be zero too

DS_combo = xr.where( (DS_yield_plant_clip > 0) & (DS_yield_harvest_clip > 0), 1, 0)


DS_yield_clip_plant_harv = DS_y_obs['Yield'].where( DS_combo == 1 )


plot_2d_map(plant_harvest_zeros.mean('time'))
plot_2d_map(plant_harvest_error.mean('time'))
plot_2d_map(DS_yield_clip_plant_harv.mean('time'))


plot_2d_map(DS_y_obs['Yield'].mean('time'))
plot_2d_map(DS_y_obs['Yield'].sel(time=2016))

# Testing for observed data following 1% cliping
test_3 = DS_y_obs['Yield'].where((DS_y_harvested_flach_clip > 0) )

plot_2d_map(test_3.mean('time'))

# Consider 0 cases where harvest is NA and planting area exists
test_4 = xr.where( plant_harvest_zeros == 1, 0, DS_y_obs['Yield'])

plot_2d_map(test_4.mean('time'))

# test_4.to_netcdf('soy_ibge_yield_1981_2019_clip.nc')


#TODO: CHECK THIS
# Select only grid cells without NAs along time axis:
test_5 = xr.where( np.isnan(DS_y_obs).any(dim = 'time') == False, DS_y_obs, np.nan )
# test_5.drop('spatial_ref')
# test_5 = test_4.where(np.isnan(test_4).any() == False)
plot_2d_map(test_5['Yield'].mean('time'))

# Select only grid cells with at least 10 years of data:
test_6 = xr.where( (np.isnan(DS_y_obs['Yield']) == False ).sum(dim='time') >= 5, DS_y_obs, np.nan )
plot_2d_map(test_6['Yield'].mean('time'))
if len(test_6.coords) >3 :
    test_6=test_6.drop('spatial_ref')
test_6.to_netcdf('soy_ibge_yield_1981_2019_5.nc')

# Final filter method

# 1) Select only grid cells with > 1% Harvest area fractions
test_filter_1 = xr.where( ds_rio_2['harvested_area'] >= 0, DS_y_obs['Yield'], np.nan )

# 2) Planting area > 0 & Harvest area is NA: Yield = 0
test_filter_2 = xr.where( plant_harvest_zeros == 1, 0, DS_y_obs['Yield'])

# 3) Remove grid cells with values repeating more than 3 times
# test_filter_3 = TODO

# 4) Select only grid cells with minimum of 5 years per grid cell
test_filter_4 = xr.where( (np.isnan(DS_y_obs['Yield']) == False ).sum(dim='time') >= 5, DS_y_obs['Yield'], np.nan )

plot_2d_map(DS_y_obs['Yield'].mean('time'))
plot_2d_map(test_filter_2.mean('time'))
plot_2d_map(plant_harvest_zeros.mean('time'))
test_filter_2 = test_filter_2.drop('spatial_ref')
test_filter_2.to_netcdf('soy_yield_1981_2019_1prc_harvplan.nc')

#%%
# hectares
DSharvest_area_spam = xr.open_dataset("spam2010v1r0_global_harvested-area_soyb_r.nc", decode_times=True)

DS_yield_error_harv = xr.where( (DS_y_obs['Yield'] > 0) & (DS_harvest_count==0), 1, 0) # This should be zero
DS_yield_error_harv = xr.where( (DS_yield_count == 0) & (DS_harvest_count > 0), 1, 0) # This should be zero


da_area = area_grid(DSharvest_area_spam.Band1['lat'], DSharvest_area_spam.Band1['lon'])

#Check if all values along time are equal -> True expected
print(np.all(np.diff(da_area, axis=0) == 0))

# Final mask with the fraction of harvested area by total grid area
area_frac = DSharvest_area_spam.Band1/da_area * 10000

area_frac_perc = area_frac * 100
area_frac_perc_nozero = area_frac_perc.where(area_frac_perc > 0)
area_frac_perc_nozero.plot.hist( bins = 100)
sns.histplot(data=area_frac_perc)

plot_2d_map(DSharvest_area_spam.Band1)



# (Yield per grid * Harvested area) / sum( harvested _ area)
# This should be equal to mean(yield per grid)
total_area = DS_y_harvested_flach['Harvested_area'].sum(("lon", "lat"))
yield_weighted = (DS_y_obs['Yield'] * DS_y_harvested_flach['Harvested_area']) / total_area

# Check if it matches
weights = DS_y_harvested_flach['Harvested_area'] / total_area
yield_weighted_2 = weights * DS_y_obs['Yield']
yield_weighted_mean_2 = yield_weighted_2.sum(("lon", "lat"))  # Should be identical to #1


yield_weighted_mean = yield_weighted.sum(("lon", "lat"))
yield_mean = DS_y_obs['Yield'].mean(("lon", "lat"))

yield_weighted_mean.plot(label = 'weighted')
yield_mean.plot(label = 'non-weighted') 
plt.legend()
yield_weighted_mean.name = 'Yield'
if len(yield_weighted_mean.coords) >1 :
    yield_weighted_mean=yield_weighted_mean.drop('spatial_ref')
df_yield_weighted_mean = yield_weighted_mean.to_dataframe()

# removal with a 2nd order based on the CO2 levels
coeff = np.polyfit(df_co2.values.ravel(), df_yield_weighted_mean.values, 1)
trend = np.polyval(coeff, df_co2.values.ravel())
df_yield_weighted_mean_det =  pd.DataFrame( df_yield_weighted_mean['Yield'] - trend, index = df_yield_weighted_mean.index, columns = df_yield_weighted_mean.columns) + df_yield_weighted_mean.mean() 


#####
test_stack = DS_y_harvested_flach['Harvested_area'][:1,:2,:2].copy()
test_stack.values = [[[10, 10],[10, 10]]]
#####


#%% GEnerate tiff or netcdf files from the vectors we have for each area
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box, mapping
import json
   
    
from geocube.api.core import make_geocube

input_geopackage = gpd.read_file('soy_harvest_area_arg_1978_2019.gpkg')


out_grid = make_geocube(
    vector_data=input_geopackage,measurements=["anio","superficie_cosechada_ha"],datetime_measurements ='anio',
    group_by="anio",geom=json.dumps(mapping(box(-180, -90, 180, 90))), resolution=(-0.5, 0.5),
)

out_grid["column_name"].rio.to_raster("my_rasterized_column.tif")

out_grid = out_grid.rename({'y': 'lat', 'x': 'lon', 'anio':'time'})
out_grid['time'].attrs = {'units':'years since 1978-01-01'}
out_grid['lat'].attrs = {'units':'degrees_north'}
out_grid['lon'].attrs = {'units':'degrees_east'}


plot_2d_am_map(out_grid['superficie_cosechada_ha'].sel(time = 2006))

out_grid['superficie_cosechada_ha'].sel(time = 2000).sum() / 10**6
