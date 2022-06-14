# -*- coding: utf-8 -*-
"""
Load Harvest data and process them to become a single harvest area file
Created on Fri Apr  8 12:02:01 2022

@author: morenodu
# """
# Load observed national data from FAOSTAT to check if results make sense.

faostat_harvest = pd.read_csv('FAOSTAT_data_4-11-2022.csv')
faostat_harvest = faostat_harvest.loc[:,['Year','Area','Value']]
faostat_harvest['Value'] = faostat_harvest['Value']
faostat_harvest = faostat_harvest.loc[faostat_harvest['Year']>=1980]
faostat_harvest = faostat_harvest.loc[faostat_harvest['Year']<=2016]

sns.lineplot(data = faostat_harvest, hue = 'Area',x = 'Year', y= 'Value')



#%%
DS_harvest_are_us = xr.load_dataset("soy_harvest_area_US_all_1980_2020_05x05_density.nc", decode_times=False)
DS_harvest_are_us['time'] = pd.date_range(start='1980', periods=DS_harvest_are_us.sizes['time'], freq='YS').year

plot_2d_am_map(DS_harvest_are_us['harvest_area'].sel(time = 2000))

DS_harvest_are_arg = xr.load_dataset("soy_harvest_area_arg_1978_2019_05x05_density.nc", decode_times=False)
DS_harvest_are_arg['time'] = pd.date_range(start='1978', periods=DS_harvest_are_arg.sizes['time'], freq='YS').year
# SHift Argentina one year forward because of calendar
DS_harvest_are_arg = DS_harvest_are_arg.copy().shift(time = 1) # SHIFT AGRNEITNA ONE YeAR FORWARD
plot_2d_am_map(DS_harvest_are_arg['harvest_area'].sel(time = 2000))

DS_harvest_are_br = xr.load_dataset("soy_harvest_area_br_1980_2016_05x05_density.nc", decode_times=False)
DS_harvest_are_br['time'] = pd.date_range(start='1980', periods=DS_harvest_are_br.sizes['time'], freq='YS').year


plot_2d_am_map(DS_harvest_are_br['harvest_area'].sel(time = 1980))
plot_2d_am_map(DS_harvest_are_br['harvest_area'].sel(time = 2000))
plot_2d_am_map(DS_harvest_are_br['harvest_area'].sel(time = 2016))


sns.lineplot(data = faostat_harvest, hue = 'Area',x = 'Year', y= 'Value')
DS_harvest_are_us['harvest_area'].sum(['lat','lon']).plot(label = 'US')
DS_harvest_are_br['harvest_area'].sum(['lat','lon']).plot(label = 'BR')
DS_harvest_are_arg['harvest_area'].sum(['lat','lon']).plot(label = 'ARG')
plt.legend()
plt.show()
# IT SHOULD LOOK SIMILAR

# COMBINE
DS_harvest_area_am = DS_harvest_are_us.combine_first(DS_harvest_are_br)
DS_harvest_area_am = DS_harvest_area_am.combine_first(DS_harvest_are_arg)
DS_harvest_area_am = rearrange_latlot(DS_harvest_area_am)
plot_2d_am_map(DS_harvest_area_am['harvest_area'].sel(time = 1980))
DS_harvest_area_am.to_netcdf("soybean_harvest_area_calculated_americas_hg.nc")





#%% SPAM 2010

DS_harvest_area_spam = xr.load_dataset("soy_harvest_spam.nc", decode_times=True)
DS_harvest_area_spam['harvest_area'] = DS_harvest_area_spam['harvest_area'].where(DS_harvest_area_spam['harvest_area'] > 0)
plot_2d_am_map(DS_harvest_area_spam['harvest_area'])


