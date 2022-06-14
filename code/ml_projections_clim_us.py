# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 09:53:59 2021

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
from mask_shape_border import mask_shape_border
from failure_probability import feature_importance_selection, failure_probability
from stochastic_optimization_Algorithm import stochastic_optimization_Algorithm
from shap_prop import shap_prop
from bias_correction_masked import *
import matplotlib as mpl
import pickle

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['figure.dpi'] = 144
mpl.rcParams.update({'font.size': 14})

#%% Load climatic data

start_date, end_date = '01-01-2015','31-12-2100'

# MODEL
model = 'ukesm'
# one year change
rcp_scenario = 'ssp585' # 'ssp126', 'ssp585'
region = "_us" #""

DS_clim_ext_projections = xr.open_mfdataset('monthly_'+ model +'_'+ rcp_scenario + region +'/*.nc').sel(time=slice(start_date, end_date))

# Clean
DS_clim_ext_projections = DS_clim_ext_projections.drop_vars('fd') # Always zero
DS_clim_ext_projections = DS_clim_ext_projections.drop_vars('id') # Always zero
DS_clim_ext_projections = DS_clim_ext_projections.drop_vars('time_bnds') # Always zero
# DS_clim_ext_projections = DS_clim_ext_projections.drop_vars('spi') # Always zero
# DS_clim_ext_projections = DS_clim_ext_projections.drop_vars('spei') # Always zero
# DS_clim_ext_projections = DS_clim_ext_projections.drop('scale') # Always zero

DS_clim_ext_projections = DS_clim_ext_projections[list_features]

da_list = []
for feature in list(DS_clim_ext_projections.keys()):
    if (type(DS_clim_ext_projections[feature].values[0,0,0]) == type(DS_clim_ext_projections.r10mm.values[0,0,0])):
        print('Time')
        DS = timedelta_to_int(DS_clim_ext_projections, feature)
    else:
        print('Integer')
        DS = DS_clim_ext_projections[feature]
    
    da_list.append(DS)

DS_clim_ext_projections_combined = xr.merge(da_list)    
DS_clim_ext_projections_combined = DS_clim_ext_projections_combined.drop_vars('r10mm') # Always zero

DS_clim_ext_projections_combined.coords['lon'] = (DS_clim_ext_projections_combined.coords['lon'] + 180) % 360 - 180
DS_clim_ext_projections_combined = DS_clim_ext_projections_combined.sortby(DS_clim_ext_projections_combined.lon)
DS_clim_ext_projections_combined = DS_clim_ext_projections_combined.reindex(lat=DS_clim_ext_projections_combined.lat[::-1])
if len(DS_clim_ext_projections_combined.coords) >3 :
    DS_clim_ext_projections_combined=DS_clim_ext_projections_combined.drop('spatial_ref')
    
DS_clim_ext_projections_us = DS_clim_ext_projections_combined.where(DS_y_obs_us['usda_yield'].mean('time') >= 0.0 )

plot_2d_us_map(DS_clim_ext_projections_us['prcptot'].mean('time'))
DS_clim_ext_projections_us['prcptot'].mean(['lat','lon']).plot()

DS_clim_ext_projections_us_det = detrend_dataset(DS_clim_ext_projections_us)
DS_clim_ext_projections_us_det['prcptot'].mean(['lat','lon']).plot()

# There is a very annoying missing grid cell on the verge of the Michigan lake that requires to mask on the calendar too
DS_clim_ext_projections_us_det = DS_clim_ext_projections_us_det.where( DS_cal_sachs_month['plant'] >= 0 )
plot_2d_us_map(DS_clim_ext_projections_us_det['prcptot'].mean('time'))

# For loop along features to obtain 24 months of climatic data for each year
list_features_reshape_shift = []
for feature in list(DS_clim_ext_projections_us_det.keys()):
    ### Reshape
    df_test_shift = reshape_shift(DS_clim_ext_projections_us_det[feature])
    
    ### Join and change name to S for the shift values
    df_feature_reshape_shift = (df_test_shift.dropna().join(df_cal_sachs_month_harvest)
                                .rename(columns={'plant':'s'}))
    # Move 
    col = df_feature_reshape_shift.pop("s")
    df_feature_reshape_shift.insert(0, col.name, col)
    df_feature_reshape_shift[['s']].isna().sum()
    nan_rows = df_feature_reshape_shift[['s']][df_feature_reshape_shift[['s']].isnull().T.any()]
    if nan_rows.empty == False:
        print('Missing crop calendar values')
    
    # Shift accoording to month indicator (hence +1)
    df_feature_reshape_shift = (df_feature_reshape_shift.apply(lambda x : x.shift(-(int(x['s'])) + 1) , axis=1)
                                .drop(columns=['s']))
    
    
    list_features_reshape_shift.append(df_feature_reshape_shift)

# Transform into dataframe
df_proj_features_reshape_shift = pd.concat(list_features_reshape_shift, axis=1)

### Select specific months
suffixes = tuple(["_"+str(j) for j in range(3,6)])
df_proj_feature_season_6mon = df_proj_features_reshape_shift.loc[:,df_proj_features_reshape_shift.columns.str.endswith(suffixes)]
df_proj_feature_season_6mon = df_proj_feature_season_6mon.rename_axis(index={'year':'time'}).reorder_levels(['time','lat','lon']).sort_index()
df_proj_feature_season_6mon = df_proj_feature_season_6mon.dropna()
#%% EPIC projections

model_full = 'ukesm1-0-ll' #ukesm1-0-ll,gfdl-esm4
co2_scenario = '2015co2' # 'default' '2015co2'
DS_y_epic_proj = xr.open_dataset("epic-iiasa_"+ model_full +"_w5e5_"+rcp_scenario+"_2015soc_"+co2_scenario+"_yield-soy-noirr_global_annual_2015_2100.nc", decode_times=False)


# Convert time unit
units, reference_date = DS_y_epic_proj.time.attrs['units'].split('since')
DS_y_epic_proj['time'] = pd.date_range(start=' 2015-01-01, 00:00:00', periods=DS_y_epic_proj.sizes['time'], freq='YS')
DS_y_epic_proj['time'] = DS_y_epic_proj['time'].dt.year

DS_y_epic_proj['yield-soy-noirr'].mean(['lat', 'lon']).plot()
plt.show()

DS_y_epic_proj_us = DS_y_epic_proj.where(DS_y_obs_us['usda_yield'].mean('time') >= 0.0 )

DS_y_epic_proj_us_det = xr.DataArray( detrend_dim(DS_y_epic_proj_us['yield-soy-noirr'], 'time') + DS_y_epic_proj_us['yield-soy-noirr'].mean('time'), name= DS_y_epic_proj_us['yield-soy-noirr'].name, attrs = DS_y_epic_proj_us['yield-soy-noirr'].attrs)

plot_2d_us_map(DS_y_epic_proj_us_det.mean('time'))

DS_y_epic_proj_us['yield-soy-noirr'].mean(['lat', 'lon']).plot()
DS_y_epic_proj_us_det.mean(['lat', 'lon']).plot()
plt.show()

df_y_epic_proj_us = DS_y_epic_proj_us_det.to_dataframe().dropna()
df_y_epic_proj_us = df_y_epic_proj_us.reorder_levels(['time','lat','lon']).sort_index()

#%% Doubts on season calendar

df_hybrid_proj_2 = pd.concat([df_y_epic_proj_us, df_proj_feature_season_6mon], axis = 1 )
df_hybrid_proj_test_2 = df_hybrid_proj_2.query("time>=2015 and time <= 2100")

df_prediction_proj_2 = df_y_epic_proj_us.query("time>=2015 and time <= 2100").copy() 
predic_model_test_2 = model_hyb_us.predict(df_hybrid_proj_test_2).copy()
df_prediction_proj_2.loc[:,'yield-soy-noirr'] = predic_model_test_2

DS_hybrid_proj_2 = xr.Dataset.from_dataframe(df_prediction_proj_2)
DS_hybrid_proj_2 = DS_hybrid_proj_2.sortby('lat')
DS_hybrid_proj_2 = DS_hybrid_proj_2.sortby('lon')

plt.plot(DS_hybrid_proj_2.time.values, DS_hybrid_proj_2['yield-soy-noirr'].mean(['lat','lon']), label = 'Hybrid')
plt.plot(DS_y_epic_proj_us_det.time.values, DS_y_epic_proj_us_det.mean(['lat','lon']), label = 'EPIC pure')
plt.title(rcp_scenario+"_"+co2_scenario)
plt.legend()
plt.show()


plot_2d_us_map(DS_y_epic_proj_us_det.sel(time=2084))
plot_2d_us_map(DS_hybrid_proj_2['yield-soy-noirr'].sel(time=2084))

#%% EPIC SIM
df_predic_epic_test = df_y_epic_proj_us.copy()
predic_epic_test = full_model_epic_us.predict(df_y_epic_proj_us)
df_predic_epic_test.loc[:,'yield-soy-noirr'] = full_model_epic_us.predict(df_y_epic_proj_us)

DS_pred_epic_proj = xr.Dataset.from_dataframe(df_predic_epic_test)
DS_pred_epic_proj = DS_pred_epic_proj.sortby('lat')
DS_pred_epic_proj = DS_pred_epic_proj.sortby('lon')

# CLIMATIC model
df_pred_clim_proj_test = df_y_epic_proj_us.copy()
df_pred_clim_proj_test.loc[:,'yield-soy-noirr'] = full_model_exclim_dyn_us.predict(df_proj_feature_season_6mon)

DS_pred_clim_proj = xr.Dataset.from_dataframe(df_pred_clim_proj_test)
DS_pred_clim_proj = DS_pred_clim_proj.sortby('lat')
DS_pred_clim_proj = DS_pred_clim_proj.sortby('lon')

# WEIGHTING SCHEME
DS_harvest_area_globiom = xr.open_dataset('soy_usa_harvest_area_05x05.nc')
DS_harvest_area_globiom['harvest_area'] = DS_harvest_area_globiom['harvest_area'].where(DS_hybrid_proj_2['yield-soy-noirr'].mean('time')>0)
plot_2d_us_map(DS_harvest_area_globiom['harvest_area'])
plot_2d_us_map(DS_pred_clim_proj['yield-soy-noirr'].mean('time'))

total_area = DS_harvest_area_globiom['harvest_area'].sum()
DS_hybrid_weighted = ((DS_hybrid_proj_2['yield-soy-noirr'] * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'yield-soy-noirr')
DS_epic_weighted = ((DS_pred_epic_proj['yield-soy-noirr'] * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'yield-soy-noirr')
DS_clim_weighted = ((DS_pred_clim_proj['yield-soy-noirr'] * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'yield-soy-noirr')

plt.plot(DS_hybrid_proj_2['yield-soy-noirr'].mean(['lat','lon']), label = 'unweighted')
plt.plot(DS_hybrid_weighted['yield-soy-noirr'].sum(['lat','lon']), label = 'weighted')
plt.ylabel('yield')
plt.xlabel('years')
plt.legend()
plt.show()


# plt.plot(DS_y_epic_proj_us_det.time.values, DS_y_epic_proj_us_det.mean(['lat','lon']), label = 'Pure EPIC')
plt.plot(DS_pred_epic_proj.time.values, DS_pred_epic_proj["yield-soy-noirr"].mean(['lat','lon']), label = 'EPIC-RF')
plt.plot(DS_hybrid_proj_2.time.values, DS_hybrid_proj_2["yield-soy-noirr"].mean(['lat','lon']), label = 'Hybrid-RF')
plt.legend()
plt.show()

plt.plot(DS_pred_epic_proj.time.values, DS_epic_weighted["yield-soy-noirr"].sum(['lat','lon'])/np.mean(DS_epic_weighted["yield-soy-noirr"].sum(['lat','lon'])), label = 'EPIC-RF')
plt.plot(DS_pred_clim_proj.time.values, DS_clim_weighted["yield-soy-noirr"].sum(['lat','lon'])/np.mean(DS_clim_weighted["yield-soy-noirr"].sum(['lat','lon'])), label = 'CLIM-RF')
plt.plot(DS_hybrid_proj_2.time.values, DS_hybrid_weighted['yield-soy-noirr'].sum(['lat','lon'])/np.mean(DS_hybrid_weighted["yield-soy-noirr"].sum(['lat','lon'])), label = 'Hybrid')
plt.title(rcp_scenario+"_"+co2_scenario)
plt.ylabel("Shock (%)")
plt.legend()
plt.show()


DS_hybrid_proj_2.to_netcdf('output_models'+region +'/hybrid_'+model_full+'_'+rcp_scenario+'_'+co2_scenario+'_yield_soybean_2015_2100.nc')
DS_pred_clim_proj.to_netcdf('output_models'+region +'/clim_'+model_full+'_'+rcp_scenario+'_'+co2_scenario+'_yield_soybean_2015_2100.nc')
DS_pred_epic_proj.to_netcdf('output_models'+region +'/epic_'+model_full+'_'+rcp_scenario+'_'+co2_scenario+'_yield_soybean_2015_2100.nc')

#%% Extrapolation tests

def exceedance_freq(exceedance_data, ref_data):
    # Test how many times variables are extrapolated outside of the training range for each variable.
    test_max = exceedance_data > ref_data.max(axis=0)
    exceedance_percentage_max = test_max.sum(axis=0)*100/len(test_max)
    
    test_min = exceedance_data < ref_data.min(axis=0)
    exceedance_percentage_min = test_min.sum(axis=0)*100/len(test_min)
    
    max_min = pd.concat([exceedance_percentage_max, exceedance_percentage_min], axis = 1)    
    max_min.columns = ['% Above', '% Below']

    return max_min

df_hybrid_us_2 = df_hybrid_us.copy()   
df_hybrid_us_2 = df_hybrid_us_2.rename(columns={'yield':'yield-soy-noirr'})
max_min_US = exceedance_freq(df_hybrid_proj_test_2, df_hybrid_us_2)








#%% HYBRID PROJECTIONS - REDO THIS

def synthetic_timeseries(chosen_type = 'hybrid'):
    DS_hyb_ipsl_585_2015co2 = xr.open_dataset(chosen_type + "_ipsl-cm6a-lr_ssp585_2015co2_yield_soybean_2015_2100.nc", decode_times=False)
    DS_hyb_ipsl_585_default = xr.open_dataset(chosen_type + "_ipsl-cm6a-lr_ssp585_default_yield_soybean_2015_2100.nc", decode_times=False)
    DS_hyb_ipsl_126_2015co2 = xr.open_dataset(chosen_type + "_ipsl-cm6a-lr_ssp126_2015co2_yield_soybean_2015_2100.nc", decode_times=False)
    DS_hyb_ipsl_126_default = xr.open_dataset(chosen_type + "_ipsl-cm6a-lr_ssp126_default_yield_soybean_2015_2100.nc", decode_times=False)
    
    DS_hyb_ukesm_585_2015co2 = xr.open_dataset(chosen_type + "_ukesm1-0-ll_ssp585_2015co2_yield_soybean_2015_2100.nc", decode_times=False)
    DS_hyb_ukesm_585_default = xr.open_dataset(chosen_type + "_ukesm1-0-ll_ssp585_default_yield_soybean_2015_2100.nc", decode_times=False)
    DS_hyb_ukesm_126_2015co2 = xr.open_dataset(chosen_type + "_ukesm1-0-ll_ssp126_2015co2_yield_soybean_2015_2100.nc", decode_times=False)
    DS_hyb_ukesm_126_default = xr.open_dataset(chosen_type + "_ukesm1-0-ll_ssp126_default_yield_soybean_2015_2100.nc", decode_times=False)
    
    DS_hyb_gfdl_585_2015co2 = xr.open_dataset(chosen_type + "_gfdl-esm4_ssp585_2015co2_yield_soybean_2015_2100.nc", decode_times=False)
    DS_hyb_gfdl_585_default = xr.open_dataset(chosen_type + "_gfdl-esm4_ssp585_default_yield_soybean_2015_2100.nc", decode_times=False)
    DS_hyb_gfdl_126_2015co2 = xr.open_dataset(chosen_type + "_gfdl-esm4_ssp126_2015co2_yield_soybean_2015_2100.nc", decode_times=False)
    DS_hyb_gfdl_126_default = xr.open_dataset(chosen_type + "_gfdl-esm4_ssp126_default_yield_soybean_2015_2100.nc", decode_times=False)
    
    plt.figure(figsize = (10,6),dpi=300)
    plt.plot(DS_hyb_ipsl_585_2015co2["yield-soy-noirr"].mean(['lat','lon']), label = 'ipsl 8p5 2015CO2')
    plt.plot(DS_hyb_ipsl_126_2015co2["yield-soy-noirr"].mean(['lat','lon']), label = 'ipsl 2p6 2015CO2', linestyle=':')
    plt.plot(DS_hyb_ukesm_585_2015co2["yield-soy-noirr"].mean(['lat','lon']), label = 'ukesm 8p5 2015CO2')
    plt.plot(DS_hyb_ukesm_126_2015co2["yield-soy-noirr"].mean(['lat','lon']), label = 'ukesm 2p6 2015CO2', linestyle=':')
    plt.plot(DS_hyb_gfdl_585_2015co2["yield-soy-noirr"].mean(['lat','lon']), label = 'gfdl 8p5 2015CO2')
    plt.plot(DS_hyb_gfdl_126_2015co2["yield-soy-noirr"].mean(['lat','lon']), label = 'gfdl 2p6 2015CO2', linestyle=':')
    plt.ylabel("Yield (ton/ha)")
    plt.xlabel("Years")
    plt.legend()
    plt.show()
    
    ### WIEGHTED ANALYSIS
    DS_harvest_area_globiom = xr.open_dataset('../../paper_hybrid_agri/data/soy_usa_harvest_area_05x05.nc')
    DS_harvest_area_globiom['harvest_area'] = DS_harvest_area_globiom['harvest_area'].where(DS_hyb_ipsl_585_2015co2['yield-soy-noirr']>0)
    total_area = DS_harvest_area_globiom['harvest_area'].sum(['lat','lon'])
    
    # RCP  8.5
    DS_hyb_ipsl_585_2015co2_weighted = ((DS_hyb_ipsl_585_2015co2['yield-soy-noirr'] * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'Yield')
    DS_hyb_ukesm_585_2015co2_weighted = ((DS_hyb_ukesm_585_2015co2['yield-soy-noirr'] * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'Yield')
    DS_hyb_gfdl_585_2015co2_weighted = ((DS_hyb_gfdl_585_2015co2['yield-soy-noirr'] * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'Yield')
    
    # RCP  2.6
    DS_hyb_ipsl_126_2015co2_weighted = ((DS_hyb_ipsl_126_2015co2['yield-soy-noirr'] * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'Yield')
    DS_hyb_ukesm_126_2015co2_weighted = ((DS_hyb_ukesm_126_2015co2['yield-soy-noirr'] * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'Yield')
    DS_hyb_gfdl_126_2015co2_weighted = ((DS_hyb_gfdl_126_2015co2['yield-soy-noirr'] * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'Yield')
    
    DS_hyb_585_2015co2 = pd.concat([DS_hyb_ipsl_585_2015co2_weighted.sum(['lat','lon']).to_dataframe(),
                                    DS_hyb_ukesm_585_2015co2_weighted.sum(['lat','lon']).to_dataframe(),
                                    DS_hyb_gfdl_585_2015co2_weighted.sum(['lat','lon']).to_dataframe()], axis=0)
    
    DS_hyb_585_2015co2['RCP'] = '8.5'
    
    DS_hyb_126_2015co2 = pd.concat([DS_hyb_ipsl_126_2015co2_weighted.sum(['lat','lon']).to_dataframe(),
                                    DS_hyb_ukesm_126_2015co2_weighted.sum(['lat','lon']).to_dataframe(),
                                    DS_hyb_gfdl_126_2015co2_weighted.sum(['lat','lon']).to_dataframe()], axis=0)
    
    DS_hyb_126_2015co2['RCP'] = '2.6'
    
    
    df_hyb_2015co2 = pd.concat([DS_hyb_126_2015co2,DS_hyb_585_2015co2])
    
    return df_hyb_2015co2

df_hyb_2015co2 = synthetic_timeseries(chosen_type = 'hybrid')
df_hyb_2015co2['model'] = 'hybrid'

df_epic_2015co2 = synthetic_timeseries(chosen_type = 'epic')
df_epic_2015co2['model'] = 'epic'

df_clim_2015co2 = synthetic_timeseries(chosen_type = 'clim')


df_2015co2 = pd.concat([df_hyb_2015co2,df_epic_2015co2])
df_2015co2['RCP_model'] = df_2015co2['RCP'] + "_"+ df_2015co2['model']

plt.figure(figsize = (8,6),dpi=300)
sns.lineplot(data = df_hyb_2015co2, x = df_hyb_2015co2.index, y = df_hyb_2015co2['Yield'], hue = df_hyb_2015co2['RCP'])
plt.ylabel("Yield (ton/ha)")
plt.xlabel("Years")
plt.show()

sns.kdeplot(data = df_hyb_2015co2, x = df_hyb_2015co2['Yield'], hue = df_hyb_2015co2['RCP'], fill = True)


plt.figure(figsize = (8,6),dpi=300)
sns.lineplot(data = df_2015co2, x = df_2015co2.index, y = df_2015co2['Yield'], hue = df_2015co2['RCP_model'])
plt.ylabel("Yield (ton/ha)")
plt.xlabel("Years")
plt.show()

plt.figure(figsize = (10,6),dpi=300)
sns.kdeplot(data = df_2015co2, x = df_2015co2['Yield'], hue = df_2015co2['RCP_model'], fill = True)
plt.show()


#%% GLOBIOM SHIFTERS

def local_minima_30years(dataset, no_cycles = 3, weights = 'NO'):
    
    cycle_period = ( dataset.time.max() - dataset.time.min() ) // no_cycles
    year0 = dataset.time[0].values
    
    list_shifters_cycle = []
    for cycle in range(no_cycles):
        
        dataset_cycle = dataset.sel( time = slice( year0 + cycle_period * cycle, year0 + 1 + cycle_period * ( cycle+1 ) ) )
        
        if weights == 'NO':

            year_min = dataset_cycle.time[0].values + dataset_cycle['yield-soy-noirr'].mean(['lat','lon']).argmin(dim='time').values
            ds_local_minima = dataset_cycle.sel(time=year_min)
            ds_shifter_cycle = ds_local_minima / dataset_cycle.mean(['time'])
            print('year_min:',dataset_cycle.time[0].values, 'year_max:',dataset_cycle.time[-1].values, 'min_pos',year_min, 'shifter', list(ds_shifter_cycle.to_dataframe().mean().values))
            
        elif weights == 'YES':
            
            total_area = DS_harvest_area_globiom['harvest_area'].sum()
            DS_weighted = ((dataset['yield-soy-noirr'] * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'yield-soy-noirr')            
            dataset_cycle_weight = DS_weighted.sel( time = slice( year0 + cycle_period * cycle, year0 + 1 + cycle_period * ( cycle+1 ) ) )
 
            year_min = dataset_cycle_weight.time[0].values + dataset_cycle_weight['yield-soy-noirr'].sum(['lat','lon']).argmin(dim='time').values
            ds_local_minima = dataset_cycle.sel(time=year_min)
            ds_shifter_cycle = ds_local_minima / dataset_cycle.mean(['time'])     
            print('year_min:', dataset_cycle.time[0].values, 'year_max:',dataset_cycle.time[-1].values, 'min_pos',year_min, 'shifter', list(ds_shifter_cycle.to_dataframe().mean().values))
            
        list_shifters_cycle.append(ds_shifter_cycle)
        
    return list_shifters_cycle
        
# list_test = local_minima_30years(DS_hybrid_proj_2, weights = 'NO') 
list_test = local_minima_30years(DS_hybrid_proj_2, weights = 'YES') 

list_test[0].to_netcdf("hybrid_"+model_full+'_'+rcp_scenario+"_"+co2_scenario+"_yield_soybean_shift_2017-2044.nc")
list_test[1].to_netcdf("hybrid_"+model_full+'_'+rcp_scenario+"_"+co2_scenario+"_yield_soybean_shift_2044-2071.nc")
list_test[2].to_netcdf("hybrid_"+model_full+'_'+rcp_scenario+"_"+co2_scenario+"_yield_soybean_shift_2071-2098.nc")

plot_2d_map(list_test[0]["yield-soy-noirr"])

list_shift_epic_proj = local_minima_30years(DS_pred_epic_proj, weights = 'YES') 

list_shift_epic_proj[0].to_netcdf("epic_"+model_full+'_'+rcp_scenario+"_"+co2_scenario+"_yield_soybean_shift_2017-2044.nc")
list_shift_epic_proj[1].to_netcdf("epic_"+model_full+'_'+rcp_scenario+"_"+co2_scenario+"_yield_soybean_shift_2044-2071.nc")
list_shift_epic_proj[2].to_netcdf("epic_"+model_full+'_'+rcp_scenario+"_"+co2_scenario+"_yield_soybean_shift_2071-2098.nc")


list_shift_clim_proj = local_minima_30years(DS_pred_clim_proj, weights = 'YES') 

list_shift_clim_proj[0].to_netcdf("clim_"+model_full+'_'+rcp_scenario+"_"+co2_scenario+"_yield_soybean_shift_2017-2044.nc")
list_shift_clim_proj[1].to_netcdf("clim_"+model_full+'_'+rcp_scenario+"_"+co2_scenario+"_yield_soybean_shift_2044-2071.nc")
list_shift_clim_proj[2].to_netcdf("clim_"+model_full+'_'+rcp_scenario+"_"+co2_scenario+"_yield_soybean_shift_2071-2098.nc")



def shocks_series(dataset, start_year, end_year, ref_start_year, ref_end_year):
    
    dataset_mean_general = dataset.sel( time = slice( ref_start_year, ref_end_year ) ).mean(['time'])
    dataset_cycle = dataset.sel( time = slice( start_year, end_year ) )
         
    shocks_series = dataset_cycle / dataset_mean_general

    total_area = DS_harvest_area_globiom['harvest_area'].sum()
    DS_weighted = ((dataset['yield-soy-noirr'] * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'yield-soy-noirr')            
    dataset_cycle_weight = DS_weighted.sel( time = slice( start_year, end_year ))
    
    shocks_series_weight = dataset_cycle_weight['yield-soy-noirr'].sum(['lat','lon']) / dataset_mean_general.mean()

    df_shocks = pd.DataFrame(index = dataset_cycle.time.values, data = shocks_series_weight.to_dataframe().values)

    print('Weighted shocks:', df_shocks)
        
        
    return shocks_series
        
shcoks_series_hybrid = shocks_series(DS_hybrid_proj_2, 2095, 2099, 2070, 2100)
shcoks_series_epic = shocks_series(DS_pred_epic_proj, 2095, 2099, 2070, 2100)
shcoks_series_clim = shocks_series(DS_pred_clim_proj, 2095, 2099, 2070, 2100)

shcoks_series_test.to_netcdf("hybrid_"+model_full+'_'+rcp_scenario+"_"+co2_scenario+"_yield_soybean_shocks_series_2095_2099.nc")
shcoks_series_test.to_netcdf("epic_"+model_full+'_'+rcp_scenario+"_"+co2_scenario+"_yield_soybean_shocks_series_2095_2099.nc")
shcoks_series_test.to_netcdf("clim_"+model_full+'_'+rcp_scenario+"_"+co2_scenario+"_yield_soybean_shocks_series_2095_2099.nc")
