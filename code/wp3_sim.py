# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 16:45:24 2022

@author: morenodu
"""
import os
os.chdir('C:/Users/morenodu/OneDrive - Stichting Deltares/Documents/PhD/paper_hybrid_agri/data')
import xarray as xr 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import cartopy
import cartopy.crs as ccrs
import matplotlib as mplf

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['figure.dpi'] = 144
mpl.rcParams.update({'font.size': 14})


#%% 1) Determine the climatic shifters of 2012

def hybrid_prediction(DS_input_hybrid):    
    df_input_hybrid_fut_test_2 = DS_input_hybrid.to_dataframe().dropna()
    df_input_hybrid_fut_test_2 = df_input_hybrid_fut_test_2.rename_axis(list(DS_input_hybrid.coords))
    
    if 'time' in list(DS_input_hybrid.coords):
        df_input_hybrid_fut_test_2 = df_input_hybrid_fut_test_2.reorder_levels(['time','lat','lon']).sort_index()
    else:
        df_input_hybrid_fut_test_2 = df_input_hybrid_fut_test_2.reorder_levels(['lat','lon']).sort_index()

    # Predicting hybrid results
    df_prediction_proj_2 = df_input_hybrid_fut_test_2[['yield-soy-noirr']].copy() 
    predic_model_test_2 = full_model_hyb_am2.predict(df_input_hybrid_fut_test_2.copy() )
    df_prediction_proj_2.loc[:,'yield-soy-noirr'] = predic_model_test_2
    
    DS_hybrid_proj_2 = xr.Dataset.from_dataframe(df_prediction_proj_2)
    # Reindex to avoid missnig coordinates and dimension values
    DS_hybrid_proj_2 = rearrange_latlot(DS_hybrid_proj_2)

    return DS_hybrid_proj_2

def shift_comparison(DS_epic, DS_clim):
    DS_input_hybrid = convert_detrend_fut(DS_epic, DS_clim, detrend = 'historical')
    
    DS_input_hybrid_mean = DS_input_hybrid.mean('time')
    DS_input_hybrid_shift = DS_input_hybrid_mean + DS_input_anomaly_2012.drop('time')
    
    DS_output_hybrid = hybrid_prediction(DS_input_hybrid)
    DS_output_hybrid_shift = hybrid_prediction(DS_input_hybrid_shift)
    
    DS_output_hybrid_weighted = weighted_conversion(DS_output_hybrid['yield-soy-noirr'], DS_harvest_area_fut, name_ds = 'yield-soy-noirr')
    DS_output_hybrid_shift_weighted = weighted_conversion(DS_output_hybrid_shift['yield-soy-noirr'], DS_harvest_area_fut, name_ds = 'yield-soy-noirr')
    
    plt.plot(DS_output_hybrid.time, DS_output_hybrid_weighted['yield-soy-noirr'])
    plt.axhline(DS_output_hybrid_shift_weighted['yield-soy-noirr'], linestyle = 'dashed')
    plt.show()
    
    difference = (DS_output_hybrid_weighted['yield-soy-noirr'].min() - DS_output_hybrid_hist_weighted['yield-soy-noirr'].mean().values.item())/DS_output_hybrid_hist_weighted['yield-soy-noirr'].mean().values.item() 
    
    print('The difference between the lowest point in the series and the 2012 shock was:',difference)
    
    return DS_output_hybrid, difference


DS_input = xr.Dataset.from_dataframe(df_input_hybrid_am).sel(time = slice(2000,2015))
DS_input = rearrange_latlot(DS_input)

DS_input_2 = DS_input.rename({'yield':'yield-soy-noirr'})
DS_output_hybrid_hist = hybrid_prediction(DS_input_2)
DS_output_hybrid_hist_weighted = weighted_conversion(DS_output_hybrid_hist['yield-soy-noirr'], DS_harvest_area_fut, name_ds = 'yield-soy-noirr')



DS_input_2012 = DS_input.sel(time =2012) 

DS_input_anomaly_2012 = DS_input_2012 - DS_input.mean('time')
DS_input_anomaly_2012 = DS_input_anomaly_2012.rename({'yield': 'yield-soy-noirr'})
plot_2d_am_map(DS_input_anomaly_2012['txm_3'], title = 'Soybean yield in 2012')

DS_input_2012_2 = DS_input_2012.copy()
DS_input_2012_2 = DS_input_2012_2.rename({'yield':'yield-soy-noirr'})
DS_output_hybrid_2012 = hybrid_prediction(DS_input_2012_2.drop('time'))
DS_output_hybrid_2012_weighted = weighted_conversion(DS_output_hybrid_2012['yield-soy-noirr'], DS_harvest_area_fut, name_ds = 'yield-soy-noirr')

plot_2d_am_map(DS_output_hybrid_2012['yield-soy-noirr'], title = 'Soybean yield in 2012')

shift_2012_value = DS_output_hybrid_2012_weighted['yield-soy-noirr'].values.item() - DS_output_hybrid_hist_weighted['yield-soy-noirr'].mean().values.item()
plot_2d_am_map(DS_output_hybrid_2012['yield-soy-noirr'] - DS_output_hybrid_hist['yield-soy-noirr'].mean('time'), title = 'Soybean yield anomaly in 2012',colormap = 'RdBu', label_cbar = 'Yield (ton/ha)')


# 2) Get the 2C adn 3C for each GCM - 30 year period and average them.

import glob
import re

def open_csv_timeseries(path, pre_ind_tmep = 13.8):
    files = glob.glob(path)
    df = []
    for f in files:
        csv = pd.read_csv(f, header=None, index_col = 0)
        csv = csv.rename(columns={1:'tas'})
        csv.index.name = 'time'
        df.append(csv)
        df_2 = pd.concat(df) - 273.15 - pre_ind_tmep
        df_2.index = pd.to_datetime(df_2.index)
        
        name = re.split('/|_', path)        
        df_2.to_csv("projections_global_mean/"+name[7]+"_"+name[3]+"_"+name[6]+".csv")

    return df_2


df_ipsl_26 = open_csv_timeseries("projections_global_mean/ipsl-cm6a-lr_r1i1p1f1_w5e5_ssp126_tas_*.csv")
df_ipsl_85 = open_csv_timeseries("projections_global_mean/ipsl-cm6a-lr_r1i1p1f1_w5e5_ssp585_tas_*.csv")

df_ukesm_26 = open_csv_timeseries("projections_global_mean/ukesm1-0-ll_r1i1p1f2_w5e5_ssp126_tas_*.csv")
df_ukesm_85 = open_csv_timeseries("projections_global_mean/ukesm1-0-ll_r1i1p1f2_w5e5_ssp585_tas_*.csv")

df_gfdl_26 = open_csv_timeseries("projections_global_mean/gfdl-esm4_r1i1p1f1_w5e5_ssp126_tas_*.csv")
df_gfdl_85 = open_csv_timeseries("projections_global_mean/gfdl-esm4_r1i1p1f1_w5e5_ssp585_tas_*.csv")

averaging_function_days = '1825d'
df_ipsl_26_5y = df_ipsl_26.rolling(averaging_function_days).mean()
df_ipsl_85_5y = df_ipsl_85.rolling(averaging_function_days).mean()
df_gfdl_26_5y = df_gfdl_26.rolling(averaging_function_days).mean()
df_gfdl_85_5y = df_gfdl_85.rolling(averaging_function_days).mean()
df_ukesm_26_5y = df_ukesm_26.rolling(averaging_function_days).mean()
df_ukesm_85_5y = df_ukesm_85.rolling(averaging_function_days).mean()

plt.figure(figsize = (10,8),dpi=300)
plt.plot(df_ipsl_26_5y['2020':'2100-12-31'], label = 'ipsl_26')
plt.plot(df_ipsl_85_5y['2020':'2100-12-31'], label = 'ipsl_85')
plt.plot(df_gfdl_26_5y['2020':'2100-12-31'], label = 'gfdl_26')
plt.plot(df_gfdl_85_5y['2020':'2100-12-31'], label = 'gfdl_85')
plt.plot(df_ukesm_26_5y['2020':'2100-12-31'], label = 'ukesm_26')
plt.plot(df_ukesm_85_5y['2020':'2100-12-31'], label = 'ukesm_85')
plt.axhline(1.5, linestyle = 'dashed')
plt.axhline(2.0, linestyle = 'dashed')
plt.axhline(3.0, linestyle = 'dashed',c = 'red')
plt.yticks((np.arange(0,5,0.5)))
plt.ylim(0,5)
plt.legend()
plt.show()


print('2 degree for ipsl_26 at', df_ipsl_26_5y[df_ipsl_26_5y['tas'] >= 2].index[0] if np.max(df_ipsl_26_5y['tas']) >= 2 else 'No 2 degree')
print('2 degree for ipsl_85 at', df_ipsl_85_5y[df_ipsl_85_5y['tas'] >= 2].index[0] if np.max(df_ipsl_85_5y['tas']) >= 2 else 'No 2 degree')
print('2 degree for gfdl_26 at', df_gfdl_26_5y[df_gfdl_26_5y['tas'] >= 2].index[0] if np.max(df_gfdl_26_5y['tas']) >= 2 else 'No 2 degree')
print('2 degree for gfdl_85 at', df_gfdl_85_5y[df_gfdl_85_5y['tas'] >= 2].index[0] if np.max(df_gfdl_85_5y['tas']) >= 2 else 'No 2 degree')
print('2 degree for ukesm_26 at', df_ukesm_26_5y[df_ukesm_26_5y['tas'] >= 2].index[0] if np.max(df_ukesm_26_5y['tas']) >= 2 else 'No 2 degree')
print('2 degree for ukesm_85 at', df_ukesm_85_5y[df_ukesm_85_5y['tas'] >= 2].index[0] if np.max(df_ukesm_85_5y['tas']) >= 2 else 'No 2 degree')

print('3 degree for ipsl_26 at', df_ipsl_26_5y[df_ipsl_26_5y['tas'] >= 3].index[0] if np.max(df_ipsl_26_5y['tas']) >= 3 else 'No 3 degree')
print('3 degree for ipsl_85 at',df_ipsl_85_5y[df_ipsl_85_5y['tas'] >= 3].index[0] if np.max(df_ipsl_85_5y['tas']) >= 3 else 'No 3 degree')
print('3 degree for gfdl_26 at',df_gfdl_26_5y[df_gfdl_26_5y['tas'] >= 3].index[0] if np.max(df_gfdl_26_5y['tas']) >= 3 else 'No 3 degree')
print('3 degree for gfdl_85 at',df_gfdl_85_5y[df_gfdl_85_5y['tas'] >= 3].index[0] if np.max(df_gfdl_85_5y['tas']) >= 3 else 'No 3 degree')
print('3 degree for ukesm_26 at',df_ukesm_26_5y[df_ukesm_26_5y['tas'] >= 3].index[0] if np.max(df_ukesm_26_5y['tas']) >= 3 else 'No 3 degree')
print('3 degree for ukesm_85 at',df_ukesm_85_5y[df_ukesm_85_5y['tas'] >= 3].index[0] if np.max(df_ukesm_85_5y['tas']) >= 3 else 'No 3 degree')


def ds_levels_selection(DS_epic, DS_clim):
    DS_epic_mid = DS_epic.sel(time = slice(2035,2065))
    DS_epic_late = DS_epic.sel(time = slice(2066,2096))
    
    DS_clim_mid = DS_clim.sel(time = slice(2035,2065))
    DS_clim_late = DS_clim.sel(time = slice(2066,2096))
    
    return DS_epic_mid, DS_epic_late, DS_clim_mid, DS_clim_late

DS_y_epic_proj_ukesm_585_am_mid, DS_y_epic_proj_ukesm_585_am_late, DS_feature_proj_ukesm_585_am_mid, DS_feature_proj_ukesm_585_am_late = ds_levels_selection(DS_y_epic_proj_ukesm_585_am, DS_feature_proj_ukesm_585_am)
DS_y_epic_proj_ukesm_126_am_mid, DS_y_epic_proj_ukesm_126_am_late, DS_feature_proj_ukesm_126_am_mid, DS_feature_proj_ukesm_126_am_late = ds_levels_selection(DS_y_epic_proj_ukesm_126_am, DS_feature_proj_ukesm_126_am)

DS_y_epic_proj_gfdl_585_am_mid, DS_y_epic_proj_gfdl_585_am_late, DS_feature_proj_gfdl_585_am_mid, DS_feature_proj_gfdl_585_am_late = ds_levels_selection(DS_y_epic_proj_gfdl_585_am, DS_feature_proj_gfdl_585_am)
DS_y_epic_proj_gfdl_126_am_mid, DS_y_epic_proj_gfdl_126_am_late, DS_feature_proj_gfdl_126_am_mid, DS_feature_proj_gfdl_126_am_late = ds_levels_selection(DS_y_epic_proj_gfdl_126_am, DS_feature_proj_gfdl_126_am)

DS_y_epic_proj_ipsl_585_am_mid, DS_y_epic_proj_ipsl_585_am_late, DS_feature_proj_ipsl_585_am_mid, DS_feature_proj_ipsl_585_am_late = ds_levels_selection(DS_y_epic_proj_ipsl_585_am, DS_feature_proj_ipsl_585_am)
DS_y_epic_proj_ipsl_126_am_mid, DS_y_epic_proj_ipsl_126_am_late, DS_feature_proj_ipsl_126_am_mid, DS_feature_proj_ipsl_126_am_late = ds_levels_selection(DS_y_epic_proj_ipsl_126_am, DS_feature_proj_ipsl_126_am)


def convert_detrend_fut(DS_epic, DS_clim, detrend = 'historical'):
    # Detrend clim
    ### Choose historical mean so there is no mean deviation between the historical and future timelines
    if detrend == 'historical':
        DS_feature_proj_6mon_am_det = detrend_dataset(DS_clim, deg = 'free', mean_data = DS_feature_season_6mon_am_det.sel(time = slice(2005,2015))) 
    elif detrend == 'mean future':
        DS_feature_proj_6mon_am_det = detrend_dataset(DS_clim, deg = 'free') 
    elif detrend == 'no detrend':
        DS_feature_proj_6mon_am_det = DS_clim
    
    df_feature_proj_6mon_am_det = DS_feature_proj_6mon_am_det.to_dataframe().dropna()
    df_feature_proj_6mon_am_det = df_feature_proj_6mon_am_det.rename_axis(list(DS_clim.coords)).reorder_levels(['time','lat','lon']).sort_index()

    list_feat_precipitation = [s for s in df_feature_proj_6mon_am_det.keys() if "prcptot" in s]
    for feature in list_feat_precipitation:
        df_feature_proj_6mon_am_det[feature][df_feature_proj_6mon_am_det[feature] < 0] = 0
    
    # Detrend epic
    DS_detrended, DS_fit = detrend_dim_2(DS_epic['yield-soy-noirr'], 'time')
    DS_fit_mean = xr.DataArray( DS_fit + DS_detrended.mean(['time']), name= DS_epic['yield-soy-noirr'].name, attrs = DS_epic['yield-soy-noirr'].attrs)
    
    ####
    if detrend == 'historical':
        DS_y_epic_proj_am_det = xr.DataArray( DS_detrended + DS_y_epic_am_det.sel(time = slice(2005, 2015)).mean('time'), name= DS_epic['yield-soy-noirr'].name, attrs = DS_epic['yield-soy-noirr'].attrs)
    elif detrend == 'mean future':
        DS_y_epic_proj_am_det = xr.DataArray( DS_detrended + DS_fit_mean.mean('time'), name= DS_epic['yield-soy-noirr'].name, attrs = DS_epic['yield-soy-noirr'].attrs)
    elif detrend == 'no detrend':
        DS_y_epic_proj_am_det = DS_epic['yield-soy-noirr']
    
    
    df_y_epic_proj_am = DS_y_epic_proj_am_det.to_dataframe().dropna()
    df_y_epic_proj_am = df_y_epic_proj_am.reorder_levels(['time','lat','lon']).sort_index()
    
    df_y_epic_proj_am = df_y_epic_proj_am.where(df_feature_proj_6mon_am_det['prcptot_3'] > -100).dropna()
    df_feature_proj_6mon_am_det = df_feature_proj_6mon_am_det.where(df_y_epic_proj_am['yield-soy-noirr'] > -100).dropna()
    
    df_input_hybrid_fut_test_2 = pd.concat([df_y_epic_proj_am, df_feature_proj_6mon_am_det], axis = 1 )
    
    DS_input_hybrid = xr.Dataset.from_dataframe(df_input_hybrid_fut_test_2)
    # Reindex to avoid missnig coordinates and dimension values
    DS_input_hybrid_2 = rearrange_latlot(DS_input_hybrid)

    return DS_input_hybrid_2

DS_output_hybrid_ukesm_585_am_late, difference_ukesm_585_am_late = shift_comparison(DS_y_epic_proj_ukesm_585_am_late, DS_feature_proj_ukesm_585_am_late)
DS_output_hybrid_gfdl_585_am_late, difference_gfdl_585_am_late = shift_comparison(DS_y_epic_proj_gfdl_585_am_late, DS_feature_proj_gfdl_585_am_late)
DS_output_hybrid_ipsl_585_am_late, difference_ipsl_585_am_late = shift_comparison(DS_y_epic_proj_ipsl_585_am_late, DS_feature_proj_ipsl_585_am_late)

DS_output_hybrid_ukesm_585_am_mid, difference_ukesm_585_am_mid = shift_comparison(DS_y_epic_proj_ukesm_585_am_mid, DS_feature_proj_ukesm_585_am_mid)
DS_output_hybrid_gfdl_585_am_mid, difference_gfdl_585_am_mid = shift_comparison(DS_y_epic_proj_gfdl_585_am_mid, DS_feature_proj_gfdl_585_am_mid)
DS_output_hybrid_ipsl_585_am_mid, difference_ipsl_585_am_mid = shift_comparison(DS_y_epic_proj_ipsl_585_am_mid, DS_feature_proj_ipsl_585_am_mid)

################################
DS_output_hybrid_ukesm_126_am_late, difference_ukesm_126_am_late = shift_comparison(DS_y_epic_proj_ukesm_126_am_late, DS_feature_proj_ukesm_126_am_late)
DS_output_hybrid_gfdl_126_am_late, difference_gfdl_126_am_late = shift_comparison(DS_y_epic_proj_gfdl_126_am_late, DS_feature_proj_gfdl_126_am_late)
DS_output_hybrid_ipsl_126_am_late, difference_ipsl_126_am_late = shift_comparison(DS_y_epic_proj_ipsl_126_am_late, DS_feature_proj_ipsl_126_am_late)

DS_output_hybrid_ukesm_126_am_mid, difference_ukesm_126_am_mid = shift_comparison(DS_y_epic_proj_ukesm_126_am_mid, DS_feature_proj_ukesm_126_am_mid)
DS_output_hybrid_gfdl_126_am_mid, difference_gfdl_126_am_mid = shift_comparison(DS_y_epic_proj_gfdl_126_am_mid, DS_feature_proj_gfdl_126_am_mid)
DS_output_hybrid_ipsl_126_am_mid, difference_ipsl_126_am_mid = shift_comparison(DS_y_epic_proj_ipsl_126_am_mid, DS_feature_proj_ipsl_126_am_mid)

# Compare temperrature in 3C with historical

def gw_levels_run(DS_epic, df_clim, df_gw):
    
    if np.max(df_gw['tas']) >= 2:
        ref_year_2c = df_gw[df_gw['tas'] >= 2].index[0].year
        DS_2C = DS_epic.sel(time = slice(ref_year_2c - 10, ref_year_2c + 10) )
        
    if np.max(df_gw['tas']) >= 3:
        ref_year_3c = df_gw[df_gw['tas'] >= 3].index[0].year 
        DS_3C = DS_epic.sel(time = slice(ref_year_3c - 10, ref_year_3c + 10) )

    return DS_2C if "DS_2C" in locals() else None, DS_3C if "DS_3C" in locals() else None


DS_y_epic_proj_ukesm_585_am_2C, DS_y_epic_proj_ukesm_585_am_3C = gw_levels_run(DS_y_epic_proj_ukesm_585_am, df_feature_proj_ukesm_585_am, df_ukesm_85_5y)
DS_y_epic_proj_ukesm_126_am_2C, DS_y_epic_proj_ukesm_126_am_3C = gw_levels_run(DS_y_epic_proj_ukesm_126_am, df_feature_proj_ukesm_126_am, df_ukesm_26_5y)

DS_y_epic_proj_gfdl_585_am_2C, DS_y_epic_proj_gfdl_585_am_3C = gw_levels_run(DS_y_epic_proj_gfdl_585_am, df_feature_proj_gfdl_585_am, df_gfdl_85_5y)
DS_y_epic_proj_gfdl_126_am_2C, DS_y_epic_proj_gfdl_126_am_3C = gw_levels_run(DS_y_epic_proj_gfdl_126_am, df_feature_proj_gfdl_126_am, df_gfdl_26_5y)

DS_y_epic_proj_ipsl_585_am_2C, DS_y_epic_proj_ipsl_585_am_3C = gw_levels_run(DS_y_epic_proj_ipsl_585_am, df_feature_proj_ipsl_585_am, df_ipsl_85_5y)
DS_y_epic_proj_ipsl_126_am_2C, DS_y_epic_proj_ipsl_126_am_3C = gw_levels_run(DS_y_epic_proj_ipsl_126_am, df_feature_proj_ipsl_126_am, df_ipsl_26_5y)

projection_data_clim_diff_3C_2012 = (DS_feature_proj_ukesm_585_am.where(DS_y_epic_proj_ukesm_585_am_3C['yield-soy-noirr'] > - 1000).mean('time')) - (DS_input_2.sel(time = 2012))
projection_data_clim_diff_3C_2012.to_netcdf('projection_data_clim_anomaly_3C_2012.nc')

plot_2d_am_map( projection_data_clim_diff_3C_2012['txm_2'], colormap = 'RdBu_r', vmin = -4, vmax = 4, title = "Temperature difference 3C and 2012 temperature" )
plot_2d_am_map( projection_data_clim_diff_3C_2012['prcptot_2'], colormap = 'RdBu_r', vmin = -4, vmax = 4, title = "Temperature difference 3C and 2012 temperature" )


#%% 3) Compare the shift in 2012 with the shifts in those years

#### Trends
DS_hybrid_trend_gfdl_26 = xr.load_dataset("output_models_am/hybrid_trends/hybrid_trend_gfdl-esm4_ssp126_default_yield_soybean_2015_2100.nc")
DS_hybrid_trend_gfdl_85 = xr.load_dataset("output_models_am/hybrid_trends/hybrid_trend_gfdl-esm4_ssp585_default_yield_soybean_2015_2100.nc")
DS_hybrid_trend_ipsl_26 = xr.load_dataset("output_models_am/hybrid_trends/hybrid_trend_ipsl-cm6a-lr_ssp126_default_yield_soybean_2015_2100.nc")
DS_hybrid_trend_ipsl_85 = xr.load_dataset("output_models_am/hybrid_trends/hybrid_trend_ipsl-cm6a-lr_ssp585_default_yield_soybean_2015_2100.nc")
DS_hybrid_trend_ukesm_26 = xr.load_dataset("output_models_am/hybrid_trends/hybrid_trend_ukesm1-0-ll_ssp126_default_yield_soybean_2015_2100.nc")
DS_hybrid_trend_ukesm_85 = xr.load_dataset("output_models_am/hybrid_trends/hybrid_trend_ukesm1-0-ll_ssp585_default_yield_soybean_2015_2100.nc")
# Merge all scenarios
DS_hybrid_trend_all = xr.merge([DS_hybrid_trend_gfdl_26.rename({'yield-soy-noirr':'GFDL-esm4_1-2.6'}),DS_hybrid_trend_gfdl_85.rename({'yield-soy-noirr':'GFDL-esm4_5-8.5'}),
                          DS_hybrid_trend_ipsl_26.rename({'yield-soy-noirr':'IPSL-cm6a-lr_1-2.6'}),DS_hybrid_trend_ipsl_85.rename({'yield-soy-noirr':'IPSL-cm6a-lr_5-8.5'}),
                          DS_hybrid_trend_ukesm_26.rename({'yield-soy-noirr':'UKESM1-0-ll_1-2.6'}),DS_hybrid_trend_ukesm_85.rename({'yield-soy-noirr':'UKESM1-0-ll_5-8.5'})])



def ds_levels_hybrid_selection(DS):
    DS_mid = DS.sel(time = slice(2035,2065))
    DS_late = DS.sel(time = slice(2066,2096))
    
    return DS_mid, DS_late


DS_hybrid_trend_all_weighted = weighted_conversion(DS_hybrid_trend_all, DS_harvest_area_fut, name_ds = 'yield-soy-noirr')

DS_hybrid_trend_all_mid, DS_hybrid_trend_all_late = ds_levels_hybrid_selection(DS_hybrid_trend_all_weighted)

df_hybrid_trend_all_mid = DS_hybrid_trend_all_mid.to_dataframe().mean().to_frame('value')
df_hybrid_trend_all_mid['GCM']= df_hybrid_trend_all_mid.index.str.split('_').str[0]
df_hybrid_trend_all_mid['SSP'] = df_hybrid_trend_all_mid.index.str.split('_').str[1]
df_hybrid_trend_all_mid['Time period'] = 'Mid century'
df_hybrid_trend_all_mid['value'] = (df_hybrid_trend_all_mid['value'] - DS_output_hybrid_hist_weighted['yield-soy-noirr'].mean().values.item())/ DS_output_hybrid_hist_weighted['yield-soy-noirr'].mean().values.item()

df_hybrid_trend_all_late = DS_hybrid_trend_all_late.to_dataframe().mean().to_frame('value')
df_hybrid_trend_all_late['GCM']= df_hybrid_trend_all_late.index.str.split('_').str[0]
df_hybrid_trend_all_late['SSP'] = df_hybrid_trend_all_late.index.str.split('_').str[1]
df_hybrid_trend_all_late['Time period'] = 'Late century'
df_hybrid_trend_all_late['value'] = (df_hybrid_trend_all_late['value'] - DS_output_hybrid_hist_weighted['yield-soy-noirr'].mean().values.item())/ DS_output_hybrid_hist_weighted['yield-soy-noirr'].mean().values.item()

df_hybrid_trend_all_merged = pd.concat([df_hybrid_trend_all_mid, df_hybrid_trend_all_late])
df_hybrid_trend_all_merged['value'] = df_hybrid_trend_all_merged['value']*100

df_hybrid_trend_all_126 = df_hybrid_trend_all_merged.where(df_hybrid_trend_all_merged['SSP'] == '1-2.6').dropna()
df_hybrid_trend_all_585 = df_hybrid_trend_all_merged.where(df_hybrid_trend_all_merged['SSP'] == '5-8.5').dropna()


# Plot shifters
fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2, figsize=(12,6),sharey = True )

ax1.axhline(y = (shift_2012_value/DS_output_hybrid_hist_weighted['yield-soy-noirr'].mean().values.item())*100, linestyle = 'dashed', color = 'k', label = '2012 event', linewidth = 4 )
ax2.axhline(y = (shift_2012_value/DS_output_hybrid_hist_weighted['yield-soy-noirr'].mean().values.item())*100, linestyle = 'dashed', color = 'k', label = '2012 event', linewidth = 4 )
sns.barplot(x =df_hybrid_trend_all_126['Time period'], hue = df_hybrid_trend_all_126['GCM'], y = df_hybrid_trend_all_126['value'], ax=ax1)
sns.barplot(x = df_hybrid_trend_all_585['Time period'], hue = df_hybrid_trend_all_585['GCM'], y = df_hybrid_trend_all_585['value'], ax=ax2)

ax1.set_title('SSP 1-2.6')
# ax1.set_ylim(-30, 30)
ax2.set_title('SSP 5-8.5')
ax1.set_ylabel('difference to 2012 (%)')
ax2.set_ylabel('')
ax1.get_legend().remove()
ax2.get_legend().remove()
ax1.set_xlabel('')
ax2.set_xlabel('')
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.7, 0), ncol=3, frameon=False)
fig.suptitle('Difference between the mean yield and the 2012 event')
plt.tight_layout()
plt.show()



### Just climate variability and shocks

difference_shocks_585 = pd.DataFrame([[difference_ukesm_585_am_mid.values.item(), difference_gfdl_585_am_mid.values.item(), difference_ipsl_585_am_mid.values.item()], 
              [difference_ukesm_585_am_late.values.item(), difference_gfdl_585_am_late.values.item(), difference_ipsl_585_am_late.values.item()]],
             index = ['Mid century','Late century'], columns = ['UKESM1-0-ll','GFDL-esm4','IPSL-cm6a-lr'])
difference_shocks_585 = difference_shocks_585*100

difference_shocks_126 = pd.DataFrame([[difference_ukesm_126_am_mid.values.item(), difference_gfdl_126_am_mid.values.item(), difference_ipsl_126_am_mid.values.item()], 
              [difference_ukesm_126_am_late.values.item(), difference_gfdl_126_am_late.values.item(), difference_ipsl_126_am_late.values.item()]],
             index = ['Mid century','Late century'], columns = ['UKESM1-0-ll','GFDL-esm4','IPSL-cm6a-lr'])
difference_shocks_126 = difference_shocks_126*100

shock_585 = difference_shocks_585.melt(ignore_index= False)
shock_126 = difference_shocks_126.melt(ignore_index= False)
shocks = pd.concat([shock_126, shock_585])

# Plot shifters
fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2, figsize=(12,6),sharey = True )

ax1.axhline(y = (shift_2012_value/DS_output_hybrid_hist_weighted['yield-soy-noirr'].mean().values.item())*100, linestyle = 'dashed', color = 'k', label = '2012 event', linewidth = 4 )
ax2.axhline(y = (shift_2012_value/DS_output_hybrid_hist_weighted['yield-soy-noirr'].mean().values.item())*100, linestyle = 'dashed', color = 'k', label = '2012 event', linewidth = 4 )
sns.barplot(x = shock_126.index, hue = shock_126['variable'], y = shock_126['value'], ax=ax1, hue_order = ['GFDL-esm4', 'IPSL-cm6a-lr', 'UKESM1-0-ll'])
sns.barplot(x = shock_585.index, hue = shock_585['variable'], y = shock_585['value'], ax=ax2, hue_order = ['GFDL-esm4', 'IPSL-cm6a-lr', 'UKESM1-0-ll'])
ax1.set_title('RCP 2.6')
ax1.set_ylim(-50,0)
ax2.set_title('RCP 8.5')
ax1.set_ylabel('Yield loss to historical climatology (%)')
ax2.set_ylabel('')
ax1.get_legend().remove()
ax2.get_legend().remove()
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.7, 0), ncol=4, frameon=False)
fig.suptitle('a) Difference between the largest shock in the future and the 2012 event')
plt.tight_layout()
plt.show()


# Add 2012 to the mean values

df_hybrid_trend_comb_mid = DS_hybrid_trend_all_mid.to_dataframe().mean().to_frame('value') + shift_2012_value
df_hybrid_trend_comb_mid['GCM']= df_hybrid_trend_comb_mid.index.str.split('_').str[0]
df_hybrid_trend_comb_mid['SSP'] = df_hybrid_trend_comb_mid.index.str.split('_').str[1]
df_hybrid_trend_comb_mid['Time period'] = 'Mid century'
df_hybrid_trend_comb_mid['value'] = (df_hybrid_trend_comb_mid['value'] - DS_output_hybrid_hist_weighted['yield-soy-noirr'].mean().values.item())/ DS_output_hybrid_hist_weighted['yield-soy-noirr'].mean().values.item()

df_hybrid_trend_comb_late = DS_hybrid_trend_all_late.to_dataframe().mean().to_frame('value') + shift_2012_value
df_hybrid_trend_comb_late['GCM']= df_hybrid_trend_comb_late.index.str.split('_').str[0]
df_hybrid_trend_comb_late['SSP'] = df_hybrid_trend_comb_late.index.str.split('_').str[1]
df_hybrid_trend_comb_late['Time period'] = 'Late century'
df_hybrid_trend_comb_late['value'] = (df_hybrid_trend_comb_late['value'] - DS_output_hybrid_hist_weighted['yield-soy-noirr'].mean().values.item())/ DS_output_hybrid_hist_weighted['yield-soy-noirr'].mean().values.item()

df_hybrid_trend_comb_merged = pd.concat([df_hybrid_trend_comb_mid, df_hybrid_trend_comb_late])
df_hybrid_trend_comb_merged['value'] = df_hybrid_trend_comb_merged['value']*100

df_hybrid_trend_comb_126 = df_hybrid_trend_comb_merged.where(df_hybrid_trend_comb_merged['SSP'] == '1-2.6').dropna()
df_hybrid_trend_comb_585 = df_hybrid_trend_comb_merged.where(df_hybrid_trend_comb_merged['SSP'] == '5-8.5').dropna()


# Plot shifters
fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2, figsize=(12,6),sharey = True )

ax1.axhline(y = (shift_2012_value/DS_output_hybrid_hist_weighted['yield-soy-noirr'].mean().values.item())*100, linestyle = 'dashed', color = 'k', label = '2012 event', linewidth = 4 )
ax2.axhline(y = (shift_2012_value/DS_output_hybrid_hist_weighted['yield-soy-noirr'].mean().values.item())*100, linestyle = 'dashed', color = 'k', label = '2012 event', linewidth = 4 )
sns.barplot(x = df_hybrid_trend_comb_126['Time period'], hue = df_hybrid_trend_comb_126['GCM'], y = df_hybrid_trend_comb_126['value'], ax=ax1, hue_order = ['GFDL-esm4', 'IPSL-cm6a-lr', 'UKESM1-0-ll'])
sns.barplot(x = df_hybrid_trend_comb_585['Time period'], hue = df_hybrid_trend_comb_585['GCM'], y = df_hybrid_trend_comb_585['value'], ax=ax2, hue_order = ['GFDL-esm4', 'IPSL-cm6a-lr', 'UKESM1-0-ll'])
ax1.set_title('RCP 2.6')
ax1.set_ylim(-50,0)
ax2.set_title('RCP 8.5')
ax1.set_ylabel('Yield loss to historical climatology (%)')
ax2.set_ylabel('')
ax1.get_legend().remove()
ax2.get_legend().remove()
ax1.set_xlabel('')
ax2.set_xlabel('')
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.7, 0), ncol=4, frameon=False)
fig.suptitle('b) 2012 yield anomaly in future scenarios')
plt.tight_layout()
plt.show()


# 4) Take the trend and add the shifters
# Trends corresponding to mean climate in each block



# 5) Analysis


# Add anomaly to trends



