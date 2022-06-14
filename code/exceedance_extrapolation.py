# -*- coding: utf-8 -*-
"""
EXTRAPOLTAION TEST

CHECK FOR THE AMOUNT OF TIMES EACH VARIABLES IS OUT OF THE CALIBRATION RANGE.
RESULTS IN %.

Created on Tue Nov  2 17:00:43 2021

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
import matplotlib as mpl
import pickle
import joblib
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error,  explained_variance_score

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['figure.dpi'] = 144
mpl.rcParams.update({'font.size': 14})

# #%% Load the pipeline first:
# full_model_hyb2 = joblib.load('sklearn_pipeline_us_BN.pkl')

# # Then, load the Keras model:
# full_model_hyb2['estimator'].model = load_model('keras_model_us_BN.h5')

# # Data for testing: 
# X = pd.read_csv('dataset_input_hybrid_forML.csv', index_col=[0,1,2],)
# y = pd.read_csv('dataset_obs_yield_forML.csv', index_col=[0,1,2],)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# y_pred = full_model_hyb2.predict(X_test)

# # report performance
# print("R2 on test set:", round(r2_score(y_test, y_pred),2))
# print("Var score on test set:", round(explained_variance_score(y_test, y_pred),2))
# print("MAE on test set:", round(mean_absolute_error(y_test, y_pred),5))
# print("RMSE on test set:",round(mean_squared_error(y_test, y_pred, squared=False),5))


#%% functions 

# Detrend Dataset
def detrend_dataset(DS, deg = 'free', dim = 'time', print_res = True, mean_data = None):
            
    if deg == 'free':
        da_list = []
        for feature in list(DS.keys()):
            da = DS[feature]
            print(feature)
            
            if mean_data is None:
                mean_dataarray = da.mean('time')
            else:
                mean_dataarray = mean_data[feature].mean('time') #da.mean('time') - ( da.mean() - mean_data[feature].mean() )
            
            da_zero_mean = da.where( da < np.nanmin(da.values), other = 0 )
    
            dict_res = {}
            for degree in [1,2]:
                # detrend along a single dimension
                p = da.polyfit(dim=dim, deg=degree)
                fit = xr.polyval(da[dim], p.polyfit_coefficients)
                
                da_det = da - fit
                
                res_detrend = np.nansum((da_zero_mean.mean(['lat','lon'])-da_det.mean(['lat','lon']))**2)
                dict_res.update({degree:res_detrend})
            if print_res == True:
                print(dict_res)
            deg = min(dict_res, key=dict_res.get) # minimum degree   
            
            # detrend along a single dimension
            print('Chosen degree is ', deg)
            p = da.polyfit(dim=dim, deg=deg)
            fit = xr.polyval(da[dim], p.polyfit_coefficients)
        
            da_det = da - fit + mean_dataarray
            da_det.name = feature
            da_list.append(da_det)
        DS_det = xr.merge(da_list) 
    
    else:       
        px= DS.polyfit(dim='time', deg=deg)
        fitx = xr.polyval(DS['time'], px)
        dict_name = dict(zip(list(fitx.keys()), list(DS.keys())))
        fitx = fitx.rename(dict_name)
        DS_det  = (DS - fitx) + mean_data
        
    return DS_det


# Different ways to detrend, select the best one
def detrend_dim(da, dim, deg = 'free', print_res = True):        
    if deg == 'free':
        
        da_zero_mean = da.where( da < np.nanmin(da.values), other = 0 )

        dict_res = {}
        for degree in [1,2]:
            # detrend along a single dimension
            p = da.polyfit(dim=dim, deg=degree)
            fit = xr.polyval(da[dim], p.polyfit_coefficients)
            
            da_det = da - fit
            res_detrend = np.nansum((da_zero_mean.mean(['lat','lon'])-da_det.mean(['lat','lon']))**2)
            dict_res_in = {degree:res_detrend}
            dict_res.update(dict_res_in)
        if print_res == True:
            print(dict_res)
        deg = min(dict_res, key=dict_res.get) # minimum degree        
    
    # detrend along a single dimension
    print('Chosen degree is ', deg)
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    
    da_det = da - fit   
    return da_det

# Convert datetime values (ns) into days
def timedelta_to_int(DS, var):
    da_timedelta = DS[var].dt.days
    da_timedelta = da_timedelta.rename(var)
    da_timedelta.attrs["units"] = 'days'    
    return da_timedelta



#%% FUNCTION TO GENERATE PROJECTIONS BASED ON RANDOM FOREST
def projections_generation_hybrid(model, rcp_scenario, region, hybrid_model_full, start_date, end_date, co2_scen='both'):
    DS_clim_ext_projections = xr.open_mfdataset('monthly_'+ model +'_'+ rcp_scenario + region +'/*.nc').sel(time=slice(start_date, end_date))
    
    if model == 'ukesm':
        model_full = 'ukesm1-0-ll'
    elif model == 'gfdl':
        model_full = 'gfdl-esm4'
    elif model == 'ipsl':
        model_full = 'ipsl-cm6a-lr'

### Climatic variables - Extreme weather    
    # Clean
    DS_clim_ext_projections = DS_clim_ext_projections.drop_vars('fd') # Always zero
    DS_clim_ext_projections = DS_clim_ext_projections.drop_vars('id') # Always zero
    DS_clim_ext_projections = DS_clim_ext_projections.drop_vars('time_bnds') # Always zero
    
    # Selected features
    list_features = ['prcptot', 'r10mm','txm' ]# 'dtr', 'tnm', 'txge35', 'tr', 'txm', 'tmm', 'tnn'
    DS_clim_ext_projections = DS_clim_ext_projections[list_features]
    DS_clim_ext_projections = DS_clim_ext_projections.where(DS_y_obs_us['usda_yield'].mean('time') >= -5.0 )

    
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
        DS_clim_ext_projections_combined = DS_clim_ext_projections_combined.drop('spatial_ref')
        
    DS_clim_ext_projections_us = DS_clim_ext_projections_combined.where( DS_chosen_calendar >= 0 )
    
    plot_2d_us_map(DS_clim_ext_projections_us['prcptot'].mean('time'))
    
    DS_clim_ext_projections_us_det = DS_clim_ext_projections_us #detrend_dataset(DS_clim_ext_projections_us, mean_data = DS_exclim_us_det_clip)
    DS_clim_ext_projections_us['prcptot'].mean(['lat','lon']).plot()
    DS_clim_ext_projections_us_det['prcptot'].mean(['lat','lon']).plot()
    plt.show()
    
    DS_clim_ext_projections_us['txm'].mean(['lat','lon']).plot()
    DS_clim_ext_projections_us_det['txm'].mean(['lat','lon']).plot()
    plt.show()
    
    
    # For loop along features to obtain 24 months of climatic data for each year
    list_features_reshape_shift = []
    for feature in list(DS_clim_ext_projections_us_det.keys()):
        ### Reshape
        df_test_shift = reshape_shift(DS_clim_ext_projections_us_det[feature])
        
        ### Join and change name to S for the shift values
        df_feature_reshape_shift = (df_test_shift.dropna().join(df_calendar_month_us)
                                    .rename(columns={'plant':'s'}))
        # Move 
        col = df_feature_reshape_shift.pop("s")
        df_feature_reshape_shift.insert(0, col.name, col)
        df_feature_reshape_shift[['s']].isna().sum()
        nan_rows = df_feature_reshape_shift[['s']][df_feature_reshape_shift[['s']].isnull().T.any()]
        if nan_rows.empty == False:
            print('Missing crop calendar values!')
        
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
    df_proj_feature_season_6mon=df_proj_feature_season_6mon.astype(float)

    # SECOND DETRENDING PART - SEASONAL
    DS_feature_proj_6mon_us = xr.Dataset.from_dataframe(df_proj_feature_season_6mon)
    DS_feature_proj_6mon_us_det = detrend_dataset(DS_feature_proj_6mon_us, deg = 'free', mean_data = DS_feature_proj_6mon_us )
    df_feature_proj_6mon_us_det = DS_feature_proj_6mon_us_det.to_dataframe().dropna()

    for feature in df_feature_proj_6mon_us_det.columns:
        df_proj_feature_season_6mon[feature].groupby('time').mean().plot(label = 'old')
        df_feature_proj_6mon_us_det[feature].groupby('time').mean().plot(label = 'detrend')
        plt.title(f'{feature}')
        plt.legend()
        plt.show()
    
    for feature in ['prcptot_3', 'prcptot_4', 'prcptot_5']:
        df_feature_proj_6mon_us_det[feature][df_feature_proj_6mon_us_det[feature] < 0] = 0

    # UPDATE DETRENDED VALUES
    df_proj_feature_season_6mon = df_feature_proj_6mon_us_det
    
    #%% EPIC projections
    
    def epic_projections_function_co2(co2_scenario):
        DS_y_epic_proj = xr.open_dataset("epic-iiasa_"+ model_full +"_w5e5_"+rcp_scenario+"_2015soc_"+co2_scenario+"_yield-soy-noirr_global_annual_2015_2100.nc", decode_times=False)
        # Convert time unit
        units, reference_date = DS_y_epic_proj.time.attrs['units'].split('since')
        DS_y_epic_proj['time'] = pd.date_range(start=' 2015-01-01, 00:00:00', periods=DS_y_epic_proj.sizes['time'], freq='YS')
        DS_y_epic_proj['time'] = DS_y_epic_proj['time'].dt.year
        
        DS_y_epic_proj['yield-soy-noirr'].mean(['lat', 'lon']).plot()
        plt.title( model + "_"+ rcp_scenario+"_"+co2_scenario)
        plt.show()
        
        DS_y_epic_proj_us = DS_y_epic_proj.where(DS_y_obs_us['usda_yield'].mean('time') >= -5.0 )
        DS_y_epic_proj_us = DS_y_epic_proj_us.where(DS_clim_ext_projections_us['prcptot'].mean('time') >= -100.0 )
        
        DS_y_epic_proj_us_det = xr.DataArray( detrend_dim(DS_y_epic_proj_us['yield-soy-noirr'], 'time') + DS_y_epic_us_det.mean('time'), name= DS_y_epic_proj_us['yield-soy-noirr'].name, attrs = DS_y_epic_proj_us['yield-soy-noirr'].attrs)
        
        plot_2d_us_map(DS_y_epic_proj_us_det.mean('time'))
        
        DS_y_epic_proj_us['yield-soy-noirr'].mean(['lat', 'lon']).plot()
        DS_y_epic_proj_us_det.mean(['lat', 'lon']).plot()
        plt.title('Detrend'+ "_"+ model + "_"+ rcp_scenario+"_"+co2_scenario)
        plt.show()
        
        df_y_epic_proj_us = DS_y_epic_proj_us_det.to_dataframe().dropna()
        df_y_epic_proj_us = df_y_epic_proj_us.reorder_levels(['time','lat','lon']).sort_index()
        
        #%% HYBRID
        
        df_hybrid_proj_2 = pd.concat([df_y_epic_proj_us, df_proj_feature_season_6mon], axis = 1 )
        df_hybrid_proj_test_2 = df_hybrid_proj_2.query("time>=2015 and time <= 2100")
                
        # Save this for future operations:
        # df_hybrid_proj_test_2.to_csv('dataset_future_input_hybrid_forML.csv')

        # Predicting hybrid results
        df_prediction_proj_2 = df_y_epic_proj_us.query("time>=2015 and time <= 2100").copy() 
        predic_model_test_2 = hybrid_model_full.predict(df_hybrid_proj_test_2.values).copy()
        df_prediction_proj_2.loc[:,'yield-soy-noirr'] = predic_model_test_2
        
        DS_hybrid_proj_2 = xr.Dataset.from_dataframe(df_prediction_proj_2)
        DS_hybrid_proj_2 = DS_hybrid_proj_2.sortby('lat')
        DS_hybrid_proj_2 = DS_hybrid_proj_2.sortby('lon')
        
        plt.plot(DS_hybrid_proj_2.time.values, DS_hybrid_proj_2['yield-soy-noirr'].mean(['lat','lon']), label = 'Hybrid')
        plt.plot(DS_y_epic_proj_us_det.time.values, DS_y_epic_proj_us_det.mean(['lat','lon']), label = 'EPIC pure')
        plt.title('Hybrid'+ "_"+ model + "_"+rcp_scenario+"_"+co2_scenario)
        plt.legend()
        plt.show()
        
        plt.plot(DS_hybrid_proj_2.time.values, DS_hybrid_proj_2['yield-soy-noirr'].mean(['lat','lon'])/ DS_hybrid_proj_2['yield-soy-noirr'].mean(), label = 'Hybrid')
        plt.plot(DS_y_epic_proj_us_det.time.values, DS_y_epic_proj_us_det.mean(['lat','lon'])/DS_y_epic_proj_us_det.mean(), label = 'EPIC pure')
        plt.title('Hybrid'+ "_"+ model + "_"+rcp_scenario+"_"+co2_scenario)
        plt.ylabel('Shock (0-1)')
        plt.legend()
        plt.show()
               
        plot_2d_us_map(DS_hybrid_proj_2['yield-soy-noirr'].sel(time=2084))
        plot_2d_us_map(DS_hybrid_proj_2['yield-soy-noirr'].mean('time'))
        
        #%% EPIC SIM
        df_predic_epic_test = df_y_epic_proj_us.copy()
        predic_epic_test = full_model_epic_us.predict(df_y_epic_proj_us.values)
        df_predic_epic_test.loc[:,'yield-soy-noirr'] = full_model_epic_us.predict(df_y_epic_proj_us.values)
        
        DS_pred_epic_proj = xr.Dataset.from_dataframe(df_predic_epic_test)
        DS_pred_epic_proj = DS_pred_epic_proj.sortby('lat')
        DS_pred_epic_proj = DS_pred_epic_proj.sortby('lon')
        
        # CLIMATIC model
        df_pred_clim_proj_test = df_y_epic_proj_us.copy()
        df_pred_clim_proj_test.loc[:,'yield-soy-noirr'] = full_model_exclim_dyn_us.predict(df_proj_feature_season_6mon.values)
        
        DS_pred_clim_proj = xr.Dataset.from_dataframe(df_pred_clim_proj_test)
        DS_pred_clim_proj = DS_pred_clim_proj.sortby('lat')
        DS_pred_clim_proj = DS_pred_clim_proj.sortby('lon')
        
        # WEIGHTING SCHEME
        DS_harvest_area_globiom =  xr.open_dataset("../../paper_hybrid_agri/data/americas_mask_ha.nc", decode_times=False).sel(longitude=slice(-120,-60), latitude=slice(50,30)).rename({'latitude': 'lat', 'longitude': 'lon'}) # xr.open_dataset('../../paper_hybrid_agri/data/soy_harvest_area_globiom_05x05_2b.nc').mean('time')
        DS_harvest_area_globiom['harvest_area'] = DS_harvest_area_globiom['annual_area_harvested_rfc_crop08_ha_30mn'].where(DS_hybrid_proj_2['yield-soy-noirr'].mean('time')>-2)
        
        total_area =  DS_harvest_area_globiom['harvest_area'].sum(['lat','lon'])
        
        DS_hybrid_weighted = ((DS_hybrid_proj_2['yield-soy-noirr'] * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'yield-soy-noirr')
        DS_epic_weighted = ((DS_pred_epic_proj['yield-soy-noirr'] * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'yield-soy-noirr')
        DS_clim_weighted = ((DS_pred_clim_proj['yield-soy-noirr'] * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'yield-soy-noirr')
        DS_epic_pure_weighted = ((DS_y_epic_proj_us_det * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'yield-soy-noirr')
        
        plt.plot(DS_hybrid_proj_2['yield-soy-noirr'].mean(['lat','lon']), label = 'unweighted')
        plt.plot(DS_hybrid_weighted['yield-soy-noirr'].sum(['lat','lon']), label = 'weighted')
        plt.ylabel('yield')
        plt.xlabel('years')
        plt.legend()
        plt.show()
        
        
        plt.plot(DS_y_epic_proj_us_det.time.values, DS_y_epic_proj_us_det.mean(['lat','lon']), label = 'Pure EPIC')
        plt.plot(DS_pred_epic_proj.time.values, DS_pred_epic_proj["yield-soy-noirr"].mean(['lat','lon']), label = 'EPIC-RF')
        plt.plot(DS_hybrid_proj_2.time.values, DS_hybrid_proj_2["yield-soy-noirr"].mean(['lat','lon']), label = 'Hybrid-RF')
        plt.title('Non-weighted comparison')
        plt.legend()
        plt.show()
        
        plt.plot(DS_y_epic_proj_us_det.time.values, DS_y_epic_proj_us_det.mean(['lat','lon']), label = 'Pure EPIC')
        plt.plot(DS_pred_epic_proj.time.values, DS_epic_weighted["yield-soy-noirr"].sum(['lat','lon']), label = 'EPIC-RF')
        plt.plot(DS_hybrid_proj_2.time.values, DS_hybrid_weighted['yield-soy-noirr'].sum(['lat','lon']), label = 'Hybrid-RF')
        plt.title('Weighted comparison')
        plt.legend()
        plt.show()
        
        plt.plot(DS_pred_epic_proj.time.values, DS_epic_weighted["yield-soy-noirr"].sum(['lat','lon'])/np.mean(DS_epic_weighted["yield-soy-noirr"].sum(['lat','lon'])), label = 'EPIC-RF')
        plt.plot(DS_pred_clim_proj.time.values, DS_clim_weighted["yield-soy-noirr"].sum(['lat','lon'])/np.mean(DS_clim_weighted["yield-soy-noirr"].sum(['lat','lon'])), label = 'CLIM-RF')
        plt.plot(DS_hybrid_proj_2.time.values, DS_hybrid_weighted['yield-soy-noirr'].sum(['lat','lon'])/np.mean(DS_hybrid_weighted["yield-soy-noirr"].sum(['lat','lon'])), label = 'Hybrid')
        plt.title("Weighted"+ "_" + model + "_"+ rcp_scenario+"_"+co2_scenario)
        plt.ylabel("Shock (%)")
        plt.legend()
        plt.show()
        
        # FINAL COMPARISON EPIC WITH HYBRID
        plt.plot(DS_hybrid_proj_2.time.values, DS_hybrid_weighted['yield-soy-noirr'].sum(['lat','lon'])/np.mean(DS_hybrid_weighted["yield-soy-noirr"].sum(['lat','lon'])), label = 'Hybrid')
        plt.plot(DS_epic_pure_weighted.time.values, DS_epic_pure_weighted['yield-soy-noirr'].sum(['lat','lon'])/np.mean(DS_epic_pure_weighted["yield-soy-noirr"].sum(['lat','lon'])), label = 'EPIC PURE')
        plt.title("Weighted"+ "_" + model + "_"+ rcp_scenario+"_"+co2_scenario)
        plt.ylabel("Shock (%)")
        plt.legend()
        plt.show()
        
        
        # difference shock
        weighted_timeseries = DS_hybrid_weighted['yield-soy-noirr'].sum(['lat','lon'])/np.mean(DS_hybrid_weighted["yield-soy-noirr"].sum(['lat','lon']))
        weighted_timeseries_min = weighted_timeseries.idxmin()       
        plot_2d_us_map(DS_y_epic_proj_us_det.sel(time=weighted_timeseries_min)/DS_y_epic_proj_us_det.mean('time'))
        plot_2d_us_map(DS_hybrid_proj_2['yield-soy-noirr'].sel(time=weighted_timeseries_min)/DS_hybrid_proj_2["yield-soy-noirr"].mean('time'))
        plot_2d_us_map(DS_hybrid_proj_2['yield-soy-noirr'].sel(time=weighted_timeseries_min)/DS_hybrid_proj_2["yield-soy-noirr"].mean('time') - DS_y_epic_proj_us_det.sel(time=weighted_timeseries_min)/DS_y_epic_proj_us_det.mean('time'))
                                                                      
        
        
        #%% Extrapolation tests

        def exceedance_freq(exceedance_data, ref_data):
            # Test how many times variables are extrapolated outside of the training range for each variable.
            test_max = exceedance_data > ref_data.max(axis=0)
            exceedance_percentage_max = test_max.sum(axis=0)*100/len(exceedance_data)
            
            test_min = exceedance_data < ref_data.min(axis=0)
            exceedance_percentage_min = test_min.sum(axis=0)*100/len(exceedance_data)
            
            max_min = pd.concat([exceedance_percentage_max, exceedance_percentage_min], axis = 1)    
            max_min.columns = [model_full + rcp_scenario + co2_scenario+'_prc_above', model_full + rcp_scenario + co2_scenario+'_prc_below']
        
            return max_min
        
        df_hybrid_us_2 = df_hybrid_us.copy()   
        df_hybrid_us_2 = df_hybrid_us_2.rename(columns={'yield':'yield-soy-noirr'})
        max_min_US = exceedance_freq(df_hybrid_proj_test_2, df_hybrid_us_2)

        df_hybrid_proj_test_2 = df_hybrid_proj_test_2.rename(columns = {f'yield-soy-noirr':f'yield-soy-noirr_{co2_scenario}'})
        df_prediction_proj_2 = df_prediction_proj_2.rename(columns = {f'yield-soy-noirr':f'yield-soy-noirr_{co2_scenario}'})
        return max_min_US, df_hybrid_proj_test_2, df_prediction_proj_2
    
    if co2_scen == 'both':
        max_min_US_default, df_hybrid_proj_test_2_default, df_prediction_proj_2_default = epic_projections_function_co2(co2_scenario = 'default')
        max_min_US_2015co2, df_hybrid_proj_test_2_2015co2, df_prediction_proj_2_2015co2 = epic_projections_function_co2(co2_scenario = '2015co2')
        
        max_min_US_both = pd.concat([max_min_US_default, max_min_US_2015co2], axis = 1)    
        df_hybrid_proj_test_2_both = pd.concat([df_hybrid_proj_test_2_default, df_hybrid_proj_test_2_2015co2['yield-soy-noirr_2015co2']], axis = 1)    
        df_prediction_proj_2_both = pd.concat([df_prediction_proj_2_default['yield-soy-noirr_default'], df_prediction_proj_2_2015co2['yield-soy-noirr_2015co2']], axis = 1)    
    
    else:
        co2_scenario = co2_scen
        max_min_US_scen = epic_projections_function_co2(co2_scenario)
    
    return max_min_US_both, df_hybrid_proj_test_2_both, df_prediction_proj_2_both

#%% Run functions to determine exceedance levels of extrapolation
# ----------------------------------------------------------------------------------------------

max_min_US_ukesm_585,df_hybrid_proj_us_ukesm_585, df_prediction_proj_us_ukesm_585 = projections_generation_hybrid(model = 'ukesm', rcp_scenario = 'ssp585', region = "_us", hybrid_model_full = full_model_hyb2_am2, start_date='01-01-2015', end_date='31-12-2100', co2_scen = 'both')
max_min_US_ukesm_126,df_hybrid_proj_us_ukesm_126, df_prediction_proj_us_ukesm_126 = projections_generation_hybrid(model = 'ukesm', rcp_scenario = 'ssp126', region = "_us", hybrid_model_full = full_model_hyb2_am2, start_date='01-01-2015', end_date='31-12-2100', co2_scen = 'both')

max_min_US_gfdl_585,df_hybrid_proj_gfdl_585, df_prediction_proj_gfdl_585 = projections_generation_hybrid(model = 'gfdl', rcp_scenario = 'ssp585', region = "_us", hybrid_model_full = full_model_hyb2,start_date='01-01-2015', end_date='31-12-2100', co2_scen = 'both')
max_min_US_gfdl_126,df_hybrid_proj_gfdl_126, df_prediction_proj_gfdl_126 = projections_generation_hybrid(model = 'gfdl', rcp_scenario = 'ssp126', region = "_us", hybrid_model_full = full_model_hyb2,start_date='01-01-2015', end_date='31-12-2100', co2_scen = 'both')

max_min_US_ipsl_585,df_hybrid_proj_ipsl_585, df_prediction_proj_ipsl_585 = projections_generation_hybrid(model = 'ipsl', rcp_scenario = 'ssp585', region = "_us", hybrid_model_full = full_model_hyb2,start_date='01-01-2015', end_date='31-12-2100', co2_scen = 'both')
max_min_US_ipsl_126,df_hybrid_proj_ipsl_126, df_prediction_proj_ipsl_126 = projections_generation_hybrid(model = 'ipsl', rcp_scenario = 'ssp126', region = "_us", hybrid_model_full = full_model_hyb2,start_date='01-01-2015', end_date='31-12-2100', co2_scen = 'both')

final_exceedance_table = pd.concat([max_min_US_ukesm_126,max_min_US_ukesm_585,max_min_US_gfdl_126,max_min_US_gfdl_585,max_min_US_ipsl_126,max_min_US_ipsl_585], axis = 1)

### PERCENTAGE OF CASES OUTSIDE THE CALIBRATION ZONE, IF HIGH IT'S BAD
print(final_exceedance_table.max(axis=0))
print(final_exceedance_table.max(axis=1))

#%% Plots for each projections
df_hybrid_us_2 = df_hybrid_us.copy()
df_hybrid_us_2 = df_hybrid_us_2.rename(columns={'yield':'yield-soy-noirr_default'})
df_hybrid_us_2['yield-soy-noirr_2015co2'] = df_hybrid_us_2['yield-soy-noirr_default']

for feature in df_hybrid_proj_us_ukesm_585.columns:
    fig1 = plt.figure(figsize=(10,5))
    plt.axvspan(df_hybrid_us_2[feature].min(), df_hybrid_us_2[feature].max(), facecolor='0.2', alpha=0.3)
    
    plt.hist(df_hybrid_proj_us_ukesm_585[feature],bins=50, label = 'ukesm_585', alpha = 0.9)
    plt.hist(df_hybrid_proj_ukesm_126[feature],bins=50, label = 'ukesm_126', alpha = 0.9)
    plt.hist(df_hybrid_proj_ipsl_585[feature],bins=50, label = 'ipsl_585', alpha = 0.9)
    plt.hist(df_hybrid_proj_ipsl_126[feature],bins=50, label = 'ipsl_126', alpha = 0.9)
    plt.hist(df_hybrid_proj_gfdl_585[feature],bins=50, label = 'gfdl_585', alpha = 0.9)
    plt.hist(df_hybrid_proj_gfdl_126[feature],bins=50, label = 'gfdl_126', alpha = 0.9)
    plt.hist(df_hybrid_us_2[feature], bins=50, label = 'Historical')

    plt.legend(loc="upper left")
    plt.title(f'Histogram of {feature} for GCM-RCPs')
    plt.ylabel('Frequency')
    if feature in ['txm_3','txm_4','txm_5']:
        x_label = 'Temperature (°C)'
    elif feature in ['prcptot_3', 'prcptot_4', 'prcptot_5']:
        x_label = 'Precipitation (mm/month)'
    else:
        x_label = 'Yield (ton/ha)'
          
    plt.xlabel(x_label)
    # fig1.savefig('paper_figures/ec_earth_moving_average.png', format='png', dpi=250)
    plt.show()


for feature in df_hybrid_proj_us_ukesm_585.columns:
    fig1 = plt.figure(figsize=(10,5))
    df_hybrid_proj_us_ukesm_585[feature].where(df_hybrid_proj_us_ukesm_585[feature] > df_hybrid_us_2[feature].max()).dropna()
    
    plt.hist(df_hybrid_proj_us_ukesm_585[feature].where(df_hybrid_proj_us_ukesm_585[feature] > df_hybrid_us_2[feature].max()).dropna(),bins=50, label = 'ukesm_585', alpha = 0.9)
    plt.hist(df_hybrid_proj_ukesm_126[feature].where(df_hybrid_proj_ukesm_126[feature] > df_hybrid_us_2[feature].max()).dropna(),bins=50, label = 'ukesm_126', alpha = 0.9)
    plt.hist(df_hybrid_proj_ipsl_585[feature].where(df_hybrid_proj_ipsl_585[feature] > df_hybrid_us_2[feature].max()).dropna(),bins=50, label = 'ipsl_585', alpha = 0.9)
    plt.hist(df_hybrid_proj_ipsl_126[feature].where(df_hybrid_proj_ipsl_126[feature] > df_hybrid_us_2[feature].max()).dropna(),bins=50, label = 'ipsl_126', alpha = 0.9)
    plt.hist(df_hybrid_proj_gfdl_585[feature].where(df_hybrid_proj_gfdl_585[feature] > df_hybrid_us_2[feature].max()).dropna(),bins=50, label = 'gfdl_585', alpha = 0.9)
    plt.hist(df_hybrid_proj_gfdl_126[feature].where(df_hybrid_proj_gfdl_126[feature] > df_hybrid_us_2[feature].max()).dropna(),bins=50, label = 'gfdl_126', alpha = 0.9)

    plt.legend(loc="upper right")
    plt.title(f'Histogram of {feature} for GCM-RCPs')
    plt.ylabel('Frequency')
    if feature in ['txm_3','txm_4','txm_5']:
        x_label = 'Temperature (°C)'
    elif feature in ['prcptot_3', 'prcptot_4', 'prcptot_5']:
        x_label = 'Precipitation (mm/month)'
    else:
        x_label = 'Yield (ton/ha)'
          
    plt.xlabel(x_label)
    # fig1.savefig('paper_figures/ec_earth_moving_average.png', format='png', dpi=250)
    plt.show()

for feature in df_hybrid_proj_us_ukesm_585.columns:
    print(feature, 'max:',df_hybrid_us_2[feature].max()  , feature, 'min:',df_hybrid_us_2[feature].min()  )
for feature in df_hybrid_proj_us_ukesm_585.columns:
    print(feature, '-', round(df_hybrid_proj_us_ukesm_585[feature].max() - df_hybrid_us_2[feature].max(), 3) )


#%% TEST partial dependence plot for out of calibration zone cases

# EPIC
plt.plot(df_hybrid_us_2['yield-soy-noirr_2015co2'].groupby('time').mean(), label = 'History')
plt.plot(df_hybrid_proj_us_ukesm_585['yield-soy-noirr_2015co2'].groupby('time').mean(), label = 'Future')
plt.title('EPIC predictions')
plt.legend()
plt.show()

# TMX
plt.plot(df_hybrid_us_2['txm_4'].groupby('time').mean().index, df_hybrid_us_2['txm_4'].groupby('time').mean(), label = 'History')
df_hybrid_proj_us_ukesm_585['txm_4'].groupby('time').mean().plot(label = 'Future')
plt.title('TXM_4 predictions')
plt.legend()
plt.show()

# HYBRID
plt.plot(df_predict_test_hist['Yield'].groupby('time').mean(), label = 'History')
plt.plot(df_prediction_proj_us_ukesm_585['yield-soy-noirr_2015co2'].groupby('time').mean(), label = 'Future')
plt.axvline(df_hybrid_proj_us_ukesm_585['yield-soy-noirr_2015co2'].groupby('time').mean().idxmin(), linestyle = 'dashed')
plt.axvline(df_hybrid_proj_us_ukesm_585['yield-soy-noirr_2015co2'].groupby('time').mean().nsmallest(5).index[1], linestyle = 'dashed')
plt.title('Hybrid predictions')
plt.legend()
plt.show()

# Plots for points distribution
for feature in df_hybrid_proj_us_ukesm_585.columns:
    df_clim_extrapolated = df_hybrid_proj_us_ukesm_585[feature].where(df_hybrid_proj_us_ukesm_585[feature] > df_hybrid_us_2[feature].max()).dropna()
    df_y_extrapolated = df_prediction_proj_us_ukesm_585['yield-soy-noirr_2015co2'].where(df_hybrid_proj_us_ukesm_585[feature] > df_hybrid_us_2[feature].max()).dropna()

    plt.scatter(df_hybrid_us_2[feature], df_predict_test_hist['Yield'], color = 'k', label = 'History')    
    plt.scatter(df_hybrid_proj_us_ukesm_585[feature], df_prediction_proj_us_ukesm_585['yield-soy-noirr_2015co2'], alpha = 0.8, label = 'Projection')
    plt.hlines(df_predict_test_hist['Yield'].mean(), df_hybrid_us_2[feature].min(), df_hybrid_us_2[feature].max(), color = 'k')
    plt.scatter(df_clim_extrapolated, df_y_extrapolated, alpha = 0.8, label = 'Extrapolation')
    plt.legend(loc="upper right")
    plt.title(f'Scatterplot of {feature} for GCM-RCPs')
    plt.ylabel('Yield')
    if feature in ['txm_3','txm_4','txm_5']:
        x_label = 'Temperature (°C)'
    elif feature in ['prcptot_3', 'prcptot_4', 'prcptot_5']:
        x_label = 'Precipitation (mm/month)'
    else:
        x_label = 'Yield (ton/ha)'
          
    plt.xlabel(x_label)
    plt.show()
    

for feature in df_hybrid_proj_us_ukesm_585.columns:
    df_clim_extrapolated = df_hybrid_proj_us_ukesm_585[feature].where(df_hybrid_proj_us_ukesm_585[feature] < df_hybrid_us_2[feature].min()).dropna()
    df_y_extrapolated = df_prediction_proj_us_ukesm_585['yield-soy-noirr_2015co2'].where(df_hybrid_proj_us_ukesm_585[feature] < df_hybrid_us_2[feature].min()).dropna()

    plt.scatter(df_hybrid_us_2[feature], df_predict_test_hist['Yield'], color = 'k')    
    plt.scatter(df_hybrid_proj_us_ukesm_585[feature], df_prediction_proj_us_ukesm_585['yield-soy-noirr_2015co2'], alpha = 0.8)
    plt.hlines(df_predict_test_hist['Yield'].mean(), df_hybrid_us_2[feature].min(), df_hybrid_us_2[feature].max(), color = 'k')
    plt.scatter(df_clim_extrapolated, df_y_extrapolated, alpha = 0.8)
    # plt.legend(loc="upper right")
    plt.title(f'Scatterplot of {feature} for GCM-RCPs')
    plt.ylabel('Yield')
    if feature in ['txm_3','txm_4','txm_5']:
        x_label = 'Temperature (°C)'
    elif feature in ['prcptot_3', 'prcptot_4', 'prcptot_5']:
        x_label = 'Precipitation (mm/month)'
    else:
        x_label = 'Yield (ton/ha)'
          
    plt.xlabel(x_label)
    plt.show()
    

for feature in df_hybrid_proj_us_ukesm_585.columns:   
    sns.kdeplot(df_hybrid_us_2[feature],fill=True, alpha = 0.3, label = 'History')
    sns.kdeplot(df_hybrid_proj_us_ukesm_585[feature].astype(np.float64), fill=True, alpha = 0.3, label = 'Proj')
    plt.legend()
    plt.show()
    
    plt.plot(df_hybrid_us_2[feature].groupby('time').mean(), label = 'History')
    plt.plot(df_hybrid_proj_us_ukesm_585[feature].groupby('time').mean().astype(np.float64), label = 'Future')
    plt.axvline(df_hybrid_proj_us_ukesm_585['yield-soy-noirr_2015co2'].groupby('time').mean().idxmin(), linestyle = 'dashed')
    plt.axvline(df_hybrid_proj_us_ukesm_585['yield-soy-noirr_2015co2'].groupby('time').mean().nsmallest(5).index[1], linestyle = 'dashed')
    plt.title(f'{feature} predictions')
    plt.legend()
    plt.show()
    
sns.kdeplot(df_predict_test_hist['Yield'], fill=True, alpha = 0.3, label = 'History')
sns.kdeplot(df_prediction_proj_us_ukesm_585['yield-soy-noirr_2015co2'],fill=True, alpha = 0.3, label = 'Proj')
plt.legend()
plt.show()

### Partial dependence plots
from sklearn.inspection import PartialDependenceDisplay

features_to_plot = [0,6]
fig, ax1 = plt.subplots(1, 1, figsize=(10, 8), dpi=500)
disp1 = PartialDependenceDisplay.from_estimator(full_model_hyb2_am2, df_hybrid_proj_us_ukesm_585.iloc[:,:-1], features_to_plot, pd_line_kw={'color':'r'},percentiles=(0.01,0.99), ax = ax1)
disp2 = PartialDependenceDisplay.from_estimator(full_model_hyb2_am2, df_hybrid_us_2.iloc[:,:-1], features_to_plot, ax = disp1.axes_,percentiles=(0.01,0.99), pd_line_kw={'color':'k'})
plt.setp(disp1.deciles_vlines_, visible=False)
plt.setp(disp2.deciles_vlines_, visible=False)
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
disp1.plot(ax=[ax1, ax2], line_kw={"label": "Extrapolation", "color": "red"})
disp2.plot(ax=[ax1, ax2], line_kw={"label": "Training", "color": "black"})
ax1.set_ylim(0.0, 2.8)
ax2.set_ylim(0.0, 2.8)
plt.setp(disp1.deciles_vlines_, visible=False)
plt.setp(disp2.deciles_vlines_, visible=False)
ax1.legend()
plt.show()

features_to_plot = [1,2,3]
fig, ax1 = plt.subplots(1, 1, figsize=(10, 8), dpi=500)
disp1 = PartialDependenceDisplay.from_estimator(full_model_hyb2_am, df_hybrid_proj_us_ukesm_585.iloc[:,:-1], features_to_plot, pd_line_kw={'color':'r'},percentiles=(0.01,0.99), ax = ax1)
disp2 = PartialDependenceDisplay.from_estimator(full_model_hyb2_am, df_hybrid_us_2.iloc[:,:-1], features_to_plot, ax = disp1.axes_,percentiles=(0.01,0.99),pd_line_kw={'color':'k'})
plt.ylim(0, 2.6)
plt.setp(disp1.deciles_vlines_, visible=False)
plt.setp(disp2.deciles_vlines_, visible=False)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 8))
disp1.plot(ax=[ax1, ax2, ax3], line_kw={"label": "Extrapolation", "color": "red"})
disp2.plot(
    ax=[ax1, ax2, ax3], line_kw={"label": "Training", "color": "black"}
)
ax1.set_ylim(1, 2.8)
ax2.set_ylim(1, 2.8)
ax3.set_ylim(1, 2.8)
plt.setp(disp1.deciles_vlines_, visible=False)
plt.setp(disp2.deciles_vlines_, visible=False)
ax1.legend()
plt.show()

features_to_plot = [4,5,6]
fig, ax1 = plt.subplots(1, 1, figsize=(10, 8), dpi=500)
disp1 = PartialDependenceDisplay.from_estimator(full_model_hyb2_am, df_hybrid_proj_us_ukesm_585.iloc[:,:-1], features_to_plot, pd_line_kw={'color':'r'},percentiles=(0.01,0.99), ax = ax1)
disp2 = PartialDependenceDisplay.from_estimator(full_model_hyb2_am, df_hybrid_us_2.iloc[:,:-1], features_to_plot, ax = disp1.axes_,percentiles=(0.01,0.99),pd_line_kw={'color':'k'})
plt.ylim(0, 2.6)
plt.setp(disp1.deciles_vlines_, visible=False)
plt.setp(disp2.deciles_vlines_, visible=False)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 8))
disp1.plot(ax=[ax1, ax2, ax3], line_kw={"label": "Extrapolation", "color": "red"})
disp2.plot(
    ax=[ax1, ax2, ax3], line_kw={"label": "Training", "color": "black"}
)
ax1.set_ylim(1, 2.8)
ax2.set_ylim(1, 2.8)
ax3.set_ylim(1, 2.8)
plt.setp(disp1.deciles_vlines_, visible=False)
plt.setp(disp2.deciles_vlines_, visible=False)
ax1.legend()
plt.show()

#%% TRYING SHAP TO EXPLAIN THE MODEL
import shap
sample = shap.sample(df_hybrid_us_2.iloc[:,:-1], len(df_hybrid_us_2)//10 )

explainer = shap.KernelExplainer(full_model_hyb2.predict,sample)

shap_values = explainer.shap_values(df_hybrid_us_2.iloc[:500,:-1],nsamples=100)

shap.summary_plot(shap_values,df_hybrid_us_2.iloc[:500,:-1])








