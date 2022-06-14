# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 13:56:44 2022

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
# full_model_hyb_br = joblib.load('sklearn_pipeline_br_noBN_57.pkl')

# # Then, load the Keras model:
# full_model_hyb_br['estimator'].model = load_model('keras_model_br_noBN_57.h5')

# # Data for testing: 
# X = pd.read_csv('dataset_input_hybrid_forML_br.csv', index_col=[0,1,2],)
# y = pd.read_csv('dataset_obs_yield_forML_br.csv', index_col=[0,1,2],)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# y_pred = full_model_hyb_br.predict(X_test)

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
def projections_generation_hybrid(model, rcp_scenario, region, hybrid_model_full, start_date, end_date, co2_scen='both', three_models = False, sim_round = '\Gen_Assem', shift_year = False):
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
    list_features_br = ['prcptot', 'r10mm', 'txm'  ]# 'dtr', 'tnm', 'txge35', 'tr', 'txm', 'tmm', 'tnn'
    DS_clim_ext_projections = DS_clim_ext_projections[list_features_br]
    DS_clim_ext_projections = DS_clim_ext_projections.where(DS_y_obs_up_clip_det.mean('time') >= -5.0 )

    plot_2d_map(DS_clim_ext_projections['prcptot'].mean('time'))

    
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
        
    DS_clim_ext_projections_br = DS_clim_ext_projections_combined.where( DS_chosen_calendar_br >= 0 )
        
    DS_clim_ext_projections_br_det = DS_clim_ext_projections_br #detrend_dataset(DS_clim_ext_projections_br, mean_data = DS_exclim_br_det_clip)
    DS_clim_ext_projections_br['prcptot'].mean(['lat','lon']).plot()
    DS_clim_ext_projections_br_det['prcptot'].mean(['lat','lon']).plot()
    plt.show()
    
    DS_clim_ext_projections_br['txm'].mean(['lat','lon']).plot()
    DS_clim_ext_projections_br_det['txm'].mean(['lat','lon']).plot()
    plt.title('temperature detrending')
    plt.show()
    
    # plot_2d_map(DS_exclim_br_det_clip['txm'].mean('time'))
    # plot_2d_map(DS_clim_ext_projections_br['txm'].mean('time'))
    plot_2d_map(DS_clim_ext_projections_br_det['txm'].mean('time'))

        
    # For loop along features to obtain 24 months of climatic data for each year
    list_features_proj_reshape_shift = []
    for feature in list(DS_clim_ext_projections_br_det.keys()):
        ### Reshape and shift for 24 months for every year.
        df_clim_proj_shift = reshape_shift(DS_clim_ext_projections_br_det[feature])
        df_clim_proj_shift_12 = reshape_shift(DS_clim_ext_projections_br_det[feature], shift_time = 12)
        # Combine both dataframes
        df_clim_proj_shift_twoyears = df_clim_proj_shift.dropna().join(df_clim_proj_shift_12)
        
        ### Join and change name to S for the shift values
        df_clim_proj_shift = (df_clim_proj_shift_twoyears.dropna().join(df_calendar_month_br)
                                    .rename(columns={'plant':'s'}))
        # Move 
        col = df_clim_proj_shift.pop("s")
        df_clim_proj_shift.insert(0, col.name, col)
        df_clim_proj_shift[['s']].isna().sum()
    
        # Shift accoording to month indicator (hence +1)
        df_clim_proj_shift = (df_clim_proj_shift.apply(lambda x : x.shift(-(int(x['s']))+1) , axis=1)
                                    .drop(columns=['s']))
        
        
        list_features_proj_reshape_shift.append(df_clim_proj_shift)
    
    # Transform into dataframe
    df_clim_proj_twoyears = pd.concat(list_features_proj_reshape_shift, axis=1)
    
    ### Select specific months
    suffixes = tuple(["_"+str(j) for j in range(3,6)])
    df_feature_proj_6mon = df_clim_proj_twoyears.loc[:,df_clim_proj_twoyears.columns.str.endswith(suffixes)]
    
    # Shift 1 year
    end_year = 2100
    if shift_year == True:
        df_feature_proj_6mon.index = df_feature_proj_6mon.index.set_levels(df_feature_proj_6mon.index.levels[2] + 1, level=2)
        end_year = 2099
    
    df_feature_proj_6mon = df_feature_proj_6mon.rename_axis(index={'year':'time'})
    df_feature_proj_6mon = df_feature_proj_6mon.reorder_levels(['time','lat','lon']).sort_index()
    df_feature_proj_6mon = df_feature_proj_6mon.dropna()
    df_feature_proj_6mon=df_feature_proj_6mon.astype(float)
    
    # SECOND DETRENDING PART - SEASONAL
    DS_feature_proj_6mon_br = xr.Dataset.from_dataframe(df_feature_proj_6mon)
    DS_feature_proj_6mon_br_det = detrend_dataset(DS_feature_proj_6mon_br, deg = 'free', mean_data = DS_feature_season_6mon_br_det )#DS_feature_season_6mon_br_det )
    df_feature_proj_6mon_br_det = DS_feature_proj_6mon_br_det.to_dataframe().dropna()
    
    for feature in ['prcptot_3', 'prcptot_4', 'prcptot_5']:
        df_feature_proj_6mon_br_det[feature][df_feature_proj_6mon_br_det[feature] < 0] = 0
    
    for feature in df_feature_proj_6mon.columns:
        df_feature_proj_6mon[feature].groupby('time').mean().plot(label = 'old')
        df_feature_proj_6mon_br_det[feature].groupby('time').mean().plot(label = 'detrend')
        plt.title(f'{feature}')
        plt.legend()
        plt.show()
    
    # UPDATE DETRENDED VALUES
    df_feature_proj_6mon = df_feature_proj_6mon_br_det

        
    #%% EPIC projections
    
    def epic_projections_function_co2(co2_scenario):
        DS_y_epic_proj = xr.open_dataset("epic-iiasa_"+ model_full +"_w5e5_"+rcp_scenario+"_2015soc_"+co2_scenario+"_yield-soy-noirr_global_annual_2015_2100.nc", decode_times=False)
        # Convert time unit
        units, reference_date = DS_y_epic_proj.time.attrs['units'].split('since')
        DS_y_epic_proj['time'] = pd.date_range(start=' 2015-01-01, 00:00:00', periods=DS_y_epic_proj.sizes['time'], freq='YS')
        DS_y_epic_proj['time'] = DS_y_epic_proj['time'].dt.year
        
        DS_y_epic_proj['yield-soy-noirr'].mean(['lat', 'lon']).plot()
        plt.title( model + "_" + rcp_scenario + "_" + co2_scenario )
        plt.show()
        
        DS_y_epic_proj_br = DS_y_epic_proj.where(DS_y_obs_up_clip_det.mean('time') >= -5.0 )
        DS_y_epic_proj_br = DS_y_epic_proj_br.where(DS_clim_ext_projections['prcptot'].mean('time') >= -100.0 )
        

        DS_y_epic_proj_br_det = xr.DataArray( detrend_dim(DS_y_epic_proj_br['yield-soy-noirr'], 'time') + DS_y_epic_br_clip_det.mean('time'), name= DS_y_epic_proj_br['yield-soy-noirr'].name, attrs = DS_y_epic_proj_br['yield-soy-noirr'].attrs)
        
        # plot_2d_map(DS_y_epic_proj_br_det.mean('time'))
        
        DS_y_epic_proj_br['yield-soy-noirr'].mean(['lat', 'lon']).plot()
        DS_y_epic_proj_br_det.mean(['lat', 'lon']).plot()
        plt.title('Detrend'+ "_"+ model + "_"+ rcp_scenario+"_"+co2_scenario)
        plt.show()
        
        df_y_epic_proj_us = DS_y_epic_proj_br_det.to_dataframe().dropna()
        df_y_epic_proj_us = df_y_epic_proj_us.reorder_levels(['time','lat','lon']).sort_index()
        
        #%% HYBRID
        
        df_hybrid_proj_2 = pd.concat([df_y_epic_proj_us, df_feature_proj_6mon], axis = 1 )
        df_hybrid_proj_test_2 = df_hybrid_proj_2.query(f"time>=2016 and time <= {end_year}")

        # Predicting hybrid results
        df_prediction_proj_2 = df_y_epic_proj_us.query(f"time>=2016 and time <= {end_year}").copy() 
        predic_model_test_2 = hybrid_model_full.predict(df_hybrid_proj_test_2).copy()
        df_prediction_proj_2.loc[:,'yield-soy-noirr'] = predic_model_test_2
        
        DS_hybrid_proj_2 = xr.Dataset.from_dataframe(df_prediction_proj_2)
        DS_hybrid_proj_2 = DS_hybrid_proj_2.sortby('lat')
        DS_hybrid_proj_2 = DS_hybrid_proj_2.sortby('lon')
        
        plt.plot(DS_hybrid_proj_2.time.values, DS_hybrid_proj_2['yield-soy-noirr'].mean(['lat','lon']), label = 'Hybrid')
        plt.plot(DS_y_epic_proj_br_det.time.values, DS_y_epic_proj_br_det.mean(['lat','lon']), label = 'EPIC pure')
        plt.title('Hybrid'+ "_"+ model + "_"+rcp_scenario+"_"+co2_scenario)
        plt.legend()
        plt.show()
        
        plt.plot(DS_hybrid_proj_2.time.values, DS_hybrid_proj_2['yield-soy-noirr'].mean(['lat','lon'])/ DS_hybrid_proj_2['yield-soy-noirr'].mean(), label = 'Hybrid')
        plt.plot(DS_y_epic_proj_br_det.time.values, DS_y_epic_proj_br_det.mean(['lat','lon'])/DS_y_epic_proj_br_det.mean(), label = 'EPIC pure')
        plt.title('Hybrid'+ "_"+ model + "_"+rcp_scenario+"_"+co2_scenario)
        plt.ylabel('Shock (0-1)')
        plt.legend()
        plt.show()
        
        plot_2d_map(DS_hybrid_proj_2['yield-soy-noirr'].mean('time'))
        
        #%% EPIC SIM
        df_predic_epic_test = df_y_epic_proj_us.copy()
        df_predic_epic_test.loc[:,'yield-soy-noirr'] = full_model_epic.predict(df_y_epic_proj_us)
        
        DS_pred_epic_proj = xr.Dataset.from_dataframe(df_predic_epic_test)
        DS_pred_epic_proj = DS_pred_epic_proj.sortby('lat')
        DS_pred_epic_proj = DS_pred_epic_proj.sortby('lon')
        
        # CLIMATIC model
        df_pred_clim_proj_test = df_y_epic_proj_us.copy()
        df_pred_clim_proj_test.loc[:,'yield-soy-noirr'] = model_exclim_dyn_br.predict(df_feature_proj_6mon)
        
        DS_pred_clim_proj = xr.Dataset.from_dataframe(df_pred_clim_proj_test)
        DS_pred_clim_proj = DS_pred_clim_proj.sortby('lat')
        DS_pred_clim_proj = DS_pred_clim_proj.sortby('lon')
        
        print("Self-Correlation is:", xr.corr(DS_pred_epic_proj["yield-soy-noirr"], DS_pred_epic_proj["yield-soy-noirr"]).values )      
        print("Correlation CLIM x EPIC is:", xr.corr(DS_pred_epic_proj["yield-soy-noirr"], DS_pred_clim_proj["yield-soy-noirr"]).values )      
        print("Correlation HYB x CLIM is:", xr.corr(DS_hybrid_proj_2["yield-soy-noirr"], DS_pred_clim_proj["yield-soy-noirr"]).values )      
        print("Correlation HYB x EPIC is:", xr.corr(DS_hybrid_proj_2["yield-soy-noirr"], DS_pred_epic_proj["yield-soy-noirr"]).values )      
        
        # WEIGHTING SCHEME
        # DS_harvest_area_globiom = xr.open_dataset('../../paper_hybrid_agri/data/soy_harvest_area_globiom_05x05_2b.nc').mean('time')
        # DS_harvest_area_globiom['harvest_area'] = DS_harvest_area_globiom['harvest_area'].where(DS_hybrid_proj_2['yield-soy-noirr'].mean('time')>0)
        # total_area = DS_harvest_area_globiom['harvest_area'].sum()

        DS_harvest_area_globiom =  xr.open_dataset("../../paper_hybrid_agri/data/americas_mask_ha.nc", decode_times=False).sel(longitude=slice(-58.25,-44)).rename({'latitude': 'lat', 'longitude': 'lon'}) # xr.open_dataset('../../paper_hybrid_agri/data/soy_harvest_area_globiom_05x05_2b.nc').mean('time')
        DS_harvest_area_globiom['harvest_area'] = DS_harvest_area_globiom['annual_area_harvested_rfc_crop08_ha_30mn'].where(DS_hybrid_proj_2['yield-soy-noirr'].mean('time')>-2)
        
        total_area =  DS_harvest_area_globiom['harvest_area'].sum(['lat','lon'])
        
        plot_2d_map(DS_harvest_area_globiom['harvest_area'])

        DS_hybrid_weighted = ((DS_hybrid_proj_2['yield-soy-noirr'] * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'yield-soy-noirr')
        DS_epic_weighted = ((DS_pred_epic_proj['yield-soy-noirr'] * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'yield-soy-noirr')
        DS_clim_weighted = ((DS_pred_clim_proj['yield-soy-noirr'] * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'yield-soy-noirr')
        DS_epic_pure_weighted = ((DS_y_epic_proj_br_det * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'yield-soy-noirr').sel(time = slice(2015,2099))
        
        plt.plot(DS_hybrid_proj_2['yield-soy-noirr'].mean(['lat','lon']), label = 'unweighted')
        plt.plot(DS_hybrid_weighted['yield-soy-noirr'].sum(['lat','lon']), label = 'weighted')
        plt.ylabel('yield')
        plt.xlabel('years')
        plt.legend()
        plt.show()
        
        
        plt.plot(DS_y_epic_proj_br_det.time.values, DS_y_epic_proj_br_det.mean(['lat','lon']), label = 'Pure EPIC')
        plt.plot(DS_pred_epic_proj.time.values, DS_pred_epic_proj["yield-soy-noirr"].mean(['lat','lon']), label = 'EPIC-RF')
        plt.plot(DS_hybrid_proj_2.time.values, DS_hybrid_proj_2["yield-soy-noirr"].mean(['lat','lon']), label = 'Hybrid-RF')
        plt.title('Non-weighted comparison')
        plt.legend()
        plt.show()
        
        plt.plot(DS_y_epic_proj_br_det.time.values, DS_y_epic_proj_br_det.mean(['lat','lon']), label = 'Pure EPIC')
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
        
        weighted_timeseries = DS_hybrid_weighted['yield-soy-noirr'].sum(['lat','lon'])/np.mean(DS_hybrid_weighted["yield-soy-noirr"].sum(['lat','lon']))
        weighted_timeseries_min = weighted_timeseries.idxmin()       
        plot_2d_map(DS_y_epic_proj_br_det.sel(time=weighted_timeseries_min)/DS_y_epic_proj_br_det.mean('time'))
        plot_2d_map(DS_hybrid_proj_2['yield-soy-noirr'].sel(time=weighted_timeseries_min)/DS_hybrid_proj_2["yield-soy-noirr"].mean('time'))
        plot_2d_map(DS_hybrid_proj_2['yield-soy-noirr'].sel(time=weighted_timeseries_min)/DS_hybrid_proj_2["yield-soy-noirr"].mean('time') - DS_y_epic_proj_br_det.sel(time=weighted_timeseries_min)/DS_y_epic_proj_br_det.mean('time'))
                                                                      
            
        DS_hybrid_proj_2.to_netcdf('output_models'+region +'/hybrid_'+model_full+'_'+rcp_scenario+'_'+co2_scenario+'_yield_soybean_2015_2100.nc')
        DS_pred_clim_proj.to_netcdf('output_models'+region +'/clim_'+model_full+'_'+rcp_scenario+'_'+co2_scenario+'_yield_soybean_2015_2100.nc')
        DS_pred_epic_proj.to_netcdf('output_models'+region +'/epic_'+model_full+'_'+rcp_scenario+'_'+co2_scenario+'_yield_soybean_2015_2100.nc')
        
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
                    
                    # Reindex to avoid missnig coordinates and dimension values
                    new_lat = np.arange(ds_shifter_cycle.lat[0], ds_shifter_cycle.lat[-1], 0.5)
                    new_lon = np.arange(ds_shifter_cycle.lon[0], ds_shifter_cycle.lon[-1], 0.5)
                    ds_shifter_cycle = ds_shifter_cycle.reindex({'lat':new_lat})
                    ds_shifter_cycle = ds_shifter_cycle.reindex({'lon':new_lon})
                    
                    print('year_min:', dataset_cycle.time[0].values, 'year_max:',dataset_cycle.time[-1].values, 'min_pos',year_min, 'shifter', list(ds_shifter_cycle.to_dataframe().mean().values))
                    
                list_shifters_cycle.append(ds_shifter_cycle)

            return list_shifters_cycle
                
        list_test = local_minima_30years(DS_hybrid_proj_2, weights = 'YES') 
        
        # Hybrid model
        list_test[0].to_netcdf('output_shocks'+region +sim_round +"/hybrid_"+model_full+'_'+rcp_scenario+"_"+co2_scenario+"_yield_soybean_shift_2017-2044.nc")
        list_test[1].to_netcdf('output_shocks'+region +sim_round +"/hybrid_"+model_full+'_'+rcp_scenario+"_"+co2_scenario+"_yield_soybean_shift_2044-2071.nc")
        list_test[2].to_netcdf('output_shocks'+region + sim_round +"/hybrid_"+model_full+'_'+rcp_scenario+"_"+co2_scenario+"_yield_soybean_shift_2071-2098.nc")
        
        df_hybrid_proj_test_2 = df_hybrid_proj_test_2.rename(columns = {f'yield-soy-noirr':f'yield-soy-noirr_{co2_scenario}'})
        df_prediction_proj_2 = df_prediction_proj_2.rename(columns = {f'yield-soy-noirr':f'yield-soy-noirr_{co2_scenario}'})
    
        
        if three_models is True:
            list_shift_epic_proj = local_minima_30years(DS_pred_epic_proj, weights = 'YES') 
            
            list_shift_epic_proj[0].to_netcdf('output_shocks'+region +sim_round +"/epic_"+model_full+'_'+rcp_scenario+"_"+co2_scenario+"_yield_soybean_shift_2017-2044.nc")
            list_shift_epic_proj[1].to_netcdf('output_shocks'+region +sim_round +"/epic_"+model_full+'_'+rcp_scenario+"_"+co2_scenario+"_yield_soybean_shift_2044-2071.nc")
            list_shift_epic_proj[2].to_netcdf('output_shocks'+region +sim_round +"/epic_"+model_full+'_'+rcp_scenario+"_"+co2_scenario+"_yield_soybean_shift_2071-2098.nc")
            
            
            list_shift_clim_proj = local_minima_30years(DS_pred_clim_proj, weights = 'YES') 
            
            list_shift_clim_proj[0].to_netcdf('output_shocks'+region +sim_round +"/clim_"+model_full+'_'+rcp_scenario+"_"+co2_scenario+"_yield_soybean_shift_2017-2044.nc")
            list_shift_clim_proj[1].to_netcdf('output_shocks'+region +sim_round +"/clim_"+model_full+'_'+rcp_scenario+"_"+co2_scenario+"_yield_soybean_shift_2044-2071.nc")
            list_shift_clim_proj[2].to_netcdf('output_shocks'+region +sim_round +"/clim_"+model_full+'_'+rcp_scenario+"_"+co2_scenario+"_yield_soybean_shift_2071-2098.nc")
            
        return df_hybrid_proj_test_2, df_prediction_proj_2            
            
# ####        TURN ON WHEN SERIES IS DECIDED
#         def shocks_series(dataset, start_year, end_year, ref_start_year, ref_end_year):
            
#             dataset_mean_general = dataset.sel( time = slice( ref_start_year, ref_end_year ) ).mean(['time'])
#             dataset_cycle = dataset.sel( time = slice( start_year, end_year ) )
                 
#             shocks_series = dataset_cycle / dataset_mean_general
        
#             total_area = DS_harvest_area_globiom['harvest_area'].sum()
#             DS_weighted = ((dataset['yield-soy-noirr'] * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'yield-soy-noirr')            
#             dataset_cycle_weight = DS_weighted.sel( time = slice( start_year, end_year ))
            
#             shocks_series_weight = dataset_cycle_weight['yield-soy-noirr'].sum(['lat','lon']) / dataset_mean_general.mean()
        
#             df_shocks = pd.DataFrame(index = dataset_cycle.time.values, data = shocks_series_weight.to_dataframe().values)
        
#             print('Weighted shocks:', df_shocks)
                
                
#             return shocks_series
                
#         shcoks_series_hybrid = shocks_series(DS_hybrid_proj_2, 2095, 2099, 2070, 2100)
        
#         shcoks_series_hybrid.to_netcdf('output_shocks'+region +"/hybrid_"+model_full+'_'+rcp_scenario+"_"+co2_scenario+"_yield_soybean_shocks_series_2095_2099.nc")

#         if three_models is True:
#             shcoks_series_epic = shocks_series(DS_pred_epic_proj, 2095, 2099, 2070, 2100)
#             shcoks_series_clim = shocks_series(DS_pred_clim_proj, 2095, 2099, 2070, 2100)
        
#             shcoks_series_epic.to_netcdf('output_shocks'+region +"/epic_"+model_full+'_'+rcp_scenario+"_"+co2_scenario+"_yield_soybean_shocks_series_2095_2099.nc")
#             shcoks_series_clim.to_netcdf('output_shocks'+region +"/clim_"+model_full+'_'+rcp_scenario+"_"+co2_scenario+"_yield_soybean_shocks_series_2095_2099.nc")
        
    
    if co2_scen == 'both':
        df_hybrid_proj_test_2_default, df_prediction_proj_2_default = epic_projections_function_co2(co2_scenario = 'default')
        df_hybrid_proj_test_2_2015co2, df_prediction_proj_2_2015co2 = epic_projections_function_co2(co2_scenario = '2015co2')
        
        df_hybrid_proj_test_2_both = pd.concat([df_hybrid_proj_test_2_default, df_hybrid_proj_test_2_2015co2['yield-soy-noirr_2015co2']], axis = 1)    
        df_prediction_proj_2_both = pd.concat([df_prediction_proj_2_default['yield-soy-noirr_default'], df_prediction_proj_2_2015co2['yield-soy-noirr_2015co2']], axis = 1)    

    
    else:
        co2_scenario = co2_scen
        epic_projections_function_co2(co2_scenario)
    
    return df_hybrid_proj_test_2_both, df_prediction_proj_2_both



#%% START MAIN SCRIPT

# UKESM model
df_hybrid_proj_ukesm_585_br, df_prediction_proj_ukesm_585_br = projections_generation_hybrid(model = 'ukesm', rcp_scenario = 'ssp585', region = "_br", hybrid_model_full = full_model_hyb2_am2, start_date='01-01-2015', end_date='31-12-2100', co2_scen = 'both')
df_hybrid_proj_ukesm_126_br, df_prediction_proj_ukesm_126_br = projections_generation_hybrid(model = 'ukesm', rcp_scenario = 'ssp126', region = "_br", hybrid_model_full = full_model_hyb2_am2, start_date='01-01-2015', end_date='31-12-2100', co2_scen = 'both')

# GFDL model
df_hybrid_proj_gfdl_585, df_prediction_proj_gfdl_585 = projections_generation_hybrid(model = 'gfdl', rcp_scenario = 'ssp585', region = "_br", hybrid_model_full = full_model_hyb_br, start_date='01-01-2015', end_date='31-12-2100', co2_scen = 'both')
df_hybrid_proj_gfdl_126, df_prediction_proj_gfdl_126 = projections_generation_hybrid(model = 'gfdl', rcp_scenario = 'ssp126', region = "_br", hybrid_model_full = full_model_hyb_br, start_date='01-01-2015', end_date='31-12-2100', co2_scen = 'both')

# IPSL model
df_hybrid_proj_ipsl_585, df_prediction_proj_ipsl_585 = projections_generation_hybrid(model = 'ipsl', rcp_scenario = 'ssp585', region = "_br", hybrid_model_full = full_model_hyb_br, start_date='01-01-2015', end_date='31-12-2100', co2_scen = 'both')
df_hybrid_proj_ipsl_126, df_prediction_proj_ipsl_126 = projections_generation_hybrid(model = 'ipsl', rcp_scenario = 'ssp126', region = "_br", hybrid_model_full = full_model_hyb_br, start_date='01-01-2015', end_date='31-12-2100', co2_scen = 'both')


#%% TEST partial dependence plot for out of calibration zone cases - 2088 should be extreme

# load future prodictions
df_predict_fut = df_prediction_proj_ukesm_585_br.iloc[:,[1]].copy() # Use 2015 case
df_predict_fut = df_predict_fut.rename(columns={'yield-soy-noirr_2015co2':'yield-soy-noirr'})

df_proj_fut = df_hybrid_proj_ukesm_585_br.rename(columns={'yield-soy-noirr_2015co2':'yield-soy-noirr'})
df_proj_fut =  df_proj_fut[ ['yield-soy-noirr'] + [ col for col in df_proj_fut.columns if col != 'yield-soy-noirr' ] ]
df_proj_fut = df_proj_fut.drop(columns='yield-soy-noirr_default')

df_hybrid_us_2_br = df_hybrid_batch.copy()


plt.plot(df_hybrid_us_2_br['yield-soy-noirr'].groupby('time').mean(), label = 'History')
plt.plot(df_proj_fut['yield-soy-noirr'].groupby('time').mean(), label = 'Future')
plt.axvline(df_proj_fut['yield-soy-noirr'].groupby('time').mean().idxmin(), linestyle = 'dashed')
plt.title('EPIC predictions')
plt.legend()
plt.show()

# TMX
plt.plot(df_hybrid_us_2_br['txm_4'].groupby('time').mean(), label = 'History')
plt.plot(df_proj_fut['txm_4'].groupby('time').mean(), label = 'Future')
plt.axvline(df_proj_fut['yield-soy-noirr'].groupby('time').mean().idxmin(), linestyle = 'dashed')
plt.title('TXM_4 predictions')
plt.legend()
plt.show()

plt.plot(df_predict_test_hist['Yield'].groupby('time').mean(), label = 'History')
plt.plot(df_predict_fut['yield-soy-noirr'].groupby('time').mean(), label = 'Future')
plt.axvline(df_proj_fut['yield-soy-noirr'].groupby('time').mean().idxmin(), linestyle = 'dashed')
plt.title('Hybrid predictions')
plt.legend()
plt.show()

# Plots for points distribution
for feature in df_proj_fut.columns:
    df_clim_extrapolated = df_proj_fut[feature].where(df_proj_fut[feature] > df_hybrid_us_2_br[feature].max()).dropna()
    df_y_extrapolated = df_predict_fut['yield-soy-noirr'].where(df_proj_fut[feature] > df_hybrid_us_2_br[feature].max()).dropna()

    plt.scatter(df_hybrid_us_2_br[feature], df_predict_test_hist['Yield'], color = 'k', label = 'History')    
    plt.scatter(df_proj_fut[feature], df_predict_fut['yield-soy-noirr'], alpha = 0.8, label = 'Projection')
    plt.hlines(df_predict_test_hist['Yield'].mean(), df_hybrid_us_2_br[feature].min(), df_hybrid_us_2_br[feature].max(), color = 'k')
    plt.scatter(df_clim_extrapolated, df_y_extrapolated, alpha = 0.8, label = 'Extrapolation')
    plt.legend(loc="upper right")
    plt.title(f'Scatterplot of {feature} for GCM-RCPs')
    plt.ylabel('Yield')
    if feature in ['tnx_3','tnx_4','tnx_5']:
        x_label = 'Temperature (°C)'
    elif feature in ['prcptot_3', 'prcptot_4', 'prcptot_5']:
        x_label = 'Precipitation (mm/month)'
    else:
        x_label = 'Yield (ton/ha)'
          
    plt.xlabel(x_label)
    plt.show()

for feature in df_proj_fut.columns:   
    sns.kdeplot(df_hybrid_us_2_br[feature],fill=True, alpha = 0.3, label = 'History')
    sns.kdeplot(df_proj_fut[feature],fill=True, alpha = 0.3, label = 'Proj')
    print('hist', df_hybrid_us_2_br[feature].mean(), 'fut', df_proj_fut[feature].mean())
    plt.legend()
    plt.show()
    
    plt.plot(df_hybrid_us_2_br[feature].groupby('time').mean(), label = 'History')
    plt.plot(df_proj_fut[feature].groupby('time').mean(), label = 'Future')
    plt.axvline(df_proj_fut['yield-soy-noirr'].groupby('time').mean().idxmin(), linestyle = 'dashed')
    plt.title(f'{feature} predictions')
    plt.legend()
    plt.show()
    
    
sns.kdeplot(df_predict_test_hist['Yield'], fill=True, alpha = 0.3, label = 'History')
sns.kdeplot(df_predict_fut['yield-soy-noirr'],fill=True, alpha = 0.3, label = 'Proj')
plt.legend()
plt.show()

plt.plot(df_predict_test_hist['Yield'].groupby('time').mean(), label = 'History')
plt.plot(df_predict_fut['yield-soy-noirr'].groupby('time').mean(), label = 'Future')
plt.axvline(df_proj_fut['yield-soy-noirr'].groupby('time').mean().idxmin(), linestyle = 'dashed')
plt.title(f'{feature} predictions')
plt.legend()
plt.show()


# for feature in df_proj_fut.columns:
#     df_clim_extrapolated = df_proj_fut[feature].where(df_proj_fut[feature] < df_hybrid_us_2_br[feature].min()).dropna()
#     df_y_extrapolated = df_predict_fut['yield-soy-noirr'].where(df_proj_fut[feature] < df_hybrid_us_2_br[feature].min()).dropna()

#     plt.scatter(df_hybrid_us_2_br[feature], df_predict_test_hist['Yield'], color = 'k')    
#     plt.scatter(df_proj_fut[feature], df_predict_fut['yield-soy-noirr'], alpha = 0.8)
#     plt.hlines(df_predict_test_hist['Yield'].mean(), df_hybrid_us_2_br[feature].min(), df_hybrid_us_2_br[feature].max(), color = 'k')
#     plt.scatter(df_clim_extrapolated, df_y_extrapolated, alpha = 0.8)
#     # plt.legend(loc="upper right")
#     plt.title(f'Scatterplot of {feature} for GCM-RCPs')
#     plt.ylabel('Yield')
#     if feature in ['tnx_3','tnx_4','tnx_5']:
#         x_label = 'Temperature (°C)'
#     elif feature in ['prcptot_3', 'prcptot_4', 'prcptot_5']:
#         x_label = 'Precipitation (mm/month)'
#     else:
#         x_label = 'Yield (ton/ha)'
          
#     plt.xlabel(x_label)
#     plt.show()
    

#%% Partial dependence plots
from sklearn.inspection import PartialDependenceDisplay

features_to_plot = ['yield-soy-noirr','txm_4']
fig, ax1 = plt.subplots(1, 1, figsize=(10, 8), dpi=500)
disp1 = PartialDependenceDisplay.from_estimator(full_model_hyb_br, df_proj_fut, features_to_plot, pd_line_kw={'color':'r'},percentiles=(0.01,0.99), ax = ax1)
disp2 = PartialDependenceDisplay.from_estimator(full_model_hyb_br, df_hybrid_us_2_br, features_to_plot, ax = disp1.axes_,percentiles=(0,1), pd_line_kw={'color':'k'})
plt.setp(disp1.deciles_vlines_, visible=False)
plt.setp(disp2.deciles_vlines_, visible=False)
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
disp1.plot(ax=[ax1, ax2], line_kw={"label": "Extrapolation", "color": "red"})
disp2.plot(ax=[ax1, ax2], line_kw={"label": "Training", "color": "black"})
ax1.set_ylim(1.0, 2.8)
ax2.set_ylim(1.0, 2.8)
plt.setp(disp1.deciles_vlines_, visible=False)
plt.setp(disp2.deciles_vlines_, visible=False)
ax1.legend()
plt.show()

features_to_plot = [1,2,3]
fig, ax1 = plt.subplots(1, 1, figsize=(10, 8), dpi=500)
disp1 = PartialDependenceDisplay.from_estimator(full_model_hyb2_am, df_proj_fut, features_to_plot, pd_line_kw={'color':'r'},percentiles=(0.01,0.99), ax = ax1)
disp2 = PartialDependenceDisplay.from_estimator(full_model_hyb2_am, df_hybrid_us_2_br, features_to_plot, ax = disp1.axes_,percentiles=(0,1), pd_line_kw={'color':'k'})
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
disp1 = PartialDependenceDisplay.from_estimator(full_model_hyb2_am, df_proj_fut, features_to_plot, pd_line_kw={'color':'r'},percentiles=(0.01,0.99), ax = ax1, method = 'brute')
disp2 = PartialDependenceDisplay.from_estimator(full_model_hyb2_am, df_hybrid_us_2_br, features_to_plot, ax = disp1.axes_,percentiles=(0,1), pd_line_kw={'color':'k'}, method = 'brute')
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

#%%
mean_stuff = df_proj_fut['prcptot_4'].groupby(['lat','lon']).mean().to_xarray()
stuff_2088 = df_proj_fut['prcptot_4'].loc[2088].groupby(['lat','lon']).mean().to_xarray()
delta = stuff_2088 - mean_stuff
delta.plot()
# test_values = df_hybrid_us_2_br.iloc[[0],:-1].copy()
# test_values.iloc[[0],:] = [0,0,0,0,50,50,50]
# test_prdict = full_model_hyb_br.predict(test_values)
# print(test_prdict)

DS_test = xr.open_mfdataset('monthly_ukesm_ssp585_br/txm_MON_climpact.ukesm1_r1i1p1f1_w5e5_ssp5-ssp5.nc')

df_test = DS_test['txm'].mean(['lat','lon']).to_dataframe()
df_test.plot()

from statsmodels.tsa.seasonal import seasonal_decompose
decompose_data = seasonal_decompose(df_test, model="additive", period = 516)
decompose_data.plot(); 

decompose_data.seasonal.plot(); 

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(df_test); 

plt.figure(figsize=(10,6))
plt.show()


























