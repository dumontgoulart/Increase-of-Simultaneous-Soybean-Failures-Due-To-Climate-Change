# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 15:19:12 2021

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

#%% Load the pipeline first:
full_model_hyb2 = joblib.load('sklearn_pipeline_us_BN.pkl')

# Then, load the Keras model:
full_model_hyb2['estimator'].model = load_model('keras_model_us_BN.h5')

# Data for testing: 
X = pd.read_csv('dataset_input_hybrid_forML.csv', index_col=[0,1,2],)
y = pd.read_csv('dataset_obs_yield_forML.csv', index_col=[0,1,2],)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

y_pred = full_model_hyb2.predict(X_test)

# report performance
print("R2 on test set:", round(r2_score(y_test, y_pred),2))
print("Var score on test set:", round(explained_variance_score(y_test, y_pred),2))
print("MAE on test set:", round(mean_absolute_error(y_test, y_pred),5))
print("RMSE on test set:",round(mean_squared_error(y_test, y_pred, squared=False),5))


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
        for degree in [1,2,3]:
            # detrend along a single dimension
            p = da.polyfit(dim=dim, deg=degree)
            fit = xr.polyval(da[dim], p.polyfit_coefficients)
            
            da_det = da - fit
            res_detrend = np.nansum((da_zero_mean-da_det)**2)
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
def projections_generation_hybrid(model, rcp_scenario, region, hybrid_model_full, start_date, end_date, co2_scen='both', three_models = False, sim_round = '\Gen_Assem'):
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

    DS_ref = DS_exclim_us_det_clip['txm'].sel(time='2012-07-16').where(DS_clim_ext_projections_us['txm'].mean('time') > -100)

    delta_temp = DS_clim_ext_projections_us['txm'].sel(time = '2056-07-16') - DS_ref.values
    
    plt.figure(figsize=(12,5)) #plot clusters
    ax=plt.axes(projection=ccrs.Mercator())
    delta_temp.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True, cmap='RdBu_r', vmin=-6, vmax=6, cbar_kwargs={"label": "Anomaly (°C)"})
    ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
    ax.set_extent([-125,-67,25,50], ccrs.Geodetic())
    ax.set_title('Temperature difference 3C and hist')
    plt.show()
    
    
    DS_clim_ext_projections_us_det = detrend_dataset(DS_clim_ext_projections_us)
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
        
        DS_y_epic_proj_us_det = xr.DataArray( detrend_dim(DS_y_epic_proj_us['yield-soy-noirr'], 'time') + DS_y_epic_proj_us['yield-soy-noirr'].mean('time'), name= DS_y_epic_proj_us['yield-soy-noirr'].name, attrs = DS_y_epic_proj_us['yield-soy-noirr'].attrs)
        
        # plot_2d_us_map(DS_y_epic_proj_us_det.mean('time'))
        
        DS_y_epic_proj_us['yield-soy-noirr'].mean(['lat', 'lon']).plot()
        DS_y_epic_proj_us_det.mean(['lat', 'lon']).plot()
        plt.title('Detrend'+ "_"+ model + "_"+ rcp_scenario+"_"+co2_scenario)
        plt.show()
        
        df_y_epic_proj_us = DS_y_epic_proj_us_det.to_dataframe().dropna()
        df_y_epic_proj_us = df_y_epic_proj_us.reorder_levels(['time','lat','lon']).sort_index()
        
        #%% HYBRID
        
        df_hybrid_proj_2 = pd.concat([df_y_epic_proj_us, df_proj_feature_season_6mon], axis = 1 )
        df_hybrid_proj_test_2 = df_hybrid_proj_2.query("time>=2015 and time <= 2100")
        
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
        
        #%% EPIC SIM
        df_predic_epic_test = df_y_epic_proj_us.copy()
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
        
        # Plot non-weighted timeseries
        plt.plot(DS_y_epic_proj_us_det.time.values, DS_y_epic_proj_us_det.mean(['lat','lon']), label = 'Pure EPIC')
        plt.plot(DS_pred_epic_proj.time.values, DS_pred_epic_proj["yield-soy-noirr"].mean(['lat','lon']), label = 'EPIC-RF')
        plt.plot(DS_hybrid_proj_2.time.values, DS_hybrid_proj_2["yield-soy-noirr"].mean(['lat','lon']), label = 'Hybrid-RF')
        plt.title('Non-weighted comparison')
        plt.legend()
        plt.show()
        
        # Plot weighted timeseries
        plt.plot(DS_y_epic_proj_us_det.time.values, DS_y_epic_proj_us_det.mean(['lat','lon']), label = 'Pure EPIC')
        plt.plot(DS_pred_epic_proj.time.values, DS_epic_weighted["yield-soy-noirr"].sum(['lat','lon']), label = 'EPIC-RF')
        plt.plot(DS_hybrid_proj_2.time.values, DS_hybrid_weighted['yield-soy-noirr'].sum(['lat','lon']), label = 'Hybrid-RF')
        plt.title('Weighted comparison')
        plt.legend()
        plt.show()
        
        # Plot weighted shocks
        plt.plot(DS_pred_epic_proj.time.values, DS_epic_weighted["yield-soy-noirr"].sum(['lat','lon'])/np.mean(DS_epic_weighted["yield-soy-noirr"].sum(['lat','lon'])), label = 'EPIC-RF')
        plt.plot(DS_pred_clim_proj.time.values, DS_clim_weighted["yield-soy-noirr"].sum(['lat','lon'])/np.mean(DS_clim_weighted["yield-soy-noirr"].sum(['lat','lon'])), label = 'CLIM-RF')
        plt.plot(DS_hybrid_proj_2.time.values, DS_hybrid_weighted['yield-soy-noirr'].sum(['lat','lon'])/np.mean(DS_hybrid_weighted["yield-soy-noirr"].sum(['lat','lon'])), label = 'Hybrid')
        plt.title("Weighted"+ "_" + model + "_"+ rcp_scenario+"_"+co2_scenario)
        plt.ylabel("Shock (%)")
        plt.legend()
        plt.show()
        
        # Plot weighted shocks HYRBID ONLY
        plt.plot(DS_hybrid_proj_2.time.values, DS_hybrid_weighted['yield-soy-noirr'].sum(['lat','lon'])/np.mean(DS_hybrid_weighted["yield-soy-noirr"].sum(['lat','lon'])), label = 'Hybrid')
        plt.title("Weighted"+ "_" + model + "_"+ rcp_scenario+"_"+co2_scenario)
        plt.ylabel("Shock (%)")
        plt.legend()
        plt.show()
            
        DS_hybrid_proj_2.to_netcdf('output_models'+region +'/hybrid_'+model_full+'_'+rcp_scenario+'_'+co2_scenario+'_yield_soybean_2015_2100.nc')
        DS_pred_clim_proj.to_netcdf('output_models'+region +'/clim_'+model_full+'_'+rcp_scenario+'_'+co2_scenario+'_yield_soybean_2015_2100.nc')
        DS_pred_epic_proj.to_netcdf('output_models'+region +'/epic_'+model_full+'_'+rcp_scenario+'_'+co2_scenario+'_yield_soybean_2015_2100.nc')
        
        print(DS_hybrid_proj_2)
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
                
        list_test = local_minima_30years(DS_hybrid_proj_2, weights = 'YES') 
        
        print(list_test[0])
        # Hybrid model
        list_test[0].to_netcdf('output_shocks'+region +sim_round +"/hybrid_"+model_full+'_'+rcp_scenario+"_"+co2_scenario+"_yield_soybean_shift_2017-2044.nc")
        list_test[1].to_netcdf('output_shocks'+region +sim_round +"/hybrid_"+model_full+'_'+rcp_scenario+"_"+co2_scenario+"_yield_soybean_shift_2044-2071.nc")
        list_test[2].to_netcdf('output_shocks'+region + sim_round +"/hybrid_"+model_full+'_'+rcp_scenario+"_"+co2_scenario+"_yield_soybean_shift_2071-2098.nc")
        
        plot_2d_us_map(list_test[1]['yield-soy-noirr'])

        
        if three_models is True:
            list_shift_epic_proj = local_minima_30years(DS_pred_epic_proj, weights = 'YES') 
            
            list_shift_epic_proj[0].to_netcdf('output_shocks'+region +sim_round +"/epic_"+model_full+'_'+rcp_scenario+"_"+co2_scenario+"_yield_soybean_shift_2017-2044.nc")
            list_shift_epic_proj[1].to_netcdf('output_shocks'+region +sim_round +"/epic_"+model_full+'_'+rcp_scenario+"_"+co2_scenario+"_yield_soybean_shift_2044-2071.nc")
            list_shift_epic_proj[2].to_netcdf('output_shocks'+region +sim_round +"/epic_"+model_full+'_'+rcp_scenario+"_"+co2_scenario+"_yield_soybean_shift_2071-2098.nc")
            
            
            list_shift_clim_proj = local_minima_30years(DS_pred_clim_proj, weights = 'YES') 
            
            list_shift_clim_proj[0].to_netcdf('output_shocks'+region +sim_round +"/clim_"+model_full+'_'+rcp_scenario+"_"+co2_scenario+"_yield_soybean_shift_2017-2044.nc")
            list_shift_clim_proj[1].to_netcdf('output_shocks'+region +sim_round +"/clim_"+model_full+'_'+rcp_scenario+"_"+co2_scenario+"_yield_soybean_shift_2044-2071.nc")
            list_shift_clim_proj[2].to_netcdf('output_shocks'+region +sim_round +"/clim_"+model_full+'_'+rcp_scenario+"_"+co2_scenario+"_yield_soybean_shift_2071-2098.nc")
            
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
            
        
        return
    
    if co2_scen == 'both':
        epic_projections_function_co2(co2_scenario = 'default')
        epic_projections_function_co2(co2_scenario = '2015co2')
    
    else:
        co2_scenario = co2_scen
        epic_projections_function_co2(co2_scenario)
    
    return



#%% START MAIN SCRIPT

# UKESM model
projections_generation_hybrid(model = 'ukesm', rcp_scenario = 'ssp585', region = "_us", hybrid_model_full = full_model_hyb2, start_date='01-01-2015', end_date='31-12-2100', co2_scen = 'both')
projections_generation_hybrid(model = 'ukesm', rcp_scenario = 'ssp126', region = "_us", hybrid_model_full = full_model_hyb2, start_date='01-01-2015', end_date='31-12-2100', co2_scen = 'both')

# GFDL model
projections_generation_hybrid(model = 'gfdl', rcp_scenario = 'ssp585', region = "_us", hybrid_model_full = full_model_hyb2, start_date='01-01-2015', end_date='31-12-2100', co2_scen = 'both')
projections_generation_hybrid(model = 'gfdl', rcp_scenario = 'ssp126', region = "_us", hybrid_model_full = full_model_hyb2, start_date='01-01-2015', end_date='31-12-2100', co2_scen = 'both')

# IPSL model
projections_generation_hybrid(model = 'ipsl', rcp_scenario = 'ssp585', region = "_us", hybrid_model_full = full_model_hyb2, start_date='01-01-2015', end_date='31-12-2100', co2_scen = 'both')
projections_generation_hybrid(model = 'ipsl', rcp_scenario = 'ssp126', region = "_us", hybrid_model_full = full_model_hyb2, start_date='01-01-2015', end_date='31-12-2100', co2_scen = 'both')


##### Shifters weighted yield

# # UKESM 585
# year_min: 2015 year_max: 2044 min_pos 2041 shifter [2041.0, 0.7732478380203247]
# year_min: 2043 year_max: 2072 min_pos 2056 shifter [2056.0, 0.814445436000824]
# year_min: 2071 year_max: 2100 min_pos 2086 shifter [2086.0, 0.745225727558136]

# # UKESM 126
# year_min: 2015 year_max: 2044 min_pos 2040 shifter [2040.0, 0.8048002123832703]
# year_min: 2043 year_max: 2072 min_pos 2054 shifter [2054.0, 0.7392540574073792]
# year_min: 2071 year_max: 2100 min_pos 2098 shifter [2098.0, 0.8095159530639648]


# # GFDL model 585
# year_min: 2015 year_max: 2044 min_pos 2034 shifter [2034.0, 0.8223055005073547]
# year_min: 2043 year_max: 2072 min_pos 2062 shifter [2062.0, 0.751900851726532]
# year_min: 2071 year_max: 2100 min_pos 2093 shifter [2093.0, 0.5870298743247986]

# # GFDL model 126
# year_min: 2015 year_max: 2044 min_pos 2015 shifter [2015.0, 0.8127191066741943]
# year_min: 2043 year_max: 2072 min_pos 2047 shifter [2047.0, 0.9065349698066711]
# year_min: 2071 year_max: 2100 min_pos 2083 shifter [2083.0, 0.8539149761199951]

# # IPSL model 585
# year_min: 2015 year_max: 2044 min_pos 2026 shifter [2026.0, 0.8059989809989929]
# year_min: 2043 year_max: 2072 min_pos 2062 shifter [2062.0, 0.7766252160072327]
# year_min: 2071 year_max: 2100 min_pos 2096 shifter [2096.0, 0.7489902377128601]

# # IPSL model 126
# year_min: 2015 year_max: 2044 min_pos 2026 shifter [2026.0, 0.8544405102729797]
# year_min: 2043 year_max: 2072 min_pos 2060 shifter [2060.0, 0.8020848631858826]
# year_min: 2071 year_max: 2100 min_pos 2092 shifter [2092.0, 0.8079625964164734]


















