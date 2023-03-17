# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:01:10 2022

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
import cartopy.io.shapereader as shpreader
import matplotlib as mpl

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['figure.dpi'] = 144
mpl.rcParams.update({'font.size': 14})

# =============================================================================
#  load local Functions 
# =============================================================================
       
from functions_p2 import plot_2d_am_map, detrend_dataset, detrend_dim_2, timedelta_to_int, convert_timeunit, rearrange_latlot
from functions_p2 import weighted_conversion, dynamic_calendar


#%% FUNCTION TO GENERATE PROJECTIONS BASED ON RANDOM FOREST
def projections_generation_hybrid(model, rcp_scenario, region, hybrid_model_full, start_date, end_date, co2_scen='both', three_models = False, sim_round = '\Gen_Assem'):
	# load data from historical period to use as reference for the future projectons - covered areas, detrended values
	DS_obs_ref = DS_predict_test_hist # DS_predict_clim_2012_mask DS_predict_test_hist
	DS_features_hist = DS_feature_season_6mon_am_det # DS_feature_season_6mon_2012 DS_feature_season_6mon_am_det
	DS_y_epic_hist = DS_y_epic_am_det_regul # DS_y_epic_am_2012 DS_y_epic_am_det_regul
    
	list_features_am = list_features # From historical time
	DS_calendar_hist = DS_chosen_calendar_am
	df_calendar_hist = df_calendar_month_am # not the same as DS

	if region == '_am' or 'am':
        #COMBINE THE THREE REGIONS WITH SOUTH AMERICA 12 MONTHS SHIFTED
		DS_clim_ext_projections_us = xr.open_mfdataset('monthly_'+ model +'_'+ rcp_scenario + "_us" +'/*.nc')
        
        # Shift br data one year ahead
		DS_clim_ext_projections_br = xr.open_mfdataset('monthly_'+ model +'_'+ rcp_scenario + "_br" +'/*.nc')
		DS_clim_ext_projections_br = DS_clim_ext_projections_br.copy().shift(time = 12) # SHIFT EPIC BR ONE YEAR FORWARD
       
        # Shift ARG data one year ahead
		DS_clim_ext_projections_arg = xr.open_mfdataset('monthly_'+ model +'_'+ rcp_scenario + "_arg" +'/*.nc')
		DS_clim_ext_projections_arg = DS_clim_ext_projections_arg.copy().shift(time = 12) # SHIFT EPIC BR ONE YEAR FORWARD
        
        # Combine all grids
		DS_clim_ext_projections = DS_clim_ext_projections_us.combine_first(DS_clim_ext_projections_br)
		DS_clim_ext_projections = DS_clim_ext_projections.combine_first(DS_clim_ext_projections_arg)
		DS_clim_ext_projections = DS_clim_ext_projections.sel(time=slice(start_date, end_date))
        
        # Reindex to avoid missnig coordinates and dimension values
		DS_clim_ext_projections = rearrange_latlot(DS_clim_ext_projections)

        #control
		plot_2d_am_map(DS_clim_ext_projections['txm'].isel(time = 0), title = 'First year projection')
		plot_2d_am_map(DS_clim_ext_projections['txm'].isel(time = -1), title = 'Lat year projection')

	if model == 'ukesm':
		model_full = 'ukesm1-0-ll'
	elif model == 'gfdl':
		model_full = 'gfdl-esm4'
	elif model == 'ipsl':
		model_full = 'ipsl-cm6a-lr'

    ### Climatic variables
    # Clean
	DS_clim_ext_projections = DS_clim_ext_projections.drop_vars(['fd','id','time_bnds']) # Always zero
    # Selected features
	DS_clim_ext_projections = DS_clim_ext_projections[list_features_am]
    
	DS_clim_ext_projections = DS_clim_ext_projections.where(DS_obs_ref['Yield'].mean('time') >= -5.0)
	plot_2d_am_map(DS_clim_ext_projections['prcptot'].mean('time'))
      
	da_list = []
	for feature in list(DS_clim_ext_projections.keys()):
		if (type(DS_clim_ext_projections[feature].values[0,0,0]) == np.timedelta64):
			print('Time')
			DS = timedelta_to_int(DS_clim_ext_projections, feature)
		else:
			print('Integer')
			DS = DS_clim_ext_projections[feature]
		da_list.append(DS)
    
	DS_clim_ext_projections_combined = xr.merge(da_list)    

	DS_clim_ext_projections_combined = rearrange_latlot(DS_clim_ext_projections_combined)
	DS_clim_ext_projections_combined = DS_clim_ext_projections_combined.reindex(lat=DS_clim_ext_projections_combined.lat[::-1])
	if len(DS_clim_ext_projections_combined.coords) >3 :
		DS_clim_ext_projections_combined = DS_clim_ext_projections_combined.drop('spatial_ref')
        
	DS_clim_ext_projections_am = DS_clim_ext_projections_combined.where( DS_calendar_hist >= 0 )
    
    # =============================================================================
    # CONVERT CLIMATIC VARIABLES ACCORDING TO THE SOYBEAN GROWING SEASON PER GRIDCELL 
    # =============================================================================
	df_clim_proj_twoyears = dynamic_calendar(DS_clim_ext_projections_am, df_calendar_hist)
        
    ### Select specific months
	suffixes = tuple(["_"+str(j) for j in range(1,4)])
	df_feature_proj_6mon = df_clim_proj_twoyears.loc[:,df_clim_proj_twoyears.columns.str.endswith(suffixes)]
    
	df_feature_proj_6mon = df_feature_proj_6mon.rename_axis(index={'year':'time'})
	df_feature_proj_6mon = df_feature_proj_6mon.reorder_levels(['time','lat','lon']).sort_index()
	df_feature_proj_6mon = df_feature_proj_6mon.dropna()
	df_feature_proj_6mon = df_feature_proj_6mon.astype(float)

    # =============================================================================
    # # SECOND DETRENDING PART - SEASONAL CYCLE
    # # If to keep historical mean values: DS_feature_season_6mon_am_det, if to keep projections mean: DS_feature_proj_6mon_am
    # =============================================================================
	DS_feature_proj_6mon_am = xr.Dataset.from_dataframe(df_feature_proj_6mon)
	DS_feature_proj_6mon_am = rearrange_latlot(DS_feature_proj_6mon_am)
    
    # Choose historical mean so there is no mean deviation between the historical and future timelines
	DS_feature_proj_6mon_am_det = detrend_dataset(DS_feature_proj_6mon_am, deg = 'free', 
                                                  mean_data = DS_features_hist.sel(time = slice(2000,2015))) #<<-- important to align values
    
	df_feature_proj_6mon_am_det = DS_feature_proj_6mon_am_det.to_dataframe().dropna()
	df_feature_proj_6mon_am_det = df_feature_proj_6mon_am_det.rename_axis(list(DS_features_hist.coords)).reorder_levels(['time','lat','lon']).sort_index()
    
	for feature in df_feature_proj_6mon.columns:
		df_feature_proj_6mon[feature].groupby('time').mean().plot(label = 'old')
		df_feature_proj_6mon_am_det[feature].groupby('time').mean().plot(label = 'detrend')
		plt.title(f'{feature} for {model_full}_{rcp_scenario}')
		plt.legend()
		plt.show()
    
	list_feat_precipitation = [s for s in df_feature_proj_6mon_am_det.keys() if "prcptot" in s]
	for feature in list_feat_precipitation:
		df_feature_proj_6mon_am_det[feature][df_feature_proj_6mon_am_det[feature] < 0] = 0
    
    # UPDATE DETRENDED VALUES
	df_feature_proj_6mon = df_feature_proj_6mon_am_det
    # DS_features = xr.Dataset.from_dataframe(df_feature_proj_6mon)
    # DS_features.to_netcdf('output_clim_indices_'+region +'/clim_indices_'+model_full+'_'+rcp_scenario+'_soybean_2015_2100.nc')
    
    # =============================================================================
    #     #%% EPIC projections
    # =============================================================================
	def epic_projections_function_co2(co2_scenario):        
		if region == '_am' or 'am':
			DS_y_epic_proj = xr.open_dataset("epic-iiasa_"+ model_full +"_w5e5_"+rcp_scenario+"_2015soc_"+co2_scenario+"_yield-soy-noirr_global_annual_2015_2100.nc", decode_times=False)
			DS_y_epic_proj = convert_timeunit(DS_y_epic_proj)
            
            # EPIC DATA USA - keep as it is
			DS_y_epic_proj_us = DS_y_epic_proj.where(DS_y_epic_us['yield'].mean('time') >= -5.0 )
            
            # Shift br data one year ahead
			DS_y_epic_proj_br = DS_y_epic_proj.where(DS_y_epic_br['yield'].mean('time') >= -5.0 )
			DS_y_epic_proj_br = DS_y_epic_proj_br.shift(time = 1) # SHIFT EPIC BR ONE YEAR FORWARD
            
            # Shift ARG data one year ahead
			DS_y_epic_proj_arg = DS_y_epic_proj.where(DS_y_epic_arg['yield'].mean('time') >= -5.0 )
			DS_y_epic_proj_arg = DS_y_epic_proj_arg.shift(time = 1) # SHIFT EPIC BR ONE YEAR FORWARD
                        
            # Combine all grids
			DS_y_epic_proj_am = DS_y_epic_proj_us.combine_first(DS_y_epic_proj_br)
			DS_y_epic_proj_am = DS_y_epic_proj_am.combine_first(DS_y_epic_proj_arg)
            
            # Reindex to avoid missnig coordinates and dimension values
			DS_y_epic_proj_am = DS_y_epic_proj_am.reindex(lat=DS_y_epic_proj_am.lat[::-1])
			DS_y_epic_proj_am = DS_y_epic_proj_am.sortby(['time','lat','lon'])
			DS_y_epic_proj_am = rearrange_latlot(DS_y_epic_proj_am)
            
			DS_y_epic_proj_am = DS_y_epic_proj_am.where(DS_y_epic_hist.mean('time') >= -5.0 )
            
		else:
			DS_y_epic_proj = xr.open_dataset("epic-iiasa_"+ model_full +"_w5e5_"+rcp_scenario+"_2015soc_"+co2_scenario+"_yield-soy-noirr_global_annual_2015_2100.nc", decode_times=False)
			DS_y_epic_proj_am = DS_y_epic_proj.where(DS_y_epic_hist.mean('time') >= -5.0 )
                    
		return DS_y_epic_proj_am.sel(time = slice(2016,2100)), DS_feature_proj_6mon_am, df_feature_proj_6mon
    # =============================================================================    
    
	if (co2_scen == 'default') or (co2_scen == '2015co2'):
		DS_y_epic_proj_am, DS_feature_proj_6mon_am, df_feature_proj_6mon2 = epic_projections_function_co2(co2_scenario = co2_scen)
        
	else:
		raise ValueError('Only forms of accepted co2_scen: <default>, <2015co2> or <both>') 
    
	return DS_y_epic_proj_am, DS_feature_proj_6mon_am, df_feature_proj_6mon2
    
        
#%% HYBRID model projections
def hybrid_predictions_future(DS_y_epic_proj_am, df_feature_proj_6mon2, model, rcp_scenario, hybrid_model_full, region = "_am", co2_scen = 'Default'): #- DS_fit.sel(time = 2016)
    end_year = 2100
    DS_y_epic_hist = DS_y_epic_am_det_regul.where(DS_y_epic_proj_am['yield-soy-noirr'].mean('time') > -100) # DS_y_epic_am_2012 #DS_y_epic_am_det_regul
    DS_features_hist = DS_feature_season_6mon_am_det.where(DS_y_epic_proj_am['yield-soy-noirr'].mean('time') > -100) # DS_feature_season_6mon_2012 DS_feature_season_6mon_am_det
 
    if model == 'ukesm':
        model_full = 'ukesm1-0-ll'
    elif model == 'gfdl':
        model_full = 'gfdl-esm4'
    elif model == 'ipsl':
        model_full = 'ipsl-cm6a-lr'
        
    # load EPIC - Mean values depend on what is porjected: future mean or historical mean
    DS_detrended, DS_fit = detrend_dim_2(DS_y_epic_proj_am['yield-soy-noirr'], 'time')
    
    DS_fit_mean = xr.DataArray( DS_fit + DS_detrended.mean(['time']), name= DS_y_epic_proj_am['yield-soy-noirr'].name, 
                               attrs = DS_y_epic_proj_am['yield-soy-noirr'].attrs)

	# Detrend & adjust according to historical mean values per grid cell
    DS_y_epic_proj_am_det = xr.DataArray( DS_detrended + DS_y_epic_hist.sel(time = slice(2000, 2015)).mean('time'), 
                                         name= DS_y_epic_proj_am['yield-soy-noirr'].name, attrs = DS_y_epic_proj_am['yield-soy-noirr'].attrs)

    DS_y_epic_proj_am['yield-soy-noirr'].mean(['lat', 'lon']).plot(label = 'EPIC')
    DS_fit_mean.mean(['lat', 'lon']).plot(label = 'EPIC trend')
    DS_y_epic_proj_am_det.mean(['lat', 'lon']).plot(label = 'EPIC detrended')
    plt.legend()
    plt.title('Detrend'+ "_"+ model + "_"+ rcp_scenario+"_"+co2_scen)
    plt.tight_layout()
    plt.show()
        
    df_y_epic_proj_am = DS_y_epic_proj_am_det.to_dataframe().dropna()
    df_y_epic_proj_am = df_y_epic_proj_am.reorder_levels(['time','lat','lon']).sort_index()
    
    df_y_epic_proj_am = df_y_epic_proj_am.where(df_feature_proj_6mon2['prcptot_3'] > -100).dropna()
    df_feature_proj_6mon2 = df_feature_proj_6mon2.where(df_y_epic_proj_am['yield-soy-noirr'] > -100).dropna()
    
    ####   Prepare data for hybrid model
    df_input_hybrid_fut_test_2 = pd.concat([df_y_epic_proj_am, df_feature_proj_6mon2], axis = 1 ).query(f"time>=2016 and time <= {end_year}")
    
    # Add country information to dataframe
    DS_epic_us = DS_y_epic_proj_am_det.where(DS_y_obs_us_all['Yield'].sel(time = 2012) > -10)
    DS_epic_br = DS_y_epic_proj_am_det.where(DS_y_obs_br['Yield'].sel(time = 2012) > -10)
    DS_epic_arg = DS_y_epic_proj_am_det.where(DS_y_obs_arg['Yield'].sel(time = 2012) > -10)
    
    def country_location_add(df, US = DS_y_obs_us_all['Yield'], BR = DS_y_obs_br['Yield'], ARG = DS_y_obs_arg['Yield']):
        DS_country_us = xr.where(US >-100, 'US', np.nan)
        DS_country_br = xr.where(BR >-100, 'BR', np.nan)
        DS_country_arg = xr.where(ARG >-100, 'ARG', np.nan)
        DS_country_all = xr.merge([DS_country_us, DS_country_br, DS_country_arg], compat='no_conflicts')
        df_country_all = DS_country_all.to_dataframe().dropna().reorder_levels(['time','lat','lon']).sort_index()
        df_country_am = df_country_all.where(df[df.columns[0]] > -100).dropna()
        df = pd.concat([df, df_country_am], axis =1 )
        df = pd.get_dummies(df)
        return df
    
    df_input_hybrid_fut_test_2 = country_location_add(df_input_hybrid_fut_test_2, US = DS_epic_us, BR = DS_epic_br, ARG = DS_epic_arg)
    
    # Predicting hybrid results
    df_prediction_proj_2 = df_y_epic_proj_am.query(f"time>=2016 and time <= {end_year}").copy() 
    predic_model_test_2 = hybrid_model_full.predict(df_input_hybrid_fut_test_2.query(f"time>=2016 and time <= {end_year}")).copy()
    df_prediction_proj_2.loc[:,'yield-soy-noirr'] = predic_model_test_2
    
    DS_hybrid_proj_2 = xr.Dataset.from_dataframe(df_prediction_proj_2)
    # Reindex to avoid missnig coordinates and dimension values
    DS_hybrid_proj_2 = rearrange_latlot(DS_hybrid_proj_2)
  
    plt.plot(DS_hybrid_proj_2.time.values, DS_hybrid_proj_2['yield-soy-noirr'].mean(['lat','lon']), label = 'Hybrid')
    plt.plot(DS_y_epic_proj_am_det.time.values, DS_y_epic_proj_am_det.mean(['lat','lon']), label = 'EPIC pure')
    plt.title('Hybrid'+ "_"+ model + "_"+rcp_scenario+"_"+co2_scen)
    plt.legend()
    plt.show()
    
    plt.plot(DS_hybrid_proj_2.time.values, DS_hybrid_proj_2['yield-soy-noirr'].mean(['lat','lon'])/ DS_hybrid_proj_2['yield-soy-noirr'].mean(), label = 'Hybrid')
    plt.plot(DS_y_epic_proj_am_det.time.values, DS_y_epic_proj_am_det.mean(['lat','lon'])/DS_y_epic_proj_am_det.mean(), label = 'EPIC pure')
    plt.title('Hybrid'+ "_"+ model + "_"+rcp_scenario+"_"+co2_scen)
    plt.ylabel('Shock (0-1)')
    plt.legend()
    plt.show()
    
    # EPIC model
    df_y_epic_proj_am_loc = country_location_add(df_y_epic_proj_am, US = DS_epic_us, BR = DS_epic_br, ARG = DS_epic_arg)
    df_predic_epic_test = df_y_epic_proj_am.query(f"time>=2016 and time <= {end_year}").copy()
    df_predic_epic_test.loc[:,'yield-soy-noirr'] = full_model_epic_am.predict(df_y_epic_proj_am_loc.query(f"time>=2016 and time <= {end_year}"))
    
    DS_pred_epic_proj = xr.Dataset.from_dataframe(df_predic_epic_test)
    DS_pred_epic_proj = rearrange_latlot(DS_pred_epic_proj)

    # CLIMATIC model
    df_feature_proj_6mon2_loc = country_location_add(df_feature_proj_6mon2, US = DS_epic_us, BR = DS_epic_br, ARG = DS_epic_arg)
    df_pred_clim_proj_test = df_y_epic_proj_am.query(f"time>=2016 and time <= {end_year}").copy()
    df_pred_clim_proj_test.loc[:,'yield-soy-noirr'] = full_model_exclim_dyn_am.predict(df_feature_proj_6mon2_loc.query(f"time>=2016 and time <= {end_year}"))
    
    DS_pred_clim_proj = xr.Dataset.from_dataframe(df_pred_clim_proj_test)
    DS_pred_clim_proj = rearrange_latlot(DS_pred_clim_proj)

# =============================================================================
#     #%% Full analysis
# =============================================================================
    
    # WEIGHTING SCHEME
    DS_mirca_test = xr.open_dataset("../../paper_hybrid_agri/data/americas_mask_ha.nc", decode_times=False).rename({'latitude': 'lat', 'longitude': 'lon'})
    DS_mirca_test = DS_mirca_test.rename({'annual_area_harvested_rfc_crop08_ha_30mn':'harvest_area'}).where(DS_hybrid_proj_2['yield-soy-noirr'].mean('time')>-2)
    # TEST WITH SPAM 2010
    DS_mirca_test = xr.open_dataset("../../paper_hybrid_agri/data/soy_harvest_spam_native_05x05.nc", decode_times=False)

    #### HARVEST DATA
    DS_harvest_area_globiom = xr.load_dataset("../../paper_hybrid_agri/data/soybean_harvest_area_calculated_americas_hg.nc", decode_times=False)
    DS_harvest_area_globiom = DS_harvest_area_globiom.sel(time=2012)
    DS_harvest_area_globiom = DS_harvest_area_globiom.where(DS_mirca_test['harvest_area'] > 0 )
    DS_harvest_area_globiom = rearrange_latlot(DS_harvest_area_globiom)
    # plot_2d_am_map(DS_harvest_area_globiom['harvest_area'])
               
    DS_hybrid_weighted = weighted_conversion(DS_hybrid_proj_2['yield-soy-noirr'], DS_harvest_area_globiom, name_ds = 'yield-soy-noirr')
    DS_epic_weighted = weighted_conversion(DS_pred_epic_proj['yield-soy-noirr'], DS_harvest_area_globiom, name_ds = 'yield-soy-noirr')
    DS_clim_weighted = weighted_conversion(DS_pred_clim_proj['yield-soy-noirr'], DS_harvest_area_globiom, name_ds = 'yield-soy-noirr')
    DS_epic_pure_weighted = weighted_conversion(DS_y_epic_proj_am_det, DS_harvest_area_globiom, name_ds = 'yield-soy-noirr')
    DS_epic_pure_weighted = DS_epic_pure_weighted.sel(time = slice(2015,2099))
    
    # Country analysis
    # US
    DS_historical_hybrid_us = xr.load_dataset("output_models_am/hybrid_epic_us-iiasa_gswp3-w5e5_obsclim_2015soc_default_yield-soy-noirr_global_annual_1901_2016.nc")
    DS_historical_hybrid_br = xr.load_dataset("output_models_am/hybrid_epic_br-iiasa_gswp3-w5e5_obsclim_2015soc_default_yield-soy-noirr_global_annual_1901_2016.nc")
    DS_historical_hybrid_arg = xr.load_dataset("output_models_am/hybrid_epic_arg-iiasa_gswp3-w5e5_obsclim_2015soc_default_yield-soy-noirr_global_annual_1901_2016.nc")

    for country in [DS_historical_hybrid_us, DS_historical_hybrid_br, DS_historical_hybrid_arg]:
        DS_mirca_us = DS_harvest_area_globiom.where(country['Yield'].sel(time = 2012) > -10)
        
        DS_epic_hist_weighted = weighted_conversion(DS_y_epic_hist, DS_mirca_us, name_ds = 'yield-soy-noirr')
        DS_epic_proj_det_weighted = weighted_conversion(DS_y_epic_proj_am_det, DS_mirca_us, name_ds = 'yield-soy-noirr')
        
        DS_obs_hist_weighted = weighted_conversion(DS_predict_test_hist['Yield'], DS_mirca_us, name_ds = 'yield-soy-noirr')
        DS_hybrid_us_weighted = weighted_conversion(DS_hybrid_proj_2['yield-soy-noirr'], DS_mirca_us, name_ds = 'yield-soy-noirr')
        print("difference between mean EPIC projection and climatology:", round(DS_epic_proj_det_weighted['yield-soy-noirr'].mean().values / DS_epic_hist_weighted['yield-soy-noirr'].sel(time=slice(2000,2015)).mean().values, 3))
        
        DS_feature_proj_6mon_am_det = xr.Dataset.from_dataframe(df_feature_proj_6mon2)
        DS_feature_proj_6mon_am_det = rearrange_latlot(DS_feature_proj_6mon_am_det)
        for feature in df_feature_proj_6mon2.columns:
            DS_features_hist_weighted = weighted_conversion(DS_features_hist[feature], DS_mirca_us, name_ds = feature)
            DS_feature_proj_6mon_am_det_weighted = weighted_conversion(DS_feature_proj_6mon_am_det[feature], DS_mirca_us, name_ds = feature)
        
            print(f"difference between mean {feature} projection and climatology:", round(DS_feature_proj_6mon_am_det_weighted[feature].mean().values / DS_features_hist_weighted[feature].sel(time=slice(2000,2015)).mean().values, 3))
            
        print("difference between mean Hybrid projection and observed climatology:", round(DS_hybrid_us_weighted['yield-soy-noirr'].mean().values / DS_obs_hist_weighted['yield-soy-noirr'].sel(time=slice(2000,2015)).mean().values, 3))
            
    DS_dif_fut_past_epic = (DS_y_epic_proj_am_det.mean('time') / DS_y_epic_hist.sel(time=slice(2000,2015)).mean('time') ) - 1
    DS_dif_fut_past_obs = (DS_hybrid_proj_2['yield-soy-noirr'].mean('time') / DS_predict_test_hist['Yield'].sel(time=slice(2000,2015)).mean('time') ) - 1
    
    for feature in df_feature_proj_6mon2.columns:
        
        DS_dif_fut_past_prcptot = (DS_feature_proj_6mon_am_det[feature].mean('time') / DS_features_hist[feature].sel(time=slice(2000,2015)).mean('time') ) - 1
        plot_2d_am_map(DS_dif_fut_past_prcptot, title = f'Dif fut {feature} and past hybrid - bias', vmin = -0.1, vmax = 0.1, colormap = 'RdBu' )

    plot_2d_am_map(DS_dif_fut_past_epic, title = 'Dif EPIC - fut and past hybrid - bias', vmin = -0.1, vmax = 0.1, colormap = 'RdBu' )
    plot_2d_am_map(DS_dif_fut_past_obs, title = 'Dif fut and past hybrid - bias', vmin = -0.1, vmax = 0.1, colormap = 'RdBu' )

    # plt.plot(DS_hybrid_proj_2['yield-soy-noirr'].mean(['lat','lon']), label = 'unweighted')
    # plt.plot(DS_hybrid_weighted['yield-soy-noirr'], label = 'weighted')
    # plt.ylabel('yield')
    # plt.xlabel('years')
    # plt.title('Difference weighted and non-weighted')
    # plt.legend()
    # plt.show()
            
    # plt.plot(DS_y_epic_proj_am_det.time.values, DS_y_epic_proj_am_det.mean(['lat','lon']), label = 'Pure EPIC')
    # plt.plot(DS_pred_epic_proj.time.values, DS_pred_epic_proj["yield-soy-noirr"].mean(['lat','lon']), label = 'EPIC-RF')
    # plt.plot(DS_hybrid_proj_2.time.values, DS_hybrid_proj_2["yield-soy-noirr"].mean(['lat','lon']), label = 'Hybrid-RF')
    # plt.title('Non-weighted comparison')
    # plt.legend()
    # # plt.show()
    
    # plt.plot(DS_y_epic_proj_am_det.time.values, DS_y_epic_proj_am_det.mean(['lat','lon']), label = 'Pure EPIC')
    # plt.plot(DS_pred_epic_proj.time.values, DS_epic_weighted["yield-soy-noirr"], label = 'EPIC-RF')
    # plt.plot(DS_hybrid_proj_2.time.values, DS_hybrid_weighted['yield-soy-noirr'], label = 'Hybrid-RF')
    # plt.title('Weighted comparison')
    # plt.legend()
    # plt.show()
    
    # plt.figure(figsize=(7,7), dpi=250) #plot clusters
    plt.plot(DS_pred_epic_proj.time.values, DS_epic_weighted["yield-soy-noirr"]/np.mean(DS_epic_weighted["yield-soy-noirr"]), label = 'EPIC-RF')
    plt.plot(DS_pred_clim_proj.time.values, DS_clim_weighted["yield-soy-noirr"]/np.mean(DS_clim_weighted["yield-soy-noirr"]), label = 'CLIM-RF')
    plt.plot(DS_hybrid_proj_2.time.values, DS_hybrid_weighted['yield-soy-noirr']/np.mean(DS_hybrid_weighted["yield-soy-noirr"]), label = 'Hybrid')
    plt.title("Weighted"+ "_" + model + "_"+ rcp_scenario+"_"+co2_scen)
    plt.ylabel("Shock (%)")
    plt.legend()
    plt.show()
    
    # FINAL COMPARISON EPIC WITH HYBRID
    # plt.figure(figsize=(7,7), dpi=250) #plot clusters
    plt.plot(DS_hybrid_proj_2.time.values, DS_hybrid_weighted['yield-soy-noirr']/np.mean(DS_hybrid_weighted["yield-soy-noirr"]), label = 'Hybrid')
    plt.plot(DS_epic_pure_weighted.time.values, DS_epic_pure_weighted['yield-soy-noirr']/np.mean(DS_epic_pure_weighted["yield-soy-noirr"]), label = 'EPIC PURE')
    plt.title("Weighted"+ "_" + model + "_"+ rcp_scenario+"_"+co2_scen)
    plt.ylabel("Shock (%)")
    plt.legend()
    plt.show()
    
    # # difference shock
    # weighted_timeseries = DS_hybrid_weighted['yield-soy-noirr']/np.mean(DS_hybrid_weighted["yield-soy-noirr"])
    # weighted_timeseries_min = weighted_timeseries.idxmin()       
    # plot_2d_am_map(DS_y_epic_proj_am_det.sel(time=weighted_timeseries_min)/DS_y_epic_proj_am_det.mean('time'), title = 'Orig. EPIC projection')
    # plot_2d_am_map(DS_hybrid_proj_2['yield-soy-noirr'].sel(time=weighted_timeseries_min)/DS_hybrid_proj_2["yield-soy-noirr"].mean('time'), title = 'Hybrid projection')
    # plot_2d_am_map(DS_hybrid_proj_2['yield-soy-noirr'].sel(time=weighted_timeseries_min)/DS_hybrid_proj_2["yield-soy-noirr"].mean('time') - DS_y_epic_proj_am_det.sel(time=weighted_timeseries_min)/DS_y_epic_proj_am_det.mean('time'), title = 'Difference Hybrid and EPIC')
    
    ### Save the data
    df_input_hybrid_fut_test_2.to_csv('output_models'+region +'/climatic_projections/model_input_'+model_full+'_'+rcp_scenario+'_'+co2_scen+'_2015_2100.csv')
    
    DS_hybrid_proj_2.to_netcdf('output_models'+region +'/hybrid_'+model_full+'_'+rcp_scenario+'_'+co2_scen+'_yield_soybean_2015_2100.nc')
    DS_pred_clim_proj.to_netcdf('output_models'+region +'/clim_'+model_full+'_'+rcp_scenario+'_'+co2_scen+'_yield_soybean_2015_2100.nc')
    DS_pred_epic_proj.to_netcdf('output_models'+region +'/epic_'+model_full+'_'+rcp_scenario+'_'+co2_scen+'_yield_soybean_2015_2100.nc')
    # plot_2d_am_map(DS_hybrid_proj_2['yield-soy-noirr'].mean('time'))

    ### Save trends
    DS_hybrid_proj_trend = DS_hybrid_proj_2 + DS_fit - DS_fit.sel(time = slice(2016,2020)).mean('time')
    DS_hybrid_proj_trend.to_netcdf('output_models'+region +'/hybrid_trends/hybrid_trend_'+model_full+'_'+rcp_scenario+'_'+co2_scen+'_yield_soybean_2015_2100.nc')

    
# =============================================================================
# GLOBIOM SHIFTERS
# =============================================================================
   
    # def local_minima_30years(dataset, no_cycles = 3, weights = 'YES'):

    #     cycle_period = (( dataset.time.max() - dataset.time.min() ) // no_cycles ) + 1
    #     year0 = dataset.time[0].values
        
    #     list_shifters_cycle = []
    #     for cycle in range(no_cycles):
            
    #         dataset_cycle = dataset.sel( time = slice( year0 + cycle_period * cycle, year0 -1 + cycle_period * ( cycle+1 ) ) )
            
    #         if weights == 'NO':
    
    #             year_min = dataset_cycle.time[0].values + dataset_cycle['yield-soy-noirr'].mean(['lat','lon']).argmin(dim='time').values
    #             ds_local_minima = dataset_cycle.sel(time=year_min)
    #             ds_shifter_cycle = ds_local_minima / dataset_cycle.mean(['time'])
    #             print('year_min:',dataset_cycle.time[0].values, 'year_max:',dataset_cycle.time[-1].values, 'min_pos',year_min, 'shifter', list(ds_shifter_cycle.to_dataframe().mean().values))
                
    #         elif weights == 'YES':
                
    #             DS_weighted_cycle = weighted_conversion(dataset, DS_harvest_area_globiom, name_ds = 'yield-soy-noirr')
    #             dataset_cycle_weight = DS_weighted_cycle.sel( time = slice( year0 + cycle_period * cycle, year0 - 1 + cycle_period * ( cycle+1 ) ) )
     
    #             year_min = dataset_cycle_weight.time[0].values + dataset_cycle_weight['yield-soy-noirr'].argmin(dim='time').values
    #             ds_local_minima = dataset_cycle.sel(time=year_min)
    #             ds_shifter_cycle = ds_local_minima / dataset_cycle.mean(['time'])     
                
    #             # Reindex to avoid missnig coordinates and dimension values
    #             new_lat = np.arange(ds_shifter_cycle.lat[0], ds_shifter_cycle.lat[-1], 0.5)
    #             new_lon = np.arange(ds_shifter_cycle.lon[0], ds_shifter_cycle.lon[-1], 0.5)
    #             ds_shifter_cycle = ds_shifter_cycle.reindex({'lat':new_lat})
    #             ds_shifter_cycle = ds_shifter_cycle.reindex({'lon':new_lon})
                
    #             print('year_min:', dataset_cycle_weight.time[0].values, 'year_max:',dataset_cycle_weight.time[-1].values, '/ Minimum year:',year_min, '/ Shifter value:', dataset_cycle_weight['yield-soy-noirr'].min(dim='time').values/dataset_cycle_weight['yield-soy-noirr'].mean(dim='time').values)
                
    #         list_shifters_cycle.append(ds_shifter_cycle)

    #     return list_shifters_cycle
            
    # list_test = local_minima_30years(DS_hybrid_proj_2, weights = 'YES') 
    
    # sim_round = '/scenarios_forum'
    # # Hybrid model
    # list_test[0].to_netcdf('output_shocks'+region +sim_round +"/hybrid_"+model_full+'_'+rcp_scenario+"_"+co2_scen+"_yield_soybean_shift_2017-2044.nc")
    # list_test[1].to_netcdf('output_shocks'+region +sim_round +"/hybrid_"+model_full+'_'+rcp_scenario+"_"+co2_scen+"_yield_soybean_shift_2044-2071.nc")
    # list_test[2].to_netcdf('output_shocks'+region + sim_round +"/hybrid_"+model_full+'_'+rcp_scenario+"_"+co2_scen+"_yield_soybean_shift_2071-2098.nc")
    
    df_input_hybrid_fut_test_2 = df_input_hybrid_fut_test_2.rename(columns = {'yield-soy-noirr':f'yield-soy-noirr_{co2_scen}'})
    df_prediction_proj_2 = df_prediction_proj_2.rename(columns = {'yield-soy-noirr':f'yield-soy-noirr_{co2_scen}'})
        
    return df_input_hybrid_fut_test_2, df_prediction_proj_2            

#%% START MAIN SCRIPT

# Runs future scenarios
model_to_be_used = full_model_hyb_am2 #full_model_hyb # or: full_model_hyb_am2
co2_scen = 'default' # 'both'
# =============================================================================
# First function - data preparation for future scenarios - dynamic calendar and more
# =============================================================================# UKESM model
DS_y_epic_proj_ukesm_585_am, DS_feature_proj_ukesm_585_am, df_feature_proj_ukesm_585_am = projections_generation_hybrid(model = 'ukesm', rcp_scenario = 'ssp585', region = "_am", hybrid_model_full = model_to_be_used, start_date='01-01-2016', end_date='31-12-2100', co2_scen = co2_scen)
DS_y_epic_proj_ukesm_126_am, DS_feature_proj_ukesm_126_am, df_feature_proj_ukesm_126_am = projections_generation_hybrid(model = 'ukesm', rcp_scenario = 'ssp126', region = "_am", hybrid_model_full = model_to_be_used, start_date='01-01-2016', end_date='31-12-2100', co2_scen = co2_scen)

# GFDL model
DS_y_epic_proj_gfdl_585_am, DS_feature_proj_gfdl_585_am, df_feature_proj_gfdl_585_am = projections_generation_hybrid(model = 'gfdl', rcp_scenario = 'ssp585', region = "_am", hybrid_model_full = model_to_be_used, start_date='01-01-2016', end_date='31-12-2100', co2_scen = co2_scen)
DS_y_epic_proj_gfdl_126_am, DS_feature_proj_gfdl_126_am, df_feature_proj_gfdl_126_am = projections_generation_hybrid(model = 'gfdl', rcp_scenario = 'ssp126', region = "_am", hybrid_model_full = model_to_be_used, start_date='01-01-2016', end_date='31-12-2100', co2_scen = co2_scen)

# IPSL model
DS_y_epic_proj_ipsl_585_am, DS_feature_proj_ipsl_585_am, df_feature_proj_ipsl_585_am = projections_generation_hybrid(model = 'ipsl', rcp_scenario = 'ssp585', region = "_am", hybrid_model_full = model_to_be_used, start_date='01-01-2016', end_date='31-12-2100', co2_scen = co2_scen)
DS_y_epic_proj_ipsl_126_am, DS_feature_proj_ipsl_126_am, df_feature_proj_ipsl_126_am = projections_generation_hybrid(model = 'ipsl', rcp_scenario = 'ssp126', region = "_am", hybrid_model_full = model_to_be_used, start_date='01-01-2016', end_date='31-12-2100', co2_scen = co2_scen)

# =============================================================================
# Second function - hybrid prediction 
# =============================================================================
# UKESM model
df_input_hybrid_fut_ukesm_585_am, df_prediction_proj_ukesm_585_am = hybrid_predictions_future(DS_y_epic_proj_ukesm_585_am, df_feature_proj_ukesm_585_am, model = 'ukesm', rcp_scenario = 'ssp585', region = "_am", hybrid_model_full = full_model_hyb_am2, co2_scen = co2_scen)
df_input_hybrid_fut_ukesm_126_am, df_prediction_proj_ukesm_126_am = hybrid_predictions_future(DS_y_epic_proj_ukesm_126_am, df_feature_proj_ukesm_126_am, model = 'ukesm', rcp_scenario = 'ssp126', region = "_am", hybrid_model_full = full_model_hyb_am2, co2_scen = co2_scen)

# GFDL model
df_input_hybrid_fut_gfdl_585_am, df_prediction_proj_gfdl_585_am = hybrid_predictions_future(DS_y_epic_proj_gfdl_585_am, df_feature_proj_gfdl_585_am, model = 'gfdl', rcp_scenario = 'ssp585', region = "_am", hybrid_model_full = full_model_hyb_am2, co2_scen = co2_scen)
df_input_hybrid_fut_gfdl_126_am, df_prediction_proj_gfdl_126_am = hybrid_predictions_future(DS_y_epic_proj_gfdl_126_am, df_feature_proj_gfdl_126_am, model = 'gfdl', rcp_scenario = 'ssp126', region = "_am", hybrid_model_full = full_model_hyb_am2, co2_scen = co2_scen)

# IPSL model
df_input_hybrid_fut_ipsl_585_am, df_prediction_proj_ipsl_585_am = hybrid_predictions_future(DS_y_epic_proj_ipsl_585_am, df_feature_proj_ipsl_585_am, model = 'ipsl', rcp_scenario = 'ssp585', region = "_am", hybrid_model_full = full_model_hyb_am2, co2_scen = co2_scen)
df_input_hybrid_fut_ipsl_126_am, df_prediction_proj_ipsl_126_am = hybrid_predictions_future(DS_y_epic_proj_ipsl_126_am, df_feature_proj_ipsl_126_am, model = 'ipsl', rcp_scenario = 'ssp126', region = "_am", hybrid_model_full = full_model_hyb_am2, co2_scen = co2_scen)

