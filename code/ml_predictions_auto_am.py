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
import matplotlib as mpl

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['figure.dpi'] = 144
mpl.rcParams.update({'font.size': 14})

#%% Functions 
# Plot 2D projections of the americas:
def plot_2d_am_map(dataarray_2d, title = None, colormap = None, vmin = None, vmax = None):
    # Plot 2D map of DataArray, remember to average along time or select one temporal interval
    plt.figure(figsize=(12,10)) #plot clusters
    ax=plt.axes(projection=ccrs.LambertAzimuthalEqualArea(central_longitude=-74.5, central_latitude=0))
    ax.add_feature(cartopy.feature.LAND, facecolor='gray', alpha=0.1)
    ax.add_geometries([usa_border0], ccrs.PlateCarree(),edgecolor='black', facecolor='grey', alpha=0.5, lw=0.7, zorder=0)
    ax.add_geometries([bra_border0], ccrs.PlateCarree(),edgecolor='black', facecolor='grey', alpha=0.5, lw=0.7, zorder=0)
    ax.add_geometries([arg_border0], ccrs.PlateCarree(),edgecolor='black', facecolor='grey', alpha=0.5, lw=0.0, zorder=0)
    if colormap is None:
        dataarray_2d.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True, vmin = vmin, vmax = vmax, zorder=20)
    elif colormap is not None:
        dataarray_2d.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True, cmap= colormap, vmin = vmin, vmax = vmax, zorder=20)
    ax.add_geometries([arg_border0], ccrs.PlateCarree(),edgecolor='black', facecolor='None', alpha=0.5, lw=0.7, zorder=21)
    ax.set_extent([-115,-34,-41,44])
    if title is not None:
        plt.title(title)
    plt.show()
# Detrend Dataset
def detrend_dataset(DS, deg = 'free', dim = 'time', print_res = True, mean_data = None):
    # Is the polynomial degree for detrending fixed or should it be automatically defined?        
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
def detrend_dim_2(da, dim, deg = 'free', print_res = True):        
    # Is the polynomial degree for detrending fixed or should it be automatically defined?        
    if deg == 'free':      
        da_zero_mean = da.where( da < np.nanmin(da.values), other = 0 )

        dict_res = {}
        for degree in [1,2]:
            # detrend along a single dimension
            p = da.polyfit(dim=dim, deg=degree)
            fit = xr.polyval(da[dim], p.polyfit_coefficients) 
            da_det = da - fit
            res_detrend = np.nansum((da_zero_mean.mean(['lat','lon']) - da_det.mean(['lat','lon']))**2)
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
    return da_det, fit

# Convert datetime values (ns) into days
def timedelta_to_int(DS, var):
    da_timedelta = DS[var].dt.days
    da_timedelta = da_timedelta.rename(var)
    da_timedelta.attrs["units"] = 'days'    
    return da_timedelta

 # Convert time unit
def convert_timeunit(DS):
    units, reference_date = DS.time.attrs['units'].split('since')
    DS['time'] = pd.date_range(start=' 2015-01-01, 00:00:00', periods=DS.sizes['time'], freq='YS')
    DS['time'] = DS['time'].dt.year
    return DS

def rearrange_latlot(DS, resolution = 0.5):
    DS = DS.sortby('lat')
    DS = DS.sortby('lon')
    new_lat = np.arange(DS.lat.min(), DS.lat.max() + resolution, resolution)
    new_lon = np.arange(DS.lon.min(), DS.lon.max() + resolution, resolution)
    DS = DS.reindex({'lat':new_lat})
    DS = DS.reindex({'lon':new_lon})
    return DS

def weighted_conversion(DS, DS_area, name_ds = 'Yield'):
    if type(DS) == xr.core.dataarray.DataArray:
        DS_weighted = ((DS * DS_area['harvest_area'].where(DS > -10) ) / DS_area['harvest_area'].where(DS > -10).sum(['lat','lon'])).to_dataset(name = name_ds)
    elif type(DS) == xr.core.dataarray.Dataset:
        DS_weighted = ((DS * DS_area['harvest_area'].where(DS > -10) / DS_area['harvest_area'].where(DS > -10).sum(['lat','lon'])))
    return DS_weighted.sum(['lat','lon'])

def dynamic_calendar(DS):
    # First reshape each year to make a 24 month calendar
    df_clim_shift = reshape_shift(DS)
    df_clim_shift_12 = reshape_shift(DS, shift_time = 12)
    # Combine both dataframes and constraint it to be below 2016 just in case
    df_test_reshape_twoyears = df_clim_shift.dropna().join(df_clim_shift_12)
    ### Join and change name to S for the shift values
    df_feature_reshape_shift = df_test_reshape_twoyears.dropna().join(df_calendar_month_am)
    
    list_df_feature_reshape_shift = []
    # Divide the dataset by climatic feature so the shifting does not mix the different variables together
    for feature in list(DS.keys()):
    
        df_feature_reshape_shift_var = pd.concat([df_feature_reshape_shift.loc[:,'plant'], df_feature_reshape_shift.filter(like=feature)], axis = 1)
        
        # Shift accoording to month indicator (hence +1) - SLOW   
        list_shifted_variables = [df_feature_reshape_shift_var.shift(-(int(indicator))+1, axis = 1).where(indicator == df_feature_reshape_shift_var['plant']).dropna( how = 'all') 
                                  for indicator in np.unique(df_feature_reshape_shift_var['plant'])]
        
        df_feature_reshape_shift_var = pd.concat(list_shifted_variables).sort_index().drop(columns=['plant'])
        list_df_feature_reshape_shift.append(df_feature_reshape_shift_var)

    return  pd.concat(list_df_feature_reshape_shift, axis=1)


#%% FUNCTION TO GENERATE PROJECTIONS BASED ON RANDOM FOREST
def projections_generation_hybrid(model, rcp_scenario, region, hybrid_model_full, start_date, end_date, co2_scen='both', three_models = False, sim_round = '\Gen_Assem'):
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
    list_features_am = list_features # From historical time
    DS_clim_ext_projections = DS_clim_ext_projections[list_features_am]

    DS_clim_ext_projections = DS_clim_ext_projections.where(DS_y_obs_am_det_regul.mean('time') >= -5.0)
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
        
    DS_clim_ext_projections_am = DS_clim_ext_projections_combined.where( DS_chosen_calendar_am >= 0 )
    
    # =============================================================================
    # CONVERT CLIMATIC VARIABLES ACCORDING TO THE SOYBEAN GROWING SEASON PER GRIDCELL 
    # =============================================================================
    df_clim_proj_twoyears = dynamic_calendar(DS_clim_ext_projections_am)
        
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
    DS_feature_proj_6mon_am_det = detrend_dataset(DS_feature_proj_6mon_am, deg = 'free', mean_data = DS_feature_season_6mon_am_det.sel(time = slice(2005,2015))) 

    df_feature_proj_6mon_am_det = DS_feature_proj_6mon_am_det.to_dataframe().dropna()
    df_feature_proj_6mon_am_det = df_feature_proj_6mon_am_det.rename_axis(list(DS_feature_proj_6mon_am_det.coords)).reorder_levels(['time','lat','lon']).sort_index()
    
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
        if region == '_am' or 'am': #Write down a way of selecting each file and merging them toegether with the right windows 
        
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
            
            DS_y_epic_proj_am = DS_y_epic_proj_am.where(DS_y_epic_am_det_regul.mean('time') >= -5.0 )
            
        else:
            DS_y_epic_proj = xr.open_dataset("epic-iiasa_"+ model_full +"_w5e5_"+rcp_scenario+"_2015soc_"+co2_scenario+"_yield-soy-noirr_global_annual_2015_2100.nc", decode_times=False)
            DS_y_epic_proj_am = DS_y_epic_proj.where(DS_y_epic_am_det_regul.mean('time') >= -5.0 )
                    
        return DS_y_epic_proj_am.sel(time = slice(2016,2100)), df_feature_proj_6mon
# =============================================================================    

    if (co2_scen == 'default') or (co2_scen == '2015co2'):
        DS_y_epic_proj_am, df_feature_proj_6mon2 = epic_projections_function_co2(co2_scenario = co2_scen)
        
    else:
        raise ValueError('Only forms of accepted co2_scen: <default>, <2015co2> or <both>') 
    
    return DS_y_epic_proj_am, df_feature_proj_6mon2

        
#%% HYBRID model projections
def hybrid_predictions_future(DS_y_epic_proj_am, df_feature_proj_6mon2, model, rcp_scenario, hybrid_model_full, region = "_am", co2_scen = 'Default'): #- DS_fit.sel(time = 2016)
    end_year = 2100
    
    if model == 'ukesm':
        model_full = 'ukesm1-0-ll'
    elif model == 'gfdl':
        model_full = 'gfdl-esm4'
    elif model == 'ipsl':
        model_full = 'ipsl-cm6a-lr'
        
    # load EPIC - Mean values depend on what is porjected: future mean or historical mean
    DS_detrended, DS_fit = detrend_dim_2(DS_y_epic_proj_am['yield-soy-noirr'], 'time')
    DS_fit_mean = xr.DataArray( DS_fit + DS_detrended.mean(['time']), name= DS_y_epic_proj_am['yield-soy-noirr'].name, attrs = DS_y_epic_proj_am['yield-soy-noirr'].attrs)

    DS_y_epic_proj_am_det = xr.DataArray( DS_detrended + DS_y_epic_am_det.sel(time = slice(2005, 2015)).mean('time'), name= DS_y_epic_proj_am['yield-soy-noirr'].name, attrs = DS_y_epic_proj_am['yield-soy-noirr'].attrs)

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
    df_predic_epic_test = df_y_epic_proj_am.query(f"time>=2016 and time <= {end_year}").copy()
    df_predic_epic_test.loc[:,'yield-soy-noirr'] = full_model_epic_am.predict(df_y_epic_proj_am.query(f"time>=2016 and time <= {end_year}"))
    
    DS_pred_epic_proj = xr.Dataset.from_dataframe(df_predic_epic_test)
    DS_pred_epic_proj = rearrange_latlot(DS_pred_epic_proj)

    # CLIMATIC model
    df_pred_clim_proj_test = df_y_epic_proj_am.query(f"time>=2016 and time <= {end_year}").copy()
    df_pred_clim_proj_test.loc[:,'yield-soy-noirr'] = full_model_exclim_dyn_am.predict(df_feature_proj_6mon2.query(f"time>=2016 and time <= {end_year}"))
    
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
    plot_2d_am_map(DS_hybrid_proj_2['yield-soy-noirr'].mean('time'))

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
DS_y_epic_proj_ukesm_585_am, df_feature_proj_ukesm_585_am = projections_generation_hybrid(model = 'ukesm', rcp_scenario = 'ssp585', region = "_am", hybrid_model_full = model_to_be_used, start_date='01-01-2016', end_date='31-12-2100', co2_scen = co2_scen)
DS_y_epic_proj_ukesm_126_am, df_feature_proj_ukesm_126_am = projections_generation_hybrid(model = 'ukesm', rcp_scenario = 'ssp126', region = "_am", hybrid_model_full = model_to_be_used, start_date='01-01-2016', end_date='31-12-2100', co2_scen = co2_scen)

# GFDL model
DS_y_epic_proj_gfdl_585_am, df_feature_proj_gfdl_585_am = projections_generation_hybrid(model = 'gfdl', rcp_scenario = 'ssp585', region = "_am", hybrid_model_full = model_to_be_used, start_date='01-01-2016', end_date='31-12-2100', co2_scen = co2_scen)
DS_y_epic_proj_gfdl_126_am, df_feature_proj_gfdl_126_am = projections_generation_hybrid(model = 'gfdl', rcp_scenario = 'ssp126', region = "_am", hybrid_model_full = model_to_be_used, start_date='01-01-2016', end_date='31-12-2100', co2_scen = co2_scen)

# IPSL model
DS_y_epic_proj_ipsl_585_am, df_feature_proj_ipsl_585_am = projections_generation_hybrid(model = 'ipsl', rcp_scenario = 'ssp585', region = "_am", hybrid_model_full = model_to_be_used, start_date='01-01-2016', end_date='31-12-2100', co2_scen = co2_scen)
DS_y_epic_proj_ipsl_126_am, df_feature_proj_ipsl_126_am = projections_generation_hybrid(model = 'ipsl', rcp_scenario = 'ssp126', region = "_am", hybrid_model_full = model_to_be_used, start_date='01-01-2016', end_date='31-12-2100', co2_scen = co2_scen)

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









# # #%%
# # # UKESM model
# # df_input_hybrid_fut_ukesm_585_am, df_prediction_proj_ukesm_585_am = projections_generation_hybrid(model = 'ukesm', rcp_scenario = 'ssp585', region = "_am", hybrid_model_full = model_to_be_used, start_date='01-01-2015', end_date='31-12-2100', co2_scen = co2_scen)
# # df_input_hybrid_fut_ukesm_126_am, df_prediction_proj_ukesm_126_am = projections_generation_hybrid(model = 'ukesm', rcp_scenario = 'ssp126', region = "_am", hybrid_model_full = model_to_be_used, start_date='01-01-2015', end_date='31-12-2100', co2_scen = co2_scen)

# # # GFDL model
# # df_input_hybrid_fut_gfdl_585_am, df_prediction_proj_gfdl_585_am = projections_generation_hybrid(model = 'gfdl', rcp_scenario = 'ssp585', region = "_am", hybrid_model_full = model_to_be_used, start_date='01-01-2015', end_date='31-12-2100', co2_scen = co2_scen)
# # df_input_hybrid_fut_gfdl_126_am, df_prediction_proj_gfdl_126_am = projections_generation_hybrid(model = 'gfdl', rcp_scenario = 'ssp126', region = "_am", hybrid_model_full = model_to_be_used, start_date='01-01-2015', end_date='31-12-2100', co2_scen = co2_scen)

# # # IPSL model
# # df_input_hybrid_fut_ipsl_585_am, df_prediction_proj_ipsl_585_am = projections_generation_hybrid(model = 'ipsl', rcp_scenario = 'ssp585', region = "_am", hybrid_model_full = model_to_be_used, start_date='01-01-2015', end_date='31-12-2100', co2_scen = co2_scen)
# # df_input_hybrid_fut_ipsl_126_am, df_prediction_proj_ipsl_126_am = projections_generation_hybrid(model = 'ipsl', rcp_scenario = 'ssp126', region = "_am", hybrid_model_full = model_to_be_used, start_date='01-01-2015', end_date='31-12-2100', co2_scen = co2_scen)


# # #%% TEST partial dependence plot for out of calibration zone cases - 2088 should be extreme

# # load future prodictions
# df_predict_fut = df_prediction_proj_ukesm_585_am.iloc[:,[0]].copy() # Use 2015 case
# df_predict_fut = df_predict_fut.rename(columns={'yield-soy-noirr_default':'yield-soy-noirr'})

# df_proj_fut = df_input_hybrid_fut_ukesm_585_am.rename(columns={'yield-soy-noirr_default':'yield-soy-noirr'})
# df_proj_fut =  df_proj_fut[ ['yield-soy-noirr'] + [ col for col in df_proj_fut.columns if col != 'yield-soy-noirr' ] ]
# # df_proj_fut = df_proj_fut.drop(columns='yield-soy-noirr_default')

# df_hybrid_am_test = df_input_hybrid_am.copy()
# df_hybrid_am_test = df_hybrid_am_test.rename(columns={'yield':'yield-soy-noirr'})

# # EPIC
# plt.plot(df_hybrid_am_test['yield-soy-noirr'].groupby('time').mean(), label = 'History')
# plt.plot(df_proj_fut['yield-soy-noirr'].groupby('time').mean(), label = 'Future')
# plt.axvline(df_proj_fut['yield-soy-noirr'].groupby('time').mean().idxmin(), linestyle = 'dashed')
# plt.title('EPIC predictions')
# plt.legend()
# plt.show()

# # TMX
# plt.plot(df_hybrid_am_test['txm_4'].groupby('time').mean(), label = 'History')
# plt.plot(df_proj_fut['txm_4'].groupby('time').mean(), label = 'Future')
# plt.title('TXM_4 predictions')
# plt.axvline(df_proj_fut['yield-soy-noirr'].groupby('time').mean().idxmin(), linestyle = 'dashed')
# plt.axvline(df_proj_fut['txm_4'].groupby('time').mean().idxmax(), c = 'r', linestyle = 'dashed')
# plt.legend()
# plt.show()

# # HYBRID
# plt.plot(df_predict_hyb_am['Yield'].groupby('time').mean(), label = 'History')
# plt.plot(df_predict_fut['yield-soy-noirr'].groupby('time').mean(), label = 'Future')
# plt.title('Hybrid predictions')
# plt.axvline(df_proj_fut['yield-soy-noirr'].groupby('time').mean().idxmin(), linestyle = 'dashed')
# plt.legend()
# plt.show()

# # Plots for points distribution
# for feature in df_proj_fut.columns:
#     df_clim_extrapolated = df_proj_fut[feature].where(df_proj_fut[feature] > df_hybrid_am_test[feature].max()).dropna()
#     df_y_extrapolated = df_predict_fut['yield-soy-noirr'].where(df_proj_fut[feature] > df_hybrid_am_test[feature].max()).dropna()

#     plt.scatter(df_hybrid_am_test[feature], df_predict_hyb_am['Yield'], color = 'k', label = 'History')    
#     plt.scatter(df_proj_fut[feature], df_predict_fut['yield-soy-noirr'], alpha = 0.8, label = 'Projection')
#     sns.regplot(df_hybrid_am_test[feature], df_predict_hyb_am['Yield'], color = 'k', label = 'History', scatter = False)    
#     sns.regplot(df_proj_fut[feature], df_predict_fut['yield-soy-noirr'], label = 'Projection', scatter = False)    
#     plt.scatter(df_clim_extrapolated, df_y_extrapolated, alpha = 0.8, label = 'Extrapolation')
#     plt.legend(loc="upper right")
#     plt.title(f'Scatterplot of {feature} for GCM-RCPs')
#     plt.ylabel('Yield')
#     if feature in ['txm_3','txm_4','txm_5']:
#         x_label = 'Temperature (°C)'
#     elif feature in ['prcptot_2', 'prcptot_3', 'prcptot_4', 'prcptot_5']:
#         x_label = 'Precipitation (mm/month)'
#     else:
#         x_label = 'Yield (ton/ha)'
          
#     plt.xlabel(x_label)
#     plt.show()

# for feature in df_proj_fut.columns:   
#     sns.kdeplot(df_hybrid_am_test[feature],fill=True, alpha = 0.3, label = 'History')
#     sns.kdeplot(df_proj_fut[feature],fill=True, alpha = 0.3, label = 'Proj')
#     print('hist', np.round(df_hybrid_am_test[feature].mean(), 3), 'fut', np.round(df_proj_fut[feature].mean(),3))
#     plt.legend()
#     plt.show()
    
#     plt.plot(df_hybrid_am_test[feature].groupby('time').mean(), label = 'History')
#     plt.plot(df_proj_fut[feature].groupby('time').mean(), label = 'Future')
#     plt.axvline(df_proj_fut['yield-soy-noirr'].groupby('time').mean().idxmin(), linestyle = 'dashed')
#     plt.axvline(df_proj_fut['yield-soy-noirr'].groupby('time').mean().nsmallest(5).index[1], linestyle = 'dashed')
#     plt.axvline(df_predict_fut['yield-soy-noirr'].groupby('time').mean().idxmin(), color = 'red',linestyle = 'dashed')
#     plt.title(f'{feature} predictions')
#     plt.legend()
#     plt.show()
    
# sns.kdeplot(df_predict_hyb_am['Yield'], fill=True, alpha = 0.3, label = 'History')
# sns.kdeplot(df_predict_fut['yield-soy-noirr'],fill=True, alpha = 0.3, label = 'Proj')
# plt.title('Hybrid predictions')
# plt.legend()
# plt.show()

# plt.plot(df_predict_hyb_am['Yield'].groupby('time').mean(), label = 'History')
# plt.plot(df_predict_fut['yield-soy-noirr'].groupby('time').mean(), label = 'Future')
# plt.title('Hybrid predictions')
# plt.axvline(df_proj_fut['yield-soy-noirr'].groupby('time').mean().idxmin(), linestyle = 'dashed')
# plt.axvline(df_proj_fut['yield-soy-noirr'].groupby('time').mean().nsmallest(5).index[1], linestyle = 'dashed')
# plt.legend()
# plt.show()

# # for feature in df_proj_fut.columns:
# #     df_clim_extrapolated = df_proj_fut[feature].where(df_proj_fut[feature] < df_hybrid_am_test[feature].min()).dropna()
# #     df_y_extrapolated = df_predict_fut['yield-soy-noirr'].where(df_proj_fut[feature] < df_hybrid_am_test[feature].min()).dropna()

# #     plt.scatter(df_hybrid_am_test[feature], df_predict_hyb_am['Yield'], color = 'k')    
# #     plt.scatter(df_proj_fut[feature], df_predict_fut['yield-soy-noirr'], alpha = 0.8)
# #     plt.hlines(df_predict_hyb_am['Yield'].mean(), df_hybrid_am_test[feature].min(), df_hybrid_am_test[feature].max(), color = 'k')
# #     plt.scatter(df_clim_extrapolated, df_y_extrapolated, alpha = 0.8)
# #     # plt.legend(loc="upper right")
# #     plt.title(f'Scatterplot of {feature} for GCM-RCPs')
# #     plt.ylabel('Yield')
# #     if feature in ['tnx_3','tnx_4','tnx_5']:
# #         x_label = 'Temperature (°C)'
# #     elif feature in ['prcptot_3', 'prcptot_4', 'prcptot_5']:
# #         x_label = 'Precipitation (mm/month)'
# #     else:
# #         x_label = 'Yield (ton/ha)'
          
# #     plt.xlabel(x_label)
# #     plt.show()
    

# #%% Partial dependence plots
# from sklearn.inspection import PartialDependenceDisplay

# features_to_plot = ['yield-soy-noirr','txm_3']
# fig, ax1 = plt.subplots(1, 1, figsize=(10, 8), dpi=500)
# disp1 = PartialDependenceDisplay.from_estimator(model_to_be_used, df_proj_fut, features_to_plot, pd_line_kw={'color':'r'},percentiles=(0.01,0.99), ax = ax1)
# disp2 = PartialDependenceDisplay.from_estimator(model_to_be_used, df_hybrid_am_test, features_to_plot, ax = disp1.axes_,percentiles=(0,1), pd_line_kw={'color':'k'})
# plt.setp(disp1.deciles_vlines_, visible=False)
# plt.setp(disp2.deciles_vlines_, visible=False)
# plt.show()

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
# disp1.plot(ax=[ax1, ax2], line_kw={"label": "Extrapolation", "color": "red"})
# disp2.plot(ax=[ax1, ax2], line_kw={"label": "Training", "color": "black"})
# ax1.set_ylim(1.0, 2.8)
# ax2.set_ylim(1.0, 2.8)
# plt.setp(disp1.deciles_vlines_, visible=False)
# plt.setp(disp2.deciles_vlines_, visible=False)
# ax1.legend()
# plt.show()

# features_to_plot = [1,2,3]
# fig, ax1 = plt.subplots(1, 1, figsize=(10, 8), dpi=500)
# disp3 = PartialDependenceDisplay.from_estimator(model_to_be_used, df_proj_fut, features_to_plot, pd_line_kw={'color':'r'},percentiles=(0.01,0.99), ax = ax1)
# disp4 = PartialDependenceDisplay.from_estimator(model_to_be_used, df_hybrid_am_test, features_to_plot, ax = disp3.axes_,percentiles=(0,1), pd_line_kw={'color':'k'})
# plt.ylim(0, 2.6)
# plt.setp(disp3.deciles_vlines_, visible=False)
# plt.setp(disp4.deciles_vlines_, visible=False)

# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 8))
# disp3.plot(ax=[ax1, ax2, ax3], line_kw={"label": "Extrapolation", "color": "red"})
# disp4.plot(
#     ax=[ax1, ax2, ax3], line_kw={"label": "Training", "color": "black"}
# )
# ax1.set_ylim(1, 2.8)
# ax2.set_ylim(1, 2.8)
# ax3.set_ylim(1, 2.8)
# plt.setp(disp3.deciles_vlines_, visible=False)
# plt.setp(disp4.deciles_vlines_, visible=False)
# ax1.legend()
# plt.show()

# features_to_plot = [4,5,6]
# fig, ax1 = plt.subplots(1, 1, figsize=(10, 8), dpi=500)
# disp5 = PartialDependenceDisplay.from_estimator(model_to_be_used, df_proj_fut, features_to_plot, pd_line_kw={'color':'r'},percentiles=(0.01,0.99), ax = ax1, method = 'brute')
# disp6 = PartialDependenceDisplay.from_estimator(model_to_be_used, df_hybrid_am_test, features_to_plot, ax = disp5.axes_,percentiles=(0,1), pd_line_kw={'color':'k'}, method = 'brute')
# plt.ylim(0, 2.6)
# plt.setp(disp5.deciles_vlines_, visible=False)
# plt.setp(disp6.deciles_vlines_, visible=False)

# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 8))
# disp5.plot(ax=[ax1, ax2, ax3], line_kw={"label": "Extrapolation", "color": "red"})
# disp6.plot(
#     ax=[ax1, ax2, ax3], line_kw={"label": "Training", "color": "black"}
# )
# ax1.set_ylim(1, 2.8)
# ax2.set_ylim(1, 2.8)
# ax3.set_ylim(1, 2.8)
# plt.setp(disp5.deciles_vlines_, visible=False)
# plt.setp(disp6.deciles_vlines_, visible=False)
# ax1.legend()
# plt.show()

# #%%
# mean_stuff = df_proj_fut['prcptot_4'].groupby(['lat','lon']).mean().to_xarray()
# stuff_2088 = df_proj_fut['prcptot_4'].loc[2088].groupby(['lat','lon']).mean().to_xarray()
# delta = stuff_2088 - mean_stuff
# delta.plot()
# # test_values = df_hybrid_am_test.iloc[[0],:-1].copy()
# # test_values.iloc[[0],:] = [0,0,0,0,50,50,50]
# # test_prdict = full_model_hyb.predict(test_values)
# # print(test_prdict)

# DS_test = xr.open_mfdataset('monthly_ukesm_ssp585_am/txm_MON_climpact.ukesm1-0-ll_r1i1p1f2_w5e5_ssp5-ssp5.nc').where(DS_y_epic_am_det_regul.mean('time') >= -5.0)

# # txm 2 is peaking one year after txm_4
# DS_test['txm'].mean(['lat','lon']).plot()
# DS_test['txm'].sel(time = slice('2030-01-16','2040-10-16')).mean(['lat','lon']).plot()
# DS_test['txm'].sel(time = slice('2037-01-01','2037-01-30')).mean(['lat','lon']).values  
# DS_test['txm'].sel(time = slice('2036-11-01','2037-03-30')).mean(['lat','lon']).values  

# DS_test['txm'].sel(time = slice('2034-11-01','2035-03-30')).mean(['lat','lon']).values 
# DS_test['txm'].sel(time = slice('2035-11-01','2036-03-30')).mean(['lat','lon']).values # March SHould be equal to TXM_4 peak
# DS_test['txm'].sel(time = slice('2036-11-01','2037-03-30')).mean(['lat','lon']).values # January SHould be equal to TXM_2 peak

# df_test = DS_test['txm'].mean(['lat','lon']).to_dataframe()
# df_test

# from statsmodels.tsa.seasonal import seasonal_decompose
# decompose_data = seasonal_decompose(df_test, model="additive", period = 516)
# decompose_data.plot(); 

# decompose_data.seasonal.plot(); 

# from statsmodels.graphics.tsaplots import plot_acf
# plot_acf(df_test); 

# plt.figure(figsize=(10,6))
# plt.show()

# DS_test['txm'].sel(time = '2100-12-16').mean(['time']).plot()


# DS_test2 = xr.open_mfdataset('epic-iiasa_ukesm1-0-ll_w5e5_ssp585_2015soc_2015co2_yield-soy-noirr_global_annual_2015_2100.nc', decode_times= False)
# DS_test2 = DS_test2.where(DS_y_epic_am_det.mean('time') >= -5.0)


# DS_test2['yield-soy-noirr'].sel(time = 438).plot()
# plot_2d_am_map(DS_test2['yield-soy-noirr'].sel(time = 438))
# #%%














DS_y_epic_proj = xr.open_dataset("epic-iiasa_ukesm1-0-ll_w5e5_ssp585_2015soc_default_yield-soy-noirr_global_annual_2015_2100.nc", decode_times=False)








