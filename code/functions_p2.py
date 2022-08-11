# -*- coding: utf-8 -*-
"""
Functions for paper 2 - hybrid model

Created on Mon Jul 25 15:23:23 2022

@author: morenodu
"""

import xarray as xr 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import cartopy
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import seaborn as sns
import matplotlib as mpl

# LOAD FUNCTIONS
def weighted_prod_conversion(DS, DS_area, name_ds = 'Yield', mode = 'production'):
    if mode == 'yield':
        dividing_factor = DS_area['harvest_area'].where(DS > -10).sum(['lat','lon'])
    elif mode == 'production':
        dividing_factor = 10**6
    
    if type(DS) == xr.core.dataarray.DataArray:
        DS_weighted = ((DS * DS_area['harvest_area'].where(DS > -10) ) / dividing_factor ).to_dataset(name = name_ds)
    elif type(DS) == xr.core.dataarray.Dataset:
        DS_weighted = (DS * DS_area['harvest_area'].where(DS > -10) / dividing_factor )
        
    return DS_weighted.sum(['lat','lon'])

def rearrange_latlot(DS, resolution = 0.5):
    DS = DS.sortby('lon')
    DS = DS.sortby('lat')
    new_lat = np.arange(DS.lat.min(), DS.lat.max() + resolution, resolution)
    new_lon = np.arange(DS.lon.min(), DS.lon.max() + resolution, resolution)
    DS = DS.reindex({'lat':new_lat})
    DS = DS.reindex({'lon':new_lon})
    return DS

countries = shpreader.natural_earth(resolution='50m',category='cultural',name='admin_0_countries')
# Find the boundary polygon.
for country in shpreader.Reader(countries).records():
    if country.attributes['SU_A3'] == 'ARG':
        arg_border0 = country.geometry
    elif country.attributes['SU_A3'] == 'BRA':
        bra_border0 = country.geometry
    elif country.attributes['SU_A3'] == 'USA':
        usa_border0 = country.geometry
        
# Plots for spatial distribution of anomalies
def plot_2d_am_map(dataarray_2d, title = None, colormap = None, vmin = None, vmax = None, save_fig = None, label_cbar = None):
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
    if save_fig is not None:
        fig.canvas.draw()
        # plt.tight_layout()
        plt.savefig(f'paper_figures_production/{save_fig}.png', format='png', dpi=300, bbox_inches='tight' )

    plt.show()

# plot multiple figures in a row - counterfactuals
def plot_2d_am_multi(DS, map_proj = None, map_title = None):
    if map_proj is None:
        map_proj = ccrs.LambertAzimuthalEqualArea(central_longitude=-74.5, central_latitude=0)
    else:
        map_proj = map_proj
    
    if len(DS.time) > 1:
        p = DS.plot(transform=ccrs.PlateCarree(), col="time", col_wrap=5, subplot_kws={"projection": map_proj}, 
                    robust=True, cmap = 'RdBu', vmin=-1., vmax=1., zorder=20, aspect=0.9)
        
        for ax in p.axes.flat:
            ax.axis('off')
            ax.add_feature(cartopy.feature.LAND, facecolor='gray', alpha=0.1)
            # ax.add_feature(cartopy.feature.BORDERS, alpha = 0.1)
            ax.add_geometries([usa_border0], ccrs.PlateCarree(),edgecolor='black', facecolor='grey', alpha=0.5, lw=0.7, zorder=0)
            ax.add_geometries([bra_border0], ccrs.PlateCarree(),edgecolor='black', facecolor='grey', alpha=0.5, lw=0.7, zorder=0)
            ax.add_geometries([arg_border0], ccrs.PlateCarree(),edgecolor='black', facecolor='grey', alpha=0.5, lw=0.7, zorder=0)
            # ax.add_geometries([arg_border0], ccrs.PlateCarree(),edgecolor='black', facecolor='None', alpha=0.5, lw=0.7, zorder=21)
            ax.set_extent([-113,-36,-38,45])
            ax.set_aspect('equal', 'box')
            
            if map_title is not None:
                plt.suptitle(map_title)        
        plt.show()
            
    elif len(DS.time) == 1:
        plot_2d_am_map(DS, title = map_title, colormap = 'RdBu', vmin = None, vmax = None)
        plt.show()
    
    elif len(DS.time) == 0:
        print(f'No counterfactuals for this scenario {DS.name}')
        
        
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
                
                res_detrend = np.nansum((da_zero_mean.mean(['lat','lon']) - da_det.mean(['lat','lon']))**2)
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
    return da_det
        
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

def reshape_data(dataframe):  #converts and reshape data
    #If already dataframe, skip the convertsion
    if isinstance(dataframe, pd.Series):    
        dataframe = dataframe.to_frame()
        
    dataframe['month'] = dataframe.index.get_level_values('time').month
    dataframe['year'] = dataframe.index.get_level_values('time').year
    dataframe.set_index('month', append=True, inplace=True)
    dataframe.set_index('year', append=True, inplace=True)
    # dataframe = dataframe.reorder_levels(['time', 'year','month'])
    dataframe.index = dataframe.index.droplevel('time')
    dataframe = dataframe.unstack('month')
    dataframe.columns = dataframe.columns.droplevel()
    return dataframe

def reshape_shift(dataset, shift_time=0):
    ### Convert to dataframe and shift according to input -> if shift time is 0, then nothing is shifted
    dataframe_1 = dataset.shift(time=-shift_time).to_dataframe()    
    
    # Define the column names based on the type of data - dataframe or dataarray
    if type(dataset) == xr.core.dataset.Dataset:
        print('dataset mode')
        column_names = [var_name +"_"+str(j) for var_name in dataset.data_vars 
                        for j in range(1 + shift_time, 13 + shift_time)]
        
    elif type(dataset) == xr.core.dataarray.DataArray: 
        print('dataArray mode') 
        column_names = [dataset.name +"_"+str(j) for j in range(1+shift_time,13+shift_time)]
    else:
        raise ValueError('Data must be either Dataset ot DataArray.')
        
    # Reshape dataframe
    dataframe_reshape = reshape_data(dataframe_1)
    dataframe_reshape.columns = column_names      
    return dataframe_reshape

def weighted_conversion(DS, DS_area, name_ds = 'Yield'):
    if type(DS) == xr.core.dataarray.DataArray:
        DS_weighted = ((DS * DS_area['harvest_area'].where(DS > -10) ) / DS_area['harvest_area'].where(DS > -10).sum(['lat','lon'])).to_dataset(name = name_ds)
    elif type(DS) == xr.core.dataarray.Dataset:
        DS_weighted = ((DS * DS_area['harvest_area'].where(DS > -10) / DS_area['harvest_area'].where(DS > -10).sum(['lat','lon'])))
    return DS_weighted.sum(['lat','lon'])

def dynamic_calendar(DS, df_calendar_month_am):
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


#%%

        
# def convert_clim_weighted_ensemble(df_clim, DS_counterfactuals_weighted_country, feature, DS_area, mode = 'production'):
#     DS_clim = xr.Dataset.from_dataframe(df_clim)
#     DS_clim = rearrange_latlot(DS_clim)
#     # Countefactuals only
#     DS_clim_counter = DS_clim.where(DS_counterfactuals_weighted_country[feature].time.where(DS_counterfactuals_weighted_country[feature] > -10).dropna(dim = 'time'))
   
#     DS_clim_counter_weight_country = weighted_prod_conversion(DS_clim_counter, DS_area = DS_area, mode = mode)
#     df_clim_counter_weight_country = DS_clim_counter_weight_country.to_dataframe()
#     df_clim_counter_weight_country['scenario'] = 'Analogues'
#     df_clim_counter_weight_country['model_used'] = feature
    
#     return df_clim_counter_weight_country

# def country_scale_conversion(country, DS_harvest_area_sim, DS_counterfactuals_spatial, mode='production'):
#     # Generate the calendars for both the historical and future periods, then the 2012 analogues in each country and the historical values per country
#     # Determine country level yields for historical period
#     DS_historical_hybrid_country = xr.load_dataset(f"output_models_am/hybrid_epic_{country}-iiasa_gswp3-w5e5_obsclim_2015soc_default_yield-soy-noirr_global_annual_1901_2016.nc")
#     DS_mirca_country_hist = DS_harvest_area_sim.where(DS_historical_hybrid_country['Yield'] > -10)
    
#     # Determine country level yields for future projections
#     DS_counterfactual_country = DS_counterfactuals_spatial.where(DS_historical_hybrid_country['Yield'].sel(time = 2012) > -10)
#     DS_mirca_country = DS_harvest_area_fut.where(DS_historical_hybrid_country['Yield'].sel(time = 2012) > -10)
              
#     # Weighted analysis historical
#     DS_historical_hybrid_country_weight = weighted_prod_conversion(DS_historical_hybrid_country, DS_area = DS_mirca_country_hist, mode=mode)
    
#     return DS_mirca_country_hist, DS_mirca_country, DS_counterfactual_country, DS_historical_hybrid_country_weight


# # Plot production - TEST
# def production(DS, DS_area):
#     if type(DS) == xr.core.dataarray.DataArray:
#         DS_weighted = ((DS * DS_area['harvest_area'] ) ).to_dataset(name = 'Yield')
#     elif type(DS) == xr.core.dataarray.Dataset:
#         DS_weighted = ((DS * DS_area['harvest_area'] ) )
#     return DS_weighted.sum(['lat','lon'])

# def counterfactuals_per_scenario(DS):
#     list_ds_counterfactuals = []
#     for feature in list(DS.keys()):
#         counterfactuals_by_rcp = DS[feature].sel(time = DS_counterfactuals_weighted_am[feature].time.where(DS_counterfactuals_weighted_am[feature] > -10).dropna(dim = 'time'))
#         plot_2d_am_multi(counterfactuals_by_rcp, map_title = feature )
#         list_ds_counterfactuals.append(counterfactuals_by_rcp)
#     combined = xr.concat(list_ds_counterfactuals, dim='time')
#     ds_combined = combined.to_dataset(name='Yield (ton/ha)')
#     return ds_combined



# def figure_timeseries_per_country(DS_timeseries):
#     DS_hybrid_trend_us_weighted = weighted_prod_conversion(DS_timeseries, DS_area = DS_mirca_us)
#     DS_hybrid_trend_br_weighted = weighted_prod_conversion(DS_timeseries, DS_area = DS_mirca_br)
#     DS_hybrid_trend_arg_weighted = weighted_prod_conversion(DS_timeseries, DS_area = DS_mirca_arg)
    
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 8), sharex=True, sharey=True) # marker='o'marker='^',marker='s',
#     ax1.axhline(y = DS_historical_hybrid_us_weight['Yield'].sel(time = 2012).values, linestyle = 'dashed', color = 'k', label = '2012 event', linewidth = 2 )
#     DS_hybrid_trend_us_weighted['GFDL-esm4_1-2.6'].plot(color='tab:blue',marker='o', ax = ax1, linewidth = 2 )
#     DS_hybrid_trend_us_weighted['GFDL-esm4_5-8.5'].plot( color='tab:orange',marker='o', ax = ax1, linewidth = 2 )
#     DS_hybrid_trend_us_weighted['IPSL-cm6a-lr_1-2.6'].plot( color='tab:blue', marker='^',ax = ax1, linewidth = 2 )
#     DS_hybrid_trend_us_weighted['IPSL-cm6a-lr_5-8.5'].plot( marker='^', color='tab:orange', ax = ax1, linewidth = 2 )
#     DS_hybrid_trend_us_weighted['UKESM1-0-ll_1-2.6'].plot( color='tab:blue',marker='s', ax = ax1, linewidth = 2 )
#     DS_hybrid_trend_us_weighted['UKESM1-0-ll_5-8.5'].plot( marker='s', color='tab:orange', ax = ax1, linewidth = 2 )
#     ax1.set_title("a) US")
#     ax1.set_ylabel('Production (Mt)')
    
#     # ax1.legend()
#     ax2.axhline(y = DS_historical_hybrid_br_weight['Yield'].sel(time = 2012).values, linestyle = 'dashed', color = 'k', label = '2012 event', linewidth = 2 )
#     DS_hybrid_trend_br_weighted['GFDL-esm4_1-2.6'].plot(color='tab:blue',marker='o', ax = ax2, linewidth = 2 )
#     DS_hybrid_trend_br_weighted['GFDL-esm4_5-8.5'].plot( color='tab:orange',marker='o', ax = ax2, linewidth = 2 )
#     DS_hybrid_trend_br_weighted['IPSL-cm6a-lr_1-2.6'].plot( color='tab:blue', marker='^',ax = ax2, linewidth = 2 )
#     DS_hybrid_trend_br_weighted['IPSL-cm6a-lr_5-8.5'].plot( marker='^', color='tab:orange', ax = ax2, linewidth = 2 )
#     DS_hybrid_trend_br_weighted['UKESM1-0-ll_1-2.6'].plot( color='tab:blue',marker='s', ax = ax2, linewidth = 2 )
#     DS_hybrid_trend_br_weighted['UKESM1-0-ll_5-8.5'].plot( marker='s', color='tab:orange', ax = ax2, linewidth = 2 )
#     ax2.set_title("b) Brazil")
    
#     lines = ax2.get_lines()
#     legend1 = ax2.legend([dummy_lines[i] for i in range(0,6)], ["2012 event", "SSP1-2.6", "SSP5-8.5","GFDL-esm4", "IPSL-cm6a-lr", "UKESM1-0-ll"], frameon = False,loc = 3, ncol=2)
#     # legend2 = ax2.legend([dummy_lines2[i] for i in [0,1,2]], ["GFDL-esm4", "IPSL-cm6a-lr", "UKESM1-0-ll"], frameon = False,loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
#     ax2.set_ylabel('')
    
#     ax3.axhline(y = DS_historical_hybrid_arg_weight['Yield'].sel(time = 2012).values, linestyle = 'dashed', color = 'k', label = '2012 event', linewidth = 2 )
#     DS_hybrid_trend_arg_weighted['GFDL-esm4_1-2.6'].plot(color='tab:blue',marker='o', ax = ax3, linewidth = 2 )
#     DS_hybrid_trend_arg_weighted['GFDL-esm4_5-8.5'].plot( color='tab:orange',marker='o', ax = ax3, linewidth = 2 )
#     DS_hybrid_trend_arg_weighted['IPSL-cm6a-lr_1-2.6'].plot( color='tab:blue', marker='^',ax = ax3, linewidth = 2 )
#     DS_hybrid_trend_arg_weighted['IPSL-cm6a-lr_5-8.5'].plot( marker='^', color='tab:orange', ax = ax3, linewidth = 2 )
#     DS_hybrid_trend_arg_weighted['UKESM1-0-ll_1-2.6'].plot( color='tab:blue',marker='s', ax = ax3, linewidth = 2 )
#     DS_hybrid_trend_arg_weighted['UKESM1-0-ll_5-8.5'].plot( marker='s', color='tab:orange', ax = ax3, linewidth = 2 )
#     ax3.set_title("a) Argentina")
#     ax3.set_ylabel('')
#     plt.tight_layout()
#     plt.show()
    
    
#  # Local counterfactuals
# def counterfactual_generation(DS_yields, DS_mirca_country, local_factual):
#     # Define the extend of the future timeseries based on the 2012 year
#     DS_hybrid_country = DS_yields.where(DS_mirca_country['harvest_area'] > 0)
#     # Make conversion to weighted timeseries
#     DS_projections_weighted_country = weighted_prod_conversion(DS_hybrid_country, DS_area = DS_mirca_country)
#     # Isolate the years with counterfactuals and remove the ones without
#     DS_projections_weighted_country_counterfactual = DS_projections_weighted_country.where(DS_projections_weighted_country <= local_factual).dropna('time', how = 'all')
#     return DS_projections_weighted_country_counterfactual


# def counterfactuals_country_level(DS_projections_weighted_country_counterfactual, local_factual):
    
#     print("Number of impact analogues per scenario for the country:", (DS_projections_weighted_country_counterfactual > -10).sum())
    
#     # Check years of counterfactuals
#     list_counterfactuals_scenarios = []
#     for feature in list(DS_projections_weighted_country_counterfactual.keys()):
#         feature_counterfactuals = DS_projections_weighted_country_counterfactual[feature].dropna(dim = 'time')
#         print(feature_counterfactuals.time.values)
#         list_counterfactuals_scenarios.append(feature_counterfactuals.time.values)
#         number_counter = len(np.hstack(list_counterfactuals_scenarios)) 
#     print('total counterfactuals: ', number_counter)
        
#     return DS_projections_weighted_country_counterfactual, number_counter

    

# def co_occurrences_analogues(feature, ax, title = ''):
#     # Create figure with co occurrence of analogues for each region and RCP.

#     domain = years_countefactuals_merged.where( (years_countefactuals_merged['variable'] == feature)).dropna()
#     domain = domain.sort_values(by = 'time')
#     domain = domain.pivot( columns='region', values = 'value')
    
#     def domain_country_locator(country):
#         if country in domain.columns:
#             domain_country = domain.where(domain[country] > 0, 0)[country]
#         else:
#             domain_country = 0
#             print(f'no {country} for this RCP:', feature)         
#         return domain_country
    
#     # Establish the local domains
#     domain_us = domain_country_locator('US')
#     domain_br = domain_country_locator('Brazil')
#     domain_arg = domain_country_locator('Argentina')
#     domain_am = domain_country_locator('AM')

#     # Figure structure
#     ax = ax
#     ax.bar(domain.index, domain_us, label = 'US') #, bottom = domain.where(domain['region'] == 'AM')['value'])
#     ax.bar(domain.index, domain_br, bottom = domain_us, label = 'Brazil')
#     ax.bar(domain.index, domain_arg, bottom = domain_us, label = 'Argentina')
#     ax.bar(domain.index, domain_am, bottom = domain_us + domain_arg + domain_br, label = 'AM', hatch="//" ) 
#     ax.set_ylim(0,3)
#     ax.set_title(f"{title} {feature}", loc='left' )

#     return ax


# def clim_conditions_analogues_anom(DS_area_hist, DS_area_fut, DS_counterfactuals_weighted_country, country, option = '2012', plot_legend = False, plot_yaxis = False):
#     # Conversion of historical series to weighted timeseries    
#     DS_conditions_hist_weighted_country = weighted_prod_conversion(DS_conditions_hist, DS_area = DS_area_hist, mode = 'yield')
#     # DS to df
#     df_clim_hist_weighted = DS_conditions_hist_weighted_country.sel(time = slice(1972,2015)).to_dataframe()
#     df_clim_hist_weighted['scenario'] = 'Climatology'
#     df_clim_hist_weighted['model_used'] = 'Climatology'
    
#     # Conversion of future series to weighted timeseries    
#     df_clim_counter_ukesm_85 = convert_clim_weighted_ensemble(df_hybrid_fut_ukesm_585_am, DS_counterfactuals_weighted_country, 'UKESM1-0-ll_5-8.5', DS_area_fut, mode = 'yield')    
#     df_clim_counter_ukesm_26 = convert_clim_weighted_ensemble(df_hybrid_fut_ukesm_126_am, DS_counterfactuals_weighted_country, 'UKESM1-0-ll_1-2.6', DS_area_fut, mode = 'yield')    
#     df_clim_counter_gfdl_85 = convert_clim_weighted_ensemble(df_hybrid_fut_gfdl_585_am, DS_counterfactuals_weighted_country, 'GFDL-esm4_5-8.5', DS_area_fut, mode = 'yield')    
#     df_clim_counter_gfdl_26 = convert_clim_weighted_ensemble(df_hybrid_fut_gfdl_126_am, DS_counterfactuals_weighted_country, 'GFDL-esm4_1-2.6', DS_area_fut, mode = 'yield')    
#     df_clim_counter_ipsl_85 = convert_clim_weighted_ensemble(df_hybrid_fut_ipsl_585_am, DS_counterfactuals_weighted_country, 'IPSL-cm6a-lr_5-8.5', DS_area_fut, mode = 'yield')    
#     df_clim_counter_ipsl_26 = convert_clim_weighted_ensemble(df_hybrid_fut_ipsl_126_am, DS_counterfactuals_weighted_country, 'IPSL-cm6a-lr_1-2.6', DS_area_fut, mode = 'yield')    
    
#     # Merge dataframes with different names
#     df_clim_counterfactuals_weighted_all_country = pd.concat([df_clim_hist_weighted, df_clim_counter_ukesm_85, df_clim_counter_ukesm_26, 
#                                                       df_clim_counter_gfdl_85, df_clim_counter_gfdl_26,
#                                                       df_clim_counter_ipsl_85, df_clim_counter_ipsl_26])
    
#     df_clim_counterfactuals_weighted_all_country['level'] = 'Country-level analogues' 
    
    
#     df_clim_counter_ukesm_85_am = convert_clim_weighted_ensemble(df_hybrid_fut_ukesm_585_am, DS_counterfactuals_weighted_am, 'UKESM1-0-ll_5-8.5', DS_area_fut, mode = 'yield')    
#     df_clim_counter_ukesm_26_am = convert_clim_weighted_ensemble(df_hybrid_fut_ukesm_126_am, DS_counterfactuals_weighted_am, 'UKESM1-0-ll_1-2.6', DS_area_fut, mode = 'yield')    
#     df_clim_counter_gfdl_85_am = convert_clim_weighted_ensemble(df_hybrid_fut_gfdl_585_am, DS_counterfactuals_weighted_am, 'GFDL-esm4_5-8.5', DS_area_fut, mode = 'yield')    
#     df_clim_counter_gfdl_26_am = convert_clim_weighted_ensemble(df_hybrid_fut_gfdl_126_am, DS_counterfactuals_weighted_am, 'GFDL-esm4_1-2.6', DS_area_fut, mode = 'yield')    
#     df_clim_counter_ipsl_85_am = convert_clim_weighted_ensemble(df_hybrid_fut_ipsl_585_am, DS_counterfactuals_weighted_am, 'IPSL-cm6a-lr_5-8.5', DS_area_fut, mode = 'yield')    
#     df_clim_counter_ipsl_26_am = convert_clim_weighted_ensemble(df_hybrid_fut_ipsl_126_am, DS_counterfactuals_weighted_am, 'IPSL-cm6a-lr_1-2.6', DS_area_fut, mode = 'yield')    
    
#     # Merge dataframes with different names
#     df_clim_counterfactuals_weighted_all_am = pd.concat([df_clim_counter_ukesm_85_am, df_clim_counter_ukesm_26_am, 
#                                                       df_clim_counter_gfdl_85_am, df_clim_counter_gfdl_26_am,
#                                                       df_clim_counter_ipsl_85_am, df_clim_counter_ipsl_26_am])
    
#     df_clim_counterfactuals_weighted_all_am['level'] = '2012 analogues' 

    
#     df_clim_analogues_levels = pd.concat([df_clim_counterfactuals_weighted_all_country, df_clim_counterfactuals_weighted_all_am])
#     df_clim_analogues_levels['country'] = country
#     return df_clim_analogues_levels
    
    
    















