# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 09:30:07 2022

@author: morenodu
"""
import xarray as xr 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import cartopy
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

def mask_2012_crops(start_date = 1970, end_date = 2015):
    
    countries = shpreader.natural_earth(resolution='50m',category='cultural',name='admin_0_countries')
    # Find the boundary polygon.
    for country in shpreader.Reader(countries).records():
        if country.attributes['SU_A3'] == 'ARG':
            arg_border0 = country.geometry
        elif country.attributes['SU_A3'] == 'BRA':
            bra_border0 = country.geometry
        elif country.attributes['SU_A3'] == 'USA':
            usa_border0 = country.geometry

    # Function for state mask and mapping
    def states_mask(input_gdp_shp, state_names = None) :
        country = gpd.read_file(input_gdp_shp, crs="epsg:4326") 
        country_shapes = list(shpreader.Reader(input_gdp_shp).geometries())
        if state_names is not None:
            soy_states = country[country['NAME_1'].isin(state_names)]
            states_area = soy_states['geometry'].to_crs({'proj':'cea'}) 
            states_area_sum = (sum(states_area.area / 10**6))
            return soy_states, country_shapes, states_area_sum
        else:
            return country_shapes

    def plot_2d_am_map(dataarray_2d, title = None, colormap = None, vmin = None, vmax = None, label_cbar = None):
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
            dataarray_2d.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True, cmap= colormap, vmin = vmin, vmax = vmax, cbar_kwargs={'label':label_cbar}, zorder=20)
        ax.add_geometries([arg_border0], ccrs.PlateCarree(),edgecolor='black', facecolor='None', alpha=0.5, lw=0.7, zorder=21)
        ax.set_extent([-115,-34,-41,44])
        if title is not None:
            plt.title(title)
        plt.show()
        
    def plot_2d_map(dataarray_2d):
        # Plot 2D map of DataArray, remember to average along time or select one temporal interval
        plt.figure(figsize=(12,5)) #plot clusters
        ax=plt.axes(projection=ccrs.Mercator())
        dataarray_2d.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
        ax.add_geometries(br1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
        ax.set_extent([-80.73,-34,-45,6], ccrs.PlateCarree())
        plt.show()

    def plot_2d_us_map(dataarray_2d):
        # Plot 2D map of DataArray, remember to average along time or select one temporal interval
        plt.figure(figsize=(12,5)) #plot clusters
        ax=plt.axes(projection=ccrs.Mercator())
        dataarray_2d.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
        ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
        ax.set_extent([-125,-67,25,50], ccrs.Geodetic())
        plt.show()

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

    def rearrange_latlot(DS, resolution = 0.5):
        DS = DS.sortby('lat')
        DS = DS.sortby('lon')
        new_lat = np.arange(DS.lat.min(), DS.lat.max() + resolution, resolution)
        new_lon = np.arange(DS.lon.min(), DS.lon.max() + resolution, resolution)
        DS = DS.reindex({'lat':new_lat})
        DS = DS.reindex({'lon':new_lon})
        return DS

    def timedelta_to_int(DS, var):
        da_timedelta = DS[var].dt.days
        da_timedelta = da_timedelta.rename(var)
        da_timedelta.attrs["units"] = 'days'
        
        return da_timedelta

    def weighted_conversion(DS, DS_area, name_ds = 'Yield'):
        if type(DS) == xr.core.dataarray.DataArray:
            DS_weighted = ((DS * DS_area['harvest_area'].where(DS > -10) ) / DS_area['harvest_area'].where(DS > -10).sum(['lat','lon'])).to_dataset(name = name_ds)
        elif type(DS) == xr.core.dataarray.Dataset:
            DS_weighted = ((DS * DS_area['harvest_area'].where(DS > -10) / DS_area['harvest_area'].where(DS > -10).sum(['lat','lon'])))
        return DS_weighted.sum(['lat','lon'])

    #%% LOADING MAIN DATA
    # Load Country shapes
    us1_shapes = states_mask('../../Paper_drought/data/gadm36_USA_1.shp')
    br1_shapes = states_mask('../../Paper_drought/data/gadm36_BRA_1.shp')
    arg_shapes = states_mask('GIS/gadm36_ARG_1.shp')
    
    # =============================================================================
    # USe MIRCA to isolate the rainfed 90% soybeans
    # =============================================================================
    # DS_mirca_test = xr.open_dataset("../../paper_hybrid_agri/data/americas_mask_ha.nc", decode_times=False).rename({'latitude': 'lat', 'longitude': 'lon','annual_area_harvested_rfc_crop08_ha_30mn':'harvest_area'})
    DS_mirca_test = xr.open_dataset("../../paper_hybrid_agri/data/soy_harvest_spam_native_05x05.nc", decode_times=False)
    
    #### HARVEST DATA
    DS_harvest_area_sim = xr.load_dataset("../../paper_hybrid_agri/data/soybean_harvest_area_calculated_americas_hg.nc", decode_times=False)
    DS_harvest_area_sim = DS_harvest_area_sim.sel(time=2012) 
    DS_harvest_area_sim = DS_harvest_area_sim.where(DS_mirca_test['harvest_area'] > 0 )
    DS_harvest_area_sim = rearrange_latlot(DS_harvest_area_sim)
    DS_mirca_test = DS_harvest_area_sim
    plot_2d_am_map(DS_mirca_test['harvest_area'])
    plot_2d_am_map(DS_harvest_area_sim['harvest_area'])
    
    # =============================================================================
    # GLOBAL - EPIC 
    # =============================================================================
    DS_y_epic = xr.open_dataset("epic-iiasa_gswp3-w5e5_obsclim_2015soc_default_yield-soy-noirr_global_annual_1901_2016.nc", decode_times=False)
    # Convert time unit
    units, reference_date = DS_y_epic.time.attrs['units'].split('since')
    DS_y_epic['time'] = pd.date_range(start=reference_date, periods=DS_y_epic.sizes['time'], freq='YS')
    DS_y_epic['time'] = DS_y_epic['time'].dt.year 
    DS_y_epic = DS_y_epic.sel(time=slice('1969-12-12','2016-12-12'))
    DS_y_epic = DS_y_epic.rename({'yield-soy-noirr':'yield'})
    
    DS_y_epic_am = DS_y_epic.where(DS_mirca_test['harvest_area'] > 0 )
    plot_2d_am_map(DS_y_epic_am['yield'].isel(time=0))
    plot_2d_am_map(DS_y_epic_am['yield'].sel(time=2016))
    # remove weird data point on north of brazil that seems to be off-calendar
    DS_epic_2016_south = DS_y_epic_am.sel(time = 2016).where(DS_y_epic_am.sel(time = 2016).lat < 10)
    DS_y_epic_am = xr.where(DS_epic_2016_south['yield'] > -100, np.nan, DS_y_epic_am)
    DS_y_epic_am['yield'] = DS_y_epic_am['yield'].transpose('time','lat','lon')
    DS_y_epic_am = rearrange_latlot(DS_y_epic_am)
    
    # =============================================================================
    # OBSERVED CENSUS DATA
    # =============================================================================
    # start_date, end_date = 1972, 2016
    # # US -------------------------------------------------------------
    DS_y_obs_us_all = xr.open_dataset("../../paper_hybrid_agri/data/soy_yields_US_all_1975_2020_05x05.nc", decode_times=False).sel(lon=slice(-160,-10))
    # Convert time unit --- # units, reference_date = DS_y_obs_us_all.time.attrs['units'].split('since') #pd.date_range(start=reference_date, periods=DS_y_obs_us_all.sizes['time'], freq='YS').year
    DS_y_obs_us_all['time'] = DS_y_obs_us_all['time'].astype(int) 
    DS_y_obs_us_all = DS_y_obs_us_all.sel(time=slice(start_date, end_date))
    
    DS_y_obs_us_all = DS_y_obs_us_all.where(DS_mirca_test['harvest_area'] > 0 )
    DS_y_obs_us_all.to_netcdf("soybean_yields_US_1978_2016.nc")
    plot_2d_am_map(DS_y_obs_us_all['Yield'].mean('time'))
    
    DS_y_epic_us = DS_y_epic_am.where(DS_y_obs_us_all['Yield'].mean('time') > - 5)
    
    # =============================================================================
    # # BRAZIL --------------------------------------------------------
    # =============================================================================
    DS_y_obs_br = xr.open_dataset("../../paper_hybrid_agri/data/soy_yield_1975_2016_05x05_1prc.nc", decode_times=False) 
    DS_y_obs_br=DS_y_obs_br.sel(time = slice(start_date, end_date))
    DS_y_obs_br = DS_y_obs_br.where(DS_mirca_test['harvest_area'] > 0 )
    plot_2d_am_map(DS_y_obs_br['Yield'].mean('time'))
    
    DS_y_obs_br.to_netcdf("soybean_yields_BR_1978_2016.nc")
    
    # SHIFT EPIC FOR BRAZIL ONE YEAR FORWARD TO MATCH INTERNATIONAL CALENDARS
    DS_y_epic_br = DS_y_epic_am.where(DS_y_obs_br['Yield'].mean('time') > - 5)
    DS_y_epic_br = DS_y_epic_br.copy().shift(time = 1) # SHIFT EPIC BR ONE YEAR FORWARD
    # DS_y_epic_br = DS_y_epic_br.where(DS_y_obs_br['Yield'] > - 5)
    # plot_2d_am_map(DS_y_epic_br['yield'].sel(time=1979))
    
    # =============================================================================
    # # AREGNTINA --------------------------------------------------------
    # =============================================================================
    DS_y_obs_arg = xr.open_dataset("../../paper_hybrid_agri/data/soy_yield_arg_1974_2019_05x05.nc", decode_times=False)#soy_yield_1980_2016_1prc05x05 / soy_yield_1980_2016_all_filters05x05
    DS_y_obs_arg=DS_y_obs_arg.sel(time = slice(start_date-1, end_date)) # get one year earlier to shift them forward
    
    # SHIFT OBSERVED DATA FOR ARGENTINA ONE YEAR FORWARD TO MATCH INTERNATIONAL CALENDARS - from planting year to harvest year
    DS_y_obs_arg = DS_y_obs_arg.copy().shift(time = 1) # SHIFT AGRNEITNA ONE YeAR FORWARD
    DS_y_obs_arg = DS_y_obs_arg.where(DS_mirca_test['harvest_area'] > 0 ).sel(time=slice(start_date, end_date))
    DS_y_obs_arg.to_netcdf("soybean_yields_ARG_1978_2016.nc")
    
    # SHIFT EPIC DATA FOR ARGENTINA ONE YEAR FORWARD TO MATCH INTERNATIONAL CALENDARS
    DS_y_epic_arg = DS_y_epic_am.where(DS_y_obs_arg['Yield'].mean('time') > - 5)
    DS_y_epic_arg = DS_y_epic_arg.copy().shift(time = 1) # SHIFT EPIC ARG ONE YeAR FORWARD
    # DS_y_epic_arg = DS_y_epic_arg.where(DS_y_obs_arg['Yield'] > - 5)
    
    # =============================================================================
    # # Plots for analysis
    # =============================================================================
    plt.plot(DS_y_obs_arg['Yield'].time, DS_y_obs_arg['Yield'].mean(['lat','lon']), label = 'ARG')
    plt.plot(DS_y_obs_br.time, DS_y_obs_br['Yield'].mean(['lat','lon']), label = 'BR')
    plt.plot(DS_y_obs_us_all.time, DS_y_obs_us_all['Yield'].mean(['lat','lon']), label = 'US')
    plt.vlines(DS_y_obs_us_all.time, 1,3.5, linestyles ='dashed', colors = 'k')
    plt.legend()
    plt.show()
    
    plt.plot(DS_y_epic_arg.time, DS_y_epic_arg['yield'].mean(['lat','lon']), label = 'ARG')
    plt.plot(DS_y_epic_br.time, DS_y_epic_br['yield'].mean(['lat','lon']), label = 'BR')
    plt.plot(DS_y_epic_us.time, DS_y_epic_us['yield'].mean(['lat','lon']), label = 'US')
    plt.vlines(DS_y_epic_us.time, 2,4.5, linestyles ='dashed', colors = 'k')
    plt.legend()
    plt.show()
    
    # =============================================================================
    # # Combine the datsaets:
    # =============================================================================
    DS_y_obs_am = DS_y_obs_us_all.combine_first(DS_y_obs_arg)
    DS_y_obs_am = DS_y_obs_am.combine_first(DS_y_obs_br)
    DS_y_obs_am = rearrange_latlot(DS_y_obs_am)
    
    plot_2d_am_map(DS_y_obs_am["Yield"].mean('time'))
    
    DS_y_epic_am_2 = DS_y_epic_us.combine_first(DS_y_epic_br)
    DS_y_epic_am_2 = DS_y_epic_am_2.combine_first(DS_y_epic_arg)
    
    plt.plot(DS_y_epic_am.time, DS_y_epic_am['yield'].mean(['lat','lon']), label = "no shift")
    plt.plot(DS_y_epic_am_2.time, DS_y_epic_am_2['yield'].mean(['lat','lon']), label = "shift")
    plt.legend()
    plt.show()
    
    DS_y_epic_am = DS_y_epic_am_2.copy()
    
    plot_2d_am_map(DS_y_epic_am["yield"].mean('time'))
    plot_2d_am_map(DS_y_obs_am["Yield"].mean('time'))
    
    ##### concatenate the two types of data
    DS_y_obs_am_clip = DS_y_obs_am.where(DS_y_epic_am['yield'].mean('time') >= 0.0 )
    DS_y_epic_am_clip = DS_y_epic_am.where(DS_y_obs_am['Yield'].mean('time') >= 0.0 )
    
    # Compare
    df_epic_am = DS_y_obs_am_clip.to_dataframe().dropna()
    df_obs_am = DS_y_obs_am_clip.to_dataframe().dropna()
    
    start_date_det, end_date_det = start_date, end_date
    DS_y_epic_am_det = xr.DataArray( detrend_dim(DS_y_epic_am_clip["yield"], 'time') + DS_y_epic_am_clip["yield"].mean('time'), name= DS_y_epic_am_clip["yield"].name, attrs = DS_y_epic_am_clip["yield"].attrs)
    DS_y_epic_am_det = DS_y_epic_am_det.sel(time = slice(start_date_det, end_date_det))
    
    plt.plot(DS_y_epic_am_det.time, DS_y_epic_am_det.mean(['lat','lon']), label='EPIC')
    plt.legend()
    plt.show()
        
    
    # Compare EPIC with Observed dataset
    df_epic_am_det_2012 = DS_y_epic_am_det.to_dataframe().dropna().sort_index(ascending = [True,False,True])
    
    # # Plot each country detrended and weighted
    # DS_y_obs_det_weighted_am = weighted_conversion(DS_y_obs_am_det.sel(time = slice(1975,2016)), DS_area = DS_mirca_test.where(DS_y_obs_am['Yield'] > -1))
    # DS_y_obs_det_weighted_us = weighted_conversion(DS_y_obs_am_det, DS_area = DS_mirca_test.where(DS_y_obs_us_all['Yield'] > -1))
    # DS_y_obs_det_weighted_br = weighted_conversion(DS_y_obs_am_det, DS_area = DS_mirca_test.where(DS_y_obs_br['Yield'] > -1))
    # DS_y_obs_det_weighted_arg = weighted_conversion(DS_y_obs_am_det, DS_area = DS_mirca_test.where(DS_y_obs_arg['Yield'] > -1))
    
    # # Plot historical timeline of weighted soybean yield
    # plt.plot(DS_y_obs_det_weighted_us.time, DS_y_obs_det_weighted_us['Yield'], label = 'US')
    # plt.plot(DS_y_obs_det_weighted_br.time, DS_y_obs_det_weighted_br['Yield'], label = 'BR')
    # plt.plot(DS_y_obs_det_weighted_arg.time, DS_y_obs_det_weighted_arg['Yield'], label = 'ARG')
    # plt.legend()
    # plt.title('Weighted averages of soybean yields')
    # plt.ylabel('Yield (ton/ha)')
    # plt.tight_layout()
    # plt.show()
    
    
    #%% EXTREME CLIMATE INDICES
     
    def units_conversion(DS_exclim):
        da_list = []
        for feature in list(DS_exclim.keys()):
            if (type(DS_exclim[feature].values[0,0,0]) == np.timedelta64):
                print('Time')
                DS = timedelta_to_int(DS_exclim, feature)
            else:
                print('Integer')
                DS = DS_exclim[feature]
            
            da_list.append(DS)
        return xr.merge(da_list)    
    
    # Start - take the initial year of the observed dataset - one year so we can shift the epic data 12 months forwards
    start_date_clim, end_date_clim = f'01-01-{DS_y_epic_am_det.time[0].values - 1}','31-12-2016'
    
    DS_exclim_us = xr.open_mfdataset('../../paper_hybrid_agri/data/climpact-master/climpact-master/www/output_historical_us/monthly_data/*.nc').sel(time=slice(start_date_clim, end_date_clim))
    DS_exclim_arg = xr.open_mfdataset('../../paper_hybrid_agri/data/climpact-master/climpact-master/www/output_historical_arg/monthly_data/*.nc').sel(time=slice(start_date_clim, end_date_clim))
    DS_exclim_br = xr.open_mfdataset('../../paper_hybrid_agri/data/climpact-master/climpact-master/www/output_gswp3/monthly_data/*.nc').sel(time=slice(start_date_clim, end_date_clim))
    
    #SHIFT 12 months forward
    DS_exclim_br = DS_exclim_br.shift(time = 12)
    DS_exclim_arg = DS_exclim_arg.shift(time = 12)
    
    # COMBINE
    DS_exclim_am = DS_exclim_us.combine_first(DS_exclim_arg)
    DS_exclim_am = DS_exclim_am.combine_first(DS_exclim_br)
    DS_exclim_am = DS_exclim_am.where(DS_y_epic_am_det.mean('time') > -10)
    DS_exclim_am = rearrange_latlot(DS_exclim_am)
    
    plot_2d_am_map(DS_exclim_am['txm'].mean('time'))
    
    # New dataset
    DS_exclim_am = DS_exclim_am.drop_vars(['fd','id','spei','time_bnds','spi','dtr','tr','tnlt2','tmlt10','tnltm2','tnltm20','txn', 'tmlt5', 'tmge5','tx10p', 'tmge10','tn10p', 'tnm', 'tnn', 'tmm']) # Always zero
    # DS_exclim_am['spei'] = DS_exclim_am['spei'].sel(scale = 1)
    list_features = ['prcptot', 'txm'] 
    DS_exclim_am = DS_exclim_am[list_features] 
    df_list_features = list(DS_exclim_am.keys())
    
    # Convert data from time to values (number of days)
    DS_exclim_am_comb = units_conversion(DS_exclim_am)
    
    # Adjust data
    DS_exclim_am_comb = rearrange_latlot(DS_exclim_am_comb)
    DS_exclim_am_comb = DS_exclim_am_comb.reindex(lat=DS_exclim_am_comb.lat[::-1])
    if len(DS_exclim_am_comb.coords) >3 :
        DS_exclim_am_comb=DS_exclim_am_comb.drop('spatial_ref')
        
    DS_exclim_am_det = DS_exclim_am_comb
    
    # =============================================================================
    # Relative dates functions - shift and reshape
    # =============================================================================
    # Reshape to have each calendar year on the columns (1..12)
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
    
    def calendar_multiyear_adjust(month_list, df_entry, mode = "shift_one_year"):
        df = df_entry.copy()
        month_list = np.sort(month_list)[::-1]
        for month in month_list:
            if df.loc[df == month].sum() > 0:
                print(f'There are planting dates on the following year (y+1) for month {month}')
                if mode == "shift_one_year":
                    df.loc[df == month] = 12 + month
                elif mode == 'erase':
                    df.loc[df == month] = np.nan
            else:
                print(f'No planting dates for month {month}')
        return df.to_frame().dropna()
    
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
    
    
    # =============================================================================
    # Load calendars ############################################################
    # =============================================================================
    DS_cal_ggcmi = xr.open_dataset('../../paper_hybrid_agri/data/soy_rf_ggcmi_crop_calendar_phase3_v1.01.nc4') / (365/12)
    DS_cal_sachs = xr.open_dataset('../../paper_hybrid_agri/data/Soybeans.crop.calendar_sachs_05x05.nc') / (365/12) 
    DS_cal_mirca = xr.open_dataset('../../paper_hybrid_agri/data/mirca2000_soy_calendar.nc') # 
    DS_cal_ggcmi_test = xr.open_dataset('../../paper_hybrid_agri/data/soy_rf_ggcmi_crop_calendar_phase3_v1.01.nc4').where(DS_y_epic_am_det.mean('time') >= -10 )
    
    # =============================================================================
    # TEST CALENDARS
    # =============================================================================
    DS_cal_mirca_subset_us = DS_cal_mirca.where(DS_y_obs_us_all['Yield'].mean('time') >= -10)
    DS_cal_ggcmi_subset_us_test = DS_cal_ggcmi.where(DS_y_obs_us_all['Yield'].mean('time') >= -10)
    DS_cal_ggcmi_subset_arg = DS_cal_ggcmi.where(DS_y_obs_arg['Yield'].mean('time') >= -10)
    DS_cal_mirca_subset_arg_test = DS_cal_mirca.where(DS_y_obs_arg['Yield'].mean('time') >= -10)
    DS_cal_ggcmi_subset_br = DS_cal_ggcmi.where(DS_y_obs_br['Yield'].mean('time') >= -10)
    DS_cal_mirca_subset_br_test = DS_cal_mirca.where(DS_y_obs_br['Yield'].mean('time') >= -10)
    
    plot_2d_am_map(DS_cal_mirca_subset_us['start'], title = 'MIRCA start')
    plot_2d_am_map(DS_cal_ggcmi_subset_br['planting_day'], title = 'MIRCA start')
    plot_2d_am_map(DS_cal_ggcmi_subset_arg['planting_day'], title = 'MIRCA start')
    
    DS_calendar_combined = DS_cal_mirca_subset_us['start'].combine_first(DS_cal_ggcmi_subset_br['planting_day'])
    DS_calendar_combined = DS_calendar_combined.combine_first(DS_cal_ggcmi_subset_arg['planting_day'])
    DS_calendar_combined = rearrange_latlot(DS_calendar_combined)
    plot_2d_am_map(DS_calendar_combined, title = 'combined start')
    
    DS_cal_mirca_subset = DS_cal_mirca.where(DS_y_epic_am_det.mean('time') >= -10 )
    DS_cal_sachs_month_subset = DS_cal_sachs.where(DS_y_epic_am_det.mean('time') >= -10)
    DS_cal_ggcmi_subset = DS_cal_ggcmi.where(DS_y_epic_am_det.mean('time') >= -10)
    
    ### Chose calendar:
    DS_chosen_calendar_am = DS_cal_ggcmi_subset['maturity_day'].round() #DS_cal_mirca_subset['start'] #DS_calendar_combined # [ DS_cal_ggcmi_subset['planting_day']  #DS_cal_sachs_month_subset['plant'] DS_cal_mirca_subset['start'] ]
    if DS_chosen_calendar_am.name != 'plant':
        DS_chosen_calendar_am = DS_chosen_calendar_am.rename('plant')
    
    # # ATTENTION, REMOVING GRID CELLS THAT LOOK WEIRD AND IRRELEVANT. Specific to GGCMI
    # DS_chosen_calendar_am = DS_chosen_calendar_am.where(DS_chosen_calendar_am != 2, drop = True)
    
    # Convert DS to df
    df_chosen_calendar = DS_chosen_calendar_am.to_dataframe().dropna()
    # Rounding up planting dates to closest integer in the month scale
    df_calendar_month_am = df_chosen_calendar[['plant']].apply(np.rint).astype(np.float32)
    
    # transform the months that are early in the year to the next year (ex 1 -> 13). Attention as this should be done only for the south america region
    df_calendar_month_am = calendar_multiyear_adjust([1,2,3,4,5], df_calendar_month_am['plant'])
    
    # Define the maturity date and then subtract X months
    df_calendar_month_am['plant'] = df_calendar_month_am['plant'] - 1
    
    ### LOAD climate date and clip to the calendar cells    
    DS_exclim_am_det_clip = DS_exclim_am_det.sel(time=slice(f'{DS_y_epic_am_det.time[0].values}-01-01','2016-12-31')).where(DS_chosen_calendar_am >= 0 )
    plot_2d_am_map(DS_exclim_am_det_clip['prcptot'].mean('time'))
    DS_exclim_am_det_clip.resample(time="1MS").mean(dim="time")
    
    # =============================================================================
    # CONVERT CLIMATIC VARIABLES ACCORDING TO THE SOYBEAN GROWING SEASON PER GRIDCELL 
    # =============================================================================
    df_features_reshape_2years_am = dynamic_calendar(DS_exclim_am_det_clip)
    ###################################################
    
    ### Select specific months ###################################################
    suffixes = tuple(["_"+str(j) for j in range(1,4)])
    df_feature_season_6mon_am = df_features_reshape_2years_am.loc[:,df_features_reshape_2years_am.columns.str.endswith(suffixes)]
    
    # Organising the data
    df_feature_season_6mon_am = df_feature_season_6mon_am.rename_axis(index={'year':'time'}).reorder_levels(['time','lat','lon']).sort_index()
    # df_feature_season_6mon_am = df_feature_season_6mon_am.where(df_obs_am_det['Yield'].mean>=0).dropna().astype(float)
    
    # SECOND DETRENDING PART - SEASONAL
    DS_feature_season_6mon_am = xr.Dataset.from_dataframe(df_feature_season_6mon_am)
    DS_feature_season_6mon_am = rearrange_latlot(DS_feature_season_6mon_am)
    
    DS_feature_season_6mon_am_det = detrend_dataset(DS_feature_season_6mon_am, deg = 'free')
    
    df_feature_season_6mon_am_det = DS_feature_season_6mon_am_det.to_dataframe().dropna()
    df_feature_season_6mon_am_det = df_feature_season_6mon_am_det.rename_axis(list(DS_feature_season_6mon_am_det.coords)).reorder_levels(['time','lat','lon']).sort_index()
    
    for feature in df_feature_season_6mon_am_det.columns:
        df_feature_season_6mon_am[feature].groupby('time').mean().plot(label = 'old')
        df_feature_season_6mon_am_det[feature].groupby('time').mean().plot(label = 'detrend')
        # df_feature_season_6mon_us_nodet[feature].groupby('time').mean().plot(label = '1 detrend')
        plt.title(f'{feature}')
        plt.legend()
        plt.show()
        print(np.round(df_feature_season_6mon_am_det[feature].groupby('time').mean().max(),3))
    
    # # Detrending in season
    df_feature_season_6mon_am_2012_mask = df_feature_season_6mon_am_det
    
    list_feat_precipitation = [s for s in df_feature_season_6mon_am_2012_mask.keys() if "prcptot" in s]
    for feature in list_feat_precipitation:
        df_feature_season_6mon_am_2012_mask[feature][df_feature_season_6mon_am_2012_mask[feature] < 0] = 0
    
    plot_2d_am_map(DS_feature_season_6mon_am_det['prcptot_3'].sel(time = start_date))
    plot_2d_am_map(DS_y_epic_am_det.sel(time = start_date))
    
    df_epic_am_det_2012 = df_epic_am_det_2012.where(df_feature_season_6mon_am['prcptot'+suffixes[0]] > -100).dropna().reorder_levels(['time','lat','lon']).sort_index()
    df_epic_am_det_2012 = df_epic_am_det_2012.rename(columns={'yield':'Yield'})
    
    df_input_hybrid_am_2012_mask = pd.concat([df_epic_am_det_2012, df_feature_season_6mon_am_2012_mask.reorder_levels(['time','lat','lon']).sort_index()], axis = 1)
    
    def country_location_add(df, US = DS_y_obs_us_all['Yield'], BR = DS_y_obs_br['Yield'], ARG = DS_y_obs_arg['Yield']):
        DS_country_us = xr.where(US >-100, 'US', np.nan)
        DS_country_br = xr.where(BR >-100, 'BR', np.nan)
        DS_country_arg = xr.where(ARG >-100, 'ARG', np.nan)
        DS_country_all = xr.merge([DS_country_us, DS_country_br, DS_country_arg], compat='no_conflicts')
        df_country_all = DS_country_all.to_dataframe().dropna().reorder_levels(['time','lat','lon']).sort_index()
        df_country_am = df_country_all.where(df['prcptot_1'] > -100).dropna()
        df = pd.concat([df, df_country_am], axis =1 )
        df = pd.get_dummies(df)
        return df
    df_input_hybrid_am_2012_mask = country_location_add(df_input_hybrid_am_2012_mask)

    return df_input_hybrid_am_2012_mask, df_epic_am_det_2012, DS_y_epic_am_det, DS_feature_season_6mon_am_det

if __name__ == "__main__":
    run()

