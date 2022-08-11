# -*- coding: utf-8 -*-
"""
Load factual
Load counterfactuals
Statistical comparisons

Created on Tue Mar 29 10:15:06 2022

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
import seaborn as sns
import matplotlib as mpl

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['figure.dpi'] = 144
mpl.rcParams.update({'font.size': 14})

# LOAD FUNCTIONS
def weighted_conversion(DS, DS_area, name_ds = 'Yield'):
    if type(DS) == xr.core.dataarray.DataArray:
        DS_weighted = ((DS * DS_area['harvest_area'].where(DS > -10) ) / DS_area['harvest_area'].where(DS > -10).sum(['lat','lon'])).to_dataset(name = name_ds)
    elif type(DS) == xr.core.dataarray.Dataset:
        DS_weighted = ((DS * DS_area['harvest_area'].where(DS > -10) / DS_area['harvest_area'].where(DS > -10).sum(['lat','lon'])))
    return DS_weighted.sum(['lat','lon'])

def rearrange_latlot(DS, resolution = 0.5):
    DS = DS.sortby('lat')
    DS = DS.sortby('lon')
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
def plot_2d_am_map(dataarray_2d, title = None, colormap = None, vmin = None, vmax = None, save_fig = None):
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
        plt.tight_layout()
        plt.savefig(f'paper_figures/{save_fig}.png', format='png', dpi=500)

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
    
#%% Load historical case
DS_historical_hybrid = xr.load_dataset("output_models_am/hybrid_epic-iiasa_gswp3-w5e5_obsclim_2015soc_default_yield-soy-noirr_global_annual_1901_2016.nc")
DS_historical_hybrid = rearrange_latlot(DS_historical_hybrid)

### Load future hybrid runs - detrended:
DS_hybrid_gfdl_26 = xr.load_dataset("output_models_am/hybrid_gfdl-esm4_ssp126_default_yield_soybean_2015_2100.nc")
DS_hybrid_gfdl_85 = xr.load_dataset("output_models_am/hybrid_gfdl-esm4_ssp585_default_yield_soybean_2015_2100.nc")
DS_hybrid_ipsl_26 = xr.load_dataset("output_models_am/hybrid_ipsl-cm6a-lr_ssp126_default_yield_soybean_2015_2100.nc")
DS_hybrid_ipsl_85 = xr.load_dataset("output_models_am/hybrid_ipsl-cm6a-lr_ssp585_default_yield_soybean_2015_2100.nc")
DS_hybrid_ukesm_26 = xr.load_dataset("output_models_am/hybrid_ukesm1-0-ll_ssp126_default_yield_soybean_2015_2100.nc")
DS_hybrid_ukesm_85 = xr.load_dataset("output_models_am/hybrid_ukesm1-0-ll_ssp585_default_yield_soybean_2015_2100.nc")

# Merge all scenarios
DS_hybrid_all = xr.merge([DS_hybrid_gfdl_26.rename({'yield-soy-noirr':'GFDL-esm4_1-2.6'}),DS_hybrid_gfdl_85.rename({'yield-soy-noirr':'GFDL-esm4_5-8.5'}),
                          DS_hybrid_ipsl_26.rename({'yield-soy-noirr':'IPSL-cm6a-lr_1-2.6'}),DS_hybrid_ipsl_85.rename({'yield-soy-noirr':'IPSL-cm6a-lr_5-8.5'}),
                          DS_hybrid_ukesm_26.rename({'yield-soy-noirr':'UKESM1-0-ll_1-2.6'}),DS_hybrid_ukesm_85.rename({'yield-soy-noirr':'UKESM1-0-ll_5-8.5'})])

#### Load future hybrid runs:
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

# Mean and standard deviation of each projections at grid level:
print(f'Mean historical:{round(DS_historical_hybrid.to_dataframe().mean().values.item(),2)} and Std: {round(DS_historical_hybrid.to_dataframe().std().values.item(),2)}')
print(f'Mean:{round(DS_hybrid_gfdl_85.to_dataframe().mean().values.item(),2)} and Std: {round(DS_hybrid_gfdl_85.to_dataframe().std().values.item(),2)}')
print(f'Mean:{round(DS_hybrid_ipsl_85.to_dataframe().mean().values.item(),2)} and Std: {round(DS_hybrid_ipsl_85.to_dataframe().std().values.item(),2)}')
print(f'Mean:{round(DS_hybrid_ukesm_85.to_dataframe().mean().values.item(),2)} and Std: {round(DS_hybrid_ukesm_85.to_dataframe().std().values.item(),2)}')

#%% WIEGHTED ANALYSIS
# =============================================================================

### Use MIRCA to isolate the rainfed 90% soybeans
# DS_mirca = xr.open_dataset("../../paper_hybrid_agri/data/americas_mask_ha.nc", decode_times=False).rename({'latitude': 'lat', 'longitude': 'lon','annual_area_harvested_rfc_crop08_ha_30mn':'harvest_area'})
DS_mirca = xr.open_dataset("../../paper_hybrid_agri/data/soy_harvest_spam_native_05x05.nc", decode_times=False)

#### HARVEST DATA
DS_harvest_area_sim = xr.load_dataset("../../paper_hybrid_agri/data/soybean_harvest_area_calculated_americas_hg.nc", decode_times=False)
DS_harvest_area_sim = DS_harvest_area_sim.sel(time = 2012) #.mean('time') 
DS_harvest_area_sim = DS_harvest_area_sim.where(DS_mirca['harvest_area'] > 0 )

# Historical, change with year
DS_harvest_area_hist = DS_harvest_area_sim.where(DS_historical_hybrid['Yield']> -5)
DS_harvest_area_hist = rearrange_latlot(DS_harvest_area_hist)

# Future, it works as the constant area throught the 21st century based on 2014/15/16
DS_harvest_area_fut = DS_harvest_area_fut = DS_harvest_area_hist.sel(time = 2012) #.sel(time=slice(2014,2016)).mean(['time'])
DS_harvest_area_fut = rearrange_latlot(DS_harvest_area_fut)

# Test plots to check for problems
plot_2d_am_map(DS_harvest_area_hist['harvest_area'].isel(time = 0))
plot_2d_am_map(DS_harvest_area_hist['harvest_area'].sel(time = 2012))
plot_2d_am_map(DS_harvest_area_fut['harvest_area'], title = 'Future projections')

# Weighted comparison for each model - degree of explanation
DS_historical_hybrid_weighted = weighted_conversion(DS_historical_hybrid['Yield'], DS_area = DS_harvest_area_hist)

# Future projections and transform into weighted timeseries
DS_hybrid_gfdl_26_weighted = weighted_conversion(DS_hybrid_gfdl_26['yield-soy-noirr'], DS_area = DS_harvest_area_fut)
DS_hybrid_gfdl_85_weighted = weighted_conversion(DS_hybrid_gfdl_85['yield-soy-noirr'], DS_area = DS_harvest_area_fut)
DS_hybrid_ipsl_26_weighted = weighted_conversion(DS_hybrid_ipsl_26['yield-soy-noirr'], DS_area = DS_harvest_area_fut)
DS_hybrid_ipsl_85_weighted = weighted_conversion(DS_hybrid_ipsl_85['yield-soy-noirr'], DS_area = DS_harvest_area_fut)
DS_hybrid_ukesm_26_weighted = weighted_conversion(DS_hybrid_ukesm_26['yield-soy-noirr'], DS_area = DS_harvest_area_fut)
DS_hybrid_ukesm_85_weighted = weighted_conversion(DS_hybrid_ukesm_85['yield-soy-noirr'], DS_area = DS_harvest_area_fut)

DS_historical_hybrid_weighted['Yield'].plot(label = 'history')
DS_hybrid_ukesm_85_weighted['Yield'].plot(label = 'UKESM1-0-ll_5-8.5')
plt.legend()
plt.show()

plt.figure(figsize=(8,6), dpi=300) #plot clusters
plt.axhline(y = DS_historical_hybrid_weighted['Yield'].sel(time = 2012).values, linestyle = 'dashed', label = 'Factual')
DS_hybrid_gfdl_26_weighted['Yield'].plot(label = 'GFDL-esm4_1-2.6')
DS_hybrid_gfdl_85_weighted['Yield'].plot(label = 'GFDL-esm4_5-8.5')
DS_hybrid_ipsl_26_weighted['Yield'].plot(label = 'IPSL-cm6a-lr_1-2.6')
DS_hybrid_ipsl_85_weighted['Yield'].plot(label = 'IPSL-cm6a-lr_5-8.5')
DS_hybrid_ukesm_26_weighted['Yield'].plot(label = 'UKESM1-0-ll_1-2.6')
DS_hybrid_ukesm_85_weighted['Yield'].plot(label = 'UKESM1-0-ll_5-8.5')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5), dpi=300) #plot clusters
sns.kdeplot(DS_historical_hybrid_weighted['Yield'], label = 'History', fill = True)
sns.kdeplot(DS_hybrid_gfdl_26_weighted['Yield'], label = 'GFDL-esm4_1-2.6', fill = True)
sns.kdeplot(DS_hybrid_gfdl_85_weighted['Yield'], label = 'GFDL-esm4_5-8.5', fill = True)
sns.kdeplot(DS_hybrid_ipsl_26_weighted['Yield'], label = 'IPSL-cm6a-lr_1-2.6', fill = True)
sns.kdeplot(DS_hybrid_ipsl_85_weighted['Yield'], label = 'IPSL-cm6a-lr_5-8.5', fill = True)
sns.kdeplot(DS_hybrid_ukesm_26_weighted['Yield'], label = 'UKESM1-0-ll_1-2.6', fill = True)
sns.kdeplot(DS_hybrid_ukesm_85_weighted['Yield'], label = 'UKESM1-0-ll_5-8.5', fill = True)
plt.legend()
plt.show()

# put the scenarios all together
DS_hybrid_all_weighted = weighted_conversion(DS_hybrid_all, DS_area = DS_harvest_area_fut)
df_hybrid_weighted_melt = pd.melt(DS_hybrid_all_weighted.to_dataframe(),ignore_index= False )
df_hybrid_weighted_melt_counterfactuals = df_hybrid_weighted_melt.where(df_hybrid_weighted_melt['value'] <= DS_historical_hybrid_weighted['Yield'].sel(time=2012).values )

df_hybrid_weighted_melt_counterfactuals_split = df_hybrid_weighted_melt_counterfactuals[df_hybrid_weighted_melt_counterfactuals['value'] > - 10].copy()
df_hybrid_weighted_melt_counterfactuals_split['SSP'] = df_hybrid_weighted_melt_counterfactuals_split.variable.str.split('_').str[-1]


# put the scenarios all together
DS_hybrid_trend_all_weighted = weighted_conversion(DS_hybrid_trend_all, DS_area = DS_harvest_area_fut)
df_hybrid_trend_weighted_melt = pd.melt(DS_hybrid_trend_all_weighted.to_dataframe(),ignore_index= False )
df_hybrid_trend_weighted_melt_counterfactuals = df_hybrid_trend_weighted_melt.where(df_hybrid_trend_weighted_melt['value'] <= DS_historical_hybrid_weighted['Yield'].sel(time=2012).values )

df_hybrid_trend_weighted_melt_counterfactuals_split = df_hybrid_trend_weighted_melt_counterfactuals[df_hybrid_trend_weighted_melt_counterfactuals['value'] > - 10].copy()
df_hybrid_trend_weighted_melt_counterfactuals_split['SSP'] = df_hybrid_trend_weighted_melt_counterfactuals_split.variable.str.split('_').str[-1]

print("Number of impact analogues:", (df_hybrid_weighted_melt_counterfactuals['value'] > -10).sum(),
      'An average per scenario of',(df_hybrid_weighted_melt_counterfactuals['value'] > -10).sum()/ (len(df_hybrid_weighted_melt_counterfactuals.variable.unique())-1) )

years_counterfactuals = df_hybrid_weighted_melt_counterfactuals[df_hybrid_weighted_melt_counterfactuals['value'] > -10]

plt.figure(figsize=(8,5), dpi=300) #plot clusters
plt.axhline(y = DS_historical_hybrid_weighted['Yield'].sel(time=2012).values)
sns.scatterplot(data = df_hybrid_weighted_melt_counterfactuals, 
                x = df_hybrid_weighted_melt_counterfactuals.index, y='value', hue = 'variable')


#%% Figure 2 - timeseries with factual baseline
# =============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
ax1.axhline(y = DS_historical_hybrid_weighted['Yield'].sel(time=2012).values, linestyle = 'dashed', label = '2012 event', linewidth = 2 )
DS_hybrid_trend_all_weighted['GFDL-esm4_1-2.6'].plot(label = 'GFDL-esm4 SSP126',marker='o', color='tab:blue', ax = ax1, linewidth = 2 )
DS_hybrid_trend_all_weighted['GFDL-esm4_5-8.5'].plot(label = 'GFDL-esm4 SSP585',marker='o', color='tab:orange', ax = ax1, linewidth = 2 )
DS_hybrid_trend_all_weighted['IPSL-cm6a-lr_1-2.6'].plot(label = 'IPSL-cm6a-lr SSP126',marker='^', color='tab:blue', ax = ax1, linewidth = 2 )
DS_hybrid_trend_all_weighted['IPSL-cm6a-lr_5-8.5'].plot(label = 'IPSL-cm6a-lr SSP585',marker='^', color='tab:orange', ax = ax1, linewidth = 2 )
DS_hybrid_trend_all_weighted['UKESM1-0-ll_1-2.6'].plot(label = 'UKESM1-0-ll SSP126',marker='s', color='tab:blue', ax = ax1, linewidth = 2 )
DS_hybrid_trend_all_weighted['UKESM1-0-ll_5-8.5'].plot(label = 'UKESM1-0-ll SSP585',marker='s', color='tab:orange', ax = ax1, linewidth = 2 )
ax1.set_title("a) Soybean timeseries in no-adaptation scenario")
ax1.legend(frameon = False)
plt.ylim(0.8,2.9)
ax1.set_ylabel('')

ax2.axhline(y = DS_historical_hybrid_weighted['Yield'].sel(time=2012).values, linestyle = 'dashed', label = '2012 event', linewidth = 2 )
DS_hybrid_gfdl_26_weighted['Yield'].plot(label = 'GFDL-esm4 SSP126',marker='o', color='tab:blue', ax = ax2, linewidth = 2 )
DS_hybrid_gfdl_85_weighted['Yield'].plot(label = 'GFDL-esm4 SSP585',marker='o', color='tab:orange', ax = ax2, linewidth = 2 )
DS_hybrid_ipsl_26_weighted['Yield'].plot(label = 'IPSL-cm6a-lr SSP126',marker='^', color='tab:blue', ax = ax2, linewidth = 2 )
DS_hybrid_ipsl_85_weighted['Yield'].plot(label = 'IPSL-cm6a-lr SSP585',marker='^', color='tab:orange', ax = ax2, linewidth = 2 )
DS_hybrid_ukesm_26_weighted['Yield'].plot(label = 'UKESM1-0-ll SSP126',marker='s', color='tab:blue', ax = ax2, linewidth = 2 )
DS_hybrid_ukesm_85_weighted['Yield'].plot(label = 'UKESM1-0-ll SSP585',marker='s', color='tab:orange', ax = ax2, linewidth = 2 )
ax2.set_title("b) Soybean timeseries in full-adaptation scenario")

plt.ylabel('')
fig.supylabel('Yield (ton/ha)')
plt.tight_layout()
plt.savefig('paper_figures/timeseries_projections_ab.png', format='png', dpi=500)
plt.show()


# =============================================================================
# # Fig A1 - supplementary figure with the frequency of analogues per RCP, and the magnitude of the analogues (measured as analogue - historical event)
# =============================================================================
length_index = len(df_hybrid_trend_weighted_melt_counterfactuals_split[df_hybrid_trend_weighted_melt_counterfactuals_split['SSP']=='5-8.5'])
fig,((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(12, 8), sharex=True)
 
sns.histplot( x = df_hybrid_trend_weighted_melt_counterfactuals_split[['SSP','value']]['SSP'].sort_values().values, ax =ax1)
ax1.set_title('a)', loc='left')
ax1.set_ylim(0,length_index+5)

sns.histplot( x = df_hybrid_weighted_melt_counterfactuals_split[['SSP','value']]['SSP'].sort_values().values, ax=ax2)
ax2.set_title('b)', loc='left')
ax2.set_ylim(0,length_index+5)

bar = sns.boxplot(data = df_hybrid_trend_weighted_melt_counterfactuals_split, y= df_hybrid_trend_weighted_melt_counterfactuals_split['value'] - DS_historical_hybrid_weighted['Yield'].min().values , x= 'SSP', ax = ax3, order= ['1-2.6','5-8.5'])
ax3.set_ylabel('Yield anomaly (ton/ha)')
ax3.set_title('c)', loc='left')
ax3.set_ylim(-1.2,0.2)

bar = sns.boxplot(data = df_hybrid_weighted_melt_counterfactuals_split, y= df_hybrid_weighted_melt_counterfactuals_split['value'] - DS_historical_hybrid_weighted['Yield'].min().values , x= 'SSP', ax = ax4, order= ['1-2.6','5-8.5'])
ax4.set_ylabel('Yield anomaly (ton/ha)')
ax4.set_title('d)', loc='left')
ax4.set_ylim(-1.2,0.2)

plt.tight_layout()
plt.savefig('paper_figures/fig_sup_timeseries_adap.png', format='png', dpi=500)
plt.show()

diff_rcp85 = df_hybrid_trend_weighted_melt_counterfactuals_split.where(df_hybrid_trend_weighted_melt_counterfactuals_split['SSP'] == '5-8.5').mean().values - DS_historical_hybrid_weighted['Yield'].mean().values
diff_rcp26 = df_hybrid_trend_weighted_melt_counterfactuals_split.where(df_hybrid_trend_weighted_melt_counterfactuals_split['SSP'] == '1-2.6').mean().values - DS_historical_hybrid_weighted['Yield'].mean().values

print('Difference mean rcp85 losses to the 2012 event is:', diff_rcp85 / (DS_historical_hybrid_weighted['Yield'].sel(time=2012).values - DS_historical_hybrid_weighted['Yield'].mean().values) )
print('Difference mean rcp26 losses to the 2012 event is:', diff_rcp26 / (DS_historical_hybrid_weighted['Yield'].sel(time=2012).values - DS_historical_hybrid_weighted['Yield'].mean().values) )

print('Number of rcp26 analogues:', len(df_hybrid_weighted_melt_counterfactuals_split.where(df_hybrid_weighted_melt_counterfactuals_split['SSP'] == '1-2.6').dropna()))
print('Number of rcp85 analogues:', len(df_hybrid_weighted_melt_counterfactuals_split.where(df_hybrid_weighted_melt_counterfactuals_split['SSP'] == '5-8.5').dropna()))

print('Number of trend rcp26 analogues:', len(df_hybrid_trend_weighted_melt_counterfactuals_split.where(df_hybrid_trend_weighted_melt_counterfactuals_split['SSP'] == '1-2.6').dropna()))
print('Number of trend rcp85 analogues:', len(df_hybrid_trend_weighted_melt_counterfactuals_split.where(df_hybrid_trend_weighted_melt_counterfactuals_split['SSP'] == '5-8.5').dropna()))
      
#%% Comparison of factual and counterfactuals

# =============================================================================
# # Identify the FACTUAL case - Minimum value with respect to the mean
# =============================================================================
yield_factual = DS_historical_hybrid_weighted.sel(time = 2012)['Yield'].values

# =============================================================================
# COUNTERFACTUALS
# =============================================================================
# Find impact analogues based on the yield value of the factual case.
DS_counterfactuals_weighted_am = DS_hybrid_all_weighted.where(DS_hybrid_all_weighted <= yield_factual)
print("Number of impact analogues per scenario:", (DS_counterfactuals_weighted_am > -10).sum())
# Check years of counterfactuals
list_counterfactuals_scenarios = []
for feature in list(DS_counterfactuals_weighted_am.keys()):
    feature_counterfactuals = DS_counterfactuals_weighted_am[feature].dropna(dim = 'time')
    print(feature_counterfactuals.time.values)
    list_counterfactuals_scenarios.append(feature_counterfactuals.time.values)
print('Total counterfactuals: ', len(np.hstack(list_counterfactuals_scenarios)) )

# Create dataset with spatailly explict counterfactuals only 
DS_counterfactuals_spatial = DS_hybrid_all.where(DS_counterfactuals_weighted_am > -10)

# Find the counterfactual shocks using a baseline as reference, either historical yields or the factual as reference
DS_hybrid_counterfactuals_spatial_shock = DS_counterfactuals_spatial.dropna('time', how='all') - DS_historical_hybrid['Yield'].mean('time')
DS_hybrid_counterfactuals_spatial_shock_2012 = DS_counterfactuals_spatial.dropna('time', how='all') - DS_historical_hybrid['Yield'].sel(time = 2012)

# =============================================================================
# # Plots the counterfactuals per scenario 
# =============================================================================
list_ds_counterfactuals = []
for feature in list(DS_hybrid_counterfactuals_spatial_shock.keys()):
    counterfactuals_by_rcp = DS_hybrid_counterfactuals_spatial_shock[feature].sel(time = DS_counterfactuals_weighted_am[feature].time.where(DS_counterfactuals_weighted_am[feature] > -10).dropna(dim = 'time'))
    plot_2d_am_multi(counterfactuals_by_rcp, map_title = feature )
    list_ds_counterfactuals.append(counterfactuals_by_rcp)
combined = xr.concat(list_ds_counterfactuals, dim='time')
ds_combined = combined.to_dataset(name='yield (ton/ha)')

# spatial distribution of the future analogues with respect to the 2012 year
plot_2d_am_map(ds_combined['yield (ton/ha)'].mean('time'), colormap = 'RdBu', save_fig = 'mean_fut_analogues_spatial')

# for feature in list(DS_hybrid_counterfactuals_spatial_shock_2012.keys()):
#     plot_2d_am_multi(DS_hybrid_counterfactuals_spatial_shock_2012[feature].sel(time = DS_counterfactuals_weighted_am[feature].time.where(DS_counterfactuals_weighted_am[feature] > -10).dropna(dim = 'time')), map_title = feature )
   
# =============================================================================
# Climatic analysis of counterfactuals
# =============================================================================
# Function converting the climate data for the counterfactuals into weighted timeseries 
def convert_clim_weighted_ensemble(df_clim, DS_counterfactuals_weighted_country, feature, DS_area):
    DS_clim = xr.Dataset.from_dataframe(df_clim)
    DS_clim = rearrange_latlot(DS_clim)
    # Countefactuals only
    DS_clim_counter = DS_clim.where(DS_counterfactuals_weighted_country[feature].time.where(DS_counterfactuals_weighted_country[feature] > -10).dropna(dim = 'time'))
   
    DS_clim_counter_weight_country = weighted_conversion(DS_clim_counter, DS_area = DS_area)
    df_clim_counter_weight_country = DS_clim_counter_weight_country.to_dataframe()
    df_clim_counter_weight_country['scenario'] = 'Analogues'
    df_clim_counter_weight_country['model_used'] = feature
    
    return df_clim_counter_weight_country

# Load historical climatic information - input
df_hybrid_am = pd.read_csv('dataset_input_hybrid_am_forML.csv', index_col=[0,1,2],).iloc[:, 1:].copy()

# Plot the mean temperature values for all counterfactuals and subtract from the 2012 year.
DS_conditions_hist = xr.Dataset.from_dataframe(df_hybrid_am)
DS_conditions_hist = rearrange_latlot(DS_conditions_hist)

# Load clim projections and Remove yield information
df_hybrid_fut_ukesm_585_am = pd.read_csv('output_models_am/climatic_projections/model_input_ukesm1-0-ll_ssp585_default_2015_2100.csv', index_col=[0,1,2],).iloc[:, 1:]
df_hybrid_fut_ukesm_126_am = pd.read_csv('output_models_am/climatic_projections/model_input_ukesm1-0-ll_ssp126_default_2015_2100.csv', index_col=[0,1,2],).iloc[:, 1:]
df_hybrid_fut_gfdl_585_am = pd.read_csv('output_models_am/climatic_projections/model_input_gfdl-esm4_ssp585_default_2015_2100.csv', index_col=[0,1,2],).iloc[:, 1:]
df_hybrid_fut_gfdl_126_am = pd.read_csv('output_models_am/climatic_projections/model_input_gfdl-esm4_ssp126_default_2015_2100.csv', index_col=[0,1,2],).iloc[:, 1:]
df_hybrid_fut_ipsl_585_am = pd.read_csv('output_models_am/climatic_projections/model_input_ipsl-cm6a-lr_ssp585_default_2015_2100.csv', index_col=[0,1,2],).iloc[:, 1:]
df_hybrid_fut_ipsl_126_am = pd.read_csv('output_models_am/climatic_projections/model_input_ipsl-cm6a-lr_ssp126_default_2015_2100.csv', index_col=[0,1,2],).iloc[:, 1:]

# Conversion of historical series to weighted timeseries    
DS_conditions_hist_weighted_am = weighted_conversion(DS_conditions_hist, DS_area = DS_harvest_area_hist)
DS_conditions_2012_weighted_am = DS_conditions_hist_weighted_am.sel(time=2012)
# DS to df
df_clim_hist_weighted = DS_conditions_hist_weighted_am.to_dataframe()
df_clim_hist_weighted['scenario'] = 'Climatology'
df_clim_hist_weighted['model_used'] = 'Climatology'

# Conversion of future series to weighted timeseries    
df_clim_counter_ukesm_85 = convert_clim_weighted_ensemble(df_hybrid_fut_ukesm_585_am, DS_counterfactuals_weighted_am, 'UKESM1-0-ll_5-8.5', DS_harvest_area_fut)    
df_clim_counter_ukesm_26 = convert_clim_weighted_ensemble(df_hybrid_fut_ukesm_126_am, DS_counterfactuals_weighted_am, 'UKESM1-0-ll_1-2.6', DS_harvest_area_fut)    
df_clim_counter_gfdl_85 = convert_clim_weighted_ensemble(df_hybrid_fut_gfdl_585_am, DS_counterfactuals_weighted_am, 'GFDL-esm4_5-8.5', DS_harvest_area_fut)    
df_clim_counter_gfdl_26 = convert_clim_weighted_ensemble(df_hybrid_fut_gfdl_126_am, DS_counterfactuals_weighted_am, 'GFDL-esm4_1-2.6', DS_harvest_area_fut)    
df_clim_counter_ipsl_85 = convert_clim_weighted_ensemble(df_hybrid_fut_ipsl_585_am, DS_counterfactuals_weighted_am, 'IPSL-cm6a-lr_5-8.5', DS_harvest_area_fut)    
df_clim_counter_ipsl_26 = convert_clim_weighted_ensemble(df_hybrid_fut_ipsl_126_am, DS_counterfactuals_weighted_am, 'IPSL-cm6a-lr_1-2.6', DS_harvest_area_fut)    

# Merge dataframes with different names
df_clim_counterfactuals_weighted_all_am = pd.concat([df_clim_hist_weighted, df_clim_counter_ukesm_85, df_clim_counter_ukesm_26, 
                                                  df_clim_counter_gfdl_85, df_clim_counter_gfdl_26,
                                                  df_clim_counter_ipsl_85, df_clim_counter_ipsl_26])


# =============================================================================
# # Fig 3: Plot boxplots comparing the 2012 event and the analogues
# =============================================================================
names = df_clim_counterfactuals_weighted_all_am.columns.drop(['scenario', 'model_used'])
ncols = len(names)
fig, axes  = plt.subplots(2,int(np.ceil(len(df_clim_counterfactuals_weighted_all_am.columns)/3)), figsize=(10, 8), dpi=300, gridspec_kw=dict(height_ratios=[1,1]))

for name, ax in zip(names, axes.flatten()):
    df_merge_subset_am = df_clim_counterfactuals_weighted_all_am[df_clim_counterfactuals_weighted_all_am.index != 2012].loc[:,[name,'scenario']]
    df_merge_subset_am['variable'] = name
    # ax.axhspan(df_merge_subset_am[name].where(df_merge_subset_am['scenario'] == 'Climatology').dropna().quantile(0.05), df_merge_subset_am[name].where(df_merge_subset_am['scenario'] == 'Climatology').dropna().quantile(0.95), facecolor='0.2', alpha=0.3, zorder = 0, label = 'Climatology')
    g1 = sns.boxplot(y=name, x = 'variable', hue = 'scenario', data=df_merge_subset_am.where(df_merge_subset_am['scenario'] == 'Analogues').dropna(), orient='v', ax=ax)
    # g1 = sns.scatterplot(y=name, x = 'variable', data=df_merge_subset_am.where(df_merge_subset_am['scenario'] == 'Analogues' ), ax=ax, color = 'orange', s=60, label = 'Analogues', zorder = 20)
    ax.axhline( y = DS_conditions_2012_weighted_am[name].mean(), color = 'red', linestyle = 'dashed',linewidth =2, label = '2012 event', zorder = 19)
    g1.set(xticklabels=[])  # remove the tick labels
    g1.set(xlabel= name)
    if name in names[0:3]:
        g1.set(ylabel= 'Precipitation (mm/month)')  # remove the axis label  
    elif name in names[3:6]:
        g1.set(ylabel='Temperature (°C)' )  # remove the axis label   
    ax.get_legend().remove()
    g1.tick_params(bottom=False)  # remove the ticks
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc=[0.3, -0.01], ncol=3, frameon=False)
# plt.suptitle('2012 analogues')
plt.tight_layout()
plt.savefig('paper_figures/clim_conditions_2012.png', format='png', dpi=500)
plt.show()


# =============================================================================
# # Fig SI: Plot boxplots comparing the historical events to the 2012 analogues
# =============================================================================
names = df_clim_counterfactuals_weighted_all_am.columns.drop(['scenario', 'model_used'])
ncols = len(names)
fig, axes  = plt.subplots(2,int(np.ceil(len(df_clim_counterfactuals_weighted_all_am.columns)/3)), figsize=(10, 8), dpi=300, gridspec_kw=dict(height_ratios=[1,1]))

for name, ax in zip(names, axes.flatten()):
    df_merge_subset_am = df_clim_counterfactuals_weighted_all_am[df_clim_counterfactuals_weighted_all_am.index != 2012].loc[:,[name,'scenario']]
    df_merge_subset_am['variable'] = name
    # ax.axhspan(df_merge_subset_am[name].where(df_merge_subset_am['scenario'] == 'Climatology').dropna().quantile(0.05), df_merge_subset_am[name].where(df_merge_subset_am['scenario'] == 'Climatology').dropna().quantile(0.95), facecolor='0.2', alpha=0.3, zorder = 0, label = 'Climatology')
    g1 = sns.boxplot(y=name, x = 'variable', hue = 'scenario', data=df_merge_subset_am, orient='v', ax=ax)
    # g1 = sns.scatterplot(y=name, x = 'variable', data=df_merge_subset_am.where(df_merge_subset_am['scenario'] == 'Analogues' ), ax=ax, color = 'orange', s=60, label = 'Analogues', zorder = 20)
    # ax.axhline( y = DS_conditions_2012_weighted_am[name].mean(), color = 'red', linestyle = 'dashed',linewidth =2, label = '2012 event', zorder = 19)
    g1.set(xticklabels=[])  # remove the tick labels
    g1.set(xlabel= name)
    if name in names[0:3]:
        g1.set(ylabel= 'Precipitation (mm/month)')  # remove the axis label  
    elif name in names[3:6]:
        g1.set(ylabel='Temperature (°C)' )  # remove the axis label   
    ax.get_legend().remove()
    g1.tick_params(bottom=False)  # remove the ticks
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc=[0.3, -0.01], ncol=3, frameon=False)
# plt.suptitle('2012 analogues')
plt.tight_layout()
plt.savefig('paper_figures/fig_si_clim_conditions_2012_climatology.png', format='png', dpi=500)
plt.show()


# =============================================================================
# # Fig SI: Plot boxplots comparing the historical events to the 2012 analogues
# =============================================================================
names = df_clim_counterfactuals_weighted_all_am.columns.drop(['scenario', 'model_used'])
ncols = len(names)
fig, axes  = plt.subplots(2,int(np.ceil(len(df_clim_counterfactuals_weighted_all_am.columns)/3)), figsize=(10, 8), dpi=300, gridspec_kw=dict(height_ratios=[1,1]))

for name, ax in zip(names, axes.flatten()):
    df_merge_subset_am = df_clim_counterfactuals_weighted_all_am[df_clim_counterfactuals_weighted_all_am.index != 2012].loc[:,[name,'scenario']]
    df_merge_subset_am['variable'] = name
    # ax.axhspan(df_merge_subset_am[name].where(df_merge_subset_am['scenario'] == 'Climatology').dropna().quantile(0.05), df_merge_subset_am[name].where(df_merge_subset_am['scenario'] == 'Climatology').dropna().quantile(0.95), facecolor='0.2', alpha=0.3, zorder = 0, label = 'Climatology')
    g1 = sns.boxplot(y=name, x = 'variable', hue = 'scenario', data=df_merge_subset_am.where(df_merge_subset_am['scenario'] == 'Climatology').dropna(), orient='v', ax=ax)
    # g1 = sns.scatterplot(y=name, x = 'variable', data=df_merge_subset_am.where(df_merge_subset_am['scenario'] == 'Analogues' ), ax=ax, color = 'orange', s=60, label = 'Analogues', zorder = 20)
    ax.axhline( y = DS_conditions_2012_weighted_am[name].mean(), color = 'red', linestyle = 'dashed',linewidth =2, label = '2012 event', zorder = 19)
    g1.set(xticklabels=[])  # remove the tick labels
    g1.set(xlabel= name)
    if name in names[0:3]:
        g1.set(ylabel= 'Precipitation (mm/month)')  # remove the axis label  
    elif name in names[3:6]:
        g1.set(ylabel='Temperature (°C)' )  # remove the axis label   
    ax.get_legend().remove()
    g1.tick_params(bottom=False)  # remove the ticks
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc=[0.3, -0.01], ncol=3, frameon=False)
# plt.suptitle('2012 analogues')
plt.tight_layout()
# plt.savefig('paper_figures/fig_si_clim_conditions_2012_climatology.png', format='png', dpi=500)
plt.show()


#%% 2012 analogues at a spatial level - how each country is affected by the 2012 analogues
# =============================================================================
'''
Try to compare the counterfatuals from a country perspective, does it mean one of the three countries is more prone to losses / failures?
'''
# Determine country level yields for historical period
DS_historical_hybrid_us = xr.load_dataset("output_models_am/hybrid_epic_us-iiasa_gswp3-w5e5_obsclim_2015soc_default_yield-soy-noirr_global_annual_1901_2016.nc")
DS_historical_hybrid_br = xr.load_dataset("output_models_am/hybrid_epic_br-iiasa_gswp3-w5e5_obsclim_2015soc_default_yield-soy-noirr_global_annual_1901_2016.nc")
DS_historical_hybrid_arg = xr.load_dataset("output_models_am/hybrid_epic_arg-iiasa_gswp3-w5e5_obsclim_2015soc_default_yield-soy-noirr_global_annual_1901_2016.nc")

DS_mirca_us_hist = DS_harvest_area_sim.where(DS_historical_hybrid_us['Yield'] > -10)
DS_mirca_br_hist = DS_harvest_area_sim.where(DS_historical_hybrid_br['Yield'] > -10)
DS_mirca_arg_hist = DS_harvest_area_sim.where(DS_historical_hybrid_arg['Yield'] > -10)

# Determine country level yields for future projections
DS_counterfactual_us = DS_counterfactuals_spatial.where(DS_historical_hybrid_us['Yield'].sel(time = 2012) > -10)
DS_mirca_us = DS_harvest_area_fut.where(DS_historical_hybrid_us['Yield'].sel(time = 2012) > -10)

DS_counterfactual_br = DS_counterfactuals_spatial.where(DS_historical_hybrid_br['Yield'].sel(time = 2012) > -10)
DS_mirca_br = DS_harvest_area_fut.where(DS_historical_hybrid_br['Yield'].sel(time = 2012) > -10)

DS_counterfactual_arg = DS_counterfactuals_spatial.where(DS_historical_hybrid_arg['Yield'].sel(time = 2012) > -10)
DS_mirca_arg = DS_harvest_area_fut.where(DS_historical_hybrid_arg['Yield'].sel(time = 2012) > -10)

plot_2d_am_map(DS_mirca_arg.harvest_area)

# Weighted analysis historical
DS_historical_hybrid_us_weight = weighted_conversion(DS_historical_hybrid_us, DS_area = DS_mirca_us_hist)
DS_historical_hybrid_br_weight = weighted_conversion(DS_historical_hybrid_br, DS_area = DS_mirca_br_hist)
DS_historical_hybrid_arg_weight = weighted_conversion(DS_historical_hybrid_arg, DS_area = DS_mirca_arg_hist)

# Plot historical timeline of weighted soybean yield
plt.plot(DS_historical_hybrid_us_weight.time, DS_historical_hybrid_us_weight['Yield'], label = 'US')
plt.plot(DS_historical_hybrid_br_weight.time, DS_historical_hybrid_br_weight['Yield'], label = 'Brazil')
plt.plot(DS_historical_hybrid_arg_weight.time, DS_historical_hybrid_arg_weight['Yield'], label = 'Argentina')
plt.title('Historical hybrid data')
plt.ylim(1,3)
plt.legend()
plt.show()

# Plot aggregated historical timeseries - equivalent to production / global contribution
DS_mirca_us['harvest_area'].sum() + DS_mirca_br['harvest_area'].sum() + DS_mirca_arg['harvest_area'].sum() - DS_harvest_area_hist.mean('time')['harvest_area'].sum()

# Plot production - TEST
def production(DS, DS_area):
    if type(DS) == xr.core.dataarray.DataArray:
        DS_weighted = ((DS * DS_area['harvest_area'] ) ).to_dataset(name = 'Yield')
    elif type(DS) == xr.core.dataarray.Dataset:
        DS_weighted = ((DS * DS_area['harvest_area'] ) )
    return DS_weighted.sum(['lat','lon'])

DS_produc_am = production(DS_historical_hybrid, DS_harvest_area_fut)
DS_produc_us = production(DS_historical_hybrid_us, DS_mirca_us_hist)
DS_produc_br = production(DS_historical_hybrid_br, DS_mirca_br_hist)
DS_produc_arg = production(DS_historical_hybrid_arg, DS_mirca_arg_hist)

plt.plot(DS_produc_am.time, DS_produc_am['Yield'], label = 'AM')
plt.stackplot(DS_produc_us.time, DS_produc_us['Yield'],
              DS_produc_br['Yield'], DS_produc_arg['Yield'], labels = ['US', 'Brazil', 'Argentina'])
plt.legend()
plt.tight_layout()
plt.show()

# Weighted analysis future projections
DS_counterfactual_us_weighted = weighted_conversion(DS_counterfactual_us, DS_area = DS_mirca_us)
DS_counterfactual_us_weighted = DS_counterfactual_us_weighted.where(DS_counterfactual_us_weighted > 0).dropna('time', how = 'all')

DS_counterfactual_br_weighted = weighted_conversion(DS_counterfactual_br, DS_area = DS_mirca_br)
DS_counterfactual_br_weighted = DS_counterfactual_br_weighted.where(DS_counterfactual_br_weighted > 0).dropna('time', how = 'all')

DS_counterfactual_arg_weighted = weighted_conversion(DS_counterfactual_arg, DS_area = DS_mirca_arg)
DS_counterfactual_arg_weighted = DS_counterfactual_arg_weighted.where(DS_counterfactual_arg_weighted > 0).dropna('time', how = 'all')

df_mean_yields = pd.DataFrame( [DS_counterfactual_us_weighted.to_dataframe().mean(), DS_counterfactual_br_weighted.to_dataframe().mean(),
                               DS_counterfactual_arg_weighted.to_dataframe().mean()], 
                                          index = ['US', 'Brazil', 'Argentina'])

# =============================================================================
# # Plot accumulated mean losses of projected counterfactuals per country with reference to aggregated series
# =============================================================================
df_mean_yields.plot.bar(figsize = (12,8))
plt.axhline(y = yield_factual, linestyle = 'dashed', label = 'Factual')
plt.show()

# Plot accumulated mean losses per country - Absolute shock
sns.barplot(data = df_mean_yields.T )
plt.axhline(y = yield_factual, color = 'black', linestyle = 'dashed', label = '2012 event')
plt.legend(frameon=False)
plt.title('Counterfactuals absolute values')
plt.ylabel('ton/ha')
plt.show()

# Plot accumulated mean losses per country - Absolute shock
sns.barplot(data = df_mean_yields.T - DS_historical_hybrid_weighted['Yield'].mean().values)
plt.axhline(y = yield_factual - DS_historical_hybrid_weighted['Yield'].mean().values, color = 'black', linestyle = 'dashed', label = '2012 event')
plt.legend(frameon=False)
plt.title('Analogues of the 2012 event in each country')
plt.ylabel('Yield anomaly (ton/ha)')
plt.show()

# Plot accumulated mean losses per country - Relative shock
sns.barplot(data = df_mean_yields.T / DS_historical_hybrid_weighted['Yield'].mean().values - 1)
plt.axhline(y = yield_factual / DS_historical_hybrid_weighted['Yield'].mean().values - 1, color = 'black', linestyle = 'dashed', label = '2012 event')
plt.legend(frameon=False)
plt.title('Analogues of the 2012 event in each country (relative)')
plt.show()

#%% Analysis at a country level - 2012 event from a country perspective, aggregation at country level

# =============================================================================
# Failures per country
yield_factual_2012_us = DS_historical_hybrid_us_weight.sel(time = 2012).Yield.values
yield_factual_2012_br = DS_historical_hybrid_br_weight.sel(time = 2012).Yield.values
yield_factual_2012_arg = DS_historical_hybrid_arg_weight.sel(time = 2012).Yield.values
yield_factual_2012_am = pd.DataFrame([yield_factual_2012_us, yield_factual_2012_br, yield_factual_2012_arg], index = ['US', 'Brazil', 'Argentina'])

mean_historical_values_per_country = pd.DataFrame([DS_historical_hybrid_us_weight.Yield.mean('time').values, DS_historical_hybrid_br_weight.Yield.mean('time').values,  DS_historical_hybrid_arg_weight.Yield.mean('time').values], index = ['US', 'Brazil', 'Argentina'])

show_historical_data = False
if show_historical_data == True:
    # =============================================================================
    # # Historical failures at a country level
    # =============================================================================
    plt.bar(x = 'US', height = yield_factual_2012_us)
    plt.bar(x = 'Brazil', height = yield_factual_2012_br)
    plt.bar(x = 'Argentina', height = yield_factual_2012_arg)
    plt.axhline(y = yield_factual, color = 'black', linestyle = 'dashed', label = 'Aggregated')
    plt.title('2012 yield shock')
    plt.legend()
    plt.ylabel('ton/ha')
    plt.show()
    
    # Absolute shocks
    plt.bar(x = 'US', height = yield_factual_2012_us - DS_historical_hybrid_us_weight.Yield.mean('time').values)
    plt.bar(x = 'Brazil', height = yield_factual_2012_br - DS_historical_hybrid_br_weight.Yield.mean('time').values)
    plt.bar(x = 'Argentina', height = yield_factual_2012_arg- DS_historical_hybrid_arg_weight.Yield.mean('time').values)
    plt.axhline(y = yield_factual - DS_historical_hybrid_weighted.Yield.mean('time').values, color = 'black', linestyle = 'dashed', label = 'Aggregated')
    plt.title('2012 absolute shock')
    plt.ylabel('Yield anomaly (ton/ha)')
    plt.legend()
    plt.show()
    
    # Relative shocks
    plt.bar(x = 'US', height = yield_factual_2012_us / DS_historical_hybrid_us_weight.Yield.mean('time').values - 1)
    plt.bar(x = 'Brazil', height = yield_factual_2012_br / DS_historical_hybrid_br_weight.Yield.mean('time').values - 1)
    plt.bar(x = 'Argentina', height = yield_factual_2012_arg / DS_historical_hybrid_arg_weight.Yield.mean('time').values - 1)
    plt.axhline(y = yield_factual / DS_historical_hybrid_weighted.Yield.mean('time').values - 1, color = 'black', linestyle = 'dashed', label = 'Aggregated')
    plt.title('2012 relative shock')
    plt.ylabel('0-1')
    plt.legend()
    plt.show()
    
    # =============================================================================
    # 2012 analogues with respect to historical country levels
    # =============================================================================
   
    # Counterfactuals of 2012 event in absolute values
    sns.barplot(data = (df_mean_yields.T ))
    plt.axhline(y = yield_factual, color = 'black', linestyle = 'dashed', label = '2012 event')
    plt.legend()
    plt.title('Counterfactuals absolute values')
    plt.ylabel('ton/ha')
    plt.show()
    
    # Counterfactuals of 2012 event in absolute shock
    sns.barplot(data = (df_mean_yields.T - mean_historical_values_per_country.T.values))
    plt.axhline(y = yield_factual - DS_historical_hybrid_weighted['Yield'].mean().values, color = 'black', linestyle = 'dashed', label = '2012 event')
    plt.legend()
    plt.title('Local analogues of each country')
    plt.ylabel('ton/ha')
    plt.show()
    
    # Plot accumulated mean losses per country - Relative shock
    sns.barplot(data = (df_mean_yields.T / mean_historical_values_per_country.T.values - 1))
    plt.axhline(y = yield_factual / DS_historical_hybrid_weighted['Yield'].mean().values - 1, color = 'black', linestyle = 'dashed', label = '2012 event')
    plt.legend()
    plt.title('Counterfactuals relative shock')
    plt.show()

#%% Number of counterfactuals per country level, which scenarios and years they appear.

# Local counterfactuals
def counterfactual_generation(DS_yields, DS_mirca_country, local_factual):
    # Define the extend of the future timeseries based on the 2012 year
    DS_hybrid_country = DS_yields.where(DS_mirca_country['harvest_area'] > 0)
    # Make conversion to weighted timeseries
    DS_projections_weighted_country = weighted_conversion(DS_hybrid_country, DS_area = DS_mirca_country)
    # Isolate the years with counterfactuals and remove the ones without
    DS_projections_weighted_country_counterfactual = DS_projections_weighted_country.where(DS_projections_weighted_country <= local_factual).dropna('time', how = 'all')
    return DS_projections_weighted_country_counterfactual


def counterfactuals_country_level(DS_projections_weighted_country_counterfactual, local_factual):
    
    print("Number of impact analogues per scenario for the country:", (DS_projections_weighted_country_counterfactual > -10).sum())
    
    # Check years of counterfactuals
    list_counterfactuals_scenarios = []
    for feature in list(DS_projections_weighted_country_counterfactual.keys()):
        feature_counterfactuals = DS_projections_weighted_country_counterfactual[feature].dropna(dim = 'time')
        print(feature_counterfactuals.time.values)
        list_counterfactuals_scenarios.append(feature_counterfactuals.time.values)
        number_counter = len(np.hstack(list_counterfactuals_scenarios)) 
    print('total counterfactuals: ', number_counter)
        
    return DS_projections_weighted_country_counterfactual, number_counter


DS_projections_weighted_us_counterfactual = counterfactual_generation(DS_hybrid_all, DS_mirca_us, yield_factual_2012_us)
DS_projections_weighted_br_counterfactual = counterfactual_generation(DS_hybrid_all, DS_mirca_br, yield_factual_2012_br)
DS_projections_weighted_arg_counterfactual = counterfactual_generation(DS_hybrid_all, DS_mirca_arg, yield_factual_2012_arg)
    
# Local analogues comparison bars
local_analogues = pd.DataFrame([DS_projections_weighted_us_counterfactual.to_dataframe().mean(), DS_projections_weighted_br_counterfactual.to_dataframe().mean(), DS_projections_weighted_arg_counterfactual.to_dataframe().mean()],
                               index = ['US','Brazil','Argentina'])

DS_counterfactuals_weighted_us, number_counter_us = counterfactuals_country_level(DS_projections_weighted_us_counterfactual, yield_factual_2012_us)
DS_counterfactuals_weighted_br, number_counter_br = counterfactuals_country_level(DS_projections_weighted_br_counterfactual, yield_factual_2012_br)
DS_counterfactuals_weighted_arg, number_counter_arg = counterfactuals_country_level(DS_projections_weighted_arg_counterfactual, yield_factual_2012_arg)

# =============================================================================
# FIG 4: Local analogues
# =============================================================================
fig, (ax1,ax2) = plt.subplots(nrows=2, figsize=(8,8),sharex = True )

ax1.bar(x = ['US','Brazil','Argentina'], height = [number_counter_us, number_counter_br, number_counter_arg])
ax1.set_title('a) Number of local analogues per country')
ax1.set_ylabel('Count')
# Counterfactuals of 2012 event in absolute shock
sns.barplot(data = (local_analogues.T - mean_historical_values_per_country.T.values), ax=ax2)
# ax2.axhline(y = yield_factual - DS_historical_hybrid_weighted['Yield'].mean().values, color = 'black', linestyle = 'dashed', label = '2012 event')
ax2.set_title('b) Magnitude of local analogues per country')
ax2.set_ylabel('Yield (ton/ha)')
plt.tight_layout()
plt.savefig('paper_figures/number_analogues.png', format='png', dpi=500)
plt.show()


# Supplementary information on local analogues - co-occurrence

years_counterfactuals_am = df_hybrid_weighted_melt_counterfactuals[df_hybrid_weighted_melt_counterfactuals['value'] > -10].copy()
years_counterfactuals_us = DS_counterfactuals_weighted_us.to_dataframe().melt(ignore_index = False).dropna()
years_counterfactuals_br = DS_counterfactuals_weighted_br.to_dataframe().melt(ignore_index = False).dropna()
years_counterfactuals_arg = DS_counterfactuals_weighted_arg.to_dataframe().melt(ignore_index = False).dropna()
years_counterfactuals_am['region'] = 'AM'
years_counterfactuals_us['region'] = 'US'
years_counterfactuals_br['region'] = 'Brazil'
years_counterfactuals_arg['region'] = 'Argentina'
years_countefactuals_merged = pd.concat([years_counterfactuals_am,years_counterfactuals_us,years_counterfactuals_br,years_counterfactuals_arg ], axis =0)
years_countefactuals_merged['value'] = 1

def co_occurrences_analogues(feature, ax, title = ''):
    # Create figure with co occurrence of analogues for each region and RCP.

    domain = years_countefactuals_merged.where( (years_countefactuals_merged['variable'] == feature)).dropna()
    domain = domain.sort_values(by = 'time')
    domain = domain.pivot( columns='region', values = 'value')
    
    def domain_country_locator(country):
        if country in domain.columns:
            domain_country = domain.where(domain[country] > 0, 0)[country]
        else:
            domain_country = 0
            print(f'no {country} for this RCP:', feature)         
        return domain_country
    
    # Establish the local domains
    domain_us = domain_country_locator('US')
    domain_br = domain_country_locator('Brazil')
    domain_arg = domain_country_locator('Argentina')
    domain_am = domain_country_locator('AM')

    # Figure structure
    ax = ax
    ax.bar(domain.index, domain_us, label = 'US') #, bottom = domain.where(domain['region'] == 'AM')['value'])
    ax.bar(domain.index, domain_br, bottom = domain_us, label = 'Brazil')
    ax.bar(domain.index, domain_arg, bottom = domain_us, label = 'Argentina')
    ax.bar(domain.index, domain_am, bottom = domain_us + domain_arg + domain_br, label = 'AM', hatch="//" ) 
    ax.set_ylim(0,3)
    ax.set_title(f"{title} {feature}", loc='left' )

    return ax

# =============================================================================
# Fig A2: Co-occurrence of 2012 analogues atb oth local and simulatenous levels
# =============================================================================
fig, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(nrows=6, figsize=(8,12))
fig_gfdl_26 = co_occurrences_analogues('GFDL-esm4_1-2.6', ax = ax1, title = 'a)')
fig_gfdl_85 = co_occurrences_analogues('GFDL-esm4_5-8.5', ax = ax2, title = 'b)')
fig_ipsl_26 = co_occurrences_analogues('IPSL-cm6a-lr_1-2.6', ax = ax3, title = 'c)')
fig_ipsl_85 = co_occurrences_analogues('IPSL-cm6a-lr_5-8.5', ax = ax4, title = 'd)')
fig_ukesm_26 = co_occurrences_analogues('UKESM1-0-ll_1-2.6', ax = ax5, title = 'e)')
fig_ukesm_85 = co_occurrences_analogues('UKESM1-0-ll_5-8.5', ax = ax6, title = 'f)')
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc=[0.1,-0.001], ncol=4, frameon=False )
plt.tight_layout()
plt.savefig('paper_figures/co_occurrence.png', format='png', dpi=500)
plt.show()

#%%

def figure_timeseries_per_country(DS_timeseries):
    DS_hybrid_trend_us_weighted = weighted_conversion(DS_timeseries, DS_area = DS_mirca_us)
    DS_hybrid_trend_br_weighted = weighted_conversion(DS_timeseries, DS_area = DS_mirca_br)
    DS_hybrid_trend_arg_weighted = weighted_conversion(DS_timeseries, DS_area = DS_mirca_arg)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 8), sharex=True, sharey=True) # marker='o'marker='^',marker='s',
    ax1.axhline(y = DS_historical_hybrid_us_weight['Yield'].sel(time = 2012).values, linestyle = 'dashed', color = 'k', label = '2012 event', linewidth = 2 )
    DS_hybrid_trend_us_weighted['GFDL-esm4_1-2.6'].plot(color='tab:blue',marker='o', ax = ax1, linewidth = 2 )
    DS_hybrid_trend_us_weighted['GFDL-esm4_5-8.5'].plot( color='tab:orange',marker='o', ax = ax1, linewidth = 2 )
    DS_hybrid_trend_us_weighted['IPSL-cm6a-lr_1-2.6'].plot( color='tab:blue', marker='^',ax = ax1, linewidth = 2 )
    DS_hybrid_trend_us_weighted['IPSL-cm6a-lr_5-8.5'].plot( marker='^', color='tab:orange', ax = ax1, linewidth = 2 )
    DS_hybrid_trend_us_weighted['UKESM1-0-ll_1-2.6'].plot( color='tab:blue',marker='s', ax = ax1, linewidth = 2 )
    DS_hybrid_trend_us_weighted['UKESM1-0-ll_5-8.5'].plot( marker='s', color='tab:orange', ax = ax1, linewidth = 2 )
    ax1.set_title("a) US")
    ax1.set_ylabel('Production (Mt)')
    
    # ax1.legend()
    ax2.axhline(y = DS_historical_hybrid_br_weight['Yield'].sel(time = 2012).values, linestyle = 'dashed', color = 'k', label = '2012 event', linewidth = 2 )
    DS_hybrid_trend_br_weighted['GFDL-esm4_1-2.6'].plot(color='tab:blue',marker='o', ax = ax2, linewidth = 2 )
    DS_hybrid_trend_br_weighted['GFDL-esm4_5-8.5'].plot( color='tab:orange',marker='o', ax = ax2, linewidth = 2 )
    DS_hybrid_trend_br_weighted['IPSL-cm6a-lr_1-2.6'].plot( color='tab:blue', marker='^',ax = ax2, linewidth = 2 )
    DS_hybrid_trend_br_weighted['IPSL-cm6a-lr_5-8.5'].plot( marker='^', color='tab:orange', ax = ax2, linewidth = 2 )
    DS_hybrid_trend_br_weighted['UKESM1-0-ll_1-2.6'].plot( color='tab:blue',marker='s', ax = ax2, linewidth = 2 )
    DS_hybrid_trend_br_weighted['UKESM1-0-ll_5-8.5'].plot( marker='s', color='tab:orange', ax = ax2, linewidth = 2 )
    ax2.set_title("b) Brazil")
    
    lines = ax2.get_lines()
    legend1 = ax2.legend([dummy_lines[i] for i in range(0,6)], ["2012 event", "SSP1-2.6", "SSP5-8.5","GFDL-esm4", "IPSL-cm6a-lr", "UKESM1-0-ll"], frameon = False,loc = 3, ncol=2)
    # legend2 = ax2.legend([dummy_lines2[i] for i in [0,1,2]], ["GFDL-esm4", "IPSL-cm6a-lr", "UKESM1-0-ll"], frameon = False,loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    ax2.set_ylabel('')
    
    ax3.axhline(y = DS_historical_hybrid_arg_weight['Yield'].sel(time = 2012).values, linestyle = 'dashed', color = 'k', label = '2012 event', linewidth = 2 )
    DS_hybrid_trend_arg_weighted['GFDL-esm4_1-2.6'].plot(color='tab:blue',marker='o', ax = ax3, linewidth = 2 )
    DS_hybrid_trend_arg_weighted['GFDL-esm4_5-8.5'].plot( color='tab:orange',marker='o', ax = ax3, linewidth = 2 )
    DS_hybrid_trend_arg_weighted['IPSL-cm6a-lr_1-2.6'].plot( color='tab:blue', marker='^',ax = ax3, linewidth = 2 )
    DS_hybrid_trend_arg_weighted['IPSL-cm6a-lr_5-8.5'].plot( marker='^', color='tab:orange', ax = ax3, linewidth = 2 )
    DS_hybrid_trend_arg_weighted['UKESM1-0-ll_1-2.6'].plot( color='tab:blue',marker='s', ax = ax3, linewidth = 2 )
    DS_hybrid_trend_arg_weighted['UKESM1-0-ll_5-8.5'].plot( marker='s', color='tab:orange', ax = ax3, linewidth = 2 )
    ax3.set_title("a) Argentina")
    ax3.set_ylabel('')
    plt.tight_layout()
    plt.show()

figure_timeseries_per_country(DS_hybrid_trend_all)
figure_timeseries_per_country(DS_hybrid_all)



#%% Yield - Climate interaction
# RESEARCH QUESTION: What are the climatic conditions leading to the failures? 

import matplotlib as mpl
backend = mpl.get_backend()
mpl.use('agg')

# =============================================================================
# # Comparing the weather conditions between historic times, 2012 and counterfactuals for each climatic variables and for each country
# =============================================================================

def clim_conditions_analogues_country(DS_area_hist, DS_area_fut, DS_counterfactuals_weighted_country, country, option = '2012', plot_legend = False, plot_yaxis = False):
    # Show letter per country
    letters_to_country = {'US':'a) US', 'Brazil':'b) Brazil', 'Argentina':'c) Argentina'}
    
    # Conversion of historical series to weighted timeseries    
    DS_conditions_hist_weighted_country = weighted_conversion(DS_conditions_hist, DS_area = DS_area_hist)
    DS_conditions_2012_weighted_country = DS_conditions_hist_weighted_country.sel(time=2012)
    # DS to df
    df_clim_hist_weighted = DS_conditions_hist_weighted_country.to_dataframe()
    df_clim_hist_weighted['scenario'] = 'Climatology'
    df_clim_hist_weighted['model_used'] = 'Climatology'
    
    # Conversion of future series to weighted timeseries    
    df_clim_counter_ukesm_85 = convert_clim_weighted_ensemble(df_hybrid_fut_ukesm_585_am, DS_counterfactuals_weighted_country, 'UKESM1-0-ll_5-8.5', DS_area_fut)    
    df_clim_counter_ukesm_26 = convert_clim_weighted_ensemble(df_hybrid_fut_ukesm_126_am, DS_counterfactuals_weighted_country, 'UKESM1-0-ll_1-2.6', DS_area_fut)    
    df_clim_counter_gfdl_85 = convert_clim_weighted_ensemble(df_hybrid_fut_gfdl_585_am, DS_counterfactuals_weighted_country, 'GFDL-esm4_5-8.5', DS_area_fut)    
    df_clim_counter_gfdl_26 = convert_clim_weighted_ensemble(df_hybrid_fut_gfdl_126_am, DS_counterfactuals_weighted_country, 'GFDL-esm4_1-2.6', DS_area_fut)    
    df_clim_counter_ipsl_85 = convert_clim_weighted_ensemble(df_hybrid_fut_ipsl_585_am, DS_counterfactuals_weighted_country, 'IPSL-cm6a-lr_5-8.5', DS_area_fut)    
    df_clim_counter_ipsl_26 = convert_clim_weighted_ensemble(df_hybrid_fut_ipsl_126_am, DS_counterfactuals_weighted_country, 'IPSL-cm6a-lr_1-2.6', DS_area_fut)    
    
    # Merge dataframes with different names
    df_clim_counterfactuals_weighted_all_country = pd.concat([df_clim_hist_weighted, df_clim_counter_ukesm_85, df_clim_counter_ukesm_26, 
                                                      df_clim_counter_gfdl_85, df_clim_counter_gfdl_26,
                                                      df_clim_counter_ipsl_85, df_clim_counter_ipsl_26])
    
# =============================================================================
#     # Plot boxplots comparing the historical events, the 2012 event and the counterfactuals
# =============================================================================
    names = df_clim_counterfactuals_weighted_all_country.columns.drop(['scenario', 'model_used'])
    fig, axes  = plt.subplots(2,int(np.ceil(len(df_clim_counterfactuals_weighted_all_country.columns)/3)), figsize=(5, 6), dpi=300, gridspec_kw=dict(height_ratios=[1,1]))
    
    for name, ax in zip(names, axes.flatten()):
        df_merge_subset = df_clim_counterfactuals_weighted_all_country[df_clim_counterfactuals_weighted_all_country.index != 2012].loc[:,[name,'scenario']]
        df_merge_subset['variable'] = name
        
        if option == '2012':
            g1 = sns.boxplot(y=name, x = 'variable', hue = 'scenario', data=df_merge_subset.where(df_merge_subset['scenario'] == 'Analogues').dropna(), orient='v', ax=ax)
            # g1 = sns.scatterplot(y=name, x = 'variable', data=df_merge_subset.where(df_merge_subset['scenario'] == 'Analogues' ), ax=ax, color = 'orange', s=60, label = 'Analogues', zorder = 20)
            ax.axhline( y = DS_conditions_2012_weighted_country[name].mean(), color = 'red', linestyle = 'dashed',linewidth =2, label = '2012 event', zorder = 19)
        elif option == 'climatology':
            g1 = sns.boxplot(y=name, x = 'variable', hue = 'scenario', data=df_merge_subset, orient='v', ax=ax)
        elif option == 'historical':
            g1 = sns.boxplot(y=name, x = 'variable', hue = 'scenario', data=df_merge_subset.where(df_merge_subset['scenario'] == 'Climatology').dropna(), orient='v', ax=ax)
            ax.axhline( y = DS_conditions_2012_weighted_country[name].mean(), color = 'red', linestyle = 'dashed',linewidth =2, label = '2012 event', zorder = 19)
        
        g1.set(xticklabels=[])  # remove the tick labels
        g1.set(xlabel= name)
        if plot_yaxis == True:
            if name in names[0:1]:
                g1.set(ylabel= 'Precipitation (mm/month)')  # remove the axis label  
            elif name in names[3:4]:
                g1.set(ylabel='Temperature (°C)' )  # remove the axis label   
            else:
                g1.set(ylabel='' )
        else:
            g1.set(ylabel='' )
            
        ax.get_legend().remove()
        g1.tick_params(bottom=False)  # remove the ticks
    
    if plot_legend == True:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc=[0.2,-0.01], ncol=2, frameon=False)
        
    plt.suptitle(f'{letters_to_country[country]}', x=0.15, y=.97)
    plt.tight_layout()
    # plt.show()
    return fig

def edit_figs(fig):
    c1 = fig.canvas
    c1.draw()
    a1 = np.array(c1.buffer_rgba())
    return a1

# =============================================================================
# Fig 5: climatic conditions of the country-level analogues with respect to the 2012 event
# =============================================================================
fig_us_test = clim_conditions_analogues_country(DS_mirca_us_hist, DS_mirca_us, DS_counterfactuals_weighted_us, country = 'US', plot_legend= False, plot_yaxis = True)
fig_br_test = clim_conditions_analogues_country(DS_mirca_br_hist, DS_mirca_br, DS_counterfactuals_weighted_br, country = 'Brazil', plot_legend= True)
fig_arg_test = clim_conditions_analogues_country(DS_mirca_arg_hist, DS_mirca_arg, DS_counterfactuals_weighted_arg, country = 'Argentina', plot_legend= False)  

fig_us_draw = edit_figs(fig_us_test)
fig_br_draw = edit_figs(fig_br_test)
fig_arg_draw = edit_figs(fig_arg_test)

a = np.hstack((fig_us_draw,fig_br_draw,fig_arg_draw))

mpl.use(backend)
fig,ax = plt.subplots(figsize=(15, 6), dpi=200)
fig.subplots_adjust(0, 0, 1, 1)
ax.set_axis_off()
plt.tight_layout()
plt.draw()
ax.matshow(a)
fig.savefig('paper_figures/clim_conditions_countries_2012.png', format='png', dpi=500)

# =============================================================================
# Fig SI:5: climatic conditions of the country-level analogues with respect to climatology
# =============================================================================
fig_us_sup = clim_conditions_analogues_country(DS_mirca_us_hist, DS_mirca_us, DS_counterfactuals_weighted_us, country = 'US', option = 'climatology', plot_yaxis = True)
fig_br_sup = clim_conditions_analogues_country(DS_mirca_br_hist, DS_mirca_br, DS_counterfactuals_weighted_br, country = 'Brazil', option = 'climatology', plot_legend= True)
fig_arg_sup = clim_conditions_analogues_country(DS_mirca_arg_hist, DS_mirca_arg, DS_counterfactuals_weighted_arg, country = 'Argentina', option = 'climatology')  

fig_us_sup_draw = edit_figs(fig_us_sup)
fig_br_sup_draw = edit_figs(fig_br_sup)
fig_arg_sup_draw = edit_figs(fig_arg_sup)

a_sup = np.hstack((fig_us_sup_draw,fig_br_sup_draw,fig_arg_sup_draw))

fig_sup, ax = plt.subplots(figsize=(15, 6), dpi=200)
fig_sup.subplots_adjust(0, 0, 1, 1)
ax.set_axis_off()
plt.tight_layout()
plt.draw()
ax.matshow(a_sup)
fig_sup.savefig('paper_figures/fig_sup_clim_conditions_countries_climatology.png', format='png', dpi=500)


# =============================================================================
# Fig SI:5: climatic conditions of the country-level analogues with respect to climatology
# =============================================================================
fig_us_sup = clim_conditions_analogues_country(DS_mirca_us_hist, DS_mirca_us, DS_counterfactuals_weighted_us, country = 'US', option = 'historical', plot_yaxis = True)
fig_br_sup = clim_conditions_analogues_country(DS_mirca_br_hist, DS_mirca_br, DS_counterfactuals_weighted_br, country = 'Brazil', option = 'historical', plot_legend= True)
fig_arg_sup = clim_conditions_analogues_country(DS_mirca_arg_hist, DS_mirca_arg, DS_counterfactuals_weighted_arg, country = 'Argentina', option = 'historical')  

fig_us_sup_draw = edit_figs(fig_us_sup)
fig_br_sup_draw = edit_figs(fig_br_sup)
fig_arg_sup_draw = edit_figs(fig_arg_sup)

a_sup = np.hstack((fig_us_sup_draw,fig_br_sup_draw,fig_arg_sup_draw))

fig_sup, ax = plt.subplots(figsize=(15, 6), dpi=200)
fig_sup.subplots_adjust(0, 0, 1, 1)
ax.set_axis_off()
plt.tight_layout()
plt.draw()
ax.matshow(a_sup)
# fig_sup.savefig('paper_figures/fig_sup_clim_conditions_countries_climatology_past.png', format='png', dpi=500)

#%%


#####

# def counterfactuals_country_level(DS_yields, DS_mirca_country, DS_historical_hybrid_country, local_factual):
#     DS_projections_country = DS_yields.where(DS_mirca_country['harvest_area'].mean() > 0)
    
#     DS_historical_hybrid_country_weight = weighted_conversion(DS_historical_hybrid_country, DS_area = DS_mirca_country)
#     yield_factual_2012 = local_factual
#     print(DS_historical_hybrid_country_weight.sel(time = 2012).Yield.values)
#     print(local_factual)
    
#     DS_projections_country_weighted = weighted_conversion(DS_projections_country, DS_area = DS_mirca_country.mean('time'))
#     DS_projections_country_weighted = DS_projections_country_weighted.where(DS_projections_country_weighted > 0).dropna('time', how = 'all')
    
#     DS_counterfactuals_weighted_country = DS_projections_country_weighted.where(DS_projections_country_weighted <= yield_factual_2012)
#     print("Number of impact analogues per scenario for the country:", (DS_counterfactuals_weighted_country > -10).sum())
    
#     # Check years of counterfactuals
#     list_counterfactuals_scenarios = []
#     for feature in list(DS_counterfactuals_weighted_country.keys()):
#         feature_counterfactuals = DS_counterfactuals_weighted_country[feature].dropna(dim = 'time')
#         print(feature_counterfactuals.time.values)
#         list_counterfactuals_scenarios.append(feature_counterfactuals.time.values)
#         number_counter = len(np.hstack(list_counterfactuals_scenarios)) 
#     print('total counterfactuals: ', number_counter)
        
#     # Create dataset with spatailly explict counterfactuals only 
#     DS_counterfactuals_spatial_country = DS_projections_country.where(DS_counterfactuals_weighted_country > -10)

#     # Find the counterfactual shocks using a baseline as reference, either historical yields or the factual as reference
#     DS_hybrid_counterfactuals_spatial_shock_country = DS_counterfactuals_spatial_country.dropna('time', how='all') - DS_historical_hybrid_country['Yield'].mean('time')
#     # DS_hybrid_counterfactuals_spatial_shock_country_2012 = DS_counterfactuals_spatial_country.dropna('time', how='all') - DS_historical_hybrid_country['Yield'].sel(time = 2012)

#     # =============================================================================
#     # # Plots the counterfactuals per scenario 
#     # =============================================================================
#     # for feature in list(DS_hybrid_counterfactuals_spatial_shock_country.keys()):
#     #     plot_2d_am_multi(DS_hybrid_counterfactuals_spatial_shock_country[feature].sel(time = DS_counterfactuals_weighted_country[feature].time.where(DS_counterfactuals_weighted_country[feature] > -10).dropna(dim = 'time')), map_title = feature )
    
#     # for feature in list(DS_hybrid_counterfactuals_spatial_shock_country_2012.keys()):
#     #     plot_2d_am_multi(DS_hybrid_counterfactuals_spatial_shock_country_2012[feature].sel(time = DS_counterfactuals_weighted_country[feature].time.where(DS_counterfactuals_weighted_country[feature] > -10).dropna(dim = 'time')), map_title = feature )
#     return DS_counterfactuals_weighted_country, number_counter
