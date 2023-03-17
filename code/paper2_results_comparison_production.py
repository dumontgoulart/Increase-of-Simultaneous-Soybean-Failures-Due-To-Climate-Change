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
    
#%% Load historical case
start_hist_date = 2000
end_hist_date = 2015

DS_historical_hybrid = xr.load_dataset("output_models_am/hybrid_epic-iiasa_gswp3-w5e5_obsclim_2015soc_default_yield-soy-noirr_global_annual_1901_2016.nc").sel(time = slice (start_hist_date, end_hist_date))
# DS_historical_hybrid = xr.load_dataset("output_models_am/hybrid_epic-iiasa_gswp3-w5e5_obsclim_2015soc_default_yield-soy-noirr_2012_mask.nc")
DS_historical_hybrid = rearrange_latlot(DS_historical_hybrid)

### Load future hybrid runs - detrended:
DS_hybrid_gfdl_26 = xr.load_dataset("output_models_am/hybrid_gfdl-esm4_ssp126_default_yield_soybean_2015_2100.nc")
DS_hybrid_gfdl_85 = xr.load_dataset("output_models_am/hybrid_gfdl-esm4_ssp585_default_yield_soybean_2015_2100.nc")
DS_hybrid_ipsl_26 = xr.load_dataset("output_models_am/hybrid_ipsl-cm6a-lr_ssp126_default_yield_soybean_2015_2100.nc")
DS_hybrid_ipsl_85 = xr.load_dataset("output_models_am/hybrid_ipsl-cm6a-lr_ssp585_default_yield_soybean_2015_2100.nc")
DS_hybrid_ukesm_26 = xr.load_dataset("output_models_am/hybrid_ukesm1-0-ll_ssp126_default_yield_soybean_2015_2100.nc")
DS_hybrid_ukesm_85 = xr.load_dataset("output_models_am/hybrid_ukesm1-0-ll_ssp585_default_yield_soybean_2015_2100.nc")

#### Load future hybrid runs - trends:
DS_hybrid_trend_gfdl_26 = xr.load_dataset("output_models_am/hybrid_trends/hybrid_trend_gfdl-esm4_ssp126_default_yield_soybean_2015_2100.nc")
DS_hybrid_trend_gfdl_85 = xr.load_dataset("output_models_am/hybrid_trends/hybrid_trend_gfdl-esm4_ssp585_default_yield_soybean_2015_2100.nc")
DS_hybrid_trend_ipsl_26 = xr.load_dataset("output_models_am/hybrid_trends/hybrid_trend_ipsl-cm6a-lr_ssp126_default_yield_soybean_2015_2100.nc")
DS_hybrid_trend_ipsl_85 = xr.load_dataset("output_models_am/hybrid_trends/hybrid_trend_ipsl-cm6a-lr_ssp585_default_yield_soybean_2015_2100.nc")
DS_hybrid_trend_ukesm_26 = xr.load_dataset("output_models_am/hybrid_trends/hybrid_trend_ukesm1-0-ll_ssp126_default_yield_soybean_2015_2100.nc")
DS_hybrid_trend_ukesm_85 = xr.load_dataset("output_models_am/hybrid_trends/hybrid_trend_ukesm1-0-ll_ssp585_default_yield_soybean_2015_2100.nc")

# Merge all scenarios
DS_hybrid_all = xr.merge([DS_hybrid_gfdl_26.rename({'yield-soy-noirr':'GFDL-esm4_1-2.6'}),DS_hybrid_gfdl_85.rename({'yield-soy-noirr':'GFDL-esm4_5-8.5'}),
                          DS_hybrid_ipsl_26.rename({'yield-soy-noirr':'IPSL-cm6a-lr_1-2.6'}),DS_hybrid_ipsl_85.rename({'yield-soy-noirr':'IPSL-cm6a-lr_5-8.5'}),
                          DS_hybrid_ukesm_26.rename({'yield-soy-noirr':'UKESM1-0-ll_1-2.6'}),DS_hybrid_ukesm_85.rename({'yield-soy-noirr':'UKESM1-0-ll_5-8.5'})])


DS_hybrid_trend_all = xr.merge([DS_hybrid_trend_gfdl_26.rename({'yield-soy-noirr':'GFDL-esm4_1-2.6'}),DS_hybrid_trend_gfdl_85.rename({'yield-soy-noirr':'GFDL-esm4_5-8.5'}),
                          DS_hybrid_trend_ipsl_26.rename({'yield-soy-noirr':'IPSL-cm6a-lr_1-2.6'}),DS_hybrid_trend_ipsl_85.rename({'yield-soy-noirr':'IPSL-cm6a-lr_5-8.5'}),
                          DS_hybrid_trend_ukesm_26.rename({'yield-soy-noirr':'UKESM1-0-ll_1-2.6'}),DS_hybrid_trend_ukesm_85.rename({'yield-soy-noirr':'UKESM1-0-ll_5-8.5'})])

### Use MIRCA to isolate the rainfed 90% soybeans
# DS_mirca = xr.open_dataset("../../paper_hybrid_agri/data/americas_mask_ha.nc", decode_times=False).rename({'latitude': 'lat', 'longitude': 'lon','annual_area_harvested_rfc_crop08_ha_30mn':'harvest_area'})
DS_mirca = xr.open_dataset("../../paper_hybrid_agri/data/soy_harvest_spam_native_05x05.nc", decode_times=False)

#### HARVEST DATA
DS_harvest_area_sim = xr.load_dataset("../../paper_hybrid_agri/data/soybean_harvest_area_calculated_americas_hg.nc", decode_times=False).sel(time = slice(start_hist_date,end_hist_date))

#%% WIEGHTED ANALYSIS
# process harvest area
DS_harvest_area_sim = DS_harvest_area_sim.sel(time = 2012) 
DS_harvest_area_sim = DS_harvest_area_sim.where(DS_mirca['harvest_area'] > 0 )

# Historical, change with year
DS_harvest_area_hist = DS_harvest_area_sim.where(DS_historical_hybrid['Yield']> -5)
DS_harvest_area_hist = rearrange_latlot(DS_harvest_area_hist)

# Future, it works as the constant area throught the 21st century based on 2014/15/16
DS_harvest_area_fut = DS_harvest_area_hist.sel(time = 2012)
DS_harvest_area_fut = rearrange_latlot(DS_harvest_area_fut)

# Test plots to check for problems
plot_2d_am_map(DS_harvest_area_hist['harvest_area'].isel(time = 0))
plot_2d_am_map(DS_harvest_area_hist['harvest_area'].sel(time = 2012))
plot_2d_am_map(DS_harvest_area_fut['harvest_area'], title = 'Future projections')

# Mean and standard deviation of each projections at grid level:
print(f'Mean historical:{round(DS_historical_hybrid.to_dataframe().mean().values.item(),2)} and Std: {round(DS_historical_hybrid.to_dataframe().std().values.item(),2)}')
print(f'Mean:{round(DS_hybrid_gfdl_85.to_dataframe().mean().values.item(),2)} and Std: {round(DS_hybrid_gfdl_85.to_dataframe().std().values.item(),2)}')
print(f'Mean:{round(DS_hybrid_ipsl_85.to_dataframe().mean().values.item(),2)} and Std: {round(DS_hybrid_ipsl_85.to_dataframe().std().values.item(),2)}')
print(f'Mean:{round(DS_hybrid_ukesm_85.to_dataframe().mean().values.item(),2)} and Std: {round(DS_hybrid_ukesm_85.to_dataframe().std().values.item(),2)}')

# Weighted comparison for each model - degree of explanation
DS_historical_hybrid_weighted = weighted_prod_conversion(DS_historical_hybrid['Yield'], DS_area = DS_harvest_area_hist)

# Future projections and transform into weighted timeseries
DS_hybrid_gfdl_26_weighted = weighted_prod_conversion(DS_hybrid_gfdl_26['yield-soy-noirr'], DS_area = DS_harvest_area_fut)
DS_hybrid_gfdl_85_weighted = weighted_prod_conversion(DS_hybrid_gfdl_85['yield-soy-noirr'], DS_area = DS_harvest_area_fut)
DS_hybrid_ipsl_26_weighted = weighted_prod_conversion(DS_hybrid_ipsl_26['yield-soy-noirr'], DS_area = DS_harvest_area_fut)
DS_hybrid_ipsl_85_weighted = weighted_prod_conversion(DS_hybrid_ipsl_85['yield-soy-noirr'], DS_area = DS_harvest_area_fut)
DS_hybrid_ukesm_26_weighted = weighted_prod_conversion(DS_hybrid_ukesm_26['yield-soy-noirr'], DS_area = DS_harvest_area_fut)
DS_hybrid_ukesm_85_weighted = weighted_prod_conversion(DS_hybrid_ukesm_85['yield-soy-noirr'], DS_area = DS_harvest_area_fut)

plt.figure(figsize=(8,5), dpi=300) #plot clusters
DS_historical_hybrid_weighted['Yield'].plot(label = 'history')
DS_hybrid_gfdl_85_weighted['Yield'].plot(label = 'GFDL_5-8.5')
DS_hybrid_gfdl_26_weighted['Yield'].plot(label = 'GFDLl_1-2.6')
DS_hybrid_ukesm_85_weighted['Yield'].plot(label = 'UKESM1-0-ll_5-8.5')
DS_hybrid_ukesm_26_weighted['Yield'].plot(label = 'UKESM1-0-ll_1-2.6')
plt.legend()
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
factual_event_value = DS_historical_hybrid_weighted['Yield'].sel(time = 2012).values

DS_hybrid_all_weighted = weighted_prod_conversion(DS_hybrid_all, DS_area = DS_harvest_area_fut)
df_hybrid_weighted_melt = pd.melt(DS_hybrid_all_weighted.to_dataframe(),ignore_index= False )
df_hybrid_weighted_melt_counterfactuals = df_hybrid_weighted_melt.where(df_hybrid_weighted_melt['value'] <= factual_event_value )

df_hybrid_weighted_melt_counterfactuals_split = df_hybrid_weighted_melt_counterfactuals[df_hybrid_weighted_melt_counterfactuals['value'] > - 10].copy()
df_hybrid_weighted_melt_counterfactuals_split['SSP'] = df_hybrid_weighted_melt_counterfactuals_split.variable.str.split('_').str[-1]


# put the scenarios all together
DS_hybrid_trend_all_weighted = weighted_prod_conversion(DS_hybrid_trend_all, DS_area = DS_harvest_area_fut)
df_hybrid_trend_weighted_melt = pd.melt(DS_hybrid_trend_all_weighted.to_dataframe(),ignore_index= False )
df_hybrid_trend_weighted_melt_counterfactuals = df_hybrid_trend_weighted_melt.where(df_hybrid_trend_weighted_melt['value'] <= factual_event_value )

df_hybrid_trend_weighted_melt_counterfactuals_split = df_hybrid_trend_weighted_melt_counterfactuals[df_hybrid_trend_weighted_melt_counterfactuals['value'] > - 10].copy()
df_hybrid_trend_weighted_melt_counterfactuals_split['SSP'] = df_hybrid_trend_weighted_melt_counterfactuals_split.variable.str.split('_').str[-1]

print("Number of impact analogues:", (df_hybrid_weighted_melt_counterfactuals['value'] > -10).sum(),
      'An average per scenario of',(df_hybrid_weighted_melt_counterfactuals['value'] > -10).sum()/ (len(df_hybrid_weighted_melt_counterfactuals.variable.unique())-1) )

years_counterfactuals = df_hybrid_weighted_melt_counterfactuals[df_hybrid_weighted_melt_counterfactuals['value'] > -10]


#%% Figure 2 - timeseries with factual baseline
# =============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True) # marker='o'marker='^',marker='s',
ax1.axhline(y = factual_event_value, linestyle = 'dashed', color = 'k', label = '2012 event', linewidth = 2 )
DS_hybrid_trend_all_weighted['GFDL-esm4_1-2.6'].plot(color='tab:blue',marker='o', ax = ax1, linewidth = 2 )
DS_hybrid_trend_all_weighted['GFDL-esm4_5-8.5'].plot( color='tab:orange',marker='o', ax = ax1, linewidth = 2 )
DS_hybrid_trend_all_weighted['IPSL-cm6a-lr_1-2.6'].plot( color='tab:blue', marker='^',ax = ax1, linewidth = 2 )
DS_hybrid_trend_all_weighted['IPSL-cm6a-lr_5-8.5'].plot( marker='^', color='tab:orange', ax = ax1, linewidth = 2 )
DS_hybrid_trend_all_weighted['UKESM1-0-ll_1-2.6'].plot( color='tab:blue',marker='s', ax = ax1, linewidth = 2 )
DS_hybrid_trend_all_weighted['UKESM1-0-ll_5-8.5'].plot( marker='s', color='tab:orange', ax = ax1, linewidth = 2 )
ax1.set_title("a) Soybean timeseries in no-adaptation scenario")
# ax1.legend()

# CUstomise legends
dummy_lines = []
dummy_lines.append(ax1.plot([],[], color="black", ls = 'dashed' )[0])
dummy_lines.append(ax1.plot([],[], color="tab:blue", ls = 'solid' )[0])
dummy_lines.append(ax1.plot([],[], color="tab:orange", ls = 'solid' )[0])
dummy_lines.append(ax1.plot([],[], color="black", ls = 'solid',marker='o' )[0])
dummy_lines.append(ax1.plot([],[], color="black", ls = 'solid', marker='^')[0])
dummy_lines.append(ax1.plot([],[], color="black", ls = 'solid', marker='s' )[0])

lines = ax1.get_lines()
legend1 = ax1.legend([dummy_lines[i] for i in range(0,6)], ["2012 event", "SSP1-2.6", "SSP5-8.5","GFDL-esm4", "IPSL-cm6a-lr", "UKESM1-0-ll"], frameon = False,loc = 3, ncol=2)
# legend2 = ax2.legend([dummy_lines2[i] for i in [0,1,2]], ["GFDL-esm4", "IPSL-cm6a-lr", "UKESM1-0-ll"], frameon = False,loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
ax1.set_ylabel('')

ax2.axhline(y = factual_event_value, linestyle = 'dashed',color = 'k', label = '2012 event', linewidth = 2 )
DS_hybrid_gfdl_26_weighted['Yield'].plot(color='tab:blue',marker='o', ax = ax2, linewidth = 2 )
DS_hybrid_gfdl_85_weighted['Yield'].plot(marker='o', color='tab:orange', ax = ax2, linewidth = 2 )
DS_hybrid_ipsl_26_weighted['Yield'].plot(color='tab:blue',marker='^', ax = ax2, linewidth = 2 )
DS_hybrid_ipsl_85_weighted['Yield'].plot(color='tab:orange',marker='^', ax = ax2, linewidth = 2)
DS_hybrid_ukesm_26_weighted['Yield'].plot(color='tab:blue',marker='s', ax = ax2, linewidth = 2 )
DS_hybrid_ukesm_85_weighted['Yield'].plot(marker='s', color='tab:orange', ax = ax2, linewidth = 2  )
ax2.set_title("b) Soybean timeseries in adaptation scenario")

plt.ylabel('')
fig.supylabel('Production (Megatonne)')
plt.tight_layout()
plt.savefig('paper_figures_production/timeseries_projections_ab.png', format='png', dpi=300)
plt.show()


# =============================================================================
# # Fig A1 - supplementary figure with the frequency of analogues per RCP, and the magnitude of the analogues (measured as analogue - historical event)
# =============================================================================
length_index = len(df_hybrid_trend_weighted_melt_counterfactuals_split[df_hybrid_trend_weighted_melt_counterfactuals_split['SSP']=='5-8.5'])
magnitutde_counterfactuals = df_hybrid_trend_weighted_melt_counterfactuals_split['value'] - DS_historical_hybrid_weighted['Yield'].sel(time = 2012).values

fig,((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(12, 8), sharex=True)
 
sns.histplot( x = df_hybrid_trend_weighted_melt_counterfactuals_split[['SSP','value']]['SSP'].sort_values().values, ax =ax1, hue = df_hybrid_trend_weighted_melt_counterfactuals_split[['SSP','value']]['SSP'].sort_values().values)
ax1.set_title('a)', loc='left')
ax1.set_ylim(0,length_index+5)
ax1.legend([], frameon=False)

sns.histplot( x = df_hybrid_weighted_melt_counterfactuals_split[['SSP','value']]['SSP'].sort_values().values, ax=ax2, hue = df_hybrid_weighted_melt_counterfactuals_split[['SSP','value']]['SSP'].sort_values().values)
ax2.set_title('b)', loc='left')
ax2.set_ylim(0,length_index+5)
ax2.set_ylabel('')
ax2.set_yticklabels([])
ax2.legend([], frameon=False)

bar = sns.boxplot(data = df_hybrid_trend_weighted_melt_counterfactuals_split, y= df_hybrid_trend_weighted_melt_counterfactuals_split['value'] - DS_historical_hybrid_weighted['Yield'].sel(time = 2012).values , x= 'SSP', ax = ax3, order= ['1-2.6','5-8.5'])
ax3.set_ylabel('Production (Megatonne)')
ax3.set_title('c)', loc='left')
ax3.set_ylim(magnitutde_counterfactuals.min() - 5,magnitutde_counterfactuals.max())

bar = sns.boxplot(data = df_hybrid_weighted_melt_counterfactuals_split, y= df_hybrid_weighted_melt_counterfactuals_split['value'] - DS_historical_hybrid_weighted['Yield'].sel(time = 2012).values , x= 'SSP', ax = ax4, order= ['1-2.6','5-8.5'])
ax4.set_ylabel('')
ax4.set_title('d)', loc='left')
ax4.set_ylim(magnitutde_counterfactuals.min() - 5,magnitutde_counterfactuals.max())
ax4.set_yticklabels([])

plt.tight_layout()
plt.savefig('paper_figures_production/fig_sup_timeseries_adap.png', format='png', dpi=300)
plt.show()

diff_rcp85 = df_hybrid_weighted_melt_counterfactuals_split.where(df_hybrid_weighted_melt_counterfactuals_split['SSP'] == '5-8.5').mean().values- DS_historical_hybrid_weighted['Yield'].sel(time = 2012).values
diff_rcp26 = df_hybrid_weighted_melt_counterfactuals_split.where(df_hybrid_weighted_melt_counterfactuals_split['SSP'] == '1-2.6').mean().values- DS_historical_hybrid_weighted['Yield'].sel(time = 2012).values

diff_rcp85_trend = df_hybrid_trend_weighted_melt_counterfactuals_split.where(df_hybrid_trend_weighted_melt_counterfactuals_split['SSP'] == '5-8.5').mean().values- DS_historical_hybrid_weighted['Yield'].sel(time = 2012).values
diff_rcp26_trend = df_hybrid_trend_weighted_melt_counterfactuals_split.where(df_hybrid_trend_weighted_melt_counterfactuals_split['SSP'] == '1-2.6').mean().values- DS_historical_hybrid_weighted['Yield'].sel(time = 2012).values

print('Number of rcp26 analogues:', len(df_hybrid_weighted_melt_counterfactuals_split.where(df_hybrid_weighted_melt_counterfactuals_split['SSP'] == '1-2.6').dropna()))
print('Number of rcp85 analogues:', len(df_hybrid_weighted_melt_counterfactuals_split.where(df_hybrid_weighted_melt_counterfactuals_split['SSP'] == '5-8.5').dropna()))

print('Number of trend rcp26 analogues:', len(df_hybrid_trend_weighted_melt_counterfactuals_split.where(df_hybrid_trend_weighted_melt_counterfactuals_split['SSP'] == '1-2.6').dropna()))
print('Number of trend rcp85 analogues:', len(df_hybrid_trend_weighted_melt_counterfactuals_split.where(df_hybrid_trend_weighted_melt_counterfactuals_split['SSP'] == '5-8.5').dropna()))

print('Ratio of analogues caused by climate variability and by trends:', round(len(df_hybrid_weighted_melt_counterfactuals_split)/len(df_hybrid_trend_weighted_melt_counterfactuals_split),2) )

# magnitude differences
print('Difference trends mean rcp85 losses to the 2012 event is:', diff_rcp85_trend / (DS_historical_hybrid_weighted['Yield'].sel(time = 2012).values) )
print('Difference trends mean rcp26 losses to the 2012 event is:', diff_rcp26_trend / (DS_historical_hybrid_weighted['Yield'].sel(time = 2012).values) )

print('Difference mean rcp85 losses to the 2012 event is:', diff_rcp85 / (DS_historical_hybrid_weighted['Yield'].sel(time = 2012).values) )
print('Difference mean rcp26 losses to the 2012 event is:', diff_rcp26 / (DS_historical_hybrid_weighted['Yield'].sel(time = 2012).values) )

#%% Comparison of factual and counterfactuals

# <> ADD timeseriees, pdfs and  STD of the hybrid model for country

# Identify the FACTUAL case - Minimum value with respect to the mean
yield_factual = DS_historical_hybrid_weighted.sel(time = 2012)['Yield'].values

factual_loss_2012_spatial = DS_historical_hybrid['Yield'].sel(time=2012) - DS_historical_hybrid['Yield'].sel(time = slice(start_hist_date,end_hist_date)).mean('time')

plot_2d_am_map(factual_loss_2012_spatial, colormap = 'RdBu', vmin = -1, vmax = 1, title = "b) 2012 prediction")

factual_loss_2012_weighted_production = yield_factual - DS_historical_hybrid_weighted.sel(time = slice(start_hist_date,end_hist_date)).mean('time')['Yield'].values

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

# =============================================================================
# Climatic analysis of counterfactuals
# =============================================================================
# Function converting the climate data for the counterfactuals into weighted timeseries 
def convert_clim_weighted_ensemble(df_clim, DS_counterfactuals_weighted_country, feature, DS_area, mode = 'production'):
    DS_clim = xr.Dataset.from_dataframe(df_clim)
    DS_clim = rearrange_latlot(DS_clim)
    # Countefactuals only
    DS_clim_counter = DS_clim.where(DS_counterfactuals_weighted_country[feature].time.where(DS_counterfactuals_weighted_country[feature] > -10).dropna(dim = 'time'))
   
    DS_clim_counter_weight_country = weighted_prod_conversion(DS_clim_counter, DS_area = DS_area, mode = mode)
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
columns_to_be_used = ['prcptot_1', 'prcptot_2', 'prcptot_3','txm_1', 'txm_2', 'txm_3']
df_hybrid_fut_ukesm_585_am = pd.read_csv('output_models_am/climatic_projections/model_input_ukesm1-0-ll_ssp585_default_2015_2100.csv', index_col=[0,1,2],).loc[:,columns_to_be_used]
df_hybrid_fut_ukesm_126_am = pd.read_csv('output_models_am/climatic_projections/model_input_ukesm1-0-ll_ssp126_default_2015_2100.csv', index_col=[0,1,2],).loc[:,columns_to_be_used]
df_hybrid_fut_gfdl_585_am = pd.read_csv('output_models_am/climatic_projections/model_input_gfdl-esm4_ssp585_default_2015_2100.csv', index_col=[0,1,2],).loc[:,columns_to_be_used]
df_hybrid_fut_gfdl_126_am = pd.read_csv('output_models_am/climatic_projections/model_input_gfdl-esm4_ssp126_default_2015_2100.csv', index_col=[0,1,2],).loc[:,columns_to_be_used]
df_hybrid_fut_ipsl_585_am = pd.read_csv('output_models_am/climatic_projections/model_input_ipsl-cm6a-lr_ssp585_default_2015_2100.csv', index_col=[0,1,2],).loc[:,columns_to_be_used]
df_hybrid_fut_ipsl_126_am = pd.read_csv('output_models_am/climatic_projections/model_input_ipsl-cm6a-lr_ssp126_default_2015_2100.csv', index_col=[0,1,2],).loc[:,columns_to_be_used]

# Conversion of historical series to weighted timeseries    
DS_conditions_hist_weighted_am = weighted_prod_conversion(DS_conditions_hist, DS_area = DS_harvest_area_hist , mode = 'yield')
DS_conditions_2012_weighted_am = DS_conditions_hist_weighted_am.sel(time=2012)
# DS to df
df_clim_hist_weighted = DS_conditions_hist_weighted_am.to_dataframe()
df_clim_hist_weighted['scenario'] = 'Climatology'
df_clim_hist_weighted['model_used'] = 'Climatology'

# Conversion of future series to weighted timeseries    
df_clim_counter_ukesm_85 = convert_clim_weighted_ensemble(df_hybrid_fut_ukesm_585_am, DS_counterfactuals_weighted_am, 'UKESM1-0-ll_5-8.5', DS_harvest_area_fut, mode = 'yield')    
df_clim_counter_ukesm_26 = convert_clim_weighted_ensemble(df_hybrid_fut_ukesm_126_am, DS_counterfactuals_weighted_am, 'UKESM1-0-ll_1-2.6', DS_harvest_area_fut, mode = 'yield')    
df_clim_counter_gfdl_85 = convert_clim_weighted_ensemble(df_hybrid_fut_gfdl_585_am, DS_counterfactuals_weighted_am, 'GFDL-esm4_5-8.5', DS_harvest_area_fut, mode = 'yield')    
df_clim_counter_gfdl_26 = convert_clim_weighted_ensemble(df_hybrid_fut_gfdl_126_am, DS_counterfactuals_weighted_am, 'GFDL-esm4_1-2.6', DS_harvest_area_fut, mode = 'yield')    
df_clim_counter_ipsl_85 = convert_clim_weighted_ensemble(df_hybrid_fut_ipsl_585_am, DS_counterfactuals_weighted_am, 'IPSL-cm6a-lr_5-8.5', DS_harvest_area_fut, mode = 'yield')  
df_clim_counter_ipsl_26 = convert_clim_weighted_ensemble(df_hybrid_fut_ipsl_126_am, DS_counterfactuals_weighted_am, 'IPSL-cm6a-lr_1-2.6', DS_harvest_area_fut, mode = 'yield')    

# Merge dataframes with different names
df_clim_counterfactuals_weighted_all_am = pd.concat([df_clim_hist_weighted.loc[start_hist_date:end_hist_date], df_clim_counter_ukesm_85, df_clim_counter_ukesm_26, 
                                                  df_clim_counter_gfdl_85, df_clim_counter_gfdl_26,
                                                  df_clim_counter_ipsl_85, df_clim_counter_ipsl_26])


#%% Series of bboxplots to explain climatic conditions leading to analogues
# =============================================================================
# # Fig 3: Plot boxplots comparing the 2012 event and the analogues
# =============================================================================
names = df_clim_counterfactuals_weighted_all_am.columns.drop(['scenario', 'model_used'])
ncols = len(names)
fig, axes  = plt.subplots(2,int(np.ceil(len(df_clim_counterfactuals_weighted_all_am.columns)/3)), figsize=(8,10), dpi=300, sharey='row')

for name, ax in zip(names, axes.flatten()):
    df_merge_subset_am = df_clim_counterfactuals_weighted_all_am.loc[:,[name,'model_used','scenario']]
    df_merge_subset_am['variable'] = name
    df_merge_subset_am[name] = df_merge_subset_am[name] - df_clim_hist_weighted.loc[start_hist_date:end_hist_date][name].mean()
    axline_2012_ref = 0 # DS_conditions_2012_weighted_am[name].mean() - DS_conditions_hist_weighted_am[name].sel(time=slice(2000,end_hist_date)).mean()
    axline_2012_anomaly =  DS_conditions_2012_weighted_am[name].mean() - DS_conditions_hist_weighted_am[name].sel(time=slice(start_hist_date,end_hist_date)).mean()
    
    # ax.axhline( y = 0, color = 'black', zorder = 0, linestyle = 'dotted')
    ax.axhline( y = axline_2012_anomaly, color = 'firebrick', linestyle = 'dashed',linewidth =2, label = '2012 event', zorder = 19)
    # ax.axhspan(df_merge_subset_am[name].where(df_merge_subset_am['scenario'] == 'Climatology').dropna().quantile(0.05), df_merge_subset_am[name].where(df_merge_subset_am['scenario'] == 'Climatology').dropna().quantile(0.95), facecolor='0.2', alpha=0.3, zorder = 0, label = 'Climatology')
    g1 = sns.boxplot(y=name, x = 'variable', hue = 'scenario', data=df_merge_subset_am.where(df_merge_subset_am['scenario'] == 'Analogues').dropna(), palette = ['tab:orange'], orient='v', ax=ax)
    # g1 = sns.scatterplot(y=name, x = 'variable', data=df_merge_subset_am.where(df_merge_subset_am['scenario'] == 'Analogues' ), ax=ax, color = 'orange', s=60, label = 'Analogues', zorder = 20)
    g1.set(xticklabels=[])  # remove the tick labels
    g1.set(xlabel= name)
    if name in names[0:1]:
        g1.set(ylabel= 'Precipitation anomaly (mm)')  # remove the axis label  
    elif name in names[3:4]:
        g1.set(ylabel='Temperature anomaly (°C)' )  # remove the axis label 
    else:
        g1.set(ylabel='' ) 
    ax.get_legend().remove()
    g1.tick_params(bottom=False)  # remove the ticks
    # Change the visualization to put the 2012 event as central to the plot
    lower_boundary = axline_2012_ref - np.min([axline_2012_ref - df_merge_subset_am.where(df_merge_subset_am['scenario'] == 'Analogues').dropna()[name].min(), axline_2012_anomaly.values])
    higher_boundary =  np.max( [df_merge_subset_am.where(df_merge_subset_am['scenario'] == 'Analogues').dropna()[name].max(), axline_2012_anomaly.values] ) - axline_2012_ref
    buffer_zone = max(lower_boundary, higher_boundary)
    ax.set_ylim(axline_2012_ref - buffer_zone*1.3, axline_2012_ref + buffer_zone*1.3)
    ax.xaxis.set_label_position('top') 
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.8, 0.01), ncol=3, frameon=False)
# plt.suptitle('2012 analogues')
plt.tight_layout()
plt.savefig('paper_figures_production/clim_conditions_2012.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# Grouped boxplots, similar figure to the previous one
# =============================================================================
df_copy = df_clim_counterfactuals_weighted_all_am.dropna().copy()
df_copy[['prcptot_1','prcptot_2','prcptot_3','txm_1','txm_2','txm_3']] = df_copy[['prcptot_1','prcptot_2','prcptot_3','txm_1','txm_2','txm_3']] - df_copy.query('scenario == "Climatology"').mean()
df_copy['year'] = df_copy.index

df_copy_long = pd.wide_to_long(df_copy, stubnames=['prcptot','txm'], i=['year', 'model_used', 'scenario'], sep='_', j='month')

df_analogues_grouped = df_copy_long.query('scenario == "Analogues" ') #
df_analogues_grouped.loc[:,'month'] = df_analogues_grouped.index.get_level_values(level = 'month').copy()
df_analogues_grouped.loc[:,'scenario'] = df_analogues_grouped.index.get_level_values(level = 'scenario').copy()

df_climatology_grouped = df_copy_long.query(' scenario == "Climatology" & year == 2012')
df_climatology_grouped.loc[:,'month'] = df_climatology_grouped.index.get_level_values(level = 'month').copy()
df_climatology_grouped.loc[:,'scenario'] = df_climatology_grouped.index.get_level_values(level = 'scenario').copy()

fig, axes  = plt.subplots(2,1, figsize=(8,10), dpi=300, sharex='all')
for name in ['prcptot', 'txm']: #, ax in zip(names, axes.flatten()):
    if name in ['prcptot']:
        ax = axes[0]
    elif name in ['txm']:
        ax = axes[1]
            
    ax.hlines(y=df_climatology_grouped[name], xmin=df_climatology_grouped['month']-1.5, xmax=df_climatology_grouped['month']-0.5, colors='firebrick', linestyles='--', lw=2, label = '2012 event')
    g1 = sns.boxplot(ax = ax, x="month", y=name, data=df_analogues_grouped, palette = ['tab:orange'], hue = 'scenario') #drawstyle = 'steps-mid', err_kws= {'step':'mid'}
    g1 = sns.swarmplot(ax = ax, x="month", y=name, data=df_analogues_grouped, palette = ['firebrick'], size = 10, alpha=0.5, hue = 'scenario') #drawstyle = 'steps-mid', err_kws= {'step':'mid'}
    # g1.set(xticklabels=[])  # remove the tick labels
    # g1.set(xlabel= '')
    g1.set(ylabel= '' )
    ax.get_legend().remove()

axes[0].set_ylabel('Precipitation anomaly (mm)')
axes[0].set_xlabel("")
axes[0].set_title("a) Precipitation",loc='left')

axes[1].set_ylabel('Temperature anomaly (°C)')
axes[1].set_title("b) Temperature",loc='left')

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles[:2], labels[:2], bbox_to_anchor=(0.8, 0.01), ncol=3, frameon=False)
plt.tight_layout()
plt.savefig('paper_figures_production/clim_conditions_grouped_2012.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# # Fig SI: Plot boxplots comparing the historical events to the 2012 analogues
# =============================================================================
names = df_clim_counterfactuals_weighted_all_am.columns.drop(['scenario', 'model_used'])
ncols = len(names)
fig, axes  = plt.subplots(2,int(np.ceil(len(df_clim_counterfactuals_weighted_all_am.columns)/3)), figsize=(8,10), dpi=300, sharey='row')

for name, ax in zip(names, axes.flatten()):
    df_merge_subset_am = df_clim_counterfactuals_weighted_all_am.loc[:,[name,'model_used','scenario']]
    df_merge_subset_am['variable'] = name
    df_merge_subset_am[name] = df_merge_subset_am[name] - df_clim_hist_weighted.loc[start_hist_date:end_hist_date][name].mean()
    axline_2012_ref = 0 # DS_conditions_2012_weighted_am[name].mean() - DS_conditions_hist_weighted_am[name].sel(time=slice(start_hist_date,end_hist_date)).mean()
    
    # ax.axhline( y = 0, color = 'black', zorder = 0, linestyle = 'dotted')
    # ax.axhline( y = DS_conditions_2012_weighted_am[name].mean() - DS_conditions_hist_weighted_am[name].sel(time=slice(start_hist_date,end_hist_date)).mean(), color = 'firebrick', linestyle = 'dashed',linewidth =2, label = '2012 event', zorder = 19)
    # ax.axhspan(df_merge_subset_am[name].where(df_merge_subset_am['scenario'] == 'Climatology').dropna().quantile(0.05), df_merge_subset_am[name].where(df_merge_subset_am['scenario'] == 'Climatology').dropna().quantile(0.95), facecolor='0.2', alpha=0.3, zorder = 0, label = 'Climatology')
    g1 = sns.boxplot(y=name, x = 'variable', hue = 'model_used', data=df_merge_subset_am.where(df_merge_subset_am['scenario'] == 'Analogues').dropna(), orient='v', ax=ax)
    # g1 = sns.scatterplot(y=name, x = 'variable', data=df_merge_subset_am.where(df_merge_subset_am['scenario'] == 'Analogues' ), ax=ax, color = 'orange', s=60, label = 'Analogues', zorder = 20)
    g1.set(xticklabels=[])  # remove the tick labels
    g1.set(xlabel= name)
    if name in names[0:1]:
        g1.set(ylabel= 'Precipitation anomaly (mm)')  # remove the axis label  
    elif name in names[3:4]:
        g1.set(ylabel='Temperature anomaly (°C)' )  # remove the axis label 
    else:
        g1.set(ylabel='' ) 
    ax.get_legend().remove()
    g1.tick_params(bottom=False)  # remove the ticks
    # Change the visualization to put the 2012 event as central to the plot
    lower_boundary = axline_2012_ref - df_merge_subset_am.where(df_merge_subset_am['scenario'] == 'Analogues').dropna()[name].min()
    higher_boundary = df_merge_subset_am.where(df_merge_subset_am['scenario'] == 'Analogues').dropna()[name].max() - axline_2012_ref
    buffer_zone = max(lower_boundary, higher_boundary)
    ax.set_ylim(axline_2012_ref - buffer_zone*1.3, axline_2012_ref + buffer_zone*1.3)
    ax.xaxis.set_label_position('top') 
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.8, 0.01), ncol=2, frameon=False)
# plt.suptitle('2012 analogues')
plt.tight_layout()
plt.savefig('paper_figures_production/clim_conditions_2012_gcms.png', format='png', dpi=300, bbox_inches='tight')
plt.show()


# =============================================================================
# Grouped boxplots, similar figure to the previous one
# =============================================================================
df_copy = df_clim_counterfactuals_weighted_all_am.dropna().copy()
df_copy[['prcptot_1','prcptot_2','prcptot_3','txm_1','txm_2','txm_3']] = df_copy[['prcptot_1','prcptot_2','prcptot_3','txm_1','txm_2','txm_3']] - df_copy.query('scenario == "Climatology"').mean()
df_copy['year'] = df_copy.index

df_copy_long = pd.wide_to_long(df_copy, stubnames=['prcptot','txm'], i=['year', 'model_used', 'scenario'], sep='_', j='month')

df_copy_long['month'] = df_copy_long.index.get_level_values(level = 'month')
df_copy_long['scenario'] = df_copy_long.index.get_level_values(level = 'scenario')

fig, axes  = plt.subplots(2,1, figsize=(8,10), dpi=300, sharex='all')
for name in ['prcptot', 'txm']: #, ax in zip(names, axes.flatten()):
    if name in ['prcptot']:
        ax = axes[0]
    elif name in ['txm']:
        ax = axes[1]
            
    # ax.hlines(y=df_climatology_grouped[name], xmin=df_climatology_grouped['month']-1.5, xmax=df_climatology_grouped['month']-0.5, colors='firebrick', linestyles='--', lw=2, label = '2012 event')
    g1 = sns.boxplot(ax = ax, x="month", y=name, data=df_copy_long, hue = 'scenario') #drawstyle = 'steps-mid', err_kws= {'step':'mid'}
    # g1.set(xticklabels=[])  # remove the tick labels
    # g1.set(xlabel= '')
    g1.set(ylabel= '' )
    ax.get_legend().remove()

axes[0].set_ylabel('Precipitation anomaly (mm)')
axes[0].set_xlabel("")
axes[0].set_title("a) Precipitation",loc='left')

axes[1].set_ylabel('Temperature anomaly (°C)')
axes[1].set_title("b) Temperature",loc='left')

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.8, 0.01), ncol=3, frameon=False)
plt.tight_layout()
plt.savefig('paper_figures_production/fig_si_clim_conditions_2012_climatology_grouped.png', format='png', dpi=300, bbox_inches='tight')
plt.show()



# # =============================================================================
# # # Fig SI: Plot boxplots comparing the historical events to the 2012 analogues
# # =============================================================================
def symmetrize_y_axis(axes):
    y_max = np.abs(axes.get_ylim()).max()
    axes.set_ylim(ymin=-y_max, ymax=y_max)

# names = df_clim_counterfactuals_weighted_all_am.columns.drop(['scenario', 'model_used'])
# ncols = len(names)
# fig, axes  = plt.subplots(2,int(np.ceil(len(df_clim_counterfactuals_weighted_all_am.columns)/3)), figsize=(8, 8), dpi=300, sharey='row')

# for name, ax in zip(names, axes.flatten()):
#     df_merge_subset_am = df_clim_counterfactuals_weighted_all_am.loc[:,[name,'model_used','scenario']]
#     df_merge_subset_am['variable'] = name
#     df_merge_subset_am[name] = df_merge_subset_am[name] - df_clim_hist_weighted.loc[start_hist_date:end_hist_date][name].mean()

#     # ax.axhspan(df_merge_subset_am[name].where(df_merge_subset_am['scenario'] == 'Climatology').dropna().quantile(0.05), df_merge_subset_am[name].where(df_merge_subset_am['scenario'] == 'Climatology').dropna().quantile(0.95), facecolor='0.2', alpha=0.3, zorder = 0, label = 'Climatology')
#     g1 = sns.boxplot(y=name, x = 'variable', hue = 'scenario', data=df_merge_subset_am, orient='v', ax=ax)
#     # g1 = sns.scatterplot(y=name, x = 'variable', data=df_merge_subset_am.where(df_merge_subset_am['scenario'] == 'Analogues' ), ax=ax, color = 'orange', s=60, label = 'Analogues', zorder = 20)
#     # ax.axhline( y = DS_conditions_2012_weighted_am[name].mean(), color = 'red', linestyle = 'dashed',linewidth =2, label = '2012 event', zorder = 19)
#     g1.set(xticklabels=[])  # remove the tick labels
#     g1.set(xlabel= name)
#     if name in names[0:3]:
#         g1.set(ylabel= 'Precipitation anomaly (mm)')  # remove the axis label  
#     elif name in names[3:6]:
#         g1.set(ylabel='Temperature anomaly (°C)' )  # remove the axis label   
#     ax.get_legend().remove()
#     g1.tick_params(bottom=False)  # remove the ticks
#     symmetrize_y_axis(ax)
#     ax.xaxis.set_label_position('top') 
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels,bbox_to_anchor=(0.8, 0), ncol=3, frameon=False)
# # plt.suptitle('2012 analogues')
# plt.tight_layout()
# plt.savefig('paper_figures_production/fig_si_clim_conditions_2012_climatology.png', format='png', dpi=300, bbox_inches='tight')
# plt.show()


# # =============================================================================
# # # Fig SI: Plot boxplots comparing the historical events to the 2012 analogues
# # =============================================================================
# names = df_clim_counterfactuals_weighted_all_am.columns.drop(['scenario', 'model_used'])
# ncols = len(names)
# fig, axes  = plt.subplots(2,int(np.ceil(len(df_clim_counterfactuals_weighted_all_am.columns)/3)), figsize=(8, 8), dpi=300, sharey='row')

# for name, ax in zip(names, axes.flatten()):
#     df_merge_subset_am = df_clim_counterfactuals_weighted_all_am.loc[:,[name,'scenario']]
#     df_merge_subset_am['variable'] = name
#     df_merge_subset_am[name] = df_merge_subset_am[name] - df_clim_hist_weighted.loc[start_hist_date:end_hist_date][name].mean()
#     axline_2012_ref = DS_conditions_2012_weighted_am[name].mean() - DS_conditions_hist_weighted_am[name].sel(time=slice(start_hist_date,end_hist_date)).mean()
    
#     # ax.axhspan(df_merge_subset_am[name].where(df_merge_subset_am['scenario'] == 'Climatology').dropna().quantile(0.05), df_merge_subset_am[name].where(df_merge_subset_am['scenario'] == 'Climatology').dropna().quantile(0.95), facecolor='0.2', alpha=0.3, zorder = 0, label = 'Climatology')
#     g1 = sns.boxplot(y=name, x = 'variable', hue = 'scenario', data=df_merge_subset_am.where(df_merge_subset_am['scenario'] == 'Climatology').dropna(), orient='v', ax=ax)
#     # g1 = sns.scatterplot(y=name, x = 'variable', data=df_merge_subset_am.where(df_merge_subset_am['scenario'] == 'Analogues' ), ax=ax, color = 'orange', s=60, label = 'Analogues', zorder = 20)
#     ax.axhline( y = axline_2012_ref, color = 'firebrick', linestyle = 'dashed',linewidth =2, label = '2012 event', zorder = 19)
#     g1.set(xticklabels=[])  # remove the tick labels
#     g1.set(xlabel= name)
#     if name in names[0:3]:
#         g1.set(ylabel= 'Precipitation anomaly (mm)')  # remove the axis label  
#     elif name in names[3:6]:
#         g1.set(ylabel='Temperature anomaly (°C)' )  # remove the axis label   
#     ax.get_legend().remove()
#     g1.tick_params(bottom=False)  # remove the ticks

# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, bbox_to_anchor=(0.8, 0), ncol=3, frameon=False)
# # plt.suptitle('2012 analogues')
# plt.tight_layout()
# plt.savefig('paper_figures_production/fig_si_clim_conditions_2012_climatology_2.png', format='png', dpi=300, bbox_inches='tight')
# plt.show()

### Future scenarios climatology
# Conversion of future series to weighted timeseries    
df_clim_counter_ukesm_85_average = convert_clim_weighted_ensemble(df_hybrid_fut_ukesm_585_am, DS_hybrid_all_weighted, 'UKESM1-0-ll_5-8.5', DS_harvest_area_fut, mode = 'yield')    
df_clim_counter_ukesm_26_average = convert_clim_weighted_ensemble(df_hybrid_fut_ukesm_126_am, DS_hybrid_all_weighted, 'UKESM1-0-ll_1-2.6', DS_harvest_area_fut, mode = 'yield')    
df_clim_counter_gfdl_85_average = convert_clim_weighted_ensemble(df_hybrid_fut_gfdl_585_am, DS_hybrid_all_weighted, 'GFDL-esm4_5-8.5', DS_harvest_area_fut, mode = 'yield')    
df_clim_counter_gfdl_26_average = convert_clim_weighted_ensemble(df_hybrid_fut_gfdl_126_am, DS_hybrid_all_weighted, 'GFDL-esm4_1-2.6', DS_harvest_area_fut, mode = 'yield')    
df_clim_counter_ipsl_85_average = convert_clim_weighted_ensemble(df_hybrid_fut_ipsl_585_am, DS_hybrid_all_weighted, 'IPSL-cm6a-lr_5-8.5', DS_harvest_area_fut, mode = 'yield')  
df_clim_counter_ipsl_26_average = convert_clim_weighted_ensemble(df_hybrid_fut_ipsl_126_am, DS_hybrid_all_weighted, 'IPSL-cm6a-lr_1-2.6', DS_harvest_area_fut, mode = 'yield')    

# Merge dataframes with different names
df_clim_counterfactuals_weighted_all_am_average = pd.concat([df_clim_hist_weighted.loc[start_hist_date:end_hist_date], df_clim_counter_ukesm_85_average, df_clim_counter_ukesm_26_average, 
                                                  df_clim_counter_gfdl_85_average, df_clim_counter_gfdl_26_average,
                                                  df_clim_counter_ipsl_85_average, df_clim_counter_ipsl_26_average])
# =============================================================================
# # Fig SI: Plot boxplots comparing the historical events to the 2012 analogues
# =============================================================================
# names = df_clim_counterfactuals_weighted_all_am_average.columns.drop(['scenario', 'model_used'])
# ncols = len(names)
# fig, axes  = plt.subplots(2,int(np.ceil(len(df_clim_counterfactuals_weighted_all_am_average.columns)/3)), figsize=(8, 8), dpi=300, sharey='row')

# for name, ax in zip(names, axes.flatten()):
#     df_merge_subset_am = df_clim_counterfactuals_weighted_all_am_average.loc[:,[name,'scenario']]
#     df_merge_subset_am['variable'] = name
#     df_merge_subset_am[name] = df_merge_subset_am[name] - df_clim_hist_weighted.loc[start_hist_date:end_hist_date][name].mean()

#     # ax.axhspan(df_merge_subset_am[name].where(df_merge_subset_am['scenario'] == 'Climatology').dropna().quantile(0.05), df_merge_subset_am[name].where(df_merge_subset_am['scenario'] == 'Climatology').dropna().quantile(0.95), facecolor='0.2', alpha=0.3, zorder = 0, label = 'Climatology')
#     g1 = sns.boxplot(y=name, x = 'variable', hue = 'scenario', data=df_merge_subset_am, orient='v', ax=ax)
#     # g1 = sns.scatterplot(y=name, x = 'variable', data=df_merge_subset_am.where(df_merge_subset_am['scenario'] == 'Analogues' ), ax=ax, color = 'orange', s=60, label = 'Analogues', zorder = 20)
#     # ax.axhline( y = DS_conditions_2012_weighted_am[name].mean(), color = 'red', linestyle = 'dashed',linewidth =2, label = '2012 event', zorder = 19)
#     g1.set(xticklabels=[])  # remove the tick labels
#     g1.set(xlabel= name)
#     if name in names[0:3]:
#         g1.set(ylabel= 'Precipitation anomaly (mm)')  # remove the axis label  
#     elif name in names[3:6]:
#         g1.set(ylabel='Temperature anomaly (°C)' )  # remove the axis label   
#     ax.get_legend().remove()
#     g1.tick_params(bottom=False)  # remove the ticks
#     symmetrize_y_axis(ax)
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels,bbox_to_anchor=(0.8, 0), ncol=3, frameon=False)
# # plt.suptitle('2012 analogues')
# plt.tight_layout()
# # plt.savefig('paper_figures_production/fig_si_clim_conditions_2012_climatology.png', format='png', dpi=300, bbox_inches='tight')
# plt.show()


#%% 2012 analogues at a spatial level - how each country is affected by the 2012 analogues
# =============================================================================
'''
Try to compare the counterfatuals from a country perspective, does it mean one of the three countries is more prone to losses / failures?
'''
DS_historical_hybrid_us = xr.load_dataset("output_models_am/hybrid_epic_us-iiasa_gswp3-w5e5_obsclim_2015soc_default_yield-soy-noirr_global_annual_1901_2016.nc")
DS_historical_hybrid_br = xr.load_dataset("output_models_am/hybrid_epic_br-iiasa_gswp3-w5e5_obsclim_2015soc_default_yield-soy-noirr_global_annual_1901_2016.nc")
DS_historical_hybrid_arg = xr.load_dataset("output_models_am/hybrid_epic_arg-iiasa_gswp3-w5e5_obsclim_2015soc_default_yield-soy-noirr_global_annual_1901_2016.nc")

def country_scale_conversion(country, DS_harvest_area_sim, DS_counterfactuals_spatial, mode='production'):
    # Generate the calendars for both the historical and future periods, then the 2012 analogues in each country and the historical values per country
    # Determine country level yields for historical period
    DS_historical_hybrid_country = xr.load_dataset(f"output_models_am/hybrid_epic_{country}-iiasa_gswp3-w5e5_obsclim_2015soc_default_yield-soy-noirr_global_annual_1901_2016.nc")
    DS_mirca_country_hist = DS_harvest_area_sim.where(DS_historical_hybrid_country['Yield'] > -10)
    
    # Determine country level yields for future projections
    DS_counterfactual_country = DS_counterfactuals_spatial.where(DS_historical_hybrid_country['Yield'].sel(time = 2012) > -10)
    DS_mirca_country = DS_harvest_area_fut.where(DS_historical_hybrid_country['Yield'].sel(time = 2012) > -10)
              
    # Weighted analysis historical
    DS_historical_hybrid_country_weight = weighted_prod_conversion(DS_historical_hybrid_country, DS_area = DS_mirca_country_hist, mode=mode)
    
    return DS_mirca_country_hist, DS_mirca_country, DS_counterfactual_country, DS_historical_hybrid_country_weight

DS_mirca_us_hist, DS_mirca_us, DS_counterfactual_us, DS_historical_hybrid_us_weight = country_scale_conversion('us', DS_harvest_area_sim, DS_counterfactuals_spatial)
DS_mirca_br_hist, DS_mirca_br, DS_counterfactual_br, DS_historical_hybrid_br_weight = country_scale_conversion('br', DS_harvest_area_sim, DS_counterfactuals_spatial)
DS_mirca_arg_hist, DS_mirca_arg, DS_counterfactual_arg, DS_historical_hybrid_arg_weight = country_scale_conversion('arg', DS_harvest_area_sim, DS_counterfactuals_spatial)

# Plot historical timeline of weighted soybean yield
plt.plot(DS_historical_hybrid_us_weight.time, DS_historical_hybrid_us_weight['Yield'], label = 'US')
plt.plot(DS_historical_hybrid_br_weight.time, DS_historical_hybrid_br_weight['Yield'], label = 'Brazil')
plt.plot(DS_historical_hybrid_arg_weight.time, DS_historical_hybrid_arg_weight['Yield'], label = 'Argentina')
plt.title('Historical hybrid data')
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

plt.figure(figsize=(8,6))
plt.plot(DS_produc_am.time, DS_produc_am['Yield']/10**6, label = 'Americas', linewidth = 4, color= 'black')
plt.stackplot(DS_produc_us.time, DS_produc_us['Yield']/10**6,
              DS_produc_br['Yield']/10**6, DS_produc_arg['Yield']/10**6, labels = ['US', 'Brazil', 'Argentina'])
plt.axvline(2012, color = 'black', linestyle = 'dashed', label = '2012 event')
plt.title('Historical simulated soybean production for each country')
plt.ylabel('Soybean production (megaton)')
plt.legend()
plt.tight_layout()
plt.show()

print("relative shock (%) for the 2012 event with respect to climatology (2000-2015):", (DS_produc_am.sel(time = 2012)/10**6 - DS_produc_am.mean()/10**6)/(DS_produc_am.mean()/10**6))

# Find the counterfactual shocks using a baseline as reference, either historical yields or the factual as reference
DS_hybrid_counterfactuals_spatial_shock = DS_counterfactuals_spatial.dropna('time', how='all') - DS_historical_hybrid['Yield'].mean('time')
DS_hybrid_counterfactuals_spatial_shock_2012 = DS_counterfactuals_spatial.dropna('time', how='all') - DS_historical_hybrid['Yield'].sel(time = 2012)

def counterfactuals_per_scenario(DS):
    list_ds_counterfactuals = []
    for feature in list(DS.keys()):
        counterfactuals_by_rcp = DS[feature].sel(time = DS_counterfactuals_weighted_am[feature].time.where(DS_counterfactuals_weighted_am[feature] > -10).dropna(dim = 'time'))
        plot_2d_am_multi(counterfactuals_by_rcp, map_title = feature )
        list_ds_counterfactuals.append(counterfactuals_by_rcp)
    combined = xr.concat(list_ds_counterfactuals, dim='time')
    ds_combined = combined.to_dataset(name='Yield (ton/ha)')
    return ds_combined

# Spatial distribution of the future analogues with respect to the 2012 year
DS_counterfactuals_spatal_together_climatology = counterfactuals_per_scenario(DS_hybrid_counterfactuals_spatial_shock)
DS_counterfactuals_spatal_together_2012 = counterfactuals_per_scenario(DS_hybrid_counterfactuals_spatial_shock_2012)


# =============================================================================
# # FIGURE 4: Plots the counterfactuals per scenario 
# =============================================================================
plot_2d_am_map(DS_counterfactuals_spatal_together_climatology['Yield (ton/ha)'].mean('time'), colormap = 'RdBu', save_fig = 'mean_fut_analogues_spatial')
plot_2d_am_map(DS_counterfactuals_spatal_together_climatology['Yield (ton/ha)'].std('time'), colormap = 'Blues')
plot_2d_am_map(DS_counterfactuals_spatal_together_2012['Yield (ton/ha)'].mean('time'), colormap = 'RdBu', save_fig = 'mean_fut_analogues_spatial_2012')
plot_2d_am_map(DS_counterfactuals_spatal_together_2012['Yield (ton/ha)'].std('time'), colormap = 'Blues')

DS_counterfactuals_climatology_valuesforus = weighted_prod_conversion(DS_counterfactuals_spatal_together_climatology.rename({'Yield (ton/ha)':'US'}), DS_area = DS_mirca_us)
DS_counterfactuals_climatology_valuesforbr = weighted_prod_conversion(DS_counterfactuals_spatal_together_climatology.rename({'Yield (ton/ha)':'BR'}), DS_area = DS_mirca_br)
DS_counterfactuals_climatology_valuesforarg = weighted_prod_conversion(DS_counterfactuals_spatal_together_climatology.rename({'Yield (ton/ha)':'ARG'}), DS_area = DS_mirca_arg)

DS_counterfactuals_2012_valuesforus = weighted_prod_conversion(DS_counterfactuals_spatal_together_2012.rename({'Yield (ton/ha)':'US'}), DS_area = DS_mirca_us)
DS_counterfactuals_2012_valuesforbr = weighted_prod_conversion(DS_counterfactuals_spatal_together_2012.rename({'Yield (ton/ha)':'BR'}), DS_area = DS_mirca_br)
DS_counterfactuals_2012_valuesforarg = weighted_prod_conversion(DS_counterfactuals_spatal_together_2012.rename({'Yield (ton/ha)':'ARG'}), DS_area = DS_mirca_arg)

from statistics import NormalDist

def confidence_interval(data, confidence=0.95):
  dist = NormalDist.from_samples(data)
  z = NormalDist().inv_cdf((1 + confidence) / 2.)
  h = dist.stdev * z / ((len(data) - 1) ** .5)
  return dist.mean - h, dist.mean + h

analogues_0025_us, analogues_095_us = confidence_interval(DS_counterfactuals_2012_valuesforus.to_dataframe()['US'], confidence=0.95)
analogues_0025_br, analogues_095_br = confidence_interval(DS_counterfactuals_2012_valuesforbr.to_dataframe()['BR'], confidence=0.95)
analogues_0025_arg, analogues_095_arg = confidence_interval(DS_counterfactuals_2012_valuesforarg.to_dataframe()['ARG'], confidence=0.95)

print(f'\n The average anomalies of the 2012 analogues compared to the climatology is for the US: {DS_counterfactuals_climatology_valuesforus["US"].mean().values}, BR: {DS_counterfactuals_climatology_valuesforbr["BR"].mean().values}, and ARG: {DS_counterfactuals_climatology_valuesforarg["ARG"].mean().values}')
print(f'\n The 97.5p anomalies of the 2012 analogues compared to the climatology is for the US: {DS_counterfactuals_climatology_valuesforus["US"].quantile(1-0.025).values}, BR: {DS_counterfactuals_climatology_valuesforbr["BR"].quantile(1-0.025).values}, and ARG: {DS_counterfactuals_climatology_valuesforarg["ARG"].quantile(1-0.025).values}')
print(f'\n The 2.5p anomalies of the 2012 analogues compared to the climatology is for the US: {DS_counterfactuals_climatology_valuesforus["US"].quantile(0.025).values}, BR: {DS_counterfactuals_climatology_valuesforbr["BR"].quantile(0.025).values}, and ARG: {DS_counterfactuals_climatology_valuesforarg["ARG"].quantile(0.025).values}')


print(f'\n The average anomalies of the 2012 analogues compared to the original event (2012) is for the US: {DS_counterfactuals_2012_valuesforus["US"].mean().values}, BR: {DS_counterfactuals_2012_valuesforbr["BR"].mean().values}, and ARG: {DS_counterfactuals_2012_valuesforarg["ARG"].mean().values}')
# print(f'\n The STD anomalies of the 2012 analogues compared to the original event (2012) is for the US: {DS_counterfactuals_2012_valuesforus["US"].std().values}, BR: {DS_counterfactuals_2012_valuesforbr["BR"].std().values}, and ARG: {DS_counterfactuals_2012_valuesforarg["ARG"].std().values}')
print(f'\n The 97.5p anomalies of the 2012 analogues compared to the original event (2012) is for the US: {analogues_095_us}, BR: {analogues_095_br}, and ARG: {analogues_095_arg}')
print(f'\n The 2.5p anomalies of the 2012 analogues compared to the original event (2012) is for the US: {analogues_0025_us}, BR: {analogues_0025_br}, and ARG: {analogues_0025_arg}')

anomalies2012_percountry = pd.concat([DS_counterfactuals_2012_valuesforus['US'].to_dataframe(),DS_counterfactuals_2012_valuesforbr['BR'].to_dataframe(),DS_counterfactuals_2012_valuesforarg['ARG'].to_dataframe()], axis = 1)

# Plot accumulated mean losses per country - Relative shock
sns.barplot(data = anomalies2012_percountry )
plt.show()

#%% Timeseries of soybean production by country with trends
def figure_timeseries_per_country(DS_timeseries):
    DS_hybrid_trend_us_weighted = weighted_prod_conversion(DS_timeseries, DS_area = DS_mirca_us)
    DS_hybrid_trend_br_weighted = weighted_prod_conversion(DS_timeseries, DS_area = DS_mirca_br)
    DS_hybrid_trend_arg_weighted = weighted_prod_conversion(DS_timeseries, DS_area = DS_mirca_arg)
    
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
    ax3.set_ylim(10,80)
    plt.tight_layout()
    plt.show()
    
    # Determine the amount of country-level analogues
    analogues_us = DS_hybrid_trend_us_weighted.where(DS_hybrid_trend_us_weighted < DS_historical_hybrid_us_weight['Yield'].sel(time = 2012).values).to_dataframe().count()
    analogues_br = DS_hybrid_trend_br_weighted.where(DS_hybrid_trend_br_weighted < DS_historical_hybrid_br_weight['Yield'].sel(time = 2012).values).to_dataframe().count()
    analogues_arg = DS_hybrid_trend_arg_weighted.where(DS_hybrid_trend_arg_weighted < DS_historical_hybrid_arg_weight['Yield'].sel(time = 2012).values).to_dataframe().count()
    
    analogues_country = pd.concat([analogues_us,analogues_br, analogues_arg], axis = 1)
    analogues_country.columns = ['US', 'BR', 'ARG']
        
    DS_historical_hybrid_us_weight['Yield'].sel(time = slice(2000,2015)).plot()
    DS_hybrid_trend_us_weighted['GFDL-esm4_1-2.6'].plot()
    DS_hybrid_trend_us_weighted['UKESM1-0-ll_5-8.5'].plot()
    plt.show()
    DS_historical_hybrid_br_weight['Yield'].sel(time = slice(2000,2015)).plot()
    DS_hybrid_trend_br_weighted['GFDL-esm4_1-2.6'].plot()
    DS_hybrid_trend_br_weighted['UKESM1-0-ll_5-8.5'].plot()
    plt.show()
    DS_historical_hybrid_arg_weight['Yield'].sel(time = slice(2000,2015)).plot()
    DS_hybrid_trend_arg_weighted['GFDL-esm4_1-2.6'].plot()
    DS_hybrid_trend_arg_weighted['UKESM1-0-ll_5-8.5'].plot()
    plt.show()
    
    print('mean deviation in US')
    print(DS_hybrid_trend_us_weighted['GFDL-esm4_1-2.6'].mean().values/DS_historical_hybrid_us_weight['Yield'].sel(time = slice(2000,2015)).mean().values)
    print(DS_hybrid_trend_us_weighted['IPSL-cm6a-lr_1-2.6'].mean().values/DS_historical_hybrid_us_weight['Yield'].sel(time = slice(2000,2015)).mean().values)
    print(DS_hybrid_trend_us_weighted['UKESM1-0-ll_5-8.5'].mean().values/DS_historical_hybrid_us_weight['Yield'].sel(time = slice(2000,2015)).mean().values)
    
    print('mean deviation in BR')
    print(DS_hybrid_trend_br_weighted['GFDL-esm4_1-2.6'].mean().values/DS_historical_hybrid_br_weight['Yield'].sel(time = slice(2000,2015)).mean().values)
    print(DS_hybrid_trend_br_weighted['IPSL-cm6a-lr_1-2.6'].mean().values/DS_historical_hybrid_br_weight['Yield'].sel(time = slice(2000,2015)).mean().values)
    print(DS_hybrid_trend_br_weighted['UKESM1-0-ll_5-8.5'].mean().values/DS_historical_hybrid_br_weight['Yield'].sel(time = slice(2000,2015)).mean().values)
    
    print('mean deviation in ARG')
    print(DS_hybrid_trend_arg_weighted['GFDL-esm4_1-2.6'].mean().values/DS_historical_hybrid_arg_weight['Yield'].sel(time = slice(2000,2015)).mean().values)
    print(DS_hybrid_trend_arg_weighted['IPSL-cm6a-lr_1-2.6'].mean().values/DS_historical_hybrid_arg_weight['Yield'].sel(time = slice(2000,2015)).mean().values)
    print(DS_hybrid_trend_arg_weighted['UKESM1-0-ll_5-8.5'].mean().values/DS_historical_hybrid_arg_weight['Yield'].sel(time = slice(2000,2015)).mean().values)
    
    # Number of analogues per country
    print(np.sum(analogues_country))

    return analogues_country
    

analogues_country_trend = figure_timeseries_per_country(DS_hybrid_trend_all)
analogues_country_notrend = figure_timeseries_per_country(DS_hybrid_all)

print('Ratio of analogues caused by climate variability and by trends:',np.sum(analogues_country_notrend)/np.sum(analogues_country_trend))


#%% Analysis at a country level - 2012 event from a country perspective, aggregation at country level

# Failures per country
yield_factual_2012_us = DS_historical_hybrid_us_weight.sel(time = 2012).Yield.values
yield_factual_2012_br = DS_historical_hybrid_br_weight.sel(time = 2012).Yield.values
yield_factual_2012_arg = DS_historical_hybrid_arg_weight.sel(time = 2012).Yield.values

# Group
yield_factual_2012_am = pd.DataFrame([yield_factual_2012_us, yield_factual_2012_br, yield_factual_2012_arg], index = ['US', 'Brazil', 'Argentina'])
mean_historical_values_per_country = pd.DataFrame([DS_historical_hybrid_us_weight.Yield.mean('time').values, DS_historical_hybrid_br_weight.Yield.mean('time').values,  DS_historical_hybrid_arg_weight.Yield.mean('time').values], index = ['US', 'Brazil', 'Argentina'])

# 2012 analogues for each country
DS_counterfactual_us_weighted = weighted_prod_conversion(DS_counterfactual_us, DS_area = DS_mirca_us)
DS_counterfactual_us_weighted = DS_counterfactual_us_weighted.where(DS_counterfactual_us_weighted > 0).dropna('time', how = 'all')

DS_counterfactual_br_weighted = weighted_prod_conversion(DS_counterfactual_br, DS_area = DS_mirca_br)
DS_counterfactual_br_weighted = DS_counterfactual_br_weighted.where(DS_counterfactual_br_weighted > 0).dropna('time', how = 'all')

DS_counterfactual_arg_weighted = weighted_prod_conversion(DS_counterfactual_arg, DS_area = DS_mirca_arg)
DS_counterfactual_arg_weighted = DS_counterfactual_arg_weighted.where(DS_counterfactual_arg_weighted > 0).dropna('time', how = 'all')

df_mean_yields = pd.DataFrame( [DS_counterfactual_us_weighted.to_dataframe().mean(), DS_counterfactual_br_weighted.to_dataframe().mean(),
                               DS_counterfactual_arg_weighted.to_dataframe().mean()], 
                                          index = ['US', 'Brazil', 'Argentina'])


yield_climatology_2000_2015_us = DS_historical_hybrid_us_weight.sel(time = slice(start_hist_date,end_hist_date)).Yield.mean('time').values
yield_climatology_2000_2015_br = DS_historical_hybrid_br_weight.sel(time = slice(start_hist_date,end_hist_date)).Yield.mean('time').values
yield_climatology_2000_2015_arg = DS_historical_hybrid_arg_weight.sel(time = slice(start_hist_date,end_hist_date)).Yield.mean('time').values

loss_country_2012_us = yield_factual_2012_us - yield_climatology_2000_2015_us
loss_country_2012_br = yield_factual_2012_br - yield_climatology_2000_2015_br
loss_country_2012_arg = yield_factual_2012_arg - yield_climatology_2000_2015_arg
print(f'2012 year loss was for the US: {loss_country_2012_us}, BR: {loss_country_2012_br}, ARG: {loss_country_2012_arg} \n Total loss: {loss_country_2012_us+loss_country_2012_br+loss_country_2012_arg}')
print(f'2012 prcentage year loss was for the US: {loss_country_2012_us/yield_climatology_2000_2015_us}, BR: {loss_country_2012_br/yield_climatology_2000_2015_br}, ARG: {loss_country_2012_arg/yield_climatology_2000_2015_arg} \n Total loss: {(loss_country_2012_us+loss_country_2012_br+loss_country_2012_arg)/(yield_climatology_2000_2015_us + yield_climatology_2000_2015_br + yield_climatology_2000_2015_arg)}')

print( "US the amount of impact analogues at a country level is: ",len(DS_historical_hybrid_us_weight.sel(time = slice(start_hist_date,end_hist_date)).where(DS_historical_hybrid_us_weight['Yield'] <= yield_factual_2012_us).to_dataframe().dropna()))
print( "BR the amount of impact analogues at a country level is: ",len(DS_historical_hybrid_br_weight.sel(time = slice(start_hist_date,end_hist_date)).where(DS_historical_hybrid_br_weight['Yield'] <= yield_factual_2012_br).to_dataframe().dropna()))
print( "ARG the amount of impact analogues at a country level is: ",len(DS_historical_hybrid_arg_weight.sel(time = slice(start_hist_date,end_hist_date)).where(DS_historical_hybrid_arg_weight['Yield'] <= yield_factual_2012_arg).to_dataframe().dropna()))


# Test with only weighted yield
DS_mirca_us_hist_yield, DS_mirca_us_yield, DS_counterfactual_us_yield, DS_historical_hybrid_us_weight_yield = country_scale_conversion('us', DS_harvest_area_sim, DS_counterfactuals_spatial, mode='yield')
DS_mirca_br_hist_yield, DS_mirca_br_yield, DS_counterfactual_br_yield, DS_historical_hybrid_br_weight_yield = country_scale_conversion('br', DS_harvest_area_sim, DS_counterfactuals_spatial, mode='yield')
DS_mirca_arg_hist_yield, DS_mirca_arg_yield, DS_counterfactual_arg_yield, DS_historical_hybrid_arg_weight_yield = country_scale_conversion('arg', DS_harvest_area_sim, DS_counterfactuals_spatial, mode='yield')

print( "US the amount of impact analogues at a country level is: ",len(DS_historical_hybrid_us_weight_yield.where(DS_historical_hybrid_us_weight_yield['Yield'] <= DS_historical_hybrid_us_weight_yield['Yield'].sel(time = 2012)).to_dataframe().dropna()))
print( "BR the amount of impact analogues at a country level is: ",len(DS_historical_hybrid_br_weight_yield.where(DS_historical_hybrid_br_weight_yield['Yield'] <= DS_historical_hybrid_br_weight_yield['Yield'].sel(time = 2012)).to_dataframe().dropna()))
print( "ARG the amount of impact analogues at a country level is: ",len(DS_historical_hybrid_arg_weight_yield.where(DS_historical_hybrid_arg_weight_yield['Yield'] <= DS_historical_hybrid_arg_weight_yield['Yield'].sel(time = 2012)).to_dataframe().dropna()))

# Extra information but not really useful
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
    plt.bar(x = 'US', height = loss_country_2012_us)
    plt.bar(x = 'Brazil', height = loss_country_2012_br)
    plt.bar(x = 'Argentina', height = loss_country_2012_arg)
    plt.title('2012 absolute shock')
    plt.ylabel('Production anomaly (Megatonne)')
    plt.show()
    
    # Relative shocks
    plt.bar(x = 'US', height = yield_factual_2012_us /yield_climatology_2000_2015_us - 1)
    plt.bar(x = 'Brazil', height = yield_factual_2012_br / yield_climatology_2000_2015_br - 1)
    plt.bar(x = 'Argentina', height = yield_factual_2012_arg / yield_climatology_2000_2015_arg - 1)
    plt.title('2012 relative shock')
    plt.ylabel('Relative anomaly')
    plt.show()
    
    # =============================================================================
    # 2012 analogues with respect to historical country levels
    # =============================================================================
   
    # Counterfactuals of 2012 event in absolute values
    sns.barplot(data = (df_mean_yields.T ))
    plt.title('Counterfactuals absolute values')
    plt.ylabel('ton/ha')
    plt.show()
    
    # Counterfactuals of 2012 event in absolute shock
    sns.barplot(data = (df_mean_yields.T - mean_historical_values_per_country.T.values))
    plt.title('Local analogues of each country')
    plt.ylabel('ton/ha')
    plt.show()
    
    # Plot accumulated mean losses per country - Relative shock
    sns.barplot(data = (df_mean_yields.T / mean_historical_values_per_country.T.values - 1))
    plt.title('Counterfactuals relative shock')
    plt.show()

#%% Number of counterfactuals per country level, which scenarios and years they appear.

# Local counterfactuals
def counterfactual_generation(DS_yields, DS_mirca_country, local_factual):
    # Define the extend of the future timeseries based on the 2012 year
    DS_hybrid_country = DS_yields.where(DS_mirca_country['harvest_area'] > 0)
    # Make conversion to weighted timeseries
    DS_projections_weighted_country = weighted_prod_conversion(DS_hybrid_country, DS_area = DS_mirca_country)
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

local_analogues_2 = pd.concat([DS_projections_weighted_us_counterfactual.to_dataframe().melt().dropna()['value'], 
                                  DS_projections_weighted_br_counterfactual.to_dataframe().melt().dropna()['value'],
                                  DS_projections_weighted_arg_counterfactual.to_dataframe().melt().dropna()['value']],axis=1)
                              
local_analogues_2.columns=['US','Brazil','Argentina']

DS_counterfactuals_weighted_us, number_counter_us = counterfactuals_country_level(DS_projections_weighted_us_counterfactual, yield_factual_2012_us)
DS_counterfactuals_weighted_br, number_counter_br = counterfactuals_country_level(DS_projections_weighted_br_counterfactual, yield_factual_2012_br)
DS_counterfactuals_weighted_arg, number_counter_arg = counterfactuals_country_level(DS_projections_weighted_arg_counterfactual, yield_factual_2012_arg)

# =============================================================================
# FIG 4: Local analogues
# =============================================================================
fig, (ax1,ax2) = plt.subplots(nrows=2, figsize=(6,8),sharex = True )
# Occurrances
ax1.bar(x = ['US','Brazil','Argentina'], height = [number_counter_us, number_counter_br, number_counter_arg], color=['tab:blue','tab:orange','tab:green'])
ax1.set_title('a) Number of country-level analogues')
ax1.set_ylabel('Count')

local_analogues_deficit = local_analogues_2 - yield_factual_2012_am.T.values
# Counterfactuals of 2012 event in absolute shock
sns.barplot(data = (local_analogues_deficit), ax=ax2,ci=None)
ax2.set_title('b) Magnitude of country-level analogues')
ax2.set_ylabel('Production anomaly (Megatonne)')
plt.tight_layout()
plt.savefig('paper_figures_production/number_analogues.png', format='png', dpi=300)
plt.show()

local_analogues_0025_us, local_analogues_095_us = confidence_interval(local_analogues_deficit['US'].dropna(), confidence=0.95)
local_analogues_0025_br, local_analogues_095_br = confidence_interval(local_analogues_deficit['Brazil'].dropna(), confidence=0.95)
local_analogues_0025_arg, local_analogues_095_arg = confidence_interval(local_analogues_deficit['Argentina'].dropna(), confidence=0.95)

print('The average losses per country are ',local_analogues_2.mean().values - yield_factual_2012_am.T.values)
print('The 2.5% average losses per country are ',local_analogues_0025_us, local_analogues_0025_br, local_analogues_0025_arg)
print('The 97.5% losses per country are ',local_analogues_095_us, local_analogues_095_br, local_analogues_095_arg)


# Supplementary information on local analogues - co-occurrence
years_counterfactuals_am = df_hybrid_weighted_melt_counterfactuals[df_hybrid_weighted_melt_counterfactuals['value'] > -10].copy()
years_counterfactuals_us = DS_counterfactuals_weighted_us.to_dataframe().melt(ignore_index = False).dropna()
years_counterfactuals_br = DS_counterfactuals_weighted_br.to_dataframe().melt(ignore_index = False).dropna()
years_counterfactuals_arg = DS_counterfactuals_weighted_arg.to_dataframe().melt(ignore_index = False).dropna()
years_counterfactuals_am['region'] = 'Americas'
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
    domain_am = domain_country_locator('Americas')

    # Figure structure
    ax = ax
    ax.bar(domain.index, domain_us, label = 'US') #, bottom = domain.where(domain['region'] == 'AM')['value'])
    ax.bar(domain.index, domain_br, bottom = domain_us, label = 'Brazil')
    ax.bar(domain.index, domain_arg, bottom = domain_us, label = 'Argentina')
    ax.bar(domain.index, domain_am, bottom = domain_us + domain_arg + domain_br, label = 'Americas', hatch="//" ) 
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
fig.legend(handles, labels, loc=[0.1,-0.004], ncol=4, frameon=False )
plt.tight_layout()
plt.savefig('paper_figures_production/co_occurrence.png', format='png', dpi=300)
plt.show()


#%% OPPOSITE OF PREVIOUS SECTON, COMPARE THE LOCAL ANALOGUES WITH THE AGGREAGTE CONDITIONS IN EACH COUNTRY

df_clim_counterfactuals_weighted_all_am.where(df_clim_counterfactuals_weighted_all_am['scenario'] == 'Analogues').dropna()

# =============================================================================
# # Comparing the weather conditions between historic times, 2012 and counterfactuals for each climatic variables and for each country
# =============================================================================

def clim_conditions_analogues_anom(DS_area_hist, DS_area_fut, DS_counterfactuals_weighted_country, country, option = '2012', plot_legend = False, plot_yaxis = False):
    # Conversion of historical series to weighted timeseries    
    DS_conditions_hist_weighted_country = weighted_prod_conversion(DS_conditions_hist, DS_area = DS_area_hist, mode = 'yield')
    # DS to df
    df_clim_hist_weighted = DS_conditions_hist_weighted_country.sel(time = slice(start_hist_date,end_hist_date)).to_dataframe()
    df_clim_hist_weighted['scenario'] = 'Climatology'
    df_clim_hist_weighted['model_used'] = 'Climatology'
    
    # Conversion of future series to weighted timeseries    
    df_clim_counter_ukesm_85 = convert_clim_weighted_ensemble(df_hybrid_fut_ukesm_585_am, DS_counterfactuals_weighted_country, 'UKESM1-0-ll_5-8.5', DS_area_fut, mode = 'yield')    
    df_clim_counter_ukesm_26 = convert_clim_weighted_ensemble(df_hybrid_fut_ukesm_126_am, DS_counterfactuals_weighted_country, 'UKESM1-0-ll_1-2.6', DS_area_fut, mode = 'yield')    
    df_clim_counter_gfdl_85 = convert_clim_weighted_ensemble(df_hybrid_fut_gfdl_585_am, DS_counterfactuals_weighted_country, 'GFDL-esm4_5-8.5', DS_area_fut, mode = 'yield')    
    df_clim_counter_gfdl_26 = convert_clim_weighted_ensemble(df_hybrid_fut_gfdl_126_am, DS_counterfactuals_weighted_country, 'GFDL-esm4_1-2.6', DS_area_fut, mode = 'yield')    
    df_clim_counter_ipsl_85 = convert_clim_weighted_ensemble(df_hybrid_fut_ipsl_585_am, DS_counterfactuals_weighted_country, 'IPSL-cm6a-lr_5-8.5', DS_area_fut, mode = 'yield')    
    df_clim_counter_ipsl_26 = convert_clim_weighted_ensemble(df_hybrid_fut_ipsl_126_am, DS_counterfactuals_weighted_country, 'IPSL-cm6a-lr_1-2.6', DS_area_fut, mode = 'yield')    
    
    # Merge dataframes with different names
    df_clim_counterfactuals_weighted_all_country = pd.concat([df_clim_hist_weighted, df_clim_counter_ukesm_85, df_clim_counter_ukesm_26, 
                                                      df_clim_counter_gfdl_85, df_clim_counter_gfdl_26,
                                                      df_clim_counter_ipsl_85, df_clim_counter_ipsl_26])
    
    df_clim_counterfactuals_weighted_all_country['level'] = 'Country-level analogues' 
    
    
    df_clim_counter_ukesm_85_am = convert_clim_weighted_ensemble(df_hybrid_fut_ukesm_585_am, DS_counterfactuals_weighted_am, 'UKESM1-0-ll_5-8.5', DS_area_fut, mode = 'yield')    
    df_clim_counter_ukesm_26_am = convert_clim_weighted_ensemble(df_hybrid_fut_ukesm_126_am, DS_counterfactuals_weighted_am, 'UKESM1-0-ll_1-2.6', DS_area_fut, mode = 'yield')    
    df_clim_counter_gfdl_85_am = convert_clim_weighted_ensemble(df_hybrid_fut_gfdl_585_am, DS_counterfactuals_weighted_am, 'GFDL-esm4_5-8.5', DS_area_fut, mode = 'yield')    
    df_clim_counter_gfdl_26_am = convert_clim_weighted_ensemble(df_hybrid_fut_gfdl_126_am, DS_counterfactuals_weighted_am, 'GFDL-esm4_1-2.6', DS_area_fut, mode = 'yield')    
    df_clim_counter_ipsl_85_am = convert_clim_weighted_ensemble(df_hybrid_fut_ipsl_585_am, DS_counterfactuals_weighted_am, 'IPSL-cm6a-lr_5-8.5', DS_area_fut, mode = 'yield')    
    df_clim_counter_ipsl_26_am = convert_clim_weighted_ensemble(df_hybrid_fut_ipsl_126_am, DS_counterfactuals_weighted_am, 'IPSL-cm6a-lr_1-2.6', DS_area_fut, mode = 'yield')    
    
    # Merge dataframes with different names
    df_clim_counterfactuals_weighted_all_am = pd.concat([df_clim_counter_ukesm_85_am, df_clim_counter_ukesm_26_am, 
                                                      df_clim_counter_gfdl_85_am, df_clim_counter_gfdl_26_am,
                                                      df_clim_counter_ipsl_85_am, df_clim_counter_ipsl_26_am])
    
    df_clim_counterfactuals_weighted_all_am['level'] = '2012 analogues' 

    
    df_clim_analogues_levels = pd.concat([df_clim_counterfactuals_weighted_all_country, df_clim_counterfactuals_weighted_all_am])
    df_clim_analogues_levels['country'] = country
    return df_clim_analogues_levels


df_clim_analogues_levels_us = clim_conditions_analogues_anom(DS_mirca_us_hist, DS_mirca_us, DS_counterfactuals_weighted_us, country = 'US')
df_clim_analogues_levels_br = clim_conditions_analogues_anom(DS_mirca_br_hist, DS_mirca_br, DS_counterfactuals_weighted_br, country = 'Brazil')
df_clim_analogues_levels_arg = clim_conditions_analogues_anom(DS_mirca_arg_hist, DS_mirca_arg, DS_counterfactuals_weighted_arg, country = 'Argentina')  


df_analogues_levels_all = pd.concat([df_clim_analogues_levels_us, df_clim_analogues_levels_br, df_clim_analogues_levels_arg])

# =============================================================================
# Fig 6: climatic conditions of the country-level analogues with respect to the 2012 event
# =============================================================================

# Plot boxplots comparing the historical events, the 2012 event and the counterfactuals
def figure_anomalies_country_analogues(name_file, option = 2012):
    names = df_analogues_levels_all.columns.drop(['scenario', 'model_used','level','country'])
    position_i = {'US':0,'Brazil':3,'Argentina':6}
    fig, axes  = plt.subplots(2, int(len(names) * 3/2), figsize=(14, 8), dpi=300, sharey='row' )
    
    for country in df_analogues_levels_all.country.unique():
        
        for name in names: #, ax in zip(names, axes.flatten()):
            if name in ['prcptot_1', 'prcptot_2', 'prcptot_3']:
                ax = axes[0,position_i[country]+int(name[-1]) - 1]
            elif name in ['txm_1', 'txm_2', 'txm_3']:
                ax = axes[1,position_i[country]+int(name[-1]) - 1]
    
            df_merge_subset = df_analogues_levels_all.loc[:,[name,'scenario','level','country']]
            df_merge_subset = df_merge_subset.where(df_merge_subset['country'] == country).dropna()
            df_merge_subset['variable'] = name
            df_merge_subset[name] = df_merge_subset[name] - df_merge_subset.where(df_merge_subset['scenario'] == 'Climatology').dropna().loc[start_hist_date:end_hist_date][name].mean()
            axline_2012_ref = 0 # df_merge_subset.loc[2012][name] - df_merge_subset.where(df_merge_subset['scenario'] == 'Climatology').dropna().loc[start_hist_date:end_hist_date][name].mean()
            
            if option == 2012:
                g1 = sns.boxplot(y=name, x = 'variable', hue = 'level', data=df_merge_subset.where(df_merge_subset['scenario'] == 'Analogues').dropna(), orient='v', ax=ax)
                ax.axhline( y = df_merge_subset.loc[2012][name], color = 'firebrick', linestyle = 'dashed',linewidth =2, label = '2012 event', zorder = 19)
                # Change the visualization to put the 2012 event as central to the plot
                lower_boundary = axline_2012_ref - df_merge_subset.where(df_merge_subset['scenario'] == 'Analogues').dropna()[name].min()
                higher_boundary = df_merge_subset.where(df_merge_subset['scenario'] == 'Analogues').dropna()[name].max() - axline_2012_ref
                buffer_zone = max(lower_boundary, higher_boundary)
                ax.set_ylim(axline_2012_ref - buffer_zone*1.2, axline_2012_ref + buffer_zone*1.2)
            
            if option == 'climatology':
                g1 = sns.boxplot(y=name, x = 'variable', hue = 'scenario', data=df_merge_subset.where(df_merge_subset['level'] == 'Country-level analogues').dropna(), orient='v', ax=ax)
                lower_boundary = axline_2012_ref - df_merge_subset.where(df_merge_subset['scenario'] == 'Analogues').dropna()[name].min()
                higher_boundary = df_merge_subset.where(df_merge_subset['scenario'] == 'Analogues').dropna()[name].max() - axline_2012_ref
                buffer_zone = max(lower_boundary, higher_boundary)
                ax.set_ylim(axline_2012_ref - buffer_zone*1.2, axline_2012_ref + buffer_zone*1.2)
                
            if option == 'simple':
                g1 = sns.boxplot(y=name, x = 'variable', hue = 'level', data=df_merge_subset.query('level == "Country-level analogues" & scenario == "Analogues" ').dropna(), orient='v', ax=ax, showfliers = False)
                ax.axhline( y = df_merge_subset.loc[2012][name], color = 'firebrick', linestyle = 'dashed',linewidth =2, label = '2012 event', zorder = 19)
                # Change the visualization to put the 2012 event as central to the plot
                lower_boundary = axline_2012_ref - df_merge_subset.where(df_merge_subset['scenario'] == 'Analogues').dropna()[name].min()
                higher_boundary = df_merge_subset.where(df_merge_subset['scenario'] == 'Analogues').dropna()[name].max() - axline_2012_ref
                buffer_zone = max(lower_boundary, higher_boundary)
                # ax.set_ylim(axline_2012_ref - buffer_zone*1.2, axline_2012_ref + buffer_zone*1.2)
           
            
            g1.set(xticklabels=[])  # remove the tick labels
            g1.set(xlabel= name)
            g1.set(ylabel='' )
            ax.get_legend().remove()
            ax.xaxis.set_label_position('top') 

    
    axes[0,0].set_ylabel('Precipitation anomaly (mm)')
    axes[1,0].set_ylabel('Temperature anomaly (°C)')
    
    axes[0,0].set_title('a) US')
    axes[0,3].set_title('b) Brazil')
    axes[0,6].set_title('c) Argentina')
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.7, 0.04), ncol=3, frameon=False)
    plt.tight_layout()
    fig.savefig(f'paper_figures_production/{name_file}.png', format='png', dpi=300,bbox_inches='tight')


figure_anomalies_country_analogues(name_file = 'clim_conditions_countries_2012_anomaly', option = 2012)
figure_anomalies_country_analogues(name_file = 'fig_sup_clim_conditions_countries_climatology', option = 'climatology')     
figure_anomalies_country_analogues(name_file = 'fig_sup_clim_conditions_countries_simple', option = 'simple')     









#%%

def clim_conditions_analogues_anom_2(DS_area_hist, DS_area_fut, DS_counterfactuals_weighted_country, country):
    # Conversion of historical series to weighted timeseries    
    DS_conditions_hist_weighted_country = weighted_prod_conversion(DS_conditions_hist, DS_area = DS_area_hist, mode = 'yield')
    # DS to df
    df_clim_hist_weighted = DS_conditions_hist_weighted_country.sel(time = slice(start_hist_date,end_hist_date)).to_dataframe()
    df_clim_hist_weighted_mean = df_clim_hist_weighted.mean()

    df_clim_hist_weighted['scenario'] = 'Climatology'
    df_clim_hist_weighted['model_used'] = 'Climatology'
    
    # Conversion of future series to weighted timeseries    
    df_clim_counter_ukesm_85 = convert_clim_weighted_ensemble(df_hybrid_fut_ukesm_585_am, DS_counterfactuals_weighted_country, 'UKESM1-0-ll_5-8.5', DS_area_fut, mode = 'yield')    
    df_clim_counter_ukesm_26 = convert_clim_weighted_ensemble(df_hybrid_fut_ukesm_126_am, DS_counterfactuals_weighted_country, 'UKESM1-0-ll_1-2.6', DS_area_fut, mode = 'yield')    
    df_clim_counter_gfdl_85 = convert_clim_weighted_ensemble(df_hybrid_fut_gfdl_585_am, DS_counterfactuals_weighted_country, 'GFDL-esm4_5-8.5', DS_area_fut, mode = 'yield')    
    df_clim_counter_gfdl_26 = convert_clim_weighted_ensemble(df_hybrid_fut_gfdl_126_am, DS_counterfactuals_weighted_country, 'GFDL-esm4_1-2.6', DS_area_fut, mode = 'yield')    
    df_clim_counter_ipsl_85 = convert_clim_weighted_ensemble(df_hybrid_fut_ipsl_585_am, DS_counterfactuals_weighted_country, 'IPSL-cm6a-lr_5-8.5', DS_area_fut, mode = 'yield')    
    df_clim_counter_ipsl_26 = convert_clim_weighted_ensemble(df_hybrid_fut_ipsl_126_am, DS_counterfactuals_weighted_country, 'IPSL-cm6a-lr_1-2.6', DS_area_fut, mode = 'yield')    
    
    # Merge dataframes with different names
    df_clim_counterfactuals_weighted_all_country = pd.concat([df_clim_hist_weighted, df_clim_counter_ukesm_85, df_clim_counter_ukesm_26, 
                                                      df_clim_counter_gfdl_85, df_clim_counter_gfdl_26,
                                                      df_clim_counter_ipsl_85, df_clim_counter_ipsl_26])
    
    df_clim_counterfactuals_weighted_all_country[['prcptot_1','prcptot_2','prcptot_3','txm_1','txm_2','txm_3']] = df_clim_counterfactuals_weighted_all_country[['prcptot_1','prcptot_2','prcptot_3','txm_1','txm_2','txm_3']] - df_clim_hist_weighted_mean
    df_clim_counterfactuals_weighted_all_country['level'] = 'Country-level analogues' 
    
    df_clim_counter_ukesm_85_am = convert_clim_weighted_ensemble(df_hybrid_fut_ukesm_585_am, DS_counterfactuals_weighted_am, 'UKESM1-0-ll_5-8.5', DS_area_fut, mode = 'yield')    
    df_clim_counter_ukesm_26_am = convert_clim_weighted_ensemble(df_hybrid_fut_ukesm_126_am, DS_counterfactuals_weighted_am, 'UKESM1-0-ll_1-2.6', DS_area_fut, mode = 'yield')    
    df_clim_counter_gfdl_85_am = convert_clim_weighted_ensemble(df_hybrid_fut_gfdl_585_am, DS_counterfactuals_weighted_am, 'GFDL-esm4_5-8.5', DS_area_fut, mode = 'yield')    
    df_clim_counter_gfdl_26_am = convert_clim_weighted_ensemble(df_hybrid_fut_gfdl_126_am, DS_counterfactuals_weighted_am, 'GFDL-esm4_1-2.6', DS_area_fut, mode = 'yield')    
    df_clim_counter_ipsl_85_am = convert_clim_weighted_ensemble(df_hybrid_fut_ipsl_585_am, DS_counterfactuals_weighted_am, 'IPSL-cm6a-lr_5-8.5', DS_area_fut, mode = 'yield')    
    df_clim_counter_ipsl_26_am = convert_clim_weighted_ensemble(df_hybrid_fut_ipsl_126_am, DS_counterfactuals_weighted_am, 'IPSL-cm6a-lr_1-2.6', DS_area_fut, mode = 'yield')    
    
    # Merge dataframes with different names
    df_clim_counterfactuals_weighted_all_am = pd.concat([df_clim_counter_ukesm_85_am, df_clim_counter_ukesm_26_am, 
                                                      df_clim_counter_gfdl_85_am, df_clim_counter_gfdl_26_am,
                                                      df_clim_counter_ipsl_85_am, df_clim_counter_ipsl_26_am])
    
    df_clim_counterfactuals_weighted_all_am[['prcptot_1','prcptot_2','prcptot_3','txm_1','txm_2','txm_3']] = df_clim_counterfactuals_weighted_all_am[['prcptot_1','prcptot_2','prcptot_3','txm_1','txm_2','txm_3']] - df_clim_hist_weighted_mean
    df_clim_counterfactuals_weighted_all_am['level'] = '2012 analogues' 

    df_clim_analogues_levels = pd.concat([df_clim_counterfactuals_weighted_all_country, df_clim_counterfactuals_weighted_all_am])
    df_clim_analogues_levels['country'] = country
    return df_clim_analogues_levels


df_clim_analogues_levels_us2 = clim_conditions_analogues_anom_2(DS_mirca_us_hist, DS_mirca_us, DS_counterfactuals_weighted_us, country = 'US')
df_clim_analogues_levels_br2 = clim_conditions_analogues_anom_2(DS_mirca_br_hist, DS_mirca_br, DS_counterfactuals_weighted_br, country = 'Brazil')
df_clim_analogues_levels_arg2 = clim_conditions_analogues_anom_2(DS_mirca_arg_hist, DS_mirca_arg, DS_counterfactuals_weighted_arg, country = 'Argentina')  


df_analogues_levels_all_2 = pd.concat([df_clim_analogues_levels_us2, df_clim_analogues_levels_br2, df_clim_analogues_levels_arg2])



test = df_analogues_levels_all_2.dropna()
test['year'] = test.index
test_long = pd.wide_to_long(test, stubnames=['prcptot','txm'], i=['year', 'model_used', 'country', 'level', 'scenario'], sep='_', j='month')




fig, axes  = plt.subplots(2, int(len(df_analogues_levels_all.country.unique())), figsize=(10, 8), dpi=300, sharey='row', sharex = 'all' )
position_i = {'US':0,'Brazil':1,'Argentina':2}

for country in test_long.index.get_level_values(level = 'country').unique():
    data_country = test_long.query('country == @country & scenario == "Analogues" & level == "Country-level analogues" ') #
    data_country.loc[:,'month'] = data_country.index.get_level_values(level = 'month')
    data_country.loc[:,'level'] = data_country.index.get_level_values(level = 'level')
    
    data_country_climatology = test_long.query('country == @country & scenario == "Climatology" ')
    data_country_climatology.loc[:, 'month'] = data_country_climatology.index.get_level_values(level = 'month')
    data_country_climatology.loc[:, 'level'] = data_country_climatology.index.get_level_values(level = 'level')
    
    for name in ['prcptot', 'txm']: #, ax in zip(names, axes.flatten()):
        if name in ['prcptot']:
            ax = axes[0,position_i[country]]
        elif name in ['txm']:
            ax = axes[1,position_i[country]]
                
        ax.hlines(y=data_country_climatology.query('year == 2012')[name], xmin=data_country_climatology.query('year == 2012')['month']-1.5, xmax=data_country_climatology.query('year == 2012')['month']-0.5, colors='firebrick', linestyles='--', lw=2, label = '2012 event')
        g1 = sns.boxplot(ax = ax, x="month", y=name, data=data_country, hue = 'level', palette=['tab:orange']) #drawstyle = 'steps-mid', err_kws= {'step':'mid'}
        # g1 = sns.swarmplot(ax = ax, x="month", y=name, data=data_country, hue = 'level', palette=['firebrick']) #drawstyle = 'steps-mid', err_kws= {'step':'mid'}
        g1.set(ylabel= '' )
        ax.get_legend().remove()


axes[0,0].set_ylabel('Precipitation anomaly (mm)')
axes[1,0].set_ylabel('Temperature anomaly (°C)')

axes[0,0].set_xlabel("")
axes[0,1].set_xlabel("")
axes[0,2].set_xlabel("")

axes[0,0].set_title('a) US')
axes[0,1].set_title('b) Brazil')
axes[0,2].set_title('c) Argentina')

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.8, 0.03), ncol=3, frameon=False)
plt.tight_layout()
fig.savefig('paper_figures_production/clim_conditions_grouped_countries_2012_anomaly.png', format='png', dpi=300,bbox_inches='tight')
plt.show()



fig, axes  = plt.subplots(2, int(len(df_analogues_levels_all.country.unique())), figsize=(10, 8), dpi=300, sharey='row', sharex = 'all' )
position_i = {'US':0,'Brazil':1,'Argentina':2}

for country in test_long.index.get_level_values(level = 'country').unique():
    data_country = test_long.query('country == @country & level == "Country-level analogues" ') #
    data_country.loc[:, 'month'] = data_country.index.get_level_values(level = 'month')
    data_country.loc[:, 'level'] = data_country.index.get_level_values(level = 'level')
    data_country.loc[:, 'scenario'] = data_country.index.get_level_values(level = 'scenario')

   
    for name in ['prcptot', 'txm']: #, ax in zip(names, axes.flatten()):
        if name in ['prcptot']:
            ax = axes[0,position_i[country]]
        elif name in ['txm']:
            ax = axes[1,position_i[country]]
                
        # ax.hlines(y=data_country_climatology[name], xmin=data_country_climatology['month']-1.5, xmax=data_country_climatology['month']-0.5, colors='firebrick', linestyles='--', lw=2, label = '2012 event')
        g1 = sns.boxplot(ax = ax, x="month", y=name, data=data_country, hue = 'scenario') #drawstyle = 'steps-mid', err_kws= {'step':'mid'}
        g1.set(ylabel= '' )
        ax.get_legend().remove()


axes[0,0].set_ylabel('Precipitation anomaly (mm)')
axes[1,0].set_ylabel('Temperature anomaly (°C)')

axes[0,0].set_xlabel("")
axes[0,1].set_xlabel("")
axes[0,2].set_xlabel("")

axes[0,0].set_title('a) US')
axes[0,1].set_title('b) Brazil')
axes[0,2].set_title('c) Argentina')

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.8, 0.03), ncol=3, frameon=False)
plt.tight_layout()
fig.savefig('paper_figures_production/clim_conditions_grouped_countries_2012_anomaly_climatology.png', format='png', dpi=300,bbox_inches='tight')
plt.show()
#%% Yield - Climate interaction
# # RESEARCH QUESTION: What are the climatic conditions leading to the failures? 

# import matplotlib as mpl
# backend = mpl.get_backend()
# mpl.use('agg')

# # =============================================================================
# # # Comparing the weather conditions between historic times, 2012 and counterfactuals for each climatic variables and for each country
# # =============================================================================

# def clim_conditions_analogues_country(DS_area_hist, DS_area_fut, DS_counterfactuals_weighted_country, country, option = '2012', plot_legend = False, plot_yaxis = False):
#     # Show letter per country
#     letters_to_country = {'US':'a) US', 'Brazil':'b) Brazil', 'Argentina':'c) Argentina'}
    
#     # Conversion of historical series to weighted timeseries    
#     DS_conditions_hist_weighted_country = weighted_prod_conversion(DS_conditions_hist, DS_area = DS_area_hist, mode = 'yield')
#     DS_conditions_2012_weighted_country = DS_conditions_hist_weighted_country.sel(time=2012)
#     # DS to df
#     df_clim_hist_weighted = DS_conditions_hist_weighted_country.sel(time = slice(start_hist_date,end_hist_date)).to_dataframe()
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
    
#     # Plot boxplots comparing the historical events, the 2012 event and the counterfactuals
#     names = df_clim_counterfactuals_weighted_all_country.columns.drop(['scenario', 'model_used'])
#     fig, axes  = plt.subplots(2,int(np.ceil(len(df_clim_counterfactuals_weighted_all_country.columns)/3)), figsize=(4, 8), dpi=300)
    
#     for name, ax in zip(names, axes.flatten()):
#         df_merge_subset = df_clim_counterfactuals_weighted_all_country.loc[:,[name,'scenario']]
#         df_merge_subset['variable'] = name
        
#         if option == '2012':
#             g1 = sns.boxplot(y=name, x = 'variable', hue = 'scenario', data=df_merge_subset.where(df_merge_subset['scenario'] == 'Analogues').dropna(), palette=['tab:orange'], orient='v', ax=ax)
#             ax.axhline( y = DS_conditions_2012_weighted_country[name].mean(), color = 'firebrick', linestyle = 'dashed',linewidth =2, label = '2012 event', zorder = 19)
            
#             # Change the visualization to put the 2012 event as central to the plot
#             lower_boundary = DS_conditions_2012_weighted_country[name].mean() - df_merge_subset.where(df_merge_subset['scenario'] == 'Analogues').dropna()[name].min()
#             higher_boundary = df_merge_subset.where(df_merge_subset['scenario'] == 'Analogues').dropna()[name].max() - DS_conditions_2012_weighted_country[name].mean()
#             buffer_zone = max(lower_boundary, higher_boundary)
#             ax.set_ylim(DS_conditions_2012_weighted_country[name].mean() - buffer_zone, DS_conditions_2012_weighted_country[name].mean()+buffer_zone)
        
#         elif option == 'climatology':
#             g1 = sns.boxplot(y=name, x = 'variable', hue = 'scenario', data=df_merge_subset, orient='v', ax=ax)

#         elif option == 'historical':
#             g1 = sns.boxplot(y=name, x = 'variable', hue = 'scenario', data=df_merge_subset.where(df_merge_subset['scenario'] == 'Climatology').dropna(), orient='v', ax=ax)
#             ax.axhline( y = DS_conditions_2012_weighted_country[name].mean(), color = 'firebrick', linestyle = 'dashed',linewidth =2, label = '2012 event', zorder = 19)
        
#         g1.set(xticklabels=[])  # remove the tick labels
#         g1.set(xlabel= name)
#         if plot_yaxis == True:
#             if name in names[0:1]:
#                 g1.set(ylabel= 'Precipitation (mm)')  # remove the axis label  
#             elif name in names[3:4]:
#                 g1.set(ylabel='Temperature (°C)' )  # remove the axis label   
#             else:
#                 g1.set(ylabel='' )
#         else:
#             g1.set(ylabel='' )
            
#         ax.get_legend().remove()
#         g1.tick_params(bottom=False)  # remove the ticks
    
#     if plot_legend == True:
#         handles, labels = ax.get_legend_handles_labels()
#         fig.legend(handles, labels, bbox_to_anchor=(1, 0.04), ncol=2, frameon=False)
        
#     plt.suptitle(f'{letters_to_country[country]}', x=0.2, y=.97)
#     plt.tight_layout()
#     # plt.show()
#     return fig

# def edit_figs(fig):
#     c1 = fig.canvas
#     c1.draw()
#     a1 = np.array(c1.buffer_rgba())
#     return a1

# # =============================================================================
# # Fig 5: climatic conditions of the country-level analogues with respect to the 2012 event
# # =============================================================================
# fig_us_test = clim_conditions_analogues_country(DS_mirca_us_hist, DS_mirca_us, DS_counterfactuals_weighted_us, country = 'US', plot_legend= False, plot_yaxis = True)
# fig_br_test = clim_conditions_analogues_country(DS_mirca_br_hist, DS_mirca_br, DS_counterfactuals_weighted_br, country = 'Brazil', plot_legend= True)
# fig_arg_test = clim_conditions_analogues_country(DS_mirca_arg_hist, DS_mirca_arg, DS_counterfactuals_weighted_arg, country = 'Argentina', plot_legend= False)  

# fig_us_draw = edit_figs(fig_us_test)
# fig_br_draw = edit_figs(fig_br_test)
# fig_arg_draw = edit_figs(fig_arg_test)

# a = np.hstack((fig_us_draw,fig_br_draw,fig_arg_draw))

# mpl.use(backend)
# fig,ax = plt.subplots(figsize=(12, 8), dpi=200)
# fig.subplots_adjust(0, 0, 1, 1)
# ax.set_axis_off()
# plt.draw()
# ax.matshow(a)
# plt.tight_layout()
# fig.savefig('paper_figures_production/clim_conditions_countries_2012.png', format='png', dpi=300)

# # =============================================================================
# # Fig SI:5: climatic conditions of the country-level analogues with respect to climatology
# # =============================================================================
# fig_us_sup = clim_conditions_analogues_country(DS_mirca_us_hist, DS_mirca_us, DS_counterfactuals_weighted_us, country = 'US', option = 'climatology', plot_yaxis = True)
# fig_br_sup = clim_conditions_analogues_country(DS_mirca_br_hist, DS_mirca_br, DS_counterfactuals_weighted_br, country = 'Brazil', option = 'climatology', plot_legend= True)
# fig_arg_sup = clim_conditions_analogues_country(DS_mirca_arg_hist, DS_mirca_arg, DS_counterfactuals_weighted_arg, country = 'Argentina', option = 'climatology')  

# fig_us_sup_draw = edit_figs(fig_us_sup)
# fig_br_sup_draw = edit_figs(fig_br_sup)
# fig_arg_sup_draw = edit_figs(fig_arg_sup)

# a_sup = np.hstack((fig_us_sup_draw,fig_br_sup_draw,fig_arg_sup_draw))

# fig_sup, ax = plt.subplots(figsize=(12, 8), dpi=200)
# fig_sup.subplots_adjust(0, 0, 1, 1)
# ax.set_axis_off()
# plt.tight_layout()
# plt.draw()
# ax.matshow(a_sup)
# fig_sup.savefig('paper_figures_production/fig_sup_clim_conditions_countries_climatology.png', format='png', dpi=300)


# # =============================================================================
# # Fig SI:5: climatic conditions of the country-level analogues with respect to climatology
# # =============================================================================
# fig_us_sup_2 = clim_conditions_analogues_country(DS_mirca_us_hist, DS_mirca_us, DS_counterfactuals_weighted_us, country = 'US', option = 'historical', plot_yaxis = True)
# fig_br_sup_2 = clim_conditions_analogues_country(DS_mirca_br_hist, DS_mirca_br, DS_counterfactuals_weighted_br, country = 'Brazil', option = 'historical', plot_legend= True)
# fig_arg_sup_2 = clim_conditions_analogues_country(DS_mirca_arg_hist, DS_mirca_arg, DS_counterfactuals_weighted_arg, country = 'Argentina', option = 'historical')  

# fig_us_sup_draw_2 = edit_figs(fig_us_sup_2)
# fig_br_sup_draw_2 = edit_figs(fig_br_sup_2)
# fig_arg_sup_draw_2 = edit_figs(fig_arg_sup_2)

# a_sup_2 = np.hstack((fig_us_sup_draw_2,fig_br_sup_draw_2,fig_arg_sup_draw_2))

# fig_sup_2, ax_2 = plt.subplots(figsize=(12, 8), dpi=200)
# fig_sup_2.subplots_adjust(0, 0, 1, 1)
# ax_2.set_axis_off()
# plt.tight_layout()
# plt.draw()
# ax_2.matshow(a_sup_2)
# fig_sup_2.savefig('paper_figures_production/fig_sup_clim_conditions_countries_climatology_history.png', format='png', dpi=300)

#%% COMPARING ANOMALIES IN CLIMATIC CONDITIONS BETWEEN THE COUNTRY-LEVEL ANALOGUES AND THE AGGREGATED ANALOGUES.

# def country_level_analogues_conditions(DS_area_hist, DS_area_fut, DS_counterfactuals_weighted_country):
#     # Conversion of historical series to weighted timeseries    
#     DS_conditions_hist_weighted_country = weighted_prod_conversion(DS_conditions_hist, DS_area = DS_area_hist, mode = 'yield')
#     DS_conditions_2012_weighted_country = DS_conditions_hist_weighted_country.sel(time=2012)
#     # DS to df
#     df_clim_hist_weighted = DS_conditions_hist_weighted_country.sel(time = slice(start_hist_date,end_hist_date)).to_dataframe()
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
    
#     df_conditions_analogues = df_clim_counterfactuals_weighted_all_country.iloc[:,0:-2].where(df_clim_counterfactuals_weighted_all_country['scenario'] == 'Analogues').dropna()
#     df_conditions_climatology_mean = df_clim_counterfactuals_weighted_all_country.where(df_clim_counterfactuals_weighted_all_country['scenario'] == 'Climatology').dropna().mean()

#     df_anomalies = df_conditions_analogues - df_conditions_climatology_mean
    
#     return df_clim_counterfactuals_weighted_all_country, df_anomalies

# df_clim_counterfactuals_weighted_all_us, df_anomalies_us = country_level_analogues_conditions(DS_mirca_us_hist, DS_mirca_us, DS_counterfactuals_weighted_us)
# df_clim_counterfactuals_weighted_all_br, df_anomalies_br = country_level_analogues_conditions(DS_mirca_br_hist, DS_mirca_br, DS_counterfactuals_weighted_br)
# df_clim_counterfactuals_weighted_all_arg, df_anomalies_arg = country_level_analogues_conditions(DS_mirca_arg_hist, DS_mirca_arg, DS_counterfactuals_weighted_arg)
# df_clim_counterfactuals_weighted_all_am, df_anomalies_am = country_level_analogues_conditions(DS_harvest_area_hist, DS_harvest_area_fut, DS_counterfactuals_weighted_am)

# def anomaly_comparison(df_anomalies_country, df_anomalies_overall, country, plot_legend = False, plot_yaxis = True):
        
#     letters_to_country = {'US':'a) US', 'Brazil':'b) Brazil', 'Argentina':'c) Argentina', 'country-level':''}
#     df_anomalies_country= df_anomalies_country.copy()
#     df_anomalies_overall = df_anomalies_overall.copy()
    
#     df_anomalies_country['type'] = country
#     df_anomalies_overall['type'] = '2012 analogues'
    
#     df_anomalies_all = pd.concat([df_anomalies_country, df_anomalies_overall])
    
#     # Plot boxplots comparing the historical events, the 2012 event and the counterfactuals
#     names = df_anomalies_all.columns.drop(['type'])
#     if country in ['US','Brazil','Argentina']:
#         fig, axes  = plt.subplots(2,int(np.ceil(len(df_anomalies_all.columns)/3)), figsize=(4, 6), dpi=300)
#     if country == 'country-level':
#         fig, axes  = plt.subplots(2,int(np.ceil(len(df_anomalies_all.columns)/3)), figsize=(8, 8), dpi=300)

#     for name, ax in zip(names, axes.flatten()):
#         df_merge_subset = df_anomalies_all.loc[:,[name,'type']]
#         df_merge_subset['variable'] = name
#         #PLOT
#         ax.axhline( y = 0, color = 'firebrick', linestyle = 'dashed',linewidth =2, zorder = 19)
#         g1 = sns.boxplot(y=name, x = 'variable', hue = 'type', data=df_merge_subset, palette=['tab:blue','tab:orange'], orient='v', ax=ax)
#         yabs_max = abs(max(ax.get_ylim(), key=abs))
#         ax.set_ylim(ymin=-yabs_max, ymax=yabs_max)
        
#         g1.set(xticklabels=[])  # remove the tick labels
#         g1.set(xlabel= name)
#         if plot_yaxis == True:
#             if name in names[0:1]:
#                 g1.set(ylabel= 'Precip. anomaly (mm)')  # remove the axis label  
#             elif name in names[3:4]:
#                 g1.set(ylabel='Temp. anomaly (°C)' )  # remove the axis label   
#             else:
#                 g1.set(ylabel='' )
#         else:
#             g1.set(ylabel='' )
            
#         ax.get_legend().remove()
           
#         ax.get_legend()
#         g1.tick_params(bottom=False)  # remove the ticks
    
#     if plot_legend == True:
#         handles, labels = ax.get_legend_handles_labels()
#         fig.legend(handles, labels, bbox_to_anchor=(0.8, 0), ncol=2, frameon=False)
        
#     plt.suptitle(f'{letters_to_country[country]}', x=0.2, y=.97)
#     plt.tight_layout()
#     # plt.show()
#     return fig



# anomaly_comparison(df_anomalies_us, df_anomalies_am, 'US',plot_legend = False, plot_yaxis = True)
# anomaly_comparison(df_anomalies_br, df_anomalies_am, 'Brazil',plot_legend = True, plot_yaxis = False)
# anomaly_comparison(df_anomalies_arg, df_anomalies_am, 'Argentina',plot_legend = False, plot_yaxis = False)


# df_anomalies_all_countries = pd.concat([df_anomalies_us,df_anomalies_br, df_anomalies_arg])
# anomaly_comparison(df_anomalies_all_countries, df_anomalies_am, 'country-level',plot_legend = True, plot_yaxis = True)


# fig, axes  = plt.subplots(2,int(np.ceil(len(df_anomalies_am.columns)/2)), figsize=(12, 6), dpi=300)
# for var, ax in zip(df_anomalies_am.columns, axes.flatten()):
#     sns.kdeplot(df_anomalies_am[var], fill = True, color = 'black', label = '2012 analogues', ax=ax)
#     sns.kdeplot(df_anomalies_br.reset_index()[var], fill = True, label = 'US analogues', ax=ax)
#     sns.kdeplot(df_anomalies_us.reset_index()[var], fill = True,label = 'BR analogues', ax=ax)
#     sns.kdeplot(df_anomalies_arg.reset_index()[var],fill = True, label = 'ARG analogues', ax=ax)
#     if var in df_anomalies_am.columns[0:3]:
#         ax.set_xlabel('Precip. anomaly (mm)') # remove the axis label  
#     elif var in df_anomalies_am.columns[3:6]:
#         ax.set_xlabel('Temp. anomaly (°C)') # remove the axis label   
#     else:
#         g1.set(ylabel='' )
#     ax.get_legend()
# handles_1, labels_1 = ax.get_legend_handles_labels()
# fig.legend(handles_1, labels_1, bbox_to_anchor=(0.9, 0), ncol=4, frameon=False)
# # plt.suptitle(f'{letters_to_country[country]}', x=0.2, y=.97)
# plt.tight_layout()
# plt.show()
    
# fig, axes  = plt.subplots(2,int(np.ceil(len(df_anomalies_am.columns)/2)), figsize=(12, 6), dpi=300)
# for var, ax in zip(df_anomalies_am.columns, axes.flatten()):
#     sns.kdeplot(df_anomalies_am[var], fill = True, color = 'black', label = '2012 analogues', ax=ax)
#     sns.kdeplot(df_anomalies_all_countries.reset_index()[var],fill = True, label = 'country-level analogues', ax=ax)
#     if var in df_anomalies_am.columns[0:3]:
#         ax.set_xlabel('Precip. anomaly (mm)') # remove the axis label  
#     elif var in df_anomalies_am.columns[3:6]:
#         ax.set_xlabel('Temp. anomaly (°C)') # remove the axis label   
#     else:
#         g1.set(ylabel='' )
#     ax.get_legend()
# handles_1, labels_1 = ax.get_legend_handles_labels()
# lgd = fig.legend(handles_1, labels_1, bbox_to_anchor=(0.8, 0), ncol=4, frameon=False)
# # plt.suptitle(f'{letters_to_country[country]}', x=0.2, y=.97)
# plt.tight_layout()
# fig.savefig('samplefigure', bbox_inches='tight')
# plt.show()

