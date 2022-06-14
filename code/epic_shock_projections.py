# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 16:16:53 2021

@author: morenodu
"""

# 2015co2 RCP 26
DS_hybrid_2015co2_rcp26 = xr.open_dataset('hybrid_ssp126_2015co2_yield_soybean_2015_2100.nc', decode_times=True)
DS_hybrid_2015co2_rcp26 = DS_hybrid_2015co2_rcp26.rename_vars({'yield-soy-noirr':f'{DS_hybrid_2015co2_rcp26=}'.split('=')[0][3:]})

DS_epic_2015co2_rcp26 = xr.open_dataset('epic_ssp126_2015co2_yield_soybean_2015_2100.nc', decode_times=True)
DS_epic_2015co2_rcp26 = DS_epic_2015co2_rcp26.rename_vars({'yield-soy-noirr':f'{DS_epic_2015co2_rcp26=}'.split('=')[0][3:]})

DS_clim_2015co2_rcp26 = xr.open_dataset('clim_ssp126_2015co2_yield_soybean_2015_2100.nc', decode_times=True)
DS_clim_2015co2_rcp26 = DS_clim_2015co2_rcp26.rename_vars({'yield-soy-noirr':f'{DS_clim_2015co2_rcp26=}'.split('=')[0][3:]})


# 2015co2 RCP 85
DS_hybrid_2015co2_rcp85 = xr.open_dataset('hybrid_ssp585_2015co2_yield_soybean_2015_2100.nc', decode_times=True)
DS_hybrid_2015co2_rcp85 = DS_hybrid_2015co2_rcp85.rename_vars({'yield-soy-noirr':f'{DS_hybrid_2015co2_rcp85=}'.split('=')[0][3:]})

DS_epic_2015co2_rcp85 = xr.open_dataset('epic_ssp585_2015co2_yield_soybean_2015_2100.nc', decode_times=True)
DS_epic_2015co2_rcp85 = DS_epic_2015co2_rcp85.rename_vars({'yield-soy-noirr':f'{DS_epic_2015co2_rcp85=}'.split('=')[0][3:]})

DS_clim_2015co2_rcp85 = xr.open_dataset('clim_ssp585_2015co2_yield_soybean_2015_2100.nc', decode_times=True)
DS_clim_2015co2_rcp85 = DS_clim_2015co2_rcp85.rename_vars({'yield-soy-noirr':f'{DS_clim_2015co2_rcp85=}'.split('=')[0][3:]})

# Default CO2 RCP 26
DS_hybrid_default_rcp26 = xr.open_dataset('hybrid_ssp126_default_yield_soybean_2015_2100.nc', decode_times=True)
DS_hybrid_default_rcp26 = DS_hybrid_default_rcp26.rename_vars({'yield-soy-noirr':f'{DS_hybrid_default_rcp26=}'.split('=')[0][3:]})

DS_epic_default_rcp26 = xr.open_dataset('epic_ssp126_default_yield_soybean_2015_2100.nc', decode_times=True)
DS_epic_default_rcp26 = DS_epic_default_rcp26.rename_vars({'yield-soy-noirr':f'{DS_epic_default_rcp26=}'.split('=')[0][3:]})

DS_clim_default_rcp26 = xr.open_dataset('clim_ssp126_default_yield_soybean_2015_2100.nc', decode_times=True)
DS_clim_default_rcp26 = DS_clim_default_rcp26.rename_vars({'yield-soy-noirr':f'{DS_clim_default_rcp26=}'.split('=')[0][3:]})

# Default CO2 RCP 85
DS_hybrid_default_rcp85 = xr.open_dataset('hybrid_ssp585_default_yield_soybean_2015_2100.nc', decode_times=True)
DS_hybrid_default_rcp85 = DS_hybrid_default_rcp85.rename_vars({'yield-soy-noirr':f'{DS_hybrid_default_rcp85=}'.split('=')[0][3:]})

DS_epic_default_rcp85 = xr.open_dataset('epic_ssp585_default_yield_soybean_2015_2100.nc', decode_times=True)
DS_epic_default_rcp85 = DS_epic_default_rcp85.rename_vars({'yield-soy-noirr':f'{DS_epic_default_rcp85=}'.split('=')[0][3:]})

DS_clim_default_rcp85 = xr.open_dataset('clim_ssp585_default_yield_soybean_2015_2100.nc', decode_times=True)
DS_clim_default_rcp85 = DS_clim_default_rcp85.rename_vars({'yield-soy-noirr':f'{DS_clim_default_rcp85=}'.split('=')[0][3:]})


DS_merged_projections = xr.merge([ DS_hybrid_2015co2_rcp26[list(DS_hybrid_2015co2_rcp26.keys())],
          DS_epic_2015co2_rcp26[list(DS_epic_2015co2_rcp26.keys())],
          DS_clim_2015co2_rcp26[list(DS_clim_2015co2_rcp26.keys())],
          DS_hybrid_2015co2_rcp85[list(DS_hybrid_2015co2_rcp85.keys())],
          DS_epic_2015co2_rcp85[list(DS_epic_2015co2_rcp85.keys())],
          DS_clim_2015co2_rcp85[list(DS_clim_2015co2_rcp85.keys())],
          DS_hybrid_default_rcp26[list(DS_hybrid_default_rcp26.keys())],
          DS_epic_default_rcp26[list(DS_epic_default_rcp26.keys())],
          DS_clim_default_rcp26[list(DS_clim_default_rcp26.keys())],
          DS_hybrid_default_rcp85[list(DS_hybrid_default_rcp85.keys())],
          DS_epic_default_rcp85[list(DS_epic_default_rcp85.keys())],
          DS_clim_default_rcp85[list(DS_clim_default_rcp85.keys())]])

####
df_merged_projections = DS_merged_projections.mean(['lat','lon']).to_dataframe()

df_merged_projections_melt = df_merged_projections.melt(ignore_index = False)

list_test_slow = ['epic_2015co2_rcp26', 'clim_2015co2_rcp26','hybrid_2015co2_rcp26']

df_merged_projections_melt_red = df_merged_projections_melt[df_merged_projections_melt.variable.isin(list_test_slow)]

sns.kdeplot(x = df_merged_projections_melt_red.value, hue = df_merged_projections_melt_red.variable, fill = True)

df_merged_projections_melt_split = df_merged_projections_melt['variable'].str.split('_', expand=True)

df_merged_projections_melt_mult = pd.concat([df_merged_projections_melt, df_merged_projections_melt_split], axis =1 )
df_merged_projections_melt_mult = df_merged_projections_melt_mult.rename({0: 'model', 1: 'co2', 2:'rcp'}, axis='columns')
df_merged_projections_melt_mult = df_merged_projections_melt_mult[df_merged_projections_melt_mult['co2'] == 'default']

def plot_hues_2(dataframe, list_scen):
    dataframe = dataframe[dataframe.variable.isin(list_scen)]

    
    plt.figure(figsize = (6,6),dpi=200)
    ax = sns.kdeplot(x = dataframe.value, hue = dataframe.variable, fill = True)
    ax.legend_.set_bbox_to_anchor((0.05, 0.99))
    ax.legend_._set_loc(2)
    plt.title(f"Distribution of projections")
    plt.xlabel('Yield (ton/ha)')
    plt.show()
    
list_test_model = ['epic_2015co2_rcp26', 'clim_2015co2_rcp26','hybrid_2015co2_rcp26']
list_test_model_default = ['epic_default_rcp26', 'clim_default_rcp26','hybrid_default_rcp26']
list_85 = ['epic_2015co2_rcp85', 'clim_2015co2_rcp85','hybrid_2015co2_rcp85']
list_85_default = ['epic_default_rcp85', 'clim_default_rcp85','hybrid_default_rcp85']
list_rcp_default = ['hybrid_default_rcp26','hybrid_default_rcp85']

plot_hues_2(df_merged_projections_melt, list_test_model)
plot_hues_2(df_merged_projections_melt, list_test_model_default)
plot_hues_2(df_merged_projections_melt, list_85)
plot_hues_2(df_merged_projections_melt, list_85_default)
plot_hues_2(df_merged_projections_melt, list_rcp_default)

g = sns.kdeplot(x = df_merged_projections_melt_red.value, hue = df_merged_projections_melt_red.index, fill = True)
g.legend_.remove()

sns.lineplot(x=df_merged_projections_melt_mult.index, y=df_merged_projections_melt_mult.value, hue = df_merged_projections_melt_mult.rcp)
plt.ylabel('Yield (ton/ha)')
sns.lineplot(x=df_merged_projections_melt_mult.index, y=df_merged_projections_melt_mult.value, hue = df_merged_projections_melt_mult.co2)
sns.lineplot(x=df_merged_projections_melt_mult.index, y=df_merged_projections_melt_mult.value, hue = df_merged_projections_melt_mult.model)


df_merged_projections_melt.plot()

df_merged_projections.plot()

plt.figure(figsize = (6,6),dpi=200)
plt.plot(df_merged_projections)
plt.legend()
ax.legend_.set_bbox_to_anchor((0.05, 0.99))
ax.legend_._set_loc(2)
plt.title(f"Distribution of projections")
plt.xlabel('Yield (ton/ha)')
plt.show()



DS_hybrid_2015co2_rcp26['yield-soy-noirr'].mean(['lat','lon']).plot(label = 'hybrid')
DS_epic_2015co2_rcp26['yield-soy-noirr'].mean(['lat','lon']).plot(label = 'epic')
DS_clim_2015co2_rcp26['yield-soy-noirr'].mean(['lat','lon']).plot(label = 'clim')
plt.legend()
plt.show()



DS_hybrid_default_rcp26['yield-soy-noirr'].mean(['lat','lon']).plot(label = '2.6 default')
DS_hybrid_2015co2_rcp26['yield-soy-noirr'].mean(['lat','lon']).plot(label = '2.6 stat', linestyle = ":")
DS_hybrid_default_rcp85['yield-soy-noirr'].mean(['lat','lon']).plot(label = '8.5 default')
DS_hybrid_2015co2_rcp85['yield-soy-noirr'].mean(['lat','lon']).plot(label = '8.5 stat', linestyle = ":")
plt.legend()
plt.show()


DS_hybrid_default_rcp85['yield-soy-noirr'].mean(['lat','lon']).plot(label = 'default hybrid')
DS_hybrid_2015co2_rcp85['yield-soy-noirr'].mean(['lat','lon']).plot(label = 'stat hybrid', linestyle = ":")
DS_clim_default_rcp85['yield-soy-noirr'].mean(['lat','lon']).plot(label = 'default clim')
DS_clim_2015co2_rcp85['yield-soy-noirr'].mean(['lat','lon']).plot(label = ' stat clim', linestyle = ":")
DS_epic_default_rcp85['yield-soy-noirr'].mean(['lat','lon']).plot(label = 'default epic')
DS_epic_2015co2_rcp85['yield-soy-noirr'].mean(['lat','lon']).plot(label = 'stat epic', linestyle = ":")
plt.legend()
plt.show()

sns.kdeplot(data=DS_hybrid_2015co2_rcp26['yield-soy-noirr'].to_dataframe(),label = '2p6', fill = True)
sns.kdeplot(data=DS_hybrid_2015co2_rcp85['yield-soy-noirr'].to_dataframe(),label = '8p5', fill = True)
plt.legend()
plt.show()


sns.kdeplot(data=DS_hybrid_default_rcp26['yield-soy-noirr'].to_dataframe(),label = '2p6', fill = True)
sns.kdeplot(data=DS_hybrid_default_rcp85['yield-soy-noirr'].to_dataframe(),label = '8p5', fill = True)
plt.legend()
plt.show()


sns.kdeplot(data=DS_hybrid_default_rcp85['yield-soy-noirr'].to_dataframe(),label = 'hyb', fill = True)
sns.kdeplot(data=DS_epic_default_rcp85['yield-soy-noirr'].to_dataframe(),label = 'epic', fill = True)
sns.kdeplot(data=DS_clim_default_rcp85['yield-soy-noirr'].to_dataframe(),label = 'clim', fill = True)
plt.legend()
plt.show()

## KEEP SCALE CONSTANT


test = sns.load_dataset("penguins")

