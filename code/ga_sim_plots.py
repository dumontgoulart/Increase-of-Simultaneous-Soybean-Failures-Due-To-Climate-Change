# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 19:59:15 2022

@author: morenodu
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 10:13:29 2021

@author: morenodu
"""

import os
os.chdir('C:/Users/morenodu/OneDrive - Stichting Deltares/Documents/PhD/Paper_drought/data')

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
from stochastic_optimization_Algorithm import stochastic_optimization_Algorithm
from shap_prop import shap_prop
from bias_correction_masked import *
import matplotlib as mpl
import pickle

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['figure.dpi'] = 144
mpl.rcParams.update({'font.size': 14})

os.chdir('C:/Users/morenodu/OneDrive - Stichting Deltares/Documents/PhD/paper_hybrid_agri/data/ga_simulations')

df_soy_glob = pd.read_csv('ga_simulations_br_us_soy.csv', usecols = ['IEA_SCEN', 'VAR_ID', 'VAR_UNIT', 'MACROSCEN', 'BIOENSCEN','ANYREGION', 'ALLYEAR', 'VALUE'])
df_soy_glob.set_index('ALLYEAR', inplace=True)

df_soy_glob_subset = df_soy_glob[ (df_soy_glob['VAR_ID'] == 'YEXO') & (df_soy_glob['VAR_UNIT'] == 'fm t/ha') & (df_soy_glob['MACROSCEN'] == 'SSP2') ] 


df_soy_glob_subset_hist = df_soy_glob_subset[df_soy_glob_subset['BIOENSCEN'] == 'HistShock'] 
df_soy_glob_subset_hist = df_soy_glob_subset_hist.loc[[2010,2011],:]

df_soy_glob_subset_hist_shock_us = df_soy_glob_subset_hist[df_soy_glob_subset_hist['ANYREGION'] == 'USAReg']
df_soy_glob_subset_hist_shock_us['VALUE'] = df_soy_glob_subset_hist_shock_us.loc[:,'VALUE'].div(df_soy_glob_subset_hist_shock_us.loc[:,'VALUE'].shift(1))
df_soy_glob_subset_hist_shock_us['VALUE'] = (df_soy_glob_subset_hist_shock_us['VALUE'] - 1) * 100


df_soy_glob_subset_hist_shock_br = df_soy_glob_subset_hist[df_soy_glob_subset_hist['ANYREGION'] == 'BrazilReg']
df_soy_glob_subset_hist_shock_br['VALUE'] = df_soy_glob_subset_hist_shock_br.loc[:,'VALUE'].div(df_soy_glob_subset_hist_shock_br.loc[:,'VALUE'].shift(1))
df_soy_glob_subset_hist_shock_br['VALUE'] = (df_soy_glob_subset_hist_shock_br['VALUE'] - 1) * 100


def shock_conversion(prod_soya_br_subset, chosen_model = None, co2_scenario = None, shifter_calc = 'divide'): 

    prod_soya_br_subset_model = prod_soya_br_subset.copy()
    
    prod_soya_br_subset_model = prod_soya_br_subset_model[prod_soya_br_subset_model['IEA_SCEN'].str.len() >20]
    prod_soya_br_subset_model['gcm'] = prod_soya_br_subset_model.IEA_SCEN.str.split('_').str[0]
    prod_soya_br_subset_model['rcp'] = prod_soya_br_subset_model.IEA_SCEN.str.split('_').str[1]
    prod_soya_br_subset_model['shock_year'] = prod_soya_br_subset_model.IEA_SCEN.str.split('MIN_').str[0].str.slice(-2)
    prod_soya_br_subset_model['shock_year'] = prod_soya_br_subset_model['shock_year'].replace(['30','50','70'], [2031,2051,2071])
    
    prod_soya_br_subset_model['gcm_rcp_year'] = prod_soya_br_subset_model['gcm'] + "_" +prod_soya_br_subset_model['rcp'] +"_" + prod_soya_br_subset_model.index.astype(str)
    prod_soya_br_subset_model['rcp_year'] = prod_soya_br_subset_model['rcp'] +"_" + prod_soya_br_subset_model.index.astype(str)

    prod_soya_br_subset_model_dif = prod_soya_br_subset_model[['VAR_ID','ANYREGION','IEA_SCEN','gcm', 'rcp', 'shock_year', 'gcm_rcp_year','rcp_year','VALUE']]
    
    if chosen_model is not None:
        prod_soya_br_subset_model_dif = prod_soya_br_subset_model_dif[prod_soya_br_subset_model_dif['model'] == chosen_model]
    
    if shifter_calc == 'divide':
        prod_soya_br_subset_model_dif['VALUE'] = prod_soya_br_subset_model_dif.loc[:,'VALUE'].div(prod_soya_br_subset_model_dif.loc[:,'VALUE'].shift(1)) #prod_soya_br_subset_model_dif['VALUE'].diff()# 
        prod_soya_br_subset_model_dif_sub = prod_soya_br_subset_model_dif[prod_soya_br_subset_model_dif.index.astype(str).str.strip().str[-1] == '1'] 
        prod_soya_br_subset_model_dif_sub['VALUE'] = (prod_soya_br_subset_model_dif_sub['VALUE'] - 1) * 100

    elif shifter_calc == 'diff':
        prod_soya_br_subset_model_dif['VALUE'] = prod_soya_br_subset_model_dif['VALUE'].diff()# 
        prod_soya_br_subset_model_dif_sub = prod_soya_br_subset_model_dif[prod_soya_br_subset_model_dif.index.astype(str).str.strip().str[-1] == '1'] 
    
    
    
    return prod_soya_br_subset_model_dif_sub


df_soy_glob_subset_shock = shock_conversion(df_soy_glob_subset)

df_soy_glob_subset_shock_years = df_soy_glob_subset_shock[(df_soy_glob_subset_shock.index == df_soy_glob_subset_shock['shock_year']) &
                                                          (df_soy_glob_subset_shock['ANYREGION'] == 'BrazilReg')]

df_soy_glob_subset_shock_years = df_soy_glob_subset_shock_years[df_soy_glob_subset_shock_years['gcm'] == 'UKESM1-0-LL' ]

# BAR PLOTS BRAZIL
sns.barplot(x=df_soy_glob_subset_shock_years["rcp"], y=df_soy_glob_subset_shock_years["VALUE"], order = ['rcp2p6', 'rcp8p5'])
plt.title(f"Variable: {df_soy_glob_subset_shock_years['VAR_ID'].values[0]} in in {df_soy_glob_subset_shock_years['ANYREGION'].str.slice(0,-3).values[0]}")
plt.show()

sns.barplot(x=df_soy_glob_subset_shock_years.index, y=df_soy_glob_subset_shock_years["VALUE"])
plt.title(f"Variable: {df_soy_glob_subset_shock_years['VAR_ID'].values[0]} in in {df_soy_glob_subset_shock_years['ANYREGION'].str.slice(0,-3).values[0]}")
plt.show()

hatches = ['', '\\','', '\\', '', '\\']
colors = {'rcp2p6_2031': "#70A0CD", 'rcp8p5_2031': "#70A0CD", 'rcp2p6_2051': "#C47901", 'rcp8p5_2051': "#C47901", 'rcp2p6_2071': "#990102", 'rcp8p5_2071': "#990102"}

plt.figure(figsize = (12,6),dpi=200)
bar = sns.barplot(x=df_soy_glob_subset_shock_years["rcp_year"], y=df_soy_glob_subset_shock_years["VALUE"], palette=colors, order = ['rcp2p6_2031', 'rcp8p5_2031', 'rcp2p6_2051', 'rcp8p5_2051', 'rcp2p6_2071', 'rcp8p5_2071'], )#ci = None
bar.axhline(-2, linestyle='dashed', label = '2012 shock')
plt.legend()
plt.title(f"Shock for variable: {df_soy_glob_subset_shock_years['VAR_ID'].values[0]} in {df_soy_glob_subset_shock_years['ANYREGION'].str.slice(0,-3).values[0]}")
plt.ylabel('Shock (%)')
for i, thisbar in enumerate(bar.patches):
    thisbar.set_hatch(hatches[i])
plt.show()

df_soy_glob_subset_shock_years_sel = df_soy_glob_subset_shock_years[df_soy_glob_subset_shock_years['shock_year'] == 2051]
df_soy_glob_subset_shock_years_sel['GW degree'] =np.where(df_soy_glob_subset_shock_years_sel['rcp'] == 'rcp2p6', '2°C', np.where(df_soy_glob_subset_shock_years_sel['rcp'] =='rcp8p5', '3°C', ''))
df_soy_glob_subset_shock_years_sel = df_soy_glob_subset_shock_years_sel[['GW degree','ANYREGION','VALUE']]
df_soy_glob_subset_shock_years_sel = df_soy_glob_subset_shock_years_sel.append( pd.DataFrame([['Hist',df_soy_glob_subset_hist_shock_br['VALUE'].iloc[1]]], columns=['GW degree','VALUE'], index=[2051]) )
df_soy_glob_subset_shock_years_sel['ANYREGION'] = 'Brazil'

sns.barplot(x=df_soy_glob_subset_shock_years_sel['GW degree'], y=df_soy_glob_subset_shock_years_sel["VALUE"], order = ['Hist','2°C','3°C'] )
plt.show()


# BAR PLOTS US
df_soy_glob_subset_shock_years = df_soy_glob_subset_shock[(df_soy_glob_subset_shock.index == df_soy_glob_subset_shock['shock_year']) &
                                                          (df_soy_glob_subset_shock['ANYREGION'] == 'USAReg')]

df_soy_glob_subset_shock_years = df_soy_glob_subset_shock_years[df_soy_glob_subset_shock_years['gcm'] == 'UKESM1-0-LL' ]

sns.barplot(x=df_soy_glob_subset_shock_years["rcp"], y=df_soy_glob_subset_shock_years["VALUE"], order = ['rcp2p6', 'rcp8p5'])
plt.title(f"Variable: {df_soy_glob_subset_shock_years['VAR_ID'].values[0]} in in {df_soy_glob_subset_shock_years['ANYREGION'].str.slice(0,-3).values[0]}")
plt.show()

sns.barplot(x=df_soy_glob_subset_shock_years.index, y=df_soy_glob_subset_shock_years["VALUE"])
plt.title(f"Variable: {df_soy_glob_subset_shock_years['VAR_ID'].values[0]} in in {df_soy_glob_subset_shock_years['ANYREGION'].str.slice(0,-3).values[0]}")
plt.show()

hatches = ['', '\\','', '\\', '', '\\']
colors = {'rcp2p6_2031': "#70A0CD", 'rcp8p5_2031': "#70A0CD", 'rcp2p6_2051': "#C47901", 'rcp8p5_2051': "#C47901", 'rcp2p6_2071': "#990102", 'rcp8p5_2071': "#990102"}

plt.figure(figsize = (12,6),dpi=200)
bar = sns.barplot(x=df_soy_glob_subset_shock_years["rcp_year"], y=df_soy_glob_subset_shock_years["VALUE"], palette=colors, order = ['rcp2p6_2031', 'rcp8p5_2031', 'rcp2p6_2051', 'rcp8p5_2051', 'rcp2p6_2071', 'rcp8p5_2071'], )#ci = None
bar.axhline(-2, linestyle='dashed', label = '2012 shock')
plt.legend()
plt.title(f"Shock for variable: {df_soy_glob_subset_shock_years['VAR_ID'].values[0]} in {df_soy_glob_subset_shock_years['ANYREGION'].str.slice(0,-3).values[0]}")
plt.ylabel('Shock (%)')
for i, thisbar in enumerate(bar.patches):
    thisbar.set_hatch(hatches[i])
plt.show()

df_soy_glob_subset_shock_years_sel_us = df_soy_glob_subset_shock_years[df_soy_glob_subset_shock_years['shock_year'] == 2051]
df_soy_glob_subset_shock_years_sel_us['GW degree'] =np.where(df_soy_glob_subset_shock_years_sel_us['rcp'] == 'rcp2p6', '2°C', np.where(df_soy_glob_subset_shock_years_sel_us['rcp'] =='rcp8p5', '3°C', ''))
df_soy_glob_subset_shock_years_sel_us = df_soy_glob_subset_shock_years_sel_us[['GW degree','ANYREGION','VALUE']]
df_soy_glob_subset_shock_years_sel_us = df_soy_glob_subset_shock_years_sel_us.append( pd.DataFrame([['Hist',df_soy_glob_subset_hist_shock_us['VALUE'].iloc[1]]], columns=['GW degree','VALUE'], index=[2051]) )
df_soy_glob_subset_shock_years_sel_us['ANYREGION'] = 'US'
sns.barplot(x=df_soy_glob_subset_shock_years_sel_us['GW degree'], y=df_soy_glob_subset_shock_years_sel_us["VALUE"], order = ['Hist','2°C','3°C'] )
plt.show()



df_bar_br_us = pd.concat([df_soy_glob_subset_shock_years_sel_us, df_soy_glob_subset_shock_years_sel])
df_bar_br_us['GW degree'][df_bar_br_us['GW degree'] == '2°C'] = '2.5°C'
# PLOT FINAL FIGURE GA #############################################################################
plt.figure(figsize = (5,5),dpi=300)
sns.barplot(x=df_bar_br_us['GW degree'], y=df_bar_br_us["VALUE"], hue= df_bar_br_us['ANYREGION'], order = ['Hist','2.5°C','3°C'] )
plt.ylabel('Shock (%)')
plt.title('Soybean yield shock')
plt.legend( title="",loc=3, fontsize='small', fancybox=True)
plt.show()

#%%

df_soy_glob_subset_xprp = df_soy_glob[ (df_soy_glob['VAR_ID'] == 'XPRP') & (df_soy_glob['VAR_UNIT'] == 'fm t/ha') & (df_soy_glob['MACROSCEN'] == 'SSP2') ] 

df_soy_glob_subset_shock = shock_conversion(df_soy_glob_subset)


#%%
test_plot = df_soy_glob_subset_shock_years.reset_index(drop=True)

ax = sns.kdeplot(x= test_plot["VALUE"], hue = test_plot["shock_year"], fill = True)
plt.title(f"Variable: {df_soy_glob_subset_shock_years['VAR_ID'].values[0]}")
ax.legend_.set_bbox_to_anchor((0.05, 0.99))
ax.legend_._set_loc(2)
plt.title(f"Shock for variable: {test_plot['VAR_ID'].values[0]} in {test_plot['ANYREGION'].str.slice(0,-3).values[0]}")
plt.xlabel('Shock impact (%)')
plt.show()


ax = sns.kdeplot(x= test_plot["VALUE"], hue = test_plot["rcp"], fill = True)
plt.title(f"Variable: {df_soy_glob_subset_shock_years['VAR_ID'].values[0]}")
ax.legend_.set_bbox_to_anchor((0.05, 0.99))
ax.legend_._set_loc(2)
plt.title(f"Shock for variable: {test_plot['VAR_ID'].values[0]} in {test_plot['ANYREGION'].str.slice(0,-3).values[0]}")
plt.xlabel('Shock impact (%)')
plt.show()



# Conditioned on single RCPs
plt.figure(figsize = (5,5),dpi=300)
ax = sns.kdeplot(x= test_plot[test_plot['rcp'] == 'rcp8p5']["VALUE"], hue = test_plot[test_plot['rcp'] == 'rcp8p5']["shock_year"], fill = True)
plt.title(f"RCP 8.5")
ax.legend_.set_bbox_to_anchor((0.05, 0.99))
ax.legend_._set_loc(2)
plt.title(f"Shock for variable: {test_plot['VAR_ID'].values[0]} in {test_plot['ANYREGION'].str.slice(0,-3).values[0]}")
plt.xlabel('Shock impact (%)')
plt.show()


plt.figure(figsize = (5,5),dpi=300)
ax = sns.kdeplot(x= test_plot[test_plot['rcp'] == 'rcp2p6']["VALUE"], hue = test_plot[test_plot['rcp'] == 'rcp2p6']["shock_year"], fill = True)
plt.title(f"RCP 8.5")
ax.legend_.set_bbox_to_anchor((0.05, 0.99))
ax.legend_._set_loc(2)
plt.title(f"Shock for variable: {test_plot['VAR_ID'].values[0]} in {test_plot['ANYREGION'].str.slice(0,-3).values[0]}")
plt.xlabel('Shock impact (%)')
plt.show()

# Each shock year
for shock_year_sel in test_plot['shock_year'].unique():
    plt.figure(figsize = (5,5),dpi=300)
    ax = sns.kdeplot(x= test_plot[test_plot['shock_year'] == shock_year_sel]["VALUE"], hue = test_plot[test_plot['shock_year'] == shock_year_sel]["rcp"], fill = True, hue_order = ['rcp2p6', 'rcp8p5'])
    plt.title(f"RCP 8.5")
    ax.legend_.set_bbox_to_anchor((0.05, 0.99))
    ax.legend_._set_loc(2)
    ax.set_xlim(-80,50)
    plt.title(f"Shock for variable: {test_plot['VAR_ID'].values[0]} in {test_plot['ANYREGION'].str.slice(0,-3).values[0]}")
    plt.xlabel('Shock impact (%)')
    plt.show()
    

# Group by warming level
# 2 degree: IPSL_26_2070, IPSL_85_2030, GFDL_85_2050, UKESM_26_2030, UKESM_85_2030
# 3 degree:  IPSL_85_2050, GFDL_85_2070, UKESM_85_2050
list_2c_gw = ['IPSL-CM6A-LR_rcp2p6_2071', 'IPSL-CM6A-LR_rcp8p5_2031', 'GFDL-ESM4_rcp8p5_2051', 'UKESM1-0-LL_rcp8p5_2031', 'UKESM1-0-LL_rcp2p6_2031']
list_3c_gw = ['IPSL-CM6A-LR_rcp8p5_2051', 'GFDL-ESM4_rcp8p5_2071', 'UKESM1-0-LL_rcp8p5_2051']

df_soy_glob_subset_shock_years['GW_level'] = np.where(df_soy_glob_subset_shock_years['gcm_rcp_year'].isin(list_2c_gw), '2°C', np.where(df_soy_glob_subset_shock_years['gcm_rcp_year'].isin(list_3c_gw), '3°C', ''))


# BAR PLOTS BRAZIL
sns.barplot(x=df_soy_glob_subset_shock_years[(df_soy_glob_subset_shock_years["GW_level"] =='2°C') |(df_soy_glob_subset_shock_years["GW_level"] =='3°C' )]["GW_level"], 
            y=df_soy_glob_subset_shock_years[(df_soy_glob_subset_shock_years["GW_level"] =='2°C') |(df_soy_glob_subset_shock_years["GW_level"] =='3°C' )]["VALUE"])
plt.title(f"Variable: {df_soy_glob_subset_shock_years['VAR_ID'].values[0]} in in {df_soy_glob_subset_shock_years['ANYREGION'].str.slice(0,-3).values[0]}")
plt.show()

#%% EUROPE  - GA ESTHER

df_europe_imp = pd.read_csv('ga_simulations_europe_soy.csv', usecols = ['IEA_SCEN', 'VAR_ID','Item', 'VAR_UNIT', 'BIOENSCEN', 'MACROSCEN','ANYREGION', 'ALLYEAR', 'VALUE'])
df_europe_imp['VALUE'] = (df_europe_imp['VALUE'] - 1) *100
df_europe_imp.set_index('ALLYEAR', inplace=True)
df_europe_imp['gcm'] = df_europe_imp.IEA_SCEN.str.split('_').str[0]

df_europe_imp_ref = df_europe_imp[(df_europe_imp['BIOENSCEN'] == 'HistShock')]

df_europe_imp_subset = df_europe_imp[ (df_europe_imp['IEA_SCEN'] == 'UKESM1-0-LL_rcp2p6_50MIN_hybrid') | (df_europe_imp['IEA_SCEN'] == 'UKESM1-0-LL_rcp8p5_50MIN_hybrid')]
df_europe_imp_subset['GW degree'] = np.where(df_europe_imp_subset['IEA_SCEN'] == 'UKESM1-0-LL_rcp2p6_50MIN_hybrid', '2.5°C', np.where(df_europe_imp_subset['IEA_SCEN'] =='UKESM1-0-LL_rcp8p5_50MIN_hybrid', '3°C', ''))

def plot_shocks_socio(Item_sel, Var_id_sel, Region = 'EU'):
    df_europe_imp_scen = df_europe_imp_subset[ (df_europe_imp_subset['VAR_ID'] == Var_id_sel) & (df_europe_imp_subset['Item'] == Item_sel)  & (df_europe_imp_subset['ANYREGION'] == Region)] 
    df_europe_imp_ref_sub = df_europe_imp_ref[(df_europe_imp_ref['VAR_ID'] == Var_id_sel) & (df_europe_imp_ref['Item'] == Item_sel)& (df_europe_imp_ref['ANYREGION'] == Region)]  
    print(df_europe_imp_scen)
    # BAR PLOTS EUROPE
    plt.figure(figsize = (7,5),dpi=300)
    sns.barplot(x=df_europe_imp_scen["GW degree"], y=df_europe_imp_scen["VALUE"], hue = df_europe_imp_scen['BIOENSCEN'], order = ['2.5°C', '3°C'])
    plt.axhline(df_europe_imp_ref_sub['VALUE'].values, c = 'k', linestyle = 'dashed', label = 'Hist')
    plt.title(f"{df_europe_imp_scen['VAR_ID'].values[0]} of {df_europe_imp_scen['Item'].values[0]} in {df_europe_imp_scen['ANYREGION'].values[0]}")
    plt.ylabel('Shock (%)')
    plt.legend()
    plt.show()
    
plot_shocks_socio('Soya','CONS')
plot_shocks_socio('Soya','XPRP')
plot_shocks_socio('Soya','FOOD')
plot_shocks_socio('Soya','FEED')
plot_shocks_socio('Meat','XPRP')

plot_shocks_socio('Soya','PROD', 'USA_BRA')
plot_shocks_socio('Meat','XPRP', 'USA_BRA')
plot_shocks_socio('Meat','FOOD', 'USA_BRA')
plot_shocks_socio('Meat','PROD', 'USA_BRA')
plot_shocks_socio('Meat','CALO', 'USA_BRA')
plot_shocks_socio('Soya','CALO', 'USA_BRA')
plot_shocks_socio('Meat','CONS', 'USA_BRA')
plot_shocks_socio('Soya','CONS', 'USA_BRA')


df_europe_imp = pd.read_csv('ga_simulations_europe_soy.csv', usecols = ['IEA_SCEN', 'VAR_ID','Item', 'VAR_UNIT', 'BIOENSCEN', 'MACROSCEN','ANYREGION', 'ALLYEAR', 'VALUE'])
df_europe_imp['VALUE'] = (df_europe_imp['VALUE'] - 1) *100
df_europe_imp.set_index('ALLYEAR', inplace=True)
df_europe_imp['gcm'] = df_europe_imp.IEA_SCEN.str.split('_').str[0]



#%% Change visualization to degree of GW
import glob
def open_csv_timeseries(path= "projections_global_mean/ipsl-cm6a-lr_r1i1p1f1_w5e5_ssp126_tas_*.csv", pre_ind_tmep = 13.8):
    files = glob.glob(path)
    df = []
    for f in files:
        csv = pd.read_csv(f, header=None, index_col = 0)
        csv = csv.rename(columns={1:'tas'})
        csv.index.name = 'time'
        df.append(csv)
        df_2 = pd.concat(df) - 273.15 - pre_ind_tmep
        df_2.index = pd.to_datetime(df_2.index)

    return df_2

df_ipsl_26 = open_csv_timeseries("../projections_global_mean/ipsl-cm6a-lr_r1i1p1f1_w5e5_ssp126_tas_*.csv")
df_ipsl_85 = open_csv_timeseries("../projections_global_mean/ipsl-cm6a-lr_r1i1p1f1_w5e5_ssp585_tas_*.csv")


df_ukesm_26 = open_csv_timeseries("../projections_global_mean/ukesm1-0-ll_r1i1p1f2_w5e5_ssp126_tas_*.csv")
df_ukesm_85 = open_csv_timeseries("../projections_global_mean/ukesm1-0-ll_r1i1p1f2_w5e5_ssp585_tas_*.csv")


df_gfdl_26 = open_csv_timeseries("../projections_global_mean/gfdl-esm4_r1i1p1f1_w5e5_ssp126_tas_*.csv")
df_gfdl_85 = open_csv_timeseries("../projections_global_mean/gfdl-esm4_r1i1p1f1_w5e5_ssp585_tas_*.csv")

df_ipsl_26.groupby(df_ipsl_26.index.year)['tas'].transform('mean').plot()
df_ipsl_85.groupby(df_ipsl_85.index.year)['tas'].transform('mean').plot()
df_gfdl_26.groupby(df_gfdl_26.index.year)['tas'].transform('mean').plot()
df_gfdl_85.groupby(df_gfdl_85.index.year)['tas'].transform('mean').plot()
df_ukesm_26.groupby(df_ukesm_26.index.year)['tas'].transform('mean').plot()
df_ukesm_85.groupby(df_ukesm_85.index.year)['tas'].transform('mean').plot()
plt.show()

df_ipsl_26_5y = df_ipsl_26.rolling('1826d').mean()
df_ipsl_85_5y = df_ipsl_85.rolling('1826d').mean()
df_gfdl_26_5y = df_gfdl_26.rolling('1826d').mean()
df_gfdl_85_5y = df_gfdl_85.rolling('1826d').mean()
df_ukesm_26_5y = df_ukesm_26.rolling('1826d').mean()
df_ukesm_85_5y = df_ukesm_85.rolling('1826d').mean()

plt.figure(figsize = (6,6),dpi=300)
# plt.plot(df_ipsl_26.rolling('1826d').mean(), label = 'ipsl_26')
# plt.plot(df_ipsl_85.rolling('1826d').mean(), label = 'ipsl_85')
# plt.plot(df_gfdl_26.rolling('1826d').mean(), label = 'gfdl_26')
# plt.plot(df_gfdl_85.rolling('1826d').mean(), label = 'gfdl_85')
plt.plot(df_ukesm_26_5y['2020':'2100-12-31'], label = 'ukesm_26')
plt.plot(df_ukesm_85_5y['2020':'2100-12-31'], label = 'ukesm_85')
# plt.axhline(2.5, linestyle = 'dashed')
# plt.axhline(3, linestyle = 'dashed', c = 'red')
plt.axvline(len(df_ukesm_26)*0.93, linestyle = 'dashed', c = 'black')
plt.yticks((np.arange(0,6,0.5)))
plt.ylim(0,6)
plt.legend()
plt.show()


print('2 degree at', df_ipsl_26_5y[df_ipsl_26_5y['tas'] >= 2].index[0] if np.max(df_ipsl_26_5y['tas']) >= 2 else 'No 2 degree')
print('2 degree at', df_ipsl_85_5y[df_ipsl_85_5y['tas'] >= 2].index[0] if np.max(df_ipsl_85_5y['tas']) >= 2 else 'No 2 degree')
print('2 degree at', df_gfdl_26_5y[df_gfdl_26_5y['tas'] >= 2].index[0] if np.max(df_gfdl_26_5y['tas']) >= 2 else 'No 2 degree')
print('2 degree at', df_gfdl_85_5y[df_gfdl_85_5y['tas'] >= 2].index[0] if np.max(df_gfdl_85_5y['tas']) >= 2 else 'No 2 degree')
print('2 degree at', df_ukesm_26_5y[df_ukesm_26_5y['tas'] >= 2].index[0] if np.max(df_ukesm_26_5y['tas']) >= 2 else 'No 2 degree')
print('2 degree at', df_ukesm_85_5y[df_ukesm_85_5y['tas'] >= 2].index[0] if np.max(df_ukesm_85_5y['tas']) >= 2 else 'No 2 degree')

print('3 degree at', df_ipsl_26_5y[df_ipsl_26_5y['tas'] >= 3].index[0] if np.max(df_ipsl_26_5y['tas']) >= 3 else 'No 3 degree')
print('3 degree at',df_ipsl_85_5y[df_ipsl_85_5y['tas'] >= 3].index[0] if np.max(df_ipsl_85_5y['tas']) >= 3 else 'No 3 degree')
print('3 degree at',df_gfdl_26_5y[df_gfdl_26_5y['tas'] >= 3].index[0] if np.max(df_gfdl_26_5y['tas']) >= 3 else 'No 3 degree')
print('3 degree at',df_gfdl_85_5y[df_gfdl_85_5y['tas'] >= 3].index[0] if np.max(df_gfdl_85_5y['tas']) >= 3 else 'No 3 degree')
print('3 degree at',df_ukesm_26_5y[df_ukesm_26_5y['tas'] >= 3].index[0] if np.max(df_ukesm_26_5y['tas']) >= 3 else 'No 3 degree')
print('3 degree at',df_ukesm_85_5y[df_ukesm_85_5y['tas'] >= 3].index[0] if np.max(df_ukesm_85_5y['tas']) >= 3 else 'No 3 degree')

# 2 degree: IPSL_26_2070, IPSL_85_2030, GFDL_85_2050, UKESM_26_2030, UKESM_85_2030
# 3 degree:  IPSL_85_2050, GFDL_85_2070, UKESM_85_2050

#%%


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



shifters_proj = pd.DataFrame( [[0.7732478380203247, 0.814445436000824, 0.745225727558136], 
                               [0.8048002123832703, 0.7392540574073792, 0.8095159530639648], 
                               [0.8223055005073547, 0.751900851726532, 0.5870298743247986],
                               [0.8127191066741943, 0.9065349698066711, 0.8539149761199951],
                               [0.8059989809989929, 0.7766252160072327, 0.7489902377128601],
                               [0.8544405102729797, 0.8020848631858826, 0.8079625964164734 ]], 
                             index = ['UKESM_585','UKESM_126','GFDL_585', 'GFDL_126', 'IPSL_585', 'IPSL_126'], columns = [2030,2050,2070]).T


sns.barplot(x= shifters_proj.index, y=shifters_proj['UKESM_585'])
plt.show()

shifters_proj = (shifters_proj - 1) * 100
shock_2012 = (0.87534434 - 1) * 100

ax = sns.kdeplot(x= test_plot["VALUE"], hue = test_plot["rcp"], fill = True)
plt.title(f"Variable: {df_soy_glob_subset_shock_years['VAR_ID'].values[0]}")
ax.legend_.set_bbox_to_anchor((0.05, 0.99))
ax.legend_._set_loc(2)
plt.title(f"Shock for variable: {test_plot['VAR_ID'].values[0]} in {test_plot['ANYREGION'].str.slice(0,-3).values[0]}")
plt.xlabel('Shock impact (%)')
plt.show()

shifters_proj['shock_year'] = shifters_proj.index

    

shifters_proj_long = pd.melt(shifters_proj, id_vars = ['shock_year'], ignore_index=False )
shifters_proj_long['rcp'] = shifters_proj_long['variable'].str.split('_').str[1]
shifters_proj_long['gcm'] = shifters_proj_long['variable'].str.split('_').str[0]

# Bar plots yield shocks
plt.figure(figsize = (5,5),dpi=300)
sns.barplot(x= shifters_proj_long.index, y=shifters_proj_long['value'], hue = shifters_proj_long['rcp'], hue_order = ['126','585'] )
plt.axhline(shock_2012, linestyle = 'dashed', c = 'k', label = '2012')
plt.ylabel('Shock (%)')
plt.title('Soybean yield shock')
plt.legend()
plt.show()

plt.figure(figsize = (5,5),dpi=300)
sns.barplot(x= shifters_proj_long.rcp, y=shifters_proj_long['value'], hue = shifters_proj_long.index, order = ['126','585'])
plt.axhline(shock_2012, linestyle = 'dashed', c = 'k', label = '2012')
plt.ylabel('Shock (%)')
plt.title('Soybean yield shock')
plt.legend()
plt.show()


# Test only for two scenarios:
shifters_proj_long_two = shifters_proj_long[((shifters_proj_long['variable'] == 'IPSL_585')|(shifters_proj_long['variable'] == 'IPSL_126')) & (shifters_proj_long['shock_year'] == 2050)]

shifters_proj_long_two['GW degree'] =np.where(shifters_proj_long_two['rcp'] == '126', '2°C', np.where(shifters_proj_long_two['rcp'] =='585', '3°C', ''))
shifters_proj_long_two=shifters_proj_long_two[['GW degree','value']]
shifters_proj_long_two = shifters_proj_long_two.append( pd.DataFrame([['Hist',shock_2012]], columns=['GW degree','value'], index=[2050]) )
shifters_proj_long_two['Country'] ='USA'

plt.figure(figsize = (5,5),dpi=300)
sns.barplot(x= shifters_proj_long_two['GW degree'], y=shifters_proj_long_two['value'], order = ['Hist','2°C', '3°C'])
plt.ylabel('Shock (%)')
plt.title('Soybean yield shock')
plt.show()



#%% Brazil

# # # UKESM 585
# year_min: 2016 year_max: 2044 min_pos 2033 shifter [2033.0, 0.8912345767021179]
# year_min: 2043 year_max: 2071 min_pos 2067 shifter [2067.0, 0.8426542282104492]
# year_min: 2070 year_max: 2098 min_pos 2094 shifter [2094.0, 0.8382395505905151]

# # # UKESM 126
# year_min: 2016 year_max: 2044 min_pos 2018 shifter [2018.0, 0.8274495601654053]
# year_min: 2043 year_max: 2071 min_pos 2043 shifter [2043.0, 0.924953043460846]
# year_min: 2070 year_max: 2098 min_pos 2096 shifter [2096.0, 0.9216389060020447]


# # # GFDL model 585
# year_min: 2016 year_max: 2044 min_pos 2020 shifter [2020.0, 0.8650361895561218]
# year_min: 2043 year_max: 2071 min_pos 2048 shifter [2048.0, 0.6792255640029907]
# year_min: 2070 year_max: 2098 min_pos 2095 shifter [2095.0, 0.7575809955596924]

# # # GFDL model 126
# year_min: 2016 year_max: 2044 min_pos 2025 shifter [2025.0, 0.7959718704223633]
# year_min: 2043 year_max: 2071 min_pos 2060 shifter [2060.0, 0.8679850101470947]
# year_min: 2070 year_max: 2098 min_pos 2089 shifter [2089.0, 0.9291833639144897]

# # # IPSL model 585
# year_min: 2016 year_max: 2044 min_pos 2035 shifter [2035.0, 0.8600314259529114]
# year_min: 2043 year_max: 2071 min_pos 2046 shifter [2046.0, 0.9106815457344055]
# year_min: 2070 year_max: 2098 min_pos 2077 shifter [2077.0, 0.9405683875083923]

# # # IPSL model 126
# year_min: 2016 year_max: 2044 min_pos 2031 shifter [2031.0, 0.8844877481460571]
# year_min: 2043 year_max: 2071 min_pos 2049 shifter [2049.0, 0.7847091555595398]
# year_min: 2070 year_max: 2098 min_pos 2071 shifter [2071.0, 0.9701869487762451]


shifters_proj_br = pd.DataFrame( [[0.8912345767021179, 0.8426542282104492, 0.8382395505905151], 
                               [0.8274495601654053, 0.924953043460846,0.9216389060020447], 
                               [0.8650361895561218, 0.6792255640029907, 0.7575809955596924],
                               [0.7959718704223633, 0.8679850101470947, 0.9291833639144897],
                               [ 0.8600314259529114, 0.9106815457344055, 0.9405683875083923],
                               [ 0.8844877481460571, 0.7847091555595398, 0.9701869487762451 ]], 
                             index = ['UKESM_585','UKESM_126','GFDL_585', 'GFDL_126', 'IPSL_585', 'IPSL_126'], columns = [2030,2050,2070]).T




shifters_proj_br = (shifters_proj_br - 1) * 100
shock_2012 = (0.84798976 - 1) * 100


shifters_proj_br['shock_year'] = shifters_proj_br.index

    

shifters_proj_long = pd.melt(shifters_proj_br, id_vars = ['shock_year'], ignore_index=False )
shifters_proj_long['rcp'] = shifters_proj_long['variable'].str.split('_').str[1]
shifters_proj_long['gcm'] = shifters_proj_long['variable'].str.split('_').str[0]

# Bar plots yield shocks
plt.figure(figsize = (5,5),dpi=300)
sns.barplot(x= shifters_proj_long.index, y=shifters_proj_long['value'], hue = shifters_proj_long['rcp'], hue_order = ['126','585'] )
plt.axhline(shock_2012, linestyle = 'dashed', c = 'k', label = '2012')
plt.ylabel('Shock (%)')
plt.title('Soybean yield shock')
plt.legend()
plt.show()

plt.figure(figsize = (5,5),dpi=300)
sns.barplot(x= shifters_proj_long.rcp, y=shifters_proj_long['value'], hue = shifters_proj_long.index, order = ['126','585'])
plt.axhline(shock_2012, linestyle = 'dashed', c = 'k', label = '2012')
plt.ylabel('Shock (%)')
plt.title('Soybean yield shock')
plt.legend()
plt.show()


# Test only for two scenarios:
shifters_proj_long_two_br = shifters_proj_long[((shifters_proj_long['variable'] == 'IPSL_585')|(shifters_proj_long['variable'] == 'IPSL_126')) & (shifters_proj_long['shock_year'] == 2050)]

shifters_proj_long_two_br['GW degree'] =np.where(shifters_proj_long_two_br['rcp'] == '126', '2°C', np.where(shifters_proj_long_two_br['rcp'] =='585', '3°C', ''))
shifters_proj_long_two_br=shifters_proj_long_two_br[['GW degree','value']]
shifters_proj_long_two_br = shifters_proj_long_two_br.append( pd.DataFrame([['Hist',shock_2012]], columns=['GW degree','value'], index=[2050]) )
shifters_proj_long_two_br['Country'] ='Brazil'

plt.figure(figsize = (5,5),dpi=300)
sns.barplot(x= shifters_proj_long_two_br['GW degree'], y=shifters_proj_long_two['value'], order = ['Hist','2°C', '3°C'])
plt.ylabel('Shock (%)')
plt.title('Soybean yield shock')
plt.show()

shifters_proj_long_countries = pd.concat([shifters_proj_long_two,shifters_proj_long_two_br], axis = 0)


plt.figure(figsize = (5,5),dpi=300)
sns.barplot(x= shifters_proj_long_countries['GW degree'], y=shifters_proj_long_countries['value'], hue=shifters_proj_long_countries['Country'], order = ['Hist','2°C', '3°C'])
plt.ylabel('Shock (%)')
plt.title('Soybean yield shock')
plt.show()