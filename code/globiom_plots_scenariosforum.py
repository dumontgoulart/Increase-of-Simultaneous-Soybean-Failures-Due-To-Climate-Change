# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 17:10:26 2022

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

os.chdir('C:/Users/morenodu/OneDrive - Stichting Deltares/Documents/PhD/paper_hybrid_agri/data')

#%%
prod_soya_br = pd.read_csv('scenarios_forum_simulations/ga_simulations_physical.csv') #, usecols = ['IEA_SCEN', 'VAR_ID', "ANYREGION", 'VAR_UNIT', 'ALLYEAR', 'VALUE']
# prod_soya_br.set_index('ALLYEAR', inplace=True)

label_shock = 'Yield (t/ha)'


prod_soya_br_subset = prod_soya_br[ (prod_soya_br['VAR_ID'] == 'PROD') & (prod_soya_br['VAR_UNIT'] == '1000 t dm') ] 


def plot_hues(dataframe, list_scen):
    dataframe = dataframe[dataframe['IEA_SCEN'].isin(list_scen)]
    
    plt.figure(figsize = (10,10),dpi=200)
    dataframe.plot(kind = 'line')
    plt.ylabel(label_shock)
    plt.xlabel('Years')
    plt.legend()
    plt.show()
    
    plt.figure(figsize = (10,10),dpi=200)
    sns.lineplot(data=dataframe, x='ALLYEAR', y='VALUE', hue='IEA_SCEN')
    plt.title(f"Variable: {prod_soya_br_subset['VAR_ID'].values[0]}")
    plt.ylabel(label_shock)
    plt.xlabel('Years')
    plt.show()


def plot_for_scenarios(conditions, title = None):   
    prod_soya_br_subset_2 = prod_soya_br[ conditions ] 

    prod_soya_br_subset_model = prod_soya_br_subset_2.copy()
    # prod_soya_br_subset_model = prod_soya_br_subset_model.loc[[2010,2011,2050,2051]]
    prod_soya_br_subset_model = prod_soya_br_subset_model[prod_soya_br_subset_model['ALLYEAR'].isin([2010,2011,2050,2051])]
    prod_soya_br_subset_model['ANYREGION'] = prod_soya_br_subset_model['ANYREGION'].str.slice(0,-3)
    
    prod_soya_br_subset_model['VALUE_div'] = prod_soya_br_subset_model.loc[:,'VALUE'].div(prod_soya_br_subset_model.loc[:,'VALUE'].shift(1)) #
    prod_soya_br_subset_model['VALUE_def'] = prod_soya_br_subset_model['VALUE'].diff()

    prod_soya_br_subset_model_dif_sub = prod_soya_br_subset_model[prod_soya_br_subset_model['ALLYEAR'].astype(str).str.strip().str[-1] == '1']
    prod_soya_br_subset_model_dif_sub['VALUE_div'] = (prod_soya_br_subset_model_dif_sub['VALUE_div'] - 1) * 100
    
    
    prod_soya_br_subset_model_dif_sub['model'] = prod_soya_br_subset_model_dif_sub.IEA_SCEN.str.split('MIN_').str[-1]
    prod_soya_br_subset_model_dif_sub['gcm'] = prod_soya_br_subset_model_dif_sub.IEA_SCEN.str.split('_').str[0]
    prod_soya_br_subset_model_dif_sub['rcp'] = prod_soya_br_subset_model_dif_sub.IEA_SCEN.str.split('_').str[1].str.slice(3,6).replace([np.nan, '2p6','8p5'],['Hist', '2.6','8.5'])
    prod_soya_br_subset_model_dif_sub['shock_year'] = prod_soya_br_subset_model_dif_sub.IEA_SCEN.str.split('_').str[2].str.slice(0,2)
    prod_soya_br_subset_model_dif_sub['shock_year'] = prod_soya_br_subset_model_dif_sub['shock_year'].replace(['30','50','70'], [2031,2051,2071])
    prod_soya_br_subset_model_dif_sub = prod_soya_br_subset_model_dif_sub[prod_soya_br_subset_model_dif_sub['gcm'].isin(['scenRCPref', 'UKESM1-0-LL'])]

    # prod_soya_br_subset_model_dif_sub = prod_soya_br_subset_model_dif_sub[prod_soya_br_subset_model_dif_sub['ALLYEAR'] == prod_soya_br_subset_model_dif_sub['shock_year']]
    
    
    df_shock_hist_yexo = prod_soya_br_subset_model_dif_sub[(prod_soya_br_subset_model_dif_sub['ALLYEAR'] == 2011) & (prod_soya_br_subset_model_dif_sub['model']  == 'scenRCPref') & (prod_soya_br_subset_model_dif_sub['BIOENSCEN']  == 'HistShock')]
    df_shock_fut_yexo = prod_soya_br_subset_model_dif_sub[(prod_soya_br_subset_model_dif_sub['ALLYEAR'] == 2051) & (prod_soya_br_subset_model_dif_sub['model']  == 'hybrid')]
    
    df_shock_all_yexo = pd.concat([df_shock_hist_yexo, df_shock_fut_yexo])
    
    sns.barplot(x=df_shock_all_yexo["ANYREGION"], y=df_shock_all_yexo["VALUE_div"], hue = df_shock_all_yexo['rcp'], hue_order = ['Hist','2.6','8.5'])
    plt.title(f"{df_shock_all_yexo['VAR_ID'].values[0]} shocks")
    if title is not None:
        plt.title(f"{title} shocks")

    plt.ylabel('Magnitude (%)')
    plt.xlabel('Country')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon = False, ncol = 5)
    plt.tight_layout()
    plt.show()
    
    sns.barplot(x=df_shock_all_yexo["ANYREGION"], y=df_shock_all_yexo["VALUE_def"], hue = df_shock_all_yexo['rcp'], hue_order = ['Hist','2.6','8.5'])
    plt.title(f"{df_shock_all_yexo['VAR_ID'].values[0]} shocks")
    if title is not None:
        plt.title(f"{title} shocks")

    plt.ylabel('Deficit (ton/ha)')
    plt.xlabel('Country')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon = False, ncol = 5)
    plt.tight_layout()
    plt.show()
    
    
plot_for_scenarios( (prod_soya_br['VAR_ID'] == 'YEXO') & (prod_soya_br['VAR_UNIT'] == 'fm t/ha') , title = 'Yield')
plot_for_scenarios( (prod_soya_br['VAR_ID'] == 'PROD') & (prod_soya_br['VAR_UNIT'] == '1000 t dm') )
plot_for_scenarios( (prod_soya_br['VAR_ID'] == 'CONS') )
plot_for_scenarios( (prod_soya_br['VAR_ID'] == 'XPRP'))
plot_for_scenarios( (prod_soya_br['VAR_ID'] == 'FOOD'))
plot_for_scenarios( (prod_soya_br['VAR_ID'] == 'FEED'))



# Second trial

prod_soya_br_change = pd.read_csv('scenarios_forum_simulations/ga_simulations_physical_change.csv') #, usecols = ['IEA_SCEN', 'VAR_ID', "ANYREGION", 'VAR_UNIT', 'ALLYEAR', 'VALUE']

def plot_for_scenarios_change(conditions):   
    prod_soya_br_subset_2 = prod_soya_br_change[ conditions ] 

    prod_soya_br_subset_model = prod_soya_br_subset_2.copy()
            
    
    prod_soya_br_subset_model['model'] = prod_soya_br_subset_model.IEA_SCEN.str.split('MIN_').str[-1]
    prod_soya_br_subset_model['gcm'] = prod_soya_br_subset_model.IEA_SCEN.str.split('_').str[0]
    prod_soya_br_subset_model['rcp'] = prod_soya_br_subset_model.IEA_SCEN.str.split('_').str[1].str.slice(3,6)
    prod_soya_br_subset_model['rcp'] = prod_soya_br_subset_model['rcp'].replace(['2p6','8p5'],['2.6','8.5'])
    prod_soya_br_subset_model = prod_soya_br_subset_model[(prod_soya_br_subset_model['ANYREGION'].isin(['Brazil', 'Argentina', 'USA']))]
    prod_soya_br_subset_model['VALUE'] = (prod_soya_br_subset_model['VALUE'] - 1) * 100
                                                          
        
    df_shock_fut_yexo = prod_soya_br_subset_model[(prod_soya_br_subset_model['ALLYEAR'] == 2050) & (prod_soya_br_subset_model['model']  == 'hybrid')]
    
    
    sns.barplot(x=df_shock_fut_yexo["ANYREGION"], y = df_shock_fut_yexo["VALUE"], hue = df_shock_fut_yexo['rcp'], hue_order = ['2.6','8.5'])
    plt.title(f"{df_shock_fut_yexo['VAR_ID'].values[0]} anomalies")
    plt.ylabel('Yield magnitude (%)')
    plt.xlabel('Country')
    plt.tight_layout()
    plt.show()

plot_for_scenarios_change( (prod_soya_br_change['VAR_ID'] == 'PROD') )
plot_for_scenarios_change( (prod_soya_br_change['VAR_ID'] == 'YEXO') & (prod_soya_br_change['VAR_UNIT'] == 'fm t/ha') )
plot_for_scenarios_change( (prod_soya_br_change['VAR_ID'] == 'CONS') )
plot_for_scenarios_change( (prod_soya_br_change['VAR_ID'] == 'XPRP'))
plot_for_scenarios_change( (prod_soya_br_change['VAR_ID'] == 'FOOD'))
plot_for_scenarios_change( (prod_soya_br_change['VAR_ID'] == 'FEED'))




# =============================================================================
# Load bilateral flows
# =============================================================================
df_bilateral_flow = pd.read_csv('scenarios_forum_simulations/bilateral_flows_europe.csv') #, usecols = ['IEA_SCEN', 'VAR_ID', "ANYREGION", 'VAR_UNIT', 'ALLYEAR', 'VALUE']

df_bilateral_flow_copy = df_bilateral_flow.copy()
df_bilateral_flow_copy = df_bilateral_flow_copy[df_bilateral_flow_copy['ALLYEAR'].isin([2010,2011,2050,2051])]


df_bilateral_flow_copy['model'] = df_bilateral_flow_copy.IEA_SCEN.str.split('MIN_').str[-1]
df_bilateral_flow_copy['gcm'] = df_bilateral_flow_copy.IEA_SCEN.str.split('_').str[0]
df_bilateral_flow_copy['rcp'] = df_bilateral_flow_copy.IEA_SCEN.str.split('_').str[1].str.slice(3,6).replace([np.nan, '2p6','8p5'],['Hist', '2.6','8.5'])
df_bilateral_flow_copy = df_bilateral_flow_copy[df_bilateral_flow_copy['model'].isin(['hybrid', 'scenRCPref'])]

df_bilateral_flow_copy['VALUE_div'] = df_bilateral_flow_copy['VALUE'].diff()# 


df_bilateral_flow_copy['shock_year'] = df_bilateral_flow_copy.IEA_SCEN.str.split('_').str[2].str.slice(0,2)
df_bilateral_flow_copy['shock_year'] = df_bilateral_flow_copy['shock_year'].replace(['30','50','70', np.nan], [2031,2051,2071,2011])
df_bilateral_flow_copy = df_bilateral_flow_copy[df_bilateral_flow_copy['ALLYEAR'] == df_bilateral_flow_copy['shock_year']]

df_bilateral_flow_copy = df_bilateral_flow_copy[df_bilateral_flow_copy['export_country'].isin(['USAReg','ArgentinaReg','BrazilReg'])]

df_bilateral_flow_hist = df_bilateral_flow_copy[(df_bilateral_flow_copy['rcp'] == 'Hist') & (df_bilateral_flow_copy['BIOENSCEN'] == 'HistShock')]
df_bilateral_flow_26 = df_bilateral_flow_copy[df_bilateral_flow_copy['rcp'] == '2.6']
df_bilateral_flow_85 = df_bilateral_flow_copy[df_bilateral_flow_copy['rcp'] == '8.5']

df_bilateral_flow_copy = pd.concat([df_bilateral_flow_hist, df_bilateral_flow_26, df_bilateral_flow_85])
# df_bilateral_flow_copy = df_bilateral_flow_copy[df_bilateral_flow_copy['gcm'] == 'UKESM1-0-LL']



sns.barplot(x=df_bilateral_flow_copy["BIOENSCEN"], y=df_bilateral_flow_copy["VALUE_div"], hue = df_bilateral_flow_copy['rcp'], hue_order = ['Hist','2.6','8.5'])
plt.title(f"Change in EU soybean imports")
plt.ylabel('Yield magnitude (%)')
plt.xlabel('Country')
plt.tight_layout()
plt.show()

df_bilateral_flow_copy_plot = df_bilateral_flow_copy[df_bilateral_flow_copy['rcp'].isin(['Hist', '2.6'])]
sns.barplot(x=df_bilateral_flow_copy_plot["export_country"], y=df_bilateral_flow_copy_plot["VALUE_div"], hue = df_bilateral_flow_copy_plot['BIOENSCEN'])
plt.title(f"Change in EU soybean imports")
plt.ylabel('Yield magnitude (%)')
plt.xlabel('Country')
plt.tight_layout()
plt.show()

df_bilateral_flow_copy_plot = df_bilateral_flow_copy[df_bilateral_flow_copy['rcp'].isin(['Hist', '8.5'])]
sns.barplot(x=df_bilateral_flow_copy_plot["export_country"], y=df_bilateral_flow_copy_plot["VALUE_div"], hue = df_bilateral_flow_copy_plot['BIOENSCEN'])
plt.title(f"Change in EU soybean imports")
plt.ylabel('Yield magnitude (%)')
plt.xlabel('Country')
plt.tight_layout()
plt.show()


sns.barplot(x=df_bilateral_flow_26["BIOENSCEN"], y=df_bilateral_flow_copy["VALUE_div"], hue = df_bilateral_flow_copy['export_country'])
plt.title(f"Change in EU soybean imports RCP 2.6")
plt.ylabel('Anomaly (1000 t)')
plt.xlabel('Country')
plt.tight_layout()
plt.show()


sns.barplot(x=df_bilateral_flow_85["BIOENSCEN"], y=df_bilateral_flow_copy["VALUE_div"], hue = df_bilateral_flow_copy['export_country'])
plt.title(f"Change in EU soybean imports RCP 8.5")
plt.ylabel('Anomaly (1000 t)')
plt.xlabel('Country')
plt.tight_layout()
plt.show()



# =============================================================================
# Load bilateral flows
# =============================================================================
df_bilateral_flow = pd.read_csv('scenarios_forum_simulations/bilateral_flows_europe_change.csv') #, usecols = ['IEA_SCEN', 'VAR_ID', "ANYREGION", 'VAR_UNIT', 'ALLYEAR', 'VALUE']

df_bilateral_flow_copy = df_bilateral_flow.copy()
# df_bilateral_flow_copy = df_bilateral_flow_copy[df_bilateral_flow_copy['ALLYEAR'].isin([2010,2011,2050,2051])]


df_bilateral_flow_copy['model'] = df_bilateral_flow_copy.IEA_SCEN.str.split('MIN_').str[-1]
df_bilateral_flow_copy['gcm'] = df_bilateral_flow_copy.IEA_SCEN.str.split('_').str[0]
df_bilateral_flow_copy['rcp'] = df_bilateral_flow_copy.IEA_SCEN.str.split('_').str[1].str.slice(3,6).replace(['2p6','8p5'],['2.6','8.5'])
df_bilateral_flow_copy = df_bilateral_flow_copy[df_bilateral_flow_copy['model'].isin(['hybrid'])]
df_bilateral_flow_copy = df_bilateral_flow_copy[df_bilateral_flow_copy['gcm'] == 'UKESM1-0-LL']
df_bilateral_flow_copy['export_country'] = df_bilateral_flow_copy['export_country'].str.slice(0,-3)


df_bilateral_flow_copy = df_bilateral_flow_copy[df_bilateral_flow_copy['export_country'].isin(['USA','Argentina','Brazil'])]
df_bilateral_flow_copy_default = df_bilateral_flow_copy[df_bilateral_flow_copy['BIOENSCEN'] == 'Default']
df_bilateral_flow_26 = df_bilateral_flow_copy[df_bilateral_flow_copy['rcp'] == '2.6']
df_bilateral_flow_85 = df_bilateral_flow_copy[df_bilateral_flow_copy['rcp'] == '8.5']


sns.barplot(x=df_bilateral_flow_copy_default["export_country"].sort_values(), y=df_bilateral_flow_copy_default["VALUE"], hue = df_bilateral_flow_copy_default['rcp'].sort_values())
plt.title(f"Change in EU soybean imports")
plt.ylabel('Anomaly (1000 t)')
plt.xlabel('RCP')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon = False, ncol = 5)
plt.tight_layout()
plt.show()

sns.barplot(x=df_bilateral_flow_26["export_country"].sort_values(), y=df_bilateral_flow_copy["VALUE"], hue = df_bilateral_flow_copy['BIOENSCEN'])
plt.title(f"Change in EU soybean imports RCP 2.6")
plt.ylabel('Anomaly (1000 t)')
plt.xlabel('Socio-economic scenario')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon = False, ncol = 5)
plt.tight_layout()
plt.show()


sns.barplot(x=df_bilateral_flow_85["export_country"].sort_values(), y=df_bilateral_flow_copy["VALUE"], hue = df_bilateral_flow_copy['BIOENSCEN'])
plt.title(f"Change in EU soybean imports RCP 8.5")
plt.ylabel('Anomaly (1000 t)')
plt.xlabel('Socio-economic scenario')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon = False, ncol = 5)
plt.tight_layout()
plt.show()


sns.barplot(x=df_bilateral_flow_copy["export_country"].sort_values(), y=df_bilateral_flow_copy["VALUE"], hue = df_bilateral_flow_copy['BIOENSCEN'])
plt.title(f"Change in EU soybean imports")
plt.ylabel('Anomaly (1000 t)')
plt.xlabel('Socio-economic scenario')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon = False, ncol = 5)
plt.tight_layout()
plt.show()







# =============================================================================
# Final socio-economic impacts inth eEU
# =============================================================================
df_eu_impacts = pd.read_csv('scenarios_forum_simulations/ga_simulations_EU.csv') #, usecols = ['IEA_SCEN', 'VAR_ID', "ANYREGION", 'VAR_UNIT', 'ALLYEAR', 'VALUE']

df_eu_impacts_copy = df_eu_impacts.copy()

df_eu_impacts_copy['model'] = df_eu_impacts_copy.IEA_SCEN.str.split('MIN_').str[-1]
df_eu_impacts_copy['gcm'] = df_eu_impacts_copy.IEA_SCEN.str.split('_').str[0]
df_eu_impacts_copy['rcp'] = df_eu_impacts_copy.IEA_SCEN.str.split('_').str[1].str.slice(3,6).replace([np.nan, '2p6','8p5'],['Hist','2.6','8.5'])
df_eu_impacts_copy = df_eu_impacts_copy[df_eu_impacts_copy['model'].isin(['hybrid', 'scenRCPref'])]
df_eu_impacts_copy = df_eu_impacts_copy[df_eu_impacts_copy['gcm'].isin(['UKESM1-0-LL','scenRCPref'])]
df_eu_impacts_copy_eu = df_eu_impacts_copy[df_eu_impacts_copy['ANYREGION'] == 'Europe']
df_eu_impacts_copy_eu['VALUE'] = (df_eu_impacts_copy_eu['VALUE'] - 1 )* 100


trial = df_eu_impacts_copy_eu[(df_eu_impacts_copy_eu['VAR_ID'] == 'XPRP') & (df_eu_impacts_copy_eu['Item'] == 'Soya')]

sns.barplot(x=trial["rcp"], order=['2.6', '8.5'], y=trial["VALUE"], hue = trial['BIOENSCEN'])
plt.axhline(y= list(trial[(trial['model']=='scenRCPref')&(trial['BIOENSCEN']=='Default')]['VALUE']), color = 'black', linestyle = 'dashed', label = 'Hist')
plt.title(f"Shock on soybean prices in the EU")
plt.ylabel('Price change (%)')
plt.xlabel('RCP')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon = False, ncol = 5)
plt.tight_layout()
plt.show()

trial = df_eu_impacts_copy_eu[(df_eu_impacts_copy_eu['VAR_ID'] == 'CONS') & (df_eu_impacts_copy_eu['Item'] == 'Soya')]
sns.barplot(x=trial["rcp"], order=['2.6', '8.5'], y=trial["VALUE"], hue = trial['BIOENSCEN'])
plt.axhline(y= list(trial[(trial['model']=='scenRCPref')&(trial['BIOENSCEN']=='Default')]['VALUE']), color = 'black', linestyle = 'dashed', label = 'Hist')
plt.title(f"Shock on soybean consumption in the EU")
plt.ylabel('Consumption change (%)')
plt.xlabel('RCP')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon = False, ncol = 5)
plt.tight_layout()
plt.show()


trial = df_eu_impacts_copy_eu[(df_eu_impacts_copy_eu['VAR_ID'] == 'FOOD') & (df_eu_impacts_copy_eu['Item'] == 'Soya')]
sns.barplot(x=trial["rcp"], order=['2.6', '8.5'], y=trial["VALUE"], hue = trial['BIOENSCEN'])
plt.axhline(y= list(trial[(trial['model']=='scenRCPref')&(trial['BIOENSCEN']=='Default')]['VALUE']), color = 'black', linestyle = 'dashed', label = 'Hist')
plt.title(f"Shock on soybean human consumption in the EU")
plt.ylabel('Consumption change (%)')
plt.xlabel('RCP')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon = False, ncol = 5)
plt.tight_layout()
plt.show()


trial = df_eu_impacts_copy_eu[(df_eu_impacts_copy_eu['VAR_ID'] == 'FEED') & (df_eu_impacts_copy_eu['Item'] == 'Soya')]
sns.barplot(x=trial["rcp"], order=['2.6', '8.5'], y=trial["VALUE"], hue = trial['BIOENSCEN'])
plt.axhline(y= list(trial[(trial['model']=='scenRCPref')&(trial['BIOENSCEN']=='Default')]['VALUE']), color = 'black', linestyle = 'dashed', label = 'Hist')
plt.title(f"Shock on animal soybean consumption in the EU")
plt.ylabel('Consumption change (%)')
plt.xlabel('RCP')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon = False, ncol = 5)
plt.tight_layout()
plt.show()


trial = df_eu_impacts_copy_eu[(df_eu_impacts_copy_eu['VAR_ID'] == 'XPRP') & (df_eu_impacts_copy_eu['Item'] == 'Meat')]
sns.barplot(x=trial["rcp"], order=['2.6', '8.5'], y=trial["VALUE"], hue = trial['BIOENSCEN'])
plt.axhline(y= list(trial[(trial['model']=='scenRCPref')&(trial['BIOENSCEN']=='Default')]['VALUE']), color = 'black', linestyle = 'dashed', label = 'Hist')
plt.title(f"Shock on meat prices in the EU")
plt.ylabel('Price change (%)')
plt.xlabel('RCP')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon = False, ncol = 5)
plt.tight_layout()
plt.show()

trial = df_eu_impacts_copy_eu[(df_eu_impacts_copy_eu['VAR_ID'] == 'CONS') & (df_eu_impacts_copy_eu['Item'] == 'Meat')]
sns.barplot(x=trial["rcp"], order=['2.6', '8.5'], y=trial["VALUE"], hue = trial['BIOENSCEN'])
plt.axhline(y= list(trial[(trial['model']=='scenRCPref')&(trial['BIOENSCEN']=='Default')]['VALUE']), color = 'black', linestyle = 'dashed', label = 'Hist')
plt.title(f"Shock on meat consumption in the EU")
plt.ylabel('Consumption change (%)')
plt.xlabel('RCP')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon = False, ncol = 5)
plt.tight_layout()
plt.show()


trial = df_eu_impacts_copy_eu[(df_eu_impacts_copy_eu['VAR_ID'] == 'FOOD') & (df_eu_impacts_copy_eu['Item'] == 'Meat')]
sns.barplot(x=trial["rcp"], order=['2.6', '8.5'], y=trial["VALUE"], hue = trial['BIOENSCEN'])
plt.axhline(y= list(trial[(trial['model']=='scenRCPref')&(trial['BIOENSCEN']=='Default')]['VALUE']), color = 'black', linestyle = 'dashed', label = 'Hist')
plt.title(f"Shock on meat human consumption in the EU")
plt.ylabel('Consumption change (%)')
plt.xlabel('RCP')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon = False, ncol = 5)
plt.tight_layout()
plt.show()


#YEXO

#barplots 31-30
def pdf_plots(prod_soya_br_subset, chosen_model = None, co2_scenario = None): 
    prod_soya_br_subset_model = prod_soya_br_subset.copy()
    
    prod_soya_br_subset_model['model'] = prod_soya_br_subset_model.IEA_SCEN.str.split('MIN_').str[-1]
    prod_soya_br_subset_model['rcp'] = prod_soya_br_subset_model.IEA_SCEN.str.slice(13,16)
    prod_soya_br_subset_model['co2'] = prod_soya_br_subset_model.IEA_SCEN.str.slice(10,13)
    prod_soya_br_subset_model['co2'] = prod_soya_br_subset_model['co2'].replace(['rcp','noC'], ['default','2015co2'])
    
    prod_soya_br_subset_model_dif = prod_soya_br_subset_model[(prod_soya_br_subset_model['model'] == 'hybrid') | (prod_soya_br_subset_model['model'] == 'EPIC') | (prod_soya_br_subset_model['model'] == 'clim')]
    
    if chosen_model is not None:
        prod_soya_br_subset_model_dif = prod_soya_br_subset_model_dif[prod_soya_br_subset_model_dif['model'] == chosen_model]
    
    if co2_scenario is not None:
        prod_soya_br_subset_model_dif = prod_soya_br_subset_model_dif[prod_soya_br_subset_model_dif['co2'] == co2_scenario]
        
    prod_soya_br_subset_model_dif['VALUE'] = prod_soya_br_subset_model_dif.loc[:,'VALUE'].div(prod_soya_br_subset_model_dif.loc[:,'VALUE'].shift(1)) #prod_soya_br_subset_model_dif['VALUE'].diff()# 
    
    prod_soya_br_subset_model_dif_sub = prod_soya_br_subset_model_dif[prod_soya_br_subset_model_dif.index.astype(str).str.strip().str[-1] == '1']
    prod_soya_br_subset_model_dif_sub['VALUE'] = (prod_soya_br_subset_model_dif_sub['VALUE'] - 1) * 100
    
    
    sns.violinplot(x=prod_soya_br_subset_model_dif_sub["model"], y=prod_soya_br_subset_model_dif_sub["VALUE"])
    plt.show()
    
    sns.violinplot(x=prod_soya_br_subset_model_dif_sub["rcp"], y=prod_soya_br_subset_model_dif_sub["VALUE"])
    plt.show()
    
    sns.violinplot(x=prod_soya_br_subset_model_dif_sub.index, y=prod_soya_br_subset_model_dif_sub["VALUE"])
    plt.show()
    
    ax = sns.kdeplot(x=prod_soya_br_subset_model_dif_sub["VALUE"], hue =prod_soya_br_subset_model_dif_sub["model"], fill=True)
    ax.legend_.set_bbox_to_anchor((0.05, 0.99))
    ax.legend_._set_loc(2)
    plt.xlabel('Shock impact (%)')
    plt.title('Model')
    plt.show()
    
    
    ax = sns.kdeplot(x=prod_soya_br_subset_model_dif_sub["VALUE"], hue =prod_soya_br_subset_model_dif_sub.index, fill=True)
    ax.legend_.set_bbox_to_anchor((0.05, 0.99))
    ax.legend_._set_loc(2)
    plt.xlabel('Shock impact (%)')
    plt.title('Year')
    plt.show()
    
    ax = sns.kdeplot(x=prod_soya_br_subset_model_dif_sub["VALUE"], hue = prod_soya_br_subset_model_dif_sub["rcp"], fill=True)
    ax.legend_.set_bbox_to_anchor((0.05, 0.99))
    ax.legend_._set_loc(2)
    plt.xlabel('Shock impact (%)')
    plt.title('RCP')
    plt.show()


pdf_plots(prod_soya_br_subset,co2_scenario = 'default')

pdf_plots(prod_soya_br_subset, chosen_model = 'hybrid', co2_scenario = 'default')
pdf_plots(prod_soya_br_subset, chosen_model = 'hybrid', co2_scenario = '2015co2')


def shock_plots(prod_soya_br_subset, chosen_model = None, co2_scenario = None): 

    prod_soya_br_subset_model = prod_soya_br_subset.copy()
    
    prod_soya_br_subset_model = prod_soya_br_subset_model[prod_soya_br_subset_model['IEA_SCEN'].str.len() >20]
    prod_soya_br_subset_model['model'] = prod_soya_br_subset_model.IEA_SCEN.str.split('MIN_').str[-1]
    prod_soya_br_subset_model['rcp'] = prod_soya_br_subset_model.IEA_SCEN.str.slice(13,16)
    prod_soya_br_subset_model['co2'] = prod_soya_br_subset_model.IEA_SCEN.str.slice(10,13)
    prod_soya_br_subset_model['co2'] = prod_soya_br_subset_model['co2'].replace(['rcp','noC'], ['default','2015co2'])
    prod_soya_br_subset_model['shock_year'] =prod_soya_br_subset_model.index

    
    prod_soya_br_subset_model['rcp_year'] = prod_soya_br_subset_model['rcp'] +"_" + prod_soya_br_subset_model.index.astype(str)
    
    prod_soya_br_subset_model_dif = prod_soya_br_subset_model[(prod_soya_br_subset_model['model'] == 'hybrid') | (prod_soya_br_subset_model['model'] == 'EPIC') | (prod_soya_br_subset_model['model'] == 'clim')]
    
    if chosen_model is not None:
        prod_soya_br_subset_model_dif = prod_soya_br_subset_model_dif[prod_soya_br_subset_model_dif['model'] == chosen_model]
    
    if co2_scenario is not None:
        prod_soya_br_subset_model_dif = prod_soya_br_subset_model_dif[prod_soya_br_subset_model_dif['co2'] == co2_scenario]
    
    prod_soya_br_subset_model_dif['VALUE'] = prod_soya_br_subset_model_dif.loc[:,'VALUE'].div(prod_soya_br_subset_model_dif.loc[:,'VALUE'].shift(1)) #prod_soya_br_subset_model_dif['VALUE'].diff()# 
    
    prod_soya_br_subset_model_dif_sub = prod_soya_br_subset_model_dif[prod_soya_br_subset_model_dif.index.astype(str).str.strip().str[-1] == '1'] 
    
    prod_soya_br_subset_model_dif_sub['VALUE'] = (prod_soya_br_subset_model_dif_sub['VALUE'] - 1) * 100
    
    sns.barplot(x=prod_soya_br_subset_model_dif_sub["rcp"], y=prod_soya_br_subset_model_dif_sub["VALUE"])
    plt.title(f"Variable: {prod_soya_br_subset['VAR_ID'].values[0]}")
    plt.show()
    
    sns.barplot(x=prod_soya_br_subset_model_dif_sub.index, y=prod_soya_br_subset_model_dif_sub["VALUE"])
    plt.title(f"Variable: {prod_soya_br_subset['VAR_ID'].values[0]}")
    plt.show()
    
    hatches = ['', '\\','', '\\', '', '\\']
    colors = {'2p6_2031': "#70A0CD", '8p5_2031': "#70A0CD", '2p6_2051': "#C47901", '8p5_2051': "#C47901", '2p6_2071': "#990102", '8p5_2071': "#990102"}
    
    plt.figure(figsize = (8,6),dpi=200)
    bar = sns.barplot(x=prod_soya_br_subset_model_dif_sub["rcp_year"], y=prod_soya_br_subset_model_dif_sub["VALUE"], palette=colors, order = ['2p6_2031', '8p5_2031', '2p6_2051', '8p5_2051', '2p6_2071', '8p5_2071'], ci = None)
    plt.title(f"Shock for variable: {prod_soya_br_subset['VAR_ID'].values[0]}")
    plt.ylabel('Percentage change')
    for i, thisbar in enumerate(bar.patches):
        thisbar.set_hatch(hatches[i])
    plt.show()
    
    ax = sns.kdeplot(x= prod_soya_br_subset_model_dif_sub["VALUE"], hue = prod_soya_br_subset_model_dif_sub["shock_year"], fill = True)
    plt.title(f"Variable: {prod_soya_br_subset['VAR_ID'].values[0]}")
    ax.legend_.set_bbox_to_anchor((0.05, 0.99))
    ax.legend_._set_loc(2)
    plt.show()
    

soya_br_subset_prod = prod_soya_br[ (prod_soya_br['VAR_ID'] == 'Prod') & (prod_soya_br['VAR_UNIT'] == '1000 t') ] 
shock_plots(soya_br_subset_prod, chosen_model = 'hybrid')
shock_plots(soya_br_subset_prod, chosen_model = 'hybrid', co2_scenario='2015co2' )
shock_plots(soya_br_subset_prod, chosen_model = 'hybrid', co2_scenario='default' )

soya_br_subset_cons = prod_soya_br[ (prod_soya_br['VAR_ID'] == 'CONS') & (prod_soya_br['VAR_UNIT'] == '1000 t') ] 
shock_plots(soya_br_subset_cons, chosen_model = 'hybrid',co2_scenario='2015co2')
shock_plots(soya_br_subset_cons, chosen_model = 'hybrid',co2_scenario='default')

soya_br_subset_nett = prod_soya_br[ (prod_soya_br['VAR_ID'] == 'NETT') & (prod_soya_br['VAR_UNIT'] == '1000 t') ] 
shock_plots(soya_br_subset_nett, chosen_model = 'hybrid',co2_scenario='2015co2')
shock_plots(soya_br_subset_nett, chosen_model = 'hybrid',co2_scenario='default')

soya_br_subset_xprp = prod_soya_br[ (prod_soya_br['VAR_ID'] == 'XPRP') ]# & (prod_soya_br['VAR_UNIT'] == 'fm t/ha') ] 
shock_plots(soya_br_subset_xprp, chosen_model = 'hybrid')
shock_plots(soya_br_subset_xprp, chosen_model = 'hybrid', co2_scenario='2015co2' )
shock_plots(soya_br_subset_xprp, chosen_model = 'hybrid', co2_scenario='default' )

soya_br_subset_calo = prod_soya_br[ (prod_soya_br['VAR_ID'] == 'CALO') ] 
shock_plots(soya_br_subset_calo, chosen_model = 'hybrid')
shock_plots(soya_br_subset_calo, chosen_model = 'hybrid', co2_scenario='2015co2' )
shock_plots(soya_br_subset_calo, chosen_model = 'hybrid', co2_scenario='default' )

soya_br_subset_yexo = prod_soya_br[ (prod_soya_br['VAR_ID'] == 'YEXO')  & (prod_soya_br['VAR_UNIT'] == 'fm t/ha') ] 
shock_plots(soya_br_subset_yexo, chosen_model = 'hybrid')
shock_plots(soya_br_subset_yexo, chosen_model = 'hybrid', co2_scenario='2015co2' )
shock_plots(soya_br_subset_yexo, chosen_model = 'hybrid', co2_scenario='default' )

shock_plots(soya_br_subset_yexo, co2_scenario='default')


price_soy = prod_soya_br[ (prod_soya_br['VAR_ID'] == 'XPRP') ]# & (prod_soya_br['VAR_UNIT'] == 'fm t/ha') ] 
prod_soy = prod_soya_br[ (prod_soya_br['VAR_ID'] == 'Prod') & (prod_soya_br['VAR_UNIT'] == '1000 t') ] 
value_added_soy = price_soy.copy()
value_added_soy['VALUE'] = price_soy['VALUE'] * prod_soy['VALUE']

value_added_soy_shift = value_added_soy.copy()
value_added_soy_shift['VALUE'] = value_added_soy['VALUE'].diff()
value_added_soy_shift_shock = value_added_soy_shift[value_added_soy_shift.index.astype(str).str.strip().str[-1] == '1'] 
value_added_soy_shift_shock['rcp'] = value_added_soy_shift_shock.IEA_SCEN.str.slice(13,16)
value_added_soy_shift_shock['shock_year'] =value_added_soy_shift_shock.IEA_SCEN.str.slice(17,19)
value_added_soy_shift_shock = value_added_soy_shift_shock[value_added_soy_shift_shock['IEA_SCEN'].str.len() >20]
value_added_soy_shift_shock['model'] = value_added_soy_shift_shock.IEA_SCEN.str.split('MIN_').str[-1]
value_added_soy_shift_shock['co2'] = value_added_soy_shift_shock.IEA_SCEN.str.slice(10,13)
value_added_soy_shift_shock['co2'] = value_added_soy_shift_shock['co2'].replace(['rcp','noC'], ['default','2015co2'])

value_added_soy_shift_shock_2p6_2030 = value_added_soy_shift_shock[(value_added_soy_shift_shock.index == 2031) & 
                                                                   (value_added_soy_shift_shock.shock_year == '30') & 
                                                                   (value_added_soy_shift_shock.rcp == '2p6') &
                                                                   (value_added_soy_shift_shock.model == 'hybrid') &
                                                                   (value_added_soy_shift_shock.co2 == 'default')]


value_added_soy_shift_shock_2p6_2050 = value_added_soy_shift_shock[(value_added_soy_shift_shock.index == 2051) & 
                                                                   (value_added_soy_shift_shock.shock_year == '50') & 
                                                                   (value_added_soy_shift_shock.rcp == '2p6') &
                                                                   (value_added_soy_shift_shock.model == 'hybrid') &
                                                                   (value_added_soy_shift_shock.co2 == 'default')]

value_added_soy_shift_shock_2p6_2070 = value_added_soy_shift_shock[(value_added_soy_shift_shock.index == 2071) & 
                                                                   (value_added_soy_shift_shock.shock_year == '70') & 
                                                                   (value_added_soy_shift_shock.rcp == '2p6') &
                                                                   (value_added_soy_shift_shock.model == 'hybrid') &
                                                                   (value_added_soy_shift_shock.co2 == 'default')]


value_added_soy_shift_shock_8p5_2030 = value_added_soy_shift_shock[(value_added_soy_shift_shock.index == 2031) & 
                                                                   (value_added_soy_shift_shock.shock_year == '30') & 
                                                                   (value_added_soy_shift_shock.rcp == '8p5') &
                                                                   (value_added_soy_shift_shock.model == 'hybrid') &
                                                                   (value_added_soy_shift_shock.co2 == 'default')]


value_added_soy_shift_shock_8p5_2050 = value_added_soy_shift_shock[(value_added_soy_shift_shock.index == 2051) & 
                                                                   (value_added_soy_shift_shock.shock_year == '50') & 
                                                                   (value_added_soy_shift_shock.rcp == '8p5') &
                                                                   (value_added_soy_shift_shock.model == 'hybrid') &
                                                                   (value_added_soy_shift_shock.co2 == 'default')]

value_added_soy_shift_shock_8p5_2070 = value_added_soy_shift_shock[(value_added_soy_shift_shock.index == 2071) & 
                                                                   (value_added_soy_shift_shock.shock_year == '70') & 
                                                                   (value_added_soy_shift_shock.rcp == '8p5') &
                                                                   (value_added_soy_shift_shock.model == 'hybrid') &
                                                                   (value_added_soy_shift_shock.co2 == 'default')]


print(value_added_soy_shift_shock_2p6_2030['VALUE']*1000 / 10**9)
print(value_added_soy_shift_shock_2p6_2050['VALUE']*1000 / 10**9)
print(value_added_soy_shift_shock_2p6_2070['VALUE']*1000 / 10**9)
print(value_added_soy_shift_shock_8p5_2030['VALUE']*1000 / 10**9)
print(value_added_soy_shift_shock_8p5_2050['VALUE']*1000 / 10**9)
print(value_added_soy_shift_shock_8p5_2070['VALUE']*1000 / 10**9)









price_soy = prod_soya_br[ (prod_soya_br['VAR_ID'] == 'CALO') ]


value_added_soy_shift = price_soy.copy()
value_added_soy_shift['VALUE'] = price_soy['VALUE'].diff()
value_added_soy_shift_shock = value_added_soy_shift[value_added_soy_shift.index.astype(str).str.strip().str[-1] == '1'] 
value_added_soy_shift_shock['rcp'] = value_added_soy_shift_shock.IEA_SCEN.str.slice(13,16)
value_added_soy_shift_shock['shock_year'] =value_added_soy_shift_shock.IEA_SCEN.str.slice(17,19)
value_added_soy_shift_shock = value_added_soy_shift_shock[value_added_soy_shift_shock['IEA_SCEN'].str.len() >20]
value_added_soy_shift_shock['model'] = value_added_soy_shift_shock.IEA_SCEN.str.split('MIN_').str[-1]
value_added_soy_shift_shock['co2'] = value_added_soy_shift_shock.IEA_SCEN.str.slice(10,13)
value_added_soy_shift_shock['co2'] = value_added_soy_shift_shock['co2'].replace(['rcp','noC'], ['default','2015co2'])

value_added_soy_shift_shock_2p6_2030 = value_added_soy_shift_shock[(value_added_soy_shift_shock.index == 2031) & 
                                                                   (value_added_soy_shift_shock.shock_year == '30') & 
                                                                   (value_added_soy_shift_shock.rcp == '2p6') &
                                                                   (value_added_soy_shift_shock.model == 'hybrid') &
                                                                   (value_added_soy_shift_shock.co2 == 'default')]


value_added_soy_shift_shock_2p6_2050 = value_added_soy_shift_shock[(value_added_soy_shift_shock.index == 2051) & 
                                                                   (value_added_soy_shift_shock.shock_year == '50') & 
                                                                   (value_added_soy_shift_shock.rcp == '2p6') &
                                                                   (value_added_soy_shift_shock.model == 'hybrid') &
                                                                   (value_added_soy_shift_shock.co2 == 'default')]

value_added_soy_shift_shock_2p6_2070 = value_added_soy_shift_shock[(value_added_soy_shift_shock.index == 2071) & 
                                                                   (value_added_soy_shift_shock.shock_year == '70') & 
                                                                   (value_added_soy_shift_shock.rcp == '2p6') &
                                                                   (value_added_soy_shift_shock.model == 'hybrid') &
                                                                   (value_added_soy_shift_shock.co2 == 'default')]


value_added_soy_shift_shock_8p5_2030 = value_added_soy_shift_shock[(value_added_soy_shift_shock.index == 2031) & 
                                                                   (value_added_soy_shift_shock.shock_year == '30') & 
                                                                   (value_added_soy_shift_shock.rcp == '8p5') &
                                                                   (value_added_soy_shift_shock.model == 'hybrid') &
                                                                   (value_added_soy_shift_shock.co2 == 'default')]


value_added_soy_shift_shock_8p5_2050 = value_added_soy_shift_shock[(value_added_soy_shift_shock.index == 2051) & 
                                                                   (value_added_soy_shift_shock.shock_year == '50') & 
                                                                   (value_added_soy_shift_shock.rcp == '8p5') &
                                                                   (value_added_soy_shift_shock.model == 'hybrid') &
                                                                   (value_added_soy_shift_shock.co2 == 'default')]

value_added_soy_shift_shock_8p5_2070 = value_added_soy_shift_shock[(value_added_soy_shift_shock.index == 2071) & 
                                                                   (value_added_soy_shift_shock.shock_year == '70') & 
                                                                   (value_added_soy_shift_shock.rcp == '8p5') &
                                                                   (value_added_soy_shift_shock.model == 'hybrid') &
                                                                   (value_added_soy_shift_shock.co2 == 'default')]


print(value_added_soy_shift_shock_2p6_2030['VALUE'])
print(value_added_soy_shift_shock_2p6_2050['VALUE'])
print(value_added_soy_shift_shock_2p6_2070['VALUE'])
print(value_added_soy_shift_shock_8p5_2030['VALUE'])
print(value_added_soy_shift_shock_8p5_2050['VALUE'])
print(value_added_soy_shift_shock_8p5_2070['VALUE'])



# prod_soya_br_melt = prod_soya_br.copy()
# prod_soya_br_melt = pd.pivot_table(prod_soya_br_melt, index = ['ALLYEAR','IEA_SCEN', 'VAR_UNIT'],columns = 'VAR_ID', values = 'VALUE')



prod_soya_br_subset_model = prod_soya_br_subset_model[prod_soya_br_subset_model['IEA_SCEN'].str.len() >20]
prod_soya_br_subset_model['model'] = prod_soya_br_subset_model.IEA_SCEN.str.split('MIN_').str[-1]
prod_soya_br_subset_model['rcp'] = prod_soya_br_subset_model.IEA_SCEN.str.slice(13,16)
prod_soya_br_subset_model['co2'] = prod_soya_br_subset_model.IEA_SCEN.str.slice(10,13)
prod_soya_br_subset_model['co2'] = prod_soya_br_subset_model['co2'].replace(['rcp','noC'], ['default','2015co2'])




def shock_conversion(prod_soya_br_subset, chosen_model = None, co2_scenario = None): 

    prod_soya_br_subset_model = prod_soya_br_subset.copy()
    
    prod_soya_br_subset_model = prod_soya_br_subset_model[prod_soya_br_subset_model['IEA_SCEN'].str.len() >20]
    prod_soya_br_subset_model['model'] = prod_soya_br_subset_model.IEA_SCEN.str.split('MIN_').str[-1]
    prod_soya_br_subset_model['rcp'] = prod_soya_br_subset_model.IEA_SCEN.str.slice(13,16)
    prod_soya_br_subset_model['co2'] = prod_soya_br_subset_model.IEA_SCEN.str.slice(10,13)
    prod_soya_br_subset_model['co2'] = prod_soya_br_subset_model['co2'].replace(['rcp','noC'], ['default','2015co2'])
    prod_soya_br_subset_model['shock_year'] = prod_soya_br_subset_model.index
    
    prod_soya_br_subset_model['rcp_year'] = prod_soya_br_subset_model['rcp'] +"_" + prod_soya_br_subset_model.index.astype(str)
    
    prod_soya_br_subset_model_dif = prod_soya_br_subset_model[(prod_soya_br_subset_model['model'] == 'hybrid') | (prod_soya_br_subset_model['model'] == 'EPIC') | (prod_soya_br_subset_model['model'] == 'clim')]
    
    if chosen_model is not None:
        prod_soya_br_subset_model_dif = prod_soya_br_subset_model_dif[prod_soya_br_subset_model_dif['model'] == chosen_model]
    
    if co2_scenario is not None:
        prod_soya_br_subset_model_dif = prod_soya_br_subset_model_dif[prod_soya_br_subset_model_dif['co2'] == co2_scenario]
    
    prod_soya_br_subset_model_dif['VALUE'] = prod_soya_br_subset_model_dif.loc[:,'VALUE'].div(prod_soya_br_subset_model_dif.loc[:,'VALUE'].shift(1)) #prod_soya_br_subset_model_dif['VALUE'].diff()# 
    
    prod_soya_br_subset_model_dif_sub = prod_soya_br_subset_model_dif[prod_soya_br_subset_model_dif.index.astype(str).str.strip().str[-1] == '1'] 
    
    prod_soya_br_subset_model_dif_sub['VALUE'] = (prod_soya_br_subset_model_dif_sub['VALUE'] - 1) * 100
    
    return prod_soya_br_subset_model_dif_sub

### PAIRPLOT WITH ALL VARIABLES
co2_scen = None
soya_br_prod_shock = shock_conversion(soya_br_subset_prod, chosen_model = 'hybrid', co2_scenario = co2_scen)
soya_br_cons_shock = shock_conversion(soya_br_subset_cons, chosen_model = 'hybrid', co2_scenario=co2_scen)
soya_br_nett_shock = shock_conversion(soya_br_subset_nett, chosen_model = 'hybrid', co2_scenario=co2_scen)
soya_br_xprp_shock = shock_conversion(soya_br_subset_xprp, chosen_model = 'hybrid', co2_scenario=co2_scen)
soya_br_calo_shock = shock_conversion(soya_br_subset_calo, chosen_model = 'hybrid', co2_scenario=co2_scen)
soya_br_yexo_shock = shock_conversion(soya_br_subset_yexo, chosen_model = 'hybrid', co2_scenario=co2_scen)


soya_br_prod_shock = soya_br_prod_shock.rename({'VALUE':'production'}, axis = 'columns')
soya_br_prod_shock = soya_br_prod_shock.drop(['shock_year'], axis = 1)

# soya_br_subset_prod['rcp'] = soya_br_subset_prod.IEA_SCEN.str.slice(13,16)
# soya_br_subset_prod['shock_year'] =soya_br_subset_prod.IEA_SCEN.str.slice(17,19)
# soya_br_subset_prod['model'] = soya_br_subset_prod.IEA_SCEN.str.split('MIN_').str[-1]
# soya_br_subset_prod['co2'] = soya_br_subset_prod.IEA_SCEN.str.slice(10,13)
# soya_br_subset_prod['co2'] = soya_br_subset_prod['co2'].replace(['rcp','noC'], ['default','2015co2'])



soya_br_cons_shock = soya_br_cons_shock.rename({'VALUE':'consumption'}, axis = 'columns')
soya_br_nett_shock = soya_br_nett_shock.rename({'VALUE':'nett'}, axis = 'columns')
soya_br_xprp_shock = soya_br_xprp_shock.rename({'VALUE':'xprp'}, axis = 'columns')
soya_br_calo_shock = soya_br_calo_shock.rename({'VALUE':'calo'}, axis = 'columns')
soya_br_yexo_shock = soya_br_yexo_shock.rename({'VALUE':'yexo'}, axis = 'columns')



soya_br_vars = pd.concat([soya_br_prod_shock,soya_br_yexo_shock['yexo'],soya_br_cons_shock['consumption'],soya_br_xprp_shock['xprp'],
                          soya_br_nett_shock['nett'], soya_br_calo_shock['calo']], axis = 1)
soya_br_vars = soya_br_vars[soya_br_vars['IEA_SCEN'].str.len() >20]


sns.pairplot(soya_br_vars, hue = 'rcp')




# CHECK BEST POSITION FOR CALENDAR - which year to select? 

# Compare Reanalysis EPIC with GFDL bias correctin

#  Compare the two CO2 scenarios


# Give RF_EPIC, RF_climate, RF_hybrid



# ANALYSIS

# Validation

# R2, MAE, MSE, 
# Maps of error for events - historical obs - RFs
# Weighted area yield projections (lines)

# 2 - Quantify the differences and contributions/explanatory power of each contributor

# PDFs of porjections, tails

# 3 - COmparison projections - RCPs, models, and slcies of 30 years




# delivery: RF_epic, RF_clim, RF_hybrid
# RCP: 2.6, 8.5
# CO2 fix, dyn



#%%

prod_soya_br_models = pd.read_csv('globiom_results/Soya_BR_globiom_ukesm_ipsl_v3.csv', usecols = ['IEA_SCEN', 'VAR_ID', 'VAR_UNIT', 'ALLYEAR', 'VALUE'])
prod_soya_br_models.set_index('ALLYEAR', inplace=True)

label_shock = 'Yield (t/ha)'

prod_soya_br_models.plot()

prod_soya_br_models_subset = prod_soya_br_models[ (prod_soya_br_models['VAR_ID'] == 'YEXO') & (prod_soya_br_models['VAR_UNIT'] == 'fm t/ha') ] 

def plot_hues(dataframe, list_scen):
    dataframe = dataframe[dataframe['IEA_SCEN'].isin(list_scen)]
    
    plt.figure(figsize = (10,10),dpi=200)
    sns.lineplot(data=dataframe, x='ALLYEAR', y='VALUE', hue='IEA_SCEN')
    plt.title(f"Variable: {dataframe['VAR_ID'].values[0]}")
    plt.ylabel(label_shock)
    plt.xlabel('Years')
    plt.show()


prod_test = prod_soya_br_models_subset[prod_soya_br_models_subset['IEA_SCEN'].isin([ 'GFDL-ESM4_noC2p6', 'GFDL-ESM4_noC8p5', 'IPSL-CM6A-LR_noC2p6', 'IPSL-CM6A-LR_noC8p5', 'UKESM1-0-LL_noC2p6', 'UKESM1-0-LL_noC8p5'])]
prod_test['GCM'] = prod_test.IEA_SCEN.str.split('_').str[0]
prod_test['RCP'] = prod_test.IEA_SCEN.str.split('_').str[1].str.slice(3,6)

plt.figure(figsize = (6,6),dpi=300)
sns.lineplot(data = prod_test, x = prod_test.index, y = prod_test['VALUE'], hue = prod_test['RCP'])
plt.ylabel("Yield (ton/ha)")
plt.xlabel("Years")
plt.show()

plt.figure(figsize = (6,6),dpi=300)
sns.lineplot(data = prod_test, x = prod_test.index, y = prod_test['VALUE'], hue = prod_test['GCM'])
plt.ylabel("Yield (ton/ha)")
plt.xlabel("Years")
plt.show()


list_test_slow = [ 'GFDL-ESM4_noC2p6', 'GFDL-ESM4_noC8p5', 'IPSL-CM6A-LR_noC2p6', 'IPSL-CM6A-LR_noC8p5', 'UKESM1-0-LL_noC2p6', 'UKESM1-0-LL_noC8p5']
plot_hues(prod_soya_br_models_subset, list_test_slow)

model_name_plot = "IPSL-CM6A-LR" #'UKESM1-0-LL', 'GFDL-ESM4', IPSL-CM6A-LR
list_test_slow = [ model_name_plot+'_noC2p6', model_name_plot+'_noC8p5']
plot_hues(prod_soya_br_models_subset, list_test_slow)

list_test_noC = ['UKESM1-0-LL_noC2p6_30MIN_clim', 'UKESM1-0-LL_noC2p6_30MIN_EPIC', 'UKESM1-0-LL_noC2p6_30MIN_hybrid']
plot_hues(prod_soya_br_models_subset,list_test_noC)


list_test_all = ['UKESM1-0-LL_rcp2p6_70MIN_hybrid','UKESM1-0-LL_rcp8p5_70MIN_hybrid','UKESM1-0-LL_noC2p6_70MIN_hybrid','UKESM1-0-LL_noC8p5_70MIN_hybrid']
plot_hues(prod_soya_br_models_subset, list_test_all)

model_name_plot = "UKESM1-0-LL" #'UKESM1-0-LL', 'GFDL-ESM4', IPSL-CM6A-LR
list_test_years = [model_name_plot+'_rcp2p6_30MIN_hybrid',model_name_plot+'_rcp2p6_50MIN_hybrid',model_name_plot+'_rcp2p6_70MIN_hybrid', model_name_plot+'_noC2p6_30MIN_hybrid',model_name_plot+'_noC2p6_50MIN_hybrid',model_name_plot+'_noC2p6_70MIN_hybrid']
plot_hues(prod_soya_br_models_subset, list_test_years)

list_test_rcp = ['UKESM1-0-LL_rcp8p5_30MIN_hybrid','UKESM1-0-LL_rcp8p5_30MIN_clim','UKESM1-0-LL_rcp8p5_30MIN_EPIC',
'UKESM1-0-LL_noC8p5_30MIN_hybrid', 'UKESM1-0-LL_noC8p5_30MIN_clim', 'UKESM1-0-LL_noC8p5_30MIN_EPIC']

plot_hues(prod_soya_br_models_subset, list_test_rcp)


prod_soya_br_subset_model = prod_soya_br_models_subset.copy()


#YEXO

#barplots 31-30
def pdf_plots(prod_soya_br_subset, chosen_model = None, co2_scenario = None): 
    prod_soya_br_subset_model = prod_soya_br_subset.copy()
    
    prod_soya_br_subset_model = prod_soya_br_subset_model[prod_soya_br_subset_model['IEA_SCEN'].str.len() >20]
    prod_soya_br_subset_model['gcm'] = prod_soya_br_subset_model.IEA_SCEN.str.split('_').str[0]
    prod_soya_br_subset_model['model'] = prod_soya_br_subset_model.IEA_SCEN.str.split('MIN_').str[-1]
    prod_soya_br_subset_model['rcp'] = prod_soya_br_subset_model.IEA_SCEN.str.split('_').str[1].str.slice(3,6)
    prod_soya_br_subset_model['co2'] = prod_soya_br_subset_model.IEA_SCEN.str.split('_').str[1].str.slice(0,3)
    prod_soya_br_subset_model['co2'] = prod_soya_br_subset_model['co2'].replace(['rcp','noC'], ['default','2015co2'])
    prod_soya_br_subset_model['shock_year'] = prod_soya_br_subset_model.IEA_SCEN.str.split('_').str[2].str.slice(0,2)
    prod_soya_br_subset_model['shock_year'] = prod_soya_br_subset_model['shock_year'].replace(['30','50','70'], [2031,2051,2071])
    
    prod_soya_br_subset_model['rcp_year'] = prod_soya_br_subset_model['rcp'] +"_" + prod_soya_br_subset_model.index.astype(str)
    
    prod_soya_br_subset_model_dif = prod_soya_br_subset_model[(prod_soya_br_subset_model['model'] == 'hybrid') | (prod_soya_br_subset_model['model'] == 'EPIC') | (prod_soya_br_subset_model['model'] == 'clim')]
    
    if chosen_model is not None:
        prod_soya_br_subset_model_dif = prod_soya_br_subset_model_dif[prod_soya_br_subset_model_dif['model'] == chosen_model]
    
    if co2_scenario is not None:
        prod_soya_br_subset_model_dif = prod_soya_br_subset_model_dif[prod_soya_br_subset_model_dif['co2'] == co2_scenario]
        
    prod_soya_br_subset_model_dif['VALUE'] = prod_soya_br_subset_model_dif.loc[:,'VALUE'].div(prod_soya_br_subset_model_dif.loc[:,'VALUE'].shift(1)) #prod_soya_br_subset_model_dif['VALUE'].diff()# 
    
    prod_soya_br_subset_model_dif_sub = prod_soya_br_subset_model_dif[prod_soya_br_subset_model_dif.index.astype(str).str.strip().str[-1] == '1']
    prod_soya_br_subset_model_dif_sub['VALUE'] = (prod_soya_br_subset_model_dif_sub['VALUE'] - 1) * 100
    prod_soya_br_subset_model_dif_sub = prod_soya_br_subset_model_dif_sub[prod_soya_br_subset_model_dif_sub.index == prod_soya_br_subset_model_dif_sub['shock_year']]

    
    sns.violinplot(x=prod_soya_br_subset_model_dif_sub["model"], y=prod_soya_br_subset_model_dif_sub["VALUE"])
    plt.show()
    
    sns.violinplot(x=prod_soya_br_subset_model_dif_sub["rcp"], y=prod_soya_br_subset_model_dif_sub["VALUE"])
    plt.show()
    
    sns.violinplot(x=prod_soya_br_subset_model_dif_sub.index, y=prod_soya_br_subset_model_dif_sub["VALUE"])
    plt.show()
    
    ax = sns.kdeplot(x=prod_soya_br_subset_model_dif_sub["VALUE"], hue =prod_soya_br_subset_model_dif_sub["model"], fill=True)
    ax.legend_.set_bbox_to_anchor((0.05, 0.99))
    ax.legend_._set_loc(2)
    plt.xlabel('Shock impact (%)')
    plt.title('Model')
    plt.show()
    
    ax = sns.kdeplot(x=prod_soya_br_subset_model_dif_sub["VALUE"], hue =prod_soya_br_subset_model_dif_sub.index, fill=True)
    ax.legend_.set_bbox_to_anchor((0.05, 0.99))
    ax.legend_._set_loc(2)
    plt.xlabel('Shock impact (%)')
    plt.title('Year')
    plt.show()
    
    ax = sns.kdeplot(x=prod_soya_br_subset_model_dif_sub["VALUE"], hue = prod_soya_br_subset_model_dif_sub["rcp"], fill=True)
    ax.legend_.set_bbox_to_anchor((0.05, 0.99))
    ax.legend_._set_loc(2)
    plt.xlabel('Shock impact (%)')
    plt.title('RCP')
    plt.show()
    
    ax = sns.kdeplot(x=prod_soya_br_subset_model_dif_sub["VALUE"], hue = prod_soya_br_subset_model_dif_sub["gcm"], fill=True)
    ax.legend_.set_bbox_to_anchor((0.05, 0.99))
    ax.legend_._set_loc(2)
    plt.xlabel('Shock impact (%)')
    plt.title('GCM')
    plt.show()


pdf_plots(prod_soya_br_subset_model, co2_scenario = 'default')
pdf_plots(prod_soya_br_subset_model, co2_scenario = '2015co2')

pdf_plots(prod_soya_br_subset_model, chosen_model = 'hybrid', co2_scenario = 'default')
pdf_plots(prod_soya_br_subset_model, chosen_model = 'hybrid', co2_scenario = '2015co2')


def shock_plots(prod_soya_br_subset, chosen_model = None, co2_scenario = None, chosen_rcp = None): 

    prod_soya_br_subset_model = prod_soya_br_subset.copy()
    
    prod_soya_br_subset_model = prod_soya_br_subset_model[prod_soya_br_subset_model['IEA_SCEN'].str.len() >20]
    prod_soya_br_subset_model['gcm'] = prod_soya_br_subset_model.IEA_SCEN.str.split('_').str[0]
    prod_soya_br_subset_model['model'] = prod_soya_br_subset_model.IEA_SCEN.str.split('MIN_').str[-1]
    prod_soya_br_subset_model['rcp'] = prod_soya_br_subset_model.IEA_SCEN.str.split('_').str[1].str.slice(3,6)
    prod_soya_br_subset_model['co2'] = prod_soya_br_subset_model.IEA_SCEN.str.split('_').str[1].str.slice(0,3)
    prod_soya_br_subset_model['co2'] = prod_soya_br_subset_model['co2'].replace(['rcp','noC'], ['default','2015co2'])
    prod_soya_br_subset_model['shock_year'] = prod_soya_br_subset_model.IEA_SCEN.str.split('_').str[2].str.slice(0,2)
    prod_soya_br_subset_model['shock_year'] = prod_soya_br_subset_model['shock_year'].replace(['30','50','70'], [2031,2051,2071])
    
    
    prod_soya_br_subset_model['rcp_year'] = prod_soya_br_subset_model['rcp'] +"_" + prod_soya_br_subset_model.index.astype(str)
        
    prod_soya_br_subset_model_dif = prod_soya_br_subset_model[(prod_soya_br_subset_model['model'] == 'hybrid') | (prod_soya_br_subset_model['model'] == 'EPIC') | (prod_soya_br_subset_model['model'] == 'clim')]
    
    if chosen_model is not None:
        prod_soya_br_subset_model_dif = prod_soya_br_subset_model_dif[prod_soya_br_subset_model_dif['model'] == chosen_model]
    
    if chosen_rcp is not None:
        prod_soya_br_subset_model_dif = prod_soya_br_subset_model_dif[prod_soya_br_subset_model_dif['rcp'] == chosen_rcp]
    
    if co2_scenario is not None:
        prod_soya_br_subset_model_dif = prod_soya_br_subset_model_dif[prod_soya_br_subset_model_dif['co2'] == co2_scenario]
    
    prod_soya_br_subset_model_dif['VALUE'] = prod_soya_br_subset_model_dif.loc[:,'VALUE'].div(prod_soya_br_subset_model_dif.loc[:,'VALUE'].shift(1)) #prod_soya_br_subset_model_dif['VALUE'].diff()# 
    
    prod_soya_br_subset_model_dif_sub = prod_soya_br_subset_model_dif[prod_soya_br_subset_model_dif.index.astype(str).str.strip().str[-1] == '1'] 
    
    prod_soya_br_subset_model_dif_sub['VALUE'] = (prod_soya_br_subset_model_dif_sub['VALUE'] - 1) * 100
    prod_soya_br_subset_model_dif_sub = prod_soya_br_subset_model_dif_sub[prod_soya_br_subset_model_dif_sub.index == prod_soya_br_subset_model_dif_sub['shock_year']]
   
    sns.barplot(x=prod_soya_br_subset_model_dif_sub["rcp"], y=prod_soya_br_subset_model_dif_sub["VALUE"])
    plt.title(f"Variable: {prod_soya_br_subset['VAR_ID'].values[0]}")
    plt.show()
    
    sns.barplot(x=prod_soya_br_subset_model_dif_sub.index, y=prod_soya_br_subset_model_dif_sub["VALUE"])
    plt.title(f"Variable: {prod_soya_br_subset['VAR_ID'].values[0]}")
    plt.show()
    
    sns.barplot(x=prod_soya_br_subset_model_dif_sub['gcm'], y=prod_soya_br_subset_model_dif_sub["VALUE"])
    plt.title(f"Variable: {prod_soya_br_subset['VAR_ID'].values[0]}")
    plt.show()
    
    hatches = ['', '\\','', '\\', '', '\\']
    colors = {'2p6_2031': "#70A0CD", '8p5_2031': "#70A0CD", '2p6_2051': "#C47901", '8p5_2051': "#C47901", '2p6_2071': "#990102", '8p5_2071': "#990102"}
    
    plt.figure(figsize = (8,6),dpi=200)
    bar = sns.barplot(x=prod_soya_br_subset_model_dif_sub["rcp_year"], y=prod_soya_br_subset_model_dif_sub["VALUE"], palette=colors, order = ['2p6_2031', '8p5_2031', '2p6_2051', '8p5_2051', '2p6_2071', '8p5_2071'])
    plt.title(f"Shock for variable: {prod_soya_br_subset['VAR_ID'].values[0]}")
    plt.ylabel('Percentage change')
    for i, thisbar in enumerate(bar.patches):
        thisbar.set_hatch(hatches[i])
    plt.show()
    
    ax = sns.kdeplot(x= prod_soya_br_subset_model_dif_sub["VALUE"], hue = prod_soya_br_subset_model_dif_sub["shock_year"], fill = True)
    plt.title(f"Variable: {prod_soya_br_subset['VAR_ID'].values[0]}")
    ax.legend_.set_bbox_to_anchor((0.05, 0.99))
    ax.legend_._set_loc(2)
    plt.show()
    

soya_br_subset_prod = prod_soya_br_models[ (prod_soya_br_models['VAR_ID'] == 'Prod') & (prod_soya_br_models['VAR_UNIT'] == '1000 t') ] 
shock_plots(soya_br_subset_prod, chosen_model = 'hybrid')
shock_plots(soya_br_subset_prod, chosen_model = 'hybrid', chosen_rcp='2p6' )
shock_plots(soya_br_subset_prod, chosen_model = 'hybrid', co2_scenario='2015co2')
shock_plots(soya_br_subset_prod, chosen_model = 'hybrid', co2_scenario='default' )

soya_br_subset_cons = prod_soya_br_models[ (prod_soya_br_models['VAR_ID'] == 'CONS') & (prod_soya_br_models['VAR_UNIT'] == '1000 t') ] 
shock_plots(soya_br_subset_cons, chosen_model = 'hybrid',co2_scenario='2015co2')
shock_plots(soya_br_subset_cons, chosen_model = 'hybrid',co2_scenario='default')

soya_br_subset_nett = prod_soya_br_models[ (prod_soya_br_models['VAR_ID'] == 'NETT') & (prod_soya_br_models['VAR_UNIT'] == '1000 t') ] 
shock_plots(soya_br_subset_nett, chosen_model = 'hybrid',co2_scenario='2015co2')
shock_plots(soya_br_subset_nett, chosen_model = 'hybrid',co2_scenario='default')

soya_br_subset_xprp = prod_soya_br_models[ (prod_soya_br_models['VAR_ID'] == 'XPRP') ]# & (prod_soya_br['VAR_UNIT'] == 'fm t/ha') ] 
shock_plots(soya_br_subset_xprp, chosen_model = 'hybrid')
shock_plots(soya_br_subset_xprp, chosen_model = 'hybrid', co2_scenario='2015co2' )
shock_plots(soya_br_subset_xprp, chosen_model = 'hybrid', co2_scenario='default' )

soya_br_subset_calo = prod_soya_br_models[ (prod_soya_br_models['VAR_ID'] == 'CALO') ] 
shock_plots(soya_br_subset_calo, chosen_model = 'hybrid')
shock_plots(soya_br_subset_calo, chosen_model = 'hybrid', co2_scenario='2015co2' )
shock_plots(soya_br_subset_calo, chosen_model = 'hybrid', co2_scenario='default' )

soya_br_subset_yexo = prod_soya_br_models[ (prod_soya_br_models['VAR_ID'] == 'YEXO')  & (prod_soya_br_models['VAR_UNIT'] == 'fm t/ha') ] 
shock_plots(soya_br_subset_yexo, chosen_model = 'hybrid')
shock_plots(soya_br_subset_yexo, chosen_model = 'hybrid', co2_scenario='2015co2' )
shock_plots(soya_br_subset_yexo, chosen_model = 'hybrid', co2_scenario='default' )



def shock_conversion(prod_soya_br_subset, chosen_model = None, co2_scenario = None): 

    prod_soya_br_subset_model = prod_soya_br_subset.copy()
    
    prod_soya_br_subset_model = prod_soya_br_subset_model[prod_soya_br_subset_model['IEA_SCEN'].str.len() >20]
    prod_soya_br_subset_model['gcm'] = prod_soya_br_subset_model.IEA_SCEN.str.split('_').str[0]
    prod_soya_br_subset_model['model'] = prod_soya_br_subset_model.IEA_SCEN.str.split('MIN_').str[-1]
    prod_soya_br_subset_model['rcp'] = prod_soya_br_subset_model.IEA_SCEN.str.split('_').str[1].str.slice(3,6)
    prod_soya_br_subset_model['co2'] = prod_soya_br_subset_model.IEA_SCEN.str.split('_').str[1].str.slice(0,3)
    prod_soya_br_subset_model['co2'] = prod_soya_br_subset_model['co2'].replace(['rcp','noC'], ['default','2015co2'])
    prod_soya_br_subset_model['shock_year'] = prod_soya_br_subset_model.IEA_SCEN.str.split('_').str[2].str.slice(0,2)
    prod_soya_br_subset_model['shock_year'] = prod_soya_br_subset_model['shock_year'].replace(['30','50','70'], [2031,2051,2071])
    
    prod_soya_br_subset_model['rcp_year'] = prod_soya_br_subset_model['rcp'] +"_" + prod_soya_br_subset_model.index.astype(str)
    
    prod_soya_br_subset_model_dif = prod_soya_br_subset_model[(prod_soya_br_subset_model['model'] == 'hybrid') | (prod_soya_br_subset_model['model'] == 'EPIC') | (prod_soya_br_subset_model['model'] == 'clim')]
    
    if chosen_model is not None:
        prod_soya_br_subset_model_dif = prod_soya_br_subset_model_dif[prod_soya_br_subset_model_dif['model'] == chosen_model]
    
    if co2_scenario is not None:
        prod_soya_br_subset_model_dif = prod_soya_br_subset_model_dif[prod_soya_br_subset_model_dif['co2'] == co2_scenario]
    
    prod_soya_br_subset_model_dif['VALUE'] = prod_soya_br_subset_model_dif.loc[:,'VALUE'].div(prod_soya_br_subset_model_dif.loc[:,'VALUE'].shift(1)) #prod_soya_br_subset_model_dif['VALUE'].diff()# 
    
    prod_soya_br_subset_model_dif_sub = prod_soya_br_subset_model_dif[prod_soya_br_subset_model_dif.index.astype(str).str.strip().str[-1] == '1'] 
    
    prod_soya_br_subset_model_dif_sub['VALUE'] = (prod_soya_br_subset_model_dif_sub['VALUE'] - 1) * 100
    
    prod_soya_br_subset_model_dif_sub = prod_soya_br_subset_model_dif_sub[prod_soya_br_subset_model_dif_sub.index == prod_soya_br_subset_model_dif_sub['shock_year']]

    return prod_soya_br_subset_model_dif_sub
### PAIRPLOT WITH ALL VARIABLES
co2_scen = None
soya_br_prod_shock = shock_conversion(prod_soya_br_subset_model, chosen_model = 'hybrid', co2_scenario = '2015co2')


ax = sns.kdeplot(x= soya_br_prod_shock["VALUE"], hue = soya_br_prod_shock["rcp_year"], fill = True)
plt.title(f"Variable: {soya_br_prod_shock['VAR_ID'].values[0]}")
ax.legend_.set_bbox_to_anchor((0.05, 0.99))
ax.legend_._set_loc(2)
plt.xlabel('Shock impact (%)')
plt.show()

plt.figure(figsize = (5,5),dpi=300)
ax = sns.kdeplot(x= soya_br_prod_shock["VALUE"], hue = soya_br_prod_shock["rcp"], fill = True)
# plt.title(f"Soybean yields")
ax.legend_.set_bbox_to_anchor((0.05, 0.99))
ax.legend_._set_loc(2)
plt.xlabel('Shock impact (%)')
plt.show()

plt.figure(figsize = (5,5),dpi=300)
ax = sns.kdeplot(x= soya_br_prod_shock["VALUE"], hue = soya_br_prod_shock["gcm"], fill = True)
# plt.title(f"Soybean yields")
ax.legend_.set_bbox_to_anchor((0.05, 0.99))
ax.legend_._set_loc(2)
plt.xlabel('Shock impact (%)')
plt.show()


plt.figure(figsize = (5,5),dpi=300)
ax = sns.kdeplot(x= soya_br_prod_shock["VALUE"], hue = soya_br_prod_shock["shock_year"], fill = True)
# plt.title(f"Soybean yields")
ax.legend_.set_bbox_to_anchor((0.05, 0.99))
ax.legend_._set_loc(2)
plt.xlabel('Shock impact (%)')
plt.show()

plt.figure(figsize = (5,5),dpi=300)
ax = sns.kdeplot(x= soya_br_prod_shock[soya_br_prod_shock['rcp'] == '8p5']["VALUE"], hue = soya_br_prod_shock[soya_br_prod_shock['rcp'] == '8p5']["shock_year"], fill = True)
plt.title(f"RCP 8.5")
ax.legend_.set_bbox_to_anchor((0.05, 0.99))
ax.legend_._set_loc(2)
plt.xlabel('Shock impact (%)')
plt.show()

plt.figure(figsize = (5,5),dpi=300)
ax = sns.kdeplot(x= soya_br_prod_shock[soya_br_prod_shock['rcp'] == '2p6']["VALUE"], hue = soya_br_prod_shock[soya_br_prod_shock['rcp'] == '2p6']["shock_year"], fill = True)
plt.title(f"RCP 2.6")
ax.legend_.set_bbox_to_anchor((0.05, 0.99))
ax.legend_._set_loc(2)
plt.xlabel('Shock impact (%)')
plt.show()








