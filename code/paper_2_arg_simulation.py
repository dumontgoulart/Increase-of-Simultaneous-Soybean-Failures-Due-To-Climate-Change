# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 10:08:36 2022

@author: morenodu
"""

import os
os.chdir('C:/Users/morenodu/OneDrive - Stichting Deltares/Documents/PhD/Paper_drought/data')
from mask_shape_border import mask_shape_border
from failure_probability import feature_importance_selection, failure_probability

os.chdir('C:/Users/morenodu/OneDrive - Stichting Deltares/Documents/PhD/paper_hybrid_agri/data')
from sklearnex import patch_sklearn
patch_sklearn()
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
import matplotlib as mpl
import pickle

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['figure.dpi'] = 144
mpl.rcParams.update({'font.size': 14})
#%% First step, load data
# Function
def states_mask(input_gdp_shp, state_names) :
    country = gpd.read_file(input_gdp_shp, crs="epsg:4326") 
    country_shapes = list(shpreader.Reader(input_gdp_shp).geometries())
    soy_states = country[country['NAME_1'].isin(state_names)]
    states_area = soy_states['geometry'].to_crs({'proj':'cea'}) 
    states_area_sum = (sum(states_area.area / 10**6))
    return soy_states, country_shapes, states_area_sum

# Group and detrend - .groupby('time').mean(...)
def detrending(df):
    df_det = pd.DataFrame( 
    signal.detrend(df, axis=0), index=df.index,
    columns = df.columns ) + df.mean(axis=0)
    return df_det

def plot_2d_map(dataarray_2d):
    # Plot 2D map of DataArray, remember to average along time or select one temporal interval
    plt.figure(figsize=(12,5)) #plot clusters
    ax=plt.axes(projection=ccrs.Mercator())
    dataarray_2d.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
    ax.add_geometries(arg_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
    ax.set_extent([-85,-42,-52,-5], ccrs.Geodetic())
    plt.show()
    
soy_arg_states, arg_shapes, arg_states_area_sum = states_mask('GIS/gadm36_ARG_1.shp', ['Buenos Aires'])

# plot_2d_map(DS_hadex['prcptot'].mean('time')) 

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
        for degree in [1,2]:
            # detrend along a single dimension
            p = da.polyfit(dim=dim, deg=degree)
            fit = xr.polyval(da[dim], p.polyfit_coefficients)
            
            da_det = da - fit
            res_detrend = np.nansum((da_zero_mean.mean(['lat','lon'])-da_det.mean(['lat','lon']))**2)
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


#%% Open observed yield

DS_y_obs_arg = xr.open_dataset("soy_yield_arg_1978_2019_05x05.nc", decode_times=False) #* 0.06725106937  #soy_yields_argda_05x05
DS_y_obs_arg = DS_y_obs_arg.sel(time=slice(1980,2016))
DS_y_obs_arg = DS_y_obs_arg.reindex(lat=DS_y_obs_arg.lat[::-1])
plot_2d_map(DS_y_obs_arg["Yield"].mean('time'))

    
# Mask for MIRCA 2000 each tile >0.9 rainfed
DS_mirca_test = xr.open_dataset("../../paper_hybrid_agri/data/americas_mask_ha.nc", decode_times=False)
DS_mirca_test = DS_mirca_test.rename({'latitude': 'lat', 'longitude': 'lon'})
plot_2d_map(DS_mirca_test['annual_area_harvested_rfc_crop08_ha_30mn'])

DS_y_obs_arg = DS_y_obs_arg.where(DS_mirca_test['annual_area_harvested_rfc_crop08_ha_30mn'] > 0 )
DS_y_obs_arg = DS_y_obs_arg.dropna(dim = 'lon', how='all')
DS_y_obs_arg = DS_y_obs_arg.dropna(dim = 'lat', how='all')
if len(DS_y_obs_arg.coords) >3 :
    DS_y_obs_arg=DS_y_obs_arg.drop('spatial_ref')
  
# # EPIC
# DS_y_epic_arg = xr.open_dataset("../../Paper_drought/data/ACY_gswp3-w5e5_obsclim_2015soc_default_soy_noirr.nc", decode_times=True).sel(time=slice('1959-12-12','2016-12-12'), lon=slice(-160,-10)) 

# EPIC 2
DS_y_epic = xr.open_dataset("epic-iiasa_gswp3-w5e5_obsclim_2015soc_default_yield-soy-noirr_global_annual_1901_2016.nc", decode_times=False)
DS_y_epic = DS_y_epic.reindex(lat=DS_y_epic.lat[::-1])
# Convert time unit
units, reference_date = DS_y_epic.time.attrs['units'].split('since')
DS_y_epic['time'] = pd.date_range(start=reference_date, periods=DS_y_epic.sizes['time'], freq='YS')
DS_y_epic['time'] = DS_y_epic['time'].dt.year 

DS_y_epic["yield-soy-noirr"].sel(lon=slice(-125,-67), lat=slice(25,50)).mean('time').plot()
plot_2d_map(DS_y_epic["yield-soy-noirr"].mean('time'))

DS_y_epic_arg = DS_y_epic.where(DS_y_obs_arg['Yield'] > -4.0 )
# DS_y_epic_arg = DS_y_epic_arg.rename({'yield-soy-noirr':'yield'})
DS_y_obs_arg = DS_y_obs_arg.where(DS_y_epic_arg['yield-soy-noirr'] > -4.0 )



plot_2d_map(DS_y_obs_arg["Yield"].sel(time=2014))
plot_2d_map(DS_y_epic_arg["yield-soy-noirr"].sel(time=2014))

corr_3d = xr.corr(DS_y_epic_arg["yield-soy-noirr"], DS_y_obs_arg["Yield"], dim="time", )
plt.figure(figsize=(12,5)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
corr_3d.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True, levels = 10)
ax.add_geometries(arg_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-80,-40,-50,-20], ccrs.PlateCarree())
plt.show()
corr_3d_high = corr_3d.where(corr_3d > 0.4)
plot_2d_map(corr_3d_high)

# Compare
df_epic_arg = DS_y_epic_arg.to_dataframe().dropna()
df_obs_arg = DS_y_obs_arg.to_dataframe().dropna()

DS_y_obs_arg_det = xr.DataArray( detrend_dim(DS_y_obs_arg['Yield'], 'time') + DS_y_obs_arg['Yield'].mean('time'), name= DS_y_obs_arg['Yield'].name, attrs = DS_y_obs_arg['Yield'].attrs)
plot_2d_map(DS_y_obs_arg_det.mean('time'))
    
DS_y_epic_arg_det = xr.DataArray( detrend_dim(DS_y_epic_arg["yield-soy-noirr"], 'time') + DS_y_epic_arg["yield-soy-noirr"].mean('time'), name= DS_y_epic_arg["yield-soy-noirr"].name, attrs = DS_y_epic_arg["yield-soy-noirr"].attrs)

plt.plot(DS_y_obs_arg.time, DS_y_obs_arg['Yield'].mean(['lat','lon']))
plt.plot(DS_y_obs_arg.time, DS_y_obs_arg_det.mean(['lat','lon']))
plt.title('Observed data detrending')
plt.show()

plt.plot(DS_y_epic_arg["yield-soy-noirr"].time, DS_y_epic_arg["yield-soy-noirr"].mean(['lat','lon']))
plt.plot(DS_y_epic_arg_det.time, DS_y_epic_arg_det.mean(['lat','lon']))
plt.title('EPIC data detrending')
plt.show()

# Compare EPIC with Observed dataset
df_epic_arg_det = DS_y_epic_arg_det.to_dataframe().dropna()
df_obs_arg_det = DS_y_obs_arg_det.to_dataframe().dropna()

plt.plot(DS_y_epic_arg_det.time, DS_y_epic_arg_det.mean(['lat','lon']))
plt.plot(DS_y_obs_arg_det.time, DS_y_obs_arg_det.mean(['lat','lon']))
plt.vlines(DS_y_epic_arg_det.time, 1,5.5, linestyles ='dashed', colors = 'k')
plt.title('epic vs obs comparison')
plt.show()

# Pearson's correlation
from scipy.stats import pearsonr

corr_grouped, _ = pearsonr(df_obs_arg_det['Yield'], df_epic_arg_det['yield-soy-noirr'])
print('Pearsons correlation: %.3f' % corr_grouped)

corr_grouped, _ = pearsonr(DS_y_epic_arg_det.mean(['lat','lon']).to_dataframe().dropna()['yield-soy-noirr'], DS_y_obs_arg_det.mean(['lat','lon']).to_dataframe().dropna()['Yield'])
print('Pearsons correlation: %.3f' % corr_grouped)


DS_y_iizumi = xr.open_dataset("soybean_iizumi_1981_2016.nc", decode_times=True)
DS_y_iizumi = DS_y_iizumi.rename({'latitude': 'lat', 'longitude': 'lon'})

DS_y_iizumi_test = DS_y_iizumi.where(DS_y_obs_arg_det >= -5.0 )
df_iizumi = DS_y_iizumi_test.to_dataframe().dropna()
df_iizumi_mean = df_iizumi.groupby('time').mean(...)
df_iizumi_mean_det = detrending(df_iizumi_mean)

# Plot time series
plt.figure(figsize=(10,6))
plt.plot(DS_y_epic_arg_det.time, DS_y_epic_arg_det.mean(['lat','lon']), label = 'EPIC')
plt.plot(DS_y_obs_arg_det.time, DS_y_obs_arg_det.mean(['lat','lon']), label = 'OBS')
plt.vlines(DS_y_epic_arg_det.time, 1,5.5, linestyles ='dashed', colors = 'k')
plt.plot(df_iizumi_mean_det, label = 'Iizumi')
plt.title('epic vs obs comparison')
plt.legend()
plt.show()


#%% Machine learning model training
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error,  explained_variance_score
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from keras.layers import Activation
from keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from scikeras.wrappers import KerasRegressor
import lightgbm as lgb

import os
os.environ['PYTHONHASHSEED']= '123'
os.environ['TF_CUDNN_DETERMINISTIC']= '1'
import random as python_random
np.random.seed(1)
python_random.seed(1)
tf.random.set_seed(1)

def calibration(X,y,type_of_model='RF', params = None, stack_model = False):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
       
    if params is None:
        # model_rf = make_pipeline(StandardScaler(), MLPRegressor(random_state=0, hidden_layer_sizes= (200,200,200), alpha = 0.0001,verbose=1, max_iter=400,learning_rate = 'adaptive') ) #
        # full_model_rf = make_pipeline(StandardScaler(), MLPRegressor(random_state=0, hidden_layer_sizes= (200,200,200), alpha = 0.0001,verbose=1, max_iter=400,learning_rate = 'adaptive') ) #
        
        if type_of_model == 'RF':
            model_rf = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1,
                                              max_depth = 20, max_features = 'auto',
                                              min_samples_leaf = 1, min_samples_split=2)
            
            full_model_rf = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1,
                                              max_depth = 20, max_features = 'auto',
                                              min_samples_leaf = 1, min_samples_split=2)
            
        
        elif type_of_model == 'lightgbm':
            model_rf = Pipeline([
                ('scaler', StandardScaler()),
                ('estimator', lgb.LGBMRegressor(linear_tree= True, max_depth = 20, num_leaves = 50, min_data_in_leaf = 100, 
                                                random_state=0, learning_rate = 0.01, n_estimators = 1000 ) )
            ])
            
            
            full_model_rf = Pipeline([
                ('scaler', StandardScaler()),
                ('estimator', lgb.LGBMRegressor(linear_tree= True, max_depth = 20, num_leaves = 50, min_data_in_leaf = 100, 
                                                random_state=0, learning_rate = 0.01, n_estimators = 1000 ) )
            ])
        
        elif type_of_model == 'DNN':
            def create_model():
                model = Sequential()
                model.add(Dense(200, input_dim=len(X_train.columns))) #,kernel_regularizer=regularizers.l2(0.0001)
                model.add(BatchNormalization())
                model.add(Activation('relu'))
                # model.add(Dropout(0.2))
    
                model.add(Dense(200))
                model.add(BatchNormalization())
                model.add(Activation('relu'))
                # model.add(Dropout(0.2))
    
                model.add(Dense(200))
                model.add(BatchNormalization())
                model.add(Activation('relu'))
                # model.add(Dropout(0.2))
                
                model.add(Dense(200))
                model.add(BatchNormalization())
                model.add(Activation('relu'))
                # model.add(Dropout(0.2))
    
                # model.add(Dense(50))
                # model.add(BatchNormalization())
                # model.add(Activation('relu'))
                # model.add(Dense(50))
                # model.add(BatchNormalization())
                # model.add(Activation('relu'))
                model.add(Dense(1, activation='linear'))
                # compile the keras model
                model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.01), metrics=['mean_squared_error','mean_absolute_error'])
                return model
                    
            model_rf = Pipeline([
                ('scaler', StandardScaler()),
                ('estimator', KerasRegressor(model=create_model, epochs=200, batch_size=512, verbose=1))
            ])
    
            full_model_rf = Pipeline([
                ('scaler', StandardScaler()),
                ('estimator', KerasRegressor(model=create_model, epochs=200, batch_size=512, verbose=1))
            ])
            
    
    elif params is not None:
        model_rf = RandomForestRegressor(n_estimators=params['n_estimators'], random_state=0, n_jobs=-1, 
                                      max_depth = params['max_depth'], max_features = params['max_features'],
                                      min_samples_leaf = params['min_samples_leaf'], min_samples_split = params['min_samples_split'])
        
        full_model_rf = RandomForestRegressor(n_estimators=params['n_estimators'], random_state=0, n_jobs=-1, 
                                      max_depth = params['max_depth'], max_features = params['max_features'],
                                      min_samples_leaf = params['min_samples_leaf'], min_samples_split = params['min_samples_split'])
        
    if stack_model is False:
        model = model_rf.fit(X_train, y_train)
        
        full_model = full_model_rf.fit(X, y)
        
    elif stack_model is True:
        print('Model: stacked')
        
        estimators = [
            ('RF', make_pipeline(StandardScaler(), model_rf)), #make_pipeline(StandardScaler(),
            ('MLP', make_pipeline(StandardScaler(), MLPRegressor(random_state=0, hidden_layer_sizes= (100,100,100), alpha = 0.0001,verbose=1, max_iter=400,learning_rate = 'adaptive')) )
            ]
        
        estimators_full = [
            ('RF', make_pipeline(StandardScaler(), full_model_rf)), #make_pipeline(StandardScaler(),
            ('MLP', make_pipeline(StandardScaler(), MLPRegressor(random_state=0, hidden_layer_sizes= (100,100,100), alpha = 0.0001, max_iter=400))) #
            ]
        
        # Get together the models:
        stacking_regressor = StackingRegressor(estimators=estimators, final_estimator = LinearRegression() ) # MLPRegressor(random_state=0, max_iter=500) #SVR() # GaussianProcessRegressor(kernel = 1**2 * RationalQuadratic(alpha=1, length_scale=1)) #StackingRegressor(estimators=estimators)
        stacking_regressor_full = StackingRegressor(estimators=estimators_full, final_estimator = LinearRegression() ) # MLPRegressor(random_state=0, max_iter=500) #SVR() # GaussianProcessRegressor(kernel = 1**2 * RationalQuadratic(alpha=1, length_scale=1)) #StackingRegressor(estimators=estimators)

        model = stacking_regressor.fit(X_train, y_train)

        full_model = stacking_regressor_full.fit(X, y)
        
       
        
    def MBE(y_true, y_pred):
        '''
        Parameters:
            y_true (array): Array of observed values
            y_pred (array): Array of prediction values
    
        Returns:
            mbe (float): Bias score
        '''
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_true = y_true.reshape(len(y_true),1)
        y_pred = y_pred.reshape(len(y_pred),1)   
        diff = (y_true-y_pred)
        mbe = diff.mean()
        return mbe
    
    # Test performance
    y_pred = model.predict(X_test)
    
    # report performance
    print("R2 on test set:", round(r2_score(y_test, y_pred),2))
    print("Var score on test set:", round(explained_variance_score(y_test, y_pred),2))
    print("MAE on test set:", round(mean_absolute_error(y_test, y_pred),5))
    print("RMSE on test set:",round(mean_squared_error(y_test, y_pred, squared=False),5))
    print("MBE on test set:", round(MBE(y_test, y_pred),5))
    print("______")
    
    y_pred_total = full_model.predict(X)
    
    
    plt.figure(figsize=(5,5), dpi=250) #plot clusters
    plt.scatter(y_test, y_pred)
    plt.plot(y_test, y_test, color = 'black', label = '1:1 line')
    plt.ylabel('Predicted yield')
    plt.xlabel('Observed yield')
    plt.title('Scatter plot - test set')
    plt.legend()
    # plt.savefig('paper_figures/epic_argda_validation.png', format='png', dpi=500)
    plt.show()

    # # perform permutation importance
    # results = permutation_importance(model, X_test, y_test, scoring='neg_mean_squared_error', n_repeats=5, random_state=0, n_jobs=-1)
    # # get importance
    # df_importance = pd.DataFrame(results.importances_mean)
    # df_importance.index = X.columns
    # print("Mutual importance:",df_importance)
    # # summarize feature importance
    # plt.figure(figsize=(12,5)) #plot clusters
    # plt.bar(df_importance.index, df_importance[0])
    # plt.show()
    
    return y_pred, y_pred_total, model, full_model 


from sklearn.model_selection import GridSearchCV

def hyper_param_tuning(X,y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    
    # Create the parameter grid based on the results of random search 
    param_grid = {
        'max_depth': [5,6,10,15,20], #list(range(5,15))
        'max_features': ['auto'],
        'min_samples_leaf': [1,2,3,4],
        'min_samples_split': [2,3,4,5],
        'n_estimators': [100, 200, 300,500]
    }
    # Create a based model
    rf = RandomForestRegressor()# Instantiate the grid search model #scoring='neg_mean_absolute_error',
    grid_search = GridSearchCV(estimator = rf,  param_grid = param_grid, scoring = 'neg_root_mean_squared_error', cv = 5, n_jobs = -1, verbose = 2) 
    grid_search.fit(X_train, y_train)
    print("Best parameters set found on development set:")
    print(grid_search.best_params_)
    means = grid_search.cv_results_["mean_test_score"]
    stds = grid_search.cv_results_["std_test_score"]
    for mean, std, params in zip(means, stds, grid_search.cv_results_["params"]):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    
    params_cv_chosen = grid_search.best_params_
    best_grid = grid_search.best_estimator_
    
    return params_cv_chosen, best_grid

#%% EPIC RF

X, y = df_epic_arg_det, df_obs_arg_det['Yield'].values.flatten().ravel()

for test_size in [0.1,0.2,0.3,0.4,0.5]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    
    regr_rf = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1, 
                                      max_depth = 20, max_features = 'auto',
                                           min_samples_leaf = 1, min_samples_split=2)
    regr_rf.fit(X_train, y_train)
    
    y_rf = regr_rf.predict(X_test)
    
    print(f"R2 {test_size} OBS-RF:EPIC:",round(r2_score(y_test, y_rf),2))

# Standard model
print('Standard Epic results:')
y_pred_epic_arg, y_pred_total_epic_arg, model_epic_arg, full_model_epic_arg  = calibration(X, y, stack_model = False)



#%% EXTREME CLIMATE INDICES

def timedelta_to_int(DS, var):
    da_timedelta = DS[var].dt.days
    da_timedelta = da_timedelta.rename(var)
    da_timedelta.attrs["units"] = 'days'
    
    return da_timedelta

# Select data according to months
def is_month(month, ref_in, ref_out):
    return (month >= ref_in) & (month <= ref_out)


# Start
start_date, end_date = '31-12-1979','31-12-2016'

DS_exclim_arg = xr.open_mfdataset('../../paper_hybrid_agri/data/climpact-master/climpact-master/www/output_historical_arg/monthly_data/*.nc').sel(time=slice(start_date, end_date))

# New dataset
DS_exclim_arg = DS_exclim_arg.drop_vars(['fd','id','time_bnds']) # Always zero
list_features_arg = ['prcptot', 'r10mm','txm' ] #, 'tmge5', 'txgt50p',  'su']# 'dtr', 'tnm', 'txge35', 'tr', 'txm', 'tmm', 'tnn'
DS_exclim_arg = DS_exclim_arg[list_features_arg] 
DS_exclim_arg = DS_exclim_arg.where(DS_y_obs_arg_det.mean('time') > -5)

def units_conversion(DS_exclim_arg):
    da_list = []
    for feature in list(DS_exclim_arg.keys()):
        if (type(DS_exclim_arg[feature].values[0,0,0]) == np.timedelta64):
            print('Time')
            DS = timedelta_to_int(DS_exclim_arg, feature)
        else:
            print('Integer')
            DS = DS_exclim_arg[feature]
        
        da_list.append(DS)
    return xr.merge(da_list)    

DS_exclim_arg_comb = units_conversion(DS_exclim_arg)
DS_exclim_arg_comb = DS_exclim_arg_comb.drop_vars('r10mm') # Always zero

DS_exclim_arg_comb.coords['lon'] = (DS_exclim_arg_comb.coords['lon'] + 180) % 360 - 180
DS_exclim_arg_comb = DS_exclim_arg_comb.sortby(DS_exclim_arg_comb.lon)
DS_exclim_arg_comb = DS_exclim_arg_comb.reindex(lat=DS_exclim_arg_comb.lat[::-1])
if len(DS_exclim_arg_comb.coords) >3 :
    DS_exclim_arg_comb=DS_exclim_arg_comb.drop('spatial_ref')
    
DS_exclim_arg_det = DS_exclim_arg_comb #detrend_dataset(DS_exclim_arg_comb) #DS_exclim_arg_comb #
# DS_exclim_arg_det = DS_exclim_arg_det.where(DS_y_obs_arg['Yield'].mean('time') >= 0.0 )
# DS_exclim_arg_det = mask_shape_border(DS_exclim_arg_det, soy_arg_states)

### SELECT MONTHS TO BE CONSIDERED
DS_exclim_arg_det_season = DS_exclim_arg_det.sel(time=is_month(DS_exclim_arg_det['time.month'], 3,6)) #['dtr', 'tnm', 'tr', 'txm', 'txge35', 'tmm', 'tnn'] // ['tnm', 'txge35', 'tnn', 'tmm', 'txm', 'txge30', 'tmlt10']

# Average across season
DS_exclim_arg_det_season = DS_exclim_arg_det_season.groupby('time.year').mean('time')
DS_exclim_arg_det_season = DS_exclim_arg_det_season.rename({'year':'time'})
DS_exclim_arg_det_season = DS_exclim_arg_det_season.reindex(lat=DS_exclim_arg_det_season.lat[::-1])
DS_exclim_arg_det_season = DS_exclim_arg_det_season.where(DS_y_obs_arg['Yield'] >= -5.0 )
DS_exclim_arg_det_season = DS_exclim_arg_det_season.sortby(['time','lat','lon'])
plot_2d_map(DS_exclim_arg_det_season['prcptot'].mean('time'))

# DS_exclim_arg_det_season = xr.merge([DS_exclim_arg_det_season,DS_exclim_ann_arg_det])

df_exclim_arg_det_season = DS_exclim_arg_det_season.to_dataframe().dropna().reorder_levels(['time','lat','lon']).sort_index()
# df_obs_arg_det_2 = df_obs_arg_det.where(df_exclim_arg_det_season['spei'] >= -100000.0 ).dropna()

feature_importance_selection(df_exclim_arg_det_season, df_obs_arg_det)

print('Standard ECE results:')
X, y = df_exclim_arg_det_season, df_obs_arg_det['Yield'].values.flatten().ravel()
y_pred_exclim, y_pred_total_exclim, model_exclim, full_model_exclim = calibration(X, y)




#%% Relative dates calendar

# Functions
# Reshape to have each calendar year on the columns (1..12)
def reshape_data(dataarray):  #converts and reshape data
    if isinstance(dataarray, pd.DataFrame): #If already dataframe, skip the convertsion
        dataframe = dataarray
    elif isinstance(dataarray, pd.Series):    
        dataframe = dataarray.to_frame()
        
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
    ### Convert to dataframe
    if shift_time == 0:     
        dataframe_1 = dataset.to_dataframe()
        # Define the column names
        column_names = [dataset.name +"_"+str(j) for j in range(1,13)]   
    else:
        dataframe_1 = dataset.shift(time=-shift_time).to_dataframe()    
        # Define the column names
        column_names = [dataset.name +"_"+str(j) for j in range(1+shift_time,13+shift_time)]
    # Reshape
    dataframe_reshape = reshape_data(dataframe_1)
    dataframe_reshape.columns = column_names  
    return dataframe_reshape


### Load calendars ############################################################
DS_cal_sachs = xr.open_dataset('../../paper_hybrid_agri/data/Soybeans.crop.calendar_sachs_05x05.nc') / (365/12) # 0.72 for Sachs // best type of calendar is plant
DS_cal_mirca = xr.open_dataset('../../paper_hybrid_agri/data/mirca2000_soy_calendar.nc') # 
# DS_cal_plant = xr.open_dataset('../../Paper_drought/data/soy_rf_pd_2015soc.nc').mean('time') / (365/12)
DS_cal_ggcmi = xr.open_dataset('../../paper_hybrid_agri/data/soy_rf_ggcmi_crop_calendar_phase3_v1.01.nc4') / (365/12)

DS_cal_mirca_subset = DS_cal_mirca.where(DS_exclim_arg_det['prcptot'].mean('time') >= -10)
DS_cal_sachs_month_subset = DS_cal_sachs.where(DS_exclim_arg_det['prcptot'].mean('time') >= -10)
DS_cal_ggcmi_subset = DS_cal_ggcmi.where(DS_exclim_arg_det['prcptot'].mean('time') >= -10)

plot_2d_map(DS_cal_mirca_subset['start'])
plot_2d_map(DS_cal_ggcmi_subset['planting_day'])
plot_2d_map(DS_cal_ggcmi_subset['maturity_day'])
plot_2d_map(DS_cal_ggcmi_subset['maturity_day']-DS_cal_ggcmi_subset['planting_day'])

### Chose calendar:
DS_chosen_calendar_arg = DS_cal_mirca_subset['start']   # [ DS_cal_mirca_subset['start'] #DS_cal_ggcmi_subset['planting_day']  
if DS_chosen_calendar_arg.name != 'plant':
    DS_chosen_calendar_arg = DS_chosen_calendar_arg.rename('plant')
# Convert DS to df
df_chosen_calendar = DS_chosen_calendar_arg.to_dataframe().dropna()
# Convert planting days to beginning of the month
df_calendar_month_arg = df_chosen_calendar[['plant']].apply(np.rint).astype('Int64')


### LOAD climate date and clip to the calendar cells    
DS_exclim_arg_det_clip = DS_exclim_arg_det.where(DS_chosen_calendar_arg >= 0 )
DS_exclim_arg_det_clip.resample(time="1MS").mean(dim="time")
DS_exclim_arg_det_clip = DS_exclim_arg_det_clip.where(DS_y_obs_arg_det.mean('time') > -5)
plot_2d_map(DS_exclim_arg_det_clip['prcptot'].mean('time'))

# DS_exclim_arg_det_clip['time'] = DS_exclim_arg_det_clip['time'] + pd.Timedelta(days=365)

# For loop along features to obtain 24 months of climatic data for each year
list_features_reshape_shift_arg = []
for feature in list(DS_exclim_arg_det_clip.keys()):
    print(feature)
    ### Reshape and shift for 24 months for every year.
    df_test_shift = reshape_shift(DS_exclim_arg_det_clip[feature])
    df_test_shift_12 = reshape_shift(DS_exclim_arg_det_clip[feature], shift_time = 12)
    # Combine both dataframes
    df_test_reshape_twoyears = df_test_shift.dropna().join(df_test_shift_12)
    # Remove last year, because we do not have two years for it
    df_test_reshape_twoyears = df_test_reshape_twoyears.query('year <= 2016')
    ### Join and change name to S for the shift values
    df_feature_reshape_shift = (df_test_reshape_twoyears.dropna().join(df_calendar_month_arg)
                                .rename(columns={'plant':'s'}))
    # Move 
    col = df_feature_reshape_shift.pop("s")
    df_feature_reshape_shift.insert(0, col.name, col)
    # Activate this if error "TypeError: int() argument must be a string, a bytes-like object or a number, not 'NAType'" occurs
    
    # print(df_feature_reshape_shift[['s']].isna().sum())

    # Shift accoording to month indicator (hence +1)
    df_feature_reshape_shift = (df_feature_reshape_shift.apply(lambda x : x.shift(-(int(x['s']))+1) , axis=1)
                                .drop(columns=['s']))
    
    
    list_features_reshape_shift_arg.append(df_feature_reshape_shift)

# Transform into dataframe
df_features_reshape_2years = pd.concat(list_features_reshape_shift_arg, axis=1)

### Select specific months ###################################################
suffixes = tuple(["_"+str(j) for j in range(2,5)])
df_feature_season_6mon_arg = df_features_reshape_2years.loc[:, df_features_reshape_2years.columns.str.endswith(suffixes)]

# Shifting one year?  NO, Argentina does not need that
df_feature_season_6mon_arg.index = df_feature_season_6mon_arg.index.set_levels(df_feature_season_6mon_arg.index.levels[2] + 0, level=2)

df_feature_season_6mon_arg = df_feature_season_6mon_arg.rename_axis(index={'year':'time'}).reorder_levels(['time','lat','lon']).sort_index()
df_feature_season_6mon_arg = df_feature_season_6mon_arg.where(df_exclim_arg_det_season['prcptot']>=-20).dropna().astype(float)
   
# SECOND DETRENDING PART - SEASONAL
DS_feature_season_6mon_arg = xr.Dataset.from_dataframe(df_feature_season_6mon_arg)
DS_feature_season_6mon_arg_det = detrend_dataset(DS_feature_season_6mon_arg, deg = 'free')
df_feature_season_6mon_arg_det = DS_feature_season_6mon_arg_det.to_dataframe().dropna()

for feature in df_feature_season_6mon_arg.columns:
    df_feature_season_6mon_arg[feature].groupby('time').mean().plot(label = 'old')
    df_feature_season_6mon_arg_det[feature].groupby('time').mean().plot(label = 'detrend')
    plt.title(f'{feature}')
    plt.legend()
    plt.show()

# =============================================================================
# # ATTENTION HERE - update second detrending scheme
# =============================================================================
df_feature_season_6mon_arg = df_feature_season_6mon_arg_det

for feature in ['prcptot_2', 'prcptot_3', 'prcptot_4']:
    df_feature_season_6mon_arg[feature][df_feature_season_6mon_arg[feature] < 0] = 0

df_obs_arg_det_clip = df_obs_arg_det.where(df_feature_season_6mon_arg['prcptot'+suffixes[0]] > -100).dropna()

feature_importance_selection(df_feature_season_6mon_arg, df_obs_arg_det_clip['Yield'])

print('Dynamic ECE results:')
X, y = df_feature_season_6mon_arg, df_obs_arg_det_clip['Yield'].values.flatten().ravel()
y_pred_exclim_dyn_arg, y_pred_total_exclim_dyn_arg, model_exclim_dyn_arg, full_model_exclim_dyn_arg = calibration(X, y, stack_model = False)


# =============================================================================
# #### BENCHMARK FOR CALENDAR CHANGES ####################################################################
# =============================================================================
# For loop along features to obtain 24 months of climatic data for each year
# -------------------------------------------------------------------------------------------------------
list_static_calendar = []
for feature in list(DS_exclim_arg_det_clip.keys()):
    ### Reshape and shift for 24 months for every year.
    df_test_shift = reshape_shift(DS_exclim_arg_det_clip[feature])
    df_test_shift_12 = reshape_shift(DS_exclim_arg_det_clip[feature], shift_time = 12)
    # Combine both dataframes
    df_test_reshape_twoyears = df_test_shift.dropna().join(df_test_shift_12)
    list_static_calendar.append(df_test_reshape_twoyears)
    
# Transform into dataframe
df_cal_benchmark = pd.concat(list_static_calendar, axis=1)

### Select specific months
suffixes_stat = tuple(["_"+str(j) for j in range(13,16)])
df_cal_benchmark_season = df_cal_benchmark.loc[:,df_cal_benchmark.columns.str.endswith(suffixes_stat)]
df_cal_benchmark_season = df_cal_benchmark_season.rename_axis(index={'year':'time'}).reorder_levels(['time','lat','lon']).sort_index()
df_cal_benchmark_season = df_cal_benchmark_season.where(df_obs_arg_det['Yield']>=-10).dropna().astype(float)
# df_cal_benchmark_season = pd.DataFrame(df_cal_benchmark_season.iloc[:,0:3]) - if you want only precipitation

feature_importance_selection(df_cal_benchmark_season, df_obs_arg_det['Yield'])

print('Static ECE results:')
X, y = df_cal_benchmark_season, df_obs_arg_det['Yield'].values.flatten().ravel()
y_pred_exclim_stat, y_pred_total_exclim_stat, model_exclim_stat, full_model_exclim_stat = calibration(X, y)


#%% Hybrid model
df_epic_arg_det_clip = df_epic_arg_det.where(df_feature_season_6mon_arg['prcptot'+suffixes[0]] > -100).dropna()

# Combine the EPIC output with the Extreme climate indices to generate the input dataset for the hybrid model
df_hybrid_arg = pd.concat([df_epic_arg_det_clip, df_feature_season_6mon_arg], axis = 1)
# df_hybrid_arg['lat']=df_hybrid_arg.index.get_level_values(1)

# Save this for future operations:
df_hybrid_arg.to_csv('dataset_input_hybrid_arg_forML.csv')
df_obs_arg_det_clip.to_csv('dataset_obs_yield_arg_forML.csv')

X, y = df_hybrid_arg, df_obs_arg_det_clip['Yield'].values.flatten().ravel()


# Feature selection
feature_importance_selection(df_hybrid_arg, df_obs_arg_det_clip)

for test_size in [0.1,0.2]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    
    regr_rf = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1, 
                                      max_depth = 20, max_features = 'auto',
                                           min_samples_leaf = 1, min_samples_split=2)
    regr_rf.fit(X_train, y_train)
    
    y_rf = regr_rf.predict(X_test)
    
    print(f"R2 {test_size} OBS-RF:EPIC:",round(r2_score(y_test, y_rf),2))

# Evaluate Model
print('Standard Hybrid results:')
# y_pred_hyb_arg, y_pred_total_hyb_arg, model_hyb_arg, full_model_hyb_arg = calibration(X, y, type_of_model='DNN', stack_model = False)
y_pred_hyb_arg_rf, y_pred_total_hyb_arg_rf, model_hyb_arg_rf, full_model_hyb2_rf = calibration(X, y, type_of_model='RF', stack_model = False)
# y_pred_hyb_arg_rf, y_pred_total_hyb_arg_rf, model_hyb_arg_rf, full_model_hyb2_rf = calibration(X, y, type_of_model='lightgbm', stack_model = False)


# ################ saving just the keras model, still needs to be scaled. <<<<<<<<<<<<<<<<
# full_model_hyb_arg['estimator'].model_.save('/hybrid_model/hybrid_arg_ANN_67')

# # Save the Keras model first:
# full_model_hyb_arg['estimator'].model_.save('hybrid_arg_ANN_67.h5')

# # This hack allows us to save the sklearn pipeline:
# full_model_hyb_arg['estimator'].model = None

# import joblib
# from keras.models import load_model
# # Finally, save the pipeline:
# joblib.dump(full_model_hyb_arg, 'sklearn_pipeline_arg_ANN_67.pkl')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Test performance
y_pred = model_hyb_am2.predict(X_test)

# report performance
print("R2 on test set:", round(r2_score(y_test, y_pred),2))
print("Var score on test set:", round(explained_variance_score(y_test, y_pred),2))
print("MAE on test set:", round(mean_absolute_error(y_test, y_pred),5))
print("RMSE on test set:",round(mean_squared_error(y_test, y_pred, squared=False),5))

#%%
for feature in df_hybrid_arg.columns:   
    plt.plot(df_hybrid_arg[feature].groupby('time').mean(), label = 'History')
    plt.axvline(df_hybrid_arg['yield-soy-noirr'].groupby('time').mean().idxmin(), linestyle = 'dashed')
    plt.axvline(df_hybrid_arg['yield-soy-noirr'].groupby('time').mean().nsmallest(5).index[1], linestyle = 'dashed')
    plt.title(f'{feature} predictions')
    plt.legend()
    plt.show()

plt.plot(df_obs_arg_det_clip['Yield'].groupby('time').mean(), label = 'History')
plt.axvline(df_hybrid_arg['yield-soy-noirr'].groupby('time').mean().idxmin(), linestyle = 'dashed')
plt.axvline(df_hybrid_arg['yield-soy-noirr'].groupby('time').mean().nsmallest(5).index[1], linestyle = 'dashed')

# Compare timelines
df_predict_hist_arg = df_obs_arg_det_clip.copy()
predict_test_hist = model_hyb_arg.predict(df_hybrid_arg.values)
df_predict_hist_arg.loc[:,"Yield"] = predict_test_hist
DS_predict_test_hist = xr.Dataset.from_dataframe(df_predict_hist_arg)
# DS_predict_test_hist.to_netcdf('netcdf_present.nc')

shift_2012 = DS_predict_test_hist['Yield'].sel(time=2012) / DS_predict_test_hist['Yield'].mean(['time']) 
plot_2d_map(shift_2012)
# shift_2012.to_netcdf('shifter_2012_arg.nc')


# RF:EPIC
df_predict_epic_hist = df_obs_arg_det_clip.copy()
predict_epic_hist = model_epic_arg.predict(df_epic_arg_det_clip.values)
df_predict_epic_hist.loc[:,"Yield"] = predict_epic_hist
DS_predict_epic_hist = xr.Dataset.from_dataframe(df_predict_epic_hist)


# RF:ECE
df_predict_clim_hist = df_obs_arg_det_clip.copy()
predict_clim_hist = model_exclim_dyn_arg.predict(df_feature_season_6mon_arg.values)
df_predict_clim_hist.loc[:,"Yield"] = predict_clim_hist
DS_predict_clim_hist = xr.Dataset.from_dataframe(df_predict_clim_hist)

# PLOTS
plt.figure(figsize=(10,6), dpi=300) #plot clusters
plt.plot(DS_y_epic_arg_det.time, DS_y_epic_arg_det.mean(['lat','lon']), label = 'Original EPIC',linestyle='dashed',linewidth=3)
plt.plot(DS_y_obs_arg_det.time, DS_y_obs_arg_det.mean(['lat','lon']), label = 'Obs', linestyle='dashed',linewidth=3)
plt.plot(DS_predict_epic_hist.time, DS_predict_epic_hist['Yield'].mean(['lat','lon']), label = 'RF:EPIC')
plt.plot(DS_predict_clim_hist.time, DS_predict_clim_hist['Yield'].mean(['lat','lon']), label = 'RF:ECE')
plt.plot(DS_predict_test_hist.time, DS_predict_test_hist['Yield'].mean(['lat','lon']), label = 'RF:Hybrid')
plt.title('Absolute shocks')
plt.legend()
plt.show()

# Scaled
plt.figure(figsize=(10,6), dpi=300) #plot clusters
plt.plot(DS_y_epic_arg_det.time, DS_y_epic_arg_det.mean(['lat','lon'])/DS_y_epic_arg_det.mean(), label = 'Original EPIC',linestyle='dashed',linewidth=3)
plt.plot(DS_y_obs_arg_det.time, DS_y_obs_arg_det.mean(['lat','lon'])/DS_y_obs_arg_det.mean(), label = 'Obs', linestyle='dashed',linewidth=3)
plt.plot(DS_predict_epic_hist.time, DS_predict_epic_hist['Yield'].mean(['lat','lon'])/DS_predict_epic_hist['Yield'].mean(), label = 'RF:EPIC')
plt.plot(DS_predict_clim_hist.time, DS_predict_clim_hist['Yield'].mean(['lat','lon'])/DS_predict_clim_hist['Yield'].mean(), label = 'RF:ECE')
plt.plot(DS_predict_test_hist.time, DS_predict_test_hist['Yield'].mean(['lat','lon'])/DS_predict_test_hist['Yield'].mean(), label = 'RF:Hybrid')
plt.title('Scaled shocks')
plt.legend()
plt.show()


### WIEGHTED ANALYSIS
DS_harvest_area_globiom =  xr.open_dataset("../../paper_hybrid_agri/data/americas_mask_ha.nc", decode_times=False).rename({'latitude': 'lat', 'longitude': 'lon'}) # xr.open_dataset('../../paper_hybrid_agri/data/soy_harvest_area_globiom_05x05_2b.nc').mean('time')
DS_harvest_area_globiom = DS_harvest_area_globiom.rename({'annual_area_harvested_rfc_crop08_ha_30mn':'harvest_area'}).where(DS_predict_test_hist['Yield']>=0)
plot_2d_am_map(DS_harvest_area_globiom['harvest_area'].isel(time = 0))
plot_2d_am_map(DS_harvest_area_globiom['harvest_area'].isel(time = -1))

total_area = DS_harvest_area_globiom['harvest_area'].sum(['lat','lon'])
DS_obs_weighted = ((DS_y_obs_arg_det * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'Yield')
DS_hybrid_weighted = ((DS_predict_test_hist['Yield'] * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'Yield')
DS_epic_weighted = ((DS_predict_epic_hist['Yield'] * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'Yield')
DS_clim_weighted = ((DS_predict_clim_hist['Yield'] * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'Yield')
DS_epic_orig_weighted =((DS_y_epic_arg_det * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'Yield') 

# Weighted plot
plt.figure(figsize=(8,5), dpi=300) #plot clusters
# plt.plot(DS_predict_epic_hist.time, DS_epic_orig_weighted['Yield'].sum(['lat','lon']), label = 'Obs', linestyle='dashed',linewidth=3)
plt.plot(DS_predict_epic_hist.time, DS_obs_weighted['Yield'].sum(['lat','lon']), label = 'Obs', linestyle='dashed',linewidth=3)
plt.plot(DS_predict_epic_hist.time, DS_epic_weighted['Yield'].sum(['lat','lon']), label = 'RF:EPIC')
plt.plot(DS_predict_epic_hist.time, DS_clim_weighted['Yield'].sum(['lat','lon']), label = 'RF:CLIM')
plt.plot(DS_predict_epic_hist.time, DS_hybrid_weighted['Yield'].sum(['lat','lon']), label = 'RF:hybrid')
plt.title('Weighted comparisons')
plt.ylabel('Yield (ton/ha)')
plt.xlabel('Years')
plt.legend()
plt.show()

print("Weighted R2 OBS-EPIC:",round(r2_score(DS_obs_weighted['Yield'].sum(['lat','lon']), DS_epic_orig_weighted['Yield'].sum(['lat','lon'])),2))
print("Weighted R2 OBS-RF:EPIC:",round(r2_score(DS_obs_weighted['Yield'].sum(['lat','lon']), DS_epic_weighted['Yield'].sum(['lat','lon'])),2))
print("Weighted R2 OBS-Clim:",round(r2_score(DS_obs_weighted['Yield'].sum(['lat','lon']), DS_clim_weighted['Yield'].sum(['lat','lon'])),2))
print("Weighted R2 OBS-Hybrid:",round(r2_score(DS_obs_weighted['Yield'].sum(['lat','lon']), DS_hybrid_weighted['Yield'].sum(['lat','lon'])),2))
print("Weighted R2 Hybrid-clim:",round(r2_score(DS_hybrid_weighted['Yield'].sum(['lat','lon']), DS_clim_weighted['Yield'].sum(['lat','lon'])),2))


# Scatter plots
plt.figure(figsize=(5,5), dpi=250) #plot clusters
plt.scatter(df_obs_arg_det['Yield'],df_epic_arg_det['yield-soy-noirr'])
plt.plot(df_obs_arg_det['Yield'].sort_values(), df_obs_arg_det['Yield'].sort_values(), linestyle = '--' , color = 'black', label = '1:1 line')
plt.ylabel('Original EPIC predicted yield')
plt.xlabel('Observed yield')
plt.legend()
# plt.savefig('paper_figures/epic_argda_validation.png', format='png', dpi=500)
plt.show()

plt.figure(figsize=(5,5), dpi=250) #plot clusters
plt.scatter(df_obs_arg_det_clip['Yield'],df_predict_hist_arg['Yield'])
plt.plot(df_obs_arg_det['Yield'].sort_values(), df_obs_arg_det['Yield'].sort_values(), linestyle = '--' , color = 'black', label = '1:1 line')
plt.ylabel('Hybrid predicted yield')
plt.xlabel('Observed yield')
plt.legend()
# plt.savefig('paper_figures/epic_argda_validation.png', format='png', dpi=500)
plt.show()


plt.figure(figsize=(5,5), dpi=250) #plot clusters
plt.scatter(df_obs_arg_det_clip['Yield'],df_predict_epic_hist['Yield'])
plt.plot(df_obs_arg_det['Yield'].sort_values(), df_obs_arg_det['Yield'].sort_values(), linestyle = '--' , color = 'black', label = '1:1 line')
plt.ylabel('EPIC predicted yield')
plt.xlabel('Observed yield')
plt.legend()
# plt.savefig('paper_figures/epic_argda_validation.png', format='png', dpi=500)
plt.show()


plt.figure(figsize=(5,5), dpi=250) #plot clusters
plt.scatter(df_obs_arg_det_clip['Yield'],df_predict_clim_hist['Yield'])
plt.plot(df_obs_arg_det['Yield'].sort_values(), df_obs_arg_det['Yield'].sort_values(), linestyle = '--' , color = 'black', label = '1:1 line')
plt.ylabel('RF clim predicted yield')
plt.xlabel('Observed yield')
plt.legend()
# plt.savefig('paper_figures/epic_argda_validation.png', format='png', dpi=500)
plt.show()

print("R2 OBS-EPIC_original:",round(r2_score(df_obs_arg_det['Yield'], df_epic_arg_det['yield-soy-noirr']),2))
print("R2 OBS-RF:EPIC:",round(r2_score(df_obs_arg_det_clip['Yield'].sort_values(), df_predict_epic_hist['Yield'].sort_values()),2))
print("R2 OBS-RF:Clim:",round(r2_score(df_obs_arg_det_clip['Yield'].sort_values(), df_predict_clim_hist['Yield'].sort_values()),2))
print("R2 OBS-Hybrid:",round(r2_score(df_obs_arg_det_clip['Yield'].sort_values(), df_predict_hist_arg['Yield'].sort_values()),2))




