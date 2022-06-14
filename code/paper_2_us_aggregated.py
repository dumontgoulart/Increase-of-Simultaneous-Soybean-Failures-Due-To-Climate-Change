# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 17:33:36 2021

@author: morenodu
"""
df_obs_agg_us= DS_y_obs_us_det.mean(['lat','lon']).to_dataframe().dropna()
df_epic_agg_us = DS_y_epic_us_det.mean(['lat','lon']).to_dataframe().dropna()

DS_y_obs_us_det.mean('time').plot()
plt.show()
DS_y_epic_us_det.mean('time').plot()
### WIEGHTED ANALYSIS
DS_harvest_area_globiom = xr.open_dataset('../../paper_hybrid_agri/data/soy_usa_harvest_area_05x05.nc')
DS_harvest_area_globiom['harvest_area'] = DS_harvest_area_globiom['harvest_area'].where(DS_predict_test_hist['Yield']>0)

total_area = DS_harvest_area_globiom['harvest_area'].sum(['lat','lon'])
DS_obs_weighted = ((DS_y_obs_us_det * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'Yield')
DS_epic_orig_weighted =((DS_y_epic_us_det * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'Yield') 

# SUM IT ALL
df_obs_weighted = DS_obs_weighted['Yield'].sum(['lat','lon']).to_dataframe()
df_obs_weighted=df_obs_weighted.rename(columns={'Yield': 'usda_yield'})
df_epic_weighted = DS_epic_orig_weighted['Yield'].sum(['lat','lon']).to_dataframe()

# Plot
plt.figure(figsize=(8,5), dpi=300) #plot clusters
plt.plot(df_obs_agg_us, label = 'Observed')
plt.plot(df_epic_agg_us, label = 'EPIC')
plt.vlines(df_epic_agg_us.index, 1.5,4.5, linestyles ='dashed', colors = 'k')
plt.legend()
plt.show()

# Weighted plot
plt.figure(figsize=(8,5), dpi=300) #plot clusters
plt.plot(df_obs_weighted.index, df_obs_weighted, label = 'Obs', linestyle='dashed',linewidth=3)
plt.plot(df_epic_weighted.index, df_epic_weighted, label = 'RF:EPIC')
plt.vlines(df_epic_agg_us.index, 1.5,4.5, linestyles ='dashed', colors = 'k')
plt.ylabel('Yield (ton/ha)')
plt.xlabel('Years')
plt.legend()
plt.show()

# Pearson's correlation
from scipy.stats import pearsonr

corr_grouped, _ = pearsonr(df_obs_agg_us.values.flatten(), df_epic_agg_us.values.flatten())
print('Pearsons correlation mean: %.3f' % corr_grouped)

corr_grouped, _ = pearsonr(df_obs_weighted.values.flatten(), df_epic_weighted.values.flatten())
print('Pearsons correlation weighted: %.3f' % corr_grouped)


#%% Machine learning model training
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error,  explained_variance_score
from sklearn.inspection import permutation_importance

def calibration(X,y, params = None):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    
    # define the model
    if params is None:
        model = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1, 
                                      max_depth = 20, max_features = 'auto',
                                           min_samples_leaf = 1, min_samples_split=2)
        model.fit(X_train, y_train)
        
        full_model = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1, 
                                           max_depth = 20, max_features = 'auto',
                                           min_samples_leaf = 1, min_samples_split=2).fit(X, y) 

    
    if params is not None:
        model = RandomForestRegressor(n_estimators=params['n_estimators'], random_state=0, n_jobs=-1, 
                                      max_depth = params['max_depth'], max_features = params['max_features'],
                                      min_samples_leaf = params['min_samples_leaf'], min_samples_split = params['min_samples_split'])
        model.fit(X_train, y_train)
        
        full_model = RandomForestRegressor(n_estimators=params['n_estimators'], random_state=0, n_jobs=-1, 
                                      max_depth = params['max_depth'], max_features = params['max_features'],
                                      min_samples_leaf = params['min_samples_leaf'], min_samples_split = params['min_samples_split']).fit(X, y) 

    def MBE(y_true, y_pred):
        '''
        Parameters:
            y_true (array): Array of observed values
            y_pred (array): Array of prediction values
    
        Returns:
            mbe (float): Biais score
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
    y_pred_total = model.predict(X)
    
    

    # perform permutation importance
    results = permutation_importance(model, X_test, y_test, scoring='neg_mean_squared_error', n_repeats=10, random_state=0, n_jobs=-1)
    # get importance
    df_importance = pd.DataFrame(results.importances_mean)
    df_importance.index = X.columns
    print(df_importance)
    # summarize feature importance
    plt.figure(figsize=(12,5)) #plot clusters
    plt.bar(df_importance.index, df_importance[0])
    plt.show()
    
    return y_pred, y_pred_total, model, full_model 


from sklearn.model_selection import GridSearchCV

def hyper_param_tuning(X,y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    
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

X, y = df_epic_weighted, df_obs_weighted.values.ravel()

# ------------------------------------------------------------------------------------------------------------------
# Tune hyper-parameters
params_cv_chosen_epic_agg_us, best_grid_epic_us = hyper_param_tuning(X, y)

# Save hyper-parameters
with open('params_cv_chosen_epic_agg_us.pickle', 'wb') as f:
    pickle.dump(params_cv_chosen_epic_agg_us, f)
# ------------------------------------------------------------------------------------------------------------------

# Standard model
print('Standard EPIC results:')
y_pred_epic_agg_us, y_pred_total_epic_agg_us, model_epic_agg_us, full_model_agg_epic_agg_us  = calibration(X, y)

# Tunned model
with open('params_cv_chosen_epic_agg_us.pickle', 'rb') as f:
    params_cv_chosen_epic_agg_us = pickle.load(f)
    
y_pred_epic_agg_us, y_pred_total_epic_agg_us, model_epic_agg_us, full_model_epic_agg_us = calibration(X, y, params = params_cv_chosen_epic_agg_us )

#%% EXTREME CLIMATE

df_clim_agg_us = df_feature_season_6mon.groupby(level='time').mean()


DS_clim_weighted =((df_feature_season_6mon.to_xarray() * DS_harvest_area_globiom['harvest_area'] ) / total_area)

# SUM IT ALL
df_clim_weighted = DS_clim_weighted.sum(['lat','lon']).to_dataframe()

#plot
df_clim_agg_us['txm_3'].plot()
df_clim_weighted['txm_3'].plot()


feature_importance_selection(df_clim_weighted, df_obs_weighted)
X, y = df_clim_weighted, df_obs_weighted.values.flatten().ravel()
print('Standard ECE results:')

y_pred_exclim_dyn_agg_us, y_pred_total_exclim_dyn_agg_us, model_exclim_dyn_agg_us, full_model_exclim_dyn_agg_us = calibration(X, y)

#%% HYBRID

df_hybrid_agg_us = pd.concat([df_epic_weighted, df_clim_weighted], axis = 1)
X, y = df_hybrid_agg_us, df_obs_weighted.values.flatten().ravel()

## Tune hyper-parameters ###################################################
params_cv_chosen_hybrid_agg_us, best_grid_hybrid_agg_us = hyper_param_tuning(X, y)
###
# Save hyper-parameters
with open('params_cv_chosen_hybrid_agg_us.pickle', 'wb') as f:
    pickle.dump(params_cv_chosen_hybrid_agg_us, f)
#############################################################################

# Feature selection
feature_importance_selection(df_hybrid_agg_us, df_obs_weighted)

# Evaluate Model
print('Standard Hybrid results:')
y_pred_hyb_agg_us, y_pred_total_hyb_agg_us, model_hyb_agg_us, full_model_hyb_agg_us = calibration(X, y)


# Tunned model
with open('params_cv_chosen_hybrid_agg_us.pickle', 'rb') as f:
    params_cv_chosen_hybrid_us = pickle.load(f)
y_pred_hyb2, y_pred_total_hyb2, model_hyb_agg_us2, full_model_hyb2 = calibration(X, y, params_cv_chosen_hybrid_agg_us)

for test_size in [0.1,0.2,0.3,0.4,0.5]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    
    regr_rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=0)
    regr_rf.fit(X_train, y_train)
    
    y_rf = regr_rf.predict(X_test)
    
    print(f"R2 {test_size} OBS-RF:EPIC:",round(r2_score(y_test, y_rf),2))


#%%
# Compare timelines
df_predict_test_agg_hist = df_epic_weighted.copy()
predict_test_hist = model_hyb_agg_us2.predict(df_hybrid_agg_us)
df_predict_test_agg_hist.loc[:,"Yield"] = predict_test_hist
DS_predict_test_agg_hist = xr.Dataset.from_dataframe(df_predict_test_agg_hist)

# RF:EPIC
df_predict_epic_agg_hist = df_epic_weighted.copy()
predict_epic_hist = model_epic_agg_us.predict(df_epic_weighted)
df_predict_epic_agg_hist.loc[:,"Yield"] = predict_epic_hist
DS_predict_epic_agg_hist = xr.Dataset.from_dataframe(df_predict_epic_agg_hist)


# RF:ECE
df_predict_clim_agg_hist = df_epic_weighted.copy()
predict_clim_hist = model_exclim_dyn_agg_us.predict(df_clim_weighted)
df_predict_clim_agg_hist.loc[:,"Yield"] = predict_clim_hist
DS_predict_clim_agg_hist = xr.Dataset.from_dataframe(df_predict_clim_agg_hist)

# PLOTS
plt.figure(figsize=(10,6), dpi=300) #plot clusters
plt.plot(df_epic_weighted.index, df_epic_weighted, label = 'Original EPIC',linestyle='dashed',linewidth=3)
plt.plot(df_obs_weighted.index, df_obs_weighted, label = 'Obs', linestyle='dashed',linewidth=3)
plt.plot(DS_predict_epic_agg_hist.time, DS_predict_epic_agg_hist['Yield'], label = 'RF:EPIC')
plt.plot(DS_predict_clim_agg_hist.time, DS_predict_clim_agg_hist['Yield'], label = 'RF:ECE')
plt.plot(DS_predict_test_agg_hist.time, DS_predict_test_agg_hist['Yield'], label = 'RF:Hybrid')
plt.legend()
plt.show()


# R2 score full dataset
print("R2 OBS-EPIC_original:",round(r2_score(df_obs_weighted['usda_yield'], df_epic_weighted['Yield']),2))
print("R2 OBS-RF:EPIC:",round(r2_score(df_obs_weighted['usda_yield'], df_predict_epic_agg_hist['Yield']),2))
print("R2 OBS-RF:Clim:",round(r2_score(df_obs_weighted['usda_yield'], df_predict_clim_agg_hist['Yield']),2))
print("R2 OBS-Hybrid:",round(r2_score(df_obs_weighted['usda_yield'], df_predict_test_agg_hist['Yield']),2))

#%%
df_obs_train_us, df_obs_test_us = train_test_split(df_obs_weighted, test_size=0.4, random_state=0)

def score_list(metric, name):
    print('ALL YEARS (mixed)')
    print(f"{name} EPIC:",round(metric(df_obs_weighted, df_predict_epic_agg_hist),3))
    # print(f"{name} climatic Indices:",round(metric(df_obs_mean_det, df_pred_clim_total),3))
    print(f"{name} Extreme Indices:",round(metric(df_obs_weighted, df_predict_clim_agg_hist),3))
    print(f"{name} Hybrid:",round(metric(df_obs_weighted, df_predict_test_agg_hist),3), '\n')
    
def score_list_oos(metric, name):
    print('TEST YEARS (Out of sample)')
    print(f"{name} EPIC :",round(metric(df_obs_test_us, pd.DataFrame(y_pred_epic_agg_us, index = df_obs_test_us.index)),3))
    # print(f"{name} climatic Indices:",round(metric(df_obs_test, df_pred_clim),3))
    print(f"{name} Extreme Indices:",round(metric(df_obs_test_us, pd.DataFrame(y_pred_exclim_dyn_agg_us, index = df_obs_test_us.index)),3))
    print(f"{name} Hybrid:",round(metric(df_obs_test_us, pd.DataFrame(y_pred_hyb_agg_us, index = df_obs_test_us.index)),3), '\n')

score_list(r2_score, 'R2')
score_list(mean_absolute_error, 'Mean absolute error')
score_list(mean_squared_error, 'Mean squared error')
    
score_list_oos(r2_score, 'R2')
score_list_oos(mean_absolute_error, 'Mean absolute error')
score_list_oos(mean_squared_error, 'Mean squared error')
    

