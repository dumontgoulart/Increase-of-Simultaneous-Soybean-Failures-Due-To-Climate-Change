# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 09:59:37 2021

@author: morenodu
"""
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

X, y = df_hybrid_us.copy(), df_obs_us_det_clip['usda_yield'].copy().values.flatten().ravel()

# scaler = StandardScaler().fit(X)
# X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#%%


parameter_space = {
    'hidden_layer_sizes': [ (1000,1000,1000), (400,400,400,400),(600,600,600,600)],
    'max_iter': [500],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [0.0001,0.05],
    'learning_rate': ['constant'],
}

model = MLPRegressor(random_state=0)

# # define grid 'kernel': 1**2 * RationalQuadratic(alpha=1, length_scale=1) <<< It's not working, too much memory
# # define search
search = GridSearchCV(model, parameter_space, cv=3, n_jobs=-1,  verbose = 2)
# perform the search
results = search.fit(X_train, y_train)
# summarize best
print('Best Mean Accuracy: %.3f' % results.best_score_)
print('Best Config: %s' % results.best_params_)
# summarize all
means = results.cv_results_['mean_test_score']
params = results.cv_results_['params']
for mean, param in zip(means, params):
    print(">%.3f with: %r" % (mean, param))


# model_rf = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1, 
#                               max_depth = 20, max_features = 'auto',
#                                    min_samples_leaf = 1, min_samples_split=2)

# estimators = [
#     ('RF',make_pipeline(StandardScaler(), model_rf)), #make_pipeline(StandardScaler(),
#     ('MLP', make_pipeline(StandardScaler(),MLPRegressor(random_state=0, max_iter= 500, alpha= 0.05,hidden_layer_sizes= (50, 100, 50)))) #
#                ]
# Get together the models:
    # Best results so far : random_state=0, hidden_layer_sizes =  (200,200,200,200), alpha = 0.0001, max_iter=400
stacking_regressor = (make_pipeline(StandardScaler(), MLPRegressor(random_state=0, hidden_layer_sizes =  (400,400,400, 400), alpha = 0.0001, max_iter=400, verbose =2 ) )) #StackingRegressor(estimators=estimators) #  #SVR() # GaussianProcessRegressor(kernel = 1**2 * RationalQuadratic(alpha=1, length_scale=1)) #StackingRegressor(estimators=estimators)
stacking_regressor_full = (make_pipeline(StandardScaler(), MLPRegressor(random_state=0, hidden_layer_sizes =  (400,400,400, 400), alpha = 0.0001, max_iter=400, verbose =2 ) )) #StackingRegressor(estimators=estimators) #  #SVR() # GaussianProcessRegressor(kernel = 1**2 * RationalQuadratic(alpha=1, length_scale=1)) #StackingRegressor(estimators=estimators)

model = stacking_regressor.fit(X_train, y_train)
full_model = stacking_regressor_full.fit(X, y)

# Test performance
y_pred = model.predict(X_test)

# report performance
print("R2 on test set:", round(r2_score(y_test, y_pred),2))
print("Var score on test set:", round(explained_variance_score(y_test, y_pred),2))
print("MAE on test set:", round(mean_absolute_error(y_test, y_pred),5))
print("RMSE on test set:",round(mean_squared_error(y_test, y_pred, squared=False),5))


X_delta = X.copy() * 1.1
# X_delta['txm_5'] = X['txm_5'].copy() * 1.5



from sklearn.inspection import PartialDependenceDisplay

features_to_plot = [0,6]
fig, ax1 = plt.subplots(1, 1, figsize=(10, 8), dpi=500)
disp1 = PartialDependenceDisplay.from_estimator(full_model, X_delta, features_to_plot, pd_line_kw={'color':'r'},percentiles=(0.01,0.99),  ax = ax1)
disp2 = PartialDependenceDisplay.from_estimator(full_model, X, features_to_plot, ax = disp1.axes_,percentiles=(0.01,0.99),  pd_line_kw={'color':'k'})
plt.setp(disp1.deciles_vlines_, visible=False)
plt.setp(disp2.deciles_vlines_, visible=False)
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
disp1.plot(ax=[ax1, ax2], line_kw={"label": "Extrapolation", "color": "red"})
disp2.plot(ax=[ax1, ax2], line_kw={"label": "Training", "color": "black"})
ax1.set_ylim(1, 2.8)
ax2.set_ylim(1, 2.8)
plt.setp(disp1.deciles_vlines_, visible=False)
plt.setp(disp2.deciles_vlines_, visible=False)
ax1.legend()
plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
disp1.plot(ax=[ax1, ax2], line_kw={"label": "Extrapolation", "color": "red"})
disp2.plot(ax=[ax1, ax2], line_kw={"label": "Training", "color": "black"})
ax1.set_ylim(-1, 2.8)
ax2.set_ylim(-1, 2.8)
plt.setp(disp1.deciles_vlines_, visible=False)
plt.setp(disp2.deciles_vlines_, visible=False)
ax1.legend()
plt.show()


#%% FUNCTION TO GENERATE PROJECTIONS BASED ON RANDOM FOREST
def projections_generation(model, rcp_scenario, region, start_date, end_date, co2_scen='both'):
    DS_clim_ext_projections = xr.open_mfdataset('monthly_'+ model +'_'+ rcp_scenario + region +'/*.nc').sel(time=slice(start_date, end_date))
    
    if model == 'ukesm':
        model_full = 'ukesm1-0-ll'
    elif model == 'gfdl':
        model_full = 'gfdl-esm4'
    elif model == 'ipsl':
        model_full = 'ipsl-cm6a-lr'

#%% Climatic variables - Extreme weather    
    # Clean
    DS_clim_ext_projections = DS_clim_ext_projections.drop_vars('fd') # Always zero
    DS_clim_ext_projections = DS_clim_ext_projections.drop_vars('id') # Always zero
    DS_clim_ext_projections = DS_clim_ext_projections.drop_vars('time_bnds') # Always zero
    # DS_clim_ext_projections = DS_clim_ext_projections.drop_vars('spi') # Always zero
    # DS_clim_ext_projections = DS_clim_ext_projections.drop_vars('spei') # Always zero
    # DS_clim_ext_projections = DS_clim_ext_projections.drop('scale') # Always zero
    
    # Selected features
    DS_clim_ext_projections = DS_clim_ext_projections[list_features]
    DS_clim_ext_projections = DS_clim_ext_projections.where(DS_y_obs_us['usda_yield'].mean('time') >= -5.0 )

    
    da_list = []
    for feature in list(DS_clim_ext_projections.keys()):
        if (type(DS_clim_ext_projections[feature].values[0,0,0]) == type(DS_clim_ext_projections.r10mm.values[0,0,0])):
            print('Time')
            DS = timedelta_to_int(DS_clim_ext_projections, feature)
        else:
            print('Integer')
            DS = DS_clim_ext_projections[feature]
        
        da_list.append(DS)
    
    DS_clim_ext_projections_combined = xr.merge(da_list)    
    DS_clim_ext_projections_combined = DS_clim_ext_projections_combined.drop_vars('r10mm') # Always zero

    DS_clim_ext_projections_combined.coords['lon'] = (DS_clim_ext_projections_combined.coords['lon'] + 180) % 360 - 180
    DS_clim_ext_projections_combined = DS_clim_ext_projections_combined.sortby(DS_clim_ext_projections_combined.lon)
    DS_clim_ext_projections_combined = DS_clim_ext_projections_combined.reindex(lat=DS_clim_ext_projections_combined.lat[::-1])
    if len(DS_clim_ext_projections_combined.coords) >3 :
        DS_clim_ext_projections_combined=DS_clim_ext_projections_combined.drop('spatial_ref')
        
    DS_clim_ext_projections_us = DS_clim_ext_projections_combined.where( DS_cal_sachs_month['plant'] >= 0 )
    
    plot_2d_us_map(DS_clim_ext_projections_us['prcptot'].mean('time'))
    
    DS_clim_ext_projections_us_det = detrend_dataset(DS_clim_ext_projections_us)
    DS_clim_ext_projections_us['prcptot'].mean(['lat','lon']).plot()
    DS_clim_ext_projections_us_det['prcptot'].mean(['lat','lon']).plot()
    plt.show()
    
    DS_clim_ext_projections_us['txm'].mean(['lat','lon']).plot()
    DS_clim_ext_projections_us_det['txm'].mean(['lat','lon']).plot()
    plt.show()
    
    
    # For loop along features to obtain 24 months of climatic data for each year
    list_features_reshape_shift = []
    for feature in list(DS_clim_ext_projections_us_det.keys()):
        ### Reshape
        df_test_shift = reshape_shift(DS_clim_ext_projections_us_det[feature])
        
        ### Join and change name to S for the shift values
        df_feature_reshape_shift = (df_test_shift.dropna().join(df_cal_sachs_month_harvest)
                                    .rename(columns={'plant':'s'}))
        # Move 
        col = df_feature_reshape_shift.pop("s")
        df_feature_reshape_shift.insert(0, col.name, col)
        df_feature_reshape_shift[['s']].isna().sum()
        nan_rows = df_feature_reshape_shift[['s']][df_feature_reshape_shift[['s']].isnull().T.any()]
        if nan_rows.empty == False:
            print('Missing crop calendar values!')
        
        # Shift accoording to month indicator (hence +1)
        df_feature_reshape_shift = (df_feature_reshape_shift.apply(lambda x : x.shift(-(int(x['s'])) + 1) , axis=1)
                                    .drop(columns=['s']))
        
        
        list_features_reshape_shift.append(df_feature_reshape_shift)
    
    # Transform into dataframe
    df_proj_features_reshape_shift = pd.concat(list_features_reshape_shift, axis=1)
    
    ### Select specific months
    suffixes = tuple(["_"+str(j) for j in range(3,6)])
    df_proj_feature_season_6mon = df_proj_features_reshape_shift.loc[:,df_proj_features_reshape_shift.columns.str.endswith(suffixes)]
    df_proj_feature_season_6mon = df_proj_feature_season_6mon.rename_axis(index={'year':'time'}).reorder_levels(['time','lat','lon']).sort_index()
    df_proj_feature_season_6mon = df_proj_feature_season_6mon.dropna()
    
    def epic_projections_function_co2(co2_scenario):
        DS_y_epic_proj = xr.open_dataset("epic-iiasa_"+ model_full +"_w5e5_"+rcp_scenario+"_2015soc_"+co2_scenario+"_yield-soy-noirr_global_annual_2015_2100.nc", decode_times=False)
        # Convert time unit
        units, reference_date = DS_y_epic_proj.time.attrs['units'].split('since')
        DS_y_epic_proj['time'] = pd.date_range(start=' 2015-01-01, 00:00:00', periods=DS_y_epic_proj.sizes['time'], freq='YS')
        DS_y_epic_proj['time'] = DS_y_epic_proj['time'].dt.year
        
        DS_y_epic_proj['yield-soy-noirr'].mean(['lat', 'lon']).plot()
        plt.title( model + "_"+ rcp_scenario+"_"+co2_scenario)
        plt.show()
        
        DS_y_epic_proj_us = DS_y_epic_proj.where(DS_y_obs_us['usda_yield'].mean('time') >= -5.0 )
        DS_y_epic_proj_us = DS_y_epic_proj_us.where(DS_clim_ext_projections_us['prcptot'].mean('time') >= -100.0 )
        
        DS_y_epic_proj_us_det = xr.DataArray( detrend_dim(DS_y_epic_proj_us['yield-soy-noirr'], 'time') + DS_y_epic_proj_us['yield-soy-noirr'].mean('time'), name= DS_y_epic_proj_us['yield-soy-noirr'].name, attrs = DS_y_epic_proj_us['yield-soy-noirr'].attrs)
        
        plot_2d_us_map(DS_y_epic_proj_us_det.mean('time'))
        
        DS_y_epic_proj_us['yield-soy-noirr'].mean(['lat', 'lon']).plot()
        DS_y_epic_proj_us_det.mean(['lat', 'lon']).plot()
        plt.title('Detrend'+ "_"+ model + "_"+ rcp_scenario+"_"+co2_scenario)
        plt.show()
        
        df_y_epic_proj_us = DS_y_epic_proj_us_det.to_dataframe().dropna()
        df_y_epic_proj_us = df_y_epic_proj_us.reorder_levels(['time','lat','lon']).sort_index()
        
        df_hybrid_proj_2 = pd.concat([df_y_epic_proj_us, df_proj_feature_season_6mon], axis = 1 )
        df_hybrid_proj_test_2 = df_hybrid_proj_2.query("time>=2015 and time <= 2100")
        return df_hybrid_proj_test_2
    if co2_scen == 'both':
        df_hybrid_proj_test_2_default = epic_projections_function_co2(co2_scenario = 'default')
        df_hybrid_proj_test_2_2015co2 = epic_projections_function_co2(co2_scenario = '2015co2')
        
    
    return df_hybrid_proj_test_2_default, df_hybrid_proj_test_2_2015co2


df_hybrid_proj_test_2_default, df_hybrid_proj_test_2_2015co2 = projections_generation(model = 'ukesm', rcp_scenario = 'ssp585', region = "_us", start_date='01-01-2015', end_date='31-12-2100', co2_scen = 'both')


# Test performance
df_hybrid_predictions = full_model.predict(df_hybrid_proj_test_2_2015co2.values)


df_hybrid_us_2 = df_hybrid_us.copy()
df_hybrid_us_2 = df_hybrid_us_2.rename(columns={'yield':'yield-soy-noirr_default'})
df_hybrid_us_2['yield-soy-noirr_2015co2'] = df_hybrid_us_2['yield-soy-noirr_default']



from sklearn.inspection import PartialDependenceDisplay

features_to_plot = [0,6]
fig, ax1 = plt.subplots(1, 1, figsize=(10, 8), dpi=500)
disp1 = PartialDependenceDisplay.from_estimator(full_model, df_hybrid_proj_test_2_2015co2, features_to_plot, pd_line_kw={'color':'r'},percentiles=(0.01,0.99), ax = ax1)
disp2 = PartialDependenceDisplay.from_estimator(full_model, df_hybrid_us_2.iloc[:,:-1], features_to_plot, ax = disp1.axes_,percentiles=(0.01,0.99), pd_line_kw={'color':'k'})
plt.setp(disp1.deciles_vlines_, visible=False)
plt.setp(disp2.deciles_vlines_, visible=False)
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
disp1.plot(ax=[ax1, ax2], line_kw={"label": "Extrapolation", "color": "red"})
disp2.plot(ax=[ax1, ax2], line_kw={"label": "Training", "color": "black"})
ax1.set_ylim(1, 2.8)
ax2.set_ylim(1, 2.8)
plt.setp(disp1.deciles_vlines_, visible=False)
plt.setp(disp2.deciles_vlines_, visible=False)
ax1.legend()
plt.show()