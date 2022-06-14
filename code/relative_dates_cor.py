# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 09:37:00 2021

@author: morenodu
"""

import os
os.chdir('C:/Users/morenodu/OneDrive - Stichting Deltares/Documents/PhD/paper_hybrid_agri/data')

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

#%% HADEX V3 - NEW TEST

DS_calendar_plant = xr.open_dataset('../../Paper_drought/data/soy_rf_pd_2015soc.nc').mean('time') / (365/12)
plot_2d_map(DS_calendar_plant['Calendar'])

DS_calendar_mature = xr.open_dataset('soy_rf_md_2015soc_2.nc').mean('time') / (365/12)
plot_2d_map(DS_calendar_mature['Calendar'])

DS_cal_sachs = xr.open_dataset('../../paper_hybrid_agri/data/Soybeans.crop.calendar_sachs_05x05.nc') 
DS_cal_sachs = DS_cal_sachs.where(DS_y_obs_up_clip_det.mean('time') >= 0.0 )
DS_cal_sachs_month = DS_cal_sachs / (365/12)
plot_2d_map( (DS_cal_sachs_month['plant'] ))

DS_cal_abr = xr.open_dataset('../../paper_hybrid_agri/data/calendar_soybeans/calendar_v15_05x05_2.nc')
DS_cal_abr['time'] = pd.date_range(start='1973', periods=DS_cal_abr.sizes['time'], freq='YS').year
# crop to obs yield values
DS_cal_abr = DS_cal_abr.where(DS_y_obs_up_clip_det >= 0.0 )
DS_cal_abr_mean = DS_cal_abr.mean('time') / (365/12)
DS_cal_abr_mean['mraendcal'].attrs = {'long_name': 'Last planting date', 'units':'month'}

df_cal_abr_mean = DS_cal_abr_mean.to_dataframe()
df_cal_sachs_month = DS_cal_sachs_month['plant'].to_dataframe().dropna()

plot_2d_map( (DS_cal_abr_mean['mraendcal'] ))
# plot_2d_map(DS_cal_abr_clip['metendcal'].sel(time=2000))
plot_2d_map(DS_cal_abr['metendcal'].sel(time=2000))


start_date, end_date = '01-01-1980','30-12-2016'

DS_hadex = xr.open_mfdataset('../../paper_hybrid_agri/data/climpact-master/climpact-master/www/output_gswp3/monthly_data/*.nc').sel(time=slice(start_date, end_date))

# New dataset
DS_hadex = DS_hadex.drop_vars('fd') # Always zero
DS_hadex = DS_hadex.drop_vars('id') # Always zero
DS_hadex = DS_hadex.drop_vars('time_bnds') # Always zero
DS_hadex = DS_hadex.drop_vars('spi') # Always zero
DS_hadex = DS_hadex.drop_vars('spei') # Always zero
DS_hadex = DS_hadex.drop('scale') # Always zero

DS_hadex = DS_hadex[list_features_br] 

plot_2d_map(DS_hadex['prcptot'].mean('time'))

def timedelta_to_int(DS, var):
    da_timedelta = DS[var].dt.days
    da_timedelta = da_timedelta.rename(var)
    da_timedelta.attrs["units"] = 'days'
    
    return da_timedelta

da_list = []
for feature in list(DS_hadex.keys()):
    if (type(DS_hadex[feature].values[0,0,0]) == type(DS_hadex.r10mm.values[0,0,0])):
        print('Time')
        DS = timedelta_to_int(DS_hadex, feature)
    else:
        print('Integer')
        DS = DS_hadex[feature]
    
    da_list.append(DS)

DS_hadex_combined = xr.merge(da_list)    

# DS_hadex_combined = DS_hadex_combined.rename({'latitude': 'lat', 'longitude': 'lon'})
DS_hadex_combined.coords['lon'] = (DS_hadex_combined.coords['lon'] + 180) % 360 - 180
DS_hadex_combined = DS_hadex_combined.sortby(DS_hadex_combined.lon)
DS_hadex_combined = DS_hadex_combined.reindex(lat=DS_hadex_combined.lat[::-1])
if len(DS_hadex_combined.coords) >3 :
    DS_hadex_combined=DS_hadex_combined.drop('spatial_ref')
    
DS_hadex_combined_det = detrend_dataset(DS_hadex_combined)

DS_hadex_combined_br_det = mask_shape_border(DS_hadex_combined_det, soy_brs_states)
plot_2d_map(DS_hadex_combined_br['prcptot'].mean('time'))


DS_hadex_combined_br_det_test_2 = DS_hadex_combined_br_det
DS_hadex_combined_br_det_test_2 = DS_hadex_combined_br_det_test_2.where(DS_cal_abr_mean['metendcal'] > 0)
plot_2d_map(DS_hadex_combined_br_det_test_2['prcptot'].mean('time'))
plot_2d_map(DS_cal_abr_mean['metendcal'] )

### Convert planting days to beginning of the month
df_call_br_month = (df_cal_abr_mean[['metendcal']] / (365/12) ).apply(np.floor).astype('Int64')
# df_call_br_month = df_call_br_month.where(df_call_br_month.isna(), 13 ) # TEST FOR ONE VALUE 13 #############
# df_call_br_month = DS_calendar_plant['Calendar'].to_dataframe().apply(np.floor).astype('Int64')

df_call_br_sachs = (df_cal_sachs_month ).apply(np.floor).astype('Int64')

# # Convert e-18 to NAs
# for feature in list(DS_hadex_combined_br_det_test_2.keys()):
#     print(DS_hadex_combined_br_det_test_2[feature].name, DS_hadex_combined_br_det_test_2[feature].min().values)
# DS_hadex_combined_br_det_test_2 = DS_hadex_combined_br_det_test_2.where(DS_hadex_combined_br_det_test_2['r10mm'] > -10000)
# for feature in list(DS_hadex_combined_br_det_test_2.keys()):
#     print(DS_hadex_combined_br_det_test_2[feature].name, DS_hadex_combined_br_det_test_2[feature].min().values)

# For loop along features to obtain 24 months of climatic data for each year
list_features_reshape_shift = []
for feature in list(DS_hadex_combined_br_det_test_2.keys()):
    ### Reshape and shift for 24 months for every year.
    df_test_shift = reshape_shift(DS_hadex_combined_br_det_test_2[feature])
    df_test_shift_12 = reshape_shift(DS_hadex_combined_br_det_test_2[feature], shift_time = 12)
    # Combine both dataframes
    df_test_reshape_twoyears = df_test_shift.dropna().join(df_test_shift_12)
    # Remove last year, because we do not have two years for it
    df_test_reshape_twoyears = df_test_reshape_twoyears.query('year < 2016')
    ### Join and change name to S for the shift values
    df_feature_reshape_shift = (df_test_reshape_twoyears.dropna().join(df_call_br_month)
                                .rename(columns={'metendcal':'s'}))
    # Move 
    col = df_feature_reshape_shift.pop("s")
    df_feature_reshape_shift.insert(0, col.name, col)
    # Activate this if error "TypeError: int() argument must be a string, a bytes-like object or a number, not 'NAType'" occurs
    
    # print(df_feature_reshape_shift[['s']].isna().sum())

    # Shift accoording to month indicator (hence +1)
    df_feature_reshape_shift = (df_feature_reshape_shift.apply(lambda x : x.shift(-(int(x['s']))+1) , axis=1)
                                .drop(columns=['s']))
    
    
    list_features_reshape_shift.append(df_feature_reshape_shift)

# Transform into dataframe
df_features_reshape_2years = pd.concat(list_features_reshape_shift, axis=1)

### Select specific months
suffixes = tuple(["_"+str(j) for j in range(2,5)])
df_feature_season_6mon = df_features_reshape_2years.loc[:,df_features_reshape_2years.columns.str.endswith(suffixes)]

# # Subjective to monthly means, but we can think of different techniques
# df_feature_season_2mon_mean = df_feature_season_6mon.groupby(np.arange(len(df_feature_season_6mon.columns))// 2, axis=1).mean()
# df_feature_season_2mon_mean.columns = df_hadex_combined_br_season.columns
# df_feature_season_2mon_mean.index.names = df_features_reshape_2years.index.names

df_feature_season_6mon_test = df_feature_season_6mon

# Shift 1 year
df_feature_season_6mon_test.index = df_feature_season_6mon_test.index.set_levels(df_feature_season_6mon_test.index.levels[2] + 1, level=2)



df_feature_season_6mon_test = df_feature_season_6mon_test.rename_axis(index={'year':'time'})
df_feature_season_6mon_test = df_feature_season_6mon_test.reorder_levels(['time','lat','lon']).sort_index()
df_feature_season_6mon_test = df_feature_season_6mon_test.where(df_hadex_combined_br_season['prcptot']>=0).dropna().astype(float)
df_test2 = df_test2.where(df_feature_season_6mon_test['prcptot_2']>=0).dropna()
df_feature_season_6mon_test = df_feature_season_6mon_test.where(df_test2['Yield']>=0).dropna()

feature_importance_selection(df_feature_season_6mon_test, df_test2)
X, y = df_feature_season_6mon_test.values, df_test2.values.ravel()

# # Tune hyper-parameters #######################################
# params_cv_chosen_clim, best_grid_clim = hyper_param_tuning(X,y)

# # Save hyper-parameters #################################
# with open('params_cv_chosen_clim.pickle', 'wb') as f:
#     pickle.dump(params_cv_chosen_clim, f)

y_pred_clim, y_pred_total_clim, model_clim, full_model_clim = calibration(X,y)

# # Tunned model
# with open('../../paper_hybrid_agri/data/params_cv_chosen_clim.pickle', 'rb') as f:
#     params_cv_chosen_clim = pickle.load(f)
    
# y_pred_clim, y_pred_total_clim, model_clim, full_model_clim = calibration(X,y,params_cv_chosen_clim)



#### BENCHMARK FOR CALENDAR CHANGES ##############################################################################################
# For loop along features to obtain 24 months of climatic data for each year
list_static_calendar = []
for feature in list(DS_hadex_combined_br_det_test_2.keys()):
    ### Reshape and shift for 24 months for every year.
    df_test_shift = reshape_shift(DS_hadex_combined_br_det_test_2[feature])
    df_test_shift_12 = reshape_shift(DS_hadex_combined_br_det_test_2[feature], shift_time = 12)
    # Combine both dataframes
    df_test_reshape_twoyears = df_test_shift.dropna().join(df_test_shift_12)
    list_static_calendar.append(df_test_reshape_twoyears)


# Transform into dataframe
df_cal_benchmark = pd.concat(list_static_calendar, axis=1)

### Select specific months
suffixes_stat = tuple(["_"+str(j) for j in range(13,16)])
df_cal_benchmark_season = df_cal_benchmark.loc[:,df_cal_benchmark.columns.str.endswith(suffixes_stat)]
df_cal_benchmark_season = df_cal_benchmark_season.rename_axis(index={'year':'time'}).reorder_levels(['time','lat','lon']).sort_index()
df_cal_benchmark_season = df_cal_benchmark_season.where(df_hadex_combined_br_season['prcptot']>=-10).dropna().astype(float)

df_test3 = test2.to_dataframe().dropna()
df_test3 = df_test3.reorder_levels(['time','lat','lon']).sort_index()
df_test3 = df_test3.where(df_cal_benchmark_season['prcptot_13']>=-10).dropna()


# # TEST TO SEE IF AGGREGATING IMPROVES --------------- alternative formation to aggregate values and improve performance
# df_cal_benchmark_season_mean =  pd.concat([df_cal_benchmark_season.iloc[:,0:2].mean(axis=1),
#                                           df_cal_benchmark_season.iloc[:,2:4].mean(axis=1), 
#                                           df_cal_benchmark_season.iloc[:,4:6].mean(axis=1),
#                                           df_cal_benchmark_season.iloc[:,6:8].mean(axis=1),
#                                           df_cal_benchmark_season.iloc[:,8:10].mean(axis=1),
#                                           df_cal_benchmark_season.iloc[:,10:12].mean(axis=1),
#                                           df_cal_benchmark_season.iloc[:,12:14].mean(axis=1),
#                                           df_cal_benchmark_season.iloc[:,14:16].mean(axis=1),
#                                           df_cal_benchmark_season.iloc[:,16:18].mean(axis=1),
#                                           df_cal_benchmark_season.iloc[:,18:20].mean(axis=1)],axis=1)
# df_cal_benchmark_season_mean.columns=['dtr_7_8', 'precip_7_8', 'r10mm_7_8','rx7day_7_8','tnm_7_8', 'tr_7_8', 'txge35_7_8', 'txm_7_8', 'tmm_7_8','tnn_7_8']


feature_importance_selection(df_cal_benchmark_season, df_test3)
X, y = df_cal_benchmark_season.values, df_test3.values.ravel()

y_pred_exclim_stat, y_pred_total_exclim_stat, model_exclim_stat, full_model_exclim_stat = calibration(X, y)



# Subjective to monthly means, but we can think of different techniques
df_feature_season_2mon_mean = df_feature_season_6mon_test.groupby(np.arange(len(df_feature_season_6mon.columns))// 3, axis=1).mean()
df_feature_season_2mon_mean.columns = df_hadex_combined_br_season.columns
# df_feature_season_2mon_mean.index.names = df_features_reshape_2years.index.names

# aggregate
feature_importance_selection(df_feature_season_2mon_mean, df_test2)
X, y = df_feature_season_2mon_mean.values, df_test2.values.ravel()
y_pred_clim_red, y_pred_total_clim_red, model_clim_red = calibration(X,y)

### VALIDATION

# This should be false, but if true: Shift years
df_hadex_combined_br_season.loc[(1991, -33.25, -53.25)] == df_feature_season_2mon_mean.loc[(-33.25, -53.25,1990)]

# Shift 1 year
df_feature_season_2mon_mean.index = df_feature_season_2mon_mean.index.set_levels(df_feature_season_2mon_mean.index.levels[2] + 1, level=2)

#  This should be true for validation
df_hadex_combined_br_season.loc[(1990, -33.25, -53.25)] == df_feature_season_2mon_mean.loc[(-33.25, -53.25,1990)]






