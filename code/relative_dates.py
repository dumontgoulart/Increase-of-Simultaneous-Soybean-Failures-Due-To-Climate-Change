# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 16:23:57 2021

@author: morenodu
"""
# Identify the months of growing season, first and last ---- A_new = A.where(A.time.isin(B.time), drop=True)

DS_calendar = xr.open_dataset('../../paper_hybrid_agri/data/calendar_soybeans/Planting_st_SOJAGRUPO I.nc')/ (365/12)
DS_harvest = xr.open_dataset('../../paper_hybrid_agri/data/calendar_soybeans/Harvesting_SOJAGRUPO_II.nc')/ (365/12)
DS_calendar['plant.start'].plot()
DS_harvest['harvest.end'].plot()
# Create a mask with the selected months
mask_months = DS_hadex_combined_br_det.sel(time=is_month(DS_hadex_combined_br_det['time.month'], 1, 2))
mask_months2 = DS_hadex_combined_br_det.sel(time=DS_hadex_combined_br_det.time.dt.month.isin([1,2]))

df_calendar = DS_calendar['plant.start'].to_dataframe()
df_harvest = DS_harvest['harvest.end'].to_dataframe()

# Test
test_mask = df_harvest.where(df_harvest['harvest.end'] > df_calendar['plant.start'], True, False)

xr.where( DS_hadex_combined_br_det.time )

# Cut the dataset to the selected months chosen for each location
DS_hadex_combined_br_det_test = DS_hadex_combined_br_det.where(DS_hadex_combined_br_det.time.isin( mask_months.time), drop = True )
DS_hadex_combined_br_det_test = DS_hadex_combined_br_det_test.groupby('time.year').mean('time')
DS_hadex_combined_br_det_test = DS_hadex_combined_br_det_test.rename({'year':'time'})
DS_hadex_combined_br_det_test = DS_hadex_combined_br_det_test.reindex(lat=DS_hadex_combined_br_det_test.lat[::-1])
DS_hadex_combined_br_det_test = DS_hadex_combined_br_det_test.where(DS_y_obs_up_clip_det >= 0.0 )
# DS_hadex_combined_br_season.to_netcdf('ds_clim.nc')



month_to_consider = [1,2]

DS_hadex_combined_br_det = detrend_dataset(DS_hadex_combined_br)
# DS_hadex_combined_br_det = DS_hadex_combined_br_det[['ETR','DTR', 'R10mm', 'Rx5day']]
# Select months
DS_hadex_combined_br_season = DS_hadex_combined_br_det.sel(time=is_month(DS_hadex_combined_br_det['time.month'], 1,2))
# Average across season
DS_hadex_combined_br_season = DS_hadex_combined_br_season.groupby('time.year').mean('time')
DS_hadex_combined_br_season = DS_hadex_combined_br_season.rename({'year':'time'})
DS_hadex_combined_br_season = DS_hadex_combined_br_season.reindex(lat=DS_hadex_combined_br_season.lat[::-1])
DS_hadex_combined_br_season = DS_hadex_combined_br_season.where(DS_y_obs_up_clip_det >= 0.0 )
# DS_hadex_combined_br_season.to_netcdf('ds_clim.nc')


DS_hadex_combined_br_det_test.txx.groupby('time').mean(...).plot()
DS_hadex_combined_br_season.txx.groupby('time').mean(...).plot()
plot_2d_map(DS_hadex_combined_br_det_test['tx90p'].mean('time'))
plot_2d_map(DS_hadex_combined_br_season['tx90p'].mean('time'))


#### 2nd trial
# Monthly climate data -> 

DS_cal_abr = xr.open_dataset('../../paper_hybrid_agri/data/calendar_soybeans/calendar_v15_05x05.nc')
DS_cal_abr['time'] = pd.date_range(start='1973', periods=DS_cal_abr.sizes['time'], freq='YS').year
# crop to obs yield values
DS_cal_abr_clip = DS_cal_abr.where(DS_y_obs_up_clip_det >= 0.0 )
DS_cal_abr_mean = DS_cal_abr_clip.mean('time')
df_cal_abr_mean = DS_cal_abr_mean.to_dataframe()

df_hadex_combined_br_det = DS_hadex_combined_br_det.to_dataframe()


plot_2d_map(DS_cal_abr_clip['metendcal'].mean('time') / (365/12))

df_test = df_hadex_combined_br_det

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

# columns dataframe
# column_names = [i+"_"+str(j) for i in list(df_test['dtr'].columns) for j in range(1,13)]
column_names = [df_test['dtr'].name +"_"+str(j) for j in range(1,13)]
    
# Reshape
df_feature_reshape = reshape_data(df_test['dtr'])
df_feature_reshape.columns = column_names

# Should be checked if the join approach is working as it should be
df_reshape_cal = df_feature_reshape.join(df_cal_abr_mean['metstacal'] / (365/12) )
df_reshape_cal = df_reshape_cal.join(df_cal_abr_mean['metendcal'] / (365/12) )

df_reshape_cal_nextyear = df_reshape_cal.shift(1)
# Change months to follow the initial date according to the beginning of planting date.
# if suffix  -2 of columns > metendcal, 
df_reshape_cal

df_reshape_cal.loc[(-27.25, -53.25)]


df_reshape_cal.loc[:, df_reshape_cal.columns.str.endswith("_2")]


df_reshape_cal.loc[:, df_reshape_cal.gt(2).any()]

df_test.loc[df_test['metstacal']==df_test['metstacal'].max()]


test_1 = pd.DataFrame([[1,2,3,4], [10,12,13,14], [20, 22, 23,24]])
test_1.columns = ['a','b','c','d']
test_1.shift(-1, axis=1)

df_x = pd.DataFrame([[0],[1],[2]], index = test_1.index, columns = ['ab'])

test_1.apply(lambda x: x.shift(periods = -df_x.loc[x,'ab']), axis = 1)

m=test_1.index.isin(df_x['ab'].index)
test_2=test_1.loc[m].apply(lambda x:x.shift(-(x.name)),axis=1)

test_1 = test_1.T

for val in df_x.values:
    test_2[val['ab']] = test_1[val['ab']].shift(periods=-val['ab'])



test_1 = pd.DataFrame([[1,2,3,4], [10,12,13,14], [20, 22, 23, 24]])
# test_1 = test_1.T
df_x = pd.DataFrame([[1],[1],[1]])
# for val in df_x.values:
#     test_1[val[0]] = test_1[val[0]].shift(periods=-val[0])
# test_1 = test_1.T


test_2=(df_x.rename(columns={0:'s'})
            .join(test_1)
            .apply(lambda x:x.shift(-(x['s'])),axis=1)
            .drop(columns=['s']))





#%% Pipeline for each variable - for loop on each column

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

### Shift dates because we want to work with previous years
DS_hadex_combined_br_det_test_2 = DS_hadex_combined_br_det.shift(time=12)

### Convert planting days to beginning of the month
df_call_br_month = (df_cal_abr_mean[['metendcal']] / (365/12) ).apply(np.floor).astype('Int64')
df_call_br_month = df_call_br_month.where(df_call_br_month.isna(), 13 ) # TEST FOR ONE VALUE 13 #############

# Convert e-18 to NAs
for feature in list(DS_hadex_combined_br_det_test_2.keys()):
    print(DS_hadex_combined_br_det_test_2[feature].name, DS_hadex_combined_br_det_test_2[feature].min().values)
DS_hadex_combined_br_det_test_2 = DS_hadex_combined_br_det_test_2.where(DS_hadex_combined_br_det_test_2 > -10000)


# For loop along features to obtain 24 months of climatic data for each year
list_features_reshape_shift = []
for feature in list(DS_hadex_combined_br_det_test_2.keys()):
    ### Reshape and shift for 24 months for every year.
    df_test_shift = reshape_shift(DS_hadex_combined_br_det_test_2[feature])
    df_test_shift_12 = reshape_shift(DS_hadex_combined_br_det_test_2[feature], shift_time = 12)
    # Combine both dataframes
    df_test_reshape_twoyears = df_test_shift.dropna().join(df_test_shift_12)
    
    ### Join and change name to S for the shift values
    df_feature_reshape_shift = (df_test_reshape_twoyears.dropna().join(df_call_br_month)
                                .rename(columns={'metendcal':'s'}))
    # Move 
    col = df_feature_reshape_shift.pop("s")
    df_feature_reshape_shift.insert(0, col.name, col)
    
    # Shift accoording to month indicator (hence +1)
    df_feature_reshape_shift = (df_feature_reshape_shift.apply(lambda x : x.shift(-(int(x['s']))+1) , axis=1)
                                .drop(columns=['s']))
    
    
    list_features_reshape_shift.append(df_feature_reshape_shift)

# Transform into dataframe
df_features_reshape_2years = pd.concat(list_features_reshape_shift, axis=1)

### Select specific months
suffixes = tuple(["_"+str(j) for j in range(1,3)])
df_feature_season_6mon = df_features_reshape_2years.loc[:,df_features_reshape_2years.columns.str.endswith(suffixes)]
# Subjective to monthly means, but we can think of different techniques
df_feature_season_2mon_mean = df_feature_season_6mon.groupby(np.arange(len(df_feature_season_6mon.columns))// 2, axis=1).mean()
df_feature_season_2mon_mean.columns = df_hadex_combined_br_season.columns




### VALIDATION
# This should be true for validation
df_hadex_combined_br_season.loc[(1991, -33.25, -53.25)] == df_feature_season_2mon_mean.loc[(-33.25, -53.25,1990)]

# This should be false, but if true: Shift years
df_hadex_combined_br_season.loc[(1991, -33.25, -53.25)] == df_feature_season_2mon_mean.loc[(-33.25, -53.25,1991)]










### Reshape and shift for 24 months for every year.
df_test_shift = reshape_shift(DS_hadex_combined_br_det['dtr'])
df_test_shift_12 = reshape_shift(DS_hadex_combined_br_det['dtr'], shift_time = 12)
# Combine both dataframes
df_test_reshape_twoyears = df_test_shift.dropna().join(df_test_shift_12)


### Convert planting days to beginning of the month
df_call_br_month = (df_cal_abr_mean[['metendcal']] / (365/12) ).apply(np.floor).astype('Int64')


### Join and change name to S for the shift values
df_feature_reshape_shift = (df_test_reshape_twoyears.dropna().join(df_call_br_month)
                            .rename(columns={'metendcal':'s'}))
# Move 
col = df_feature_reshape_shift.pop("s")
df_feature_reshape_shift.insert(0, col.name, col)

# Shift accoording to month indicator (hence +1)
df_feature_reshape_shift = (df_feature_reshape_shift.apply(lambda x : x.shift(-(int(x['s']))+1) , axis=1)
                            .drop(columns=['s']))

df_feature_season_6mon = df_feature_reshape_shift.loc[:,'dtr_1':'dtr_3']




# DUMB SCRIPT BUT WORKING - validation 
### Convert to dataframe
df_hadex_combined_br_det = DS_hadex_combined_br_det['dtr'].to_dataframe()

# Define the column names
column_names = [df_hadex_combined_br_det['dtr'].name +"_"+str(j) for j in range(1,13)]
    
# Reshape
df_feature_reshape = reshape_data(df_hadex_combined_br_det['dtr'])
df_feature_reshape.columns = column_names

### One year later dataframe
df_hadex_combined_br_det_nextyear = DS_hadex_combined_br_det.shift(time=-12).to_dataframe()

# Define the column names
column_names_next_year = [df_hadex_combined_br_det_nextyear['dtr'].name +"_"+str(j) for j in range(13,25)]

# Reshape
df_hadex_combined_br_det_nextyear_reshape = reshape_data(df_hadex_combined_br_det_nextyear['dtr'])
df_hadex_combined_br_det_nextyear_reshape.columns = column_names_next_year

# Combine both dataframes
df_feature_reshape_twoyears = df_feature_reshape.dropna().join(df_hadex_combined_br_det_nextyear_reshape)


### Convert planting days to beginning of the month
df_call_br_month = (df_cal_abr_mean[['metendcal']] / (365/12) ).apply(np.floor).astype('Int64')


### Join and change name to S for the shift values
df_feature_reshape_shift = (df_feature_reshape_twoyears.dropna().join(df_call_br_month)
                            .rename(columns={'metendcal':'s'}))
# Move 
col = df_feature_reshape_shift.pop("s")
df_feature_reshape_shift.insert(0, col.name, col)

# Shift accoording to month indicator (hence +1)
df_feature_reshape_shift = (df_feature_reshape_shift.apply(lambda x : x.shift(-(int(x['s']))+1) , axis=1)
                            .drop(columns=['s']))

df_feature_season_6mon = df_feature_reshape_shift.loc[:,'dtr_1':'dtr_3']










test_1 = DS_hadex_combined_br_det['tnx'].sel(lat=slice(-27.75, -27.25), lon=slice(-53.75,-53.25), time=slice('01-01-1989','31-12-1990'))
test_1_shift = test_1.shift(time=-12)

df_feature_reshape_shift = df_feature_reshape.T 
for val in list(df_call_br_month.values.flatten()): 
    df_feature_reshape_shift = df_feature_reshape.shift(periods=-val)
    line+=1  
df_feature_reshape_shift = df_feature_reshape_shift.T         
df_feature_reshape_shift.loc[(-25.25, -48.75, 2014)]

# THIS WORKS BUT TAKES A LOT OF TIME
df_feature_reshape_shift_list = []
for coords in df_feature_reshape.index:
    coords_notime = (coords[0], coords[1])
    if df_call_br_month.loc[coords_notime].isna().values == True:
        continue
        
    df_feature_reshape_shift = df_feature_reshape.loc[[coords]].shift(periods = -df_call_br_month.loc[coords_notime][0], axis = 1)
    df_feature_reshape_shift_list.append(df_feature_reshape_shift)

df_feature_reshape_shift_df = pd.concat(df_feature_reshape_shift_list, axis=1)
df_feature_reshape_shift_df = df_feature_reshape_shift_df.T

df_feature_reshape.loc[[(-27.25, -53.25, 2000)]].shift(periods = -df_call_br_month.loc[(-27.25, -53.25)][0], axis = 1)


for i in df_feature_reshape_shift['s']:
    print(df_feature_reshape_shift['s'].iloc[0])
    