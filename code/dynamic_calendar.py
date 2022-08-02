# =============================================================================
# Relative dates functions - shift and reshape
# =============================================================================
# Reshape to have each calendar year on the columns (1..12)
def reshape_data(dataframe):  #converts and reshape data
    #If already dataframe, skip the convertsion
    if isinstance(dataframe, pd.Series):    
        dataframe = dataframe.to_frame()
        
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
    ### Convert to dataframe and shift according to input -> if shift time is 0, then nothing is shifted
    dataframe_1 = dataset.shift(time=-shift_time).to_dataframe()    
    
    # Define the column names based on the type of data - dataframe or dataarray
    if type(dataset) == xr.core.dataset.Dataset:
        print('dataset mode')
        column_names = [var_name +"_"+str(j) for var_name in dataset.data_vars 
                        for j in range(1 + shift_time, 13 + shift_time)]
        
    elif type(dataset) == xr.core.dataarray.DataArray: 
        print('dataArray mode') 
        column_names = [dataset.name +"_"+str(j) for j in range(1+shift_time,13+shift_time)]
    else:
        raise ValueError('Data must be either Dataset ot DataArray.')
        
    # Reshape dataframe
    dataframe_reshape = reshape_data(dataframe_1)
    dataframe_reshape.columns = column_names      
    return dataframe_reshape

def calendar_multiyear_adjust(month_list, df_entry, mode = "shift_one_year"):
    df = df_entry.copy()
    month_list = np.sort(month_list)[::-1]
    for month in month_list:
        if df.loc[df == month].sum() > 0:
            print(f'There are planting dates on the following year (y+1) for month {month}')
            if mode == "shift_one_year":
                df.loc[df == month] = 12 + month
            elif mode == 'erase':
                df.loc[df == month] = np.nan
        else:
            print(f'No planting dates for month {month}')
    return df.to_frame().dropna()

def dynamic_calendar(DS, df_calendar_month_am):
    # First reshape each year to make a 24 month calendar
    df_clim_shift = reshape_shift(DS)
    df_clim_shift_12 = reshape_shift(DS, shift_time = 12)
    # Combine both dataframes and constraint it to be below 2016 just in case
    df_clim_reshape_twoyears = df_clim_shift.dropna().join(df_clim_shift_12)
    ### Join and change name to S for the shift values
    df_feature_reshape_shift = df_clim_reshape_twoyears.dropna().join(df_calendar_month_am)
    
    list_df_feature_reshape_shift = []
    # Divide the dataset by climatic feature so the shifting does not mix the different variables together
    for feature in list(DS.keys()):
    
        df_feature_reshape_shift_var = pd.concat([df_feature_reshape_shift.loc[:,'plant'], df_feature_reshape_shift.filter(like=feature)], axis = 1)
        
        # Shift accoording to month indicator (hence +1) - SLOW   // tried with query but also slow.. what to do?
        list_shifted_variables = [df_feature_reshape_shift_var.shift(-(int(indicator))+1, axis = 1).where(indicator == df_feature_reshape_shift_var['plant']).dropna( how = 'all') 
                                  for indicator in np.unique(df_feature_reshape_shift_var['plant'])]
        
        df_feature_reshape_shift_var = pd.concat(list_shifted_variables).sort_index().drop(columns=['plant'])
        list_df_feature_reshape_shift.append(df_feature_reshape_shift_var)

    return  pd.concat(list_df_feature_reshape_shift, axis=1)


# =============================================================================
# Load calendars ############################################################
# =============================================================================
DS_cal_ggcmi_test = xr.open_dataset('../../paper_hybrid_agri/data/soy_rf_ggcmi_crop_calendar_phase3_v1.01.nc4')



# Convert DS to df
df_chosen_calendar = DS_cal_ggcmi_test.to_dataframe().dropna()
# Rounding up planting dates to closest integer in the month scale
df_calendar_month_am = df_chosen_calendar[['plant']].apply(np.rint).astype(np.float32)

# transform the months that are early in the year to the next year (ex 1 -> 13). Attention as this should be done only for the south america region
df_calendar_month_am = calendar_multiyear_adjust([1,2,3,4,5], df_calendar_month_am['plant'])

# Define the maturity date and then subtract X months
df_calendar_month_am['plant'] = df_calendar_month_am['plant'] - 1

### LOAD climate date and clip to the calendar cells    
DS_exclim_am_det_clip = DS_exclim_am_det.sel(time=slice(f'{DS_y_obs_am_det.time[0].values}-01-01','2016-12-31')).where(DS_cal_ggcmi_test >= 0 )
plot_2d_am_map(DS_exclim_am_det_clip['prcptot'].mean('time'))
DS_exclim_am_det_clip.resample(time="1MS").mean(dim="time")

# =============================================================================
# CONVERT CLIMATIC VARIABLES ACCORDING TO THE SOYBEAN GROWING SEASON PER GRIDCELL 
# =============================================================================
df_features_reshape_2years_am = dynamic_calendar(DS_exclim_am_det_clip, df_calendar_month_am)
