# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 10:11:24 2022

@author: morenodu
"""

# Different ways to detrend, select the best one
def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    
    da_det = da - fit
    
    
    residual = np.nansum((da-fit)**2)
    print(f'Residuals for degree {deg}: {residual}')
    return da_det





list_residuals_detrend =[]
for degree in [1,2,3]:
    
    p = np.polyfit(x= x, y = line_to_be_used, deg = degree)
    fit = np.polyval(p, x)
    
    detrend = line_to_be_used - fit
    
    
    plt.plot(x, line_to_be_used, 'r-', linewidth=3)
    plt.plot(x, detrend, 'b-', linewidth=3)
    
    
    res_detrend = np.nansum((line_test_notrend-detrend)**2)
    
    dict_res = {degree:res_detrend}
    list_residuals_detrend.append(dict_res)
    print(res_detrend)

from collections import ChainMap
dict_res = dict(ChainMap(*list_residuals_detrend))

min_degree = min(dict_res, key=dict_res.get)




dict_res = {}
for degree in [1,2,3]:
    
    p = np.polyfit(x= x, y = line_to_be_used, deg = degree)
    fit = np.polyval(p, x)
    
    detrend = line_to_be_used - fit
    
    
    plt.plot(x, line_to_be_used, 'r-', linewidth=3)
    plt.plot(x, detrend, 'b-', linewidth=3)
    
    
    res_detrend = np.nansum((line_test_notrend-detrend)**2)
    
    dict_res_in = {degree:res_detrend}
    dict_res.update(dict_res_in)
    print(res_detrend)

min_degree = min(dict_res, key=dict_res.get)


# Different ways to detrend, select the best one
def detrend_dim(da, dim, deg = 'free', print_res = True):        
    if deg == 'free':
        
        da_zero_mean = da.where( da < np.nanmin(da.values), other = 0 )

        dict_res = {}
        for degree in [1,2,3]:
            # detrend along a single dimension
            p = da.polyfit(dim=dim, deg=degree)
            fit = xr.polyval(da[dim], p.polyfit_coefficients)
            
            da_det = da - fit
            res_detrend = np.nansum((da_zero_mean-da_det)**2)
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


detrend_1 = detrend_dim(DS_clim_ext_projections['prcptot'], 'time', deg = 'free')


test_x = DS_clim_ext_projections['prcptot'].where(
    DS_clim_ext_projections['prcptot'] < np.nanmin(DS_clim_ext_projections['prcptot'].values), other = 0 )


# MODEL
model = 'ukesm'
# one year change
rcp_scenario = 'ssp585' # 'ssp126', 'ssp585'
region = "_us" #""
start_date, end_date = '01-01-2015','31-12-2100'

DS_clim_ext_projections = xr.open_mfdataset('monthly_'+ model +'_'+ rcp_scenario + region +'/*.nc').sel(time=slice(start_date, end_date))
DS_clim_ext_projections = DS_clim_ext_projections.where( DS_chosen_calendar >= 0 )

DS_clim_ext_projections['txx'].groupby('time.year').mean().mean(['lat','lon']).plot()
DS_clim_ext_projections['prcptot'].groupby('time.year').mean().mean(['lat','lon']).plot()
DS_clim_ext_projections['dtr'].groupby('time.year').mean().mean(['lat','lon']).plot()

detrend_1 = detrend_dim(DS_clim_ext_projections['prcptot'], 'time', deg = 1)
detrend_2 = detrend_dim(DS_clim_ext_projections['prcptot'], 'time', deg = 2)
detrend_3 = detrend_dim(DS_clim_ext_projections['prcptot'], 'time', deg = 3)

detrend_1.groupby('time.year').mean().mean(['lat','lon']).plot()
detrend_2.groupby('time.year').mean().mean(['lat','lon']).plot()
detrend_3.groupby('time.year').mean().mean(['lat','lon']).plot()

test = (DS_clim_ext_projections['prcptot'] - detrend_1)**2
np.nansum(test.values)


line_test = np.linspace(0.0, 4.0, num=10)
line_test_sq = np.square(line_test)
line_test_3 = line_test**3
x = range(0,10)

line_to_be_used = line_test_3
line_test_notrend = np.linspace(0,0, num=10)

list_residuals_detrend =[]
for degree in [1,2,3]:
    
    p = np.polyfit(x= x, y = line_to_be_used, deg = degree)
    fit = np.polyval(p, x)
    
    detrend = line_to_be_used - fit
    
    
    plt.plot(x, line_to_be_used, 'r-', linewidth=3)
    plt.plot(x, detrend, 'b-', linewidth=3)
    
    
    residual = np.nansum((detrend)**2)
    res_detrend = np.nansum((line_test_notrend-detrend)**2)
    dict_res = {degree:res_detrend}
    list_residuals_detrend.append(dict_res)
    print(res_detrend)

from collections import ChainMap
dict_res = dict(ChainMap(*list_residuals_detrend))

min_degree = min(dict_res, key=dict_res.get)



# Detrend Dataset
def detrend_dataset(DS):
    px= DS.polyfit(dim='time', deg=1)
    fitx = xr.polyval(DS['time'], px)
    dict_name = dict(zip(list(fitx.keys()), list(DS.keys())))
    fitx = fitx.rename(dict_name)
    DS_det  = (DS - fitx) + DS.mean('time')
    return DS_det




# Detrend Dataset
def detrend_dataset(DS, deg = 'free', dim = 'time', print_res = True):
    
    if deg == 'free':
        da_list = []
        for feature in list(DS.keys()):
            da = DS[feature]
            print(feature)
            da_zero_mean = da.where( da < np.nanmin(da.values), other = 0 )
    
            dict_res = {}
            for degree in [1,2,3]:
                # detrend along a single dimension
                p = da.polyfit(dim=dim, deg=degree)
                fit = xr.polyval(da[dim], p.polyfit_coefficients)
                
                da_det = da - fit
                
                res_detrend = np.nansum((da_zero_mean-da_det)**2)
                dict_res.update({degree:res_detrend})
            if print_res == True:
                print(dict_res)
            deg = min(dict_res, key=dict_res.get) # minimum degree   
            
            # detrend along a single dimension
            print('Chosen degree is ', deg)
            p = da.polyfit(dim=dim, deg=deg)
            fit = xr.polyval(da[dim], p.polyfit_coefficients)
        
            da_det = da - fit
            da_det.name = feature
            da_list.append(da_det)
        DS_det = xr.merge(da_list) + DS.mean('time')
    
    else:       
        px= DS.polyfit(dim='time', deg=deg)
        fitx = xr.polyval(DS['time'], px)
        dict_name = dict(zip(list(fitx.keys()), list(DS.keys())))
        fitx = fitx.rename(dict_name)
        DS_det  = (DS - fitx) + DS.mean('time')
        
    return DS_det

test_ds = DS_exclim_us[['prcptot','txm']]

ds_detrended = detrend_dataset(test_ds)

ds_detrended_fix = detrend_dataset(test_ds, deg = 1)

timeseries_free = ds_detrended['prcptot'].groupby('time.year').mean().mean(['lat','lon'])
timeseries_fix = ds_detrended_fix['prcptot'].groupby('time.year').mean().mean(['lat','lon'])


ds_detrended['txm'].groupby('time.year').mean().mean(['lat','lon']).plot(label = 'free')
ds_detrended_fix['txm'].groupby('time.year').mean().mean(['lat','lon']).plot(label = f'deg 1')
plt.legend()

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(ds_detrended['prcptot'].time.values.reshape(-1, 1), ds_detrended['prcptot'].mean(['lat','lon']).values)
trend = model.predict(ds_detrended['prcptot'].time.values.reshape(-1, 1))

model2 = LinearRegression()
model2.fit(timeseries_free.year.values.reshape(-1, 1), timeseries_free.values)
trend2 = model2.predict(timeseries_free.year.values.reshape(-1, 1))


plt.plot(trend)
plt.plot(trend2)

ds_detrended['prcptot'].groupby('time.year').mean().mean(['lat','lon']).plot(label = 'free')
ds_detrended_fix['prcptot'].groupby('time.year').mean().mean(['lat','lon']).plot(label = f'deg 1')
plt.plot(timeseries_free.year, trend)
plt.plot(timeseries_free.year, trend2)
plt.legend()
