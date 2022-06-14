# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 20:04:22 2022

@author: morenodu
"""

import os
os.chdir('C:/Users/morenodu/OneDrive - Stichting Deltares/Documents/PhD/Paper_drought/data')
from mask_shape_border import mask_shape_border
from failure_probability import feature_importance_selection, failure_probability

os.chdir('C:/Users/morenodu/OneDrive - Stichting Deltares/Documents/PhD/paper_hybrid_agri/data')
import xarray as xr 
import rioxarray
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


# load netcdf file
import xarray as xr 
import rioxarray

xds = xr.open_dataset('output_shocks_br/Gen_Assem/hybrid_ukesm1-0-ll_ssp585_2015co2_yield_soybean_shift_2017-2044.nc')
xds = xds.rename({'lat':'y','lon':'x', 'time':'band'})

new_dim = np.arange(xds.y[0], xds.y[-1], 0.5)

xds_new = xds.reindex({'y':new_dim})

corr_test = xr.corr(xds["yield-soy-noirr"], xds_new["yield-soy-noirr"])

corr_test.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True, levels = 10)
# Add CRS
xds.rio.write_crs("epsg:4326", inplace=True)

# Convert to geotiff
xds["yield-soy-noirr"].rio.to_raster('output_shocks_br/Gen_Assem/hybrid_ukesm1-0-ll_ssp585_2015co2_yield_soybean_shift_2017-2044.tif')
rio = xr.open_rasterio("output_shocks_br/Gen_Assem/hybrid_ukesm1-0-ll_ssp585_2015co2_yield_soybean_shift_2017-2044.tif")

!rio info hybrid_gfdl-esm4_ssp126_2015co2_yield_soybean_shift_2017-2044_test.tif

# Check sizes
size_x = (xds.x[-1].values-xds.x[0].values) / (len(xds.x) - 1)
size_y = (xds.y[-1].values-xds.y[0].values) / (len(xds.y) - 1)


xds2 = xr.open_dataset('hybrid_yield_soybean_shift_2017-2044.nc')
xds2 = xds2.rename({'lat':'y','lon':'x', 'time':'band'})

# Add CRS
xds2.rio.write_crs("epsg:4326", inplace=True)

# Convert to geotiff
xds2["yield-soy-noirr"].rio.to_raster('hybrid_gfdl-esm4_ssp126_2015co2_yield_soybean_shift_2017-2044_test.tif')
rio2 = xr.open_rasterio("hybrid_gfdl-esm4_ssp126_2015co2_yield_soybean_shift_2017-2044_test.tif")

!rio2 info hybrid_gfdl-esm4_ssp126_2015co2_yield_soybean_shift_2017-2044_test.tif

# Check sizes
size_x2 = (xds2.x[-1].values-xds2.x[0].values) / (len(xds2.x) - 1)
size_y2 = (xds2.y[-1].values-xds2.y[0].values) / (len(xds2.y) - 1)
