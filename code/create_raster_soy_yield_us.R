library(terra)
library(tidyr)
library(dplyr)
library(tidyverse)
library(USAboundaries)
library(stars)

# FOR HARVEST YIELD_us
setwd("~/PhD/paper_hybrid_agri/data")

crop_dir <- ("C:/users/morenodu/OneDrive - Stichting Deltares/Documents/PhD/Paper_drought/data/usda_data")

# usda_yields <- read.csv("soy_usda_obs_1960_2019.csv") %>% dplyr::select(County,State,Year,Value)

usda_yields <- list.files(crop_dir, "usda", full.names = TRUE) %>%
  map_dfr(read.csv)  %>%
  arrange(Year)  %>% dplyr::select(County,State,Year,Value)

usda_yields$Value <- usda_yields$Value * 0.06725106937

usda_yields <- subset(usda_yields, (County !="SEVIER" | State  != 'ARKANSAS' | Year != '1989')) 

us_spatial_sf<-  function(x) {
  #x is either us_counties or us_states from usaboundaries package
  x %>%
    filter(!state_name  %in%
             c("Alaska","Puerto Rico","Hawaii","Washington",
               "Oregon","California","Nevada","Idaho","Utah",
               "Arizona","New Mexico","Colorado","Wyoming","Montana"))%>%
    rename(County = name,
           State  = state_name) %>%
    dplyr::select(County,State,aland,geometry) %>%
    mutate_at(vars(State,County), ~toupper(.))
}

us_counties_subset <- us_counties() %>% dplyr::select(c(6,9,11,15))

usda_spatial_full <- us_counties_subset %>%
  us_spatial_sf %>%
  left_join(usda_yields, by = c("County", "State")) %>%
  drop_na()

usda_spatial_yield <-usda_spatial_full %>%
  dplyr::select(County,State,Year,Value,geometry)

usda_spatial_2010 <- usda_spatial_yield[usda_spatial_yield$Year == 1980,]
usda_spatial_2010["Value"] %>% plot()

extent_us = c(-180, 180, -90, 90)
resolution_us = 0.05
ncol_us = (extent_us[2]-extent_us[1])/resolution_us
nrows_us = (extent_us[4]-extent_us[3])/resolution_us

# Raster creation
baserast <- rast(nrows=nrows_us, ncol=ncol_us,
                 extent= extent_us, # CHANGE TO US
                 crs="+proj=longlat +datum=WGS84",
                 vals=NA)

rasters <- rast(lapply(1975:2020,
                       function(x)
                         rasterize(vect(usda_spatial_yield %>%
                                          filter(Year==x)),baserast,"Value")))
names(rasters) <- 1975:2020
varnames(rasters) <- paste0("soy_yield_usda",1975:2020)

writeRaster(rasters,"soy_yields_US_all_1975_2020_001.tif",overwrite=TRUE)

#########################################################################################
#### ADD filter for 1% of removal - needs file create_raster_harvest_area_us.R loaded
# setwd("~/PhD/paper_hybrid_agri/data")

# Calculate harvest area %
total_area <- usda_spatial_full %>%
  dplyr::select(County,State,Year,aland,geometry) %>% dplyr::mutate(aland = aland/10000)


usda_spatial_yield$ratio_area <- usda_spatial_area$Value / total_area$aland

sum(usda_spatial_yield$ratio_area < 0.01, na.rm=T) 
sum(usda_spatial_yield$Value[usda_spatial_yield$ratio_area < 0.01], na.rm=T)

usda_spatial_yield$Value <- with(usda_spatial_yield,replace(Value, ratio_area < 0.01, NA))
sum(usda_spatial_yield$Value[usda_spatial_yield$ratio_area < 0.01], na.rm=T)


usda_spatial_2010 <- usda_spatial_yield[usda_spatial_yield$Year == 1980,]
usda_spatial_2010["Value"] %>% plot()

rasters <- rast(lapply(1975:2020,
                       function(x)
                         rasterize(vect(usda_spatial_yield %>%
                                          filter(Year==x)),baserast,"Value")))
names(rasters) <- 1975:2020
varnames(rasters) <- paste0("soy_yield_usda",1975:2020)

writeRaster(rasters,"soy_yields_US_all_1975_2020_1prc_002.tif",overwrite=TRUE)




# 
# ################################################################################
# # VECTORISE EPIC OUTPUT
# rdir <- ("C:/users/morenodu/OneDrive - Stichting Deltares/Documents/PhD/Paper_drought/data")
# epic_yield_us = read_ncdf(paste(rdir,"ACY_gswp3-w5e5_obsclim_2015soc_default_soy_noirr.nc",sep=""))
# 
# 
# us_climate <- us_climate_xy%>%
#   st_as_sf(x = ., coords = c("x", "y"),
#            crs = "+proj=longlat +datum=WGS84 +ellps=WGS84 +towgs84=0,0,0")
# 
# 
# #======================================================================
# #Aggregate grid based climate information to county scale
# df_point_list <- split(dplyr::select(us_climate, -Year),
#                        us_climate$Year)
# #Split sf usda_resid country by Years
# df_poly_list <- split(usda_resid_county, usda_resid_county$Year)
# #join yield and climate sf, na are reproduced as climate_df is filtered
# full_Sf_yield_climate <- map2_dfr(df_poly_list, df_point_list,
#                                   ~ .x %>%
#                                     st_join(.y, left =FALSE) %>%
#                                     group_by(County,Year,State) %>%
#                                     mutate_at(vars(-geometry,-State,-County,-Year),~(mean(.,na.rm = TRUE))))
# #do you want a simple spatial averages?
# #summarize_at(vars(-geometry),~(mean(.,na.rm = TRUE))))
# ####
# 
