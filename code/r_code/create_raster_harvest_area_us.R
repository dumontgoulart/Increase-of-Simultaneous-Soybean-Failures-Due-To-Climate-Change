library(terra)
library(tidyr)
library(dplyr)
library(tidyverse)
library(USAboundaries)
library(stars)

# FOR HARVEST AREA_us
setwd("~/PhD/paper_hybrid_agri/data")

crop_dir <- ("C:/users/morenodu/OneDrive - Stichting Deltares/Documents/PhD/paper_hybrid_agri/data")

usda_area <- read.csv("soy_us_usda_harvest_area_1980_2020.csv") %>% dplyr::select(County,State,Year,Value)
usda_area<- usda_area %>% arrange(Year)  %>% dplyr::select(County,State,Year,Value)
# 
# usda_area <- list.files(crop_dir, "usda", full.names = TRUE) %>%
#   map_dfr(read.csv)  %>%
#   arrange(Year)  %>% dplyr::select(County,State,Year,Value)

usda_area$Value <- usda_area$Value * 0.4046856422 # Convert from acres to hectare * 4046.86 * 0.0001

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
  left_join(usda_area, by = c("County", "State")) %>%
  drop_na()

usda_spatial_area <-usda_spatial_full %>%
  dplyr::select(County,State,Year,Value,geometry)

usda_spatial_area_2010 <- usda_spatial_area[usda_spatial_area$Year == 2010,]
usda_spatial_area_2010["Value"] %>% plot()

extent_us = c(-180, 180, -90, 90)
resolution_us = 0.5
ncol_us = (extent_us[2]-extent_us[1])/resolution_us
nrows_us = (extent_us[4]-extent_us[3])/resolution_us


# Raster creation
baserast <- rast(nrows=nrows_us, ncol=ncol_us,
                 extent= extent_us, # CHANGE TO US
                 crs="+proj=longlat +datum=WGS84")

rasters <- rast(lapply(1980:2020,
                       function(x)
                         rasterize(vect(usda_spatial_area %>%
                                          filter(Year==x)),baserast,"Value")))
names(rasters) <- 1980:2020
varnames(rasters) <- paste0("soy_harvest_area_usda",1980:2020)

writeRaster(rasters,"soy_harvest_area_US_all_1980_2020_05x05_07.tiff",overwrite=TRUE)

###### Density for harvest area
r <- baserast
shp.soy.yld.us_subset <- subset(usda_spatial_area, Year < 2017 )
v <- vect(shp.soy.yld.us_subset)
ra <- cellSize(r, unit="ha")         
e <- expanse(v, unit="ha") 
v$density <- v$Value / e

years <- str_sort(unique(v$Year))
out <- list()
for (i in 1:length(years)) {
  vv <- v[v$Year == years[i], ]
  x <- rasterize(vv, r, "density")
  out[[i]] <- x * ra
}
out <- rast(out)
names(out) <- years

writeRaster(out,"soy_harvest_area_us_1980_2016_05x05_density_02.tif", overwrite=TRUE)


############### spam 2010

library(raster)
library(terra)
#LOAD
str_name_all<-'spam2010V2r0_global_H_SOYB_A.tif' 
str_name_rainfed<-'spam2010V2r0_global_H_SOYB_R.tif' 

r_rain=raster(str_name_rainfed)
r_all=raster(str_name_all)

#FIND MASK > 0.9
r_ratio_mask <- (r_rain / r_all) > 0.95
r_rain_masked <- r_rain
r_rain_masked[r_ratio_mask < 1] = NA
r_rain_masked[r_rain_masked < 1] = NA

#plot
my_window <- extent(-100,-40,-38,45)
plot(my_window, col=NA)
plot(r_rain_masked, add=T)


# Full resolution
extent_us = c(-180, 180, -90, 90)
resolution_us = res(r_rain)[1]
ncol_us = (extent_us[2]-extent_us[1])/resolution_us
nrows_us = (extent_us[4]-extent_us[3])/resolution_us

# Raster creation
r_base <- rast(nrows=nrows_us, ncol=ncol_us,
               extent= extent_us, # CHANGE TO US
               crs="+proj=longlat +datum=WGS84")

raster_rain_mask_grid <- terra::project(rast(r_rain_masked), r_base)
writeRaster(raster_rain_mask_grid,"soy_harvest_spam_native.tif", overwrite=TRUE)



# Aggregate level
imported_raster_aggregated <- aggregate(r_rain_masked, fun = sum, fact=(0.5/res(r_all)[1]))

extent_us = c(-180, 180, -90, 90)
resolution_us = 0.5
ncol_us = (extent_us[2]-extent_us[1])/resolution_us
nrows_us = (extent_us[4]-extent_us[3])/resolution_us

# Raster creation
r_base <- rast(nrows=nrows_us, ncol=ncol_us,
               extent= extent_us, # CHANGE TO US
               crs="+proj=longlat +datum=WGS84")

imported_raster_2_aggregated <- terra::project(rast(imported_raster_aggregated), r_base)
writeRaster(imported_raster_2_aggregated,"soy_harvest_spam_agg_resamp.tif", overwrite=TRUE)



## 2

extent_us = c(-180, 180, -90, 90)
resolution_us = res(r_rain)[1]
ncol_us = (extent_us[2]-extent_us[1])/resolution_us
nrows_us = (extent_us[4]-extent_us[3])/resolution_us

# Raster creation
r_base <- rast(nrows=nrows_us, ncol=ncol_us,
                 extent= extent_us, # CHANGE TO US
                 crs="+proj=longlat +datum=WGS84")

imported_raster_2_resamp <- terra::project(rast(r_rain_masked), r_base)

imported_raster_3_aggregated <- aggregate(imported_raster_2_resamp, fun = sum, fact= 6 )

writeRaster(imported_raster_3_aggregated,"soy_harvest_spam_resamp_agg.tif", overwrite=TRUE)

#plot
my_window <- extent(-100,-40,-38,45)
plot(my_window, col=NA)
plot(r_all, add=T)

my_window <- extent(-100,-40,-38,45)
plot(my_window, col=NA)
plot(r_rain, add=T)

my_window <- extent(-100,-40,-38,45)
plot(my_window, col=NA)
plot(r_rain_masked, add=T)

my_window <- extent(-100,-40,-38,45)
plot(my_window, col=NA)
plot(imported_raster_2_aggregated, add=T)

#plot
my_window <- extent(-100,-40,-38,45)
plot(my_window, col=NA)
plot(imported_raster_3_aggregated, add=T)

