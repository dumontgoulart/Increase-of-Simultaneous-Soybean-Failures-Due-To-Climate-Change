library(ncdf4) # package for netcdf manipulation
library(raster) # package for raster manipulation
library(rgdal) # package for geospatial analysis
library(ggplot2) # package for plotting
library(terra)
library(tidyr)
library(dplyr)
library(sf)
library(sp)

rdir <- ("C:/users/morenodu/OneDrive - Stichting Deltares/Documents/PhD/paper_hybrid_agri/data/output_shocks_am/scenarios_forum")
setwd(rdir)

# Load global grid matrix GLOBIOM is based on:
globiom_grid <- st_read("../../COLROW/COLROW_GLOBIOMcode.shp") 
# plot(globiom_grid)

# Load a given globiom csv data (harvest area 2050)
globiom_grid_2050 <- read.csv("c:/users/morenodu/OneDrive - Stichting Deltares/Documents/PhD/other_papers/Soy_Area_2050.csv") %>% dplyr::select(CR_ID,X2050) %>% mutate(X2050 = X2050 * 1000) %>% rename(area = X2050)

# merge the global grid with the data above on a common variable, here called 'CR_ID'
globiom_grid_new_area <- merge(globiom_grid, globiom_grid_2050, by='CR_ID')
st_write(globiom_grid_new_area, "soy_harvest_area_globiom_2050.shp")
#plot(globiom_grid_new_area)

summary(globiom_grid_new_area$area )

# Load reference:
refe_grid <- raster('hybrid_ukesm1-0-ll_ssp585_default_yield_soybean_shift_2017-2044.tif')

ext <- extent(globiom_grid_new_area)
rr <- raster(ext, res=0.5)
crs(rr) <- "+proj=longlat"
rrr <- rasterize(globiom_grid_new_area, rr,"area")

rrr_resamp <- resample(rrr, refe_grid)

writeRaster(rrr_resamp,"soy_harvest_area_globiom_2050.tiff",overwrite=TRUE)

writeRaster(rrr_resamp,"soy_harvest_area_globiom_2050.nc",overwrite=TRUE, format="CDF", varname="area", varunit="ha", 
            longname="harvest area", xname="lon",   yname="lat")

