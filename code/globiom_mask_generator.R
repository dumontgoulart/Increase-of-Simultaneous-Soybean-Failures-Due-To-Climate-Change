library(ncdf4) # package for netcdf manipulation
library(raster) # package for raster manipulation
library(rgdal) # package for geospatial analysis
library(ggplot2) # package for plotting
library(terra)
library(tidyr)
library(dplyr)
library(sf)
library(sp)

rdir <- ("C:/users/morenodu/OneDrive - Stichting Deltares/Documents/PhD/paper_hybrid_agri/data")
setwd(rdir)

globiom_grid_am <- vect("./soy_americas_harvest_area.shp")

# Short code to generate masks for globiom


### MAIN CODE
create_shifters_globiom_2012 <- function(path) {
  
  # Load .nc file
  nc_file <- raster(paste(path,'.nc', sep = ""))
  
  writeRaster(x = nc_file, filename = paste(path,'.tif', sep = ""), format = 'GTiff', overwrite = TRUE)
  
  raster_2012_hybrid <- rast(paste(path,'.tif', sep = ""))
  names(raster_2012_hybrid) <- 'hybrid_shift'
  
  v_all <- terra::extract(raster_2012_hybrid, globiom_grid_am, fun = mean)
  
  output_hybrid = unlist(lapply(v_all$hybrid_shift, function(x) if (!is.null(x)) mean(x, na.rm=TRUE) else NA ))
  
  globiom_grid_am$shift_hybrid <- output_hybrid
  
  shift_2012_soybean_am <- data.frame(globiom_grid_am$CR_ID, globiom_grid_am$shift_hybrid)
  names(shift_2012_soybean_am) <- c('ALLCOLROW','hybrid')
  shift_2012_soybean_am_sub <- na.omit(shift_2012_soybean_am, cols = c("hybrid"))

  
  # Save shifters to csv
  write.csv(shift_2012_soybean_am_sub, paste(path,'.csv', sep = ""), row.names = FALSE)
  
}

# main code, choose file desired
path_1 = 'soybean_yields_america_detrended_2012'
path_2 = 'soybean_harvested_area_america_2012'

create_shifters_globiom_2012(path_2)
