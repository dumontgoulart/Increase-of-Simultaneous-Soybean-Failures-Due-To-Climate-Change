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

globiom_grid_am <- vect("../../soy_americas_harvest_area.shp")


### MAIN CODE
create_shifters_globiom <- function(model, co2_scenario, ssp, years, three_models = FALSE) {
  
  # Load .nc file
  nc_file <- raster(paste('hybrid_', model, '_',ssp,'_',co2_scenario,'_yield_soybean_shift_',years,'.nc', sep = ""))
  
  writeRaster(x = nc_file, filename = paste('hybrid_', model, '_',ssp,'_',co2_scenario,'_yield_soybean_shift_',years,'.tif', sep = ""), format = 'GTiff', overwrite = TRUE)
  
  raster_2012_hybrid <- rast(paste('hybrid_', model, '_',ssp,'_',co2_scenario,'_yield_soybean_shift_',years,'.tif', sep = ""))
  names(raster_2012_hybrid) <- 'hybrid_shift'
  
  # If only using Hybrid
  if (three_models == FALSE) {
    v_all <- terra::extract(raster_2012_hybrid, globiom_grid_am, fun = mean)
    
    output_hybrid = unlist(lapply(v_all$hybrid_shift, function(x) if (!is.null(x)) mean(x, na.rm=TRUE) else NA ))
    
    globiom_grid_am$shift_hybrid <- output_hybrid
    
    shift_2012_soybean_am <- data.frame(globiom_grid_am$CR_ID, globiom_grid_am$shift_hybrid)
    names(shift_2012_soybean_am) <- c('ALLCOLROW','hybrid')
    shift_2012_soybean_am_sub <- na.omit(shift_2012_soybean_am, cols = c("hybrid"))
  }
  
  
  if (three_models == TRUE) {
    writeRaster(x =raster(paste('epic_', model, '_',ssp,'_',co2_scenario,'_yield_soybean_shift_',years,'.nc', sep = "")), filename = paste('epic_', model, '_',ssp,'_',co2_scenario,'_yield_soybean_shift_',years,'.tif', sep = ""), format = 'GTiff', overwrite = TRUE)
    
    writeRaster(x =raster(paste('clim_', model, '_',ssp,'_',co2_scenario,'_yield_soybean_shift_',years,'.nc', sep = "")), filename = paste('clim_', model, '_',ssp,'_',co2_scenario,'_yield_soybean_shift_',years,'.tif', sep = ""), format = 'GTiff', overwrite = TRUE)
    
    raster_2012_epic <- rast(paste('epic_', model, '_',ssp,'_',co2_scenario,'_yield_soybean_shift_',years,'.tif', sep = ""))
    names(raster_2012_epic) <- 'epic_shift'
    
    raster_2012_clim <- rast(paste('clim_', model, '_',ssp,'_',co2_scenario,'_yield_soybean_shift_',years,'.tif', sep = ""))
    names(raster_2012_clim) <- 'clim_shift'
    
    all_rasters <- c(raster_2012_hybrid,raster_2012_epic,raster_2012_clim )
    
    v_all <- terra::extract(all_rasters, globiom_grid_am, fun = mean)
    
    output_hybrid = unlist(lapply(v_all$hybrid_shift, function(x) if (!is.null(x)) mean(x, na.rm=TRUE) else NA ))
    output_epic = unlist(lapply(v_all$epic_shift, function(x) if (!is.null(x)) mean(x, na.rm=TRUE) else NA ))
    output_clim = unlist(lapply(v_all$clim_shift, function(x) if (!is.null(x)) mean(x, na.rm=TRUE) else NA ))
    
    globiom_grid_am$shift_hybrid <- output_hybrid
    globiom_grid_am$shift_epic <- output_epic
    globiom_grid_am$shift_clim <- output_clim
    
    shift_2012_soybean_am <- data.frame(globiom_grid_am$CR_ID, globiom_grid_am$shift_hybrid, globiom_grid_am$shift_epic, globiom_grid_am$shift_clim)
    names(shift_2012_soybean_am) <- c('ALLCOLROW','hybrid', 'epic', 'clim')
    shift_2012_soybean_am_sub <- na.omit(shift_2012_soybean_am, cols = c("hybrid"))
  }
  
  # Save shifters to csv
  write.csv(shift_2012_soybean_am_sub, paste('output_csv/yield_', model, '_',ssp,'_',co2_scenario,'_yield_soybean_shift_',years,'.csv', sep = ""), row.names = FALSE)
  
}

#####  MERGE CSV FILES
merge_shifters <- function(model,co2_scenario) {
  
  group_per_time <- function(model, co2_scenario, ssp) {
    
    csv_shift_2017 <- read.csv(paste('output_csv/yield_', model, '_',ssp,'_',co2_scenario,'_yield_soybean_shift_2017-2044.csv', sep = ""))
    csv_shift_2044 <- read.csv(paste('output_csv/yield_', model, '_',ssp,'_',co2_scenario,'_yield_soybean_shift_2044-2071.csv', sep = ""))
    csv_shift_2071 <- read.csv(paste('output_csv/yield_', model, '_',ssp,'_',co2_scenario,'_yield_soybean_shift_2071-2098.csv', sep = ""))
    
    csv_shift_2017$ALLYEAR = 2030
    csv_shift_2044$ALLYEAR = 2050
    csv_shift_2071$ALLYEAR = 2070
    
    csv_combined <- bind_rows(csv_shift_2017,csv_shift_2044,csv_shift_2071 )
    
    csv_combined <- csv_combined %>% select(ALLYEAR, everything())
    
    if(co2_scenario == '2015co2' && ssp == 'ssp585' ){
      csv_combined$rcp = 'noC8p5'
    } else if (co2_scenario == 'default' && ssp == 'ssp585') {
      csv_combined$rcp = 'rcp8p5'
    } else if (co2_scenario == '2015co2' && ssp == 'ssp126') {
      csv_combined$rcp = 'noC2p6'
    } else if (co2_scenario == 'default' && ssp == 'ssp126') {
      csv_combined$rcp = 'rcp2p6'
    }
    
    write.csv(csv_combined, paste('output_csv/yield_', model, '_',ssp,'_',co2_scenario,'_soybean_shift_2015-2100.csv', sep = ""), row.names = FALSE)
    
  }
  
  group_per_time(model, co2_scenario, 'ssp585')
  group_per_time(model, co2_scenario, 'ssp126')
  
  ###### COMBINE TWO RCPs
  csv_shift_85 <- read.csv(paste('output_csv/yield_', model, '_ssp585_',co2_scenario,'_soybean_shift_2015-2100.csv', sep = ""))
  csv_shift_26 <- read.csv(paste('output_csv/yield_', model, '_ssp126_',co2_scenario,'_soybean_shift_2015-2100.csv', sep = ""))
  
  csv_comb_rcps <- bind_rows(csv_shift_26,csv_shift_85 )
  csv_comb_rcps <- csv_comb_rcps %>% select(rcp, everything())
  
  indir  <- paste(rdir,"/results_shifters_am_",model,'_ssp126_585',"/",sep="")
  dir.create(indir)
  
  write.csv(csv_comb_rcps, paste(indir, 'yield_', model, '_ssp126_585_',co2_scenario,'_soybean_shift_2015-2100.csv', sep = ""), row.names = FALSE)
  
}

merge_models <- function(model1,model2,model3, co2_scenario) {
  csv_shift_gfdl <- read.csv(paste(rdir,"/results_shifters_am_",model1,'_ssp126_585',"/",'yield_', model1, '_ssp126_585_',co2_scenario,'_soybean_shift_2015-2100.csv',sep=""))
  csv_shift_ukesm <- read.csv(paste(rdir,"/results_shifters_am_",model2,'_ssp126_585',"/",'yield_', model2, '_ssp126_585_',co2_scenario,'_soybean_shift_2015-2100.csv',sep=""))
  csv_shift_ipsl <- read.csv(paste(rdir,"/results_shifters_am_",model3,'_ssp126_585',"/",'yield_', model3, '_ssp126_585_',co2_scenario,'_soybean_shift_2015-2100.csv',sep=""))
  
  csv_shift_gfdl$GCM = 'gfdl-esm4'
  csv_shift_ukesm$GCM = 'ukesm1-0-ll'
  csv_shift_ipsl$GCM = 'ipsl-cm6a-lr'
  
  csv_combined <- bind_rows(csv_shift_gfdl,csv_shift_ukesm,csv_shift_ipsl )
  csv_combined <- csv_combined %>% select(GCM, everything())
  
  indir  <- paste(rdir,"/results_shifters_am_all_gcms_rcps","/",sep="")
  dir.create(indir)
  
  write.csv(csv_combined, paste(indir, 'yield_all_models_rcps_soybean_shift_2015-2100.csv', sep = ""), row.names = FALSE)
}

# 
# # 2015co2
# create_shifters_globiom('gfdl-esm4', '2015co2', 'ssp126' ,'2017-2044')
# create_shifters_globiom('gfdl-esm4', '2015co2','ssp126' ,'2044-2071')
# create_shifters_globiom('gfdl-esm4', '2015co2','ssp126' ,'2071-2098')
# 
# create_shifters_globiom('gfdl-esm4', '2015co2','ssp585' ,'2017-2044')
# create_shifters_globiom('gfdl-esm4', '2015co2','ssp585' ,'2044-2071')
# create_shifters_globiom('gfdl-esm4', '2015co2','ssp585' ,'2071-2098')
# 
# 
# create_shifters_globiom('ukesm1-0-ll', '2015co2', 'ssp126' ,'2017-2044')
# create_shifters_globiom('ukesm1-0-ll', '2015co2','ssp126' ,'2044-2071')
# create_shifters_globiom('ukesm1-0-ll', '2015co2','ssp126' ,'2071-2098')
# 
# create_shifters_globiom('ukesm1-0-ll', '2015co2','ssp585' ,'2017-2044')
# create_shifters_globiom('ukesm1-0-ll', '2015co2','ssp585' ,'2044-2071')
# create_shifters_globiom('ukesm1-0-ll', '2015co2','ssp585' ,'2071-2098')
# 
# 
# create_shifters_globiom('ipsl-cm6a-lr', '2015co2','ssp126' ,'2017-2044')
# create_shifters_globiom('ipsl-cm6a-lr', '2015co2','ssp126' ,'2044-2071')
# create_shifters_globiom('ipsl-cm6a-lr', '2015co2','ssp126' ,'2071-2098')
# 
# create_shifters_globiom('ipsl-cm6a-lr', '2015co2','ssp585' ,'2017-2044')
# create_shifters_globiom('ipsl-cm6a-lr', '2015co2','ssp585' ,'2044-2071')
# create_shifters_globiom('ipsl-cm6a-lr', '2015co2','ssp585' ,'2071-2098')


# DEFAULT co2
create_shifters_globiom('gfdl-esm4', 'default','ssp126' ,'2017-2044')
create_shifters_globiom('gfdl-esm4', 'default','ssp126' ,'2044-2071')
create_shifters_globiom('gfdl-esm4', 'default','ssp126' ,'2071-2098')

create_shifters_globiom('gfdl-esm4', 'default','ssp585' ,'2017-2044')
create_shifters_globiom('gfdl-esm4', 'default','ssp585' ,'2044-2071')
create_shifters_globiom('gfdl-esm4', 'default','ssp585' ,'2071-2098')

create_shifters_globiom('ukesm1-0-ll', 'default','ssp126' ,'2017-2044')
create_shifters_globiom('ukesm1-0-ll', 'default','ssp126' ,'2044-2071')
create_shifters_globiom('ukesm1-0-ll', 'default','ssp126' ,'2071-2098')

create_shifters_globiom('ukesm1-0-ll', 'default','ssp585' ,'2017-2044')
create_shifters_globiom('ukesm1-0-ll', 'default','ssp585' ,'2044-2071')
create_shifters_globiom('ukesm1-0-ll', 'default','ssp585' ,'2071-2098')

create_shifters_globiom('ipsl-cm6a-lr', 'default','ssp126' ,'2017-2044')
create_shifters_globiom('ipsl-cm6a-lr', 'default','ssp126' ,'2044-2071')
create_shifters_globiom('ipsl-cm6a-lr', 'default','ssp126' ,'2071-2098')

create_shifters_globiom('ipsl-cm6a-lr', 'default','ssp585' ,'2017-2044')
create_shifters_globiom('ipsl-cm6a-lr', 'default','ssp585' ,'2044-2071')
create_shifters_globiom('ipsl-cm6a-lr', 'default','ssp585' ,'2071-2098')

# 
# ### FINAL SHIFTERS FOR MODEL AND CO2 Scenarios
# merge_shifters('gfdl-esm4', '2015co2')
# merge_shifters('ukesm1-0-ll', '2015co2')
# merge_shifters('ipsl-cm6a-lr', '2015co2')

merge_shifters('gfdl-esm4', 'default')
merge_shifters('ukesm1-0-ll', 'default')
merge_shifters('ipsl-cm6a-lr', 'default')


# MERGE MODEL
merge_models('gfdl-esm4','ukesm1-0-ll','ipsl-cm6a-lr', 'default')
  

############### Historical  run - 2012
# load file and concvert to tiff.
nc_file_hist <- raster(paste('../../output_models_am/shifters_2012_hybrid_climatology_syn_2050area.nc', sep = ""))
writeRaster(x = nc_file_hist, filename = paste('../../output_models_am/shifters_2012_hybrid_climatology_syn_2050area','.tif', sep = ""), format = 'GTiff', overwrite = TRUE)
raster_2012_hybrid <- rast(paste('../../output_models_am/shifters_2012_hybrid_climatology_syn_2050area','.tif', sep = ""))
names(raster_2012_hybrid) <- 'hybrid_shift'
# Process the data and convert to csv according to globiom grid
v_all <- terra::extract(raster_2012_hybrid, globiom_grid_am, fun = mean)
output_hybrid = unlist(lapply(v_all$hybrid_shift, function(x) if (!is.null(x)) mean(x, na.rm=TRUE) else NA ))
globiom_grid_am$shift_hybrid <- output_hybrid
shift_2012_soybean_am <- data.frame(globiom_grid_am$CR_ID, globiom_grid_am$shift_hybrid)
names(shift_2012_soybean_am) <- c('ALLCOLROW','hybrid')
shift_2012_soybean_am_sub <- na.omit(shift_2012_soybean_am, cols = c("hybrid"))

# Save shifters to csv
write.csv(shift_2012_soybean_am_sub, paste('output_csv/shifters_2012_hybrid_climatology_syn_2050area','.csv', sep = ""), row.names = FALSE)


## ESTHER - future shifters
nc_file_hist <- raster(paste('../../output_shocks_am/esther_paper/hybrid_ukesm1-0-ll_ssp126_default_yield_soybean_shift_2044-2071.nc', sep = ""))
writeRaster(x = nc_file_hist, filename = paste('../../output_shocks_am/esther_paper/hybrid_ukesm1-0-ll_ssp126_default_yield_soybean_shift_2044-2071','.tif', sep = ""), format = 'GTiff', overwrite = TRUE)
raster_2012_hybrid <- rast(paste('../../output_shocks_am/esther_paper/hybrid_ukesm1-0-ll_ssp126_default_yield_soybean_shift_2044-2071','.tif', sep = ""))
names(raster_2012_hybrid) <- 'hybrid_shift'
# Process the data and convert to csv according to globiom grid
v_all <- terra::extract(raster_2012_hybrid, globiom_grid_am, fun = mean)
output_hybrid = unlist(lapply(v_all$hybrid_shift, function(x) if (!is.null(x)) mean(x, na.rm=TRUE) else NA ))
globiom_grid_am$shift_hybrid <- output_hybrid
shift_2012_soybean_am <- data.frame(globiom_grid_am$CR_ID, globiom_grid_am$shift_hybrid)
names(shift_2012_soybean_am) <- c('ALLCOLROW','hybrid')
shift_2012_soybean_am_sub <- na.omit(shift_2012_soybean_am, cols = c("hybrid"))
# Save shifters to csv
write.csv(shift_2012_soybean_am_sub, paste('../../output_shocks_am/esther_paper/hybrid_ukesm1-0-ll_ssp126_default_yield_soybean_shift_2044-2071','.csv', sep = ""), row.names = FALSE)

nc_file_hist <- raster(paste('../../output_shocks_am/esther_paper/hybrid_ukesm1-0-ll_ssp585_default_yield_soybean_shift_2044-2071.nc', sep = ""))
writeRaster(x = nc_file_hist, filename = paste('../../output_shocks_am/esther_paper/hybrid_ukesm1-0-ll_ssp585_default_yield_soybean_shift_2044-2071','.tif', sep = ""), format = 'GTiff', overwrite = TRUE)
raster_2012_hybrid <- rast(paste('../../output_shocks_am/esther_paper/hybrid_ukesm1-0-ll_ssp585_default_yield_soybean_shift_2044-2071','.tif', sep = ""))
names(raster_2012_hybrid) <- 'hybrid_shift'
# Process the data and convert to csv according to globiom grid
v_all <- terra::extract(raster_2012_hybrid, globiom_grid_am, fun = mean)
output_hybrid = unlist(lapply(v_all$hybrid_shift, function(x) if (!is.null(x)) mean(x, na.rm=TRUE) else NA ))
globiom_grid_am$shift_hybrid <- output_hybrid
shift_2012_soybean_am <- data.frame(globiom_grid_am$CR_ID, globiom_grid_am$shift_hybrid)
names(shift_2012_soybean_am) <- c('ALLCOLROW','hybrid')
shift_2012_soybean_am_sub <- na.omit(shift_2012_soybean_am, cols = c("hybrid"))
# Save shifters to csv
write.csv(shift_2012_soybean_am_sub, paste('../../output_shocks_am/esther_paper/hybrid_ukesm1-0-ll_ssp585_default_yield_soybean_shift_2044-2071','.csv', sep = ""), row.names = FALSE)





############### SAVE GLOBIOM GRID
globiom_grid_am <- vect("../../soy_americas_harvest_area.shp")
# load file and concvert to tiff.
nc_file_hist <- raster(paste('../../soy_harvest_spam_native_05x05.nc', sep = ""))
writeRaster(x = nc_file_hist, filename = paste('../../soy_harvest_spam_native_05x05','.tif', sep = ""), format = 'GTiff', overwrite = TRUE)

raster_2012_hybrid <- rast(paste('../../soy_harvest_spam_native','.tif', sep = ""))
names(raster_2012_hybrid) <- 'hybrid_shift'

# Process the data and convert to csv according to globiom grid
v_all <- terra::extract(raster_2012_hybrid, globiom_grid_am, fun = sum)
output_hybrid = unlist(lapply(v_all$hybrid_shift, function(x) if (!is.null(x)) mean(x, na.rm=TRUE) else NA ))
globiom_grid_am$shift_hybrid <- output_hybrid
shift_2012_soybean_am <- data.frame(globiom_grid_am$CR_ID, globiom_grid_am$shift_hybrid)
names(shift_2012_soybean_am) <- c('ALLCOLROW','harvest_area')
shift_2012_soybean_am_sub3 <- na.omit(shift_2012_soybean_am, cols = c("hybrid"))

# Save shifters to csv
write.csv(shift_2012_soybean_am_sub, paste('../../soy_harvest_spam_globiom','.csv', sep = ""), row.names = FALSE)



# Sendin the minimum shocks in the entire series to Esther
create_shifters_globiom_min <- function(input) {
  nc_file_hist <- raster(paste(input,'.nc', sep = ""))
  writeRaster(x = nc_file_hist, filename = paste(input,'.tif', sep = ""), format = 'GTiff', overwrite = TRUE)
  raster_2012_hybrid <- rast(paste(input,'.tif', sep = ""))
  names(raster_2012_hybrid) <- 'hybrid_shift'
  # Process the data and convert to csv according to globiom grid
  v_all <- terra::extract(raster_2012_hybrid, globiom_grid_am, fun = mean)
  output_hybrid = unlist(lapply(v_all$hybrid_shift, function(x) if (!is.null(x)) mean(x, na.rm=TRUE) else NA ))
  globiom_grid_am$shift_hybrid <- output_hybrid
  shift_2012_soybean_am <- data.frame(globiom_grid_am$CR_ID, globiom_grid_am$shift_hybrid)
  names(shift_2012_soybean_am) <- c('ALLCOLROW','hybrid')
  shift_2012_soybean_am_sub <- na.omit(shift_2012_soybean_am, cols = c("hybrid"))
  
  str_parts = strsplit(input, "/")
  # Save shifters to csv
  write.csv(shift_2012_soybean_am_sub, paste(str_parts[[1]][1], '/', str_parts[[1]][2], '/output_csv/', str_parts[[1]][3],'.csv', sep = ""), row.names = FALSE)}
  

create_shifters_globiom_min('../esther_paper/hybrid_gfdl-esm4_ssp126_default_yield_soybean_min_shift_2015_2098')
create_shifters_globiom_min('../esther_paper/hybrid_gfdl-esm4_ssp585_default_yield_soybean_min_shift_2015_2098')
create_shifters_globiom_min('../esther_paper/hybrid_ipsl-cm6a-lr_ssp126_default_yield_soybean_min_shift_2015_2098')
create_shifters_globiom_min('../esther_paper/hybrid_ipsl-cm6a-lr_ssp585_default_yield_soybean_min_shift_2015_2098')
create_shifters_globiom_min('../esther_paper/hybrid_ukesm1-0-ll_ssp126_default_yield_soybean_min_shift_2015_2098')
create_shifters_globiom_min('../esther_paper/hybrid_ukesm1-0-ll_ssp585_default_yield_soybean_min_shift_2015_2098')


nc_file_hist <- raster(paste('../esther_paper/hybrid_gfdl-esm4_ssp126_default_yield_soybean_min_shift_2015_2098', '.nc', sep = ""))
writeRaster(x = nc_file_hist, filename = paste('../esther_paper/hybrid_gfdl-esm4_ssp126_default_yield_soybean_min_shift_2015_2098','.tif', sep = ""), format = 'GTiff', overwrite = TRUE)
raster_2012_hybrid <- rast(paste('../esther_paper/hybrid_gfdl-esm4_ssp126_default_yield_soybean_min_shift_2015_2098','.tif', sep = ""))
names(raster_2012_hybrid) <- 'hybrid_shift'
# Process the data and convert to csv according to globiom grid
v_all <- terra::extract(raster_2012_hybrid, globiom_grid_am, fun = mean)
output_hybrid = unlist(lapply(v_all$hybrid_shift, function(x) if (!is.null(x)) mean(x, na.rm=TRUE) else NA ))
globiom_grid_am$shift_hybrid <- output_hybrid
shift_2012_soybean_am <- data.frame(globiom_grid_am$CR_ID, globiom_grid_am$shift_hybrid)
names(shift_2012_soybean_am) <- c('ALLCOLROW','hybrid')
shift_2012_soybean_am_sub <- na.omit(shift_2012_soybean_am, cols = c("hybrid"))

my_parts <- strsplit(test, "/")
paste(my_parts[[1]][1], '/', my_parts[[1]][2], '/output_csv/', my_parts[[1]][3],'.csv', sep = "")
