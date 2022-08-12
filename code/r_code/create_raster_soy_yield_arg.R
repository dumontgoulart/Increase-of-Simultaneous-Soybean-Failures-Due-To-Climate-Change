library(terra)
library(tidyr)
library(dplyr)
library(sf)
library(data.table)
setwd("~/PhD/paper_hybrid_agri/data")

ARGmun <- st_read("GIS/arg_admbnda_adm2_unhcr2017.shp") %>% 
  mutate(ADM2_PCODE = substr(ADM2_PCODE,4,9)) %>%
  mutate(ADM2_PCODE = as.integer(ADM2_PCODE))
  
arg.soy.yield <- read.csv('soja-serie-1969-2019.csv') %>% 
  select(-c(cultivo_nombre)) %>%
  rename(ADM2_PCODE = `departamento_id`) %>%
  filter(anio > 1973) 

arg.soy.yield <- arg.soy.yield %>% dplyr::select(anio,provincia_nombre, ADM2_PCODE,rendimiento_kgxha)

arg.soy.yield$rendimiento_kgxha[arg.soy.yield$rendimiento_kgxha > 5000] <- NA

arg.soy.yld.shp <- left_join(ARGmun, arg.soy.yield) #%>% drop_na(anio )

arg.soy.yld.shp <- subset(arg.soy.yld.shp, anio > 1973 )
arg.soy.yld.shp <- subset(arg.soy.yld.shp, anio < 2017 )
arg.soy.yld.shp <- arg.soy.yld.shp %>% dplyr::select(ADM2_REF, anio,rendimiento_kgxha, geometry) 

arg.soy.yld.shp.2009 <- arg.soy.yld.shp %>% filter(anio== 1974) 
plot(arg.soy.yld.shp.2009['rendimiento_kgxha'])

  
# Convert raster
init_year = 1974
end_year = 2016
extent_arg = c(-180, 180, -90, 90)
resolution_arg = 0.05
ncol_arg = (extent_arg[2]-extent_arg[1])/resolution_arg
nrows_arg = (extent_arg[4]-extent_arg[3])/resolution_arg

baserast <- rast(nrows=nrows_arg, ncol=ncol_arg,
                 extent= extent_arg,
                 crs="+proj=longlat +datum=WGS84",
                 vals=NA)

rasters <- rast(lapply(init_year:end_year, 
                       function(x)
                         rasterize(vect(arg.soy.yld.shp %>% 
                                          filter(anio==x)),baserast,"rendimiento_kgxha")))

names(rasters) <-init_year:end_year
varnames(rasters) <- paste0("soy_yield_", init_year:end_year)

writeRaster(rasters,"soy_yield_arg_1978_2019_005.tif", overwrite=TRUE)


#### ADD filter for 1% of removal - needs file create_raster_harvest_area_us.R loaded

# Calculate harvest area %
arg.soy.yld.shp$ratio_area <- shp.soy.harvest.arg$superficie_cosechada_ha / shp.soy.harvest.arg$total_area

sum(arg.soy.yld.shp$ratio_area < 0.01, na.rm=T) 
sum(arg.soy.yld.shp$rendimiento_kgxha[arg.soy.yld.shp$ratio_area < 0.01], na.rm=T)

arg.soy.yld.shp$rendimiento_kgxha <- with(arg.soy.yld.shp,replace(rendimiento_kgxha, ratio_area < 0.01, NA))
sum(arg.soy.yld.shp$rendimiento_kgxha[arg.soy.yld.shp$ratio_area < 0.01], na.rm=T)


rasters <- rast(lapply(1978:2019, 
                       function(x)
                         rasterize(vect(arg.soy.yld.shp %>% 
                                          filter(anio==x)),baserast,"rendimiento_kgxha")))

names(rasters) <-1978:2019
varnames(rasters) <- paste0("soy_yield_", 1978:2019)

writeRaster(rasters,"soy_yield_arg_1978_2019_1prc_001.tif", overwrite=TRUE)


################################
# TESTS
################################

###### Filter for at least 5 years of data in each county 
df_duration <- arg.soy.yield %>%  
  group_by(ADM2_PCODE) %>% 
  count(duration_year = max(anio)-min(anio) & !is.na(rendimiento_kgxha) & rendimiento_kgxha > 0 )  %>%
  filter(duration_year == TRUE)  %>%
  mutate(time_mask = ifelse(n >= 5, TRUE, FALSE))


