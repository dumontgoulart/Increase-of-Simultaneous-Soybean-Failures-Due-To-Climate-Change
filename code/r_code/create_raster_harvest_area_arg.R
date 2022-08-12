library(terra)
library(tidyr)
library(dplyr)
library(sf)
library(data.table)
setwd("~/PhD/paper_hybrid_agri/data")

ARGmun <- st_read("GIS/arg_admbnda_adm2_unhcr2017.shp") %>% 
  mutate(ADM2_PCODE = substr(ADM2_PCODE,4,9)) %>%
  mutate(ADM2_PCODE = as.integer(ADM2_PCODE))

arg.soy.harvest <- read.csv('soja-serie-1969-2019.csv') %>% 
  # select(-c(cultivo_nombre)) %>%
  rename(ADM2_PCODE = `departamento_id`) %>%
  filter(anio > 1977) 

arg.soy.harvest <- arg.soy.harvest %>% dplyr::select(anio,provincia_nombre, ADM2_PCODE, superficie_cosechada_ha)

shp.soy.harvest.arg <- left_join(ARGmun, arg.soy.harvest) #%>% drop_na(anio )

shp.soy.harvest.arg <- subset(shp.soy.harvest.arg, anio > 1976 )
shp.soy.harvest.arg <- shp.soy.harvest.arg %>% dplyr::select(ADM2_REF, anio,superficie_cosechada_ha, geometry) 

shp.soy.harvest.arg.simp <- shp.soy.harvest.arg %>% dplyr::select(anio,superficie_cosechada_ha) #
st_geometry(shp.soy.harvest.arg.simp) <- NULL # remove geometry, coerce to data.frame

# plot(shp.soy.harvest.arg['superficie_cosechada_ha'])

shp.soy.harvest.arg.2009 <- shp.soy.harvest.arg %>% filter(anio== 2009) 
# plot(shp.soy.harvest.arg.2009['superficie_cosechada_ha'])

sum(shp.soy.harvest.arg$superficie_cosechada_ha[shp.soy.harvest.arg$anio == "1990"], na.rm = T) / 10**6
sum(shp.soy.harvest.arg$superficie_cosechada_ha[shp.soy.harvest.arg$anio == "2016"], na.rm = T) / 10**6

# Save as shapefile
st_write(shp.soy.harvest.arg, paste0(getwd(), "/", "soy_harvest_area_arg_1978_2019.gpkg"), append = TRUE)

# Convert raster
extent_arg = c(-180, 180, -90, 90)
resolution_arg = 0.5
ncol_arg = (extent_arg[2]-extent_arg[1])/resolution_arg
nrows_arg = (extent_arg[4]-extent_arg[3])/resolution_arg

baserast <- rast(nrows=nrows_arg, ncol=ncol_arg,
                 extent= extent_arg,
                 crs="+proj=longlat +datum=WGS84",
                 )

# if error shows up, use summary(shp.soy.harvest.arg$anio) to check for possible issues with the dates established.
rasters <- rast(lapply(1978:2019, 
                       function(x)
                         rasterize(vect(shp.soy.harvest.arg %>% 
                                          filter(anio==x)), baserast, "superficie_cosechada_ha", fun = 'sum')))

names(rasters) <-1978:2019
varnames(rasters) <- paste0("soy_harvest_area_", 1978:2019)

writeRaster(rasters,"soy_harvest_area_arg_1978_2019_05x05_03.tif", overwrite=TRUE)

###### Density for harvest area
r <- baserast
shp.soy.yld.arg_subset <- subset(shp.soy.harvest.arg, anio < 2017 )
v <- vect(shp.soy.yld.arg_subset)
ra <- cellSize(r, unit="ha")         
e <- expanse(v, unit="ha") 
v$density <- v$superficie_cosechada_ha / e

years <- str_sort(unique(v$anio))
out <- list()
for (i in 1:length(years)) {
  vv <- v[v$anio == years[i], ]
  x <- rasterize(vv, r, "density")
  out[[i]] <- x * ra
}
out <- rast(out)
names(out) <- years

writeRaster(out,"soy_harvest_area_arg_1978_2000_05x05_density_02.tif", overwrite=TRUE)


##############

s.sf <- st_transform(shp.soy.harvest.arg, "+proj=tmerc +lat_0=-90 +lon_0=-72 +k=1 +x_0=1500000 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs ") 
s.sf$total_area <- st_area(s.sf)

shp.soy.harvest.arg$total_area <- s.sf$total_area #(shp.soy.harvest.arg)
shp.soy.harvest.arg$total_area <- units::set_units(x = shp.soy.harvest.arg$total_area, value = ha)
shp.soy.harvest.arg$total_area <- units::drop_units(shp.soy.harvest.arg$total_area)


