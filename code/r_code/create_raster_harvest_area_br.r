library(sidrar)
library(terra)
library(tidyr)
library(dplyr)
library(sf)
library(ncdf4)
library(raster)
library(rgdal)
library(stringr)
setwd("~/PhD/paper_hybrid_agri/data")

# period.list <- list(1975:1978)
# 
# # 112 for yield, 109 for planting area, 216 for harvesting area
# mun.soy.has <- lapply(period.list, function(x)
#   get_sidra(1612, variable = 216,
#             period = as.character(x),
#             geo = "City",
#             classific = c("c81"),
#             category = list(2713)))
# 
# test_2 <- do.call(rbind,mun.soy.has)
# write.csv(test_2,"soy_harv_area_munic_br_1975_1978.csv")

# mun.soy.has_2 <- do.call(rbind,mun.soy.has) %>%
#   rename(ADM2_PCODE = `Município (Código)`) %>%
#   dplyr::select(Ano,ADM2_PCODE,Valor)

mun.soy.has_2 <- read.csv("soy_harv_area_munic_br_1975:1978.csv")
mun.soy.has_2 <- mun.soy.has_2 %>% rename(ADM2_PCODE = 'Município..Código.') %>%
  dplyr::select(Ano,ADM2_PCODE,Valor) 

mun.soy.has_2 <- read.csv("soy_harv_area_munic_br_1975_1978.csv")

mun.soy.has_3 <- read.csv("soy_harv_area_munic_br.csv")
mun.soy.has <- rbind(mun.soy.has_2, mun.soy.has_3)

mun.soy.has <- mun.soy.has %>% rename(ADM2_PCODE = 'Município..Código.') %>%
  dplyr::select(Ano,ADM2_PCODE,Valor) 
mun.soy.has <- dplyr::arrange(mun.soy.has , ADM2_PCODE, Ano)

BRmun <- st_read("GIS/bra_admbnda_adm2_ibge_2020.shp") %>% 
  mutate(ADM2_PCODE = substr(ADM2_PCODE,3,9)) %>%
  mutate_at(c('ADM2_PCODE'), as.numeric)
st_crs(BRmun) <- 4326
BRmun <- st_transform(BRmun, crs = 4326)

shp.soy.has <- left_join(BRmun, mun.soy.has) %>% drop_na()


# ---------------------------------------------------
# Save as shapefile
# ---------------------------------------------------
# st_write(shp.soy.has, paste0(getwd(), "/", "soy_ibge_br_harvest_area.shp"))

# Calculate harvest area %
areas <- read.csv("GIS/areas.csv",sep = ";") %>%
  rename(ADM2_PCODE = CD_GCMUN) %>% rename(area = AR_MUN_2019) %>%
  mutate(area = 100*as.integer(str_replace(area,",",".")))

area_municipality <- inner_join(areas,shp.soy.has)
area_municipality$harv_area_frac <- area_municipality$Valor / area_municipality$area
area_municipality$Valor[area_municipality$Valor==0] <- NA
area_municipality = area_municipality %>% drop_na()

# Check for values above 1 - error
#sum(area_municipality$harv_area_frac > 1, na.rm=T) # SHOULD BE ZERO
area_municipality %>% filter(harv_area_frac < 0.01)
sum(area_municipality$harv_area_frac < 0.01, na.rm=T) 
#unique(area_municipality$ADM2_PT[which(area_municipality$harv_area_frac > 1, arr.ind=TRUE)])
#summary(area_municipality$harv_area_frac, na.rm = TRUE )
#hist(area_municipality$harv_area_frac)

# Check summary data for harvest area
# summary(shp.soy.has$Valor )
# hist(area_municipality$Valor)
# test_2 <-area_municipality$Valor
# test_3 <- with(area_municipality,replace(Valor, harv_area_frac < 0.01, NA))
# summary(test_2 )
# summary(test_3 )
# hist(test_2)
# hist(test_3)
# all.equal(test_2,test_3)
# sum(test_2 > 0, na.rm=T) - sum(test_3 > 0, na.rm=T) # number of cases that disappeared

shp.soy.has$harv_area_frac <- area_municipality$harv_area_frac
shp.soy.has$Valor <- with(area_municipality,replace(Valor, harv_area_frac < 0.01, NA))
sum(shp.soy.has$Valor[shp.soy.has$harv_area_frac < 0.01], na.rm=T)  # IT should be zero to show there is no value below 1%

summary(shp.soy.has$Valor )
summary(shp.soy.has$harv_area_frac )


shp.soy.has.2009 <- shp.soy.has %>% filter(Ano == 2016) 
# plot(shp.soy.has.2009['Valor'])

# Save shapefile with the cells only with > 1%

####
extent_br = c(-180, 180, -90, 90)
resolution_br = 0.5
ncol_br = (extent_br[2]-extent_br[1])/resolution_br
nrows_br = (extent_br[4]-extent_br[3])/resolution_br

# Raster creation
baserast <- rast(nrows=nrows_br, ncol=ncol_br,
                 extent= extent_br,
                 crs="+proj=longlat +datum=WGS84")

# rasters <- rast(lapply(1980:2016, 
#                        function(x)
#                          rasterize(vect(shp.soy.has %>% 
#                                           filter(Ano==x)),baserast,"Valor", fun = 'mean')))
# names(rasters) <- 1980:2016
# varnames(rasters) <- paste0("soy_yield_",1980:2016)
# 
# writeRaster(rasters,"soy_harvest_area_city_1980_2016_1prc_05x05.tiff",overwrite=TRUE)
# 
# plot(rasters)



###### Density for harvest area
r <- baserast
shp.soy.yld.br_subset <- subset(shp.soy.has, Ano < 2017 )
v <- vect(shp.soy.yld.br_subset)
ra <- cellSize(r, unit="ha")         
e <- expanse(v, unit="ha") 
v$density <- v$Valor / e

years <- str_sort(unique(v$Ano))
out <- list()
for (i in 1:length(years)) {
  vv <- v[v$Ano == years[i], ]
  x <- rasterize(vv, r, "density")
  out[[i]] <- x * ra
}
out <- rast(out)
names(out) <- years

writeRaster(out,"soy_harvest_area_br_1980_2016_05x05_density_03.tif", overwrite=TRUE)


