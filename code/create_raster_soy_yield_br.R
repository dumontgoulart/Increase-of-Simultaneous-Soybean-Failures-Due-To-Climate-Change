library(sidrar)
library(terra)
library(tidyr)
library(dplyr)
library(sf)
library(data.table)
setwd("~/PhD/paper_hybrid_agri/data")
dir_crop_ibge <- ("C:/users/morenodu/OneDrive - Stichting Deltares/Documents/PhD/paper_hybrid_agri/data")

init_date <- 1975
end_date <- 2016

# period.list <- list(1975:1979)
# # info_sidra(1612)[[3]]
# var_list <- c(112)
# 
# mun.soy.yld <- lapply(period.list, function(x)
#   get_sidra(1612, variable = var_list,
#             period = as.character(x),
#             geo = "City",
#             classific = c("c81"),
#             category = list(2713)))
# 
# test_2 <- do.call(rbind,mun.soy.yld)
# write.csv(test_2,"soy_yield_munic_br_1975_1978.csv")

mun.soy.yld1 <- read.csv("soy_yield_munic_br_1975_1978.csv")
mun.soy.yld1 <- subset(mun.soy.yld1, Ano..Código. < 1979)

mun.soy.yld2 <- read.csv("soy_yield_munic_br.csv")
mun.soy.yld <- rbind(mun.soy.yld1, mun.soy.yld2)

# mun.soy.yld3 <- list.files(dir_crop_ibge, "soy_yield_munic_br", full.names = TRUE) %>%
#   map_dfr(read.csv) 

mun.soy.yld$Valor <- mun.soy.yld$Valor / 1000

mun.soy.yld <- mun.soy.yld %>% rename(ADM2_PCODE = 'Município..Código.') %>%
  dplyr::select(Ano,ADM2_PCODE,Valor) 

mun.soy.yld <- dplyr::arrange(mun.soy.yld , ADM2_PCODE, Ano)


BRmun <- st_read("GIS/bra_admbnda_adm2_ibge_2020.shp") %>% 
  mutate(ADM2_PCODE = substr(ADM2_PCODE,3,9)) %>%
  mutate_at(c('ADM2_PCODE'), as.numeric)

st_crs(BRmun) <- 4326
BRmun <- st_transform(BRmun, crs = 4326)

shp.soy.yld.br <- left_join(BRmun,mun.soy.yld) %>% drop_na()

# RASTER BASE EXTENTION
extent_br = c(-180, 180, -90, 90)
resolution_br = 0.05
ncol_br = (extent_br[2]-extent_br[1])/resolution_br
nrows_br = (extent_br[4]-extent_br[3])/resolution_br

# Raster creation
baserast <- rast(nrows=nrows_br, ncol=ncol_br,
                 extent= extent_br,
                 crs="+proj=longlat +datum=WGS84",
                 vals=NA)

rasters <- rast(lapply(init_date:end_date, 
                       function(x)
                         rasterize(vect(shp.soy.yld.br %>% 
                                          filter(Ano==x)),baserast,"Valor")))
names(rasters) <- init_date:end_date
varnames(rasters) <- paste0("soy_yield_",init_date:end_date)


writeRaster(rasters,"soy_yield_1975_2016_005.tif", overwrite = TRUE)



### Add the conditions for harvested area > 1% #################################
test <- shp.soy.yld.br %>% st_drop_geometry()
test_2 <- shp.soy.has %>% st_drop_geometry()
remove_cities <- anti_join(test_2, test, by = c('ADM2_PCODE','Ano','ADM1_PCODE'))

shp.soy.yld3 <- shp.soy.yld.br 

hist(shp.soy.yld3$Valor)
summary(shp.soy.yld3$Valor)

shp.soy.yld3$harv_area_frac <- area_municipality$harv_area_frac
sum(shp.soy.yld3$harv_area_frac < 0.01, na.rm=T) 
sum(shp.soy.yld3$Valor[shp.soy.yld3$harv_area_frac < 0.01], na.rm=T)

shp.soy.yld3$Valor[shp.soy.yld3$harv_area_frac < 0.01] <- NA

sum(shp.soy.yld3$harv_area_frac < 0.01, na.rm=T) 
sum(shp.soy.yld3$Valor[shp.soy.yld3$harv_area_frac < 0.01], na.rm=T)

extent_br = c(-180, 180, -90, 90)
resolution_br = 0.05
ncol_br = (extent_br[2]-extent_br[1])/resolution_br
nrows_br = (extent_br[4]-extent_br[3])/resolution_br

# Raster creation
baserast <- rast(nrows=nrows_br, ncol=ncol_br,
                 extent= extent_br,
                 crs="+proj=longlat +datum=WGS84",
                 vals=NA)

rasters <- rast(lapply(init_date:end_date, 
                       function(x)
                         rasterize(vect(shp.soy.yld3 %>% 
                                          filter(Ano==x)),baserast,"Valor")))
names(rasters) <- init_date:end_date
varnames(rasters) <- paste0("soy_yield_",init_date:end_date)

writeRaster(rasters,"soy_yield_1975_2016_005_1prc.tif", overwrite=TRUE)


### Filters
# (1) counties with more than 3 years (???4) of repeated values were considered to contain artificial data and were excluded from the analysis; 
# (2) only counties with more than 4 years (???5) of consecutive records were selected for the analysis, 

### Remove counties that have 6 or more times repeated the same value #################################
df_repeat <- shp.soy.yld3 %>% 
  count(Valor, ADM2_PCODE)  %>%
  filter(!is.na(Valor)) %>%
  group_by(ADM2_PCODE) %>%
  summarise(Value = max(n, na.rm = TRUE)) %>%
  mutate(repeat_mask = ifelse(Value >= 6, TRUE, FALSE))

### Remove counties where there are 4 or more consecutive identical values #################################
df_repeat2 <- shp.soy.yld3 %>%
  filter(!is.na(Valor)) %>%
  group_by(ADM2_PCODE) %>%
  summarise(max_consecutive_values = max(rle(Valor)$lengths, na.rm = TRUE)) %>%
  mutate(repeat_mask = ifelse(max_consecutive_values >= 4, TRUE, FALSE))

### Remove counties with 10 or less years #################################
df_minimum_years <- shp.soy.yld3 %>%
  filter(!is.na(Valor)) %>%
  group_by(ADM2_PCODE) %>%
  count(duration_year = max(as.integer(Ano))-min(as.integer(Ano)))  %>%
  mutate(time_mask = ifelse(n >= 10, FALSE, TRUE))
hist(df_minimum_years$n)

# Check amount of counties being removed for each criterium
sum(df_repeat$repeat_mask == TRUE)
sum(df_repeat2$repeat_mask == TRUE)
sum(df_minimum_years$time_mask == TRUE)

df_repeat <- data.frame(df_repeat$ADM2_PCODE,df_repeat$repeat_mask)
names(df_repeat) <- c("ADM2_PCODE", "repeat_mask")

df_repeat2 <- data.frame(df_repeat2$ADM2_PCODE,df_repeat2$repeat_mask)
names(df_repeat2) <- c("ADM2_PCODE", "consecutive_mask")

df_minimum_years <- data.frame(df_minimum_years$ADM2_PCODE,df_minimum_years$time_mask)
names(df_minimum_years) <- c("ADM2_PCODE", "minimum_time_mask")

df_all_filters <- data.frame(df_repeat$ADM2_PCODE, df_repeat$repeat_mask, df_repeat2$minimum_time_mask,df_minimum_years$minimum_time_mask) 
names(df_all_filters) <- c("ADM2_PCODE","repeat_mask","consecutive_mask", "minimum_time_mask")

# df_soy <- data.frame(shp.soy.yld3$ADM2_PCODE,shp.soy.yld3$Valor)
# names(df_soy) <- c("ADM2_PCODE", "Valor")
# 
# df_repeat3 <- df_soy %>%
#   filter(!is.na(Valor)) %>%
#   group_by(ADM2_PCODE) %>%
#   mutate(grp = rleid(Valor)) %>%
#   count(grp) %>%
#   summarise(max_consecutive_values = max(n))

shp.soy.yld4 <- left_join(shp.soy.yld3,df_all_filters)
shp.soy.yld4$Valor[shp.soy.yld4$repeat_mask == TRUE ] <- NA 
shp.soy.yld4$Valor[shp.soy.yld4$consecutive_mask == TRUE ] <- NA 
shp.soy.yld4$Valor[shp.soy.yld4$minimum_time_mask == TRUE ] <- NA 
hist(shp.soy.yld3$Valor)
hist(shp.soy.yld4$Valor)
summary(shp.soy.yld3$Valor)
summary(shp.soy.yld4$Valor)


### Create Raster
baserast <- rast(nrows=4860, ncol=5040,
                 extent= c(-74.25, -32.25, -34.25, 6.25),
                 crs="+proj=longlat +datum=WGS84",
                 vals=NA)

rasters <- rast(lapply(1980:2016, 
                       function(x)
                         rasterize(vect(shp.soy.yld4 %>% 
                                          filter(Ano==x)),baserast,"Valor")))
names(rasters) <- 1980:2016
varnames(rasters) <- paste0("soy_yield_",1980:2016)

writeRaster(rasters,"soy_yield_1980_2016_all_filters.tiff")

