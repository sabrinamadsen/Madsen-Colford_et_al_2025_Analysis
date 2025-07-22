# This code is used to compare biogenic fluxes from SMUrF and UrbanVPRM to 
# anthropogenic fluxes estimated by the EDGAR and ODIAC inventories. The code
# takes the average fluxes (biogenic or anthropogenic) for each hour of the day
# for the specified season and saves it as a .csv file. 
# See Madsen-Colford et al. 2025 for details
# Portions of the code with *** are sections that should be changed by the user
# (e.g. path & filenames, which season to use, etc.)

memory.limit(size=5e5)
library("raster")
library("ggplot2")
library('ncdf4')
library('data.table')
library('sf')
library('latex2exp')
library("dplyr")
library("cowplot")
options(dplyr.summarise.inform = FALSE)

# *** Set to T if you want to compute both anthropogenic and biogenic fluxes (takes longer!)
# Set to F is you only want biogenic fluxes ***
anthro <- F

## Import monthly EDGAR data for resampling biogenic fluxes
# *** CHANGE PATH & FILENAME ***
EDGAR_path = 'E:/Research/Emission_Inventories/EDGAR_v8/Monthly_gridmaps/'
EDGAR_mth_dat = nc_open(paste0(EDGAR_path,'Transport/v8.0_FT2022_GHG_CO2_2018_TRANSPORT_flx.nc'))

## Extract lat, lon and time
lons = as.vector(ncvar_get(EDGAR_mth_dat,'lon'))
lats = as.vector(ncvar_get(EDGAR_mth_dat,'lat'))
time = as.vector(ncvar_get(EDGAR_mth_dat,'time'))

## Find indices in Toronto and 2018
lonIdx <- which( lons >= -80 & lons < -78.5)
latIdx <- which( lats >= 43 & lats < 44)
timeIdx <- which( time >= 0 & time < 365)

if(anthro==F){
  # NOTE: THIS IS ONLY FLUXES FROM TRANSPORTATION
  tra = ncvar_get(EDGAR_mth_dat,'fluxes')[lonIdx,latIdx,timeIdx]
  
  ## Make a grid of indices in space & time & combine indices with transport emissions
  indices <- expand.grid(lons[lonIdx],lats[latIdx],time[timeIdx])
  EDGAR_df <- data.frame(cbind(indices,as.vector(tra)))
  names(EDGAR_df)  <- c("lon","lat","doy","total")
  rm(tra,EDGAR_mth_dat,indices,latIdx,lonIdx,timeIdx,lats,lons,time)
}else{
  ## Extract each category of emissions
  tra <- ncvar_get(EDGAR_mth_dat,"fluxes")[lonIdx,latIdx,timeIdx]
  nc_close(EDGAR_mth_dat)
  EDGAR_mth_dat = nc_open(paste0(EDGAR_path,'Agriculture/v8.0_FT2022_GHG_CO2_2018_AGRICULTURE_flx.nc'))
  ag <- ncvar_get(EDGAR_mth_dat,"fluxes")[lonIdx,latIdx,timeIdx]
  nc_close(EDGAR_mth_dat)
  EDGAR_mth_dat = nc_open(paste0(EDGAR_path,'Building/v8.0_FT2022_GHG_CO2_2018_BUILDINGS_flx.nc'))
  bld <- ncvar_get(EDGAR_mth_dat,"fluxes")[lonIdx,latIdx,timeIdx]
  nc_close(EDGAR_mth_dat)
  EDGAR_mth_dat = nc_open(paste0(EDGAR_path,'Fuel_exploitation/v8.0_FT2022_GHG_CO2_2018_FUEL_EXPLOITATION_flx.nc'))
  fu_ex <- ncvar_get(EDGAR_mth_dat,"fluxes")[lonIdx,latIdx,timeIdx]
  nc_close(EDGAR_mth_dat)
  EDGAR_mth_dat = nc_open(paste0(EDGAR_path,'Industry/v8.0_FT2022_GHG_CO2_2018_IND_PROCESSES_flx.nc'))
  in_pr <- ncvar_get(EDGAR_mth_dat,"fluxes")[lonIdx,latIdx,timeIdx]
  nc_close(EDGAR_mth_dat)
  EDGAR_mth_dat = nc_open(paste0(EDGAR_path,'Industry_combustion/v8.0_FT2022_GHG_CO2_2018_IND_COMBUSTION_flx.nc'))
  in_co <- ncvar_get(EDGAR_mth_dat,"fluxes")[lonIdx,latIdx,timeIdx]
  nc_close(EDGAR_mth_dat)
  EDGAR_mth_dat = nc_open(paste0(EDGAR_path,'Power_industry/v8.0_FT2022_GHG_CO2_2018_POWER_INDUSTRY_flx.nc'))
  pwr <- ncvar_get(EDGAR_mth_dat,"fluxes")[lonIdx,latIdx,timeIdx]
  nc_close(EDGAR_mth_dat)
  EDGAR_mth_dat = nc_open(paste0(EDGAR_path,'Waste/v8.0_FT2022_GHG_CO2_2018_WASTE_flx.nc'))
  wst <- ncvar_get(EDGAR_mth_dat,"fluxes")[lonIdx,latIdx,timeIdx]
  nc_close(EDGAR_mth_dat)
  
  ## Make a grid of indices in space & time & combine indices with categories of emissions
  indices <- expand.grid(lons[lonIdx],lats[latIdx],time[timeIdx])
  EDGAR_df <- data.frame(cbind(indices,as.vector(ag),as.vector(bld),as.vector(fu_ex),as.vector(in_co),as.vector(in_pr),as.vector(pwr),as.vector(tra),as.vector(wst)))
  names(EDGAR_df)  <- c("lon","lat","doy","ag","bld","fu_ex","in_co","in_pr","pwr","tra","wst")
  EDGAR_df <- EDGAR_df %>% rowwise() %>% mutate(total = sum(c_across(ag:wst)))
  
  rm(tra,ag,bld,fu_ex,in_pr,in_co,pwr,wst,EDGAR_mth_dat,indices,latIdx,lonIdx,
     timeIdx,lats,lons,time)
  
  ## Import TIMES diurnal scaling factors
  # *** CHANGE PATH ***
  TIMES_path <- 'E:/Research/Emission_Inventories/TIMES/'
  TIMES_fctrs <- nc_open(paste0(TIMES_path,'diurnal_scale_factors.nc'))
  
  ## Define TIME lats, lons and time
  TIMES_lons = seq(from = -179.875, to = 180.125, by = 0.25)
  TIMES_lats = seq(from = -89.875, to = 90.125, by = 0.25)
  TIMES_time = c(0:23)
  
  ## Find indices within Toronto
  TIMES_lonIdx <- which( TIMES_lons >= -80 & TIMES_lons < -78.5)
  TIMES_latIdx <- which( TIMES_lats >= 43 & TIMES_lats < 44)
  TIMES_timeIdx <- which( TIMES_time>= 0 & TIMES_time< 24)
  
  ## Extract TIMES factors in Toronto
  TIMES_scl <- ncvar_get(TIMES_fctrs,"diurnal_scale_factors")[TIMES_timeIdx,TIMES_lonIdx,TIMES_latIdx]
  indices <- expand.grid(TIMES_time[TIMES_timeIdx],TIMES_lons[TIMES_lonIdx],TIMES_lats[TIMES_latIdx])
  scl_df <- data.frame(cbind(indices,as.vector(TIMES_scl)))
  names(scl_df)  <- c("hr","lon","lat","scl")
  
  ## Import stdev of TIMES diurnal scaling factors
  TIMES_std <- nc_open(paste0(TIMES_path,'diurnal_std_dev.nc'))
  ## Extract TIMES factors std in Toronto
  TIMES_scl_std <- ncvar_get(TIMES_std,"stddev_diurnal_scale_factors")[TIMES_lonIdx,TIMES_latIdx]
  lat_lon_indices <- expand.grid(TIMES_lons[TIMES_lonIdx],TIMES_lats[TIMES_latIdx])
  scl_std_df <- data.frame(cbind(lat_lon_indices,as.vector(TIMES_scl_std)))
  names(scl_std_df)  <- c("lon","lat","std")
  nc_close(TIMES_std)
  
  rm(indices,TIMES_fctrs,TIMES_scl_std,TIMES_std,TIMES_timeIdx,TIMES_time,
     TIMES_scl)
  
  ## Extract weekly TIMES scale factors
  TIMES_wkly_fctrs <- nc_open(paste0(TIMES_path,'weekly_factors_scale_China.nc'))
  TIMES_wkly_time = c(1:7)
  
  TIMES_wkly_timeIdx <- which( TIMES_wkly_time>= 0 & TIMES_wkly_time< 8)
  
  TIMES_wkly_scl <- ncvar_get(TIMES_wkly_fctrs,"weekly_scale_factors")[TIMES_wkly_timeIdx,TIMES_lonIdx,TIMES_latIdx]
  wkly_indices <- expand.grid(TIMES_wkly_time[TIMES_wkly_timeIdx],TIMES_lons[TIMES_lonIdx],TIMES_lats[TIMES_latIdx])
  scl_wkly_df <- data.frame(cbind(wkly_indices,as.vector(TIMES_wkly_scl)))
  names(scl_wkly_df)  <- c("wday","lon","lat","scl")
  
  scl_wkly_df <- scl_wkly_df[,c(2,3,4,1)]
  
  ## Import stdev of TIMES diurnal scaling factors
  TIMES_wkly_std <- nc_open(paste0(TIMES_path,'weekly_std_dev_scale_China.nc'))
  ## Extract TIMES factors std in Toronto
  TIMES_wkly_scl_std <- ncvar_get(TIMES_wkly_std,"stddev_weekly_scale_factors")[TIMES_lonIdx,TIMES_latIdx]
  scl_wkly_std_df <- data.frame(cbind(lat_lon_indices,as.vector(TIMES_wkly_scl_std)))
  names(scl_wkly_std_df)  <- c("lon","lat","std")
  nc_close(TIMES_wkly_std)
  
  rm(lat_lon_indices,TIMES_latIdx,TIMES_lonIdx,TIMES_lats,TIMES_lons,
     TIMES_wkly_std,TIMES_wkly_scl,TIMES_wkly_time,TIMES_wkly_timeIdx,
     wkly_indices,TIMES_wkly_scl_std)
  
  scl_w_std = as.numeric(names(sort(-table(scl_wkly_std_df$std[scl_wkly_std_df$std!=0])))[1])
  
  scl_w <- NULL
  for(w in 1:7){
    if(length(scl_w)==0){
      scl_w = as.numeric(names(sort(-table(scl_wkly_df$scl[scl_wkly_df$wday==w][scl_wkly_df$scl[scl_wkly_df$wday==w]!=1])))[1])
    }else if(w<7){
      scl_w = append(scl_w,as.numeric(names(sort(-table(scl_wkly_df$scl[scl_wkly_df$wday==w][scl_wkly_df$scl[scl_wkly_df$wday==w]!=1])))[1]))
    }else{
      scl_w = c(as.numeric(names(sort(-table(scl_wkly_df$scl[scl_wkly_df$wday==w][scl_wkly_df$scl[scl_wkly_df$wday==w]!=1])))[1]),scl_w)
    }
  }
  
  nc_close(TIMES_wkly_fctrs)
  rm(TIMES_wkly_fctrs,w)
  
  #Define ODIAC files
  inDIR <- paste0('E:/Research/Emission_Inventories/ODIAC/2018')
  ODIAC_2018_files <- list.files(path=inDIR,pattern='.tif')
  
}

## Bring in UrbanVPRM data 
# *** CHANGE PATH & FILENAME ***
vprm_dir <- "E:/Research/UrbanVPRM/dataverse_files/GTA_V061_500m_2018/"
vprm_files <- list.files(vprm_dir,pattern = "vprm_GMIS_Toronto_ACI_SOLRIS_ISA_500m_GTA_V061_2018_no_PScale_adjusted_Topt_Ra_URB_parameters_fixed_gapfilled_LSWI_filtered_bilinear_PAR_block")

VPRM_data <- NULL
for(f in vprm_files){
  if(length(VPRM_data)==0){
    VPRM_data <- fread(paste0(vprm_dir,f),select=c("HoY","Index","GEE","Re"),data.table = FALSE)
  }else{
    VPRM_data2 <- fread(paste0(vprm_dir,f),select=c("HoY","Index","GEE","Re"),data.table = FALSE)
    VPRM_data <- rbind(VPRM_data,VPRM_data2)
  }
}
rm(f,vprm_files)

## Bring in index, x, & y data to assign VPRM fluxes to x & y values
vprm_lat_lon <- fread(paste0(vprm_dir,"adjusted_evi_lswi_interpolated_modis_v061_qc_filtered_LSWI_filtered.csv"),select=c("Index","DOY","x","y"))
vprm_lat_lon <- vprm_lat_lon[vprm_lat_lon$DOY==1]
vprm_lat_lon <- as.data.frame(vprm_lat_lon)
vprm_lat_lon$DOY <- NULL

VPRM_merge <- merge(VPRM_data,vprm_lat_lon,by="Index")

## Calculate VPRM NEE
VPRM_merge$NEE <- VPRM_merge$GEE+VPRM_merge$Re

## Create datetime to define month & hour and add to VPRM dataframe
dt <- as.POSIXct((unique(VPRM_merge$HoY)-1)*3600,origin="2018-01-01-01-00-00",tz="UTC")
VPRM_merge$mo <- as.numeric(substr(dt,6,7))
VPRM_merge$hr <- (VPRM_merge$HoY-1) %% 24

if(anthro==T){
  #also include a column for day of the week
  VPRM_merge$wday <- wday(dt)
  VPRM_merge$wday <- VPRM_merge$wday-1
  VPRM_merge$wday[VPRM_merge$wday==0] <- 7
  VPRM_df <- VPRM_merge[,c(5,6,7,8,2,9,10)]
}else{
  VPRM_df <- VPRM_merge[,c(5,6,7,8,2,9)]
}

rm(vprm_dir,VPRM_data,VPRM_data2,vprm_lat_lon,VPRM_merge,dt)

#Define number of days in each month
mon_num_days = c(31,28,31,30,31,30,31,31,30,31,30,31)
#Jn,Fb,Mr,Ap,Ma,Jn,Jl,Au,sp,Oc,Nv,Dc

## Bring in Toronto SMUrF for cropping other data
# *** CHANGE PATH & FILENAME ***
SMUrF_NEE_Toronto=raster('C:/Users/kitty/Documents/Research/SIF/SMUrF/SMUrF_Toronto_2018_2021_average_fluxes.nc',varname='NEE')
SMUrF_NEE_Toronto_resamp <- resample(SMUrF_NEE_Toronto,crop(rasterFromXYZ(EDGAR_df[c(1,2,4)]),SMUrF_NEE_Toronto))
rm(SMUrF_NEE_Toronto)

if(anthro==F){
  # If only running biogenic fluxes create a dataframe for EDGAR emissions
  # for cropping biogenic fluxes to EDGAR resolution
  EDGAR_mth_df <- data.frame(cbind(EDGAR_df$lon[EDGAR_df$doy==unique(EDGAR_df$doy)[1]],
                                   EDGAR_df$lat[EDGAR_df$doy==unique(EDGAR_df$doy)[1]],
                                   EDGAR_df$doy[EDGAR_df$doy==unique(EDGAR_df$doy)[1]],
                                   EDGAR_df$total[EDGAR_df$doy==unique(EDGAR_df$doy)[1]]))
  names(EDGAR_mth_df) <- c("lon","lat","doy","total")
  
  #convert it to a raster
  EDGAR_rstr <- rasterFromXYZ(EDGAR_mth_df)
  EDGAR_rstr <- EDGAR_rstr$total #only keep total emissions
  crs(EDGAR_rstr) <- '+proj=longlat +datum=WGS84 +no_defs'
  rm(EDGAR_mth_df,EDGAR_df)
  
  #Crop the EDGAR data to Toronto
  EDGAR_2018_cropped = crop(EDGAR_rstr,SMUrF_NEE_Toronto_resamp)*1000*10^6/44.0095 #convert to umol/m2/s
  rm(EDGAR_rstr)
  
  # Remove values outside of Toronto
  EDGAR_2018_cropped[is.na(SMUrF_NEE_Toronto_resamp)]<-NA
}

# *** CHANGE PATH & FILENAMES*** 
SMUrF_path <- 'C:/Users/kitty/Documents/Research/SIF/SMUrF/output2018_500m_CSIF_to_TROPOMI_CSIF_ALL_converted_slps_V3_temp_impervious_R_shore_corr_V061_8day/easternCONUS/hourly_flux_GMIS_Toronto_fixed_border_ISA_a_w_sd_era5/'
#Filename for fluxes (WITHOUT month)
SMUrF_fn <- 'hrly_mean_GPP_Reco_NEE_easternCONUS_2018'
# Filename for uncertainty (WITHOUT month)
SMUrF_sd_fn <- 'hrly_mean_GPP_Reco_NEE_sd_easternCONUS_2018'

# define a function for converting a dataframe to a raster stack 
# (with each layer representing an hour of the month)
rstr_stk_fun<- function(h){
  df_hour <- df_split[[h]]
  r <- rasterFromXYZ(df_hour[,c('x','y','NEE')])
  names(r) <- paste0("X",as.POSIXct((as.numeric(h)-1)*3600,origin="2018-01-01-00-00",tz="UTC"))
  return(r)
}

# *** SELECT WHICH SEASON YOU WANT TO RUN ***
# Options are: 'MAM' (March-May), 'JJA' (July-August), 
# 'SON' (September-November), 'DJF' (January, February & December)  
season <- 'DJF'

if(season=='MAM'){
  sel_mons <- c(3:5)
}else if(season=='JJA'){
  sel_mons <- c(6:8)
}else if(season=='SON'){
  sel_mons <- c(9:11)
}else if(season=='DJF'){
  sel_mons <- c(1,2,12)
}

for (i in sel_mons){
  if(i<10){
    SMUrF_NEE=stack(paste0(SMUrF_path,SMUrF_fn,'0',i,'.nc'),varname='NEE_mean')
    #units of umol/m2/s
    SMUrF_NEE_sd=stack(paste0(SMUrF_path,SMUrF_sd_fn,'0',i,'.nc'),varname='NEE_sd')
    #units of umol/m2/s
  }else{
    SMUrF_NEE=stack(paste0(SMUrF_path,SMUrF_fn,i,'.nc'),varname='NEE_mean')
    #units of umol/m2/s
    SMUrF_NEE_sd=stack(paste0(SMUrF_path,SMUrF_sd_fn,i,'.nc'),varname='NEE_sd')
  }
  
  #Crop to data to Toronto 
  SMUrF_NEE_crop <- crop(SMUrF_NEE,SMUrF_NEE_Toronto_resamp)
  SMUrF_NEE_sd_crop <- crop(SMUrF_NEE_sd,SMUrF_NEE_Toronto_resamp)
  rm(SMUrF_NEE_sd,SMUrF_NEE)
  
  #Select UrbanVPRM data for the specified month
  VPRM_df_m <- VPRM_df[VPRM_df$mo==i,]
  
  if (anthro==T){
    #Create a dataframe for total EDGAR emissions
    EDGAR_mth_df <- data.frame(cbind(EDGAR_df$lon[EDGAR_df$doy==unique(EDGAR_df$doy)[i]],
                                     EDGAR_df$lat[EDGAR_df$doy==unique(EDGAR_df$doy)[i]],
                                     EDGAR_df$doy[EDGAR_df$doy==unique(EDGAR_df$doy)[i]],
                                     EDGAR_df$total[EDGAR_df$doy==unique(EDGAR_df$doy)[i]]))
    names(EDGAR_mth_df) <- c("lon","lat","doy","total")
    
    #convert it to a raster
    EDGAR_rstr <- rasterFromXYZ(EDGAR_mth_df)
    EDGAR_rstr <- EDGAR_rstr$total #only keep total emissions
    rm(EDGAR_mth_df)
    
    #Crop the EDGAR data to Toronto
    EDGAR_2018_cropped = crop(EDGAR_rstr,SMUrF_NEE_Toronto_resamp)*1000*10^6/44.0095 #convert to umol/m2/s
    rm(EDGAR_rstr)
    
    #Bring in and crop ODIAC data:
    ODIAC_2018_cropped = raster(paste0(inDIR,'/',ODIAC_2018_files[i]))*10^6*11/(3*44.0095*mon_num_days[i]*24*3600)
    ODIAC_2018_cropped = crop(ODIAC_2018_cropped,SMUrF_NEE_Toronto_resamp) #*scl_fct_ont
    
    #resample ODIAC to the same resolution as EDGAR
    ODIAC_2018_resamp = aggregate(ODIAC_2018_cropped,12,mean)
    rm(ODIAC_2018_cropped)
    
    # Remove data outside of the city of Toronto
    EDGAR_2018_cropped[is.na(SMUrF_NEE_Toronto_resamp)]<-NA
    ODIAC_2018_resamp[is.na(SMUrF_NEE_Toronto_resamp)]<-NA
    
    #Make a stack of EDGAR and ODIAC values for each hour of the day
    EDGAR_2018_stk <- stack(replicate(24,EDGAR_2018_cropped))
    ODIAC_2018_stk <- stack(replicate(24,ODIAC_2018_resamp))
    
    EDGAR_2018_std_stk <- stack(replicate(24,EDGAR_2018_cropped))
    ODIAC_2018_std_stk <- stack(replicate(24,ODIAC_2018_resamp))
    rm(ODIAC_2018_resamp)
    
    #Multiply by hourly scale factors
    # Since the scale factor is the same for all data over land in Ontario 
    # Choose some pixel that falls over land to extract hourly scale factor
    scl_h <- scl_df$scl[(scl_df$lon==scl_df$lon[433]) & (scl_df$lat==scl_df$lat[433])]
    EDGAR_2018_stk <- EDGAR_2018_stk*scl_h
    ODIAC_2018_stk <- ODIAC_2018_stk*scl_h
    EDGAR_2018_h_std_stk <- EDGAR_2018_stk*scl_std_df$std[20]
    ODIAC_2018_h_std_stk <- ODIAC_2018_stk*scl_std_df$std[20]
    
    rm(scl_h)
    
    #Make stacks of EDGAR and ODIAC values for each hour of the month
    EDGAR_2018_stk <- stack(replicate(length(unique(VPRM_df_m$HoY))/24,EDGAR_2018_stk))
    names(EDGAR_2018_stk) <- names(SMUrF_NEE_crop)
    EDGAR_2018_std_stk <- stack(replicate(length(unique(VPRM_df_m$HoY))/24,EDGAR_2018_h_std_stk))
    names(EDGAR_2018_std_stk) <- names(SMUrF_NEE_crop)
    EDGAR_2018_h_std_stk <- EDGAR_2018_std_stk
    
    ODIAC_2018_stk <- stack(replicate(length(unique(VPRM_df_m$HoY))/24,ODIAC_2018_stk))
    names(ODIAC_2018_stk) <- names(SMUrF_NEE_crop)
    ODIAC_2018_std_stk <- stack(replicate(length(unique(VPRM_df_m$HoY))/24,ODIAC_2018_h_std_stk))
    names(ODIAC_2018_std_stk) <- names(SMUrF_NEE_crop)
    ODIAC_2018_h_std_stk <- ODIAC_2018_std_stk
    
    #Determine the day of the week for each stack layer:
    dow <- wday(as.Date(paste0(substr(names(EDGAR_2018_stk),2,5),"-",substr(names(EDGAR_2018_stk),7,8),"-",substr(names(EDGAR_2018_stk),10,11))))
    
    #scale EDGAR emissions based on weekly TIMES data
    for(w in 1:7){
      l_replace<-which(dow==w)
      for(ind in l_replace){
        EDGAR_2018_stk[[ind]] <- EDGAR_2018_stk[[ind]]*scl_w[w]
        EDGAR_2018_std_stk[[ind]] <- sqrt((EDGAR_2018_stk[[ind]]*scl_w_std)^2+(EDGAR_2018_h_std_stk[[ind]]*scl_w[w])^2)
        ODIAC_2018_stk[[ind]] <- ODIAC_2018_stk[[ind]]*scl_w[w]
        ODIAC_2018_std_stk[[ind]] <- sqrt((ODIAC_2018_stk[[ind]]*scl_w_std)^2+(ODIAC_2018_h_std_stk[[ind]]*scl_w[w])^2)
      }
    }
    
    rm(ODIAC_2018_h_std_stk,EDGAR_2018_h_std_stk)
    
    # Save the anthropogenic data to a raster stack 
    if(i==sel_mons[1]){
      EDGAR_stk <- EDGAR_2018_stk
      EDGAR_std_stk <- EDGAR_2018_std_stk
      ODIAC_stk <- ODIAC_2018_stk
      ODIAC_std_stk <- ODIAC_2018_std_stk
    }else{
      EDGAR_stk <- stack(EDGAR_stk,EDGAR_2018_stk)
      EDGAR_std_stk <- stack(EDGAR_std_stk,EDGAR_2018_std_stk)
      ODIAC_stk <- stack(ODIAC_stk,ODIAC_2018_stk)
      ODIAC_std_stk <- stack(ODIAC_std_stk,ODIAC_2018_std_stk)
    }
    
  }else if (i==sel_mons[1]){
    #Otherwise just create a raster stack for cropping (not for computation)
    #Create a raster stack for each hour of the month
    EDGAR_2018_stk <- stack(replicate(24,EDGAR_2018_cropped))
    EDGAR_2018_stk <- stack(replicate(length(unique(VPRM_df_m$HoY))/24,EDGAR_2018_stk))
    names(EDGAR_2018_stk) <- names(SMUrF_NEE_crop)
  }
  
  #Convert VPRM dataframe to a raster stack
  df_split<-split(VPRM_df_m, VPRM_df_m$HoY)
  VPRM_rstr_stk <- stack(lapply(names(df_split),rstr_stk_fun))
  names(VPRM_rstr_stk) <- names(SMUrF_NEE_crop)
  crs(VPRM_rstr_stk) <- crs(SMUrF_NEE_crop)
  
  # Aggregate fluxes to EDGAR resolution
  SMUrF_NEE_resamp <- aggregate(SMUrF_NEE_crop,24,mean)
  SMUrF_NEE_resamp <- resample(SMUrF_NEE_resamp,EDGAR_2018_cropped)
  SMUrF_NEE_sd_resamp <- aggregate(SMUrF_NEE_sd_crop,24,mean)
  SMUrF_NEE_sd_resamp <- resample(SMUrF_NEE_sd_resamp,EDGAR_2018_cropped)
  VPRM_NEE_resamp <- aggregate(VPRM_rstr_stk,24,mean)
  VPRM_NEE_resamp <- resample(VPRM_NEE_resamp,EDGAR_2018_cropped)
  
  #SMUrF_NEE_max_resamp <- aggregate(SMUrF_NEE_crop+SMUrF_NEE_sd_crop,24,mean)
  #SMUrF_NEE_max_resamp <- resample(SMUrF_NEE_max_resamp,EDGAR_2018_cropped)
  VPRM_NEE_max_resamp <- aggregate((VPRM_rstr_stk+SMUrF_NEE_sd_crop),24,mean)
  VPRM_NEE_max_resamp <- resample(VPRM_NEE_max_resamp,EDGAR_2018_cropped)
  
  #Set values outside of the city to NaN
  SMUrF_NEE_resamp[is.na(SMUrF_NEE_Toronto_resamp)] <- NA
  SMUrF_NEE_sd_resamp[is.na(SMUrF_NEE_Toronto_resamp)] <- NA
  VPRM_NEE_resamp[is.na(SMUrF_NEE_Toronto_resamp)] <- NA
  
  #SMUrF_NEE_max_resamp[is.na(SMUrF_NEE_Toronto_resamp)] <- NA
  VPRM_NEE_max_resamp[is.na(SMUrF_NEE_Toronto_resamp)] <- NA
  
  # Save the biogenic data to a raster stack 
  if(i==sel_mons[1]){
    SMUrF_stk <- SMUrF_NEE_resamp
    SMUrF_std_stk <- SMUrF_NEE_sd_resamp
    VPRM_stk <- VPRM_NEE_resamp
    
    #SMUrF_max_stk <- SMUrF_NEE_max_resamp
    VPRM_max_stk <- VPRM_NEE_max_resamp
  }else{
    SMUrF_stk <- stack(SMUrF_stk,SMUrF_NEE_resamp)
    SMUrF_std_stk <- stack(SMUrF_std_stk,SMUrF_NEE_sd_resamp)
    VPRM_stk <- stack(VPRM_stk,VPRM_NEE_resamp)
    
    #SMUrF_max_stk <- stack(SMUrF_max_stk,SMUrF_NEE_max_resamp)
    VPRM_max_stk <- stack(VPRM_max_stk,VPRM_NEE_max_resamp)
  }
  rm(VPRM_df_m,df_split,VPRM_rstr_stk,SMUrF_NEE_resamp,SMUrF_NEE_sd_resamp,VPRM_NEE_resamp)
  
  print(paste0("month: ",i," done"))
  removeTmpFiles(h=0.25)
}


#calculate the percentage of each EDGAR pixel that falls within Toronto's bounds

Toronto_sf <- st_read("C:/Users/kitty/Documents/Research/SIF/Shape_files/Toronto/Toronto_BB.shp")

EDGAR_coords <-  xyFromCell(EDGAR_2018_stk[[1]],1:ncell(EDGAR_2018_stk[[1]]))

int_perc <-NULL
for (i in 1:length(EDGAR_coords[,1])){
  xval <- EDGAR_coords[i,1]
  yval <- EDGAR_coords[i,2]
  pix <- data.frame(PID=rep(1,4),POS=1:4, X=c(xval-0.05,xval-0.05,xval+0.05,xval+0.05),Y=c(yval-0.05,yval+0.05,yval+0.05,yval-0.05))
  pix_xy <- pix[,c(3,4)]
  pix_poly  <- pix_xy %>% st_as_sf(coords = c("X","Y"),crs=crs(Toronto_sf)) %>% summarise(geometry = st_combine(geometry)) %>% st_cast("POLYGON")
  
  pix_Toronto_inter <- st_intersection(Toronto_sf$geometry,pix_poly)
  
  if (length(int_perc)==0){
    if (length(pix_Toronto_inter)==0){
      int_perc <- 0      
    }else{
      int_perc <- st_area(pix_Toronto_inter)/st_area(pix_poly)
    }
  }else{
    if (length(pix_Toronto_inter)==0){
      int_perc <-append(int_perc,0)
    }else{
      int_perc <- append(int_perc, st_area(pix_Toronto_inter)/st_area(pix_poly))
    }
  }
}

HoD <- seq(0,23)
if(season=='MAM'){
  sel_HoY <- as.vector(t(seq(1:(744+720+744))))
}else if(season=='JJA'){
  sel_HoY <- as.vector(t(seq(1:(720+744+744))))
}else if(season=='SON'){
  sel_HoY <- as.vector(t(seq(1:(720+744+720))))
}else if(season=='DJF'){
  sel_HoY <- as.vector(t(seq(1:(744+672+744))))
}

#Loop over each hour to find average fluxes for each hour for the specified season
for (h in 0:23){
  #Select the same hour each day
  hvals <- which(sel_HoY %% 24 == (h+5) %% 24)
  mean_SMUrF <- sum(values(calc(SMUrF_stk[[hvals]],mean)*int_perc),na.rm=TRUE)/sum(int_perc)
  mean_std_SMUrF <- sum(values(calc(SMUrF_std_stk[[hvals]],mean)*int_perc),na.rm=TRUE)/sum(int_perc)
  
  mean_VPRM <- sum(values(calc(VPRM_stk[[hvals]],mean)*int_perc),na.rm=TRUE)/sum(int_perc)
  
  max_VPRM <- sum(values(calc(VPRM_max_stk[[hvals]],mean)*int_perc),na.rm=TRUE)/sum(int_perc)
  
  if(anthro==T){
    mean_std_ODIAC <- sum(values(calc(ODIAC_std_stk[[hvals]],mean)*int_perc),na.rm=TRUE)/sum(int_perc)
    mean_ODIAC <- sum(values(calc(ODIAC_stk[[hvals]],mean)*int_perc),na.rm=TRUE)/sum(int_perc)
    
    mean_std_EDGAR <- sum(values(calc(EDGAR_std_stk[[hvals]],mean)*int_perc),na.rm=TRUE)/sum(int_perc)
    mean_EDGAR <- sum(values(calc(EDGAR_stk[[hvals]],mean)*int_perc),na.rm=TRUE)/sum(int_perc)
  }
  if (h==0){
    SMUrF_sd_diurnal <- mean_std_SMUrF
    SMUrF_mean_diurnal <- mean_SMUrF
    
    VPRM_mean_diurnal <- mean_VPRM
    
    VPRM_sd_diurnal <- max_VPRM-mean_VPRM
    
    if(anthro==T){
      ODIAC_sd_diurnal <- mean_std_ODIAC
      ODIAC_mean_diurnal <- mean_ODIAC
      
      EDGAR_sd_diurnal <- mean_std_EDGAR
      EDGAR_mean_diurnal <- mean_EDGAR
    }
  }else{
    SMUrF_sd_diurnal <- append(SMUrF_sd_diurnal, mean_std_SMUrF) 
    SMUrF_mean_diurnal <- append(SMUrF_mean_diurnal, mean_SMUrF)
    
    VPRM_mean_diurnal <- append(VPRM_mean_diurnal, mean_VPRM)
    
    VPRM_sd_diurnal <- append(VPRM_sd_diurnal, max_VPRM-mean_VPRM)
    
    if(anthro==T){
      ODIAC_sd_diurnal <- append(ODIAC_sd_diurnal, mean_std_ODIAC) 
      ODIAC_mean_diurnal <- append(ODIAC_mean_diurnal, mean_ODIAC)
    
      EDGAR_sd_diurnal <- append(EDGAR_sd_diurnal, mean_std_EDGAR) 
      EDGAR_mean_diurnal <- append(EDGAR_mean_diurnal, mean_EDGAR)
    }
  }
}

if(anthro==T){
  df <- data.frame(x = HoD, S_NEE = SMUrF_mean_diurnal, 
                   S_NEE_sd = SMUrF_sd_diurnal, V_NEE = VPRM_mean_diurnal,
                   O_flx = ODIAC_mean_diurnal, O_flx_sd = ODIAC_sd_diurnal,
                   E_flx = EDGAR_mean_diurnal, E_flx_sd = EDGAR_sd_diurnal)
  # *** CHANGE PATH & FILENAME ***
  write.csv(df, file = paste0('C:/Users/kitty/Documents/Research/SIF/Emission_inventories/Emission_inventory_comparison/',season,'_anthro_bio_diurnal_sys_errs_all_VPRM_abs_errs_SMUrF_fix.csv'), row.names = FALSE)
}else{
  df <- data.frame(x = HoD, S_NEE = SMUrF_mean_diurnal,
                   S_NEE_sd = SMUrF_sd_diurnal, V_NEE = VPRM_mean_diurnal,
                   V_NEE_sd = VPRM_sd_diurnal)
  # *** CHANGE PATH & FILENAME ***
  write.csv(df, file = paste0('C:/Users/kitty/Documents/Research/SIF/Emission_inventories/Emission_inventory_comparison/',season,'_bio_diurnal_sys_errs_all_VPRM_abs_errs_SMUrF_fix.csv'), row.names = FALSE)
}




ggplot(df, aes(x=x,y=S_NEE))+geom_point(aes(color='SMUrF'),size=1)+
  geom_errorbar(aes(ymin=S_NEE-S_NEE_sd,ymax=S_NEE+S_NEE_sd),width=0.5,col='red')+
  geom_point(aes(x=x,y=V_NEE,color='UrbanVPRM'),size=1)+
  geom_errorbar(aes(ymin=V_NEE-V_NEE_sd,ymax=V_NEE+V_NEE_sd),width=0.5,col='blue')+
  #geom_point(aes(x=x,y=E_flx,color='EDGAR'),size=1)+
  #geom_errorbar(aes(ymin=E_flx-E_flx_sd,ymax=E_flx+E_flx_sd),width=0.5,col='orange')+
  #geom_point(aes(x=x,y=O_flx,color='ODIAC'),size=1)+
  #geom_errorbar(aes(x=x,ymin=O_flx-O_flx_sd,ymax=O_flx+O_flx_sd,color='ODIAC'),width=0.5,col='purple')+
  theme_bw()+
  theme(legend.position = "none",legend.title=element_blank(),
        plot.title = element_text(hjust = 0.5),
        axis.text.x=element_blank(),
        panel.grid.major = element_blank(),panel.grid.minor = element_blank(),
        panel.border = element_rect(colour = "black", fill=NA, size=1))+
  labs(title = "Winter",x=element_blank(),y=TeX(r'(CO$_2$ Fluxes ($\mu$mol m$^{-2}$ s$^{-1}$))'))+
  #ylim(min(df$V_NEE-df$V_NEE_sd),max(df$S_NEE+df$S_NEE_sd))+
  #theme(plot.margin = unit(c(0.1,-0.05,-0.3,0.18), "cm"))+
  scale_color_manual(values=c('red','blue'),name=NULL)
