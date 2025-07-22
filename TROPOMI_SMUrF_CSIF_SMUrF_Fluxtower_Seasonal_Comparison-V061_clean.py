#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This code compares & fits biogenic fluxes from the original and updated SMUrF model to 3 eddy covariance fluxtowers
# for each season of 2018 & 2019.

#  This creates Figs. 3 d-f of Madsen-Colford et al. 2025

# *** denotes areas that the user should change


# In[1]:


import numpy as np #numerical python
import matplotlib.pyplot as plt #for plotting
from matplotlib.cm import get_cmap #import colour maps for contour plots
import netCDF4
from netCDF4 import Dataset, date2num #for reading netCDF data files and their date (not sure if I need the later)
import datetime as dt
import pandas as pd
import math
from scipy import optimize as opt 
from scipy import odr
from scipy import stats
from datetime import datetime, timedelta
#from datetime import datetime as dt
import matplotlib.patheffects as pe
from sklearn import linear_model #for robust fitting
from sklearn.metrics import r2_score, mean_squared_error #for analyzing robust fits
import matplotlib.colors as clrs #for log color scale


# In[2]:


# Import first CSIF data file to get the start of the year & convert to days since 1970

# *** CHANGE PATH & FILENAME ***
g=Dataset('E:/Research/SMUrF/output2018_CSIF_V061/easternCONUS/daily_mean_Reco_neuralnet/era5/2018/daily_mean_Reco_uncert_easternCONUS_20180101.nc')
start_of_year=g.variables['time'][0]/3600/24-1 #convert seconds since 1970 to days (minus one)
g.close()


# In[3]:


# *** CHANGE PATH ***
C_path = 'E:/Research/SMUrF/output2018_CSIF_V061/easternCONUS/hourly_flux_era5/'
# *** CHANGE FILENAME ***
C_fn = 'hrly_mean_GPP_Reco_NEE_easternCONUS_2018'

C_time=[]
C_Reco=[]
C_NEE=[]
C_GPP=[]
C_lats=[]
C_lons=[]
for j in range(1,13):
    try:
        #Import original SMUrF data (using CSIF)
        if j<10:
            f=Dataset(C_path+C_fn+'0'+str(j)+'.nc')
        else:
            # *** CHANGE PATH ***
            f=Dataset(C_path+C_fn+str(j)+'.nc')
        if len(C_time)==0:
            # If it is the first file create arrays for each variable and save lat/lon
            C_lats=f.variables['lat'][:]
            C_lons=f.variables['lon'][:]
            C_Reco=f.variables['Reco_mean'][:]
            C_GPP=f.variables['GPP_mean'][:]
            C_NEE=f.variables['NEE_mean'][:]
            C_time=f.variables['time'][:]/24/3600-start_of_year-5/24 #convert seconds since 1970 to days and subtract start of year and adjust to local time
        else:
            # Otherwise append the data to the arrays
            C_Reco=np.concatenate((C_Reco,f.variables['Reco_mean'][:]),axis=0)
            C_GPP=np.concatenate((C_GPP,f.variables['GPP_mean'][:]),axis=0)
            C_NEE=np.concatenate((C_NEE,f.variables['NEE_mean'][:]),axis=0)
            C_time=np.concatenate((C_time,(f.variables['time'][:]/24/3600-start_of_year-5/24)),axis=0)
        f.close()
    except FileNotFoundError:
        pass


# In[4]:


# Replace fill values with NaN
C_Reco[C_Reco==-999]=np.nan
C_NEE[C_NEE==-999]=np.nan
C_GPP[C_GPP==-999]=np.nan


# In[5]:


# Select the pixel over Borden forest
C_GPP_array=np.zeros(8765)*np.nan
C_NEE_array=np.zeros(8765)*np.nan
C_Reco_array=np.zeros(8765)*np.nan
C_time_array=np.zeros(8765)*np.nan
for i in range(len(C_GPP[:,0,0])):
    C_time_array[i]=C_time[i]
    C_GPP_array[i]=C_GPP[i,36,15]
    C_NEE_array[i]=C_NEE[i,36,15]
    C_Reco_array[i]=C_Reco[i,36,15]


# In[6]:


#Import Borden Fluxtower data

# *** CHANGE PATH ***
Borden_Fluxes=pd.read_csv('/Users/kitty/Documents/Research/SIF/Flux_Tower/2018_NEP_GPP_Borden.csv', index_col=0)

#Extract Borden flux data into arrays:
Borden_dates_fluxes=np.zeros([17520])*np.nan
Borden_NEEgf_fluxes=np.zeros([17520])*np.nan
Borden_NEE_fluxes=np.zeros([17520])*np.nan
Borden_R_fluxes=np.zeros([17520])*np.nan
Borden_Rgf_fluxes=np.zeros([17520])*np.nan
Borden_GEP_fluxes=np.zeros([17520])*np.nan
Borden_GEPgf_fluxes=np.zeros([17520])*np.nan
n=0
m=0
date=1
for i in range(0,17520):
    Borden_dates_fluxes[i]=Borden_Fluxes.iat[i,0]-5/24 #adjust to local time
    Borden_NEEgf_fluxes[i]=-Borden_Fluxes.iat[i,5] # NEE (gap filled)
    Borden_NEE_fluxes[i]=-Borden_Fluxes.iat[i,1] # NEE (non-gap filled)
    Borden_Rgf_fluxes[i]=Borden_Fluxes.iat[i,6]
    Borden_R_fluxes[i]=Borden_Fluxes.iat[i,3]
    Borden_GEP_fluxes[i]=Borden_Fluxes.iat[i,4]
    Borden_GEPgf_fluxes[i]=Borden_Fluxes.iat[i,7]
    
del Borden_Fluxes #Remove remaining data to save memory


# In[7]:


# Convert half-hourly flux tower data to hourly averages to match resolution of SMUrF 
# Average hour and the next half hour period
Borden_NEE=np.zeros(np.shape(C_GPP_array))*np.nan
Borden_GEPgf=np.zeros(np.shape(C_GPP_array))*np.nan
Borden_NEEgf=np.zeros(np.shape(C_GPP_array))*np.nan
Borden_Rgf=np.zeros(np.shape(C_GPP_array))*np.nan
for i in range(np.int(len(Borden_dates_fluxes)/2)):
    Borden_NEE[i]=np.nanmean([Borden_NEE_fluxes[i*2],Borden_NEE_fluxes[i*2+1]])
    Borden_GEPgf[i]=np.nanmean([Borden_GEPgf_fluxes[i*2],Borden_GEPgf_fluxes[i*2+1]])
    Borden_NEEgf[i]=np.nanmean([Borden_NEEgf_fluxes[i*2],Borden_NEEgf_fluxes[i*2+1]])
    Borden_Rgf[i]=np.nanmean([Borden_Rgf_fluxes[i*2],Borden_Rgf_fluxes[i*2+1]])


# In[9]:


#Load in SMUrF data and crop to flux tower locations for 2018:

S_time=[]
S_lats=[]
S_lons=[]

S_Reco_Borden=[]
S_Reco_std_Borden=[]
S_NEE_Borden=[]
S_NEE_std_Borden=[]
S_GPP_Borden=[]
S_GPP_std_Borden=[]

S_Reco_TP39=[]
S_Reco_std_TP39=[]
S_NEE_TP39=[]
S_NEE_std_TP39=[]
S_GPP_TP39=[]
S_GPP_std_TP39=[]

S_Reco_TPD=[]
S_Reco_std_TPD=[]
S_NEE_TPD=[]
S_NEE_std_TPD=[]
S_GPP_TPD=[]
S_GPP_std_TPD=[]

# *** CHANGE PATH ***
S_path = 'C:/Users/kitty/Documents/Research/SIF/SMUrF/output2018_500m_CSIF_to_TROPOMI_CSIF_ALL_converted_slps_V3_temp_impervious_R_shore_corr_V061_8day/easternCONUS/hourly_flux_GMIS_combined_ISA_a_w_sd_era5/'
# *** CHANGE FILENAME ***
S_fn = 'hrly_mean_GPP_Reco_NEE_easternCONUS_2018' # filename (without the month)

for j in range(1,13):
    try:
        #load in the data
        if j<10:
            f=Dataset(S_path+S_fn+'0'+str(j)+'.nc')
        else:
            f=Dataset(S_path+S_fn+str(j)+'.nc')
        
        S_Reco=f.variables['Reco_mean'][:]
        S_GPP=f.variables['GPP_mean'][:]
        S_NEE=f.variables['NEE_mean'][:]
        
        if len(S_time)==0:
            # If it is the first file start an array for each variable and save lat/lon & fluxes
            S_lats=f.variables['lat'][:]
            S_lons=f.variables['lon'][:]
            S_time=f.variables['time'][:]/24/3600-start_of_year-5/24 #convert seconds since 1970 to days and subtract start of year and adjust to local time
                        
            S_GPP_Borden = np.nanmean([S_GPP[:,458,230],S_GPP[:,458,231],S_GPP[:,458,232],S_GPP[:,459,230],S_GPP[:,459,231],S_GPP[:,459,232],S_GPP[:,460,232]],axis=0)
            S_GPP_std_Borden = np.nanstd([S_GPP[:,458,230],S_GPP[:,458,231],S_GPP[:,458,232],S_GPP[:,459,230],S_GPP[:,459,231],S_GPP[:,459,232],S_GPP[:,460,232]],axis=0)
            S_Reco_Borden = np.nanmean([S_Reco[:,458,230],S_Reco[:,458,231],S_Reco[:,458,232],S_Reco[:,459,230],S_Reco[:,459,231],S_Reco[:,459,232],S_Reco[:,460,232]],axis=0)
            S_Reco_std_Borden = np.nanstd([S_Reco[:,458,230],S_Reco[:,458,231],S_Reco[:,458,232],S_Reco[:,459,230],S_Reco[:,459,231],S_Reco[:,459,232],S_Reco[:,460,232]],axis=0)
            S_NEE_Borden = np.nanmean([S_NEE[:,458,230],S_NEE[:,458,231],S_NEE[:,458,232],S_NEE[:,459,230],S_NEE[:,459,231],S_NEE[:,459,232],S_NEE[:,460,232]],axis=0)
            S_NEE_std_Borden = np.nanstd([S_NEE[:,458,230],S_NEE[:,458,231],S_NEE[:,458,232],S_NEE[:,459,230],S_NEE[:,459,231],S_NEE[:,459,232],S_NEE[:,460,232]],axis=0)

            S_GPP_TP39 = np.nanmean(S_GPP[:,73:75,129:131],axis=(1,2))
            S_GPP_std_TP39 = np.nanstd(S_GPP[:,73:75,129:131],axis=(1,2))
            S_Reco_TP39 = np.nanmean(S_Reco[:,73:75,129:131],axis=(1,2))
            S_Reco_std_TP39 = np.nanstd(S_Reco[:,73:75,129:131],axis=(1,2))
            S_NEE_TP39 = np.nanmean(S_NEE[:,73:75,129:131],axis=(1,2))
            S_NEE_std_TP39 = np.nanstd(S_NEE[:,73:75,129:131],axis=(1,2))
            
            S_GPP_TPD = np.nanmean(S_GPP[:,55:58,80:83],axis=(1,2))
            S_GPP_std_TPD = np.nanstd(S_GPP[:,55:58,80:83],axis=(1,2))
            S_Reco_TPD = np.nanmean(S_Reco[:,55:58,80:83],axis=(1,2))
            S_Reco_std_TPD = np.nanstd(S_Reco[:,55:58,80:83],axis=(1,2))
            S_NEE_TPD = np.nanmean(S_NEE[:,55:58,80:83],axis=(1,2))
            S_NEE_std_TPD = np.nanstd(S_NEE[:,55:58,80:83],axis=(1,2))
            
        else:
            #Otherwise append fluxes to the array
            S_GPP_Borden = np.concatenate((S_GPP_Borden,np.nanmean([S_GPP[:,458,230],S_GPP[:,458,231],S_GPP[:,458,232],S_GPP[:,459,230],S_GPP[:,459,231],S_GPP[:,459,232],S_GPP[:,460,232]],axis=0)),axis=0)
            S_GPP_std_Borden = np.concatenate((S_GPP_std_Borden,np.nanstd([S_GPP[:,458,230],S_GPP[:,458,231],S_GPP[:,458,232],S_GPP[:,459,230],S_GPP[:,459,231],S_GPP[:,459,232],S_GPP[:,460,232]],axis=0)),axis=0)
            S_Reco_Borden = np.concatenate((S_Reco_Borden,np.nanmean([S_Reco[:,458,230],S_Reco[:,458,231],S_Reco[:,458,232],S_Reco[:,459,230],S_Reco[:,459,231],S_Reco[:,459,232],S_Reco[:,460,232]],axis=0)),axis=0)
            S_Reco_std_Borden = np.concatenate((S_Reco_std_Borden,np.nanstd([S_Reco[:,458,230],S_Reco[:,458,231],S_Reco[:,458,232],S_Reco[:,459,230],S_Reco[:,459,231],S_Reco[:,459,232],S_Reco[:,460,232]],axis=0)),axis=0)
            S_NEE_Borden = np.concatenate((S_NEE_Borden,np.nanmean([S_NEE[:,458,230],S_NEE[:,458,231],S_NEE[:,458,232],S_NEE[:,459,230],S_NEE[:,459,231],S_NEE[:,459,232],S_NEE[:,460,232]],axis=0)),axis=0)
            S_NEE_std_Borden = np.concatenate((S_NEE_std_Borden,np.nanstd([S_NEE[:,458,230],S_NEE[:,458,231],S_NEE[:,458,232],S_NEE[:,459,230],S_NEE[:,459,231],S_NEE[:,459,232],S_NEE[:,460,232]],axis=0)),axis=0)

            S_GPP_TP39 = np.concatenate((S_GPP_TP39,np.nanmean(S_GPP[:,73:75,129:131],axis=(1,2))),axis=0)
            S_GPP_std_TP39 = np.concatenate((S_GPP_std_TP39,np.nanstd(S_GPP[:,73:75,129:131],axis=(1,2))),axis=0)
            S_Reco_TP39 = np.concatenate((S_Reco_TP39,np.nanmean(S_Reco[:,73:75,129:131],axis=(1,2))),axis=0)
            S_Reco_std_TP39 = np.concatenate((S_Reco_std_TP39,np.nanstd(S_Reco[:,73:75,129:131],axis=(1,2))),axis=0)
            S_NEE_TP39 = np.concatenate((S_NEE_TP39,np.nanmean(S_NEE[:,73:75,129:131],axis=(1,2))),axis=0)
            S_NEE_std_TP39 = np.concatenate((S_NEE_std_TP39,np.nanstd(S_NEE[:,73:75,129:131],axis=(1,2))),axis=0)

            S_GPP_TPD = np.concatenate((S_GPP_TPD,np.nanmean(S_GPP[:,55:58,80:83],axis=(1,2))),axis=0)
            S_GPP_std_TPD = np.concatenate((S_GPP_std_TPD,np.nanstd(S_GPP[:,55:58,80:83],axis=(1,2))),axis=0)
            S_Reco_TPD = np.concatenate((S_Reco_TPD,np.nanmean(S_Reco[:,55:58,80:83],axis=(1,2))),axis=0)
            S_Reco_std_TPD = np.concatenate((S_Reco_std_TPD,np.nanstd(S_Reco[:,55:58,80:83],axis=(1,2))),axis=0)
            S_NEE_TPD = np.concatenate((S_NEE_TPD,np.nanmean(S_NEE[:,55:58,80:83],axis=(1,2))),axis=0)
            S_NEE_std_TPD = np.concatenate((S_NEE_std_TPD,np.nanstd(S_NEE[:,55:58,80:83],axis=(1,2))),axis=0)

            S_time=np.concatenate((S_time,(f.variables['time'][:]/24/3600-start_of_year-5/24)),axis=0)
        f.close()
    except FileNotFoundError:
        pass
    
del(S_GPP,S_Reco,S_NEE)

S_GPP_Borden = np.concatenate((S_GPP_Borden,np.ones(5)*np.nan),axis=0)
S_GPP_std_Borden = np.concatenate((S_GPP_std_Borden,np.ones(5)*np.nan),axis=0)
S_Reco_Borden = np.concatenate((S_Reco_Borden,np.ones(5)*np.nan),axis=0)
S_Reco_std_Borden = np.concatenate((S_Reco_std_Borden,np.ones(5)*np.nan),axis=0)
S_NEE_Borden = np.concatenate((S_NEE_Borden,np.ones(5)*np.nan),axis=0)
S_NEE_std_Borden = np.concatenate((S_NEE_std_Borden,np.ones(5)*np.nan),axis=0)

S_GPP_TP39 = np.concatenate((S_GPP_TP39,np.ones(5)*np.nan),axis=0)
S_GPP_std_TP39 = np.concatenate((S_GPP_std_TP39,np.ones(5)*np.nan),axis=0)
S_Reco_TP39 = np.concatenate((S_Reco_TP39,np.ones(5)*np.nan),axis=0)
S_Reco_std_TP39 = np.concatenate((S_Reco_std_TP39,np.ones(5)*np.nan),axis=0)
S_NEE_TP39 = np.concatenate((S_NEE_TP39,np.ones(5)*np.nan),axis=0)
S_NEE_std_TP39 = np.concatenate((S_NEE_std_TP39,np.ones(5)*np.nan),axis=0)

S_GPP_TPD = np.concatenate((S_GPP_TPD,np.ones(5)*np.nan),axis=0)
S_GPP_std_TPD = np.concatenate((S_GPP_std_TPD,np.ones(5)*np.nan),axis=0)
S_Reco_TPD = np.concatenate((S_Reco_TPD,np.ones(5)*np.nan),axis=0)
S_Reco_std_TPD = np.concatenate((S_Reco_std_TPD,np.ones(5)*np.nan),axis=0)
S_NEE_TPD = np.concatenate((S_NEE_TPD,np.ones(5)*np.nan),axis=0)
S_NEE_std_TPD = np.concatenate((S_NEE_std_TPD,np.ones(5)*np.nan),axis=0)

S_time=np.concatenate((S_time,np.ones(5)*np.nan),axis=0)


# In[ ]:





# In[9]:


# *** Optionally (uncomment) save the data over fluxtowers for faster loading in the future ***

#g = Dataset('E:/Research/SMUrF/output2018_500m_CSIF_to_TROPOMI_CSIF_ALL_converted_slps_V3_temp_impervious_R_shore_corr_V061_8day/SMUrF_Borden_fluxes.nc','w', format='NETCDF4')
##asif,sweight_tot,aweight_grid=aggregate(OCOf.variables['lon'][:],OCOf.variables['lat'][:], 0.05, f.variables['lon'][:],f.variables['lat'][:],f.variables['daily_sif'][:],step)
#g.createDimension('time',len(S_time_array))

## define variables to save in the file
#NEE = g.createVariable('NEE',np.float32,'time')
#NEE_std = g.createVariable('NEE_std',np.float32,'time')
#GPP = g.createVariable('GPP',np.float32,'time')
#GPP_std = g.createVariable('GPP_std',np.float32,'time')
#Reco = g.createVariable('Reco',np.float32,'time')
#Reco_std = g.createVariable('Reco_std',np.float32,'time')
#t = g.createVariable('time',np.float32,'time')

#NEE[:]=S_NEE_Borden
#NEE_std[:]=S_NEE_std_Borden
#GPP[:]=S_GPP_Borden
#GPP_std[:]=S_GPP_std_Borden
#Reco[:]=S_Reco_Borden
#Reco_std[:]=S_Reco_std_Borden
#t[:] = S_time
##sif_error[:,:]=np.array(f.variables['Errors'][::-1])

##close the file
#g.close()


# In[10]:


#Load in SMUrF data and crop to flux tower locations for 2019:

S_time=[]
S_lats=[]
S_lons=[]

S_Reco_TP39_2019=[]
S_Reco_std_TP39_2019=[]
S_NEE_TP39_2019=[]
S_NEE_std_TP39_2019=[]
S_GPP_TP39_2019=[]
S_GPP_std_TP39_2019=[]

S_Reco_TPD_2019=[]
S_Reco_std_TPD_2019=[]
S_NEE_TPD_2019=[]
S_NEE_std_TPD_2019=[]
S_GPP_TPD_2019=[]
S_GPP_std_TPD_2019=[]

# *** CHANGE PATH ***
S_path = 'E:/Research/SMUrF/output2019_500m_CSIF_to_TROPOMI_CSIF_ALL_converted_slps_V3_temp_impervious_R_shore_corr_V061_8day/easternCONUS/hourly_flux_GMIS_Toronto_fixed_border_ISA_a_w_sd_era5/'
# *** CHANGE FILENAME ***
S_fn = 'hrly_mean_GPP_Reco_NEE_easternCONUS_2019' # filename (without the month)

for j in range(1,13):
    try:
        #load in the data
        if j<10:
            f=Dataset(S_path+S_fn+'0'+str(j)+'.nc')
        else:
            f=Dataset(S_path+S_fn+str(j)+'.nc')
        
        S_Reco=f.variables['Reco_mean'][:]
        S_GPP=f.variables['GPP_mean'][:]
        S_NEE=f.variables['NEE_mean'][:]
        
        if len(S_time)==0:
            # If it is the first file start an array for each variable and save lat/lon
            S_lats=f.variables['lat'][:]
            S_lons=f.variables['lon'][:]
            S_time=f.variables['time'][:]/24/3600-start_of_year-5/24 #convert seconds since 1970 to days and subtract start of year and adjust to local time
                        
            S_GPP_TP39_2019 = np.nanmean(S_GPP[:,73:75,129:131],axis=(1,2))
            S_GPP_std_TP39_2019 = np.nanstd(S_GPP[:,73:75,129:131],axis=(1,2))
            S_Reco_TP39_2019 = np.nanmean(S_Reco[:,73:75,129:131],axis=(1,2))
            S_Reco_std_TP39_2019 = np.nanstd(S_Reco[:,73:75,129:131],axis=(1,2))
            S_NEE_TP39_2019 = np.nanmean(S_NEE[:,73:75,129:131],axis=(1,2))
            S_NEE_std_TP39_2019 = np.nanstd(S_NEE[:,73:75,129:131],axis=(1,2))
            
            S_GPP_TPD_2019 = np.nanmean(S_GPP[:,55:58,80:83],axis=(1,2))
            S_GPP_std_TPD_2019 = np.nanstd(S_GPP[:,55:58,80:83],axis=(1,2))
            S_Reco_TPD_2019 = np.nanmean(S_Reco[:,55:58,80:83],axis=(1,2))
            S_Reco_std_TPD_2019 = np.nanstd(S_Reco[:,55:58,80:83],axis=(1,2))
            S_NEE_TPD_2019 = np.nanmean(S_NEE[:,55:58,80:83],axis=(1,2))
            S_NEE_std_TPD_2019 = np.nanstd(S_NEE[:,55:58,80:83],axis=(1,2))
            
        else:
            #Otherwise append to the array
            S_GPP_TP39_2019 = np.concatenate((S_GPP_TP39_2019,np.nanmean(S_GPP[:,73:75,129:131],axis=(1,2))),axis=0)
            S_GPP_std_TP39_2019 = np.concatenate((S_GPP_std_TP39_2019,np.nanstd(S_GPP[:,73:75,129:131],axis=(1,2))),axis=0)
            S_Reco_TP39_2019 = np.concatenate((S_Reco_TP39_2019,np.nanmean(S_Reco[:,73:75,129:131],axis=(1,2))),axis=0)
            S_Reco_std_TP39_2019 = np.concatenate((S_Reco_std_TP39_2019,np.nanstd(S_Reco[:,73:75,129:131],axis=(1,2))),axis=0)
            S_NEE_TP39_2019 = np.concatenate((S_NEE_TP39_2019,np.nanmean(S_NEE[:,73:75,129:131],axis=(1,2))),axis=0)
            S_NEE_std_TP39_2019 = np.concatenate((S_NEE_std_TP39_2019,np.nanstd(S_NEE[:,73:75,129:131],axis=(1,2))),axis=0)

            S_GPP_TPD_2019 = np.concatenate((S_GPP_TPD_2019,np.nanmean(S_GPP[:,55:58,80:83],axis=(1,2))),axis=0)
            S_GPP_std_TPD_2019 = np.concatenate((S_GPP_std_TPD_2019,np.nanstd(S_GPP[:,55:58,80:83],axis=(1,2))),axis=0)
            S_Reco_TPD_2019 = np.concatenate((S_Reco_TPD_2019,np.nanmean(S_Reco[:,55:58,80:83],axis=(1,2))),axis=0)
            S_Reco_std_TPD_2019 = np.concatenate((S_Reco_std_TPD_2019,np.nanstd(S_Reco[:,55:58,80:83],axis=(1,2))),axis=0)
            S_NEE_TPD_2019 = np.concatenate((S_NEE_TPD_2019,np.nanmean(S_NEE[:,55:58,80:83],axis=(1,2))),axis=0)
            S_NEE_std_TPD_2019 = np.concatenate((S_NEE_std_TPD_2019,np.nanstd(S_NEE[:,55:58,80:83],axis=(1,2))),axis=0)

            S_time=np.concatenate((S_time,(f.variables['time'][:]/24/3600-start_of_year-5/24)),axis=0)
        f.close()
    except FileNotFoundError:
        pass
    
del(S_GPP,S_Reco,S_NEE)

S_GPP_TP39_2019 = np.concatenate((S_GPP_TP39_2019,np.ones(5)*np.nan),axis=0)
S_GPP_std_TP39_2019 = np.concatenate((S_GPP_std_TP39_2019,np.ones(5)*np.nan),axis=0)
S_Reco_TP39_2019 = np.concatenate((S_Reco_TP39_2019,np.ones(5)*np.nan),axis=0)
S_Reco_std_TP39_2019 = np.concatenate((S_Reco_std_TP39_2019,np.ones(5)*np.nan),axis=0)
S_NEE_TP39_2019 = np.concatenate((S_NEE_TP39_2019,np.ones(5)*np.nan),axis=0)
S_NEE_std_TP39_2019 = np.concatenate((S_NEE_std_TP39_2019,np.ones(5)*np.nan),axis=0)

S_GPP_TPD_2019 = np.concatenate((S_GPP_TPD_2019,np.ones(5)*np.nan),axis=0)
S_GPP_std_TPD_2019 = np.concatenate((S_GPP_std_TPD_2019,np.ones(5)*np.nan),axis=0)
S_Reco_TPD_2019 = np.concatenate((S_Reco_TPD_2019,np.ones(5)*np.nan),axis=0)
S_Reco_std_TPD_2019 = np.concatenate((S_Reco_std_TPD_2019,np.ones(5)*np.nan),axis=0)
S_NEE_TPD_2019 = np.concatenate((S_NEE_TPD_2019,np.ones(5)*np.nan),axis=0)
S_NEE_std_TPD_2019 = np.concatenate((S_NEE_std_TPD_2019,np.ones(5)*np.nan),axis=0)

S_time=np.concatenate((S_time,np.ones(5)*np.nan),axis=0)


# In[ ]:





# In[11]:


# *** If you have already saved the SMUrF data over flux towers uncomment this (faster): ***

#g = Dataset('E:/Research/SMUrF/output2018_500m_CSIF_to_TROPOMI_CSIF_ALL_converted_slps_temp_impervious_R_V061_8day/SMUrF_Borden_fluxes.nc')
#S_NEE_Borden=g.variables['NEE'][:]
#S_NEE_std_Borden = g.variables['NEE_std'][:]
#S_GPP_Borden=g.variables['GPP'][:]
#S_GPP_std_Borden = g.variables['GPP_std'][:]
#S_Reco_Borden=g.variables['Reco'][:]
#S_Reco_std_Borden = g.variables['Reco_std'][:]
#S_time=g.variables['time'][:]
#g.close()

#g = Dataset('E:/Research/SMUrF/output2018_500m_CSIF_to_TROPOMI_CSIF_ALL_converted_slps_temp_impervious_R_V061_8day/SMUrF_TP39_fluxes.nc')
#S_NEE_TP39=g.variables['NEE'][:]
#S_NEE_std_TP39 = g.variables['NEE_std'][:]
#S_GPP_TP39=g.variables['GPP'][:]
#S_GPP_std_TP39 = g.variables['GPP_std'][:]
#S_Reco_TP39=g.variables['Reco'][:]
#S_Reco_std_TP39 = g.variables['Reco_std'][:]
#S_time=g.variables['time'][:]
#g.close()

#g = Dataset('E:/Research/SMUrF/output2018_500m_CSIF_to_TROPOMI_CSIF_ALL_converted_slps_temp_impervious_R_V061_8day/SMUrF_TPD_fluxes_2018.nc')
#S_NEE_TPD=g.variables['NEE'][:]
#S_NEE_std_TPD = g.variables['NEE_std'][:]
#S_GPP_TPD=g.variables['GPP'][:]
#S_GPP_std_TPD = g.variables['GPP_std'][:]
#S_Reco_TPD=g.variables['Reco'][:]
#S_Reco_std_TPD = g.variables['Reco_std'][:]
#S_time=g.variables['time'][:]
#g.close()

#g = Dataset('E:/Research/SMUrF/output2019_500m_CSIF_to_TROPOMI_CSIF_ALL_converted_slps_temp_impervious_R_V061_8day/SMUrF_Borden_fluxes.nc')
#S_NEE_Borden_2019=g.variables['NEE'][:]
#S_NEE_2019_std_Borden = g.variables['NEE_std'][:]
#S_GPP_Borden_2019=g.variables['GPP'][:]
#S_GPP_2019_std_Borden = g.variables['GPP_std'][:]
#S_Reco_Borden_2019=g.variables['Reco'][:]
#S_Reco_2019_std_Borden = g.variables['Reco_std'][:]
#S_time=g.variables['time'][:]
#g.close()

#g = Dataset('E:/Research/SMUrF/output2019_500m_CSIF_to_TROPOMI_CSIF_ALL_converted_slps_temp_impervious_R_V061_8day/SMUrF_TP39_fluxes.nc')
#S_NEE_TP39_2019=g.variables['NEE'][:]
#S_NEE_2019_std_TP39 = g.variables['NEE_std'][:]
#S_GPP_TP39_2019=g.variables['GPP'][:]
#S_GPP_2019_std_TP39 = g.variables['GPP_std'][:]
#S_Reco_TP39_2019=g.variables['Reco'][:]
#S_Reco_2019_std_TP39 = g.variables['Reco_std'][:]
#S_time=g.variables['time'][:]
#g.close()

#g = Dataset('E:/Research/SMUrF/output2019_500m_CSIF_to_TROPOMI_CSIF_ALL_converted_slps_temp_impervious_R_V061_8day/SMUrF_TPD_fluxes_2019.nc')
#S_NEE_TPD_2019=g.variables['NEE'][:]
#S_NEE_2019_std_TPD = g.variables['NEE_std'][:]
#S_GPP_TPD_2019=g.variables['GPP'][:]
#S_GPP_2019_std_TPD = g.variables['GPP_std'][:]
#S_Reco_TPD_2019=g.variables['Reco'][:]
#S_Reco_2019_std_TPD = g.variables['Reco_std'][:]
#S_time=g.variables['time'][:]
#g.close()


# In[ ]:





# In[11]:


# Isolate the spring (March, April, May) data
#MAM: Day of year 60 - 151 inclusive

with np.errstate(invalid='ignore'):
    MAM_time=C_time_array[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    Borden_GPPgf_MAM=Borden_GEPgf[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    C_GPP_Borden_MAM=C_GPP_array[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    S_GPP_Borden_MAM=S_GPP_Borden[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]

    Borden_Rgf_MAM=Borden_Rgf[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    C_Reco_Borden_MAM=C_Reco_array[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    S_Reco_Borden_MAM=S_Reco_Borden[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]

    Borden_NEEgf_MAM=Borden_NEEgf[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    Borden_NEE_MAM=Borden_NEE[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    C_NEE_Borden_MAM=C_NEE_array[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    S_NEE_Borden_MAM=S_NEE_Borden[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]


# In[ ]:





# In[12]:


# Import 2019 Original SMUrF (using CSIF)

# *** CHANGE PATH & FILENAME ***
g=Dataset('E:/Research/SMUrF/output2019_CSIF_V061/easternCONUS/daily_mean_Reco_neuralnet/era5/2019/daily_mean_Reco_uncert_easternCONUS_20190101.nc')
start_of_year_2019=g.variables['time'][0]/3600/24-1 #convert seconds since 1970 to days (minus one)
g.close()

# *** CHANGE PATH ***
C_path = 'E:/Research/SMUrF/output2019_CSIF_V061/easternCONUS/hourly_flux_era5/'
# *** CHANGE FILENAME ***
C_fn = 'hrly_mean_GPP_Reco_NEE_easternCONUS_2019' #filename (without month)

C_time_2019=[]
C_Reco_2019=[]
C_NEE_2019=[]
C_GPP_2019=[]
C_lats_2019=[]
C_lons_2019=[]
for j in range(1,13):
    try:
        if j<10:
            # 2019:
            f=Dataset(C_path+C_fn+'0'+str(j)+'.nc')
        else:
            # 2019:
            f=Dataset(C_path+C_fn+str(j)+'.nc')
        if len(C_time_2019)==0:
            C_lats_2019=f.variables['lat'][:]
            C_lons_2019=f.variables['lon'][:]
            C_Reco_2019=f.variables['Reco_mean'][:]
            C_GPP_2019=f.variables['GPP_mean'][:]
            C_NEE_2019=f.variables['NEE_mean'][:]
            C_time_2019=f.variables['time'][:]/24/3600-start_of_year_2019-5/24 #convert seconds since 1970 to days and subtract start of year and adjust to local time
        else:
            C_Reco_2019=np.concatenate((C_Reco_2019,f.variables['Reco_mean'][:]),axis=0)
            C_GPP_2019=np.concatenate((C_GPP_2019,f.variables['GPP_mean'][:]),axis=0)
            C_NEE_2019=np.concatenate((C_NEE_2019,f.variables['NEE_mean'][:]),axis=0)
            C_time_2019=np.concatenate((C_time_2019,(f.variables['time'][:]/24/3600-start_of_year_2019-5/24)),axis=0)
        f.close()
    except FileNotFoundError:
        pass

C_Reco_2019[C_Reco_2019==-999]=np.nan
C_NEE_2019[C_NEE_2019==-999]=np.nan
C_GPP_2019[C_GPP_2019==-999]=np.nan


# In[13]:


#Crop CSIF-SMUrF data over TP39

C_GPP_TP39_array=np.zeros(8765)*np.nan
C_NEE_TP39_array=np.zeros(8765)*np.nan
C_Reco_TP39_array=np.zeros(8765)*np.nan

for i in range(len(C_GPP[:,0,0])):
    C_GPP_TP39_array[i]=C_GPP[i,4,6]
    C_NEE_TP39_array[i]=C_NEE[i,4,6]
    C_Reco_TP39_array[i]=C_Reco[i,4,6]


# In[14]:


# Import 2018 TP39 flux tower data & average to hourly resolution

# *** CHANGE PATH & FILENAME ***
TP39_2018_data=pd.read_csv('C:/Users/kitty/Documents/Research/SIF/Flux_Tower/Turkey_Point/TP39_HH_2018.csv', usecols=[0,1,2,6,77,78,79]) #header=1

TP39_2018_dates=np.zeros([17520])*np.nan
TP39_2018_NEE=np.zeros([17520])*np.nan #gapfilled NEE
TP39_2018_NEE2=np.zeros([17520])*np.nan #non-gapfilled NEE
TP39_2018_GPP=np.zeros([17520])*np.nan
TP39_2018_R=np.zeros([17520])*np.nan
n=0
m=0
#date=1
for i in range(17520):
    if 201801010000<=TP39_2018_data.iat[i,0]<202001010000:
        TP39_2018_dates[i]=datetime.strptime(str(int(TP39_2018_data.iat[i,0])),'%Y%m%d%H%M').timetuple().tm_yday+(float(str(TP39_2018_data.iat[i,0])[8:10])+float(str(TP39_2018_data.iat[i,0])[10:12])/60)/24
        #check that the value is greater than -9999 (value for empty measurements)
        if TP39_2018_data.iat[i,2]>-9999:
            TP39_2018_NEE2[i]=TP39_2018_data.iat[i,2] # save the non-gapfilled NEE value
        if TP39_2018_data.iat[i,6]>-9999:
            TP39_2018_NEE[i]=TP39_2018_data.iat[i,6] # save the gapfilled NEE value
        if TP39_2018_data.iat[i,4]>-9999:
            TP39_2018_GPP[i]=TP39_2018_data.iat[i,4] # save the GPP value
        if TP39_2018_data.iat[i,5]>-9999:
            TP39_2018_R[i]=TP39_2018_data.iat[i,5] # save the Reco value

TP39_GPP=np.zeros(np.shape(C_GPP_TP39_array))*np.nan
TP39_NEEgf=np.zeros(np.shape(C_GPP_TP39_array))*np.nan
TP39_NEE=np.zeros(np.shape(C_GPP_TP39_array))*np.nan
TP39_R=np.zeros(np.shape(C_GPP_TP39_array))*np.nan
for i in range(len(C_time_array)-5):
    with np.errstate(invalid='ignore'):
        TP39_GPP[i+5]=np.nanmean([TP39_2018_GPP[i*2],TP39_2018_GPP[i*2+1]])
        TP39_NEEgf[i+5]=np.nanmean([TP39_2018_NEE[i*2],TP39_2018_NEE[i*2+1]])
        TP39_NEE[i+5]=np.nanmean([TP39_2018_NEE2[i*2],TP39_2018_NEE2[i*2+1]])
        TP39_R[i+5]=np.nanmean([TP39_2018_R[i*2],TP39_2018_R[i*2+1]])


# In[ ]:





# In[15]:


#Select spring 2018 TP39 data

#MAM: Doy 60 - 151 inclusive

with np.errstate(invalid='ignore'):
    TP39_GPP_MAM=TP39_GPP[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    C_GPP_TP39_MAM=C_GPP_TP39_array[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    S_GPP_TP39_MAM=S_GPP_TP39[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]

    TP39_R_MAM=TP39_R[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    C_Reco_TP39_MAM=C_Reco_TP39_array[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    S_Reco_TP39_MAM=S_Reco_TP39[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]

    TP39_NEEgf_MAM=TP39_NEEgf[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    TP39_NEE_MAM=TP39_NEE[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    C_NEE_TP39_MAM=C_NEE_TP39_array[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    S_NEE_TP39_MAM=S_NEE_TP39[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]


# In[16]:


# load in 2019 TP39 data & average to hourly resolution

# *** CHANGE PATH & FILENAME ***
TP39_2019_data=pd.read_csv('C:/Users/kitty/Documents/Research/SIF/Flux_Tower/Turkey_Point/TP39_HH_2019.csv', usecols=[0,1,2,6,77,78,79]) #header=1

TP39_2019_dates=np.zeros([17520])*np.nan
TP39_2019_dates2=np.zeros([17520])*np.nan
TP39_2019_NEE=np.zeros([17520])*np.nan
TP39_2019_NEE2=np.zeros([17520])*np.nan
TP39_2019_GPP=np.zeros([17520])*np.nan
TP39_2019_R=np.zeros([17520])*np.nan
n=0
m=0
date=1
for i in range(17520):
    if 201901010000<=TP39_2019_data.iat[i,0]<202001010000:
        TP39_2019_dates[i]=datetime.strptime(str(int(TP39_2019_data.iat[i,0])),'%Y%m%d%H%M').timetuple().tm_yday+(float(str(TP39_2019_data.iat[i,0])[8:10])+float(str(TP39_2019_data.iat[i,0])[10:12])/60)/24

        #check that the value is greater than -9999 (value for empty measurements)
        if TP39_2019_data.iat[i,2]>-9999:
            TP39_2019_NEE2[i]=TP39_2019_data.iat[i,2] # save the non-gapfilled NEE value
        if TP39_2019_data.iat[i,6]>-9999:
            TP39_2019_NEE[i]=TP39_2019_data.iat[i,6] # save the gapfilled NEE value
        if TP39_2019_data.iat[i,4]>-9999:
            TP39_2019_GPP[i]=TP39_2019_data.iat[i,4] # save the GPP value
        if TP39_2019_data.iat[i,5]>-9999:
            TP39_2019_R[i]=TP39_2019_data.iat[i,5] # save the Reco value
        
# Select Original SMUrF data over TP39
C_GPP_TP39_2019_array=np.zeros(8765)*np.nan
C_NEE_TP39_2019_array=np.zeros(8765)*np.nan
C_Reco_TP39_2019_array=np.zeros(8765)*np.nan
C_time_2019_array=np.zeros(8765)*np.nan
for i in range(len(C_GPP_2019[:,0,0])):
    C_time_2019_array[i]=C_time[i]
    C_GPP_TP39_2019_array[i]=C_GPP_2019[i,4,6]
    C_NEE_TP39_2019_array[i]=C_NEE_2019[i,4,6]
    C_Reco_TP39_2019_array[i]=C_Reco_2019[i,4,6]

TP39_2019_hrly_GPP=np.zeros(np.shape(C_GPP_TP39_2019_array))*np.nan
TP39_2019_hrly_NEEgf=np.zeros(np.shape(C_GPP_TP39_2019_array))*np.nan
TP39_2019_hrly_NEE=np.zeros(np.shape(C_GPP_TP39_2019_array))*np.nan
TP39_2019_hrly_R=np.zeros(np.shape(C_GPP_TP39_2019_array))*np.nan
for i in range(len(C_time_array)-5):
    with np.errstate(invalid='ignore'):
        TP39_2019_hrly_GPP[i+5]=np.nanmean([TP39_2019_GPP[i*2],TP39_2019_GPP[i*2+1]])
        TP39_2019_hrly_NEEgf[i+5]=np.nanmean([TP39_2019_NEE[i*2],TP39_2019_NEE[i*2+1]])
        TP39_2019_hrly_NEE[i+5]=np.nanmean([TP39_2019_NEE2[i*2],TP39_2019_NEE2[i*2+1]])
        TP39_2019_hrly_R[i+5]=np.nanmean([TP39_2019_R[i*2],TP39_2019_R[i*2+1]])


# In[17]:


#Filter out erroneous NEE values between doy 195 and 198

## ***Optional: uncomment to visualize erroneous values
#with np.errstate(invalid='ignore'):
#    plt.figure()
#    plt.xlim(193,200)
#    plt.scatter(C_time_array,TP39_2019_hrly_NEE,label='TP39 NEE')
#    plt.scatter(C_time_array[(C_time_array>195.05-5/24) & (C_time_array<198.6-5/24)],TP39_2019_hrly_NEE[(C_time_array>195.05-5/24) & (C_time_array<198.6-5/24)],label='Erroneous TP39 NEE')
#    plt.scatter(C_time_array,S_NEE_TP39_2019,marker='*',label='SMUrF NEE')
#    plt.legend()
#    plt.xlabel('Day of year, 2019')
#    plt.ylabel('NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
#    plt.title('Erroneous TP39 flux tower NEE values')
#    plt.show()
# ***

with np.errstate(invalid='ignore'):
    TP39_2019_hrly_NEE[(C_time_array>195.05-5/24) & (C_time_array<198.6-5/24)]= np.nan


# In[ ]:





# In[18]:


# Select spring data over TP39

##MAM: Doy 60 - 151 inclusive
with np.errstate(invalid='ignore'):
    TP39_2019_GPP_MAM=TP39_2019_hrly_GPP[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    C_GPP_TP39_2019_MAM=C_GPP_TP39_2019_array[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    S_GPP_TP39_2019_MAM=S_GPP_TP39_2019[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]

    TP39_2019_R_MAM=TP39_2019_hrly_R[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    C_Reco_TP39_2019_MAM=C_Reco_TP39_2019_array[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    S_Reco_TP39_2019_MAM=S_Reco_TP39_2019[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]

    TP39_2019_NEEgf_MAM=TP39_2019_hrly_NEEgf[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    TP39_2019_NEE_MAM=TP39_2019_hrly_NEE[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    C_NEE_TP39_2019_MAM=C_NEE_TP39_2019_array[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    S_NEE_TP39_2019_MAM=S_NEE_TP39_2019[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]


# In[19]:


# Load in TPD 2018 flux tower data & average to hourly resolution

# *** CHANGE PATH & FILENAME ***
TPD_2018_data=pd.read_csv('C:/Users/kitty/Documents/Research/SIF/Flux_Tower/Turkey_Point/TPD_HH_2018.csv', usecols=[0,1,2,6,74,75,76]) #header=1

TPD_2018_dates=np.zeros([17520])*np.nan
TPD_2018_dates2=np.zeros([17520])*np.nan
TPD_2018_NEE=np.zeros([17520])*np.nan
TPD_2018_NEE2=np.zeros([17520])*np.nan
TPD_2018_GPP=np.zeros([17520])*np.nan
TPD_2018_R=np.zeros([17520])*np.nan
n=0
m=0
date=1
for i in range(17520):
    if 201801010000<=TPD_2018_data.iat[i,0]<202001010000:
        TPD_2018_dates[i]=datetime.strptime(str(int(TPD_2018_data.iat[i,0])),'%Y%m%d%H%M').timetuple().tm_yday+(float(str(TPD_2018_data.iat[i,0])[8:10])+float(str(TPD_2018_data.iat[i,0])[10:12])/60)/24 #save the current date (and time)
        #check that the value is greater than -9999 (value for empty measurements)
        if TPD_2018_data.iat[i,2]>-9999:
            TPD_2018_NEE2[i]=TPD_2018_data.iat[i,2] # save the NEE value
        if TPD_2018_data.iat[i,6]>-9999:
            TPD_2018_NEE[i]=TPD_2018_data.iat[i,6] # save the NEE value
        if TPD_2018_data.iat[i,4]>-9999:
            TPD_2018_GPP[i]=TPD_2018_data.iat[i,4] # save the NEE value
        if TPD_2018_data.iat[i,5]>-9999:
            TPD_2018_R[i]=TPD_2018_data.iat[i,5] # save the NEE value

# Select Original SMUrF data over TPD
C_GPP_TPD_array=np.zeros(8765)*np.nan
C_NEE_TPD_array=np.zeros(8765)*np.nan
C_Reco_TPD_array=np.zeros(8765)*np.nan
for i in range(len(C_GPP[:,0,0])):
    C_GPP_TPD_array[i]=np.nanmean([C_GPP[i,2,2]])
    C_NEE_TPD_array[i]=np.nanmean([C_NEE[i,2,2]])
    C_Reco_TPD_array[i]=np.nanmean([C_Reco[i,2,2]])

# Average TPD fluxtower data to hourly resolution
TPD_GPP=np.zeros(np.shape(C_GPP_TPD_array))*np.nan
TPD_NEE=np.zeros(np.shape(C_GPP_TPD_array))*np.nan
TPD_NEEgf=np.zeros(np.shape(C_GPP_TPD_array))*np.nan
TPD_R=np.zeros(np.shape(C_GPP_TPD_array))*np.nan
for i in range(len(C_time_array)-5):
    with np.errstate(invalid='ignore'):
        TPD_GPP[i+5]=np.nanmean([TPD_2018_GPP[i*2],TPD_2018_GPP[i*2+1]])
        TPD_NEE[i+5]=np.nanmean([TPD_2018_NEE2[i*2],TPD_2018_NEE2[i*2+1]])
        TPD_NEEgf[i+5]=np.nanmean([TPD_2018_NEE[i*2],TPD_2018_NEE[i*2+1]])
        TPD_R[i+5]=np.nanmean([TPD_2018_R[i*2],TPD_2018_R[i*2+1]])


# In[20]:


# Select spring (March-May) 2018 data over TPD
#MAM: Doy 60 - 151 inclusive

with np.errstate(invalid='ignore'):
    TPD_GPP_MAM=TPD_GPP[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    C_GPP_TPD_MAM=C_GPP_TPD_array[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    S_GPP_TPD_MAM=S_GPP_TPD[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]

    TPD_R_MAM=TPD_R[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    C_Reco_TPD_MAM=C_Reco_TPD_array[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    S_Reco_TPD_MAM=S_Reco_TPD[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]

    TPD_NEEgf_MAM=TPD_NEEgf[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    TPD_NEE_MAM=TPD_NEE[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    C_NEE_TPD_MAM=C_NEE_TPD_array[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    S_NEE_TPD_MAM=S_NEE_TPD[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]


# In[21]:


# Load in 2019 TPD fluxtower data

# *** CHANGE PATH AND FILENAME ***
TPD_2019_data=pd.read_csv('C:/Users/kitty/Documents/Research/SIF/Flux_Tower/Turkey_Point/TPD_HH_2019.csv', usecols=[0,1,2,6,74,75,76]) #header=1

TPD_2019_dates=np.zeros([17520])*np.nan
TPD_2019_dates2=np.zeros([17520])*np.nan
TPD_2019_NEE=np.zeros([17520])*np.nan
TPD_2019_NEE2=np.zeros([17520])*np.nan
TPD_2019_GPP=np.zeros([17520])*np.nan
TPD_2019_R=np.zeros([17520])*np.nan
n=0
m=0
date=1
for i in range(17520):
    if 201901010000<=TPD_2019_data.iat[i,0]<202001010000:
        TPD_2019_dates[i]=TPD_2019_data.iat[i,0] #save the current date (and time)
        TPD_2019_dates2[i]=date+n/48
        #check that the value is greater than -9999 (value for empty measurements)
        if TPD_2019_data.iat[i,2]>-9999:
            TPD_2019_NEE2[i]=TPD_2019_data.iat[i,2] # save the NEE value
        if TPD_2019_data.iat[i,6]>-9999:
            TPD_2019_NEE[i]=TPD_2019_data.iat[i,6] # save the NEE value
        if TPD_2019_data.iat[i,4]>-9999:
            TPD_2019_GPP[i]=TPD_2019_data.iat[i,4] # save the NEE value
        if TPD_2019_data.iat[i,5]>-9999:
            TPD_2019_R[i]=TPD_2019_data.iat[i,5] # save the NEE value

#Select original SMUrF pixel over TPD
C_GPP_TPD_2019_array=np.zeros(8765)*np.nan
C_NEE_TPD_2019_array=np.zeros(8765)*np.nan
C_Reco_TPD_2019_array=np.zeros(8765)*np.nan

for i in range(len(C_GPP_2019[:,0,0])):
    C_GPP_TPD_2019_array[i]=np.nanmean([C_GPP_2019[i,2,2]])
    C_NEE_TPD_2019_array[i]=np.nanmean([C_NEE_2019[i,2,2]])
    C_Reco_TPD_2019_array[i]=np.nanmean([C_Reco_2019[i,2,2]])

#Average 30-minute fluxtower data to hourly resolution:
TPD_2019_hrly_GPP=np.zeros(np.shape(C_GPP_TPD_2019_array))*np.nan
TPD_2019_hrly_NEE=np.zeros(np.shape(C_GPP_TPD_2019_array))*np.nan
TPD_2019_hrly_NEEgf=np.zeros(np.shape(C_GPP_TPD_2019_array))*np.nan
TPD_2019_hrly_R=np.zeros(np.shape(C_GPP_TPD_2019_array))*np.nan

for i in range(len(C_time_array)-6):
    with np.errstate(invalid='ignore'):
        TPD_2019_hrly_GPP[i+5]=np.nanmean([TPD_2019_GPP[i*2],TPD_2019_GPP[i*2+1]])
        TPD_2019_hrly_NEE[i+5]=np.nanmean([TPD_2019_NEE2[i*2],TPD_2019_NEE2[i*2+1]])
        TPD_2019_hrly_NEEgf[i+5]=np.nanmean([TPD_2019_NEE[i*2],TPD_2019_NEE[i*2+1]])
        TPD_2019_hrly_R[i+5]=np.nanmean([TPD_2019_R[i*2],TPD_2019_R[i*2+1]])


# In[22]:


#Select only spring data over TPD in 2019
#MAM: Doy 60 - 151 inclusive

with np.errstate(invalid='ignore'):
    TPD_2019_GPP_MAM=TPD_2019_hrly_GPP[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    C_GPP_TPD_2019_MAM=C_GPP_TPD_2019_array[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    S_GPP_TPD_2019_MAM=S_GPP_TPD_2019[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]

    TPD_2019_R_MAM=TPD_2019_hrly_R[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    C_Reco_TPD_2019_MAM=C_Reco_TPD_2019_array[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    S_Reco_TPD_2019_MAM=S_Reco_TPD_2019[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]

    TPD_2019_NEEgf_MAM=TPD_2019_hrly_NEEgf[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    TPD_2019_NEE_MAM=TPD_2019_hrly_NEE[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    C_NEE_TPD_2019_MAM=C_NEE_TPD_2019_array[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]
    S_NEE_TPD_2019_MAM=S_NEE_TPD_2019[(np.round(C_time_array,5)>=60) & (np.round(C_time_array,5)<152)]


# In[23]:


# Combine data over all fluxtowers in spring
Fluxtower_GPP_MAM_tot = np.concatenate([Borden_GPPgf_MAM,TP39_GPP_MAM,TPD_GPP_MAM,TP39_2019_GPP_MAM,TPD_2019_GPP_MAM])
C_GPP_MAM_tot = np.concatenate([C_GPP_Borden_MAM,C_GPP_TP39_MAM,C_GPP_TPD_MAM,C_GPP_TP39_2019_MAM,C_GPP_TPD_2019_MAM])
S_GPP_MAM_tot = np.concatenate([S_GPP_Borden_MAM,S_GPP_TP39_MAM,S_GPP_TPD_MAM,S_GPP_TP39_2019_MAM,S_GPP_TPD_2019_MAM])

Fluxtower_Reco_MAM_tot = np.concatenate([Borden_Rgf_MAM,TP39_R_MAM,TPD_R_MAM,TP39_2019_R_MAM,TPD_2019_R_MAM])
C_Reco_MAM_tot = np.concatenate([C_Reco_Borden_MAM,C_Reco_TP39_MAM,C_Reco_TPD_MAM,C_Reco_TP39_2019_MAM,C_Reco_TPD_2019_MAM])
S_Reco_MAM_tot = np.concatenate([S_Reco_Borden_MAM,S_Reco_TP39_MAM,S_Reco_TPD_MAM,S_Reco_TP39_2019_MAM,S_Reco_TPD_2019_MAM])

Fluxtower_NEEgf_MAM_tot = np.concatenate([Borden_NEEgf_MAM,TP39_NEEgf_MAM,TPD_NEEgf_MAM,TP39_2019_NEEgf_MAM,TPD_2019_NEEgf_MAM])
Fluxtower_NEE_MAM_tot = np.concatenate([Borden_NEE_MAM,TP39_NEE_MAM,TPD_NEE_MAM,TP39_2019_NEE_MAM,TPD_2019_NEE_MAM])
C_NEE_MAM_tot = np.concatenate([C_NEE_Borden_MAM,C_NEE_TP39_MAM,C_NEE_TPD_MAM,C_NEE_TP39_2019_MAM,C_NEE_TPD_2019_MAM])
S_NEE_MAM_tot = np.concatenate([S_NEE_Borden_MAM,S_NEE_TP39_MAM,S_NEE_TPD_MAM,S_NEE_TP39_2019_MAM,S_NEE_TPD_2019_MAM])


# In[ ]:





# In[24]:


# Define linear function for plotting
line1_1=np.arange(-100,100)

def func2(x,m,b):
    return m*x+b


# In[25]:


#Fit original SMUrF data to fluxtower data using a bootstrapped Huber fit

finitemask0 = np.isfinite(Fluxtower_NEE_MAM_tot)
Fluxtower_NEE_MAMclean0 = Fluxtower_NEE_MAM_tot[finitemask0]
C_NEE_MAMclean0 = C_NEE_MAM_tot[finitemask0]

finitemask2 = np.isfinite(C_NEE_MAMclean0)
C_NEE_MAMclean1 = C_NEE_MAMclean0[finitemask2]
Fluxtower_NEE_MAMclean1 = Fluxtower_NEE_MAMclean0[finitemask2]

Huber_tot_MAM_C_NEE_slps=[]
Huber_tot_MAM_C_NEE_ints=[]
Huber_tot_MAM_C_NEE_R2=[]

#try bootstrapping
indx_list=list(range(0,len(C_NEE_MAMclean1)))
for i in range(1,1001):
    #sub selection of points
    NEE_indx=np.random.choice(indx_list,size=len(C_NEE_MAMclean1))

    Huber_model = linear_model.HuberRegressor(fit_intercept=True)
    Huber_fit=Huber_model.fit((Fluxtower_NEE_MAMclean1[NEE_indx]).reshape(-1,1),C_NEE_MAMclean1[NEE_indx])
    H_m=Huber_fit.coef_
    H_c=Huber_fit.intercept_
    x_accpt, y_accpt = Fluxtower_NEE_MAMclean1, C_NEE_MAMclean1
    y_predict = H_m * x_accpt + H_c
    H_R2=r2_score(y_accpt, y_predict)
    Huber_tot_MAM_C_NEE_slps.append(H_m)
    Huber_tot_MAM_C_NEE_ints.append(H_c)
    Huber_tot_MAM_C_NEE_R2.append(H_R2)
    
y_predict = np.nanmean(Huber_tot_MAM_C_NEE_slps) * x_accpt + np.nanmean(Huber_tot_MAM_C_NEE_ints)
Huber_MAM_C_NEE_R2=r2_score(y_accpt, y_predict)

print('Original SMUrF MAM slope: '+str(np.round(np.nanmean(Huber_tot_MAM_C_NEE_slps),3))+' +/- '+str(np.round(np.nanstd(Huber_tot_MAM_C_NEE_slps),3)))
print('Original SMUrF MAM intercept: '+str(np.round(np.nanmean(Huber_tot_MAM_C_NEE_ints),3))+' +/- '+str(np.round(np.nanstd(Huber_tot_MAM_C_NEE_ints),3)))

print('Original SMUrF MAM R^2: '+str(np.round(np.nanmean(Huber_MAM_C_NEE_R2),3)))


# In[26]:


#Fit Modified SMUrF data to fluxtower data using a bootstrapped Huber fit
          
finitemask0 = np.isfinite(Fluxtower_NEE_MAM_tot)
Fluxtower_NEE_MAMclean0 = Fluxtower_NEE_MAM_tot[finitemask0]
S_NEE_MAMclean0 = S_NEE_MAM_tot[finitemask0]

finitemask2 = np.isfinite(S_NEE_MAMclean0)
S_NEE_MAMclean1 = S_NEE_MAMclean0[finitemask2]
Fluxtower_NEE_MAMclean1 = Fluxtower_NEE_MAMclean0[finitemask2]

Huber_tot_MAM_S_NEE_slps=[]
Huber_tot_MAM_S_NEE_ints=[]
Huber_tot_MAM_S_NEE_R2=[]

#try bootstrapping
indx_list=list(range(0,len(S_NEE_MAMclean1)))
for i in range(1,1001):
    #sub selection of points
    NEE_indx=np.random.choice(indx_list,size=len(S_NEE_MAMclean1))

    Huber_model = linear_model.HuberRegressor(fit_intercept=True)
    Huber_fit=Huber_model.fit((Fluxtower_NEE_MAMclean1[NEE_indx]).reshape(-1,1),S_NEE_MAMclean1[NEE_indx])
    H_m=Huber_fit.coef_
    H_c=Huber_fit.intercept_
    x_accpt, y_accpt = Fluxtower_NEE_MAMclean1, S_NEE_MAMclean1
    y_predict = H_m * x_accpt + H_c
    H_R2=r2_score(y_accpt, y_predict)
    Huber_tot_MAM_S_NEE_slps.append(H_m)
    Huber_tot_MAM_S_NEE_ints.append(H_c)
    Huber_tot_MAM_S_NEE_R2.append(H_R2)
    
#print(np.nanmean(Huber_tot_MAM_S_NEE_slps),np.nanstd(Huber_tot_MAM_S_NEE_slps),np.nanmean(Huber_tot_MAM_S_NEE_ints),np.nanstd(Huber_tot_MAM_S_NEE_ints),np.nanmean(Huber_tot_MAM_S_NEE_R2),np.nanstd(Huber_tot_MAM_S_NEE_R2))

x_accpt, y_accpt = Fluxtower_NEE_MAMclean1, S_NEE_MAMclean1
y_predict = np.nanmean(Huber_tot_MAM_S_NEE_slps) * x_accpt + np.nanmean(Huber_tot_MAM_S_NEE_ints)
Huber_MAM_S_NEE_R2=r2_score(y_accpt, y_predict)
#print(Huber_MAM_S_NEE_R2)

print('Updated SMUrF MAM slope: '+str(np.round(np.nanmean(Huber_tot_MAM_S_NEE_slps),3))+' +/- '+str(np.round(np.nanstd(Huber_tot_MAM_S_NEE_slps),3)))
print('Updated SMUrF MAM intercept: '+str(np.round(np.nanmean(Huber_tot_MAM_S_NEE_ints),3))+' +/- '+str(np.round(np.nanstd(Huber_tot_MAM_S_NEE_ints),3)))

print('Updated SMUrF MAM R^2: '+str(np.round(np.nanmean(Huber_MAM_S_NEE_R2),3)))


# In[ ]:





# In[27]:


# Get fluxes over towers for summer

#JJA: Doy 152 - 223 inclusive
with np.errstate(invalid='ignore'):
    JJA_time=C_time_array[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    Borden_GPPgf_JJA=Borden_GEPgf[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    C_GPP_Borden_JJA=C_GPP_array[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    S_GPP_Borden_JJA=S_GPP_Borden[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]

    Borden_Rgf_JJA=Borden_Rgf[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    C_Reco_Borden_JJA=C_Reco_array[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    S_Reco_Borden_JJA=S_Reco_Borden[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]

    Borden_NEEgf_JJA=Borden_NEEgf[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    Borden_NEE_JJA=Borden_NEE[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    C_NEE_Borden_JJA=C_NEE_array[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    S_NEE_Borden_JJA=S_NEE_Borden[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]

    TP39_GPP_JJA=TP39_GPP[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    C_GPP_TP39_JJA=C_GPP_TP39_array[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    S_GPP_TP39_JJA=S_GPP_TP39[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]

    TP39_R_JJA=TP39_R[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    C_Reco_TP39_JJA=C_Reco_TP39_array[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    S_Reco_TP39_JJA=S_Reco_TP39[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]

    TP39_NEEgf_JJA=TP39_NEEgf[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    TP39_NEE_JJA=TP39_NEE[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    C_NEE_TP39_JJA=C_NEE_TP39_array[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    S_NEE_TP39_JJA=S_NEE_TP39[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]

    TP39_2019_GPP_JJA=TP39_2019_hrly_GPP[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    C_GPP_TP39_2019_JJA=C_GPP_TP39_2019_array[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    S_GPP_TP39_2019_JJA=S_GPP_TP39_2019[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]

    TP39_2019_R_JJA=TP39_2019_hrly_R[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    C_Reco_TP39_2019_JJA=C_Reco_TP39_2019_array[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    S_Reco_TP39_2019_JJA=S_Reco_TP39_2019[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]

    TP39_2019_NEEgf_JJA=TP39_2019_hrly_NEEgf[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    TP39_2019_NEE_JJA=TP39_2019_hrly_NEE[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    C_NEE_TP39_2019_JJA=C_NEE_TP39_2019_array[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    S_NEE_TP39_2019_JJA=S_NEE_TP39_2019[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]

    TPD_GPP_JJA=TPD_GPP[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    C_GPP_TPD_JJA=C_GPP_TPD_array[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    S_GPP_TPD_JJA=S_GPP_TPD[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]

    TPD_R_JJA=TPD_R[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    C_Reco_TPD_JJA=C_Reco_TPD_array[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    S_Reco_TPD_JJA=S_Reco_TPD[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]

    TPD_NEEgf_JJA=TPD_NEEgf[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    TPD_NEE_JJA=TPD_NEE[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    C_NEE_TPD_JJA=C_NEE_TPD_array[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    S_NEE_TPD_JJA=S_NEE_TPD[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]

    TPD_2019_GPP_JJA=TPD_2019_hrly_GPP[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    C_GPP_TPD_2019_JJA=C_GPP_TPD_2019_array[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    S_GPP_TPD_2019_JJA=S_GPP_TPD_2019[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]

    TPD_2019_R_JJA=TPD_2019_hrly_R[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    C_Reco_TPD_2019_JJA=C_Reco_TPD_2019_array[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    S_Reco_TPD_2019_JJA=S_Reco_TPD_2019[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]

    TPD_2019_NEEgf_JJA=TPD_2019_hrly_NEEgf[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    TPD_2019_NEE_JJA=TPD_2019_hrly_NEE[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    C_NEE_TPD_2019_JJA=C_NEE_TPD_2019_array[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]
    S_NEE_TPD_2019_JJA=S_NEE_TPD_2019[(np.round(C_time_array,5)>=152) & (np.round(C_time_array,5)<224)]


# In[28]:


# Combine data over all fluxtowers in summer
Fluxtower_GPP_JJA_tot = np.concatenate([Borden_GPPgf_JJA,TP39_GPP_JJA,TPD_GPP_JJA,TP39_2019_GPP_JJA,TPD_2019_GPP_JJA])
C_GPP_JJA_tot = np.concatenate([C_GPP_Borden_JJA,C_GPP_TP39_JJA,C_GPP_TPD_JJA,C_GPP_TP39_2019_JJA,C_GPP_TPD_2019_JJA])
S_GPP_JJA_tot = np.concatenate([S_GPP_Borden_JJA,S_GPP_TP39_JJA,S_GPP_TPD_JJA,S_GPP_TP39_2019_JJA,S_GPP_TPD_2019_JJA])

Fluxtower_Reco_JJA_tot = np.concatenate([Borden_Rgf_JJA,TP39_R_JJA,TPD_R_JJA,TP39_2019_R_JJA,TPD_2019_R_JJA])
C_Reco_JJA_tot = np.concatenate([C_Reco_Borden_JJA,C_Reco_TP39_JJA,C_Reco_TPD_JJA,C_Reco_TP39_2019_JJA,C_Reco_TPD_2019_JJA])
S_Reco_JJA_tot = np.concatenate([S_Reco_Borden_JJA,S_Reco_TP39_JJA,S_Reco_TPD_JJA,S_Reco_TP39_2019_JJA,S_Reco_TPD_2019_JJA])

Fluxtower_NEEgf_JJA_tot = np.concatenate([Borden_NEEgf_JJA,TP39_NEEgf_JJA,TPD_NEEgf_JJA,TP39_2019_NEEgf_JJA,TPD_2019_NEEgf_JJA])
Fluxtower_NEE_JJA_tot = np.concatenate([Borden_NEE_JJA,TP39_NEE_JJA,TPD_NEE_JJA,TP39_2019_NEE_JJA,TPD_2019_NEE_JJA])
C_NEE_JJA_tot = np.concatenate([C_NEE_Borden_JJA,C_NEE_TP39_JJA,C_NEE_TPD_JJA,C_NEE_TP39_2019_JJA,C_NEE_TPD_2019_JJA])
S_NEE_JJA_tot = np.concatenate([S_NEE_Borden_JJA,S_NEE_TP39_JJA,S_NEE_TPD_JJA,S_NEE_TP39_2019_JJA,S_NEE_TPD_2019_JJA])


# In[29]:


#Fit Original SMUrF data to fluxtower data using a bootstrapped Huber fit
          
finitemask0 = np.isfinite(Fluxtower_NEE_JJA_tot)
Fluxtower_NEE_JJAclean0 = Fluxtower_NEE_JJA_tot[finitemask0]
C_NEE_JJAclean0 = C_NEE_JJA_tot[finitemask0]

finitemask2 = np.isfinite(C_NEE_JJAclean0)
C_NEE_JJAclean1 = C_NEE_JJAclean0[finitemask2]
Fluxtower_NEE_JJAclean1 = Fluxtower_NEE_JJAclean0[finitemask2]

Huber_tot_JJA_C_NEE_slps=[]
Huber_tot_JJA_C_NEE_ints=[]
Huber_tot_JJA_C_NEE_R2=[]

#try bootstrapping
indx_list=list(range(0,len(C_NEE_JJAclean1)))
for i in range(1,1001):
    #sub selection of points
    NEE_indx=np.random.choice(indx_list,size=len(C_NEE_JJAclean1))

    Huber_model = linear_model.HuberRegressor(fit_intercept=True)
    Huber_fit=Huber_model.fit((Fluxtower_NEE_JJAclean1[NEE_indx]).reshape(-1,1),C_NEE_JJAclean1[NEE_indx])
    H_m=Huber_fit.coef_
    H_c=Huber_fit.intercept_
    x_accpt, y_accpt = Fluxtower_NEE_JJAclean1, C_NEE_JJAclean1
    y_predict = H_m * x_accpt + H_c
    H_R2=r2_score(y_accpt, y_predict)
    Huber_tot_JJA_C_NEE_slps.append(H_m)
    Huber_tot_JJA_C_NEE_ints.append(H_c)
    Huber_tot_JJA_C_NEE_R2.append(H_R2)
    
y_predict = np.nanmean(Huber_tot_JJA_C_NEE_slps) * x_accpt + np.nanmean(Huber_tot_JJA_C_NEE_ints)
Huber_JJA_C_NEE_R2=r2_score(y_accpt, y_predict)

print('Original SMUrF JJA slope: '+str(np.round(np.nanmean(Huber_tot_JJA_C_NEE_slps),3))+' +/- '+str(np.round(np.nanstd(Huber_tot_JJA_C_NEE_slps),3)))
print('Original SMUrF JJA intercept: '+str(np.round(np.nanmean(Huber_tot_JJA_C_NEE_ints),3))+' +/- '+str(np.round(np.nanstd(Huber_tot_JJA_C_NEE_ints),3)))

print('Original SMUrF JJA R^2: '+str(np.round(np.nanmean(Huber_JJA_C_NEE_R2),3)))


# In[35]:


# Fit Modified SMUrF data to fluxtower data using bootstrapped Huber fit
          
finitemask0 = np.isfinite(Fluxtower_NEE_JJA_tot)
Fluxtower_NEE_JJAclean0 = Fluxtower_NEE_JJA_tot[finitemask0]
S_NEE_JJAclean0 = S_NEE_JJA_tot[finitemask0]

finitemask2 = np.isfinite(S_NEE_JJAclean0)
S_NEE_JJAclean1 = S_NEE_JJAclean0[finitemask2]
Fluxtower_NEE_JJAclean1 = Fluxtower_NEE_JJAclean0[finitemask2]

Huber_tot_JJA_S_NEE_slps=[]
Huber_tot_JJA_S_NEE_ints=[]
Huber_tot_JJA_S_NEE_R2=[]

#try bootstrapping
indx_list=list(range(0,len(S_NEE_JJAclean1)))
for i in range(1,1001):
    #sub selection of points
    NEE_indx=np.random.choice(indx_list,size=len(S_NEE_JJAclean1))

    Huber_model = linear_model.HuberRegressor(fit_intercept=True)
    Huber_fit=Huber_model.fit((Fluxtower_NEE_JJAclean1[NEE_indx]).reshape(-1,1),S_NEE_JJAclean1[NEE_indx])
    H_m=Huber_fit.coef_
    H_c=Huber_fit.intercept_
    x_accpt, y_accpt = Fluxtower_NEE_JJAclean1, S_NEE_JJAclean1
    y_predict = H_m * x_accpt + H_c
    H_R2=r2_score(y_accpt, y_predict)
    Huber_tot_JJA_S_NEE_slps.append(H_m)
    Huber_tot_JJA_S_NEE_ints.append(H_c)
    Huber_tot_JJA_S_NEE_R2.append(H_R2)
    
y_predict = np.nanmean(Huber_tot_JJA_S_NEE_slps) * x_accpt + np.nanmean(Huber_tot_JJA_S_NEE_ints)
Huber_JJA_S_NEE_R2=r2_score(y_accpt, y_predict)

print('Updated SMUrF JJA slope: '+str(np.round(np.nanmean(Huber_tot_JJA_S_NEE_slps),3))+' +/- '+str(np.round(np.nanstd(Huber_tot_JJA_S_NEE_slps),3)))
print('Updated SMUrF JJA intercept: '+str(np.round(np.nanmean(Huber_tot_JJA_S_NEE_ints),3))+' +/- '+str(np.round(np.nanstd(Huber_tot_JJA_S_NEE_ints),3)))

print('Updated SMUrF JJA R^2: '+str(np.round(np.nanmean(Huber_JJA_S_NEE_R2),3)))


# In[ ]:





# In[31]:


#SON: Doy 224 - 334 inclusive
with np.errstate(invalid='ignore'):
    SON_time=C_time_array[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    
    #Borden Forest 2018:
    Borden_GPPgf_SON=Borden_GEPgf[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    C_GPP_Borden_SON=C_GPP_array[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    S_GPP_Borden_SON=S_GPP_Borden[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]

    Borden_Rgf_SON=Borden_Rgf[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    C_Reco_Borden_SON=C_Reco_array[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    S_Reco_Borden_SON=S_Reco_Borden[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]

    Borden_NEEgf_SON=Borden_NEEgf[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    Borden_NEE_SON=Borden_NEE[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    C_NEE_Borden_SON=C_NEE_array[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    S_NEE_Borden_SON=S_NEE_Borden[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]

    #TP39 2018:
    TP39_GPP_SON=TP39_GPP[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    C_GPP_TP39_SON=C_GPP_TP39_array[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    S_GPP_TP39_SON=S_GPP_TP39[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]

    TP39_R_SON=TP39_R[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    C_Reco_TP39_SON=C_Reco_TP39_array[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    S_Reco_TP39_SON=S_Reco_TP39[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]

    TP39_NEEgf_SON=TP39_NEEgf[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    TP39_NEE_SON=TP39_NEE[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    C_NEE_TP39_SON=C_NEE_TP39_array[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    S_NEE_TP39_SON=S_NEE_TP39[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]

    #TP39 2019:
    TP39_2019_GPP_SON=TP39_2019_hrly_GPP[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    C_GPP_TP39_2019_SON=C_GPP_TP39_2019_array[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    S_GPP_TP39_2019_SON=S_GPP_TP39_2019[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]

    TP39_2019_R_SON=TP39_2019_hrly_R[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    C_Reco_TP39_2019_SON=C_Reco_TP39_2019_array[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    S_Reco_TP39_2019_SON=S_Reco_TP39_2019[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]

    TP39_2019_NEEgf_SON=TP39_2019_hrly_NEEgf[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    TP39_2019_NEE_SON=TP39_2019_hrly_NEE[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    C_NEE_TP39_2019_SON=C_NEE_TP39_2019_array[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    S_NEE_TP39_2019_SON=S_NEE_TP39_2019[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]

    #TPD 2018:
    TPD_GPP_SON=TPD_GPP[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    C_GPP_TPD_SON=C_GPP_TPD_array[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    S_GPP_TPD_SON=S_GPP_TPD[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]

    TPD_R_SON=TPD_R[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    C_Reco_TPD_SON=C_Reco_TPD_array[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    S_Reco_TPD_SON=S_Reco_TPD[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]

    TPD_NEEgf_SON=TPD_NEEgf[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    TPD_NEE_SON=TPD_NEE[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    C_NEE_TPD_SON=C_NEE_TPD_array[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    S_NEE_TPD_SON=S_NEE_TPD[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]

    #TPD 2019:
    TPD_2019_GPP_SON=TPD_2019_hrly_GPP[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    C_GPP_TPD_2019_SON=C_GPP_TPD_2019_array[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    S_GPP_TPD_2019_SON=S_GPP_TPD_2019[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]

    TPD_2019_R_SON=TPD_2019_hrly_R[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    C_Reco_TPD_2019_SON=C_Reco_TPD_2019_array[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    S_Reco_TPD_2019_SON=S_Reco_TPD_2019[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]

    TPD_2019_NEEgf_SON=TPD_2019_hrly_NEEgf[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    TPD_2019_NEE_SON=TPD_2019_hrly_NEE[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    C_NEE_TPD_2019_SON=C_NEE_TPD_2019_array[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]
    S_NEE_TPD_2019_SON=S_NEE_TPD_2019[(np.round(C_time_array,5)>=224) & (np.round(C_time_array,5)<335)]


# In[32]:


# Combine data over all fluxtowers:
Fluxtower_GPP_SON_tot = np.concatenate([Borden_GPPgf_SON,TP39_GPP_SON,TPD_GPP_SON,TP39_2019_GPP_SON,TPD_2019_GPP_SON])
C_GPP_SON_tot = np.concatenate([C_GPP_Borden_SON,C_GPP_TP39_SON,C_GPP_TPD_SON,C_GPP_TP39_2019_SON,C_GPP_TPD_2019_SON])
S_GPP_SON_tot = np.concatenate([S_GPP_Borden_SON,S_GPP_TP39_SON,S_GPP_TPD_SON,S_GPP_TP39_2019_SON,S_GPP_TPD_2019_SON])

Fluxtower_Reco_SON_tot = np.concatenate([Borden_Rgf_SON,TP39_R_SON,TPD_R_SON,TP39_2019_R_SON,TPD_2019_R_SON])
C_Reco_SON_tot = np.concatenate([C_Reco_Borden_SON,C_Reco_TP39_SON,C_Reco_TPD_SON,C_Reco_TP39_2019_SON,C_Reco_TPD_2019_SON])
S_Reco_SON_tot = np.concatenate([S_Reco_Borden_SON,S_Reco_TP39_SON,S_Reco_TPD_SON,S_Reco_TP39_2019_SON,S_Reco_TPD_2019_SON])

Fluxtower_NEEgf_SON_tot = np.concatenate([Borden_NEEgf_SON,TP39_NEEgf_SON,TPD_NEEgf_SON,TP39_2019_NEEgf_SON,TPD_2019_NEEgf_SON])
Fluxtower_NEE_SON_tot = np.concatenate([Borden_NEE_SON,TP39_NEE_SON,TPD_NEE_SON,TP39_2019_NEE_SON,TPD_2019_NEE_SON])
C_NEE_SON_tot = np.concatenate([C_NEE_Borden_SON,C_NEE_TP39_SON,C_NEE_TPD_SON,C_NEE_TP39_2019_SON,C_NEE_TPD_2019_SON])
S_NEE_SON_tot = np.concatenate([S_NEE_Borden_SON,S_NEE_TP39_SON,S_NEE_TPD_SON,S_NEE_TP39_2019_SON,S_NEE_TPD_2019_SON])


# In[33]:


##Fit Original SMUrF data to fluxtower data using bootstrapped Huber fit
          
finitemask0 = np.isfinite(Fluxtower_NEE_SON_tot)
Fluxtower_NEE_SONclean0 = Fluxtower_NEE_SON_tot[finitemask0]
C_NEE_SONclean0 = C_NEE_SON_tot[finitemask0]

finitemask2 = np.isfinite(C_NEE_SONclean0)
C_NEE_SONclean1 = C_NEE_SONclean0[finitemask2]
Fluxtower_NEE_SONclean1 = Fluxtower_NEE_SONclean0[finitemask2]

Huber_tot_SON_C_NEE_slps=[]
Huber_tot_SON_C_NEE_ints=[]
Huber_tot_SON_C_NEE_R2=[]

#try bootstrapping
indx_list=list(range(0,len(C_NEE_SONclean1)))
for i in range(1,1001):
    #sub selection of points
    NEE_indx=np.random.choice(indx_list,size=len(C_NEE_SONclean1))

    Huber_model = linear_model.HuberRegressor(fit_intercept=True)
    Huber_fit=Huber_model.fit((Fluxtower_NEE_SONclean1[NEE_indx]).reshape(-1,1),C_NEE_SONclean1[NEE_indx])
    H_m=Huber_fit.coef_
    H_c=Huber_fit.intercept_
    x_accpt, y_accpt = Fluxtower_NEE_SONclean1, C_NEE_SONclean1
    y_predict = H_m * x_accpt + H_c
    H_R2=r2_score(y_accpt, y_predict)
    Huber_tot_SON_C_NEE_slps.append(H_m)
    Huber_tot_SON_C_NEE_ints.append(H_c)
    Huber_tot_SON_C_NEE_R2.append(H_R2)
    
y_predict = np.nanmean(Huber_tot_SON_C_NEE_slps) * x_accpt + np.nanmean(Huber_tot_SON_C_NEE_ints)
Huber_SON_C_NEE_R2=r2_score(y_accpt, y_predict)

print('Original SMUrF SON slope: '+str(np.round(np.nanmean(Huber_tot_SON_C_NEE_slps),3))+' +/- '+str(np.round(np.nanstd(Huber_tot_SON_C_NEE_slps),3)))
print('Original SMUrF SON intercept: '+str(np.round(np.nanmean(Huber_tot_SON_C_NEE_ints),3))+' +/- '+str(np.round(np.nanstd(Huber_tot_SON_C_NEE_ints),3)))

print('Original SMUrF SON R^2: '+str(np.round(np.nanmean(Huber_SON_C_NEE_R2),3)))


# In[34]:


# Fit updated SMUrF model to flux tower data in autumn using a bootstrapped Huber fit
finitemask0 = np.isfinite(Fluxtower_NEE_SON_tot)
Fluxtower_NEE_SONclean0 = Fluxtower_NEE_SON_tot[finitemask0]
S_NEE_SONclean0 = S_NEE_SON_tot[finitemask0]

finitemask2 = np.isfinite(S_NEE_SONclean0)
S_NEE_SONclean1 = S_NEE_SONclean0[finitemask2]
Fluxtower_NEE_SONclean1 = Fluxtower_NEE_SONclean0[finitemask2]

Huber_tot_SON_S_NEE_slps=[]
Huber_tot_SON_S_NEE_ints=[]
Huber_tot_SON_S_NEE_R2=[]

#try bootstrapping
indx_list=list(range(0,len(S_NEE_SONclean1)))
for i in range(1,1001):
    #sub selection of points
    NEE_indx=np.random.choice(indx_list,size=len(S_NEE_SONclean1))

    Huber_model = linear_model.HuberRegressor(fit_intercept=True)
    Huber_fit=Huber_model.fit((Fluxtower_NEE_SONclean1[NEE_indx]).reshape(-1,1),S_NEE_SONclean1[NEE_indx])
    H_m=Huber_fit.coef_
    H_c=Huber_fit.intercept_
    x_accpt, y_accpt = Fluxtower_NEE_SONclean1, S_NEE_SONclean1
    y_predict = H_m * x_accpt + H_c
    H_R2=r2_score(y_accpt, y_predict)
    Huber_tot_SON_S_NEE_slps.append(H_m)
    Huber_tot_SON_S_NEE_ints.append(H_c)
    Huber_tot_SON_S_NEE_R2.append(H_R2)

y_predict = np.nanmean(Huber_tot_SON_S_NEE_slps) * x_accpt + np.nanmean(Huber_tot_SON_S_NEE_ints)
Huber_SON_S_NEE_R2=r2_score(y_accpt, y_predict)

print('Updated SMUrF SON slope: '+str(np.round(np.nanmean(Huber_tot_SON_S_NEE_slps),3))+' +/- '+str(np.round(np.nanstd(Huber_tot_SON_S_NEE_slps),3)))
print('Updated SMUrF SON intercept: '+str(np.round(np.nanmean(Huber_tot_SON_S_NEE_ints),3))+' +/- '+str(np.round(np.nanstd(Huber_tot_SON_S_NEE_ints),3)))

print('Updated SMUrF SON R^2: '+str(np.round(np.nanmean(Huber_SON_S_NEE_R2),3)))


# In[ ]:





# In[36]:


# Isolate winter data (DJF): Doy 224 - 334 inclusive
with np.errstate(invalid='ignore'):
    DJF_time=C_time_array[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    Borden_GPPgf_DJF=Borden_GEPgf[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    C_GPP_Borden_DJF=C_GPP_array[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    S_GPP_Borden_DJF=S_GPP_Borden[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]

    Borden_Rgf_DJF=Borden_Rgf[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    C_Reco_Borden_DJF=C_Reco_array[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    S_Reco_Borden_DJF=S_Reco_Borden[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]

    Borden_NEEgf_DJF=Borden_NEEgf[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    Borden_NEE_DJF=Borden_NEE[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    C_NEE_Borden_DJF=C_NEE_array[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    S_NEE_Borden_DJF=S_NEE_Borden[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]

    TP39_GPP_DJF=TP39_GPP[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    C_GPP_TP39_DJF=C_GPP_TP39_array[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    S_GPP_TP39_DJF=S_GPP_TP39[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]

    TP39_R_DJF=TP39_R[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    C_Reco_TP39_DJF=C_Reco_TP39_array[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    S_Reco_TP39_DJF=S_Reco_TP39[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]

    TP39_NEEgf_DJF=TP39_NEEgf[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    TP39_NEE_DJF=TP39_NEE[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    C_NEE_TP39_DJF=C_NEE_TP39_array[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    S_NEE_TP39_DJF=S_NEE_TP39[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]

    TP39_2019_GPP_DJF=TP39_2019_hrly_GPP[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    C_GPP_TP39_2019_DJF=C_GPP_TP39_2019_array[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    S_GPP_TP39_2019_DJF=S_GPP_TP39_2019[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]

    TP39_2019_R_DJF=TP39_2019_hrly_R[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    C_Reco_TP39_2019_DJF=C_Reco_TP39_2019_array[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    S_Reco_TP39_2019_DJF=S_Reco_TP39_2019[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]

    TP39_2019_NEEgf_DJF=TP39_2019_hrly_NEEgf[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    TP39_2019_NEE_DJF=TP39_2019_hrly_NEE[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    C_NEE_TP39_2019_DJF=C_NEE_TP39_2019_array[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    S_NEE_TP39_2019_DJF=S_NEE_TP39_2019[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]

    TPD_GPP_DJF=TPD_GPP[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    C_GPP_TPD_DJF=C_GPP_TPD_array[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    S_GPP_TPD_DJF=S_GPP_TPD[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]

    TPD_R_DJF=TPD_R[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    C_Reco_TPD_DJF=C_Reco_TPD_array[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    S_Reco_TPD_DJF=S_Reco_TPD[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]

    TPD_NEEgf_DJF=TPD_NEEgf[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    TPD_NEE_DJF=TPD_NEE[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    C_NEE_TPD_DJF=C_NEE_TPD_array[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    S_NEE_TPD_DJF=S_NEE_TPD[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]

    TPD_2019_GPP_DJF=TPD_2019_hrly_GPP[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    C_GPP_TPD_2019_DJF=C_GPP_TPD_2019_array[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    S_GPP_TPD_2019_DJF=S_GPP_TPD_2019[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]

    TPD_2019_R_DJF=TPD_2019_hrly_R[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    C_Reco_TPD_2019_DJF=C_Reco_TPD_2019_array[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    S_Reco_TPD_2019_DJF=S_Reco_TPD_2019[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]

    TPD_2019_NEEgf_DJF=TPD_2019_hrly_NEEgf[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    TPD_2019_NEE_DJF=TPD_2019_hrly_NEE[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    C_NEE_TPD_2019_DJF=C_NEE_TPD_2019_array[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]
    S_NEE_TPD_2019_DJF=S_NEE_TPD_2019[(np.round(C_time_array,5)>=335) | (np.round(C_time_array,5)<60)]


# In[37]:


Fluxtower_GPP_DJF_tot = np.concatenate([Borden_GPPgf_DJF,TP39_GPP_DJF,TPD_GPP_DJF,TP39_2019_GPP_DJF,TPD_2019_GPP_DJF])
C_GPP_DJF_tot = np.concatenate([C_GPP_Borden_DJF,C_GPP_TP39_DJF,C_GPP_TPD_DJF,C_GPP_TP39_2019_DJF,C_GPP_TPD_2019_DJF])
S_GPP_DJF_tot = np.concatenate([S_GPP_Borden_DJF,S_GPP_TP39_DJF,S_GPP_TPD_DJF,S_GPP_TP39_2019_DJF,S_GPP_TPD_2019_DJF])

Fluxtower_Reco_DJF_tot = np.concatenate([Borden_Rgf_DJF,TP39_R_DJF,TPD_R_DJF,TP39_2019_R_DJF,TPD_2019_R_DJF])
C_Reco_DJF_tot = np.concatenate([C_Reco_Borden_DJF,C_Reco_TP39_DJF,C_Reco_TPD_DJF,C_Reco_TP39_2019_DJF,C_Reco_TPD_2019_DJF])
S_Reco_DJF_tot = np.concatenate([S_Reco_Borden_DJF,S_Reco_TP39_DJF,S_Reco_TPD_DJF,S_Reco_TP39_2019_DJF,S_Reco_TPD_2019_DJF])

Fluxtower_NEEgf_DJF_tot = np.concatenate([Borden_NEEgf_DJF,TP39_NEEgf_DJF,TPD_NEEgf_DJF,TP39_2019_NEEgf_DJF,TPD_2019_NEEgf_DJF])
Fluxtower_NEE_DJF_tot = np.concatenate([Borden_NEE_DJF,TP39_NEE_DJF,TPD_NEE_DJF,TP39_2019_NEE_DJF,TPD_2019_NEE_DJF])
C_NEE_DJF_tot = np.concatenate([C_NEE_Borden_DJF,C_NEE_TP39_DJF,C_NEE_TPD_DJF,C_NEE_TP39_2019_DJF,C_NEE_TPD_2019_DJF])
S_NEE_DJF_tot = np.concatenate([S_NEE_Borden_DJF,S_NEE_TP39_DJF,S_NEE_TPD_DJF,S_NEE_TP39_2019_DJF,S_NEE_TPD_2019_DJF])


# In[38]:


# Fit Original SMUrF data using bootstrapped Huber fit 
finitemask0 = np.isfinite(Fluxtower_NEE_DJF_tot)
Fluxtower_NEE_DJFclean0 = Fluxtower_NEE_DJF_tot[finitemask0]
C_NEE_DJFclean0 = C_NEE_DJF_tot[finitemask0]

finitemask2 = np.isfinite(C_NEE_DJFclean0)
C_NEE_DJFclean1 = C_NEE_DJFclean0[finitemask2]
Fluxtower_NEE_DJFclean1 = Fluxtower_NEE_DJFclean0[finitemask2]

Huber_tot_DJF_C_NEE_slps=[]
Huber_tot_DJF_C_NEE_ints=[]
Huber_tot_DJF_C_NEE_R2=[]

#try bootstrapping
indx_list=list(range(0,len(C_NEE_DJFclean1)))
for i in range(1,1001):
    #sub selection of points
    NEE_indx=np.random.choice(indx_list,size=len(C_NEE_DJFclean1))

    Huber_model = linear_model.HuberRegressor(fit_intercept=True)
    Huber_fit=Huber_model.fit((Fluxtower_NEE_DJFclean1[NEE_indx]).reshape(-1,1),C_NEE_DJFclean1[NEE_indx])
    H_m=Huber_fit.coef_
    H_c=Huber_fit.intercept_
    x_accpt, y_accpt = Fluxtower_NEE_DJFclean1, C_NEE_DJFclean1
    y_predict = H_m * x_accpt + H_c
    H_R2=r2_score(y_accpt, y_predict)
    Huber_tot_DJF_C_NEE_slps.append(H_m)
    Huber_tot_DJF_C_NEE_ints.append(H_c)
    Huber_tot_DJF_C_NEE_R2.append(H_R2)
    
print('Original SMUrF DJF slope: '+str(np.round(np.nanmean(Huber_tot_DJF_C_NEE_slps),3))+' +/- '+str(np.round(np.nanstd(Huber_tot_DJF_C_NEE_slps),3)))
print('Original SMUrF DJF intercept: '+str(np.round(np.nanmean(Huber_tot_DJF_C_NEE_ints),3))+' +/- '+str(np.round(np.nanstd(Huber_tot_DJF_C_NEE_ints),3)))

y_predict = np.nanmean(Huber_tot_DJF_C_NEE_slps) * x_accpt + np.nanmean(Huber_tot_DJF_C_NEE_ints)
Huber_DJF_C_NEE_R2=r2_score(y_accpt, y_predict)
print('Original SMUrF DJF R^2: '+str(np.round(np.nanmean(Huber_DJF_C_NEE_R2),3)))


# In[39]:


##Fit SMUrF data to fluxtower data using bootstrapped Huber fit (with downscaling & fluxtower fix)
          
finitemask0 = np.isfinite(Fluxtower_NEE_DJF_tot)
Fluxtower_NEE_DJFclean0 = Fluxtower_NEE_DJF_tot[finitemask0]
S_NEE_DJFclean0 = S_NEE_DJF_tot[finitemask0]

finitemask2 = np.isfinite(S_NEE_DJFclean0)
S_NEE_DJFclean1 = S_NEE_DJFclean0[finitemask2]
Fluxtower_NEE_DJFclean1 = Fluxtower_NEE_DJFclean0[finitemask2]

Huber_tot_DJF_S_NEE_slps=[]
Huber_tot_DJF_S_NEE_ints=[]
Huber_tot_DJF_S_NEE_R2=[]

#try bootstrapping
indx_list=list(range(0,len(S_NEE_DJFclean1)))
for i in range(1,1001):
    #sub selection of points
    NEE_indx=np.random.choice(indx_list,size=len(S_NEE_DJFclean1))

    Huber_model = linear_model.HuberRegressor(fit_intercept=True)
    Huber_fit=Huber_model.fit((Fluxtower_NEE_DJFclean1[NEE_indx]).reshape(-1,1),S_NEE_DJFclean1[NEE_indx])
    H_m=Huber_fit.coef_
    H_c=Huber_fit.intercept_
    x_accpt, y_accpt = Fluxtower_NEE_DJFclean1, S_NEE_DJFclean1
    y_predict = H_m * x_accpt + H_c
    H_R2=r2_score(y_accpt, y_predict)
    Huber_tot_DJF_S_NEE_slps.append(H_m)
    Huber_tot_DJF_S_NEE_ints.append(H_c)
    Huber_tot_DJF_S_NEE_R2.append(H_R2)
    
print('Updated SMUrF DJF slope: '+str(np.round(np.nanmean(Huber_tot_DJF_S_NEE_slps),3))+' +/- '+str(np.round(np.nanstd(Huber_tot_DJF_S_NEE_slps),3)))
print('Updated SMUrF DJF intercept: '+str(np.round(np.nanmean(Huber_tot_DJF_S_NEE_ints),3))+' +/- '+str(np.round(np.nanstd(Huber_tot_DJF_S_NEE_ints),3)))

y_predict = np.nanmean(Huber_tot_DJF_S_NEE_slps) * x_accpt + np.nanmean(Huber_tot_DJF_S_NEE_ints)
Huber_DJF_S_NEE_R2=r2_score(y_accpt, y_predict)
print('Updated SMUrF DJF R^2: '+str(np.round(np.nanmean(Huber_DJF_S_NEE_R2),3)))


# In[45]:


# with downscaling & fluxtower fix

plt.style.use('tableau-colorblind10')

plt.rc('font',size=21.5)
fig, ax = plt.subplots(1,4,sharex=True,sharey=True,figsize=(24,6))
ax[0].set_xlim(-69,25)
ax[0].set_ylim(-69,25)

ax[0].axvline(0,linestyle=(0, (3, 2, 1, 2, 1, 2)),c='k')
ax[0].axhline(0,linestyle=(0, (3, 2, 1, 2, 1, 2)),c='k')
ax[0].scatter(Fluxtower_NEE_MAM_tot,C_NEE_MAM_tot,s=5)
ax[0].scatter(Fluxtower_NEE_MAM_tot,S_NEE_MAM_tot,s=5)
ax[0].plot(line1_1,func2(line1_1,np.nanmean(Huber_tot_MAM_C_NEE_slps),np.nanmean(Huber_tot_MAM_C_NEE_ints)),linestyle='--',label=str(np.round(np.nanmean(Huber_tot_MAM_C_NEE_slps),2))+'$\cdot$x+'+str(np.round(np.nanmean(Huber_tot_MAM_C_NEE_ints),2))+', R$^2$ = '+str(np.round(Huber_MAM_C_NEE_R2,2)),c='k', path_effects=[pe.Stroke(linewidth=5, foreground='#006BA4'), pe.Normal()])
ax[0].plot(line1_1,func2(line1_1,np.nanmean(Huber_tot_MAM_S_NEE_slps),np.nanmean(Huber_tot_MAM_S_NEE_ints)),linestyle='-.',label=str(np.round(np.nanmean(Huber_tot_MAM_S_NEE_slps),2))+'$\cdot$x+'+str(np.round(np.nanmean(Huber_tot_MAM_S_NEE_ints),2))+', R$^2$ = '+str(np.round(Huber_MAM_S_NEE_R2,2)),c='k', path_effects=[pe.Stroke(linewidth=5, foreground='#FF800E'), pe.Normal()])
ax[0].plot(line1_1,line1_1,linestyle=':',c='k')

ax[0].legend(loc='lower center')
ax[0].set_title('Spring')

ax[0].axvline(0,linestyle=(0, (3, 2, 1, 2, 1, 2)),c='k')
ax[0].axhline(0,linestyle=(0, (3, 2, 1, 2, 1, 2)),c='k')
ax[1].scatter(Fluxtower_NEE_JJA_tot,C_NEE_JJA_tot,s=5)
ax[1].scatter(Fluxtower_NEE_JJA_tot,S_NEE_JJA_tot,s=5)
ax[1].plot(line1_1,func2(line1_1,np.nanmean(Huber_tot_JJA_C_NEE_slps),np.nanmean(Huber_tot_JJA_C_NEE_ints)),linestyle='--',label=str(np.round(np.nanmean(Huber_tot_JJA_C_NEE_slps),2))+'$\cdot$x+'+str(np.round(np.nanmean(Huber_tot_JJA_C_NEE_ints),2))+', R$^2$ = '+str(np.round(Huber_JJA_C_NEE_R2,2)),c='k', path_effects=[pe.Stroke(linewidth=5, foreground='#006BA4'), pe.Normal()])
ax[1].plot(line1_1,func2(line1_1,np.nanmean(Huber_tot_JJA_S_NEE_slps),np.nanmean(Huber_tot_JJA_S_NEE_ints)),linestyle='-.',label=str(np.round(np.nanmean(Huber_tot_JJA_S_NEE_slps),2))+'$\cdot$x+'+str(np.round(np.nanmean(Huber_tot_JJA_S_NEE_ints),2))+', R$^2$ = '+str(np.round(Huber_JJA_S_NEE_R2,2)),c='k', path_effects=[pe.Stroke(linewidth=5, foreground='#FF800E'), pe.Normal()])

ax[1].plot(line1_1,line1_1,linestyle=':',c='k')
ax[1].legend(loc='lower center')

ax[1].set_title('Summer')

ax[1].axvline(0,linestyle=(0, (3, 2, 1, 2, 1, 2)),c='k')
ax[1].axhline(0,linestyle=(0, (3, 2, 1, 2, 1, 2)),c='k')
ax[2].scatter(Fluxtower_NEE_SON_tot,C_NEE_SON_tot,s=5)
ax[2].scatter(Fluxtower_NEE_SON_tot,S_NEE_SON_tot,s=5)
ax[2].plot(line1_1,func2(line1_1,np.nanmean(Huber_tot_SON_C_NEE_slps),np.nanmean(Huber_tot_SON_C_NEE_ints)),linestyle='--',label=str(np.round(np.nanmean(Huber_tot_SON_C_NEE_slps),2))+'$\cdot$x - '+str(np.round(-np.nanmean(Huber_tot_SON_C_NEE_ints),2))+', R$^2$ = '+str(np.round(Huber_SON_C_NEE_R2,2)),c='k', path_effects=[pe.Stroke(linewidth=5, foreground='#006BA4'), pe.Normal()])
ax[2].plot(line1_1,func2(line1_1,np.nanmean(Huber_tot_SON_S_NEE_slps),np.nanmean(Huber_tot_SON_S_NEE_ints)),linestyle='-.',label=str(np.round(np.nanmean(Huber_tot_SON_S_NEE_slps),2))+'$\cdot$x+'+str(np.round(np.nanmean(Huber_tot_SON_S_NEE_ints),2))+', R$^2$ = '+str(np.round(Huber_SON_S_NEE_R2,2)),c='k', path_effects=[pe.Stroke(linewidth=5, foreground='#FF800E'), pe.Normal()])

ax[2].plot(line1_1,line1_1,linestyle=':',c='k')
ax[2].legend(loc='lower center')
ax[2].set_title('Autumn')

ax[2].axvline(0,linestyle=(0, (3, 2, 1, 2, 1, 2)),c='k')
ax[2].axhline(0,linestyle=(0, (3, 2, 1, 2, 1, 2)),c='k')
ax[3].scatter(-100,-100,label='Original SMUrF',c='#006BA4')
ax[3].scatter(-100,-100,label='Updated SMUrF',c='#FF800E')
ax[3].scatter(Fluxtower_NEE_DJF_tot,C_NEE_DJF_tot,s=5)
ax[3].scatter(Fluxtower_NEE_DJF_tot,S_NEE_DJF_tot,s=5)

ax[3].axvline(0,linestyle=(0, (3, 2, 1, 2, 1, 2)),c='k')
ax[3].axhline(0,linestyle=(0, (3, 2, 1, 2, 1, 2)),c='k')
ax[3].plot(line1_1,line1_1,linestyle=':',c='k')
ax[3].legend(loc='lower right',fontsize=23)
ax[3].set_title('Winter')
ax[0].set_ylabel('Modelled NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')

ax[0].text(-67,17.5,'(e)',c='k',fontsize=26)
ax[1].text(-67,17.5,'(f)',c='k',fontsize=26)
ax[2].text(-67,17.5,'(g)',c='k',fontsize=26)
ax[3].text(-67,17.5,'(h)',c='k',fontsize=26)

fig.text(0.5, 0.01, 'Flux Tower NEE ($\mu$mol m$^{-2}$ s$^{-1}$)', ha='center')
fig.subplots_adjust(hspace=0,wspace=0)

# *** UNCOMMENT TO SAVE FIGURE (CHANGE FILENAME) ***
plt.savefig('Seasonal_Original_fixed_Updated_SMUrF_vs_fixed2_fluxtower_Huber_fit_NEE_0_lines_larger_font_cb_friendly_labelled.pdf',bbox_inches='tight')
plt.savefig('Seasonal_Original_fixed_Updated_SMUrF_vs_fixed2_fluxtower_Huber_fit_NEE_0_lines_larger_font_cb_friendly_labelled.png',bbox_inches='tight')
fig.show()


# In[ ]:




