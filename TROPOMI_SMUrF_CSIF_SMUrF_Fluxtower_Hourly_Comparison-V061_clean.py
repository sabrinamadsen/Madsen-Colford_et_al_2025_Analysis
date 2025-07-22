#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This code compares biogenic fluxes estimated by the Original and Updated SMUrF models to those
# estimated by three eddy-covariance flux towers in Southern Ontario: Borden Forest Mixed Deciduous,
# Turkey Point 1939 Pine (TP39), and Turkey Point Deciduous (TPD). 

# The code loads in the data
# It then generates timeseries of daily-averaged flux tower & SMUrF (original and updated) NEE, GPP & Reco,

# The code also applies bootstrapped Huber fits to the hourly NEE fluxes estimated by SMUrF and the 
# non-gapfilled NEE from all 3 flux towers, for both the original and modified SMUrF

# This code generates figures 2 c & d of Madsen-Colford et al. 2025

# Comments including '***' indicate parts of the code that may need to be changed by the user (e.g. path/filenames)


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
from datetime import datetime, timedelta
#from datetime import datetime as dt
import matplotlib.patheffects as pe
from sklearn import linear_model #for robust fitting
from sklearn.metrics import r2_score, mean_squared_error #for analyzing robust fits
import matplotlib.colors as clrs #for log color scale


# In[2]:


# Load in the first Reco file to extract the start time of 2018, in seconds since 1970, & convert to days since 1970

# *** CHANGE PATH & FILENAME ***
g=Dataset('E:/Research/SMUrF/output2018_CSIF_V061/easternCONUS/daily_mean_Reco_neuralnet/era5/2018/daily_mean_Reco_uncert_easternCONUS_20180101.nc')
start_of_year=g.variables['time'][0]/3600/24-1 #convert seconds since 1970 to days (minus one)
g.close()


# In[3]:


# Load in the original SMUrF fluxes

# *** CHANGE PATH ***
C_path = 'E:/Research/SMUrF/output2018_CSIF_V061/easternCONUS/hourly_flux_era5/'
# *** CHANGE FILENAME ***
C_fn = 'hrly_mean_GPP_Reco_NEE_easternCONUS_2018' #filename WITHOUT the month

C_time=[]
C_Reco=[]
C_NEE=[]
C_GPP=[]
C_lats=[]
C_lons=[]
for j in range(1,13):
    try:
        if j<10:
            f=Dataset(C_path+C_fn+'0'+str(j)+'.nc')
        else:
            # 2018:
            f=Dataset(C_path+C_fn+str(j)+'.nc')
        if len(C_time)==0: #if it is the first month, make arrays of data & save lat/lon
            C_lats=f.variables['lat'][:]
            C_lons=f.variables['lon'][:]
            C_Reco=f.variables['Reco_mean'][:]
            C_GPP=f.variables['GPP_mean'][:]
            C_NEE=f.variables['NEE_mean'][:]
            C_time=f.variables['time'][:]/24/3600-start_of_year-5/24 #convert seconds since 1970 to days and subtract start of year and adjust to local time
        else: #otherwise append this month's data to the arrays
            C_Reco=np.concatenate((C_Reco,f.variables['Reco_mean'][:]),axis=0)
            C_GPP=np.concatenate((C_GPP,f.variables['GPP_mean'][:]),axis=0)
            C_NEE=np.concatenate((C_NEE,f.variables['NEE_mean'][:]),axis=0)
            C_time=np.concatenate((C_time,(f.variables['time'][:]/24/3600-start_of_year-5/24)),axis=0)
        f.close()
    except FileNotFoundError:
        pass
    
# Set fill values to NaN
C_Reco[C_Reco==-999]=np.nan
C_NEE[C_NEE==-999]=np.nan
C_GPP[C_GPP==-999]=np.nan


# In[4]:


# Select original SMUrF pixel that lands over the Borden Forest flux tower
C_GPP_array=np.zeros(8765)*np.nan
C_NEE_array=np.zeros(8765)*np.nan
C_Reco_array=np.zeros(8765)*np.nan
C_time_array=np.zeros(8765)*np.nan
for i in range(len(C_GPP[:,0,0])):
    C_time_array[i]=C_time[i]
    C_GPP_array[i]=C_GPP[i,36,15]
    C_NEE_array[i]=C_NEE[i,36,15]
    C_Reco_array[i]=C_Reco[i,36,15]


# In[ ]:





# In[5]:


# Load in Borden Forest fluxes

#*** CHANGE PATH & FILENAME ***
Borden_Fluxes=pd.read_csv('/Users/kitty/Documents/Research/SIF/Flux_Tower/2018_NEP_GPP_Borden.csv', index_col=0)


# In[6]:


Borden_dates_fluxes=np.zeros([17520])*np.nan
Borden_NEEgf_fluxes=np.zeros([17520])*np.nan
Borden_NEE_fluxes=np.zeros([17520])*np.nan
Borden_Rgf_fluxes=np.zeros([17520])*np.nan
Borden_GEPgf_fluxes=np.zeros([17520])*np.nan
for i in range(0,17520):
    Borden_dates_fluxes[i]=Borden_Fluxes.iat[i,0]-5/24 #adjust to local time
    Borden_NEEgf_fluxes[i]=-Borden_Fluxes.iat[i,5] #NEE (gap filled)
    Borden_NEE_fluxes[i]=-Borden_Fluxes.iat[i,1] # NEE (non gap-filled)
    Borden_Rgf_fluxes[i]=Borden_Fluxes.iat[i,6] #Reco
    Borden_GEPgf_fluxes[i]=Borden_Fluxes.iat[i,7] #GPP
del Borden_Fluxes


# In[ ]:





# In[7]:


#Take daily average of fluxtower fluxes
days_of_year=np.arange(1,366)+0.5 #Make the day of year centered on the middle of the day (i.e. noon)
Borden_daily_NEE=np.zeros(365)*np.nan
Borden_daily_NEEgf=np.zeros(365)*np.nan
Borden_daily_GPPgf=np.zeros(365)*np.nan
Borden_daily_Rgf=np.zeros(365)*np.nan
date=0
daily_NEE=[]
daily_NEEgf=[]
daily_GPPgf=[]
daily_Rgf=[]
for i in range(len(Borden_dates_fluxes)):
    if np.round(Borden_dates_fluxes[i],4)>=1:
        if np.round(Borden_dates_fluxes[i],4)<365: #The last day is not complete (in local time) so skip it in daily average
            # If it is not the last day, append the data to the flux lists
            daily_NEE.append(Borden_NEE_fluxes[i])
            daily_NEEgf.append(Borden_NEEgf_fluxes[i])
            daily_GPPgf.append(Borden_GEPgf_fluxes[i])
            daily_Rgf.append(Borden_Rgf_fluxes[i])
            #If it is the last hour of the day, average the lists of fluxes and save to an array & empty the list
            if np.floor(np.round(Borden_dates_fluxes[i],4))<np.floor(np.round(Borden_dates_fluxes[i+1],4)):
                Borden_daily_NEE[date]=np.mean(daily_NEE)

                Borden_daily_NEEgf[date]=np.mean(daily_NEEgf)
                Borden_daily_GPPgf[date]=np.mean(daily_GPPgf)
                Borden_daily_Rgf[date]=np.mean(daily_Rgf)

                date+=1
                daily_NEE=[]
                daily_NEEgf=[]
                daily_GPPgf=[]
                daily_Rgf=[]


# In[ ]:





# In[8]:


# Define a straight line and a linear function for plotting fits
line1_1=np.arange(-100,100)

def func2(x,m,b):
    return m*x+b


# In[9]:


#Define an array for the date & time of year (in decimal day of year)
date_array=np.arange(np.nanmin(C_time),366,1/24)


# In[10]:


#Average Original SMUrF to daily resolution

C_daily_NEE=np.zeros(365)*np.nan
C_daily_GPP=np.zeros(365)*np.nan
C_daily_R=np.zeros(365)*np.nan
date=0
daily_NEE_C=[]
daily_GPP_C=[]
daily_R_C=[]
for i in range(len(date_array)):
    if np.round(date_array[i],4)>=1:
        if date+1>=365:
            daily_NEE_C.append(C_NEE_array[i])
            daily_GPP_C.append(C_GPP_array[i])
            daily_R_C.append(C_Reco_array[i])
            if i==len(date_array)-1:
                C_daily_NEE[date]=np.mean(daily_NEE_C)
                C_daily_GPP[date]=np.mean(daily_GPP_C)
                C_daily_R[date]=np.mean(daily_R_C)
                date+=1
        else:
            daily_NEE_C.append(C_NEE_array[i])
            daily_GPP_C.append(C_GPP_array[i])
            daily_R_C.append(C_Reco_array[i])
            if np.floor(np.round(date_array[i],4))<np.floor(np.round(date_array[i+1],4)):
                C_daily_NEE[date]=np.mean(daily_NEE_C)
                C_daily_GPP[date]=np.mean(daily_GPP_C)
                C_daily_R[date]=np.mean(daily_R_C)
            
                date+=1
                daily_NEE_C=[]
                daily_GPP_C=[]
                daily_R_C=[]


# In[11]:


# Take hourly average of Borden flux tower data (current half-hour and the next half-hour)

Borden_NEE=np.zeros(np.shape(C_GPP_array))*np.nan
Borden_GEPgf=np.zeros(np.shape(C_GPP_array))*np.nan
Borden_NEEgf=np.zeros(np.shape(C_GPP_array))*np.nan
Borden_Rgf=np.zeros(np.shape(C_GPP_array))*np.nan

for i in range(np.int(len(Borden_dates_fluxes)/2)):
    with np.errstate(invalid='ignore'):
        Borden_NEE[i]=np.nanmean([Borden_NEE_fluxes[i*2],Borden_NEE_fluxes[i*2+1]])
        Borden_GEPgf[i]=np.nanmean([Borden_GEPgf_fluxes[i*2],Borden_GEPgf_fluxes[i*2+1]])
        Borden_NEEgf[i]=np.nanmean([Borden_NEEgf_fluxes[i*2],Borden_NEEgf_fluxes[i*2+1]])
        Borden_Rgf[i]=np.nanmean([Borden_Rgf_fluxes[i*2],Borden_Rgf_fluxes[i*2+1]])


# In[ ]:





# In[12]:


# Load in original SMUrF fluxes for 2019

# load in first Reco file to get the time of the first day of the year (since 1970)
g=Dataset('E:/Research/SMUrF/output2019_CSIF_V061/easternCONUS/daily_mean_Reco_neuralnet/era5/2019/daily_mean_Reco_uncert_easternCONUS_20190101.nc')
start_of_year_2019=g.variables['time'][0]/3600/24-1 #convert seconds since 1970 to days (minus one)
g.close()

# *** CHANGE PATH ***
C_path = 'E:/Research/SMUrF/output2019_CSIF_V061/easternCONUS/hourly_flux_era5/'
# *** CHANGE FILENAME ***
C_fn= 'hrly_mean_GPP_Reco_NEE_easternCONUS_2019' # filename without month (added in loop)

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
        if len(C_time_2019)==0: #if it is the first month save the flux data to an arrays
            C_lats_2019=f.variables['lat'][:]
            C_lons_2019=f.variables['lon'][:]
            C_Reco_2019=f.variables['Reco_mean'][:]
            C_GPP_2019=f.variables['GPP_mean'][:]
            C_NEE_2019=f.variables['NEE_mean'][:]
            C_time_2019=f.variables['time'][:]/24/3600-start_of_year_2019-5/24 #convert seconds since 1970 to days and subtract start of year and adjust to local time
        else: #otherwise add to the arrays
            C_Reco_2019=np.concatenate((C_Reco_2019,f.variables['Reco_mean'][:]),axis=0)
            C_GPP_2019=np.concatenate((C_GPP_2019,f.variables['GPP_mean'][:]),axis=0)
            C_NEE_2019=np.concatenate((C_NEE_2019,f.variables['NEE_mean'][:]),axis=0)
            C_time_2019=np.concatenate((C_time_2019,(f.variables['time'][:]/24/3600-start_of_year_2019-5/24)),axis=0)
        f.close()
    except FileNotFoundError:
        pass

# Replace fill values with NaN
C_Reco_2019[C_Reco_2019==-999]=np.nan
C_NEE_2019[C_NEE_2019==-999]=np.nan
C_GPP_2019[C_GPP_2019==-999]=np.nan


# In[ ]:





# ### Do the same for TP39

# In[13]:


# Load in TP39 2018 flux tower data

TP39_2018_data=pd.read_csv('C:/Users/kitty/Documents/Research/SIF/Flux_Tower/Turkey_Point/TP39_HH_2018.csv', usecols=[0,1,2,77,78,79]) #header=1


# In[14]:


TP39_2018_dates=np.zeros([17520])*np.nan
TP39_2018_NEE=np.zeros([17520])*np.nan
TP39_2018_NEE2=np.zeros([17520])*np.nan
TP39_2018_GPP=np.zeros([17520])*np.nan
TP39_2018_R=np.zeros([17520])*np.nan
for i in range(17520):
    if 201801010000<=TP39_2018_data.iat[i,0]<201901010000:
        TP39_2018_dates[i]=datetime.strptime(str(int(TP39_2018_data.iat[i,0])),'%Y%m%d%H%M').timetuple().tm_yday+(float(str(TP39_2018_data.iat[i,0])[8:10])+float(str(TP39_2018_data.iat[i,0])[10:12])/60)/24
        #check that the value is greater than -9999 (value for empty measurements)
        if TP39_2018_data.iat[i,2]>-9999:
            TP39_2018_NEE2[i]=TP39_2018_data.iat[i,2] # save the NEE (non-gapfilled) data
        if TP39_2018_data.iat[i,5]>-9999:
            TP39_2018_NEE[i]=TP39_2018_data.iat[i,5] # save the NEE (gap-filled) 
        if TP39_2018_data.iat[i,3]>-9999:
            TP39_2018_GPP[i]=TP39_2018_data.iat[i,3] # save the GPP
        if TP39_2018_data.iat[i,4]>-9999:
            TP39_2018_R[i]=TP39_2018_data.iat[i,4] # save the Reco


# In[15]:


# Select Original SMUrF pixel over TP39 flux tower
C_GPP_TP39_array=np.zeros(8765)*np.nan
C_NEE_TP39_array=np.zeros(8765)*np.nan
C_Reco_TP39_array=np.zeros(8765)*np.nan

for i in range(len(C_GPP[:,0,0])):
    C_GPP_TP39_array[i]=C_GPP[i,4,6]
    C_NEE_TP39_array[i]=C_NEE[i,4,6]
    C_Reco_TP39_array[i]=C_Reco[i,4,6]


# In[16]:


#Take the daily average of TP39 flux tower data

days_of_year=np.arange(1,366)+0.5
TP39_daily_NEE=np.zeros(365)*np.nan
TP39_daily_NEEgf=np.zeros(365)*np.nan
TP39_daily_GPP=np.zeros(365)*np.nan
TP39_daily_R=np.zeros(365)*np.nan
date=0
daily_NEE=[]
daily_NEEgf=[]
daily_GPP=[]
daily_R=[]
for i in range(len(TP39_2018_dates)):
    if TP39_2018_dates[i]>=1:
        if date+1>=365:
            daily_NEE.append(TP39_2018_NEE2[i])
            daily_NEEgf.append(TP39_2018_NEE[i])
            daily_GPP.append(TP39_2018_GPP[i])
            daily_R.append(TP39_2018_R[i])
            if i==len(TP39_2018_dates)-1:
                TP39_daily_NEE[date]=np.mean(daily_NEE)
                TP39_daily_NEEgf[date]=np.mean(daily_NEEgf)
                TP39_daily_GPP[date]=np.mean(daily_GPP)
                TP39_daily_R[date]=np.mean(daily_R)
                
                date+=1
        else:
            daily_NEE.append(TP39_2018_NEE2[i])
            daily_NEEgf.append(TP39_2018_NEE[i])
            daily_GPP.append(TP39_2018_GPP[i])
            daily_R.append(TP39_2018_R[i])
            if np.floor(np.round(TP39_2018_dates[i],4))<np.floor(np.round(TP39_2018_dates[i+1],4)):
                TP39_daily_NEE[date]=np.mean(daily_NEE)
                TP39_daily_NEEgf[date]=np.mean(daily_NEEgf)
                TP39_daily_GPP[date]=np.mean(daily_GPP)
                TP39_daily_R[date]=np.mean(daily_R)

                date+=1
                daily_NEE=[]
                daily_NEEgf=[]
                daily_GPP=[]
                daily_R=[]


# In[17]:


#Convert half-hourly to hourly data
TP39_GPP=np.zeros(np.shape(C_GPP_TP39_array))*np.nan
TP39_NEEgf=np.zeros(np.shape(C_GPP_TP39_array))*np.nan
TP39_NEE=np.zeros(np.shape(C_GPP_TP39_array))*np.nan
TP39_R=np.zeros(np.shape(C_GPP_TP39_array))*np.nan
for i in range(len(date_array)-5):
    TP39_GPP[i+5]=np.nanmean([TP39_2018_GPP[i*2],TP39_2018_GPP[i*2+1]])
    TP39_NEEgf[i+5]=np.nanmean([TP39_2018_NEE[i*2],TP39_2018_NEE[i*2+1]])
    TP39_NEE[i+5]=np.nanmean([TP39_2018_NEE2[i*2],TP39_2018_NEE2[i*2+1]])
    TP39_R[i+5]=np.nanmean([TP39_2018_R[i*2],TP39_2018_R[i*2+1]])


# In[18]:


# Take the daily average of the original SMUrF 2018 fluxes over TP39

C_daily_NEE_TP39=np.zeros(365)*np.nan
C_daily_GPP_TP39=np.zeros(365)*np.nan
C_daily_R_TP39=np.zeros(365)*np.nan

date=0
daily_NEE_TP39_S=[]
daily_GPP_TP39_S=[]
daily_R_TP39_S=[]
for i in range(len(date_array)):
    if np.round(date_array[i],4)>=1:
        if np.round(date_array[i],4)+1>=365:
            daily_NEE_TP39_S.append(C_NEE_TP39_array[i])
            daily_GPP_TP39_S.append(C_GPP_TP39_array[i])
            daily_R_TP39_S.append(C_Reco_TP39_array[i])
            if i==len(date_array)-1:
                C_daily_NEE_TP39[date]=np.mean(daily_NEE_TP39_S)
                C_daily_GPP_TP39[date]=np.mean(daily_GPP_TP39_S)
                C_daily_R_TP39[date]=np.mean(daily_R_TP39_S)
                date+=1
        else:
            daily_NEE_TP39_S.append(C_NEE_TP39_array[i])
            daily_GPP_TP39_S.append(C_GPP_TP39_array[i])
            daily_R_TP39_S.append(C_Reco_TP39_array[i])
            if np.floor(np.round(date_array[i],4))<np.floor(np.round(date_array[i+1],4)):
                C_daily_NEE_TP39[date]=np.mean(daily_NEE_TP39_S)
                C_daily_GPP_TP39[date]=np.mean(daily_GPP_TP39_S)
                C_daily_R_TP39[date]=np.mean(daily_R_TP39_S)
                date+=1
                daily_NEE_TP39_S=[]
                daily_GPP_TP39_S=[]
                daily_R_TP39_S=[]


# In[ ]:





# In[19]:


#Load in TP39 2019 flux tower data

#*** CHANGE PATH & FILENAME ***
TP39_2019_data=pd.read_csv('C:/Users/kitty/Documents/Research/SIF/Flux_Tower/Turkey_Point/TP39_HH_2019.csv', usecols=[0,1,2,77,78,79]) #header=1


# In[20]:


TP39_2019_dates=np.zeros([17520])*np.nan
TP39_2019_NEE=np.zeros([17520])*np.nan
TP39_2019_NEE2=np.zeros([17520])*np.nan
TP39_2019_GPP=np.zeros([17520])*np.nan
TP39_2019_R=np.zeros([17520])*np.nan

for i in range(17520):
    if 201901010000<=TP39_2019_data.iat[i,0]<202001010000:
        TP39_2019_dates[i]=datetime.strptime(str(int(TP39_2019_data.iat[i,0])),'%Y%m%d%H%M').timetuple().tm_yday+(float(str(TP39_2019_data.iat[i,0])[8:10])+float(str(TP39_2019_data.iat[i,0])[10:12])/60)/24
        #check that the value is greater than -9999 (value for empty measurements)
        if TP39_2019_data.iat[i,2]>-9999:
            TP39_2019_NEE2[i]=TP39_2019_data.iat[i,2] # save the NEE (non-filled)
        if TP39_2019_data.iat[i,5]>-9999:
            TP39_2019_NEE[i]=TP39_2019_data.iat[i,5] # save the gap-filled NEE 
        if TP39_2019_data.iat[i,3]>-9999:
            TP39_2019_GPP[i]=TP39_2019_data.iat[i,3] # save the GPP
        if TP39_2019_data.iat[i,4]>-9999:
            TP39_2019_R[i]=TP39_2019_data.iat[i,4] # save the Reco


# In[21]:


#Save Original SMUrF 2019 fluxes over TP39
C_GPP_TP39_2019_array=np.zeros(8765)*np.nan
C_NEE_TP39_2019_array=np.zeros(8765)*np.nan
C_Reco_TP39_2019_array=np.zeros(8765)*np.nan
C_time_2019_array=np.zeros(8765)*np.nan
for i in range(len(C_GPP_2019[:,0,0])):
    C_time_2019_array[i]=C_time[i]
    C_GPP_TP39_2019_array[i]=C_GPP_2019[i,4,6]
    C_NEE_TP39_2019_array[i]=C_NEE_2019[i,4,6]
    C_Reco_TP39_2019_array[i]=C_Reco_2019[i,4,6]


# In[22]:


#Filter out erroneous NEE values between doy 195 and 198

plt.rc('font',size=10)
## ***Optional: uncomment to visualize erroneous values
#with np.errstate(invalid='ignore'):
#    plt.figure()
#    plt.xlim(193,200)
#    plt.scatter(TP39_2019_dates,TP39_2019_NEE2,label='TP39 NEE')
#    plt.scatter(TP39_2019_dates[(TP39_2019_dates>195.04-5/24) & (TP39_2019_dates<198.6-5/24)],TP39_2019_NEE2[(TP39_2019_dates>195.04-5/24) & (TP39_2019_dates<198.6-5/24)],label='Erroneous TP39 NEE')
#    plt.scatter(C_time_2019_array,C_NEE_TP39_2019_array,marker='*',label='SMUrF NEE')
#    plt.legend()
#    plt.xlabel('Day of year, 2019')
#    plt.ylabel('NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
#    plt.title('Erroneous TP39 flux tower NEE values')
#    plt.show()
    
#    plt.figure()
#    plt.xlim(193,200)
#    plt.scatter(TP39_2019_dates,TP39_2019_NEE,label='TP39 NEEgf')
#    plt.scatter(TP39_2019_dates[(TP39_2019_dates>195.04-5/24) & (TP39_2019_dates<198.6-5/24)],TP39_2019_NEE[(TP39_2019_dates>195.04-5/24) & (TP39_2019_dates<198.6-5/24)],label='Erroneous TP39 NEEgf')
#    plt.scatter(C_time_2019_array,C_NEE_TP39_2019_array,marker='*',label='SMUrF NEE')
#    plt.legend()
#    plt.xlabel('Day of year, 2019')
#    plt.ylabel('NEEgf ($\mu$mol m$^{-2}$ s$^{-1}$)')
#    plt.title('Erroneous TP39 flux tower NEEgf values')
#    plt.show()
    
#    plt.figure()
#    plt.xlim(193,200)
#    plt.scatter(TP39_2019_dates,TP39_2019_GPP,label='TP39 GPP')
#    plt.scatter(TP39_2019_dates[(TP39_2019_dates>195.04-5/24) & (TP39_2019_dates<198.6-5/24)],TP39_2019_GPP[(TP39_2019_dates>195.04-5/24) & (TP39_2019_dates<198.6-5/24)],label='Erroneous TP39 GPP')
#    plt.scatter(C_time_2019_array,C_GPP_TP39_2019_array,marker='*',label='SMUrF GPP')
#    plt.legend()
#    plt.xlabel('Day of year, 2019')
#    plt.ylabel('GPP ($\mu$mol m$^{-2}$ s$^{-1}$)')
#    plt.title('Erroneous TP39 flux tower GPP values')
#    plt.show()
    
#    plt.figure()
#    plt.xlim(193,200)
#    plt.scatter(TP39_2019_dates,TP39_2019_R,label='TP39 Reco')
#    plt.scatter(TP39_2019_dates[(TP39_2019_dates>195.04-5/24) & (TP39_2019_dates<198.6-5/24)],TP39_2019_R[(TP39_2019_dates>195.04-5/24) & (TP39_2019_dates<198.6-5/24)],label='Erroneous TP39 Reco')
#    plt.scatter(C_time_2019_array,C_Reco_TP39_2019_array,marker='*',label='SMUrF Reco')
#    plt.legend()
#    plt.xlabel('Day of year, 2019')
#    plt.ylabel('Reco ($\mu$mol m$^{-2}$ s$^{-1}$)')
#    plt.title('Erroneous TP39 flux tower Reco values')
#    plt.show()
# ***

with np.errstate(invalid='ignore'):
    TP39_2019_NEE2[(TP39_2019_dates>195.04-5/24) & (TP39_2019_dates<198.6-5/24)]= np.nan
    TP39_2019_NEE[(TP39_2019_dates>195.04-5/24) & (TP39_2019_dates<198.6-5/24)]= np.nan
    TP39_2019_GPP[(TP39_2019_dates>195.04-5/24) & (TP39_2019_dates<198.6-5/24)]= np.nan
    TP39_2019_R[(TP39_2019_dates>195.04-5/24) & (TP39_2019_dates<198.6-5/24)]= np.nan


# In[ ]:





# In[23]:


TP39_2019_hrly_GPP=np.zeros(np.shape(C_GPP_TP39_2019_array))*np.nan
TP39_2019_hrly_NEEgf=np.zeros(np.shape(C_GPP_TP39_2019_array))*np.nan
TP39_2019_hrly_NEE=np.zeros(np.shape(C_GPP_TP39_2019_array))*np.nan
TP39_2019_hrly_R=np.zeros(np.shape(C_GPP_TP39_2019_array))*np.nan
for i in range(len(date_array)-5):
    with np.errstate(invalid='ignore'):
        TP39_2019_hrly_GPP[i+5]=np.nanmean([TP39_2019_GPP[i*2],TP39_2019_GPP[i*2+1]])
        TP39_2019_hrly_NEEgf[i+5]=np.nanmean([TP39_2019_NEE[i*2],TP39_2019_NEE[i*2+1]])
        TP39_2019_hrly_NEE[i+5]=np.nanmean([TP39_2019_NEE2[i*2],TP39_2019_NEE2[i*2+1]])
        TP39_2019_hrly_R[i+5]=np.nanmean([TP39_2019_R[i*2],TP39_2019_R[i*2+1]])


# In[24]:


#Take daily average of TP39 2019 flux tower data
days_of_year=np.arange(1,366)+0.5
TP39_daily_2019_NEE=np.zeros(365)*np.nan
TP39_daily_2019_NEEgf=np.zeros(365)*np.nan
TP39_daily_2019_GPP=np.zeros(365)*np.nan
TP39_daily_2019_R=np.zeros(365)*np.nan

date=0
daily_NEE=[]
daily_NEEgf=[]
daily_GPP=[]
daily_R=[]
for i in range(len(TP39_2019_dates)):
    if TP39_2019_dates[i]>=1:
        if date+1>=365:
            daily_NEE.append(TP39_2019_NEE2[i])
            daily_NEEgf.append(TP39_2019_NEE[i])
            daily_GPP.append(TP39_2019_GPP[i])
            daily_R.append(TP39_2019_R[i])
            if i==len(TP39_2019_dates)-1:
                TP39_daily_2019_NEE[date]=np.mean(daily_NEE)
                TP39_daily_2019_NEEgf[date]=np.mean(daily_NEEgf)
                TP39_daily_2019_GPP[date]=np.mean(daily_GPP)
                TP39_daily_2019_R[date]=np.mean(daily_R)
                date+=1
        else:
            daily_NEE.append(TP39_2019_NEE2[i])
            daily_NEEgf.append(TP39_2019_NEE[i])
            daily_GPP.append(TP39_2019_GPP[i])
            daily_R.append(TP39_2019_R[i])
            if np.floor(np.round(TP39_2019_dates[i],4))<np.floor(np.round(TP39_2019_dates[i+1],4)):
                TP39_daily_2019_NEE[date]=np.mean(daily_NEE)
                TP39_daily_2019_NEEgf[date]=np.mean(daily_NEEgf)
                TP39_daily_2019_GPP[date]=np.mean(daily_GPP)
                TP39_daily_2019_R[date]=np.mean(daily_R)
                date+=1


# In[ ]:





# In[25]:


C_daily_2019_NEE_TP39=np.zeros(365)*np.nan
C_daily_2019_GPP_TP39=np.zeros(365)*np.nan
C_daily_2019_R_TP39=np.zeros(365)*np.nan

#Take daily average of original SMUrF 2019 fluxes over TP39
date=0
daily_2019_NEE_TP39_C=[]
daily_2019_GPP_TP39_C=[]
daily_2019_R_TP39_C=[]
for i in range(len(date_array)):
    if np.round(date_array[i],4)>=1:
        if np.round(date_array[i],4)+1>=365:
            daily_2019_NEE_TP39_C.append(C_NEE_TP39_2019_array[i])
            daily_2019_GPP_TP39_C.append(C_GPP_TP39_2019_array[i])
            daily_2019_R_TP39_C.append(C_Reco_TP39_2019_array[i])
            if i==len(date_array)-1:
                C_daily_2019_NEE_TP39[date]=np.mean(daily_2019_NEE_TP39_C)
                C_daily_2019_GPP_TP39[date]=np.mean(daily_2019_GPP_TP39_C)
                C_daily_2019_R_TP39[date]=np.mean(daily_2019_R_TP39_C)
                date+=1
        else:
            daily_2019_NEE_TP39_C.append(C_NEE_TP39_2019_array[i])
            daily_2019_GPP_TP39_C.append(C_GPP_TP39_2019_array[i])
            daily_2019_R_TP39_C.append(C_Reco_TP39_2019_array[i])
            if np.floor(np.round(date_array[i],4))<np.floor(np.round(date_array[i+1],4)):
                C_daily_2019_NEE_TP39[date]=np.mean(daily_2019_NEE_TP39_C)
                C_daily_2019_GPP_TP39[date]=np.mean(daily_2019_GPP_TP39_C)
                C_daily_2019_R_TP39[date]=np.mean(daily_2019_R_TP39_C)
                date+=1
                daily_2019_NEE_TP39_C=[]
                daily_2019_GPP_TP39_C=[]
                daily_2019_R_TP39_C=[]


# In[ ]:





# ### Look at TPD

# In[26]:


#Load in TPD flux tower data

# *** CHANGE PATH & FILENAME ***
TPD_2018_data=pd.read_csv('C:/Users/kitty/Documents/Research/SIF/Flux_Tower/Turkey_Point/TPD_HH_2018.csv', usecols=[0,1,2,74,75,76]) #header=1


# In[27]:


TPD_2018_dates=np.zeros([17520])*np.nan
TPD_2018_NEE=np.zeros([17520])*np.nan
TPD_2018_NEE2=np.zeros([17520])*np.nan
TPD_2018_GPP=np.zeros([17520])*np.nan
TPD_2018_R=np.zeros([17520])*np.nan
for i in range(17520):
    if 201801010000<=TPD_2018_data.iat[i,0]<202001010000:
        TPD_2018_dates[i]=datetime.strptime(str(int(TPD_2018_data.iat[i,0])),'%Y%m%d%H%M').timetuple().tm_yday+(float(str(TPD_2018_data.iat[i,0])[8:10])+float(str(TPD_2018_data.iat[i,0])[10:12])/60)/24 #save the current date (and time)
        #check that the value is greater than -9999 (value for empty measurements)
        if TPD_2018_data.iat[i,2]>-9999:
            TPD_2018_NEE2[i]=TPD_2018_data.iat[i,2] # save the NEE value
        if TPD_2018_data.iat[i,5]>-9999:
            TPD_2018_NEE[i]=TPD_2018_data.iat[i,5] # save the NEE value
        if TPD_2018_data.iat[i,3]>-9999:
            TPD_2018_GPP[i]=TPD_2018_data.iat[i,3] # save the NEE value
        if TPD_2018_data.iat[i,4]>-9999:
            TPD_2018_R[i]=TPD_2018_data.iat[i,4] # save the NEE value


# In[28]:


#Select original SMUrF data over TPD

C_GPP_TPD_array=np.zeros(8765)*np.nan
C_NEE_TPD_array=np.zeros(8765)*np.nan
C_Reco_TPD_array=np.zeros(8765)*np.nan

for i in range(len(C_GPP[:,0,0])):
    C_GPP_TPD_array[i]=np.nanmean([C_GPP[i,2,2]])
    C_NEE_TPD_array[i]=np.nanmean([C_NEE[i,2,2]])
    C_Reco_TPD_array[i]=np.nanmean([C_Reco[i,2,2]])


# In[29]:


#Take daily average of TPD flux tower data

days_of_year=np.arange(1,366)+0.5
TPD_daily_NEE=np.zeros(365)*np.nan
TPD_daily_NEEgf=np.zeros(365)*np.nan
TPD_daily_GPP=np.zeros(365)*np.nan
TPD_daily_R=np.zeros(365)*np.nan
date=0
daily_NEE=[]
daily_NEEgf=[]
daily_GPP=[]
daily_R=[]
for i in range(len(TPD_2018_dates)):
    if TPD_2018_dates[i]>=1:
        if date+1>=365:
            daily_NEE.append(TPD_2018_NEE2[i])
            daily_NEEgf.append(TPD_2018_NEE[i])
            daily_GPP.append(TPD_2018_GPP[i])
            daily_R.append(TPD_2018_R[i])
            if i==len(TPD_2018_dates)-1:
                TPD_daily_NEE[date]=np.mean(daily_NEE)
                TPD_daily_NEEgf[date]=np.mean(daily_NEEgf)
                TPD_daily_GPP[date]=np.mean(daily_GPP)
                TPD_daily_R[date]=np.mean(daily_R)
                date+=1
        else:
            daily_NEE.append(TPD_2018_NEE2[i])
            daily_NEEgf.append(TPD_2018_NEE[i])
            daily_GPP.append(TPD_2018_GPP[i])
            daily_R.append(TPD_2018_R[i])
            if np.floor(np.round(TPD_2018_dates[i],4))<np.floor(np.round(TPD_2018_dates[i+1],4)):
                TPD_daily_NEE[date]=np.mean(daily_NEE)
                TPD_daily_NEEgf[date]=np.mean(daily_NEEgf)
                TPD_daily_GPP[date]=np.mean(daily_GPP)
                TPD_daily_R[date]=np.mean(daily_R)

                date+=1
                daily_NEE=[]
                daily_NEEgf=[]
                daily_GPP=[]
                daily_R=[]


# In[30]:


#Take hourly-average of TPD data

TPD_GPP=np.zeros(np.shape(C_GPP_TPD_array))*np.nan
TPD_NEE=np.zeros(np.shape(C_GPP_TPD_array))*np.nan
TPD_NEEgf=np.zeros(np.shape(C_GPP_TPD_array))*np.nan
TPD_R=np.zeros(np.shape(C_GPP_TPD_array))*np.nan
for i in range(len(date_array)-5):
    TPD_GPP[i+5]=np.nanmean([TPD_2018_GPP[i*2],TPD_2018_GPP[i*2+1]])
    TPD_R[i+5]=np.nanmean([TPD_2018_R[i*2],TPD_2018_R[i*2+1]])
    TPD_NEE[i+5]=np.nanmean([TPD_2018_NEE2[i*2],TPD_2018_NEE2[i*2+1]])
    TPD_NEEgf[i+5]=np.nanmean([TPD_2018_NEE[i*2],TPD_2018_NEE[i*2+1]])


# In[ ]:





# In[31]:


# Take daily average of Original SMUrF 2018 data over TPD
C_daily_NEE_TPD=np.zeros(365)*np.nan
C_daily_GPP_TPD=np.zeros(365)*np.nan
C_daily_R_TPD=np.zeros(365)*np.nan
date=0
daily_NEE_TPD_S=[]
daily_GPP_TPD_S=[]
daily_R_TPD_S=[]
for i in range(len(date_array)):
    if date_array[i]>=1:
        if date+1>=365:
            daily_NEE_TPD_S.append(C_NEE_TPD_array[i])
            daily_GPP_TPD_S.append(C_GPP_TPD_array[i])
            daily_R_TPD_S.append(C_Reco_TPD_array[i])
            if i==len(date_array)-1:
                C_daily_NEE_TPD[date]=np.mean(daily_NEE_TPD_S)
                C_daily_GPP_TPD[date]=np.mean(daily_GPP_TPD_S)
                C_daily_R_TPD[date]=np.mean(daily_R_TPD_S)
                date+=1
        else:
            daily_NEE_TPD_S.append(C_NEE_TPD_array[i])
            daily_GPP_TPD_S.append(C_GPP_TPD_array[i])
            daily_R_TPD_S.append(C_Reco_TPD_array[i])
            if np.floor(np.round(date_array[i],4))<np.floor(np.round(date_array[i+1],4)):
                C_daily_NEE_TPD[date]=np.mean(daily_NEE_TPD_S)
                C_daily_GPP_TPD[date]=np.mean(daily_GPP_TPD_S)
                C_daily_R_TPD[date]=np.mean(daily_R_TPD_S)
                date+=1
                daily_NEE_TPD_S=[]
                daily_GPP_TPD_S=[]
                daily_R_TPD_S=[]


# In[ ]:





# In[32]:


#Import TPD 2019 flux tower data

# *** CHANGE PATH & FILENAME ***
TPD_2019_data=pd.read_csv('C:/Users/kitty/Documents/Research/SIF/Flux_Tower/Turkey_Point/TPD_HH_2019.csv', usecols=[0,1,2,74,75,76]) #header=1

TPD_2019_dates=np.zeros([17520])*np.nan
TPD_2019_NEE=np.zeros([17520])*np.nan
TPD_2019_NEE2=np.zeros([17520])*np.nan
TPD_2019_GPP=np.zeros([17520])*np.nan
TPD_2019_R=np.zeros([17520])*np.nan
n=0
m=0
date=1
for i in range(17520):
    if 201901010000<=TPD_2019_data.iat[i,0]<202001010000:
        TPD_2019_dates[i]=datetime.strptime(str(int(TPD_2019_data.iat[i,0])),'%Y%m%d%H%M').timetuple().tm_yday+(float(str(TPD_2019_data.iat[i,0])[8:10])+float(str(TPD_2019_data.iat[i,0])[10:12])/60)/24 #save the current date (and time)
        #check that the value is greater than -9999 (value for empty measurements)
        if TPD_2019_data.iat[i,2]>-9999:
            TPD_2019_NEE2[i]=TPD_2019_data.iat[i,2] # save the NEE (non-gapfilled)
        if TPD_2019_data.iat[i,5]>-9999:
            TPD_2019_NEE[i]=TPD_2019_data.iat[i,5] # save the gap-filled NEE
        if TPD_2019_data.iat[i,3]>-9999:
            TPD_2019_GPP[i]=TPD_2019_data.iat[i,3] # save the GPP
        if TPD_2019_data.iat[i,4]>-9999:
            TPD_2019_R[i]=TPD_2019_data.iat[i,4] # save the Reco

days_of_year=np.arange(1,366)+0.5
TPD_daily_2019_NEE=np.zeros(365)*np.nan
TPD_daily_2019_NEEgf=np.zeros(365)*np.nan
TPD_daily_2019_GPP=np.zeros(365)*np.nan
TPD_daily_2019_R=np.zeros(365)*np.nan
TPD_daily_2019_NEE_std=np.zeros(365)*np.nan
TPD_daily_2019_NEEgf_std=np.zeros(365)*np.nan
TPD_daily_2019_GPP_std=np.zeros(365)*np.nan
TPD_daily_2019_R_std=np.zeros(365)*np.nan
date=0
daily_2019_NEE=[]
daily_2019_NEEgf=[]
daily_2019_GPP=[]
daily_2019_R=[]
for i in range(len(TPD_2019_dates)):
    if TPD_2019_dates[i]>=1:
        if date+1>=365:
            daily_2019_NEE.append(TPD_2019_NEE2[i])
            daily_2019_NEEgf.append(TPD_2019_NEE[i])
            daily_2019_GPP.append(TPD_2019_GPP[i])
            daily_2019_R.append(TPD_2019_R[i])
            if i==len(TPD_2019_dates)-1:
                TPD_daily_2019_NEE[date]=np.mean(daily_2019_NEE)
                TPD_daily_2019_NEEgf[date]=np.mean(daily_2019_NEEgf)
                TPD_daily_2019_GPP[date]=np.mean(daily_2019_GPP)
                TPD_daily_2019_R[date]=np.mean(daily_2019_R)
                date+=1
        else:
            daily_2019_NEE.append(TPD_2019_NEE2[i])
            daily_2019_NEEgf.append(TPD_2019_NEE[i])
            daily_2019_GPP.append(TPD_2019_GPP[i])
            daily_2019_R.append(TPD_2019_R[i])
            if np.floor(np.round(TPD_2019_dates[i],4))<np.floor(np.round(TPD_2019_dates[i+1],4)):
                TPD_daily_2019_NEE[date]=np.mean(daily_2019_NEE)
                TPD_daily_2019_NEEgf[date]=np.mean(daily_2019_NEEgf)
                TPD_daily_2019_GPP[date]=np.mean(daily_2019_GPP)
                TPD_daily_2019_R[date]=np.mean(daily_2019_R)
                date+=1
                daily_2019_NEE=[]
                daily_2019_NEEgf=[]
                daily_2019_GPP=[]
                daily_2019_R=[]


# In[33]:


# Select Original SMUrF 2019 fluxes over TPD
C_GPP_TPD_2019_array=np.zeros(8765)*np.nan
C_NEE_TPD_2019_array=np.zeros(8765)*np.nan
C_Reco_TPD_2019_array=np.zeros(8765)*np.nan

for i in range(len(C_GPP_2019[:,0,0])):
    C_GPP_TPD_2019_array[i]=np.nanmean([C_GPP_2019[i,2,2]])
    C_NEE_TPD_2019_array[i]=np.nanmean([C_NEE_2019[i,2,2]])
    C_Reco_TPD_2019_array[i]=np.nanmean([C_Reco_2019[i,2,2]])


# In[34]:


#Average TPD 2019 flux tower data to hourly resolution

TPD_2019_hrly_GPP=np.zeros(np.shape(C_GPP_TPD_2019_array))*np.nan
TPD_2019_hrly_NEE=np.zeros(np.shape(C_GPP_TPD_2019_array))*np.nan
TPD_2019_hrly_NEEgf=np.zeros(np.shape(C_GPP_TPD_2019_array))*np.nan
TPD_2019_hrly_R=np.zeros(np.shape(C_GPP_TPD_2019_array))*np.nan
for i in range(len(date_array)-5):
    TPD_2019_hrly_GPP[i+5]=np.nanmean([TPD_2019_GPP[i*2],TPD_2019_GPP[i*2+1]])
    TPD_2019_hrly_NEE[i+5]=np.nanmean([TPD_2019_NEE2[i*2],TPD_2019_NEE2[i*2+1]])
    TPD_2019_hrly_NEEgf[i+5]=np.nanmean([TPD_2019_NEE[i*2],TPD_2019_NEE[i*2+1]])
    TPD_2019_hrly_R[i+5]=np.nanmean([TPD_2019_R[i*2],TPD_2019_R[i*2+1]])


# In[ ]:





# In[35]:


# **** If you have not saved the Updated SMUrF data over the flux towers (see 
#'TROPOMI_SMUrF_CSIF_SMUrF_Fluxtower_Seasonal_Comparison-V061_clean.py') run this block of code:

#Load in SMUrF data and crop to flux tower locations for 2018:
S_time=[]
S_lats=[]
S_lons=[]

S_Reco_Borden=[]
S_NEE_Borden=[]
S_GPP_Borden=[]

S_Reco_TP39=[]
S_NEE_TP39=[]
S_GPP_TP39=[]

S_Reco_TPD=[]
S_NEE_TPD=[]
S_GPP_TPD=[]

# *** CHANGE PATH ***
S_path = 'C:/Users/kitty/Documents/Research/SIF/SMUrF/output2018_500m_CSIF_to_TROPOMI_CSIF_ALL_converted_slps_V3_temp_impervious_R_shore_corr_V061_8day/easternCONUS/hourly_flux_GMIS_Toronto_fixed_border_ISA_a_w_sd_era5/'
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
              
            # *** NOTE: if you changed the extent from the default settings when running SMUrF you will need to 
            # change the indices below to find the correct pixels over the flux towers ***
            S_GPP_Borden = np.nanmean([S_GPP[:,458,230],S_GPP[:,458,231],S_GPP[:,458,232],S_GPP[:,459,230],S_GPP[:,459,231],S_GPP[:,459,232],S_GPP[:,460,232]],axis=0)
            S_Reco_Borden = np.nanmean([S_Reco[:,458,230],S_Reco[:,458,231],S_Reco[:,458,232],S_Reco[:,459,230],S_Reco[:,459,231],S_Reco[:,459,232],S_Reco[:,460,232]],axis=0)
            S_NEE_Borden = np.nanmean([S_NEE[:,458,230],S_NEE[:,458,231],S_NEE[:,458,232],S_NEE[:,459,230],S_NEE[:,459,231],S_NEE[:,459,232],S_NEE[:,460,232]],axis=0)
            
            S_GPP_TP39 = np.nanmean(S_GPP[:,73:75,129:131],axis=(1,2))
            S_Reco_TP39 = np.nanmean(S_Reco[:,73:75,129:131],axis=(1,2))
            S_NEE_TP39 = np.nanmean(S_NEE[:,73:75,129:131],axis=(1,2))
            
            S_GPP_TPD = np.nanmean(S_GPP[:,55:58,80:83],axis=(1,2))
            S_Reco_TPD = np.nanmean(S_Reco[:,55:58,80:83],axis=(1,2))
            S_NEE_TPD = np.nanmean(S_NEE[:,55:58,80:83],axis=(1,2))
            
        else:
            #Otherwise append fluxes to the array
            S_GPP_Borden = np.concatenate((S_GPP_Borden,np.nanmean([S_GPP[:,458,230],S_GPP[:,458,231],S_GPP[:,458,232],S_GPP[:,459,230],S_GPP[:,459,231],S_GPP[:,459,232],S_GPP[:,460,232]],axis=0)),axis=0)
            S_Reco_Borden = np.concatenate((S_Reco_Borden,np.nanmean([S_Reco[:,458,230],S_Reco[:,458,231],S_Reco[:,458,232],S_Reco[:,459,230],S_Reco[:,459,231],S_Reco[:,459,232],S_Reco[:,460,232]],axis=0)),axis=0)
            S_NEE_Borden = np.concatenate((S_NEE_Borden,np.nanmean([S_NEE[:,458,230],S_NEE[:,458,231],S_NEE[:,458,232],S_NEE[:,459,230],S_NEE[:,459,231],S_NEE[:,459,232],S_NEE[:,460,232]],axis=0)),axis=0)
            
            S_GPP_TP39 = np.concatenate((S_GPP_TP39,np.nanmean(S_GPP[:,73:75,129:131],axis=(1,2))),axis=0)
            S_Reco_TP39 = np.concatenate((S_Reco_TP39,np.nanmean(S_Reco[:,73:75,129:131],axis=(1,2))),axis=0)
            S_NEE_TP39 = np.concatenate((S_NEE_TP39,np.nanmean(S_NEE[:,73:75,129:131],axis=(1,2))),axis=0)
            
            S_GPP_TPD = np.concatenate((S_GPP_TPD,np.nanmean(S_GPP[:,55:58,80:83],axis=(1,2))),axis=0)
            S_Reco_TPD = np.concatenate((S_Reco_TPD,np.nanmean(S_Reco[:,55:58,80:83],axis=(1,2))),axis=0)
            S_NEE_TPD = np.concatenate((S_NEE_TPD,np.nanmean(S_NEE[:,55:58,80:83],axis=(1,2))),axis=0)
            
            S_time=np.concatenate((S_time,(f.variables['time'][:]/24/3600-start_of_year-5/24)),axis=0)
        f.close()
    except FileNotFoundError:
        pass
    
del(S_GPP,S_Reco,S_NEE)

S_GPP_Borden = np.concatenate((S_GPP_Borden,np.ones(5)*np.nan),axis=0)
S_Reco_Borden = np.concatenate((S_Reco_Borden,np.ones(5)*np.nan),axis=0)
S_NEE_Borden = np.concatenate((S_NEE_Borden,np.ones(5)*np.nan),axis=0)

S_GPP_TP39 = np.concatenate((S_GPP_TP39,np.ones(5)*np.nan),axis=0)
S_Reco_TP39 = np.concatenate((S_Reco_TP39,np.ones(5)*np.nan),axis=0)
S_NEE_TP39 = np.concatenate((S_NEE_TP39,np.ones(5)*np.nan),axis=0)

S_GPP_TPD = np.concatenate((S_GPP_TPD,np.ones(5)*np.nan),axis=0)
S_Reco_TPD = np.concatenate((S_Reco_TPD,np.ones(5)*np.nan),axis=0)
S_NEE_TPD = np.concatenate((S_NEE_TPD,np.ones(5)*np.nan),axis=0)

S_time=np.concatenate((S_time,np.ones(5)*np.nan),axis=0)


# In[36]:


#Load in SMUrF data and crop to flux tower locations for 2019:
S_time=[]
S_lats=[]
S_lons=[]

S_Reco_TP39_2019=[]
S_NEE_TP39_2019=[]
S_GPP_TP39_2019=[]

S_Reco_TPD_2019=[]
S_NEE_TPD_2019=[]
S_GPP_TPD_2019=[]

# *** CHANGE PATH ***
S_path = 'C:/Users/kitty/Documents/Research/SIF/SMUrF/output2019_500m_CSIF_to_TROPOMI_CSIF_ALL_converted_slps_V3_temp_impervious_R_shore_corr_V061_8day/easternCONUS/hourly_flux_GMIS_Toronto_fixed_border_ISA_a_w_sd_era5/'
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
                 
            # *** NOTE: if you changed the extent from the default settings when running SMUrF you will need to 
            # change the indices below to find the correct pixels over the flux towers ***
            S_GPP_TP39_2019 = np.nanmean(S_GPP[:,73:75,129:131],axis=(1,2))
            S_Reco_TP39_2019 = np.nanmean(S_Reco[:,73:75,129:131],axis=(1,2))
            S_NEE_TP39_2019 = np.nanmean(S_NEE[:,73:75,129:131],axis=(1,2))
            
            S_GPP_TPD_2019 = np.nanmean(S_GPP[:,55:58,80:83],axis=(1,2))
            S_Reco_TPD_2019 = np.nanmean(S_Reco[:,55:58,80:83],axis=(1,2))
            S_NEE_TPD_2019 = np.nanmean(S_NEE[:,55:58,80:83],axis=(1,2))
            
        else:
            #Otherwise append to the array
            S_GPP_TP39_2019 = np.concatenate((S_GPP_TP39_2019,np.nanmean(S_GPP[:,73:75,129:131],axis=(1,2))),axis=0)
            S_Reco_TP39_2019 = np.concatenate((S_Reco_TP39_2019,np.nanmean(S_Reco[:,73:75,129:131],axis=(1,2))),axis=0)
            S_NEE_TP39_2019 = np.concatenate((S_NEE_TP39_2019,np.nanmean(S_NEE[:,73:75,129:131],axis=(1,2))),axis=0)
            
            S_GPP_TPD_2019 = np.concatenate((S_GPP_TPD_2019,np.nanmean(S_GPP[:,55:58,80:83],axis=(1,2))),axis=0)
            S_Reco_TPD_2019 = np.concatenate((S_Reco_TPD_2019,np.nanmean(S_Reco[:,55:58,80:83],axis=(1,2))),axis=0)
            S_NEE_TPD_2019 = np.concatenate((S_NEE_TPD_2019,np.nanmean(S_NEE[:,55:58,80:83],axis=(1,2))),axis=0)
            
            S_time=np.concatenate((S_time,(f.variables['time'][:]/24/3600-start_of_year-5/24)),axis=0)
        f.close()
    except FileNotFoundError:
        pass
    
del(S_GPP,S_Reco,S_NEE)

S_GPP_2019_TP39 = np.concatenate((S_GPP_TP39_2019,np.ones(5)*np.nan),axis=0)
S_Reco_2019_TP39 = np.concatenate((S_Reco_TP39_2019,np.ones(5)*np.nan),axis=0)
S_NEE_2019_TP39 = np.concatenate((S_NEE_TP39_2019,np.ones(5)*np.nan),axis=0)

S_GPP_2019_TPD = np.concatenate((S_GPP_TPD_2019,np.ones(5)*np.nan),axis=0)
S_Reco_2019_TPD = np.concatenate((S_Reco_TPD_2019,np.ones(5)*np.nan),axis=0)
S_NEE_2019_TPD = np.concatenate((S_NEE_TPD_2019,np.ones(5)*np.nan),axis=0)

S_time=np.concatenate((S_time,np.ones(5)*np.nan),axis=0)

#End of load in data


# In[ ]:





# In[242]:


# *** IF you previously saved SMUrF fluxes over flux towers (see 
# 'TROPOMI_SMUrF_CSIF_SMUrF_Fluxtower_Seasonal_Comparison-V061_clean.py') uncomment the lines below to load it in (faster)
# instead of running the load in block above ***

## Load in Updated SMUrF fluxes over Borden Forest
## *** CHANGE PATH & FILENAME ***
#g = Dataset('E:/Research/SMUrF/output2018_500m_CSIF_to_TROPOMI_CSIF_ALL_converted_slps_temp_impervious_R_V061_8day/SMUrF_Borden_fluxes.nc')
#S_test_NEE_Borden=g.variables['NEE'][:]
#S_test_GPP_Borden=g.variables['GPP'][:]
#S_test_Reco_Borden=g.variables['Reco'][:]
#S_test_time=g.variables['time'][:]
#g.close()

## Load in Updated SMUrF fluxes over TP39
## *** CHANGE PATH & FILENAME ***
#g = Dataset('E:/Research/SMUrF/output2018_500m_CSIF_to_TROPOMI_CSIF_ALL_converted_slps_temp_impervious_R_V061_8day/SMUrF_TP39_fluxes.nc')
#S_test_NEE_TP39=g.variables['NEE'][:]
#S_test_GPP_TP39=g.variables['GPP'][:]
#S_test_Reco_TP39=g.variables['Reco'][:]
#S_test_time=g.variables['time'][:]
#g.close()

## Load in Updated SMUrF fluxes over TPD
## *** CHANGE PATH & FILENAME ***
#g = Dataset('E:/Research/SMUrF/output2018_500m_CSIF_to_TROPOMI_CSIF_ALL_converted_slps_temp_impervious_R_V061_8day/SMUrF_TPD_fluxes_2018.nc')
#S_test_NEE_TPD=g.variables['NEE'][:]
#S_test_GPP_TPD=g.variables['GPP'][:]
#S_test_Reco_TPD=g.variables['Reco'][:]
#S_test_time=g.variables['time'][:]
#g.close()

## Load in Updated SMUrF fluxes over TPD & TP39 for 2019
## *** CHANGE PATH & FILENAME ***
#g = Dataset('E:/Research/SMUrF/output2019_500m_CSIF_to_TROPOMI_CSIF_ALL_converted_slps_temp_impervious_R_V061_8day/SMUrF_TP39_fluxes.nc')
#S_test_NEE_2019_TP39=g.variables['NEE'][:]
#S_test_GPP_2019_TP39=g.variables['GPP'][:]
#S_test_Reco_2019_TP39=g.variables['Reco'][:]
#S_test_time=g.variables['time'][:]
#g.close()

#g = Dataset('E:/Research/SMUrF/output2019_500m_CSIF_to_TROPOMI_CSIF_ALL_converted_slps_temp_impervious_R_V061_8day/SMUrF_TPD_fluxes_2019.nc')
#S_test_NEE_2019_TPD=g.variables['NEE'][:]
#S_test_GPP_2019_TPD=g.variables['GPP'][:]
#S_test_Reco_2019_TPD=g.variables['Reco'][:]
#S_test_time=g.variables['time'][:]
#g.close()


# In[244]:





# In[37]:


#Take the daily average Updated SMUrF over each of the fluxtowers

S_daily_NEE=np.zeros(365)*np.nan
S_daily_GPP=np.zeros(365)*np.nan
S_daily_R=np.zeros(365)*np.nan
daily_NEE_S=[]
daily_GPP_S=[]
daily_R_S=[]

S_daily_NEE_TP39=np.zeros(365)*np.nan
S_daily_GPP_TP39=np.zeros(365)*np.nan
S_daily_R_TP39=np.zeros(365)*np.nan
daily_NEE_TP39_S=[]
daily_GPP_TP39_S=[]
daily_R_TP39_S=[]

S_daily_NEE_TPD=np.zeros(365)*np.nan
S_daily_GPP_TPD=np.zeros(365)*np.nan
S_daily_R_TPD=np.zeros(365)*np.nan
daily_NEE_TPD_S=[]
daily_GPP_TPD_S=[]
daily_R_TPD_S=[]

date=0
for i in range(len(date_array)):
    if np.round(date_array[i],4)>=1:
        if date+1>=365:
            daily_NEE_S.append(S_NEE_Borden[i])
            daily_GPP_S.append(S_GPP_Borden[i])
            daily_R_S.append(S_Reco_Borden[i])
            
            daily_NEE_TP39_S.append(S_NEE_TP39[i])
            daily_GPP_TP39_S.append(S_GPP_TP39[i])
            daily_R_TP39_S.append(S_Reco_TP39[i])
            
            daily_NEE_TPD_S.append(S_NEE_TPD[i])
            daily_GPP_TPD_S.append(S_GPP_TPD[i])
            daily_R_TPD_S.append(S_Reco_TPD[i])
            if i==len(date_array)-1:
                S_daily_NEE[date]=np.mean(daily_NEE_S)
                S_daily_GPP[date]=np.mean(daily_GPP_S)
                S_daily_R[date]=np.mean(daily_R_S)
                
                S_daily_NEE_TP39[date]=np.mean(daily_NEE_TP39_S)
                S_daily_GPP_TP39[date]=np.mean(daily_GPP_TP39_S)
                S_daily_R_TP39[date]=np.mean(daily_R_TP39_S)
                
                S_daily_NEE_TPD[date]=np.mean(daily_NEE_TPD_S)
                S_daily_GPP_TPD[date]=np.mean(daily_GPP_TPD_S)
                S_daily_R_TPD[date]=np.mean(daily_R_TPD_S)
                date+=1
        else:
            daily_NEE_S.append(S_NEE_Borden[i])
            daily_GPP_S.append(S_GPP_Borden[i])
            daily_R_S.append(S_Reco_Borden[i])
            
            daily_NEE_TP39_S.append(S_NEE_TP39[i])
            daily_GPP_TP39_S.append(S_GPP_TP39[i])
            daily_R_TP39_S.append(S_Reco_TP39[i])
            
            daily_NEE_TPD_S.append(S_NEE_TPD[i])
            daily_GPP_TPD_S.append(S_GPP_TPD[i])
            daily_R_TPD_S.append(S_Reco_TPD[i])
            if np.floor(np.round(date_array[i],4))<np.floor(np.round(date_array[i+1],4)):
                S_daily_NEE[date]=np.mean(daily_NEE_S)
                S_daily_GPP[date]=np.mean(daily_GPP_S)
                S_daily_R[date]=np.mean(daily_R_S)
                
                S_daily_NEE_TP39[date]=np.mean(daily_NEE_TP39_S)
                S_daily_GPP_TP39[date]=np.mean(daily_GPP_TP39_S)
                S_daily_R_TP39[date]=np.mean(daily_R_TP39_S)
                
                S_daily_NEE_TPD[date]=np.mean(daily_NEE_TPD_S)
                S_daily_GPP_TPD[date]=np.mean(daily_GPP_TPD_S)
                S_daily_R_TPD[date]=np.mean(daily_R_TPD_S)
            
                date+=1
                daily_NEE_S=[]
                daily_GPP_S=[]
                daily_R_S=[]
                
                daily_NEE_TP39_S=[]
                daily_GPP_TP39_S=[]
                daily_R_TP39_S=[]
                
                daily_NEE_TPD_S=[]
                daily_GPP_TPD_S=[]
                daily_R_TPD_S=[]


# In[38]:


# *** Optional: Uncomment to viusalize 2018-2019 hourly fluxes

#plt.xlim(1,365*2)
#plt.scatter(C_time_array,S_NEE_Borden,label='TROPOMI-SMUrF',s=8)
#plt.scatter(C_time_array,C_NEE_array,label='CSIF-SMUrF',s=4)
#plt.scatter(C_time_array,Borden_NEE,label='Fluxtower',c='k',s=2)
#plt.legend()
#plt.xlabel('Day of Year')
#plt.ylabel('NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
#plt.title('Borden Modelled NEE')
#plt.show()

#plt.xlim(1,365*2)
#plt.scatter(C_time_array,S_NEE_TP39,label='TROPOMI-SMUrF',s=8)
#plt.scatter(C_time_array,C_NEE_TP39_array,label='CSIF-SMUrF',s=4)
#plt.scatter(C_time_array,TP39_NEE,label='Fluxtower',c='k',s=2)
#plt.scatter(C_time_array+365,S_NEE_2019_TP39,c='tab:blue',s=8)
#plt.scatter(C_time_array+365,C_NEE_TP39_2019_array,c='tab:orange',s=4)
#plt.scatter(C_time_array+365,TP39_2019_hrly_NEE,c='k',s=2)
#plt.xlabel('Day of Year')
#plt.ylabel('NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
#plt.title('TP39 Modelled NEE')
#plt.show()

#plt.xlim(1,365*2)
#plt.scatter(C_time_array,S_NEE_TPD,label='TROPOMI-SMUrF',s=8)
#plt.scatter(C_time_array,C_NEE_TPD_array,label='CSIF-SMUrF',s=4)
#plt.scatter(C_time_array,TPD_NEE,label='Fluxtower',c='k',s=2)
#plt.scatter(C_time_array+365,S_NEE_2019_TPD,c='tab:blue',s=8)
#plt.scatter(C_time_array+365,C_NEE_TPD_2019_array,c='tab:orange',s=4)
#plt.scatter(C_time_array+365,TPD_2019_hrly_NEE,c='k',s=2)
#plt.xlabel('Day of Year')
#plt.ylabel('NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
#plt.title('TPD Modelled NEE')
#plt.show()

# End of uncomment ***


# In[ ]:





# In[40]:


plt.rc('font',size=22)

fig, ax = plt.subplots(3,3,sharex=True,figsize=(11,8))
ax[0,0].set_xlim(1,365)
ax[0,0].set_ylim(-2,22)
ax[0,1].set_ylim(-2,22)
ax[0,2].set_ylim(-2,22)

l0,=ax[0,0].plot(days_of_year,Borden_daily_GPPgf+50,label='Borden Fluxtower',c='k')
ax[0,0].plot(days_of_year,Borden_daily_GPPgf,label='Borden 2018',c='k')

ls0,=ax[0,0].plot(days_of_year,days_of_year+50,label='Original SMUrF',c='#006BA4')
ax[0,0].plot(days_of_year,C_daily_GPP,label='Original SMUrF',c='#006BA4',alpha=0.75)

ls1,=ax[0,0].plot(days_of_year,days_of_year+50,label='Updated SMUrF',c='#FF800E',linestyle='--')
ax[0,0].plot(days_of_year,S_daily_GPP,label='Updated SMUrF',c='#FF800E', alpha=0.75,linestyle='--')
ax[0,0].set_title('GPP')

l1=ax[0,1].scatter(days_of_year,TP39_daily_GPP+50,label='TP39 Fluxtower',c='k')
ax[0,1].plot(days_of_year,TP39_daily_GPP,label='TP39 2018-2019',c='k')
ax[0,1].plot(days_of_year,C_daily_GPP_TP39,label='Original SMUrF', c='#006BA4',alpha=0.75)
ax[0,1].plot(days_of_year,S_daily_GPP_TP39,label='Updated SMUrF',c='#FF800E',alpha=0.75,linestyle='--')
ax[0,0].set_ylabel('GPP')

l2=ax[0,2].scatter(days_of_year,TPD_daily_GPP+50,label='TPD Fluxtower',c='k')
ax[0,2].plot(days_of_year,TPD_daily_GPP,label='TPD 2018-2019',c='k')
ax[0,2].plot(days_of_year,C_daily_GPP_TPD,label='Original SMUrF', c='#006BA4',alpha=0.75)
ax[0,2].plot(days_of_year,S_daily_GPP_TPD,label='Updated SMUrF',c='#FF800E',alpha=0.75,linestyle='--')

ax[1,0].set_ylim(-1,22)
ax[1,1].set_ylim(-1,22)
ax[1,2].set_ylim(-1,22)

ax[1,0].plot(days_of_year,Borden_daily_Rgf,label='Borden 2018-2020',c='k')
ax[1,0].plot(days_of_year,C_daily_R,label='Original SMUrF', c='#006BA4',alpha=0.75)
ax[1,0].plot(days_of_year,S_daily_R,label='Updated SMUrF',c='#FF800E', alpha=0.75,linestyle='--')

ax[0,0].set_title('GPP')

ax[1,1].plot(days_of_year,TP39_daily_R,label='TP39 2018-2019',c='k')
ax[1,1].plot(days_of_year,C_daily_R_TP39,label='Original SMUrF', c='#006BA4',alpha=0.75)
ax[1,1].plot(days_of_year,S_daily_R_TP39,label='Updated SMUrF',c='#FF800E',alpha=0.75,linestyle='--')

l2=ax[1,2].scatter(days_of_year,TPD_daily_R+50,label='TPD Fluxtower',c='k')
ax[1,2].plot(days_of_year,TPD_daily_R,label='TPD 2018-2019',c='k')
ax[1,2].plot(days_of_year,C_daily_R_TPD,label='Original SMUrF', c='#006BA4',alpha=0.75)
ax[1,2].plot(days_of_year,S_daily_R_TPD,label='Updated SMUrF',c='#FF800E',alpha=0.75,linestyle='--')

ax[1,0].set_ylabel('R$_{eco}$')

ax[2,0].set_ylim(-16,6)
ax[2,1].set_ylim(-16,6)
ax[2,2].set_ylim(-16,6)

ax[2,0].plot(days_of_year,Borden_daily_NEEgf,label='Borden 2018-2020',c='k')
ax[2,0].plot(days_of_year,C_daily_NEE,label='Original SMUrF', c='#006BA4',alpha=0.75)
ax[2,0].plot(days_of_year,S_daily_NEE,label='Updated SMUrF',c='#FF800E', alpha=0.75,linestyle='--')

ax[0,0].set_title('Borden Forest')
ax[0,1].set_title('TP39')
ax[0,2].set_title('TPD')

ax[2,1].plot(days_of_year,TP39_daily_NEEgf,label='TP39 2018-2019',c='k')
ax[2,1].plot(days_of_year,C_daily_NEE_TP39,label='Original SMUrF', c='#006BA4',alpha=0.75)
ax[2,1].plot(days_of_year,S_daily_NEE_TP39,label='Updated SMUrF',c='#FF800E',alpha=0.75,linestyle='--')

ax[2,2].plot(days_of_year,TPD_daily_NEEgf,label='TPD 2018-2019',c='k')
ax[2,2].plot(days_of_year,C_daily_NEE_TPD,label='Original SMUrF', c='#006BA4',alpha=0.75)
ax[2,2].plot(days_of_year,S_daily_NEE_TPD,label='Updated SMUrF',c='#FF800E',alpha=0.75,linestyle='--')

ax[2,0].set_ylabel('NEE')

ax[0,1].set_yticks([])
ax[0,2].set_yticks([])
ax[1,1].set_yticks([])
ax[1,2].set_yticks([])
ax[2,1].set_yticks([])
ax[2,2].set_yticks([])

ax[1,0].legend([l0,ls0,ls1],['Flux Tower','Original SMUrF','Updated SMUrF'],loc='upper left',fontsize=16)
ax[2,1].set_xlabel('Day of Year')
fig.subplots_adjust(hspace=0,wspace=0)
# *** Uncomment next two lines to save figure. CHANGE PATHS & FILENAMES ***
#plt.savefig('fixed_SMUrF_V061_vs_fixed_fluxtower_Comparison_All_fluxes_2018_larger_font_cb_friendly.pdf',bbox_inches='tight')
#plt.savefig('fixed_SMUrF_V061_vs_fixed_fluxtower_Comparison_All_fluxes_2018_larger_font_cb_friendly.png',bbox_inches='tight')
fig.show()


# In[ ]:





# In[41]:


#Concatenate the hourly data over all flux towers 
All_S_NEE=np.concatenate([S_NEE_Borden,S_NEE_TP39,S_NEE_TPD,S_NEE_2019_TP39,S_NEE_2019_TPD])
All_fluxtower_NEE=np.concatenate([Borden_NEE,TP39_NEE,TPD_NEE,TP39_2019_hrly_NEE,TPD_2019_hrly_NEE])
All_C_NEE=np.concatenate([C_NEE_array,C_NEE_TP39_array,C_NEE_TPD_array,C_NEE_TP39_2019_array,C_NEE_TPD_2019_array])


# In[43]:


#With fluxtower & downscaling fixes
finitemask1 = np.isfinite(All_fluxtower_NEE)
All_fluxtower_NEEclean0 = All_fluxtower_NEE[finitemask1]
All_S_NEEclean0 = All_S_NEE[finitemask1]

finitemask2 = np.isfinite(All_S_NEEclean0)
Total_S_NEE_2018_2019 = All_S_NEEclean0[finitemask2]
Total_fluxtower_S_NEE_2018_2019 = All_fluxtower_NEEclean0[finitemask2]


# In[ ]:





# In[44]:


#With fluxtower fix
finitemask1 = np.isfinite(All_fluxtower_NEE)
All_fluxtower_NEEclean0 = All_fluxtower_NEE[finitemask1]
All_C_NEEclean0 = All_C_NEE[finitemask1]

finitemask2 = np.isfinite(All_C_NEEclean0)
Total_C_NEE_2018_2019 = All_C_NEEclean0[finitemask2]
Total_fluxtower_C_NEE_2018_2019 = All_fluxtower_NEEclean0[finitemask2]


# In[ ]:





# In[45]:


#Fit the Original SMUrF NEE to non-gapfilled flux tower NEE

Huber_Tot_C_NEE_slps=[]
Huber_Tot_C_NEE_ints=[]
Huber_Tot_C_NEE_R2=[]

#try bootstrapping
indx_list=list(range(0,len(Total_C_NEE_2018_2019)))
for i in range(1,1001):
    #sub selection of points
    NEE_indx=np.random.choice(indx_list,size=len(Total_C_NEE_2018_2019))
    
    try:
        Huber_model = linear_model.HuberRegressor(fit_intercept=True)
        Huber_fit=Huber_model.fit((Total_fluxtower_C_NEE_2018_2019[NEE_indx]).reshape(-1,1),Total_C_NEE_2018_2019[NEE_indx])
        H_m=Huber_fit.coef_
        H_c=Huber_fit.intercept_
        x_accpt, y_accpt = Total_fluxtower_C_NEE_2018_2019, Total_C_NEE_2018_2019
        y_predict = H_m * x_accpt + H_c
        H_R2=r2_score(y_accpt, y_predict)
        Huber_Tot_C_NEE_slps.append(H_m)
        Huber_Tot_C_NEE_ints.append(H_c)
        Huber_Tot_C_NEE_R2.append(H_R2)
    except ValueError: #if Huber fit can't find a solution for the subset, skip it
        pass
    
y_predict = np.nanmean(Huber_Tot_C_NEE_slps) * x_accpt + np.nanmean(Huber_Tot_C_NEE_ints)
Huber_C_NEE_R2=r2_score(y_accpt, y_predict)

print('Original SMUrF slope: '+str(np.round(np.nanmean(Huber_Tot_C_NEE_slps),3))+' +/- '+str(np.round(np.nanstd(Huber_Tot_C_NEE_slps),3)))
print('Original SMUrF intercept: '+str(np.round(np.nanmean(Huber_Tot_C_NEE_ints),3))+' +/- '+str(np.round(np.nanstd(Huber_Tot_C_NEE_ints),3)))

print('Original SMUrF R^2: '+str(np.round(np.nanmean(Huber_C_NEE_R2),3)))


# In[ ]:





# In[46]:


#Fit the Updated SMUrF NEE to non-gapfilled flux tower NEE
#WITH downscaling, MODIS shift, & fluxtower fixes
Huber_Tot_S_NEE_slps=[]
Huber_Tot_S_NEE_ints=[]
Huber_Tot_S_NEE_R2=[]

#try bootstrapping
indx_list=list(range(0,len(Total_S_NEE_2018_2019)))
for i in range(1,1000):
    #sub selection of points
    NEE_indx=np.random.choice(indx_list,size=len(Total_S_NEE_2018_2019))
    
    try:
        Huber_model = linear_model.HuberRegressor(fit_intercept=True)
        Huber_fit=Huber_model.fit((Total_fluxtower_S_NEE_2018_2019[NEE_indx]).reshape(-1,1),Total_S_NEE_2018_2019[NEE_indx])
        H_m=Huber_fit.coef_
        H_c=Huber_fit.intercept_
        x_accpt, y_accpt = Total_fluxtower_S_NEE_2018_2019, Total_S_NEE_2018_2019
        y_predict = H_m * x_accpt + H_c
        H_R2=r2_score(y_accpt, y_predict)
        Huber_Tot_S_NEE_slps.append(H_m)
        Huber_Tot_S_NEE_ints.append(H_c)
        Huber_Tot_S_NEE_R2.append(H_R2)
    except ValueError: #if Huber fit can't find a solution for the subset, skip it
        pass
    
y_predict = np.nanmean(Huber_Tot_S_NEE_slps) * x_accpt + np.nanmean(Huber_Tot_S_NEE_ints)
Huber_S_NEE_R2=r2_score(y_accpt, y_predict)

print('Updated SMUrF slope: '+str(np.round(np.nanmean(Huber_Tot_S_NEE_slps),3))+' +/- '+str(np.round(np.nanstd(Huber_Tot_S_NEE_slps),3)))
print('Updated SMUrF intercept: '+str(np.round(np.nanmean(Huber_Tot_S_NEE_ints),3))+' +/- '+str(np.round(np.nanstd(Huber_Tot_S_NEE_ints),3)))

print('Updated SMUrF R^2: '+str(np.round(np.nanmean(Huber_S_NEE_R2),3)))


# In[ ]:





# In[48]:


plt.style.use('tableau-colorblind10')
plt.rc('font',size=18)
plt.figure(figsize=(8,6))
plt.xlim(-80,20)
plt.ylim(-80,20)
plt.axis('scaled')

plt.scatter(100,100,label='Original SMUrF')
plt.scatter(100,100,label='Updated SMUrF')
plt.scatter(Total_fluxtower_C_NEE_2018_2019,Total_C_NEE_2018_2019,s=5,c='#006BA4')
plt.scatter(Total_fluxtower_S_NEE_2018_2019,Total_S_NEE_2018_2019,s=5,c='#FF800E',alpha=0.5)

plt.plot(line1_1,func2(line1_1,np.nanmean(Huber_Tot_C_NEE_slps),np.nanmean(Huber_Tot_C_NEE_ints)),linestyle='--',label=str(np.round(np.nanmean(Huber_Tot_C_NEE_slps),2))+'$\cdot$x+'+str(np.round(np.nanmean(Huber_Tot_C_NEE_ints),2))+', R$^2$ = '+str(np.round(Huber_C_NEE_R2,2)),c='k', path_effects=[pe.Stroke(linewidth=5, foreground='#006BA4'), pe.Normal()])
plt.plot(line1_1,func2(line1_1,np.nanmean(Huber_Tot_S_NEE_slps),np.nanmean(Huber_Tot_S_NEE_ints)),linestyle='-.',label=str(np.round(np.nanmean(Huber_Tot_S_NEE_slps),2))+'$\cdot$x+'+str(np.round(np.nanmean(Huber_Tot_S_NEE_ints),2))+', R$^2$ = '+str(np.round(Huber_S_NEE_R2,2)),c='k', path_effects=[pe.Stroke(linewidth=5, foreground='#FF800E'), pe.Normal()])

plt.plot(line1_1,line1_1,linestyle=':',c='k')
plt.title('SMUrF vs Flux Tower NEE')
plt.xlabel('Flux Tower NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
plt.ylabel('Modelled NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
plt.legend()
# *** Uncomment to save figure. CHANGE FILENAME ***
#plt.savefig('fixed_SMUrF_V061_vs_fixed_fluxtower_NEE_non_gapfilled_hrly_fit_Huber_correlation_All_fluxes_2018_2019_larger_font_cb_friendly.pdf',bbox_inches='tight')
#plt.savefig('fixed_SMUrF_V061_vs_fixed_fluxtower_NEE_non_gapfilled_hrly_fit_Huber_correlation_All_fluxes_2018_2019_larger_font_cb_friendly.png',bbox_inches='tight')
plt.show()


# In[ ]:





# In[ ]:




