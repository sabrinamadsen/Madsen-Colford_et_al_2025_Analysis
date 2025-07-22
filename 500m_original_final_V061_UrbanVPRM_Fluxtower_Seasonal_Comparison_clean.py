#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This code compares & fits fluxes from the original and updated UrbanVPRM to 3 flux towers in Southern Ontario 
# for each season of 2018 & 2019

# Code used to generate figure 3.a-d and S3 of Madsen-Colford et al.
# *** Denotes portions of the code the user should change


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from scipy import optimize as opt 
from scipy import odr
from scipy import stats as sts
from datetime import datetime, timedelta
from sklearn import linear_model #for robust fitting
from sklearn.metrics import r2_score, mean_squared_error #for analyzing robust fits
import matplotlib.patheffects as pe
import matplotlib.colors as clrs #for log color scale
from netCDF4 import Dataset, date2num #for reading netCDF data files and their date (not sure if I need the later)


# In[2]:


#Load in original UrbanVPRM over Borden forest fluxtower (2018)

# *** CHANGE PATH & FILENAME ***
#2018:
VPRM_data=pd.read_csv('Borden_500m_V061_no_adjustments_2018/vprm_mixed_ISA_Borden_500m_V061_2018_no_adjustments.csv')


# In[3]:


#Load in updated UrbanVPRM over Borden forest fluxtower (2018)

# *** CHANGE PATH & FILENAME ***
Updated_VPRM_data=pd.read_csv('Borden_V061_500m_2018/vprm_GMIS_Toronto_ACI_SOLRIS_ISA_500m_Borden_V061_2018_no_PScale_adjusted_Topt_Ra_URB_parameters_fixed_gapfilled_LSWI_filtered.csv')


# In[ ]:





# In[4]:


#Format UrbanVPRM flux data as arrays & select data within the Borden forest flux tower's footprint.

VPRM_HoY0=np.zeros([8760,6])*np.nan
VPRM_Index0=np.zeros([8760,6])*np.nan
VPRM_GEE0=np.zeros([8760,6])*np.nan
VPRM_Reco0=np.zeros([8760,6])*np.nan

Updated_VPRM_HoY0=np.zeros([8760,6])*np.nan
Updated_VPRM_Index0=np.zeros([8760,6])*np.nan
Updated_VPRM_GEE0=np.zeros([8760,6])*np.nan
Updated_VPRM_Reco0=np.zeros([8760,6])*np.nan

h=0
l=0
for i in range(8760*105,8760*106):
    VPRM_HoY0[h,l]=VPRM_data.iat[i,1] # Time is in UTC
    VPRM_Index0[h,l]=VPRM_data.iat[i,2]
    VPRM_GEE0[h,l]=VPRM_data.iat[i,3]
    VPRM_Reco0[h,l]=VPRM_data.iat[i,9]
    
    Updated_VPRM_HoY0[h,l]=Updated_VPRM_data.iat[i,1]
    Updated_VPRM_Index0[h,l]=Updated_VPRM_data.iat[i,2]
    Updated_VPRM_GEE0[h,l]=Updated_VPRM_data.iat[i,3]
    Updated_VPRM_Reco0[h,l]=Updated_VPRM_data.iat[i,9]
    h+=1
    if VPRM_data.iat[i+1,2]>VPRM_data.iat[i,2]:
        l+=1
        h=0
for i in range(8760*120,8760*122):
    VPRM_HoY0[h,l]=VPRM_data.iat[i,1]
    VPRM_Index0[h,l]=VPRM_data.iat[i,2]
    VPRM_GEE0[h,l]=VPRM_data.iat[i,3]
    VPRM_Reco0[h,l]=VPRM_data.iat[i,9]
    
    Updated_VPRM_HoY0[h,l]=Updated_VPRM_data.iat[i,1]
    Updated_VPRM_Index0[h,l]=Updated_VPRM_data.iat[i,2]
    Updated_VPRM_GEE0[h,l]=Updated_VPRM_data.iat[i,3]
    Updated_VPRM_Reco0[h,l]=Updated_VPRM_data.iat[i,9]
    h+=1
    if VPRM_data.iat[i+1,2]>VPRM_data.iat[i,2]:
        l+=1
        h=0
for i in range(8760*135,8760*138):
    VPRM_HoY0[h,l]=VPRM_data.iat[i,1]
    VPRM_Index0[h,l]=VPRM_data.iat[i,2]
    VPRM_GEE0[h,l]=VPRM_data.iat[i,3]
    VPRM_Reco0[h,l]=VPRM_data.iat[i,9]
    
    Updated_VPRM_HoY0[h,l]=Updated_VPRM_data.iat[i,1]
    Updated_VPRM_Index0[h,l]=Updated_VPRM_data.iat[i,2]
    Updated_VPRM_GEE0[h,l]=Updated_VPRM_data.iat[i,3]
    Updated_VPRM_Reco0[h,l]=Updated_VPRM_data.iat[i,9]
    h+=1
    if VPRM_data.iat[i+1,2]>VPRM_data.iat[i,2]:
        l+=1
        h=0


# In[5]:


#For original VPRM only:
#Original UrbanVPRM saves -GEE instead of GEE, multiply by -1
VPRM_GEE0 = -VPRM_GEE0

# Compute NEE from Reco and GEE
VPRM_NEE0=VPRM_Reco0+VPRM_GEE0
Updated_VPRM_NEE0=Updated_VPRM_Reco0+Updated_VPRM_GEE0


# In[6]:


#Average all pixels falling within fluxtower footprint:

VPRM_Borden_2018_avg_DoY=np.mean(VPRM_HoY0, axis=1)/24+23/24 #time is in UTC
VPRM_Borden_2018_avg_Index=np.mean(VPRM_Index0, axis=1)
VPRM_Borden_2018_avg_GPP=-np.mean(VPRM_GEE0, axis=1)
VPRM_Borden_2018_avg_Reco=np.mean(VPRM_Reco0, axis=1)
VPRM_Borden_2018_avg_NEE=np.mean(VPRM_NEE0, axis=1)

Updated_VPRM_Borden_2018_avg_DoY=np.mean(Updated_VPRM_HoY0, axis=1)/24+23/24
Updated_VPRM_Borden_2018_avg_Index=np.mean(Updated_VPRM_Index0, axis=1)
Updated_VPRM_Borden_2018_avg_GPP=-np.mean(Updated_VPRM_GEE0, axis=1)
Updated_VPRM_Borden_2018_avg_Reco=np.mean(Updated_VPRM_Reco0, axis=1)
Updated_VPRM_Borden_2018_avg_NEE=np.mean(Updated_VPRM_NEE0, axis=1)


# In[ ]:





# In[7]:


#Load Borden flux tower values
# *** CHANGE PATH & FILENAME ***
Borden_Fluxes=pd.read_csv('/Users/kitty/Documents/Research/SIF/Flux_Tower/2018_NEP_GPP_Borden.csv', index_col=0)

Borden_dates=np.zeros([17521])*np.nan
Borden_NEEgf_fluxes=np.zeros([17521])*np.nan
Borden_NEE_fluxes=np.zeros([17521])*np.nan
Borden_Rgf_fluxes=np.zeros([17521])*np.nan
Borden_GEPgf_fluxes=np.zeros([17521])*np.nan
n=0
m=0
date=1
for i in range(0,17520):
    Borden_dates[i+1]=Borden_Fluxes.iat[i,0] #Time in UTC
    Borden_NEEgf_fluxes[i+1]=-Borden_Fluxes.iat[i,5] #NEE (gap filled)
    Borden_NEE_fluxes[i+1]=-Borden_Fluxes.iat[i,1]
    Borden_Rgf_fluxes[i+1]=Borden_Fluxes.iat[i,6]
    Borden_GEPgf_fluxes[i+1]=Borden_Fluxes.iat[i,7]
    
del Borden_Fluxes


# In[ ]:





# In[8]:


sunrise_set=pd.read_csv('C:/Users/kitty/Documents/Research/SIF/NOAA_Solar_Calculations_year_2018.csv').loc[:,('Date','Sunrise Time (LST)','Sunset Time (LST)')]

yr = 2018
sunrise_dates=np.zeros([366])*np.nan
sunset_dates=np.zeros([366])*np.nan
for i in range(0,366):
    #Note: dates are in local time!
    sunrise_dates[i]= datetime.strptime(sunrise_set.iloc[i,0],"%Y-%m-%d").timetuple().tm_yday+int(sunrise_set.iloc[i,1][0])/24+int(sunrise_set.iloc[i,1][2:4])/24/60+int(sunrise_set.iloc[i,1][5:7])/24/60/60
    sunset_dates[i]=datetime.strptime(sunrise_set.iloc[i,0],"%Y-%m-%d").timetuple().tm_yday+int(sunrise_set.iloc[i,2][0:2])/24+int(sunrise_set.iloc[i,2][3:5])/24/60+int(sunrise_set.iloc[i,2][6:8])/24/60/60
    if int(sunrise_set.iloc[i,0][0:4])>yr:
        sunrise_dates[i]+=365
        sunset_dates[i]+=365
    
del sunrise_set


# In[9]:


# Convert half-hourly flux tower data to hourly averages to match resolution of SMUrF 
# time given in files represents the begining of each 30 minute period: average hour & next half-hour to get full hour
Borden_NEE=np.zeros(np.shape(VPRM_Borden_2018_avg_GPP))*np.nan
Borden_GEPgf=np.zeros(np.shape(VPRM_Borden_2018_avg_GPP))*np.nan
Borden_NEEgf=np.zeros(np.shape(VPRM_Borden_2018_avg_GPP))*np.nan
Borden_Rgf=np.zeros(np.shape(VPRM_Borden_2018_avg_GPP))*np.nan
for i in range(np.int(len(Borden_dates)/2)):
    Borden_NEE[i]=np.nanmean([Borden_NEE_fluxes[i*2],Borden_NEE_fluxes[i*2+1]])
    Borden_GEPgf[i]=np.nanmean([Borden_GEPgf_fluxes[i*2],Borden_GEPgf_fluxes[i*2+1]])
    Borden_NEEgf[i]=np.nanmean([Borden_NEEgf_fluxes[i*2],Borden_NEEgf_fluxes[i*2+1]])
    Borden_Rgf[i]=np.nanmean([Borden_Rgf_fluxes[i*2],Borden_Rgf_fluxes[i*2+1]])


# In[ ]:





# In[10]:


#Select only spring data (March-May)
#MAM: Doy 60 - 151 inclusive

MAM_time=VPRM_Borden_2018_avg_DoY[(np.round(VPRM_Borden_2018_avg_DoY,5)>=60) & (np.round(VPRM_Borden_2018_avg_DoY,5)<152)]
Borden_GPPgf_MAM=Borden_GEPgf[(np.round(VPRM_Borden_2018_avg_DoY,5)>=60) & (np.round(VPRM_Borden_2018_avg_DoY,5)<152)]
VPRM_Borden_GPP_MAM=VPRM_Borden_2018_avg_GPP[(np.round(VPRM_Borden_2018_avg_DoY,5)>=60) & (np.round(VPRM_Borden_2018_avg_DoY,5)<152)]
Updated_VPRM_Borden_GPP_MAM=Updated_VPRM_Borden_2018_avg_GPP[(np.round(VPRM_Borden_2018_avg_DoY,5)>=60) & (np.round(VPRM_Borden_2018_avg_DoY,5)<152)]

Borden_Rgf_MAM=Borden_Rgf[(np.round(VPRM_Borden_2018_avg_DoY,5)>=60) & (np.round(VPRM_Borden_2018_avg_DoY,5)<152)]
VPRM_Borden_Reco_MAM=VPRM_Borden_2018_avg_Reco[(np.round(VPRM_Borden_2018_avg_DoY,5)>=60) & (np.round(VPRM_Borden_2018_avg_DoY,5)<152)]
Updated_VPRM_Borden_Reco_MAM=Updated_VPRM_Borden_2018_avg_Reco[(np.round(VPRM_Borden_2018_avg_DoY,5)>=60) & (np.round(VPRM_Borden_2018_avg_DoY,5)<152)]

Borden_NEEgf_MAM=Borden_NEEgf[(np.round(VPRM_Borden_2018_avg_DoY,5)>=60) & (np.round(VPRM_Borden_2018_avg_DoY,5)<152)]
Borden_NEE_MAM=Borden_NEE[(np.round(VPRM_Borden_2018_avg_DoY,5)>=60) & (np.round(VPRM_Borden_2018_avg_DoY,5)<152)]
VPRM_Borden_NEE_MAM=VPRM_Borden_2018_avg_NEE[(np.round(VPRM_Borden_2018_avg_DoY,5)>=60) & (np.round(VPRM_Borden_2018_avg_DoY,5)<152)]
Updated_VPRM_Borden_NEE_MAM=Updated_VPRM_Borden_2018_avg_NEE[(np.round(VPRM_Borden_2018_avg_DoY,5)>=60) & (np.round(VPRM_Borden_2018_avg_DoY,5)<152)]


# In[ ]:





# In[11]:


# Load in original UrbanVPRM 2018 fluxes over TP39
# *** CHANGE PATH & FILENAME ***
VPRM_data=pd.read_csv('TP39_500m_V061_no_adjustments_2018/vprm_mixed_ISA_TP39_500m_V061_2018_no_adjustments.csv')


# In[12]:


# Load in original UrbanVPRM 2018 fluxes over TP39
# *** CHANGE PATH & FILENAME ***
Updated_VPRM_data=pd.read_csv('TP39_V061_500m_2018/vprm_GMIS_Toronto_ACI_SOLRIS_ISA_500m_TP39_V061_2018_no_PScale_adjusted_Topt_Ra_URB_parameters_fixed_gapfilled_LSWI_filtered.csv')


# In[13]:


#Select data in footprint of the TP39 tower
VPRM_HoY0=np.zeros([8760,4])*np.nan
VPRM_Index0=np.zeros([8760,4])*np.nan
VPRM_GEE0=np.zeros([8760,4])*np.nan
VPRM_Reco0=np.zeros([8760,4])*np.nan

Updated_VPRM_HoY0=np.zeros([8760,4])*np.nan
Updated_VPRM_Index0=np.zeros([8760,4])*np.nan
Updated_VPRM_GEE0=np.zeros([8760,4])*np.nan
Updated_VPRM_Reco0=np.zeros([8760,4])*np.nan
h=0
l=0
for i in range(8760*119,8760*121):
    VPRM_HoY0[h,l]=VPRM_data.iat[i,1]
    VPRM_Index0[h,l]=VPRM_data.iat[i,2]
    VPRM_GEE0[h,l]=VPRM_data.iat[i,3]
    VPRM_Reco0[h,l]=VPRM_data.iat[i,9]
    
    Updated_VPRM_HoY0[h,l]=Updated_VPRM_data.iat[i,1]
    Updated_VPRM_Index0[h,l]=Updated_VPRM_data.iat[i,2]
    Updated_VPRM_GEE0[h,l]=Updated_VPRM_data.iat[i,3]
    Updated_VPRM_Reco0[h,l]=Updated_VPRM_data.iat[i,9]
    h+=1
    if VPRM_data.iat[i+1,2]>VPRM_data.iat[i,2]:
        l+=1
        h=0
for i in range(8760*135,8760*137):
    VPRM_HoY0[h,l]=VPRM_data.iat[i,1]
    VPRM_Index0[h,l]=VPRM_data.iat[i,2]
    VPRM_GEE0[h,l]=VPRM_data.iat[i,3]
    VPRM_Reco0[h,l]=VPRM_data.iat[i,9]
    
    Updated_VPRM_HoY0[h,l]=Updated_VPRM_data.iat[i,1]
    Updated_VPRM_Index0[h,l]=Updated_VPRM_data.iat[i,2]
    Updated_VPRM_GEE0[h,l]=Updated_VPRM_data.iat[i,3]
    Updated_VPRM_Reco0[h,l]=Updated_VPRM_data.iat[i,9]
    h+=1
    if VPRM_data.iat[i+1,2]>VPRM_data.iat[i,2]:
        l+=1
        h=0


# In[14]:


#Compute NEE
VPRM_GEE0=-VPRM_GEE0 #ONLY FOR ORIGINAL VPRM - fix sign of GPP
VPRM_NEE0=VPRM_Reco0+VPRM_GEE0
Updated_VPRM_NEE0=Updated_VPRM_Reco0+Updated_VPRM_GEE0


# In[15]:


# Average the data within the footprint
VPRM_TP39_2018_avg_DoY=np.mean(VPRM_HoY0, axis=1)/24+23/24
VPRM_TP39_2018_avg_Index=np.mean(VPRM_Index0, axis=1)
VPRM_TP39_2018_avg_GPP=-np.mean(VPRM_GEE0, axis=1)
VPRM_TP39_2018_avg_Reco=np.mean(VPRM_Reco0, axis=1)
VPRM_TP39_2018_avg_NEE=np.mean(VPRM_NEE0, axis=1)

Updated_VPRM_TP39_2018_avg_DoY=np.mean(Updated_VPRM_HoY0, axis=1)/24+23/24
Updated_VPRM_TP39_2018_avg_Index=np.mean(Updated_VPRM_Index0, axis=1)
Updated_VPRM_TP39_2018_avg_GPP=-np.mean(Updated_VPRM_GEE0, axis=1)
Updated_VPRM_TP39_2018_avg_Reco=np.mean(Updated_VPRM_Reco0, axis=1)
Updated_VPRM_TP39_2018_avg_NEE=np.mean(Updated_VPRM_NEE0, axis=1)


# In[ ]:





# In[16]:


# Load in TP39 2018 data (in local time)

# *** CHANGE PATH & FILENAME ***
TP39_Fluxes=pd.read_csv('/Users/kitty/Documents/Research/SIF/Flux_Tower/Turkey_Point/TP39_HH_2018.csv',usecols=(0,2,77,78,79))

TP39_dates=np.zeros([17520])*np.nan
TP39_NEEgf_fluxes=np.zeros([17520])*np.nan
TP39_NEE_fluxes=np.zeros([17520])*np.nan
TP39_Rgf_fluxes=np.zeros([17520])*np.nan
TP39_GPPgf_fluxes=np.zeros([17520])*np.nan

for i in range(0,17520):
    if 201801010000<=TP39_Fluxes.iat[i,0]<201901010000:
        #TP is 5 hours behind UTC adjust to UTC
        TP39_dates[i]=datetime.strptime(str(int(TP39_Fluxes.iat[i,0])),'%Y%m%d%H%M').timetuple().tm_yday+(float(str(TP39_Fluxes.iat[i,0])[8:10])+float(str(TP39_Fluxes.iat[i,0])[10:12])/60)/24+5/24
        TP39_NEEgf_fluxes[i]=TP39_Fluxes.iat[i,4] #NEE (gap filled)
        if TP39_Fluxes.iat[i,1]>-9999:
            TP39_NEE_fluxes[i]=TP39_Fluxes.iat[i,1]
        TP39_Rgf_fluxes[i]=TP39_Fluxes.iat[i,3]
        TP39_GPPgf_fluxes[i]=TP39_Fluxes.iat[i,2]


# In[ ]:





# In[17]:


#Take hourly average
TP39_GPP=np.zeros(np.shape(VPRM_TP39_2018_avg_GPP))*np.nan
TP39_NEE=np.zeros(np.shape(VPRM_TP39_2018_avg_GPP))*np.nan
TP39_NEEgf=np.zeros(np.shape(VPRM_TP39_2018_avg_GPP))*np.nan
TP39_R=np.zeros(np.shape(VPRM_TP39_2018_avg_GPP))*np.nan
for i in range(np.int(len(TP39_dates)/2)):
    if i<8755:
        TP39_GPP[i+5]=np.nanmean([TP39_GPPgf_fluxes[i*2],TP39_GPPgf_fluxes[i*2+1]])
        TP39_NEE[i+5]=np.nanmean([TP39_NEE_fluxes[i*2],TP39_NEE_fluxes[i*2+1]])
        TP39_NEEgf[i+5]=np.nanmean([TP39_NEEgf_fluxes[i*2],TP39_NEEgf_fluxes[i*2+1]])
        TP39_R[i+5]=np.nansum([TP39_Rgf_fluxes[i*2],TP39_Rgf_fluxes[i*2+1]])


# In[ ]:





# In[18]:


# Select only spring data over TP39
#MAM: Doy 60 - 151 inclusive

MAM_time=VPRM_TP39_2018_avg_DoY[(np.round(VPRM_TP39_2018_avg_DoY,5)>=60) & (np.round(VPRM_TP39_2018_avg_DoY,5)<152)]
TP39_GPPgf_MAM=TP39_GPP[(np.round(VPRM_TP39_2018_avg_DoY,5)>=60) & (np.round(VPRM_TP39_2018_avg_DoY,5)<152)]
VPRM_TP39_GPP_MAM=VPRM_TP39_2018_avg_GPP[(np.round(VPRM_TP39_2018_avg_DoY,5)>=60) & (np.round(VPRM_TP39_2018_avg_DoY,5)<152)]
Updated_VPRM_TP39_GPP_MAM=Updated_VPRM_TP39_2018_avg_GPP[(np.round(VPRM_TP39_2018_avg_DoY,5)>=60) & (np.round(VPRM_TP39_2018_avg_DoY,5)<152)]

TP39_Rgf_MAM=TP39_R[(np.round(VPRM_TP39_2018_avg_DoY,5)>=60) & (np.round(VPRM_TP39_2018_avg_DoY,5)<152)]
VPRM_TP39_Reco_MAM=VPRM_TP39_2018_avg_Reco[(np.round(VPRM_TP39_2018_avg_DoY,5)>=60) & (np.round(VPRM_TP39_2018_avg_DoY,5)<152)]
Updated_VPRM_TP39_Reco_MAM=Updated_VPRM_TP39_2018_avg_Reco[(np.round(VPRM_TP39_2018_avg_DoY,5)>=60) & (np.round(VPRM_TP39_2018_avg_DoY,5)<152)]

TP39_NEEgf_MAM=TP39_NEEgf[(np.round(VPRM_TP39_2018_avg_DoY,5)>=60) & (np.round(VPRM_TP39_2018_avg_DoY,5)<152)]
TP39_NEE_MAM=TP39_NEE[(np.round(VPRM_TP39_2018_avg_DoY,5)>=60) & (np.round(VPRM_TP39_2018_avg_DoY,5)<152)]
VPRM_TP39_NEE_MAM=VPRM_TP39_2018_avg_NEE[(np.round(VPRM_TP39_2018_avg_DoY,5)>=60) & (np.round(VPRM_TP39_2018_avg_DoY,5)<152)]
Updated_VPRM_TP39_NEE_MAM=Updated_VPRM_TP39_2018_avg_NEE[(np.round(VPRM_TP39_2018_avg_DoY,5)>=60) & (np.round(VPRM_TP39_2018_avg_DoY,5)<152)]


# In[ ]:





# In[19]:


# Load in VPRM flux data over TP39 in 2019

# *** CHANGE PATHS & FILENAMES ***
VPRM_data=pd.read_csv('TP39_500m_V061_no_adjustments_2019/vprm_mixed_ISA_TP39_500m_V061_2019_no_adjustments.csv')#vprm_GMIS_Toronto_ACI_SOLRIS_ISA_500m_TP39_V061_2019_no_PScale_adjusted_Topt_Ra_URB_parameters_fixed_gapfilled_LSWI_filtered.csv')
Updated_VPRM_data=pd.read_csv('TP39_V061_500m_2019/vprm_GMIS_Toronto_ACI_SOLRIS_ISA_500m_TP39_V061_2019_no_PScale_adjusted_Topt_Ra_URB_parameters_fixed_gapfilled_LSWI_filtered.csv')

#Select data in footprint of tower
VPRM_HoY0=np.zeros([8760,4])*np.nan
VPRM_Index0=np.zeros([8760,4])*np.nan
VPRM_GEE0=np.zeros([8760,4])*np.nan
VPRM_Reco0=np.zeros([8760,4])*np.nan

Updated_VPRM_HoY0=np.zeros([8760,4])*np.nan
Updated_VPRM_Index0=np.zeros([8760,4])*np.nan
Updated_VPRM_GEE0=np.zeros([8760,4])*np.nan
Updated_VPRM_Reco0=np.zeros([8760,4])*np.nan
h=0
l=0
for i in range(8760*119,8760*121):
    VPRM_HoY0[h,l]=VPRM_data.iat[i,1]
    VPRM_Index0[h,l]=VPRM_data.iat[i,2]
    VPRM_GEE0[h,l]=VPRM_data.iat[i,3]
    VPRM_Reco0[h,l]=VPRM_data.iat[i,9]
    
    Updated_VPRM_HoY0[h,l]=Updated_VPRM_data.iat[i,1]
    Updated_VPRM_Index0[h,l]=Updated_VPRM_data.iat[i,2]
    Updated_VPRM_GEE0[h,l]=Updated_VPRM_data.iat[i,3]
    Updated_VPRM_Reco0[h,l]=Updated_VPRM_data.iat[i,9]
    h+=1
    if VPRM_data.iat[i+1,2]>VPRM_data.iat[i,2]:
        l+=1
        h=0
for i in range(8760*135,8760*137):
    VPRM_HoY0[h,l]=VPRM_data.iat[i,1]
    VPRM_Index0[h,l]=VPRM_data.iat[i,2]
    VPRM_GEE0[h,l]=VPRM_data.iat[i,3]
    VPRM_Reco0[h,l]=VPRM_data.iat[i,9]
    
    Updated_VPRM_HoY0[h,l]=Updated_VPRM_data.iat[i,1]
    Updated_VPRM_Index0[h,l]=Updated_VPRM_data.iat[i,2]
    Updated_VPRM_GEE0[h,l]=Updated_VPRM_data.iat[i,3]
    Updated_VPRM_Reco0[h,l]=Updated_VPRM_data.iat[i,9]    
    h+=1
    if VPRM_data.iat[i+1,2]>VPRM_data.iat[i,2]:
        l+=1
        h=0

VPRM_GEE0=-VPRM_GEE0 #ONLY FOR ORIGINAL UrbanVPRM 
VPRM_NEE0=VPRM_Reco0+VPRM_GEE0 #Compute NEE
Updated_VPRM_NEE0=Updated_VPRM_Reco0+Updated_VPRM_GEE0


# In[20]:


#Average all pixels that fall in the footprint

#Original UrbanVPRM
#MODIS V061
VPRM_TP39_2019_avg_DoY=np.mean(VPRM_HoY0, axis=1)/24+23/24
VPRM_TP39_2019_avg_Index=np.mean(VPRM_Index0, axis=1)
VPRM_TP39_2019_avg_GPP=-np.mean(VPRM_GEE0, axis=1)
VPRM_TP39_2019_avg_Reco=np.mean(VPRM_Reco0, axis=1)
VPRM_TP39_2019_avg_NEE=np.mean(VPRM_NEE0, axis=1)

#Updated UrbanVPRM
#MODIS V061
Updated_VPRM_TP39_2019_avg_DoY=np.mean(Updated_VPRM_HoY0, axis=1)/24+23/24
Updated_VPRM_TP39_2019_avg_Index=np.mean(Updated_VPRM_Index0, axis=1)
Updated_VPRM_TP39_2019_avg_GPP=-np.mean(Updated_VPRM_GEE0, axis=1)
Updated_VPRM_TP39_2019_avg_Reco=np.mean(Updated_VPRM_Reco0, axis=1)
Updated_VPRM_TP39_2019_avg_NEE=np.mean(Updated_VPRM_NEE0, axis=1)


# In[21]:


# *** CHANGE PATH ***
sunrise_set=pd.read_csv('C:/Users/kitty/Documents/Research/SIF/NOAA_Solar_Calculations_year_2019.csv').loc[:,('Date','Sunrise Time (LST)','Sunset Time (LST)')]

yr = 2019
sunrise_dates_2019=np.zeros([366])*np.nan
sunset_dates_2019=np.zeros([366])*np.nan
for i in range(0,366):
    #sunrise/set times are in local time
    sunrise_dates_2019[i]= datetime.strptime(sunrise_set.iloc[i,0],"%Y-%m-%d").timetuple().tm_yday+int(sunrise_set.iloc[i,1][0])/24+int(sunrise_set.iloc[i,1][2:4])/24/60+int(sunrise_set.iloc[i,1][5:7])/24/60/60
    sunset_dates_2019[i]=datetime.strptime(sunrise_set.iloc[i,0],"%Y-%m-%d").timetuple().tm_yday+int(sunrise_set.iloc[i,2][0:2])/24+int(sunrise_set.iloc[i,2][3:5])/24/60+int(sunrise_set.iloc[i,2][6:8])/24/60/60
    if int(sunrise_set.iloc[i,0][0:4])>yr:
        sunrise_dates_2019[i]+=365
        sunset_dates_2019[i]+=365
    
del sunrise_set


# In[ ]:





# In[22]:


# Load in TP39 2019 fluxtower data data (in local time)
# *** CHANGE PATH & FILENAME ***
TP39_2019_Fluxes=pd.read_csv('C:/Users/kitty/Documents/Research/SIF/Flux_Tower/Turkey_Point/TP39_HH_2019.csv', usecols=(0,2,77,78,79))

TP39_2019_dates=np.zeros([17520])*np.nan
TP39_2019_NEEgf_fluxes=np.zeros([17520])*np.nan
TP39_2019_NEE_fluxes=np.zeros([17520])*np.nan
TP39_2019_Rgf_fluxes=np.zeros([17520])*np.nan
TP39_2019_GPPgf_fluxes=np.zeros([17520])*np.nan
for i in range(0,17520):
    if 201901010000<=TP39_2019_Fluxes.iat[i,0]<202001010000:
        #TP is 5 hours behind UTC #adjust to UTC
        TP39_2019_dates[i]=datetime.strptime(str(int(TP39_2019_Fluxes.iat[i,0])),'%Y%m%d%H%M').timetuple().tm_yday+(float(str(TP39_2019_Fluxes.iat[i,0])[8:10])+float(str(TP39_2019_Fluxes.iat[i,0])[10:12])/60)/24+5/24

        if TP39_2019_Fluxes.iat[i,4] > -9999:
            TP39_2019_NEEgf_fluxes[i]=TP39_2019_Fluxes.iat[i,4] #NEE (gap filled)
        if TP39_2019_Fluxes.iat[i,1]> -9999:
            TP39_2019_NEE_fluxes[i]=TP39_2019_Fluxes.iat[i,1]
        if TP39_2019_Fluxes.iat[i,3] > -9999:
            TP39_2019_Rgf_fluxes[i]=TP39_2019_Fluxes.iat[i,3]
        if TP39_2019_Fluxes.iat[i,2] > -9999:
            TP39_2019_GPPgf_fluxes[i]=TP39_2019_Fluxes.iat[i,2]


# In[ ]:





# In[23]:


#Take hourly average
TP39_2019_GPP=np.zeros(np.shape(VPRM_TP39_2019_avg_GPP))*np.nan
TP39_2019_NEE=np.zeros(np.shape(VPRM_TP39_2019_avg_GPP))*np.nan
TP39_2019_NEEgf=np.zeros(np.shape(VPRM_TP39_2019_avg_GPP))*np.nan
TP39_2019_R=np.zeros(np.shape(VPRM_TP39_2019_avg_GPP))*np.nan

for i in range(np.int(len(TP39_2019_dates)/2)):
    with np.errstate(invalid='ignore'):
        if i<8755:
            TP39_2019_GPP[i+5]=np.nanmean([TP39_2019_GPPgf_fluxes[i*2],TP39_2019_GPPgf_fluxes[i*2+1]])
            TP39_2019_NEE[i+5]=np.nanmean([TP39_2019_NEE_fluxes[i*2],TP39_2019_NEE_fluxes[i*2+1]])
            TP39_2019_NEEgf[i+5]=np.nanmean([TP39_2019_NEEgf_fluxes[i*2],TP39_2019_NEEgf_fluxes[i*2+1]])
            TP39_2019_R[i+5]=np.nanmean([TP39_2019_Rgf_fluxes[i*2],TP39_2019_Rgf_fluxes[i*2+1]])


# In[24]:


#Filter out erroneous NEE values between doy 195 and 198

# ***Optional: uncomment to visualize erroneous values
#plt.figure()
#plt.xlim(193,200)
#plt.scatter(VPRM_TP39_2019_avg_DoY,TP39_2019_NEE,label='TP39 NEE')
#plt.scatter(VPRM_TP39_2019_avg_DoY[(VPRM_TP39_2019_avg_DoY>195.05) & (VPRM_TP39_2019_avg_DoY<198.6)],TP39_2019_NEE[(VPRM_TP39_2019_avg_DoY>195.05) & (VPRM_TP39_2019_avg_DoY<198.6)],label='Erroneous TP39 NEE')
#plt.scatter(VPRM_TP39_2019_avg_DoY,VPRM_TP39_2019_avg_NEE,marker='*',label='UrbanVPRM NEE')
#plt.legend()
#plt.xlabel('Day of year, 2019')
#plt.ylabel('NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
#plt.title('Erroneous TP39 flux tower NEE values')
#plt.show()

# ***

TP39_2019_NEE[(VPRM_TP39_2019_avg_DoY>195.05) & (VPRM_TP39_2019_avg_DoY<198.6)]= np.nan


# In[25]:


# Select spring data over TP39 in 2019
#MAM: Doy 60 - 151 inclusive

MAM_time=VPRM_TP39_2019_avg_DoY[(np.round(VPRM_TP39_2019_avg_DoY,5)>=60) & (np.round(VPRM_TP39_2019_avg_DoY,5)<152)]
TP39_2019_GPPgf_MAM=TP39_2019_GPP[(np.round(VPRM_TP39_2019_avg_DoY,5)>=60) & (np.round(VPRM_TP39_2019_avg_DoY,5)<152)]
VPRM_TP39_2019_GPP_MAM=VPRM_TP39_2019_avg_GPP[(np.round(VPRM_TP39_2019_avg_DoY,5)>=60) & (np.round(VPRM_TP39_2019_avg_DoY,5)<152)]
Updated_VPRM_TP39_2019_GPP_MAM=Updated_VPRM_TP39_2019_avg_GPP[(np.round(VPRM_TP39_2019_avg_DoY,5)>=60) & (np.round(VPRM_TP39_2019_avg_DoY,5)<152)]

TP39_2019_Rgf_MAM=TP39_2019_R[(np.round(VPRM_TP39_2019_avg_DoY,5)>=60) & (np.round(VPRM_TP39_2019_avg_DoY,5)<152)]
VPRM_TP39_2019_Reco_MAM=VPRM_TP39_2019_avg_Reco[(np.round(VPRM_TP39_2019_avg_DoY,5)>=60) & (np.round(VPRM_TP39_2019_avg_DoY,5)<152)]
Updated_VPRM_TP39_2019_Reco_MAM=Updated_VPRM_TP39_2019_avg_Reco[(np.round(VPRM_TP39_2019_avg_DoY,5)>=60) & (np.round(VPRM_TP39_2019_avg_DoY,5)<152)]

TP39_2019_NEEgf_MAM=TP39_2019_NEEgf[(np.round(VPRM_TP39_2019_avg_DoY,5)>=60) & (np.round(VPRM_TP39_2019_avg_DoY,5)<152)]
TP39_2019_NEE_MAM=TP39_2019_NEE[(np.round(VPRM_TP39_2019_avg_DoY,5)>=60) & (np.round(VPRM_TP39_2019_avg_DoY,5)<152)]
VPRM_TP39_2019_NEE_MAM=VPRM_TP39_2019_avg_NEE[(np.round(VPRM_TP39_2019_avg_DoY,5)>=60) & (np.round(VPRM_TP39_2019_avg_DoY,5)<152)]
Updated_VPRM_TP39_2019_NEE_MAM=Updated_VPRM_TP39_2019_avg_NEE[(np.round(VPRM_TP39_2019_avg_DoY,5)>=60) & (np.round(VPRM_TP39_2019_avg_DoY,5)<152)]


# In[ ]:





# In[26]:


#Load in UrbanVPRM fluxes over TPD for 2018

#*** CHANGE PATHS & FILENAMES ***
VPRM_data=pd.read_csv('TPD_500m_V061_no_adjustments_2018/vprm_mixed_ISA_TPD_500m_V061_2018_no_adjustments.csv')
Updated_VPRM_data=pd.read_csv('TPD_V061_500m_2018/vprm_GMIS_Toronto_ACI_SOLRIS_ISA_500m_TPD_V061_2018_no_PScale_adjusted_Topt_Ra_URB_parameters_fixed_gapfilled_LSWI_filtered.csv')


# In[27]:


#Select pixels that fall within the flux tower footprint

VPRM_HoY0=np.zeros([8760,9])*np.nan
VPRM_Index0=np.zeros([8760,9])*np.nan
VPRM_GEE0=np.zeros([8760,9])*np.nan
VPRM_Reco0=np.zeros([8760,9])*np.nan

Updated_VPRM_HoY0=np.zeros([8760,9])*np.nan
Updated_VPRM_Index0=np.zeros([8760,9])*np.nan
Updated_VPRM_GEE0=np.zeros([8760,9])*np.nan
Updated_VPRM_Reco0=np.zeros([8760,9])*np.nan
h=0
l=0
for i in range(8760*103,8760*106):
    VPRM_HoY0[h,l]=VPRM_data.iat[i,1]
    VPRM_Index0[h,l]=VPRM_data.iat[i,2]
    VPRM_GEE0[h,l]=VPRM_data.iat[i,3]
    VPRM_Reco0[h,l]=VPRM_data.iat[i,9]
    
    Updated_VPRM_HoY0[h,l]=Updated_VPRM_data.iat[i,1]
    Updated_VPRM_Index0[h,l]=Updated_VPRM_data.iat[i,2]
    Updated_VPRM_GEE0[h,l]=Updated_VPRM_data.iat[i,3]
    Updated_VPRM_Reco0[h,l]=Updated_VPRM_data.iat[i,9]
    h+=1
    if VPRM_data.iat[i+1,2]>VPRM_data.iat[i,2]:
        l+=1
        h=0
for i in range(8760*119,8760*122):
    VPRM_HoY0[h,l]=VPRM_data.iat[i,1]
    VPRM_Index0[h,l]=VPRM_data.iat[i,2]
    VPRM_GEE0[h,l]=VPRM_data.iat[i,3]
    VPRM_Reco0[h,l]=VPRM_data.iat[i,9]
    
    Updated_VPRM_HoY0[h,l]=Updated_VPRM_data.iat[i,1]
    Updated_VPRM_Index0[h,l]=Updated_VPRM_data.iat[i,2]
    Updated_VPRM_GEE0[h,l]=Updated_VPRM_data.iat[i,3]
    Updated_VPRM_Reco0[h,l]=Updated_VPRM_data.iat[i,9]
    h+=1
    if VPRM_data.iat[i+1,2]>VPRM_data.iat[i,2]:
        l+=1
        h=0
for i in range(8760*135,8760*138):
    VPRM_HoY0[h,l]=VPRM_data.iat[i,1]
    VPRM_Index0[h,l]=VPRM_data.iat[i,2]
    VPRM_GEE0[h,l]=VPRM_data.iat[i,3]
    VPRM_Reco0[h,l]=VPRM_data.iat[i,9]
    
    Updated_VPRM_HoY0[h,l]=Updated_VPRM_data.iat[i,1]
    Updated_VPRM_Index0[h,l]=Updated_VPRM_data.iat[i,2]
    Updated_VPRM_GEE0[h,l]=Updated_VPRM_data.iat[i,3]
    Updated_VPRM_Reco0[h,l]=Updated_VPRM_data.iat[i,9]
    h+=1
    if VPRM_data.iat[i+1,2]>VPRM_data.iat[i,2]:
        l+=1
        h=0


# In[28]:


# Average pixels inside the footprint for each timestep

VPRM_TPD_avg_DoY=np.nanmean(VPRM_HoY0, axis=1)/24+23/24 #convert from hour 1 to DoY=1
VPRM_TPD_avg_Index=np.nanmean(VPRM_Index0, axis=1)
VPRM_TPD_avg_GPP=np.nanmean(VPRM_GEE0, axis=1) #- for updated version
VPRM_TPD_avg_Reco=np.nanmean(VPRM_Reco0, axis=1)
VPRM_TPD_avg_NEE=np.nanmean(VPRM_Reco0-VPRM_GEE0, axis=1) #switch sign for updated version

Updated_VPRM_TPD_avg_DoY=np.nanmean(Updated_VPRM_HoY0, axis=1)/24+23/24 #convert from hour 1 to DoY=1
Updated_VPRM_TPD_avg_Index=np.nanmean(Updated_VPRM_Index0, axis=1)
Updated_VPRM_TPD_avg_GPP=np.nanmean(-Updated_VPRM_GEE0, axis=1) #- for updated version
Updated_VPRM_TPD_avg_Reco=np.nanmean(Updated_VPRM_Reco0, axis=1)
Updated_VPRM_TPD_avg_NEE=np.nanmean(Updated_VPRM_Reco0+Updated_VPRM_GEE0, axis=1) #switch sign for updated version


# In[29]:


# Load in TPD 2018 flux tower fluxes
# *** CHANGE PATH & FILENAME ***
TPD_Fluxes=pd.read_csv('/Users/kitty/Documents/Research/SIF/Flux_Tower/Turkey_Point/TPD_HH_2018.csv', usecols=(0,2,74,75,76))

TPD_dates=np.zeros([17520])*np.nan
TPD_NEEgf_fluxes=np.zeros([17520])*np.nan
TPD_NEE_fluxes=np.zeros([17520])*np.nan
TPD_Rgf_fluxes=np.zeros([17520])*np.nan
TPD_GPPgf_fluxes=np.zeros([17520])*np.nan

for i in range(0,17520):
    if 201801010000<=TPD_Fluxes.iat[i,0]<201901010000:
        #TP is 5 hours behind UTC #adjust to UTC
        TPD_dates[i]=datetime.strptime(str(int(TPD_Fluxes.iat[i,0])),'%Y%m%d%H%M').timetuple().tm_yday+(float(str(TPD_Fluxes.iat[i,0])[8:10])+float(str(TPD_Fluxes.iat[i,0])[10:12])/60)/24+5/24

        TPD_NEEgf_fluxes[i]=TPD_Fluxes.iat[i,4] #NEE (gap filled)
        if TPD_Fluxes.iat[i,1]>-9999:
            TPD_NEE_fluxes[i]=TPD_Fluxes.iat[i,1]
        TPD_Rgf_fluxes[i]=TPD_Fluxes.iat[i,3]
        TPD_GPPgf_fluxes[i]=TPD_Fluxes.iat[i,2]


# In[ ]:





# In[30]:


TPD_GPP=np.zeros(np.shape(VPRM_TPD_avg_GPP))*np.nan
TPD_NEE=np.zeros(np.shape(VPRM_TPD_avg_GPP))*np.nan
TPD_NEEgf=np.zeros(np.shape(VPRM_TPD_avg_GPP))*np.nan
TPD_R=np.zeros(np.shape(VPRM_TPD_avg_GPP))*np.nan

for i in range(np.int(len(TPD_dates)/2)):
    if i<8755:
        with np.errstate(invalid='ignore'):
            TPD_GPP[i+5]=np.nanmean([TPD_GPPgf_fluxes[i*2],TPD_GPPgf_fluxes[i*2+1]])
            TPD_NEE[i+5]=np.nanmean([TPD_NEE_fluxes[i*2],TPD_NEE_fluxes[i*2+1]])
            TPD_NEEgf[i+5]=np.nanmean([TPD_NEEgf_fluxes[i*2],TPD_NEEgf_fluxes[i*2+1]])
            TPD_R[i+5]=np.nanmean([TPD_Rgf_fluxes[i*2],TPD_Rgf_fluxes[i*2+1]])


# In[31]:


#Select only spring (March-May) data over TPD
#MAM: Doy 60 - 151 inclusive

MAM_time=VPRM_TPD_avg_DoY[(np.round(VPRM_TPD_avg_DoY,5)>=60) & (np.round(VPRM_TPD_avg_DoY,5)<152)]
TPD_GPPgf_MAM=TPD_GPP[(np.round(VPRM_TPD_avg_DoY,5)>=60) & (np.round(VPRM_TPD_avg_DoY,5)<152)]
VPRM_TPD_GPP_MAM=VPRM_TPD_avg_GPP[(np.round(VPRM_TPD_avg_DoY,5)>=60) & (np.round(VPRM_TPD_avg_DoY,5)<152)]
Updated_VPRM_TPD_GPP_MAM=Updated_VPRM_TPD_avg_GPP[(np.round(VPRM_TPD_avg_DoY,5)>=60) & (np.round(VPRM_TPD_avg_DoY,5)<152)]

TPD_Rgf_MAM=TPD_R[(np.round(VPRM_TPD_avg_DoY,5)>=60) & (np.round(VPRM_TPD_avg_DoY,5)<152)]
VPRM_TPD_Reco_MAM=VPRM_TPD_avg_Reco[(np.round(VPRM_TPD_avg_DoY,5)>=60) & (np.round(VPRM_TPD_avg_DoY,5)<152)]
Updated_VPRM_TPD_Reco_MAM=Updated_VPRM_TPD_avg_Reco[(np.round(VPRM_TPD_avg_DoY,5)>=60) & (np.round(VPRM_TPD_avg_DoY,5)<152)]

TPD_NEEgf_MAM=TPD_NEEgf[(np.round(VPRM_TPD_avg_DoY,5)>=60) & (np.round(VPRM_TPD_avg_DoY,5)<152)]
TPD_NEE_MAM=TPD_NEE[(np.round(VPRM_TPD_avg_DoY,5)>=60) & (np.round(VPRM_TPD_avg_DoY,5)<152)]
VPRM_TPD_NEE_MAM=VPRM_TPD_avg_NEE[(np.round(VPRM_TPD_avg_DoY,5)>=60) & (np.round(VPRM_TPD_avg_DoY,5)<152)]
Updated_VPRM_TPD_NEE_MAM=Updated_VPRM_TPD_avg_NEE[(np.round(VPRM_TPD_avg_DoY,5)>=60) & (np.round(VPRM_TPD_avg_DoY,5)<152)]


# In[ ]:





# In[32]:


# Load in UrbanVPRM 2019 fluxes over TPD

# ***CHANGE PATHS & FILENAMES ***
VPRM_data=pd.read_csv('TPD_500m_V061_no_adjustments_2019/vprm_mixed_ISA_TPD_500m_V061_2019_no_adjustments.csv')

Updated_VPRM_data=pd.read_csv('TPD_V061_500m_2019/vprm_GMIS_Toronto_ACI_SOLRIS_ISA_500m_TPD_V061_2019_no_PScale_adjusted_Topt_Ra_URB_parameters_fixed_gapfilled_LSWI_filtered.csv')

VPRM_HoY0=np.zeros([8760,9])*np.nan
VPRM_Index0=np.zeros([8760,9])*np.nan
VPRM_GEE0=np.zeros([8760,9])*np.nan
VPRM_Reco0=np.zeros([8760,9])*np.nan

Updated_VPRM_HoY0=np.zeros([8760,9])*np.nan
Updated_VPRM_Index0=np.zeros([8760,9])*np.nan
Updated_VPRM_GEE0=np.zeros([8760,9])*np.nan
Updated_VPRM_Reco0=np.zeros([8760,9])*np.nan
h=0
l=0
for i in range(8760*103,8760*106):
    VPRM_HoY0[h,l]=VPRM_data.iat[i,1]
    VPRM_Index0[h,l]=VPRM_data.iat[i,2]
    VPRM_GEE0[h,l]=VPRM_data.iat[i,3]
    VPRM_Reco0[h,l]=VPRM_data.iat[i,9]
    
    Updated_VPRM_HoY0[h,l]=Updated_VPRM_data.iat[i,1]
    Updated_VPRM_Index0[h,l]=Updated_VPRM_data.iat[i,2]
    Updated_VPRM_GEE0[h,l]=Updated_VPRM_data.iat[i,3]
    Updated_VPRM_Reco0[h,l]=Updated_VPRM_data.iat[i,9]
    h+=1
    if VPRM_data.iat[i+1,2]>VPRM_data.iat[i,2]:
        l+=1
        h=0
for i in range(8760*119,8760*122):
    VPRM_HoY0[h,l]=VPRM_data.iat[i,1]
    VPRM_Index0[h,l]=VPRM_data.iat[i,2]
    VPRM_GEE0[h,l]=VPRM_data.iat[i,3]
    VPRM_Reco0[h,l]=VPRM_data.iat[i,9]
    
    Updated_VPRM_HoY0[h,l]=Updated_VPRM_data.iat[i,1]
    Updated_VPRM_Index0[h,l]=Updated_VPRM_data.iat[i,2]
    Updated_VPRM_GEE0[h,l]=Updated_VPRM_data.iat[i,3]
    Updated_VPRM_Reco0[h,l]=Updated_VPRM_data.iat[i,9]
    h+=1
    if VPRM_data.iat[i+1,2]>VPRM_data.iat[i,2]:
        l+=1
        h=0
for i in range(8760*135,8760*138):
    VPRM_HoY0[h,l]=VPRM_data.iat[i,1]
    VPRM_Index0[h,l]=VPRM_data.iat[i,2]
    VPRM_GEE0[h,l]=VPRM_data.iat[i,3]
    VPRM_Reco0[h,l]=VPRM_data.iat[i,9]
    
    Updated_VPRM_HoY0[h,l]=Updated_VPRM_data.iat[i,1]
    Updated_VPRM_Index0[h,l]=Updated_VPRM_data.iat[i,2]
    Updated_VPRM_GEE0[h,l]=Updated_VPRM_data.iat[i,3]
    Updated_VPRM_Reco0[h,l]=Updated_VPRM_data.iat[i,9]
    h+=1
    if VPRM_data.iat[i+1,2]>VPRM_data.iat[i,2]:
        l+=1
        h=0


# In[33]:


# Average fluxes within the footprint

#Original UrbanVPRM MODIS V061

VPRM_GEE0=VPRM_GEE0
VPRM_TPD_2019_avg_DoY=np.nanmean(VPRM_HoY0, axis=1)/24+23/24 #convert from hour 1 to DoY=1
VPRM_TPD_2019_avg_Index=np.nanmean(VPRM_Index0, axis=1)
VPRM_TPD_2019_avg_GPP=np.nanmean(VPRM_GEE0, axis=1)
VPRM_TPD_2019_avg_Reco=np.nanmean(VPRM_Reco0, axis=1)
VPRM_TPD_2019_avg_NEE=np.nanmean(VPRM_Reco0-VPRM_GEE0, axis=1)

#Updated UrbanVPRM MODIS V061
Updated_VPRM_TPD_2019_avg_DoY=np.nanmean(Updated_VPRM_HoY0, axis=1)/24+23/24 #convert from hour 1 to DoY=1
Updated_VPRM_TPD_2019_avg_Index=np.nanmean(Updated_VPRM_Index0, axis=1)
Updated_VPRM_TPD_2019_avg_GPP=-np.nanmean(Updated_VPRM_GEE0, axis=1)
Updated_VPRM_TPD_2019_avg_Reco=np.nanmean(Updated_VPRM_Reco0, axis=1)
Updated_VPRM_TPD_2019_avg_NEE=np.nanmean(Updated_VPRM_Reco0+Updated_VPRM_GEE0, axis=1)


# In[ ]:





# In[34]:


#Load in TPD 2019 fluxtower fluxes

# *** CHANGE PATH & FILENAME ***
TPD_2019_Fluxes=pd.read_csv('/Users/kitty/Documents/Research/SIF/Flux_Tower/Turkey_Point/TPD_HH_2019.csv',usecols=(0,2,74,75,76))

TPD_2019_dates=np.zeros([17520])*np.nan
TPD_2019_NEEgf_fluxes=np.zeros([17520])*np.nan
TPD_2019_NEE_fluxes=np.zeros([17520])*np.nan
TPD_2019_Rgf_fluxes=np.zeros([17520])*np.nan
TPD_2019_GPPgf_fluxes=np.zeros([17520])*np.nan
for i in range(0,17520):
    if 201901010000<=TPD_2019_Fluxes.iat[i,0]<202001010000:
        #TP is 5 hours behind UTC #adjust to UTC
        TPD_2019_dates[i]=datetime.strptime(str(int(TPD_2019_Fluxes.iat[i,0])),'%Y%m%d%H%M').timetuple().tm_yday+(float(str(TPD_2019_Fluxes.iat[i,0])[8:10])+float(str(TPD_2019_Fluxes.iat[i,0])[10:12])/60)/24+5/24

        if TPD_2019_Fluxes.iat[i,4]>-9999:
            TPD_2019_NEEgf_fluxes[i]=TPD_2019_Fluxes.iat[i,4] #NEE (gap filled)
        if TPD_2019_Fluxes.iat[i,1]>-9999:
            TPD_2019_NEE_fluxes[i]=TPD_2019_Fluxes.iat[i,1]
        if TPD_2019_Fluxes.iat[i,3]>-9999:
            TPD_2019_Rgf_fluxes[i]=TPD_2019_Fluxes.iat[i,3]
        if  TPD_2019_Fluxes.iat[i,2]>-9999:
            TPD_2019_GPPgf_fluxes[i]=TPD_2019_Fluxes.iat[i,2]


# In[ ]:





# In[35]:


TPD_2019_GPP=np.zeros(np.shape(VPRM_TPD_2019_avg_GPP))*np.nan
TPD_2019_NEE=np.zeros(np.shape(VPRM_TPD_2019_avg_GPP))*np.nan
TPD_2019_NEEgf=np.zeros(np.shape(VPRM_TPD_2019_avg_GPP))*np.nan
TPD_2019_R=np.zeros(np.shape(VPRM_TPD_2019_avg_GPP))*np.nan

for i in range(np.int(len(TPD_2019_dates)/2)):
    if i<8755:
        with np.errstate(invalid='ignore'):
            TPD_2019_GPP[i+5]=np.nansum([TPD_2019_GPPgf_fluxes[i*2],0.5*TPD_2019_GPPgf_fluxes[i*2+1]])/((~np.isnan(TPD_2019_GPPgf_fluxes[i*2])).sum()+0.5*(~np.isnan(TPD_2019_GPPgf_fluxes[i*2+1])).sum())
            TPD_2019_NEE[i+5]=np.nansum([TPD_2019_NEE_fluxes[i*2],0.5*TPD_2019_NEE_fluxes[i*2+1]])/((~np.isnan(TPD_2019_NEE_fluxes[i*2])).sum()+0.5*(~np.isnan(TPD_2019_NEE_fluxes[i*2+1])).sum())
            TPD_2019_NEEgf[i+5]=np.nansum([TPD_2019_NEEgf_fluxes[i*2],0.5*TPD_2019_NEEgf_fluxes[i*2+1]])/((~np.isnan(TPD_2019_NEEgf_fluxes[i*2])).sum()+0.5*(~np.isnan(TPD_2019_NEEgf_fluxes[i*2+1])).sum())
            TPD_2019_R[i+5]=np.nansum([TPD_2019_Rgf_fluxes[i*2],0.5*TPD_2019_Rgf_fluxes[i*2+1]])/((~np.isnan(TPD_2019_Rgf_fluxes[i*2])).sum()+0.5*(~np.isnan(TPD_2019_Rgf_fluxes[i*2+1])).sum())


# In[36]:


#MAM: Doy 60 - 151 inclusive

MAM_time=VPRM_TPD_2019_avg_DoY[(np.round(VPRM_TPD_2019_avg_DoY,5)>=60) & (np.round(VPRM_TPD_2019_avg_DoY,5)<152)]
TPD_2019_GPPgf_MAM=TPD_2019_GPP[(np.round(VPRM_TPD_2019_avg_DoY,5)>=60) & (np.round(VPRM_TPD_2019_avg_DoY,5)<152)]
VPRM_TPD_2019_GPP_MAM=VPRM_TPD_2019_avg_GPP[(np.round(VPRM_TPD_2019_avg_DoY,5)>=60) & (np.round(VPRM_TPD_2019_avg_DoY,5)<152)]
Updated_VPRM_TPD_2019_GPP_MAM=Updated_VPRM_TPD_2019_avg_GPP[(np.round(VPRM_TPD_2019_avg_DoY,5)>=60) & (np.round(VPRM_TPD_2019_avg_DoY,5)<152)]

TPD_2019_Rgf_MAM=TPD_2019_R[(np.round(VPRM_TPD_2019_avg_DoY,5)>=60) & (np.round(VPRM_TPD_2019_avg_DoY,5)<152)]
VPRM_TPD_2019_Reco_MAM=VPRM_TPD_2019_avg_Reco[(np.round(VPRM_TPD_2019_avg_DoY,5)>=60) & (np.round(VPRM_TPD_2019_avg_DoY,5)<152)]
Updated_VPRM_TPD_2019_Reco_MAM=Updated_VPRM_TPD_2019_avg_Reco[(np.round(VPRM_TPD_2019_avg_DoY,5)>=60) & (np.round(VPRM_TPD_2019_avg_DoY,5)<152)]

TPD_2019_NEEgf_MAM=TPD_2019_NEEgf[(np.round(VPRM_TPD_2019_avg_DoY,5)>=60) & (np.round(VPRM_TPD_2019_avg_DoY,5)<152)]
TPD_2019_NEE_MAM=TPD_2019_NEE[(np.round(VPRM_TPD_2019_avg_DoY,5)>=60) & (np.round(VPRM_TPD_2019_avg_DoY,5)<152)]
VPRM_TPD_2019_NEE_MAM=VPRM_TPD_2019_avg_NEE[(np.round(VPRM_TPD_2019_avg_DoY,5)>=60) & (np.round(VPRM_TPD_2019_avg_DoY,5)<152)]
Updated_VPRM_TPD_2019_NEE_MAM=Updated_VPRM_TPD_2019_avg_NEE[(np.round(VPRM_TPD_2019_avg_DoY,5)>=60) & (np.round(VPRM_TPD_2019_avg_DoY,5)<152)]


# In[ ]:





# In[37]:


# Combine data from all sites in spring
MAM_VPRM_GPP=np.concatenate([VPRM_Borden_GPP_MAM,VPRM_TP39_GPP_MAM,VPRM_TPD_GPP_MAM,VPRM_TP39_2019_GPP_MAM,VPRM_TPD_2019_GPP_MAM])
MAM_Updated_VPRM_GPP=np.concatenate([Updated_VPRM_Borden_GPP_MAM,Updated_VPRM_TP39_GPP_MAM,Updated_VPRM_TPD_GPP_MAM,Updated_VPRM_TP39_2019_GPP_MAM,Updated_VPRM_TPD_2019_GPP_MAM])
MAM_GPP=np.concatenate([Borden_GPPgf_MAM,TP39_GPPgf_MAM,TPD_GPPgf_MAM,TP39_2019_GPPgf_MAM,TPD_2019_GPPgf_MAM])

MAM_VPRM_Reco=np.concatenate([VPRM_Borden_Reco_MAM,VPRM_TP39_Reco_MAM,VPRM_TPD_Reco_MAM,VPRM_TP39_2019_Reco_MAM,VPRM_TPD_2019_Reco_MAM])
MAM_Updated_VPRM_Reco=np.concatenate([Updated_VPRM_Borden_Reco_MAM,Updated_VPRM_TP39_Reco_MAM,Updated_VPRM_TPD_Reco_MAM,Updated_VPRM_TP39_2019_Reco_MAM,Updated_VPRM_TPD_2019_Reco_MAM])
MAM_Reco=np.concatenate([Borden_Rgf_MAM,TP39_Rgf_MAM,TPD_Rgf_MAM,TP39_2019_Rgf_MAM,TPD_2019_Rgf_MAM])

MAM_VPRM_NEE=np.concatenate([VPRM_Borden_NEE_MAM,VPRM_TP39_NEE_MAM,VPRM_TPD_NEE_MAM,VPRM_TP39_2019_NEE_MAM,VPRM_TPD_2019_NEE_MAM])
MAM_Updated_VPRM_NEE=np.concatenate([Updated_VPRM_Borden_NEE_MAM,Updated_VPRM_TP39_NEE_MAM,Updated_VPRM_TPD_NEE_MAM,Updated_VPRM_TP39_2019_NEE_MAM,Updated_VPRM_TPD_2019_NEE_MAM])
MAM_NEEgf=np.concatenate([Borden_NEEgf_MAM,TP39_NEEgf_MAM,TPD_NEEgf_MAM,TP39_2019_NEEgf_MAM,TPD_2019_NEEgf_MAM])
MAM_NEE=np.concatenate([Borden_NEE_MAM,TP39_NEE_MAM,TPD_NEE_MAM,TP39_2019_NEE_MAM,TPD_2019_NEE_MAM])


# In[ ]:





# In[38]:


# define a linear function (used for plotting fits)

def func2(x,m,b):
    return m*x+b

line1_1=np.arange(-100,100)


# In[39]:


#Fit original UrbanVPRM NEE to non-gapfilled flux tower NEE for spring using a bootstrapped Huber fit

finitemask0=np.isfinite(MAM_NEE)
MAM_NEEclean0=MAM_NEE[finitemask0]
MAM_VPRM_NEEclean0=MAM_VPRM_NEE[finitemask0]

finitemask1=np.isfinite(MAM_VPRM_NEEclean0)
MAM_NEEclean1=MAM_NEEclean0[finitemask1]
MAM_VPRM_NEEclean1=MAM_VPRM_NEEclean0[finitemask1]

Huber_MAM_NEE_slps=[]
Huber_MAM_NEE_ints=[]
Huber_MAM_NEE_R2=[]

#try bootstrapping
indx_list=list(range(0,len(MAM_VPRM_NEEclean1)))
for i in range(1,1000):
    #sub selection of points
    NEE_indx=np.random.choice(indx_list,size=len(MAM_VPRM_NEEclean1))
    
    try:
        Huber_model = linear_model.HuberRegressor(fit_intercept=True)
        Huber_fit=Huber_model.fit((MAM_NEEclean1[NEE_indx]).reshape(-1,1),MAM_VPRM_NEEclean1[NEE_indx])
        H_m=Huber_fit.coef_
        H_c=Huber_fit.intercept_
        x_accpt, y_accpt = MAM_NEEclean1, MAM_VPRM_NEEclean1
        y_predict = H_m * x_accpt + H_c
        H_R2=r2_score(y_accpt, y_predict)
        Huber_MAM_NEE_slps.append(H_m)
        Huber_MAM_NEE_ints.append(H_c)
        Huber_MAM_NEE_R2.append(H_R2)
    except ValueError: #if Huber fit can't find a solution for the subset, skip it
        pass
    
Huber_MAM_R2 = r2_score(MAM_VPRM_NEEclean1, MAM_NEEclean1*np.nanmean(Huber_MAM_NEE_slps)+np.nanmean(Huber_MAM_NEE_ints))

print('Original UrbanVPRM MAM slope: '+str(np.round(np.nanmean(Huber_MAM_NEE_slps),3))+' +/- '+str(np.round(np.nanstd(Huber_MAM_NEE_slps),3)))
print('Original UrbanVPRM MAM intercept: '+str(np.round(np.nanmean(Huber_MAM_NEE_ints),3))+' +/- '+str(np.round(np.nanstd(Huber_MAM_NEE_ints),3)))

print('Original UrbanVPRM MAM R^2: '+str(np.round(np.nanmean(Huber_MAM_R2),3)))


# In[ ]:





# In[40]:


#Fit Updated UrbanVPRM NEE to non-gapfilled flux tower NEE for spring using a bootstrapped Huber fit

finitemask0=np.isfinite(MAM_NEE)
MAM_NEEclean0=MAM_NEE[finitemask0]
MAM_Updated_VPRM_NEEclean0=MAM_Updated_VPRM_NEE[finitemask0]

finitemask1=np.isfinite(MAM_Updated_VPRM_NEEclean0)
MAM_NEEclean1=MAM_NEEclean0[finitemask1]
MAM_Updated_VPRM_NEEclean1=MAM_Updated_VPRM_NEEclean0[finitemask1]

Huber_MAM_Updated_NEE_slps=[]
Huber_MAM_Updated_NEE_ints=[]
Huber_MAM_Updated_NEE_R2=[]

#try bootstrapping
indx_list=list(range(0,len(MAM_Updated_VPRM_NEEclean1)))
for i in range(1,1000):
    #sub selection of points
    NEE_indx=np.random.choice(indx_list,size=len(MAM_Updated_VPRM_NEEclean1))
    
    try:
        Huber_model = linear_model.HuberRegressor(fit_intercept=True)
        Huber_fit=Huber_model.fit((MAM_NEEclean1[NEE_indx]).reshape(-1,1),MAM_Updated_VPRM_NEEclean1[NEE_indx])
        H_m=Huber_fit.coef_
        H_c=Huber_fit.intercept_
        x_accpt, y_accpt = MAM_NEEclean1, MAM_Updated_VPRM_NEEclean1
        y_predict = H_m * x_accpt + H_c
        H_R2=r2_score(y_accpt, y_predict)
        Huber_MAM_Updated_NEE_slps.append(H_m)
        Huber_MAM_Updated_NEE_ints.append(H_c)
        Huber_MAM_Updated_NEE_R2.append(H_R2)
    except ValueError: #if Huber fit can't find a solution for the subset, skip it
        pass
    

Huber_MAM_Updated_R2 = r2_score(MAM_Updated_VPRM_NEEclean1, MAM_NEEclean1*np.nanmean(Huber_MAM_Updated_NEE_slps)+np.nanmean(Huber_MAM_Updated_NEE_ints))

print('Updated UrbanVPRM MAM slope: '+str(np.round(np.nanmean(Huber_MAM_Updated_NEE_slps),3))+' +/- '+str(np.round(np.nanstd(Huber_MAM_Updated_NEE_slps),3)))
print('Updated UrbanVPRM MAM intercept: '+str(np.round(np.nanmean(Huber_MAM_Updated_NEE_ints),3))+' +/- '+str(np.round(np.nanstd(Huber_MAM_Updated_NEE_ints),3)))

print('Updated UrbanVPRM MAM R^2: '+str(np.round(np.nanmean(Huber_MAM_Updated_R2),3)))


# In[146]:





# ### Now look at Summer (JJA)

# In[41]:


#JJA: Doy 152 - 223 inclusive

JJA_time=VPRM_Borden_2018_avg_DoY[(np.round(VPRM_Borden_2018_avg_DoY,5)>=152) & (np.round(VPRM_Borden_2018_avg_DoY,5)<224)]
Borden_GPPgf_JJA=Borden_GEPgf[(np.round(VPRM_Borden_2018_avg_DoY,5)>=152) & (np.round(VPRM_Borden_2018_avg_DoY,5)<224)]
VPRM_Borden_GPP_JJA=VPRM_Borden_2018_avg_GPP[(np.round(VPRM_Borden_2018_avg_DoY,5)>=152) & (np.round(VPRM_Borden_2018_avg_DoY,5)<224)]
Updated_VPRM_Borden_GPP_JJA=Updated_VPRM_Borden_2018_avg_GPP[(np.round(VPRM_Borden_2018_avg_DoY,5)>=152) & (np.round(VPRM_Borden_2018_avg_DoY,5)<224)]

Borden_Rgf_JJA=Borden_Rgf[(np.round(VPRM_Borden_2018_avg_DoY,5)>=152) & (np.round(VPRM_Borden_2018_avg_DoY,5)<224)]
VPRM_Borden_Reco_JJA=VPRM_Borden_2018_avg_Reco[(np.round(VPRM_Borden_2018_avg_DoY,5)>=152) & (np.round(VPRM_Borden_2018_avg_DoY,5)<224)]
Updated_VPRM_Borden_Reco_JJA=Updated_VPRM_Borden_2018_avg_Reco[(np.round(VPRM_Borden_2018_avg_DoY,5)>=152) & (np.round(VPRM_Borden_2018_avg_DoY,5)<224)]

Borden_NEEgf_JJA=Borden_NEEgf[(np.round(VPRM_Borden_2018_avg_DoY,5)>=152) & (np.round(VPRM_Borden_2018_avg_DoY,5)<224)]
Borden_NEE_JJA=Borden_NEE[(np.round(VPRM_Borden_2018_avg_DoY,5)>=152) & (np.round(VPRM_Borden_2018_avg_DoY,5)<224)]
VPRM_Borden_NEE_JJA=VPRM_Borden_2018_avg_NEE[(np.round(VPRM_Borden_2018_avg_DoY,5)>=152) & (np.round(VPRM_Borden_2018_avg_DoY,5)<224)]
Updated_VPRM_Borden_NEE_JJA=Updated_VPRM_Borden_2018_avg_NEE[(np.round(VPRM_Borden_2018_avg_DoY,5)>=152) & (np.round(VPRM_Borden_2018_avg_DoY,5)<224)]


# In[42]:


#JJA: Doy 152 - 223 inclusive

JJA_time=VPRM_TP39_2018_avg_DoY[(np.round(VPRM_TP39_2018_avg_DoY,5)>=152) & (np.round(VPRM_TP39_2018_avg_DoY,5)<224)]
TP39_GPPgf_JJA=TP39_GPP[(np.round(VPRM_TP39_2018_avg_DoY,5)>=152) & (np.round(VPRM_TP39_2018_avg_DoY,5)<224)]
VPRM_TP39_GPP_JJA=VPRM_TP39_2018_avg_GPP[(np.round(VPRM_TP39_2018_avg_DoY,5)>=152) & (np.round(VPRM_TP39_2018_avg_DoY,5)<224)]
Updated_VPRM_TP39_GPP_JJA=Updated_VPRM_TP39_2018_avg_GPP[(np.round(VPRM_TP39_2018_avg_DoY,5)>=152) & (np.round(VPRM_TP39_2018_avg_DoY,5)<224)]

TP39_Rgf_JJA=TP39_R[(np.round(VPRM_TP39_2018_avg_DoY,5)>=152) & (np.round(VPRM_TP39_2018_avg_DoY,5)<224)]
VPRM_TP39_Reco_JJA=VPRM_TP39_2018_avg_Reco[(np.round(VPRM_TP39_2018_avg_DoY,5)>=152) & (np.round(VPRM_TP39_2018_avg_DoY,5)<224)]
Updated_VPRM_TP39_Reco_JJA=Updated_VPRM_TP39_2018_avg_Reco[(np.round(VPRM_TP39_2018_avg_DoY,5)>=152) & (np.round(VPRM_TP39_2018_avg_DoY,5)<224)]

TP39_NEEgf_JJA=TP39_NEEgf[(np.round(VPRM_TP39_2018_avg_DoY,5)>=152) & (np.round(VPRM_TP39_2018_avg_DoY,5)<224)]
TP39_NEE_JJA=TP39_NEE[(np.round(VPRM_TP39_2018_avg_DoY,5)>=152) & (np.round(VPRM_TP39_2018_avg_DoY,5)<224)]
VPRM_TP39_NEE_JJA=VPRM_TP39_2018_avg_NEE[(np.round(VPRM_TP39_2018_avg_DoY,5)>=152) & (np.round(VPRM_TP39_2018_avg_DoY,5)<224)]
Updated_VPRM_TP39_NEE_JJA=Updated_VPRM_TP39_2018_avg_NEE[(np.round(VPRM_TP39_2018_avg_DoY,5)>=152) & (np.round(VPRM_TP39_2018_avg_DoY,5)<224)]


# In[43]:


#JJA: Doy 152 - 223 inclusive

JJA_time=VPRM_TP39_2019_avg_DoY[(np.round(VPRM_TP39_2019_avg_DoY,5)>=152) & (np.round(VPRM_TP39_2019_avg_DoY,5)<224)]
TP39_2019_GPPgf_JJA=TP39_2019_GPP[(np.round(VPRM_TP39_2019_avg_DoY,5)>=152) & (np.round(VPRM_TP39_2019_avg_DoY,5)<224)]
VPRM_TP39_2019_GPP_JJA=VPRM_TP39_2019_avg_GPP[(np.round(VPRM_TP39_2019_avg_DoY,5)>=152) & (np.round(VPRM_TP39_2019_avg_DoY,5)<224)]
Updated_VPRM_TP39_2019_GPP_JJA=Updated_VPRM_TP39_2019_avg_GPP[(np.round(VPRM_TP39_2019_avg_DoY,5)>=152) & (np.round(VPRM_TP39_2019_avg_DoY,5)<224)]

TP39_2019_Rgf_JJA=TP39_2019_R[(np.round(VPRM_TP39_2019_avg_DoY,5)>=152) & (np.round(VPRM_TP39_2019_avg_DoY,5)<224)]
VPRM_TP39_2019_Reco_JJA=VPRM_TP39_2019_avg_Reco[(np.round(VPRM_TP39_2019_avg_DoY,5)>=152) & (np.round(VPRM_TP39_2019_avg_DoY,5)<224)]
Updated_VPRM_TP39_2019_Reco_JJA=Updated_VPRM_TP39_2019_avg_Reco[(np.round(VPRM_TP39_2019_avg_DoY,5)>=152) & (np.round(VPRM_TP39_2019_avg_DoY,5)<224)]

TP39_2019_NEEgf_JJA=TP39_2019_NEEgf[(np.round(VPRM_TP39_2019_avg_DoY,5)>=152) & (np.round(VPRM_TP39_2019_avg_DoY,5)<224)]
TP39_2019_NEE_JJA=TP39_2019_NEE[(np.round(VPRM_TP39_2019_avg_DoY,5)>=152) & (np.round(VPRM_TP39_2019_avg_DoY,5)<224)]
VPRM_TP39_2019_NEE_JJA=VPRM_TP39_2019_avg_NEE[(np.round(VPRM_TP39_2019_avg_DoY,5)>=152) & (np.round(VPRM_TP39_2019_avg_DoY,5)<224)]
Updated_VPRM_TP39_2019_NEE_JJA=Updated_VPRM_TP39_2019_avg_NEE[(np.round(VPRM_TP39_2019_avg_DoY,5)>=152) & (np.round(VPRM_TP39_2019_avg_DoY,5)<224)]


# In[44]:


#JJA: Doy 152 - 223 inclusive

JJA_time=VPRM_TPD_avg_DoY[(np.round(VPRM_TPD_avg_DoY,5)>=152) & (np.round(VPRM_TPD_avg_DoY,5)<224)]
TPD_GPPgf_JJA=TPD_GPP[(np.round(VPRM_TPD_avg_DoY,5)>=152) & (np.round(VPRM_TPD_avg_DoY,5)<224)]
VPRM_TPD_GPP_JJA=VPRM_TPD_avg_GPP[(np.round(VPRM_TPD_avg_DoY,5)>=152) & (np.round(VPRM_TPD_avg_DoY,5)<224)]
Updated_VPRM_TPD_GPP_JJA=Updated_VPRM_TPD_avg_GPP[(np.round(VPRM_TPD_avg_DoY,5)>=152) & (np.round(VPRM_TPD_avg_DoY,5)<224)]

TPD_Rgf_JJA=TPD_R[(np.round(VPRM_TPD_avg_DoY,5)>=152) & (np.round(VPRM_TPD_avg_DoY,5)<224)]
VPRM_TPD_Reco_JJA=VPRM_TPD_avg_Reco[(np.round(VPRM_TPD_avg_DoY,5)>=152) & (np.round(VPRM_TPD_avg_DoY,5)<224)]
Updated_VPRM_TPD_Reco_JJA=Updated_VPRM_TPD_avg_Reco[(np.round(VPRM_TPD_avg_DoY,5)>=152) & (np.round(VPRM_TPD_avg_DoY,5)<224)]

TPD_NEEgf_JJA=TPD_NEEgf[(np.round(VPRM_TPD_avg_DoY,5)>=152) & (np.round(VPRM_TPD_avg_DoY,5)<224)]
TPD_NEE_JJA=TPD_NEE[(np.round(VPRM_TPD_avg_DoY,5)>=152) & (np.round(VPRM_TPD_avg_DoY,5)<224)]
VPRM_TPD_NEE_JJA=VPRM_TPD_avg_NEE[(np.round(VPRM_TPD_avg_DoY,5)>=152) & (np.round(VPRM_TPD_avg_DoY,5)<224)]
Updated_VPRM_TPD_NEE_JJA=Updated_VPRM_TPD_avg_NEE[(np.round(VPRM_TPD_avg_DoY,5)>=152) & (np.round(VPRM_TPD_avg_DoY,5)<224)]


# In[45]:


#JJA: Doy 152 - 223 inclusive

JJA_time=VPRM_TPD_2019_avg_DoY[(np.round(VPRM_TPD_2019_avg_DoY,5)>=152) & (np.round(VPRM_TPD_2019_avg_DoY,5)<224)]
TPD_2019_GPPgf_JJA=TPD_2019_GPP[(np.round(VPRM_TPD_2019_avg_DoY,5)>=152) & (np.round(VPRM_TPD_2019_avg_DoY,5)<224)]
VPRM_TPD_2019_GPP_JJA=VPRM_TPD_2019_avg_GPP[(np.round(VPRM_TPD_2019_avg_DoY,5)>=152) & (np.round(VPRM_TPD_2019_avg_DoY,5)<224)]
Updated_VPRM_TPD_2019_GPP_JJA=Updated_VPRM_TPD_2019_avg_GPP[(np.round(VPRM_TPD_2019_avg_DoY,5)>=152) & (np.round(VPRM_TPD_2019_avg_DoY,5)<224)]

TPD_2019_Rgf_JJA=TPD_2019_R[(np.round(VPRM_TPD_2019_avg_DoY,5)>=152) & (np.round(VPRM_TPD_2019_avg_DoY,5)<224)]
VPRM_TPD_2019_Reco_JJA=VPRM_TPD_2019_avg_Reco[(np.round(VPRM_TPD_2019_avg_DoY,5)>=152) & (np.round(VPRM_TPD_2019_avg_DoY,5)<224)]
Updated_VPRM_TPD_2019_Reco_JJA=Updated_VPRM_TPD_2019_avg_Reco[(np.round(VPRM_TPD_2019_avg_DoY,5)>=152) & (np.round(VPRM_TPD_2019_avg_DoY,5)<224)]

TPD_2019_NEEgf_JJA=TPD_2019_NEEgf[(np.round(VPRM_TPD_2019_avg_DoY,5)>=152) & (np.round(VPRM_TPD_2019_avg_DoY,5)<224)]
TPD_2019_NEE_JJA=TPD_2019_NEE[(np.round(VPRM_TPD_2019_avg_DoY,5)>=152) & (np.round(VPRM_TPD_2019_avg_DoY,5)<224)]
VPRM_TPD_2019_NEE_JJA=VPRM_TPD_2019_avg_NEE[(np.round(VPRM_TPD_2019_avg_DoY,5)>=152) & (np.round(VPRM_TPD_2019_avg_DoY,5)<224)]
Updated_VPRM_TPD_2019_NEE_JJA=Updated_VPRM_TPD_2019_avg_NEE[(np.round(VPRM_TPD_2019_avg_DoY,5)>=152) & (np.round(VPRM_TPD_2019_avg_DoY,5)<224)]


# In[46]:


#Combine all summer data
JJA_VPRM_GPP=np.concatenate([VPRM_Borden_GPP_JJA,VPRM_TP39_GPP_JJA,VPRM_TPD_GPP_JJA,VPRM_TP39_2019_GPP_JJA,VPRM_TPD_2019_GPP_JJA])
JJA_Updated_VPRM_GPP=np.concatenate([Updated_VPRM_Borden_GPP_JJA,Updated_VPRM_TP39_GPP_JJA,Updated_VPRM_TPD_GPP_JJA,Updated_VPRM_TP39_2019_GPP_JJA,Updated_VPRM_TPD_2019_GPP_JJA])
JJA_GPP=np.concatenate([Borden_GPPgf_JJA,TP39_GPPgf_JJA,TPD_GPPgf_JJA,TP39_2019_GPPgf_JJA,TPD_2019_GPPgf_JJA])

JJA_VPRM_Reco=np.concatenate([VPRM_Borden_Reco_JJA,VPRM_TP39_Reco_JJA,VPRM_TPD_Reco_JJA,VPRM_TP39_2019_Reco_JJA,VPRM_TPD_2019_Reco_JJA])
JJA_Updated_VPRM_Reco=np.concatenate([Updated_VPRM_Borden_Reco_JJA,Updated_VPRM_TP39_Reco_JJA,Updated_VPRM_TPD_Reco_JJA,Updated_VPRM_TP39_2019_Reco_JJA,Updated_VPRM_TPD_2019_Reco_JJA])
JJA_Reco=np.concatenate([Borden_Rgf_JJA,TP39_Rgf_JJA,TPD_Rgf_JJA,TP39_2019_Rgf_JJA,TPD_2019_Rgf_JJA])

JJA_VPRM_NEE=np.concatenate([VPRM_Borden_NEE_JJA,VPRM_TP39_NEE_JJA,VPRM_TPD_NEE_JJA,VPRM_TP39_2019_NEE_JJA,VPRM_TPD_2019_NEE_JJA])
JJA_Updated_VPRM_NEE=np.concatenate([Updated_VPRM_Borden_NEE_JJA,Updated_VPRM_TP39_NEE_JJA,Updated_VPRM_TPD_NEE_JJA,Updated_VPRM_TP39_2019_NEE_JJA,Updated_VPRM_TPD_2019_NEE_JJA])
JJA_NEEgf=np.concatenate([Borden_NEEgf_JJA,TP39_NEEgf_JJA,TPD_NEEgf_JJA,TP39_2019_NEEgf_JJA,TPD_2019_NEEgf_JJA])
JJA_NEE=np.concatenate([Borden_NEE_JJA,TP39_NEE_JJA,TPD_NEE_JJA,TP39_2019_NEE_JJA,TPD_2019_NEE_JJA])


# In[47]:


#Fit original UrbanVPRM NEE to non-gapfilled flux tower NEE for summer using a bootstrapped Huber fit

finitemask0=np.isfinite(JJA_NEE)
JJA_NEEclean0=JJA_NEE[finitemask0]
JJA_VPRM_NEEclean0=JJA_VPRM_NEE[finitemask0]

finitemask1=np.isfinite(JJA_VPRM_NEEclean0)
JJA_NEEclean1=JJA_NEEclean0[finitemask1]
JJA_VPRM_NEEclean1=JJA_VPRM_NEEclean0[finitemask1]

Huber_JJA_NEE_slps=[]
Huber_JJA_NEE_ints=[]
Huber_JJA_NEE_R2=[]

#try bootstrapping
indx_list=list(range(0,len(JJA_VPRM_NEEclean1)))
for i in range(1,1000):
    #sub selection of points
    NEE_indx=np.random.choice(indx_list,size=len(JJA_VPRM_NEEclean1))
    
    try:
        Huber_model = linear_model.HuberRegressor(fit_intercept=True)
        Huber_fit=Huber_model.fit((JJA_NEEclean1[NEE_indx]).reshape(-1,1),JJA_VPRM_NEEclean1[NEE_indx])
        H_m=Huber_fit.coef_
        H_c=Huber_fit.intercept_
        x_accpt, y_accpt = JJA_NEEclean1, JJA_VPRM_NEEclean1
        y_predict = H_m * x_accpt + H_c
        H_R2=r2_score(y_accpt, y_predict)
        Huber_JJA_NEE_slps.append(H_m)
        Huber_JJA_NEE_ints.append(H_c)
        Huber_JJA_NEE_R2.append(H_R2)
    except ValueError: #if Huber fit can't find a solution for the subset, skip it
        pass
    
Huber_JJA_R2 = r2_score(JJA_VPRM_NEEclean1, JJA_NEEclean1*np.nanmean(Huber_JJA_NEE_slps)+np.nanmean(Huber_JJA_NEE_ints))

print('Original UrbanVPRM JJA slope: '+str(np.round(np.nanmean(Huber_JJA_NEE_slps),3))+' +/- '+str(np.round(np.nanstd(Huber_JJA_NEE_slps),3)))
print('Original UrbanVPRM JJA intercept: '+str(np.round(np.nanmean(Huber_JJA_NEE_ints),3))+' +/- '+str(np.round(np.nanstd(Huber_JJA_NEE_ints),3)))

print('Original UrbanVPRM JJA R^2: '+str(np.round(np.nanmean(Huber_JJA_R2),3)))


# In[48]:


#Fit updated UrbanVPRM NEE to non-gapfilled flux tower NEE for summer using a bootstrapped Huber fit

finitemask0=np.isfinite(JJA_NEE)
JJA_NEEclean0=JJA_NEE[finitemask0]
JJA_Updated_VPRM_NEEclean0=JJA_Updated_VPRM_NEE[finitemask0]

finitemask1=np.isfinite(JJA_Updated_VPRM_NEEclean0)
JJA_NEEclean1=JJA_NEEclean0[finitemask1]
JJA_Updated_VPRM_NEEclean1=JJA_Updated_VPRM_NEEclean0[finitemask1]

Huber_JJA_Updated_NEE_slps=[]
Huber_JJA_Updated_NEE_ints=[]
Huber_JJA_Updated_NEE_R2=[]

#try bootstrapping
indx_list=list(range(0,len(JJA_Updated_VPRM_NEEclean1)))
for i in range(1,1000):
    #sub selection of points
    NEE_indx=np.random.choice(indx_list,size=len(JJA_Updated_VPRM_NEEclean1))
    
    try:
        Huber_model = linear_model.HuberRegressor(fit_intercept=True)
        Huber_fit=Huber_model.fit((JJA_NEEclean1[NEE_indx]).reshape(-1,1),JJA_Updated_VPRM_NEEclean1[NEE_indx])
        H_m=Huber_fit.coef_
        H_c=Huber_fit.intercept_
        x_accpt, y_accpt = JJA_NEEclean1, JJA_Updated_VPRM_NEEclean1
        y_predict = H_m * x_accpt + H_c
        H_R2=r2_score(y_accpt, y_predict)
        Huber_JJA_Updated_NEE_slps.append(H_m)
        Huber_JJA_Updated_NEE_ints.append(H_c)
        Huber_JJA_Updated_NEE_R2.append(H_R2)
    except ValueError: #if Huber fit can't find a solution for the subset, skip it
        pass

Huber_JJA_Updated_R2 = r2_score(JJA_Updated_VPRM_NEEclean1, JJA_NEEclean1*np.nanmean(Huber_JJA_Updated_NEE_slps)+np.nanmean(Huber_JJA_Updated_NEE_ints))

print('Updated UrbanVPRM JJA slope: '+str(np.round(np.nanmean(Huber_JJA_Updated_NEE_slps),3))+' +/- '+str(np.round(np.nanstd(Huber_JJA_Updated_NEE_slps),3)))
print('Updated UrbanVPRM JJA intercept: '+str(np.round(np.nanmean(Huber_JJA_Updated_NEE_ints),3))+' +/- '+str(np.round(np.nanstd(Huber_JJA_Updated_NEE_ints),3)))

print('Updated UrbanVPRM JJA R^2: '+str(np.round(np.nanmean(Huber_JJA_Updated_R2),3)))


# In[ ]:





# In[49]:


# *** Optional: Bring in SMUrF data (needed for fig S3) 
# *** NEED TO RUN 'TROPOMI_SMUrF_CSIF_SMUrF_Fluxtower_Seasonal_Comparison-V061_clean.py' & SAVE SMURF FLUXES OVER BORDEN FOREST ***

#*** CHANGE PATH & FILENAME ****
g = Dataset('E:/Research/SMUrF/output2018_500m_CSIF_to_TROPOMI_CSIF_ALL_converted_slps_temp_impervious_R_V061_8day/SMUrF_Borden_fluxes.nc')
S_NEE_Borden=g.variables['NEE'][:]
S_NEE_std_Borden = g.variables['NEE_std'][:]
S_GPP_Borden=g.variables['GPP'][:]
S_GPP_std_Borden = g.variables['GPP_std'][:]
S_Reco_Borden=g.variables['Reco'][:]
S_Reco_std_Borden = g.variables['Reco_std'][:]
S_time=g.variables['time'][:]+5/24
g.close()

with np.errstate(invalid='ignore'):
    S_Borden_GPP_JJA=S_GPP_Borden[(np.round(S_time,5)>=152) & (np.round(S_time,5)<224)]
    S_Borden_NEE_JJA=S_NEE_Borden[(np.round(S_time,5)>=152) & (np.round(S_time,5)<224)]
    S_Borden_Reco_JJA=S_Reco_Borden[(np.round(S_time,5)>=152) & (np.round(S_time,5)<224)]


# In[49]:


import matplotlib.patches as ptchs

plt.style.use('tableau-colorblind10')

fig,ax=plt.subplots(figsize=(10,6))
plt.xlim(192.1,195)
plt.ylim(-4,63)

for i in range(len(sunrise_dates)):
    plt.axvline(sunrise_dates[i],c='red',linestyle=':')
    plt.axvline(sunset_dates[i],c='k',linestyle=':')
    
plt.axvline(sunrise_dates[0]+5/24,c='red',linestyle=':',label='sunrise')
plt.axvline(sunset_dates[0]+5/24,c='k',linestyle=':',label='sunset')
plt.scatter(JJA_time-5/24,Updated_VPRM_Borden_GPP_JJA,marker='*',label='UrbanVPRM')
plt.scatter(JJA_time-5/24,S_Borden_GPP_JJA,marker='d',label='SMUrF')
plt.scatter(JJA_time-5/24,Borden_GPPgf_JJA,c='#595959',label='Fluxtower GPP')
circ=ptchs.Ellipse((192.43-5/24,1.5),0.1,7,fill=False,color='g')
ax.add_artist(circ)
circ=ptchs.Ellipse((193.02-5/24,3.1),0.12,12,fill=False,color='g')
ax.add_artist(circ)
circ=ptchs.Ellipse((193.43-5/24,2.3),0.1,10,fill=False,color='g')
ax.add_artist(circ)
circ=ptchs.Ellipse((194-5/24,1.25),0.1,6,fill=False,color='g')
ax.add_artist(circ)
circ=ptchs.Ellipse((194.43-5/24,1.5),0.1,6,fill=False,color='g')
ax.add_artist(circ)
circ=ptchs.Ellipse((195.02-5/24,1.8),0.17,8.5,fill=False,color='g')
ax.add_artist(circ)
plt.legend(fontsize=14,loc='upper left')
plt.xlabel('Day of the Year',fontsize=14)
plt.ylabel('GPP ($\mu$mol m$^{-2}$ s$^{-1}$)',fontsize=14)
plt.title('GPP at Borden Forest, 2018',fontsize=14)
# *** Uncomment to save figure as png and pdf CHANGE FILENAME ***
#plt.savefig('Diurnal_UrbanVPRM_SMUrF_vs_Borden_fixed_GPP.png',bbox_inches='tight')
#plt.savefig('Diurnal_UrbanVPRM_SMUrF_vs_Borden_fixed_GPP.pdf',bbox_inches='tight')
plt.show()

# *** End of optional


# In[ ]:





# ### Now look at Autumn (SON)

# In[50]:


#SON: Doy 224 - 334 inclusive

SON_time=VPRM_Borden_2018_avg_DoY[(np.round(VPRM_Borden_2018_avg_DoY,5)>=224) & (np.round(VPRM_Borden_2018_avg_DoY,5)<335)]
Borden_GPPgf_SON=Borden_GEPgf[(np.round(VPRM_Borden_2018_avg_DoY,5)>=224) & (np.round(VPRM_Borden_2018_avg_DoY,5)<335)]
VPRM_Borden_GPP_SON=VPRM_Borden_2018_avg_GPP[(np.round(VPRM_Borden_2018_avg_DoY,5)>=224) & (np.round(VPRM_Borden_2018_avg_DoY,5)<335)]
Updated_VPRM_Borden_GPP_SON=Updated_VPRM_Borden_2018_avg_GPP[(np.round(VPRM_Borden_2018_avg_DoY,5)>=224) & (np.round(VPRM_Borden_2018_avg_DoY,5)<335)]

Borden_Rgf_SON=Borden_Rgf[(np.round(VPRM_Borden_2018_avg_DoY,5)>=224) & (np.round(VPRM_Borden_2018_avg_DoY,5)<335)]
VPRM_Borden_Reco_SON=VPRM_Borden_2018_avg_Reco[(np.round(VPRM_Borden_2018_avg_DoY,5)>=224) & (np.round(VPRM_Borden_2018_avg_DoY,5)<335)]
Updated_VPRM_Borden_Reco_SON=Updated_VPRM_Borden_2018_avg_Reco[(np.round(VPRM_Borden_2018_avg_DoY,5)>=224) & (np.round(VPRM_Borden_2018_avg_DoY,5)<335)]

Borden_NEEgf_SON=Borden_NEEgf[(np.round(VPRM_Borden_2018_avg_DoY,5)>=224) & (np.round(VPRM_Borden_2018_avg_DoY,5)<335)]
Borden_NEE_SON=Borden_NEE[(np.round(VPRM_Borden_2018_avg_DoY,5)>=224) & (np.round(VPRM_Borden_2018_avg_DoY,5)<335)]
VPRM_Borden_NEE_SON=VPRM_Borden_2018_avg_NEE[(np.round(VPRM_Borden_2018_avg_DoY,5)>=224) & (np.round(VPRM_Borden_2018_avg_DoY,5)<335)]
Updated_VPRM_Borden_NEE_SON=Updated_VPRM_Borden_2018_avg_NEE[(np.round(VPRM_Borden_2018_avg_DoY,5)>=224) & (np.round(VPRM_Borden_2018_avg_DoY,5)<335)]


# In[ ]:





# In[51]:


#SON: Doy 224 - 223 inclusive

SON_time=VPRM_TP39_2018_avg_DoY[(np.round(VPRM_TP39_2018_avg_DoY,5)>=224) & (np.round(VPRM_TP39_2018_avg_DoY,5)<335)]
TP39_GPPgf_SON=TP39_GPP[(np.round(VPRM_TP39_2018_avg_DoY,5)>=224) & (np.round(VPRM_TP39_2018_avg_DoY,5)<335)]
VPRM_TP39_GPP_SON=VPRM_TP39_2018_avg_GPP[(np.round(VPRM_TP39_2018_avg_DoY,5)>=224) & (np.round(VPRM_TP39_2018_avg_DoY,5)<335)]
Updated_VPRM_TP39_GPP_SON=Updated_VPRM_TP39_2018_avg_GPP[(np.round(VPRM_TP39_2018_avg_DoY,5)>=224) & (np.round(VPRM_TP39_2018_avg_DoY,5)<335)]

TP39_Rgf_SON=TP39_R[(np.round(VPRM_TP39_2018_avg_DoY,5)>=224) & (np.round(VPRM_TP39_2018_avg_DoY,5)<335)]
VPRM_TP39_Reco_SON=VPRM_TP39_2018_avg_Reco[(np.round(VPRM_TP39_2018_avg_DoY,5)>=224) & (np.round(VPRM_TP39_2018_avg_DoY,5)<335)]
Updated_VPRM_TP39_Reco_SON=Updated_VPRM_TP39_2018_avg_Reco[(np.round(VPRM_TP39_2018_avg_DoY,5)>=224) & (np.round(VPRM_TP39_2018_avg_DoY,5)<335)]

TP39_NEEgf_SON=TP39_NEEgf[(np.round(VPRM_TP39_2018_avg_DoY,5)>=224) & (np.round(VPRM_TP39_2018_avg_DoY,5)<335)]
TP39_NEE_SON=TP39_NEE[(np.round(VPRM_TP39_2018_avg_DoY,5)>=224) & (np.round(VPRM_TP39_2018_avg_DoY,5)<335)]
VPRM_TP39_NEE_SON=VPRM_TP39_2018_avg_NEE[(np.round(VPRM_TP39_2018_avg_DoY,5)>=224) & (np.round(VPRM_TP39_2018_avg_DoY,5)<335)]
Updated_VPRM_TP39_NEE_SON=Updated_VPRM_TP39_2018_avg_NEE[(np.round(VPRM_TP39_2018_avg_DoY,5)>=224) & (np.round(VPRM_TP39_2018_avg_DoY,5)<335)]


# In[52]:


#SON: Doy 224 - 223 inclusive

SON_time=VPRM_TP39_2019_avg_DoY[(np.round(VPRM_TP39_2019_avg_DoY,5)>=224) & (np.round(VPRM_TP39_2019_avg_DoY,5)<335)]
TP39_2019_GPPgf_SON=TP39_2019_GPP[(np.round(VPRM_TP39_2019_avg_DoY,5)>=224) & (np.round(VPRM_TP39_2019_avg_DoY,5)<335)]
VPRM_TP39_2019_GPP_SON=VPRM_TP39_2019_avg_GPP[(np.round(VPRM_TP39_2019_avg_DoY,5)>=224) & (np.round(VPRM_TP39_2019_avg_DoY,5)<335)]
Updated_VPRM_TP39_2019_GPP_SON=Updated_VPRM_TP39_2019_avg_GPP[(np.round(VPRM_TP39_2019_avg_DoY,5)>=224) & (np.round(VPRM_TP39_2019_avg_DoY,5)<335)]

TP39_2019_Rgf_SON=TP39_2019_R[(np.round(VPRM_TP39_2019_avg_DoY,5)>=224) & (np.round(VPRM_TP39_2019_avg_DoY,5)<335)]
VPRM_TP39_2019_Reco_SON=VPRM_TP39_2019_avg_Reco[(np.round(VPRM_TP39_2019_avg_DoY,5)>=224) & (np.round(VPRM_TP39_2019_avg_DoY,5)<335)]
Updated_VPRM_TP39_2019_Reco_SON=Updated_VPRM_TP39_2019_avg_Reco[(np.round(VPRM_TP39_2019_avg_DoY,5)>=224) & (np.round(VPRM_TP39_2019_avg_DoY,5)<335)]

TP39_2019_NEEgf_SON=TP39_2019_NEEgf[(np.round(VPRM_TP39_2019_avg_DoY,5)>=224) & (np.round(VPRM_TP39_2019_avg_DoY,5)<335)]
TP39_2019_NEE_SON=TP39_2019_NEE[(np.round(VPRM_TP39_2019_avg_DoY,5)>=224) & (np.round(VPRM_TP39_2019_avg_DoY,5)<335)]
VPRM_TP39_2019_NEE_SON=VPRM_TP39_2019_avg_NEE[(np.round(VPRM_TP39_2019_avg_DoY,5)>=224) & (np.round(VPRM_TP39_2019_avg_DoY,5)<335)]
Updated_VPRM_TP39_2019_NEE_SON=Updated_VPRM_TP39_2019_avg_NEE[(np.round(VPRM_TP39_2019_avg_DoY,5)>=224) & (np.round(VPRM_TP39_2019_avg_DoY,5)<335)]


# In[53]:


#SON: Doy 224 - 223 inclusive

SON_time=VPRM_TPD_avg_DoY[(np.round(VPRM_TPD_avg_DoY,5)>=224) & (np.round(VPRM_TPD_avg_DoY,5)<335)]
TPD_GPPgf_SON=TPD_GPP[(np.round(VPRM_TPD_avg_DoY,5)>=224) & (np.round(VPRM_TPD_avg_DoY,5)<335)]
VPRM_TPD_GPP_SON=VPRM_TPD_avg_GPP[(np.round(VPRM_TPD_avg_DoY,5)>=224) & (np.round(VPRM_TPD_avg_DoY,5)<335)]
Updated_VPRM_TPD_GPP_SON=Updated_VPRM_TPD_avg_GPP[(np.round(VPRM_TPD_avg_DoY,5)>=224) & (np.round(VPRM_TPD_avg_DoY,5)<335)]

TPD_Rgf_SON=TPD_R[(np.round(VPRM_TPD_avg_DoY,5)>=224) & (np.round(VPRM_TPD_avg_DoY,5)<335)]
VPRM_TPD_Reco_SON=VPRM_TPD_avg_Reco[(np.round(VPRM_TPD_avg_DoY,5)>=224) & (np.round(VPRM_TPD_avg_DoY,5)<335)]
Updated_VPRM_TPD_Reco_SON=Updated_VPRM_TPD_avg_Reco[(np.round(VPRM_TPD_avg_DoY,5)>=224) & (np.round(VPRM_TPD_avg_DoY,5)<335)]

TPD_NEEgf_SON=TPD_NEEgf[(np.round(VPRM_TPD_avg_DoY,5)>=224) & (np.round(VPRM_TPD_avg_DoY,5)<335)]
TPD_NEE_SON=TPD_NEE[(np.round(VPRM_TPD_avg_DoY,5)>=224) & (np.round(VPRM_TPD_avg_DoY,5)<335)]
VPRM_TPD_NEE_SON=VPRM_TPD_avg_NEE[(np.round(VPRM_TPD_avg_DoY,5)>=224) & (np.round(VPRM_TPD_avg_DoY,5)<335)]
Updated_VPRM_TPD_NEE_SON=Updated_VPRM_TPD_avg_NEE[(np.round(VPRM_TPD_avg_DoY,5)>=224) & (np.round(VPRM_TPD_avg_DoY,5)<335)]


# In[54]:


#SON: Doy 224 - 223 inclusive

SON_time=VPRM_TPD_2019_avg_DoY[(np.round(VPRM_TPD_2019_avg_DoY,5)>=224) & (np.round(VPRM_TPD_2019_avg_DoY,5)<335)]
TPD_2019_GPPgf_SON=TPD_2019_GPP[(np.round(VPRM_TPD_2019_avg_DoY,5)>=224) & (np.round(VPRM_TPD_2019_avg_DoY,5)<335)]
VPRM_TPD_2019_GPP_SON=VPRM_TPD_2019_avg_GPP[(np.round(VPRM_TPD_2019_avg_DoY,5)>=224) & (np.round(VPRM_TPD_2019_avg_DoY,5)<335)]
Updated_VPRM_TPD_2019_GPP_SON=Updated_VPRM_TPD_2019_avg_GPP[(np.round(VPRM_TPD_2019_avg_DoY,5)>=224) & (np.round(VPRM_TPD_2019_avg_DoY,5)<335)]

TPD_2019_Rgf_SON=TPD_2019_R[(np.round(VPRM_TPD_2019_avg_DoY,5)>=224) & (np.round(VPRM_TPD_2019_avg_DoY,5)<335)]
VPRM_TPD_2019_Reco_SON=VPRM_TPD_2019_avg_Reco[(np.round(VPRM_TPD_2019_avg_DoY,5)>=224) & (np.round(VPRM_TPD_2019_avg_DoY,5)<335)]
Updated_VPRM_TPD_2019_Reco_SON=Updated_VPRM_TPD_2019_avg_Reco[(np.round(VPRM_TPD_2019_avg_DoY,5)>=224) & (np.round(VPRM_TPD_2019_avg_DoY,5)<335)]

TPD_2019_NEEgf_SON=TPD_2019_NEEgf[(np.round(VPRM_TPD_2019_avg_DoY,5)>=224) & (np.round(VPRM_TPD_2019_avg_DoY,5)<335)]
TPD_2019_NEE_SON=TPD_2019_NEE[(np.round(VPRM_TPD_2019_avg_DoY,5)>=224) & (np.round(VPRM_TPD_2019_avg_DoY,5)<335)]
VPRM_TPD_2019_NEE_SON=VPRM_TPD_2019_avg_NEE[(np.round(VPRM_TPD_2019_avg_DoY,5)>=224) & (np.round(VPRM_TPD_2019_avg_DoY,5)<335)]
Updated_VPRM_TPD_2019_NEE_SON=Updated_VPRM_TPD_2019_avg_NEE[(np.round(VPRM_TPD_2019_avg_DoY,5)>=224) & (np.round(VPRM_TPD_2019_avg_DoY,5)<335)]


# In[55]:


#Combine all fall data
SON_VPRM_GPP=np.concatenate([VPRM_Borden_GPP_SON,VPRM_TP39_GPP_SON,VPRM_TPD_GPP_SON,VPRM_TP39_2019_GPP_SON,VPRM_TPD_2019_GPP_SON])
SON_Updated_VPRM_GPP=np.concatenate([Updated_VPRM_Borden_GPP_SON,Updated_VPRM_TP39_GPP_SON,Updated_VPRM_TPD_GPP_SON,Updated_VPRM_TP39_2019_GPP_SON,Updated_VPRM_TPD_2019_GPP_SON])
SON_GPP=np.concatenate([Borden_GPPgf_SON,TP39_GPPgf_SON,TPD_GPPgf_SON,TP39_2019_GPPgf_SON,TPD_2019_GPPgf_SON])

SON_VPRM_Reco=np.concatenate([VPRM_Borden_Reco_SON,VPRM_TP39_Reco_SON,VPRM_TPD_Reco_SON,VPRM_TP39_2019_Reco_SON,VPRM_TPD_2019_Reco_SON])
SON_Updated_VPRM_Reco=np.concatenate([Updated_VPRM_Borden_Reco_SON,Updated_VPRM_TP39_Reco_SON,Updated_VPRM_TPD_Reco_SON,Updated_VPRM_TP39_2019_Reco_SON,Updated_VPRM_TPD_2019_Reco_SON])
SON_Reco=np.concatenate([Borden_Rgf_SON,TP39_Rgf_SON,TPD_Rgf_SON,TP39_2019_Rgf_SON,TPD_2019_Rgf_SON])

SON_VPRM_NEE=np.concatenate([VPRM_Borden_NEE_SON,VPRM_TP39_NEE_SON,VPRM_TPD_NEE_SON,VPRM_TP39_2019_NEE_SON,VPRM_TPD_2019_NEE_SON])
SON_Updated_VPRM_NEE=np.concatenate([Updated_VPRM_Borden_NEE_SON,Updated_VPRM_TP39_NEE_SON,Updated_VPRM_TPD_NEE_SON,Updated_VPRM_TP39_2019_NEE_SON,Updated_VPRM_TPD_2019_NEE_SON])
SON_NEEgf=np.concatenate([Borden_NEEgf_SON,TP39_NEEgf_SON,TPD_NEEgf_SON,TP39_2019_NEEgf_SON,TPD_2019_NEEgf_SON])
SON_NEE=np.concatenate([Borden_NEE_SON,TP39_NEE_SON,TPD_NEE_SON,TP39_2019_NEE_SON,TPD_2019_NEE_SON])


# In[ ]:





# In[56]:


#Fit original UrbanVPRM NEE to non-gapfilled flux tower NEE for autumn using a bootstrapped Huber fit

finitemask0=np.isfinite(SON_NEE)
SON_NEEclean0=SON_NEE[finitemask0]
SON_VPRM_NEEclean0=SON_VPRM_NEE[finitemask0]

finitemask1=np.isfinite(SON_VPRM_NEEclean0)
SON_NEEclean1=SON_NEEclean0[finitemask1]
SON_VPRM_NEEclean1=SON_VPRM_NEEclean0[finitemask1]

Huber_SON_NEE_slps=[]
Huber_SON_NEE_ints=[]
Huber_SON_NEE_R2=[]

#try bootstrapping
indx_list=list(range(0,len(SON_VPRM_NEEclean1)))
for i in range(1,1000):
    #sub selection of points
    NEE_indx=np.random.choice(indx_list,size=len(SON_VPRM_NEEclean1))
    
    try:
        Huber_model = linear_model.HuberRegressor(fit_intercept=True)
        Huber_fit=Huber_model.fit((SON_NEEclean1[NEE_indx]).reshape(-1,1),SON_VPRM_NEEclean1[NEE_indx])
        H_m=Huber_fit.coef_
        H_c=Huber_fit.intercept_
        x_accpt, y_accpt = SON_NEEclean1, SON_VPRM_NEEclean1
        y_predict = H_m * x_accpt + H_c
        H_R2=r2_score(y_accpt, y_predict)
        Huber_SON_NEE_slps.append(H_m)
        Huber_SON_NEE_ints.append(H_c)
        Huber_SON_NEE_R2.append(H_R2)
    except ValueError: #if Huber fit can't find a solution for the subset, skip it
        pass

Huber_SON_R2 = r2_score(SON_VPRM_NEEclean1, SON_NEEclean1*np.nanmean(Huber_SON_NEE_slps)+np.nanmean(Huber_SON_NEE_ints))

print('Original UrbanVPRM SON slope: '+str(np.round(np.nanmean(Huber_SON_NEE_slps),3))+' +/- '+str(np.round(np.nanstd(Huber_SON_NEE_slps),3)))
print('Original UrbanVPRM SON intercept: '+str(np.round(np.nanmean(Huber_SON_NEE_ints),3))+' +/- '+str(np.round(np.nanstd(Huber_SON_NEE_ints),3)))

print('Original UrbanVPRM SON R^2: '+str(np.round(np.nanmean(Huber_SON_R2),3)))


# In[ ]:





# In[57]:


#Fit updated UrbanVPRM NEE to non-gapfilled flux tower NEE for autumn using a bootstrapped Huber fit

finitemask0=np.isfinite(SON_NEE)
SON_NEEclean0=SON_NEE[finitemask0]
SON_Updated_VPRM_NEEclean0=SON_Updated_VPRM_NEE[finitemask0]

finitemask1=np.isfinite(SON_Updated_VPRM_NEEclean0)
SON_NEEclean1=SON_NEEclean0[finitemask1]
SON_Updated_VPRM_NEEclean1=SON_Updated_VPRM_NEEclean0[finitemask1]

Huber_SON_Updated_NEE_slps=[]
Huber_SON_Updated_NEE_ints=[]
Huber_SON_Updated_NEE_R2=[]

#try bootstrapping
indx_list=list(range(0,len(SON_Updated_VPRM_NEEclean1)))
for i in range(1,1000):
    #sub selection of points
    NEE_indx=np.random.choice(indx_list,size=len(SON_Updated_VPRM_NEEclean1))
    
    try:
        Huber_model = linear_model.HuberRegressor(fit_intercept=True)
        Huber_fit=Huber_model.fit((SON_NEEclean1[NEE_indx]).reshape(-1,1),SON_Updated_VPRM_NEEclean1[NEE_indx])
        H_m=Huber_fit.coef_
        H_c=Huber_fit.intercept_
        x_accpt, y_accpt = SON_NEEclean1, SON_Updated_VPRM_NEEclean1
        y_predict = H_m * x_accpt + H_c
        H_R2=r2_score(y_accpt, y_predict)
        Huber_SON_Updated_NEE_slps.append(H_m)
        Huber_SON_Updated_NEE_ints.append(H_c)
        Huber_SON_Updated_NEE_R2.append(H_R2)
    except ValueError: #if Huber fit can't find a solution for the subset, skip it
        pass

Huber_SON_Updated_R2 = r2_score(SON_Updated_VPRM_NEEclean1, SON_NEEclean1*np.nanmean(Huber_SON_Updated_NEE_slps)+np.nanmean(Huber_SON_Updated_NEE_ints))

print('Updated UrbanVPRM SON slope: '+str(np.round(np.nanmean(Huber_SON_Updated_NEE_slps),3))+' +/- '+str(np.round(np.nanstd(Huber_SON_Updated_NEE_slps),3)))
print('Updated UrbanVPRM SON intercept: '+str(np.round(np.nanmean(Huber_SON_Updated_NEE_ints),3))+' +/- '+str(np.round(np.nanstd(Huber_SON_Updated_NEE_ints),3)))

print('Updated UrbanVPRM SON R^2: '+str(np.round(np.nanmean(Huber_SON_Updated_R2),3)))


# In[ ]:





# ### Now look at Winter (DJF - all of same year)

# In[58]:


#DJF: Doy 0-59 & 335-365 inclusive

DJF_time=VPRM_Borden_2018_avg_DoY[(np.round(VPRM_Borden_2018_avg_DoY,5)>=335) | (np.round(VPRM_Borden_2018_avg_DoY,5)<60)]
Borden_GPPgf_DJF=Borden_GEPgf[(np.round(VPRM_Borden_2018_avg_DoY,5)>=335) | (np.round(VPRM_Borden_2018_avg_DoY,5)<60)]
VPRM_Borden_GPP_DJF=VPRM_Borden_2018_avg_GPP[(np.round(VPRM_Borden_2018_avg_DoY,5)>=335) | (np.round(VPRM_Borden_2018_avg_DoY,5)<60)]
Updated_VPRM_Borden_GPP_DJF=Updated_VPRM_Borden_2018_avg_GPP[(np.round(VPRM_Borden_2018_avg_DoY,5)>=335) | (np.round(VPRM_Borden_2018_avg_DoY,5)<60)]

Borden_Rgf_DJF=Borden_Rgf[(np.round(VPRM_Borden_2018_avg_DoY,5)>=335) | (np.round(VPRM_Borden_2018_avg_DoY,5)<60)]
VPRM_Borden_Reco_DJF=VPRM_Borden_2018_avg_Reco[(np.round(VPRM_Borden_2018_avg_DoY,5)>=335) | (np.round(VPRM_Borden_2018_avg_DoY,5)<60)]
Updated_VPRM_Borden_Reco_DJF=Updated_VPRM_Borden_2018_avg_Reco[(np.round(VPRM_Borden_2018_avg_DoY,5)>=335) | (np.round(VPRM_Borden_2018_avg_DoY,5)<60)]

Borden_NEEgf_DJF=Borden_NEEgf[(np.round(VPRM_Borden_2018_avg_DoY,5)>=335) | (np.round(VPRM_Borden_2018_avg_DoY,5)<60)]
Borden_NEE_DJF=Borden_NEE[(np.round(VPRM_Borden_2018_avg_DoY,5)>=335) | (np.round(VPRM_Borden_2018_avg_DoY,5)<60)]
VPRM_Borden_NEE_DJF=VPRM_Borden_2018_avg_NEE[(np.round(VPRM_Borden_2018_avg_DoY,5)>=335) | (np.round(VPRM_Borden_2018_avg_DoY,5)<60)]
Updated_VPRM_Borden_NEE_DJF=Updated_VPRM_Borden_2018_avg_NEE[(np.round(VPRM_Borden_2018_avg_DoY,5)>=335) | (np.round(VPRM_Borden_2018_avg_DoY,5)<60)]


# In[59]:


#DJF: Doy 224 - 223 inclusive

DJF_time=VPRM_TP39_2018_avg_DoY[(np.round(VPRM_TP39_2018_avg_DoY,5)>=335) | (np.round(VPRM_TP39_2018_avg_DoY,5)<60)]
TP39_GPPgf_DJF=TP39_GPP[(np.round(VPRM_TP39_2018_avg_DoY,5)>=335) | (np.round(VPRM_TP39_2018_avg_DoY,5)<60)]
VPRM_TP39_GPP_DJF=VPRM_TP39_2018_avg_GPP[(np.round(VPRM_TP39_2018_avg_DoY,5)>=335) | (np.round(VPRM_TP39_2018_avg_DoY,5)<60)]
Updated_VPRM_TP39_GPP_DJF=Updated_VPRM_TP39_2018_avg_GPP[(np.round(VPRM_TP39_2018_avg_DoY,5)>=335) | (np.round(VPRM_TP39_2018_avg_DoY,5)<60)]

TP39_Rgf_DJF=TP39_R[(np.round(VPRM_TP39_2018_avg_DoY,5)>=335) | (np.round(VPRM_TP39_2018_avg_DoY,5)<60)]
VPRM_TP39_Reco_DJF=VPRM_TP39_2018_avg_Reco[(np.round(VPRM_TP39_2018_avg_DoY,5)>=335) | (np.round(VPRM_TP39_2018_avg_DoY,5)<60)]
Updated_VPRM_TP39_Reco_DJF=Updated_VPRM_TP39_2018_avg_Reco[(np.round(VPRM_TP39_2018_avg_DoY,5)>=335) | (np.round(VPRM_TP39_2018_avg_DoY,5)<60)]

TP39_NEEgf_DJF=TP39_NEEgf[(np.round(VPRM_TP39_2018_avg_DoY,5)>=335) | (np.round(VPRM_TP39_2018_avg_DoY,5)<60)]
TP39_NEE_DJF=TP39_NEE[(np.round(VPRM_TP39_2018_avg_DoY,5)>=335) | (np.round(VPRM_TP39_2018_avg_DoY,5)<60)]
VPRM_TP39_NEE_DJF=VPRM_TP39_2018_avg_NEE[(np.round(VPRM_TP39_2018_avg_DoY,5)>=335) | (np.round(VPRM_TP39_2018_avg_DoY,5)<60)]
Updated_VPRM_TP39_NEE_DJF=Updated_VPRM_TP39_2018_avg_NEE[(np.round(VPRM_TP39_2018_avg_DoY,5)>=335) | (np.round(VPRM_TP39_2018_avg_DoY,5)<60)]


# In[60]:


#DJF: Doy 224 - 223 inclusive

DJF_time=VPRM_TP39_2019_avg_DoY[(np.round(VPRM_TP39_2019_avg_DoY,5)>=335) | (np.round(VPRM_TP39_2019_avg_DoY,5)<60)]
TP39_2019_GPPgf_DJF=TP39_2019_GPP[(np.round(VPRM_TP39_2019_avg_DoY,5)>=335) | (np.round(VPRM_TP39_2019_avg_DoY,5)<60)]
VPRM_TP39_2019_GPP_DJF=VPRM_TP39_2019_avg_GPP[(np.round(VPRM_TP39_2019_avg_DoY,5)>=335) | (np.round(VPRM_TP39_2019_avg_DoY,5)<60)]
Updated_VPRM_TP39_2019_GPP_DJF=Updated_VPRM_TP39_2019_avg_GPP[(np.round(VPRM_TP39_2019_avg_DoY,5)>=335) | (np.round(VPRM_TP39_2019_avg_DoY,5)<60)]

TP39_2019_Rgf_DJF=TP39_2019_R[(np.round(VPRM_TP39_2019_avg_DoY,5)>=335) | (np.round(VPRM_TP39_2019_avg_DoY,5)<60)]
VPRM_TP39_2019_Reco_DJF=VPRM_TP39_2019_avg_Reco[(np.round(VPRM_TP39_2019_avg_DoY,5)>=335) | (np.round(VPRM_TP39_2019_avg_DoY,5)<60)]
Updated_VPRM_TP39_2019_Reco_DJF=Updated_VPRM_TP39_2019_avg_Reco[(np.round(VPRM_TP39_2019_avg_DoY,5)>=335) | (np.round(VPRM_TP39_2019_avg_DoY,5)<60)]

TP39_2019_NEEgf_DJF=TP39_2019_NEEgf[(np.round(VPRM_TP39_2019_avg_DoY,5)>=335) | (np.round(VPRM_TP39_2019_avg_DoY,5)<60)]
TP39_2019_NEE_DJF=TP39_2019_NEE[(np.round(VPRM_TP39_2019_avg_DoY,5)>=335) | (np.round(VPRM_TP39_2019_avg_DoY,5)<60)]
VPRM_TP39_2019_NEE_DJF=VPRM_TP39_2019_avg_NEE[(np.round(VPRM_TP39_2019_avg_DoY,5)>=335) | (np.round(VPRM_TP39_2019_avg_DoY,5)<60)]
Updated_VPRM_TP39_2019_NEE_DJF=Updated_VPRM_TP39_2019_avg_NEE[(np.round(VPRM_TP39_2019_avg_DoY,5)>=335) | (np.round(VPRM_TP39_2019_avg_DoY,5)<60)]


# In[61]:


#DJF: Doy 224 - 223 inclusive

DJF_time=VPRM_TPD_avg_DoY[(np.round(VPRM_TPD_avg_DoY,5)>=335) | (np.round(VPRM_TPD_avg_DoY,5)<60)]
TPD_GPPgf_DJF=TPD_GPP[(np.round(VPRM_TPD_avg_DoY,5)>=335) | (np.round(VPRM_TPD_avg_DoY,5)<60)]
VPRM_TPD_GPP_DJF=VPRM_TPD_avg_GPP[(np.round(VPRM_TPD_avg_DoY,5)>=335) | (np.round(VPRM_TPD_avg_DoY,5)<60)]
Updated_VPRM_TPD_GPP_DJF=Updated_VPRM_TPD_avg_GPP[(np.round(VPRM_TPD_avg_DoY,5)>=335) | (np.round(VPRM_TPD_avg_DoY,5)<60)]

TPD_Rgf_DJF=TPD_R[(np.round(VPRM_TPD_avg_DoY,5)>=335) | (np.round(VPRM_TPD_avg_DoY,5)<60)]
VPRM_TPD_Reco_DJF=VPRM_TPD_avg_Reco[(np.round(VPRM_TPD_avg_DoY,5)>=335) | (np.round(VPRM_TPD_avg_DoY,5)<60)]
Updated_VPRM_TPD_Reco_DJF=Updated_VPRM_TPD_avg_Reco[(np.round(VPRM_TPD_avg_DoY,5)>=335) | (np.round(VPRM_TPD_avg_DoY,5)<60)]

TPD_NEEgf_DJF=TPD_NEEgf[(np.round(VPRM_TPD_avg_DoY,5)>=335) | (np.round(VPRM_TPD_avg_DoY,5)<60)]
TPD_NEE_DJF=TPD_NEE[(np.round(VPRM_TPD_avg_DoY,5)>=335) | (np.round(VPRM_TPD_avg_DoY,5)<60)]
VPRM_TPD_NEE_DJF=VPRM_TPD_avg_NEE[(np.round(VPRM_TPD_avg_DoY,5)>=335) | (np.round(VPRM_TPD_avg_DoY,5)<60)]
Updated_VPRM_TPD_NEE_DJF=Updated_VPRM_TPD_avg_NEE[(np.round(VPRM_TPD_avg_DoY,5)>=335) | (np.round(VPRM_TPD_avg_DoY,5)<60)]


# In[62]:


#DJF: Doy 224 - 223 inclusive

DJF_time=VPRM_TPD_2019_avg_DoY[(np.round(VPRM_TPD_2019_avg_DoY,5)>=335) | (np.round(VPRM_TPD_2019_avg_DoY,5)<60)]
TPD_2019_GPPgf_DJF=TPD_2019_GPP[(np.round(VPRM_TPD_2019_avg_DoY,5)>=335) | (np.round(VPRM_TPD_2019_avg_DoY,5)<60)]
VPRM_TPD_2019_GPP_DJF=VPRM_TPD_2019_avg_GPP[(np.round(VPRM_TPD_2019_avg_DoY,5)>=335) | (np.round(VPRM_TPD_2019_avg_DoY,5)<60)]
Updated_VPRM_TPD_2019_GPP_DJF=Updated_VPRM_TPD_2019_avg_GPP[(np.round(VPRM_TPD_2019_avg_DoY,5)>=335) | (np.round(VPRM_TPD_2019_avg_DoY,5)<60)]

TPD_2019_Rgf_DJF=TPD_2019_R[(np.round(VPRM_TPD_2019_avg_DoY,5)>=335) | (np.round(VPRM_TPD_2019_avg_DoY,5)<60)]
VPRM_TPD_2019_Reco_DJF=VPRM_TPD_2019_avg_Reco[(np.round(VPRM_TPD_2019_avg_DoY,5)>=335) | (np.round(VPRM_TPD_2019_avg_DoY,5)<60)]
Updated_VPRM_TPD_2019_Reco_DJF=Updated_VPRM_TPD_2019_avg_Reco[(np.round(VPRM_TPD_2019_avg_DoY,5)>=335) | (np.round(VPRM_TPD_2019_avg_DoY,5)<60)]

TPD_2019_NEEgf_DJF=TPD_2019_NEEgf[(np.round(VPRM_TPD_2019_avg_DoY,5)>=335) | (np.round(VPRM_TPD_2019_avg_DoY,5)<60)]
TPD_2019_NEE_DJF=TPD_2019_NEE[(np.round(VPRM_TPD_2019_avg_DoY,5)>=335) | (np.round(VPRM_TPD_2019_avg_DoY,5)<60)]
VPRM_TPD_2019_NEE_DJF=VPRM_TPD_2019_avg_NEE[(np.round(VPRM_TPD_2019_avg_DoY,5)>=335) | (np.round(VPRM_TPD_2019_avg_DoY,5)<60)]
Updated_VPRM_TPD_2019_NEE_DJF=Updated_VPRM_TPD_2019_avg_NEE[(np.round(VPRM_TPD_2019_avg_DoY,5)>=335) | (np.round(VPRM_TPD_2019_avg_DoY,5)<60)]


# In[63]:


#Combine all winter data
DJF_VPRM_GPP=np.concatenate([VPRM_Borden_GPP_DJF,VPRM_TP39_GPP_DJF,VPRM_TPD_GPP_DJF,VPRM_TP39_2019_GPP_DJF,VPRM_TPD_2019_GPP_DJF])
DJF_Updated_VPRM_GPP=np.concatenate([Updated_VPRM_Borden_GPP_DJF,Updated_VPRM_TP39_GPP_DJF,Updated_VPRM_TPD_GPP_DJF,Updated_VPRM_TP39_2019_GPP_DJF,Updated_VPRM_TPD_2019_GPP_DJF])
DJF_GPP=np.concatenate([Borden_GPPgf_DJF,TP39_GPPgf_DJF,TPD_GPPgf_DJF,TP39_2019_GPPgf_DJF,TPD_2019_GPPgf_DJF])

DJF_VPRM_Reco=np.concatenate([VPRM_Borden_Reco_DJF,VPRM_TP39_Reco_DJF,VPRM_TPD_Reco_DJF,VPRM_TP39_2019_Reco_DJF,VPRM_TPD_2019_Reco_DJF])
DJF_Updated_VPRM_Reco=np.concatenate([Updated_VPRM_Borden_Reco_DJF,Updated_VPRM_TP39_Reco_DJF,Updated_VPRM_TPD_Reco_DJF,Updated_VPRM_TP39_2019_Reco_DJF,Updated_VPRM_TPD_2019_Reco_DJF])
DJF_Reco=np.concatenate([Borden_Rgf_DJF,TP39_Rgf_DJF,TPD_Rgf_DJF,TP39_2019_Rgf_DJF,TPD_2019_Rgf_DJF])

DJF_VPRM_NEE=np.concatenate([VPRM_Borden_NEE_DJF,VPRM_TP39_NEE_DJF,VPRM_TPD_NEE_DJF,VPRM_TP39_2019_NEE_DJF,VPRM_TPD_2019_NEE_DJF])
DJF_Updated_VPRM_NEE=np.concatenate([Updated_VPRM_Borden_NEE_DJF,Updated_VPRM_TP39_NEE_DJF,Updated_VPRM_TPD_NEE_DJF,Updated_VPRM_TP39_2019_NEE_DJF,Updated_VPRM_TPD_2019_NEE_DJF])
DJF_NEEgf=np.concatenate([Borden_NEEgf_DJF,TP39_NEEgf_DJF,TPD_NEEgf_DJF,TP39_2019_NEEgf_DJF,TPD_2019_NEEgf_DJF])
DJF_NEE=np.concatenate([Borden_NEE_DJF,TP39_NEE_DJF,TPD_NEE_DJF,TP39_2019_NEE_DJF,TPD_2019_NEE_DJF])


# In[ ]:





# In[64]:


# Optional: Fit original UrbanVPRM NEE to non-gapfilled flux tower NEE for winter using a bootstrapped Huber fit

finitemask0=np.isfinite(DJF_NEE)
DJF_NEEclean0=DJF_NEE[finitemask0]
DJF_VPRM_NEEclean0=DJF_VPRM_NEE[finitemask0]

finitemask1=np.isfinite(DJF_VPRM_NEEclean0)
DJF_NEEclean1=DJF_NEEclean0[finitemask1]
DJF_VPRM_NEEclean1=DJF_VPRM_NEEclean0[finitemask1]

Huber_DJF_NEE_slps=[]
Huber_DJF_NEE_ints=[]
Huber_DJF_NEE_R2=[]

#try bootstrapping
indx_list=list(range(0,len(DJF_VPRM_NEEclean1)))
for i in range(1,1000):
    #sub selection of points
    NEE_indx=np.random.choice(indx_list,size=len(DJF_VPRM_NEEclean1))
    
    try:
        Huber_model = linear_model.HuberRegressor(fit_intercept=True)
        Huber_fit=Huber_model.fit((DJF_NEEclean1[NEE_indx]).reshape(-1,1),DJF_VPRM_NEEclean1[NEE_indx])
        H_m=Huber_fit.coef_
        H_c=Huber_fit.intercept_
        x_accpt, y_accpt = DJF_NEEclean1, DJF_VPRM_NEEclean1
        y_predict = H_m * x_accpt + H_c
        H_R2=r2_score(y_accpt, y_predict)
        Huber_DJF_NEE_slps.append(H_m)
        Huber_DJF_NEE_ints.append(H_c)
        Huber_DJF_NEE_R2.append(H_R2)
    except ValueError: #if Huber fit can't find a solution for the subset, skip it
        pass

Huber_DJF_R2 = r2_score(DJF_VPRM_NEEclean1, DJF_NEEclean1*np.nanmean(Huber_DJF_NEE_slps)+np.nanmean(Huber_DJF_NEE_ints))

print('Original UrbanVPRM DJF slope: '+str(np.round(np.nanmean(Huber_DJF_NEE_slps),3))+' +/- '+str(np.round(np.nanstd(Huber_DJF_NEE_slps),3)))
print('Original UrbanVPRM DJF intercept: '+str(np.round(np.nanmean(Huber_DJF_NEE_ints),3))+' +/- '+str(np.round(np.nanstd(Huber_DJF_NEE_ints),3)))

print('Original UrbanVPRM DJF R^2: '+str(np.round(np.nanmean(Huber_DJF_R2),3)))


# In[65]:


# Optional: Fit updated UrbanVPRM NEE to non-gapfilled flux tower NEE for winter using a bootstrapped Huber fit

finitemask0=np.isfinite(DJF_NEE)
DJF_NEEclean0=DJF_NEE[finitemask0]
DJF_Updated_VPRM_NEEclean0=DJF_Updated_VPRM_NEE[finitemask0]

finitemask1=np.isfinite(DJF_Updated_VPRM_NEEclean0)
DJF_NEEclean1=DJF_NEEclean0[finitemask1]
DJF_Updated_VPRM_NEEclean1=DJF_Updated_VPRM_NEEclean0[finitemask1]

Huber_DJF_Updated_NEE_slps=[]
Huber_DJF_Updated_NEE_ints=[]
Huber_DJF_Updated_NEE_R2=[]

#try bootstrapping
indx_list=list(range(0,len(DJF_Updated_VPRM_NEEclean1)))
for i in range(1,1000):
    #sub selection of points
    NEE_indx=np.random.choice(indx_list,size=len(DJF_Updated_VPRM_NEEclean1))
    
    try:
        Huber_model = linear_model.HuberRegressor(fit_intercept=True)
        Huber_fit=Huber_model.fit((DJF_NEEclean1[NEE_indx]).reshape(-1,1),DJF_Updated_VPRM_NEEclean1[NEE_indx])
        H_m=Huber_fit.coef_
        H_c=Huber_fit.intercept_
        x_accpt, y_accpt = DJF_NEEclean1, DJF_Updated_VPRM_NEEclean1
        y_predict = H_m * x_accpt + H_c
        H_R2=r2_score(y_accpt, y_predict)
        Huber_DJF_Updated_NEE_slps.append(H_m)
        Huber_DJF_Updated_NEE_ints.append(H_c)
        Huber_DJF_Updated_NEE_R2.append(H_R2)
    except ValueError: #if Huber fit can't find a solution for the subset, skip it
        pass

Huber_DJF_Updated_R2 = r2_score(DJF_Updated_VPRM_NEEclean1, DJF_NEEclean1*np.nanmean(Huber_DJF_Updated_NEE_slps)+np.nanmean(Huber_DJF_Updated_NEE_ints))

print('Updated UrbanVPRM DJF slope: '+str(np.round(np.nanmean(Huber_DJF_Updated_NEE_slps),3))+' +/- '+str(np.round(np.nanstd(Huber_DJF_Updated_NEE_slps),3)))
print('Updated UrbanVPRM DJF intercept: '+str(np.round(np.nanmean(Huber_DJF_Updated_NEE_ints),3))+' +/- '+str(np.round(np.nanstd(Huber_DJF_Updated_NEE_ints),3)))

print('Updated UrbanVPRM DJF R^2: '+str(np.round(np.nanmean(Huber_DJF_Updated_R2),3)))


# In[ ]:





# In[ ]:





# In[75]:



plt.style.use('tableau-colorblind10')

plt.rc('font',size=21.5)
fig, ax = plt.subplots(1,4,sharex=True,sharey=True,figsize=(24,6))
ax[0].set_xlim(-59,25)
ax[0].set_ylim(-59,25)

ax[0].scatter(MAM_NEE,MAM_VPRM_NEE,s=5)
ax[0].scatter(MAM_NEE,MAM_Updated_VPRM_NEE,s=5)
ax[0].plot(line1_1,line1_1*np.nanmean(Huber_MAM_NEE_slps)+np.nanmean(Huber_MAM_NEE_ints),linestyle='--',linewidth=2,label=str(np.round(np.nanmean(Huber_MAM_NEE_slps),2))+'$\cdot$x+'+str(np.round(np.nanmean(Huber_MAM_NEE_ints),2))+', R$^2$ = '+str(np.round(Huber_MAM_R2,2)),c='k', path_effects=[pe.Stroke(linewidth=5, foreground='#006BA4'), pe.Normal()])
ax[0].plot(line1_1,line1_1*np.nanmean(Huber_MAM_Updated_NEE_slps)+np.nanmean(Huber_MAM_Updated_NEE_ints),linestyle='-.',linewidth=2,label=str(np.round(np.nanmean(Huber_MAM_Updated_NEE_slps),2))+'$\cdot$x+'+str(np.round(np.nanmean(Huber_MAM_Updated_NEE_ints),2))+', R$^2$ = '+str(np.round(Huber_MAM_Updated_R2,2)),c='k', path_effects=[pe.Stroke(linewidth=5, foreground='#FF800E'), pe.Normal()])

ax[0].axvline(0,linestyle=(0, (3, 2, 1, 2, 1, 2)),c='k')
ax[0].axhline(0,linestyle=(0, (3, 2, 1, 2, 1, 2)),c='k')
ax[0].plot(line1_1,line1_1,linestyle=':',c='k')
ax[0].legend(loc='lower center')
ax[0].set_title('Spring')

ax[1].scatter(JJA_NEE,JJA_VPRM_NEE,s=5)
ax[1].scatter(JJA_NEE,JJA_Updated_VPRM_NEE,s=5)
ax[1].plot(line1_1,line1_1*np.nanmean(Huber_JJA_NEE_slps)+np.nanmean(Huber_JJA_NEE_ints),linestyle='--',linewidth=2,label=str(np.round(np.nanmean(Huber_JJA_NEE_slps),2))+'$\cdot$x - '+str(np.round(abs(np.nanmean(Huber_JJA_NEE_ints)),2))+', R$^2$ = '+str(np.round(Huber_JJA_R2,2)),c='k', path_effects=[pe.Stroke(linewidth=5, foreground='#006BA4'), pe.Normal()])
ax[1].plot(line1_1,line1_1*np.nanmean(Huber_JJA_Updated_NEE_slps)+np.nanmean(Huber_JJA_Updated_NEE_ints),linestyle='-.',linewidth=2,label=str(np.round(np.nanmean(Huber_JJA_Updated_NEE_slps),2))+'$\cdot$x+'+str(np.round(np.nanmean(Huber_JJA_Updated_NEE_ints),2))+', R$^2$ = '+str(np.round(Huber_JJA_Updated_R2,2)),c='k', path_effects=[pe.Stroke(linewidth=5, foreground='#FF800E'), pe.Normal()])

ax[1].axvline(0,linestyle=(0, (3, 2, 1, 2, 1, 2)),c='k')
ax[1].axhline(0,linestyle=(0, (3, 2, 1, 2, 1, 2)),c='k')
ax[1].plot(line1_1,line1_1,linestyle=':',c='k')
ax[1].legend(loc='lower center')
ax[1].set_title('Summer')

ax[2].scatter(SON_NEE,SON_VPRM_NEE,s=5)
ax[2].scatter(SON_NEE,SON_Updated_VPRM_NEE,s=5)
ax[2].plot(line1_1,line1_1*np.nanmean(Huber_SON_NEE_slps)+np.nanmean(Huber_SON_NEE_ints),linestyle='--',linewidth=2,label=str(np.round(np.nanmean(Huber_SON_NEE_slps),2))+'$\cdot$x+'+str(np.round(np.nanmean(Huber_SON_NEE_ints),2))+', R$^2$ = '+str(np.round(Huber_SON_R2,2)),c='k', path_effects=[pe.Stroke(linewidth=5, foreground='#006BA4'), pe.Normal()])
ax[2].plot(line1_1,line1_1*np.nanmean(Huber_SON_Updated_NEE_slps)+np.nanmean(Huber_SON_Updated_NEE_ints),linestyle='-.',linewidth=2,label=str(np.round(np.nanmean(Huber_SON_Updated_NEE_slps),2))+'$\cdot$x+'+str(np.round(np.nanmean(Huber_SON_Updated_NEE_ints),2))+', R$^2$ = '+str(np.round(Huber_SON_Updated_R2,2)),c='k', path_effects=[pe.Stroke(linewidth=5, foreground='#FF800E'), pe.Normal()])

ax[2].axvline(0,linestyle=(0, (3, 2, 1, 2, 1, 2)),c='k')
ax[2].axhline(0,linestyle=(0, (3, 2, 1, 2, 1, 2)),c='k')
ax[2].plot(line1_1,line1_1,linestyle=':',c='k')
ax[2].legend(loc='lower center')
ax[2].set_title('Autumn')

ax[3].scatter(-100,-100,label='Original UrbanVPRM')
ax[3].scatter(-100,-100,label='Updated UrbanVPRM')
ax[3].scatter(DJF_NEE,DJF_VPRM_NEE,c='#006BA4',s=5)
ax[3].scatter(DJF_NEE,DJF_Updated_VPRM_NEE,c='#FF800E',s=5)

ax[3].axvline(0,linestyle=(0, (3, 2, 1, 2, 1, 2)),c='k')
ax[3].axhline(0,linestyle=(0, (3, 2, 1, 2, 1, 2)),c='k')
ax[3].plot(line1_1,line1_1,linestyle=':',c='k')
ax[3].legend(loc='lower right')
ax[3].set_title('Winter')

ax[0].set_ylabel('Modelled NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')

ax[0].text(-57.5,18,'(a)',c='k',fontsize=26)
ax[1].text(-57.5,18,'(b)',c='k',fontsize=26)
ax[2].text(-57.5,18,'(c)',c='k',fontsize=26)
ax[3].text(-57.5,18,'(d)',c='k',fontsize=26)

fig.text(0.5, 0.01, 'Flux Tower NEE ($\mu$mol m$^{-2}$ s$^{-1}$)', ha='center')
fig.subplots_adjust(hspace=0,wspace=0)
# *** Uncomment below lines to save figure as pdf (next line) & as png (second line) ***
plt.savefig('Seasonal_Original_Updated_VPRM_vs_fixed_fluxtower_Huber_fit_NEE_0_lines_larger_font_cb_friendly_labelled.pdf',bbox_inches='tight')
plt.savefig('Seasonal_Original_Updated_VPRM_vs_fixed_fluxtower_Huber_fit_NEE_0_lines_larger_font_cb_friendly_labelled.png',bbox_inches='tight')
fig.show()


# In[ ]:





# In[ ]:




