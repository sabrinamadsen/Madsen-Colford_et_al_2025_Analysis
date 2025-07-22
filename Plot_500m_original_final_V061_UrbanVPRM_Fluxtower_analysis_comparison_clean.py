#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Code used to plot & fit fluxes from the original and updated UrbanVPRM to 3 eddy-covariance flux towers in 2018 & 2019

# Generates figure 2 a & b of Madsen-Colford et al. 2025
# *** Denotes sections of the code that should be changed by the user.


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from scipy import optimize as opt 
from scipy import odr
from datetime import datetime, timedelta
from sklearn import linear_model #for robust fitting
from sklearn.metrics import r2_score, mean_squared_error #for analyzing robust fits
import matplotlib.colors as clrs #for log color scale
import matplotlib.patheffects as pe


# In[2]:


#Load in the original & updated UrbanVPRM 2018 fluxes over Borden Forest flux tower

#*** CHANGE PATHS & FILENAMES ***
VPRM_data=pd.read_csv('Borden_500m_V061_no_adjustments_2018/vprm_mixed_ISA_Borden_500m_V061_2018_no_adjustments.csv')
Updated_VPRM_data=pd.read_csv('Borden_V061_500m_2018/vprm_GMIS_Toronto_ACI_SOLRIS_ISA_500m_Borden_V061_2018_no_PScale_adjusted_Topt_Ra_URB_parameters_fixed_gapfilled_LSWI_filtered.csv')


# In[3]:


#Select the pixels that fall within the Borden Forest footprint & omit data to the NW

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
for i in range(8760*105,8760*106): # *** NOTE: if extent is changed in UrbanVPRM code these indices will need to be changed ***
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
for i in range(8760*120,8760*122): # *** NOTE: if extent is changed in UrbanVPRM code these indices will need to be changed ***
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
for i in range(8760*135,8760*138): # *** NOTE: if extent is changed in UrbanVPRM code these indices will need to be changed ***
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


# In[4]:


#For original VPRM only (original VPRM gives -GEE in output file, multiply by -1)
VPRM_GEE0 = -VPRM_GEE0
#Compute NEE
VPRM_NEE0=VPRM_Reco0+VPRM_GEE0
Updated_VPRM_NEE0=Updated_VPRM_Reco0+Updated_VPRM_GEE0


# In[5]:


#Average pixels inside Borden footprint

VPRM_Borden_2018_avg_DoY=np.mean(VPRM_HoY0, axis=1)/24+23/24
VPRM_Borden_2018_avg_Index=np.mean(VPRM_Index0, axis=1)
VPRM_Borden_2018_avg_GPP=-np.mean(VPRM_GEE0, axis=1)
VPRM_Borden_2018_avg_Reco=np.mean(VPRM_Reco0, axis=1)
VPRM_Borden_2018_avg_NEE=np.mean(VPRM_NEE0, axis=1)

Updated_VPRM_Borden_2018_avg_DoY=np.mean(Updated_VPRM_HoY0, axis=1)/24+23/24
Updated_VPRM_Borden_2018_avg_Index=np.mean(Updated_VPRM_Index0, axis=1)
Updated_VPRM_Borden_2018_avg_GPP=-np.mean(Updated_VPRM_GEE0, axis=1)
Updated_VPRM_Borden_2018_avg_Reco=np.mean(Updated_VPRM_Reco0, axis=1)
Updated_VPRM_Borden_2018_avg_NEE=np.mean(Updated_VPRM_NEE0, axis=1)


# In[6]:


#Load Borden Fluxtower values

#*** CHANGE PATH & FILENAME ***
Borden_Fluxes=pd.read_csv('/Users/kitty/Documents/Research/SIF/Flux_Tower/2018_NEP_GPP_Borden.csv', index_col=0)

Borden_dates=np.zeros([17520])*np.nan
Borden_NEEgf_fluxes=np.zeros([17520])*np.nan
Borden_NEE_fluxes=np.zeros([17520])*np.nan
Borden_Rgf_fluxes=np.zeros([17520])*np.nan
Borden_GEPgf_fluxes=np.zeros([17520])*np.nan
for i in range(0,17520):
    Borden_dates[i]=Borden_Fluxes.iat[i,0]#This is in UTC time
    Borden_NEEgf_fluxes[i]=-Borden_Fluxes.iat[i,5] #NEE (gap filled)
    Borden_NEE_fluxes[i]=-Borden_Fluxes.iat[i,1]
    Borden_Rgf_fluxes[i]=Borden_Fluxes.iat[i,6]
    Borden_GEPgf_fluxes[i]=Borden_Fluxes.iat[i,7]


# In[7]:


#Take hourly average of Borden fluxtower data
Borden_GEPgf=np.zeros(np.shape(VPRM_Borden_2018_avg_GPP))*np.nan
Borden_NEEgf=np.zeros(np.shape(VPRM_Borden_2018_avg_GPP))*np.nan
Borden_Rgf=np.zeros(np.shape(VPRM_Borden_2018_avg_GPP))*np.nan
Borden_NEE=np.zeros(np.shape(VPRM_Borden_2018_avg_GPP))*np.nan

for i in range(np.int(len(Borden_dates)/2)):
    with np.errstate(invalid='ignore'):
        Borden_GEPgf[i]=np.nanmean([Borden_GEPgf_fluxes[i*2],Borden_GEPgf_fluxes[i*2+1]])
        Borden_NEEgf[i]=np.nanmean([Borden_NEEgf_fluxes[i*2],Borden_NEEgf_fluxes[i*2+1]])
        Borden_Rgf[i]=np.nanmean([Borden_Rgf_fluxes[i*2],Borden_Rgf_fluxes[i*2+1]])
        Borden_NEE[i]=np.nanmean([Borden_NEE_fluxes[i*2],Borden_NEE_fluxes[i*2+1]])


# In[ ]:





# In[8]:


#Define a linear function and a straight line for plotting
def func2(x,m,b):
    return m*x+b

line1_1=np.arange(-100,100)


# In[ ]:





# In[9]:


#Take a daily average of VPRM fluxes over Borden Forest flux tower
date_array_2018=np.arange(np.nanmin(VPRM_Borden_2018_avg_DoY),366,1/24)

VPRM_daily_NEE_Borden_2018=np.zeros(365)*np.nan
VPRM_daily_GPP_Borden_2018=np.zeros(365)*np.nan
VPRM_daily_R_Borden_2018=np.zeros(365)*np.nan

Updated_VPRM_daily_NEE_Borden_2018=np.zeros(365)*np.nan
Updated_VPRM_daily_GPP_Borden_2018=np.zeros(365)*np.nan
Updated_VPRM_daily_R_Borden_2018=np.zeros(365)*np.nan
date=0

daily_NEE_VPRM=[]
daily_GPP_VPRM=[]
daily_R_VPRM=[]

daily_NEE_Updated_VPRM=[]
daily_GPP_Updated_VPRM=[]
daily_R_Updated_VPRM=[]
for i in range(len(date_array_2018)):
    if np.round(date_array_2018[i],4)>=1:
        if date+1>=365:
            daily_NEE_VPRM.append(VPRM_Borden_2018_avg_NEE[i])
            daily_GPP_VPRM.append(VPRM_Borden_2018_avg_GPP[i])
            daily_R_VPRM.append(VPRM_Borden_2018_avg_Reco[i])
            
            daily_NEE_Updated_VPRM.append(Updated_VPRM_Borden_2018_avg_NEE[i])
            daily_GPP_Updated_VPRM.append(Updated_VPRM_Borden_2018_avg_GPP[i])
            daily_R_Updated_VPRM.append(Updated_VPRM_Borden_2018_avg_Reco[i])
            if i==len(date_array_2018)-1:
                VPRM_daily_NEE_Borden_2018[date]=np.mean(daily_NEE_VPRM)
                VPRM_daily_GPP_Borden_2018[date]=np.mean(daily_GPP_VPRM)
                VPRM_daily_R_Borden_2018[date]=np.mean(daily_R_VPRM)
                
                Updated_VPRM_daily_NEE_Borden_2018[date]=np.mean(daily_NEE_Updated_VPRM)
                Updated_VPRM_daily_GPP_Borden_2018[date]=np.mean(daily_GPP_Updated_VPRM)
                Updated_VPRM_daily_R_Borden_2018[date]=np.mean(daily_R_Updated_VPRM)
                date+=1

        else:
            daily_NEE_VPRM.append(VPRM_Borden_2018_avg_NEE[i])
            daily_GPP_VPRM.append(VPRM_Borden_2018_avg_GPP[i])
            daily_R_VPRM.append(VPRM_Borden_2018_avg_Reco[i])
            
            daily_NEE_Updated_VPRM.append(Updated_VPRM_Borden_2018_avg_NEE[i])
            daily_GPP_Updated_VPRM.append(Updated_VPRM_Borden_2018_avg_GPP[i])
            daily_R_Updated_VPRM.append(Updated_VPRM_Borden_2018_avg_Reco[i])
            if np.floor(np.round(date_array_2018[i],4))<np.floor(np.round(date_array_2018[i+1],4)):
                VPRM_daily_NEE_Borden_2018[date]=np.mean(daily_NEE_VPRM)
                VPRM_daily_GPP_Borden_2018[date]=np.mean(daily_GPP_VPRM)
                VPRM_daily_R_Borden_2018[date]=np.mean(daily_R_VPRM)
                
                Updated_VPRM_daily_NEE_Borden_2018[date]=np.mean(daily_NEE_Updated_VPRM)
                Updated_VPRM_daily_GPP_Borden_2018[date]=np.mean(daily_GPP_Updated_VPRM)
                Updated_VPRM_daily_R_Borden_2018[date]=np.mean(daily_R_Updated_VPRM)
                
                date+=1
                daily_NEE_VPRM=[]
                daily_GPP_VPRM=[]
                daily_R_VPRM=[]
                
                daily_NEE_Updated_VPRM=[]
                daily_GPP_Updated_VPRM=[]
                daily_R_Updated_VPRM=[]


# In[10]:


#Take the daily average of Borden forest flux tower data
days_of_year=np.arange(1,366)+0.5

Borden_daily_mean_NEE=np.zeros(365)*np.nan
Borden_daily_mean_NEEgf=np.zeros(365)*np.nan
Borden_daily_mean_GPPgf=np.zeros(365)*np.nan
Borden_daily_mean_Rgf=np.zeros(365)*np.nan

for i in range(len(days_of_year)):
    daily_NEE_Borden_2018=Borden_NEE[np.floor(np.round(VPRM_Borden_2018_avg_DoY,4))==i+1]
    daily_NEEgf_Borden_2018=Borden_NEEgf[np.floor(np.round(VPRM_Borden_2018_avg_DoY,4))==i+1]
    daily_GPPgf_Borden_2018=Borden_GEPgf[np.floor(np.round(VPRM_Borden_2018_avg_DoY,4))==i+1]
    daily_Rgf_Borden_2018=Borden_Rgf[np.floor(np.round(VPRM_Borden_2018_avg_DoY,4))==i+1]
    if len(daily_NEE_Borden_2018)==24:
        Borden_daily_mean_NEE[i]=np.mean(daily_NEE_Borden_2018)
        Borden_daily_mean_NEEgf[i]=np.mean(daily_NEEgf_Borden_2018)
        Borden_daily_mean_GPPgf[i]=np.mean(daily_GPPgf_Borden_2018)
        Borden_daily_mean_Rgf[i]=np.mean(daily_Rgf_Borden_2018)


# In[ ]:





# In[11]:


#Load in UrbanVPRM data over TP39

#*** CHANGE PATHS & FILENAMES ***
VPRM_data=pd.read_csv('TP39_500m_V061_no_adjustments_2018/vprm_mixed_ISA_TP39_500m_V061_2018_no_adjustments.csv')
Updated_VPRM_data=pd.read_csv('TP39_V061_500m_2018/vprm_GMIS_Toronto_ACI_SOLRIS_ISA_500m_TP39_V061_2018_no_PScale_adjusted_Topt_Ra_URB_parameters_fixed_gapfilled_LSWI_filtered.csv')


# In[12]:


#Select data in the footprint of the TP39 tower
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
for i in range(8760*119,8760*121): # *** NOTE: if extent is changed in UrbanVPRM code these indices will need to be changed ***
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
for i in range(8760*135,8760*137): # *** NOTE: if extent is changed in UrbanVPRM code these indices will need to be changed ***
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

#Compute NEE
VPRM_GEE0=-VPRM_GEE0 #ONLY FOR ORIGINAL VPRM
VPRM_NEE0=VPRM_Reco0+VPRM_GEE0
Updated_VPRM_NEE0=Updated_VPRM_Reco0+Updated_VPRM_GEE0

#Take the average over all the pixels
VPRM_TP39_2018_avg_DoY=np.mean(VPRM_HoY0, axis=1)/24+23/24 #This makes it so that the HoY=1 is midnight on January 1st i.e. DoY= 1.00
VPRM_TP39_2018_avg_Index=np.mean(VPRM_Index0, axis=1)
VPRM_TP39_2018_avg_GPP=-np.mean(VPRM_GEE0, axis=1)
VPRM_TP39_2018_avg_Reco=np.mean(VPRM_Reco0, axis=1)
VPRM_TP39_2018_avg_NEE=np.mean(VPRM_NEE0, axis=1)

Updated_VPRM_TP39_2018_avg_DoY=np.mean(Updated_VPRM_HoY0, axis=1)/24+23/24 #This makes it so that the HoY=1 is midnight on January 1st i.e. DoY= 1.00
Updated_VPRM_TP39_2018_avg_Index=np.mean(Updated_VPRM_Index0, axis=1)
Updated_VPRM_TP39_2018_avg_GPP=-np.mean(Updated_VPRM_GEE0, axis=1)
Updated_VPRM_TP39_2018_avg_Reco=np.mean(Updated_VPRM_Reco0, axis=1)
Updated_VPRM_TP39_2018_avg_NEE=np.mean(Updated_VPRM_NEE0, axis=1)


# In[53]:





# In[13]:


#Import TP39 2018 flux tower data

#*** CHANGE PATH & FILENAME ***
TP39_Fluxes=pd.read_csv('/Users/kitty/Documents/Research/SIF/Flux_Tower/Turkey_Point/TP39_HH_2018.csv', usecols=[0,1,2,77,78,79])

TP39_dates=np.zeros([17520])*np.nan
TP39_NEEgf_fluxes=np.zeros([17520])*np.nan
TP39_NEE_fluxes=np.zeros([17520])*np.nan
TP39_Rgf_fluxes=np.zeros([17520])*np.nan
TP39_GPPgf_fluxes=np.zeros([17520])*np.nan
for i in range(0,17520):
    if 201801010000<=TP39_Fluxes.iat[i,0]<201901010000:
        #Convert to UTC time by adding 5/24 to date
        TP39_dates[i]= datetime.strptime(str(int(TP39_Fluxes.iat[i,0])),'%Y%m%d%H%M').timetuple().tm_yday+(float(str(TP39_Fluxes.iat[i,0])[8:10])+float(str(TP39_Fluxes.iat[i,0])[10:12])/60)/24+5/24
        TP39_NEEgf_fluxes[i]=TP39_Fluxes.iat[i,5] #NEE (gap filled)
        if TP39_Fluxes.iat[i,2]>-9999:
            TP39_NEE_fluxes[i]=TP39_Fluxes.iat[i,2]
        TP39_Rgf_fluxes[i]=TP39_Fluxes.iat[i,4]
        TP39_GPPgf_fluxes[i]=TP39_Fluxes.iat[i,3]


# In[14]:


#Take hourly average
TP39_GPP=np.zeros(np.shape(VPRM_TP39_2018_avg_GPP))*np.nan
TP39_NEE=np.zeros(np.shape(VPRM_TP39_2018_avg_GPP))*np.nan
TP39_NEEgf=np.zeros(np.shape(VPRM_TP39_2018_avg_GPP))*np.nan
TP39_R=np.zeros(np.shape(VPRM_TP39_2018_avg_GPP))*np.nan

for i in range(np.int(len(TP39_dates)/2)):
    with np.errstate(invalid='ignore'):
        if i<8755:
            TP39_GPP[i+5]=np.nanmean([TP39_GPPgf_fluxes[i*2],TP39_GPPgf_fluxes[i*2+1]])
            TP39_NEE[i+5]=np.nanmean([TP39_NEE_fluxes[i*2],TP39_NEE_fluxes[i*2+1]])
            TP39_NEEgf[i+5]=np.nanmean([TP39_NEEgf_fluxes[i*2],TP39_NEEgf_fluxes[i*2+1]])
            TP39_R[i+5]=np.nanmean([TP39_Rgf_fluxes[i*2],TP39_Rgf_fluxes[i*2+1]])


# In[ ]:





# In[15]:


#Take daily average of 2018 VPRM data at TP39

VPRM_TP39_2018_daily_NEE=np.zeros(365)*np.nan
VPRM_TP39_2018_daily_GPP=np.zeros(365)*np.nan
VPRM_TP39_2018_daily_R=np.zeros(365)*np.nan

Updated_VPRM_TP39_2018_daily_NEE=np.zeros(365)*np.nan
Updated_VPRM_TP39_2018_daily_GPP=np.zeros(365)*np.nan
Updated_VPRM_TP39_2018_daily_R=np.zeros(365)*np.nan

date=0
daily_NEE_VPRM=[]
daily_GPP_VPRM=[]
daily_R_VPRM=[]
daily_NEE_Updated_VPRM=[]
daily_GPP_Updated_VPRM=[]
daily_R_Updated_VPRM=[]
for i in range(len(VPRM_TP39_2018_avg_DoY)):
    if VPRM_TP39_2018_avg_DoY[i]>=1:
        if date+1>=365:
            daily_NEE_VPRM.append(VPRM_TP39_2018_avg_NEE[i])
            daily_GPP_VPRM.append(VPRM_TP39_2018_avg_GPP[i])
            daily_R_VPRM.append(VPRM_TP39_2018_avg_Reco[i])
            
            daily_NEE_Updated_VPRM.append(Updated_VPRM_TP39_2018_avg_NEE[i])
            daily_GPP_Updated_VPRM.append(Updated_VPRM_TP39_2018_avg_GPP[i])
            daily_R_Updated_VPRM.append(Updated_VPRM_TP39_2018_avg_Reco[i])
            if i==len(VPRM_TP39_2018_avg_DoY)-1:
                VPRM_TP39_2018_daily_NEE[date]=np.mean(daily_NEE_VPRM)
                VPRM_TP39_2018_daily_GPP[date]=np.mean(daily_GPP_VPRM)
                VPRM_TP39_2018_daily_R[date]=np.mean(daily_R_VPRM)
                 
                Updated_VPRM_TP39_2018_daily_NEE[date]=np.mean(daily_NEE_Updated_VPRM)
                Updated_VPRM_TP39_2018_daily_GPP[date]=np.mean(daily_GPP_Updated_VPRM)
                Updated_VPRM_TP39_2018_daily_R[date]=np.mean(daily_R_Updated_VPRM)
                
                date+=1
        else:
            daily_NEE_VPRM.append(VPRM_TP39_2018_avg_NEE[i])
            daily_GPP_VPRM.append(VPRM_TP39_2018_avg_GPP[i])
            daily_R_VPRM.append(VPRM_TP39_2018_avg_Reco[i])
            
            daily_NEE_Updated_VPRM.append(Updated_VPRM_TP39_2018_avg_NEE[i])
            daily_GPP_Updated_VPRM.append(Updated_VPRM_TP39_2018_avg_GPP[i])
            daily_R_Updated_VPRM.append(Updated_VPRM_TP39_2018_avg_Reco[i])
            if np.floor(np.round(VPRM_TP39_2018_avg_DoY[i],4))<np.floor(np.round(VPRM_TP39_2018_avg_DoY[i+1],4)):
                VPRM_TP39_2018_daily_NEE[date]=np.mean(daily_NEE_VPRM)
                VPRM_TP39_2018_daily_GPP[date]=np.mean(daily_GPP_VPRM)
                VPRM_TP39_2018_daily_R[date]=np.mean(daily_R_VPRM)
                
                Updated_VPRM_TP39_2018_daily_NEE[date]=np.mean(daily_NEE_Updated_VPRM)
                Updated_VPRM_TP39_2018_daily_GPP[date]=np.mean(daily_GPP_Updated_VPRM)
                Updated_VPRM_TP39_2018_daily_R[date]=np.mean(daily_R_Updated_VPRM)
                date+=1
                
                daily_NEE_VPRM=[]
                daily_GPP_VPRM=[]
                daily_R_VPRM=[]
                
                daily_NEE_Updated_VPRM=[]
                daily_GPP_Updated_VPRM=[]
                daily_R_Updated_VPRM=[]


# In[16]:


#Take daily average of 2018 TP39 flux tower data

TP39_daily_mean_NEE=np.zeros(365)*np.nan
TP39_daily_mean_GPP=np.zeros(365)*np.nan
TP39_daily_mean_R=np.zeros(365)*np.nan
TP39_daily_mean_NEEgf=np.zeros(365)*np.nan

for i in range(len(days_of_year)):
    daily_NEEgf_TP39_2018=TP39_NEEgf[np.floor(np.round(VPRM_TP39_2018_avg_DoY,4))==i+1]
    daily_NEE_TP39_2018=TP39_NEE[np.floor(np.round(VPRM_TP39_2018_avg_DoY,4))==i+1]
    daily_GPP_TP39_2018=TP39_GPP[np.floor(np.round(VPRM_TP39_2018_avg_DoY,4))==i+1]
    daily_R_TP39_2018=TP39_R[np.floor(np.round(VPRM_TP39_2018_avg_DoY,4))==i+1]
    if len(daily_NEE_TP39_2018)==24:
        TP39_daily_mean_NEEgf[i]=np.mean(daily_NEEgf_TP39_2018)
        TP39_daily_mean_NEE[i]=np.mean(daily_NEE_TP39_2018)
        TP39_daily_mean_GPP[i]=np.mean(daily_GPP_TP39_2018)
        TP39_daily_mean_R[i]=np.mean(daily_R_TP39_2018)


# In[ ]:





# In[17]:


# Load in UrbanVPRM 2019 fluxes over TP39 
# *** CHANGE PATHS & FILENAMES ***
VPRM_data=pd.read_csv('TP39_500m_V061_no_adjustments_2019/vprm_mixed_ISA_TP39_500m_V061_2019_no_adjustments.csv')
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
for i in range(8760*119,8760*121): # *** NOTE: if extent is changed in UrbanVPRM code these indices will need to be changed ***
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
for i in range(8760*135,8760*137): # *** NOTE: if extent is changed in UrbanVPRM code these indices will need to be changed ***
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
VPRM_NEE0=VPRM_Reco0+VPRM_GEE0
Updated_VPRM_NEE0=Updated_VPRM_Reco0+Updated_VPRM_GEE0


# In[18]:


#Average pixels that fall within TP39's footprint

#Original UrbanVPRM
VPRM_TP39_2019_avg_DoY=np.mean(VPRM_HoY0, axis=1)/24+23/24
VPRM_TP39_2019_avg_Index=np.mean(VPRM_Index0, axis=1)
VPRM_TP39_2019_avg_GPP=-np.mean(VPRM_GEE0, axis=1)
VPRM_TP39_2019_avg_Reco=np.mean(VPRM_Reco0, axis=1)
VPRM_TP39_2019_avg_NEE=np.mean(VPRM_NEE0, axis=1)

#Updated UrbanVPRM
Updated_VPRM_TP39_2019_avg_DoY=np.mean(Updated_VPRM_HoY0, axis=1)/24+23/24
Updated_VPRM_TP39_2019_avg_Index=np.mean(Updated_VPRM_Index0, axis=1)
Updated_VPRM_TP39_2019_avg_GPP=-np.mean(Updated_VPRM_GEE0, axis=1)
Updated_VPRM_TP39_2019_avg_Reco=np.mean(Updated_VPRM_Reco0, axis=1)
Updated_VPRM_TP39_2019_avg_NEE=np.mean(Updated_VPRM_NEE0, axis=1)


# In[19]:


# Load in TP39 2019 data (in local time)

# *** CHANGE PATH & FILENAME ***
TP39_2019_Fluxes=pd.read_csv('/Users/kitty/Documents/Research/SIF/Flux_Tower/Turkey_Point/TP39_HH_2019.csv',usecols=(0,2,77,78,79))

TP39_2019_dates=np.zeros([17520])*np.nan
TP39_2019_NEEgf_fluxes=np.zeros([17520])*np.nan
TP39_2019_NEE_fluxes=np.zeros([17520])*np.nan
TP39_2019_Rgf_fluxes=np.zeros([17520])*np.nan
TP39_2019_GPPgf_fluxes=np.zeros([17520])*np.nan

for i in range(0,17520):
    if 201901010000<=TP39_2019_Fluxes.iat[i,0]<202001010000:
        #TP is 5 hours behind UTC adjust to UTC
        TP39_2019_dates[i]=datetime.strptime(str(int(TP39_2019_Fluxes.iat[i,0])),'%Y%m%d%H%M').timetuple().tm_yday+(float(str(TP39_2019_Fluxes.iat[i,0])[8:10])+float(str(TP39_2019_Fluxes.iat[i,0])[10:12])/60)/24+5/24
        if TP39_2019_Fluxes.iat[i,4]>-9999:
            TP39_2019_NEEgf_fluxes[i]=TP39_2019_Fluxes.iat[i,4] #NEE (gap filled)
        if TP39_2019_Fluxes.iat[i,1]>-9999:
            TP39_2019_NEE_fluxes[i]=TP39_2019_Fluxes.iat[i,1]
        if TP39_2019_Fluxes.iat[i,3]>-9999:
            TP39_2019_Rgf_fluxes[i]=TP39_2019_Fluxes.iat[i,3]
        if TP39_2019_Fluxes.iat[i,2]>-9999:
            TP39_2019_GPPgf_fluxes[i]=TP39_2019_Fluxes.iat[i,2]


# In[20]:


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


# In[ ]:





# In[ ]:





# In[21]:


#Take daily average of UrbanVPRM 2019 data over TP39
VPRM_TP39_2019_daily_NEE=np.zeros(365)*np.nan
VPRM_TP39_2019_daily_GPP=np.zeros(365)*np.nan
VPRM_TP39_2019_daily_R=np.zeros(365)*np.nan

Updated_VPRM_TP39_2019_daily_NEE=np.zeros(365)*np.nan
Updated_VPRM_TP39_2019_daily_GPP=np.zeros(365)*np.nan
Updated_VPRM_TP39_2019_daily_R=np.zeros(365)*np.nan

date=0
daily_NEE_VPRM=[]
daily_GPP_VPRM=[]
daily_R_VPRM=[]
daily_NEE_Updated_VPRM=[]
daily_GPP_Updated_VPRM=[]
daily_R_Updated_VPRM=[]

for i in range(len(VPRM_TP39_2019_avg_DoY)):
    if VPRM_TP39_2019_avg_DoY[i]>=1:
        if date+1>=365:
            daily_NEE_VPRM.append(VPRM_TP39_2019_avg_NEE[i])
            daily_GPP_VPRM.append(VPRM_TP39_2019_avg_GPP[i])
            daily_R_VPRM.append(VPRM_TP39_2019_avg_Reco[i])
            daily_NEE_Updated_VPRM.append(Updated_VPRM_TP39_2019_avg_NEE[i])
            daily_GPP_Updated_VPRM.append(Updated_VPRM_TP39_2019_avg_GPP[i])
            daily_R_Updated_VPRM.append(Updated_VPRM_TP39_2019_avg_Reco[i])
            if i==len(VPRM_TP39_2019_avg_DoY)-1:
                VPRM_TP39_2019_daily_NEE[date]=np.mean(daily_NEE_VPRM)
                VPRM_TP39_2019_daily_GPP[date]=np.mean(daily_GPP_VPRM)
                VPRM_TP39_2019_daily_R[date]=np.mean(daily_R_VPRM)
                
                Updated_VPRM_TP39_2019_daily_NEE[date]=np.mean(daily_NEE_Updated_VPRM)
                Updated_VPRM_TP39_2019_daily_GPP[date]=np.mean(daily_GPP_Updated_VPRM)
                Updated_VPRM_TP39_2019_daily_R[date]=np.mean(daily_R_Updated_VPRM)
                date+=1
        else:
            daily_NEE_VPRM.append(VPRM_TP39_2019_avg_NEE[i])
            daily_GPP_VPRM.append(VPRM_TP39_2019_avg_GPP[i])
            daily_R_VPRM.append(VPRM_TP39_2019_avg_Reco[i])
        
            daily_NEE_Updated_VPRM.append(Updated_VPRM_TP39_2019_avg_NEE[i])
            daily_GPP_Updated_VPRM.append(Updated_VPRM_TP39_2019_avg_GPP[i])
            daily_R_Updated_VPRM.append(Updated_VPRM_TP39_2019_avg_Reco[i])
            if np.floor(np.round(VPRM_TP39_2019_avg_DoY[i],4))<np.floor(np.round(VPRM_TP39_2019_avg_DoY[i+1],4)):
                VPRM_TP39_2019_daily_NEE[date]=np.mean(daily_NEE_VPRM)
                VPRM_TP39_2019_daily_GPP[date]=np.mean(daily_GPP_VPRM)
                VPRM_TP39_2019_daily_R[date]=np.mean(daily_R_VPRM)
                
                Updated_VPRM_TP39_2019_daily_NEE[date]=np.mean(daily_NEE_Updated_VPRM)
                Updated_VPRM_TP39_2019_daily_GPP[date]=np.mean(daily_GPP_Updated_VPRM)
                Updated_VPRM_TP39_2019_daily_R[date]=np.mean(daily_R_Updated_VPRM)
                
                date+=1
                daily_NEE_VPRM=[]
                daily_GPP_VPRM=[]
                daily_R_VPRM=[]
                
                daily_NEE_Updated_VPRM=[]
                daily_GPP_Updated_VPRM=[]
                daily_R_Updated_VPRM=[]
         
        
#Take daily average of TP39 2019 flux tower data
TP39_2019_daily_mean_NEE=np.zeros(365)*np.nan
TP39_2019_daily_mean_GPP=np.zeros(365)*np.nan
TP39_2019_daily_mean_R=np.zeros(365)*np.nan
TP39_2019_daily_mean_NEEgf=np.zeros(365)*np.nan

for i in range(len(days_of_year)):
    daily_NEEgf_TP39_2019=TP39_2019_NEEgf[np.floor(np.round(VPRM_TP39_2019_avg_DoY,4))==i+1]
    daily_NEE_TP39_2019=TP39_2019_NEE[np.floor(np.round(VPRM_TP39_2019_avg_DoY,4))==i+1]
    daily_GPP_TP39_2019=TP39_2019_GPP[np.floor(np.round(VPRM_TP39_2019_avg_DoY,4))==i+1]
    daily_R_TP39_2019=TP39_2019_R[np.floor(np.round(VPRM_TP39_2019_avg_DoY,4))==i+1]
    if len(daily_NEE_TP39_2019)==24:
        TP39_2019_daily_mean_NEEgf[i]=np.mean(daily_NEEgf_TP39_2019)
        TP39_2019_daily_mean_NEE[i]=np.mean(daily_NEE_TP39_2019)
        TP39_2019_daily_mean_GPP[i]=np.mean(daily_GPP_TP39_2019)
        TP39_2019_daily_mean_R[i]=np.mean(daily_R_TP39_2019)


# In[ ]:





# In[229]:





# In[22]:


#Load in 2018 UrbanVPRM data over TPD

#*** CHANGE PATHS & FILENAMES
VPRM_data=pd.read_csv('TPD_500m_V061_no_adjustments_2018/vprm_mixed_ISA_TPD_500m_V061_2018_no_adjustments.csv')
Updated_VPRM_data=pd.read_csv('TPD_V061_500m_2018/vprm_GMIS_Toronto_ACI_SOLRIS_ISA_500m_TPD_V061_2018_no_PScale_adjusted_Topt_Ra_URB_parameters_fixed_gapfilled_LSWI_filtered.csv')

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
for i in range(8760*103,8760*106): # *** NOTE: If you changed the extent in UrbanVPRM you will need to adjust these indices ***
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
for i in range(8760*119,8760*122): # *** NOTE: If you changed the extent in UrbanVPRM you will need to adjust these indices *** 
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
for i in range(8760*135,8760*138): # *** NOTE: If you changed the extent in UrbanVPRM you will need to adjust these indices ***
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
        
#Take the average of all data that falls inside the TPD flux tower footprint
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


# In[ ]:





# In[23]:


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
            TPD_NEE_fluxes[i]=TPD_Fluxes.iat[i,1] #NEE (non-gapfilled)
        TPD_Rgf_fluxes[i]=TPD_Fluxes.iat[i,3]
        TPD_GPPgf_fluxes[i]=TPD_Fluxes.iat[i,2]


# In[24]:


TPD_GPP=np.zeros(np.shape(VPRM_TPD_avg_GPP))*np.nan
TPD_NEE=np.zeros(np.shape(VPRM_TPD_avg_GPP))*np.nan
TPD_NEEgf=np.zeros(np.shape(VPRM_TPD_avg_GPP))*np.nan
TPD_R=np.zeros(np.shape(VPRM_TPD_avg_GPP))*np.nan
for i in range(np.int(len(TPD_dates)/2)):
    with np.errstate(invalid='ignore'):
        if i<8755:
            TPD_GPP[i+5]=np.nanmean([TPD_GPPgf_fluxes[i*2],TPD_GPPgf_fluxes[i*2+1]])
            TPD_NEE[i+5]=np.nanmean([TPD_NEE_fluxes[i*2],TPD_NEE_fluxes[i*2+1]])
            TPD_NEEgf[i+5]=np.nanmean([TPD_NEEgf_fluxes[i*2],TPD_NEEgf_fluxes[i*2+1]])
            TPD_R[i+5]=np.nanmean([TPD_Rgf_fluxes[i*2],TPD_Rgf_fluxes[i*2+1]])


# In[ ]:





# In[25]:


# Take daily average of 2018 UrbanVPRM fluxes over TPD

VPRM_TPD_daily_NEE=np.zeros(365)*np.nan
VPRM_TPD_daily_GPP=np.zeros(365)*np.nan
VPRM_TPD_daily_R=np.zeros(365)*np.nan

Updated_VPRM_TPD_daily_NEE=np.zeros(365)*np.nan
Updated_VPRM_TPD_daily_GPP=np.zeros(365)*np.nan
Updated_VPRM_TPD_daily_R=np.zeros(365)*np.nan

date=0
daily_NEE_VPRM=[]
daily_GPP_VPRM=[]
daily_R_VPRM=[]

daily_NEE_Updated_VPRM=[]
daily_GPP_Updated_VPRM=[]
daily_R_Updated_VPRM=[]
for i in range(len(VPRM_TPD_avg_DoY)):
    if VPRM_TPD_avg_DoY[i]>=1:
        if date+1>=365:
            daily_NEE_VPRM.append(VPRM_TPD_avg_NEE[i])
            daily_GPP_VPRM.append(VPRM_TPD_avg_GPP[i])
            daily_R_VPRM.append(VPRM_TPD_avg_Reco[i])
            
            daily_NEE_Updated_VPRM.append(Updated_VPRM_TPD_avg_NEE[i])
            daily_GPP_Updated_VPRM.append(Updated_VPRM_TPD_avg_GPP[i])
            daily_R_Updated_VPRM.append(Updated_VPRM_TPD_avg_Reco[i])
            if i==len(VPRM_TPD_avg_DoY)-1:
                VPRM_TPD_daily_NEE[date]=np.nanmean(daily_NEE_VPRM)
                VPRM_TPD_daily_GPP[date]=np.nanmean(daily_GPP_VPRM)
                VPRM_TPD_daily_R[date]=np.nanmean(daily_R_VPRM)
                
                Updated_VPRM_TPD_daily_NEE[date]=np.nanmean(daily_NEE_Updated_VPRM)
                Updated_VPRM_TPD_daily_GPP[date]=np.nanmean(daily_GPP_Updated_VPRM)
                Updated_VPRM_TPD_daily_R[date]=np.nanmean(daily_R_Updated_VPRM)
                
                date+=1
        else:
            daily_NEE_VPRM.append(VPRM_TPD_avg_NEE[i])
            daily_GPP_VPRM.append(VPRM_TPD_avg_GPP[i])
            daily_R_VPRM.append(VPRM_TPD_avg_Reco[i])
            
            daily_NEE_Updated_VPRM.append(Updated_VPRM_TPD_avg_NEE[i])
            daily_GPP_Updated_VPRM.append(Updated_VPRM_TPD_avg_GPP[i])
            daily_R_Updated_VPRM.append(Updated_VPRM_TPD_avg_Reco[i])
            if np.floor(np.round(VPRM_TPD_avg_DoY[i],4))<np.floor(np.round(VPRM_TPD_avg_DoY[i+1],4)):
                VPRM_TPD_daily_NEE[date]=np.nanmean(daily_NEE_VPRM)
                VPRM_TPD_daily_GPP[date]=np.nanmean(daily_GPP_VPRM)
                VPRM_TPD_daily_R[date]=np.nanmean(daily_R_VPRM)
                
                Updated_VPRM_TPD_daily_NEE[date]=np.nanmean(daily_NEE_Updated_VPRM)
                Updated_VPRM_TPD_daily_GPP[date]=np.nanmean(daily_GPP_Updated_VPRM)
                Updated_VPRM_TPD_daily_R[date]=np.nanmean(daily_R_Updated_VPRM)
                
                date+=1
                daily_NEE_VPRM=[]
                daily_GPP_VPRM=[]
                daily_R_VPRM=[]
                
                daily_NEE_Updated_VPRM=[]
                daily_GPP_Updated_VPRM=[]
                daily_R_Updated_VPRM=[]
                
# Take daily average of TPD 2018 flux tower data
TPD_daily_mean_NEE=np.zeros(365)*np.nan
TPD_daily_mean_GPP=np.zeros(365)*np.nan
TPD_daily_mean_R=np.zeros(365)*np.nan
TPD_daily_mean_NEEgf=np.zeros(365)*np.nan

for i in range(len(days_of_year)):
    daily_NEEgf_TPD_2018=TPD_NEEgf[np.floor(np.round(VPRM_TPD_avg_DoY,4))==i+1]
    daily_NEE_TPD_2018=TPD_NEE[np.floor(np.round(VPRM_TPD_avg_DoY,4))==i+1]
    daily_GPP_TPD_2018=TPD_GPP[np.floor(np.round(VPRM_TPD_avg_DoY,4))==i+1]
    daily_R_TPD_2018=TPD_R[np.floor(np.round(VPRM_TPD_avg_DoY,4))==i+1]
    if len(daily_NEE_TPD_2018)==24:
        TPD_daily_mean_NEEgf[i]=np.mean(daily_NEEgf_TPD_2018)
        TPD_daily_mean_NEE[i]=np.mean(daily_NEE_TPD_2018)
        TPD_daily_mean_GPP[i]=np.mean(daily_GPP_TPD_2018)
        TPD_daily_mean_R[i]=np.mean(daily_R_TPD_2018)


# In[255]:





# In[ ]:





# In[26]:


#Load in UrbanVPRM 2019 data over TPD

#*** CHANGE PATHS & FILENAMES ***
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
for i in range(8760*103,8760*106): #*** NOTE: if you changed the extent in UrbanVPRM you will need to change these indices ***
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
for i in range(8760*119,8760*122): #*** NOTE: if you changed the extent in UrbanVPRM you will need to change these indices ***
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
for i in range(8760*135,8760*138): #*** NOTE: if you changed the extent in UrbanVPRM you will need to change these indices ***
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
        
#Average data falling within TPD footprint

#Original UrbanVPRM
VPRM_GEE0=VPRM_GEE0
VPRM_TPD_2019_avg_DoY=np.nanmean(VPRM_HoY0, axis=1)/24+23/24 #convert from hour 1 to DoY=1
VPRM_TPD_2019_avg_Index=np.nanmean(VPRM_Index0, axis=1)
VPRM_TPD_2019_avg_GPP=np.nanmean(VPRM_GEE0, axis=1)
VPRM_TPD_2019_avg_Reco=np.nanmean(VPRM_Reco0, axis=1)
VPRM_TPD_2019_avg_NEE=np.nanmean(VPRM_Reco0-VPRM_GEE0, axis=1)

#Updated UrbanVPRM 
Updated_VPRM_TPD_2019_avg_DoY=np.nanmean(Updated_VPRM_HoY0, axis=1)/24+23/24 #convert from hour 1 to DoY=1
Updated_VPRM_TPD_2019_avg_Index=np.nanmean(Updated_VPRM_Index0, axis=1)
Updated_VPRM_TPD_2019_avg_GPP=-np.nanmean(Updated_VPRM_GEE0, axis=1)
Updated_VPRM_TPD_2019_avg_Reco=np.nanmean(Updated_VPRM_Reco0, axis=1)
Updated_VPRM_TPD_2019_avg_NEE=np.nanmean(Updated_VPRM_Reco0+Updated_VPRM_GEE0, axis=1)


# In[ ]:





# In[27]:


# Load in TPD 2019 flux tower fluxes

# *** CHANGE PATH & FILENAME ***
TPD_2019_Fluxes=pd.read_csv('/Users/kitty/Documents/Research/SIF/Flux_Tower/Turkey_Point/TPD_HH_2019.csv', usecols=(0,2,74,75,76))

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
        if TPD_2019_Fluxes.iat[i,2]>-9999:
            TPD_2019_GPPgf_fluxes[i]=TPD_2019_Fluxes.iat[i,2]


# In[28]:


#Take the hourly average of TPD 2019 flux tower data

TPD_2019_GPP=np.zeros(np.shape(VPRM_TPD_2019_avg_GPP))*np.nan
TPD_2019_NEE=np.zeros(np.shape(VPRM_TPD_2019_avg_GPP))*np.nan
TPD_2019_NEEgf=np.zeros(np.shape(VPRM_TPD_2019_avg_GPP))*np.nan
TPD_2019_R=np.zeros(np.shape(VPRM_TPD_2019_avg_GPP))*np.nan
for i in range(np.int(len(TPD_2019_dates)/2)):
    with np.errstate(invalid='ignore'):
        if i<8755:
            TPD_2019_GPP[i+5]=np.nanmean([TPD_2019_GPPgf_fluxes[i*2],TPD_2019_GPPgf_fluxes[i*2+1]])
            TPD_2019_NEE[i+5]=np.nanmean([TPD_2019_NEE_fluxes[i*2],TPD_2019_NEE_fluxes[i*2+1]])
            TPD_2019_NEEgf[i+5]=np.nanmean([TPD_2019_NEEgf_fluxes[i*2],TPD_2019_NEEgf_fluxes[i*2+1]])
            TPD_2019_R[i+5]=np.nanmean([TPD_2019_Rgf_fluxes[i*2],TPD_2019_Rgf_fluxes[i*2+1]])


# In[ ]:





# In[29]:


#Take the daily average of UrbanVPRM 2019 data over TPD
VPRM_TPD_2019_daily_NEE=np.zeros(365)*np.nan
VPRM_TPD_2019_daily_GPP=np.zeros(365)*np.nan
VPRM_TPD_2019_daily_R=np.zeros(365)*np.nan

Updated_VPRM_TPD_2019_daily_NEE=np.zeros(365)*np.nan
Updated_VPRM_TPD_2019_daily_GPP=np.zeros(365)*np.nan
Updated_VPRM_TPD_2019_daily_R=np.zeros(365)*np.nan

date=0
daily_NEE_VPRM=[]
daily_GPP_VPRM=[]
daily_R_VPRM=[]

daily_NEE_Updated_VPRM=[]
daily_GPP_Updated_VPRM=[]
daily_R_Updated_VPRM=[]
for i in range(len(VPRM_TPD_2019_avg_DoY)):
    if VPRM_TPD_2019_avg_DoY[i]>=1:
        if date+1>=365:
            daily_NEE_VPRM.append(VPRM_TPD_2019_avg_NEE[i])
            daily_GPP_VPRM.append(VPRM_TPD_2019_avg_GPP[i])
            daily_R_VPRM.append(VPRM_TPD_2019_avg_Reco[i])
            
            daily_NEE_Updated_VPRM.append(Updated_VPRM_TPD_2019_avg_NEE[i])
            daily_GPP_Updated_VPRM.append(Updated_VPRM_TPD_2019_avg_GPP[i])
            daily_R_Updated_VPRM.append(Updated_VPRM_TPD_2019_avg_Reco[i])
            if i==len(VPRM_TPD_2019_avg_DoY)-1:
                VPRM_TPD_2019_daily_NEE[date]=np.nanmean(daily_NEE_VPRM)
                VPRM_TPD_2019_daily_GPP[date]=np.nanmean(daily_GPP_VPRM)
                VPRM_TPD_2019_daily_R[date]=np.nanmean(daily_R_VPRM)
                
                Updated_VPRM_TPD_2019_daily_NEE[date]=np.nanmean(daily_NEE_Updated_VPRM)
                Updated_VPRM_TPD_2019_daily_GPP[date]=np.nanmean(daily_GPP_Updated_VPRM)
                Updated_VPRM_TPD_2019_daily_R[date]=np.nanmean(daily_R_Updated_VPRM)
                date+=1
        else:
            daily_NEE_VPRM.append(VPRM_TPD_2019_avg_NEE[i])
            daily_GPP_VPRM.append(VPRM_TPD_2019_avg_GPP[i])
            daily_R_VPRM.append(VPRM_TPD_2019_avg_Reco[i])
            
            daily_NEE_Updated_VPRM.append(Updated_VPRM_TPD_2019_avg_NEE[i])
            daily_GPP_Updated_VPRM.append(Updated_VPRM_TPD_2019_avg_GPP[i])
            daily_R_Updated_VPRM.append(Updated_VPRM_TPD_2019_avg_Reco[i])
            if np.floor(np.round(VPRM_TPD_2019_avg_DoY[i],4))<np.floor(np.round(VPRM_TPD_2019_avg_DoY[i+1],4)):
                VPRM_TPD_2019_daily_NEE[date]=np.nanmean(daily_NEE_VPRM)
                VPRM_TPD_2019_daily_GPP[date]=np.nanmean(daily_GPP_VPRM)
                VPRM_TPD_2019_daily_R[date]=np.nanmean(daily_R_VPRM)
                
                Updated_VPRM_TPD_2019_daily_NEE[date]=np.nanmean(daily_NEE_Updated_VPRM)
                Updated_VPRM_TPD_2019_daily_GPP[date]=np.nanmean(daily_GPP_Updated_VPRM)
                Updated_VPRM_TPD_2019_daily_R[date]=np.nanmean(daily_R_Updated_VPRM)
                                
                date+=1
                daily_NEE_VPRM=[]
                daily_GPP_VPRM=[]
                daily_R_VPRM=[]
                
                daily_NEE_Updated_VPRM=[]
                daily_GPP_Updated_VPRM=[]
                daily_R_Updated_VPRM=[]


# Take the daily average of TPD 2019 flux tower data                
TPD_2019_daily_mean_NEE=np.zeros(365)*np.nan
TPD_2019_daily_mean_GPP=np.zeros(365)*np.nan
TPD_2019_daily_mean_R=np.zeros(365)*np.nan
TPD_2019_daily_mean_NEEgf=np.zeros(365)*np.nan

for i in range(len(days_of_year)):
    daily_NEEgf_TPD_2019=TPD_2019_NEEgf[np.floor(np.round(VPRM_TPD_2019_avg_DoY,4))==i+1]
    daily_NEE_TPD_2019=TPD_2019_NEE[np.floor(np.round(VPRM_TPD_2019_avg_DoY,4))==i+1]
    daily_GPP_TPD_2019=TPD_2019_GPP[np.floor(np.round(VPRM_TPD_2019_avg_DoY,4))==i+1]
    daily_R_TPD_2019=TPD_2019_R[np.floor(np.round(VPRM_TPD_2019_avg_DoY,4))==i+1]
    if len(daily_NEE_TPD_2019)==24:
        TPD_2019_daily_mean_NEEgf[i]=np.mean(daily_NEEgf_TPD_2019)
        TPD_2019_daily_mean_NEE[i]=np.mean(daily_NEE_TPD_2019)
        TPD_2019_daily_mean_GPP[i]=np.mean(daily_GPP_TPD_2019)
        TPD_2019_daily_mean_R[i]=np.mean(daily_R_TPD_2019)


# In[ ]:





# In[30]:


#Combine data from all sites
All_VPRM_NEE=np.concatenate([VPRM_Borden_2018_avg_NEE,VPRM_TP39_2018_avg_NEE,VPRM_TPD_avg_NEE,VPRM_TP39_2019_avg_NEE,VPRM_TPD_2019_avg_NEE])
All_Updated_VPRM_NEE=np.concatenate([Updated_VPRM_Borden_2018_avg_NEE,Updated_VPRM_TP39_2018_avg_NEE,Updated_VPRM_TPD_avg_NEE,Updated_VPRM_TP39_2019_avg_NEE,Updated_VPRM_TPD_2019_avg_NEE])

All_fluxtower_NEE=np.concatenate([Borden_NEE,TP39_NEE,TPD_NEE,TP39_2019_NEE,TPD_2019_NEE])


# In[31]:


finitemask1 = np.isfinite(All_fluxtower_NEE)
All_fluxtower_NEEclean0 = All_fluxtower_NEE[finitemask1]
All_VPRM_NEEclean0 = All_VPRM_NEE[finitemask1]
All_Updated_VPRM_NEEclean0 = All_Updated_VPRM_NEE[finitemask1]

finitemask2 = np.isfinite(All_VPRM_NEEclean0)
Total_fluxtower_VPRM_NEE =  All_fluxtower_NEEclean0[finitemask2]
Total_VPRM_NEE = All_VPRM_NEEclean0[finitemask2]

finitemask3 = np.isfinite(All_Updated_VPRM_NEEclean0)
Total_fluxtower_Updated_VPRM_NEE = All_fluxtower_NEEclean0[finitemask3]
Total_Updated_VPRM_NEE = All_Updated_VPRM_NEEclean0[finitemask3]


# In[32]:


# Fit the original UrbanVPRM NEE to flux tower (non-gapfilled) NEE using a bootstrapped Huber fit

Huber_Tot_NEE_slps=[]
Huber_Tot_NEE_ints=[]
Huber_Tot_NEE_R2=[]

#try bootstrapping 1000 times
indx_list=list(range(0,len(Total_VPRM_NEE)))
for i in range(1,1001):
    #sub selection of points
    NEE_indx=np.random.choice(indx_list,size=len(Total_VPRM_NEE))
    
    try:
        Huber_model = linear_model.HuberRegressor(fit_intercept=True)
        Huber_fit=Huber_model.fit((Total_fluxtower_VPRM_NEE[NEE_indx]).reshape(-1,1),Total_VPRM_NEE[NEE_indx])
        H_m=Huber_fit.coef_
        H_c=Huber_fit.intercept_
        x_accpt, y_accpt = Total_fluxtower_VPRM_NEE, Total_VPRM_NEE
        y_predict = H_m * x_accpt + H_c
        H_R2=r2_score(y_accpt, y_predict)
        Huber_Tot_NEE_slps.append(H_m)
        Huber_Tot_NEE_ints.append(H_c)
        Huber_Tot_NEE_R2.append(H_R2)
    except ValueError: #if Huber fit can't find a solution for the subset, skip it
        pass
    
Huber_Tot_R2 = r2_score(Total_VPRM_NEE, Total_fluxtower_VPRM_NEE*np.nanmean(Huber_Tot_NEE_slps)+np.nanmean(Huber_Tot_NEE_ints))

print('Original VPRM slope: '+str(np.round(np.nanmean(Huber_Tot_NEE_slps),3))+' +/- '+str(np.round(np.nanstd(Huber_Tot_NEE_slps),3)))
print('Original VPRM intercept: '+str(np.round(np.nanmean(Huber_Tot_NEE_ints),3))+' +/- '+str(np.round(np.nanstd(Huber_Tot_NEE_ints),3)))

print('Original VPRM R^2: '+str(np.round(Huber_Tot_R2,3)))


# In[33]:


# Fit the updated UrbanVPRM NEE to flux tower (non-gapfilled) NEE using a bootstrapped Huber fit

#Correct flux tower NEE & average after
Huber_Tot_Updated_NEE_slps=[]
Huber_Tot_Updated_NEE_ints=[]
Huber_Tot_Updated_NEE_R2=[]

#try bootstrapping 1000 times
indx_list=list(range(0,len(Total_Updated_VPRM_NEE)))
for i in range(1,1001):
    #sub selection of points
    NEE_indx=np.random.choice(indx_list,size=len(Total_Updated_VPRM_NEE))
    
    try:
        Huber_model = linear_model.HuberRegressor(fit_intercept=True)
        Huber_fit=Huber_model.fit((Total_fluxtower_Updated_VPRM_NEE[NEE_indx]).reshape(-1,1),Total_Updated_VPRM_NEE[NEE_indx])
        H_m=Huber_fit.coef_
        H_c=Huber_fit.intercept_
        x_accpt, y_accpt = Total_fluxtower_Updated_VPRM_NEE, Total_Updated_VPRM_NEE
        y_predict = H_m * x_accpt + H_c
        H_R2=r2_score(y_accpt, y_predict)
        Huber_Tot_Updated_NEE_slps.append(H_m)
        Huber_Tot_Updated_NEE_ints.append(H_c)
        Huber_Tot_Updated_NEE_R2.append(H_R2)
    except ValueError: #if Huber fit can't find a solution for the subset, skip it
        pass
    
Huber_Tot_Updated_R2 = r2_score(Total_Updated_VPRM_NEE, Total_fluxtower_Updated_VPRM_NEE*np.nanmean(Huber_Tot_Updated_NEE_slps)+np.nanmean(Huber_Tot_Updated_NEE_ints))

print('Updated VPRM slope: '+str(np.round(np.nanmean(Huber_Tot_Updated_NEE_slps),3))+' +/- '+str(np.round(np.nanstd(Huber_Tot_Updated_NEE_slps),3)))
print('Updated VPRM intercept: '+str(np.round(np.nanmean(Huber_Tot_Updated_NEE_ints),3))+' +/- '+str(np.round(np.nanstd(Huber_Tot_Updated_NEE_ints),3)))

print('Updated VPRM R^2: '+str(np.round(Huber_Tot_Updated_R2,3)))


# In[ ]:





# In[34]:


plt.style.use('tableau-colorblind10')
plt.rc('font',size=18)
plt.figure(figsize=(8,6))
plt.xlim(-80,20)
plt.ylim(-80,20)
plt.axis('scaled')

plt.scatter(-100,-100,label='Original UrbanVPRM')
plt.scatter(-100,-100,label='Updated UrbanVPRM')
plt.scatter(Total_fluxtower_VPRM_NEE,Total_VPRM_NEE,s=5,c='#006BA4')
plt.scatter(Total_fluxtower_Updated_VPRM_NEE,Total_Updated_VPRM_NEE,s=5,c='#FF800E',alpha=0.5)

plt.plot(line1_1,func2(line1_1,np.nanmean(Huber_Tot_NEE_slps),np.nanmean(Huber_Tot_NEE_ints)),linestyle='--',label=str(np.round(np.nanmean(Huber_Tot_NEE_slps),2))+'$\cdot$x+'+str(np.round(np.nanmean(Huber_Tot_NEE_ints),2))+', R$^2$ = '+str(np.round(np.nanmean(Huber_Tot_R2),2)),c='k', path_effects=[pe.Stroke(linewidth=5, foreground='#006BA4'), pe.Normal()])
plt.plot(line1_1,func2(line1_1,np.nanmean(Huber_Tot_Updated_NEE_slps),np.nanmean(Huber_Tot_Updated_NEE_ints)),linestyle='-.',label=str(np.round(np.nanmean(Huber_Tot_Updated_NEE_slps),2))+'$\cdot$x+'+str(np.round(np.nanmean(Huber_Tot_Updated_NEE_ints),2))+', R$^2$ = '+str(np.round(Huber_Tot_Updated_R2,2)),c='k', path_effects=[pe.Stroke(linewidth=5, foreground='#FF800E'), pe.Normal()])

plt.plot(line1_1,line1_1,linestyle=':',c='k')
plt.title('UrbanVPRM vs Flux Tower NEE')
plt.xlabel('Flux Tower NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
plt.ylabel('Modelled NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
plt.legend()
# *** Uncomment to save figure as pdf and as png. CHANGE PATHS & FILENAMES *** (part of Fig 2)
#plt.savefig('UrbanVPRM_V061_vs_fixed_fluxtower_non_gapfilled_NEE_hrly_Huber_fit_correlation_All_fluxes_2018_2019_larger_font_cb_friendly.pdf',bbox_inches='tight')
#plt.savefig('UrbanVPRM_V061_vs_fixed_fluxtower_non_gapfilled_NEE_hrly_Huber_fit_correlation_All_fluxes_2018_2019_larger_font_cb_friendly.png',bbox_inches='tight')
plt.show()


# In[35]:


plt.rc('font',size=22)

fig, ax = plt.subplots(3,3,sharex=True,figsize=(11,8))
ax[0,0].set_xlim(1,365)
ax[0,0].set_ylim(-2,22)
ax[0,1].set_ylim(-2,22)
ax[0,2].set_ylim(-2,22)

l0,=ax[0,0].plot(days_of_year,Borden_daily_mean_GPPgf+50,label='Borden Fluxtower',c='k')
ax[0,0].plot(days_of_year,Borden_daily_mean_GPPgf,label='Borden 2018-2020',c='k')

ls0,=ax[0,0].plot(days_of_year,days_of_year+50,label='Original VPRM',c='#006BA4')
ax[0,0].plot(days_of_year,VPRM_daily_GPP_Borden_2018,label='Original VPRM',c='#006BA4',alpha=0.75)

ls1,=ax[0,0].plot(days_of_year,days_of_year+50,label='Updated VPRM',c='#FF800E',linestyle='--')
ax[0,0].plot(days_of_year,Updated_VPRM_daily_GPP_Borden_2018,label='Updated VPRM',c='#FF800E', alpha=0.75,linestyle='--')
ax[0,0].set_title('GPP')

l1=ax[0,1].scatter(days_of_year,TP39_daily_mean_GPP+50,label='TP39 Fluxtower',c='k')
ax[0,1].plot(days_of_year,TP39_daily_mean_GPP,label='TP39 2018-2019',c='k')

ax[0,1].plot(days_of_year,VPRM_TP39_2018_daily_GPP,label='Original VPRM',c='#006BA4',alpha=0.75)

ax[0,1].plot(days_of_year,Updated_VPRM_TP39_2018_daily_GPP,label='Original VPRM',c='#FF800E',alpha=0.75,linestyle='--')
ax[0,0].set_ylabel('GPP')

l2=ax[0,2].scatter(days_of_year,TPD_daily_mean_GPP+50,label='TPD Fluxtower',c='k')
ax[0,2].plot(days_of_year,TPD_daily_mean_GPP,label='TPD 2018-2019',c='k')
ax[0,2].plot(days_of_year,VPRM_TPD_daily_GPP,label='Original VPRM',c='#006BA4',alpha=0.75)
ax[0,2].plot(days_of_year,Updated_VPRM_TPD_daily_GPP,label='Updated VPRM',c='#FF800E',alpha=0.75,linestyle='--')

ax[1,0].set_ylim(-1,22)
ax[1,1].set_ylim(-1,22)
ax[1,2].set_ylim(-1,22)

ax[1,0].plot(days_of_year,Borden_daily_mean_Rgf,label='Borden 2018-2020',c='k')
ax[1,0].plot(days_of_year,VPRM_daily_R_Borden_2018,label='Original VPRM',c='#006BA4',alpha=0.75)
ax[1,0].plot(days_of_year,Updated_VPRM_daily_R_Borden_2018,label='Updated VPRM',c='#FF800E', alpha=0.75,linestyle='--')
ax[0,0].set_title('GPP')

ax[1,1].plot(days_of_year,TP39_daily_mean_R,label='TP39 2018-2019',c='k')
ax[1,1].plot(days_of_year,VPRM_TP39_2018_daily_R,label='Original VPRM',c='#006BA4',alpha=0.75)
ax[1,1].plot(days_of_year,Updated_VPRM_TP39_2018_daily_R,label='Original VPRM',c='#FF800E',alpha=0.75,linestyle='--')

l2=ax[1,2].scatter(days_of_year,TPD_daily_mean_R+50,label='TPD Fluxtower',c='k')
ax[1,2].plot(days_of_year,TPD_daily_mean_R,label='TPD 2018-2019',c='k')
ax[1,2].plot(days_of_year,VPRM_TPD_daily_R,label='Original VPRM',c='#006BA4',alpha=0.75)
ax[1,2].plot(days_of_year,Updated_VPRM_TPD_daily_R,label='Updated VPRM',c='#FF800E',alpha=0.75,linestyle='--')

ax[1,0].set_ylabel('R$_{eco}$')

ax[2,0].set_ylim(-16,6)
ax[2,1].set_ylim(-16,6)
ax[2,2].set_ylim(-16,6)

ax[2,0].plot(days_of_year,Borden_daily_mean_NEEgf,label='Borden 2018-2020',c='k')
ax[2,0].plot(days_of_year,VPRM_daily_NEE_Borden_2018,label='Original VPRM',c='#006BA4',alpha=0.75)
ax[2,0].plot(days_of_year,Updated_VPRM_daily_NEE_Borden_2018,label='Updated VPRM',c='#FF800E', alpha=0.75,linestyle='--')

ax[0,0].set_title('Borden Forest')
ax[0,1].set_title('TP39')
ax[0,2].set_title('TPD')

ax[2,1].plot(days_of_year,TP39_daily_mean_NEEgf,label='TP39 2018-2019',c='k')
ax[2,1].plot(days_of_year,VPRM_TP39_2018_daily_NEE,label='Original VPRM',c='#006BA4',alpha=0.75)
ax[2,1].plot(days_of_year,Updated_VPRM_TP39_2018_daily_NEE,label='Original VPRM',c='#FF800E',alpha=0.75,linestyle='--')

ax[2,2].plot(days_of_year,TPD_daily_mean_NEEgf,label='TPD 2018-2019',c='k')
ax[2,2].plot(days_of_year,VPRM_TPD_daily_NEE,label='Original VPRM',c='#006BA4',alpha=0.75)
ax[2,2].plot(days_of_year,Updated_VPRM_TPD_daily_NEE,label='Updated VPRM',c='#FF800E',alpha=0.75,linestyle='--')

ax[2,0].set_ylabel('NEE')

ax[0,1].set_yticks([])
ax[0,2].set_yticks([])
ax[1,1].set_yticks([])
ax[1,2].set_yticks([])
ax[2,1].set_yticks([])
ax[2,2].set_yticks([])

ax[1,0].legend([l0,ls0,ls1],['Flux Tower','Original VPRM','Updated VPRM'],loc='upper left',fontsize=16)
ax[2,1].set_xlabel('Day of Year')
fig.subplots_adjust(hspace=0,wspace=0)
# *** Uncomment next two lines to save figure as pdf and png. CHANGE PATHS & FILENAMES *** (part of fig 2)
#plt.savefig('UrbanVPRM_V061_vs_fixed_fluxtower_Comparison_All_fluxes_2018_larger_font_cb_friendly.pdf',bbox_inches='tight')
#plt.savefig('UrbanVPRM_V061_vs_fixed_fluxtower_Comparison_All_fluxes_2018_larger_font_cb_friendly.png',bbox_inches='tight')
fig.show()


# In[ ]:




