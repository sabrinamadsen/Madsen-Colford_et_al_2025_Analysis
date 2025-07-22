#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Code used to compare spatial patterns in normalized SIF and EVI to those of updated UrbanVPRM - SMUrF NEE over Toronto
# We normalize EVI & SIF over 3 flux tower locations in Southern Ontario for more direct comparison

# Used to create figures 7 and S6 of Madsen-Colford et al.
# If used please cite

# *** denotes portions of the code that should be changed by the user


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colr
import csv
import pandas as pd
from scipy import optimize as opt 
from scipy import odr
import shapefile as shp # to import outline of GTA
from shapely import geometry # used to define a polygon for Toronto
import netCDF4
from netCDF4 import Dataset, date2num #for reading netCDF data files and their date (not sure if I need the later)
from sklearn import linear_model #for robust fitting
from sklearn.metrics import r2_score, mean_squared_error #for analyzing robust fits
import matplotlib.colors as clrs #for log color scale


# In[2]:


# Import EVI data

# *** CHANGE PATH AND FILENAME ***
VPRM_EVI_Borden=pd.read_csv('C:/Users/kitty/Documents/Research/SIF/UrbanVPRM/UrbanVPRM/dataverse_files/Borden_V061_500m_2018/adjusted_evi_lswi_interpolated_modis_v061_qc_filtered_LSWI_filtered.csv').loc[:,('DOY','Index','x','y','EVI_inter','LSWI_inter')]

#Create a dataframe with just Index, x, & y values
x_Borden=np.zeros(np.shape(VPRM_EVI_Borden.Index.unique()))*np.nan
y_Borden=np.zeros(np.shape(VPRM_EVI_Borden.Index.unique()))*np.nan
for i in range(len(VPRM_EVI_Borden.Index.unique())):
    x_Borden[i]=VPRM_EVI_Borden.x[0+i*365]
    y_Borden[i]=VPRM_EVI_Borden.y[0+i*365]


# In[3]:


# Shape EVI, lats & lons, into arrays
xvals_Borden = VPRM_EVI_Borden.x[VPRM_EVI_Borden.DOY==200].unique()
yvals_Borden = VPRM_EVI_Borden.y[VPRM_EVI_Borden.DOY==200].unique()
extent_Borden = np.min(xvals_Borden), np.max(xvals_Borden), np.min(yvals_Borden), np.max(yvals_Borden)
zvals_Borden= np.zeros([365,16,16])*np.nan
for i in range(1,366):
    zvals_Borden[i-1] = VPRM_EVI_Borden.EVI_inter[VPRM_EVI_Borden.DOY==i].values.reshape(len(yvals_Borden),len(xvals_Borden))


# In[4]:


#Select EVI over Borden Forest flux towers
EVI_data_Borden=np.zeros([46,16,16])
Borden_EVI =np.zeros(46)*np.nan
for i in range(46):
    EVI_data_Borden[i]=np.mean(zvals_Borden[i*8:i*8+8],axis=0)
    Borden_EVI[i]=np.mean((EVI_data_Borden[i,6,8],EVI_data_Borden[i,7,7],EVI_data_Borden[i,7,8],EVI_data_Borden[i,8,6],EVI_data_Borden[i,8,7],EVI_data_Borden[i,8,8]))


# In[5]:


#Import downscaled SIF data

# *** CHANGE PATH ***
sif_path='C:/Users/kitty/Documents/Research/SIF/SMUrF/data/downscaled_CSIF/TROPOMI_CSIF_combined_med/V061/2018/V3/'
# *** CHANGE FILENAMES ***
sif_fn_shore_corr='downscaled_V061_TROPO_CSIF_shore_weighted_corrected_8d_2018' #File name WITHOUT day of year
sif_fn_no_corr='downscaled_V061_TROPO_CSIF_8d_2018' #File name WITHOUT day of year

# Import lat & lon from the first day of the year 
g=Dataset(sif_path+sif_fn_no_corr+'001.nc')
TROPO_sif=g.variables['daily_sif'][:]
lons = g.variables['lon'][:]
lats = g.variables['lat'][:]
SIF_data = np.zeros([46,553,625])
g.close()


for i in range(1,366,8):
    
    if i<10:
        try:
            g=Dataset(sif_path+sif_fn_shore_corr+'00'+str(i)+'.nc')
            SIF_data[np.int((i-1)/8)]=g.variables['daily_sif'][:]
        except FileNotFoundError:
            g=Dataset(sif_path+sif_fn_no_corr+'00'+str(i)+'.nc')
            SIF_data[np.int((i-1)/8)]=g.variables['daily_sif'][:][::-1]
    elif i<100:
        try:
            g=Dataset(sif_path+sif_fn_shore_corr+'0'+str(i)+'.nc')
            SIF_data[np.int((i-1)/8)]=g.variables['daily_sif'][:]
        except FileNotFoundError:
            g=Dataset(sif_path+sif_fn_no_corr+'0'+str(i)+'.nc')
            SIF_data[np.int((i-1)/8)]=g.variables['daily_sif'][:][::-1]
    else:
        try:
            g=Dataset(sif_path+sif_fn_shore_corr+str(i)+'.nc')
            SIF_data[np.int((i-1)/8)]=g.variables['daily_sif'][:]
        except FileNotFoundError:
            g=Dataset(sif_path+sif_fn_no_corr+str(i)+'.nc')
            SIF_data[np.int((i-1)/8)]=g.variables['daily_sif'][:][::-1]
    g.close()


# In[6]:


#Select SIF data over Borden Forest Flux tower
Borden_SIF=np.nanmean([np.swapaxes(SIF_data,0,2)[::-1][458,230],np.swapaxes(SIF_data,0,2)[::-1][458,231],np.swapaxes(SIF_data,0,2)[::-1][458,232],np.swapaxes(SIF_data,0,2)[::-1][459,230],np.swapaxes(SIF_data,0,2)[::-1][459,231],np.swapaxes(SIF_data,0,2)[::-1][459,232],np.swapaxes(SIF_data,0,2)[::-1][460,232]],axis=0)


# In[7]:


#Define an array for day of the year
DoY=np.arange(1,365,8)


# In[8]:


#Load in EVI over TP39

# *** Change Path & Filename ***
VPRM_EVI_TP39=pd.read_csv('C:/Users/kitty/Documents/Research/SIF/UrbanVPRM/UrbanVPRM/dataverse_files/TP39_V061_500m_2018/adjusted_evi_lswi_interpolated_modis_v061_qc_filtered_LSWI_filtered.csv').loc[:,('DOY','Index','x','y','EVI_inter','LSWI_inter')]

#Create a dataframe with just Index, x, & y values
x_TP39=np.zeros(np.shape(VPRM_EVI_TP39.Index.unique()))*np.nan
y_TP39=np.zeros(np.shape(VPRM_EVI_TP39.Index.unique()))*np.nan
for i in range(len(VPRM_EVI_TP39.Index.unique())):
    x_TP39[i]=VPRM_EVI_TP39.x[0+i*365]
    y_TP39[i]=VPRM_EVI_TP39.y[0+i*365]

xvals_TP39 = VPRM_EVI_TP39.x[VPRM_EVI_TP39.DOY==200].unique()
yvals_TP39 = VPRM_EVI_TP39.y[VPRM_EVI_TP39.DOY==200].unique()
extent_TP39 = np.min(xvals_TP39), np.max(xvals_TP39), np.min(yvals_TP39), np.max(yvals_TP39)
zvals_TP39= np.zeros([365,16,16])*np.nan
for i in range(1,366):
    zvals_TP39[i-1] = VPRM_EVI_TP39.EVI_inter[VPRM_EVI_TP39.DOY==i].values.reshape(len(yvals_TP39),len(xvals_TP39))


# In[9]:


# Select EVI & SIF data over TP39 flux tower
EVI_data_TP39=np.zeros([46,16,16])
TP39_EVI =np.zeros(46)*np.nan
for i in range(len(SIF_data)):
    EVI_data_TP39[i]=np.mean(zvals_TP39[i*8:i*8+8],axis=0)
    TP39_EVI[i]=np.mean((EVI_data_TP39[i,6,7],EVI_data_TP39[i,6,8],EVI_data_TP39[i,7,7],EVI_data_TP39[i,7,8]))


# In[10]:


TP39_SIF=np.nanmean([np.swapaxes(SIF_data,0,2)[::-1][73,129],np.swapaxes(SIF_data,0,2)[::-1][73,130],np.swapaxes(SIF_data,0,2)[::-1][74,129],np.swapaxes(SIF_data,0,2)[::-1][74,130]],axis=0)


# In[ ]:





# In[11]:


#Load in EVI over TPD

# *** CHANGE PATH & FILENAME ***
VPRM_EVI_TPD=pd.read_csv('C:/Users/kitty/Documents/Research/SIF/UrbanVPRM/UrbanVPRM/dataverse_files/TPD_V061_500m_2018/adjusted_evi_lswi_interpolated_modis_v061_qc_filtered_LSWI_filtered.csv').loc[:,('DOY','Index','x','y','EVI_inter','LSWI_inter')]

#Create a dataframe with just Index, x, & y values
x_TPD=np.zeros(np.shape(VPRM_EVI_TPD.Index.unique()))*np.nan
y_TPD=np.zeros(np.shape(VPRM_EVI_TPD.Index.unique()))*np.nan
for i in range(len(VPRM_EVI_TPD.Index.unique())):
    x_TPD[i]=VPRM_EVI_TPD.x[0+i*365]
    y_TPD[i]=VPRM_EVI_TPD.y[0+i*365]

xvals_TPD = VPRM_EVI_TPD.x[VPRM_EVI_TPD.DOY==200].unique()
yvals_TPD = VPRM_EVI_TPD.y[VPRM_EVI_TPD.DOY==200].unique()
extent_TPD = np.min(xvals_TPD), np.max(xvals_TPD), np.min(yvals_TPD), np.max(yvals_TPD)
zvals_TPD= np.zeros([365,16,16])*np.nan
for i in range(1,366):
    zvals_TPD[i-1] = VPRM_EVI_TPD.EVI_inter[VPRM_EVI_TPD.DOY==i].values.reshape(len(yvals_TPD),len(xvals_TPD))


# In[12]:


#Select EVI & SIF data over TPD flux tower

EVI_data_TPD=np.zeros([46,16,16])
TPD_EVI =np.zeros(46)*np.nan
for i in range(len(SIF_data)):
    EVI_data_TPD[i]=np.mean(zvals_TPD[i*8:i*8+8],axis=0)
    TPD_EVI[i]=np.mean((EVI_data_TPD[i,6,6],EVI_data_TPD[i,6,7],EVI_data_TPD[i,6,8],EVI_data_TPD[i,7,6],EVI_data_TPD[i,7,7],EVI_data_TPD[i,7,8],EVI_data_TPD[i,8,6],EVI_data_TPD[i,8,7],EVI_data_TPD[i,8,8]))


# In[13]:


TPD_SIF=np.nanmean([np.swapaxes(SIF_data,0,2)[::-1][55,80],np.swapaxes(SIF_data,0,2)[::-1][55,81],np.swapaxes(SIF_data,0,2)[::-1][55,82],np.swapaxes(SIF_data,0,2)[::-1][56,80],np.swapaxes(SIF_data,0,2)[::-1][56,81],np.swapaxes(SIF_data,0,2)[::-1][56,82],np.swapaxes(SIF_data,0,2)[::-1][57,80],np.swapaxes(SIF_data,0,2)[::-1][57,81],np.swapaxes(SIF_data,0,2)[::-1][57,82]],axis=0)


# In[ ]:





# In[14]:


#Load in 2019 TP39 EVI data

# *** CHANGE PATH & FILENAME
VPRM_EVI_TP39_2019=pd.read_csv('C:/Users/kitty/Documents/Research/SIF/UrbanVPRM/UrbanVPRM/dataverse_files/TP39_V061_500m_2019/adjusted_evi_lswi_interpolated_modis_v061_qc_filtered_LSWI_filtered.csv').loc[:,('DOY','Index','x','y','EVI_inter','LSWI_inter')]

zvals_TP39_2019= np.zeros([365,16,16])*np.nan
for i in range(1,366):
    zvals_TP39_2019[i-1] = VPRM_EVI_TP39_2019.EVI_inter[VPRM_EVI_TP39_2019.DOY==i].values.reshape(len(yvals_TP39),len(xvals_TP39))


# In[15]:


EVI_data_TP39_2019=np.zeros([46,16,16])
TP39_2019_EVI =np.zeros(46)*np.nan
for i in range(len(SIF_data)):
    EVI_data_TP39_2019[i]=np.mean(zvals_TP39_2019[i*8:i*8+8],axis=0)
    TP39_2019_EVI[i]=np.mean((EVI_data_TP39_2019[i,6,7],EVI_data_TP39_2019[i,6,8],EVI_data_TP39_2019[i,7,7],EVI_data_TP39_2019[i,7,8]))


# In[16]:


# Load in 2019 SIF data

SIF_data_2019 = np.zeros([46,553,625])#np.zeros([46,96,144])*np.nan

# *** CHANGE PATH ***
sif_path='C:/Users/kitty/Documents/Research/SIF/SMUrF/data/downscaled_CSIF/TROPOMI_CSIF_combined_med/V061/2019/V3/'
# *** CHANGE FILENAMES ***
sif_fn_shore_corr='downscaled_V061_TROPO_CSIF_shore_weighted_corrected_8d_2019' #File name WITHOUT day of year
sif_fn_no_corr='downscaled_V061_TROPO_CSIF_8d_2019' #File name WITHOUT day of year

for i in range(1,366,8):
    
    if i<10:
        try:
            g=Dataset(sif_path+sif_fn_shore_corr+'00'+str(i)+'.nc')
            SIF_data_2019[np.int((i-1)/8)]=g.variables['daily_sif'][:]
        except FileNotFoundError:
            g=Dataset(sif_path+sif_fn_no_corr+'00'+str(i)+'.nc')
            SIF_data_2019[np.int((i-1)/8)]=g.variables['daily_sif'][:][::-1]
    elif i<100:
        try:
            g=Dataset(sif_path+sif_fn_shore_corr+'0'+str(i)+'.nc')
            print('shore line corrected')
        except FileNotFoundError:
            g=Dataset(sif_path+sif_fn_no_corr+'0'+str(i)+'.nc')
            SIF_data_2019[np.int((i-1)/8)]=g.variables['daily_sif'][:][::-1]
    else:
        try:
            g=Dataset(sif_path+sif_fn_no_corr+str(i)+'.nc')
            SIF_data_2019[np.int((i-1)/8)]=g.variables['daily_sif'][:]
        except FileNotFoundError:
            g=Dataset(sif_path+sif_fn_no_corr+str(i)+'.nc')
            SIF_data_2019[np.int((i-1)/8)]=g.variables['daily_sif'][:][::-1]
    g.close()


# In[17]:


TP39_2019_SIF=np.nanmean([np.swapaxes(SIF_data_2019,0,2)[::-1][73,129],np.swapaxes(SIF_data_2019,0,2)[::-1][73,130],np.swapaxes(SIF_data_2019,0,2)[::-1][74,129],np.swapaxes(SIF_data_2019,0,2)[::-1][74,130]],axis=0)


# In[18]:


#Import 2019 EVI over TPD

# *** CHANGE PATH & FILENAME ***
VPRM_EVI_TPD_2019=pd.read_csv('C:/Users/kitty/Documents/Research/SIF/UrbanVPRM/UrbanVPRM/dataverse_files/TPD_V061_500m_2019/adjusted_evi_lswi_interpolated_modis_v061_qc_filtered_LSWI_filtered.csv').loc[:,('DOY','Index','x','y','EVI_inter','LSWI_inter')]

zvals_TPD_2019= np.zeros([365,16,16])*np.nan
for i in range(1,366):
    zvals_TPD_2019[i-1] = VPRM_EVI_TPD_2019.EVI_inter[VPRM_EVI_TPD_2019.DOY==i].values.reshape(len(yvals_TPD),len(xvals_TPD))


# In[19]:


EVI_data_TPD_2019=np.zeros([46,16,16])
TPD_2019_EVI =np.zeros(46)*np.nan
for i in range(len(SIF_data_2019)):
    EVI_data_TPD_2019[i]=np.mean(zvals_TPD_2019[i*8:i*8+8],axis=0)
    TPD_2019_EVI[i]=np.mean((EVI_data_TPD_2019[i,6,6],EVI_data_TPD_2019[i,6,7],EVI_data_TPD_2019[i,6,8],EVI_data_TPD_2019[i,7,6],EVI_data_TPD_2019[i,7,7],EVI_data_TPD_2019[i,7,8],EVI_data_TPD_2019[i,8,6],EVI_data_TPD_2019[i,8,7],EVI_data_TPD_2019[i,8,8]))

TPD_2019_SIF=np.nanmean([np.swapaxes(SIF_data_2019,0,2)[::-1][55,80],np.swapaxes(SIF_data_2019,0,2)[::-1][55,81],np.swapaxes(SIF_data_2019,0,2)[::-1][55,82],np.swapaxes(SIF_data_2019,0,2)[::-1][56,80],np.swapaxes(SIF_data_2019,0,2)[::-1][56,81],np.swapaxes(SIF_data_2019,0,2)[::-1][56,82],np.swapaxes(SIF_data_2019,0,2)[::-1][57,80],np.swapaxes(SIF_data_2019,0,2)[::-1][57,81],np.swapaxes(SIF_data_2019,0,2)[::-1][57,82]],axis=0)


# In[ ]:





# In[20]:


# Load in EVI over Toronto

# *** CHANGE PATH & FILENAME ***
VPRM_EVI=pd.read_csv('E:/Research/UrbanVPRM/dataverse_files/GTA_V061_500m_2018/adjusted_evi_lswi_interpolated_modis_v061_qc_filtered_LSWI_filtered.csv').loc[:,('DOY','Index','x','y','EVI_inter','LSWI_inter')]

#Create a dataframe with just Index, x, & y values
x=np.zeros(np.shape(VPRM_EVI.Index.unique()))*np.nan
y=np.zeros(np.shape(VPRM_EVI.Index.unique()))*np.nan
for i in range(len(VPRM_EVI.Index.unique())):
    x[i]=VPRM_EVI.x[0+i*365]
    y[i]=VPRM_EVI.y[0+i*365]


# In[ ]:





# In[21]:


# Load in Toronto shape file

# *** CHANGE PATH ***
sf = shp.Reader("C:/Users/kitty/Documents/Research/SIF/Shape_files/Toronto/Toronto_Boundary.shp")
#Toronto_Shape
shape=sf.shape(0)
#Need to partition each individual shape
Toronto_x = np.zeros((len(shape.points),1))*np.nan #The main portion of the GTA
Toronto_y = np.zeros((len(shape.points),1))*np.nan
for i in range(len(shape.points)):
    Toronto_x[i]=shape.points[i][0]
    Toronto_y[i]=shape.points[i][1]


# In[22]:


xvals = VPRM_EVI.x[VPRM_EVI.DOY==200].unique()
yvals = VPRM_EVI.y[VPRM_EVI.DOY==200].unique()
extent = np.min(xvals), np.max(xvals), np.min(yvals), np.max(yvals)
zvals= np.zeros([365,96,144])*np.nan
for i in range(1,366):
    zvals[i-1] = VPRM_EVI.EVI_inter[VPRM_EVI.DOY==i].values.reshape(len(yvals),len(xvals))


# In[23]:


EVI_data=np.zeros([46,96,144])
for i in range(len(SIF_data)):
    EVI_data[i]=np.mean(zvals[i*8:i*8+8],axis=0)


# In[24]:


#Calculate the average offset & scaling factor to normalize EVI & SIF

avg_EVI_offset = np.nanmean([np.nanmin(Borden_EVI),np.nanmin(TP39_EVI),np.nanmin(TPD_EVI),np.nanmin(TP39_2019_EVI),np.nanmin(TPD_2019_EVI)])
avg_EVI_scl_fctr = np.nanmean([np.nanmax(Borden_EVI)-np.nanmin(Borden_EVI),np.nanmax(TP39_EVI)-np.nanmin(TP39_EVI),np.nanmax(TPD_EVI)-np.nanmin(TPD_EVI),np.nanmax(TP39_2019_EVI)-np.nanmin(TP39_2019_EVI),np.nanmax(TPD_2019_EVI)-np.nanmin(TPD_2019_EVI)])
avg_SIF_scl_fctr = np.nanmean([np.nanmax(Borden_SIF),np.nanmax(TP39_SIF),np.nanmax(TPD_SIF),np.nanmax(TP39_2019_SIF),np.nanmax(TPD_2019_SIF)])


# In[25]:


#Normalize the EVI & SIF data

EVI_data_normalized = (EVI_data-avg_EVI_offset)/avg_EVI_scl_fctr
EVI_data_normalized[:,SIF_data[25][::-1][264:360,288:432][::-1]==0]=0
SIF_data_normalized = (SIF_data)/avg_SIF_scl_fctr


# ## Now compare the EVI-SIF difference to the UrbanVPRM-SMUrF difference

# In[26]:


# Remove EVI over flux towers to save space
del VPRM_EVI_Borden, VPRM_EVI_TP39, VPRM_EVI_TP39_2019, VPRM_EVI_TPD, VPRM_EVI_TPD_2019


# In[27]:


# Load in UrbanVPRM NEE data

# *** Change Paths & Filenames ***
VPRM_data=pd.read_csv('E:/Research/UrbanVPRM/dataverse_files/GTA_V061_500m_2018/vprm_GMIS_Toronto_ACI_SOLRIS_ISA_500m_GTA_V061_2018_no_PScale_adjusted_Topt_Ra_URB_parameters_fixed_gapfilled_LSWI_filtered_bilinear_PAR_block_00000001.csv').loc[:,('HoY','Index','GEE','Re')]#,"TScale","SEoS_Scale","WScale","PAR","Tair","Re","Ra","Rh","EVI_scale")]

VPRM_data2=pd.read_csv('E:/Research/UrbanVPRM/dataverse_files/GTA_V061_500m_2018/vprm_GMIS_Toronto_ACI_SOLRIS_ISA_500m_GTA_V061_2018_no_PScale_adjusted_Topt_Ra_URB_parameters_fixed_gapfilled_LSWI_filtered_bilinear_PAR_block_00002501.csv').loc[:,('HoY','Index','GEE','Re')]#"TScale","SEoS_Scale","WScale","PAR","Tair","Re","Ra","Rh","EVI_scale")]
VPRM_data=VPRM_data.append(VPRM_data2)
del VPRM_data2

VPRM_data2=pd.read_csv('E:/Research/UrbanVPRM/dataverse_files/GTA_V061_500m_2018/vprm_GMIS_Toronto_ACI_SOLRIS_ISA_500m_GTA_V061_2018_no_PScale_adjusted_Topt_Ra_URB_parameters_fixed_gapfilled_LSWI_filtered_bilinear_PAR_block_00005001.csv').loc[:,('HoY','Index','GEE','Re')]
VPRM_data=VPRM_data.append(VPRM_data2)
del VPRM_data2

VPRM_data2=pd.read_csv('E:/Research/UrbanVPRM/dataverse_files/GTA_V061_500m_2018/vprm_GMIS_Toronto_ACI_SOLRIS_ISA_500m_GTA_V061_2018_no_PScale_adjusted_Topt_Ra_URB_parameters_fixed_gapfilled_LSWI_filtered_bilinear_PAR_block_00007501.csv').loc[:,('HoY','Index','GEE','Re')]
VPRM_data=VPRM_data.append(VPRM_data2)
del VPRM_data2

VPRM_data2=pd.read_csv('E:/Research/UrbanVPRM/dataverse_files/GTA_V061_500m_2018/vprm_GMIS_Toronto_ACI_SOLRIS_ISA_500m_GTA_V061_2018_no_PScale_adjusted_Topt_Ra_URB_parameters_fixed_gapfilled_LSWI_filtered_bilinear_PAR_block_00010001.csv').loc[:,('HoY','Index','GEE','Re')]
VPRM_data=VPRM_data.append(VPRM_data2)
del VPRM_data2

VPRM_data2=pd.read_csv('E:/Research/UrbanVPRM/dataverse_files/GTA_V061_500m_2018/vprm_GMIS_Toronto_ACI_SOLRIS_ISA_500m_GTA_V061_2018_no_PScale_adjusted_Topt_Ra_URB_parameters_fixed_gapfilled_LSWI_filtered_bilinear_PAR_block_00012501.csv').loc[:,('HoY','Index','GEE','Re')]
VPRM_data=VPRM_data.append(VPRM_data2)
del VPRM_data2


# In[28]:


#Reshape data into arrays
GPP=-VPRM_data.GEE.values.reshape(len(yvals),len(xvals),8760)#8784 for leap year
Reco=VPRM_data.Re.values.reshape(len(yvals),len(xvals),8760)


# In[29]:


# Create a mask for areas outside of Toronto

points=[]
for k in range(1,len(Toronto_x)):
    points.append(geometry.Point(Toronto_x[k],Toronto_y[k]))
poly=geometry.Polygon([[p.x, p.y] for p in points])

#Create a mask for areas outside the GTA
lons=np.ones(144)*np.nan
lats=np.ones(96)*np.nan
GPP_mask=np.ones([96,144])*np.nan
for i in range(0, len(lons)):
    for j in range(0, len(lats)):
        if poly.contains(geometry.Point([xvals[i],yvals[j]])):# or poly.contains(geometry.Point([xvals[i+126],yvals[j+129]+1/240])):
            lons[i]=xvals[i]
            lats[j]=yvals[j]
            GPP_mask[j,i]=1


# In[30]:


#Apply the Toronto maks & calculate NEE
GPP=(np.swapaxes(np.swapaxes(GPP,0,2),1,2))*GPP_mask[np.newaxis,:,:]
Reco=(np.swapaxes(np.swapaxes(Reco,0,2),1,2))*GPP_mask[np.newaxis,:,:]
NEE=(Reco-GPP)*GPP_mask[np.newaxis,:,:]


# In[31]:


# Calculate the 8-day average
VPRM_GPP_8day=np.ones((46, 96, 144))*np.nan
VPRM_Reco_8day=np.ones((46, 96, 144))*np.nan
VPRM_NEE_8day=np.ones((46, 96, 144))*np.nan
for i in range(46):
    VPRM_GPP_8day[i]=np.nanmean(GPP[i*8*24:i*8*24+8*24],axis=0)
    VPRM_Reco_8day[i]=np.nanmean(Reco[i*8*24:i*8*24+8*24],axis=0)
    VPRM_NEE_8day[i]=np.nanmean(NEE[i*8*24:i*8*24+8*24],axis=0)


# In[ ]:





# In[32]:


#now bring in the SMUrF data with ISA adjustment AND shoreline correction

# *** CHANGE PATH ***
SMUrF_path='C:/Users/kitty/Documents/Research/SIF/SMUrF/output2018_500m_CSIF_to_TROPOMI_CSIF_ALL_converted_slps_V3_temp_impervious_R_shore_corr_V061_8day/easternCONUS/'
# *** CHANGE FILE NAME ***
SMUrF_fn='daily_mean_Reco_ISA_a_neuralnet/era5/2018/daily_mean_Reco_uncert_GMIS_Toronto_t_easternCONUS_2018' #File name WITHOUT month or day

g=Dataset(SMUrF_path+SMUrF_fn+'0101.nc')
start_of_year=g.variables['time'][0]/3600/24-1 #convert seconds since 1970 to days (minus one)

g.close()

#With ISA adjustment using GMIS-Toronto-SOLRIS-ACI dataset
S_time=[]
S_Reco=[]
S_Reco_err=[]
S_lats_8day=[]
S_lons_8day=[]
for j in range(1,13):
    for i in range(1,32):
        try:
            if j<10:
                if i<10:
                    f=Dataset(SMUrF_path+SMUrF_fn+'0'+str(j)+'0'+str(i)+'.nc')
                else:
                    f=Dataset(SMUrF_path+SMUrF_fn+'0'+str(j)+str(i)+'.nc')
            else:
                if i<10:
                    f=Dataset(SMUrF_path+SMUrF_fn+str(j)+'0'+str(i)+'.nc')
                else:
                    f=Dataset(SMUrF_path+SMUrF_fn+str(j)+str(i)+'.nc')
            if len(S_time)==0:
                S_lats_8day=f.variables['lat'][:]
                S_lons_8day=f.variables['lon'][:]
                S_Reco=f.variables['Reco_mean'][:]
                S_Reco_err=f.variables['Reco_sd'][:]
                S_time=f.variables['time'][:]/24/3600-start_of_year-5/24 #convert seconds since 1970 to days and subtract start of year
            else:
                S_Reco=np.concatenate((S_Reco,f.variables['Reco_mean'][:]),axis=0)
                S_Reco_err=np.concatenate((S_Reco_err,f.variables['Reco_sd'][:]),axis=0)
                S_time=np.concatenate((S_time,(f.variables['time'][:]/24/3600-start_of_year-5/24)),axis=0)
            f.close()
        except FileNotFoundError:
            pass

        
# Load in GPP data
# *** CHANGE GPP FILE NAME ***
f=Dataset(SMUrF_path+'daily_mean_SIF_GPP_uncert_easternCONUS_2018.nc')
S_time_8day=f.variables['time'][:]/24/3600-start_of_year-5/24 #convert seconds since 1970 to days and subtract start of year

S_GPP_err_8day=f.variables['GPP_sd'][:]
S_GPP_8day=f.variables['GPP_mean'][:]

# Replace fill values with NaN
S_Reco[S_Reco==-999]=np.nan
S_Reco_err[S_Reco_err==-999]=np.nan
S_GPP_8day[S_GPP_8day==-999]=np.nan
S_GPP_err_8day[S_GPP_err_8day==-999]=np.nan

#Take 8-day average of Reco
S_Reco_8day=np.ones(np.shape(S_GPP_8day))*np.nan
S_Reco_err_8day=np.ones(np.shape(S_GPP_8day))*np.nan
S_Reco_std_8day=np.ones(np.shape(S_GPP_8day))*np.nan
for i in range(len(S_time_8day)):
    S_Reco_8day[i]=np.nanmean(S_Reco[i*8:i*8+8],axis=0)
    S_Reco_err_8day[i]=np.sqrt(np.nansum((S_Reco_err[i*8:i*8+8]/4)**2,axis=0))
    S_Reco_std_8day[i]=np.nanstd(S_Reco[i*8:i*8+8],axis=0)
    
#Compute NEE
S_NEE_8day=S_Reco_8day-S_GPP_8day
S_NEE_err_8day=np.sqrt(S_Reco_err_8day**2+S_GPP_err_8day**2)


# In[33]:


# Select data over Toronto
S_GPP_8day=S_GPP_8day[:,264:360,288:432]
S_Reco_8day=S_Reco_8day[:,264:360,288:432]
S_NEE_8day=S_NEE_8day[:,264:360,288:432]


# In[ ]:





# In[34]:


#Define a function and a straight line for plotting

def func2(x,m,b):
    return m*x+b

line1_1=np.arange(-100,100)


# In[35]:


#Format SIF & SMUrF NEE to the same format as EVI and UrbanVPRM
SIF_data_norm_flipped=np.zeros(np.shape(EVI_data_normalized))*np.nan
S_NEE_flipped=np.zeros(np.shape(VPRM_NEE_8day))*np.nan
for i in range(len(SIF_data_normalized)):
    SIF_data_norm_flipped[i] = SIF_data_normalized[i][::-1][264:360,288:432][::-1]
    S_NEE_flipped[i] = S_NEE_8day[i][::-1]


# In[36]:


#Apply 1000 times bootstrapped Huber fit to Normalized EVI-SIF and difference between SMUrF & VPRM NEE 

SIF_EVI=(EVI_data_normalized[23]-SIF_data_norm_flipped[23])*GPP_mask
S_VPRM=(S_NEE_flipped[23]-VPRM_NEE_8day[23])*GPP_mask

finitemask0=np.isfinite(SIF_EVI) & np.isfinite(S_VPRM) & (S_VPRM!=0)
SIF_EVI_clean0=SIF_EVI[finitemask0]
S_VPRM_clean0=S_VPRM[finitemask0]

Huber_slps=[]
Huber_ints=[]
Huber_R2s=[]

#try bootstrapping
indx_list=list(range(0,len(S_VPRM_clean0)))
for i in range(1,1001):
    #sub selection of points
    indx=np.random.choice(indx_list,size=len(S_VPRM_clean0))
    
    try:
        Huber_model = linear_model.HuberRegressor(fit_intercept=True)
        Huber_fit=Huber_model.fit((SIF_EVI_clean0[indx]).reshape(-1,1),S_VPRM_clean0[indx])
        H_m=Huber_fit.coef_
        H_c=Huber_fit.intercept_
        x_accpt, y_accpt = SIF_EVI_clean0, S_VPRM_clean0
        y_predict = H_m * x_accpt + H_c
        H_R2=r2_score(y_accpt, y_predict)
        Huber_slps.append(H_m)
        Huber_ints.append(H_c)
        Huber_R2s.append(H_R2)
    except ValueError: #if Huber fit can't find a solution for the subset, skip it
        pass
    
print('Slope = '+str(np.round(np.nanmean(Huber_slps),5))+' +/- '+str(np.round(np.nanstd(Huber_slps),5))+', intercept = '+str(np.round(np.nanmean(Huber_ints),5))+' +/- '+str(np.round(np.nanstd(Huber_ints),5)))
y_predict = np.nanmean(Huber_slps) * x_accpt + np.nanmean(Huber_ints)
Huber_R2=r2_score(y_accpt, y_predict)
print('R^2 = '+str(np.round(Huber_R2,5)))


# In[39]:


# *** Optional: Plot the SMUrF-VPRM NEE difference

plt.rc('font',size=26)

plt.figure(figsize=(10,5))
plt.xlim(-79.63,-79.13)
plt.ylim(43.55,43.87)
plt.axis('scaled')
plt.pcolormesh(xvals-1/240/2,yvals+1/240/2,S_NEE_flipped[23]-VPRM_NEE_8day[23],cmap='bwr',vmin=-8,vmax=8)
plt.plot(Toronto_x,Toronto_y,c='k')
plt.title('$\Delta$NEE ($\mu$mol m$^{-2}$ s$^{-1}$)',fontsize=28.5)
cbar=plt.colorbar()
#cbar.set_label('$\Delta$NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
plt.xlabel('Longitude ($^\circ$)')
plt.ylabel('Latitude ($^\circ$)')
# *** Uncomment to save figure CHANGE FILENAME ***
#plt.savefig('Fixed_DNEE_Toronto_larger_font_2.pdf',bbox_inches='tight')
#plt.savefig('Fixed_DNEE_Toronto_larger_font_2.png',bbox_inches='tight')
plt.show()

#Plot the SIF-EVI difference
plt.figure(figsize=(9.5,5))
plt.xlim(-79.63,-79.13)
plt.ylim(43.55,43.87)
plt.axis('scaled')
plt.pcolormesh(xvals-1/240/2,yvals+1/240/2,(EVI_data_normalized[23]-SIF_data_norm_flipped[23])*GPP_mask,cmap='bwr', vmin=-1,vmax=1)
plt.plot(Toronto_x,Toronto_y,c='k')

plt.title('Normalized EVI - Normalized SIF',fontsize=28.5)
cbar=plt.colorbar()
ax = plt.gca()
ax.axes.yaxis.set_ticklabels([])
#cbar.set_label('EVI - SIF')
plt.xlabel('Longitude ($^\circ$)')
plt.ylabel('Latitude ($^\circ$)')
# *** Uncomment to save figure CHANGE FILENAME ***
#plt.savefig('Normalized_EVI_Fixed_SIF_Toronto_larger_font_2.pdf',bbox_inches='tight')
#plt.savefig('Normalized_EVI_Fixed_SIF_Toronto_larger_font_2.png',bbox_inches='tight')
plt.show()

#Plot the correlation between normalized EVI-SIF & SMUrF-VPRM NEE
plt.figure(figsize=(6.1,5))
plt.xlim(-1,1)
plt.ylim(-9,9)
plt.scatter(SIF_EVI_clean0,S_VPRM_clean0,c='g',s=5)
plt.title('$\Delta$NEE vs EVI-SIF', fontsize=28.5)
plt.plot(line1_1,func2(line1_1,np.nanmean(Huber_slps),np.nanmean(Huber_ints)),c='k',linestyle='--',label=str(np.round(np.nanmean(Huber_slps),2))+'$\cdot$x+'+str(np.round(np.nanmean(Huber_ints),2))+', R$^2$ = '+str(np.round(Huber_R2,2)))

plt.axhline(0,linestyle=':',c='k')
plt.axvline(0,linestyle=':',c='k')
plt.legend(loc='lower center',fontsize=22)
plt.ylabel('$\Delta$NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
plt.xlabel('Normalized EVI - SIF')
# *** Uncomment to save figure CHANGE FILENAME ***
#plt.savefig('Fixed_DNEE_vs_EVI_SIF_Huber_fit_larger_font.pdf',bbox_inches='tight')
#plt.savefig('Fixed_DNEE_vs_EVI_SIF_Huber_fit_larger_font.png',bbox_inches='tight')
plt.show()

# *** End of optional


# In[ ]:





# In[40]:


# *** Optional: Investigate the outlier points that lie above the line of best fit (they are from the Rouge national park!)

plt.figure(figsize=(6.1,5))
plt.xlim(-1,1)
plt.ylim(-9,9)
plt.scatter(SIF_EVI,S_VPRM,c='g',s=5)
plt.scatter(SIF_EVI[S_VPRM>(SIF_EVI*np.nanmean(Huber_slps)+np.nanmean(Huber_ints)+1.5)],S_VPRM[S_VPRM>(SIF_EVI*np.nanmean(Huber_slps)+np.nanmean(Huber_ints)+1.5)],c='r',s=5)

plt.title('$\Delta$NEE vs EVI-SIF', fontsize=28.5)
plt.plot(line1_1,func2(line1_1,np.nanmean(Huber_slps),np.nanmean(Huber_ints)),c='k',linestyle='--',label=str(np.round(np.nanmean(Huber_slps),2))+'$\cdot$x+'+str(np.round(np.nanmean(Huber_ints),2))+', R$^2$ = '+str(np.round(Huber_R2,2)))
plt.plot(line1_1,func2(line1_1,np.nanmean(Huber_slps),np.nanmean(Huber_ints))+1.5,c='k',linestyle=':')
plt.plot(line1_1,func2(line1_1,np.nanmean(Huber_slps),np.nanmean(Huber_ints))-1.5,c='k',linestyle=':')

plt.axhline(0,linestyle=':',c='k')
plt.axvline(0,linestyle=':',c='k')
plt.legend(loc='lower center',fontsize=22)
plt.ylabel('$\Delta$NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
plt.xlabel('Normalized EVI - SIF')
plt.show()

# Outlier values are located over Rouge park

plt.rc('font',size=26)

plt.figure(figsize=(10,5))
plt.xlim(-79.63,-79.13)
plt.ylim(43.55,43.87)
plt.axis('scaled')
plt.pcolormesh(xvals-1/240/2,yvals+1/240/2,(S_NEE_flipped[23]-VPRM_NEE_8day[23])>(SIF_EVI*np.nanmean(Huber_slps)+np.nanmean(Huber_ints)+1.5),cmap='bwr',vmin=-1,vmax=1)
plt.plot(Toronto_x,Toronto_y,c='k')
plt.title('$\Delta$NEE ($\mu$mol m$^{-2}$ s$^{-1}$)',fontsize=28.5)
cbar=plt.colorbar()
#cbar.set_label('$\Delta$NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
plt.xlabel('Longitude ($^\circ$)')
plt.ylabel('Latitude ($^\circ$)')
# *** Uncomment to save figure CHANGE FILENAME ***
#plt.savefig('Fixed_DNEE_Toronto_larger_font_high_outliers.pdf',bbox_inches='tight')
#plt.savefig('Fixed_DNEE_Toronto_larger_font_high_outliers.png',bbox_inches='tight')
plt.show()


# In[41]:


#Investigate the outlier points that lie above the line of best fit (they are from the Rouge national park!)

plt.figure(figsize=(6.1,5))
plt.xlim(-1,1)
plt.ylim(-9,9)
plt.scatter(SIF_EVI,S_VPRM,c='g',s=5)
plt.scatter(SIF_EVI[S_VPRM<(SIF_EVI*np.nanmean(Huber_slps)+np.nanmean(Huber_ints)-1.5)],S_VPRM[S_VPRM<(SIF_EVI*np.nanmean(Huber_slps)+np.nanmean(Huber_ints)-1.5)],c='r',s=5)

plt.title('$\Delta$NEE vs EVI-SIF', fontsize=28.5)
plt.plot(line1_1,func2(line1_1,np.nanmean(Huber_slps),np.nanmean(Huber_ints)),c='k',linestyle='--',label=str(np.round(np.nanmean(Huber_slps),2))+'$\cdot$x+'+str(np.round(np.nanmean(Huber_ints),2))+', R$^2$ = '+str(np.round(Huber_R2,2)))
plt.plot(line1_1,func2(line1_1,np.nanmean(Huber_slps),np.nanmean(Huber_ints))+1.5,c='k',linestyle=':')
plt.plot(line1_1,func2(line1_1,np.nanmean(Huber_slps),np.nanmean(Huber_ints))-1.5,c='k',linestyle=':')

plt.axhline(0,linestyle=':',c='k')
plt.axvline(0,linestyle=':',c='k')
plt.legend(loc='lower center',fontsize=22)
plt.ylabel('$\Delta$NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
plt.xlabel('Normalized EVI - SIF')
plt.show()

# Outlier values are located over Rouge park

plt.rc('font',size=26)

plt.figure(figsize=(10,5))
plt.xlim(-79.63,-79.13)
plt.ylim(43.55,43.87)
plt.axis('scaled')
plt.pcolormesh(xvals-1/240/2,yvals+1/240/2,(S_NEE_flipped[23]-VPRM_NEE_8day[23])<(SIF_EVI*np.nanmean(Huber_slps)+np.nanmean(Huber_ints)-1.5),cmap='bwr',vmin=-1,vmax=1)
plt.plot(Toronto_x,Toronto_y,c='k')
plt.title('$\Delta$NEE ($\mu$mol m$^{-2}$ s$^{-1}$)',fontsize=28.5)
cbar=plt.colorbar()
#cbar.set_label('$\Delta$NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
plt.xlabel('Longitude ($^\circ$)')
plt.ylabel('Latitude ($^\circ$)')
# *** Uncomment to save figure CHANGE FILENAME ***
#plt.savefig('Fixed_DNEE_Toronto_larger_font_low_outliers.pdf',bbox_inches='tight')
#plt.savefig('Fixed_DNEE_Toronto_larger_font_low_outliers.png',bbox_inches='tight')
plt.show()

# *** End of optional


# In[ ]:





# In[71]:


plt.rc('font',size=22)

fig, ax = plt.subplots(1,2,figsize=(18,5),gridspec_kw={'width_ratios': [3,2]})
ax[0].set_xlim(-79.69,-79.06)
ax[0].set_ylim(43.5,43.9)
ax[0].axis('scaled')


ax[0].set_xlim(-79.69,-79.06)
ax[0].set_ylim(43.5,43.9)
ax[0].axis('scaled')
#ax[0].set_yticklabels([])
fig1=ax[0].pcolormesh(xvals-1/240/2,yvals+1/240/2,(EVI_data_normalized[23]-SIF_data_norm_flipped[23])*GPP_mask,cmap='bwr', vmin=-1,vmax=1)
ax[0].plot(Toronto_x,Toronto_y,c='k')

ax[0].set_title('Normalized EVI - Normalized SIF')
cbar1=plt.colorbar(fig1,ax=ax[0])

ax[1].set_xlim(-1,1)
ax[1].set_ylim(-8,8)
ax[1].scatter(SIF_EVI_clean0,S_VPRM_clean0,c='g',s=5)
ax[1].set_title('$\Delta$NEE vs EVI-SIF')
ax[1].plot(line1_1,func2(line1_1,np.nanmean(Huber_slps),np.nanmean(Huber_ints)),c='k',linestyle='--',label=str(np.round(np.nanmean(Huber_slps),2))+'$\cdot$x+'+str(np.round(np.nanmean(Huber_ints),2))+', R$^2$ = '+str(np.round(Huber_R2,2)))
ax[1].axhline(0,linestyle=':',c='k')
ax[1].axvline(0,linestyle=':',c='k')
ax[1].legend(loc='lower center')
ax[1].set_ylabel('$\Delta$NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
ax[1].set_xlabel('Normalized EVI - SIF')

ax[0].set_ylabel('Latitude ($^o$)')
ax[0].set_xlabel('Longitude ($^o$)')
ax[0].text(-79.712,43.883,'(a)',c='k')
ax[1].text(-0.965,6.55,'(b)',c='k')
#x[1].set_xlabel('Longitude ($^o$)')
# *** Uncomment to save figure CHANGE FILENAME ***
plt.savefig('Updated_SMUrF_UrbanVPRM_NEE_vs_EVI_SIF_Huber_fit_larger_font_labelled.pdf',bbox_inches='tight')
plt.savefig('Updated_SMUrF_UrbanVPRM_NEE_vs_EVI_SIF_Huber_fit_larger_font_labelled.png',bbox_inches='tight')
fig.show()


# In[45]:


# *** Optional, Uncomment to also plot NEE:

#plt.rc('font',size=22)

#fig, ax = plt.subplots(1,3,figsize=(25.5,4.3),gridspec_kw={'width_ratios': [3,2.9,2.2]})
#ax[0].set_xlim(-79.69,-79.06)
#ax[0].set_ylim(43.5,43.9)
#ax[0].axis('scaled')

#fig0=ax[0].pcolormesh(xvals-1/240/2,yvals+1/240/2,S_NEE_flipped[23]-VPRM_NEE_8day[23],cmap='bwr',vmin=-8,vmax=8)
#ax[0].plot(Toronto_x,Toronto_y,c='k')
#ax[0].set_title('$\Delta$NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
#cbar0=plt.colorbar(fig0,ax=ax[0])

#ax[1].set_xlim(-79.69,-79.06)
#ax[1].set_ylim(43.5,43.9)
#ax[1].axis('scaled')
#ax[1].set_yticklabels([])
#fig1=ax[1].pcolormesh(xvals-1/240/2,yvals+1/240/2,(EVI_data_normalized[23]-SIF_data_norm_flipped[23])*GPP_mask,cmap='bwr', vmin=-1,vmax=1)
#ax[1].plot(Toronto_x,Toronto_y,c='k')

#ax[1].set_title('Normalized EVI - Normalized SIF')
#cbar1=plt.colorbar(fig1,ax=ax[1])

#ax[2].set_xlim(-1,1)
#ax[2].set_ylim(-8,8)
#ax[2].scatter(SIF_EVI_clean0,S_VPRM_clean0,c='g',s=5)
#ax[2].set_title('$\Delta$NEE vs EVI-SIF')
#ax[2].plot(line1_1,func2(line1_1,np.nanmean(Huber_slps),np.nanmean(Huber_ints)),c='k',linestyle='--',label=str(np.round(np.nanmean(Huber_slps),2))+'$\cdot$x+'+str(np.round(np.nanmean(Huber_ints),2))+', R$^2$ = '+str(np.round(Huber_R2,2)))
#ax[2].axhline(0,linestyle=':',c='k')
#ax[2].axvline(0,linestyle=':',c='k')
#ax[2].legend(loc='lower center')
#ax[2].set_ylabel('$\Delta$NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
#ax[2].set_xlabel('Normalized EVI - SIF')

#ax[0].set_ylabel('Latitude ($^o$)')
#ax[0].set_xlabel('Longitude ($^o$)')
#ax[1].set_xlabel('Longitude ($^o$)')
## *** Uncomment to save figure CHANGE FILENAME ***
##plt.savefig('Updated_SMUrF_UrbanVPRM_NEE_vs_EVI_SIF_Huber_fit_larger_font.pdf',bbox_inches='tight')
##plt.savefig('Updated_SMUrF_UrbanVPRM_NEE_vs_EVI_SIF_Huber_fit_larger_font.png',bbox_inches='tight')
#fig.show()


# In[ ]:





# In[87]:


#Apply a 1000 times bootstrapped Huber fit EVI-SIF vs. DNEE during the entire growing season (April-November)

#With downscaling fix

SIF_EVI=EVI_data_normalized[11:42]-SIF_data_norm_flipped[11:42]
S_VPRM=S_NEE_flipped[11:42]-VPRM_NEE_8day[11:42]

finitemask0=np.isfinite(SIF_EVI) & np.isfinite(S_VPRM) & (S_VPRM!=0)
SIF_EVI_clean0=SIF_EVI[finitemask0]
S_VPRM_clean0=S_VPRM[finitemask0]

Huber_slps=[]
Huber_ints=[]
Huber_R2s=[]

#try bootstrapping
indx_list=list(range(0,len(S_VPRM_clean0)))
for i in range(1,1001):
    #sub selection of points
    indx=np.random.choice(indx_list,size=50000)
    
    try:
        Huber_model = linear_model.HuberRegressor(fit_intercept=True)
        Huber_fit=Huber_model.fit((SIF_EVI_clean0[indx]).reshape(-1,1),S_VPRM_clean0[indx])
        H_m=Huber_fit.coef_
        H_c=Huber_fit.intercept_
        x_accpt, y_accpt = SIF_EVI_clean0, S_VPRM_clean0
        y_predict = H_m * x_accpt + H_c
        H_R2=r2_score(y_accpt, y_predict)
        Huber_slps.append(H_m)
        Huber_ints.append(H_c)
        Huber_R2s.append(H_R2)
    except ValueError: #if Huber fit can't find a solution for the subset, skip it
        pass
    
print('Growing Season Slope = '+str(np.round(np.nanmean(Huber_slps),5))+' +/- '+str(np.round(np.nanstd(Huber_slps),5))+', intercept = '+str(np.round(np.nanmean(Huber_ints),5))+' +/- '+str(np.round(np.nanstd(Huber_ints),5)))
y_predict = np.nanmean(Huber_slps) * x_accpt + np.nanmean(Huber_ints)
Huber_R2=r2_score(y_accpt, y_predict)
print('R^2 = '+str(np.round(Huber_R2,5)))


# In[ ]:





# In[130]:


#Uncomment to only plot Growing Season dNEE vs EVI-SIF

#plt.rc('font',size=18)
#plt.figure(figsize=(7,5))
#plt.xlim(-1,1)
#plt.ylim(-9.5,9.5)
#plt.scatter(SIF_EVI_clean0,S_VPRM_clean0,c='g',s=5)
#plt.title('Toronto Growing Season $\Delta$NEE vs EVI-SIF')
#plt.plot(line1_1,func2(line1_1,np.nanmean(Huber_slps),np.nanmean(Huber_ints)),c='k',linestyle='--',label=str(np.round(np.nanmean(Huber_slps),2))+'$\cdot$x+'+str(np.round(np.nanmean(Huber_ints),2))+', R$^2$ = '+str(np.round(Huber_R2,2)))
#plt.axhline(0,linestyle=':',c='k')
#plt.axvline(0,linestyle=':',c='k')
#plt.legend()
#plt.ylabel('$\Delta$NEE ($\mu$mol/m$^2$/s)')
#plt.xlabel('Normalized EVI - SIF')
##plt.savefig('Fixed_SMUrF_VPRM_vs_EVI_SIF_Huber_fit_growing_season.pdf',bbox_inches='tight')
##plt.savefig('Fixed_SMUrF_VPRM_vs_EVI_SIF_Huber_fit_growing_season.png',bbox_inches='tight')
#plt.show()

# End of uncomment


# In[ ]:





# In[74]:


#With downscaling & MODIS shift fixes

#Apply a 1000 times bootstrapped Huber fit to summer (June-August) EVI-SIF vs. DNEE data

SIF_EVI_JJA=EVI_data_normalized[19:30]-SIF_data_norm_flipped[19:30]
S_VPRM_JJA=S_NEE_flipped[19:30]-VPRM_NEE_8day[19:30]

finitemask0=np.isfinite(SIF_EVI_JJA) & np.isfinite(S_VPRM_JJA) & (S_VPRM_JJA!=0)
SIF_EVI_JJA_clean0=SIF_EVI_JJA[finitemask0]
S_VPRM_JJA_clean0=S_VPRM_JJA[finitemask0]

Huber_JJA_slps=[]
Huber_JJA_ints=[]
Huber_JJA_R2s=[]

#try bootstrapping
indx_list=list(range(0,len(S_VPRM_JJA_clean0)))
for i in range(1,1000):
    #sub selection of points
    indx=np.random.choice(indx_list,size=50000)
    
    try:
        Huber_JJA_model = linear_model.HuberRegressor(fit_intercept=True)
        Huber_JJA_fit=Huber_JJA_model.fit((SIF_EVI_JJA_clean0[indx]).reshape(-1,1),S_VPRM_JJA_clean0[indx])
        H_m=Huber_JJA_fit.coef_
        H_c=Huber_JJA_fit.intercept_
        x_accpt, y_accpt = SIF_EVI_JJA_clean0, S_VPRM_JJA_clean0
        y_predict = H_m * x_accpt + H_c
        H_R2=r2_score(y_accpt, y_predict)
        Huber_JJA_slps.append(H_m)
        Huber_JJA_ints.append(H_c)
        Huber_JJA_R2s.append(H_R2)
    except ValueError: #if Huber fit can't find a solution for the subset, skip it
        pass

print('Summer Slope = '+str(np.round(np.nanmean(Huber_JJA_slps),5))+' +/- '+str(np.round(np.nanstd(Huber_JJA_slps),5))+', intercept = '+str(np.round(np.nanmean(Huber_JJA_ints),5))+' +/- '+str(np.round(np.nanstd(Huber_JJA_ints),5)))
y_predict = np.nanmean(Huber_JJA_slps) * x_accpt + np.nanmean(Huber_JJA_ints)
Huber_JJA_R2=r2_score(y_accpt, y_predict)
print('R^2 = '+str(np.round(Huber_JJA_R2,5)))


# In[ ]:





# In[88]:


# Figures S6
#plot the summer and growing season comparison between DNEE and EVI-SIF
plt.rc('font',size=20)

fig, ax = plt.subplots(1,2,sharex=True, sharey=True,figsize=(14,5))

ax[0].set_xlim(-0.95,0.95)
ax[0].set_ylim(-9.5,9.5)

ax[0].scatter(SIF_EVI_JJA_clean0,S_VPRM_JJA_clean0,c='g',s=5)
ax[0].set_title('Toronto Summer $\Delta$NEE vs EVI-SIF', fontsize=20)
ax[0].plot(line1_1,func2(line1_1,np.nanmean(Huber_JJA_slps),np.nanmean(Huber_JJA_ints)),c='k',linestyle='--',label=str(np.round(np.nanmean(Huber_JJA_slps),2))+'$\cdot$x+'+str(np.round(np.nanmean(Huber_JJA_ints),2))+', R$^2$ = '+str(np.round(Huber_JJA_R2,2)))
ax[0].axhline(0,linestyle=':',c='k')
ax[0].axvline(0,linestyle=':',c='k')
ax[0].legend()
ax[0].set_ylabel('$\Delta$NEE ($\mu$mol/m$^2$/s)')
ax[0].set_xlabel('Normalized EVI - SIF')

ax[1].scatter(SIF_EVI_clean0,S_VPRM_clean0,c='g',s=5)
ax[1].set_title('Toronto Growing Season $\Delta$NEE vs EVI-SIF',fontsize=20)
ax[1].plot(line1_1,func2(line1_1,np.nanmean(Huber_slps),np.nanmean(Huber_ints)),c='k',linestyle='--',label=str(np.round(np.nanmean(Huber_slps),2))+'$\cdot$x-'+str(np.round(-np.nanmean(Huber_ints),2))+', R$^2$ = '+str(np.round(Huber_R2,2)))
ax[1].axhline(0,linestyle=':',c='k')
ax[1].axvline(0,linestyle=':',c='k')
ax[1].legend()

ax[1].set_xlabel('Normalized EVI - SIF')
ax[0].text(-0.93,8.05,'(a)',c='k')
ax[1].text(-0.93,8.05,'(b)',c='k')

fig.subplots_adjust(wspace=0)
#*** Uncomment to save figure as pdf & png. CHANGE FILENAME ***
#fig.savefig('Fixed_SMUrF_VPRM_vs_EVI_SIF_Huber_fit_JJA_growing_season_labelled.pdf',bbox_inches='tight')
#fig.savefig('Fixed_SMUrF_VPRM_vs_EVI_SIF_Huber_fit_JJA_growing_season_labelled.png',bbox_inches='tight')
fig.show()


# In[ ]:




