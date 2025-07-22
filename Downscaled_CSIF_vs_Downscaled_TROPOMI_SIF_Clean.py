#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This code is used to directly compare the Contiguous Solar-Induced Fluorescence (CSIF) product, downscaled using the
# Near Infrared Reflectance Vegetation index (NIRv) from MODIS, to downscaled SIF from the TROPOMI instrument, using a
# bootstrapped Huber fit. We see a strong correlation between the two products.

# This code creates figure S1 of Madsen-Colford et al. 2025
# If used, please cite

# Comments labelled with *** are areas that should be changed by the user (e.g. change directory paths, etc.)


# In[1]:


#Import the required packages
import numpy as np #numerical python
import matplotlib.pyplot as plt #for plotting
from matplotlib.cm import get_cmap #import colour maps for contour plots
import netCDF4
from netCDF4 import Dataset, date2num #for reading netCDF data files and their date (not sure if I need the later)
import time #for timing how long a computation takes
import shapefile as shp # to import outline of GTA
from shapely import geometry # used to define boundaries of TROPOMI pixels
import glob
from pyhdf.SD import SD, SDC #Used for reading MODIS hdf files to a format usable by Python
from scipy import optimize as opt 
from scipy import odr
from sklearn import linear_model #for robust fitting
from sklearn.metrics import r2_score, mean_squared_error #for analyzing robust fits
import matplotlib.colors as clrs #for log color scale


# In[10]:


#Load in TROPOMI SIF data over the for the GTA:

# *** CHANGE PATH & FILENAME ***
sif_path = '/export/data2/downscaled_SIF/downscaled_TROPOSIF/2018/'
sif_fn = 'downscaled_sif_V061_filtered_fixed_os_ds_2018_8d_buff_' #filename WITHOUT day of year

TROPO_sif_err=np.zeros([46,553,625])*np.nan
TROPO_sif_data=np.zeros([46,553,625])*np.nan
TROPO_sif_date=np.zeros([46])*np.nan
time=4
for i in range(1,365,8):
    try:
        if i<10:
            f=Dataset(sif_path+sif_fn+'00'+str(i)+'.nc')
        elif i<100:
            f=Dataset(sif_path+sif_fn+'0'+str(i)+'.nc')
        else:
            f=Dataset(sif_path+sif_fn+str(i)+'.nc')
        TROPO_sif_data[np.int((i-1)/8)]=f.variables['daily_sif'][:]
        TROPO_sif_err[np.int((i-1)/8)]=f.variables['Errors'][:]
        TROPO_sif_date[np.int((i-1)/8)]=time 
        f.close()
    except FileNotFoundError and OSError:
        pass
    time+=8


# In[12]:


f=Dataset(sif_path+sif_fn+'225.nc')
lons=f.variables['lon'][:]
lats=f.variables['lat'][:]
f.close()


# In[ ]:





# In[22]:


# Load in 2018's downscaled CSIF data for the GTA:

# *** CHANGE PATH & FILENAME ***
CSIF_path = '/export/data2/downscaled_SIF/downscaled_CSIF/2018/'
CSIF_fn = 'downscaled_CSIF_V061_fixed_os_ds_2018_8d_buff_'

CSIF_data=np.zeros([46,len(lats),len(lons)])*np.nan
CSIF_date=np.zeros([46])*np.nan
time=4
for i in range(1,367,8):
    try:
        if i<10:
            # *** Change Path ***
            f=Dataset(CSIF_path+CSIF_fn+'00'+str(i)+'.nc')
        elif i<100:
            # *** Change Path ***
            f=Dataset(CSIF_path+CSIF_fn+'0'+str(i)+'.nc')
        else:
            # *** Change Path ***
            f=Dataset(CSIF_path+CSIF_fn+str(i)+'.nc')
        CSIF_data[np.int((i-1)/8)]=f.variables['daily_sif'][:]
        CSIF_date[np.int((i-1)/8)]=time 
        f.close()
    except FileNotFoundError and OSError:
        pass
    time+=8
    
# load in lats & lons from one of the files
f=Dataset(CSIF_path+CSIF_fn+'121.nc')
CSIF_lons=f.variables['lon'][:]
CSIF_lats=f.variables['lat'][:]
f.close()


# In[40]:


#Remove erroneous downscaled TROPOMI SIF data
with np.errstate(invalid='ignore'):
    TROPO_sif_data[TROPO_sif_data>100]=np.nan
    TROPO_sif_err[TROPO_sif_err==0]=np.nan


# In[ ]:





# In[56]:


#Load in landcover data

# *** CHANGE PATH & FILENAME ***
mod_land_data=Dataset('/export/data/analysis/tropomi/sif/downscaled/2018/MODIS_Land_Type.nc')
Land_Data=mod_land_data.variables['Land_Type'][:]
Land_Edge=mod_land_data.variables['Edge_Land_Type'][:]
mod_land_data.close()


# In[76]:


# Seperate data into different land cover types

CSIF_Rural=np.ones(np.shape(CSIF_data))*np.nan
CSIF_Crops=np.ones(np.shape(CSIF_data))*np.nan
CSIF_Urban=np.ones(np.shape(CSIF_data))*np.nan
for i in range(len(TROPO_sif_date)):
    CSIF_Rural[i][::-1][Land_Data.T<12]=CSIF_data[i][::-1][Land_Data.T<12]
    CSIF_Crops[i][::-1][(Land_Data.T==12) | (Land_Data.T==14)]=CSIF_data[i][::-1][(Land_Data.T==12) | (Land_Data.T==14)]
    CSIF_Urban[i][::-1][Land_Data.T==13]=CSIF_data[i][::-1][Land_Data.T==13]


# In[61]:


# Define a straight line & linear function for fitting & plotting

line1_1=np.arange(-5,5)

def func2(x,m,b):
    return m*x+b


# In[62]:


date_array=np.ones(np.shape(TROPO_sif_data))
for i in range(len(CSIF_date)):
    date_array[i]=CSIF_date[i]*date_array[i]


# In[63]:


# CSIF has been shown to understimate SIF in urban areas by 14.5%, 
# Correct for this by multiplying by 1.145
Adjusted_CSIF=np.copy(CSIF_data)
Adjusted_CSIF[i][::-1][Land_Data.T==13]=CSIF_data[i][::-1][Land_Data.T==13]*1.145


# In[ ]:





# In[67]:


# Apply a 1000 times bootstrapped Huber fit between downscaled TROPOMI SIF
# and downscaled CSIF data

# remove non-finite data
finitemask1 = np.isfinite(TROPO_sif_data)
TROPO_clean0 = TROPO_sif_data[finitemask1]
TROPO_err_clean0 = TROPO_sif_err[finitemask1]
Adjusted_CSIF_clean0 = Adjusted_CSIF[finitemask1]
dates_clean0=date_array[finitemask1]

finitemask2 = np.isfinite(Adjusted_CSIF_clean0)
TROPO_clean = TROPO_clean0[finitemask2]
TROPO_err_clean = TROPO_err_clean0[finitemask2]
Adjusted_CSIF_clean = Adjusted_CSIF_clean0[finitemask2]
dates_clean = dates_clean0[finitemask2]

Huber_slps=[]
Huber_ints=[]
Huber_R2s=[]

#try bootstrapping
indx_list=list(range(0,len(Adjusted_CSIF_clean)))
for i in range(1,1001):
    #sub selection of points
    indx=np.random.choice(indx_list,size=50000)
    
    try:
        Huber_model = linear_model.HuberRegressor(fit_intercept=True)
        Huber_fit=Huber_model.fit((TROPO_clean[indx]).reshape(-1,1),Adjusted_CSIF_clean[indx])
        H_m=Huber_fit.coef_
        H_c=Huber_fit.intercept_
        x_accpt, y_accpt = TROPO_clean, Adjusted_CSIF_clean
        y_predict = H_m * x_accpt + H_c
        H_R2=r2_score(y_accpt, y_predict)
        Huber_slps.append(H_m)
        Huber_ints.append(H_c)
        Huber_R2s.append(H_R2)
    except ValueError: #if Huber fit can't find a solution for the subset, skip it
        pass
    
y_predict = np.nanmean(Huber_slps) * x_accpt + np.nanmean(Huber_ints)
Huber_R2=r2_score(y_accpt, y_predict)
print('Huber fit slope = '+str(np.round(np.nanmean(Huber_slps),4))+' +/- '+str(np.round(np.nanstd(Huber_slps),4)))
print('Huber fit intercept = '+str(np.round(np.nanmean(Huber_ints),4))+' +/- '+str(np.round(np.nanstd(Huber_ints),4)))
print('Huber fit R2 = '+str(np.round(Huber_R2,4)))


# In[113]:


# Calculate the R2 of 1 to 1 line
R2_1_1=r2_score(Adjusted_CSIF_clean,TROPO_clean)
print('1 to 1 line R2 = '+str(np.round(R2_1_1,3)))


# In[ ]:





# In[79]:


# Calculate the number of finite (not NaN) datapoints over land
N_data=len(TROPO_sif_data[(np.isnan(TROPO_sif_data)==False) & (np.isnan(Adjusted_CSIF)==False) & (Land_Data.T<15)])


# In[81]:


#Remove NaN data from data from cropland CSIF (i.e. remove datapoints not over croplands)
CSIF_Crops_narm=CSIF_Crops[np.isnan(CSIF_Crops)==False]
#Create an array of indices of data over croplands
indx_list_Crops=list(range(0,len(CSIF_Crops_narm)))
#make the number of samples proportional to the fraction of each land cover type
CSIF_Crops_indx=np.random.choice(indx_list_Crops,size=int(50000*len(indx_list_Crops)/N_data))


# In[86]:


#Do the same for rural and Urban

CSIF_Rural_narm=CSIF_Rural[np.isnan(CSIF_Rural)==False]
indx_list_Rural=list(range(0,len(CSIF_Rural_narm)))
#make the number of samples proportional to the fraction of each land cover type
CSIF_Rural_indx=np.random.choice(indx_list_Rural,size=int(50000*len(indx_list_Rural)/N_data))


# In[87]:


CSIF_Urban_narm=CSIF_Urban[np.isnan(CSIF_Urban)==False]
indx_list_Urban=list(range(0,len(CSIF_Urban_narm)))
#make the number of samples proportional to the fraction of each land cover type
CSIF_Urban_indx=np.random.choice(indx_list_Urban,size=int(50000*len(indx_list_Urban)/N_data))


# In[131]:


plt.style.use('tableau-colorblind10')
plt.rc('font',size=16)

plt.figure(figsize=(7.75,7.75))
plt.xlim(-0.75,2)
plt.ylim(-0.75,2)
plt.axis('scaled')
plt.scatter(TROPO_sif_data[np.isnan(CSIF_Crops)==False][CSIF_Crops_indx],CSIF_Crops_narm[CSIF_Crops_indx],s=2)
plt.scatter([-10],[-10],s=20,c='#006BA4',label='Croplands')
plt.scatter(TROPO_sif_data[np.isnan(CSIF_Rural)==False][CSIF_Rural_indx],CSIF_Rural_narm[CSIF_Rural_indx],s=2)
plt.scatter([-10],[-10],s=20,c='#FF800E',label='Rural')
plt.scatter(TROPO_sif_data[np.isnan(CSIF_Urban)==False][CSIF_Urban_indx],CSIF_Urban_narm[CSIF_Urban_indx],s=2)
plt.scatter([-10],[-10],s=20,c='#ABABAB',label='Urban')
plt.axvline(0,c='k',linestyle=':')
plt.axhline(0,c='k',linestyle=':')
plt.plot(line1_1,func2(line1_1,np.nanmean(Huber_slps),np.nanmean(Huber_ints)),c='#595959',linestyle='--',label=str(np.round(np.nanmean(Huber_slps),2))+'$\cdot$x+'+str(np.round(np.nanmean(Huber_ints),2))+', R$^2$ = '+str(np.round(Huber_R2,2)))
plt.plot(line1_1,func2(line1_1,1,0),linestyle='-.',c='k',label='1:1, R$^2$ = '+str(np.round(R2_1_1,2)))
plt.legend()
plt.title('Downscaled CSIF vs. TROPOMI SIF, 2018')
plt.xlabel('Downscaled TROPOMI SIF (mW m$^{-2}$ sr$^{-1}$ nm$^{-1}$)')
plt.ylabel('Downscaled CSIF (mW m$^{-2}$ sr$^{-1}$ nm$^{-1}$)')
# *** Uncomment to save the figure as png and pdf. CHANGE FILENAMEs ***
#plt.savefig('Fixed_Downscaled_CSIF_vs_TROPOMI_SIF_V061_2018_GTA_Huber_fit_bootstraped_plot_less_data.pdf',bbox_inches='tight')
#plt.savefig('Fixed_Downscaled_CSIF_vs_TROPOMI_SIF_V061_2018_GTA_Huber_fit_bootstraped_plot_less_data.png',bbox_inches='tight')
plt.show()


# In[ ]:




