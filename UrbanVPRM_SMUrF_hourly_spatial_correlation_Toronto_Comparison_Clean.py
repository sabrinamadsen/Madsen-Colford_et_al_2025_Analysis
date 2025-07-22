#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This code takes NEE, GPP and Reco from the updated UrbanVPRM and compares them to that of the updated SMUrF 
# both with and without the impervious surface area (ISA) adjustment over the City of Toronto. 
# For each hour of the year a bootstrapped Huber fit is applied between SMUrF and UrbanVPRM for all pixels that fall
# inside the bounds of the City of Toronto to calculate the correlation coefficient (R^2) for each hour of the year.
# The Root Mean Square Error (RMSE) and Mean Bias Error (MBE) are also calculated for each hour.
# The results are plotted as time series for each of NEE, GPP, and Reco.

# This code reproduces figure 4 of Madsen-Colford et al. 2025
# If used please cite

# *** denote portions of the code that should be modified by the user


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from scipy import optimize as opt 
from scipy import odr
import shapefile as shp # to import outline of GTA
from shapely import geometry # used to define a polygon for Toronto
import netCDF4
from netCDF4 import Dataset, date2num #for reading netCDF data files and their date (not sure if I need the later)
from sklearn import linear_model #for doing robust fits
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.colors as clrs #for log color scale


# In[3]:


#Load in VPRM data
# *** CHANGE PATH *** 
VPRM_path = 'E:/Research/UrbanVPRM/dataverse_files/GTA_V061_500m_2018/'
# *** CHANGE FILE NAME ***
VPRM_fn = 'vprm_GMIS_Toronto_ACI_SOLRIS_ISA_500m_GTA_V061_2018_no_PScale_adjusted_Topt_Ra_URB_parameters_fixed_gapfilled_LSWI_filtered_bilinear_PAR_block_'

VPRM_data = pd.read_csv(VPRM_path+VPRM_fn+'00000001.csv').loc[:,('HoY','Index','GEE','Re')]


# In[4]:


VPRM_data2=pd.read_csv(VPRM_path+VPRM_fn+'00002501.csv').loc[:,('HoY','Index','GEE','Re')]
VPRM_data=VPRM_data.append(VPRM_data2)
del VPRM_data2


# In[5]:


VPRM_data2=pd.read_csv(VPRM_path+VPRM_fn+'00005001.csv').loc[:,('HoY','Index','GEE','Re')]
VPRM_data=VPRM_data.append(VPRM_data2)
del VPRM_data2


# In[6]:


VPRM_data2=pd.read_csv(VPRM_path+VPRM_fn+'00007501.csv').loc[:,('HoY','Index','GEE','Re')]
VPRM_data=VPRM_data.append(VPRM_data2)
del VPRM_data2


# In[7]:


VPRM_data2=pd.read_csv(VPRM_path+VPRM_fn+'00010001.csv').loc[:,('HoY','Index','GEE','Re')]
VPRM_data=VPRM_data.append(VPRM_data2)
del VPRM_data2


# In[8]:


VPRM_data2=pd.read_csv(VPRM_path+VPRM_fn+'00012501.csv').loc[:,('HoY','Index','GEE','Re')]
VPRM_data=VPRM_data.append(VPRM_data2)
del VPRM_data2


# In[9]:


# *** CHANGE PATH & FILE NAME ***
VPRM_EVI=pd.read_csv('E:/Research/UrbanVPRM/dataverse_files/GTA_V061_500m_2018/adjusted_evi_lswi_interpolated_modis_v061_qc_filtered_LSWI_filtered.csv').loc[:,('DOY','Index','x','y')]

#Create a dataframe with just Index, x, & y values
x=np.zeros(np.shape(VPRM_EVI.Index.unique()))*np.nan
y=np.zeros(np.shape(VPRM_EVI.Index.unique()))*np.nan
for i in range(len(VPRM_EVI.Index.unique())):
    x[i]=VPRM_EVI.x[0+i*365]
    y[i]=VPRM_EVI.y[0+i*365]

# Make a dataframe of index, x, & y and merge that with the VPRM data based on the index
VPRM_xy=pd.DataFrame({'Index':VPRM_EVI.Index.unique(), 'x':x, 'y':y})
VPRM_data=VPRM_data.merge(VPRM_xy[['Index','x','y']])
del VPRM_EVI, VPRM_xy


# In[10]:


# Extract the unique x and y values & define an extent
xvals = VPRM_data.x[VPRM_data.HoY==4800].unique()
yvals = VPRM_data.y[VPRM_data.HoY==4800].unique()
extent = np.min(xvals), np.max(xvals), np.min(yvals), np.max(yvals)


# In[11]:


# Rearrange the GPP and Reco data into an array
GPP=-VPRM_data.GEE.values.reshape(len(yvals),len(xvals),8760)#8784 for leap year
Reco=VPRM_data.Re.values.reshape(len(yvals),len(xvals),8760)


# In[12]:


#Load in a shape file for Toronto, used to select data inside the city

# *** CHANGE PATH AND FILE NAME ***
sf = shp.Reader("C:/Users/kitty/Documents/Research/SIF/Shape_files/Toronto/Toronto_Boundary.shp")
#Toronto_Shape
shape=sf.shape(0)
#Need to partition each individual shape
Toronto_x = np.zeros((len(shape.points),1))*np.nan #make arrays of x & y values for the outline
Toronto_y = np.zeros((len(shape.points),1))*np.nan
for i in range(len(shape.points)):
    Toronto_x[i]=shape.points[i][0]
    Toronto_y[i]=shape.points[i][1]
    
points=[]
for k in range(1,len(Toronto_x)): #convert x & y data to points
    points.append(geometry.Point(Toronto_x[k],Toronto_y[k]))
poly=geometry.Polygon([[p.x, p.y] for p in points]) #convert points to a polygon

#Create a mask for areas outside the city of Toronto
lons=np.ones(144)*np.nan
lats=np.ones(96)*np.nan
GPP_mask=np.ones([96,144])*np.nan
for i in range(0, len(lons)):
    for j in range(0, len(lats)):
        if poly.contains(geometry.Point([xvals[i],yvals[j]])):
            lons[i]=xvals[i]
            lats[j]=yvals[j]
            GPP_mask[j,i]=1


# In[13]:


# Apply a mask so that only areas within the city of Toronto are given
T_VPRM_GPP=GPP*GPP_mask[:,:,np.newaxis]
T_VPRM_Reco=Reco*GPP_mask[:,:,np.newaxis]
T_VPRM_NEE=Reco-GPP


# In[16]:


#now bring in the SMUrF data with ISA adjustment, shoreline correction, AND downscaling fix

# Bring in the first Reco file and extract the first day of the year (in seconds since 1970)
# ***CHANGE PATH AND FILE NAME ***
g=Dataset('C:/Users/kitty/Documents/Research/SIF/SMUrF/output2018_500m_CSIF_to_TROPOMI_CSIF_ALL_converted_slps_V3_temp_impervious_R_shore_corr_V061_8day/easternCONUS/daily_mean_Reco_ISA_a_neuralnet/era5/2018/daily_mean_Reco_uncert_GMIS_Toronto_t_easternCONUS_20180101.nc')
start_of_year=g.variables['time'][0]/3600/24-1 #convert seconds since 1970 to days (minus one)
g.close()


#Load in SMUrF NEE data
# *** CHANGE PATH ***
SMUrF_path = 'C:/Users/kitty/Documents/Research/SIF/SMUrF/output2018_500m_CSIF_to_TROPOMI_CSIF_ALL_converted_slps_V3_temp_impervious_R_shore_corr_V061_8day/easternCONUS/hourly_flux_GMIS_combined_ISA_a_w_sd_era5/'
# *** CHANGE FILE NAME ***
SMUrF_fn = 'hrly_mean_GPP_Reco_NEE_easternCONUS_2018'

S_time=[]
S_Reco=[]
S_NEE=[]
S_GPP=[]
S_lats=[]
S_lons=[]
for j in range(1,13): # *** ADJUST THIS TO USE SPECIFIC MONTHS ***
    try:
        if j<10:
            f=Dataset(SMUrF_path+SMUrF_fn+'0'+str(j)+'.nc')
        else:
            f=Dataset(SMUrF_path+SMUrF_fn+str(j)+'.nc')
        if len(S_time)==0:
            S_lats=f.variables['lat'][:]
            S_lons=f.variables['lon'][:]
            S_Reco=f.variables['Reco_mean'][:,264:360,288:432]
            S_GPP=f.variables['GPP_mean'][:,264:360,288:432]
            S_NEE=f.variables['NEE_mean'][:,264:360,288:432]
            S_time=f.variables['time'][:]/24/3600-start_of_year-5/24 #convert seconds since 1970 to days and subtract start of year
        else:
            S_Reco=np.concatenate((S_Reco,f.variables['Reco_mean'][:,264:360,288:432]),axis=0)
            S_GPP=np.concatenate((S_GPP,f.variables['GPP_mean'][:,264:360,288:432]),axis=0)
            S_NEE=np.concatenate((S_NEE,f.variables['NEE_mean'][:,264:360,288:432]),axis=0)
            S_time=np.concatenate((S_time,(f.variables['time'][:]/24/3600-start_of_year-5/24)),axis=0)
        f.close()
    except FileNotFoundError:
        print(j)
        pass


# In[17]:


# Remove fill values and replace with NAN
S_Reco[S_Reco==-999]=np.nan
S_NEE[S_NEE==-999]=np.nan
S_GPP[S_GPP==-999]=np.nan

# Apply Toronto mask to SMUrF data
T_S_GPP=S_GPP[:,::-1]*GPP_mask[np.newaxis,:,:]
T_S_Reco=S_Reco[:,::-1]*GPP_mask[np.newaxis,:,:]
T_S_NEE=S_NEE[:,::-1]*GPP_mask[np.newaxis,:,:]


# In[ ]:





# In[19]:


#Now bring in SMUrF data without ISA correction

# *** CHANGE PATH ***
SMUrF_noISA_path = 'E:/Research/SMUrF/output2018_500m_CSIF_to_TROPOMI_CSIF_ALL_converted_slps_V3_temp_impervious_R_shore_corr_V061_8day/easternCONUS/hourly_flux_GMIS_Toronto_fixed_border_no_ISA_era5/'

S_noISA_time=[]
S_noISA_Reco=[]
S_noISA_NEE=[]
S_noISA_GPP=[]
S_noISA_lats=[]
S_noISA_lons=[]
for j in range(1,13): # *** ADJUST THIS TO USE SPECIFIC MONTHS ***
    try:
        if j<10:
            f=Dataset(SMUrF_noISA_path+SMUrF_fn+'0'+str(j)+'.nc')
        else:
            f=Dataset(SMUrF_noISA_path+SMUrF_fn+str(j)+'.nc')
        if len(S_noISA_time)==0:
            S_noISA_lats=f.variables['lat'][:]
            S_noISA_lons=f.variables['lon'][:]
            S_noISA_Reco=f.variables['Reco_mean'][:,264:360,288:432]
            S_noISA_GPP=f.variables['GPP_mean'][:,264:360,288:432]
            S_noISA_NEE=f.variables['NEE_mean'][:,264:360,288:432]
            S_noISA_time=f.variables['time'][:]/24/3600-start_of_year-5/24 #convert seconds since 1970 to days and subtract start of year
        else:
            S_noISA_Reco=np.concatenate((S_noISA_Reco,f.variables['Reco_mean'][:,264:360,288:432]),axis=0)
            S_noISA_GPP=np.concatenate((S_noISA_GPP,f.variables['GPP_mean'][:,264:360,288:432]),axis=0)
            S_noISA_NEE=np.concatenate((S_noISA_NEE,f.variables['NEE_mean'][:,264:360,288:432]),axis=0)
            S_noISA_time=np.concatenate((S_noISA_time,(f.variables['time'][:]/24/3600-start_of_year-5/24)),axis=0)
        f.close()
    except FileNotFoundError:
        print(j)
        pass


# In[20]:


# Replace fill values with NAN
S_noISA_Reco[S_noISA_Reco==-999]=np.nan
S_noISA_NEE[S_noISA_NEE==-999]=np.nan
S_noISA_GPP[S_noISA_GPP==-999]=np.nan

# Apply Toronto mask
T_S_noISA_GPP=S_noISA_GPP[:,::-1]*GPP_mask[np.newaxis,:,:]
T_S_noISA_Reco=S_noISA_Reco[:,::-1]*GPP_mask[np.newaxis,:,:]
T_S_noISA_NEE=S_noISA_NEE[:,::-1]*GPP_mask[np.newaxis,:,:]


# In[ ]:





# In[21]:


# Swap the axes of UrbanVPRM fluxes to match that of SMUrF data and apply Toronto mask
T_VPRM_NEE=np.swapaxes(np.swapaxes(T_VPRM_NEE,0,2),1,2)*GPP_mask[np.newaxis,:,:]
T_VPRM_Reco=np.swapaxes(np.swapaxes(T_VPRM_Reco,0,2),1,2)*GPP_mask[np.newaxis,:,:]
T_VPRM_GPP=np.swapaxes(np.swapaxes(T_VPRM_GPP,0,2),1,2)*GPP_mask[np.newaxis,:,:]


# In[22]:


### Apply a 100 times bootstrapped Hubber fit for each hour of the year to find correlation between SMUrF and UrbanVPRM NEE

Huber_NEE_w_int_R2=np.zeros(8760)*np.nan
Huber_NEE_w_int_slp=np.zeros(8760)*np.nan
Huber_NEE_int=np.zeros(8760)*np.nan
Huber_NEE_w_int_slp_err=np.zeros(8760)*np.nan
Huber_NEE_int_err=np.zeros(8760)*np.nan

NEE_1_1_RMSE = np.zeros(8760)*np.nan
NEE_1_1_MBE = np.zeros(8760)*np.nan

for i in range(len(T_VPRM_NEE)):
    S_NEE_i = T_S_NEE[i]
    VPRM_NEE_i = T_VPRM_NEE[i]
    
    finitemask0=np.isfinite(S_NEE_i) & np.isfinite(VPRM_NEE_i)
    S_NEE_clean0=S_NEE_i[finitemask0]
    VPRM_NEE_clean0=VPRM_NEE_i[finitemask0]

    NEE_1_1_RMSE[i] = np.sqrt(np.sum((VPRM_NEE_clean0-S_NEE_clean0)**2)/(len(S_NEE_clean0)))                                
    NEE_1_1_MBE[i] = np.mean(VPRM_NEE_clean0-S_NEE_clean0)                                
    
    Huber_2018_slps=[]
    Huber_2018_ints=[]
    Huber_2018_R2=[]

    #try bootstrapping
    indx_list=list(range(0,len(S_NEE_clean0)))
    for j in range(0,100):
        #sub selection of points
        S_NEE_indx=np.random.choice(indx_list,size=len(S_NEE_clean0))
    
        try:
            Huber_model = linear_model.HuberRegressor(fit_intercept=True)
            Huber_fit=Huber_model.fit((S_NEE_clean0[S_NEE_indx]).reshape(-1,1),(VPRM_NEE_clean0[S_NEE_indx]))
            H_m=Huber_fit.coef_
            H_c=Huber_fit.intercept_
            y_predict = H_m * S_NEE_clean0 + H_c
            H_R2=r2_score(VPRM_NEE_clean0, y_predict)
            Huber_2018_slps.append(H_m)
            Huber_2018_ints.append(H_c)
            Huber_2018_R2.append(H_R2)
        except ValueError: #if Huber fit can't find a solution for the subset, skip it
            pass
    
    Huber_NEE_w_int_slp[i] = np.nanmean(Huber_2018_slps)
    Huber_NEE_w_int_slp_err[i] = np.nanstd(Huber_2018_slps)
    Huber_NEE_int[i] = np.nanmean(Huber_2018_ints)
    Huber_NEE_int_err[i] = np.nanstd(Huber_2018_ints)
    Huber_NEE_w_int_R2[i]=r2_score(VPRM_NEE_clean0, Huber_NEE_w_int_slp[i] * S_NEE_clean0 + Huber_NEE_int[i])
    if i%100==0: #Can comment out progress bar
        print(i)


# In[19]:


## *** Uncoment to show that 100 bootstraps is sufficient (values converge)
#Huber_mv_avg_slp=[]
#Huber_mv_std_slp=[]
#Huber_mv_avg_int=[]
#Huber_mv_std_int=[]
#for m in range(1,len(Huber_2018_slps)+1):
#    Huber_mv_avg_slp.append(np.nanmean(Huber_2018_slps[0:m]))
#    Huber_mv_std_slp.append(np.nanstd(Huber_2018_slps[0:m]))
#    Huber_mv_avg_int.append(np.nanmean(Huber_2018_ints[0:m]))
#    Huber_mv_std_int.append(np.nanstd(Huber_2018_ints[0:m]))

#plt.scatter(np.arange(0,100),Huber_2018_slps)
#plt.scatter(np.arange(0,100),Huber_mv_avg_slp)
#plt.axhline(Huber_NEE_w_int_slp[i],linestyle='--',c='k') #average slope
#plt.show()

#plt.scatter(np.arange(0,100),Huber_mv_std_slp)
#plt.axhline(Huber_NEE_w_int_slp_err[i],linestyle='--',c='k') #average error in slope
#plt.show()

#plt.scatter(np.arange(0,100),Huber_2018_ints)
#plt.scatter(np.arange(0,100),Huber_mv_avg_int)
#plt.axhline(Huber_NEE_int[i],linestyle='--',c='k') #average intercept
#plt.show()

#plt.scatter(np.arange(0,100),Huber_mv_std_int)
#plt.axhline(Huber_NEE_int_err[i],linestyle='--',c='k') #average error in intercept
#plt.show()

##end of uncomment


# In[ ]:





# In[23]:


Huber_NEE_noISA_w_int_R2=np.zeros(8760)*np.nan
Huber_NEE_noISA_w_int_slp=np.zeros(8760)*np.nan
Huber_NEE_noISA_int=np.zeros(8760)*np.nan
Huber_NEE_noISA_w_int_slp_err=np.zeros(8760)*np.nan
Huber_NEE_noISA_int_err=np.zeros(8760)*np.nan

NEE_noISA_1_1_RMSE = np.zeros(8760)*np.nan
NEE_noISA_1_1_MBE = np.zeros(8760)*np.nan

for i in range(len(T_VPRM_NEE)):
    S_noISA_NEE_i = T_S_noISA_NEE[i]
    VPRM_NEE_i = T_VPRM_NEE[i]
    
    finitemask0=np.isfinite(S_noISA_NEE_i) & np.isfinite(VPRM_NEE_i)
    S_noISA_NEE_clean0=S_noISA_NEE_i[finitemask0]
    VPRM_NEE_clean0=VPRM_NEE_i[finitemask0]

    NEE_noISA_1_1_RMSE[i] = np.sqrt(np.sum((VPRM_NEE_clean0-S_noISA_NEE_clean0)**2)/(len(S_noISA_NEE_clean0)))                                
    NEE_noISA_1_1_MBE[i] = np.mean(VPRM_NEE_clean0-S_noISA_NEE_clean0)                                
    
    Huber_noISA_2018_slps=[]
    Huber_noISA_2018_ints=[]
    Huber_noISA_2018_R2=[]

    #try bootstrapping
    indx_list=list(range(0,len(S_noISA_NEE_clean0)))
    for j in range(0,100):
        #sub selection of points
        S_noISA_NEE_indx=np.random.choice(indx_list,size=len(S_noISA_NEE_clean0))
    
        try:
            Huber_model = linear_model.HuberRegressor(fit_intercept=True)
            Huber_fit=Huber_model.fit((S_noISA_NEE_clean0[S_noISA_NEE_indx]).reshape(-1,1),(VPRM_NEE_clean0[S_noISA_NEE_indx]))
            H_m=Huber_fit.coef_
            H_c=Huber_fit.intercept_
            y_predict = H_m * S_noISA_NEE_clean0 + H_c
            H_R2=r2_score(VPRM_NEE_clean0, y_predict)
            Huber_noISA_2018_slps.append(H_m)
            Huber_noISA_2018_ints.append(H_c)
            Huber_noISA_2018_R2.append(H_R2)
        except ValueError: #if Huber fit can't find a solution for the subset, skip it
            pass
    
    Huber_NEE_noISA_w_int_slp[i] = np.nanmean(Huber_noISA_2018_slps)
    Huber_NEE_noISA_w_int_slp_err[i] = np.nanstd(Huber_noISA_2018_slps)
    Huber_NEE_noISA_int[i] = np.nanmean(Huber_noISA_2018_ints)
    Huber_NEE_noISA_int_err[i] = np.nanstd(Huber_noISA_2018_ints)
    Huber_NEE_noISA_w_int_R2[i]=r2_score(VPRM_NEE_clean0, Huber_NEE_noISA_w_int_slp[i] * S_noISA_NEE_clean0 + Huber_NEE_noISA_int[i])
    if i%100==0:
        print(i)


# In[24]:


Huber_GPP_w_int_R2=np.zeros(8760)*np.nan
Huber_GPP_w_int_slp=np.zeros(8760)*np.nan
Huber_GPP_int=np.zeros(8760)*np.nan
Huber_GPP_w_int_slp_err=np.zeros(8760)*np.nan
Huber_GPP_int_err=np.zeros(8760)*np.nan

GPP_1_1_RMSE = np.zeros(8760)*np.nan
GPP_1_1_MBE = np.zeros(8760)*np.nan

for i in range(len(T_VPRM_GPP)):
    with np.errstate(invalid='ignore'):
        S_GPP_i = T_S_GPP[i]
        VPRM_GPP_i = T_VPRM_GPP[i]

        finitemask0=np.isfinite(S_GPP_i) & np.isfinite(VPRM_GPP_i)
        S_GPP_clean0=S_GPP_i[finitemask0]
        VPRM_GPP_clean0=VPRM_GPP_i[finitemask0]

        try:
            GPP_1_1_RMSE[i] = np.sqrt(np.sum((VPRM_GPP_clean0-S_GPP_clean0)**2)/(len(S_GPP_clean0)))                                
            GPP_1_1_MBE[i] = np.mean(VPRM_GPP_clean0-S_GPP_clean0)                                
        except RuntimeError:
            pass

        Huber_2018_slps=[]
        Huber_2018_ints=[]
        Huber_2018_R2=[]

        #try bootstrapping
        indx_list=list(range(0,len(S_GPP_clean0)))
        for j in range(0,100):
            #sub selection of points
            S_GPP_indx=np.random.choice(indx_list,size=len(S_GPP_clean0))

            try:
                Huber_model = linear_model.HuberRegressor(fit_intercept=True)
                Huber_fit=Huber_model.fit((S_GPP_clean0[S_GPP_indx]).reshape(-1,1),(VPRM_GPP_clean0[S_GPP_indx]))
                H_m=Huber_fit.coef_
                H_c=Huber_fit.intercept_
                y_predict = H_m * S_GPP_clean0 + H_c
                H_R2=r2_score(VPRM_GPP_clean0, y_predict)
                Huber_2018_slps.append(H_m)
                Huber_2018_ints.append(H_c)
                Huber_2018_R2.append(H_R2)
            except ValueError: #if Huber fit can't find a solution for the subset, skip it
                pass

        Huber_GPP_w_int_slp[i] = np.nanmean(Huber_2018_slps)
        Huber_GPP_w_int_slp_err[i] = np.nanstd(Huber_2018_slps)
        Huber_GPP_int[i] = np.nanmean(Huber_2018_ints)
        Huber_GPP_int_err[i] = np.nanstd(Huber_2018_ints)
        Huber_GPP_w_int_R2[i]=r2_score(VPRM_GPP_clean0, Huber_GPP_w_int_slp[i] * S_GPP_clean0 + Huber_GPP_int[i])
        if i%100==0:
            print(i)


# In[25]:


#When GPP is 0 for both UrbanVPRM and SMUrF R2 becomes 1 and slope gets all wonky, remove these values & replace with nan

Huber_GPP_w_int_slp[Huber_GPP_w_int_R2==1]=np.nan
Huber_GPP_w_int_slp_err[Huber_GPP_w_int_R2==1]=np.nan
Huber_GPP_int[Huber_GPP_w_int_R2==1]=np.nan
Huber_GPP_int_err[Huber_GPP_w_int_R2==1]=np.nan
Huber_GPP_w_int_R2[Huber_GPP_w_int_R2==1]=np.nan


# In[ ]:





# In[26]:


Huber_Reco_w_int_R2=np.zeros(8760)*np.nan
Huber_Reco_w_int_slp=np.zeros(8760)*np.nan
Huber_Reco_int=np.zeros(8760)*np.nan
Huber_Reco_w_int_slp_err=np.zeros(8760)*np.nan
Huber_Reco_int_err=np.zeros(8760)*np.nan

Reco_1_1_RMSE = np.zeros(8760)*np.nan
Reco_1_1_MBE = np.zeros(8760)*np.nan

for i in range(len(T_VPRM_Reco)):
    S_Reco_i = T_S_Reco[i]
    VPRM_Reco_i = T_VPRM_Reco[i]
    
    finitemask0=np.isfinite(S_Reco_i) & np.isfinite(VPRM_Reco_i)
    S_Reco_clean0=S_Reco_i[finitemask0]
    VPRM_Reco_clean0=VPRM_Reco_i[finitemask0]

    Reco_1_1_RMSE[i] = np.sqrt(np.sum((VPRM_Reco_clean0-S_Reco_clean0)**2)/(len(S_Reco_clean0)))                                
    Reco_1_1_MBE[i] = np.mean(VPRM_Reco_clean0-S_Reco_clean0)                                
    
    Huber_2018_slps=[]
    Huber_2018_ints=[]
    Huber_2018_R2=[]

    #try bootstrapping
    indx_list=list(range(0,len(S_Reco_clean0)))
    for j in range(0,100):
        #sub selection of points
        S_Reco_indx=np.random.choice(indx_list,size=len(S_Reco_clean0))
    
        try:
            Huber_model = linear_model.HuberRegressor(fit_intercept=True)
            Huber_fit=Huber_model.fit((S_Reco_clean0[S_Reco_indx]).reshape(-1,1),(VPRM_Reco_clean0[S_Reco_indx]))
            H_m=Huber_fit.coef_
            H_c=Huber_fit.intercept_
            y_predict = H_m * S_Reco_clean0 + H_c
            H_R2=r2_score(VPRM_Reco_clean0, y_predict)
            Huber_2018_slps.append(H_m)
            Huber_2018_ints.append(H_c)
            Huber_2018_R2.append(H_R2)
        except ValueError: #if Huber fit can't find a solution for the subset, skip it
            pass
    
    Huber_Reco_w_int_slp[i] = np.nanmean(Huber_2018_slps)
    Huber_Reco_w_int_slp_err[i] = np.nanstd(Huber_2018_slps)
    Huber_Reco_int[i] = np.nanmean(Huber_2018_ints)
    Huber_Reco_int_err[i] = np.nanstd(Huber_2018_ints)
    Huber_Reco_w_int_R2[i]=r2_score(VPRM_Reco_clean0, Huber_Reco_w_int_slp[i] * S_Reco_clean0 + Huber_Reco_int[i])
    if i%100==0:
        print(i)


# In[27]:


Huber_Reco_noISA_w_int_R2=np.zeros(8760)*np.nan
Huber_Reco_noISA_w_int_slp=np.zeros(8760)*np.nan
Huber_Reco_noISA_int=np.zeros(8760)*np.nan
Huber_Reco_noISA_w_int_slp_err=np.zeros(8760)*np.nan
Huber_Reco_noISA_int_err=np.zeros(8760)*np.nan

Reco_noISA_1_1_RMSE = np.zeros(8760)*np.nan
Reco_noISA_1_1_MBE = np.zeros(8760)*np.nan

for i in range(len(T_VPRM_Reco)):
    S_noISA_Reco_i = T_S_noISA_Reco[i]
    VPRM_Reco_i = T_VPRM_Reco[i]
    
    finitemask0=np.isfinite(S_noISA_Reco_i) & np.isfinite(VPRM_Reco_i)
    S_noISA_Reco_clean0=S_noISA_Reco_i[finitemask0]
    VPRM_Reco_clean0=VPRM_Reco_i[finitemask0]

    Reco_noISA_1_1_RMSE[i] = np.sqrt(np.sum((VPRM_Reco_clean0-S_noISA_Reco_clean0)**2)/(len(S_noISA_Reco_clean0)))                                
    Reco_noISA_1_1_MBE[i] = np.mean(VPRM_Reco_clean0-S_noISA_Reco_clean0)                                
    
    Huber_noISA_2018_slps=[]
    Huber_noISA_2018_ints=[]
    Huber_noISA_2018_R2=[]

    #try bootstrapping
    indx_list=list(range(0,len(S_noISA_Reco_clean0)))
    for j in range(0,100):
        #sub selection of points
        S_noISA_Reco_indx=np.random.choice(indx_list,size=len(S_noISA_Reco_clean0))
    
        try:
            Huber_model = linear_model.HuberRegressor(fit_intercept=True)
            Huber_fit=Huber_model.fit((S_noISA_Reco_clean0[S_noISA_Reco_indx]).reshape(-1,1),(VPRM_Reco_clean0[S_noISA_Reco_indx]))
            H_m=Huber_fit.coef_
            H_c=Huber_fit.intercept_
            y_predict = H_m * S_noISA_Reco_clean0 + H_c
            H_R2=r2_score(VPRM_Reco_clean0, y_predict)
            Huber_noISA_2018_slps.append(H_m)
            Huber_noISA_2018_ints.append(H_c)
            Huber_noISA_2018_R2.append(H_R2)
        except ValueError: #if Huber fit can't find a solution for the subset, skip it
            pass
    
    Huber_Reco_noISA_w_int_slp[i] = np.nanmean(Huber_noISA_2018_slps)
    Huber_Reco_noISA_w_int_slp_err[i] = np.nanstd(Huber_noISA_2018_slps)
    Huber_Reco_noISA_int[i] = np.nanmean(Huber_noISA_2018_ints)
    Huber_Reco_noISA_int_err[i] = np.nanstd(Huber_noISA_2018_ints)
    Huber_Reco_noISA_w_int_R2[i]=r2_score(VPRM_Reco_clean0, Huber_Reco_noISA_w_int_slp[i] * S_noISA_Reco_clean0 + Huber_Reco_noISA_int[i])
    if i%100==0:
        print(i)


# In[28]:


HoY = np.arange(1,8761)


# In[68]:


plt.style.use('tableau-colorblind10')

plt.rc('font',size=27) #21

fig, ax = plt.subplots(3,3,sharex=True,sharey='row',figsize=(24,12))
ax[0,0].set_xlim(1,366)
ax[0,0].set_ylim(-0.01,0.99)

fig0=ax[0,0].scatter(HoY/24+1,Huber_NEE_noISA_w_int_R2,s=5)
ax[0,0].scatter(HoY/24+1,Huber_NEE_w_int_R2,s=5)
ax[0,1].scatter(HoY/24+1,Huber_Reco_noISA_w_int_R2,s=5)
ax[0,1].scatter(HoY/24+1,Huber_Reco_w_int_R2,s=5)
ax[0,2].scatter(HoY/24+1,Huber_GPP_w_int_R2,s=5)

ax[0,0].set_title('NEE Correlation Statistics')
ax[0,1].set_title('R$_{eco}$ Correlation Statistics')
ax[0,2].set_title('GPP Correlation Statistics')

ax[1,0].set_ylim(-0.5,12)
fig1=ax[1,0].scatter(HoY/24+1,NEE_noISA_1_1_RMSE,s=5)
ax[1,0].scatter(HoY/24+1,NEE_1_1_RMSE,s=5)
ax[1,1].scatter(HoY/24+1,Reco_noISA_1_1_RMSE,s=5)
ax[1,1].scatter(HoY/24+1,Reco_1_1_RMSE,s=5)
ax[1,2].scatter(HoY/24+1,GPP_1_1_RMSE,s=5)

ax[1,1].scatter([0],[0],c='#006BA4',label='Without ISA Adjustment')
ax[1,1].scatter([0],[0],c='#FF800E',label='With ISA Adjustment')
ax[1,1].legend(loc=(0.025,0.375))#fontsize=22)


ax[2,0].set_ylim(-9,9)
fig2=ax[2,0].scatter(HoY/24+1,NEE_noISA_1_1_MBE,s=5)
ax[2,0].scatter(HoY/24+1,NEE_1_1_MBE,s=5)
ax[2,1].scatter(HoY/24+1,Reco_noISA_1_1_MBE,s=5)
ax[2,1].scatter(HoY/24+1,Reco_1_1_MBE,s=5)
ax[2,2].scatter(HoY/24+1,GPP_1_1_MBE,s=5)
ax[2,0].axhline(0,linestyle='--',c='k')
ax[2,1].axhline(0,linestyle='--',c='k')
ax[2,2].axhline(0,linestyle='--',c='k')

ax[0,0].set_ylabel('R$^2$')
ax[1,0].set_ylabel('RMSE')# ($\mu$mol m$^{-2}$ s$^{-1}$)')
ax[2,0].set_ylabel('MBE')# ($\mu$mol m$^{-2}$ s$^{-1}$)')


ax[2,0].set_xlabel('Day of Year')
ax[2,1].set_xlabel('Day of Year')
ax[2,2].set_xlabel('Day of Year')

ax[0,0].text(4,0.875,'(a)',c='k',fontsize=26)
ax[0,1].text(4,0.875,'(b)',c='k',fontsize=26)
ax[0,2].text(4,0.875,'(c)',c='k',fontsize=26)
ax[1,0].text(4,10.5,'(d)',c='k',fontsize=26)
ax[1,1].text(4,10.5,'(e)',c='k',fontsize=26)
ax[1,2].text(4,10.5,'(f)',c='k',fontsize=26)
ax[2,0].text(4,6.75,'(g)',c='k',fontsize=26)
ax[2,1].text(4,6.75,'(h)',c='k',fontsize=26)
ax[2,2].text(4,6.75,'(i)',c='k',fontsize=26)

fig.subplots_adjust(hspace=0,wspace=0)
# *** Uncomment to save figure as pdf & png. CHANGE FILENAMES ***
#plt.savefig('Fixed_SMUrF_shore_corr_UrbanVPRM_before_after_ISA_correction_Huber_correlation_stats_all_fluxes_cb_friendly_larger_font_labelled.pdf',bbox_inches='tight')
#plt.savefig('Fixed_SMUrF_shore_corr_UrbanVPRM_before_after_ISA_correction_Huber_correlation_stats_all_fluxes_cb_friendly_larger_font_labelled.png',bbox_inches='tight')
fig.show()


# In[ ]:




