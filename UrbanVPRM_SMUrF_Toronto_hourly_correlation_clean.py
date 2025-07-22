#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This code is used to compare the spatial correlation between the updated SMUrF and UrbanVPRM at hourly, daily, monthly,
# and annual time scales. We use bootstrapped Hubber fits. Code reproduces figure S4 of Madsen-Colford et al. 2025.
# Please cite if code is used.

# *** indicates lines (below) that the user should change (e.g. path names etc.)


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
import shapefile as shp # to import outline of GTA
from shapely import geometry # used to define a polygon for Toronto
from sklearn import linear_model #for doing robust fits
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.colors as clrs #for log color scale


# In[2]:


#Load in VPRM data
# *** CHANGE PATH *** 
VPRM_path = 'E:/Research/UrbanVPRM/dataverse_files/GTA_V061_500m_2018/'
# *** CHANGE FILE NAME ***
VPRM_fn = 'vprm_GMIS_Toronto_ACI_SOLRIS_ISA_500m_GTA_V061_2018_no_PScale_adjusted_Topt_Ra_URB_parameters_fixed_gapfilled_LSWI_filtered_bilinear_PAR_block_'

VPRM_data = pd.read_csv(VPRM_path+VPRM_fn+'00000001.csv').loc[:,('HoY','Index','GEE','Re')]

VPRM_data2=pd.read_csv(VPRM_path+VPRM_fn+'00002501.csv').loc[:,('HoY','Index','GEE','Re')]
VPRM_data=VPRM_data.append(VPRM_data2)
del VPRM_data2

VPRM_data2=pd.read_csv(VPRM_path+VPRM_fn+'00005001.csv').loc[:,('HoY','Index','GEE','Re')]
VPRM_data=VPRM_data.append(VPRM_data2)
del VPRM_data2

VPRM_data2=pd.read_csv(VPRM_path+VPRM_fn+'00007501.csv').loc[:,('HoY','Index','GEE','Re')]
VPRM_data=VPRM_data.append(VPRM_data2)
del VPRM_data2

VPRM_data2=pd.read_csv(VPRM_path+VPRM_fn+'00010001.csv').loc[:,('HoY','Index','GEE','Re')]
VPRM_data=VPRM_data.append(VPRM_data2)
del VPRM_data2

VPRM_data2=pd.read_csv(VPRM_path+VPRM_fn+'00012501.csv').loc[:,('HoY','Index','GEE','Re')]
VPRM_data=VPRM_data.append(VPRM_data2)
del VPRM_data2


# In[3]:


# *** CHANGE FILE NAME ***
VPRM_EVI=pd.read_csv(VPRM_path+'adjusted_evi_lswi_interpolated_modis_v061_qc_filtered_LSWI_filtered.csv').loc[:,('Index','x','y')]

#Create a dataframe with just Index, x, & y values
x=np.zeros(np.shape(VPRM_EVI.Index.unique()))*np.nan
y=np.zeros(np.shape(VPRM_EVI.Index.unique()))*np.nan
for i in range(len(VPRM_EVI.Index.unique())):
    x[i]=VPRM_EVI.x[0+i*365]
    y[i]=VPRM_EVI.y[0+i*365]
    
VPRM_xy=pd.DataFrame({'Index':VPRM_EVI.Index.unique(), 'x':x, 'y':y})
VPRM_data=VPRM_data.merge(VPRM_xy[['Index','x','y']])
del VPRM_EVI, VPRM_xy


# In[4]:


del x, y


# In[5]:


# Create arrays of x & y values
xvals = VPRM_data.x[VPRM_data.HoY==4800].unique()
yvals = VPRM_data.y[VPRM_data.HoY==4800].unique()
extent = np.min(xvals), np.max(xvals), np.min(yvals), np.max(yvals)

# Reshape GPP and Reco into arrays
GPP=-VPRM_data.GEE.values.reshape(len(yvals),len(xvals),8760)#8784 for leap year
Reco=VPRM_data.Re.values.reshape(len(yvals),len(xvals),8760)


# In[6]:


#Load in the shape file for Toronto

# *** CHANGE PATH ***
sf = shp.Reader("C:/Users/kitty/Documents/Research/SIF/Shape_files/Toronto/Toronto_Boundary.shp") #Toronto_Shape
shape=sf.shape(0)
Toronto_x = np.zeros((len(shape.points),1))*np.nan
Toronto_y = np.zeros((len(shape.points),1))*np.nan
for i in range(len(shape.points)):
    Toronto_x[i]=shape.points[i][0]
    Toronto_y[i]=shape.points[i][1]
    
points=[]
for k in range(1,len(Toronto_x)):
    points.append(geometry.Point(Toronto_x[k],Toronto_y[k]))
poly=geometry.Polygon([[p.x, p.y] for p in points])

#Create a mask for areas outside Toronto
lons=np.ones(144)*np.nan
lats=np.ones(96)*np.nan
GPP_mask=np.ones([96,144])*np.nan
for i in range(0, len(lons)):
    for j in range(0, len(lats)):
        lons[i]=xvals[i]
        lats[j]=yvals[j]
        if poly.contains(geometry.Point([xvals[i],yvals[j]])):
            GPP_mask[j,i]=1


# In[7]:


# Apply the Toronto mask to GPP & Reco & compute NEE
T_VPRM_GPP=GPP*GPP_mask[:,:,np.newaxis]
T_VPRM_Reco=Reco*GPP_mask[:,:,np.newaxis]
T_VPRM_NEE=T_VPRM_Reco-T_VPRM_GPP

#swap the axes to match those of SMUrF
T_VPRM_NEE=np.swapaxes(np.swapaxes(T_VPRM_NEE,0,2),1,2)
del T_VPRM_GPP, T_VPRM_Reco


# In[ ]:





# In[10]:


#now bring in the SMUrF data with ISA adjustment, shoreline correction, AND downscaling fix

# Bring in the first Reco file and extract the first day of the year (in seconds since 1970)
# ***CHANGE PATH AND FILE NAME ***
g=Dataset('C:/Users/kitty/Documents/Research/SIF/SMUrF/output2018_500m_CSIF_to_TROPOMI_CSIF_ALL_converted_slps_V3_temp_impervious_R_shore_corr_V061_8day/easternCONUS/daily_mean_Reco_ISA_a_neuralnet/era5/2018/daily_mean_Reco_uncert_GMIS_Toronto_t_easternCONUS_20180101.nc')
start_of_year=g.variables['time'][0]/3600/24-1 #convert seconds since 1970 to days (minus one)
g.close()


#Load in SMUrF NEE data
# *** CHANGE PATH ***
SMUrF_path = 'E:/Research/SMUrF/output2018_500m_CSIF_to_TROPOMI_CSIF_ALL_converted_slps_V3_temp_impervious_R_shore_corr_V061_8day/easternCONUS/hourly_flux_GMIS_Toronto_fixed_border_ISA_a_w_sd_era5/'
# *** CHANGE FILE NAME ***
SMUrF_fn = 'hrly_mean_GPP_Reco_NEE_easternCONUS_2018'

S_time=[]
S_NEE=[]
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
            S_NEE=f.variables['NEE_mean'][:,264:360,288:432]
            S_time=f.variables['time'][:]/24/3600-start_of_year-5/24 #convert seconds since 1970 to days and subtract start of year
        else:
            S_NEE=np.concatenate((S_NEE,f.variables['NEE_mean'][:,264:360,288:432]),axis=0)
            S_time=np.concatenate((S_time,(f.variables['time'][:]/24/3600-start_of_year-5/24)),axis=0)
        f.close()
    except FileNotFoundError:
        print(j)
        pass


# In[11]:


# Replace fill values with NaN
S_NEE[S_NEE==-999]=np.nan

# Apply Toronto mask
T_S_NEE=S_NEE[:,::-1]*GPP_mask[np.newaxis,:,:]


# In[12]:


del S_NEE, VPRM_data, GPP, Reco


# In[ ]:





# In[13]:


# Define a function and straight line for fitting and plotting

def func2(x,m,b):
    return m*x+b

line1_1=np.arange(-100,100)


# In[14]:


# Fit the hourly data with a 1000x bootstrapped Huber fit

# WITH Shoreline and ISA correction
finitemask0=np.isfinite(T_S_NEE) & np.isfinite(T_VPRM_NEE)
T_S_NEE_clean0=T_S_NEE[finitemask0]
T_VPRM_NEE_clean0=T_VPRM_NEE[finitemask0]

Huber_2018_slps=[]
Huber_2018_ints=[]
Huber_2018_R2=[]

#try bootstrapping
indx_list=list(range(0,len(T_S_NEE_clean0[(T_S_NEE_clean0!=0) & (T_VPRM_NEE_clean0!=0)])))
for i in range(1,1001):
    #sub selection of points
    S_NEE_indx=np.random.choice(indx_list,size=int(50000))
    
    try:
        Huber_model = linear_model.HuberRegressor(fit_intercept=True)
        Huber_fit=Huber_model.fit((T_S_NEE_clean0[(T_S_NEE_clean0!=0) & (T_VPRM_NEE_clean0!=0)][S_NEE_indx]).reshape(-1,1),T_VPRM_NEE_clean0[(T_S_NEE_clean0!=0) & (T_VPRM_NEE_clean0!=0)][S_NEE_indx])
        H_m=Huber_fit.coef_
        H_c=Huber_fit.intercept_
        y_predict = H_m * T_S_NEE_clean0[(T_S_NEE_clean0!=0) & (T_VPRM_NEE_clean0!=0)] + H_c
        H_R2=r2_score(T_VPRM_NEE_clean0[(T_S_NEE_clean0!=0) & (T_VPRM_NEE_clean0!=0)], y_predict)
        Huber_2018_slps.append(H_m)
        Huber_2018_ints.append(H_c)
        Huber_2018_R2.append(H_R2)
    except ValueError: #if Huber fit can't find a solution for the subset, skip it
        pass
    
    if int(i)%10==0: #This line prints progress in %, can comment out if desired
        print(i/1000*100)


# In[15]:


# Calculate the R2 using the average slope and intercept from the bootstrapped Huber fits
Huber_avg_R2=r2_score(T_VPRM_NEE_clean0[(T_S_NEE_clean0!=0) & (T_VPRM_NEE_clean0!=0)], T_S_NEE_clean0[(T_S_NEE_clean0!=0) & (T_VPRM_NEE_clean0!=0)]*np.nanmean(Huber_2018_slps)+np.nanmean(Huber_2018_ints))
print('Hourly Huber fit R^2 = '+str(np.round(Huber_avg_R2,5)))


# In[16]:


# Calculate the R2 assuming a 1:1 correlation
NEE_2018_slope1_R2=r2_score(T_VPRM_NEE_clean0[(T_S_NEE_clean0!=0) & (T_VPRM_NEE_clean0!=0)], T_S_NEE_clean0[(T_S_NEE_clean0!=0) & (T_VPRM_NEE_clean0!=0)])
print('Hourly fit 1:1 line R^2 = '+str(np.round(NEE_2018_slope1_R2,5)))


# In[17]:


#Sanity check (calculate 1:1 R2 by hand)
#sum_tot = np.sum((T_VPRM_NEE_clean0[(T_S_NEE_clean0!=0) & (T_VPRM_NEE_clean0!=0)]-np.mean(T_VPRM_NEE_clean0[(T_S_NEE_clean0!=0) & (T_VPRM_NEE_clean0!=0)]))**2)
#sum_res= np.sum((T_VPRM_NEE_clean0[(T_S_NEE_clean0!=0) & (T_VPRM_NEE_clean0!=0)]-func2(T_S_NEE_clean0[(T_S_NEE_clean0!=0) & (T_VPRM_NEE_clean0!=0)],1,0))**2)
#print(1-sum_res/sum_tot)


# In[19]:


# *** Optional: Uncomment to plot correlation of hourly data ***

#plt.style.use('tableau-colorblind10')

#plt.figure(figsize=(8,6))
#plt.xlim(-20,7)
#plt.ylim(-20,7)
#plt.axis('scaled')
#plt.scatter(T_S_NEE_clean0[(T_S_NEE_clean0!=0) & (T_VPRM_NEE_clean0!=0)][S_NEE_indx],T_VPRM_NEE_clean0[(T_S_NEE_clean0!=0) & (T_VPRM_NEE_clean0!=0)][S_NEE_indx],alpha=0.1,s=5)
#plt.plot(line1_1,line1_1*np.nanmean(Huber_2018_slps)+np.nanmean(Huber_2018_ints),label=str(np.round(np.nanmean(Huber_2018_slps),2))+'$\cdot$x + '+str(np.round(np.nanmean(Huber_2018_ints),2))+', R$^2$ = '+str(np.round(Huber_avg_R2,2)),linestyle='--',c='#FF800E')
#plt.plot(line1_1,line1_1,linestyle='-.',c='k',label='1:1, R$^2$ = '+str(np.round(NEE_2018_slope1_R2,2)))
#plt.legend(loc='lower right')
#plt.xlabel('SMUrF NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
#plt.ylabel('UrbanVPRM NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
#plt.title('SMUrF vs. UrbanVPRM NEE over Toronto, 2018')
##*** Uncomment to save figure as png & pdf CHANGE FILENAME ***
#plt.savefig('Toronto_Fixed_SMUrF_vs_UrbanVPRM_NEE_hourly_bootstrap_Huber_correlation_no_weights_subselection.pdf',bbox_inches='tight')
#plt.savefig('Toronto_Fixed_SMUrF_vs_UrbanVPRM_NEE_hourly_bootstrap_Huber_correlation_no_weights_subselection.png',bbox_inches='tight')
#plt.show()


# In[ ]:





# In[18]:


# Calculate the daily average NEE for SMUrF and UrbanVPRM
VPRM_NEE_daily=np.ones((365, 96, 144))*np.nan
S_NEE_daily=np.ones((365, 96, 144))*np.nan
for i in range(365):
    VPRM_NEE_daily[i]=np.nanmean(T_VPRM_NEE[i*24:i*24+24],axis=0)
    S_NEE_daily[i]=np.nanmean(T_S_NEE[i*24:i*24+24],axis=0)


# In[19]:


# Apply 1000x bootstrapped Huber fit to daily average NEE

# WITH Shoreline and ISA correction
finitemask0=np.isfinite(S_NEE_daily) & np.isfinite(VPRM_NEE_daily) & (S_NEE_daily!=0) & (VPRM_NEE_daily!=0)
S_NEE_daily_clean0=S_NEE_daily[finitemask0]
VPRM_NEE_daily_clean0=VPRM_NEE_daily[finitemask0]

Huber_dly_slps=[]
Huber_dly_ints=[]
Huber_dly_R2=[]

#try bootstrapping
dly_indx_list=list(range(0,len(S_NEE_daily_clean0)))
for i in range(1,1000):
    #sub selection of points
    S_NEE_dly_indx=np.random.choice(dly_indx_list,size=int(50000))
    
    try:
        Huber_model = linear_model.HuberRegressor(fit_intercept=True)
        Huber_fit=Huber_model.fit((S_NEE_daily_clean0[S_NEE_dly_indx]).reshape(-1,1),VPRM_NEE_daily_clean0[S_NEE_dly_indx])
        H_m=Huber_fit.coef_
        H_c=Huber_fit.intercept_
        x_accpt, y_accpt = S_NEE_daily_clean0,VPRM_NEE_daily_clean0
        y_predict = H_m * x_accpt + H_c
        H_R2=r2_score(y_accpt, y_predict)
        Huber_dly_slps.append(H_m)
        Huber_dly_ints.append(H_c)
        Huber_dly_R2.append(H_R2)
    except ValueError: #if Huber fit can't find a solution for the subset, skip it
        pass
    if int(i)%10==0:
        print(i/1000*100)


# In[20]:


# Calculate the R2 using the average slope and intercept from the bootstrapped Huber fits
Huber_dly_avg_R2=r2_score(VPRM_NEE_daily_clean0, S_NEE_daily_clean0*np.nanmean(Huber_dly_slps)+np.nanmean(Huber_dly_ints))
#Huber_dly_avg_R2
print('Daily Average Huber fit R^2 = '+str(np.round(Huber_dly_avg_R2,5)))


# In[21]:


# Calculate the R2 assuming 1:1 correlation
NEE_2018_daily_slope1_R2=r2_score(VPRM_NEE_daily_clean0, S_NEE_daily_clean0)
#NEE_2018_daily_slope1_R2
print('Daily average fit 1:1 R^2 = '+str(np.round(NEE_2018_daily_slope1_R2,5)))


# In[24]:


#Sanity check (calculate 1:1 R2 by hand)
#sum_tot = np.sum((VPRM_NEE_daily_clean0-np.mean(VPRM_NEE_daily_clean0))**2)
#sum_res= np.sum((VPRM_NEE_daily_clean0-func2(S_NEE_daily_clean0,1,0))**2)
#print(1-sum_res/sum_tot)


# In[39]:


# *** Optional: Uncomment to plot correlation of daily-averaged data ***

##50000 points (3.4% of data)
#plt.rc('font',size=14)

#plt.figure(figsize=(6,8))
#plt.xlim(-35,8)
#plt.ylim(-35,8)
#plt.axis('scaled')
#plt.scatter(S_NEE_daily_clean0[S_NEE_dly_indx],VPRM_NEE_daily_clean0[S_NEE_dly_indx],s=1)
#plt.plot(line1_1,line1_1*np.nanmean(Huber_dly_slps)+np.nanmean(Huber_dly_ints),linewidth=2,linestyle='--',label=str(np.round(np.nanmean(Huber_dly_slps),2))+'x + '+str(np.round(np.nanmean(Huber_dly_ints),2))+', R$^2$ = '+str(np.round(Huber_dly_avg_R2,2)),c='#FF800E')
#plt.plot(line1_1,line1_1,linestyle=':',c='k',label='1:1, R$^2$ = '+str(np.round(NEE_2018_daily_slope1_R2,2)))
#plt.legend()
#plt.xlabel('SMUrF NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
#plt.ylabel('UrbanVPRM NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
#plt.title('SMUrF vs. UrbanVPRM Daily NEE, Toronto, 2018')
##*** Uncomment to save figure as png and pdf. CHANGE FILENAME ***
#plt.savefig('Toronto_Fixed_SMUrF_vs_UrbanVPRM_NEE_daily_Huber_no_errs_correlation_subselection.pdf',bbox_inches='tight')
#plt.savefig('Toronto_Fixed_SMUrF_vs_UrbanVPRM_NEE_daily_Huber_no_errs_correlation_subselection.png',bbox_inches='tight')
#plt.show()


# In[ ]:





# In[22]:


# Take 30 day average of SMUrF and VPRM NEE
VPRM_NEE_30d=np.ones((12, 96, 144))*np.nan
S_NEE_30d=np.ones((12, 96, 144))*np.nan
for i in range(12):
    VPRM_NEE_30d[i]=np.nanmean(T_VPRM_NEE[i*30*24:i*30*24+30*24],axis=0)
    S_NEE_30d[i]=np.nanmean(T_S_NEE[i*30*24:i*30*24+30*24],axis=0)


# In[23]:


# Apply 1000x bootstrapped Huber fit to monthly averaged data
finitemask0=np.isfinite(S_NEE_30d) & np.isfinite(VPRM_NEE_30d) & (S_NEE_30d!=0)
S_NEE_30d_clean0=S_NEE_30d[finitemask0]
VPRM_NEE_30d_clean0=VPRM_NEE_30d[finitemask0]

Huber_30d_slps=[]
Huber_30d_ints=[]
Huber_30d_R2=[]

#try bootstrapping
indx_30d_list=list(range(0,len(S_NEE_30d_clean0)))
for i in range(1,1000):
    #sub selection of points
    S_NEE_30d_indx=np.random.choice(indx_30d_list,size=int(50000))
    
    try:
        Huber_model = linear_model.HuberRegressor(fit_intercept=True)
        Huber_fit=Huber_model.fit((S_NEE_30d_clean0[S_NEE_30d_indx]).reshape(-1,1),VPRM_NEE_30d_clean0[S_NEE_30d_indx])
        H_m=Huber_fit.coef_
        H_c=Huber_fit.intercept_
        x_accpt, y_accpt = S_NEE_30d_clean0, VPRM_NEE_30d_clean0
        y_predict = H_m * x_accpt + H_c
        H_R2=r2_score(y_accpt, y_predict)
        Huber_30d_slps.append(H_m)
        Huber_30d_ints.append(H_c)
        Huber_30d_R2.append(H_R2)
    except ValueError:
        pass
    if int(i)%10==0:
        print(i/1000*100)


# In[24]:


# Calculate the R2 using the average Huber fit
Huber_30d_avg_R2=r2_score(VPRM_NEE_30d_clean0,S_NEE_30d_clean0*np.nanmean(Huber_30d_slps)+np.nanmean(Huber_30d_ints))
#Huber_30d_avg_R2
print('30-day average Huber fit R^2 = '+str(np.round(Huber_30d_avg_R2,5)))


# In[25]:


# Calculate the R2 assuming a 1:1 correlation
NEE_2018_30d_slope1_R2=r2_score(VPRM_NEE_30d_clean0,S_NEE_30d_clean0)
#NEE_2018_30d_slope1_R2
print('30-day average fit 1:1 R^2 = '+str(np.round(NEE_2018_30d_slope1_R2,5)))


# In[26]:


##Sanity check (calculate 1:1 R2 by hand)
#sum_tot = np.sum((VPRM_NEE_30d_clean0-np.mean(VPRM_NEE_30d_clean0))**2)
#sum_res= np.sum((VPRM_NEE_30d_clean0-func2(S_NEE_30d_clean0,1,0))**2)
#print(1-sum_res/sum_tot)


# In[31]:


# *** Optional: Uncomment to plot correlation of 30day averaged data ***

#Bootstrapped Huber fit
#all points
#plt.rc('font',size=14)

#plt.figure(figsize=(8,6))
#plt.xlim(-15,5)
#plt.ylim(-15,5)
#plt.axis('scaled')
#plt.scatter(S_NEE_30d_clean0,VPRM_NEE_30d_clean0,s=1,alpha=0.25)
#plt.plot(line1_1,line1_1*np.nanmean(Huber_30d_slps)+np.nanmean(Huber_30d_ints),linestyle='--',label=str(np.round(np.nanmean(Huber_30d_slps),2))+'x + '+str(np.round(np.nanmean(Huber_30d_ints),2))+', R$^2$ = '+str(np.round(Huber_30d_avg_R2,2)),c='#FF800E')
#plt.plot(line1_1,line1_1,linestyle=':',c='k',label='1:1, R$^2$ = '+str(np.round(NEE_2018_30d_slope1_R2,2)))
#plt.legend(loc='lower right')
#plt.xlabel('SMUrF NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
#plt.ylabel('UrbanVPRM NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
#plt.title('SMUrF vs. UrbanVPRM 30d avg NEE, Toronto')
## *** Uncomment to save figure as png and pdf. CHANGE FILENAMES ***
#plt.savefig('Toronto_Fixed_SMUrF_vs_UrbanVPRM_NEE_30d_Huber_no_errs_correlation_subselection.pdf',bbox_inches='tight')
#plt.savefig('Toronto_Fixed_SMUrF_vs_UrbanVPRM_NEE_30d_Huber_no_errs_correlation_subselection.png',bbox_inches='tight')
#plt.show()


# In[ ]:





# In[27]:


# Take the annual average of SMUrF and UrbanVPRM NEE data over Toronto
VPRM_NEE_2018_avg=np.nanmean(T_VPRM_NEE,axis=0)
S_NEE_2018_avg=np.nanmean(T_S_NEE,axis=0)


# In[28]:


# Apply a 1000x bootstrapped Huber fit to the annual Toronto NEE data
finitemask0=np.isfinite(S_NEE_2018_avg) & np.isfinite(VPRM_NEE_2018_avg)
S_NEE_2018_avg_clean0=S_NEE_2018_avg[finitemask0]
VPRM_NEE_2018_avg_clean0=VPRM_NEE_2018_avg[finitemask0]

Huber_2018_avg_slps=[]
Huber_2018_avg_ints=[]
Huber_2018_avg_R2=[]

#try bootstrapping
indx_2018_avg_list=list(range(0,len(S_NEE_2018_avg_clean0)))
for i in range(1,1000):
    #sub selection of points
    S_NEE_2018_avg_indx=np.random.choice(indx_2018_avg_list,size=len(indx_2018_avg_list))
    
    Huber_model = linear_model.HuberRegressor()
    Huber_fit=Huber_model.fit((S_NEE_2018_avg_clean0[S_NEE_2018_avg_indx]).reshape(-1,1),VPRM_NEE_2018_avg_clean0[S_NEE_2018_avg_indx])
    H_m=Huber_fit.coef_
    H_c=Huber_fit.intercept_
    x_accpt, y_accpt = S_NEE_2018_avg_clean0, VPRM_NEE_2018_avg_clean0
    y_predict = H_m * x_accpt + H_c
    H_R2=r2_score(y_accpt, y_predict)
    Huber_2018_avg_slps.append(H_m)
    Huber_2018_avg_ints.append(H_c)
    Huber_2018_avg_R2.append(H_R2)
    if int(i)%100==0:
        print(i/1000*100)


# In[29]:


#Compute R2 using average Huber fit for annual data
Huber_ann_R2=r2_score(VPRM_NEE_2018_avg_clean0[(S_NEE_2018_avg_clean0!=0) & (VPRM_NEE_2018_avg_clean0!=0)],S_NEE_2018_avg_clean0[(S_NEE_2018_avg_clean0!=0) & (VPRM_NEE_2018_avg_clean0!=0)]*np.nanmean(Huber_2018_avg_slps)+np.nanmean(Huber_2018_avg_ints))
#Huber_ann_R2
print('Annual average Huber fit R^2 = '+str(np.round(Huber_ann_R2,5)))


# In[47]:


#Compute R2 using average Huber fit for annual data
NEE_2018_avg_slope1_R2=r2_score(VPRM_NEE_2018_avg_clean0,S_NEE_2018_avg_clean0)
#NEE_2018_avg_slope1_R2
print('Annual average fit 1:1 R^2 = '+str(np.round(NEE_2018_avg_slope1_R2,5)))


# In[46]:


#Sanity check (calculate 1:1 R2 by hand)
#sum_tot = np.sum((VPRM_NEE_2018_avg_clean0-np.mean(VPRM_NEE_2018_avg_clean0))**2)
#sum_res= np.sum((VPRM_NEE_2018_avg_clean0-func2(S_NEE_2018_avg_clean0,1,0))**2)
#print(1-sum_res/sum_tot)


# In[37]:


# *** Optional: Uncomment to plot correlation of annual-averaged data ***
##all points
#plt.rc('font',size=14)

#plt.figure(figsize=(8,6))
#plt.xlim(-3,1)
#plt.ylim(-3,1)
#plt.axis('scaled')
#plt.scatter(S_NEE_2018_avg_clean0[(S_NEE_2018_avg_clean0!=0) & (VPRM_NEE_2018_avg_clean0!=0)],VPRM_NEE_2018_avg_clean0[(S_NEE_2018_avg_clean0!=0) & (VPRM_NEE_2018_avg_clean0!=0)],s=1)
#plt.plot(line1_1,line1_1*np.nanmean(Huber_2018_avg_slps)+np.nanmean(Huber_2018_avg_ints),linestyle='--',label=str(np.round(np.nanmean(Huber_2018_avg_slps),2))+'x + '+str(np.round(np.nanmean(Huber_2018_avg_ints),2))+', R$^2$ = '+str(np.round(Huber_ann_R2,2)),c='#FF800E')
#plt.plot(line1_1,line1_1,linestyle=':',c='k',label='1:1, R$^2$ = '+str(np.round(NEE_2018_avg_slope1_R2,2)))
#plt.legend(loc='lower right')
#plt.xlabel('SMUrF NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
#plt.ylabel('UrbanVPRM NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
#plt.title('SMUrF vs. UrbanVPRM 2018 Average NEE, Toronto')
## *** Uncomment to save figure as png and pdf. CHANGE FILENAME ***
#plt.savefig('Toronto_Fixed_SMUrF_vs_UrbanVPRM_NEE_2018_avg_Huber_no_errs_correlation.pdf',bbox_inches='tight')
#plt.savefig('Toronto_Fixed_SMUrF_vs_UrbanVPRM_NEE_2018_avg_Huber_no_errs_correlation.png',bbox_inches='tight')
#plt.show()


# In[ ]:





# In[59]:


plt.rc('font',size=24)

fig, ax = plt.subplots(2,2,sharex=True,sharey=True,figsize=(16,16))
ax[0,0].set_xlim(-25,8)
ax[0,0].set_ylim(-25,8)

ax[0,0].scatter(T_S_NEE_clean0[(T_S_NEE_clean0!=0) & (T_VPRM_NEE_clean0!=0)][S_NEE_indx],T_VPRM_NEE_clean0[(T_S_NEE_clean0!=0) & (T_VPRM_NEE_clean0!=0)][S_NEE_indx],s=1)
ax[0,0].plot(line1_1,line1_1*np.nanmean(Huber_2018_slps)+np.nanmean(Huber_2018_ints),linewidth=2,linestyle='--',label=str(np.round(np.nanmean(Huber_2018_slps),2))+'x + '+str(np.round(np.nanmean(Huber_2018_ints),2))+', R$^2$ = '+str(np.round(Huber_avg_R2,2)),c='#FF800E')
ax[0,0].plot(line1_1,line1_1,linestyle=':',c='k',label='1:1, R$^2$ = '+str(np.round(NEE_2018_slope1_R2,2)))
ax[0,0].legend(loc=(0.205,0.12),fontsize=22)
ax[0,0].set_xlabel('SMUrF NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
ax[0,0].set_ylabel('UrbanVPRM NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
ax[0,0].set_title('Hourly NEE')

ax[0,1].scatter(S_NEE_daily_clean0[(S_NEE_daily_clean0!=0) & (VPRM_NEE_daily_clean0!=0)][S_NEE_dly_indx],VPRM_NEE_daily_clean0[(S_NEE_daily_clean0!=0) & (VPRM_NEE_daily_clean0!=0)][S_NEE_dly_indx],s=1)
ax[0,1].plot(line1_1,line1_1*np.nanmean(Huber_dly_slps)+np.nanmean(Huber_dly_ints),linewidth=2,linestyle='--',label=str(np.round(np.nanmean(Huber_dly_slps),2))+'x + '+str(np.round(np.nanmean(Huber_dly_ints),2))+', R$^2$ = '+str(np.round(Huber_dly_avg_R2,2)),c='#FF800E')

ax[0,1].plot(line1_1,line1_1,linestyle=':',c='k',label='1:1, R$^2$ = '+str(np.round(NEE_2018_daily_slope1_R2,2)))
ax[0,1].legend(loc=(0.205,0.12),fontsize=22)
ax[0,1].set_xlabel('SMUrF NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
ax[0,1].set_title('Daily Average NEE')

ax[1,0].scatter(S_NEE_30d_clean0[(S_NEE_30d_clean0!=0) & (VPRM_NEE_30d_clean0!=0)][S_NEE_30d_indx],VPRM_NEE_30d_clean0[(S_NEE_30d_clean0!=0) & (VPRM_NEE_30d_clean0!=0)][S_NEE_30d_indx],s=1)
ax[1,0].plot(line1_1,line1_1*np.nanmean(Huber_30d_slps)+np.nanmean(Huber_30d_ints),linewidth=2,linestyle='--',label=str(np.round(np.nanmean(Huber_30d_slps),2))+'x + '+str(np.round(np.nanmean(Huber_30d_ints),2))+', R$^2$ = '+str(np.round(Huber_30d_avg_R2,2)),c='#FF800E')

ax[1,0].plot(line1_1,line1_1,linestyle=':',c='k',label='1:1, R$^2$ = '+str(np.round(NEE_2018_30d_slope1_R2,2)))
ax[1,0].legend(loc='lower right',fontsize=22)
ax[1,0].set_ylabel('UrbanVPRM NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
ax[1,0].set_xlabel('SMUrF NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
ax[1,0].set_title('Monthly Average NEE')

ax[1,1].scatter(S_NEE_2018_avg_clean0[(S_NEE_2018_avg_clean0!=0) & (VPRM_NEE_2018_avg_clean0!=0)],VPRM_NEE_2018_avg_clean0[(S_NEE_2018_avg_clean0!=0) & (VPRM_NEE_2018_avg_clean0!=0)],s=1)
ax[1,1].plot(line1_1,line1_1*np.nanmean(Huber_2018_avg_slps)+np.nanmean(Huber_2018_avg_ints),linestyle='--',label=str(np.round(np.nanmean(Huber_2018_avg_slps),2))+'x - '+str(np.round(-np.nanmean(Huber_2018_avg_ints),2))+', R$^2$ = '+str(np.round(Huber_ann_R2,2)),c='#FF800E')
ax[1,1].plot(line1_1,line1_1,linestyle=':',c='k',label='1:1, R$^2$ = '+str(np.round(NEE_2018_avg_slope1_R2,2)))
ax[1,1].legend(loc='lower right',fontsize=22)
ax[1,1].set_xlabel('SMUrF NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
ax[1,1].set_title('Annual Average NEE')

ax[0,0].text(-24.5,6,'(a)',c='k',fontsize=24)
ax[0,1].text(-24.5,6,'(b)',c='k',fontsize=24)
ax[1,0].text(-24.5,6,'(c)',c='k',fontsize=24)
ax[1,1].text(-24.5,6,'(d)',c='k',fontsize=24)

fig.subplots_adjust(hspace=0,wspace=0)
# Uncomment to save figures as .png and .pdf *** CHANGE PATHS & FILENAMES ***
plt.savefig('Toronto_Fixed_SMUrF_UrbanVPRM_NEE_Huber_correlation_hrly_dly_30d_annual_less_data_labelled.pdf',bbox_inches='tight')
plt.savefig('Toronto_Fixed_SMUrF_UrbanVPRM_NEE_Huber_correlation_hrly_dly_30d_annual_less_data_labelled.png',bbox_inches='tight')
fig.show()


# In[ ]:





# In[ ]:




