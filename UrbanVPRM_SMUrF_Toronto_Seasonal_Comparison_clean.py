#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This code is used to compare spatial patterns of the updated UrbanVPRM and SMUrF model both before and after the ISA
# adjustment to the SMUrF model, over the city of Toronto, Canada.
# We also compare the temporal variation of SMUrF and UrbanVPRM over locations with varying urban intensity in the city of
# Toronto, Canada.

# This code reproduces figures 5.d-f, 6, and S5 of Madsen-Colford et al. 2025.
# If code is used please cite

# Portions of the code that should be modified by the user (e.g. path names) are denoted above by ***


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from scipy import optimize as opt 
from scipy import odr
from scipy import stats
import shapefile as shp # to import outline of GTA
from shapely import geometry # used to define a polygon for Toronto
import netCDF4
from netCDF4 import Dataset, date2num #for reading netCDF data files and their date (not sure if I need the later)


# In[2]:


#Load in VPRM data over Toronto

# *** Change path ***
VPRM_path = 'E:/Research/UrbanVPRM/dataverse_files/GTA_V061_500m_2018/'
# *** Change filename ***
VPRM_fn = 'vprm_GMIS_Toronto_ACI_SOLRIS_ISA_500m_GTA_V061_2018_no_PScale_adjusted_Topt_Ra_URB_parameters_fixed_gapfilled_LSWI_filtered_bilinear_PAR_block_'
VPRM_data=pd.read_csv(VPRM_path+VPRM_fn+'00000001.csv').loc[:,('HoY','Index','GEE','Re')]


# In[3]:


VPRM_data2=pd.read_csv(VPRM_path+VPRM_fn+'00002501.csv').loc[:,('HoY','Index','GEE','Re')]
VPRM_data=VPRM_data.append(VPRM_data2)
del VPRM_data2


# In[ ]:


VPRM_data2=pd.read_csv(VPRM_path+VPRM_fn+'00005001.csv').loc[:,('HoY','Index','GEE','Re')]
VPRM_data=VPRM_data.append(VPRM_data2)
del VPRM_data2


# In[ ]:


VPRM_data2=pd.read_csv(VPRM_path+VPRM_fn+'00007501.csv').loc[:,('HoY','Index','GEE','Re')]
VPRM_data=VPRM_data.append(VPRM_data2)
del VPRM_data2


# In[ ]:


VPRM_data2=pd.read_csv(VPRM_path+VPRM_fn+'00010001.csv').loc[:,('HoY','Index','GEE','Re')]
VPRM_data=VPRM_data.append(VPRM_data2)
del VPRM_data2


# In[ ]:


VPRM_data2=pd.read_csv(VPRM_path+VPRM_fn+'00012501.csv').loc[:,('HoY','Index','GEE','Re')]
VPRM_data=VPRM_data.append(VPRM_data2)
del VPRM_data2


# In[ ]:


#Load in index, x, & y data

# *** Change path & file name ***
VPRM_EVI=pd.read_csv('E:/Research/UrbanVPRM/dataverse_files/GTA_V061_500m_2018/adjusted_evi_lswi_interpolated_modis_v061_qc_filtered_LSWI_filtered.csv').loc[:,('DOY','Index','x','y')]

#Create a dataframe with just Index, x, & y values
x=np.zeros(np.shape(VPRM_EVI.Index.unique()))*np.nan
y=np.zeros(np.shape(VPRM_EVI.Index.unique()))*np.nan
ind=np.zeros(np.shape(VPRM_EVI.Index.unique()))*np.nan
for i in range(len(VPRM_EVI.Index.unique())):
    x[i]=VPRM_EVI.x[0+i*365]
    y[i]=VPRM_EVI.y[0+i*365]
    ind[i]=VPRM_EVI.Index[0+i*365]

del VPRM_EVI

# Combine VPRM flux data with x & y data based on index
VPRM_xy=pd.DataFrame({'Index':ind, 'x':x, 'y':y})
VPRM_data=VPRM_data.merge(VPRM_xy[['Index','x','y']])
del VPRM_xy, x, y ,ind


# In[ ]:


# Get all unique x and y values
xvals = VPRM_data.x[VPRM_data.HoY==4800].unique()
yvals = VPRM_data.y[VPRM_data.HoY==4800].unique()
extent = np.min(xvals), np.max(xvals), np.min(yvals), np.max(yvals)


# In[ ]:


#Reshape the VPRM data into an array with dimensions of y,x, and hour of year
GPP=-VPRM_data.GEE.values.reshape(len(yvals),len(xvals),8760)#8784 for leap year
Reco=VPRM_data.Re.values.reshape(len(yvals),len(xvals),8760)
HoY=VPRM_data.HoY.values.reshape(len(yvals),len(xvals),8760)
DoY=np.mean(HoY,axis=(0,1))/24+23/24 # create an array of fractional day of year


# In[ ]:





# In[ ]:


#Import Toronto shape file

# *** Change Path ***
sf = shp.Reader("C:/Users/kitty/Documents/Research/SIF/Shape_files/Toronto/Toronto_Boundary.shp")
#Toronto_Shape
shape=sf.shape(0)
Toronto_x = np.zeros((len(shape.points),1))*np.nan #Make arrays of x & y data
Toronto_y = np.zeros((len(shape.points),1))*np.nan
for i in range(len(shape.points)):
    Toronto_x[i]=shape.points[i][0]
    Toronto_y[i]=shape.points[i][1]
    
points=[]
for k in range(1,len(Toronto_x)):
    points.append(geometry.Point(Toronto_x[k],Toronto_y[k])) #convert x & y data to points
poly=geometry.Polygon([[p.x, p.y] for p in points]) #convert points to polygons

#Create a mask for areas outside of Toronto
lons=np.ones(144)*np.nan
lats=np.ones(96)*np.nan
GPP_mask=np.ones([96,144])*np.nan
for i in range(0, len(lons)):
    for j in range(0, len(lats)):
        if poly.contains(geometry.Point([xvals[i],yvals[j]])):
            lons[i]=xvals[i]
            lats[j]=yvals[j]
            GPP_mask[j,i]=1


# In[ ]:





# In[ ]:


# Swap the axes of VPRM fluxes to match that of SMUrF & apply Toronto mask
VPRM_GPP=(np.swapaxes(np.swapaxes(GPP,0,2),1,2))*GPP_mask[np.newaxis,:,:]
VPRM_Reco=(np.swapaxes(np.swapaxes(Reco,0,2),1,2))*GPP_mask[np.newaxis,:,:]
VPRM_NEE=(np.swapaxes(np.swapaxes(Reco,0,2),1,2)-np.swapaxes(np.swapaxes(GPP,0,2),1,2))*GPP_mask[np.newaxis,:,:]


# In[ ]:


#Take 8-day average of UrbanVPRM fluxes to match SMUrF 8-day fluxes

VPRM_GPP_8day=np.ones((46, 96, 144))*np.nan
VPRM_Reco_8day=np.ones((46, 96, 144))*np.nan
VPRM_NEE_8day=np.ones((46, 96, 144))*np.nan
for i in range(46):
    VPRM_GPP_8day[i]=np.nanmean(VPRM_GPP[i*8*24:i*8*24+8*24],axis=0)
    VPRM_Reco_8day[i]=np.nanmean(VPRM_Reco[i*8*24:i*8*24+8*24],axis=0)
    VPRM_NEE_8day[i]=np.nanmean(VPRM_NEE[i*8*24:i*8*24+8*24],axis=0)


# In[ ]:


#now bring in the SMUrF data with ISA adjustment AND shoreline correction AND downscaling fix

# *** Change path ***
SMUrF_path = 'C:/Users/kitty/Documents/Research/SIF/SMUrF/output2018_500m_CSIF_to_TROPOMI_CSIF_ALL_converted_slps_V3_temp_impervious_R_shore_corr_V061_8day/easternCONUS/'

# *** Change file name ***
SMUrF_fn = 'daily_mean_Reco_ISA_a_neuralnet/era5/2018/daily_mean_Reco_uncert_GMIS_Toronto_t_easternCONUS_2018' #file name WITHOUT the month and day

#Bring in the first day to get the start of year (in seconds since 1970)
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
        
# *** Change GPP file name ***
f=Dataset(SMUrF_path+'daily_mean_SIF_GPP_uncert_easternCONUS_2018.nc')
S_time_8day=f.variables['time'][:]/24/3600-start_of_year-5/24 #convert seconds since 1970 to days and subtract start of year

S_GPP_err_8day=f.variables['GPP_sd'][:]
S_GPP_8day=f.variables['GPP_mean'][:]

S_Reco[S_Reco==-999]=np.nan
S_Reco_err[S_Reco_err==-999]=np.nan
S_GPP_8day[S_GPP_8day==-999]=np.nan
S_GPP_err_8day[S_GPP_err_8day==-999]=np.nan

S_Reco_8day=np.ones(np.shape(S_GPP_8day))*np.nan
S_Reco_err_8day=np.ones(np.shape(S_GPP_8day))*np.nan
S_Reco_std_8day=np.ones(np.shape(S_GPP_8day))*np.nan
for i in range(len(S_time_8day)):
    S_Reco_8day[i]=np.nanmean(S_Reco[i*8:i*8+8],axis=0)
    S_Reco_err_8day[i]=np.sqrt(np.nansum((S_Reco_err[i*8:i*8+8]/4)**2,axis=0))
    S_Reco_std_8day[i]=np.nanstd(S_Reco[i*8:i*8+8],axis=0)
    
S_NEE_8day=S_Reco_8day-S_GPP_8day
S_NEE_err_8day=np.sqrt(S_Reco_err_8day**2+S_GPP_err_8day**2)


# In[ ]:


# Select only data over Toronto
S_GPP_8day=S_GPP_8day[:,264:360,288:432]
S_Reco_8day=S_Reco_8day[:,264:360,288:432]
S_NEE_8day=S_NEE_8day[:,264:360,288:432]


# In[ ]:





# In[ ]:





# In[ ]:


#WITH downscaling fix

#now bring in the SMUrF data WITHOUT ISA adjustment AND shoreline correction

# *** Change path ***
SMUrF_path_noISA = 'E:/Research/SMUrF/output2018_500m_CSIF_to_TROPOMI_CSIF_ALL_converted_slps_V3_temp_impervious_R_shore_corr_V061_8day/easternCONUS/'
# *** Change filename ***
SMUrF_fn_noISA = 'daily_mean_Reco_no_ISA_neuralnet/era5/2018/daily_mean_Reco_uncert_no_ISA_easternCONUS_2018' #file name WITHOUT the month and day

#Bring in the first day to get the start of year (in seconds since 1970)
g=Dataset(SMUrF_path_noISA+SMUrF_fn_noISA+'0101.nc')
start_of_year=g.variables['time'][0]/3600/24-1 #convert seconds since 1970 to days (minus one)
g.close()

#With ISA adjustment using GMIS-Toronto-SOLRIS-ACI dataset
S_noISA_time=[]
S_noISA_Reco=[]
S_noISA_Reco_err=[]
S_noISA_lats_8day=[]
S_noISA_lons_8day=[]
for j in range(1,13):
    for i in range(1,32):
        try:
            if j<10:
                if i<10:
                    f=Dataset(SMUrF_path_noISA+SMUrF_fn_noISA+'0'+str(j)+'0'+str(i)+'.nc')
                else:
                    f=Dataset(SMUrF_path_noISA+SMUrF_fn_noISA+'0'+str(j)+str(i)+'.nc')
            else:
                if i<10:
                    f=Dataset(SMUrF_path_noISA+SMUrF_fn_noISA+str(j)+'0'+str(i)+'.nc')
                else:
                    f=Dataset(SMUrF_path_noISA+SMUrF_fn_noISA+str(j)+str(i)+'.nc')
            if len(S_noISA_time)==0:
                S_noISA_lats_8day=f.variables['lat'][:]
                S_noISA_lons_8day=f.variables['lon'][:]
                S_noISA_Reco=f.variables['Reco_mean'][:]
                S_noISA_Reco_err=f.variables['Reco_sd'][:]
                S_noISA_time=f.variables['time'][:]/24/3600-start_of_year-5/24 #convert seconds since 1970 to days and subtract start of year
            else:
                S_noISA_Reco=np.concatenate((S_noISA_Reco,f.variables['Reco_mean'][:]),axis=0)
                S_noISA_Reco_err=np.concatenate((S_noISA_Reco_err,f.variables['Reco_sd'][:]),axis=0)
                S_noISA_time=np.concatenate((S_noISA_time,(f.variables['time'][:]/24/3600-start_of_year-5/24)),axis=0)
            f.close()
        except FileNotFoundError:
            pass
        
# *** Change filename ***
f=Dataset(SMUrF_path+'daily_mean_SIF_GPP_uncert_easternCONUS_2018.nc')
S_noISA_time_8day=f.variables['time'][:]/24/3600-start_of_year-5/24 #convert seconds since 1970 to days and subtract start of year

S_noISA_GPP_err_8day=f.variables['GPP_sd'][:]
S_noISA_GPP_8day=f.variables['GPP_mean'][:]

S_noISA_Reco[S_noISA_Reco==-999]=np.nan
S_noISA_Reco_err[S_noISA_Reco_err==-999]=np.nan
S_noISA_GPP_8day[S_noISA_GPP_8day==-999]=np.nan
S_noISA_GPP_err_8day[S_noISA_GPP_err_8day==-999]=np.nan

S_noISA_Reco_8day=np.ones(np.shape(S_noISA_GPP_8day))*np.nan
S_noISA_Reco_err_8day=np.ones(np.shape(S_noISA_GPP_8day))*np.nan
S_noISA_Reco_std_8day=np.ones(np.shape(S_noISA_GPP_8day))*np.nan
for i in range(len(S_noISA_time_8day)):
    S_noISA_Reco_8day[i]=np.nanmean(S_noISA_Reco[i*8:i*8+8],axis=0)
    S_noISA_Reco_err_8day[i]=np.sqrt(np.nansum((S_noISA_Reco_err[i*8:i*8+8]/4)**2,axis=0))
    S_noISA_Reco_std_8day[i]=np.nanstd(S_noISA_Reco[i*8:i*8+8],axis=0)
    
S_noISA_NEE_8day=S_noISA_Reco_8day-S_noISA_GPP_8day
S_noISA_NEE_err_8day=np.sqrt(S_noISA_Reco_err_8day**2+S_noISA_GPP_err_8day**2)


# In[ ]:


# Select only data over Toronto
S_noISA_GPP_8day=S_noISA_GPP_8day[:,264:360,288:432]
S_noISA_Reco_8day=S_noISA_Reco_8day[:,264:360,288:432]
S_noISA_NEE_8day=S_noISA_NEE_8day[:,264:360,288:432]


# In[ ]:


VPRM_GPP_JJA_8day=VPRM_GPP_8day[(np.round(S_noISA_time_8day,5)>=152) & (np.round(S_noISA_time_8day,5)<244)]
VPRM_Reco_JJA_8day=VPRM_Reco_8day[(np.round(S_noISA_time_8day,5)>=152) & (np.round(S_noISA_time_8day,5)<244)]
VPRM_NEE_JJA_8day=VPRM_NEE_8day[(np.round(S_noISA_time_8day,5)>=152) & (np.round(S_noISA_time_8day,5)<244)]


# In[ ]:





# In[ ]:


# WITH downscaling & MODIS shift fix
## WITHOUT ISA correction

#JJA: Doy 60 - 151 inclusive
S_noISA_JJA_time=S_noISA_time_8day[(np.round(S_noISA_time_8day,5)>=152) & (np.round(S_noISA_time_8day,5)<244)]
S_noISA_GPP_JJA=S_noISA_GPP_8day[(np.round(S_noISA_time_8day,5)>=152) & (np.round(S_noISA_time_8day,5)<244)]
S_noISA_Reco_JJA=S_noISA_Reco_8day[(np.round(S_noISA_time_8day,5)>=152) & (np.round(S_noISA_time_8day,5)<244)]
S_noISA_NEE_JJA=S_noISA_NEE_8day[(np.round(S_noISA_time_8day,5)>=152) & (np.round(S_noISA_time_8day,5)<244)]


VPRM_avg_JJA= np.zeros(len(S_noISA_JJA_time))*np.nan
S_noISA_avg_JJA= np.zeros(len(S_noISA_JJA_time))*np.nan

for i in range(len(VPRM_NEE_JJA_8day)):
    # *** Uncomment to visualize NEE from updated UrbanVPRM and SMUrF model without ISA correction 
    #     for each 8-day period in the summer ***
    
    #fig, ax = plt.subplots(1,3,sharex=True,sharey=True,figsize=(24,6))
    #ax[0].set_xlim(-79.69,-79.06)
    #ax[0].set_ylim(43.5,43.9)

    #fig0=ax[0].imshow(S_noISA_NEE_JJA[i][::-1]*GPP_mask,extent= extent,cmap='bwr',vmin=-8,vmax=8)
    #ax[0].plot(Toronto_x,Toronto_y,c='k')
    #ax[0].text(-79.6,43.51, 'Average NEE = '+str(np.round(np.nanmean(S_noISA_NEE_JJA[i][::-1]*GPP_mask),3))+' $\mu$mol/m$^2$/s',fontsize=14)
    #ax[0].set_title('SMUrF no ISA adjustment Toronto NEE')

    #fig1=ax[1].imshow(VPRM_NEE_JJA_8day[i],extent= extent,cmap='bwr',vmin=-8,vmax=8)
    #ax[1].plot(Toronto_x,Toronto_y,c='k')
    #ax[1].text(-79.6,43.51, 'Average NEE = '+str(np.round(np.nanmean(VPRM_NEE_JJA_8day[i]*GPP_mask),3))+' $\mu$mol/m$^2$/s',fontsize=14)
    #ax[1].set_title('UrbanVPRM Toronto NEE, DoY: '+str(np.round(S_noISA_JJA_time[i])))

    #fig2=ax[2].imshow(S_noISA_NEE_JJA[i][::-1]-VPRM_NEE_JJA_8day[i],extent= extent,cmap='bwr',vmin=-8,vmax=8)
    #ax[2].plot(Toronto_x,Toronto_y,c='k')
    #ax[2].text(-79.6,43.51, 'SMUrF - UrbanVPRM = '+str(np.round(np.nanmean(S_noISA_NEE_JJA[i][::-1]*GPP_mask)-np.nanmean(VPRM_NEE_JJA_8day[i]),3))+' $\mu$mol/m$^2$/s',fontsize=14)
    #ax[2].set_title('SMUrF no ISA adjustment - UrbanVPRM Toronto NEE')

    #cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    #cbar=fig.colorbar(fig2,cax=cbar_ax)
    #cbar.set_label('NEE (MgCO$_2$ ha$^{-1}$ yr$^{-1}$)')

    #ax[0].set_ylabel('Latitude ($^o$)')

    #ax[0].set_xlabel('Longitude ($^o$)')
    #ax[1].set_xlabel('Longitude ($^o$)')
    #ax[2].set_xlabel('Longitude ($^o$)')
    
    #fig.subplots_adjust(hspace=0,wspace=0)
    #fig.show()
    # *** End of optional uncomment ***
    
    VPRM_avg_JJA[i]=np.nanmean(VPRM_NEE_JJA_8day[i]*GPP_mask)
    S_noISA_avg_JJA[i]=np.nanmean(S_noISA_NEE_JJA[i][::-1]*GPP_mask)
    
    # *** Optional uncomment to plot ratio of VPRM/SMUrF(noISA)
    #print('VPRM/SMUrF: '+str(VPRM_avg_JJA[i]/S_noISA_avg_JJA[i]))


# In[ ]:





# In[ ]:





# In[ ]:


# With downscaling & MODIS shift fix

# Using SMAPE: (|SMUrF-VPRM|)/((|SMUrF|+|VPRM|)/2)
print('Summer Updated |SMUrF (without ISA adjustment) - VPRM| % difference: '+str(np.round(np.nanmean(np.abs(S_noISA_avg_JJA-VPRM_avg_JJA)/((np.abs(S_noISA_avg_JJA)+np.abs(VPRM_avg_JJA))/2))*100,3))+' +/- '+str(np.round(np.nanstd(np.abs(S_noISA_avg_JJA-VPRM_avg_JJA)/((np.abs(S_noISA_avg_JJA)+np.abs(VPRM_avg_JJA))/2))*100,3))+' %')


# In[ ]:


#WITHOUT ISA adjustment
#JJA: Day of year 152 - 243, inclusive
S_noISA_JJA_time=S_noISA_time_8day[(np.round(S_noISA_time_8day,5)>=152) & (np.round(S_noISA_time_8day,5)<244)]
S_noISA_GPP_JJA=S_noISA_GPP_8day[(np.round(S_noISA_time_8day,5)>=152) & (np.round(S_noISA_time_8day,5)<244)]
S_noISA_Reco_JJA=S_noISA_Reco_8day[(np.round(S_noISA_time_8day,5)>=152) & (np.round(S_noISA_time_8day,5)<244)]
S_noISA_NEE_JJA=S_noISA_NEE_8day[(np.round(S_noISA_time_8day,5)>=152) & (np.round(S_noISA_time_8day,5)<244)]


## *** Optional: Uncomment to plot UrbanVPRM & SMUrF model, without ISA adjustment, NEE for the first week of July, 2018
#plt.rc('font',size=20)
#fig, ax = plt.subplots(1,3,sharex=True,sharey=True,figsize=(24,6))
#ax[0].set_xlim(-79.69,-79.06)
#ax[0].set_ylim(43.5,43.9)

#fig0=ax[0].imshow(S_noISA_NEE_JJA[4][::-1]*GPP_mask,extent= extent,cmap='bwr',vmin=-8,vmax=8)
#ax[0].plot(Toronto_x,Toronto_y,c='k')
#ax[0].text(-79.63,43.51, 'Average NEE = '+str(np.round(np.nanmean(S_noISA_NEE_JJA[4][::-1]*GPP_mask),3))+' $\mu$mol m$^{-2}$ s$^{-1}$')
#ax[0].set_title('No ISA SMUrF Toronto NEE')
    
#fig1=ax[1].imshow(VPRM_NEE_JJA_8day[4],extent= extent,cmap='bwr',vmin=-8,vmax=8)
#ax[1].plot(Toronto_x,Toronto_y,c='k')
#ax[1].text(-79.63,43.51, 'Average NEE = '+str(np.round(np.nanmean(VPRM_NEE_JJA_8day[4]),3))+' $\mu$mol m$^{-2}$ s$^{-1}$')
#ax[1].set_title('UrbanVPRM Toronto NEE')

#fig2=ax[2].imshow(S_noISA_NEE_JJA[4][::-1]-VPRM_NEE_JJA_8day[4],extent= extent,cmap='bwr',vmin=-8,vmax=8)
#ax[2].plot(Toronto_x,Toronto_y,c='k')
#ax[2].text(-79.67,43.51, 'UrbanVPRM - SMUrF = '+str(np.round(np.nanmean(VPRM_NEE_JJA_8day[4])-np.nanmean(S_noISA_NEE_JJA[4][::-1]*GPP_mask),3))+' $\mu$mol m$^{-2}$ s$^{-1}$')

#ax[2].set_title('SMUrF - UrbanVPRM Toronto NEE')

#cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
#cbar=fig.colorbar(fig2,cax=cbar_ax)
#cbar.set_label('NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')

#ax[0].text(-79.68,43.865,'(a)',c='k',fontsize=26)
#ax[1].text(-79.68,43.865,'(b)',c='k',fontsize=26)
#ax[2].text(-79.68,43.865,'(c)',c='k',fontsize=26)

#ax[0].set_ylabel('Latitude ($^o$)')

#ax[0].set_xlabel('Longitude ($^o$)')
#ax[1].set_xlabel('Longitude ($^o$)')
#ax[2].set_xlabel('Longitude ($^o$)')

#fig.subplots_adjust(hspace=0,wspace=0)
#fig.show()

# *** End of optional uncomment


# In[ ]:





# In[ ]:


## WITH ISA correction

#JJA: Doy 60 - 151 inclusive
S_JJA_time=S_time_8day[(np.round(S_time_8day,5)>=152) & (np.round(S_time_8day,5)<244)]
S_GPP_JJA=S_GPP_8day[(np.round(S_time_8day,5)>=152) & (np.round(S_time_8day,5)<244)]
S_Reco_JJA=S_Reco_8day[(np.round(S_time_8day,5)>=152) & (np.round(S_time_8day,5)<244)]
S_NEE_JJA=S_NEE_8day[(np.round(S_time_8day,5)>=152) & (np.round(S_time_8day,5)<244)]

S_avg_JJA= np.zeros(len(S_JJA_time))*np.nan

for i in range(len(VPRM_NEE_JJA_8day)):
    ## *** Optionally uncomment to plot each 8-day time period for SMUrF & UrbanVPRM & their difference
    
    #fig, ax = plt.subplots(1,3,sharex=True,sharey=True,figsize=(24,6))
    #ax[0].set_xlim(-79.69,-79.06)
    #ax[0].set_ylim(43.5,43.9)

    #fig0=ax[0].imshow(S_NEE_JJA[i][::-1]*GPP_mask,extent= extent,cmap='bwr',vmin=-8,vmax=8)
    #ax[0].plot(Toronto_x,Toronto_y,c='k')
    #ax[0].text(-79.6,43.51, 'Average NEE = '+str(np.round(np.nanmean(S_NEE_JJA[i][::-1]*GPP_mask),3))+' $\mu$mol/m$^2$/s',fontsize=14)
    #ax[0].set_title('Updated SMUrF Toronto NEE')

    #fig1=ax[1].imshow(VPRM_NEE_JJA_8day[i],extent= extent,cmap='bwr',vmin=-8,vmax=8)
    #ax[1].plot(Toronto_x,Toronto_y,c='k')
    #ax[1].text(-79.6,43.51, 'Average NEE = '+str(np.round(np.nanmean(VPRM_NEE_JJA_8day[i]*GPP_mask),3))+' $\mu$mol/m$^2$/s',fontsize=14)
    #ax[1].set_title('Updated UrbanVPRM Toronto NEE')

    #fig2=ax[2].imshow(S_NEE_JJA[i][::-1]-VPRM_NEE_JJA_8day[i],extent= extent,cmap='bwr',vmin=-8,vmax=8)
    #ax[2].plot(Toronto_x,Toronto_y,c='k')
    #ax[2].text(-79.6,43.51, 'SMUrF - UrbanVPRM = '+str(np.round(np.nanmean(S_NEE_JJA[i][::-1]*GPP_mask)-np.nanmean(VPRM_NEE_JJA_8day[i]),3))+' $\mu$mol/m$^2$/s',fontsize=14)
    #ax[2].set_title('SMUrF - UrbanVPRM Toronto NEE')

    #cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    #cbar=fig.colorbar(fig2,cax=cbar_ax)
    #cbar.set_label('NEE (MgCO$_2$ ha$^{-1}$ yr$^{-1}$)')

    #ax[0].set_ylabel('Latitude ($^o$)')

    #ax[0].set_xlabel('Longitude ($^o$)')
    #ax[1].set_xlabel('Longitude ($^o$)')
    #ax[2].set_xlabel('Longitude ($^o$)')
    
    #fig.subplots_adjust(hspace=0,wspace=0)
    #fig.show()
    ## *** End of uncomment

    S_avg_JJA[i]=np.nanmean(S_NEE_JJA[i][::-1]*GPP_mask)
    ## *** Optional uncomment to print (SMUrF-VPRM)/|SMUrF|:
    #print('(SMUrF-VPRM)/|SMUrF|: '+str((S_avg_JJA[i]-VPRM_avg_JJA[i])/np.abs(S_avg_JJA[i]))) 


# In[ ]:


#Updated SMUrF with ISA adjustment

# SMAPE: |SMUrF-VPRM|/((|SMUrF|+|VPRM|)/2)
print('Summer Updated |SMUrF - VPRM| % difference: '+str(np.round(np.nanmean(np.abs(S_avg_JJA-VPRM_avg_JJA)/((np.abs(S_avg_JJA)+np.abs(VPRM_avg_JJA))/2))*100,3))+' +/- '+str(np.round(np.nanstd(np.abs(S_avg_JJA-VPRM_avg_JJA)/((np.abs(S_avg_JJA)+np.abs(VPRM_avg_JJA))/2))*100,3))+' %')


# In[ ]:


# Figure S5 of Madsen Coford et al. 2025

#Plot the ratio of UrbanVPRM to SMUrF (with and without ISA adjustment)
plt.rc('font',size=14)
plt.style.use('tableau-colorblind10')

fig, ax = plt.subplots(1,1,sharex=True,figsize=(7,5))
ax.scatter(S_noISA_JJA_time,VPRM_avg_JJA,label='Updated VPRM')
ax.scatter(S_noISA_JJA_time,S_avg_JJA,label='Updated SMUrF with ISA adjustment')
ax.scatter(S_noISA_JJA_time,S_noISA_avg_JJA,label='Updated SMUrF without ISA adjustment')

ax.axhline(0,linestyle=':',c='k')
ax.set_title('8-day Average Summer NEE over Toronto')
ax.set_xlabel('Day of Year')
ax.set_ylabel('NEE ($\mu$mol m$^{-2}$ s$^{-1}$)')
ax.legend(bbox_to_anchor=(1,1))

# *** Uncomment to save figure. CHANGE FILE NAME ***
#plt.savefig('Fixed_SMUrF_with_without_ISA_VPRM_JJA.pdf',bbox_inches='tight')
#plt.savefig('Fixed_SMUrF_with_without_ISA_VPRM_JJA.png',bbox_inches='tight')
fig.show()


# In[ ]:





# In[ ]:





# In[ ]:


# Figure 5 d-f of Madsen-Colford et al. 2025

#WITH ISA adjustment
#JJA: Doy 60 - 151 inclusive

S_JJA_time=S_time_8day[(np.round(S_time_8day,5)>=152) & (np.round(S_time_8day,5)<244)]
S_GPP_JJA=S_GPP_8day[(np.round(S_time_8day,5)>=152) & (np.round(S_time_8day,5)<244)]
S_NEE_JJA=S_NEE_8day[(np.round(S_time_8day,5)>=152) & (np.round(S_time_8day,5)<244)]
S_NEE_JJA=S_NEE_8day[(np.round(S_time_8day,5)>=152) & (np.round(S_time_8day,5)<244)]

plt.rc('font',size=21.5)
fig, ax = plt.subplots(1,3,sharex=True,sharey=True,figsize=(24.5,6))
ax[0].set_xlim(-79.69,-79.06)
ax[0].set_ylim(43.5,43.9)


fig0=ax[0].imshow(S_NEE_JJA[4][::-1]*GPP_mask,extent= extent,cmap='bwr',vmin=-10,vmax=10)
ax[0].plot(Toronto_x,Toronto_y,c='k')
ax[0].text(-79.675,43.51, 'Average NEE = '+str(np.round(np.nanmean(S_NEE_JJA[4][::-1]*GPP_mask),3))+' $\mu$mol m$^{-2}$ s$^{-1}$',fontsize=24)

ax[0].arrow(-79.3+0.09,43.7-0.1,-0.11,0.1,length_includes_head=True,width=0.008,facecolor='k')
ax[0].text(-79.3+0.11,43.7-0.105, 'Don',weight='bold',fontsize=18)
ax[0].text(-79.3+0.1,43.7-0.13, 'Valley',weight='bold',fontsize=18)
ax[0].arrow(-79.448611-0.08,43.68639+0.11,0.08,-0.11,length_includes_head=True,width=0.008,facecolor='k')
ax[0].text(-79.448611-0.12,43.68639+0.14, 'York',weight='bold',fontsize=18)
ax[0].text(-79.448611-0.19,43.68639+0.12, 'Neighbourhood',weight='bold',fontsize=18)
ax[0].arrow(-79.31,43.86,0.105,-0.025,length_includes_head=True,width=0.008,facecolor='k')
ax[0].text(-79.3-0.18,43.855, 'Rouge Park',weight='bold',fontsize=18)
ax[0].arrow(-79.32-0.11,43.58,0.05,0.06,length_includes_head=True,width=0.008,facecolor='k')
ax[0].text(-79.32-0.15,43.56, 'Downtown',weight='bold',fontsize=18)

ax[0].set_title('Updated SMUrF Toronto NEE')
    
fig1=ax[1].imshow(VPRM_NEE_JJA_8day[4],extent= extent,cmap='bwr',vmin=-10,vmax=10)
ax[1].plot(Toronto_x,Toronto_y,c='k')
ax[1].text(-79.675,43.51, 'Average NEE = '+str(np.round(np.nanmean(VPRM_NEE_JJA_8day[4]),3))+' $\mu$mol m$^{-2}$ s$^{-1}$',fontsize=24)

ax[1].set_title('Updated UrbanVPRM Toronto NEE')


fig2=ax[2].imshow(S_NEE_JJA[4][::-1]-VPRM_NEE_JJA_8day[4],extent= extent,cmap='bwr',vmin=-10,vmax=10)
ax[2].plot(Toronto_x,Toronto_y,c='k')
ax[2].text(-79.68,43.51, 'Average $\Delta$NEE = '+str(np.round(np.nanmean(S_NEE_JJA[4][::-1]*GPP_mask)-np.nanmean(VPRM_NEE_JJA_8day[4]),3))+' $\mu$mol m$^{-2}$ s$^{-1}$',fontsize=24)

ax[2].set_title('SMUrF - UrbanVPRM Toronto NEE')

cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
cbar=fig.colorbar(fig2,cax=cbar_ax)
cbar.set_label('NEE ($\mu$mol m$^{-2}$ s$^{-1}$)',fontsize=24)

ax[0].text(-79.68,43.865,'(d)',c='k',fontsize=26)
ax[1].text(-79.68,43.865,'(e)',c='k',fontsize=26)
ax[2].text(-79.68,43.865,'(f)',c='k',fontsize=26)

ax[0].set_ylabel('Latitude ($^o$)', fontsize=24)

ax[0].set_xlabel('Longitude ($^o$)',fontsize=24)
ax[1].set_xlabel('Longitude ($^o$)',fontsize=24)
ax[2].set_xlabel('Longitude ($^o$)',fontsize=24)

fig.subplots_adjust(hspace=0,wspace=0)
# *** Uncomment to save figure CHANGE PATH & FILE NAME ***
#plt.savefig('Fixed_SMUrF_shore_corr_UrbanVPRM_V061_NEE_diff_with_updated_arrows_clim_10_larger_font_DoY_'+str(int(round(S_JJA_time[4])))+'labelled.pdf',bbox_inches='tight')
#plt.savefig('Fixed_SMUrF_shore_corr_UrbanVPRM_V061_NEE_diff_with_updated_arrows_clim_10_larger_font_DoY_'+str(int(round(S_JJA_time[4])))+'labelled.png',bbox_inches='tight')
fig.show()


# In[ ]:





# In[ ]:


#Now look at timeseries of specified areas:


# In[59]:


# *** Optional: Uncomment to visualize the different areas:
#Pixels start from top left

#plt.figure(figsize=(8,6))
#plt.xlim(-79.7,-79.1)
#plt.ylim(43.5,43.9)
#plt.axis('scaled')
#plt.imshow(GPP[:,:,4000],extent= extent)
#plt.scatter(xvals[120:128],[yvals[16]]*8,c='r',label='Rouge Park')
#plt.scatter(xvals[120:128],[yvals[17]]*8,c='r')
#plt.scatter(xvals[120:128],[yvals[15]]*8,c='r')
#plt.scatter(xvals[120:128],[yvals[14]]*8,c='r')


#plt.scatter(xvals[86:90],[yvals[45]]*4,c='k',label='Don Valley')
#plt.scatter(xvals[86:90],[yvals[46]]*4,c='k')
#plt.scatter(xvals[86:90],[yvals[47]]*4,c='k')

#plt.scatter(xvals[55:65],[yvals[50]]*10,c='b',label='York Neighbourhood')
#plt.scatter(xvals[55:65],[yvals[51]]*10,c='b')
#plt.scatter(xvals[55:65],[yvals[52]]*10,c='b')

#plt.scatter(xvals[73:80],[yvals[58]]*7,c='pink',label='Downtown')
#plt.scatter(xvals[73:80],[yvals[59]]*7,c='pink')
#plt.scatter(xvals[73:80],[yvals[60]]*7,c='pink')
#plt.scatter(xvals[73:80],[yvals[61]]*7,c='pink')

#plt.xlabel('Longitude ($^o$)')
#plt.ylabel('Latitude ($^o$)')
#plt.title('UrbanVPRM Toronto GPP 2018')
#plt.legend(bbox_to_anchor=(1,0.75))
#plt.show()

# *** End of optional uncomment


# In[ ]:


DoY=np.arange(1,366)

Rouge_area_GPP=np.mean(GPP[14:18,120:128],axis=(0,1))
Rouge_area_daily_GPP=Rouge_area_GPP.reshape(365,24)
Rouge_area_daily_GPP=np.mean(Rouge_area_daily_GPP,axis=1)

Don_area_GPP=np.mean(GPP[45:48,86:90],axis=(0,1))
Don_area_daily_GPP=Don_area_GPP.reshape(365,24)
Don_area_daily_GPP=np.mean(Don_area_daily_GPP,axis=1)

York_area_GPP=np.mean(GPP[50:53,55:65],axis=(0,1))
York_area_daily_GPP=York_area_GPP.reshape(365,24)
York_area_daily_GPP=np.mean(York_area_daily_GPP,axis=1)

Down_area_GPP=np.mean(GPP[58:62,73:80],axis=(0,1))
Down_area_daily_GPP=Down_area_GPP.reshape(365,24)
Down_area_daily_GPP=np.mean(Down_area_daily_GPP,axis=1)


# In[ ]:


Rouge_area_Reco=np.mean(Reco[14:18,120:128],axis=(0,1))
Rouge_area_daily_Reco=Rouge_area_Reco.reshape(365,24)
Rouge_area_daily_Reco=np.mean(Rouge_area_daily_Reco,axis=1)
DoY=np.arange(1,366)

Don_area_Reco=np.mean(Reco[45:48,86:90],axis=(0,1))
Don_area_daily_Reco=Don_area_Reco.reshape(365,24)
Don_area_daily_Reco=np.mean(Don_area_daily_Reco,axis=1)

York_area_Reco=np.mean(Reco[50:53,55:65],axis=(0,1))
York_area_daily_Reco=York_area_Reco.reshape(365,24)
York_area_daily_Reco=np.mean(York_area_daily_Reco,axis=1)

Down_area_Reco=np.mean(Reco[58:62,73:80],axis=(0,1))
Down_area_daily_Reco=Down_area_Reco.reshape(365,24)
Down_area_daily_Reco=np.mean(Down_area_daily_Reco,axis=1)


# In[ ]:


Rouge_area_daily_NEE=Rouge_area_daily_Reco-Rouge_area_daily_GPP
Don_area_daily_NEE=Don_area_daily_Reco-Don_area_daily_GPP
York_area_daily_NEE=York_area_daily_Reco-York_area_daily_GPP
Down_area_daily_NEE=Down_area_daily_Reco-Down_area_daily_GPP


# In[ ]:


#Take the 8-day average of UrbanVPRM fluxes in these areas

Rouge_area_8day_GPP=np.concatenate([Rouge_area_daily_GPP,[0,0,0]]).reshape(46,8)
Rouge_area_8day_GPP=np.mean(Rouge_area_8day_GPP,axis=1)

Don_area_8day_GPP=np.concatenate([Don_area_daily_GPP,[0,0,0]]).reshape(46,8)
Don_area_8day_GPP=np.mean(Don_area_8day_GPP,axis=1)

York_area_8day_GPP=np.concatenate([York_area_daily_GPP,[0,0,0]]).reshape(46,8)
York_area_8day_GPP=np.mean(York_area_8day_GPP,axis=1)

Down_area_8day_GPP=np.concatenate([Down_area_daily_GPP,[0,0,0]]).reshape(46,8)
Down_area_8day_GPP=np.mean(Down_area_8day_GPP,axis=1)

Rouge_area_8day_Reco=np.concatenate([Rouge_area_daily_Reco,[0,0,0]]).reshape(46,8)
Rouge_area_8day_Reco=np.mean(Rouge_area_8day_Reco,axis=1)

Don_area_8day_Reco=np.concatenate([Don_area_daily_Reco,[0,0,0]]).reshape(46,8)
Don_area_8day_Reco=np.mean(Don_area_8day_Reco,axis=1)

York_area_8day_Reco=np.concatenate([York_area_daily_Reco,[0,0,0]]).reshape(46,8)
York_area_8day_Reco=np.mean(York_area_8day_Reco,axis=1)

Down_area_8day_Reco=np.concatenate([Down_area_daily_Reco,[0,0,0]]).reshape(46,8)
Down_area_8day_Reco=np.mean(Down_area_8day_Reco,axis=1)

Rouge_area_8day_NEE=np.concatenate([Rouge_area_daily_NEE,[0,0,0]]).reshape(46,8)
Rouge_area_8day_NEE=np.mean(Rouge_area_8day_NEE,axis=1)

Don_area_8day_NEE=np.concatenate([Don_area_daily_NEE,[0,0,0]]).reshape(46,8)
Don_area_8day_NEE=np.mean(Don_area_8day_NEE,axis=1)

York_area_8day_NEE=np.concatenate([York_area_daily_NEE,[0,0,0]]).reshape(46,8)
York_area_8day_NEE=np.mean(York_area_8day_NEE,axis=1)

Down_area_8day_NEE=np.concatenate([Down_area_daily_NEE,[0,0,0]]).reshape(46,8)
Down_area_8day_NEE=np.mean(Down_area_8day_NEE,axis=1)


# In[ ]:


DoY_8day=np.arange(1,366,8)
S_Rouge_area_8day_GPP=np.mean(S_GPP_8day[:,::-1,:][:,14:18,120:128],axis=(1,2))
S_Don_area_8day_GPP=np.mean(S_GPP_8day[:,::-1,:][:,45:48,86:90],axis=(1,2))
S_York_area_8day_GPP=np.mean(S_GPP_8day[:,::-1,:][:,50:53,55:65],axis=(1,2))
S_Down_area_8day_GPP=np.mean(S_GPP_8day[:,::-1,:][:,58:62,73:80],axis=(1,2))

S_Rouge_area_8day_Reco=np.mean(S_Reco_8day[:,::-1,:][:,14:18,120:128],axis=(1,2))
S_Don_area_8day_Reco=np.mean(S_Reco_8day[:,::-1,:][:,45:48,86:90],axis=(1,2))
S_York_area_8day_Reco=np.mean(S_Reco_8day[:,::-1,:][:,50:53,55:65],axis=(1,2))
S_Down_area_8day_Reco=np.mean(S_Reco_8day[:,::-1,:][:,58:62,73:80],axis=(1,2))

S_Rouge_area_8day_NEE=np.mean(S_NEE_8day[:,::-1,:][:,14:18,120:128],axis=(1,2))
S_Don_area_8day_NEE=np.mean(S_NEE_8day[:,::-1,:][:,45:48,86:90],axis=(1,2))
S_York_area_8day_NEE=np.mean(S_NEE_8day[:,::-1,:][:,50:53,55:65],axis=(1,2))
S_Down_area_8day_NEE=np.mean(S_NEE_8day[:,::-1,:][:,58:62,73:80],axis=(1,2))


# In[ ]:





# In[ ]:


# Figure 6 of Madsen-Colford et al. 2025
# With ISA adjustment

#Plot fluxes from the updated SMUrF and UrbanVPRM in different areas of the city
plt.style.use('tableau-colorblind10')
plt.rc('font',size=22)

fig, ax = plt.subplots(1,3,sharex=True,sharey=True,figsize=(24,6))
ax[0].set_xlim(1,362)
ax[0].set_ylim(-8,13)

ax[0].plot([0],[0],c='k',label='UrbanVPRM')
ax[0].plot([0],[0],c='k',linestyle='--',label='SMUrF')
ax[0].plot(DoY_8day, Rouge_area_8day_GPP)
ax[0].plot(DoY_8day, S_Rouge_area_8day_GPP,c='#006BA4',linestyle='--')
ax[0].plot(DoY_8day, Don_area_8day_GPP)
ax[0].plot(DoY_8day, S_Don_area_8day_GPP,c='#FF800E',linestyle='--')
ax[0].plot(DoY_8day, York_area_8day_GPP)
ax[0].plot(DoY_8day, S_York_area_8day_GPP,c='#ABABAB',linestyle='--')
ax[0].plot(DoY_8day, Down_area_8day_GPP)
ax[0].plot(DoY_8day, S_Down_area_8day_GPP,c='#595959',linestyle='--')

ax[0].legend()
ax[0].set_xlabel('Day of year')
ax[0].set_ylabel('Modelled Fluxes ($\mu$mol m$^{-2}$ s$^{-1}$)')
ax[0].set_title('GPP')

ax[1].plot(DoY_8day, Rouge_area_8day_Reco,label='Rouge Park')
ax[1].plot(DoY_8day, S_Rouge_area_8day_Reco,c='#006BA4',linestyle='--')
ax[1].plot(DoY_8day, Don_area_8day_Reco,label='Don Valley')
ax[1].plot(DoY_8day, S_Don_area_8day_Reco,c='#FF800E',linestyle='--')
ax[1].plot(DoY_8day, York_area_8day_Reco,label='York Neighbourhood')
ax[1].plot(DoY_8day, S_York_area_8day_Reco,c='#ABABAB',linestyle='--')
ax[1].plot(DoY_8day, Down_area_8day_Reco,label='Downtown')
ax[1].plot(DoY_8day, S_Down_area_8day_Reco,c='#595959',linestyle='--')
ax[1].set_xlabel('Day of year')
ax[1].set_title('R$_{eco}$')

ax[2].plot(DoY_8day, Rouge_area_8day_NEE,label='Rouge Park')
ax[2].plot(DoY_8day, S_Rouge_area_8day_NEE,c='#006BA4',linestyle='--')
ax[2].plot(DoY_8day, Don_area_8day_NEE,label='Don Valley')
ax[2].plot(DoY_8day, S_Don_area_8day_NEE,c='#FF800E',linestyle='--')
ax[2].plot(DoY_8day, York_area_8day_NEE,label='York Neighbourhood')
ax[2].plot(DoY_8day, S_York_area_8day_NEE,c='#ABABAB',linestyle='--')
ax[2].plot(DoY_8day, Down_area_8day_NEE,label='Downtown')
ax[2].plot(DoY_8day, S_Down_area_8day_NEE,c='#595959',linestyle='--')
ax[2].legend()
ax[2].set_xlabel('Day of year')
ax[2].set_title('NEE')
fig.subplots_adjust(hspace=0,wspace=0)

ax[0].text(4,11,'(a)',c='k',fontsize=26)
ax[1].text(4,11,'(b)',c='k',fontsize=26)
ax[2].text(4,11,'(c)',c='k',fontsize=26)
# *** Uncomment to save figure as pdf and png. CHANGE FILE NAME ***
#plt.savefig('8day_fluxes_Toronto_areas_fixed_SMUrF_ISA_shore_corr_vs_VPRM_labelled.pdf',bbox_inches='tight')
#plt.savefig('8day_fluxes_Toronto_areas_fixed_SMUrF_ISA_shore_corr_vs_VPRM_labelled.png',bbox_inches='tight')
fig.show()


# In[ ]:




