#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This code plots the unmodified SMUrF and UrbanVPRM NEE over the city of Toronto for a week in July, 2018.
# Code used to produce Fig 5 a-c of Madsen-Colford et al. 
# If used please cite

# *** denotes section of the code that should be changed by the user.


# In[ ]:


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
import xarray as xr


# In[ ]:





# In[ ]:


#Load in unmodified UrbanVPRM data
# *** CHANGE PATH & FILENAME ***
VPRM_path = 'C:/Users/kitty/Documents/Research/SIF/UrbanVPRM/UrbanVPRM/dataverse_files/GTA_500m_V061_no_adjustments_2018/'
VPRM_fn = 'vprm_mixed_GTA_500m_V061_2018_no_adjustment_' #without block number

VPRM_data=pd.read_csv(VPRM_path+VPRM_fn+'00000001.csv').loc[:,('HoY','Index','GEE','Re')]
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


# In[ ]:


#Load in x & y data & combine with VPRM index data
# *** CHANGE FILENAME ***
VPRM_EVI=pd.read_csv(VPRM_path+'adjusted_evi_lswi_interpolated_modis.csv').loc[:,('DOY','Index','x','y')]

#Create a dataframe with just Index, x, & y values
x=np.zeros(np.shape(VPRM_EVI.Index.unique()))*np.nan
y=np.zeros(np.shape(VPRM_EVI.Index.unique()))*np.nan
for i in range(len(VPRM_EVI.Index.unique())):
    x[i]=VPRM_EVI.x[0+i*365]
    y[i]=VPRM_EVI.y[0+i*365]
    
VPRM_xy=pd.DataFrame({'Index':VPRM_EVI.Index.unique(), 'x':x, 'y':y})
VPRM_data=VPRM_data.merge(VPRM_xy[['Index','x','y']])
del VPRM_EVI, VPRM_xy


# In[ ]:


# Extract the x & y values
xvals = VPRM_data.x[VPRM_data.HoY==4800].unique()
yvals = VPRM_data.y[VPRM_data.HoY==4800].unique()
extent = np.min(xvals), np.max(xvals), np.min(yvals), np.max(yvals)

# Reshape the GPP and Reco data
GPP=VPRM_data.GEE.values.reshape(len(yvals),len(xvals),8760)#8784 for leap year
Reco=VPRM_data.Re.values.reshape(len(yvals),len(xvals),8760)

# Extract the hour and day of the year
HoY=VPRM_data.HoY.values.reshape(len(yvals),len(xvals),8760)
DoY=np.mean(HoY,axis=(0,1))/24+23/24


# In[ ]:


#Swap the aces to match the format of SMUrF & calculate NEE
VPRM_GPP=(np.swapaxes(np.swapaxes(GPP,0,2),1,2))
VPRM_Reco=(np.swapaxes(np.swapaxes(Reco,0,2),1,2))
VPRM_NEE=(np.swapaxes(np.swapaxes(Reco,0,2),1,2)-np.swapaxes(np.swapaxes(GPP,0,2),1,2))

#Take the 8-day average to match the temporal resolution of SMUrF
VPRM_GPP_8day=np.ones((46, 96, 144))*np.nan
VPRM_Reco_8day=np.ones((46, 96, 144))*np.nan
VPRM_NEE_8day=np.ones((46, 96, 144))*np.nan
for i in range(46):
    VPRM_GPP_8day[i]=np.nanmean(VPRM_GPP[i*8*24:i*8*24+8*24],axis=0)
    VPRM_Reco_8day[i]=np.nanmean(VPRM_Reco[i*8*24:i*8*24+8*24],axis=0)
    VPRM_NEE_8day[i]=np.nanmean(VPRM_NEE[i*8*24:i*8*24+8*24],axis=0)


# In[ ]:





# In[ ]:


#Load in the shape file for Toronto's boundary
# *** CHANGE PATH & FILENAME ***
sf = shp.Reader("C:/Users/kitty/Documents/Research/SIF/Shape_files/Toronto/Toronto_Boundary.shp")
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

#Create a mask for areas outside the GTA
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


#now bring in the original SMUrF data with V061 MODIS

# *** CHANGE PATH & FILENAME ***
SMUrF_path = 'E:/Research/SMUrF/output2018_CSIF_V061/easternCONUS/'
SMUrF_fn = 'daily_mean_Reco_neuralnet/era5/2018/daily_mean_Reco_uncert_easternCONUS_2018' # WITHOUT the month & day (added in loop below)

#load the first file to get the start of the year
g=Dataset(SMUrF_path+SMUrF_fn+'0101.nc')
start_of_year=g.variables['time'][0]/3600/24-1 #convert seconds since 1970 to days (minus one)
g.close()

#With ISA adjustment using GMIS-Toronto-SOLRIS-ACI dataset
S_time=[]
S_Reco=[]
S_Reco_err=[]
S_lats=[]
S_lons=[]
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
                S_lats=f.variables['lat'][:]
                S_lons=f.variables['lon'][:]
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
        
#Load in GPP data
f=Dataset(SMUrF_path+'daily_mean_SIF_GPP_uncert_easternCONUS_2018.nc')
S_time=f.variables['time'][:]/24/3600-start_of_year-5/24 #convert seconds since 1970 to days and subtract start of year

S_GPP_err=f.variables['GPP_sd'][:]
S_GPP=f.variables['GPP_mean'][:]

S_Reco[S_Reco==-999]=np.nan
S_Reco_err[S_Reco_err==-999]=np.nan
S_GPP[S_GPP==-999]=np.nan
S_GPP_err[S_GPP_err==-999]=np.nan

#Take 4-day average of Reco to match temporal resolution of GPP
S_Reco_4d=np.ones(np.shape(S_GPP))*np.nan
S_Reco_4d_err=np.ones(np.shape(S_GPP))*np.nan
#S_Reco_4d_std=np.ones(np.shape(S_GPP))*np.nan
for i in range(len(S_time)):
    S_Reco_4d[i]=np.nanmean(S_Reco[i*4:i*4+4],axis=0)
    S_Reco_4d_err[i]=np.sqrt(np.nansum((S_Reco_err[i*4:i*4+4]/4)**2,axis=0))
    #S_Reco_4d_std[i]=np.nanstd(S_Reco[i*4:i*4+4],axis=0)
    
S_NEE=S_Reco_4d-S_GPP
S_NEE_err=np.sqrt(S_Reco_4d_err**2+S_GPP_err**2)


# In[ ]:


#Create a mask for areas outside Toronto for SMUrF
# Also create weights for pixels that fall partially inside the boundary
S_mask_lons=np.ones(14)*np.nan
S_mask_lats=np.ones(10)*np.nan
S_GPP_mask=np.ones([10,14])*np.nan
S_mask_weight=np.ones([10,14])*np.nan
for i in range(0, 14):
    for j in range(0, 10):
        pts=[geometry.Point(S_lons[19:33][i]-0.025,S_lats[19:29][j]-0.025),geometry.Point(S_lons[19:33][i]+0.025,S_lats[19:29][j]-0.025),geometry.Point(S_lons[19:33][i]+0.025,S_lats[19:29][j]+0.025),geometry.Point(S_lons[19:33][i]-0.025,S_lats[19:29][j]+0.025),geometry.Point(S_lons[19:33][i]-0.025,S_lats[19:29][j]-0.025)]
        pixel=geometry.Polygon([[p.x, p.y] for p in pts])
        footprint=pixel.area
        intersect=pixel.intersection(poly)
        
        if intersect.area >0:
            S_mask_lons[i]=S_lons[19:33][i]
            S_mask_lats[j]=S_lats[19:29][j]
            S_GPP_mask[j,i]=1
            S_mask_weight[j,i]=intersect.area/footprint


# In[ ]:


#Take 8-day average to match temporal resolution of updated SMUrF
S_NEE_8day=np.zeros([len(VPRM_GPP_8day),len(S_NEE[0]),len(S_NEE[0,0])])*np.nan
S_GPP_8day=np.zeros([len(VPRM_GPP_8day),len(S_NEE[0]),len(S_NEE[0,0])])*np.nan
S_Reco_8day=np.zeros([len(VPRM_GPP_8day),len(S_NEE[0]),len(S_NEE[0,0])])*np.nan
S_time_8day=np.zeros(len(VPRM_GPP_8day))*np.nan

for i in range(len(VPRM_NEE_8day)):
    S_NEE_8day[i]=np.nanmean(S_NEE[i*2:i*2+2],axis=0)
    S_GPP_8day[i]=np.nanmean(S_GPP[i*2:i*2+2],axis=0)
    S_Reco_8day[i]=np.nanmean(S_Reco_4d[i*2:i*2+2],axis=0)
    S_time_8day[i] = S_time[i*2]


# In[ ]:


#Select only the summer data

#JJA: Doy 60 - 151 inclusive
VPRM_GPP_JJA_8day=VPRM_GPP_8day[(np.round(S_time_8day,5)>=152) & (np.round(S_time_8day,5)<244)]
VPRM_Reco_JJA_8day=VPRM_Reco_8day[(np.round(S_time_8day,5)>=152) & (np.round(S_time_8day,5)<244)]
VPRM_NEE_JJA_8day=VPRM_NEE_8day[(np.round(S_time_8day,5)>=152) & (np.round(S_time_8day,5)<244)]

S_JJA_time=S_time_8day[(np.round(S_time_8day,5)>=152) & (np.round(S_time_8day,5)<244)]
S_GPP_JJA=S_GPP_8day[(np.round(S_time_8day,5)>=152) & (np.round(S_time_8day,5)<244)]
S_Reco_JJA=S_Reco_8day[(np.round(S_time_8day,5)>=152) & (np.round(S_time_8day,5)<244)]
S_NEE_JJA=S_NEE_8day[(np.round(S_time_8day,5)>=152) & (np.round(S_time_8day,5)<244)]


# In[ ]:





# In[ ]:


# Resample VPRM data at SMUrF resolution

# Convert array to an xarray 'Data Array'
VPRM_doy_185= VPRM_NEE_JJA_8day[4,:,12:]
VPRM_NEE_185_da = xr.DataArray(VPRM_doy_185,coords=[yvals,xvals[12:]])
VPRM_NEE_185_da = VPRM_NEE_185_da.rename({'dim_0':'lat','dim_1':'lon'})

# Convert the NEE to a dataset
VPRM_NEE_185_ds = VPRM_NEE_185_da.to_dataset(name='VPRM_NEE')
#Resample the dataset to SMUrF's resolution
VPRM_NEE_185_resamp_ds = VPRM_NEE_185_ds.coarsen(lon=12).mean().coarsen(lat=12).mean()

# Convert it back to a Data Array
VPRM_NEE_185_resamp_da = VPRM_NEE_185_resamp_ds.to_array()
VPRM_NEE_185_resamp_da = VPRM_NEE_185_resamp_da.drop_vars('variable')
VPRM_NEE_185_resamp_da = VPRM_NEE_185_resamp_da[0]


# In[ ]:





# In[ ]:


#Create a mask for the resampled UrbanVPRM for areas outside Toronto
VPRM_resamp_mask_lons=np.ones(11)*np.nan
VPRM_resamp_mask_lats=np.ones(8)*np.nan
VPRM_resamp_GPP_mask=np.ones([8,11])*np.nan
VPRM_resamp_mask_weight=np.ones([8,11])*np.nan
for i in range(0, 11):
    for j in range(0, 8):
        pts=[geometry.Point(VPRM_NEE_185_resamp_da['lon'][i]-0.025,VPRM_NEE_185_resamp_da['lat'][j]-0.025),geometry.Point(VPRM_NEE_185_resamp_da['lon'][i]+0.025,VPRM_NEE_185_resamp_da['lat'][j]-0.025),geometry.Point(VPRM_NEE_185_resamp_da['lon'][i]+0.025,VPRM_NEE_185_resamp_da['lat'][j]+0.025),geometry.Point(VPRM_NEE_185_resamp_da['lon'][i]-0.025,VPRM_NEE_185_resamp_da['lat'][j]+0.025),geometry.Point(VPRM_NEE_185_resamp_da['lon'][i]-0.025,VPRM_NEE_185_resamp_da['lat'][j]-0.025)]
        VPRM_pixel=geometry.Polygon([[p.x, p.y] for p in pts])
        footprint=VPRM_pixel.area
        intersect=VPRM_pixel.intersection(poly)
        
        if intersect.area >0:
            VPRM_resamp_mask_lons[i]=VPRM_NEE_185_resamp_da['lon'][i]
            VPRM_resamp_mask_lats[j]=VPRM_NEE_185_resamp_da['lat'][j]
            VPRM_resamp_GPP_mask[j,i]=1
            VPRM_resamp_mask_weight[j,i]=intersect.area/footprint


# In[ ]:


# Create an extent for the resampled UrbanVPRM
VPRM_resamp_extent=np.min(S_lons[21:33])-0.025, np.max(S_lons[21:33])-0.025, np.min(S_lats[20:29])-0.025, np.max(S_lats[20:29])-0.025


# In[ ]:


#Crop the SMUrF data to Toronto
S_lons_cropped=S_lons[21:33]-0.025
S_lats_cropped=S_lats[20:29]-0.025
S_NEE_8day_cropped=S_NEE_8day[23,20:28,21:32]
S_GPP_mask_cropped=S_GPP_mask[1:9,2:13]
S_mask_weight_cropped=S_mask_weight[1:9,2:13]


# In[ ]:


# Define an extent for the cropped SMUrF data
S_cropped_extent=np.min(S_lons_cropped), np.max(S_lons_cropped), np.min(S_lats_cropped), np.max(S_lats_cropped)


# In[ ]:





# In[ ]:


# Plot the original SMUrF and UrbanVPRM and their difference for one week in July (fig 5 a-c)

plt.rc('font',size=21.5)
fig, ax = plt.subplots(1,3,sharex=True,sharey=True,figsize=(24.5,6))
ax[0].set_xlim(-79.69,-79.06)
ax[0].set_ylim(43.5,43.9)


fig0=ax[0].pcolormesh(S_lons_cropped,S_lats_cropped,S_NEE_8day_cropped*S_GPP_mask_cropped,cmap='bwr',vmin=-10,vmax=10)
ax[0].plot(Toronto_x,Toronto_y,c='k')
ax[0].text(-79.66,43.51, 'Average NEE = '+str(np.round(np.nansum(S_NEE_8day_cropped*S_mask_weight_cropped)/np.nansum(S_mask_weight_cropped),3))+' $\mu$mol m$^{-2}$ s$^{-1}$',fontsize=24)
ax[0].set_title('Unmodified SMUrF Toronto NEE',fontsize=25)
    
fig1=ax[1].pcolormesh(xvals,yvals,VPRM_NEE_JJA_8day[4]*GPP_mask,cmap='bwr',vmin=-10,vmax=10)
ax[1].plot(Toronto_x,Toronto_y,c='k')
#Using average at 0.05 res (with weights depending on how much of each pixel falls within city limits):
ax[1].text(-79.675,43.51, 'Average NEE = '+str(np.round(np.nansum(VPRM_NEE_185_resamp_da*VPRM_resamp_mask_weight)/(np.nansum(VPRM_resamp_mask_weight)),3))+' $\mu$mol m$^{-2}$ s$^{-1}$',fontsize=24)
ax[1].set_title('Unmodified UrbanVPRM Toronto NEE', fontsize=25)

fig2=ax[2].pcolormesh(S_lons_cropped,S_lats_cropped,S_NEE_8day_cropped*S_GPP_mask_cropped-(VPRM_NEE_185_resamp_da[::-1])*S_GPP_mask_cropped,cmap='bwr',vmin=-10,vmax=10)
ax[2].plot(Toronto_x,Toronto_y,c='k')
ax[2].text(-79.68,43.51, 'Average $\Delta$NEE = '+str(np.round(np.nansum(S_NEE_8day_cropped*S_mask_weight_cropped-VPRM_NEE_185_resamp_da[::-1]*S_mask_weight_cropped)/np.nansum(S_mask_weight_cropped),3))+' $\mu$mol m$^{-2}$ s$^{-1}$',fontsize=24)
ax[2].set_title('SMUrF - UrbanVPRM Toronto NEE',fontsize=25)

cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
cbar=fig.colorbar(fig1,cax=cbar_ax)
cbar.set_label('NEE ($\mu$mol m$^{-2}$ s$^{-1}$)',fontsize=24)

ax[0].text(-79.68,43.865,'(a)',c='k',fontsize=26)
ax[1].text(-79.68,43.865,'(b)',c='k',fontsize=26)
ax[2].text(-79.68,43.865,'(c)',c='k',fontsize=26)

ax[0].set_ylabel('Latitude ($^o$)',fontsize=24)

ax[0].set_xlabel('Longitude ($^o$)',fontsize=24)
ax[1].set_xlabel('Longitude ($^o$)',fontsize=24)
ax[2].set_xlabel('Longitude ($^o$)',fontsize=24)

fig.subplots_adjust(hspace=0,wspace=0)
# *** Uncomment to save figure as png and pdf. CHANGE FILENAME ***
#plt.savefig('Original_SMUrF_vs_V061_UrbanVPRM_NEE_aggregated_avg_unmodified_clim_10_larger_font_DoY_'+str(int(round(S_JJA_time[4])))+'_fixed_labelled.pdf',bbox_inches='tight')
#plt.savefig('Original_SMUrF_vs_V061_UrbanVPRM_NEE_aggregated_avg_unmodified_clim_10_larger_font_DoY_'+str(int(round(S_JJA_time[4])))+'_fixed_labelled.png',bbox_inches='tight')
fig.show()


# In[ ]:


#With minimum R value in UrbanVPRM
#V061 MODIS

VPRM_avg=np.zeros(len(S_NEE_8day))*np.nan
SMUrF_avg=np.zeros(len(S_NEE_8day))*np.nan
SMAPE=np.zeros(len(S_NEE_8day))*np.nan
diff=np.zeros(len(S_NEE_8day))*np.nan
diff_SMUrF=np.zeros(len(S_NEE_8day))*np.nan

for i in range(len(S_NEE_8day)):
    S_NEE_8day_cropped_i=S_NEE_8day[i,20:28,21:32]
    VPRM_doy_i= VPRM_NEE_8day[i,:,12:]
    VPRM_NEE_i_da = xr.DataArray(VPRM_doy_i,coords=[yvals-1/240/2,xvals[12:]-1/240/2])
    VPRM_NEE_i_da = VPRM_NEE_i_da.rename({'dim_0':'lat','dim_1':'lon'})

    VPRM_NEE_i_ds = VPRM_NEE_i_da.to_dataset(name='VPRM_NEE')

    VPRM_NEE_i_resamp_ds = VPRM_NEE_i_ds.coarsen(lon=12).mean().coarsen(lat=12).mean()

    VPRM_NEE_i_resamp_da = VPRM_NEE_i_resamp_ds.to_array()

    VPRM_NEE_i_resamp_da = VPRM_NEE_i_resamp_da.drop_vars('variable')
    VPRM_NEE_i_resamp_da = VPRM_NEE_i_resamp_da[0]
    
    VPRM_avg[i]=(np.nansum(VPRM_NEE_i_resamp_da[::-1]*S_mask_weight_cropped)/np.nansum(S_mask_weight_cropped))
    SMUrF_avg[i] = (np.nansum(S_NEE_8day_cropped_i*S_mask_weight_cropped)/np.nansum(S_mask_weight_cropped))
    SMAPE[i]=abs(SMUrF_avg[i]-VPRM_avg[i])/((abs(SMUrF_avg[i])+abs(VPRM_avg[i]))/2)
    diff[i]=(SMUrF_avg[i]-VPRM_avg[i])/((abs(SMUrF_avg[i])+abs(VPRM_avg[i]))/2)
    diff_SMUrF[i]=(SMUrF_avg[i]-VPRM_avg[i])/((abs(SMUrF_avg[i]))/2)
    
    #Uncomment to print the Percent Difference for each 8-day average over Toronto:
    #print("Doy "+str(i*8+1)+' |SMUrF-VPRM|/(|SMUrF|+|VPRM|/2) NEE: '+str(np.round(SMAPE[i],3)))


# In[ ]:


## *** Uncomment to print Summer SMUrF averages over Toronto
#print('Toronto SMUrF 8-day Summer Averages: '+
#      str(np.round(SMUrF_avg[(np.round(S_time_8day,5)>=152) & (np.round(S_time_8day,5)<244)],5)))


# In[ ]:


## *** Uncoment to print Summer UrbanVPRM averages over Toronto
#print('Toronto UrbanVPRM 8-day Summer Averages: '+
#      str(np.round(VPRM_avg[(np.round(S_time_8day,5)>=152) & (np.round(S_time_8day,5)<244)],5)))


# In[ ]:


## *** Uncoment to print mean Summer SMUrF averages over Toronto
print('JJA mean SMUrF = '+str(np.round(np.mean(SMUrF_avg[(np.round(S_time_8day,5)>=152) & (np.round(S_time_8day,5)<244)]),2))+' +/- '+str(np.round(np.std(SMUrF_avg[(np.round(S_time_8day,5)>=152) & (np.round(S_time_8day,5)<244)]),2)))


# In[ ]:


## *** Uncoment to print mean Summer UrbanVPRM averages over Toronto
print('JJA mean VPRM = '+str(np.round(np.mean(VPRM_avg[(np.round(S_time_8day,5)>=152) & (np.round(S_time_8day,5)<244)]),2))+' +/- '+str(np.round(np.std(VPRM_avg[(np.round(S_time_8day,5)>=152) & (np.round(S_time_8day,5)<244)]),2)))


# In[ ]:


print('JJA mean Percent difference = '+str(np.round(np.mean(SMAPE[(np.round(S_time_8day,5)>=152) & (np.round(S_time_8day,5)<244)])*100,2))+' +/- '+str(np.round(np.std(SMAPE[(np.round(S_time_8day,5)>=152) & (np.round(S_time_8day,5)<244)])*100,2))+' %')


# In[ ]:




