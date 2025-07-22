#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Code used to visualize difference between the downscaled SIF with and without the shoreline tree correction
# Creates figure S2 of Madsen-Colford et al. 2025

# *** Denotes parts of the code that should be modified by the user


# In[1]:


#For actual correction see Shape_files/Toronto_tree_analysis.R 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colr
import pandas as pd
from scipy import optimize as opt 
from scipy import odr
import shapefile as shp # to import outline of GTA
from shapely import geometry # used to define a polygon for Toronto
import netCDF4
from netCDF4 import Dataset, date2num #for reading netCDF data files and their date (not sure if I need the later)


# In[2]:


#Load in non-corrected SIF data & lat & lon

#*** CHANGE PATH & FILENAME ***
g=Dataset('C:/Users/kitty/Documents/Research/SIF/SMUrF/data/downscaled_CSIF/TROPOMI_CSIF_combined_med/V061/2018/V3/downscaled_V061_TROPO_CSIF_8d_2018185.nc')
TROPO_sif=g.variables['daily_sif'][:]
lons = g.variables['lon'][:]
lats = g.variables['lat'][:]
g.close()


# In[3]:


#Load in the Toronto shape file to mask values outside of the city

#*** CHANGE PATH & FILENAME ***
sf = shp.Reader("C:/Users/kitty/Documents/Research/SIF/Shape_files/Toronto/Toronto_Boundary.shp")
#Toronto_Shape
shape=sf.shape(0)
#Need to partition each individual shape
Toronto_x = np.zeros((len(shape.points),1))*np.nan #The main portion of the GTA
Toronto_y = np.zeros((len(shape.points),1))*np.nan
for i in range(len(shape.points)):
    Toronto_x[i]=shape.points[i][0]
    Toronto_y[i]=shape.points[i][1]
    
points=[]
for k in range(1,len(Toronto_x)):
    points.append(geometry.Point(Toronto_x[k],Toronto_y[k]))
poly=geometry.Polygon([[p.x, p.y] for p in points])

#Create a mask for areas outside the GTA
GPP_mask=np.ones([553,625])*np.nan
for i in range(0, len(lons)):
    for j in range(0, len(lats)):
        if poly.contains(geometry.Point([lons[i],lats[j]])):
            GPP_mask[j,i]=1


# In[4]:


#Load in corrected SIF data

#*** CHANGE PATH & FILENAME ***
g=Dataset('C:/Users/kitty/Documents/Research/SIF/SMUrF/data/downscaled_CSIF/TROPOMI_CSIF_combined_med/V061/2018/V3/downscaled_v061_TROPO_CSIF_shore_weighted_corrected_8d_2018185.nc')
Corr_sif=g.variables['daily_sif'][:]
Corr_lons = g.variables['lon'][:]
Corr_lats = g.variables['lat'][:]
g.close()


# In[ ]:





# In[10]:


plt.rc('font',size=24)
fig, ax = plt.subplots(1,2,sharex=True,sharey=True,figsize=(16,6))
#plt.figure(figsize=(10,5))
ax[0].set_xlim(-79.63,-79.12)
ax[0].set_ylim(43.57,43.87)


fig0=ax[0].pcolormesh(lons,lats,TROPO_sif*GPP_mask,vmin=-0.2,vmax=1)
#ax[0].set_clim(0,1)
ax[0].plot(Toronto_x,Toronto_y,c='k',linestyle=':')
ax[0].set_title('Downscaled SIF')
    
fig1=ax[1].pcolormesh(lons,lats,Corr_sif[::-1]*GPP_mask,vmin=-0.2,vmax=1)
ax[1].plot(Toronto_x,Toronto_y,c='k',linestyle=':')
ax[1].set_title('Shore-Corrected Downscaled SIF')

ax[0].set_ylabel('Latitude ($^o$)')
#ax[0].set_yticks([])

ax[0].set_xlabel('Longitude ($^o$)')
ax[1].set_xlabel('Longitude ($^o$)')

ax[0].text(-79.62,43.845,'(a)',c='k')
ax[1].text(-79.62,43.845,'(b)',c='k')

cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
cbar=fig.colorbar(fig1,cax=cbar_ax)
cbar.set_label('SIF (mW m$^{-2}$ s$^{-1}$ nm$^{-1}$)')

fig.subplots_adjust(hspace=0,wspace=0)

plt.savefig('shoreline_corrected_uncorrected_fixed_SIF_DoY_185_labelled.pdf',bbox_inches='tight')
plt.savefig('shoreline_corrected_uncorrected_fixed_SIF_DoY_185_labelled.png',bbox_inches='tight')
plt.show()

plt.figure(figsize=(10,6))
plt.xlim(-79.63,-79.12)
plt.ylim(43.57,43.87)
plt.pcolormesh(lons,lats,(Corr_sif[::-1]-TROPO_sif)*GPP_mask,cmap='bwr',vmin=-0.39,vmax=0.39)
cbar2=plt.colorbar()
plt.plot(Toronto_x,Toronto_y,c='k',linestyle=':')
plt.title('Corrected - Non-Corrected SIF')

#cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
#cbar=fig.colorbar(fig2,cax=cbar_ax)
cbar2.set_label('$\Delta$SIF (mW m$^{-2}$ s$^{-1}$ nm$^{-1}$)')

plt.yticks([])
#plt.ylabel('Latitude ($^o$)')

plt.text(-79.62,43.845,'(c)',c='k')

plt.xlabel('Longitude ($^o$)')
plt.savefig('shoreline_corrected_fixed_SIF_diff_DoY_185_labelled.pdf',bbox_inches='tight')
plt.savefig('shoreline_corrected_fixed_SIF_diff_DoY_185_labelled.png',bbox_inches='tight')
plt.show()


# In[ ]:




