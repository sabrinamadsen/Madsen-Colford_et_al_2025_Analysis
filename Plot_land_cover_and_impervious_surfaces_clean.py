#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This code plots figure 1 of Madsen-Colford et al. 2025: 
# Plots MODIS land cover (MCD12Q1), impervious surface area fraction (ISA) generated from the 
# Global Man-made Impervious Surface (GMIS), the Canadian Annual Crop Inventory (ACI),
# the Southern Ontario Land Resource Information System (SOLRIS V3.0), and the 
# City of Toronto's Topographic Mapping – Impermeable Surface datasets (see 'Toronto_permeability_plot_GMIS.R'),
# and canopy cover from the city of Toronto's Forest and Land Cover dataset.

# *** denotes parts of the code the user should change


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image #to read in TIFF files
from osgeo import gdal #to find lat/lon of TIFF files
import shapefile as shp # to import outline of GTA
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar #for making scalebars on maps
from netCDF4 import Dataset, date2num #for reading netCDF data files and their date (not sure if I need the later)
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.patches import Rectangle


# In[2]:


# *** CHANGE PATH/FILENAME ***
sf = shp.Reader("C:/Users/kitty/Documents/Research/SIF/Shape_files/Toronto/Toronto_Boundary.shp")
#Toronto_Shape
shape=sf.shape(0)
#Need to partition each individual shape
Toronto_x = np.zeros((len(shape.points),1))*np.nan #The main portion of the GTA
Toronto_y = np.zeros((len(shape.points),1))*np.nan
for i in range(len(shape.points)):
    Toronto_x[i]=shape.points[i][0]
    Toronto_y[i]=shape.points[i][1]


# In[3]:


# Load in the ISA data
# *** CHANGE PATH/FILENAME ***
ISA_data = Image.open("GMIS_Toronto_ACI_SOLRIS_2018_impervious_GTA.tif")
ISA_array = np.array(ISA_data)

ISA_array[ISA_array<0] = np.nan #Replace anomalous ISA values (<0) with NA


# In[4]:


# Create lat/lon grid for plotting from lat lon of tiff file
# *** CHANGE PATH/FILENAME ***
ISA = gdal.Open("GMIS_Toronto_ACI_SOLRIS_2018_impervious_GTA.tif")

width=ISA.RasterXSize
height=ISA.RasterYSize
gt = ISA.GetGeoTransform()

minx = gt[0]
miny = gt[3] + width*gt[4] + height*gt[5] 
maxx = gt[0] + width*gt[1] + height*gt[2]
maxy = gt[3] 

step=gt[1]
lon_list=np.arange(minx,maxx,step)
lat_list=np.arange(miny,maxy,step)
longrid=np.array([lon_list for i in range(len(lat_list))])
latgrid=np.array([lat_list for i in range(len(lon_list))]).T[::-1]


# In[5]:


#Load in land cover data and extract lat/lon
# *** CHANGE PATH/FILENAME ***
LC_dat=Dataset("C:/Users/kitty/documents/Research/SIF/SMUrF/data/MCD12Q1/V061_Adjusted_MODIS_Land_Type_2018.nc")

Land_Data=LC_dat.variables['Land_Type'][:]
lon_data= LC_dat.variables['Longitude'][:]
lat_data= LC_dat.variables['Latitude'][:]

step=1/240
lon_list_mod=np.arange(-80.9,-78.3,step)
lat_list_mod=np.arange(42.4,44.7,step)
longrid_mod=np.array([lon_list_mod for i in range(len(lat_list_mod))])
latgrid_mod=np.array([lat_list_mod for i in range(len(lon_list_mod))]).T[::-1]


# In[6]:


#Replace 0 ISA with NA to make it transparent
ISA_array_nan=np.copy(ISA_array)
ISA_array_nan[ISA_array_nan==0]=np.nan


# In[ ]:





# In[7]:


# Create a colour map for land cover
cmap=(mpl.colors.ListedColormap(['darkslategrey','darkgreen','seagreen','limegreen','forestgreen','mediumslateblue','pink','peru','goldenrod','yellowgreen','teal','khaki','slategrey','darkorange','white','darkred','lightblue']))

#Make a dictionary to call the colours for plotting land cover data & associate them with the labels

col_dict = {1: 'DarkGreen',
            2: 'lime',
            3: 'palegreen',
            4: 'greenyellow',
            5: 'forestgreen',
            6: 'mediumslateblue',
            7: 'pink',
            8: 'sienna',
            9: 'peru',
            10: 'yellow',
            11: 'teal',
            12: 'wheat',
            13: 'slategrey',
            14: 'darkorange',
            15: 'white',
            16: 'brown',
            17: 'lightblue'}

labels = np.array(['ENF','EBF','DNF','DBF','MF','CShr','OShr','WSav','Sav','Grs','Wet','Crp','Urb','Crp/Nat','Ice','Barren','Wtr'])
norm_bins = np.sort([*col_dict.keys()]) + 0.5
norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)

norm = mpl.colors.BoundaryNorm(norm_bins,len(labels),clip=True)

fmt = mpl.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])


# In[8]:


#Set Urban land cover to NA to make it transparent (so that can see ISA underneath)
Land_Dat_no_Urb = np.copy(Land_Data)
Land_Dat_no_Urb[Land_Dat_no_Urb==13]=np.nan


# In[ ]:





# In[9]:


#Load in aggregated tree canopy data
# *** CHANGE PATH/FILENAME ****
f = Dataset('C:/Users/kitty/Documents/ArcGIS/Projects/Toroonto_canopy_cover/Toronto_tree_cover_percent_rasmpled.nc')
TC_data = f.variables['tree_cover'][:]
TC_lats = f.variables['lat'][:]
TC_lons = f.variables['lon'][:]
TC_data[TC_data <0] = np.nan #replace fill values with NA


# In[ ]:





# In[14]:


fig,ax1 = plt.subplots(figsize=(10,6))
ax1.set_xlim(-80.8,-78.5)
ax1.set_ylim(42.5,44.6)
ax1.axis('scaled')

isa_im=ax1.pcolormesh(longrid,latgrid,ISA_array_nan,cmap='Greys',vmin=0,vmax=100,rasterized=True)
cbar=fig.colorbar(isa_im)
cbar.set_label('Impervious Surface Fraction (%)')

im=ax1.pcolormesh(longrid_mod,latgrid_mod,Land_Dat_no_Urb.T[::-1],cmap=cmap, norm=norm,rasterized=True)
diff = norm_bins[1:] - norm_bins[:-1]
tickz = norm_bins[:-1] +diff/2
fig.colorbar(im,format=fmt,ticks=tickz)

ax1.scatter(-79.9333,44.3166700,color='k') #Borden Forest
ax1.text(-80.4,44.23,'Borden \n Forest',c='k',weight='bold')
ax1.scatter(-80.357376,42.710161,color='k') #Turkey Point Research Center Pine location
ax1.text(-80.4,42.8,'TP39',c='k',weight='bold')
ax1.scatter(-80.557731,42.635328,color='k') #Turkey Point Research Center Deciduous location
ax1.text(-80.75,42.68,'TPD',c='k',weight='bold')

ax1.plot(Toronto_x,Toronto_y,c='k',rasterized=True)
ax1.text(-79.05,43.75,'Toronto',c='k',weight='bold')

ax1.text(-79.4,43.44,'Lake Ontario',c='k',weight='bold')

ax1.text(-80.75,42.45,'Lake Erie',c='k',weight='bold')

scalebar = AnchoredSizeBar(ax1.transData,1/240*40,"20 km", loc='upper right',pad=0.25,sep=10,frameon=False,size_vertical=0.01,fontproperties=plt.rc('font',size=14,weight='bold'),color='k')
ax1.add_artist(scalebar)

#add inset on plot
#ax1.add_patch(Rectangle((-80.05,42.35),2,0.95,facecolor='white',alpha=0.75))

ax2 = fig.add_axes([0.3,0.14,0.31,0.23])
ax2.set_xlim(-79.62,-79.13)
ax2.set_ylim(43.57,43.85)
ax2.axis('scaled')
TC_im = ax2.pcolormesh(TC_lons,TC_lats,TC_data*100,cmap='Greens',vmin=0,vmax=100,rasterized=True)
cbar_TC=fig.colorbar(TC_im)
ax2.set_title('   Canopy Cover (%)',weight='bold')
ax2.plot(Toronto_x+1/240,Toronto_y-1/240,c='k',rasterized=True)
ax2.set(xticklabels=[])
ax2.tick_params(bottom=False)
ax2.set(yticklabels=[])
ax2.tick_params(left=False)
mark_inset(ax1, ax2, loc1=2, loc2=1, ec='k', linestyle='--')

ax1.set_title('Southern Ontario Land Cover Type and ISA')
ax1.set_xlabel('Longitude ($^o$)')
ax1.set_ylabel('Latitude ($^o$)')

# *** Uncomment to save figure as pdf and png CHANGE FILE NAMES/PATH ***
plt.savefig('Land_cover_type_ISA_and_tree_cover_test.pdf',bbox_inches='tight')
plt.savefig('Land_cover_type_ISA_and_tree_cover_test.png',bbox_inches='tight')
fig.show()


# In[ ]:





# In[ ]:




