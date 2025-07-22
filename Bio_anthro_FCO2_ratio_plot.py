#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This code is used to plot the diurnal patterns of biogenic and anthropogenic CO2 fluxes in the city of Toronto
# as estimated by the UrbanVPRM & SMUrF vegetation models and the ODIAC & EDGAR emission inventories respectively. 
# We also calculate & plot the ratio of biogenic to EDGAR anthropogenic CO2 fluxes for both vegetation models.
# Need to first run 'Diurnal_uncertainty_seasonal.R' and 'Monthly_EDGAR_ODIAC_NEE_comparison.R' to calculate seasonal
# & monthly fluxes, respectively.

# This code is used to produce figures 8 & S7 in Madsen-Colford et al. 2025
# If used please cite

# Parts of the code with '***' indicate areas where the user should change something (e.g. paths or filenames)


# In[ ]:





# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd


# In[3]:


#Load in anthropogenic and biogenic fluxes for each season
# *** CHANGE PATHS & FILENAMES ***
# Create these files using 'Diurnal_uncertainty_seasonal.R'
anthro_data = pd.read_csv('bio_anthro_diurnal_sys_errs_all_seasons.csv')
bio_path = 'C:/Users/kitty/Documents/Research/SIF/Emission_inventories/Emission_inventory_comparison/'


# In[4]:


MAM_data = pd.read_csv(bio_path+'MAM_bio_diurnal_sys_errs_all_VPRM_abs_errs_SMUrF_fix.csv')
SMUrF_MAM=np.array(MAM_data['S_NEE'])
SMUrF_MAM_sd=np.array(MAM_data['S_NEE_sd'])
VPRM_MAM=np.array(MAM_data['V_NEE'])
VPRM_MAM_sd=np.array(MAM_data['V_NEE_sd'])
ODIAC_MAM=np.array(anthro_data['O_MAM'])
ODIAC_MAM_sd=np.array(anthro_data['O_MAM_sd'])
EDGAR_MAM=np.array(anthro_data['E_MAM'])
EDGAR_MAM_sd=np.array(anthro_data['E_MAM_sd'])


# In[5]:


JJA_data = pd.read_csv(bio_path+'JJA_bio_diurnal_sys_errs_all_VPRM_abs_errs_SMUrF_fix.csv')
SMUrF_JJA=np.array(JJA_data['S_NEE'])
SMUrF_JJA_sd=np.array(JJA_data['S_NEE_sd'])
VPRM_JJA=np.array(JJA_data['V_NEE'])
VPRM_JJA_sd=np.array(JJA_data['V_NEE_sd'])
ODIAC_JJA=np.array(anthro_data['O_JJA'])
ODIAC_JJA_sd=np.array(anthro_data['O_JJA_sd'])
EDGAR_JJA=np.array(anthro_data['E_JJA'])
EDGAR_JJA_sd=np.array(anthro_data['E_JJA_sd'])


# In[6]:


SON_data = pd.read_csv(bio_path+'SON_bio_diurnal_sys_errs_all_VPRM_abs_errs_SMUrF_fix.csv')
SMUrF_SON=np.array(SON_data['S_NEE'])
SMUrF_SON_sd=np.array(SON_data['S_NEE_sd'])
VPRM_SON=np.array(SON_data['V_NEE'])
VPRM_SON_sd=np.array(SON_data['V_NEE_sd'])
ODIAC_SON=np.array(anthro_data['O_SON'])
ODIAC_SON_sd=np.array(anthro_data['O_SON_sd'])
EDGAR_SON=np.array(anthro_data['E_SON'])
EDGAR_SON_sd=np.array(anthro_data['E_SON_sd'])


# In[7]:


DJF_data = pd.read_csv(bio_path+'DJF_bio_diurnal_sys_errs_all_VPRM_abs_errs_SMUrF_fix.csv')
SMUrF_DJF=np.array(DJF_data['S_NEE'])
SMUrF_DJF_sd=np.array(DJF_data['S_NEE_sd'])
VPRM_DJF=np.array(DJF_data['V_NEE'])
VPRM_DJF_sd=np.array(DJF_data['V_NEE_sd'])
ODIAC_DJF=np.array(anthro_data['O_DJF'])
ODIAC_DJF_sd=np.array(anthro_data['O_DJF_sd'])
EDGAR_DJF=np.array(anthro_data['E_DJF'])
EDGAR_DJF_sd=np.array(anthro_data['E_DJF_sd'])


# In[ ]:





# In[8]:


# Make an array with the time of day
HoY=np.arange(0,24)


# In[37]:


# *** Optional: Uncomment to plot the values of biogenic and anthropogenic CO2 fluxes only (not the ratio)

#plt.style.use('tableau-colorblind10')
#plt.rc('font',size=22)

#fig, ax = plt.subplots(1,4,sharex=True,sharey=True,figsize=(16,6))

#ax[0].errorbar(HoY,SMUrF_MAM,yerr=SMUrF_MAM_sd,marker='.',linestyle=' ',capsize=5,label='SMUrF')
#ax[0].errorbar(HoY,VPRM_MAM,yerr=VPRM_MAM_sd,marker='.',linestyle=' ',capsize=5,label='UrbanVPRM')
#ax[0].errorbar(HoY,EDGAR_MAM,yerr=EDGAR_MAM_sd,marker='^',linestyle=' ',capsize=5,label='EDGAR')
#ax[0].errorbar(HoY,ODIAC_MAM,yerr=ODIAC_MAM_sd,marker='^',linestyle=' ',capsize=5,label='ODIAC')

#ax[1].errorbar(HoY,SMUrF_JJA,yerr=SMUrF_JJA_sd,marker='.',linestyle=' ',capsize=5,label='SMUrF')
#ax[1].errorbar(HoY,VPRM_JJA,yerr=VPRM_JJA_sd,marker='.',linestyle=' ',capsize=5,label='UrbanVPRM')
#ax[1].errorbar(HoY,EDGAR_JJA,yerr=EDGAR_JJA_sd,marker='^',linestyle=' ',capsize=5,label='EDGAR')
#ax[1].errorbar(HoY,ODIAC_JJA,yerr=ODIAC_JJA_sd,marker='^',linestyle=' ',capsize=5,label='ODIAC')

#ax[2].errorbar(HoY,SMUrF_SON,yerr=SMUrF_SON_sd,marker='.',linestyle=' ',capsize=5,label='SMUrF')
#ax[2].errorbar(HoY,VPRM_SON,yerr=VPRM_SON_sd,marker='.',linestyle=' ',capsize=5,label='UrbanVPRM')
#ax[2].errorbar(HoY,EDGAR_SON,yerr=EDGAR_SON_sd,marker='^',linestyle=' ',capsize=5,label='EDGAR')
#ax[2].errorbar(HoY,ODIAC_SON,yerr=ODIAC_SON_sd,marker='^',linestyle=' ',capsize=5,label='ODIAC')

#ax[3].errorbar(HoY,SMUrF_DJF,yerr=SMUrF_DJF_sd,marker='.',linestyle=' ',capsize=5,label='SMUrF')
#ax[3].errorbar(HoY,VPRM_DJF,yerr=VPRM_DJF_sd,marker='.',linestyle=' ',capsize=5,label='UrbanVPRM')
#ax[3].errorbar(HoY,EDGAR_DJF,yerr=EDGAR_DJF_sd,marker='^',linestyle=' ',capsize=5,label='EDGAR')
#ax[3].errorbar(HoY,ODIAC_DJF,yerr=ODIAC_DJF_sd,marker='^',linestyle=' ',capsize=5,label='ODIAC')

#ax[1].legend(fontsize=18,loc='upper left')
#ax[0].set_xlabel('Hour of Day')
#ax[0].set_ylabel('CO$_2$ Fluxes ($\mu$mol m$^{-2}$ s$^{-1}$)')
#ax[0].set_title('Spring')
#ax[1].set_title('Summer')
#ax[2].set_title('Autumn')
#ax[3].set_title('Winter')
#fig.subplots_adjust(hspace=0,wspace=0)
#fig.show()
# End of uncomment ***


# In[9]:


# Use propagation of errors to estimate uncertainty in the ratios of biogenic to anthropogenic CO2 fluxes
SMUrF_EDGAR_MAM_err = np.sqrt((SMUrF_MAM_sd/EDGAR_MAM)**2+((SMUrF_MAM*EDGAR_MAM_sd)/(EDGAR_MAM**2))**2)
VPRM_EDGAR_MAM_err = np.sqrt((VPRM_MAM_sd/EDGAR_MAM)**2+((VPRM_MAM*EDGAR_MAM_sd)/(EDGAR_MAM**2))**2)

SMUrF_EDGAR_JJA_err = np.sqrt((SMUrF_JJA_sd/EDGAR_JJA)**2+((SMUrF_JJA*EDGAR_JJA_sd)/(EDGAR_JJA**2))**2)
VPRM_EDGAR_JJA_err = np.sqrt((VPRM_JJA_sd/EDGAR_JJA)**2+((VPRM_JJA*EDGAR_JJA_sd)/(EDGAR_JJA**2))**2)

SMUrF_EDGAR_SON_err = np.sqrt((SMUrF_SON_sd/EDGAR_SON)**2+((SMUrF_SON*EDGAR_SON_sd)/(EDGAR_SON**2))**2)
VPRM_EDGAR_SON_err = np.sqrt((VPRM_SON_sd/EDGAR_SON)**2+((VPRM_SON*EDGAR_SON_sd)/(EDGAR_SON**2))**2)

SMUrF_EDGAR_DJF_err = np.sqrt((SMUrF_DJF_sd/EDGAR_DJF)**2+((SMUrF_DJF*EDGAR_DJF_sd)/(EDGAR_DJF**2))**2)
VPRM_EDGAR_DJF_err = np.sqrt((VPRM_DJF_sd/EDGAR_DJF)**2+((VPRM_DJF*EDGAR_DJF_sd)/(EDGAR_DJF**2))**2)


# In[10]:


# Use propagation of errors to estimate uncertainty in the ratios of biogenic to anthropogenic CO2 fluxes
SMUrF_ODIAC_MAM_err = np.sqrt((SMUrF_MAM_sd/ODIAC_MAM)**2+((SMUrF_MAM*ODIAC_MAM_sd)/(ODIAC_MAM**2))**2)
VPRM_ODIAC_MAM_err = np.sqrt((VPRM_MAM_sd/ODIAC_MAM)**2+((VPRM_MAM*ODIAC_MAM_sd)/(ODIAC_MAM**2))**2)

SMUrF_ODIAC_JJA_err = np.sqrt((SMUrF_JJA_sd/ODIAC_JJA)**2+((SMUrF_JJA*ODIAC_JJA_sd)/(ODIAC_JJA**2))**2)
VPRM_ODIAC_JJA_err = np.sqrt((VPRM_JJA_sd/ODIAC_JJA)**2+((VPRM_JJA*ODIAC_JJA_sd)/(ODIAC_JJA**2))**2)

SMUrF_ODIAC_SON_err = np.sqrt((SMUrF_SON_sd/ODIAC_SON)**2+((SMUrF_SON*ODIAC_SON_sd)/(ODIAC_SON**2))**2)
VPRM_ODIAC_SON_err = np.sqrt((VPRM_SON_sd/ODIAC_SON)**2+((VPRM_SON*ODIAC_SON_sd)/(ODIAC_SON**2))**2)

SMUrF_ODIAC_DJF_err = np.sqrt((SMUrF_DJF_sd/ODIAC_DJF)**2+((SMUrF_DJF*ODIAC_DJF_sd)/(ODIAC_DJF**2))**2)
VPRM_ODIAC_DJF_err = np.sqrt((VPRM_DJF_sd/ODIAC_DJF)**2+((VPRM_DJF*ODIAC_DJF_sd)/(ODIAC_DJF**2))**2)


# In[ ]:





# In[40]:


# *** Optional uncomment to plot the ratio of biogenic to anthropogenic fluxes only

#plt.style.use('tableau-colorblind10')
#plt.rc('font',size=20)

#fig, ax = plt.subplots(1,4,sharex=True,sharey=True,figsize=(18,7))

#ax[0].errorbar(HoY,SMUrF_MAM/EDGAR_MAM,yerr=SMUrF_EDGAR_MAM_err,marker='.',linestyle=' ',capsize=5,label='SMUrF:EDGAR')
#ax[0].errorbar(HoY,VPRM_MAM/EDGAR_MAM,yerr=VPRM_EDGAR_MAM_err,marker='.',linestyle=' ',capsize=5,label='UrbanVPRM:EDGAR')

#ax[1].errorbar(HoY,SMUrF_JJA/EDGAR_JJA,yerr=SMUrF_EDGAR_JJA_err,marker='.',linestyle=' ',capsize=5,label='SMUrF:EDGAR')
#ax[1].errorbar(HoY,VPRM_JJA/EDGAR_JJA,yerr=VPRM_EDGAR_JJA_err,marker='.',linestyle=' ',capsize=5,label='UrbanVPRM:EDGAR')

#ax[2].errorbar(HoY,SMUrF_SON/EDGAR_SON,yerr=SMUrF_EDGAR_SON_err,marker='.',linestyle=' ',capsize=5,label='SMUrF:EDGAR')
#ax[2].errorbar(HoY,VPRM_SON/EDGAR_SON,yerr=VPRM_EDGAR_SON_err,marker='.',linestyle=' ',capsize=5,label='UrbanVPRM:EDGAR')

#ax[3].errorbar(HoY,SMUrF_DJF/EDGAR_DJF,yerr=SMUrF_EDGAR_DJF_err,marker='.',linestyle=' ',capsize=5,label='SMUrF:EDGAR')
#ax[3].errorbar(HoY,VPRM_DJF/EDGAR_DJF,yerr=VPRM_EDGAR_DJF_err,marker='.',linestyle=' ',capsize=5,label='UrbanVPRM:EDGAR')

#ax[3].legend(fontsize=18)
#ax[0].set_xlabel('Hour of Day')
#ax[1].set_xlabel('Hour of Day')
#ax[2].set_xlabel('Hour of Day')
#ax[3].set_xlabel('Hour of Day')
#ax[0].set_ylabel('Biogenic:Anthropogenic CO$_2$ Fluxes')
#ax[0].set_title('Spring')
#ax[1].set_title('Summer')
#ax[2].set_title('Autumn')
#ax[3].set_title('Winter')
#fig.subplots_adjust(hspace=0,wspace=0)
#fig.show()

# End of uncomment ***


# In[ ]:





# In[113]:


# Plot the values of the biogenic and anthropogenic CO2 fluxes and the ratio between biogenic & EDGAR 
# CO2 fluxes for each season (Figure 7)

plt.style.use('tableau-colorblind10')
plt.rc('font',size=22)

fig, ax = plt.subplots(2,4,sharex=True,figsize=(18,12))

ax[0,0].set_xlim(-1,24)
ax[0,0].set_ylim(-11,36)
ax[0,1].set_ylim(-11,36)
ax[0,2].set_ylim(-11,36)
ax[0,3].set_ylim(-11,36)

ax[0,0].errorbar(HoY,VPRM_MAM,yerr=VPRM_MAM_sd,marker='.',linestyle=' ',capsize=5,label='UrbanVPRM')
ax[0,0].errorbar(HoY,SMUrF_MAM,yerr=SMUrF_MAM_sd,marker='.',linestyle=' ',capsize=5,label='SMUrF')

ax[0,0].errorbar(HoY,EDGAR_MAM,yerr=EDGAR_MAM_sd,marker='^',linestyle=' ',capsize=5,label='EDGAR')
ax[0,0].errorbar(HoY,ODIAC_MAM,yerr=ODIAC_MAM_sd,marker='^',linestyle=' ',capsize=5,label='ODIAC')
ax[0,0].axhline(0,linestyle=':',c='k')
ax[0,0].text(-0.5,32.5,'(a)',c='k')


ax[0,1].scatter(-100,100,marker='o',label='UrbanVPRM')
ax[0,1].scatter(-100,100,marker='o',label='SMUrF')

ax[0,1].scatter(-100,100,marker='^',label='EDGAR')
ax[0,1].scatter(-100,100,marker='^',label='ODIAC')

ax[0,1].errorbar(HoY,VPRM_JJA,yerr=VPRM_JJA_sd,marker='.',linestyle=' ',capsize=5)
ax[0,1].errorbar(HoY,SMUrF_JJA,yerr=SMUrF_JJA_sd,marker='.',linestyle=' ',capsize=5)

ax[0,1].errorbar(HoY,EDGAR_JJA,yerr=EDGAR_JJA_sd,marker='^',linestyle=' ',capsize=5)
ax[0,1].errorbar(HoY,ODIAC_JJA,yerr=ODIAC_JJA_sd,marker='^',linestyle=' ',capsize=5)
ax[0,1].axhline(0,linestyle=':',c='k')
ax[0,1].text(-0.5,32.5,'(b)',c='k')

ax[0,2].errorbar(HoY,VPRM_SON,yerr=VPRM_SON_sd,marker='.',linestyle=' ',capsize=5,label='UrbanVPRM')
ax[0,2].errorbar(HoY,SMUrF_SON,yerr=SMUrF_SON_sd,marker='.',linestyle=' ',capsize=5,label='SMUrF')

ax[0,2].errorbar(HoY,EDGAR_SON,yerr=EDGAR_SON_sd,marker='^',linestyle=' ',capsize=5,label='EDGAR')
ax[0,2].errorbar(HoY,ODIAC_SON,yerr=ODIAC_SON_sd,marker='^',linestyle=' ',capsize=5,label='ODIAC')
ax[0,2].axhline(0,linestyle=':',c='k')
ax[0,2].text(-0.5,32.5,'(c)',c='k')

ax[0,3].errorbar(HoY,VPRM_DJF,yerr=VPRM_DJF_sd,marker='.',linestyle=' ',capsize=5,label='UrbanVPRM')
ax[0,3].errorbar(HoY,SMUrF_DJF,yerr=SMUrF_DJF_sd,marker='.',linestyle=' ',capsize=5,label='SMUrF')

ax[0,3].errorbar(HoY,EDGAR_DJF,yerr=EDGAR_DJF_sd,marker='^',linestyle=' ',capsize=5,label='EDGAR')
ax[0,3].errorbar(HoY,ODIAC_DJF,yerr=ODIAC_DJF_sd,marker='^',linestyle=' ',capsize=5,label='ODIAC')
ax[0,3].axhline(0,linestyle=':',c='k')
ax[0,3].text(-0.5,32.5,'(d)',c='k')

ax[0,1].set_yticks([])
ax[0,2].set_yticks([])
ax[0,3].set_yticks([])

ax[0,1].legend(fontsize=18,loc='upper right')
ax[0,0].set_ylabel('F$_{CO_2}$ ($\mu$mol m$^{-2}$ s$^{-1}$)')
ax[0,0].set_title('Spring')
ax[0,1].set_title('Summer')
ax[0,2].set_title('Autumn')
ax[0,3].set_title('Winter')


ax[1,0].set_ylim(-0.8,0.55)
ax[1,1].set_ylim(-0.8,0.55)
ax[1,2].set_ylim(-0.8,0.55)
ax[1,3].set_ylim(-0.8,0.55)

ax[1,0].errorbar(HoY,VPRM_MAM/EDGAR_MAM,yerr=VPRM_EDGAR_MAM_err,marker='.',linestyle=' ',capsize=5,label='UrbanVPRM:EDGAR')
ax[1,0].errorbar(HoY,SMUrF_MAM/EDGAR_MAM,yerr=SMUrF_EDGAR_MAM_err,marker='.',linestyle=' ',capsize=5,label='SMUrF:EDGAR')
ax[1,0].axhline(0,linestyle=':',c='k')
ax[1,0].text(-0.5,0.45,'(e)',c='k')

ax[1,1].errorbar(HoY,VPRM_JJA/EDGAR_JJA,yerr=VPRM_EDGAR_JJA_err,marker='.',linestyle=' ',capsize=5,label='UrbanVPRM:EDGAR')
ax[1,1].errorbar(HoY,SMUrF_JJA/EDGAR_JJA,yerr=SMUrF_EDGAR_JJA_err,marker='.',linestyle=' ',capsize=5,label='SMUrF:EDGAR')
ax[1,1].axhline(0,linestyle=':',c='k')
ax[1,1].text(-0.5,0.45,'(f)',c='k')

ax[1,2].errorbar(HoY,VPRM_SON/EDGAR_SON,yerr=VPRM_EDGAR_SON_err,marker='.',linestyle=' ',capsize=5,label='UrbanVPRM:EDGAR')
ax[1,2].errorbar(HoY,SMUrF_SON/EDGAR_SON,yerr=SMUrF_EDGAR_SON_err,marker='.',linestyle=' ',capsize=5,label='SMUrF:EDGAR')
ax[1,2].axhline(0,linestyle=':',c='k')
ax[1,2].text(-0.5,0.45,'(g)',c='k')

ax[1,3].scatter(-100,100,label='UrbanVPRM : EDGAR')
ax[1,3].scatter(-100,100,label='SMUrF : EDGAR')

ax[1,3].errorbar(HoY,VPRM_DJF/EDGAR_DJF,yerr=VPRM_EDGAR_DJF_err,marker='.',linestyle=' ',capsize=5)
ax[1,3].errorbar(HoY,SMUrF_DJF/EDGAR_DJF,yerr=SMUrF_EDGAR_DJF_err,marker='.',linestyle=' ',capsize=5)
ax[1,3].axhline(0,linestyle=':',c='k')
ax[1,3].text(-0.5,0.45,'(h)',c='k')

ax[1,1].set_yticks([])
ax[1,2].set_yticks([])
ax[1,3].set_yticks([])

ax[1,3].legend(fontsize=17,loc='lower left')
ax[1,0].set_xlabel('Hour of Day')
ax[1,1].set_xlabel('Hour of Day')
ax[1,2].set_xlabel('Hour of Day')
ax[1,3].set_xlabel('Hour of Day')
ax[1,0].set_ylabel('Biogenic : Anthropogenic F$_{CO_2}$')

fig.subplots_adjust(hspace=0,wspace=0)
## *** Uncomment to save figure as pdf and png. CHANGE FILENAME ***
#plt.savefig('Fixed_bio_anthro_CO2_flux_systematic_errs_VPRM_abs_err_python_0line_labelled.pdf',bbox_inches='tight')
#plt.savefig('Fixed_bio_anthro_CO2_flux_systematic_errs_VPRM_abs_err_python_0line_labelled.png',bbox_inches='tight')
fig.show()


# In[ ]:





# In[114]:


# For creating Fig. S7:

#Load in anthropogenic and biogenic fluxes for each month
# *** CHANGE FILENAME ***
# Create this file using 'Monthly_EDGAR_ODIAC_NEE_comparison.R'
Mo_data = pd.read_csv(bio_path+'anthro_bio_monthly_sys_errs_all_VPRM_abs_errs_SMUrF_fix.csv')
SMUrF_Mo=np.array(Mo_data['S_NEE'])
SMUrF_Mo_sd=np.array(Mo_data['S_NEE_sd'])
VPRM_Mo=np.array(Mo_data['V_NEE'])
VPRM_Mo_sd=np.array(Mo_data['V_NEE_sd'])
ODIAC_Mo=np.array(Mo_data['O_flx'])
EDGAR_Mo=np.array(Mo_data['E_flx'])


# In[115]:


# Make an array with the months
Mos=np.arange(1,13)


# In[116]:


# Use propagation of errors to estimate uncertainty in the ratios of biogenic to anthropogenic CO2 fluxes
SMUrF_EDGAR_Mo_err = np.sqrt((SMUrF_Mo_sd/EDGAR_Mo)**2)
VPRM_EDGAR_Mo_err = np.sqrt((VPRM_Mo_sd/EDGAR_Mo)**2)

SMUrF_ODIAC_Mo_err = np.sqrt((SMUrF_Mo_sd/ODIAC_Mo)**2)
VPRM_ODIAC_Mo_err = np.sqrt((VPRM_Mo_sd/ODIAC_Mo)**2)


# In[138]:


# Plot the values of the biogenic and anthropogenic CO2 fluxes and the ratio between biogenic & EDGAR 
# CO2 fluxes for each season (Figure 7)

plt.style.use('tableau-colorblind10')
plt.rc('font',size=22)

fig, ax = plt.subplots(1,2,sharex=True,figsize=(18,6))

ax[0].set_xlim(0.5,12.5)
ax[0].set_ylim(-5,30)

ax[0].errorbar(Mos,VPRM_Mo,yerr=VPRM_Mo_sd,marker='.',linestyle=' ',markersize=10,capsize=5)
ax[0].errorbar(Mos,SMUrF_Mo,yerr=SMUrF_Mo_sd,marker='.',linestyle=' ',markersize=10,capsize=5)
ax[0].scatter(-100,100,marker='o',label='UrbanVPRM')
ax[0].scatter(-100,100,marker='o',label='SMUrF')

ax[0].scatter(Mos,EDGAR_Mo,marker='^',s=60,label='EDGAR')
ax[0].scatter(Mos,ODIAC_Mo,marker='^',s=60,label='ODIAC')
ax[0].axhline(0,linestyle=':',c='k')
ax[0].text(0.65,27.25,'(a)',c='k')

ax[0].legend(fontsize=18,loc=(0.36,0.625))
ax[0].set_ylabel('F$_{CO_2}$ ($\mu$mol m$^{-2}$ s$^{-1}$)')
ax[0].set_title('Monthly-Averaged CO$_2$ Fluxes')

ax[1].set_ylim(-0.4,0.1)

ax[1].errorbar(Mos,VPRM_Mo/EDGAR_Mo,yerr=VPRM_EDGAR_Mo_err,marker='.',markersize=10,linestyle=' ',capsize=5)
ax[1].errorbar(Mos,SMUrF_Mo/EDGAR_Mo,yerr=SMUrF_EDGAR_Mo_err,marker='.',markersize=10,linestyle=' ',capsize=5)
ax[1].axhline(0,linestyle=':',c='k')
ax[1].text(0.65,0.06,'(b)',c='k')

ax[1].scatter(-100,100,label='UrbanVPRM : EDGAR')
ax[1].scatter(-100,100,label='SMUrF : EDGAR')

ax[1].legend(fontsize=17)
ax[0].set_xlabel('Month')
ax[1].set_xlabel('Month')
ax[1].set_ylabel('Biogenic : Anthropogenic F$_{CO_2}$')

ax[1].set_title('Monthly CO$_2$ Flux Ratios')

fig.subplots_adjust(wspace=0.25)
# *** Uncomment to save figure as pdf and png. CHANGE FILENAME ***
#plt.savefig('Monthly_bio_anthro_CO2_flux_systematic_errs_VPRM_abs_err_python_0line_labelled.pdf',bbox_inches='tight')
#plt.savefig('Monthly_bio_anthro_CO2_flux_systematic_errs_VPRM_abs_err_python_0line_labelled.png',bbox_inches='tight')
fig.show()


# In[ ]:




