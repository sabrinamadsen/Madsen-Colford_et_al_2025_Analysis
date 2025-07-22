# Madsen-Colford_et_al_2025_Analysis

# This document provides outline the codes used to perform analysis and create figures in the paper "Modification and comparison of two urban vegetation models over Southern Ontario, Canada" by Madsen-Colford et al.

# Codes with the same name but different file endings (i.e. .py and .ipynb) are identical.

# The plot of land cover and impervious surface area fraction (Fig. 1 of the main text) can be generated using 'Plot_land_cover_and_impervious_surfaces_clean.py'

# The files 'TROPOMI_SMUrF_CSIF_SMUrF_Fluxtower_Hourly_Comparison-V061_clean.py' and 'Plot_500m_original_final_V061_UrbanVPRM_Fluxtower_analysis_comparison_clean.py' are used to compare hourly and 8-day average biogenic CO2 fluxes from SMUrF (Figs. 2 c & d) and UrbanVPRM (Figs. 2 a & b), respectively, to those estimated by the eddy-covariance flux towers (used to generate Fig. 2 of the main text).

# 'TROPOMI_SMUrF_CSIF_SMUrF_Fluxtower_Seasonal_Comparison-V061_clean.py' and '500m_original_final_V061_UrbanVPRM_Fluxtower_Seasonal_Comparison_clean.py' are used to compare season-specific biogenic CO2 fluxes from SMUrF (Figs. 3 e-h) and UrbanVPRM (Figs. 3 a-d), respectively, to those estimated by the eddy-covariance flux towers (used to generate Fig 3 of the main text). '500m_original_final_V061_UrbanVPRM_Fluxtower_Seasonal_Comparison_clean.py' is also used to generate the plot that illustrates the GPP mismatch just after sunrise and before sunset (Fig. S3).

# 'UrbanVPRM_SMUrF_hourly_spatial_correlation_Toronto_Comparison.py' is used to compute the correlation statistics between SMUrF and UrbanVPRM over the city of Toronto in 2018 (generates Fig. 4 of the main text).

# The files 'Original_UrbanVPRM_SMUrF_Toronto_seasonal_comparison_clean.py' and 'UrbanVPRM_SMUrF_Toronto_Seasonal_Comparison_clean.py' are used to spatially compare the original and updated SMUrF and UrbanVPRM, respectively, over the city of Toronto (creates figures required for Fig. 5 of the main text). 'UrbanVPRM_SMUrF_Toronto_Seasonal_Comparison_clean.py' is also used to generate the timeseries of NEE, Reco, and GPP estimated by the updated SMUrF and UrbanVPRM in different parts of the city (Fig. 6 of the main text), and the plot comparing 8-day average NEE estimated by the updated SMUrF (before and after ISA correction) and UrbanVPRM (Fig. S5 of the supplemental material).

# Comparison between the model difference in NEE and the difference in the driving data (EVI-SIF) was done using 'SIF_vs_EVI_spatial_comparison_clean.py' (Fig. 7 of the main text). This code also generates the correlation plots between difference in NEE and EVI-SIF at different timescales (Fig. S6 of the supplement).

# Code required to estimate seasonal-averaged diurnal fluxes from biogenic (outputs of the updated SMUrF model and UrbanVPRM) and anthropogenic (from EDGAR and ODIAC, downscaled to hourly using TIMES emission factors) CO2 fluxes is found in the file 'Diurnal_uncertainty.r'. The monthly averaged anthropogenic and biogenic CO2 fluxes are computed using the file 'Monthly_EDGAR_ODIAC_NEE_comparison.r'. The results of both are plotted using the file 'Bio_anthro_FCO2_ratio_plot.py' (Fig. 8 of the main text for seasonal diurnal fluxes and Fig. S7 for monthly-averaged fluxes).

# Comparison of CSIF and TROPOMI SIF, downscaled using MODIS NIRv is available in 'Downscaled_CSIF_vs_Downscaled_TROPOMI_SIF_Clean.py' (Fig. S1 of the supplemental material)

# Code required to correct downscaled TROPOMI SIF near the shoreline of Toronto can be found in the file 'Toronto_tree_analysis_clean.r'. The spatial comparison used in Fig. S2 of the supplement was generated using 'Tree_corrected_vs_uncorrected_SIF.py'

# The file 'UrbanVPRM_SMUrF_Toronto_hourly_correlation_clean.py' was used to investigate the correlation between NEE estimated by the updated SMUrF and UrbanVPRM at different timescales (Fig. S4 of the supplemental material).

# 'Interannual_variability_VPRM_SMUrF_clean.py' compares the annual-averaged NEE, GPP, and Reco from SMUrF and UrbanVPRM from 2018-2021 (Fig. S8 of the supplemental material)

# 'Toronto_Boundary.shp' is a shapefile of the boundary of the city of Toronto. It is used in many of the scripts above to identify the boundaries of the city.
