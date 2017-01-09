# Predict-Benzene-Concentration

Predict the concentration of Benzene from a set of measurements of other particles taken from several sensors.

The goal of this challenge is to predict the Benzene concentration from several particle and climatic measurements taken from different sensors.

The dataset contains 9358 instances of hourly averaged responses from an array of 5 metal oxide chemical sensors embedded in an Air Quality Chemical Multisensor Device. Ground Truth hourly averaged concentrations for CO, Non Metanic Hydrocarbons, Benzene, Total Nitrogen Oxides (NOx) and Nitrogen Dioxide (NO2) and were provided by a co-located reference certified analyzer. Evidences of cross-sensitivities as well as both concept and sensor drifts are present as described in De Vito et al., Sens. And Act. B, Vol. 129,2,2008 eventually affecting sensors concentration estimation capabilities. Missing values are tagged with -999 value.

Attribute Information:

CO_GT: True hourly averaged concentration CO in mg/m^3 (reference analyzer) 
PT08_S1_CO (tin oxide) hourly averaged sensor response (nominally CO targeted) 
NMHC_GT: True hourly averaged overall Non Metanic HydroCarbons concentration in microg/m^3 (reference analyzer) 
PT08.S2 (titania) hourly averaged sensor response (nominally NMHC targeted) 
Nox_GT: True hourly averaged NOx concentration in ppb (reference analyzer) 
PT08_S3_Nox: (tungsten oxide) hourly averaged sensor response (nominally NOx targeted) 
NO2_GT: True hourly averaged NO2 concentration in microg/m^3 (reference analyzer) 
PT08_S4_NO2 (tungsten oxide) hourly averaged sensor response (nominally NO2 targeted) 
PT08_S5_O3 (indium oxide) hourly averaged sensor response (nominally O3 targeted) 
T2: Temperature in Â°C 
RH: Relative Humidity (%) 
AH: Absolute Humidity
C6H6_GT: True hourly averaged Benzene concentration in microg/m^3 (reference analyzer) 
