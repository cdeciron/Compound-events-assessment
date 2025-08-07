# Compound events relevant for the renewable energy system in Norway
This project aims to study climate related-hazards relevant for the renewable energy system in Norway, and specifically focuses on 3 periods: 1971-2000 (historical), 2036-2065 (mid-century) and 2071-2100 (end-centruy) under a RCP8.5 scenario. Two types of compound events can be assessed: cold-dry temporally compounding events and low solar-low wind multivariate events (_Zscheischler et al., 2020_). The project allows a regional study in Norway based on the 5 energy price regions. The two compound events studied are relevant for the hydro, solar and wind power sectors and aims to study events that can  impact the demand and/or the production 

# Definition of climate hazards and compound events 
## Cold-dry events
A cold spell is when at least 5 days where the minimum temperature is below the 10th percentile of the time serie 
Dry spells are defined based on the standardized precipitation evapotranspiration index (SPEI) (_Thornwaite48_)
A temporally compounding cold-dry event is considered when summer months are relatively drier than average and winter displays cold spells. 

## Dark doldrums 
A low wind spell is when 100m wind speed is below 3m/s for at least 5 days
A dark spell is when surface solar radiation downwards (rsds) is below 200W/m<sup>2</sup>
A dark doldrum is the co-occurence of these two events for at least 1 day in the same region 

# Data
## Climate variables
- Daily minimum temperature (tasmin)
- Daily average 2m air temperature (tas)
- Daily mean precipitation flux (pr)
- Daily 10m wind speed (sfcwind)
- 3-hourly surface solar radiation downwards (rsds)

## Geospatial data
- Shapefile of Norway (or any other country)
- Shapefiles of Norway's energy price regions + natural protected areas

## Model combinations (GCM-RCM) used for this study
Global climate models - regional climate models were chosen according _Antonini et al., 2024_ and _Climate in Norway 2100_
- CNRM-CERFACS-CNRM-CM5 - CNRM-ALADIN63 
- MPI-M-MPI-ESM-LR - ITCP-RegCM4-6
- MIROC-MIROC5 - CLM-CCLM-CLMcom4-8-17
- ICHEC-EC-EARTH - DMI-HIRHAM5
- ICHEC-EC-EARTH - SMHI-RCA4
- NorESM1-M - SMHI-RCA4
- NorESM1-M - GERICS-REMO2015
- HadGEM-2 - GERICS-REMO2015
- HadGEM-2 - SMHI-RCA4
- MPI-M-MPI-ESM-LR - CLM-CCLM-CLMcom4-8-17
- MPI-M-MPI-ESM-LR - GERICS-REMO2015

## Data storage and folder structure
A .zip file is included in this project and it is crucial to download it as it follows the exact same folder structure as used in the scripts. You will find also geospatial data/shapefiles for norway and its electricity price regions. 

# Scripts
The project consists in several scripts, each being essential to assess the probbaility of compound events as previously described in the introduction. 

1) <ins>File loading</ins>
_The script 'file loading' allows a preprocessing of the data: unit conversion, data clipping on shapefiles, regional averaging, exportation of the data as a csv file_
Input: climate variables files, 2 at the same time

2) <ins>Cold-dry events</ins>
_This script allow the calculation of the probability of a compound event to occur and provides a hypotheses test to assess the significance of the obtained results_

<ins>Input</ins>: tasmin_pr CSVs

3) <ins>Dark doldrums</ins>
_This script allows the calculation of the probability of a dark doldrum to occur and provides a hypotheses test to assess the significance of the obtained results_

<ins>Input</ins>: sfcwind and rsds CSVs








