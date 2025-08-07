#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Basic packages
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray
import os
import re
#Spatial data handling
import geopandas as gpd
import cartopy.crs as ccrs
import regionmask
import rasterio 
import rasterio.features
import pyproj 
from pyproj import CRS, Transformer
from pyproj import Proj, Transformer
#Plotting
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Time periods as written in file names from the CDS EURO-CORDEX data 
## i allows iteration
n = 5 #usually 1 or 5, 
x = 4 #usually 0 or 4, number of years contained in one file 
### Historical periods
hist_period_dates = {
    f"{1971 + i*n}_hist": f"{1971 + i*n}0101-{1971 + i*n + x}1231"
    for i in range(5) #6 or 30 years
}

### Mid-century periods
mid_period_dates = {
    f"{i+x}_mid": f"{2036 + i*n}0101-{2036 + i*n + x}1231"
    for i in range(5)
}

### End-century periods
end_period_dates = {
    f"{i+x}_end": f"{2071 + i*n}0101-{2071 + i*n + x}1231"
    for i in range(5)
}


# In[ ]:


#Path to file buildup with help of variables 
location = 'Volumes'
disk = 'LaCie 1'
folder = 'Compound_events_study'
subfolder = 'Climate_models_data'
gcm_rcm_folder = 'CNRM_CERFACS_CNRM_CM5_CNRM_ALADIN63' #or 'MIROC_MIROC5_CLM_CLMcom4_8_17'.. ect
variable_folder = 'Wind'

#RIP nomenclature of the GCM-RCM combination
rNi1p1 = 'r1i1p1' #'r1i1p1' or 'r2ip1' or 'r3ip1' or 'r2ip1'

#Version number
v_number = 'v2' #'v1' or 'v2'; in file name

#Temporal resolution of the data 
time_resol = 'day' 


# In[ ]:


# Set up dictionnaries for folders and file names 
## Climate variables
variables = {
    "wind": {
        "code": "sfcWind",       
        "folder": "Wind"     
    }
}

##Climate model combinations
###Global climate models
GCM = {
    'CNRM': 'CNRM-CERFACS-CNRM-CM5',
    #'MPI': 'MPI-M-MPI-ESM-LR', 
    #'MIROC': 'MIROC-MIROC5',
    #'EcEarth': 'ICHEC-EC-EARTH',
    #'NorESM': 'NorESM1-M'
    #'HadGEM': 'HadGEM-2'
    #'MPI': 'MPI-M-MPI-ESM-LR'
}
###Regional climate models
RCM = {
    'R_CNRM':  'CNRM-ALADIN63',
    #'ITCP': 'ICTP-RegCM4-6',
    #'CLM': 'CLM-CCLM-CLMcom4-8-17'
    #'HIRHAM': 'DMI-HIRHAM5'
    #'RCA': 'SMHI-RCA4'
    #'REMO2015': 'GERICS-REMO2015'
}
###Dates of the time preiods
period_dates = {
    'hist': hist_period_dates, 
    'mid': mid_period_dates, 
    'end': end_period_dates
    
}
###Scenarios
scenario = {
    'hist': 'historical', 
    'RCP': 'rcp85'
}


# In[ ]:


#Gather the files by period based on the dates in the file name
def generate_file_paths(var_key):
    #Create one list for each period 
    files = {'hist': [], 'mid': [], 'end': []}
    
    if var_key not in variables:
        raise ValueError(f"Variable '{var_key}' not found in variables dictionary.")

    #Name of folder and var_code based on definition 
    var_info = variables[var_key]
    var_code = var_info['code']
    folder_name = var_info['folder']

    #For each GCM_RCM combination and each period (hist, mid, end)
    for gcm_key, gcm_val in GCM.items():
        for rcm_key, rcm_val in RCM.items():
            for period_type, period_dict in period_dates.items():
                scen = scenario['hist'] if period_type == 'hist' else scenario['RCP']
                for date_range in period_dict.values():
                    file_path = (
                        f"/{location}/{disk}/{folder}/{subfolder}/{gcm_rcm_folder}/Climate_raw_data/{variable_folder}/"
                       f"{var_code}_EUR-11_{gcm_val}_{scen}_{rNi1p1}_{rcm_val}_{v_number}_{time_resol}_{date_range}.nc"
                    )
                    #File path is registered in lists for periods 
                    if os.path.exists(file_path):
                        files[period_type].append(file_path)
                    else:
                        print(f"Missing: {file_path}")

    return files


# In[ ]:


#For each period one dataset is created 
def create_wind_datasets_from_file_dict(file_dict):
    return {
        "hist": xr.open_mfdataset(file_dict["hist"], combine="by_coords", decode_coords="all"),
        "mid":  xr.open_mfdataset(file_dict["mid"],  combine="by_coords", decode_coords="all"),
        "end":  xr.open_mfdataset(file_dict["end"],  combine="by_coords", decode_coords="all"),
    }


# In[ ]:


file_dict = generate_file_paths("wind")  
wind_datasets = create_wind_datasets_from_file_dict(file_dict)

wind_ds_hist = wind_datasets["hist"]
wind_ds_mid  = wind_datasets["mid"]
wind_ds_end  = wind_datasets["end"]


# In[ ]:


wind_datasets = {
    'hist': xr.open_mfdataset(file_dict['hist'], chunks={'time': 100}),
    'mid':  xr.open_mfdataset(file_dict['mid'], chunks={'time': 100}),
    'end':  xr.open_mfdataset(file_dict['end'], chunks={'time': 100}),
}


# In[ ]:


#Path to folder of shapefiles
shp_path = '/Users/Camille//Documents/Stages/Stage S.Mayer/Cartes/Shapefiles'
#Enter the folder path to the shapefiles of Norway and for regions
norway_shp_folder = f'{shp_path}/Norway_E_maps.qgz'
elspot_regions_folder = f'{shp_path}/Elspot_regions_PostProcessed'

#Define your geographical coordinates system
crs_name = "EPSG:4326" #WGS84

#bboxes available on bbox finder to determine the square area of your regions
bbox_no  = [4.096012, 57.736234, 32.177067, 71.599506] # For all Norway (bounding box)
bbox_er1 = [6.833496, 58.688359, 13.908691, 62.885205] #bbox for NO1 region
bbox_er2 = [4.514952, 57.705340, 12.952452, 60.963527] #bbox for NO2 region
bbox_er3 = [1.593189, 58.712348, 17.149830, 65.848681] #bbox for NO3 region
bbox_er4 = [7.646484, 63.918058, 32.167969, 71.635993] #bbox for NO4 region
bbox_er5 = [-1.113567, 56.213244, 14.443073, 63.874893] #bbox for NO5 region

#Transformation of the data's crs (rotated pole) to a WGS84 (EPSG:4326)
original_crs = ccrs.RotatedPole(pole_latitude=39.25, pole_longitude=-162)
transformer = pyproj.Transformer.from_crs(crs_name, original_crs)

#New bbox coordinates matching EURO-CORDEX projection
RLON_MIN, RLAT_MIN = transformer.transform(bbox_no[1], bbox_no[0])
RLON_MAX, RLAT_MAX = transformer.transform(bbox_no[3], bbox_no[2]) 


# In[ ]:


#Enter the file path to your shapefiles
shapefiles = {
    'Norway': f'{shp_path}/Norway_E_maps.qgz/gadm41_NOR_1.shp', 
    'NO1': f'{shp_path}/Elspot_regions_PostProcessed/NO1_Land_Availability.shp', 
    'NO2': f'{shp_path}/Elspot_regions_PostProcessed/NO2_Land_Availability.shp', 
    'NO3': f'{shp_path}/Elspot_regions_PostProcessed/NO3_Land_Availability.shp', 
    'NO4': f'{shp_path}/Elspot_regions_PostProcessed/NO4_Land_Availability.shp', 
    'NO5': f'{shp_path}/Elspot_regions_PostProcessed/NO5_Land_Availability.shp'
}

bounding_boxes = {
    'NO': bbox_no, 
    'NO1': bbox_er1, 
    'NO2': bbox_er2, 
    'NO3': bbox_er3,
    'NO4': bbox_er4, 
    'NO5': bbox_er5
}


# In[ ]:


#Function to read the shapefiles paths and convert it to WGS84
shapes = {}
for name, path in shapefiles.items():
    gdf = gpd.read_file(path)
    gdf = gdf.to_crs(crs_name)
    shapes[name] = gdf

#Set crs to WGS84 for the datasets 
for name, ds in wind_datasets.items():
    wind_datasets[name] = ds.rio.write_crs(crs_name)


# In[ ]:


#Restrain the data to Norway bbox only instead of whole Europe in original downloaded data 
def clip_dataset_with_bbox(ds, region_name, bounding_boxes):
    bbox = bounding_boxes[region_name]
    coords = ds.coords.keys()

    #If coordinates are called 'rlat' and 'rlon' format
    if 'rlat' in coords and 'rlon' in coords:
        # rotated pole coordinates - use .sel()
        RLON_MIN, RLAT_MIN = transformer.transform(bbox[1], bbox[0])  # lat, lon order!
        RLON_MAX, RLAT_MAX = transformer.transform(bbox[3], bbox[2])
        ds_sliced = ds.sel(rlat=slice(RLAT_MIN, RLAT_MAX), rlon=slice(RLON_MIN, RLON_MAX))
        return ds_sliced

    #If coordinates are called 'x' and 'y' format
    elif 'x' in coords and 'y' in coords:
        # lat/lon are variables, not coordinates -> mask using .where()
        lon = ds['lon']
        lat = ds['lat']

        mask = (
            (lon >= bbox[0]) & (lon <= bbox[2]) &
            (lat >= bbox[1]) & (lat <= bbox[3])
        )

        if hasattr(mask, 'compute'):
            mask = mask.compute()

        if mask.sum().values == 0:
            print(f"✗ No data points selected for region {region_name}")
            return None

        ds_sliced = ds.where(mask, drop=True)
        return ds_sliced

    else:
        print(f"✗ Dataset does not have known spatial coords for region {region_name}")
        return None


# In[ ]:


sliced_by_region = {}
#Data is sliced for each region based on the bounding boxes coordinates
for period, ds in wind_datasets.items():
    print(f"\n=== Period: {period} ===")
    sliced_by_region[period] = {}
    for region in bounding_boxes:
        sliced = clip_dataset_with_bbox(ds, region, bounding_boxes)
        if sliced is None:
            print(f"✗ Failed to slice {period} for {region}")


# In[ ]:


#Compute regional mean time series for overlapping regions using a 3D mask.
#Returns a DataFrame with time as index and regions as columns
def compute_regional_means(ds, shapes):
    # Create regionmask.Regions object
    regions = regionmask.Regions([gdf.geometry.values[0] for gdf in shapes.values()],
                                 names=list(shapes.keys()))

    # Create a 3D mask (region × y × x)
    mask_3d = regions.mask_3D(ds)

    df = pd.DataFrame(index=pd.to_datetime(ds.time.values))

    # Iterate over regions shapefiles and compute mean
    for i, name in enumerate(shapes.keys()):
        region_mask = mask_3d.isel(region=i)
        masked_data = ds.rsds.where(region_mask)
        regional_mean = masked_data.mean(dim=("y", "x"), skipna=True)
        df[name] = regional_mean.compute().values

    return df

# ===== Run and save CSVs =====
#Enter your output file
output_dir = f"/{location}/{disk}/{folder}/{subfolder}/{gcm_rcm_folder}/Post_processed_data/sfc_wind/"
os.makedirs(output_dir, exist_ok=True)

#Create a csv for each period of time containing data for the 5 regions+Norway 
for period, ds in wind_datasets.items():
    print(f"Processing {period}...")
    df = compute_regional_means(ds, shapes)
    output_path = os.path.join(output_dir, f"regional_mean_wind_{period}.csv")
    df.to_csv(output_path)
    print(f"Saved: {output_path}")


# In[ ]:




