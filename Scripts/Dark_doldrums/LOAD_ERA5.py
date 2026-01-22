#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Basic packages
import pandas as pd
import numpy as np
import xarray as xr
import os
import re
#Spatial data handling
import geopandas as gpd
import cartopy.crs as ccrs
import pyproj
from pyproj import Proj, Transformer
import regionmask
#Plotting
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


import cdsapi

# Initialize CDS API client (make sure you have ~/.cdsapirc configured)
c = cdsapi.Client()

dataset = "derived-era5-single-levels-daily-statistics"

request = {
    "product_type": "reanalysis",
    "variable": ["surface_solar_radiation_downwards"],
    "year": ["1961"],
    "month": [
        "01", "02", "03", "04", "05", "06",
        "07", "08", "09", "10", "11", "12"
    ],
    "day": [
        "01", "02", "03", "04", "05", "06",
        "07", "08", "09", "10", "11", "12",
        "13", "14", "15", "16", "17", "18",
        "19", "20", "21", "22", "23", "24",
        "25", "26", "27", "28", "29", "30", "31"
    ],
    "daily_statistic": "daily_mean",
    "time_zone": "utc+00:00",  # CDS only supports UTC, not offset zones
    "area": [71.82, 3.06, 56.89, 34.7],  # North, West, South, East
    "format": "netcdf",  # Always specify format
}

c.retrieve(dataset, request, "era5_ssrd_1961_daily_mean.nc")


# In[ ]:


#Path to file buildup with help of variables (refer to folders structure figure)
location = 'Volumes'
disk = 'LaCie 1'
folder = 'Compound_events_study_folder'
subfolder = 'Climate_data_reanalysis_ERA5'

input_path = f'/{location}/{disk}/{folder}/{subfolder}'


# In[ ]:


crs_name = "EPSG:4326"  # Euro-CORDEX is usually WGS84 lat/lon

era5_tas2m = xr.open_dataset(f"{input_path}/data_stream-moda_stepType-avgua.nc")
era5_pr = xr.open_dataset(f"{input_path}/data_stream-moda_stepType-avgad.nc")

# Select the variable you want and ensure lat/lon are set as coordinates
tas2m_var = era5_tas2m['t2m']  # replace 'tas' with your actual variable name
pr_var = era5_pr['tp']         # replace 'pr' with your actual variable name

# Ensure the coords are ordered correctly (y=lat, x=lon)
tas2m_var = tas2m_var.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=False)
pr_var = pr_var.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=False)

# Write CRS so .rio knows the projection
tas2m_var = tas2m_var.rio.write_crs(crs_name)
pr_var = pr_var.rio.write_crs(crs_name)

# Put them in a dict
era5_datasets = {
    "tas2m": tas2m_var,
    "pr": pr_var
}


# In[ ]:


#Path to folder of shapefiles
geo_data_path = f"/{location}/{disk}/{folder}/Geospatial_data"

#Enter the folder path to the shapefiles of Norway and for regions
norway_shp_folder = f'{geo_data_path}/Norway_E_maps.qgz'
elspot_regions_folder = f'{geo_data_path}/Elspot_regions_PostProcessed'

#Define your geographical coordinates system
crs_name = "EPSG:4326"

#PROJECTION ROTATED POLE
scale=0.5

#bboxes available on bbox finder to determine the square area of your regions
bbox_no  = [4.096012, 57.736234, 32.177067, 71.599506] # For all Norway (bounding box)
bbox_er1 = [6.833496, 58.688359, 13.908691, 62.885205] #bbox for NO1 region
bbox_er2 = [4.514952, 57.705340, 12.952452, 60.963527] #bbox for NO2 region
bbox_er3 = [1.593189, 58.712348, 17.149830, 65.848681] #bbox for NO3 region
bbox_er4 = [7.646484, 63.918058, 32.167969, 71.635993] #bbox for NO4 region
bbox_er5 = [-1.113567, 56.213244, 14.443073, 63.874893] #bbox for NO5 region

#TRansformation of the crs to a more usual one 
original_crs = ccrs.RotatedPole(pole_latitude=39.25, pole_longitude=-162)
transformer = pyproj.Transformer.from_crs(crs_name, original_crs)

# New bbox coordinates matching EURO-CORDEX projection.
RLON_MIN, RLAT_MIN = transformer.transform(bbox_no[1], bbox_no[0])
RLON_MAX, RLAT_MAX = transformer.transform(bbox_no[3], bbox_no[2]) 


# In[ ]:


#Enter the file path to your shapefiles
shapefiles = {
    'Norway': f'{geo_data_path}/Norway_E_maps.qgz/gadm41_NOR_1.shp', 
    'NO1': f'{geo_data_path}/Elspot_regions_PostProcessed/NO1_Land_Availability.shp', 
    'NO2': f'{geo_data_path}/Elspot_regions_PostProcessed/NO2_Land_Availability.shp', 
    'NO3': f'{geo_data_path}/Elspot_regions_PostProcessed/NO3_Land_Availability.shp', 
    'NO4': f'{geo_data_path}/Elspot_regions_PostProcessed/NO4_Land_Availability.shp', 
    'NO5': f'{geo_data_path}/Elspot_regions_PostProcessed/NO5_Land_Availability.shp'
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


shapes = {}
for name, path in shapefiles.items():
    gdf = gpd.read_file(path)
    gdf = gdf.to_crs(crs_name)
    shapes[name] = gdf


# In[ ]:


def clip_dataset_with_bbox(ds, region_name, bounding_boxes):
    bbox = bounding_boxes[region_name]  # (min_lon, min_lat, max_lon, max_lat)
    lon_min, lat_min, lon_max, lat_max = bbox

    dims = list(ds.dims)

    # ERA5 / regular lat-lon
    if "latitude" in dims and "longitude" in dims:
        lat_min, lat_max = sorted([lat_min, lat_max])
        return ds.sel(
            latitude=slice(lat_max, lat_min),  # reversed if lat decreasing
            longitude=slice(lon_min, lon_max)
        )

    # CORDEX rotated grid
    elif "rlat" in dims and "rlon" in dims:
        return ds.sel(
            rlat=slice(lat_min, lat_max),
            rlon=slice(lon_min, lon_max)
        )

    else:
        print(f"✗ No recognized spatial dimensions in dataset for {region_name}")
        return None


# In[ ]:


sliced_by_region = {}

for period, ds in era5_datasets.items():
    print(f"\n=== Period: {period} ===")
    sliced_by_region[period] = {}
    for region in bounding_boxes:
        sliced = clip_dataset_with_bbox(ds, region, bounding_boxes)
        if sliced is not None:
            sliced_by_region[period][region] = sliced
            print(f"✓ Sliced {period} for {region} - dims: {sliced.dims}")
        else:
            print(f"✗ Failed to slice {period} for {region}")


# In[ ]:


for period, ds in pr_temp_datasets.items():
    print(f"\nVariables in {period}:")
    print(ds.data_vars)


# In[ ]:


era5_datasets_clean = {
    name: da.drop_vars("expver") if "expver" in da.coords else da
    for name, da in era5_datasets.items()
}

era5_ds = xr.Dataset(era5_datasets_clean)

for var in ['tas2m', 'pr']:
    if var not in era5_ds:
        print(f"{var} not found, skipping.")
        continue
    
    df = era5_ds[var].mean(dim=["latitude", "longitude"]).to_dataframe()
    output_path = os.path.join(input_path, f"regional_mean_{var}.csv")
    df.to_csv(output_path)
    print(f"Saved: {output_path}")


# In[ ]:


def compute_regional_means(ds, shapes):
    """
    Compute regional mean time series for 'tasmin' and 'pr' over defined regions.
    Returns a DataFrame with time as index and each column named <region>_<variable>.
    """
     # Create regionmask.Regions object from shape geometries
    regions = regionmask.Regions(
        [gdf.geometry.values[0] for gdf in shapes.values()],
        names=list(shapes.keys())
    )

    # Create a 3D mask (region × y × x)
    mask_3d = regions.mask_3D(ds)  # Do NOT pass 'overlap' here

    # Initialize output DataFrame with time index
    df = pd.DataFrame(index=ds.indexes['time'])  # ✅ uses CFTimeIndex directly


    # Variables to process
    variables = [var for var in ['tasmin', 'pr'] if var in ds]

    # Loop through regions
    for i, name in enumerate(shapes.keys()):
        region_mask = mask_3d.isel(region=i)

        for var in variables:
            masked_data = ds[var].where(region_mask)
            if var == 'tasmin':
                regional_mean = (masked_data.mean(dim=("rlat", "rlon"), skipna=True))-273.15
            if var == 'pr':
                regional_mean = (masked_data.mean(dim=("rlat", "rlon"), skipna=True))*86400
                
            df[f"{name}_{var}"] = regional_mean.compute().values

    return df






# In[ ]:




