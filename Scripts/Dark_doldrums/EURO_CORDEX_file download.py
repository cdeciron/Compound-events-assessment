#!/usr/bin/env python
# coding: utf-8

# In[2]:


#PACKAGES
from itertools import chain
import pandas as pd
import xclim
import numpy as np
import geopandas as gpd
import pooch
import cdsapi
import os
import xarray as xr
import json
import urllib
import pyproj
from pyproj import Proj, Transformer
import cartopy.crs as ccrs
import zipfile
import matplotlib.pyplot as plt
import re
import glob
import netCDF4
import cftime
import cartopy as cp
import cartopy.feature as cfeature
import plotly.express as px
import scipy.stats as st
from shapely.geometry import Point
import seaborn as sns
import math 
import ipywidgets as widgets
from localtileserver import get_leaflet_tile_layer, TileClient
from IPython.display import display
import rasterio
import plotly.graph_objects as go
from ipyleaflet import Map, DrawControl, Marker,LayersControl
import ipywidgets as widgets
import xclim.indices as xci
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
from dask.diagnostics import ProgressBar


# In[6]:


# Define the directory for the heatwave workflow preprocess
workflow_folder = "/Volumes/LaCie 1/Temperatures"  # Mac example
# Define directories for data and results within the previously defined workflow directory
data_dir = os.path.join(workflow_folder,'data')
results_dir = os.path.join(workflow_folder,'results')
# Check if the workflow directory exists, if not, create it along with subdirectories for data and results
if not os.path.exists(workflow_folder):
    os.makedirs(workflow_folder)
    os.makedirs(os.path.join(data_dir))
    os.makedirs(os.path.join(results_dir))


# In[14]:


#PROJECTION ROTATED POLE
scale=0.5

#defining region bounding box
#bbox=[np.min(coords_user[:,0]),np.min(coords_user[:,1]),np.max(coords_user[:,0]),np.max(coords_user[:,1])] #Only for the chosen region 
bbox = [4.096012, 57.736234, 32.177067, 71.599506]  # For all Norway (bounding box)

#setting up the projection transformation tool
crs = ccrs.RotatedPole(pole_latitude=39.25, pole_longitude=-162)
transformer = pyproj.Transformer.from_crs('epsg:4258',crs)

# New bbox coordinates matching EURO-CORDEX projection.
RLON_MIN, RLAT_MIN = transformer.transform(bbox[1], bbox[0])
RLON_MAX, RLAT_MAX = transformer.transform(bbox[3], bbox[2])

gcm = 'miroc_miroc5'
rcm = 'clmcom_clm_cclm4_8_17'
rcp = 'rcp8.5'


# In[16]:


#Remove previously downloaded files for the same model and scenario
pattern = f'{data_dir}/EUR-11*{gcm}*{rcp}*{rcm}*day*.nc'
for file_path in glob.glob(pattern):
    if os.path.exists(file_path):
        os.remove(file_path)

#start new download
zip_path_cordex = os.path.join(data_dir, 'cordex_data '+'miroc_miroc5'+'_'+'clmcom_clm_cclm4_8_17'+'midend'+'.zip')

URL = "https://cds.climate.copernicus.eu/api"
KEY = 'fb0bc1d0-2354-4d4b-a08a-788434d53503' # put your key here
c = cdsapi.Client(url=URL, key=KEY)

c.retrieve(
    'projections-cordex-domains-single-levels',
    {
        'domain': 'europe',
        'experiment': 'rcp_8_5',
        'horizontal_resolution': '0_11_degree_x_0_11_degree',
        'temporal_resolution': 'daily_mean',
        'variable': '2m_air_temperature',
        'gcm_model': 'miroc_miroc5',
        'rcm_model': 'clmcom_clm_cclm4_8_17',
        'ensemble_member': 'r1i1p1',
        'start_year': ['2086', '2091', '2096'],
        'end_year':  ['2090', '2095', '2100'],
        'format': 'zip',
    },
    zip_path_cordex)

with zipfile.ZipFile(zip_path_cordex, 'r') as zObject:
    zObject.extractall(path=data_dir)


# In[ ]:




