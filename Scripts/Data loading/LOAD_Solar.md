```python
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
```


```python
# Time periods as written in file names from the CDS EURO-CORDEX data 
## i is the number of files corresponding to a period. If 1 file = 1 year thn i=30. If 1 file = 5 years, i=5
n = 1 #usually 1 or 5, number of years contained in one file 
x = 0 #usually 0 or 4

### Historical periods
hist_period_dates = {
    f"{1971 + i*n}_hist": f"{1971 + i*n}01010130-{1971 + i*n + x}12312230"
    for i in range(30) #6 or 30 years
}

### Mid-century periods
mid_period_dates = {
    f"{i+x}_mid": f"{2036 + i*n}01010130-{2036 + i*n + x}12312230"
    for i in range(30)
}

### End-century periods
end_period_dates = {
    f"{i+x}_end": f"{2071 + i*n}01010130-{2071 + i*n + x}12312230"
    for i in range(30)
}
```


```python
#Path to file buildup with help of variables (refer to folders structure figure)
location = 'Volumes'
disk = 'LaCie 1'
folder = 'Compound_events_study_github'
subfolder = 'Climate_models_data'
gcm_rcm_folder = 'CNRM_CERFACS_CNRM_CM5_CNRM_ALADIN63' #or 'MIROC_MIROC5_CLM_CLMcom4_8_17'.. ect
variable_folder = 'Solar'

#Variables for the files name, to be adapted depending on the GCM-RCM combination you are using 
##RIP nomenclature of the GCM-RCM combination
rNi1p1 = 'r1i1p1' #'r1i1p1' or 'r2ip1' or 'r3ip1' or 'r2ip1'
##Version number
v_number = 'v2' #'v1' or 'v2'; in file name
##Temporal resolution of the data 
time_resol = '3hr' #'day' or 3hr for solar 
```


```python
# Set up dictionnaries for folders and files names 
## Climate variables
variables = {
    "solar": {
        "code": "rsds",       
        "folder": "Solar"     
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

## Dates of the time periods previously defined 
period_dates = {
    'hist': hist_period_dates, 
    'mid': mid_period_dates, 
    'end': end_period_dates
    
}

##Name of the time periods
period_names = {
    'hist': 'historical', 
    'mid': 'mid_century', 
    'end': 'end_century'
}

##Scenarios
scenario = {
    'hist': 'historical', 
    'RCP': 'rcp85'
}

periods_intervalls = {
    "historical": ("1971-01-01", "2000-12-31"),
    "near_future": ("2021-01-01", "2050-12-31"),
    "far_future": ("2071-01-01", "2100-12-31")
}
```


```python
def generate_file_paths(var_key):
    files = {'hist': [], 'mid': [], 'end': []}

    if var_key not in variables:
        raise ValueError(f"Variable '{var_key}' not found in variables dictionary.")

    var_info = variables[var_key]
    var_code = var_info['code']
    folder_name = var_info['folder']

    for gcm_key, gcm_val in GCM.items():
        for rcm_key, rcm_val in RCM.items():
            for period_type, period_dict in period_dates.items():
                scen = scenario['hist'] if period_type == 'hist' else scenario['RCP']

                for label, date_range in period_dict.items():
                    file_path = (
                        f"/{location}/{disk}/{folder}/{subfolder}/{gcm_rcm_folder}/Climate_raw_data/{variable_folder}/"
                       f"{var_code}_EUR-11_{gcm_val}_{scen}_{rNi1p1}_{rcm_val}_{v_number}_{time_resol}_{date_range}.nc"
                    )

                    if os.path.exists(file_path):
                        files[period_type].append(file_path)
                    else:
                        print(f"Missing: {file_path}")

    return files

```


```python
def create_solar_datasets_from_file_dict(file_dict):
    return {
        "hist": xr.open_mfdataset(file_dict["hist"], combine="by_coords", decode_coords="all"),
        "mid":  xr.open_mfdataset(file_dict["mid"],  combine="by_coords", decode_coords="all"),
        "end":  xr.open_mfdataset(file_dict["end"],  combine="by_coords", decode_coords="all"),
    }

```


```python
file_dict = generate_file_paths("solar") 

solar_datasets = create_solar_datasets_from_file_dict(file_dict)

solar_ds_hist = solar_datasets["hist"]
solar_ds_mid  = solar_datasets["mid"]
solar_ds_end  = solar_datasets["end"]
```


```python
solar_datasets = {
    'hist': xr.open_mfdataset(file_dict['hist'], chunks={'time': 100}),
    'mid':  xr.open_mfdataset(file_dict['mid'], chunks={'time': 100}),
    'end':  xr.open_mfdataset(file_dict['end'], chunks={'time': 100}),
}
```


```python
#Path to folder of shapefiles
geo_data_path = f"/{location}/{disk}/Compound_events_study_github/Geospatial_data"

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
```


```python
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
```


```python
shapes = {}
for name, path in shapefiles.items():
    gdf = gpd.read_file(path)
    gdf = gdf.to_crs(crs_name)
    shapes[name] = gdf

for name, ds in solar_datasets.items():
    solar_datasets[name] = ds.rio.write_crs(crs_name)
```


```python
def clip_dataset_with_bbox(ds, region_name, bounding_boxes):
    bbox = bounding_boxes[region_name]

    coords = ds.coords.keys()

    if 'rlat' in coords and 'rlon' in coords:
        # rotated pole coordinates - use .sel()
        print(f"→ Clipping using rotated coordinates (rlat/rlon) for region: {region_name}")
        RLON_MIN, RLAT_MIN = transformer.transform(bbox[1], bbox[0])  # lat, lon order!
        RLON_MAX, RLAT_MAX = transformer.transform(bbox[3], bbox[2])
        ds_sliced = ds.sel(rlat=slice(RLAT_MIN, RLAT_MAX), rlon=slice(RLON_MIN, RLON_MAX))
        return ds_sliced

    elif 'x' in coords and 'y' in coords:
        # lat/lon are variables, not coordinates -> mask using .where()
        print(f"→ Clipping using lat/lon data variables mask for region: {region_name}")

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

```


```python
sliced_by_region = {}

for period, ds in solar_datasets.items():
    print(f"\n=== Period: {period} ===")
    sliced_by_region[period] = {}
    for region in bounding_boxes:
        sliced = clip_dataset_with_bbox(ds, region, bounding_boxes)
        if sliced is not None:
            sliced_by_region[period][region] = sliced
            print(f"✓ Sliced {period} for {region} - dims: {sliced.dims}")
        else:
            print(f"✗ Failed to slice {period} for {region}")

```

    
    === Period: hist ===
    → Clipping using lat/lon data variables mask for region: NO
    ✓ Sliced hist for NO - dims: FrozenMappingWarningOnValuesAccess({'time': 87664, 'axis_nbounds': 2, 'y': 139, 'x': 133, 'nvertex': 4})
    → Clipping using lat/lon data variables mask for region: NO1
    ✓ Sliced hist for NO1 - dims: FrozenMappingWarningOnValuesAccess({'time': 87664, 'axis_nbounds': 2, 'y': 38, 'x': 33, 'nvertex': 4})
    → Clipping using lat/lon data variables mask for region: NO2
    ✓ Sliced hist for NO2 - dims: FrozenMappingWarningOnValuesAccess({'time': 87664, 'axis_nbounds': 2, 'y': 30, 'x': 40, 'nvertex': 4})
    → Clipping using lat/lon data variables mask for region: NO3
    ✓ Sliced hist for NO3 - dims: FrozenMappingWarningOnValuesAccess({'time': 87664, 'axis_nbounds': 2, 'y': 67, 'x': 73, 'nvertex': 4})
    → Clipping using lat/lon data variables mask for region: NO4
    ✓ Sliced hist for NO4 - dims: FrozenMappingWarningOnValuesAccess({'time': 87664, 'axis_nbounds': 2, 'y': 83, 'x': 98, 'nvertex': 4})
    → Clipping using lat/lon data variables mask for region: NO5
    ✓ Sliced hist for NO5 - dims: FrozenMappingWarningOnValuesAccess({'time': 87664, 'axis_nbounds': 2, 'y': 73, 'x': 77, 'nvertex': 4})
    
    === Period: mid ===
    → Clipping using lat/lon data variables mask for region: NO
    ✓ Sliced mid for NO - dims: FrozenMappingWarningOnValuesAccess({'time': 87664, 'axis_nbounds': 2, 'y': 139, 'x': 133, 'nvertex': 4})
    → Clipping using lat/lon data variables mask for region: NO1
    ✓ Sliced mid for NO1 - dims: FrozenMappingWarningOnValuesAccess({'time': 87664, 'axis_nbounds': 2, 'y': 38, 'x': 33, 'nvertex': 4})
    → Clipping using lat/lon data variables mask for region: NO2
    ✓ Sliced mid for NO2 - dims: FrozenMappingWarningOnValuesAccess({'time': 87664, 'axis_nbounds': 2, 'y': 30, 'x': 40, 'nvertex': 4})
    → Clipping using lat/lon data variables mask for region: NO3
    ✓ Sliced mid for NO3 - dims: FrozenMappingWarningOnValuesAccess({'time': 87664, 'axis_nbounds': 2, 'y': 67, 'x': 73, 'nvertex': 4})
    → Clipping using lat/lon data variables mask for region: NO4
    ✓ Sliced mid for NO4 - dims: FrozenMappingWarningOnValuesAccess({'time': 87664, 'axis_nbounds': 2, 'y': 83, 'x': 98, 'nvertex': 4})
    → Clipping using lat/lon data variables mask for region: NO5
    ✓ Sliced mid for NO5 - dims: FrozenMappingWarningOnValuesAccess({'time': 87664, 'axis_nbounds': 2, 'y': 73, 'x': 77, 'nvertex': 4})
    
    === Period: end ===
    → Clipping using lat/lon data variables mask for region: NO
    ✓ Sliced end for NO - dims: FrozenMappingWarningOnValuesAccess({'time': 87656, 'axis_nbounds': 2, 'y': 139, 'x': 133, 'nvertex': 4})
    → Clipping using lat/lon data variables mask for region: NO1
    ✓ Sliced end for NO1 - dims: FrozenMappingWarningOnValuesAccess({'time': 87656, 'axis_nbounds': 2, 'y': 38, 'x': 33, 'nvertex': 4})
    → Clipping using lat/lon data variables mask for region: NO2
    ✓ Sliced end for NO2 - dims: FrozenMappingWarningOnValuesAccess({'time': 87656, 'axis_nbounds': 2, 'y': 30, 'x': 40, 'nvertex': 4})
    → Clipping using lat/lon data variables mask for region: NO3
    ✓ Sliced end for NO3 - dims: FrozenMappingWarningOnValuesAccess({'time': 87656, 'axis_nbounds': 2, 'y': 67, 'x': 73, 'nvertex': 4})
    → Clipping using lat/lon data variables mask for region: NO4
    ✓ Sliced end for NO4 - dims: FrozenMappingWarningOnValuesAccess({'time': 87656, 'axis_nbounds': 2, 'y': 83, 'x': 98, 'nvertex': 4})
    → Clipping using lat/lon data variables mask for region: NO5
    ✓ Sliced end for NO5 - dims: FrozenMappingWarningOnValuesAccess({'time': 87656, 'axis_nbounds': 2, 'y': 73, 'x': 77, 'nvertex': 4})



```python
def compute_regional_means(ds, shapes):
    """
    Compute regional mean time series for overlapping regions using a 3D mask.
    Returns a DataFrame with time as index and regions as columns.
    """
    import regionmask

    # Create regionmask.Regions object
    regions = regionmask.Regions([gdf.geometry.values[0] for gdf in shapes.values()],
                                 names=list(shapes.keys()))

    # Create a 3D mask (region × y × x)
    mask_3d = regions.mask_3D(ds)

    df = pd.DataFrame(index=pd.to_datetime(ds.time.values))

    # Iterate over regions and compute mean
    for i, name in enumerate(shapes.keys()):
        region_mask = mask_3d.isel(region=i)
        masked_data = ds.rsds.where(region_mask)
        regional_mean = masked_data.mean(dim=("y", "x"), skipna=True)
        df[name] = regional_mean.compute().values

    return df

# ===== Run and save CSVs =====
output_dir = f"/Volumes/LaCie 1/{folder}/{subfolder}/{gcm_rcm_folder}/Post_processed_data/rsds_3h/"
os.makedirs(output_dir, exist_ok=True)

for period, ds in solar_datasets.items():
    print(f"Processing {period}...")
    df = compute_regional_means(ds, shapes)
    output_path = os.path.join(output_dir, f"regional_mean_rsds_{period}.csv")
    df.to_csv(output_path)
    print(f"Saved: {output_path}")

```

    Processing hist...


    /Applications/anaconda3/lib/python3.12/site-packages/regionmask/core/mask.py:406: UserWarning: Detected overlapping regions. As of v0.11.0 these are correctly taken into account. Note, however, that a different mask is returned than with older versions of regionmask. To suppress this warning, set `overlap=True` (to restore the old, incorrect, behaviour, set `overlap=False`).
      warnings.warn(



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[87], line 32
         30 for period, ds in solar_datasets.items():
         31     print(f"Processing {period}...")
    ---> 32     df = compute_regional_means(ds, shapes)
         33     output_path = os.path.join(output_dir, f"regional_mean_rsds_{period}.csv")
         34     df.to_csv(output_path)


    Cell In[87], line 22, in compute_regional_means(ds, shapes)
         20     masked_data = ds.rsds.where(region_mask)
         21     regional_mean = masked_data.mean(dim=("y", "x"), skipna=True)
    ---> 22     df[name] = regional_mean.compute().values
         24 return df


    File /Applications/anaconda3/lib/python3.12/site-packages/xarray/core/dataarray.py:1189, in DataArray.compute(self, **kwargs)
       1164 """Manually trigger loading of this array's data from disk or a
       1165 remote source into memory and return a new array.
       1166 
       (...)
       1186 dask.compute
       1187 """
       1188 new = self.copy(deep=False)
    -> 1189 return new.load(**kwargs)


    File /Applications/anaconda3/lib/python3.12/site-packages/xarray/core/dataarray.py:1157, in DataArray.load(self, **kwargs)
       1137 def load(self, **kwargs) -> Self:
       1138     """Manually trigger loading of this array's data from disk or a
       1139     remote source into memory and return this array.
       1140 
       (...)
       1155     dask.compute
       1156     """
    -> 1157     ds = self._to_temp_dataset().load(**kwargs)
       1158     new = self._from_temp_dataset(ds)
       1159     self._variable = new._variable


    File /Applications/anaconda3/lib/python3.12/site-packages/xarray/core/dataset.py:542, in Dataset.load(self, **kwargs)
        539 chunkmanager = get_chunked_array_type(*lazy_data.values())
        541 # evaluate all the chunked arrays simultaneously
    --> 542 evaluated_data: tuple[np.ndarray[Any, Any], ...] = chunkmanager.compute(
        543     *lazy_data.values(), **kwargs
        544 )
        546 for k, data in zip(lazy_data, evaluated_data, strict=False):
        547     self.variables[k].data = data


    File /Applications/anaconda3/lib/python3.12/site-packages/xarray/namedarray/daskmanager.py:85, in DaskManager.compute(self, *data, **kwargs)
         80 def compute(
         81     self, *data: Any, **kwargs: Any
         82 ) -> tuple[np.ndarray[Any, _DType_co], ...]:
         83     from dask.array import compute
    ---> 85     return compute(*data, **kwargs)


    File /Applications/anaconda3/lib/python3.12/site-packages/dask/base.py:664, in compute(traverse, optimize_graph, scheduler, get, *args, **kwargs)
        661     postcomputes.append(x.__dask_postcompute__())
        663 with shorten_traceback():
    --> 664     results = schedule(dsk, keys, **kwargs)
        666 return repack([f(r, *a) for r, (f, a) in zip(results, postcomputes)])


    File /Applications/anaconda3/lib/python3.12/queue.py:171, in Queue.get(self, block, timeout)
        169 elif timeout is None:
        170     while not self._qsize():
    --> 171         self.not_empty.wait()
        172 elif timeout < 0:
        173     raise ValueError("'timeout' must be a non-negative number")


    File /Applications/anaconda3/lib/python3.12/threading.py:355, in Condition.wait(self, timeout)
        353 try:    # restore state no matter what (e.g., KeyboardInterrupt)
        354     if timeout is None:
    --> 355         waiter.acquire()
        356         gotit = True
        357     else:


    KeyboardInterrupt: 



```python

```
