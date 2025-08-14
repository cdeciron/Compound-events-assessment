#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Packages
import pandas as pd
import numpy as np
import pooch
import cdsapi
import os
import xarray as xr
import json
import urllib
import zipfile
import matplotlib.pyplot as plt
import geopandas as gpd
import re
import glob
import cftime
import seaborn as sns
from shapely.ops import unary_union
from sklearn.preprocessing import normalize
from typing import List, Union, Optional
from rasterio.transform import from_bounds
import scipy.stats as st
from scipy.stats import bootstrap
from scipy.stats import kurtosis
#Randomization of the data
import random 


# In[97]:


#Imput variables to find the path to your files
location = 'Volumes'
disk = 'LaCie 1'
folder = 'Compound_events_study_folder'
subfolder = 'Climate_models_data'
gcm_rcm_folder = 'CNRM_CERFACS_CNRM_CM5_CNRM_ALADIN63' #depending on the gcm-rcm combination you are using here 
subfolder_2 = 'Post_processed_data'

input_path = f'/{location}/{disk}/{folder}/{subfolder}/{gcm_rcm_folder}/{subfolder_2}'
output_path = f'/{location}/{disk}/{folder}/{subfolder}/{gcm_rcm_folder}/Figures/Dark_doldrums'


# In[4]:


#Dictionnaries to find the csv files for all periods and for both variables
period_names = {
    'hist': 'historical', 
    'mid': 'mid-century', 
    'end': 'end-century'
}

variable_list = {
    'rsds': 'Solar',
    'sfcWind': 'Wind'
}


# In[5]:


#Define a function to convert 3hr solar data into daily data 'L' or 'D' (light or dark day) 
def label_dark_light_days(solar_series, solar_thresh=200):
    solar_series = pd.to_numeric(solar_series, errors='coerce')
    solar_series.index = pd.to_datetime(solar_series.index)

    daily_labels = {}
    daily_grouped = solar_series.groupby(solar_series.index.date)

    for day, group in daily_grouped:
        if (group > solar_thresh).sum() <= 2:
            daily_labels[pd.to_datetime(day)] = 'D'
        else:
            daily_labels[pd.to_datetime(day)] = 'L'

    return pd.Series(daily_labels, name="day_label")


# In[6]:


#Define the dark spell detection function 
def dark_spell(df, solar_threshold=200, count_threshold=5):
    zones = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5']
    dark_spell_count = {}
    df = df.copy()
    
    # Ensure index is datetime and normalized (midnight)
    df.index = pd.to_datetime(df.index).normalize()
    
    for zone in zones:
        solar_col = f"{zone}_solar"
        if solar_col not in df.columns:
            raise ValueError(f"Missing column: {solar_col}")
        
        # Boolean: True where 'D'
        dark_flag = (df[solar_col] == 'D')
        dark_spell_flags = pd.Series(False, index=df.index)
        
        # Identify runs
        run_ids = (dark_flag != dark_flag.shift()).cumsum()
        groups = dark_flag.groupby(run_ids)
        
        # Count spells
        spell_count = 0
        for _, group_vals in groups:
            if group_vals.iloc[0] and len(group_vals) >= count_threshold:
                spell_count += 1
                dark_spell_flags.loc[group_vals.index] = True
        
        dark_spell_count[zone] = spell_count
        df[f'{zone}_dark_spell'] = dark_spell_flags
    
    return df, dark_spell_count


# In[7]:


#Define the low wind spells detection function
def low_wind_spell(df, wind_thresh = 4, count_threshold = 5):
    zones = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5']
    low_wind_spell_count = {}
    df = df.copy()
    
    # Ensure index is datetime and normalized (midnight)
    df.index = pd.to_datetime(df.index).normalize()

    for zone in zones:
        wind_col = f"{zone}_wind"
        if wind_col not in df.columns:
            raise ValueError(f"Missing column: {wind_col}")

        # Boolean: True if wind < threshold
        low_wind_flag = (df[wind_col] < wind_thresh)
        low_wind_spell_flags = pd.Series(False, index=df.index)

        # Identify consecutive runs
        wind_runs = (low_wind_flag != low_wind_flag.shift()).cumsum()
        wind_groups = low_wind_flag.groupby(wind_runs)

        # Count spells and mark flags
        spell_count = 0
        for _, group_vals in wind_groups:
            if group_vals.iloc[0] and len(group_vals) >= count_threshold:
                spell_count += 1
                low_wind_spell_flags.loc[group_vals.index] = True

        low_wind_spell_count[zone] = spell_count
        df[f'{zone}_low_wind_spell'] = low_wind_spell_flags
        
    return df, low_wind_spell_count
        


# In[8]:


#Convert 10m height wind to 100m height wind according to the power law 
def power_law(wind_df):
    z = 100
    zref = 10
    alpha = 0.143

    dataset = wind_df.copy()
    if isinstance(dataset.index, pd.DatetimeIndex):
        dataset = dataset.sort_index()

    df_wind_100m = pd.DataFrame(index=dataset.index)
    for zone in ['NO1', 'NO2', 'NO3', 'NO4', 'NO5']:
        df_wind_100m[zone] = dataset[f'{zone}'] * (z / zref) ** alpha

    return df_wind_100m


# In[9]:


#Pre-processing the csv files: merge wind and solar data on daily basis and keeping data only for winter months
merged_daily_dict = {}  # Final output per period

for period_key, period_code in period_names.items():
    # --- Load solar (3-hourly) ---
    solar_path = f"{input_path}/rsds/regional_mean_rsds_{period_key}.csv"
    solar_df = pd.read_csv(solar_path, index_col=0, parse_dates=True)
    solar_df = solar_df.rename(columns={zone: f"{zone}_solar" for zone in solar_df.columns})

    # --- Label each day 'L' or 'D' for each zone ---
    solar_labels = {}
    for zone in solar_df.columns:
        solar_labels[zone] = label_dark_light_days(solar_df[zone], solar_thresh=200)

    solar_daily_labels = pd.DataFrame(solar_labels)

    # Ensure common datetime index with wind df(daily at 00:00)
    solar_daily_labels.index = pd.to_datetime(solar_daily_labels.index).normalize()

    # --- Load wind (daily) ---
    wind_path = f"{input_path}/sfcWind/regional_mean_sfcWind_{period_key}.csv"
    wind_df = pd.read_csv(wind_path, index_col=0, parse_dates=True)
    wind_df.index = wind_df.index.normalize()
    df_wind_100m = power_law(wind_df)  # apply the power law to the current wind_df
    df_wind_100m = df_wind_100m.rename(columns={zone: f"{zone}_wind" for zone in df_wind_100m.columns})
    
    # --- Merge on daily index ---
    merged_df = solar_daily_labels.merge(df_wind_100m, left_index=True, right_index=True, how="left")

    # --- Filter for winter months October to March ---
    merged_df = merged_df[merged_df.index.month.isin([10, 11, 12, 1, 2, 3])]

    merged_daily_dict[period_code] = merged_df


# In[10]:


#Define the compound events function 
def compound_events(df, wind_thresh=4, min_spell_length=5):
    zones = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5']

    df = df.copy()
    df.index = pd.to_datetime(df.index).normalize()

    low_wind_spell_count = {}
    dark_spell_count = {}
    compound_event_count = {}
    prob_ce = {}
    zone_dark_doldrum_days = {}
    compound_event_record = []

    for zone in zones:
        solar_col = f"{zone}_solar"
        wind_col = f"{zone}_wind"

        # --- Dark spells: runs of 5+ consecutive days where solar == 'D' ---
        dark_flag = (df[solar_col] == 'D')
        dark_spell_flags = pd.Series(False, index=df.index)

        dark_runs = (dark_flag != dark_flag.shift()).cumsum()
        dark_groups = dark_flag.groupby(dark_runs)

        dark_spell_count_zone = 0
        for group_id, group_vals in dark_groups:
            if group_vals.iloc[0] == True and len(group_vals) >= min_spell_length:
                dark_spell_count_zone += 1
                dark_spell_flags.loc[group_vals.index] = True

        df[f'{zone}_dark_spell'] = dark_spell_flags

        # --- Low wind spells: runs of 5+ consecutive days where wind < wind_thresh ---
        low_wind_flag = (df[wind_col] < wind_thresh)
        low_wind_spell_flags = pd.Series(False, index=df.index)

        wind_runs = (low_wind_flag != low_wind_flag.shift()).cumsum()
        wind_groups = low_wind_flag.groupby(wind_runs)

        low_wind_spell_count_zone = 0
        for group_id, group_vals in wind_groups:
            if group_vals.iloc[0] == True and len(group_vals) >= min_spell_length:
                low_wind_spell_count_zone += 1
                low_wind_spell_flags.loc[group_vals.index] = True

        df[f'{zone}_low_wind_spell'] = low_wind_spell_flags

        # --- Overlapping compound events ---
        overlap = dark_spell_flags & low_wind_spell_flags
        overlap_dates = df.index[overlap]

        zone_dark_doldrum_days[zone] = overlap_dates

        for dt in overlap_dates:
            try:
                idx = df.index.get_loc(dt)
                compound_event_record.append({'zone': zone, 'date': dt, 'original_index': idx})
            except KeyError:
                pass

        low_wind_spell_count[zone] = low_wind_spell_count_zone
        dark_spell_count[zone] = dark_spell_count_zone
        compound_event_count[zone] = len(overlap_dates)

        if len(df) > 0:
            prob_ce[zone] = round((len(overlap_dates) / len(df)) * 100, 3)
        else:
            prob_ce[zone] = 0

    compound_event_index_dict = {
        zone: [df.index.get_loc(dt) for dt in dates if dt in df.index]
        for zone, dates in zone_dark_doldrum_days.items()
    }

    compound_event_index_df = pd.DataFrame({
        zone: pd.Series(idxs) for zone, idxs in compound_event_index_dict.items()
    })

    compound_event_indexes = pd.DataFrame(compound_event_record)

    print('Low wind spells:', low_wind_spell_count)
    print('Dark spells:', dark_spell_count)
    print('Compound events (CEs):', compound_event_count)
    print('Probability of a CE (%):', prob_ce)

    return (
        low_wind_spell_count,
        dark_spell_count,
        prob_ce,
        compound_event_count
    )


# In[11]:


#Results for number of hazards and compound events for all periods
low_wind_count_hist, dark_days_count_hist, prob_hist, ce_count_hist = compound_events(merged_daily_dict['historical'])
low_wind_count_mid, dark_days_count_mid, prob_mid, ce_count_mid = compound_events(merged_daily_dict['mid-century'])
low_wind_count_end, dark_days_count_end, prob_end, ce_count_end = compound_events(merged_daily_dict['end-century'])


# In[17]:


#Bootstrapping experiment  
def hypotheses_test(df, n_bootstrap=1000, wind_thresh=4, min_spell_length=5):
    """
    Perform bootstrap hypothesis testing for compound events (dark + low wind spells).

    Returns a DataFrame with bootstrapped counts:
    - Rows: bootstrap iterations
    - Columns: zones NO1-NO5
    """
    zones = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5']
    results = []

    for i in range(n_bootstrap):
        # Resample rows with replacement (shuffle but keep index)
        synthetic_df = df.sample(n=len(df), replace=True).reset_index(drop=True)

        iteration_counts = {}
        for zone in zones:
            solar_series = synthetic_df[f'{zone}_solar']
            wind_series = synthetic_df[f'{zone}_wind']

            # --- Dark spells ---
            dark_flag = solar_series
            dark_runs = (dark_flag != dark_flag.shift()).cumsum()
            dark_groups = dark_flag.groupby(dark_runs)
            dark_spell_flags = pd.Series(False, index=synthetic_df.index)
            for _, group_vals in dark_groups:
                if group_vals.iloc[0] and len(group_vals) >= min_spell_length:
                    dark_spell_flags.loc[group_vals.index] = True

            # --- Low wind spells ---
            low_wind_flag = (wind_series < wind_thresh)
            wind_runs = (low_wind_flag != low_wind_flag.shift()).cumsum()
            wind_groups = low_wind_flag.groupby(wind_runs)
            low_wind_spell_flags = pd.Series(False, index=synthetic_df.index)
            for _, group_vals in wind_groups:
                if group_vals.iloc[0] and len(group_vals) >= min_spell_length:
                    low_wind_spell_flags.loc[group_vals.index] = True

            # --- Compound events ---
            overlap = dark_spell_flags & low_wind_spell_flags
            iteration_counts[zone] = overlap.sum()

        results.append(iteration_counts)

    # Convert list of dicts to DataFrame
    bootstrapped_df = pd.DataFrame(results)
    return bootstrapped_df


# In[19]:


boostrapped_hist = hypotheses_test(merged_daily_dict['historical'], 1000)


# In[20]:


boostrapped_mid = hypotheses_test(merged_daily_dict['mid-century'], 1000)


# In[21]:


boostrapped_end = hypotheses_test(merged_daily_dict['end-century'], 1000)


# In[99]:


sns.set_style("whitegrid")
sns.set_context("paper")
#Plotting frequency histograms 
def plot_hypothesis_test_histograms_freq(
    bstp_hist, bstp_mid, bstp_end, og_ce_count_hist, og_ce_count_mid, og_ce_count_end, label, output_path, n_number
):
    # Convert each bootstrap set to DataFrame
    hist_df = pd.DataFrame(bstp_hist)
    mid_df  = pd.DataFrame(bstp_mid)
    end_df  = pd.DataFrame(bstp_end)

    # Iterate over regions
    for region in hist_df.columns:
        data_hist = hist_df[region]
        data_mid  = mid_df[region]
        data_end  = end_df[region]

        fig, ax = plt.subplots(figsize=(7, 5))

        # Plot all three periods with different colors
        sns.histplot(data_hist, bins=15, kde=True, color="#4c72b0", edgecolor='black', label="Historical", ax=ax, alpha=0.5)
        sns.histplot(data_mid,  bins=15, kde=True, color="#55a868", edgecolor='black', label="Mid-century", ax=ax, alpha=0.5)
        sns.histplot(data_end,  bins=15, kde=True, color="#c44e52", edgecolor='black', label="End-century", ax=ax, alpha=0.5)

        # Add vertical line for observed/original value
        threshold_hist = og_ce_count_hist.get(region, None)
        threshold_mid = og_ce_count_mid.get(region, None)
        threshold_end = og_ce_count_end.get(region, None)
        
        ax.axvline(threshold_hist, color="#4c72b0", linestyle="--", linewidth=1.5)
        ax.axvline(threshold_mid, color="#55a868", linestyle="--", linewidth=1.5)
        ax.axvline(threshold_end, color="#c44e52", linestyle="--", linewidth=1.5)

        # Formatting
        ax.set_title(f'{region} Compound Events ({n_number})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Compound Events', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        sns.despine(trim=True)
        ax.legend(frameon=False, fontsize=10)

        # Save
        save_path = f"{output_path}/Bootstrap/{region}_CE_all_periods_{label}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)


# In[101]:


plot_hypothesis_test_histograms_freq(boostrapped_hist, boostrapped_mid, boostrapped_end, 
                                                ce_count_hist, ce_count_mid, ce_count_end,
                                                'freq', output_path, 'n=1000')


# In[102]:


sns.set_style("whitegrid")
sns.set_context("paper")
#Plotting density histograms
def plot_hypothesis_test_histograms_density(
    bstp_hist, bstp_mid, bstp_end, og_ce_count_hist, og_ce_count_mid, og_ce_count_end, label, output_path, n_number
):
    # Convert each bootstrap set to DataFrame
    hist_df = pd.DataFrame(bstp_hist)
    mid_df  = pd.DataFrame(bstp_mid)
    end_df  = pd.DataFrame(bstp_end)

    for region in hist_df.columns:
        data_hist = hist_df[region].dropna()
        data_mid  = mid_df[region].dropna()
        data_end  = end_df[region].dropna()

        # Define common bin edges for all three datasets
        all_data = pd.concat([data_hist, data_mid, data_end])
        bins = np.histogram_bin_edges(all_data, bins=15)

        fig, ax = plt.subplots(figsize=(7, 5))

        # Plot histograms with density scaling
        sns.histplot(
            data_hist, bins=bins, stat="density", kde=True,
            color="#4c72b0", edgecolor='black', label="Historical", ax=ax, alpha=0.5
        )
        sns.histplot(
            data_mid, bins=bins, stat="density", kde=True,
            color="#55a868", edgecolor='black', label="Mid-century", ax=ax, alpha=0.5
        )
        sns.histplot(
            data_end, bins=bins, stat="density", kde=True,
            color="#c44e52", edgecolor='black', label="End-century", ax=ax, alpha=0.5
        )

        # Add vertical line for observed/original value
        threshold_hist = og_ce_count_hist.get(region, None)
        threshold_mid = og_ce_count_mid.get(region, None)
        threshold_end = og_ce_count_end.get(region, None)
        
        ax.axvline(threshold_hist, color="#4c72b0", linestyle="--", linewidth=1.5)
        ax.axvline(threshold_mid, color="#55a868", linestyle="--", linewidth=1.5)
        ax.axvline(threshold_end, color="#c44e52", linestyle="--", linewidth=1.5)

        # Labels and styling
        ax.set_title(f'{region} Compound Events ({n_number})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Compound Events', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        sns.despine(trim=True)
        ax.legend(frameon=False, fontsize=10)

        # Save file
        save_path = f"{output_path}/Bootstrap/{region}_CE_all_periods_{label}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)


# In[103]:


plot_hypothesis_test_histograms_density(boostrapped_hist, boostrapped_mid, boostrapped_end, 
                                                ce_count_hist, ce_count_mid, ce_count_end,
                                                'density', output_path, 'n=1000')


# In[ ]:




