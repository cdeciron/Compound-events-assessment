#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Packages
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


# In[876]:


#Imput variables to find the path to your files
location = 'Volumes'
disk = 'LaCie 1'
folder = 'Compound_events_study_folder'
subfolder = 'Climate_models_data'
gcm_rcm_folder = 'CNRM_CERFACS_CNRM_CM5_CNRM_ALADIN63' #depending on the gcm-rcm combination you are using here 
subfolder_2 = 'Post_processed_data'
scen = 'RCP26'
input_path = f'/{location}/{disk}/{folder}/{subfolder}/{gcm_rcm_folder}/{subfolder_2}'

#Customizable path to where you want to store your results
output_path = f'/{location}/{disk}/{folder}/{subfolder}/{gcm_rcm_folder}/Figures/Dark_doldrums/Bootstrap'


# In[878]:


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


# In[880]:


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


# In[882]:


def label_dark_light_days(solar_series, solar_thresh=200):
    # Convert series to numeric
    solar_series = pd.to_numeric(solar_series, errors='coerce')
    
    # Ensure the index is datetime
    if not pd.api.types.is_datetime64_any_dtype(solar_series.index):
        solar_series.index = pd.to_datetime(solar_series.index, errors='coerce', infer_datetime_format=True)
    
    daily_labels = {}
    daily_grouped = solar_series.groupby(solar_series.index.date)
    
    for day, group in daily_grouped:
        if (group > solar_thresh).sum() <= 1:
            daily_labels[pd.to_datetime(day)] = 'D'
        else:
            daily_labels[pd.to_datetime(day)] = 'L'
            
    # Return a Series with proper datetime index
    return pd.Series(daily_labels, name="day_label")
    #return pd.Series(daily_labels, index=pd.to_datetime(list(daily_labels.keys())))
    warnings.simplefilter(action='ignore', category=FutureWarning)


# In[888]:


#Pre-processing the csv files: merge wind and solar data on daily basis and keeping data only for winter months
merged_daily_dict = {}  # Final output per period

for period_key, period_code in period_names.items():
    # --- Load solar (3-hourly) ---
    solar_path = f"{input_path}/rsds/{scen}/regional_mean_rsds_{period_key}.csv"
    solar_df = pd.read_csv(solar_path, index_col=0, parse_dates=True)
    solar_df = solar_df.rename(columns={zone: f"{zone}_solar" for zone in solar_df.columns})

    # --- Label each day 'L' or 'D' for each zone ---
    solar_labels = {}
    for zone in solar_df.columns:
        solar_labels[zone] = label_dark_light_days(solar_df[zone], solar_thresh=200)

    solar_daily_labels = pd.DataFrame(solar_labels)

    # Ensure common datetime index with wind df(daily at 00:00)
    solar_daily_labels.index = pd.to_datetime(solar_daily_labels.index).normalize()
    #solar_series.index = pd.to_datetime(solar_series.index, errors='coerce', infer_datetime_format=True)
    #solar_series.index = pd.to_datetime(solar_series.index, format="%Y-%m-%d %H:%M:%S", errors='coerce')

    
    # --- Load wind (daily) ---
    wind_path = f"{input_path}/sfc_wind/{scen}/regional_mean_wind_{period_key}_{scen}.csv"
    wind_df = pd.read_csv(wind_path, index_col=0, parse_dates=True)
    wind_df.index = wind_df.index.normalize()
    df_wind_100m = power_law(wind_df)  # apply the power law to the current wind_df
    df_wind_100m = df_wind_100m.rename(columns={zone: f"{zone}_wind" for zone in df_wind_100m.columns})
    
    # --- Merge on daily index ---
    merged_df = solar_daily_labels.merge(df_wind_100m, left_index=True, right_index=True, how="left")

    # --- Filter for winter months October to March ---
    merged_df = merged_df[merged_df.index.month.isin([10, 11, 12, 1, 2, 3])]

    merged_daily_dict[period_code] = merged_df


# In[889]:


#Function for compound events 
def compound_events(df, wind_thresh=4, min_spell_length=5):
    zones = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5']

    df = df.copy()
    df.index = pd.to_datetime(df.index).normalize()

    low_wind_spell_count = {}
    dark_spell_count = {}
    compound_event_count = {}
    prob_ce = {}
    compound_event_record = []

    # Initialize global CE binary column
    df['ce_binary'] = 0

    for zone in zones:
        solar_col = f"{zone}_solar"
        wind_col = f"{zone}_wind"

        # --- Dark spells ---
        #Identify dark spells as >5 days marked 'D'
        dark_flag = (df[solar_col] == 'D')
        dark_spell_flags = pd.Series(False, index=df.index)

        dark_runs = (dark_flag != dark_flag.shift()).cumsum()
        dark_groups = dark_flag.groupby(dark_runs)

        dark_spell_count_zone = 0
        for _, group_vals in dark_groups:
            if group_vals.iloc[0] and len(group_vals) >= min_spell_length:
                dark_spell_count_zone += 1
                dark_spell_flags.loc[group_vals.index] = True

        # --- Low wind spells ---
        #Identify low wind spells as >5 days sfcWind100m<4m/s
        low_wind_flag = (df[wind_col] < wind_thresh)
        low_wind_spell_flags = pd.Series(False, index=df.index)

        wind_runs = (low_wind_flag != low_wind_flag.shift()).cumsum()
        wind_groups = low_wind_flag.groupby(wind_runs)

        low_wind_spell_count_zone = 0
        for _, group_vals in wind_groups:
            if group_vals.iloc[0] and len(group_vals) >= min_spell_length:
                low_wind_spell_count_zone += 1
                low_wind_spell_flags.loc[group_vals.index] = True

        # --- Overlapping compound events ---
        #One compound event = >1 days overlap of dark spell and low wind spell 
        overlap = dark_spell_flags & low_wind_spell_flags
        overlap_dates = df.index[overlap]

        # Per-zone CE binary column
        #Binary column iniatited where if CE=1 if no CE=0
        df[f'{zone}_ce_binary'] = overlap.astype(int)
        # Update global CE binary column
        df.loc[overlap, 'ce_binary'] = 1

        for dt in overlap_dates:
            try:
                idx = df.index.get_loc(dt)
                compound_event_record.append({'zone': zone, 'date': dt, 'original_index': idx})
            except KeyError:
                pass

        low_wind_spell_count[zone] = low_wind_spell_count_zone
        dark_spell_count[zone] = dark_spell_count_zone
        compound_event_count[zone] = len(overlap_dates)
        prob_ce[zone] = round((len(overlap_dates) / len(df)) * 100, 3) if len(df) > 0 else 0

    # Keep only the columns we want: solar, wind, per-zone binary, and global CE binary
    solar_cols = [f"{zone}_solar" for zone in zones]
    wind_cols = [f"{zone}_wind" for zone in zones]
    ce_binary_cols = [f"{zone}_ce_binary" for zone in zones]

    df = df[solar_cols + wind_cols + ce_binary_cols]

    print('Low wind spells:', low_wind_spell_count)
    print('Dark spells:', dark_spell_count)
    print('Compound events (CEs):', compound_event_count)
    print('Probability of a CE (%):', prob_ce)

    return (
        low_wind_spell_count,
        dark_spell_count,
        prob_ce,
        compound_event_count,
        df
    )


# In[890]:


def mean_dark_spell_length(series):
    is_dark = series == "D"

    # Identify boundaries of consecutive runs
    groups = (is_dark != is_dark.shift()).cumsum()

    # Count length of each run
    run_lengths = is_dark.groupby(groups).sum()

    # Keep only dark runss
    dark_lengths = run_lengths[run_lengths > 0]

    if len(dark_lengths) == 0:
        return 0  # no dark periods
    else:
        return round(dark_lengths.mean(),0)


# In[891]:


# In 30 years of winter = 5400 
# Function to count the number of dark days in a region and at a given period 
def number_dark_days(period, region):
    num_dark_region_period = (merged_daily_dict[f'{period}'][f'{region}_solar'] == "D").sum()
    print(f"Number of dark days in {region}:", num_dark_region_period)


# In[892]:


def average_spell_length(period, region):
    mean_len_region = mean_dark_spell_length(merged_daily_dict[f"{period}"][f"{region}_solar"])
    print(f"Mean dark spell length in {region}:", mean_len_region, 'days')


# In[893]:


number_dark_days('historical', 'NO1')
number_dark_days('historical', 'NO2')
number_dark_days('historical', 'NO3')
number_dark_days('historical', 'NO4')
number_dark_days('historical', 'NO5')


# In[894]:


#Results for number of hazards and compound events for all periods
low_wind_count_hist, dark_days_count_hist, prob_hist, ce_count_hist, df_hist = compound_events(merged_daily_dict['historical'])


# In[895]:


low_wind_count_mid, dark_days_count_mid, prob_mid, ce_count_mid, df_mid = compound_events(merged_daily_dict['mid-century'])


# In[896]:


low_wind_count_end, dark_days_count_end, prob_end, ce_count_end, df_end = compound_events(merged_daily_dict['end-century'])


# In[ ]:


#Function to bootstrap the 'CE_binary' columns. Sample w/replacement, 10000 times. Confidence interval estimation
def hypotheses_test_binary(df, n_bootstrap=1000):
    zones = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5']
    ce_cols = [f"{zone}_ce_binary" for zone in zones]

    results = []

    for i in range(n_bootstrap):
        # Resample the CE binary column for each zone independently with replacement
        iteration_counts = {}
        for zone, col in zip(zones, ce_cols):
            resampled = df[col].sample(n=len(df), replace=True).reset_index(drop=True)
            iteration_counts[zone] = resampled.sum()  # total number of CEs in this resample

        results.append(iteration_counts)

    bootstrapped_df = pd.DataFrame(results)
    return bootstrapped_df


# In[ ]:


boostrapped_hist = hypotheses_test_binary(df_hist, 1000)


# In[ ]:


boostrapped_mid = hypotheses_test_binary(df_mid, 1000)


# In[ ]:


boostrapped_end = hypotheses_test_binary(df_end, 1000)


# In[ ]:


bootstrapped_results = {
    'hist': boostrapped_hist, 
    'mid': boostrapped_mid, 
    'end': boostrapped_end
}


# In[ ]:


# Set Seaborn style
sns.set(style="whitegrid", context="talk")

# Define colors for each period
period_colors = {
    'hist': "#4c72b0",
    'mid': "#55a868",
    'end': "#c44e52"
}

# Define threshold values for each period (example values, replace with yours)
thresholds = {
    'hist': ce_count_hist,
    'mid': ce_count_mid,
    'end': ce_count_end
}

period_dates = {
    'hist': '1971-2000', 
    'mid': '2036-2065', 
    'end': '2071-2100'
}
    

regions = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5']

#For each region, creation of a plot which is then stored in your output path
for region in regions:
    plt.figure(figsize=(10, 6), dpi=300)  # high resolution

    for period in period_dates:
        df_period = bootstrapped_results[period]  # DataFrame from hypotheses_test_binary
        counts = df_period[region]  # Series of counts for this region

        # Histogram
        plt.hist(counts, bins=15, alpha=0.5, color=period_colors[period],
                 label=period_dates[period])#, edgecolor='black')

        # Threshold line
        plt.axvline(thresholds[period][region], color=period_colors[period], linestyle='--', linewidth=2)

    plt.title(f'Bootstrap distribution of the number of dark doldrum days - Region {region}')
    #plt.suptitle(f'{scen}')#, fontsize = 20, x = 0.5, y = 0.995) 
    plt.xlabel('Number of dark doldrums ')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.7)

    # Save high-resolution figure
    plt.tight_layout()
    plt.savefig(f'{output_path}/{region}_CE_Bootstrap_{scen}.png', dpi=300)
    plt.show()
   


# In[ ]:





# In[ ]:




