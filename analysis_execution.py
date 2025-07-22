"""
analysis_execution.py

This script runs all computations on all events and summarises them in a combined dataframe. Unique indices for the dataframe 
are created by combining the event label and vegetation class label to a multi index. The final dataframe includes event-based 
statistics and climate predictors aggregated by land cover.

Author: Georg Brandt
Date: 2025-07-16
"""


#### Imports ####
# Standard libraries
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm

# Handling files
import requests
import zipfile
import tempfile
import io
import os
import glob

# Acces APIs
from xcube.core.store import new_data_store

# Plotting  
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Handling dates
import datetime
from datetime import timedelta

# Import own functions
from utils import event_sampling as es
from utils import event_visualization as ev
from utils import event_analysis as ea
from utils.event_visualization import plot_aggregated_ts

# Analysis
from statsmodels.tsa.seasonal import STL

PATH_LABELS = 'data/temp_zarr_extracted/mergedlabels_ranked_pot0.01_ne0.1_cmp_S1_T3_1950_2023.zarr'
PATH_LANDMASK = "data/GLDASp5_landmask_025d.nc4"
PATH_VEG = 'data/GLDASp5_domveg_NOAH3.6_025d.nc4'
PATH_TRANSLATION = "data/GLDA_veg_legend.csv"
THRESHOLD_DURATION_DAYS = 18

if __name__ == "__main__":

    # ======================================================
    # 1. LOAD DATA
    # ======================================================
    buffer = 2 * 365 # Days for the temporal buffer around each event



    ## kNDVI array
    # Import kNDVI dataset
    store = new_data_store("s3", root="deep-esdl-public", storage_options=dict(anon=True))
    ds_esdc = store.open_data('esdc-8d-0.25deg-1x720x1440-3.0.1.zarr')

    # Select kNDVI
    kndvi = ds_esdc['kndvi']



    ## Event summary dataframe
    # URL
    url_chd_summary = "https://zenodo.org/records/14884254/files/MergedEventStats_landonly_int.csv.zip?download=1"

    # Download the ZIP file
    response = requests.get(url_chd_summary)
    response.raise_for_status()

    # Unzip the content
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        csv_filename = z.namelist()[0]
        with z.open(csv_filename) as f:
            df_chd_summary = pd.read_csv(f)

    # Create new column 'duration_days' with numeric values
    df_chd_summary['duration_days'] = df_chd_summary['duration'].str.extract(r'(\d+)').astype(int)

    # Convert start_time and end_time to datetime
    df_chd_summary['start_time'] = pd.to_datetime(df_chd_summary['start_time'])
    df_chd_summary['end_time'] = pd.to_datetime(df_chd_summary['end_time'])

    # Filter dataframe for event duration
    df_chd_filtered = df_chd_summary[df_chd_summary['duration_days'] >= THRESHOLD_DURATION_DAYS]

    # Filter dataframe for period of ESDC kndvi data
    df_chd_filtered = df_chd_filtered[
        (df_chd_filtered['start_time'].dt.year >= 2002) &
        (df_chd_filtered['start_time'].dt.year <= 2019) &
        (df_chd_filtered['end_time'].dt.year >= 2002) &
        (df_chd_filtered['end_time'].dt.year <= 2019)
    ]



    ## Event cube dataset
    ds_chd_labels = xr.open_zarr(PATH_LABELS, consolidated=False)



    ## Land mask array
    # Open with xarray
    ds_landmask = xr.open_dataset(PATH_LANDMASK)

    # Open the right variable as land mask
    landmask = ds_landmask['GLDAS_mask']



    ## Land cover array and translation dataframe
    # Open with xarray
    ds_veg = xr.open_dataset(PATH_VEG)

    # Open the right variable
    vegetation = ds_veg['GLDAS_domveg']

    # Open class translation dataframe
    df_translation = pd.read_csv(PATH_TRANSLATION, index_col=0)


    # ======================================================
    # 2. ANALYSIS
    # ======================================================
    # Initialize empty list
    df_list = []

    for index, row in tqdm(df_chd_filtered.iterrows(), total=len(df_chd_filtered), desc="Processing events"):
        
        label = row['label']
        print(f"Processing event {label}...")

        # ======================================================
        # 2.1 PREPARE DATA FOR EVENT
        # ======================================================
        # Create event mask
        event_mask = es.create_event_mask(label, event_table=df_chd_filtered, labelcube=ds_chd_labels, land_mask=landmask)

        # Check if event_mask has values, skip event otherwise
        if event_mask.sum().item() == 0:
            print(f"Skipping event {label} — empty mask")
            continue

        # Create event array
        kndvi_array = es.create_event_dataarray(kndvi, event_mask, df_chd_filtered, label, time_buffer=buffer)

        # Check that the array has values and that the interpolation will not fail
        if kndvi_array.isel(time=0).size == 0 or np.isnan(kndvi_array.isel(time=0).values).all():
            print(f"Skipping event {label} — empty or invalid kNDVI array")
            continue

        # Start and end time of event
        start_time_value = df_chd_filtered.loc[df_chd_filtered['label'] == label, 'start_time'].values[0]
        end_time_value = df_chd_filtered.loc[df_chd_filtered['label'] == label, 'end_time'].values[0]

        # ======================================================
        # 2.2 AGGREGATE KNDVI TS PER LAND COVER TYPE
        # ======================================================
        df_mean_kndvi_ts = ea.aggregate_kndvi_timeseries(kndvi_array, vegetation, event_mask, method='mean')

        # ======================================================
        # 2.3 CALCULATE EVENT STATISTICS
        # ======================================================
        # Remove seasonality from time series
        df_mean_kndvi_ts_deseasonalized = ea.ts_decomposition(df_mean_kndvi_ts, method='deseasonalized')

        # Calculate event statistics for kNDVI
        df_stats_deseasonalized = ea.calculate_event_stats(df_mean_kndvi_ts_deseasonalized, start_time_value, end_time_value)

        # ======================================================
        # 2.4 PREPARE PREDICTORS
        # ======================================================
        # Load data arrays
        t2m = ds_esdc['air_temperature_2m']
        t2mmin = ds_esdc['min_air_temperature_2m']
        t2mmax = ds_esdc['max_air_temperature_2m']
        prec = ds_esdc['precipitation_era5']
        soil_moist = ds_esdc['root_moisture']
        evap = ds_esdc['evaporation']
        evap_pot = ds_esdc['potential_evaporation']

        # Create event arrays
        t2m_array = es.create_event_dataarray(d_array = t2m, event_mask = event_mask, event_label=label, event_table=df_chd_filtered, time_buffer = 0)
        t2mmin_array = es.create_event_dataarray(d_array = t2mmin, event_mask = event_mask, event_label=label, event_table=df_chd_filtered, time_buffer = 0)
        t2mmax_array = es.create_event_dataarray(d_array = t2mmax, event_mask = event_mask, event_label=label, event_table=df_chd_filtered, time_buffer = 0)
        prec_array = es.create_event_dataarray(d_array = prec, event_mask = event_mask, event_label=label, event_table=df_chd_filtered, time_buffer = 0)
        soil_moist_array = es.create_event_dataarray(d_array = soil_moist, event_mask = event_mask, event_label=label, event_table=df_chd_filtered, time_buffer = 0)
        evap_array = es.create_event_dataarray(d_array = evap, event_mask = event_mask, event_label=label, event_table=df_chd_filtered, time_buffer = 0)
        evap_pot_array = es.create_event_dataarray(d_array = evap_pot, event_mask = event_mask, event_label=label, event_table=df_chd_filtered, time_buffer = 0)
        event_count = ea.calculate_n_days_affected(start_time_value, end_time_value, event_mask, event_label=label, ds_label=ds_chd_labels) # Already tailored to specific event

        # ======================================================
        # 2.5 PUT ALL DATA IN ONE DATAFRAME
        # ======================================================
        event_count_agg = ea.aggregate_by_vegetation(vegetation=vegetation, 
                                                                        event_mask=event_mask, 
                                                                        data_array=event_count, 
                                                                        method='mean')
        t2m_agg = ea.aggregate_by_vegetation(vegetation=vegetation,
                                                        event_mask=event_mask,
                                                        data_array=t2m_array,
                                                        method='mean')
        t2mmin_agg = ea.aggregate_by_vegetation(vegetation=vegetation,
                                                        event_mask=event_mask,
                                                        data_array=t2mmin_array,
                                                        method='mean')
        t2mmax_agg = ea.aggregate_by_vegetation(vegetation=vegetation,
                                                        event_mask=event_mask,
                                                        data_array=t2mmax_array,
                                                        method='mean')
        prec_agg = ea.aggregate_by_vegetation(vegetation=vegetation,
                                                        event_mask=event_mask,
                                                        data_array=prec_array,
                                                        method='sum')
        soil_moist_agg = ea.aggregate_by_vegetation(vegetation=vegetation,
                                                        event_mask=event_mask,
                                                        data_array=soil_moist_array,
                                                        method='sum')
        evap_agg = ea.aggregate_by_vegetation(vegetation=vegetation,
                                                        event_mask=event_mask,
                                                        data_array=evap_array,
                                                        method='sum')
        evap_pot_agg = ea.aggregate_by_vegetation(vegetation=vegetation,
                                                        event_mask=event_mask,
                                                        data_array=evap_pot_array,
                                                        method='sum')

        # Put series together as dataframe
        df = pd.DataFrame({'n_days': event_count_agg,
                                't2m': t2m_agg,
                                't2mmin': t2mmin_agg,
                                't2mmax': t2mmax_agg,
                                'prec': prec_agg,
                                'soil_moist': soil_moist_agg,
                                'evap': evap_agg,
                                'evap_pot': evap_pot_agg})

        # Add event stats to dataframe
        df = df.join(df_stats_deseasonalized.T)

        # Add column with event label
        df['event_label'] = label

        # Use vegetation type and event label as multiindex
        df = df.set_index(['event_label', df.index])

        # Drop rows with NaN values
        df = df.dropna()

        # Append to list
        df_list.append(df)

    # Concatenate all dataframes
    df_all = pd.concat(df_list)

    # Save df to file
    #df_all.to_csv('data/results/analysis_execution.csv')