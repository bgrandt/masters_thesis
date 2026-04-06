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
THRESHOLD_DURATION_DAYS = 18 # May have to be adapted because not enough events are included
ESDC_SUBSET_PATH = 'data/esdc_subset.zarr' # Can be path to local directory or url to cloud storage
OUTPUT_PATH = 'data/output/26_03_11_on_residuals.csv'

if __name__ == "__main__":

    # ======================================================
    # 1. LOAD DATA
    # ======================================================
    buffer = 2 * 365 # Days for the temporal buffer around each event for the kNDVI cube AND predictor anomaly

    ## kNDVI array and climatic predictors
    # Import ESDL dataset
    ds_esdc = xr.open_zarr(ESDC_SUBSET_PATH)

    # Select kNDVI and predictorvariables
    kndvi = ds_esdc['kndvi']
    t2m = ds_esdc['air_temperature_2m']
    t2mmin = ds_esdc['min_air_temperature_2m']
    t2mmax = ds_esdc['max_air_temperature_2m']
    prec = ds_esdc['precipitation_era5']
    soil_moist = ds_esdc['root_moisture']
    evap = ds_esdc['evaporation']
    evap_pot = ds_esdc['potential_evaporation']

    # Cosine weighting
    kndvi_w = ea.apply_cosine_weighting(kndvi)
    t2m_w = ea.apply_cosine_weighting(t2m)
    t2mmin_w = ea.apply_cosine_weighting(t2mmin)
    t2mmax_w = ea.apply_cosine_weighting(t2mmax)
    prec_w = ea.apply_cosine_weighting(prec)
    soil_moist_w = ea.apply_cosine_weighting(soil_moist)
    evap_w = ea.apply_cosine_weighting(evap)
    evap_pot_w = ea.apply_cosine_weighting(evap_pot)

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

    # Create list of filtered event labels
    events = df_chd_filtered['label']
    evets = [int(e) for e in events]

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

    # Loop over events
    for index, row in tqdm(df_chd_filtered.iterrows(), total=len(df_chd_filtered), desc="Processing events"):
        
        label = row['label']
        print(f"Processing event {label}...")

        # ======================================================
        # 2.1 PREPARE DATA FOR EVENT
        # ======================================================
        # Start and end time of event
        start_time = df_chd_filtered.loc[df_chd_filtered['label'] == label, 'start_time'].values[0]
        end_time = df_chd_filtered.loc[df_chd_filtered['label'] == label, 'end_time'].values[0]

        # Dates from 365 days before event starts
        start_time_prev = start_time - pd.Timedelta(days=365)
        end_time_prev = start_time - pd.Timedelta(days=1)

        # Create event mask
        event_mask = es.create_event_mask(label, start_date=start_time, end_date=end_time, labelcube=ds_chd_labels, land_mask=landmask)

        # Check if event_mask has values, skip event otherwise
        if event_mask.sum().item() == 0:
            print(f"Skipping event {label} — empty mask")
            continue

        # ======================================================
        # 2.2 PREPARE RESPONSE VARIABLE KNDVI
        # ======================================================

        # Create event array
        kndvi_array = es.create_event_dataarray(kndvi_w, event_mask, start_date=start_time, end_date=end_time, time_buffer=buffer)

        # Check that the array has values and that the interpolation will not fail
        if kndvi_array.isel(time=0).size == 0 or np.isnan(kndvi_array.isel(time=0).values).all():
            print(f"Skipping event {label} — empty or invalid kNDVI array")
            continue

        # ======================================================
        # 2.3 AGGREGATE KNDVI TS PER LAND COVER TYPE
        # ======================================================
        df_mean_kndvi_ts = ea.aggregate_da_to_timeseries(kndvi_array, vegetation, event_mask, method='mean')

        # ======================================================
        # 2.4 CALCULATE EVENT STATISTICS (KNDVI RESPONSE METRICS)
        # ======================================================
        # Calculate event statistics based on kNDVI (target variables)
        #df_kndvi_stats = ea.calculate_event_stats(df_mean_kndvi_ts, start_time, end_time)
        df_kndvi_stats = ea.calculate_event_stats_on_residuals(df_mean_kndvi_ts, start_time, end_time)

        # ======================================================
        # 2.5 PREPARE PREDICTORS
        # ======================================================

        # Create arrays for anomaly computation AND aggregation
        t2m_w_array = es.create_event_dataarray(d_array = t2m_w, event_mask = event_mask, start_date=start_time, end_date=end_time, time_buffer=buffer) 
        t2mmin_w_array = es.create_event_dataarray(d_array = t2mmin_w, event_mask = event_mask, start_date=start_time, end_date=end_time, time_buffer=buffer)
        t2mmax_w_array = es.create_event_dataarray(d_array = t2mmax_w, event_mask = event_mask, start_date=start_time, end_date=end_time, time_buffer=buffer)
        prec_w_array = es.create_event_dataarray(d_array = prec_w, event_mask = event_mask, start_date=start_time, end_date=end_time, time_buffer=buffer)
        soil_moist_w_array = es.create_event_dataarray(d_array = soil_moist_w, event_mask = event_mask, start_date=start_time, end_date=end_time, time_buffer=buffer)
        evap_w_array = es.create_event_dataarray(d_array = evap_w, event_mask = event_mask, start_date=start_time, end_date=end_time, time_buffer=buffer)
        evap_pot_w_array = es.create_event_dataarray(d_array = evap_pot_w, event_mask = event_mask, start_date=start_time, end_date=end_time, time_buffer=buffer)

        # Create climate event arrays for aggregation of previous year's climate
        t2m_array_prev = es.create_event_dataarray(d_array = t2m_w, event_mask = event_mask,  start_date=start_time_prev, end_date=end_time_prev)
        t2mmin_array_prev = es.create_event_dataarray(d_array = t2mmin_w, event_mask = event_mask, start_date=start_time_prev, end_date=end_time_prev)
        t2mmax_array_prev = es.create_event_dataarray(d_array = t2mmax_w, event_mask = event_mask, start_date=start_time_prev, end_date=end_time_prev)
        prec_array_prev = es.create_event_dataarray(d_array = prec_w, event_mask = event_mask,start_date=start_time_prev, end_date=end_time_prev)
        soil_moist_array_prev = es.create_event_dataarray(d_array = soil_moist_w, event_mask = event_mask, start_date=start_time_prev, end_date=end_time_prev)
        evap_array_prev = es.create_event_dataarray(d_array = evap_w, event_mask = event_mask, start_date=start_time_prev, end_date=end_time_prev)
        evap_pot_array_prev = es.create_event_dataarray(d_array = evap_pot_w, event_mask = event_mask, start_date=start_time_prev, end_date=end_time_prev)

        # Number of days affected
        event_count = ea.calculate_n_days_affected(start_time, end_time, event_mask, event_label=label, ds_label=ds_chd_labels) # Already tailored to specific event


        ### Predictor anomalies
        df_mean_t2m_ts = ea.aggregate_da_to_timeseries(t2m_w_array, vegetation, event_mask, method='mean')
        df_mean_t2mmin_ts = ea.aggregate_da_to_timeseries(t2mmin_w_array, vegetation, event_mask, method='mean')
        df_mean_t2mmax_ts = ea.aggregate_da_to_timeseries(t2mmax_w_array, vegetation, event_mask, method='mean')
        df_mean_prec_ts = ea.aggregate_da_to_timeseries(prec_w_array, vegetation, event_mask, method='mean')
        df_mean_soil_moist_ts = ea.aggregate_da_to_timeseries(soil_moist_w_array, vegetation, event_mask, method='mean')
        df_mean_evap_ts = ea.aggregate_da_to_timeseries(evap_w_array, vegetation, event_mask, method='mean')
        df_mean_evap_pot_ts = ea.aggregate_da_to_timeseries(evap_pot_w_array, vegetation, event_mask, method='mean')

        # Compute (sum up residuals over event period + buffer)
        t2m_anomalies = ea.calculate_predictor_anomaly(df_mean_t2m_ts, start_time, end_time)
        t2mmin_anomalies = ea.calculate_predictor_anomaly(df_mean_t2mmin_ts, start_time, end_time)
        t2mmax_anomalies = ea.calculate_predictor_anomaly(df_mean_t2mmax_ts, start_time, end_time)
        prec_anomalies = ea.calculate_predictor_anomaly(df_mean_prec_ts, start_time, end_time)
        soil_moist_anomalies = ea.calculate_predictor_anomaly(df_mean_soil_moist_ts, start_time, end_time)
        evap_anomalies = ea.calculate_predictor_anomaly(df_mean_evap_ts, start_time, end_time)
        evap_pot_anomalies = ea.calculate_predictor_anomaly(df_mean_evap_pot_ts, start_time, end_time)

        # ======================================================
        # 2.6 ADD FLAG IF 2 YEARS AROUND THE EVENT WAS ANOTHER EVENT
        # ======================================================
        other_events = es.check_for_other_events(start_date=start_time,
                                                 end_date=end_time,
                                                 event_mask=event_mask,
                                                 labelcube=ds_chd_labels,
                                                 label=label,
                                                 events=events)

        # ======================================================
        # 2.7 PUT ALL DATA IN ONE DATAFRAME
        # ======================================================

        # Aggregations
        event_count_agg = ea.aggregate_by_vegetation(vegetation=vegetation, 
                                                                        event_mask=event_mask, 
                                                                        data_array=event_count, 
                                                                        method='mean')
        t2m_agg = ea.aggregate_by_vegetation(vegetation=vegetation,
                                                        event_mask=event_mask,
                                                        data_array=t2m_w_array,
                                                        method='mean',
                                                        subset_to_event_dates=True,
                                                        start_time=start_time,
                                                        end_time=end_time)
        t2mmin_agg = ea.aggregate_by_vegetation(vegetation=vegetation,
                                                        event_mask=event_mask,
                                                        data_array=t2mmin_w_array,
                                                        method='mean',
                                                        subset_to_event_dates=True,
                                                        start_time=start_time,
                                                        end_time=end_time)
        t2mmax_agg = ea.aggregate_by_vegetation(vegetation=vegetation,
                                                        event_mask=event_mask,
                                                        data_array=t2mmax_w_array,
                                                        method='mean',
                                                        subset_to_event_dates=True,
                                                        start_time=start_time,
                                                        end_time=end_time)
        prec_agg = ea.aggregate_by_vegetation(vegetation=vegetation,
                                                        event_mask=event_mask,
                                                        data_array=prec_w_array,
                                                        method='mean',
                                                        subset_to_event_dates=True,
                                                        start_time=start_time,
                                                        end_time=end_time)
        soil_moist_agg = ea.aggregate_by_vegetation(vegetation=vegetation,
                                                        event_mask=event_mask,
                                                        data_array=soil_moist_w_array,
                                                        method='mean',
                                                        subset_to_event_dates=True,
                                                        start_time=start_time,
                                                        end_time=end_time)
        evap_agg = ea.aggregate_by_vegetation(vegetation=vegetation,
                                                        event_mask=event_mask,
                                                        data_array=evap_w_array,
                                                        method='mean',
                                                        subset_to_event_dates=True,
                                                        start_time=start_time,
                                                        end_time=end_time)
        evap_pot_agg = ea.aggregate_by_vegetation(vegetation=vegetation,
                                                        event_mask=event_mask,
                                                        data_array=evap_pot_w_array,
                                                        method='mean',
                                                        subset_to_event_dates=True,
                                                        start_time=start_time,
                                                        end_time=end_time)
        
        # Aggregations from previous year's conditions
        t2m_agg_prev = ea.aggregate_by_vegetation(vegetation=vegetation,
                                                        event_mask=event_mask,
                                                        data_array=t2m_array_prev,
                                                        method='mean')
        
        t2mmin_agg_prev = ea.aggregate_by_vegetation(vegetation=vegetation,
                                                        event_mask=event_mask,
                                                        data_array=t2mmin_array_prev,
                                                        method='mean')
        
        t2mmax_agg_prev = ea.aggregate_by_vegetation(vegetation=vegetation,
                                                        event_mask=event_mask,
                                                        data_array=t2mmax_array_prev,
                                                        method='mean')
        
        prec_agg_prev = ea.aggregate_by_vegetation(vegetation=vegetation,
                                                        event_mask=event_mask,
                                                        data_array=prec_array_prev,
                                                        method='mean')
        
        soil_moist_agg_prev = ea.aggregate_by_vegetation(vegetation=vegetation,
                                                        event_mask=event_mask,
                                                        data_array=soil_moist_array_prev,
                                                        method='mean')
        
        evap_agg_prev = ea.aggregate_by_vegetation(vegetation=vegetation,
                                                        event_mask=event_mask,
                                                        data_array=evap_array_prev,
                                                        method='mean')
        
        evap_pot_agg_prev = ea.aggregate_by_vegetation(vegetation=vegetation,
                                                        event_mask=event_mask,
                                                        data_array=evap_pot_array_prev,
                                                        method='mean')
        
        # Check if any of the aggregations is empty
        all_aggs = [
            event_count_agg, t2m_agg, t2mmin_agg, t2mmax_agg, prec_agg, soil_moist_agg, evap_agg, evap_pot_agg,
            t2m_agg_prev, t2mmin_agg_prev, t2mmax_agg_prev, prec_agg_prev, soil_moist_agg_prev, evap_agg
        ]

        if any(agg.empty for agg in all_aggs):
            print(f"Skipping event {label} — one or more aggregations empty")
            continue

        # Put series together as dataframe
        df = pd.DataFrame({'n_days': event_count_agg,
                                't2m': t2m_agg,
                                't2mmin': t2mmin_agg,
                                't2mmax': t2mmax_agg,
                                'prec': prec_agg,
                                'soil_moist': soil_moist_agg,
                                'evap': evap_agg,
                                'evap_pot': evap_pot_agg,
                                't2m_prev': t2m_agg_prev,
                                't2mmin_prev': t2mmin_agg_prev,
                                't2mmax_prev': t2mmax_agg_prev,
                                'prec_prev': prec_agg_prev,
                                'soil_moist_prev': soil_moist_agg_prev,
                                'evap_prev': evap_agg_prev,
                                'evap_pot_prev': evap_pot_agg_prev,
                                't2m_anomaly': t2m_anomalies,
                                't2min_anomaly': t2mmin_anomalies,
                                't2mmax_anomaly': t2mmax_anomalies,
                                'prec_anomaly': prec_anomalies,
                                'soil_moist_anomaly': soil_moist_anomalies,
                                'evap_anomaly': evap_anomalies,
                                'evap_pot_anomaly': evap_pot_anomalies,
                                'other_events': other_events
                                })

        # Add event stats to dataframe
        df = df.join(df_kndvi_stats.T)

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
    df_all.to_csv(OUTPUT_PATH)