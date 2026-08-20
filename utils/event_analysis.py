from typing import Literal, Optional
import pandas as pd
import numpy as np
import xarray as xr
from statsmodels.tsa.seasonal import STL


PERIOD=47
SEASONAL=155 
TREND=49
INNER_ITER=1
OUTER_ITER=17
ROBUST= True

def apply_cosine_weighting(da):
    """
    Takes a dataarray and applies cosine weighting to it.

    Parameters: 
    - da (xr.DataArray): The DataArray to be weighted.

    Returns:    
    - da_weighted (xr.DataArray): The weighted DataArray.
    """
    
    weights = np.cos(da.lat*(np.pi/180))
    da_weighted = da * weights
    return da_weighted

def aggregate_da_to_timeseries(
    da, 
    vegetation_array,
    event_mask, 
    method: Literal['mean', 'median', 'min', 'max'] = 'mean'
):
    """
    Aggregates the input data array by computing one time series per vegetation class for a given extreme event. Method of aggregation can be chosen.
    IMPORTANT: for method 'mean', an area-weighted data array has to be used as input.

    Parameters:
        da (xarray.DataArray): The data array for the event (time, lat, lon), output from create_event_array().
        vegetation_array (xarray.DataArray): Vegetation class mask (lat, lon) or (time, lat, lon).
        event_mask (xarray.DataArray): Event mask (lat, lon) that is 1 for the event and 0 otherwise.
        method (str): The method to use for aggregation ('mean' or 'median').

    Returns:
        pandas.DataFrame: A dataframe with time (rows) and vegetation classes (columns).
    """

    #### Prepare vegetation mask ####
    # Interpolate vegetation to same grid as data array
    vegetation_interp = vegetation_array.interp_like(da.isel(time=0), method="nearest")

    # Interpolate event mask to same grid
    event_mask_interp = event_mask.interp_like(da.isel(time=0), method="nearest")

    # Apply the event mask
    vegetation_masked = vegetation_interp.where(event_mask_interp)

    # Drop time dimension if present in vegetation_masked
    if 'time' in vegetation_masked.dims:
        vegetation_masked = vegetation_masked.isel(time=0)

    #### Create dataframe from data ####
    # Stack spatial dimensions into one to enable consistent indexing
    veg_stacked = vegetation_masked.stack(space=('lat', 'lon'))           
    da_stacked = da.stack(space=('lat', 'lon'))               

    # Create valid data mask (no NaNs in veg or all-time NaNs in data array)
    valid_mask = (~np.isnan(veg_stacked)) & (~np.isnan(da_stacked).all(dim='time'))

    # Compute, if dask array
    if hasattr(valid_mask, 'compute'):
        valid_mask = valid_mask.compute()

    # Apply valid mask
    veg_valid = veg_stacked.where(valid_mask, drop=True)               
    da_valid = da_stacked.isel(space=valid_mask.values)           

    # Convert to DataFrame
    df = pd.DataFrame(
        da_valid.transpose('space', 'time').values,                   
        columns=da_valid['time'].values
    ) # rows = pixels, cols = time
    df['veg_class'] = veg_valid.values                     

    #### TS aggregation per land cover type ####
    if method == 'mean':
        df_grouped = df.groupby('veg_class').mean().T
    elif method == 'median':
        df_grouped = df.groupby('veg_class').median().T 
    elif method == 'min':
        df_grouped = df.groupby('veg_class').min().T 
    elif method == 'max':   
        df_grouped = df.groupby('veg_class').max().T 
    else:
        raise ValueError("Invalid method. Choose 'mean', 'median', 'min', or 'max'.")

    #### Adapt dataframe to purposes ####
    # Convert time index to datetime
    df_grouped.index = pd.to_datetime(df_grouped.index)

    # Remove duplicate rows
    df_grouped = df_grouped[~df_grouped.index.duplicated(keep='first')]

    return df_grouped


def ts_decomposition(df, optimized_params: bool = False):
    """
    Decomposes the aggregated kNDVI time series for each vegetation class into seasonal, trend and residual components.
    Returns the deseasonalized time series or the residuals.

    Parameters:
       df (pandas.DataFrame): DataFrame with aggregated kNDVI time series for each vegetation class.
       method (str): The method to use for decomposition ('deseasonalized' or 'residuals').
    Returns:
        df_decomposed (pandas.DataFrame): DataFrame with deseasonalized time series or residuals for each vegetation class.
    """
    # Create empty dataframe to store deseasonalized time series
    df_residuals = pd.DataFrame()

    for column in df.columns:
        # Retrieve the time series for the current vegetation class
        time_series = df[column]

        if optimized_params:
            decomp = STL(time_series, period=PERIOD, seasonal=SEASONAL, trend=TREND,robust=ROBUST).fit(inner_iter=INNER_ITER, outer_iter=OUTER_ITER) # Parameters determined by optimization
        else:
            decomp = STL(time_series, period=PERIOD, robust=True).fit()

        # Append result to dataframe
        df_residuals[column] = decomp.resid

    return df_residuals


def calculate_event_stats(df_time_series, start_time_value, end_time_value):    
    
    """
    Calculates event statistics (resilience, resistance, recovery) from a dataframe of time series and returns them in a dataframe. The resulting
    statistics shall be derived from the response variable (kNDVI in my case).
    Parameters:
    - df_time_series (pd.DataFrame): Dataframe containing time series data.
    - start_time_value (pd.Timestamp): Start time of the event.
    - end_time_value (pd.Timestamp): End time of the event.
    Returns:
    - pd.DataFrame: Dataframe containing event statistics.
    """
    # Calculate time buffer to caputre lagged responses
    event_duration = end_time_value - start_time_value
    buffer_duration = event_duration * 0.5 # 50% buffer
    extended_end_time = end_time_value + buffer_duration

    # Decompose time series
    df_residuals = ts_decomposition(df_time_series, optimized_params=True)

    df_stats = pd.DataFrame.from_dict({ # Put directly in dataframe
        'vpre': df_time_series.loc[
            df_time_series.index < start_time_value
        ].mean(), # Vpre

        'vdist': df_time_series.loc[
            (df_time_series.index >= start_time_value) & 
            (df_time_series.index <= extended_end_time)
        ].min(), # Vdist (simple version)

        'vpost': df_time_series.loc[
            df_time_series.index > end_time_value
        ].mean(), # Vpost

    }, orient='index')

    # Calculate resilience, resistance and recovery
    df_stats.loc['resilience'] = df_stats.loc['vpost'] / df_stats.loc['vpre']
    df_stats.loc['resistance'] = df_stats.loc['vdist'] / df_stats.loc['vpre']
    df_stats.loc['recovery_dep'] = df_stats.loc['vpost'] - df_stats.loc['vdist'] / df_stats.loc['vpre'] - df_stats.loc['vdist']
    df_stats.loc['recovery'] = df_stats.loc['vpost'] / df_stats.loc['vdist']

    # Calculate duration until maximum impact
    date_vdist = df_time_series[start_time_value:extended_end_time].idxmin() # Date of maximum impact
    df_stats.loc['n_days_vdist'] = (date_vdist - start_time_value) / np.timedelta64(1, 'D') # Number of days until maximum impact

    # Calculate anomaly strength
    df_stats.loc['kndvi_anomaly_strength'] = df_residuals[start_time_value:extended_end_time].sum(axis=0)
    df_stats.loc['kndvi_anomaly_strength_average'] = df_residuals[start_time_value:extended_end_time].sum(axis=0) / len(df_residuals[start_time_value:extended_end_time]) # Divided by number of data points

    return df_stats


def calculate_event_stats_on_residuals(df_time_series, start_time_value, end_time_value):    
    
    """
    Calculates event statistics (resilience, resistance, recovery) from a dataframe of time series and returns them in a dataframe. The resulting
    statistics shall be derived from the response variable (kNDVI in my case).
    Parameters:
    - df_time_series (pd.DataFrame): Dataframe containing time series data.
    - start_time_value (pd.Timestamp): Start time of the event.
    - end_time_value (pd.Timestamp): End time of the event.
    Returns:
    - pd.DataFrame: Dataframe containing event statistics.
    """
    # Calculate time buffer to capture lagged responses
    event_duration = end_time_value - start_time_value
    buffer_duration = event_duration * 0.5 # 50% buffer
    extended_end_time = end_time_value + buffer_duration

    # Decompose time series
    df_residuals = ts_decomposition(df_time_series, optimized_params=True)

    df_stats = pd.DataFrame.from_dict({ # Put directly in dataframe
        'vpre': df_residuals.loc[
            df_residuals.index < start_time_value
        ].mean(), # Vpre

        'vdist': df_residuals.loc[
            (df_residuals.index >= start_time_value) & 
            (df_residuals.index <= extended_end_time)
        ].min(), # Vdist (simple version)

        'vpost': df_residuals.loc[
            df_residuals.index > end_time_value
        ].mean(), # Vpost

    }, orient='index')

    # Calculate resilience, resistance and recovery
    df_stats.loc['resilience'] = df_stats.loc['vpost'] / df_stats.loc['vpre']
    df_stats.loc['resistance'] = df_stats.loc['vdist'] / df_stats.loc['vpre']
    df_stats.loc['recovery_dep'] = df_stats.loc['vpost'] - df_stats.loc['vdist'] / df_stats.loc['vpre'] - df_stats.loc['vdist']
    df_stats.loc['recovery'] = df_stats.loc['vpost'] / df_stats.loc['vdist']

    # Calculate duration until maximum impact
    date_vdist = df_time_series[start_time_value:extended_end_time].idxmin() # Date of maximum impact
    df_stats.loc['n_days_vdist'] = (date_vdist - start_time_value) / np.timedelta64(1, 'D') # Number of days until maximum impact

    # Calculate anomaly strength
    df_stats.loc['kndvi_anomaly_strength'] = df_residuals[start_time_value:extended_end_time].sum(axis=0)
    df_stats.loc['kndvi_anomaly_strength_average'] = df_residuals[start_time_value:extended_end_time].sum(axis=0) / len(df_residuals[start_time_value:extended_end_time]) # Divided by number of data points

    return df_stats


def calculate_n_days_affected(start_date, end_date, event_mask, event_label, ds_label):

    # Wrap label dataset to coordinates of event mask
    lon_orig = ds_label.longitude.values
    lon_orig_wrapped = np.where(lon_orig > 180, lon_orig - 360, lon_orig)
    sorted_indices = np.argsort(lon_orig_wrapped)
    lon_orig_wrapped_sorted = lon_orig_wrapped[sorted_indices]
    lat_orig = ds_label.latitude.values

    # Slice the dataset for the whole period
    ds_period = ds_label.sel(Ti=slice(start_date, end_date))

    # Compute boolean mask of all days in one vectorized operation
    labels = ds_period['labels'].values  # shape: (time, lat, lon)
    event_mask_all = (labels == event_label).astype(np.int8) 

    # Sum over time in numpy instead of looping
    event_count_array = event_mask_all.sum(axis=0)

    # Sort longitudes once
    event_count_array_sorted = event_count_array[:, sorted_indices]

    # Create new data array
    event_count_da = xr.DataArray(
        event_count_array_sorted,
        dims=["latitude", "longitude"],
        coords={"latitude": lat_orig, "longitude": lon_orig_wrapped_sorted}
    )

    # Resample to event mask grid
    event_count_da = event_count_da.interp(
        latitude=event_mask['lat'], longitude=event_mask['lon'], method='nearest'
    )

    # Apply event mask
    event_count_da = event_count_da.where(event_mask == 1)

    return event_count_da

def calculate_predictor_anomaly(df_ts, start_date, end_date, averaged=False):

    """
    Calculates the anomaly for the predictor (e.g. precipitation) for a given event.

    Parameters:
    - df_ts (pandas.DataFrame): A dataframe containing the residual component of a time series of the predictor.
    - start_date: Date of event beginning.
    - end_date: Date of event end.
    - averaged (bool): If True, the anomaly is the average deviation during event period + buffer.

    Returns:
    - anomalies (pandas.Series): A series containing the anomaly for the predictor per vegetation class.
    """

    # Initialize  dates
    event_duration = end_date - start_date
    buffer_duration = event_duration * 0.5
    extended_start_time = start_date - buffer_duration

    df_resid = ts_decomposition(df_ts)

    if averaged:
        anomalies = df_resid[extended_start_time:end_date].sum(axis=0) / len(df_ts[extended_start_time:end_date]) # Divided by number of data points
    else:    
        anomalies = df_resid[extended_start_time:end_date].sum(axis=0) 
    
    # Ensure right index name
    anomalies.index.name = "veg_class"

    return anomalies

def aggregate_by_vegetation(vegetation, event_mask, data_array, method: Literal['sum', 'mean'], subset_to_event_dates: bool = False, start_time: Optional[pd.Timestamp] = None, end_time: Optional[pd.Timestamp] = None):

    """
    Aggregate values in a data array by vegetation class, with event-based masking.

    This function computes the mean or sum of a data array variable grouped by vegetation classes.
    It supports both 2D data arrays (('lat', 'lon')) and 3D time series (('time', 'lat', 'lon')).
    Aggregation is limited to spatial locations where the event_mask is 'True'

    Parameters:
    - vegetation (xarray.DataArray): A 2D or 3D array containing categorical vegetation class labels. If 3D, only the first time slice is used.
    - event_mask (xarray.DataArray): A mask of valid areas (same spatial dimensions as data_array), where True indicates valid pixels.
    - data_array (xarray.DataArray): The data to be aggregated. Must have shape either (time, lat, lon) or (lat, lon).xarray.DataArray
    - method {'sum', 'mean'}: Aggregation method to use across time (if applicable) and across pixels within each vegetation class.
    - subset_to_event_dates (bool): If True, subset data_array to only include dates between start_time and end_time.
    - start_time (pd.Timestamp, optional): Start of event period for subsetting. Required if subset_to_event_dates=True.
    - end_time (pd.Timestamp, optional): End of event period for subsetting. Required if subset_to_event_dates=True.

    Returns: 
    -  A pandas.Series indexed by vegetation class, containing the aggregated value (sum or mean) per class.


    Notes
    -----
    - If data_array is 3D, the aggregation is performed across time for each pixel before grouping by vegetation class.
    - If data_array is 2D, values are grouped directly by vegetation class.
    - The function automatically interpolates vegetation and event_mask to match the spatial grid of data_array.
    - Pixels with NaNs in either vegetation or data_array are excluded from the aggregation.

    """
    # Subset to event date if requested
    if subset_to_event_dates:
        if start_time is None or end_time is None:
            raise ValueError("start_time and end_time must be provided when subset_to_event_dates=True")
        if 'time' in data_array.dims:
            # Select event period only
            data_array = data_array.sel(time=slice(start_time, end_time))

            if data_array.time.size == 0:
                raise ValueError(f"No data found between {start_time} and {end_time}")
            
        else:
            print("data_array has no time dimensions")


    # Interpolate vegetation and event mask to same grid as data_array
    reference_grid = data_array if 'time' not in data_array.dims else data_array.isel(time=0)
    vegetation_interp = vegetation.interp_like(reference_grid, method="nearest")
    event_mask_interp = event_mask.interp_like(reference_grid, method="nearest")

    # Apply the event mask
    vegetation_masked = vegetation_interp.where(event_mask_interp)

    # Drop time dimension if present in vegetation_masked
    if 'time' in vegetation_masked.dims:
        vegetation_masked = vegetation_masked.isel(time=0)

    # Stack spatial dimensions
    veg_stacked = vegetation_masked.stack(space=('lat', 'lon'))
    da_stacked = data_array.stack(space=('lat', 'lon'))

    # Create valid data mask
    if 'time' in da_stacked.dims:
        valid_mask = (~np.isnan(veg_stacked)) & (~np.isnan(da_stacked).all(dim='time'))
    else:
        valid_mask = (~np.isnan(veg_stacked)) & (~np.isnan(da_stacked))

    # Compute valid mask if dask
    if hasattr(valid_mask, 'compute'):
        valid_mask = valid_mask.compute()

    veg_valid = veg_stacked.where(valid_mask, drop=True)
    data_valid = da_stacked.isel(space=valid_mask.values)

    if 'time' in data_valid.dims:
        df = pd.DataFrame(
            data_valid.transpose('space', 'time').values,
            columns=data_valid['time'].values
        )
    else:
        df = pd.DataFrame({'value': data_valid.values})

    # Add vegetation class column
    df['veg_class'] = veg_valid.values

    # Drop duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    # Aggregate
    if 'time' in data_valid.dims:
        if method == 'mean':
            df['aggregation'] = df.mean(axis=1)
            df = df.groupby('veg_class')['aggregation'].mean()
        elif method == 'sum':
            df['aggregation'] = df.sum(axis=1)
            df = df.groupby('veg_class')['aggregation'].sum()
        else:
            raise ValueError("Invalid method. Choose 'mean' or 'sum'.")
    else:
        if method == 'mean':
            df = df.groupby('veg_class')['value'].mean()
        elif method == 'sum':
            df = df.groupby('veg_class')['value'].sum()
        else:
            raise ValueError("Invalid method. Choose 'mean' or 'sum'.")
    
    # Ensure index name is veg_class
    df.index.name = "veg_class"

    return df
