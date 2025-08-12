from typing import Literal
import pandas as pd
import numpy as np
import xarray as xr
from statsmodels.tsa.seasonal import STL


PERIOD=47
SEASONAL=155 
TREND=49
INNER_ITER=1
OUTER_ITER=17
ROBUST=True

def aggregate_kndvi_timeseries(
    kndvi_array, 
    vegetation_array,
    event_mask, 
    method: Literal['mean', 'median', 'min', 'max'] = 'mean',
    area_weighting=True
):
    """
    Aggregates kNDVI time series per vegetation class for a given extreme event. Method of aggregation can be chosen.

    Parameters:
        kndvi_array (xarray.DataArray): The kNDVI data for the event (time, lat, lon), output from create_event_array().
        vegetation_array (xarray.DataArray): Vegetation class mask (lat, lon) or (time, lat, lon).
        event_mask (xarray.DataArray): Event mask (lat, lon) that is 1 for the event and 0 otherwise.
        method (str): The method to use for aggregation ('mean' or 'median').
        method (str): The method to use for decomposition ('deseasonalized', 'residuals', 'trend', 'seasonal').


    Returns:
        pandas.DataFrame: A dataframe with time (rows) and vegetation classes (columns).
    """

    #### Prepare vegetation mask ####
    # Interpolate vegetation to same grid as kNDVI array
    vegetation_interp = vegetation_array.interp_like(kndvi_array.isel(time=0), method="nearest")

    # Interpolate event mask to same grid (if needed)
    event_mask_interp = event_mask.interp_like(kndvi_array.isel(time=0), method="nearest")

    # Apply the event mask
    vegetation_masked = vegetation_interp.where(event_mask_interp)

    # Drop time dimension if present in vegetation_masked
    if 'time' in vegetation_masked.dims:
        vegetation_masked = vegetation_masked.isel(time=0)

    #### Create dataframe from data ####
    # Stack spatial dimensions into one to enable consistent indexing
    veg_stacked = vegetation_masked.stack(space=('lat', 'lon'))           
    kndvi_stacked = kndvi_array.stack(space=('lat', 'lon'))               

    # Create valid data mask (no NaNs in veg or all-time NaNs in kNDVI)
    valid_mask = (~np.isnan(veg_stacked)) & (~np.isnan(kndvi_stacked).all(dim='time'))

    # Compute, if dask array
    if hasattr(valid_mask, 'compute'):
        valid_mask = valid_mask.compute()

    # Apply valid mask
    veg_valid = veg_stacked.where(valid_mask, drop=True)               
    kndvi_valid = kndvi_stacked.isel(space=valid_mask.values)           

    # Convert to DataFrame
    kndvi_df = pd.DataFrame(
        kndvi_valid.transpose('space', 'time').values,                   
        columns=kndvi_valid['time'].values
    ) # rows = pixels, cols = time
    kndvi_df['veg_class'] = veg_valid.values      # Add vegetation class column                      

    #### TS aggregation per land cover type ####

    ### If area_weighted=True
    if area_weighting and method == 'mean':
        # Compute area per latitude
        latitudes = kndvi_array['lat'].values
        R = 6371000  # Earth radius in meters
        dlat = np.deg2rad(np.abs(np.diff(latitudes).mean()))
        dlon = np.deg2rad(np.abs(np.diff(kndvi_array['lon'].values).mean()))
        lat_rad = np.deg2rad(latitudes)

        # Area for each lat (broadcast to all lon)
        area_per_lat = (R**2 * dlat * dlon * np.cos(lat_rad)) # shape: (lat,)
        area_2d= np.repeat(area_per_lat[:, np.newaxis], len(kndvi_array['lon']), axis=1)

        # Stack and filter by valid pixels
        area_flat = xr.DataArray(area_2d, coords={'lat': kndvi_array['lat'], 'lon': kndvi_array['lon']})
        area_stacked = area_flat.stack(space=('lat', 'lon'))
        area_valid = area_stacked.isel(space=valid_mask.values)
        kndvi_df['pixel_area'] = area_valid.values

    # Group by vegetation class and transpose to have time as index
    if method == 'mean' and area_weighting:
        def weighted_mean(df):
            weights = df['pixel_area'].values
            data = df.drop(columns=['veg_class', 'pixel_area']).values
            return np.average(data, axis=0, weights=weights)

        grouped = kndvi_df.groupby('veg_class').apply(weighted_mean)
        df_grouped = pd.DataFrame(
            grouped.tolist(),
            index=grouped.index,
            columns=kndvi_valid['time'].values
        ).T  # transpose to have time as index
    elif method == 'mean':
        df_grouped = kndvi_df.groupby('veg_class').mean().T
    elif method == 'median':
        df_grouped = kndvi_df.groupby('veg_class').median().T 
    elif method == 'min':
        df_grouped = kndvi_df.groupby('veg_class').min().T 
    elif method == 'max':   
        df_grouped = kndvi_df.groupby('veg_class').max().T 
    else:
        raise ValueError("Invalid method. Choose 'mean', 'median', 'min', or 'max'.")

    #### Adapt dataframe to purposes ####
    # Convert time index to datetime
    df_grouped.index = pd.to_datetime(df_grouped.index)

    # Remove duplicate rows
    df_grouped = df_grouped[~df_grouped.index.duplicated(keep='first')]

    return df_grouped


def ts_decomposition(df):
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
    df_deseasonalized = pd.DataFrame()
    df_residuals = pd.DataFrame()

    for column in df.columns:
        # Retrieve the time series for the current vegetation class
        time_series = df[column]

        # Decomposition
        decomp = STL(time_series, period=PERIOD, seasonal=SEASONAL, trend=TREND,robust=ROBUST).fit(inner_iter=INNER_ITER, outer_iter=OUTER_ITER) # Parameters determined by optimization

        # Retrieve components  
        seasonal = decomp.seasonal
        resid = decomp.resid

        result_deseason = time_series - seasonal

        # Append result to dataframe
        df_deseasonalized[column] = result_deseason
        df_residuals[column] = resid

    return df_deseasonalized, df_residuals


def calculate_event_stats(df_time_series, start_time_value, end_time_value):    
    
    """
    Calculates event statistics (resilience, resistance, recovery) from a dataframe of time series and returns them in a dataframe.
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
    df_deseasonalized, df_residuals = ts_decomposition(df_time_series)

    df_stats = pd.DataFrame.from_dict({ # Put directly in dataframe
        'vpre': df_deseasonalized.loc[
            df_deseasonalized.index < start_time_value
        ].mean(), # Vpre

        'vdist': df_deseasonalized.loc[
            (df_deseasonalized.index >= start_time_value) & 
            (df_deseasonalized.index <= extended_end_time)
        ].min(), # Vdist (simple version)

        'vpost': df_deseasonalized.loc[
            df_deseasonalized.index > end_time_value
        ].mean(), # Vpost

    }, orient='index')

    # Calculate resilience, resistance and recovery
    df_stats.loc['resilience'] = df_stats.loc['vpost'] / df_stats.loc['vpre']
    df_stats.loc['resistance'] = df_stats.loc['vdist'] / df_stats.loc['vpre']
    df_stats.loc['recovery'] = df_stats.loc['vpost'] - df_stats.loc['vdist'] / df_stats.loc['vpre'] - df_stats.loc['vdist']

    # Calculate duration until maximum impact
    date_vdist = df_deseasonalized[start_time_value:extended_end_time].idxmin() # Date of maximum impact
    df_stats.loc['n_days_vdist'] = (date_vdist - start_time_value) / np.timedelta64(1, 'D') # Number of days until maximum impact

    # Calculate anomaly strength
    df_stats.loc['anomaly_strength'] = df_residuals[start_time_value:extended_end_time].sum(axis=0)
    df_stats.loc['anomaly_strength_day'] = df_residuals[start_time_value:extended_end_time].sum(axis=0) / len(df_residuals[start_time_value:extended_end_time]) # Divided by number of data points

    return df_stats

def calculate_n_days_affected(start_date, end_date, event_mask, event_label, ds_label):

    # Create list of timestamps for event period
    timestamps = pd.date_range(start=start_date, end=end_date, freq='D')

    # Wrap label dataset to coordinates of event mask
    lon_orig = ds_label.longitude.values
    lon_orig_wrapped = np.where(lon_orig > 180, lon_orig - 360, lon_orig) # Change from 0-360 to -180-180
    sorted_indices = np.argsort(lon_orig_wrapped) # Create sorting indices array
    lon_orig_wrapped_sorted = lon_orig_wrapped[sorted_indices] # Sort longitude values

    lat_orig = ds_label.latitude.values

    # Create empty array of shape of event mask
    event_count_array = np.zeros_like(event_mask)

    # Loop label dataset over timestamps, add 1 to empty array if timestamp contains event label
    for event_day in timestamps:
        labels_slice = ds_label.sel(Ti=event_day)['labels'].values # Get one time slice of the event per iteration
        event_count_array += (labels_slice == event_label).astype(int) # Add 1 to empty array if timestamp contains event label

    # Sort the longitudes and adjust the mask data accordingly
    event_count_array_sorted = event_count_array[:, sorted_indices]

    # Create new data array for the event count
    event_count_da = xr.DataArray(event_count_array_sorted, 
                                dims=["latitude", "longitude"],
                                coords={"latitude": lat_orig, "longitude": lon_orig_wrapped_sorted})

    # Resample to same grid as event mask
    event_count_da = event_count_da.interp(latitude=event_mask['lat'], longitude=event_mask['lon'], method='nearest') 
   
    # Apply event mask to event count array
    event_count_da = event_count_da.where(event_mask == 1)

    return event_count_da



def aggregate_by_vegetation(
    vegetation: xr.DataArray,
    event_mask: xr.DataArray,
    data_array: xr.DataArray,
    method: Literal['sum', 'mean'],
    area_weighting: bool = True
) -> pd.Series:
    """
    Aggregate values in a data array by vegetation class, with event-based masking
    and optional area-weighted aggregation using cosine-latitude weights.

    Parameters:
    - vegetation: Categorical vegetation labels (2D or 3D).
    - event_mask: Boolean mask (True = include).
    - data_array: Data to aggregate (2D or 3D). Should be predictor event cube bound to event period.
    - method: 'sum' or 'mean'.
    - area_weighting: If True, apply cosine-latitude-based weighting (normalized).

    Returns:
    - pd.Series indexed by vegetation class.
    """

    # Interpolate vegetation and event mask to match data_array
    reference_grid = data_array if 'time' not in data_array.dims else data_array.isel(time=0)
    vegetation_interp = vegetation.interp_like(reference_grid, method="nearest")
    event_mask_interp = event_mask.interp_like(reference_grid, method="nearest")

    # Apply event mask
    vegetation_masked = vegetation_interp.where(event_mask_interp)
    if 'time' in vegetation_masked.dims:
        vegetation_masked = vegetation_masked.isel(time=0)

    # Stack spatial dims
    veg_stacked = vegetation_masked.stack(space=('lat', 'lon'))
    da_stacked = data_array.stack(space=('lat', 'lon'))

    # Valid data mask
    if 'time' in da_stacked.dims:
        valid_mask = (~np.isnan(veg_stacked)) & (~np.isnan(da_stacked).all(dim='time'))
    else:
        valid_mask = (~np.isnan(veg_stacked)) & (~np.isnan(da_stacked))

    if hasattr(valid_mask, 'compute'):
        valid_mask = valid_mask.compute()

    veg_valid = veg_stacked.where(valid_mask, drop=True)
    data_valid = da_stacked.isel(space=valid_mask.values)

    # Cosine-latitude weighting (normalized)
    if area_weighting:
        latitudes = data_array['lat'].values
        lat_rad = np.deg2rad(latitudes)
        cos_weights = np.cos(lat_rad)
        cos_weights /= cos_weights.max()  # normalize: equator = 1

        weight_2d = np.repeat(cos_weights[:, np.newaxis], len(data_array['lon']), axis=1)
        weight_da = xr.DataArray(weight_2d, coords={'lat': latitudes, 'lon': data_array['lon'].values})
        weight_stacked = weight_da.stack(space=('lat', 'lon'))
        weight_valid = weight_stacked.isel(space=valid_mask.values).values
    else:
        weight_valid = np.ones(data_valid.sizes['space'])

    # Create dataframe
    if 'time' in data_valid.dims:
        df = pd.DataFrame(
            data_valid.transpose('space', 'time').values,
            columns=data_valid['time'].values
        )
        df['aggregation'] = df.mean(axis=1) if method == 'mean' else df.sum(axis=1)
    else:
        df = pd.DataFrame({'aggregation': data_valid.values})

    df['veg_class'] = veg_valid.values
    df['lat_weight'] = weight_valid

    # Weighted aggregation
    if area_weighting:
        grouped = df.groupby('veg_class')
        if method == 'mean':
            result = grouped.apply(lambda g: np.average(g['aggregation'], weights=g['lat_weight']))
        elif method == 'sum':
            result = grouped.apply(lambda g: np.sum(g['aggregation'] * g['lat_weight']))
    else:
        if method == 'mean':
            result = df.groupby('veg_class')['aggregation'].mean()
        elif method == 'sum':
            result = df.groupby('veg_class')['aggregation'].sum()

    return result




'''DEPRECATED, WITHOUT COSINE WEIGHTING

def aggregate_by_vegetation(vegetation, event_mask, data_array, method: Literal['sum', 'mean']):

    """
    Aggregate values in a data array by vegetation class, with optional event-based masking.

    This function computes the mean or sum of a data array variable grouped by vegetation classes.
    It supports both 2D data arrays (('lat', 'lon')) and 3D time series (('time', 'lat', 'lon')).
    Aggregation is limited to spatial locations where the event_mask is 'True'

    Parameters:
    - vegetation (xarray.DataArray): A 2D or 3D array containing categorical vegetation class labels. If 3D, only the first time slice is used.
    - event_mask (xarray.DataArray): A mask of valid areas (same spatial dimensions as data_array), where True indicates valid pixels.
    - data_array (xarray.DataArray): The data to be aggregated. Must have shape either (time, lat, lon) or (lat, lon).xarray.DataArray
    - method {'sum', 'mean'}: Aggregation method to use across time (if applicable) and across pixels within each vegetation class.

    Returns: 
    -  A pandas.Series indexed by vegetation class, containing the aggregated value (sum or mean) per class.


    Notes
    -----
    - If data_array is 3D, the aggregation is performed across time for each pixel before grouping by vegetation class.
    - If data_array is 2D, values are grouped directly by vegetation class.
    - The function automatically interpolates vegetation and event_mask to match the spatial grid of data_array.
    - Pixels with NaNs in either vegetation or data_array are excluded from the aggregation.

    """

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

    return df'''


# BACKUP
'''def ts_decomposition(df,
                     method: Literal['deseasonalized', 'residuals', 'trend', 'seasonal'] = 'deseasonalized'
                     ):
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
    df_decomposed = pd.DataFrame()

    for column in df.columns:
        # Retrieve the time series for the current vegetation class
        time_series = df[column]

        # Decomposition
        decomp = STL(time_series, period=PERIOD, seasonal=SEASONAL, trend=TREND,robust=ROBUST).fit(inner_iter=INNER_ITER, outer_iter=OUTER_ITER) # Parameters determined by optimization

        # Retrieve components  
        seasonal = decomp.seasonal
        trend = decomp.trend
        resid = decomp.resid

        # Compute
        if method == 'deseasonalized':
            # Deseasonalize time series
            result = time_series - seasonal
        elif method == 'residuals':
            # Retrieve residuals
            result = resid
        elif method == 'trend':
            result = trend
        elif method == 'seasonal':
            result = seasonal
        else:
            raise ValueError("Invalid method. Choose 'deseasonalized', 'residuals' or 'trend'.")

        # Append result to dataframe
        df_decomposed[column] = result

    return df_decomposed
    
    
    
    
    def calculate_event_stats(df_time_series, start_time_value, end_time_value):    
    
    """
    Calculates event statistics (resilience, resistance, recovery) from a dataframe of time series and returns them in a dataframe.
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
    df_stats.loc['recovery'] = df_stats.loc['vpost'] - df_stats.loc['vdist'] / df_stats.loc['vpre'] - df_stats.loc['vdist']

    return df_stats'''