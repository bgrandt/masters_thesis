from typing import Literal
import pandas as pd
import numpy as np

def aggregate_kndvi_timeseries(
    kndvi_array, 
    vegetation_array,
    event_mask,
    df_translation, 
    method: Literal['mean', 'median', 'min', 'max'] = 'mean'
):
    """
    Aggregates kNDVI time series per vegetation class for a given extreme event. Method of aggregation can be chosen.

    Parameters:
        kndvi_array (xarray.DataArray): The kNDVI data for the event (time, lat, lon), output from create_event_array().
        vegetation_array (xarray.DataArray): Vegetation class mask (lat, lon) or (time, lat, lon).
        event_mask (xarray.DataArray): Event mask (lat, lon) that is 1 for the event and 0 otherwise.
        df_translation (pandas.DataFrame): DataFrame with columns 'class' and 'label' for translation of vegetation classes.
        method (str): The method to use for aggregation ('mean' or 'median').

    Returns:
        None. Appends result to `output_dict` under key `df_<event_label>_mean_kndvi_ts`.
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
    # Group by vegetation class and transpose to have time as index
    if method == 'mean':
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

    # Create mapping of class labels to class names
    translation_dict = dict(zip(df_translation.index, df_translation.iloc[:, 0]))

    # Rename df_grouped columns (excluding time index if it's part of the DataFrame)
    df_grouped = df_grouped.rename(columns=translation_dict)

    return df_grouped
