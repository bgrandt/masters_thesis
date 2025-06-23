import numpy as np
import pandas as pd
import xarray as xr

def create_event_mask(event_label, event_table, labelcube, land_mask):
    """ 
    Creates the affected area mask for a given event label, applying a land mask.

    Parameters:
    - event_label (int): The label of the event to create the mask for.
    - event_table (pd.DataFrame): DataFrame containing event information.
    - labelcube (xr.Dataset): Dataset containing label information (ds_chd_labels).
    - land_mask (xr.DataArray): Land mask (0 = water, 1 = land).

    Returns:
    - xr.DataArray: The affected land area mask for the given event label.
    """

    # Error handling if label is not an integer
    if not isinstance(event_label, int):
        raise TypeError(f"Expected event_label to be an int, but got {type(event_label).__name__}")

    # Get event time range
    start_date = event_table[event_table['label'] == event_label]['start_time'].iloc[0]
    end_date = event_table[event_table['label'] == event_label]['end_time'].iloc[0]
    timestamps = pd.date_range(start=start_date, end=end_date, freq="D")

    # Get the shape of the labelcube spatial dimensions
    time_slice_shape = labelcube.sizes['latitude'], labelcube.sizes['longitude']

    # Select time slices where event may have occurred
    event_labels = labelcube.sel(Ti=timestamps)['labels']

    # Identify affected grid cells
    affected_area_mask = np.any(event_labels == event_label, axis=0)

    # Replace NaNs with 0
    affected_area_mask = np.where(np.isnan(affected_area_mask), 0, affected_area_mask)

    # Longitude wrapping if needed (convert from 0–360 to -180–180)
    lon_orig = labelcube['longitude']
    lon_orig_wrapped = np.where(lon_orig > 180, lon_orig - 360, lon_orig)
    sorted_indices = np.argsort(lon_orig_wrapped)
    lon_sorted = lon_orig_wrapped[sorted_indices]
    affected_area_mask_sorted = affected_area_mask[:, sorted_indices]

    lat_orig = labelcube['latitude']

    # Convert to DataArray
    affected_area_mask_da = xr.DataArray(
        affected_area_mask_sorted,
        dims=["latitude", "longitude"],
        coords={"latitude": lat_orig, "longitude": lon_sorted}
    )

    # Rename to match the land_mask dimensions
    affected_area_mask_da = affected_area_mask_da.rename({"latitude": "lat", "longitude": "lon"})

    # Interpolate land mask to match event mask resolution/coordinates if needed
    land_mask_interp = land_mask.sel(time=land_mask.time[0]).drop("time")
    land_mask_interp = land_mask_interp.interp(lat=affected_area_mask_da.lat, lon=affected_area_mask_da.lon)

    # Apply the land mask: keep only where land_mask == 1
    land_bool_mask = (land_mask_interp == 1)
    masked_event = affected_area_mask_da.where(land_bool_mask, 0)

    return masked_event





def create_event_dataarray(d_array, event_mask, event_table, event_label, time_buffer=60):

    """
    Returns a minicube of the event with the event mask applied and a time buffer before and after the event.

    Parameters:
    - d_array (xarray.DataArray): containing the data (usually kNDVI) that has to be masked.
    - event_mask (xarray.DataArray): DataArray with the boolean information of the affected area.
    - event_label (int): label of the event.
    - time_buffer (int): number of days to be added before and after the end of the event.

    Returns:
    - xarray DataArray minicube containing data where the event mask is True and a time buffer before and after the event.
    """
############
    # Define region of interest for cropping with 5 degrees buffer
    lon_min = event_table[event_table['label'] == event_label]['longitude_min'].iloc[0] - 5
    lon_max = event_table[event_table['label'] == event_label]['longitude_max'].iloc[0] + 5 
    lat_min = event_table[event_table['label'] == event_label]['latitude_min'].iloc[0] - 5
    lat_max = event_table[event_table['label'] == event_label]['latitude_max'].iloc[0] + 5

#############

    # Regrid affected area mask to match the dataset's grid
    affected_area_mask_regridded = event_mask.interp(lat=d_array.lat, lon=d_array.lon, method="nearest")

    # Change mask to boolean
    affected_area_mask_regridded = affected_area_mask_regridded.astype(bool)

    # Apply mask to dataset
    dataset_masked = d_array.where(affected_area_mask_regridded)

    # Create a list of timestamps for one event
    start_date = event_table[event_table['label'] == event_label]['start_time'].iloc[0]
    end_date = event_table[event_table['label'] == event_label]['end_time'].iloc[0]
    timestamps = pd.date_range(start=start_date, end=end_date, freq="D")
    
    # Add a time buffer before and after the event
    timestamps = pd.date_range(start=start_date - pd.Timedelta(days=time_buffer), end=end_date + pd.Timedelta(days=time_buffer), freq="D")

    # Select the data for the event
    dataset_masked = dataset_masked.sel(time=timestamps, method='nearest')

    # Crop to bounding box
    event_cube = dataset_masked.sel(
    lon=slice(lon_min, lon_max),
    lat=slice(lat_min, lat_max))

    return event_cube


def get_data_pixel_coordinates(event_dataarray):

    """
    Extracts the coordinates of the pixels in the event dataset that actually have data. This function is used for sampling.
    Parameters:
    - event_dataset (xr.Dataset): The dataset containing the event data.
    Returns:
    - coordinates (pandas.DataFrame): A DataFrame containing the coordinates of the pixels in the event dataset that actually have values.
    """
   
    # Create a boolean mask that is True where the data is not NaN for at least one time slice
    mask = ~np.all(np.isnan(event_dataarray), axis=0).compute()
    
    # Get the coordinates of the True values in the mask
    mask = mask.where(mask, drop=True)

    # Extract coordinates of True values
    coordinates = (
    mask.to_dataframe(name="has_data")
        .query("has_data == True")
        .reset_index()[['lat', 'lon']]
    )

    return coordinates

