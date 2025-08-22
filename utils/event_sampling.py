import numpy as np
import pandas as pd
import xarray as xr
import numbers


def get_data_pixel_coordinates(event_dataarray):
    """
    Extracts the coordinates of the pixels in the event dataset that actually have data.
    
    Parameters:
    - event_dataarray (xr.DataArray): The DataArray containing the event data.
    
    Returns:
    - coordinates (pandas.DataFrame): A DataFrame with coordinates of pixels that have data.
    """

    if not isinstance(event_dataarray, xr.DataArray):
        raise TypeError("Input must be an xarray.DataArray")

    # Convert binary 0/1 array to boolean
    if set(np.unique(event_dataarray.values)).issubset({0, 1}):
        event_dataarray = event_dataarray.astype(bool)

    dims = event_dataarray.dims

    # Create mask depending on array dimensionality
    if len(dims) == 3:
        # Collapse time axis
        if event_dataarray.dtype == bool:
            mask = event_dataarray.any(dim=dims[0]).compute()
        else:
            mask = ~np.all(np.isnan(event_dataarray), axis=0).compute()
    elif len(dims) == 2:
        if event_dataarray.dtype == bool:
            mask = event_dataarray.compute()
        else:
            mask = ~np.isnan(event_dataarray).compute()
    else:
        raise ValueError(f"Expected 2D or 3D array, got {len(dims)}D.")

    # Get indices of True values in the mask
    true_idx = np.where(mask)

    # Extract coordinate names (lat/lon/x/y)
    coords = [c for c in mask.coords if c in ['lat', 'lon', 'x', 'y']]
    
    # Extract the coordinate values
    coord_data = {c: mask[c].values[true_idx[i]] for i, c in enumerate(coords)}
    
    # Convert to DataFrame
    coordinates = pd.DataFrame(coord_data)

    return coordinates

'''def get_data_pixel_coordinates(event_dataarray):
    """
    Extracts the coordinates of the pixels in the event dataset that actually have data.
    
    Parameters:
    - event_dataarray (xr.DataArray): The DataArray containing the event data.
    
    Returns:
    - coordinates (pandas.DataFrame): A DataFrame with coordinates of pixels that have data.
    """

    if not isinstance(event_dataarray, xr.DataArray):
        raise TypeError("Input must be an xarray.DataArray")

    # Convert binary 0/1 array to boolean
    if set(np.unique(event_dataarray.values)).issubset({0, 1}):
        event_dataarray = event_dataarray.astype(bool)

    dims = event_dataarray.dims

    # Create mask depending on dimensionality
    if len(dims) == 3:
        mask = event_dataarray.any(dim=dims[0])
    elif len(dims) == 2:
        mask = event_dataarray
    else:
        raise ValueError(f"Expected 2D or 3D array, got {len(dims)}D.")

    # Ensure boolean mask
    mask = mask.astype(bool).compute()

    # Stack dimensions into a single index
    stacked = mask.stack(z=mask.dims)

    # Keep only True values (valid pixels)
    valid_pixels = stacked[stacked]

    # Convert to DataFrame directly (includes coordinates)
    coords_df = valid_pixels.to_dataframe(name='mask').reset_index()

    # Drop helper columns
    return coords_df.drop(columns=['z', 'mask'])'''



def create_event_mask(event_label, start_date, end_date, labelcube, land_mask):
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
    if not isinstance(event_label, numbers.Integral):
        raise TypeError(f"Expected event_label to be an int, but got {type(event_label).__name__}")

    # Get event time range
    timestamps = pd.date_range(start=start_date, end=end_date, freq="D")

    # Select time slices where event may have occurred
    event_labels = labelcube.sel(Ti=timestamps)['labels'] # DO i need method='nearest'?

    # Deubgging: see if there are any time slices after selection
    if event_labels.size == 0:
        raise ValueError(f"No time slices found for event label {event_label}")

    # Identify affected grid cells
    affected_area_mask = np.any(event_labels == event_label, axis=0)

    # Debugging: see if there are any affected grid cells
    if not np.any(affected_area_mask):
        raise ValueError(f"No affected grid cells found for event label {event_label}. There might be no grid cells with this label.")

    # Replace NaNs with 0
    affected_area_mask = np.where(np.isnan(affected_area_mask), 0, affected_area_mask)

    # Debugging: check if there are any True values in the mask
    if not np.any(affected_area_mask):
        raise ValueError(f"Mask does not contain values after replacing NaNs.")

    # Longitude wrapping (convert from 0–360 to -180–180)
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

    # Interpolate land mask to match event mask resolution/coordinates
    land_mask_interp = land_mask.sel(time=land_mask.time[0]).drop("time")
    land_mask_interp = land_mask_interp.interp(lat=affected_area_mask_da.lat, lon=affected_area_mask_da.lon)

    # Debugging: check if interpolation was successful
    if not np.any(land_mask_interp): 
        raise ValueError("Interpolation resulted in NaNs.")

    # Apply the land mask: keep only where land_mask == 1
    land_bool_mask = (land_mask_interp == 1)
    masked_event = affected_area_mask_da.where(land_bool_mask, 0)
    
    return masked_event





def create_event_dataarray(d_array, event_mask, start_date, end_date, time_buffer=0):

    """
    Returns a minicube of the event with the event mask applied and a time buffer before and after the event.

    Parameters:
    - d_array (xarray.DataArray): containing the data (usually kNDVI) that has to be masked.
    - event_mask (xarray.DataArray): DataArray with the boolean information of the affected area.
    - start_date (str): start date of the event.
    - end_date (str): end date of the event.
    - time_buffer (int): number of days to be added before and after the end of the event.

    Returns:
    - xarray DataArray minicube containing data where the event mask is True and a time buffer before and after the event.
    """
############
    ## Crop to region of interest ###
    df_coords = get_data_pixel_coordinates(event_mask)
    lat_min = df_coords['lat'].min() - 5
    lat_max = df_coords['lat'].max() + 5
    lon_min = df_coords['lon'].min() - 5
    lon_max = df_coords['lon'].max() + 5
#############

    # Handle dimension naming 
    if 'Ti' in d_array.dims:
        d_array = d_array.rename({'Ti': 'time'})
    
    if 'latitude' in d_array.dims:
        d_array = d_array.rename({'latitude': 'lat'})

    if 'longitude' in d_array.dims:
        d_array = d_array.rename({'longitude': 'lon'})

    # Regrid event mask to match data array
    affected_area_mask_regridded = event_mask.interp(lat=d_array.lat, lon=d_array.lon, method="nearest")

    # Change mask to boolean
    affected_area_mask_regridded = affected_area_mask_regridded.astype(bool)

    # Apply mask to dataset
    dataset_masked = d_array.where(affected_area_mask_regridded)

    # Create a list of timestamps for one event
    timestamps = pd.date_range(start=start_date, end=end_date, freq="D")
    
    # Add a time buffer before and after the event
    timestamps = pd.date_range(start=start_date - pd.Timedelta(days=time_buffer), end=end_date + pd.Timedelta(days=time_buffer), freq="D")

    # Select the data for the event
    dataset_masked = dataset_masked.sel(time=timestamps, method='nearest')

    # Crop to bounding box
    if d_array.lat.values[0] > d_array.lat.values[-1]:
        # lat is descending
        event_cube = dataset_masked.sel(
            lon=slice(lon_min, lon_max),
            lat=slice(lat_max, lat_min)  # reverse bounds
        )
    else:
        # lat is ascending
        event_cube = dataset_masked.sel(
            lon=slice(lon_min, lon_max),
            lat=slice(lat_min, lat_max)
        )


    return event_cube


def check_for_other_events(start_date, end_date, event_mask, labelcube, label, events):

    # Create labelcube subset for event
    subset = create_event_dataarray(labelcube['labels'], event_mask, start_date, end_date, time_buffer=2*365)
    subset_events = np.unique(subset.values)

    # Create list with matching events despite actual event
    other_events = [int(ev) for ev in subset_events if not np.isnan(ev) and int(ev) != label]

    return any(ev in events for ev in other_events)
    

