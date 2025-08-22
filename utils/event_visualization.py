import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from typing import Literal


def plot_event_day(event_array, day):
    
    """
    Plots the event parameter (e.g. kNDVI) for a given day.

    Parameters:
        event_array (xarray.DataArray): The DataArray containing the event data. Output from create_event_dataarray.
        day (int; str): The day to plot, either as integer representing the index of the day in the time dimension (useful for looping) 
        or as string representing the date in the format 'YYYY-MM-DD' (useful for plotting a specific day).

    Returns:
        None
        
    """


    # Remove date duplicates
    event_array = event_array.sel(time=~event_array.get_index("time").duplicated())

    time_len = event_array.sizes['time']
    time_values = pd.to_datetime(event_array.time.values)

    # If day is given as time index
    if isinstance(day, int):

        # Check if given number is within the range of the time dimension
        if not 0 <= day < event_array.sizes['time']:
            raise IndexError(f"Index {day} is out of bounds for time dimension with indices 0 - {time_len - 1}.")

        # Extract date as string
        date_str = pd.to_datetime(event_array.isel(time=day).time.values).strftime('%Y-%m-%d')

        # Plotting with Cartopy
        fig = plt.figure(figsize=(11, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # Plot the data
        event_array.isel(time=day).plot(
            ax=ax,
            cmap='YlGn',
            robust=True,
            transform=ccrs.PlateCarree(),
            cbar_kwargs={'label': 'KNDVI'}
        )

    # If day is a string
    elif isinstance(day, str):

        # Extract date as string
        date_str = pd.to_datetime(day).strftime('%Y-%m-%d')
        date = pd.to_datetime(day)

        # Check if given date is within the time range of the cube
        if date not in time_values:
            valid_dates = [d.strftime('%Y-%m-%d') for d in time_values]
            raise ValueError(f"Date {date_str} is not within the time range of the cube. Valid dates are: \n {valid_dates}")

        # Plotting with Cartopy
        fig = plt.figure(figsize=(11, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # Plot the data
        event_array.sel(time=date, method='nearest').plot(
            ax=ax,
            cmap='YlGn',
            robust=True,
            transform=ccrs.PlateCarree(),
            cbar_kwargs={'label': 'KNDVI'}
        )

    else:
        # Handle invalid day input
        raise ValueError("Invalid day input. Please provide an integer index for the dimension time or a date string.")

    # Add coastlines and gridlines
    ax.coastlines(alpha=0.8)
    ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)

    # Add title and labels
    ax.set_title(f"Event data from {date_str}")
    plt.show()



def plot_kndvi_time_series(event_array, lat, lon, event_label, event_table):

    """
    Plots the KNDVI time series for a given coordinate.

    Parameters:
    - lat (float): Latitude of the coordinate.
    - lon (float): Longitude of the coordinate. 
    - event_dataset (xarray.DataArray): The DataArray for a specific event, containing event parameters like kNDVI. Output of create_event_dataarray.
    - event_label (str): The label of the event.
    - event_table (pandas.DataFrame): The event summary table. (df_chd_filtered)

    Returns:
    - None   # The function plots the time series and does not return any value.

    """
    # Extract start and end date
    start_date = event_table[event_table['label'] == event_label]['start_time'].iloc[0]
    end_date = event_table[event_table['label'] == event_label]['end_time'].iloc[0]

    # Extract the time series data for the coordinate
    time_series = event_array.sel(lat=lat, lon=lon, method="nearest")

    # Plot the time series
    time_series.plot()
    plt.title(f'Time Series for Pixel at (lat={lat}, lon={lon}) for event {event_label}')
    plt.xlabel('Time')
    plt.ylabel('KNDVI')

    # Set the x-axis limits to the full time range of kndvi_masked_event
    plt.xlim(event_array.time[0].values, event_array.time[-1].values)

    # Mark the start and end dates of the event
    plt.axvline(x=start_date, color='r', linestyle='--', label='event start')
    plt.axvline(x=end_date, color='g', linestyle='--', label='event end')

    # Add legend
    plt.legend()

    plt.show()



def plot_kndvi_time_series_with_baseline(event_array, lat, lon, event_label, event_table, full_kndvi_data):
    """
    Plots the KNDVI time series at a given coordinate with historical 20-year daily mean. This takes way longer than the
    plot_kndvi_time_series function, because it calculates the 20-year daily mean for each pixel.

    Parameters:
    - event_mask (xarray.DataArray): The DataArray for a specific event, containing event parameters like kNDVI. Output of create_event_dataarray.
    - lat, lon (float): Coordinate of the pixel for which time series shall be created.
    - event_label (str): The label of the event.
    - event_table (pandas.DataFrame): The event summary table. (df_chd_filtered)
    - full_kndvi_data (xarray.DataArray): Full unmasked kNDVI data over the years.

    Returns:
    - None
    """

    # Extract start and end date
    start_date = event_table[event_table['label'] == event_label]['start_time'].iloc[0]
    end_date = event_table[event_table['label'] == event_label]['end_time'].iloc[0]

    # Extract time series for the given point
    event_ts = event_array.sel(lat=lat, lon=lon, method="nearest")

    # Compute the historical average from previous 2 years
    first_year = start_date.year - 2
    last_year = start_date.year - 1

    # Time points from the event for day-of-year reference
    event_days = pd.to_datetime(event_ts.time.values).dayofyear

    # Extract full time series for the location
    full_ts = full_kndvi_data.sel(lat=lat, lon=lon, method="nearest")

    # Select the 2-year baseline range
    full_ts_20yr = full_ts.sel(time=slice(f"{first_year}", f"{last_year}"))

    # Ensure time is a datetime index
    full_time_dt = pd.to_datetime(full_ts_20yr.time.values)
    full_ts_20yr.coords["dayofyear"] = ("time", full_time_dt.dayofyear)

    # Match day-of-year with event days
    historical_ts = full_ts_20yr.where(full_ts_20yr.dayofyear.isin(event_days), drop=True)

    # Group by day-of-year and average over the years
    historical_mean = historical_ts.groupby("dayofyear").mean(dim="time")

    # Sort event data by day-of-year to align with historical
    event_ts_df = pd.DataFrame({
        "date": pd.to_datetime(event_ts.time.values),
        "kndvi": event_ts.values,
        "dayofyear": pd.to_datetime(event_ts.time.values).dayofyear
    }).sort_values("dayofyear")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(event_ts_df["date"], event_ts_df["kndvi"], label="Event kNDVI", color="blue")
    plt.plot(event_ts_df["date"], historical_mean.sel(dayofyear=event_ts_df["dayofyear"].values).values,
             label="20-Year Historical Mean", color="orange")

    # Mark start and end of the event
    plt.axvline(start_date, color='r', linestyle='--', label='Event Start')
    plt.axvline(end_date, color='g', linestyle='--', label='Event End')

    plt.title(f'KNDVI Time Series at (lat={lat}, lon={lon}) for event {event_label}\nEvent vs 20-Year Historical Average')
    plt.xlabel("Date")
    plt.ylabel("kNDVI")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_aggregated_ts(df_time_series, 
                       start_time_value, 
                       end_time_value, 
                       variable: str,
                       df_translation: pd.DataFrame = None):
    """
    Plots aggregated  time series per vegetation class for a given extreme event.

    Parameters:
       df_time_series (pandas.DataFrame): DataFrame with aggregated time series.
       start_time_value (datetime): Start time of the event.
       end_time_value (datetime): End time of the event.
       method (str): The method used for aggregation ('mean', 'median', 'min', 'max').
       df_translation (pandas.DataFrame): DataFrame with columns 'class' and 'label' for translation of vegetation classes. 
    """

    if df_translation is not None:
        # Create mapping of class labels to class names
        translation_dict = dict(zip(df_translation.index, df_translation.iloc[:, 0]))

        # Rename df_grouped columns (excluding time index if it's part of the DataFrame)
        df_time_series = df_time_series.rename(columns=translation_dict)

    # Calculate time buffer to caputre lagged responses
    event_duration = end_time_value - start_time_value
    buffer_duration = event_duration * 0.5
    extended_end_time = end_time_value + buffer_duration

    df_time_series.plot(figsize=(12, 6))
    plt.axvline(start_time_value, color='red', linestyle='--', label='Event Start')
    plt.axvline(end_time_value, color='blue', linestyle='--', label='Event End')
    plt.axvline(extended_end_time, color='green', linestyle='--', label='Extended End')
    plt.title(f"{variable} Time Series per Vegetation Class")
    plt.xlabel("Time")
    plt.ylabel(variable)
    # Legend outside (right side)
    plt.legend(
        title="Veg Class",
        bbox_to_anchor=(1.02, 1),  # shift right outside
        loc='upper left',
        borderaxespad=0
    )
    plt.grid(True)
    plt.tight_layout()
    plt.show()
