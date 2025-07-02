import numpy as np
import pandas as pd

def calculate_anomaly_score(residuals_normal, residuals_extreme, start_date, end_date):
    """
    Calculate the anomaly score for a given time period. Used by objetive function for decomposition optimization but maybe also for extreme event severity quantification.as_integer_ratio

    Parameters:
    - residuals_normal (pd.Series): Residuals from the normal time series.
    - residuals_extreme (pd.Series): Residuals from the extreme time
    - start_date (str): Start date of the anomaly period.
    - end_date (str): End date of the anomaly period.

    Returns:
    - anomaly_score (float): The calculated anomaly score.
    
    """
    # Sum of absolute values of trend component
    start_date_date = pd.to_datetime(start_date)
    end_date_date = pd.to_datetime(end_date)
    idx_start = residuals_extreme.index.get_indexer([start_date_date], method='nearest')[0] # Get nearest index to start date
    idx_end = residuals_extreme.index.get_indexer([end_date_date], method='nearest')[0]

    trend_slice = residuals_extreme.iloc[idx_start:idx_end + 1]
    sum_abs = abs(trend_slice.sum())

    # Standard deviation of normal residuals
    resid_std = np.std(residuals_normal)

    # Z-scores of extreme residuals in the anomaly window
    z_scores = np.abs(residuals_extreme.iloc[idx_start:idx_end + 1]) / resid_std

    z_min = z_scores.min()
    z_max = z_scores.max()

    if z_max == z_min:
        anomaly_score = 0.0 # Avoid dividing by 0
    else:
        normalized_z_scores = (z_scores - z_min) / (z_max - z_min)
        anomaly_score = normalized_z_scores.mean()

    return float(anomaly_score)
