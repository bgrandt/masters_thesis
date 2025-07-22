import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox

def calculate_anomaly_score(residuals_normal, residuals_extreme, start_date, end_date):
    """
    Calculate the anomaly score for a given time period. Used by objetive function for decomposition optimization but maybe also for extreme event severity quantification.

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

    '''trend_slice = residuals_extreme.iloc[idx_start:idx_end + 1]
    sum_abs = abs(trend_slice.sum())'''

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


def calculate_similarity(decomposition_a, decomposition_b, start_date, end_date):
    
    # Extract components
    trend_a = decomposition_a.trend
    seasonal_a = decomposition_a.seasonal
    residuals_a = decomposition_a.resid

    trend_b = decomposition_b.trend
    seasonal_b = decomposition_b.seasonal
    residuals_b = decomposition_b.resid

    # RMSE between components
    rmse_trend = np.sqrt(np.mean((trend_a - trend_b) ** 2))
    rmse_seasonal = np.sqrt(np.mean((seasonal_a - seasonal_b) ** 2))

    # Sum of absolute difference
    diff_trend = np.sum(np.abs(np.abs(trend_a) - np.abs(trend_b)))
    diff_seasonal = np.sum(np.abs(np.abs(seasonal_a) - np.abs(seasonal_b)))

    #### Objective function score ####
    # Calculate lower and upper bounds for RMSE normalization
    combined_trend = np.concatenate([trend_a, trend_b]) # Min and max of combined trend and seasonal components is required
    min_trend = combined_trend.min()
    max_trend = combined_trend.max()

    combined_seasonal = np.concatenate([seasonal_a, seasonal_b])
    min_seasonal = combined_seasonal.min()
    max_seasonal = combined_seasonal.max()

    # Compute scores
    trend_sim_score = np.sqrt(mean_squared_error(trend_a, trend_b)) / (max_trend - min_trend) # RMSE penalizes large deviations more
    seasonal_sim_score = np.sqrt(mean_squared_error(seasonal_a, seasonal_b)) / (max_seasonal - min_seasonal)
    anomaly_strength_score = calculate_anomaly_score(residuals_a, residuals_b, start_date, end_date)
    ljung_box_score = acorr_ljungbox(residuals_a, lags=[10], return_df=True)['lb_pvalue'].iloc[0] 

    score = ((trend_sim_score + seasonal_sim_score) +
                (1 - anomaly_strength_score) + 
                (1 - ljung_box_score))
    
    print(f"RMSE trend: {rmse_trend:.6f}")
    print(f"RMSE seasonal: {rmse_seasonal:.6f}")
    print(f"Sum of absolute difference trend: {diff_trend:.6f}")
    print(f"Sum of absolute difference seasonal: {diff_seasonal:.6f}")
    print(f"Score: {score:.6f}")
    
    