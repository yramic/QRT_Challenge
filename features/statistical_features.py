import numpy as np
import pandas as pd
from typing import List

from scipy.stats import skew, kurtosis

def calculate_statistics(df: pd.DataFrame) -> tuple[dict, dict, dict]:
    """Computes the statistics for every feature of the dataframe and returns dictionaries."""
    # Initialize base statistics dictionary
    def initialize_stats():
        return {
            'Mean': {},
            'Std': {},
            'Min': {},
            'Max': {},
            'Skewness': {},
            'Kurtosis': {},
            'Std_Std': {},
        }
    
    # Function to compute statistics for a given feature
    def compute_stats(feature, stats_dict, key):
        stats_dict['Mean'][key] = np.mean(feature)
        stats_dict['Std'][key] = np.std(feature)
        stats_dict['Min'][key] = np.min(feature)
        stats_dict['Max'][key] = np.max(feature)
        stats_dict['Skewness'][key] = skew(feature, nan_policy='omit')
        stats_dict['Kurtosis'][key] = kurtosis(feature, nan_policy='omit')
        stats_dict['Std_Std'][key] = np.std(feature.rolling(window=20).std().dropna())
    
    # Initialize dictionaries for original, log, and log-squared statistics
    stats = initialize_stats()
    log_stats = initialize_stats()
    log_squared_stats = initialize_stats()

    for key in df.keys():
        # Process only for VOLUME and RET columns
        if "VOLUME" in key or "RET" in key:
            # Original feature
            feature = df[key]
            compute_stats(feature, stats, key)

            # Logarithmic feature
            log_feature = feature.copy().replace(0, np.nan)  # Replace 0 with NaN
            log_feature = log_feature[log_feature > 0]  # Filter out non-positive values
            log_feature = np.log(log_feature)  # Apply the logarithm safely
            compute_stats(log_feature, log_stats, key)

            # Log-squared feature
            log_squared_feature = log_feature ** 2
            compute_stats(log_squared_feature, log_squared_stats, key)

    return stats, log_stats, log_squared_stats


def compute_conditional_features(
        df: pd.DataFrame, 
        test: pd.DataFrame, 
        shifts: int | List = [1, 2, 3, 4], 
        statistics: str | List = ['mean', 'sum'], 
        gb_features_list: str | List = [['SECTOR', 'DATE']], 
        target_features: str | List = ['RET']
    ) -> tuple:
    """
    Calculate conditional aggregated features for the given shifts, statistics, and group-by features.

    Args:
    train: training data containing target features and group-by features.
    test: testing data containing target features and group-by features.
    shifts: list, of shifts to calculate (default: [1, 2, 3, 4]).
    statistics: list, of statistics to calculate (default: ['mean']).
    gb_features_list: list of group-by feature lists (default: [['SECTOR', 'DATE']]).
    target_features: list of target features to calculate (default: ['RET']).

    Returns:
    pd.DataFrame for train and test with added conditional aggregated features.
    list of new feature names.
    """
    new_features = []

    for target_feature in target_features:
        for gb_features in gb_features_list:
            tmp_name = '_'.join(gb_features)
            for shift in shifts:
                for stat in statistics:
                    name = f'{target_feature}_{shift}_{tmp_name}_{stat.upper()}'
                    feat = f'{target_feature}_{shift}'
                    new_features.append(name)
                    for data in [df, test]:
                        data[name] = data.groupby(gb_features)[feat].transform(stat)
    
    return df, test, new_features



def compute_volatility(
        train: pd.DataFrame, 
        test: pd.DataFrame, 
        periods: int | List = [2], 
        targets: str | List = ['RET', 'VOLUME']
    ) -> tuple:
    """
    Compute volatility (standard deviation) for specified targets over given periods.

    Args:
    train: training data containing target columns.
    test: testing data containing target columns.
    periods: list of periods in weeks to calculate volatility for (default: [2]).
    targets: list of target columns to calculate volatility for (default: ['RET', 'VOLUME']).

    Returns:
    pd.DataFrame for train and test with added volatility features.
    list of new feature names.
    """
    new_features = []

    for period in periods:
        window_size = 5 * period
        for target in targets:
            name = f'{window_size}_day_mean_{target}_VOLATILITY'
            new_features.append(name)
            for data in [train, test]:
                rolling_std_target = (
                    data.groupby(['SECTOR', 'DATE'])
                    [[f'{target}_{day}' for day in range(1, window_size + 1)]]
                    .mean()
                    .std(axis=1)
                    .to_frame(name)
                )
                placeholder = data.join(rolling_std_target, on=['SECTOR', 'DATE'], how='left')
                data[name] = placeholder[name]

    return train, test, new_features