"""Summary of Functions for all Volume Related Features"""
import pandas as pd
import numpy as np

def compute_vpt(df: pd.DataFrame, window: int = 20) -> tuple:
    """Calculate the Volume Price Trend (VPT) for each stock and row."""
    
    df = df.copy()
    # Ensure the data is sorted by STOCK and DATE
    df = df.sort_values(by=['STOCK', 'DATE'])
    # Initialize the VPT column
    df['VPT'] = 0.0

    for day in reversed(range(1, window)):
        # RET today
        RET_today = df[f'RET_{day}']
        # RET yesterday
        RET_yesterday = df[f'RET_{day+1}']
        # Update VPT acccording to the formula
        df['VPT'] += df[f'VOLUME_{day}'] * (RET_today - RET_yesterday) / (RET_yesterday + 1e-8)
    
    new_features = ['VPT',]
    return (df[['STOCK', 'DATE', 'VPT']], new_features)


def compute_obv(df: pd.DataFrame, window: int = 20) -> tuple:
    
    df = df.copy()
    # Ensure the data is sorted by STOCK and DATE
    df = df.sort_values(by=['STOCK', 'DATE'])
    # Initialize the OBV column
    df['OBV'] = 0.0

    for day in reversed(range(1, window)):
        # RET today
        RET_today = df[f'RET_{day}']
        # RET yesterday
        RET_yesterday = df[f'RET_{day+1}']
        # VOLUME today
        VOLUME_today = df[f'VOLUME_{day}']

        # Update OBV according to the formula
        df['OBV'] += (RET_today > RET_yesterday) * VOLUME_today
        df['OBV'] -= (RET_today < RET_yesterday) * VOLUME_today
    
    new_features = ['OBV',]
    return (df[['STOCK', 'DATE', 'OBV']], new_features)


def compute_vwap(df: pd.DataFrame, n_days: int = 5, window: int = 20) -> tuple:
    assert window > n_days, "n_days needs to be bigger than the window size!"

    df = df.copy()

    df['PRICE'] = 1.0
    for day in reversed(range(n_days + 1, window+1)):
        df['PRICE'] *= (1 + df[f'RET_{day}'])
    
    for day_i in reversed(range(1, n_days + 1)):
        columns = [f'VOLUME_{day}' for day in range(day_i + 1, window + 1)]
        avg_vol = df[columns].mean(axis=1)
        df['PRICE'] *= (1 + df[f'RET_{day_i}']) 
        df[f'VWAP_{day_i}'] = df['PRICE'] * df[f'VOLUME_{day_i}'] / (avg_vol + 1e-8)

    columns = ['STOCK', 'DATE']
    new_features = [f'VWAP_{day}' for day in range(1, n_days + 1)]
    columns += new_features

    return (df[columns], new_features)


def compute_volume_oscillator(
        df: pd.DataFrame, 
        short_window: int = 5,
        long_window: int = 20
    ) -> tuple:
    """Compute the volume oscillator (VO)"""
    
    columns_short = [f'VOLUME_{day}' for day in range(1, short_window+1)]
    columns_long = [f'VOLUME_{day}' for day in range(1, long_window+1)]

    V_short = (
        df.loc[:, ['STOCK', 'DATE'] + columns_short]  
        .set_index(['STOCK', 'DATE'])                             
        .mean(axis=1)                            
    )

    V_long = (
        df.loc[:, ['STOCK', 'DATE'] + columns_long]  
        .set_index(['STOCK', 'DATE'])                             
        .mean(axis=1)                            
    )

    VO = 100 * (V_short - V_long) / (V_long + 1e-8)

    return (VO.reset_index(name='VO'), ['VO',])


def compute_macd_volume(df: pd.DataFrame, window_1: int = 10, window_2: int = 20) -> tuple:
    """Compute the moving average convergence divergence (MACD) for the Volume"""
    
    columns_1 = [f'VOLUME_{day}' for day in range(1, window_1+1)]
    columns_2 = [f'VOLUME_{day}' for day in range(1, window_2+1)]

    EMA_1 = (
        df.loc[:, ['STOCK', 'DATE'] + columns_1]
        .set_index(['STOCK', 'DATE'])
        .mean(axis=1)  # Compute the mean return for this window
        .ewm(span=window_1, adjust=False)  # Exponential weighting
        .mean()  # Compute the EMA
    )

    EMA_2 = (
        df.loc[:, ['STOCK', 'DATE'] + columns_2]
        .set_index(['STOCK', 'DATE'])
        .mean(axis=1)  # Compute the mean return for this window
        .ewm(span=window_2, adjust=False)  # Exponential weighting
        .mean()  # Compute the EMA
    )

    # Since values are really small they can also be scaled with a factor of 100
    MACD_VOLUME = EMA_1 - EMA_2

    # Return the ROC as a DataFrame
    return (MACD_VOLUME.reset_index(name='MACD_VOLUME'), ['MACD_VOLUME',])


def compute_volume_change_ratio(df: pd.DataFrame, lag_period: int = 1) -> tuple:
    """
    Compute the Volume Change Ratio (VROC) based on specific columns for current and lagged volumes.

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'STOCK', 'DATE', and volume columns (e.g., 'VOLUME_1', 'VOLUME_2', etc.).
        lag_period (int): Number of days to look back for previous volume (e.g., 'VOLUME_2' for lag_period=1).

    Returns:
        pd.DataFrame: DataFrame with 'STOCK', 'DATE', and 'VROC'.
    """
    df = df.copy()

    # Sort by stock and date to ensure proper ordering
    df = df.sort_values(by=['STOCK', 'DATE'])

    # Select current volume and the lagged volume based on the lag period
    df['VOLUME_LAG'] = df[f'VOLUME_{1 + lag_period}']

    # Compute the Volume Change Ratio
    df['VROC'] = (df['VOLUME_1'] - df['VOLUME_LAG']) / (df['VOLUME_LAG'] + 1e-8)

    # Return only relevant columns
    return (df[['STOCK', 'DATE', 'VROC']], ['VROC',])


def compute_pvt(df: pd.DataFrame, window: int = 20) -> tuple:
    """
    Compute the Price-Volume-Trend (PVT).

    Parameters:
        df: Input DataFrame with 'STOCK', 'DATE', and volume columns (e.g., 'VOLUME_1', 'VOLUME_2', etc.).
        window: How many steps should be rolled out.

    Returns:
        DataFrame with 'STOCK', 'DATE', and 'PVT' as well as feature list
    """
    df = df.copy()
    columns = [f'RET_{day}' for day in range(1, window+1)]
    # Sort by stock and date to ensure proper ordering
    df = df.sort_values(by=['STOCK', 'DATE'])

    df['PVT'] = 0.0

    for idx in reversed(range(1, window)):
        RET_curr = df[columns[idx-1]]
        RET_prev = df[columns[idx]]
        VOLUME_curr = df[f'VOLUME_{idx}']

        df['PVT'] += ((RET_curr - RET_prev) / (RET_prev + 1e-8)) * df[f'VOLUME_{idx}']

    return (df[['STOCK', 'DATE', 'PVT']], ['PVT',])


def compute_avg_volume(df: pd.DataFrame, window: int = 5) -> tuple:
    """
    Compute the Average Volume over a given window.

    Parameters:
        df: Input DataFrame with 'STOCK', 'DATE', and volume columns (e.g., 'VOLUME_1', 'VOLUME_2', etc.).
        window: How many steps should be rolled out. Here a short window size is more efficient: e.g. 5

    Returns:
        pd.DataFrame: DataFrame with 'STOCK', 'DATE', and 'AVG_VOLUME'.
    """
    df = df.copy()

    columns = [f'VOLUME_{day}' for day in range(1, window + 1)]
    
    AVG = (
        df.loc[:, ['STOCK', 'DATE'] + columns]
        .set_index(['STOCK', 'DATE'])
        .mean(axis=1)
    )

    return (AVG.reset_index(name='AVG_VOLUME'), ['AVG_VOLUME',])



def compute_volume_deviation(df: pd.DataFrame, window: int = 10, results: int = 5) -> tuple:
    """
    Compute the Volume Deviation for each stock over a fixed rolling window.

    Parameters:
        df: Input DataFrame with 'STOCK', 'DATE', and volume columns (e.g., 'VOLUME_1', 'VOLUME_2', etc.).
        window: Number of historical days to use for the rolling window.
        results: Number of volume deviation values to compute.

    Returns:
        pd.DataFrame: DataFrame with 'STOCK', 'DATE', and computed 'VT' columns.
    """
    # Check validity of inputs
    assert window >= results, "The window size must be greater than or equal to the result dimension."

    df = df.copy()

    # Loop to compute VT_1, VT_2, ..., VT_results
    for i in range(1, results + 1):
        # Fixed window of columns for all deviations
        columns = [f'VOLUME_{day}' for day in range(i, window + 1)]
        
        # Compute rolling average and standard deviation
        AVG = (
            df.loc[:, ['STOCK', 'DATE'] + columns]
            .set_index(['STOCK', 'DATE'])
            .mean(axis=1)
            .reset_index(drop=True)  # Reset index to match df
        )

        STD = (
            df.loc[:, ['STOCK', 'DATE'] + columns]
            .set_index(['STOCK', 'DATE'])
            .std(axis=1)
            .reset_index(drop=True)  # Reset index to match df
        )

        # Handle division by zero for standard deviation
        STD = STD.replace(0, 1)

        # Compute volume deviation for VOLUME_i
        df[f'VT_{i}'] = (df[f'VOLUME_{i}'] - AVG) / STD

    solution_columns = ['STOCK', 'DATE'] + [f'VT_{day}' for day in range(1, results + 1)]
    new_features = [f'VT_{day}' for day in range(1, results + 1)]

    return (df[solution_columns], new_features)


def compute_volume_spike_detection(
        df: pd.DataFrame, 
        window: int = 10, 
        threshold: float | None = None, 
        results: int = 5
    ) -> tuple:
    """
    Compute Volume Spike Detection.

    Parameters:
        df: Input DataFrame with 'STOCK', 'DATE', and 'VOLUME' columns.
        window: Rolling window size to compute the baseline volume.
        threshold: Threshold ratio to flag a volume spike.

    Returns:
        pd.DataFrame: DataFrame with 'STOCK', 'DATE', and 'SPIKE_DETECTED' columns.
    """
    df = df.copy()

    for i in range(1, results + 1):
        columns = [f'VOLUME_{day}' for day in range(i, window + 1)]

        df['BASELINE_VOLUME'] = (
            df.loc[:, ['STOCK', 'DATE'] + columns]
            .mean(axis=1)
            .reset_index(drop=True) # reset index to match the original dataframe
        )

        df[f'SPIKE_RATIO_{i}'] = df[f'VOLUME_{i}'] / df['BASELINE_VOLUME']

        # Compute a threshold
        if not threshold:
            threshold = df[f'SPIKE_RATIO_{i}'].mean() + df[f'SPIKE_RATIO_{i}'].std()

        df[f'SPIKE_DETECTED_{i}'] = (df[f'SPIKE_RATIO_{i}'].abs() > threshold).astype(int)

    new_features = [f'SPIKE_RATIO_{day}' for day in range(1, results + 1)] + \
        [f'SPIKE_DETECTED_{day}' for day in range(1, results + 1)]
    res_features = ['STOCK', 'DATE'] + new_features

    return (df[res_features], new_features)


def compute_relative_volume(
        df: pd.DataFrame, 
        window: int = 20,
        results: int = 5
    ) -> tuple:
    """
    Compute Volume Spike Detection.

    Parameters:
        df: Input DataFrame with 'STOCK', 'DATE', and 'VOLUME' columns.
        window: Rolling window size to compute the baseline volume.
        threshold: Threshold ratio to flag a volume spike.

    Returns:
        pd.DataFrame: DataFrame with 'STOCK', 'DATE', and 'SPIKE_DETECTED' columns.
    """
    df = df.copy()

    for i in range(1, results + 1):
        columns = [f'VOLUME_{day}' for day in range(i, window + 1)]

        df[f'AVG_VOLUME_{i}'] = (
            df.loc[:, ['STOCK', 'DATE'] + columns]
            .mean(axis=1)
            .reset_index(drop=True) # reset index to match the original dataframe
        )

        df[f'RELATIVE_VOLUME_{i}'] = df[f'VOLUME_{i}'] / df[f'AVG_VOLUME_{i}']


    new_features = [f'RELATIVE_VOLUME_{day}' for day in range(1, results + 1)] 
    res_features = ['STOCK', 'DATE'] + new_features

    return (df[res_features], new_features)
