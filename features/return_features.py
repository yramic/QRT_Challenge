"""Summary of functions for all return related features"""
import pandas as pd
import numpy as np

def compute_autocorrelation(df: pd.DataFrame) -> tuple:
    """Computes the Autocorrelation of Returns for lag = 1 for each stock"""
    auto_corr = []
    
    # Loop through each stock and compute autocorrelation for RET_1 and RET_2 (lag=1)
    for stock, group in df.groupby('STOCK'):
        ret_1 = group['RET_1']
        ret_2 = group['RET_2']

        var_1 = np.var(ret_1)
        var_2 = np.var(ret_2)

        if var_1 == 0 or var_2 == 0:
            autocorr_value = 0
        else:
            autocorr_value = ret_1.corr(ret_2)  # Autocorrelation for lag = 1
        
        # Append the result (stock, autocorr_value) to the list
        auto_corr.append((stock, autocorr_value))
    
    # Convert the result to a DataFrame
    auto_corr_df = pd.DataFrame(auto_corr, columns=['STOCK', 'AUTO_CORR'])
    new_features = ['AUTO_CORR',]

    return (auto_corr_df, new_features)


def compute_rolling_sharpe(df: pd.DataFrame, window: int = 5) -> tuple:
    """Computes the Rolling Sharpe Ratio for the last 'window' returns (e.g., 5 days)"""
    
    rolling_sharpe = []
    
    # Loop through each stock
    for stock, group in df.groupby('STOCK'):
        # Collect the relevant returns (e.g., RET_1 to RET_5)
        ret_columns = [f'RET_{i}' for i in range(1, window + 1)]
        returns = group[ret_columns]
        
        rolling_mean = returns.mean(axis=1)  # Mean of returns for each row (across RET_1 to RET_5)
        rolling_std = returns.std(axis=1)  # Standard deviation of returns for each row (across RET_1 to RET_5)
        
        # Compute the rolling Sharpe Ratio (assuming risk-free rate is 0)
        sharpe_ratio = rolling_mean / (rolling_std + 1e-8)
        
        # Append the computed Sharpe ratio for each stock
        rolling_sharpe.append(sharpe_ratio)
    
    # Concatenate the results and assign them back to the DataFrame
    rolling_sharpe_df = pd.concat(rolling_sharpe).reset_index(drop=True)
    new_features = ['ROLLING_SHARPE',]

    return (rolling_sharpe_df, new_features)

def compute_rsi(df: pd.DataFrame, window: int = 14) -> tuple:
    """Computes the Relative Strength Index (RSI) for each stock using a rolling window."""

    columns = [f'RET_{day}' for day in range(1, window+1)]

    # Compute the average positive gain directly
    avg_gain = (
        df.loc[:, ['STOCK', 'DATE'] + columns]  
        .set_index(['STOCK', 'DATE'])             
        .where(lambda x: x > 0)
        .fillna(0)                   
        .mean(axis=1)                            
        .sort_index() 
    )

    # Compute the average positive gain directly
    avg_loss = (
        df.loc[:, ['STOCK', 'DATE'] + columns]  
        .set_index(['STOCK', 'DATE'])             
        .where(lambda x: x < 0)
        .fillna(0)                   
        .mean(axis=1)                            
        .sort_index() 
    )

    # Compute the Relative Strength (RS) and RSI for each stock
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs + 1e-8))
    
    new_features = ['RSI',]
    # Return the computed RSI as a DataFrame with the same index as the original
    return (rsi.reset_index(drop=True), new_features)


def compute_roc(df: pd.DataFrame, window: int = 14) -> tuple:
    """Computes the Rate of Change (ROC) for each stock using a rolling window."""
    
    columns = [f'RET_{day}' for day in range(2, window+1)]

    # Compute the average return over the window
    avg_return = (
        df.loc[:, ['STOCK', 'DATE'] + columns]
        .set_index(['STOCK', 'DATE'])
        .mean(axis=1)
    )
    
    R_t = (
        df.loc[:, ['STOCK', 'DATE', 'RET_1']]
        .set_index(['STOCK', 'DATE'])['RET_1']
    )
    
    # Compute the ROC
    roc = 100 * (R_t - avg_return) / (avg_return + 1e-8)

    new_features = ['ROC',]

    # Return the ROC as a DataFrame
    return (roc.reset_index(name='ROC'), new_features)


def compute_momentum(df: pd.DataFrame, window: int = 14) -> tuple:
    """Compute the avg momentum"""
    
    columns = [f'RET_{day}' for day in range(1, window+1)]

    # Compute the average return over the window
    avg_momentum = (
        df.loc[:, ['STOCK', 'DATE'] + columns]
        .set_index(['STOCK', 'DATE'])
        .mean(axis=1)
    )

    new_features = ['MOMENTUM',]
    # Return the ROC as a DataFrame
    return (avg_momentum.reset_index(name='MOMENTUM'), new_features)


def compute_stochastic_oscillator(df: pd.DataFrame, window: int = 14) -> tuple:
    """Compute the stochastic oscillator (so)"""
    
    columns = [f'RET_{day}' for day in range(2, window+2)]

    R_max = (
        df.loc[:, ['STOCK', 'DATE'] + columns]  
        .set_index(['STOCK', 'DATE'])                             
        .max(axis=1)                            
    )

    R_min = (
        df.loc[:, ['STOCK', 'DATE'] + columns]  
        .set_index(['STOCK', 'DATE'])                             
        .min(axis=1)                            
    )

    R_t = (
        df.loc[:, ['STOCK', 'DATE', 'RET_1']]
        .set_index(['STOCK', 'DATE'])['RET_1']
    )

    SO = 100 * (R_t - R_min) / (R_max - R_min + 1e-8)

    new_features = ['SO',]
    # Return the ROC as a DataFrame
    return (SO.reset_index(name='SO'), new_features)


def compute_macd(df: pd.DataFrame, window_1: int = 10, window_2: int = 20) -> tuple:
    """Compute the moving average convergence divergence (MACD)"""
    
    columns_1 = [f'RET_{day}' for day in range(1, window_1+1)]
    columns_2 = [f'RET_{day}' for day in range(1, window_2+1)]

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
    MACD = EMA_1 - EMA_2

    new_features = ['MACD',]
    # Return the ROC as a DataFrame
    return (MACD.reset_index(name='MACD'), new_features)


def compute_golden_cross(df: pd.DataFrame, short_window: int = 5, long_window: int = 20) -> tuple:
    """Compute Golden Cross for returns using EMAs"""
    
    # Calculate the short-term EMA (e.g., 5-day)
    columns_short = [f'RET_{day}' for day in range(1, short_window + 1)]
    EMA_short = (
        df.loc[:, ['STOCK', 'DATE'] + columns_short]
        .set_index(['STOCK', 'DATE'])
        .mean(axis=1)  # Compute the mean return for this window
        .ewm(span=short_window, adjust=False)  # Exponential weighting
        .mean()  # Compute the EMA
    )

    # Calculate the long-term EMA (e.g., 20-day)
    columns_long = [f'RET_{day}' for day in range(1, long_window + 1)]
    EMA_long = (
        df.loc[:, ['STOCK', 'DATE'] + columns_long]
        .set_index(['STOCK', 'DATE'])
        .mean(axis=1) 
        .ewm(span=long_window, adjust=False)  # Exponential weighting
        .mean() 
    )

    # Compute Golden Cross: short-term EMA crossing above long-term EMA
    golden_cross = (EMA_short > EMA_long) & (EMA_short.shift(1) <= EMA_long.shift(1))
    new_features = ['GOLDEN_CROSS',]

    return (golden_cross.reset_index(drop=True), new_features)


def compute_bollinger_bands(df: pd.DataFrame, n: int = 20, K: int = 2) -> tuple:
    """Compute the lower, middle and upper bollinger band.

    Args:
        n: Number of datapoints used to compute the SMA
        K: Number of standard deviations used for the computation
    """
    
    # Calculate the returns columns (RETs)
    columns = [f'RET_{day}' for day in range(1, n + 1)]
    
    # Calculate the SMA (Simple Moving Average)
    SMA = (
        df.loc[:, ['STOCK', 'DATE'] + columns]
        .set_index(['STOCK', 'DATE'])
        .mean(axis=1)  # Compute the mean return for this window
        .reset_index(name='SMA')
    )

    # Calculate the STD (Standard Deviation)
    STD = (
        df.loc[:, ['STOCK', 'DATE'] + columns]
        .set_index(['STOCK', 'DATE'])
        .std(axis=1)
        .reset_index(name='STD')
    )

    # Compute Upper and Lower Bands
    Upper = SMA['SMA'] + K * STD['STD']
    Lower = SMA['SMA'] - K * STD['STD']
    
    # Compute the Distance between Upper and Lower Bands
    Distance = Upper - Lower
    
    # Combine all the components into a single DataFrame
    bollinger_df = SMA[['STOCK', 'DATE']].copy()
    bollinger_df['UPPER_BAND'] = Upper
    bollinger_df['LOWER_BAND'] = Lower
    bollinger_df['DISTANCE_BAND'] = Distance

    new_features = ['UPPER_BAND', 'LOWER_BAND', 'DISTANCE_BAND']
    return (bollinger_df, new_features)


def compute_cumsum(df: pd.DataFrame, n: int = 5) -> tuple:
    """Compute the cummulative return
    
    Args:
        - df: The dataframe where all the RET_i values are stored
        - n: How many RET values are considered for the cummulative sum
    """
    columns = [f'RET_{day}' for day in range(1, n + 1)]
    cumsum = (
        df.loc[:, ['STOCK', 'DATE'] + columns]
        .set_index(['STOCK', 'DATE'])
        .apply(lambda x: (1 + x).prod() - 1, axis=1)
        .reset_index(name='CUMMULATIVE_RETURN')
    )
    new_features = ['CUMMULATIVE_RETURN',]

    return (cumsum, new_features)


def compute_mfi(df: pd.DataFrame, window: int = 20) -> tuple:
    """
    Compute the Money Flow Index (MFI) for the given DataFrame.

    Args:
    df (pd.DataFrame): Input DataFrame with 'STOCK', 'DATE', 'RET_1' to 'RET_20', and 'VOLUME_1' to 'VOLUME_20'.
    window (int): The rolling window size for calculating the MFI.

    Returns:
        DataFrame with the computed MFI values as well as a list of new feature names.
    """
    df = df.copy()

    # Compute typical price and raw money flow
    df['typical_price'] = df[[f'RET_{day}' for day in range(1, 21)]].mean(axis=1) * df[[f'VOLUME_{day}' for day in range(1, 21)]].mean(axis=1)

    # Compute differences in typical price grouped by 'STOCK'
    df['typical_price_diff'] = df.groupby('STOCK')['typical_price'].diff()

    # Calculate positive and negative money flows
    df['positive_mf'] = np.where(df['typical_price_diff'] > 0, df['typical_price'], 0)
    df['negative_mf'] = np.where(df['typical_price_diff'] < 0, df['typical_price'], 0)

    # Calculate rolling sums of positive and negative money flows
    df['rolling_positive_mf'] = df.groupby('STOCK')['positive_mf'].transform(lambda x: x.rolling(window, min_periods=1).sum())
    df['rolling_negative_mf'] = df.groupby('STOCK')['negative_mf'].transform(lambda x: x.rolling(window, min_periods=1).sum())

    # Compute Money Flow Ratio (MFR)
    df['mfr'] = df['rolling_positive_mf'] / df['rolling_negative_mf']

    # Compute Money Flow Index (MFI)
    df['MFI'] = 100 - (100 / (1 + df['mfr']))

    # Handle potential NaN values by filling with medians grouped by 'STOCK'
    def safe_fillna_with_median(group):
        # Check if the group contains only NaN values
        if group.isna().all():
            # If all values are NaN, replace them with a default value (e.g., 0)
            return group.fillna(0)
        else:
            # Otherwise, fill NaN values with the median of the group
            return group.fillna(group.median())

    df['MFI'] = df.groupby('STOCK')['MFI'].transform(safe_fillna_with_median)
    new_features = ['MFI', ]

    # Return DataFrame with relevant columns
    return (df[['STOCK', 'DATE', 'MFI']], new_features)


def compute_adl(df: pd.DataFrame, window: int = 20) -> tuple:
    """
    Calculate ADL for each stock.
    
    Args:
    df: pd.DataFrame, containing 'STOCK', 'DATE', 'RET_i' (i from 1 to 5) and 'VOLUME_i' columns.
    
    Returns:
    pd.DataFrame with ADL values for each stock.
    list of new feature names.
    """
    df = df.copy()
    epsilon = 1e-8  # Small value to prevent division by zero

    # Calculate High, Low, and Close as the rolling max, min, and close
    df['HIGH'] = df.groupby('STOCK')['RET_1'].transform(lambda x: x.rolling(window=window, min_periods=1).max())
    df['LOW'] = df.groupby('STOCK')['RET_1'].transform(lambda x: x.rolling(window=window, min_periods=1).min())
    
    # Calculate Money Flow Multiplier
    df['MFR'] = ((df['RET_1'] - df['LOW']) - (df['HIGH'] - df['RET_1'])) / (df['HIGH'] - df['LOW'] + epsilon)

    # Calculate Money Flow Volume
    df['MFV'] = df['MFR'] * df['VOLUME_1']

    # Fill NaNs with the median of the Money Flow Volume column
    df['MFV'] = df.groupby('STOCK')['MFV'].transform(lambda x: x.fillna(x.median()))

    # Calculate ADL
    df['ADL'] = df.groupby('STOCK')['MFV'].cumsum()

    new_features = ['ADL',]
    
    return (df[['STOCK', 'DATE', 'ADL']], new_features)


def compute_atr(df: pd.DataFrame, window: int = 14) -> tuple:
    """Calculate ATR per stock."""
    df = df.copy()

    # Calculate High, Low, and Close
    df['HIGH'] = df.groupby('STOCK')['RET_1'].transform(lambda x: x.rolling(window=window, min_periods=1).max())
    df['LOW'] = df.groupby('STOCK')['RET_1'].transform(lambda x: x.rolling(window=window, min_periods=1).min())
    df['CLOSE'] = df['RET_1']

    # Calculate True Range (TR)
    df['TR'] = np.maximum(
        df['HIGH'] - df['LOW'],
        np.abs(df['HIGH'] - df['CLOSE'].shift(1)), 
        np.abs(df['LOW'] - df['CLOSE'].shift(1)) 
    )

    # Fill NaNs in TR with the median for each stock group
    df['TR'] = df.groupby('STOCK')['TR'].transform(lambda x: x.fillna(x.median()))

    # Calculate Average True Range (ATR)
    df['ATR'] = df.groupby('STOCK')['TR'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())

    # Fill NaNs with the median for each stock group
    df['ATR'] = df.groupby('STOCK')['ATR'].transform(lambda x: x.fillna(x.median()))

    if 'SECTOR' in df.columns:
        df['ATR'] = df.groupby('SECTOR')['ATR'].transform(lambda x: x.fillna(x.median()))

    new_features = ['ATR',]

    return (df[['STOCK', 'DATE', 'ATR']], new_features)
