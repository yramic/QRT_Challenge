"""Function used to return the corresponding dataframe with new features"""
import pandas as pd

from features.return_features import (
    compute_autocorrelation,
    compute_rolling_sharpe,
    compute_rsi,
    compute_roc,
    compute_momentum,
    compute_stochastic_oscillator,
    compute_macd,
    compute_golden_cross,
    compute_bollinger_bands,
    compute_cumsum,
    compute_mfi,
    compute_adl,
    compute_atr
)

from features.volume_features import (
    compute_vpt,
    compute_obv,
    compute_vwap,
    compute_volume_oscillator,
    compute_macd_volume,
    compute_volume_change_ratio,
    compute_pvt,
    compute_avg_volume,
    compute_volume_deviation,
    compute_volume_spike_detection,
    compute_relative_volume
)

from features.statistical_features import (
    compute_conditional_features,
    compute_volatility
)

def generate_return_volume_features(
        df: pd.DataFrame, 
        selected_features: list | str = 'all'
    ) -> tuple:
    """
    Generate a DataFrame with selected new features.

    Args:
        df: Input DataFrame with stock data.
        selected_features: List of feature function names to apply 
            or just the string 'all' if every feature should be added

    Returns:
        pd.DataFrame: Updated DataFrame with new features.
        list: List of added feature names.
    """

    # Dictionary mapping function names to actual functions
    feature_functions = {
        "autocorrelation": compute_autocorrelation,
        "rolling_sharpe": compute_rolling_sharpe,
        "rsi": compute_rsi,
        "roc": compute_roc,
        "momentum": compute_momentum,
        "stochastic_oscillator": compute_stochastic_oscillator,
        "macd": compute_macd,
        "golden_cross": compute_golden_cross,
        "bollinger_bands": compute_bollinger_bands,
        "cumsum": compute_cumsum,
        "mfi": compute_mfi,
        "adl": compute_adl,
        "atr": compute_atr,
        "vpt": compute_vpt,
        "obv": compute_obv,
        "vwap": compute_vwap,
        "volume_oscillator": compute_volume_oscillator,
        "macd_volume": compute_macd_volume,
        "volume_change_ratio": compute_volume_change_ratio,
        "pvt": compute_pvt,
        "avg_volume": compute_avg_volume,
        "volume_deviation": compute_volume_deviation,
        "volume_spike_detection": compute_volume_spike_detection,
        "relative_volume": compute_relative_volume,
    }

    if selected_features == 'all':
        selected_features = list(feature_functions.keys())

    new_features = []
    for feature in selected_features:

        if feature in feature_functions:
            feature_function = feature_functions[feature]
            
            # Compute feature
            feature_df, feature_list = feature_function(df)

            # If function returns a DataFrame, merge it
            if isinstance(feature_df, pd.DataFrame):
                if feature == 'autocorrelation':
                    df = df.merge(feature_df, on=['STOCK'], how='left')
                else:
                    df = df.merge(feature_df, on=['STOCK', 'DATE'], how='left')
            else:  # If function returns a Series, assign directly
                df[feature.upper()] = feature_df

            new_features += feature_list

        else:
            print(f"Warning: Feature '{feature}' not found in available functions.")

    return df, new_features


def generate_statistical_features(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        selected_features: str | list = 'all'
    ) -> tuple:
    """
    Generate training and testing DataFrame with selected new features.

    Args:
        trian_df: Input DataFrame with stock data.
        test_df: Test DataFrame with stock data.
        selected_features: List of feature function names to apply 
            or just the string 'all' if every feature should be added

    Returns:
        pd.DataFrame: Updated Train DataFrame with new features.
        pd.DataFrame: Updated Test DataFrame with new features.
        list: List of added feature names.
    """
    feature_functions = {
        "conditional": compute_conditional_features,
        "volatility": compute_volatility
    }

    if selected_features == 'all':
        selected_features = list(feature_functions.keys())

    new_features = []
    for feature in selected_features:

        if feature in feature_functions:
            feature_function = feature_functions[feature]
            
            # Compute feature
            train_df, test_df, feature_list = feature_function(train_df, test_df)
            new_features += feature_list

    return train_df, test_df, new_features


def generate_features(
        train_df: pd.DataFrame, 
        test_df: pd.DataFrame, 
        return_volume_features: str | list = 'all', 
        statistical_features: str | list = 'all'
    ) -> tuple:
    """
    Generate all selected features for both training and testing DataFrames.

    Args:
        train_df: Training DataFrame with stock data.
        test_df: Testing DataFrame with stock data.
        return_volume_features: List of return & volume feature names to apply 
            or 'all' to apply all of them.
        statistical_features: List of statistical feature names to apply 
            or 'all' to apply all of them.

    Returns:
        pd.DataFrame: Updated Train DataFrame with new features.
        pd.DataFrame: Updated Test DataFrame with new features.
        list: List of all added feature names.
    """

    # Generate return & volume-based features
    train_df, return_volume_feature_list = generate_return_volume_features(
        train_df, return_volume_features
    )

    test_df, _ = generate_return_volume_features(
        test_df, return_volume_features
    )

    # Generate statistical features
    train_df, test_df, statistical_feature_list = generate_statistical_features(
        train_df, test_df, statistical_features
    )

    # Combine feature lists
    all_features = return_volume_feature_list + statistical_feature_list

    return train_df, test_df, all_features