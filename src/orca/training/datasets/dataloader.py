import pandas as pd
import numpy as np

def train_val_test_dataset(result_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Utility function to split a dataframe into train, validation, and test datasets based on the geometry's dataset class.
    The splitting is done at the geometry level to avoid data leakage when using frequency as a feature (i.e., same geometry at different frequencies).

    Args:
        result_df (pd.DataFrame): The input dataframe containing the simulation results and geometry information.
    """
    # Shuffle the dataframe
    shuffled_df = result_df.sample(frac=1, random_state=11)

    # Define split points
    train_end = int(0.7 * len(shuffled_df))
    val_end = int(0.85 * len(shuffled_df))

    # Split using iloc to ensure we return DataFrames
    train_df = shuffled_df.iloc[:train_end]
    val_df = shuffled_df.iloc[train_end:val_end]
    test_df = shuffled_df.iloc[val_end:]

    return train_df, val_df, test_df