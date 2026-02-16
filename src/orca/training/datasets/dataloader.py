import pandas as pd
import numpy as np

def train_val_test_dataset(result_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Utility function to split a dataframe into train, validation, and test datasets based on the geometry's dataset class.
    The splitting is done at the geometry level to avoid data leakage when using frequency as a feature (i.e., same geometry at different frequencies).

    Args:
        result_df (pd.DataFrame): The input dataframe containing the simulation results and geometry information.
    """
    # Since you can only split once with train_test_split, we first split into train and val_test, then split val_test into val and test
    train_df, val_df, test_df = np.split(
        result_df.sample(frac=1, random_state=11),  # Shuffle the dataframe
        [int(0.7 * len(result_df)), int(0.85 * len(result_df))],  # 70% train, 15% val, 15% test
    )

    return train_df, val_df, test_df