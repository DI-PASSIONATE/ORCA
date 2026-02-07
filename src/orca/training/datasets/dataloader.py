from sklearn.model_selection import train_test_split
import pandas as pd

from orca.geometry.base_geometry import BaseGeometry

def train_val_test_dataset(result_df: pd.DataFrame, geometry: BaseGeometry, result_dir: str):
    """
    Utility function to split a dataframe into train, validation, and test datasets based on the geometry's dataset class.
    The splitting is done at the geometry level to avoid data leakage when using frequency as a feature (i.e., same geometry at different frequencies).
    """
    # Since you can only split once with train_test_split, we first split into train and val_test, then split val_test into val and test
    train_df, val_test_df = train_test_split(result_df, test_size=0.3, random_state=11)
    val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=11)

    train_dataset = geometry.dataset.new_split(directory=result_dir, data_df=train_df)
    val_dataset = geometry.dataset.new_split(directory=result_dir, data_df=val_df)
    test_dataset = geometry.dataset.new_split(directory=result_dir, data_df=test_df)

    return train_dataset, val_dataset, test_dataset