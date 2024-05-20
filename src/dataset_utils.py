"""
This file contains auxiliary methods to simplify the understanding of the experiment file
"""

import numpy as np

# from src.dataset_descriptors import *
from dataset_descriptors import *


def get_dataset_sample(df: DataFrame, seed: int, sample_percent: float, class_name: str, test_size: float) -> DataFrame:
    # for train test split is necessary to ensure the least populated class in y has only > 1 member
    # and the test set has at least 1 member

    # Calculate the number of samples for the least populated class after sampling
    min_class_samples_after_sampling = df[class_name].value_counts().min() * sample_percent

    # Calculate the number of samples for the least populated class after sampling
    min_class_samples_after_sampling_and_split = min_class_samples_after_sampling * test_size

    # If the least populated class will have at least 2 members after sampling, perform the sampling
    if min_class_samples_after_sampling_and_split >= 1:
        return df.groupby(class_name, group_keys=False).apply(
            lambda x: x.sample(frac=sample_percent, random_state=seed)).sort_index()
    else:
        return None


def preprocess_dataset(df: DataFrame) -> DataFrame:
    # replacing infinite values by the maximum allowed value
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if np.any(np.isinf(df[numeric_columns])):
        print(" - Replacing infinite values by the maximum allowed value")
        df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)

    # encoding all no numerical columns
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for column in df.columns:
        if not df[column].dtype.kind in ['i', 'f']:
            print(f" - Encoding column {column}")
            df[column] = le.fit_transform(df[column].astype(str))

    # replacing missing values by mean
    if df.isnull().any().any():
        print(" - Replacing missing values by mean")
        df.fillna(df.mean(), inplace=True)

    return df


def describe_raw_dataset(df: DataFrame, class_name: str, test_size: float) -> list:
    number_of_samples = get_number_of_samples(df)
    number_of_samples_testing = round(number_of_samples * test_size)
    number_of_samples_training = number_of_samples - number_of_samples_testing
    return [
        number_of_samples_training,
        number_of_samples_testing,
        get_number_of_input_features(df),
        get_class_imbalance_ratio(df, class_name),
        get_gini_impurity(df, class_name),
        get_entropy(df, class_name),
        get_completeness(df, class_name),
        get_consistency(df, class_name),
        get_uniqueness(df),
    ]


def describe_codified_dataset(df: DataFrame, class_name: str) -> list:
    avg_correlation, std_correlation = get_redundancy(df, class_name)
    avg_of_avg, std_of_avg, avg_of_std, std_of_std = get_avg_and_std_of_features_avg_and_std(df, class_name)

    return [
        avg_correlation, std_correlation,
        avg_of_avg, std_of_avg, avg_of_std, std_of_std
    ]
