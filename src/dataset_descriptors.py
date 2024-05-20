"""
This file contains auxiliary methods describe datasets
"""
import math
from typing import Dict

from pandas import DataFrame


# class distribution descriptors
def get_class_imbalance_ratio(df: DataFrame, class_name: str) -> float:
    """
        Calculate the class imbalance ratio for a given DataFrame and class name.

        Parameters:
        df (DataFrame): The pandas DataFrame containing the dataset.
        class_name (str): The name of the column representing the class labels.

        Returns:
        float: The class imbalance ratio.
        """
    # Calculate class counts
    class_counts = df[class_name].value_counts()

    # Calculate imbalance ratio
    imbalance_ratio = class_counts.min() / class_counts.max()

    return imbalance_ratio


def get_gini_impurity(df: DataFrame, class_name: str) -> float:
    """
    Calculate the Gini impurity for a given DataFrame and class name.

    Parameters:
    df (DataFrame): The pandas DataFrame containing the dataset.
    class_name (str): The name of the column representing the class labels.

    Returns:
    float: The Gini impurity.
    """
    # Calculate class counts
    class_counts = df[class_name].value_counts()

    # Calculate probabilities
    class_probabilities = class_counts / len(df)

    # Calculate Gini impurity
    gini_impurity = 1 - sum(class_probabilities ** 2)

    return gini_impurity


def get_entropy(df: DataFrame, class_name: str) -> float:
    """
    Calculate the entropy of a given class within a DataFrame.

    Parameters:
    - df: DataFrame containing the data.
    - class_name: Name of the column representing the class labels.

    Returns:
    - Entropy value, a float representing the degree of uncertainty or disorder in the class distribution.
    """
    # Count the occurrences of each class
    class_counts: Dict[str, int] = df[class_name].value_counts().to_dict()

    # Total number of instances
    total_instances = sum(class_counts.values())

    # Calculate entropy
    entropy = 0.0
    for count in class_counts.values():
        probability = count / total_instances
        entropy -= probability * math.log2(probability)

    return entropy


def get_number_of_samples(df: DataFrame) -> int:
    """
    Return the number of samples (rows) in the DataFrame.

    Parameters:
    - df: DataFrame containing the data.

    Returns:
    - Integer representing the number of samples in the DataFrame.
    """
    return df.shape[0]


def get_number_of_input_features(df: DataFrame) -> int:
    """
    Return the number of features (columns) in the DataFrame excluding the class column.

    Parameters:
    - df: DataFrame containing the data.

    Returns:
    - Integer representing the number of features in the DataFrame.
    """
    return df.shape[1] - 1


def get_completeness(df: DataFrame, class_name: str) -> float:
    """
    Return the percentage of missing values excluding the class column if specified.

    Parameters:
    - df: DataFrame containing the data.
    - class_name: (Optional) Name of the column representing the class labels. Defaults to None.

    Returns:
    - Completeness score, a float value between 0 and 100.
    """
    # Exclude the class column from completeness calculation if class_name is provided
    df_without_class = df.drop(columns=[class_name])

    total_cells = df_without_class.size
    missing_cells = df_without_class.isnull().sum().sum()
    completeness = ((total_cells - missing_cells) / total_cells) * 100

    return completeness


def get_redundancy(df: DataFrame, class_name: str) -> tuple:
    """
    Return avg and std of the values in the correlation matrix, excluding the class column.

    Parameters:
    - df: DataFrame containing the data.
    - class_name: Name of the column representing the class labels.

    Returns:
    - Tuple containing the average and standard deviation of the values in the correlation matrix.
    """
    # Drop the class column
    df_without_class = df.drop(columns=[class_name])

    # Remove columns with zero variance
    df_without_constant_columns = df_without_class.loc[:, df_without_class.var() != 0]

    # Calculate the correlation matrix
    correlation_matrix = df_without_constant_columns.corr()

    # Compute average and standard deviation of correlation values
    avg_correlation = correlation_matrix.values.mean()
    std_correlation = correlation_matrix.values.std()

    return avg_correlation, std_correlation


def get_consistency(df: DataFrame, class_name: str) -> float:
    """
    Return the quantity of non-contradictions within the data.

    Consistency is calculated as the ratio of non-contradictory rows to the total rows in the DataFrame.
    The possible minimum value is 0, indicating complete contradiction or inconsistency in the data,
    where every row has a different class value compared to at least one other row.
    The possible maximum value is 1, indicating perfect consistency where there are no contradictions.

    Parameters:
    - df: DataFrame containing the data.
    - class_name: Name of the column representing the class labels.

    Returns:
    - Consistency score, a float value between 0 and 1.
    """
    num_inconsistencies = 0

    # Group DataFrame by all columns except the class column
    grouped = df.groupby(df.columns.difference([class_name]).tolist())

    # Check for contradictions within each group
    for _, group in grouped:
        if group[class_name].nunique() > 1:
            num_inconsistencies += 1

    # Calculate consistency as the ratio of non-contradictory groups to total groups
    consistency = 1 - (num_inconsistencies / len(grouped))

    return consistency


def get_uniqueness(df: DataFrame) -> float:
    """
    Return the percentage of unique records in the DataFrame.

    Parameters:
    - df: DataFrame containing the data.
    - class_name: Name of the column representing the class labels.

    Returns:
    - Uniqueness score, a float value between 0 and 100.
    """
    # Calculate the percentage of unique records
    num_unique_records = df.drop_duplicates().shape[0]
    total_records = df.shape[0]
    uniqueness = (num_unique_records / total_records) * 100

    return uniqueness


def get_avg_and_std(df: DataFrame, class_name: str) -> tuple:
    """
    Compute the average (mean) and standard deviation of all data, excluding the class column.

    Parameters:
    - df: DataFrame containing the data.
    - class_name: Name of the column representing the class labels.

    Returns:
    - Tuple containing the average and standard deviation of all data, excluding the class column.
    """
    # Exclude the class column
    df_without_class = df.drop(columns=[class_name])

    # Calculate the average (mean) and standard deviation for all data
    avg_values = df_without_class.values.mean()
    std_values = df_without_class.values.std()

    return avg_values, std_values


def get_avg_and_std_of_features_avg_and_std(df: DataFrame, class_name: str) -> tuple:
    """
    Compute the average and standard deviation of each column (excluding the class column).
    Then, calculate the average and standard deviation of these values.

    Parameters:
    - df: DataFrame containing the data.
    - class_name: Name of the column representing the class labels.

    Returns:
    - Tuple containing the average and standard deviation of the averages of each column,
      and the average and standard deviation of the standard deviations of each column.
    """
    # Exclude the class column
    df_without_class = df.drop(columns=[class_name])

    # Compute the average and standard deviation of each column
    avg_per_column = df_without_class.mean()
    std_per_column = df_without_class.std()

    # Compute the average and standard deviation of the averages
    avg_of_avg = avg_per_column.mean()
    std_of_avg = avg_per_column.std()

    # Compute the average and standard deviation of the standard deviations
    avg_of_std = std_per_column.mean()
    std_of_std = std_per_column.std()

    return avg_of_avg, std_of_avg, avg_of_std, std_of_std
