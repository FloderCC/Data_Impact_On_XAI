"""
Script to run the experiment.
"""
import gc
import os

import pandas as pd
import tensorflow as tf
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from src.dataset_setup import dataset_setup_list
from src.dataset_utils import get_dataset_sample, preprocess_dataset, dataset_description_header, \
    describe_dataset_using_pymfe
from src.model_utils import create_models
from src.xai_methods import *
import warnings
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning

# Ignore warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
# Ignore warnings of type UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
# Ignore warnings of type ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# Ignore warnings of type FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)
# Ignore warnings of type UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

xai_algorithms = {
    'PI': permutation_importance_explanation,
    'SHAP': shap_explanation,
    'LIME': lime_explanation
}

seeds = [3, 5, 7, 11, 13, 17, 42]

dataset_percentage_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]

test_size = 0.3

results_header = ['Seed', 'Dataset', 'Sample Size'] \
                 + dataset_description_header \
                 + ['Model', 'Model Parameters', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'MCC',
                    'XAI', 'Test F Relevance']

number_of_models = 13

results = []

# counter
explanation_number = 1
explanations_quantity = len(dataset_setup_list) * len(xai_algorithms) * len(seeds) * len(
    dataset_percentage_sizes) * number_of_models

print(f"Total of {explanations_quantity} explanations to be done")

for seed in seeds:
    print(f"Seed: {seed}")
    # /// Setup begin \\\
    global_random_seed = seed
    np.random.seed(global_random_seed)
    tf.random.set_seed(global_random_seed)

    for dataset_setup in dataset_setup_list:

        dataset_name = dataset_setup[0]
        useful_columns = dataset_setup[1]
        class_name = dataset_setup[2]

        # loading the dataset
        dataset_folder = f"./datasets/{dataset_name}"
        full_df = pd.read_csv(
            f"{dataset_folder}/{[file for file in os.listdir(dataset_folder) if file.endswith('.csv')][0]}")

        if len(useful_columns) > 0:
            print(f"Removing columns {useful_columns}")
            full_df.drop(columns=useful_columns, inplace=True)

        print(f"Started execution with dataset {dataset_name} {full_df.shape}")

        # for each dataset size
        for dataset_percentage_size in dataset_percentage_sizes:
            print(
                f"\nStarted execution with dataset sample size {dataset_percentage_size} ({full_df.shape[0] * dataset_percentage_size} rows)")
            # splitting the dataset
            df = get_dataset_sample(full_df, seed, dataset_percentage_size, class_name, test_size)

            if df is None:  # because is it's too small for stratify then its discarded
                print(f"\nDiscarded dataset {dataset_name} with seed {seed} and size {dataset_percentage_size}")
                explanation_number += number_of_models
            else:

                # codify & prepare the dataset
                print("Codifying & preparing dataset ...")
                df = preprocess_dataset(df)

                # describe the dataset
                print("Describing dataset using pymfe ...")
                raw_dataset_description = describe_dataset_using_pymfe(df, class_name)

                # processing the class_name column with LabelEncoder to ensure labels starts from 0
                from sklearn.preprocessing import LabelEncoder

                df[class_name] = LabelEncoder().fit_transform(df[class_name])

                # splitting features & label
                X = df.drop(dataset_setup[2], axis=1)
                y = df[dataset_setup[2]]

                # encoding Y to make it processable with DNN models
                y_encoded = pd.get_dummies(y)

                # splitting the dataset in train and test
                x_train, x_test, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=test_size,
                                                                                    random_state=seed,
                                                                                    stratify=y_encoded)
                # parsing y_test to a multiclass target
                y_test = tf.argmax(y_test_encoded, axis=1).numpy()

                print(f"Tuning the {number_of_models} ML models with GA")
                tuned_models = create_models(seed, x_train, y_train_encoded)

                for (model_name, model, model_parameters_desc) in tuned_models:
                    print(f" - Explaining Model {explanation_number} / {explanations_quantity}. Type: {model_name}")

                    is_dnn = model_name == "DNN"
                    if is_dnn:
                        model.set_enabled_codified_predict(True)
                        print("   - Setting codified predict to True")

                    # testing
                    print("   - Testing ...")
                    y_pred = model.predict(x_test)

                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    mcc = matthews_corrcoef(y_test, y_pred)

                    print(f"MCC: {mcc}")

                    # getting the number of iterations
                    num_iters = 1
                    if hasattr(model, 'n_iter_'):
                        if isinstance(model.n_iter_, int):
                            num_iters = model.n_iter_
                        elif isinstance(model.n_iter_, list) or isinstance(model.n_iter_, np.ndarray):
                            num_iters = sum(model.n_iter_)

                    for xai_name, xai_algorithm in xai_algorithms.items():
                        # xai
                        print(f"\nExplanation  {explanation_number} / {explanations_quantity}")
                        print(f"Explaining test dataset with {xai_name}")
                        te_f_relevance = xai_algorithm(x_test, y_test, model)

                        # saving the results
                        results.append(
                            [seed, dataset_name, dataset_percentage_size] +
                            raw_dataset_description +
                            [model_name, model_parameters_desc, accuracy, precision, recall, f1, mcc, xai_name,
                             str(te_f_relevance.tolist())]
                        )

                        # dumping results for a file
                        results_df = pd.DataFrame(results, index=None, columns=results_header)

                        # Write to csv
                        results_df.to_csv(f"results/results.csv", index=False)

                        explanation_number += 1

                # cleaning up
                del x_train, x_test, y_train_encoded, y_test_encoded, y_pred, tuned_models
                gc.collect()
