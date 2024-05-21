"""File description:

Script to run the experiment.
"""

import os

import keras
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.dataset_utils import get_dataset_sample, describe_raw_dataset, preprocess_dataset, describe_codified_dataset
from src.dnn_models import *
from src.xai_methods import *

os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning

# Ignore warnings of type UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
# Ignore warnings of type ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# Ignore warnings of type FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)
# Ignore warnings of type UserWarning
warnings.filterwarnings("ignore", category=UserWarning)


# format: [name, [features to be removed], output]]
dataset_setup_list = [
    ['KPI-KQI', [], 'Service'],  # 165 x 14 yap
    ['UAC', ['file'], 'output'],  # 389 x 23 yap
    ['IoT-APD', ['second'], 'label'],  # 10845 x 17
    ['DeepSlice', ['no'], 'slice Type'],  # 63167 x 10
]

xai_algorithms = {
    'Permutation Importance': permutation_importance_explanation,
    'SHAP': shap_explanation,
    'LIME (ALL)': lime_explanation_all,
    # 'LIME (ALL)': lime_explanation_all_parallel
}

seeds = [3, 5, 7, 11, 13, 17, 42]
# seeds = [3]

dataset_percentage_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
# dataset_percentage_sizes = [1.0]

test_size = 0.3

#Running seed 3 DNNs. Then start all by seed 5

results_header = ['Seed',
                  'Dataset',
                  'Sample Size',
                  'Number of training samples',
                  'Number of testing samples',
                  'Number of features',
                  'Class Imbalance Ratio',
                  'Gini Impurity',
                  'Entropy',
                  'Completeness',
                  'Consistency',
                  'Uniqueness',

                  'Redundancy (avg)',
                  'Redundancy (std)',
                  'Avg of features\' avg',
                  'Std of features\' avg',
                  'Avg of features\' std',
                  'Std of features\' std',

                  'Model',
                  'MCC',
                  'XAI',
                  'Test F Relevance',
                  ]

results = []

# count
explanation_number = 1
explanations_quantity = len(dataset_setup_list) * len(xai_algorithms) * len(seeds) * len(dataset_percentage_sizes) * 14

print(explanations_quantity)

exit(2)

for seed in seeds:
    print(f"Seed: {seed}")
    # /// Setup begin \\\
    global_random_seed = seed
    np.random.seed(global_random_seed)
    tf.random.set_seed(global_random_seed)
    tf.config.set_visible_devices([], 'GPU')
    keras.utils.disable_interactive_logging()

    models = {
        # # Shallow
        'DT': DecisionTreeClassifier(random_state=global_random_seed),
        'KNN': KNeighborsClassifier(),
        'MLP': MLPClassifier(random_state=global_random_seed),
        'SGD': SGDClassifier(loss="modified_huber", random_state=global_random_seed),
        'GNB': GaussianNB(),
        'BNB': BernoulliNB(),
        'LR': LogisticRegression(random_state=global_random_seed),
        'SVC': SVC(random_state=global_random_seed, probability=True),

        # Voting based/ensemble
        'RF': RandomForestClassifier(random_state=global_random_seed),
        'VC': VotingClassifier(estimators=[('DT', DecisionTreeClassifier(random_state=global_random_seed)), ('GNB', GaussianNB()), ('SGD', SGDClassifier(loss="modified_huber", random_state=global_random_seed))], voting='soft'),
        'BC': BaggingClassifier(DecisionTreeClassifier(random_state=global_random_seed), random_state=global_random_seed),
        'ABC': AdaBoostClassifier(DecisionTreeClassifier(random_state=global_random_seed), random_state=global_random_seed),

        # DNN
        'DNN1': DNNClassifier1(),
        'DNN2': DNNClassifier2(),
    }
    # \\\ Setup end ///

    # aux variable for saving the features weights
    results_weights = {}
    for dataset_setup in dataset_setup_list:

        dataset_name = dataset_setup[0]
        class_name = dataset_setup[2]

        # loading the dataset
        dataset_folder = f"./datasets/{dataset_name}"
        full_df = pd.read_csv(f"{dataset_folder}/{[file for file in os.listdir(dataset_folder) if file.endswith('.csv')][0]}")

        print(f"Started execution with dataset {dataset_name} {full_df.shape}")

        # for each dataset size
        for dataset_percentage_size in dataset_percentage_sizes:
            print(
                f"\nStarted execution with dataset sample size {dataset_percentage_size} ({full_df.shape[0] * dataset_percentage_size} rows)")
            # splitting the dataset
            df = get_dataset_sample(full_df, seed, dataset_percentage_size, class_name, test_size)

            if df is None:  # because is it's too small for stratify then its discarded
                print(f"\nDiscarded dataset {dataset_name} with seed {seed} and size {dataset_percentage_size}")
            else:
                # describe the raw dataset
                print("Describing raw dataset ...")
                raw_dataset_description = describe_raw_dataset(df, class_name, test_size)

                # codify & prepare the dataset
                print("Codifying & preparing dataset ...")
                df = preprocess_dataset(df)

                # describe the codified dataset
                print("Describing codified dataset ...")
                codified_dataset_description = describe_codified_dataset(df, class_name)

                # splitting features & label
                X = df.drop(dataset_setup[2], axis=1)
                y = df[dataset_setup[2]]

                # encoding Y to make it processable with DNN models
                y = pd.get_dummies(y)

                # encoding Y to make it processable with DNN models
                y_encoded = pd.get_dummies(y)

                # splitting the dataset in train and test
                x_train, x_test, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=test_size,
                                                                                    random_state=seed,
                                                                                    stratify=y_encoded)
                # parsing y_test to a multiclass target
                y_test = y_test_encoded.idxmax(axis=1)

                results_weights[dataset_name] = {}
                for model_name, model in models.items():
                    print(f"\nDataset {dataset_name}. Model: {model_name}")

                    is_dnn = model_name.startswith("DNN")
                    if is_dnn:
                        y_train = y_train_encoded
                        # building the tf models in case of DNN
                        model.build(x_train.shape[1], y_train.shape[1])
                    else:
                        # parsing y_train to a multiclass target if the model is not DNN
                        y_train = y_train_encoded.idxmax(axis=1)

                    # training
                    print("Training")
                    model.fit(x_train, y_train)

                    # testing
                    print("Testing")
                    y_pred = model.predict(x_test)

                    if is_dnn:
                        y_train = y_train_encoded.idxmax(axis=1)

                    mcc = matthews_corrcoef(y_test, y_pred)

                    print(f"MCC: {mcc}")

                    results_weights[dataset_name][model_name] = {}
                    for xai_name, xai_algorithm in xai_algorithms.items():
                        if is_dnn and xai_name == 'Permutation Importance':
                            y_train_to_explain = y_train_encoded
                            y_test_to_explain = y_test_encoded
                            model.set_enabled_codified_predict(False)
                        else:
                            y_train_to_explain = y_train
                            y_test_to_explain = y_test

                        # xai
                        print(f"\nExplanation  {explanation_number} / {explanations_quantity}")
                        print(f"Explaining test dataset with {xai_name}")
                        te_f_relevance = xai_algorithm(x_test, y_test_to_explain, model)

                        # print(f"F Relevance: {f_relevance}")
                        print(f"F Relevance computed")

                        # restoring the codified prediction mode
                        if is_dnn and xai_name == 'Permutation Importance':
                            model.set_enabled_codified_predict(True)

                        # saving the results
                        results.append(
                            [seed, dataset_name, dataset_percentage_size] +
                            raw_dataset_description +
                            codified_dataset_description +
                            [model_name, mcc, xai_name, str(te_f_relevance.tolist())]
                        )

                        # dumping results for a file
                        results_df = pd.DataFrame(results, index=None, columns=results_header)

                        # Write to csv
                        results_df.to_csv(f'results/results.csv', index=False)

                        explanation_number += 1

