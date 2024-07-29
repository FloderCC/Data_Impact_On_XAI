"""
This file contains auxiliary methods to create and evaluate the models
"""

import copy
import gc
import random
import time

import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn_genetic import GASearchCV, ConsecutiveStopping, ThresholdStopping
from sklearn_genetic.callbacks import ProgressBar
from sklearn_genetic.callbacks.base import BaseCallback
from sklearn_genetic.space import Categorical

from src.dnn_model import DNNClassifier


def reset_setup(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# defining a new scorer using MCC metric for DNNs
def custom_mcc(y_true, y_pred):
    # Decode one-hot encoded predictions
    y_true_decoded = tf.argmax(y_true, axis=1)
    y_pred_decoded = tf.argmax(y_pred, axis=1)
    # Compute MCC
    mcc = matthews_corrcoef(y_true_decoded, y_pred_decoded)
    return mcc


sklearn_mcc_for_dnns = make_scorer(custom_mcc, greater_is_better=True)
sklearn_mcc = make_scorer(matthews_corrcoef, greater_is_better=True)


class GarbageCollector(BaseCallback):
    """
    Callback that calls the garbage collector after each generation
    """

    def on_step(self, record=None, logbook=None, estimator=None):
        gc.collect()
        print("Garbage collector called on on_step")


def create_models(global_random_seed, x_train,
                  y_train_encoded) -> list:  #list with returns model_name, model, parameters_desc
    models = {
        'LR': LogisticRegression(random_state=global_random_seed),
        # 'Ridge': RidgeClassifier(random_state=global_random_seed),
        # 'SGD': SGDClassifier(random_state=global_random_seed),
        # 'Perceptron': Perceptron(random_state=global_random_seed),
        'DT': DecisionTreeClassifier(random_state=global_random_seed),
        'ExtraTree': ExtraTreeClassifier(random_state=global_random_seed),
        'SVM': SVC(random_state=global_random_seed, probability=True),
        # 'LinearSVC': LinearSVC(random_state=global_random_seed),
        'GaussianNB': GaussianNB(),
        'BernoulliNB': BernoulliNB(),
        'RF': RandomForestClassifier(random_state=global_random_seed),
        'ExtraTrees': ExtraTreesClassifier(random_state=global_random_seed),
        'AdaBoost': AdaBoostClassifier(random_state=global_random_seed),
        'GradientBoosting': GradientBoostingClassifier(random_state=global_random_seed),
        'Bagging': BaggingClassifier(random_state=global_random_seed),
        'MLP': MLPClassifier(random_state=global_random_seed),
        'DNN': DNNClassifier(random_state=global_random_seed, num_classes=y_train_encoded.shape[1],
                             num_features=x_train.shape[1]),
        # 'DNN_T': DNNClassifierV2(random_state=global_random_seed, module__num_classes=y_train_encoded.shape[1], module__num_features=x_train.shape[1])
    }

    # Define parameter grids for each classifier
    param_grids = {
        'LR': {'solver': ['saga'],
                               'C': [0.1, 1, 10],
                               'penalty': ['l1', 'l2', None],
                               # penalty elasticnet removed because ValueError: l1_ratio must be specified when penalty is elasticnet
                               'class_weight': [None, 'balanced']},

        'Ridge': {'alpha': [0.1, 1, 10],
                  'class_weight': [None, 'balanced']},

        'SGD': {'loss': ['hinge', 'log_loss', 'squared_hinge'],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'class_weight': [None, 'balanced']},

        'Perceptron': {'penalty': [None, 'l1', 'l2', 'elasticnet'],
                       'alpha': [0.0001, 0.001, 0.01, 0.1],
                       'class_weight': [None, 'balanced']},

        'DT': {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [None, 10, 15, 20],
            'max_features': [None, 'sqrt', 'log2'],
            'class_weight': [None, 'balanced']},

        'ExtraTree': {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_features': [None, 'sqrt', 'log2'],
            'max_depth': [None, 10, 15, 20],
            'class_weight': [None, 'balanced']},

        'SVM': {'kernel': ['poly', 'rbf'],
                'C': [0.1, 1, 10],
                'class_weight': [None, 'balanced']},

        'LinearSVC': {'C': [0.1, 1, 10],
                      'penalty': ['l2'],
                      # penalty l1 removed because ValueError: Unsupported set of arguments: The combination of penalty='l1' and loss='squared_hinge' are not supported when dual=True, Parameters: penalty='l1', loss='squared_hinge', dual=True
                      'class_weight': [None, 'balanced']},

        'GaussianNB': {'var_smoothing': [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1],
                       'priors': [None]},

        'BernoulliNB': {'alpha': [0.1, 1, 10],
                        'fit_prior': [True]},

        'RF': {'criterion': ['gini', 'entropy', 'log_loss'],
               'max_features': [None, 'sqrt', 'log2'],
               'class_weight': [None, 'balanced'],
               'n_estimators': [25, 50, 100, 200],
               'max_depth': [None, 10, 15, 20]},

        'ExtraTrees': {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_features': [None, 'sqrt', 'log2'],
            'class_weight': [None, 'balanced'],
            'n_estimators': [25, 50, 100, 200],
            'max_depth': [None, 10, 15, 20]},

        'AdaBoost': {'n_estimators': [25, 50, 100, 200],
                     'learning_rate': [0.1, 0.5, 1]},

        'GradientBoosting': {
            'criterion': ['friedman_mse', 'squared_error'],
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.1, 0.5, 1],
            'max_depth': [None, 10, 15, 20],
            'max_features': [None, 'sqrt', 'log2']},

        'Bagging': {'n_estimators': [10, 50, 100],
                    'max_samples': [1.0]},

        'MLP': {'hidden_layer_sizes': [(50,), (100,), ],
                'activation': ['logistic', 'relu'],
                'alpha': [0.0001, 0.001, 0.01]},

        'DNN': {'hidden_layers_size': ["16, 16, 16", "8, 16, 8", "32, 16, 8"],
                'activation_function': ['sigmoid', 'tanh', 'relu', 'silu'],
                'loss_function': ['categorical_crossentropy'],
                'nn_optimizer': ['Adam', 'RMSprop', 'Nadam'],
                'epochs': [100, 500],
                'batch_size': [10, 50]},

        'DNN_T': {'module__hidden_layers_size': ["16, 16, 16", "8, 16, 8", "32, 16, 8"],
                  'module__activation_function': ['sigmoid', 'tanh', 'relu', 'silu'],
                  'criterion': ['categorical_crossentropy'],
                  'optimizer': ['Adam', 'RMSprop', 'Nadam'],
                  'max_epochs': [100, 500],
                  'batch_size': [10, 50]}
    }

    # search for the best possible parameter combinations for each classifier
    model_configurations = []

    cv = 4  # because its 20% for testing. And 0.25 * 0.8 = 0.2 for validation

    for model_name, model in models.items():
        print(f"Searching for best parameters for {model_name}")

        # parsing param_grid to the format expected by GASearchCV
        param_grid = param_grids[model_name]
        param_grid_o = copy.deepcopy(param_grids[model_name])
        if param_grid == {}:
            model_configurations.append((model_name, model, "unique"))
        else:
            for key, value in param_grid.items():
                param_grid[key] = Categorical(choices=value, random_state=global_random_seed)

            if model_name == 'DNN':
                y_train = y_train_encoded
                scoring = sklearn_mcc_for_dnns

                tf.keras.backend.clear_session(
                    free_memory=True
                )
                print("TF memory cleaned up")

            else:
                y_train = y_train_encoded.idxmax(axis=1)
                scoring = sklearn_mcc

            tic = time.time()

            evolved_estimator = GASearchCV(model,  # this library will update this model with the best parameters
                                           cv=cv,
                                           scoring=scoring,
                                           param_grid=param_grid,
                                           population_size=20,  # 20
                                           generations=10,  # 10
                                           n_jobs=1 if model_name == 'DNN' or model_name == 'DNN_T' else -1,
                                           verbose=False
                                           )
            evolved_estimator.fit(x_train,
                                  y_train,
                                  callbacks=[ConsecutiveStopping(generations=3, metric='fitness'), ProgressBar(),
                                             GarbageCollector(),
                                             ThresholdStopping(threshold=1.0, metric='fitness_max')]
                                  )

            toc = time.time()
            print(f"GA Elapsed Time: {toc - tic} seconds. Model: {model_name}")

            best_model = evolved_estimator.best_estimator_

            best_params = evolved_estimator.best_params_

            print(f"Best score on GA: {evolved_estimator.best_score_}")

            model_configurations.append((model_name, best_model, str(best_params)[1:-1]))

            # cleaning up memory
            del evolved_estimator
            gc.collect()

    return model_configurations
