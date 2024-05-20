import multiprocessing
from functools import partial

import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
from alibi.explainers import PartialDependenceVariance
from joblib import Parallel, delayed
from shap import KernelExplainer, kmeans
from sklearn.inspection import permutation_importance
from tqdm import tqdm


def shap_explanation(X, y, model):
    background_data = kmeans(X, 10)
    explainer = KernelExplainer(model.predict, background_data)
    shap_values = explainer.shap_values(X, nsamples=100)
    return np.mean(np.abs(shap_values), axis=0)


def pd_variance_explanation(X, y, model):
    # alibi only expects numpy arrays
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    explainer = PartialDependenceVariance(model)
    explanation = explainer.explain(X, method='importance')

    return explanation.data['feature_importance'][0]


def permutation_importance_explanation(X, y, model):
    feature_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    return feature_importance.importances_mean


import warnings

warnings.filterwarnings("ignore")


def lime_explanation_all(X, y, model):
    X_ = X.values

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_,
        feature_names=X.columns,
        mode='classification'
    )

    results = []
    for i in tqdm(range(len(X_))):
        exp = explainer.explain_instance(
            data_row=X_[i],
            predict_fn=model.predict_proba,
            num_features=X_.shape[1]
        )
        importances = exp.as_map()[1]
        importances = sorted(importances, key=lambda x: x[0])  # sort by feature index
        importances = [x[1] for x in importances]
        results.append(importances)

    return np.mean(np.abs(results), axis=0)


def lime_explanation_all_parallel(X, y, model, n_jobs=-1):
    X_ = X.values

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_,
        feature_names=X.columns,
        mode='classification'
    )

    def explain_instance(i):
        exp = explainer.explain_instance(
            data_row=X_[i],
            predict_fn=model.predict_proba,
            num_features=X_.shape[1]
        )
        importances = exp.as_map()[1]
        importances = sorted(importances, key=lambda x: x[0])  # sort by feature index
        importances = [x[1] for x in importances]
        return importances

    results = Parallel(n_jobs=n_jobs)(delayed(explain_instance)(i) for i in range(len(X_)))

    return np.mean(np.abs(results), axis=0)
