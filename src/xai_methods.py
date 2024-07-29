import lime
import lime.lime_tabular
import numpy as np
from alibi.explainers import PartialDependence, PermutationImportance
from shap import KernelExplainer, kmeans
from tqdm import tqdm


# XAI methods for any model

def shap_explanation(X, y, model):
    background_data = kmeans(X, 10)
    explainer = KernelExplainer(model.predict, background_data)
    shap_values = explainer.shap_values(X, nsamples=100)
    return np.mean(np.abs(shap_values), axis=0)


def permutation_importance_explanation(X, y, model):
    explainer = PermutationImportance(predictor=model.predict, score_fns='accuracy', feature_names=X.columns.tolist())
    explanations = explainer.explain(X=X.values, y=y, kind='difference')
    f_importance = explanations.data['feature_importance'][0]
    return np.array([f_importance['mean'] for f_importance in f_importance])


def lime_explanation(X, y, model):
    # works only for probabilistic models
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
        importance = exp.as_map()[1]
        importance = sorted(importance, key=lambda x: x[0])  # sort by feature index
        importance = [x[1] for x in importance]
        results.append(importance)

    return np.mean(np.abs(results), axis=0)
