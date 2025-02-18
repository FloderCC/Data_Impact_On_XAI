# Data Impact On XAI

### Repository of the work entitled "Understanding the Influence of Data Characteristics on Explainable AI"



[![DOI](https://zenodo.org/badge/739455853.svg)](https://doi.org/10.5281/zenodo.10968682)

## Structure

This repository has the following structure:
```
├── src/datasets
├── src/plots
├── src/results
├── src/dataset_descriptors.py
├── src/dataset_setup.py
├── src/dataset_utils.py
├── src/dnn_model.py
├── src/experiment.py
├── src/model_utils.py
├── src/results_analyzer.py
├── src/xai_methods.py
```

- src/datasets/ contains the datasets and its sources.
- src/plots/ contains all plots generated by the script src/results_analyzers.py
- src/results/ contains all logs from the experimental process
- src/dataset_descriptors.py contains the dataset descriptors
- src/dataset_setup defines the datasets to be used in the experiments
- src/dataset_utils.py contains the dataset loading and preprocessing functions
- src/dnn_model.py has the Deep Neural Networks code
- src/experiment.py has the main code of the experiment
- src/model_utils.py contains auxiliary methods to create and evaluate the models
- src/results_analyzer.py contains the script for generating the plots
- src/xai_methods.py contains the XAI methods code

## Hyperparameter space explored for the ML models

| **Model**        | **Hyperparameters**                                                                                                                                                                                                                          |
|------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| LR               | C: {0.1, 1, 10}<br>penalty: {l1, l2, None}<br>class_weight: {None, balanced}                                                                                                                                                                 |
| DT               | criterion: {gini, entropy, log_loss}<br>max_depth: {None, 10, 15, 20}<br>max_features: {None, sqrt, log2}<br>class_weight: {None, balanced}                                                                                                  |
| ExtraTree        | criterion: {gini, entropy, log_loss}<br>max_features: {None, sqrt, log2}<br>max_depth: {None, 10, 15, 20}<br>class_weight: {None, balanced}                                                                                                  |
| SVM              | kernel: {poly, rbf}<br>C: {0.1, 1, 10}<br>class_weight: {None, balanced}                                                                                                                                                                     |
| GaussianNB       | var_smoothing: {1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1}                                                                                                                                                                                     |
| BernoulliNB      | alpha: {0.1, 1, 10}                                                                                                                                                                                                                          |
| RF               | criterion: {gini, entropy, log_loss}<br>max_features: {None, sqrt, log2}<br>class_weight: {None, balanced}<br>n_estimators: {25, 50, 100, 200}<br>max_depth: {None, 10, 15, 20}                                                              |
| ExtraTrees       | criterion: {gini, entropy, log_loss}<br>max_features: {None, sqrt, log2}<br>class_weight: {None, balanced}<br>n_estimators: {25, 50, 100, 200}<br>max_depth: {None, 10, 15, 20}                                                              |
| AdaBoost         | n_estimators: {25, 50, 100, 200}<br>learning_rate: {0.1, 0.5, 1}                                                                                                                                                                             |
| GradientBoosting | criterion: {friedman_mse, squared_error}<br>n_estimators: {50, 100, 200}<br>learning_rate: {0.1, 0.5, 1}<br>max_depth: {None, 10, 15, 20}<br>max_features: {None, sqrt, log2}                                                                |
| Bagging          | n_estimators: {10, 50, 100}<br>max_samples: {1.0}                                                                                                                                                                                            |
| MLP              | hidden_layer_sizes: {(50,), (100,)}<br>activation: {logistic, relu}<br>alpha: {0.0001, 0.001, 0.01}                                                                                                                                          |
| DNN              | hidden_layers_size: {16, 16, 16; 8, 16, 8; 32, 16, 8}<br>activation_function: {sigmoid, tanh, relu, silu}<br>loss_function: {categorical_crossentropy}<br>nn_optimizer: {Adam, RMSprop, Nadam}<br>epochs: {100, 500}<br>batch_size: {10, 50} |

[Summary of Best Hyperparameter Settings](src/results/best_hyperparameters.md)
