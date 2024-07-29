"""
This file defines the datasets to be used in the experiments

format: [name, [features to be removed], output]]
"""

dataset_setup_list = [
    ['KPI-KQI', [], 'Service'],  # 165 x 14
    ['UNAC', ['file'], 'output'],  # 389 x 23
    ['NSR', [], 'slice Type'],  # 31583 x 17
    ['DeepSlice', ['no'], 'slice Type'],  # 63167 x 10
]