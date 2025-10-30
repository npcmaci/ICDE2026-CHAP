import pandas as pd
import numpy as np
import torch
from datasets import IndexedReconstructDataset
import os
from sklearn.preprocessing import StandardScaler

def load_numerical_dag_reconstruct_dataset(
    csv_path='raw_data/numerical_dag_data_5vars.csv',
    adj_path='raw_data/numerical_dag_adj_5vars.npy',
    target_idx=None,
    standardize=True
):
    """Load numerical DAG dataset (all features continuous)."""
    df = pd.read_csv(csv_path)
    feature_cols = [col for col in df.columns if col.startswith('X')]
    env_labels = df['env_label'].values if 'env_label' in df.columns else None
    df_features = df[feature_cols].copy()
    num_features = len(feature_cols)
    if target_idx is None:
        target_idx = num_features - 1
    cont_indices = list(range(num_features))
    cat_indices = []
    if standardize:
        scaler = StandardScaler()
        df_features = pd.DataFrame(
            scaler.fit_transform(df_features),
            columns=feature_cols
        )
    v = np.zeros(num_features, dtype=np.int64)
    num_classes_dict = {}
    dataset = IndexedReconstructDataset(df_features, cont_indices=cont_indices, cat_indices=cat_indices)
    adj_matrix = None
    if os.path.exists(adj_path):
        adj_matrix = np.load(adj_path)
    
    return dataset, v, num_classes_dict, adj_matrix, env_labels

def load_numerical_dag_5vars_dataset(standardize=True):
    """Load 5-variable numerical DAG dataset."""
    return load_numerical_dag_reconstruct_dataset(
        csv_path='raw_data/numerical_dag_data_5vars.csv',
        adj_path='raw_data/numerical_dag_adj_5vars.npy',
        standardize=standardize
    )

def load_numerical_dag_10vars_dataset(standardize=True):
    """Load 10-variable numerical DAG dataset."""
    return load_numerical_dag_reconstruct_dataset(
        csv_path='raw_data/numerical_dag_data_10vars.csv',
        adj_path='raw_data/numerical_dag_adj_10vars.npy',
        standardize=standardize
    )

def load_numerical_dag_custom_dataset(num_vars=5, target_idx=None, standardize=True):
    """Load custom-size numerical DAG dataset."""
    return load_numerical_dag_reconstruct_dataset(
        csv_path=f'raw_data/numerical_dag_data_{num_vars}vars.csv',
        adj_path=f'raw_data/numerical_dag_adj_{num_vars}vars.npy',
        target_idx=target_idx,
        standardize=standardize
    )