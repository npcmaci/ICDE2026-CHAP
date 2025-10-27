import pandas as pd
from datasets import IndexedReconstructDataset
import numpy as np


def load_diamonds_reconstruct_dataset(csv_path='raw_data/diamonds.csv'):
    """
    Load diamonds dataset and construct IndexedReconstructDataset and v vector.
    Diamonds dataset has 10 features:
    - All 10 features are continuous features (already standardized)

    Returns:
        dataset: IndexedReconstructDataset instance
        v: np.ndarray, vector of length F, 0 for continuous features, 1 for categorical features
        num_classes_dict: dict, keys are categorical feature indices, values are number of classes
    """
    df = pd.read_csv(csv_path, header=None)
    
    num_features = len(df.columns)
    cont_indices = list(range(num_features - 1))
    cat_indices = []

    v = np.zeros(num_features, dtype=np.int64)

    num_classes_dict = {}

    dataset = IndexedReconstructDataset(df, cont_indices=cont_indices, cat_indices=cat_indices)

    return dataset, v, num_classes_dict 