import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datasets import IndexedReconstructDataset
import numpy as np


def load_creditcard_reconstruct_dataset(csv_path='raw_data/creditcard.csv'):
    """
    Load creditcard dataset and construct IndexedReconstructDataset and v vector.
    Credit card dataset has 31 features:
    - First 30 features (Time, V1-V28, Amount) are continuous features
    - Last feature (Class) is categorical feature

    Returns:
        dataset: IndexedReconstructDataset instance
        v: np.ndarray, vector of length F, 0 for continuous features, 1 for categorical features
        num_classes_dict: dict, keys are categorical feature indices, values are number of classes
    """
    df = pd.read_csv(csv_path)
    
    num_features = len(df.columns)
    cont_indices = list(range(num_features - 1))
    cat_indices = []

    scaler = StandardScaler()
    df.iloc[:, cont_indices] = scaler.fit_transform(df.iloc[:, cont_indices])

    v = np.zeros(num_features, dtype=np.int64)

    num_classes_dict = {}

    dataset = IndexedReconstructDataset(df, cont_indices=cont_indices, cat_indices=cat_indices)

    return dataset, v, num_classes_dict 