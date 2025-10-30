import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datasets import IndexedReconstructDataset
import numpy as np


def load_adult_reconstruct_dataset(csv_path='../raw_data/adult.csv'):
    """
    Load adult dataset and construct IndexedReconstructDataset and v vector.
    Adult dataset has 14 features:
    - First 13 features are continuous features (already standardized)
    - Last feature (income) is categorical feature

    Returns:
        dataset: IndexedReconstructDataset instance
        v: np.ndarray, vector of length F, 0 indicates continuous feature, 1 indicates categorical feature
        num_classes_dict: dict, keys are categorical feature indices, values are number of classes for that feature
    """
    # Read data (no column names)
    df = pd.read_csv(csv_path, header=None)
    
    # Define continuous and categorical features by position
    num_features = len(df.columns)
    cont_indices = list(range(num_features - 1))  # All except last column are continuous features
    cat_indices = [num_features - 1]  # Last column is categorical feature

    # Continuous features already standardized, no need to process again
    # Tokenize categorical features
    label_encoder = LabelEncoder()
    df.iloc[:, cat_indices[0]] = label_encoder.fit_transform(df.iloc[:, cat_indices[0]])

    # Construct v vector: 0 indicates continuous feature, 1 indicates categorical feature
    v = np.zeros(num_features, dtype=np.int64)
    v[cat_indices] = 1

    # Construct num_classes_dict
    num_classes_dict = {
        cat_indices[0]: len(label_encoder.classes_)
    }

    dataset = IndexedReconstructDataset(df, cont_indices=cont_indices, cat_indices=cat_indices)

    return dataset, v, num_classes_dict 