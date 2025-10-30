import pandas as pd
from sklearn.preprocessing import StandardScaler
from datasets import IndexedReconstructDataset
import numpy as np

def load_elevator_reconstruct_dataset(csv_path='raw_data/elevator.csv'):
    """
    Load elevator dataset, all features are continuous variables, last column is regression target.
    Returns: dataset, v, num_classes_dict
    """
    df = pd.read_csv(csv_path, header=0)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    num_features = len(df.columns)
    cont_indices = list(range(num_features))
    cat_indices = []
    scaler = StandardScaler()
    df.iloc[:, cont_indices] = scaler.fit_transform(df.iloc[:, cont_indices])
    v = np.zeros(num_features, dtype=np.int64)
    num_classes_dict = {}
    dataset = IndexedReconstructDataset(df, cont_indices=cont_indices, cat_indices=cat_indices)
    return dataset, v, num_classes_dict 