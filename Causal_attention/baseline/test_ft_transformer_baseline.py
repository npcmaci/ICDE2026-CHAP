import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from tab_transformer_pytorch import FTTransformer
import logging
import time
import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.baseline_datasets import (
    load_creditcard_baseline_dataset,
    load_synthetic_baseline_dataset,
    load_adult_baseline_dataset,
    load_cardio_baseline_dataset,
    load_diamonds_baseline_dataset,
    load_diamonds_mixed_baseline_dataset,
    load_housing_baseline_dataset,
    load_elevator_baseline_dataset,
    load_housesale_baseline_dataset
)

def convert_dataset_to_numpy(dataset, v, target_idx):
    all_data = []
    all_targets = []
    for i in range(len(dataset)):
        x, y = dataset[i]
        all_data.append(x.numpy())
        all_targets.append(y.item())
    data_array = np.array(all_data)
    target_array = np.array(all_targets)
    return data_array, target_array

class TabDataset(Dataset):
    def __init__(self, data, targets, cat_idx, cont_idx, target_idx):
        self.data = data
        self.targets = targets
        self.cat_idx = cat_idx
        self.cont_idx = cont_idx
        self.target_idx = target_idx
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, idx):
        x = self.data[idx]
        if len(self.cat_idx) > 0:
            x_categ = x[self.cat_idx].astype(np.int64)
        else:
            x_categ = np.zeros(1, dtype=np.int64)
        
        if len(self.cont_idx) > 0:
            x_cont = x[self.cont_idx].astype(np.float32)
        else:
            x_cont = np.zeros(1, dtype=np.float32)
            
        y = self.targets[idx]
        return torch.tensor(x_categ), torch.tensor(x_cont), torch.tensor(y, dtype=torch.float32)

def test_ft_transformer_baseline(dataset="creditcard", d_token=64, n_heads=8, n_blocks=3, gpu_id=-1, 
                                batch_size=512, val_batch_size=1024, epochs=20, lr=1e-3, num_workers=4):
    torch.manual_seed(42)
    np.random.seed(42)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id >= 0 else 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Parameters: dataset={dataset}, d_token={d_token}, n_heads={n_heads}, n_blocks={n_blocks}, batch_size={batch_size}, epochs={epochs}, lr={lr}")

    if dataset.lower() == "creditcard":
        dataset_obj, v, num_classes_dict, target_idx = load_creditcard_baseline_dataset()
        dataset_name = "CreditCard"
    elif dataset.lower() == "synthetic":
        dataset_obj, v, num_classes_dict, target_idx = load_synthetic_baseline_dataset()
        dataset_name = "Synthetic"
    elif dataset.lower() == "adult":
        dataset_obj, v, num_classes_dict, target_idx = load_adult_baseline_dataset()
        dataset_name = "Adult"
    elif dataset.lower() == "cardio":
        dataset_obj, v, num_classes_dict, target_idx = load_cardio_baseline_dataset()
        dataset_name = "Cardio"
    elif dataset.lower() == "diamonds":
        dataset_obj, v, num_classes_dict, target_idx = load_diamonds_baseline_dataset()
        dataset_name = "Diamonds"
    elif dataset.lower() == "diamonds_mixed":
        dataset_obj, v, num_classes_dict, target_idx = load_diamonds_mixed_baseline_dataset()
        dataset_name = "DiamondsMixed"
    elif dataset.lower() == "housing":
        dataset_obj, v, num_classes_dict, target_idx = load_housing_baseline_dataset()
        dataset_name = "Housing"
    elif dataset.lower() == "elevator":
        dataset_obj, v, num_classes_dict, target_idx = load_elevator_baseline_dataset()
        dataset_name = "Elevator"
    elif dataset.lower() == "housesale":
        dataset_obj, v, num_classes_dict, target_idx = load_housesale_baseline_dataset()
        dataset_name = "Housesale"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v)

    logger.info(f"Dataset: {dataset_name}, size: {len(dataset_obj)}")
    logger.info(f"Feature types: {v}, target index: {target_idx}")

    total_size = len(dataset_obj)
    train_test_split_idx = int(0.8 * total_size)
    train_and_val_dataset = torch.utils.data.Subset(dataset_obj, range(0, train_test_split_idx))
    test_dataset = torch.utils.data.Subset(dataset_obj, range(train_test_split_idx, total_size))
    train_val_size = len(train_and_val_dataset)
    val_size = int(0.2 * train_val_size)
    train_size = train_val_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_and_val_dataset, [train_size, val_size])
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Feature indices - excluding target feature
    cont_idx = (v == 0).nonzero(as_tuple=True)[0]
    cat_idx = (v == 1).nonzero(as_tuple=True)[0]
    
    # Exclude target feature from categorical features
    cat_idx = cat_idx[cat_idx != target_idx]
    
    num_cont = len(cont_idx)
    num_cat = len(cat_idx)

    logger.info(f"Number of continuous features: {num_cont}, Number of categorical features: {num_cat}")
    logger.info(f"Continuous feature indices: {cont_idx}")
    logger.info(f"Categorical feature indices: {cat_idx}")
    logger.info(f"Target feature index: {target_idx}")

    # Count unique values for each categorical feature
    if num_cat > 0:
        cat_uniques = [num_classes_dict[idx.item()] for idx in cat_idx]
        logger.info(f"Number of unique values for categorical features: {cat_uniques}")
    else:
        cat_uniques = [2]  # If no categorical features, at least one default value is needed
        logger.info("No categorical features, using default value")

    # Determine task type
    is_regression = v[target_idx] == 0
    task_type = "regression" if is_regression else "classification"
    logger.info(f"Task type: {task_type}")

    # Convert to numpy
    train_data, train_targets = convert_dataset_to_numpy(train_dataset, v, target_idx)
    val_data, val_targets = convert_dataset_to_numpy(val_dataset, v, target_idx)
    test_data, test_targets = convert_dataset_to_numpy(test_dataset, v, target_idx)

    train_ds = TabDataset(train_data, train_targets, cat_idx, cont_idx, target_idx)
    val_ds = TabDataset(val_data, val_targets, cat_idx, cont_idx, target_idx)
    test_ds = TabDataset(test_data, test_targets, cat_idx, cont_idx, target_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    # Model
    model = FTTransformer(
        categories=tuple(cat_uniques),
        num_continuous=num_cont,
        dim=d_token,
        dim_out=1,  # Regression and classification both output 1 value
        depth=n_blocks,
        heads=n_heads,
        attn_dropout=0.1,
        ff_dropout=0.1
    ).to(device)

    # Calculate and display model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"FT-Transformer total parameters: {total_params:,}")
    logger.info(f"FT-Transformer trainable parameters: {trainable_params:,}")
    logger.info(f"FT-Transformer model size: {total_params * 4 / 1024 / 1024:.2f} MB")

    # Select loss function based on task type
    if is_regression:
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    def evaluate(loader):
        model.eval()
        all_labels, all_predictions = [], []
        infer_start = time.time()
        with torch.no_grad():
            for x_categ, x_cont, y in loader:
                x_categ, x_cont, y = x_categ.to(device), x_cont.to(device), y.to(device)
                preds = model(x_categ, x_cont).squeeze(-1)
                all_labels.append(y.cpu().numpy())
                all_predictions.append(preds.cpu().numpy())
        infer_end = time.time()
        infer_time = infer_end - infer_start
        num_samples = len(loader.dataset)
        logger.info(f"Inference total time: {infer_time:.2f} seconds, average per sample: {infer_time/num_samples:.4f} seconds")
        all_labels = np.concatenate(all_labels)
        all_predictions = np.concatenate(all_predictions)
        
        if is_regression:
            # Regression task: calculate MSE and R2
            from sklearn.metrics import mean_squared_error, r2_score
            mse = mean_squared_error(all_labels, all_predictions)
            r2 = r2_score(all_labels, all_predictions)
            return mse, r2
        else:
            # Classification task: calculate accuracy and AUC
            probs = torch.sigmoid(torch.tensor(all_predictions)).numpy()
            preds = (probs > 0.5).astype(int)
            acc = accuracy_score(all_labels, preds)
            auc = roc_auc_score(all_labels, probs)
            return acc, auc

    best_val_metric = 0
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        for x_categ, x_cont, y in train_loader:
            x_categ, x_cont, y = x_categ.to(device), x_cont.to(device), y.to(device)
            optimizer.zero_grad()
            predictions = model(x_categ, x_cont).squeeze(-1)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()
        # Training loop ends
        epoch_end = time.time()
        logger.info(f"Epoch {epoch+1}/{epochs} time: {epoch_end - epoch_start:.2f} seconds")
        
        val_metric1, val_metric2 = evaluate(val_loader)
        
        if is_regression:
            print(f"Epoch {epoch+1}: val_mse={val_metric1:.4f}, val_r2={val_metric2:.4f}")
            # For regression task, higher R2 is better
            if val_metric2 > best_val_metric:
                best_val_metric = val_metric2
                torch.save(model.state_dict(), "best_fttransformer.pth")
        else:
            print(f"Epoch {epoch+1}: val_acc={val_metric1:.4f}, val_auc={val_metric2:.4f}")
            # For classification task, higher AUC is better
            if val_metric2 > best_val_metric:
                best_val_metric = val_metric2
                torch.save(model.state_dict(), "best_fttransformer.pth")

    # Test set evaluation
    model.load_state_dict(torch.load("best_fttransformer.pth"))
    test_metric1, test_metric2 = evaluate(test_loader)
    
    if is_regression:
        print(f"Test MSE: {test_metric1:.4f}, Test R2: {test_metric2:.4f}")
    else:
        print(f"Test acc: {test_metric1:.4f}, Test AUC: {test_metric2:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FT-Transformer Baseline')
    parser.add_argument('--dataset', type=str, default='housesale',
                       choices=['creditcard', 'synthetic', 'adult', 'cardio', 'diamonds', 'diamonds_mixed', 'housing', 'elevator', 'housesale'],
                       help='Dataset selection')
    parser.add_argument('--d_token', type=int, default=64, 
                       help='Token dimension/embedding dimension')
    parser.add_argument('--n_heads', type=int, default=8, 
                       help='Number of attention heads')
    parser.add_argument('--n_blocks', type=int, default=1,
                       help='Number of Transformer blocks')
    parser.add_argument('--gpu_id', type=int, default=-1, 
                       help='GPU ID, -1 for automatic selection')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=1024, 
                       help='Validation batch size')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, 
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Update function call, pass all parameters
    test_ft_transformer_baseline(
        dataset=args.dataset,
        d_token=args.d_token,
        n_heads=args.n_heads,
        n_blocks=args.n_blocks,
        gpu_id=args.gpu_id,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.num_workers
    ) 