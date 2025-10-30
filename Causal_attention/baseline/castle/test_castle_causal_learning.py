"""CASTLE causal graph learning evaluation (minimal, module-level doc)."""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import sys
import argparse
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from CASTLE import CASTLE
from utils import random_dag, gen_data_nonlinear


def load_numerical_dag_data(csv_path, adj_path=None):
    """Load numerical DAG data and optional adjacency."""
    df = pd.read_csv(csv_path)
    feature_cols = [col for col in df.columns if col.startswith('X')]
    env_labels = df['env_label'].values if 'env_label' in df.columns else None
    data = df[feature_cols].values
    adj_matrix = None
    if adj_path and os.path.exists(adj_path):
        adj_matrix = np.load(adj_path)
    
    return data, adj_matrix, env_labels


def extract_causal_graph(castle_model, X, y, threshold=0.3):
    """Extract binary graph from CASTLE weight matrix."""
    W_est = castle_model.pred_W(X, y)
    causal_graph = (np.abs(W_est) >= threshold).astype(float)
    np.fill_diagonal(causal_graph, 0)
    
    return causal_graph, W_est


def compute_causal_metrics(pred_graph, true_graph):
    """Compute standard metrics for causal graphs."""
    assert pred_graph.shape == true_graph.shape, f"shape mismatch: {pred_graph.shape} vs {true_graph.shape}"
    tp = np.sum((pred_graph == 1) & (true_graph == 1))
    fp = np.sum((pred_graph == 1) & (true_graph == 0))
    fn = np.sum((pred_graph == 0) & (true_graph == 1))
    tn = np.sum((pred_graph == 0) & (true_graph == 0))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    shd = fp + fn
    
    metrics = {
        'SHD': shd,
        'TPR': tpr,
        'FPR': fpr,
        'Precision': precision,
        'F1_Score': f1_score,
        'Accuracy': accuracy,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn
    }
    
    return metrics


def plot_graph_comparison(pred_graph, true_graph, save_path=None):
    """Plot predicted vs true graphs and their difference."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    im1 = ax1.imshow(true_graph, cmap='Blues', vmin=0, vmax=1)
    ax1.set_title('True Causal Graph')
    ax1.set_xlabel('Effect')
    ax1.set_ylabel('Cause')
    
    im2 = ax2.imshow(pred_graph, cmap='Reds', vmin=0, vmax=1)
    ax2.set_title('Predicted Causal Graph')
    ax2.set_xlabel('Effect')
    ax2.set_ylabel('Cause')
    
    diff = pred_graph - true_graph
    im3 = ax3.imshow(diff, cmap='RdBu', vmin=-1, vmax=1)
    ax3.set_title('Difference (Pred - True)')
    ax3.set_xlabel('Effect')
    ax3.set_ylabel('Cause')
    
    plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.close()


def test_castle_causal_learning(csv_path, adj_path=None, threshold=0.3, 
                               reg_lambda=1.0, reg_beta=5.0, max_steps=100,
                               prefix='castle_test'):
    """Train CASTLE and evaluate causal graph learning."""
    print(f"=== CASTLE causal graph learning ===")
    print(f"CSV: {csv_path}")
    print(f"Adj: {adj_path}")
    print(f"Threshold: {threshold}")
    
    data, true_adj, env_labels = load_numerical_dag_data(csv_path, adj_path)
    print(f"Data shape: {data.shape}")
    print(f"Env dist: {np.bincount(env_labels) if env_labels is not None else 'None'}")
    
    if true_adj is not None:
        print(f"True edges: {true_adj.sum()}")
        print(f"Adj shape: {true_adj.shape}")
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    train_size = 6000
    X_train = data_scaled[:train_size]
    X_test = data_scaled[train_size:]
    y_train = data_scaled[:train_size, -1]
    y_test = data_scaled[train_size:, -1]
    
    y_train = np.expand_dims(y_train, -1)
    y_test = np.expand_dims(y_test, -1)
    
    print(f"Train/Test: {X_train.shape} / {X_test.shape}")
    
    num_inputs = data.shape[1]
    castle = CASTLE(
        num_train=X_train.shape[0],
        num_inputs=num_inputs,
        reg_lambda=reg_lambda,
        reg_beta=reg_beta,
        w_threshold=threshold,
        ckpt_file=f'tmp_{prefix}.ckpt'
    )
    
    castle.max_steps = max_steps
    
    print("Training CASTLE...")
    start_time = time.time()
    
    try:
        castle.fit(X_train, y_train, num_inputs, X_test, y_test, X_test, y_test)
        training_time = time.time() - start_time
        print(f"Training done in {training_time:.2f}s")
    except Exception as e:
        print(f"Training error: {e}")
        print("Proceed with current state...")
        training_time = time.time() - start_time
    
    print("Extracting graph...")
    causal_graph, weight_matrix = extract_causal_graph(castle, X_train, y_train, threshold)
    
    print(f"Pred edges: {causal_graph.sum()}")
    print(f"W shape: {weight_matrix.shape}")
    print(f"W range: [{weight_matrix.min():.4f}, {weight_matrix.max():.4f}]")
    
    if true_adj is not None:
        metrics = compute_causal_metrics(causal_graph, true_adj)
        
        print("\n=== Metrics ===")
        print(f"SHD (Structural Hamming Distance): {metrics['SHD']}")
        print(f"TPR (True Positive Rate): {metrics['TPR']:.4f}")
        print(f"FPR (False Positive Rate): {metrics['FPR']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"F1 Score: {metrics['F1_Score']:.4f}")
        print(f"Accuracy: {metrics['Accuracy']:.4f}")
        
        results_dir = f"causal_graph_results_{prefix}"
        os.makedirs(results_dir, exist_ok=True)
        
        np.save(f"{results_dir}/{prefix}_predicted_graph.npy", causal_graph)
        np.save(f"{results_dir}/{prefix}_weight_matrix.npy", weight_matrix)
        
        np.save(f"{results_dir}/{prefix}_true_graph.npy", true_adj)
        
        with open(f"{results_dir}/{prefix}_causal_metrics.txt", "w") as f:
            f.write("CASTLE Causal Graph Learning Results\n")
            f.write("=" * 40 + "\n")
            f.write(f"Dataset: {csv_path}\n")
            f.write(f"Threshold: {threshold}\n")
            f.write(f"Training Time: {training_time:.2f}s\n")
            f.write(f"Data Shape: {data.shape}\n")
            f.write(f"True Edges: {true_adj.sum()}\n")
            f.write(f"Predicted Edges: {causal_graph.sum()}\n")
            f.write("\nMetrics:\n")
            f.write(f"SHD (Structural Hamming Distance): {metrics['SHD']}\n")
            f.write(f"TPR (True Positive Rate): {metrics['TPR']:.4f}\n")
            f.write(f"FPR (False Positive Rate): {metrics['FPR']:.4f}\n")
            f.write(f"Precision: {metrics['Precision']:.4f}\n")
            f.write(f"F1 Score: {metrics['F1_Score']:.4f}\n")
            f.write(f"Accuracy: {metrics['Accuracy']:.4f}\n")
            f.write(f"TP: {metrics['TP']}\n")
            f.write(f"FP: {metrics['FP']}\n")
            f.write(f"FN: {metrics['FN']}\n")
            f.write(f"TN: {metrics['TN']}\n")
        
        plot_graph_comparison(
            causal_graph, true_adj, 
            save_path=f"{results_dir}/{prefix}_graph_comparison.png"
        )
        
        print(f"\nSaved to: {results_dir}/")
        
        return metrics
    else:
        print("No ground-truth adjacency provided")
        return None


def main():
    parser = argparse.ArgumentParser(description='CASTLE causal graph evaluation')
    parser.add_argument('--csv_path', type=str, 
                       default='../Causal_attention/raw_data/numerical_dag_data_5vars.csv',
                       help='CSV path')
    parser.add_argument('--adj_path', type=str,
                       default='../Causal_attention/raw_data/numerical_dag_adj_5vars.npy',
                       help='Adjacency path')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='Threshold')
    parser.add_argument('--reg_lambda', type=float, default=1.0,
                       help='reg_lambda')
    parser.add_argument('--reg_beta', type=float, default=5.0,
                       help='reg_beta')
    parser.add_argument('--max_steps', type=int, default=100,
                       help='max_steps')
    parser.add_argument('--prefix', type=str, default='castle_test',
                       help='Output prefix')
    
    args = parser.parse_args()
    
    metrics = test_castle_causal_learning(
        csv_path=args.csv_path,
        adj_path=args.adj_path,
        threshold=args.threshold,
        reg_lambda=args.reg_lambda,
        reg_beta=args.reg_beta,
        max_steps=args.max_steps,
        prefix=args.prefix
    )
    
    if metrics:
        print(f"\n=== Final ===")
        print(f"SHD: {metrics['SHD']}")
        print(f"TPR: {metrics['TPR']:.4f}")
        print(f"FPR: {metrics['FPR']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"F1 Score: {metrics['F1_Score']:.4f}")
        print(f"Accuracy: {metrics['Accuracy']:.4f}")


if __name__ == "__main__":
    main()