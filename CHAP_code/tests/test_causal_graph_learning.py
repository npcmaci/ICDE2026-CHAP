"""
Causal Graph Learning Evaluation Script

This script evaluates causal graph learning performance on synthetic numerical DAG datasets.
It supports both CHAP and CASTLE models for causal discovery tasks.
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.causal_attention_msk_model import CausalAttentionMskModel
from datasets.numerical_dag_dataset import load_numerical_dag_5vars_dataset, load_numerical_dag_10vars_dataset
from torch.utils.data import DataLoader, random_split
from utils.train_msk_utils import train_msk_model
from models.mask_generators import SigmoidMaskGenerator
import logging
import matplotlib.pyplot as plt
import numpy as np
import argparse
import subprocess
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


# ====================
# Evaluation Metrics
# ====================

def compute_causal_metrics(pred_adj, true_adj, threshold=0.5):
    """Compute causal graph evaluation metrics."""
    pred_binary = (pred_adj > threshold).astype(int)
    true_binary = true_adj.astype(int)
    
    tp = np.sum((pred_binary == 1) & (true_binary == 1))
    fp = np.sum((pred_binary == 1) & (true_binary == 0))
    fn = np.sum((pred_binary == 0) & (true_binary == 1))
    tn = np.sum((pred_binary == 0) & (true_binary == 0))
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    
    shd = fp + fn
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    return {
        'shd': shd,
        'tpr': tpr,
        'fpr': fpr,
        'precision': precision,
        'f1': f1,
        'accuracy': accuracy,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


# ====================
# Causal Graph Extraction
# ====================

def extract_causal_graph(model):
    """Extract binary causal graph from trained model."""
    causal_mask = model.get_causal_mask()
    causal_mask_transposed = causal_mask
    
    THRESHOLD_CAUSAL_ATTENTION = 0.46
    causal_graph = (causal_mask_transposed > THRESHOLD_CAUSAL_ATTENTION).astype(int)
    
    return causal_graph


# ====================
# Results Saving
# ====================

def save_causal_graph_results(pred_adj, true_adj, metrics, save_dir, prefix=""):
    """Persist predicted/true graphs, metrics, and a comparison figure."""
    os.makedirs(save_dir, exist_ok=True)
    
    pred_path = os.path.join(save_dir, f"{prefix}_predicted_causal_graph.npy")
    np.save(pred_path, pred_adj)
    
    true_path = os.path.join(save_dir, f"{prefix}_true_causal_graph.npy")
    np.save(true_path, true_adj)
    
    metrics_path = os.path.join(save_dir, f"{prefix}_causal_metrics.txt")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("Causal Graph Learning Evaluation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Structural Hamming Distance (SHD): {metrics['shd']}\n")
        f.write(f"True Positive Rate (TPR): {metrics['tpr']:.4f}\n")
        f.write(f"False Positive Rate (FPR): {metrics['fpr']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(f"True Positives: {metrics['tp']}\n")
        f.write(f"False Positives: {metrics['fp']}\n")
        f.write(f"False Negatives: {metrics['fn']}\n")
        f.write(f"True Negatives: {metrics['tn']}\n")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im1 = axes[0].imshow(true_adj, cmap='Blues', vmin=0, vmax=1)
    axes[0].set_title('True Causal Graph')
    axes[0].set_xlabel('Effect')
    axes[0].set_ylabel('Cause')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(pred_adj, cmap='Reds', vmin=0, vmax=1)
    axes[1].set_title('Predicted Causal Graph')
    axes[1].set_xlabel('Effect')
    axes[1].set_ylabel('Cause')
    plt.colorbar(im2, ax=axes[1])
    
    diff = pred_adj - true_adj
    im3 = axes[2].imshow(diff, cmap='RdBu', vmin=-1, vmax=1)
    axes[2].set_title('Difference (Pred - True)')
    axes[2].set_xlabel('Effect')
    axes[2].set_ylabel('Cause')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    viz_path = os.path.join(save_dir, f"{prefix}_causal_graph_comparison.png")
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Results saved: {save_dir}")
    print(f"  predicted: {pred_path}")
    print(f"  true: {true_path}")
    print(f"  metrics: {metrics_path}")
    print(f"  figure: {viz_path}")


# ====================
# Main Training Function
# ====================

def test_causal_graph_learning(prefix="", dataset="numerical_5vars", num_heads=4, num_layers=1, gpu_id=-1,
                              learning_rate=1e-4, weight_decay=1e-5, num_epochs=100, patience=10,
                              alpha=1.0, beta=1.0, regression_weight=1.0,
                              recon_update_strategy=(4, 5, 0), pred_update_strategy=(4, 5, 0),
                              optimizer_name='adam', scheduler_name='reduce_on_plateau',
                              d_model=64, dropout=0.1, batch_size=128, 
                              model_source="models.causal_attention_msk_model", train_source="utils.train_msk_utils",
                              seed=42):
    """Train model and evaluate causal graph learning."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}' if gpu_id >= 0 else 'cuda')
        if gpu_id >= 0:
            torch.cuda.set_device(gpu_id)
    else:
        device = torch.device('cpu')
    logger.info(f"Device: {device}")
    
    logger.info(f"Dataset: {dataset}")
    if dataset.lower() == "numerical_5vars":
        dataset_obj, v, num_classes_dict, true_adj_matrix, env_labels = load_numerical_dag_5vars_dataset()
        dataset_name = "Numerical_5vars"
    elif dataset.lower() == "numerical_10vars":
        dataset_obj, v, num_classes_dict, true_adj_matrix, env_labels = load_numerical_dag_10vars_dataset()
        dataset_name = "Numerical_10vars"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v)
    v = v.to(device)
    
    logger.info(f"Dataset name: {dataset_name}")
    logger.info(f"Dataset size: {len(dataset_obj)}")
    logger.info(f"v: {v}")
    logger.info(f"Adj shape: {true_adj_matrix.shape}")
    logger.info(f"True edges: {true_adj_matrix.sum()}")
    
    sample_x, sample_y = dataset_obj[0]
    logger.info(f"Sample x shape: {sample_x.shape}")
    logger.info(f"Sample y shape: {sample_y.shape}")
    
    target_idx = len(v) - 1
    logger.info(f"Target index: {target_idx}")
    
    total_size = len(dataset_obj)
    train_test_split_idx = int(0.8 * total_size)
    
    train_and_val_dataset = torch.utils.data.Subset(dataset_obj, range(0, train_test_split_idx))
    test_dataset = torch.utils.data.Subset(dataset_obj, range(train_test_split_idx, total_size))
    
    train_val_size = len(train_and_val_dataset)
    val_size = int(0.2 * train_val_size)
    train_size = train_val_size - val_size
    
    train_dataset, val_dataset = random_split(train_and_val_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Train/Val/Test: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")
    
    logger.info("Init mask generator...")
    mask_generator = SigmoidMaskGenerator(
        num_features=len(v),
        initial_threshold=0.2,
        final_threshold=0.2,
        threshold_multiplier=1.1
    ).to(device)
    
    logger.info(f"Import model class from {model_source}...")
    try:
        module = __import__(model_source, fromlist=['CausalAttentionMskModel'])
        ModelClass = getattr(module, 'CausalAttentionMskModel')
    except ImportError as e:
        raise ImportError(f"Failed to import model module {model_source}: {e}")
    
    logger.info(f"Import train function from {train_source}...")
    try:
        module = __import__(train_source, fromlist=['train_msk_model'])
        train_function = getattr(module, 'train_msk_model')
    except ImportError as e:
        raise ImportError(f"Failed to import train module {train_source}: {e}")
    
    logger.info("Init model...")
    model = ModelClass(
        v=v,
        num_classes_dict=num_classes_dict,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        share_embedding=True,
        mask_generator=mask_generator,
        target_idx=target_idx
    ).to(device)
    
    logger.info(f"Num params: {sum(p.numel() for p in model.parameters()):,}")
    
    logger.info("Start training...")
    trained_model = train_function(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        v=v,
        num_classes_dict=num_classes_dict,
        test_loader=test_loader,
        prediction_idx=target_idx,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        patience=patience,
        alpha=alpha,
        beta=beta,
        regression_weight=regression_weight,
        recon_update_strategy=recon_update_strategy,
        pred_update_strategy=pred_update_strategy,
        optimizer_name=optimizer_name,
        scheduler_name=scheduler_name,
        device=device,
        save_dir=f'checkpoints_causal_graph_{prefix}' if prefix else 'checkpoints_causal_graph',
        log_interval=100
    )
    
    logger.info("Extract causal graph...")
    predicted_adj_matrix = extract_causal_graph(trained_model)
    
    logger.info("Compute metrics...")
    metrics = compute_causal_metrics(predicted_adj_matrix, true_adj_matrix, threshold=0.46)
    
    logger.info(f"Metrics:")
    logger.info(f"  SHD (Structural Hamming Distance): {metrics['shd']}")
    logger.info(f"  TPR (True Positive Rate): {metrics['tpr']:.4f}")
    logger.info(f"  FPR (False Positive Rate): {metrics['fpr']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  F1 Score: {metrics['f1']:.4f}")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    
    save_dir = f'causal_graph_results_{prefix}' if prefix else 'causal_graph_results'
    save_causal_graph_results(predicted_adj_matrix, true_adj_matrix, metrics, save_dir, prefix)
    
    logger.info(f"Done. Saved to: {save_dir}")
    
    return {
        'predicted_adj_matrix': predicted_adj_matrix,
        'true_adj_matrix': true_adj_matrix,
        'metrics': metrics,
        'model': trained_model
    }


# ====================
# Command Line Interface
# ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Causal Graph Learning Test')
    
    parser.add_argument('--model', type=str, default='causal_attention',
                       choices=['causal_attention', 'castle'],
                       help='Model type')
    parser.add_argument('--prefix', type=str, default='', 
                       help='Output prefix')
    parser.add_argument('--dataset', type=str, default='numerical_10vars',
                       choices=['numerical_5vars', 'numerical_10vars'],
                       help='Dataset choice')
    parser.add_argument('--gpu_id', type=int, default=-1, 
                       help='GPU id (-1 for default)')
    
    parser.add_argument('--d_model', type=int, default=64,
                       help='Model dim')
    parser.add_argument('--num_heads', type=int, default=4, 
                       help='Num heads')
    parser.add_argument('--num_layers', type=int, default=1, 
                       help='Num layers')
    parser.add_argument('--dropout', type=float, default=0.1, 
                       help='Dropout')
    
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='LR')
    parser.add_argument('--weight_decay', type=float, default=1e-5, 
                       help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=100, 
                       help='Epochs')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    
    parser.add_argument('--optimizer', type=str, default='adam', 
                       choices=['adam', 'sgd'],
                       help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='reduce_on_plateau', 
                       choices=['reduce_on_plateau', 'cosine'],
                       help='LR scheduler')
    
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Reconstruction loss weight')
    parser.add_argument('--beta', type=float, default=0.1,
                       help='Sparsity/DAG loss weight')
    parser.add_argument('--regression_weight', type=float, default=1.0,
                       help='Regression loss weight')
    
    parser.add_argument('--recon_update_strategy', nargs=3, type=int, default=[4, 5, 0],
                       help='Recon update strategy (num interval offset)')
    parser.add_argument('--pred_update_strategy', nargs=3, type=int, default=[4, 5, 0],
                       help='Pred update strategy (num interval offset)')
    
    parser.add_argument('--castle_reg_lambda', type=float, default=0.01,
                       help='CASTLE reg_lambda')
    parser.add_argument('--castle_reg_beta', type=float, default=0.1,
                       help='CASTLE reg_beta')
    parser.add_argument('--castle_max_steps', type=int, default=50,
                       help='CASTLE max_steps')
    
    parser.add_argument('--model_source', type=str, default='models.causal_attention_msk_model',
                       help='Model class source')
    parser.add_argument('--train_source', type=str, default='utils.train_msk_utils',
                       help='Train function source')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Causal Graph Learning Test Configuration")
    print("=" * 60)
    print(f"Prefix: {args.prefix if args.prefix else 'none'}")
    print(f"Dataset: {args.dataset}")
    print(f"GPU: {args.gpu_id}")
    if args.model == 'causal_attention':
        print(f"Threshold (fixed): 0.46")
    else:
        print(f"Threshold (fixed): 0.05")
    print("-" * 30)
    print("Model params:")
    print(f"  d_model: {args.d_model}")
    print(f"  num_heads: {args.num_heads}")
    print(f"  num_layers: {args.num_layers}")
    print(f"  dropout: {args.dropout}")
    print("-" * 30)
    print("Train params:")
    print(f"  lr: {args.learning_rate}")
    print(f"  weight_decay: {args.weight_decay}")
    print(f"  epochs: {args.num_epochs}")
    print(f"  patience: {args.patience}")
    print(f"  batch_size: {args.batch_size}")
    print("=" * 60)
    
    if args.model == 'causal_attention':
        _ = test_causal_graph_learning(
            prefix=args.prefix, 
            dataset=args.dataset, 
            num_heads=args.num_heads, 
            num_layers=args.num_layers,
            gpu_id=args.gpu_id,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_epochs=args.num_epochs,
            patience=args.patience,
            alpha=args.alpha,
            beta=args.beta,
            regression_weight=args.regression_weight,
            recon_update_strategy=tuple(args.recon_update_strategy),
            pred_update_strategy=tuple(args.pred_update_strategy),
            optimizer_name=args.optimizer,
            scheduler_name=args.scheduler,
            d_model=args.d_model,
            dropout=args.dropout,
            batch_size=args.batch_size,
            model_source=args.model_source,
            train_source=args.train_source,
            seed=args.seed,
        )
    else:
        # Call CASTLE script (threshold fixed = 0.05) using project-relative path
        base_castle_script = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'castle', 'test_castle_causal_learning.py'))
        # Build project-local raw_data path (no absolute paths)
        raw_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'raw_data'))
        if args.dataset == 'numerical_5vars':
            csv_path = os.path.join(raw_data_dir, 'numerical_dag_data_5vars.csv')
            adj_path = os.path.join(raw_data_dir, 'numerical_dag_adj_5vars.npy')
        else:
            csv_path = os.path.join(raw_data_dir, 'numerical_dag_data_10vars.csv')
            adj_path = os.path.join(raw_data_dir, 'numerical_dag_adj_10vars.npy')

        cmd = [
            sys.executable, base_castle_script,
            '--csv_path', csv_path,
            '--adj_path', adj_path,
            '--threshold', '0.05',
            '--reg_lambda', str(args.castle_reg_lambda),
            '--reg_beta', str(args.castle_reg_beta),
            '--max_steps', str(args.castle_max_steps),
            '--prefix', args.prefix or 'castle_run'
        ]
        print('Running CASTLE:', ' '.join(cmd))
        try:
            completed = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            print(completed.stdout)
        except subprocess.CalledProcessError as e:
            print('CASTLE run failed:')
            print(e.stdout)

    print("Done.")