#!/bin/bash

# CHAP: Comprehensive Experiment Runner
# This script runs all experiments for the CHAP model and baselines
# Make sure all dependencies are installed and data is prepared before running

# Exit on any error
set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================="
echo "CHAP Experiment Runner"
echo "Starting experiments..."
echo "========================================="
echo ""
# Note: For standard deviation calculation, we run each experiment 3 times with random seeds 42, 123, 456


# ============================================
# 1. CHAP Model - Main Experiments
# ============================================
echo "========================================="
echo "Section 1: CHAP Model - Main Experiments"
echo "========================================="

cd tests

# CHAP hyperparameters from paper:
# - Hidden size: 32 (d_model)
# - Batch size: 128
# - Learning rate: 1e-3
# - Number of layers: 2
# - Number of heads: hidden_size / 8 = 32 / 8 = 4
# - Epochs: 20
# - Dropout: 0.1
# - Weight decay: 1e-5

datasets=("adult" "cardio" "creditcard" "diamonds" "elevator" "housesale")
for dataset in "${datasets[@]}"; do
    echo "Running CHAP on $dataset dataset..."
    python test_causal_attention_msk_model.py --dataset $dataset --d_model 32 --num_heads 4 --num_layers 2 --batch_size 128 --learning_rate 0.001 --num_epochs 20 --dropout 0.1 --weight_decay 1e-5 --gpu_id 0
    echo ""
done

cd ..

# ============================================
# 2. FT-Transformer Baseline (All Datasets)
# ============================================
echo "========================================="
echo "Section 2: FT-Transformer Baseline"
echo "========================================="

cd baseline

# FT-Transformer hyperparameters from paper:
# - Hidden size: 64 (d_token)
# - Batch size: 128
# - Learning rate: 1e-3
# - Number of layers: 2
# - Number of heads: hidden_size / 8 = 64 / 8 = 8
# - Epochs: 30

for dataset in "${datasets[@]}"; do
    echo "Running FT-Transformer on $dataset dataset..."
    python test_ft_transformer_baseline.py --dataset $dataset --d_token 64 --n_heads 8 --n_blocks 2 --batch_size 128 --lr 0.001 --epochs 30 --gpu_id 0
    echo ""
done

cd ..

echo ""

# ============================================
# 3. Tab-Transformer Baseline (All Datasets)
# ============================================
echo "========================================="
echo "Section 3: Tab-Transformer Baseline"
echo "========================================="

cd baseline

# Tab-Transformer hyperparameters from paper:
# - Hidden size: 32 (d_token)
# - Batch size: 128
# - Learning rate: 1e-3
# - Number of layers: 6
# - Number of heads: hidden_size / 8 = 32 / 8 = 4
# - Epochs: 30

for dataset in "${datasets[@]}"; do
    echo "Running Tab-Transformer on $dataset dataset..."
    python test_tabtransformer_baseline.py --dataset $dataset --d_token 32 --n_heads 4 --n_blocks 6 --batch_size 128 --lr 0.001 --epochs 30 --gpu_id 0
    echo ""
done

cd ..

echo ""

# ============================================
# 4. SAINT Baseline (All Datasets)
# ============================================
echo "========================================="
echo "Section 4: SAINT Baseline"
echo "========================================="

cd baseline/saint

# SAINT hyperparameters from paper:
# - Hidden size: 64 (embedding_size)
# - Batch size: 128
# - Learning rate: 1e-4
# - Number of layers: 4 (transformer_depth)
# - Number of heads: 4 (attention_heads)
# - Epochs: 50

for dataset in "${datasets[@]}"; do
    echo "Running SAINT on $dataset dataset..."
    python train_with_custom_data.py --dataset $dataset --embedding_size 64 --batchsize 128 --transformer_depth 4 --attention_heads 4 --lr 0.0001 --epochs 50
    echo ""
done

cd ../..
echo ""

# ============================================
# 5. CASTLE Baseline (All Datasets)
# ============================================
echo "========================================="
echo "Section 5: CASTLE Baseline"
echo "========================================="

cd baseline/castle

# CASTLE hyperparameters from paper:
# - Hidden size: 32
# - Batch size: 64
# - Learning rate: 1e-4
# - Number of MLP layers: 2
# - Alpha: 1
# - Beta: 5
# - Epochs: 50
# - n_folds: 5

classification_datasets=("adult" "cardio" "creditcard")
regression_datasets=("diamonds" "elevator" "housesale")

for dataset in "${classification_datasets[@]}"; do
    echo "Running CASTLE for classification on $dataset dataset..."
    python main_cf.py --csv ../raw_data/${dataset}.csv --n_folds 5 --reg_lambda 1.0 --reg_beta 5.0 --extension $dataset
    echo ""
done

for dataset in "${regression_datasets[@]}"; do
    echo "Running CASTLE for regression on $dataset dataset..."
    python main.py --csv ../raw_data/${dataset}.csv --n_folds 5 --reg_lambda 1.0 --reg_beta 5.0 --extension $dataset
    echo ""
done

cd ../..
echo ""

# ============================================
# 6. LogCause Baseline (All Datasets)
# ============================================
echo "========================================="
echo "Section 6: LogCause Baseline"
echo "========================================="

cd baseline

# LogCause hyperparameters from paper:
# - Hidden size: 32
# - Batch size: 64
# - Learning rate: 1e-5
# - Epochs: 200

for dataset in "${datasets[@]}"; do
    echo "Running LogCause on $dataset dataset..."
    python test_linear_baseline.py --dataset $dataset --batch_size 64 --lr 0.00001 --epochs 200
    echo ""
done

cd ..
echo ""

# ============================================
# 7. AutoML and XGBoost Baselines
# ============================================
echo "========================================="
echo "Section 7: AutoML and XGBoost Baselines"
echo "========================================="

cd baseline

# AutoML
echo "Running AutoML for classification on all classification datasets..."
cd automl
classification_datasets_for_automl=("adult" "cardio" "creditcard")
for dataset in "${classification_datasets_for_automl[@]}"; do
    echo "Running AutoML classification on $dataset dataset..."
    python test_automl_classification.py --dataset $dataset --time_limit 480
    echo ""
done
cd ..

echo "Running AutoML for regression on all regression datasets..."
cd automl
regression_datasets_for_automl=("diamonds" "elevator" "housesale")
for dataset in "${regression_datasets_for_automl[@]}"; do
    echo "Running AutoML regression on $dataset dataset..."
    python test_automl_regression.py --dataset $dataset --time_limit 480
    echo ""
done
cd ..

# XGBoost
echo "Running XGBoost baseline..."
cd xgboost
# Note: XGBoost script runs pre-configured datasets, see xgboost.py for details
python xgboost.py
cd ..

cd ..

echo ""

# ============================================
# 8. Synthetic Data Experiments
# ============================================
echo "========================================="
echo "Section 8: Synthetic Data Experiments"
echo "========================================="

cd tests

# CHAP on synthetic 5-variable data
echo "Running CHAP on numerical_5vars dataset..."
python test_causal_graph_learning.py --model causal_attention --dataset numerical_5vars
echo ""

# CHAP on synthetic 10-variable data
echo "Running CHAP on numerical_10vars dataset..."
python test_causal_graph_learning.py --model causal_attention --dataset numerical_10vars
echo ""

# CASTLE on synthetic 5-variable data
echo "Running CASTLE on numerical_5vars dataset..."
python test_causal_graph_learning.py --model castle --dataset numerical_5vars --castle_reg_lambda 0.01 --castle_reg_beta 0.1 --castle_max_steps 50
echo ""

# CASTLE on synthetic 10-variable data
echo "Running CASTLE on numerical_10vars dataset..."
python test_causal_graph_learning.py --model castle --dataset numerical_10vars --castle_reg_lambda 0.01 --castle_reg_beta 0.1 --castle_max_steps 50
echo ""

cd ..
echo ""

# ============================================
# 9. Design Choices Experiments
# ============================================
echo "========================================="
echo "Section 9: Design Choices Experiments"
echo "========================================="

cd tests

# Design choice experiments use adult and diamonds datasets with standard hyperparameters
design_params="--num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001"

design_datasets=("adult" "diamonds")
for dataset in "${design_datasets[@]}"; do
    # Add mask design
    echo "Running Add Mask design on $dataset..."
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $design_params --model_source models.causal_attention_msk_model_add_mask_design
    echo ""
    
    # Softmax mask design
    echo "Running Softmax Mask design on $dataset..."
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $design_params --model_source models.causal_attention_msk_model_inner_softmax_mask_design
    echo ""
    
    # Flattening design
    echo "Running Flattening design on $dataset..."
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $design_params --model_source models.causal_attention_msk_model_parents_predictor_design
    echo ""
    
    # Mean Pooling design
    echo "Running Mean Pooling design on $dataset..."
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $design_params --model_source models.causal_attention_msk_model
    echo ""
    
    # Parallel Reconstruction design
    echo "Running Parallel Reconstruction design on $dataset..."
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $design_params --model_source models.causal_attention_msk_model_nomsk_design
    echo ""
done

cd ..
echo ""

# ============================================
# 10. Ablation Study Experiments
# ============================================
echo "========================================="
echo "Section 10: Ablation Study Experiments"
echo "========================================="

cd tests

# Ablation experiments use adult and diamonds datasets with standard hyperparameters
ablation_params="--num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001"

ablation_datasets=("adult" "diamonds")
for dataset in "${ablation_datasets[@]}"; do
    # w/o pred reg
    echo "Running w/o pred reg on $dataset..."
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $ablation_params --model_source models.causal_attention_msk_model --train_source utils.train_msk_utils_wo_prediction_reg
    echo ""
    
    # w/o DAG loss
    echo "Running w/o DAG loss on $dataset..."
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $ablation_params --model_source models.causal_attention_msk_model --train_source utils.train_msk_utils_wo_dag_loss
    echo ""
    
    # w/o sparse loss
    echo "Running w/o sparse loss on $dataset..."
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $ablation_params --model_source models.causal_attention_msk_model --train_source utils.train_msk_utils_wo_sparse_loss
    echo ""
    
    # w/o reconstruction loss
    echo "Running w/o reconstruction loss on $dataset..."
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $ablation_params --model_source models.causal_attention_msk_model --train_source utils.train_msk_utils_wo_reconstruction_loss
    echo ""
    
    # w/o all reg loss
    echo "Running w/o all reg loss on $dataset..."
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $ablation_params --model_source models.causal_attention_msk_model --train_source utils.train_msk_utils_wo_three_loss
    echo ""
    
    # w/o causal mask
    echo "Running w/o causal mask on $dataset..."
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $ablation_params --model_source models.causal_attention_msk_model_wo_mask_attention --train_source utils.train_msk_utils_wo_three_loss
    echo ""
done

cd ..

# ============================================
# All Experiments Completed
# ============================================
echo ""
echo "========================================="
echo "All experiments completed successfully!"
echo "========================================="
echo ""
echo "Results should be available in their respective output directories."
echo "Check the logs for detailed performance metrics."
echo ""
