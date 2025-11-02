# CHAP: Efficient Causal Hierarchy-Aware Prediction for Structured Data via Masked Attention

We propose an efficient Causal Hierarchy-Aware Prediction (CHAP) framework for structured data via a masked attention mechanism, aiming to bridge the gap between high-capacity neural network models and causal analysis. CHAP is composed of three key components: a masked attention encoder, a reconstruction branch, and a hierarchy-aware predictor.

The masked attention encoder incorporates a learnable causal mask to represent causal dependencies among columns and integrates it with the attention mechanism through a dot-product formulation. The reconstruction branch leverages an auxiliary reconstruction task combined with DAG and sparsity losses to ensure accurate extraction of causal relationships from complex inter-column interactions.

Finally, the hierarchy-aware predictor aggregates the latent representations of each column via a set of learnable weights to perform prediction, where a hierarchy-aware regularization term is introduced to align each column’s importance with the learned causal hierarchy. CHAP alternately trains the reconstruction branch and the predictor. A lazy-update mechanism is further introduced to reduce the frequency of costly reconstruction updates, improving efficiency without sacrificing overall performance.

## Framework

![Framework](./README.assets/2%20(2).png)

## Project Structure

```
Causal_attention/
├── models/ #Model files, showing the main files with others being ablation experiments and design choices experiment variants
│   ├── causal_attention_msk_model.py #Main model
│   ├── embedding_layers.py #Embedding layers
│   └── mask_generators.py #Causal Mask generators
├── datasets/ #pytorch dataset classes
│   ├── adult_dataset.py
│   ├── cardio_dataset.py
│   ├── creditcard_dataset.py
│   ├── diamonds_dataset.py
│   ├── elevator_dataset.py
│   ├── housesale_dataset.py
│   ├── tabular_datasets.py
│   └── numerical_dag_dataset.py # Numerical DAG dataset for synthetic experiments
├── utils/ # Training and evaluation functions, showing the main files with others being ablation experiments and design choices experiment variants
│   ├── train_msk_utils.py
│   └── generate_numerical_dag_data.py # Synthetic DAG data generator
├── baseline/ #Baseline test code, showing the main scripts with remaining files being runtime requirements
│   ├── test_ft_transformer_baseline.py # FT-Transformer
│   ├── test_linear_baseline.py #LogCause
│   ├── test_tabtransformer_baseline.py #Tab-Transformer
│   ├── castle/ #Need to pull castle source code
│   │   ├── CASTLE.py 
│   │   ├── CASTLE_CF.py
│   │   ├── main.py #castle for regression tasks
│   │   ├── main_cf.py #castle for classification tasks
│   │   └── test_castle_causal_learning.py #causal graph learning evaluation
│   ├── saint/ #Need to pull saint source code
│   │   ├── data_adapter.py
│   │   └── train_with_custom_data.py # SAINT test script
│   ├── automl/ #AutoML baselines (AutoGluon and AutoSklearn)
│   │   ├── test_automl_classification.py # AutoML classification test script
│   │   ├── test_automl_regression.py # AutoML regression test script
│   │   ├── autog_cls.py # AutoGluon classification helper
│   │   ├── autog_reg.py # AutoGluon regression helper
│   │   ├── autos_cls.py # AutoSklearn classification helper
│   │   └── autos_reg.py # AutoSklearn regression helper
│   └── XGboost/ #XGBoost baseline
│       └── xgboost.py # XGBoost test script
├── tests/ #CHAP model main function
│   ├── test_causal_attention_msk_model.py
│   └── test_causal_graph_learning.py #Causal graph learning evaluation on synthetic data
├── raw_data/ #Process data according to dataset chapter instructions and place in this folder
│   ├── adult.csv
│   ├── cardio.csv
│   ├── creditcard.csv
│   ├── diamonds.csv
│   ├── elevator.csv
│   ├── housesale.csv
│   ├── numerical_dag_data_5vars.csv # Synthetic DAG data with 5 variables
│   ├── numerical_dag_data_10vars.csv # Synthetic DAG data with 10 variables
│   ├── numerical_dag_adj_5vars.npy # True adjacency matrix for 5-variable DAG
│   └── numerical_dag_adj_10vars.npy # True adjacency matrix for 10-variable DAG
└── README.md
```

## Requirements

- pytorch=2.2.2
  - pytorch-cuda=12.1
  - python=3.9.19
  - tab-transformer-pytorch==0.4.2 (for baselines)
  - tensorflow=1.15.0 (for Castle baseline)

## Datasets

All datasets are publicly available on Kaggle or the UCI Machine Learning. Data needs to be processed into CSV format, with some datasets requiring special processing. The specific download addresses and special processing for each dataset are as follows:

#### Adult Dataset
- **Source**: [Adult - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult)
- **Task**: Binary classification (income prediction)
- **Preprocessing**:
  - None


#### Cardio Dataset
- **Source**: [Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
- **Task**: Multi-class classification (cardiotocography classification)
- **Preprocessing**:
  - Four features: height, weight, ap_hi, ap_lo are discretized into categorical variables with intervals of 10

#### CreditCard Dataset
- **Source**: [creditcard.csv](https://www.kaggle.com/datasets/arockiaselciaa/creditcardcsv)
- **Task**: Binary classification (credit card default prediction)
- **Preprocessing**:
  - None

#### Diamonds Dataset
- **Source**: [Diamonds](https://www.kaggle.com/datasets/shivam2503/diamonds)
- **Task**: Regression (diamond price prediction)
- **Preprocessing**:
  - The price column is moved to the last column as the prediction target

#### Elevator Dataset
- **Source**: [Elevator Predictive Maintenance Dataset](https://www.kaggle.com/datasets/shivamb/elevator-predictive-maintenance-dataset)
- **Task**: Regression (elevator behavior prediction)
- **Preprocessing**:
  - The vibration column is moved to the last column as the prediction target

#### Housesale Dataset
- **Source**: [Housing Prices in Metropolitan Areas of India](https://www.kaggle.com/datasets/ruchi798/housing-prices-in-metropolitan-areas-of-india)
- **Task**: Regression (house price prediction)
- **Preprocessing**:
  - All 6 CSV files in the dataset are concatenated together, and the data is shuffled with random seed 42

## Data Preparation

Before running experiments, please prepare the datasets as follows:

1. **Download and process real-world datasets**: According to the instructions in the Datasets section, download the 6 datasets (Adult, Cardio, CreditCard, Diamonds, Elevator, Housesale) and process them into CSV format. Place the processed CSV files in the `raw_data/` directory.

2. **Generate synthetic datasets**: Run the synthetic data generator to create the numerical DAG datasets:
   ```bash
   python utils/generate_numerical_dag_data.py
   ```
   This will generate the four synthetic data files shown in the project structure:
   - `raw_data/numerical_dag_data_5vars.csv`
   - `raw_data/numerical_dag_data_10vars.csv`
   - `raw_data/numerical_dag_adj_5vars.npy`
   - `raw_data/numerical_dag_adj_10vars.npy`

## Training models

* **our model：**
  * cd tests
  * python test_causal_attention_msk_model.py --dataset creditcard --num_heads 4 --num_layers 2 --gpu_id 0 --d_model 32 --learning_rate 0.001
  * Note: The main experiments also provide efficiency results (training time and inference time) which are automatically logged during execution.
* **baselines：**
  * **FT-Transformer**
    * cd baseline
    * python test_ft_transformer_base_line.py --dataset creditcard --d_token 64 --n_heads 8 --n_blocks 2 --lr 0.001
  * **Tab-Transformer**
    * cd baseline
    * python test_tab_transformer_base_line.py --dataset creditcard --d_token 64 --n_heads 8 --n_blocks 6 --lr 0.001
  * **SAINT**
    * cd baseline/saint
    * python train_with_custom_data.py --dataset creditcard --embedding_size 64 --transformer_depth 4 --attention_heads 4 --lr 0.0001 
  * **CASTLE**
    * cd baseline/castle
    * python main_cf.py   --csv creditcard.csv  --n_folds 5   --reg_lambda 1.0   --reg_beta 5.0  --extension creditcard
    * python main.py   --csv diamonds.csv  --n_folds 5   --reg_lambda 1.0   --reg_beta 5.0  --extension diamonds
  * **LogCause**
    * cd baseline
    * python test_linear_baseline.py --dataset creditcard --learning_rate 0.0001
  * **AutoML**
    * cd baseline/automl
    * python test_automl_classification.py --dataset creditcard --time_limit 480
    * python test_automl_regression.py --dataset diamonds --time_limit 480
  * **XGBoost**
    * cd baseline/xgboost
    * python xgboost.py --dataset Diamonds --n_folds 5 --random_state 42 --test_size 0.2

## Running Experiments

To run all experiments at once, use the provided shell script:

```bash
bash run.sh
```

This script will sequentially run:
1. CHAP Model main experiments
2. Baseline experiments (FT-Transformer, Tab-Transformer, SAINT, CASTLE, LogCause, AutoML, XGBoost)
3. Synthetic data experiments
4. Design choices experiments
5. Ablation study experiments

The script automatically handles directory navigation and outputs progress messages for each section. All hyperparameters are configured according to the paper's specifications. Individual experiments can also be run using the detailed commands listed below.

## Synthetic Data Experiments

Experiments on synthetic DAG data to evaluate causal graph learning performance:

* python tests/test_causal_graph_learning.py --model causal_attention --dataset numerical_5vars
* python tests/test_causal_graph_learning.py --model causal_attention --dataset numerical_10vars
* python tests/test_causal_graph_learning.py --model castle --dataset numerical_5vars --castle_reg_lambda 0.01 --castle_reg_beta 0.1 --castle_max_steps 50
* python tests/test_causal_graph_learning.py --model castle --dataset numerical_10vars --castle_reg_lambda 0.01 --castle_reg_beta 0.1 --castle_max_steps 50

## Design Choices

Experiments evaluating different design choices in CHAP:

* **Add mask**
  * python tests/test_causal_attention_msk_model.py --prefix creditcard --dataset creditcard --num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001 --model_source models.causal_attention_msk_model_add_mask_design

* **Softmax mask**
  * python tests/test_causal_attention_msk_model.py --prefix creditcard --dataset creditcard --num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001 --model_source models.causal_attention_msk_model_inner_softmax_mask_design

* **Flattening**
  * python tests/test_causal_attention_msk_model.py --prefix creditcard --dataset creditcard --num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001 --model_source models.causal_attention_msk_model_parents_predictor_design

* **Mean Pooling**
  * python tests/test_causal_attention_msk_model.py --prefix creditcard --dataset creditcard --num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001 --model_source models.causal_attention_msk_model

* **Parallel Reconstruction**
  * python tests/test_causal_attention_msk_model.py --prefix creditcard --dataset creditcard --num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001 --model_source models.causal_attention_msk_model_nomsk_design

## Ablation Study

Experiments removing individual components to evaluate their contributions:

* **w/o pred reg**
  * python tests/test_causal_attention_msk_model.py --prefix creditcard --dataset creditcard --num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001 --model_source models.causal_attention_msk_model --train_source utils.train_msk_utils_wo_prediction_reg

* **w/o DAG loss**
  * python tests/test_causal_attention_msk_model.py --prefix creditcard --dataset creditcard --num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001 --model_source models.causal_attention_msk_model --train_source utils.train_msk_utils_wo_dag_loss

* **w/o sparse loss**
  * python tests/test_causal_attention_msk_model.py --prefix creditcard --dataset creditcard --num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001 --model_source models.causal_attention_msk_model --train_source utils.train_msk_utils_wo_sparse_loss

* **w/o reconstruction loss**
  * python tests/test_causal_attention_msk_model.py --prefix creditcard --dataset creditcard --num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001 --model_source models.causal_attention_msk_model --train_source utils.train_msk_utils_wo_reconstruction_loss

* **w/o all reg loss**
  * python tests/test_causal_attention_msk_model.py --prefix creditcard --dataset creditcard --num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001 --model_source models.causal_attention_msk_model --train_source utils.train_msk_utils_wo_three_loss

* **w/o causal mask**
  * python tests/test_causal_attention_msk_model.py --prefix creditcard --dataset creditcard --num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001 --model_source models.causal_attention_msk_model_wo_mask_attention --train_source utils.train_msk_utils_wo_three_loss

