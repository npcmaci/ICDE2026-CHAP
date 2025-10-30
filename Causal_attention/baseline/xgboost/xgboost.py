import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def auc_fn(y, p, num_class=2):
    if num_class > 2:
        return roc_auc_score(y, p, multi_class='ovo')
    else:
        return roc_auc_score(y, p[:, 1])

def ensure_all_classes(y_train, y_test):
    train_classes = set(np.unique(y_train))
    test_classes = set(np.unique(y_test))
    missing_classes = test_classes - train_classes

    if missing_classes:
        for cls in missing_classes:
            fake_sample = np.mean(X_train, axis=0).reshape(1, -1)
            X_train.append(fake_sample)
            y_train.append(cls)

    return np.array(X_train), np.array(y_train)

def load_single_data_all(table_file, target=None):
    if os.path.exists(table_file):
        print(f"Load from local data dir: {table_file}")
        df = pd.read_csv(table_file)

        if not target:
            target = df.columns.tolist()[-1]

        df.dropna(axis=0, subset=[target], inplace=True)

        y = df[target]
        X = df.drop(columns=[target])

        cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        if len(cat_cols) > 0:
            encoder = OrdinalEncoder(dtype=np.float64)
            X[cat_cols] = encoder.fit_transform(X[cat_cols])
            
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) > 0:
            for col in num_cols: 
                X[col].fillna(X[col].mode()[0], inplace=True)       
            X[num_cols] = MinMaxScaler().fit_transform(X[num_cols])

        
        scaler = StandardScaler()
        y = pd.Series(scaler.fit_transform(y.values.reshape(-1, 1)).flatten()) 
        print(f"# data: {len(X)}, # feat: {X.shape[1]}, pos rate: {(y == 1).sum() / len(y):.2f}")
        return X, y
    else:
        raise RuntimeError(f"No such data file: {table_file}")

def log_config(log_name):
    exp_dir = f"xgboost_test_{log_name}_{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}"
    exp_log_dir = os.path.join("logs", exp_dir)
    os.makedirs(exp_log_dir, exist_ok=True)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, handlers=[
        logging.FileHandler(os.path.join(exp_log_dir, "log.txt")),
        logging.StreamHandler()
    ])
    return exp_log_dir

log_name = "XGBoost_test"

task_datasets = ['OpenTabs/Test/Reg/Diamonds.csv', 'OpenTabs/Test/Reg/Elevators.csv']

exp_log_dir = log_config(log_name)
skf = KFold(n_splits=5, random_state=42, shuffle=True)
all_res = {}

for table_file_path in task_datasets:
    data_name = os.path.basename(table_file_path)
    logging.info(f"Start========>{data_name}_DataSet==========>")
    
    X, y = load_single_data_all(table_file_path)
    
    score_list = []
    idd = 0
    for trn_idx, val_idx in skf.split(X, y):
        idd += 1
        train_data = X.loc[trn_idx]
        train_label = y[trn_idx]
        X_test = X.loc[val_idx]
        y_test = y[val_idx]

        X_train, X_val, y_train, y_val = train_test_split(
            train_data, train_label, test_size=0.2, random_state=42, shuffle=True
        )

        classifier = XGBRegressor(random_state=42)

        classifier.fit(X_train, y_train)

        p_pred = classifier.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, p_pred))
        score_list.append(rmse)
        logging.info(f"Fold_{idd} AUC===>{data_name}==> {rmse:.4f}")

    all_res[data_name] = np.mean(score_list)
    logging.info(f"Mean 5-fold AUC===>{data_name}==> {np.mean(score_list):.4f}")

mean_list = []
for key, value in all_res.items():
    logging.info(f"Dataset: {key}, Mean AUC: {value:.4f}")
    mean_list.append(value)

result_df = pd.DataFrame(mean_list, index=all_res.keys(), columns=["AUC"])
res_path = os.path.join(exp_log_dir, "results.csv")
result_df.to_csv(res_path, index=True)
logging.info(f"Overall Mean RMSE: {np.mean(mean_list):.4f}")
logging.info(f"Results saved to {res_path}")