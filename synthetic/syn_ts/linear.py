import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import os
global_epochs = 300

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
seed_everything(42)

class LinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None
    
    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]  # Add intercept term
        self.coefficients = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
        self.intercept = self.coefficients[0]
        self.coefficients = self.coefficients[1:]
    
    def predict(self, X):
        return X.dot(self.coefficients) + self.intercept

class RidgeRegression:
    def __init__(self, alpha=1.0):  # 알파 값을 크게 증가
        self.alpha = alpha
        self.coefficients = None
        self.intercept = None
    
    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]  # Add intercept term
        identity = np.eye(X.shape[1])
        identity[0, 0] = 0  # Don't regularize the intercept
        self.coefficients = np.linalg.inv(X.T.dot(X) + self.alpha * identity).dot(X.T).dot(y)
        self.intercept = self.coefficients[0]
        self.coefficients = self.coefficients[1:]
    
    def predict(self, X):
        return X.dot(self.coefficients) + self.intercept

def compute_ite(model, X):
    t_0 = np.zeros((X.shape[0], 1))
    t_1 = np.ones((X.shape[0], 1))
    y_0 = model.predict(np.c_[X, t_0])
    y_1 = model.predict(np.c_[X, t_1])
    return y_1 - y_0

def compute_pehe(ite_true, ite_pred):
    return np.sqrt(np.mean((ite_true - ite_pred)**2))

def compute_ate(ite_true, ite_pred):
    return np.mean(np.abs(np.mean(ite_true) - np.mean(ite_pred)))

def prepare_data(df):
    seq_len = len(df['time'].unique())
    df_mean = df.groupby('sample').mean().reset_index()
    X = df_mean[['X1', 'X2', 'X3', 'X4']].values
    t_indirect = df_mean['Ti'].values
    t = (df_mean['Td'].values > 0.5).astype(float)
    y = df_mean['Y'].values
    treatment_effect = df_mean['treatment_effect'].values

    X_ts = df[['X1', 'X2', 'X3', 'X4']].values.reshape(-1, seq_len, 4)  
    t_indirect_ts = df['Ti'].values.reshape(-1, seq_len)
    t_ts = df['Td'].values.reshape(-1, seq_len)
    y_ts = df['Y'].values.reshape(-1, seq_len)
    treatment_effect_ts = df['treatment_effect'].values.reshape(-1, seq_len)
    return X, t_indirect, t, y, treatment_effect, X_ts, t_indirect_ts, t_ts, y_ts, treatment_effect_ts

def train_and_evaluate(X, t_indirect, t, y, treatment_effect, X_ts, t_indirect_ts, t_ts, y_ts, treatment_effect_ts):
    X_train, X_temp, t_indirect_train, t_indirect_temp, t_train, t_temp, y_train, y_temp, treatment_effect_train, treatment_effect_temp, X_ts_train, X_ts_temp, t_indirect_ts_train, t_indirect_ts_temp, t_ts_train, t_ts_temp, y_ts_train, y_ts_temp, treatment_effect_ts_train, treatment_effect_ts_temp = train_test_split(
        X, t_indirect, t, y, treatment_effect, X_ts, t_indirect_ts, t_ts, y_ts, treatment_effect_ts, 
        test_size=0.4, random_state=42
    )

    X_val, X_test, t_indirect_val, t_indirect_test, t_val, t_test, y_val, y_test, treatment_effect_val, treatment_effect_test, X_ts_val, X_ts_test, t_indirect_ts_val, t_indirect_ts_test, t_ts_val, t_ts_test, y_ts_val, y_ts_test, treatment_effect_ts_val, treatment_effect_ts_test = train_test_split(
        X_temp, t_indirect_temp, t_temp, y_temp, treatment_effect_temp, X_ts_temp, t_indirect_ts_temp, t_ts_temp, y_ts_temp, treatment_effect_ts_temp,
        test_size=0.5, random_state=42
    )

    models = {
        'Linear': LinearRegression(),
        'Ridge': RidgeRegression(alpha=1)  # 알파 값을 크게 설정
    }
    results = {}
    
    for name, model in models.items():
        model.fit(np.c_[X_train, t_train], y_train)
        
        ite_pred_test = compute_ite(model, X_test)
        
        results[name] = {
            'PEHE': compute_pehe(treatment_effect_test, ite_pred_test),
            'ATE': compute_ate(treatment_effect_test, ite_pred_test)
        }
    
    return results

# Main execution
df = pd.read_csv('./syn_ts_data_with_treatment_effect.csv')
X, t_indirect, t, y, treatment_effect, X_ts, t_indirect_ts, t_ts, y_ts, treatment_effect_ts = prepare_data(df)

results = train_and_evaluate(X, t_indirect, t, y, treatment_effect, X_ts, t_indirect_ts, t_ts, y_ts, treatment_effect_ts)

for model_name, metrics in results.items():
    print(f"{model_name}:")
    print(f"  PEHE: {metrics['PEHE']:.4f}")
    print(f"  |ATE|: {metrics['ATE']:.4f}")