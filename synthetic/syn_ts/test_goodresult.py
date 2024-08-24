import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Bernoulli
import torch.nn.functional as F
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import random
import os
global_epochs=500

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
seed_everything(42)
# TARNet 모델 정의
class TARNet(nn.Module):
    def __init__(self, input_dim):
        super(TARNet, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU()
        )
        self.t0 = nn.Linear(200, 1)
        self.t1 = nn.Linear(200, 1)
    
    def forward(self, x):
        h = self.shared(x)
        return self.t0(h), self.t1(h)

# DragonNet 모델 정의
class DragonNet(nn.Module):
    def __init__(self, input_dim):
        super(DragonNet, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU()
        )
        self.t0 = nn.Linear(200, 1)
        self.t1 = nn.Linear(200, 1)
        self.prop = nn.Linear(200, 1)
    
    def forward(self, x):
        h = self.shared(x)
        return self.t0(h), self.t1(h), torch.sigmoid(self.prop(h))


class CEVAE(nn.Module):
    def __init__(self, dim_x, dim_z=20, dim_h=20):
        super().__init__()
        self.dim_z = dim_z
        
        # Encoder networks
        self.q_t = nn.Sequential(nn.Linear(dim_x, dim_h), nn.ReLU(), nn.Linear(dim_h, 1))
        self.q_y = nn.Sequential(nn.Linear(dim_x + 1, dim_h), nn.ReLU(), nn.Linear(dim_h, 1))
        self.q_z = nn.Sequential(nn.Linear(dim_x + 2, dim_h), nn.ReLU(), nn.Linear(dim_h, 2 * dim_z))
        
        # Decoder networks
        self.p_x = nn.Sequential(nn.Linear(dim_z, dim_h), nn.ReLU(), nn.Linear(dim_h, 2 * dim_x))
        self.p_t = nn.Sequential(nn.Linear(dim_z, dim_h), nn.ReLU(), nn.Linear(dim_h, 1))
        self.p_y_t0 = nn.Sequential(nn.Linear(dim_z, dim_h), nn.ReLU(), nn.Linear(dim_h, 1))
        self.p_y_t1 = nn.Sequential(nn.Linear(dim_z, dim_h), nn.ReLU(), nn.Linear(dim_h, 1))

    def encode(self, x, t=None, y=None):
        t_pred = torch.sigmoid(self.q_t(x))
        t = t_pred if t is None else t
        y_pred = self.q_y(torch.cat([x, t], dim=1))
        y = y_pred if y is None else y
        q_z_params = self.q_z(torch.cat([x, t, y], dim=1))
        q_z_mu, q_z_logvar = q_z_params.chunk(2, dim=1)
        return t_pred, y_pred, q_z_mu, q_z_logvar

    def decode(self, z, t=None):
        p_x_params = self.p_x(z)
        p_x_mu, p_x_logvar = p_x_params.chunk(2, dim=1)
        p_t = torch.sigmoid(self.p_t(z))
        t = p_t if t is None else t
        y_t0 = self.p_y_t0(z)
        y_t1 = self.p_y_t1(z)
        y = (1 - t) * y_t0 + t * y_t1
        return p_x_mu, p_x_logvar, p_t, y, y_t0, y_t1

    def forward(self, x, t=None, y=None):
        t_pred, y_pred, q_z_mu, q_z_logvar = self.encode(x, t, y)
        z = self.reparameterize(q_z_mu, q_z_logvar)
        p_x_mu, p_x_logvar, p_t, y_pred_dec, y_t0, y_t1 = self.decode(z, t)
        return {
            't_pred': t_pred, 'y_pred': y_pred, 
            'q_z_mu': q_z_mu, 'q_z_logvar': q_z_logvar,
            'p_x_mu': p_x_mu, 'p_x_logvar': p_x_logvar, 
            'p_t': p_t, 'y_pred_dec': y_pred_dec,
            'y_t0': y_t0, 'y_t1': y_t1
        }

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


def train_cevae(model, X, t, y, epochs=100, batch_size=100):
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.01)
    for epoch in range(epochs):
        for i in range(0, len(X), batch_size):
            batch_X = torch.FloatTensor(X[i:i+batch_size])
            batch_t = torch.FloatTensor(t[i:i+batch_size]).unsqueeze(1)
            batch_y = torch.FloatTensor(y[i:i+batch_size]).unsqueeze(1)
            
            optimizer.zero_grad()
            
            results = model(batch_X, batch_t, batch_y)
            
            # Reconstruction loss
            recon_loss = torch.nn.functional.gaussian_nll_loss(results['p_x_mu'], batch_X, results['p_x_logvar'].exp())
            
            # Treatment prediction loss
            t_loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(results['p_t']), batch_t)
            
            # Outcome prediction loss
            y_loss = torch.nn.functional.mse_loss(results['y_pred_dec'], batch_y)
            
            # KL divergence
            kl_div = -0.5 * torch.sum(1 + results['q_z_logvar'] - results['q_z_mu'].pow(2) - results['q_z_logvar'].exp())
            
            # Total loss
            loss = recon_loss + t_loss + y_loss + 0.1 * kl_div
            
            loss.backward()
            optimizer.step()

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        x = self.embedding(x)
        return self.transformer_encoder(x)


class CEVT(nn.Module):
    def __init__(self, dim_x, seq_len, dim_z=10, dim_h=16, nhead=4, num_layers=2):
        super().__init__()
        self.dim_z = dim_z
        self.seq_len = seq_len
        
        # Encoder networks
        self.encoder = TransformerEncoder(dim_x, dim_h, nhead, num_layers)
        self.q_t_indirect = nn.Linear(dim_h * seq_len, 1)
        self.q_t = nn.Linear(dim_h * seq_len + 1, 1)
        self.q_y = nn.Linear(dim_h * seq_len + 2, 1)
        self.q_z = nn.Linear(dim_h * seq_len + 3, 2 * dim_z)
        
        # Decoder networks
        self.p_x = nn.Sequential(nn.Linear(dim_z, dim_h * seq_len), nn.ReLU(), nn.Linear(dim_h * seq_len, 2 * dim_x * seq_len))
        self.p_t_indirect = nn.Sequential(nn.Linear(dim_z, dim_h), nn.ReLU(), nn.Linear(dim_h, seq_len))
        self.p_t = nn.Sequential(nn.Linear(dim_z + seq_len, dim_h), nn.ReLU(), nn.Linear(dim_h, seq_len))
        self.p_y_t0 = nn.Sequential(nn.Linear(dim_z + seq_len, dim_h), nn.ReLU(), nn.Linear(dim_h, seq_len))
        self.p_y_t1 = nn.Sequential(nn.Linear(dim_z + seq_len, dim_h), nn.ReLU(), nn.Linear(dim_h, seq_len))

    def encode(self, x, t_indirect=None, t=None, y=None):
        h = self.encoder(x)  
        h = h.transpose(0, 1).reshape(-1, self.seq_len * h.shape[-1])  # shape: (batch_size, seq_len * dim_h)
        
        t_indirect_pred = self.q_t_indirect(h)
        t_indirect = t_indirect_pred if t_indirect is None else t_indirect.mean(dim=1, keepdim=True)
        
        t_pred = torch.sigmoid(self.q_t(torch.cat([h, t_indirect], dim=1)))
        t = t_pred if t is None else t.mean(dim=1, keepdim=True)
        
        y_pred = self.q_y(torch.cat([h, t_indirect, t], dim=1))
        y = y_pred if y is None else y.mean(dim=1, keepdim=True)
        
        q_z_params = self.q_z(torch.cat([h, t_indirect, t, y], dim=1))
        q_z_mu, q_z_logvar = q_z_params.chunk(2, dim=1)
        return t_indirect_pred, t_pred, y_pred, q_z_mu, q_z_logvar

    def decode(self, z, t_indirect=None, t=None):
        p_x_params = self.p_x(z)
        p_x_mu, p_x_logvar = p_x_params.chunk(2, dim=1)
        p_x_mu = p_x_mu.view(-1, self.seq_len, p_x_mu.shape[1] // self.seq_len)
        p_x_logvar = p_x_logvar.view(-1, self.seq_len, p_x_logvar.shape[1] // self.seq_len)
        
        p_t_indirect = self.p_t_indirect(z)
        t_indirect = p_t_indirect if t_indirect is None else t_indirect
        
        p_t = torch.sigmoid(self.p_t(torch.cat([z, t_indirect], dim=1)))
        t = p_t if t is None else t
        
        y_t0 = self.p_y_t0(torch.cat([z, t_indirect], dim=1))
        y_t1 = self.p_y_t1(torch.cat([z, t_indirect], dim=1))
        y = (1 - t) * y_t0 + t * y_t1
        return p_x_mu, p_x_logvar, p_t_indirect, p_t, y, y_t0, y_t1

    def forward(self, x, t_indirect=None, t=None, y=None):
        t_indirect_pred, t_pred, y_pred, q_z_mu, q_z_logvar = self.encode(x, t_indirect, t, y)
        z = self.reparameterize(q_z_mu, q_z_logvar)
        p_x_mu, p_x_logvar, p_t_indirect, p_t, y_pred_dec, y_t0, y_t1 = self.decode(z, t_indirect, t)
        return {
            't_indirect_pred': t_indirect_pred, 't_pred': t_pred, 'y_pred': y_pred, 
            'q_z_mu': q_z_mu, 'q_z_logvar': q_z_logvar,
            'p_x_mu': p_x_mu, 'p_x_logvar': p_x_logvar, 
            'p_t_indirect': p_t_indirect, 'p_t': p_t, 'y_pred_dec': y_pred_dec,
            'y_t0': y_t0, 'y_t1': y_t1
        }

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

def train_cevt(model, X, t_indirect, t, y, epochs=100, batch_size=100):
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(epochs):
        for i in range(0, len(X), batch_size):
            batch_X = torch.FloatTensor(X[i:i+batch_size])  # (batch_size, seq_len, input_dim)
            batch_t_indirect = torch.FloatTensor(t_indirect[i:i+batch_size])
            batch_t = torch.FloatTensor(t[i:i+batch_size])
            batch_y = torch.FloatTensor(y[i:i+batch_size])
            optimizer.zero_grad()
            
            results = model(batch_X, batch_t_indirect, batch_t, batch_y)
            
            # Reconstruction loss
            recon_loss = torch.nn.functional.gaussian_nll_loss(
                results['p_x_mu'], batch_X, results['p_x_logvar'].exp()
            ).mean()
            
            # Treatment prediction losses
            t_indirect_loss = torch.nn.functional.mse_loss(results['p_t_indirect'], batch_t_indirect)
            t_loss = torch.nn.functional.mse_loss(results['p_t'], batch_t)
            
            # Outcome prediction loss
            y_loss = torch.nn.functional.mse_loss(results['y_pred_dec'], batch_y)
            
            # KL divergence
            kl_div = -0.5 * torch.sum(1 + results['q_z_logvar'] - results['q_z_mu'].pow(2) - results['q_z_logvar'].exp())
            
            # Total loss
            loss = recon_loss + t_indirect_loss + t_loss + y_loss + 0.1 * kl_div
            loss.backward()
            optimizer.step()

# ITE 계산 함수 수정

def compute_ite(model, X):
    if isinstance(model, (TARNet, DragonNet)):
        model.eval()
        with torch.no_grad():
            if isinstance(model, DragonNet):
                y0, y1, _ = model(torch.FloatTensor(X))
            else:
                y0, y1 = model(torch.FloatTensor(X))
        return (y1 - y0).numpy().flatten()
    elif isinstance(model, CEVAE):
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            results_t0 = model(X_tensor, torch.zeros(X.shape[0], 1))
            results_t1 = model(X_tensor, torch.ones(X.shape[0], 1))
            ite = results_t1['y_t1'] - results_t0['y_t0']
        return ite.numpy().flatten()
    elif isinstance(model, (LinearRegression, Ridge, MLPRegressor)):
        t_0 = np.zeros((X.shape[0], 1))
        t_1 = np.ones((X.shape[0], 1))
        y_0 = model.predict(np.c_[X, t_0])
        y_1 = model.predict(np.c_[X, t_1])
        return y_1 - y_0
    
def compute_ite_cevt(model, X, t_indirect):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        t_indirect_tensor = torch.FloatTensor(t_indirect)
        
        # t=0일 때의 결과
        results_t0 = model(X_tensor, t_indirect_tensor, torch.zeros_like(t_indirect_tensor))
        y_t0 = results_t0['y_t0']
        
        # t=1일 때의 결과
        results_t1 = model(X_tensor, t_indirect_tensor, torch.ones_like(t_indirect_tensor))
        y_t1 = results_t1['y_t1']
        
        # ITE 계산
        ite = y_t1 - y_t0
        
    return ite.mean(dim=1).numpy() # TODO
    # return ite.numpy() # TODO

# PEHE 계산 함수
def compute_pehe(ite_true, ite_pred):
    return np.sqrt(np.mean((ite_true - ite_pred)**2))

# ATE 계산 함수
def compute_ate(ite_true, ite_pred):
    return np.mean(np.abs(ite_true - ite_pred))        

def prepare_data(df):
    seq_len = len(df['time'].unique())
    # Group by sample and calculate mean for all variables except CEVT
    df_mean = df.groupby('sample').mean().reset_index()
    X = df_mean[['X1', 'X2', 'X3', 'X4']].values # remove Z
    t_indirect = df_mean['Ti'].values
    # t = df_mean['Td'].values
    t = (df_mean['Td'].values>0.5).astype(float)
    y = df_mean['Y'].values
    treatment_effect = df_mean['treatment_effect'].values

    # Prepare time series data for CEVT
    X_ts = df[['X1', 'X2', 'X3', 'X4']].values.reshape(-1, seq_len, 4)  # (n_samples, seq_len, n_features)
    t_indirect_ts = df['Ti'].values.reshape(-1, seq_len)
    t_ts = df['Td'].values.reshape(-1, seq_len)
    y_ts = df['Y'].values.reshape(-1, seq_len)
    treatment_effect_ts = df['treatment_effect'].values.reshape(-1,seq_len)
    # import pdb;pdb.set_trace()
    return X, t_indirect, t, y, treatment_effect, X_ts, t_indirect_ts, t_ts, y_ts, treatment_effect_ts


# 모델 학습 및 평가 함수
def train_and_evaluate(X_train, t_indirect_train, t_train, y_train, X_test, t_indirect_test, t_test, ite_true, X_ts_train, t_indirect_ts_train, t_ts_train, y_ts_train, X_ts_test, t_indirect_ts_test):
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=0.5),
        'MLP': MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=global_epochs),
        'TARNet': TARNet(X_train.shape[1]),
        'DragonNet': DragonNet(X_train.shape[1]),
        'CEVAE': CEVAE(X_train.shape[1]),
        'CEVT': CEVT(X_ts_train.shape[2], X_ts_train.shape[1])  # dim_x, seq_len
    }
    
    results = {}
    
    for name, model in models.items():
        if name in ['TARNet', 'DragonNet']:
            model.train()
            optimizer = optim.Adam(model.parameters())
            criterion = nn.MSELoss()
            for _ in range(global_epochs):  # epochs
                if name == 'DragonNet':
                    y0, y1, prop = model(torch.FloatTensor(X_train))
                    loss = 0
                    if (t_train == 0).any():
                        loss += criterion(y0[t_train==0].squeeze(), torch.FloatTensor(y_train[t_train==0]))
                    if (t_train == 1).any():
                        loss += criterion(y1[t_train==1].squeeze(), torch.FloatTensor(y_train[t_train==1]))
                    loss += nn.BCELoss()(prop.squeeze(), torch.FloatTensor(t_train))
                    # loss = criterion(y0[t_train==0].squeeze(), torch.FloatTensor(y_train[t_train==0])) + \
                    #        criterion(y1[t_train==1].squeeze(), torch.FloatTensor(y_train[t_train==1])) + \
                        #    nn.BCELoss()(prop.squeeze(), torch.FloatTensor(t_train))
                else:
                    y0, y1 = model(torch.FloatTensor(X_train))
                    loss = 0
                    if (t_train == 0).any():
                        loss += criterion(y0[t_train==0].squeeze(), torch.FloatTensor(y_train[t_train==0]))
                    if (t_train == 1).any():
                        loss += criterion(y1[t_train==1].squeeze(), torch.FloatTensor(y_train[t_train==1]))
                    # loss = criterion(y0[t_train==0].squeeze(), torch.FloatTensor(y_train[t_train==0])) + \
                    #        criterion(y1[t_train==1].squeeze(), torch.FloatTensor(y_train[t_train==1]))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        elif name == 'CEVAE':
            train_cevae(model, X_train, t_train, y_train, epochs=global_epochs)
        elif name == 'CEVT':
            # X_ts_train, t_indirect_ts_train, t_ts_train, y_ts_train, X_ts_test, t_indirect_ts_test
            train_cevt(model, X_ts_train, t_indirect_ts_train, t_ts_train, y_ts_train, epochs=global_epochs)
        else:
            model.fit(np.c_[X_train, t_train], y_train)
        
        if name == 'CEVT':
            # X_ts_train, t_indirect_ts_train, t_ts_train, y_ts_train, X_ts_test, t_indirect_ts_test
            ite_pred = compute_ite_cevt(model, X_ts_test, t_indirect_ts_test)
        else:
            ite_pred = compute_ite(model, X_test)
            
        results[name] = {
            'PEHE': compute_pehe(ite_true, ite_pred),
            'ATE': compute_ate(ite_true, ite_pred)
        } if name != 'CEVT' else {
            'PEHE': compute_pehe(ite_true, ite_pred),
            'ATE': compute_ate(ite_true, ite_pred)
        }
    
    return results
# Update main execution code
df = pd.read_csv('./syn_ts_data_with_treatment_effect.csv')  # Load your data
X, t_indirect, t, y, treatment_effect, X_ts, t_indirect_ts, t_ts, y_ts, treatment_effect_ts = prepare_data(df)

(X_train, X_test, 
 t_indirect_train, t_indirect_test, 
 t_train, t_test, 
 y_train, y_test, 
 treatment_effect_train, treatment_effect_test,
 X_ts_train, X_ts_test, 
 t_indirect_ts_train, t_indirect_ts_test, 
 t_ts_train, t_ts_test, 
 y_ts_train, y_ts_test, 
 treatment_effect_ts_train, treatment_effect_ts_test) = train_test_split(
    X, t_indirect, t, y, treatment_effect, 
    X_ts, t_indirect_ts, t_ts, y_ts, treatment_effect_ts, 
    test_size=0.3, random_state=42
)

# Real Label
ite_true = treatment_effect_test
# Train and evaluate models
results = train_and_evaluate(X_train, t_indirect_train, t_train, y_train, X_test, t_indirect_test, t_test, ite_true,
                             X_ts_train, t_indirect_ts_train, t_ts_train, y_ts_train, X_ts_test, t_indirect_ts_test)

# Print results
for model_name, metrics in results.items():
    print(f"{model_name}:")
    print(f"  PEHE: {metrics['PEHE']:.4f}")
    print(f"  |ATE|: {metrics['ATE']:.4f}")
