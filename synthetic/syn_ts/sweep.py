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

global_epochs=500

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
    optimizer = torch.optim.Adam(model.parameters())
    t = torch.sigmoid(torch.FloatTensor(t))
    for epoch in range(epochs):
        for i in range(0, len(X), batch_size):
            batch_X = torch.FloatTensor(X[i:i+batch_size])
            batch_t = t[i:i+batch_size].unsqueeze(1)
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
    def __init__(self, dim_x, seq_len, dim_z=20, dim_h=64, nhead=4, num_layers=2):
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
            batch_X = torch.FloatTensor(X[i:i+batch_size]).transpose(1, 2)  # (batch_size, seq_len, input_dim)
            batch_t_indirect = torch.FloatTensor(t_indirect[i:i+batch_size])
            batch_t = torch.FloatTensor(t[i:i+batch_size])
            batch_y = torch.FloatTensor(y[i:i+batch_size])
            
            optimizer.zero_grad()
            
            results = model(batch_X, batch_t_indirect, batch_t, batch_y)
            
            # Reconstruction loss
            recon_loss = torch.nn.functional.gaussian_nll_loss(
                results['p_x_mu'], batch_X.transpose(1, 2), results['p_x_logvar'].exp()
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
        X_tensor = torch.FloatTensor(X).transpose(1, 2)
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
    # Group by sample and calculate mean for all variables except CEVT
    df_mean = df.groupby('sample').mean().reset_index()
    
    X = df_mean[['Z', 'X1', 'X2', 'X3', 'X4']].values
    t_indirect = df_mean['Ti'].values
    t = df_mean['Td'].values
    y = df_mean['Y'].values
    treatment_effect = df_mean['treatment_effect'].values

    # Prepare time series data for CEVT
    X_ts = df[['Z', 'X1', 'X2', 'X3', 'X4']].values.reshape(-1, 5, 5)  # (n_samples, seq_len, n_features)
    t_indirect_ts = df['Ti'].values.reshape(-1, 5)
    t_ts = df['Td'].values.reshape(-1, 5)
    y_ts = df['Y'].values.reshape(-1, 5)
    treatment_effect_ts = df['treatment_effect'].values.reshape(-1,5)
    
    return X, t_indirect, t, y, treatment_effect, X_ts, t_indirect_ts, t_ts, y_ts, treatment_effect_ts

def train_model(model, X_train, t_train, y_train, X_val, t_val, y_val, lr, wd, epochs):
    if isinstance(model, (TARNet, DragonNet, CEVAE, CEVT)):
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        best_model = None
        
        for epoch in range(epochs):
            model.train()
            if isinstance(model, DragonNet):
                t_train = (torch.sigmoid(torch.FloatTensor(t_train))>= 0.5).float()
                y0, y1, prop = model(torch.FloatTensor(X_train))
                loss = criterion(y0[t_train==0].squeeze(), torch.FloatTensor(y_train[t_train==0])) + \
                       criterion(y1[t_train==1].squeeze(), torch.FloatTensor(y_train[t_train==1])) + \
                       nn.BCELoss()(prop.squeeze(), torch.FloatTensor(t_train))
            elif isinstance(model, TARNet):
                t_train = (torch.sigmoid(torch.FloatTensor(t_train))>= 0.5).float()
                y0, y1 = model(torch.FloatTensor(X_train))
                loss = criterion(y0[t_train==0].squeeze(), torch.FloatTensor(y_train[t_train==0])) + \
                       criterion(y1[t_train==1].squeeze(), torch.FloatTensor(y_train[t_train==1]))
            elif isinstance(model, CEVAE):
                t_train = (torch.sigmoid(torch.FloatTensor(t_train))>= 0.5).float()
                results = model(torch.FloatTensor(X_train), torch.FloatTensor(t_train), torch.FloatTensor(y_train))
                loss = criterion(results['y_pred_dec'], torch.FloatTensor(y_train))
            elif isinstance(model, CEVT):
                results = model(torch.FloatTensor(X_train).transpose(1, 2), torch.FloatTensor(t_train[:, :-1]), torch.FloatTensor(t_train[:, -1:]), torch.FloatTensor(y_train))
                loss = criterion(results['y_pred_dec'], torch.FloatTensor(y_train))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                if isinstance(model, DragonNet):
                    y0, y1, _ = model(torch.FloatTensor(X_val))
                elif isinstance(model, TARNet):
                    y0, y1 = model(torch.FloatTensor(X_val))
                elif isinstance(model, CEVAE):
                    results = model(torch.FloatTensor(X_val), torch.FloatTensor(t_val), torch.FloatTensor(y_val))
                    val_loss = criterion(results['y_pred_dec'], torch.FloatTensor(y_val))
                elif isinstance(model, CEVT):
                    results = model(torch.FloatTensor(X_val).transpose(1, 2), torch.FloatTensor(t_val[:, :-1]), torch.FloatTensor(t_val[:, -1:]), torch.FloatTensor(y_val))
                    val_loss = criterion(results['y_pred_dec'], torch.FloatTensor(y_val))
                
                if not isinstance(model, (CEVAE, CEVT)):
                    val_loss = 0
                    t_val = (torch.sigmoid(torch.FloatTensor(t_val))>= 0.5).float()
                    if (t_val == 0).any():
                        val_loss += criterion(y0[t_val==0].squeeze(), torch.FloatTensor(y_val[t_val==0]))
                    if (t_val == 1).any():
                        val_loss += criterion(y1[t_val==1].squeeze(), torch.FloatTensor(y_val[t_val==1]))
                    # val_loss = criterion(y0[t_val==0].squeeze(), torch.FloatTensor(y_val[t_val==0])) + \
                    #            criterion(y1[t_val==1].squeeze(), torch.FloatTensor(y_val[t_val==1]))
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model.state_dict()
        import pdb;pdb.set_trace()
        
        model.load_state_dict(best_model)
    else:  # For sklearn models
        model.fit(np.c_[X_train, t_train], y_train)
        best_model = model
    
    return best_model, best_val_loss

def hyperparameter_sweep(model_class, X_train, t_train, y_train, X_val, t_val, y_val):
    lr_range = [0.001, 0.01, 0.1]
    wd_range = [0.0001, 0.001, 0.01]
    epochs = 100
    
    best_model = None
    best_val_loss = float('inf')
    
    for lr in lr_range:
        for wd in wd_range:
            model = model_class(X_train.shape[1])  # Assuming the model takes input dimension as argument
            model, val_loss = train_model(model, X_train, t_train, y_train, X_val, t_val, y_val, lr, wd, epochs)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
    
    return best_model

def train_and_evaluate(X_train, t_train, y_train, X_val, t_val, y_val, X_test, t_test, ite_true):
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(),
        'MLP': MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000),
        'TARNet': TARNet,
        'DragonNet': DragonNet,
        'CEVAE': CEVAE,
        'CEVT': CEVT
    }
    
    results = {}
    
    for name, model_class in models.items():
        if name in ['Linear', 'Ridge', 'MLP']:
            best_model = model_class
            best_model.fit(np.c_[X_train, t_train], y_train)
        else:
            best_model = hyperparameter_sweep(model_class, X_train, t_train, y_train, X_val, t_val, y_val)
        
        if name == 'CEVT':
            ite_pred = compute_ite_cevt(best_model, X_test, t_test[:, :-1])
        else:
            ite_pred = compute_ite(best_model, X_test)
        
        results[name] = {
            'PEHE': compute_pehe(ite_true, ite_pred),
            'ATE': compute_ate(ite_true, ite_pred)
        }
    
    return results

# Main execution
df = pd.read_csv('./syn_ts_data_with_treatment_effect.csv')
X, t_indirect, t, y, treatment_effect, X_ts, t_indirect_ts, t_ts, y_ts, treatment_effect_ts = prepare_data(df)

# Split data into train, validation, and test sets
X_train_val, X_test, t_train_val, t_test, y_train_val, y_test, treatment_effect_train_val, treatment_effect_test = train_test_split(
    X, t, y, treatment_effect, test_size=0.2, random_state=42
)

X_train, X_val, t_train, t_val, y_train, y_val, treatment_effect_train, treatment_effect_val = train_test_split(
    X_train_val, t_train_val, y_train_val, treatment_effect_train_val, test_size=0.25, random_state=42
)

# Train and evaluate models
results = train_and_evaluate(X_train, t_train, y_train, X_val, t_val, y_val, X_test, t_test, treatment_effect_test)

# Print results
for model_name, metrics in results.items():
    print(f"{model_name}:")
    print(f"  PEHE: {metrics['PEHE']:.4f}")
    print(f"  |ATE|: {metrics['ATE']:.4f}")