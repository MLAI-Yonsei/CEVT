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
import copy
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
global_epochs=300

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


class Transformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, seq_len, num_outputs=1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.encoder = TransformerEncoder(d_model, d_model, nhead, num_layers)
        self.regression_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.linear_head = nn.Linear(d_model, num_outputs)
        self.seq_len = seq_len

    def forward(self, x, t):
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        
        x = x + t.unsqueeze(-1).expand(-1, -1, x.size(-1))  # [batch_size, seq_len, d_model]
        
        regression_tokens = self.regression_token.expand(x.size(0), -1, -1)
        x = torch.cat([regression_tokens, x], dim=1)  # [batch_size, seq_len+1, d_model]
        
        x = self.encoder(x)  # [batch_size, seq_len+1, d_model]
        
        return self.linear_head(x[:, 0, :])

class iTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, seq_len):
        super(iTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, 1)
        self.q_t = nn.Linear(1, d_model)  

    def forward(self, x, t):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        
        t_embedded = self.q_t(t.unsqueeze(-1)) 
        
        x = self.transformer_encoder(x)
        
        last_token = x[:, -1, :]
        output = self.output_layer(last_token)
        
        return output

def train_transformer(model, X, t, y, epochs=100, batch_size=32):
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    device = next(model.parameters()).device
    
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X), batch_size):
            batch_X = torch.FloatTensor(X[i:i+batch_size]).to(device)
            batch_t = torch.FloatTensor(t[i:i+batch_size]).to(device)
            batch_y = torch.FloatTensor(y[i:i+batch_size]).to(device)

            optimizer.zero_grad()
            outputs = model(batch_X, batch_t)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()

def compute_ite_transformer(model, X):
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        t0_tensor = torch.zeros_like(X_tensor.mean(-1)).cuda()
        t1_tensor = torch.ones_like(X_tensor.mean(-1)).cuda()
        
        y0_pred = model(X_tensor, t0_tensor)
        y1_pred = model(X_tensor, t1_tensor)
        
        ite = y1_pred - y0_pred

    return ite.cpu().numpy().flatten()

class CEVAE(nn.Module):
    def __init__(self, dim_x, dim_z=200, d_model=200):
        super().__init__()
        self.dim_z = dim_z
        
        # Encoder networks
        self.q_t = nn.Sequential(nn.Linear(dim_x, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.q_y = nn.Sequential(nn.Linear(dim_x + 1, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.q_z = nn.Sequential(nn.Linear(dim_x + 2, d_model), nn.ReLU(), nn.Linear(d_model, 2 * dim_z))
        
        # Decoder networks
        self.p_x = nn.Sequential(nn.Linear(dim_z, d_model), nn.ReLU(), nn.Linear(d_model, 2 * dim_x))
        self.p_t = nn.Sequential(nn.Linear(dim_z, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.p_y_t0 = nn.Sequential(nn.Linear(dim_z, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.p_y_t1 = nn.Sequential(nn.Linear(dim_z, d_model), nn.ReLU(), nn.Linear(d_model, 1))

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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.01)
    for epoch in range(epochs):
        for i in range(0, len(X), batch_size):
            batch_X = torch.FloatTensor(X[i:i+batch_size]).to(device)
            batch_t = torch.FloatTensor(t[i:i+batch_size]).to(device).unsqueeze(1)
            batch_y = torch.FloatTensor(y[i:i+batch_size]).to(device).unsqueeze(1)
            
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
    def __init__(self, input_dim, d_model, nhead, num_layers, max_len=5000):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        return self.transformer_encoder(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model).cuda()
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# class customTransformerEncoder(nn.TransformerEncoder):
#     def __init__(self, encoder_layer, num_layers, d_model, seq_len, norm=None):
#         super(customTransformerEncoder, self).__init__(encoder_layer, num_layers, norm)
#         self.d_model = d_model
#         self.x2t1 = nn.Linear(d_model, seq_len)  # Predict indirect treatment
#         self.xt12t2 = nn.Linear(d_model, seq_len)  # Predict direct treatment
#         self.xt2yd = nn.Linear(d_model, seq_len)  # Predict outcome

#     def forward(self, src, mask=None, src_key_padding_mask=None, val_len=None, intervene_t=None):
#         output = src

#         for i, mod in enumerate(self.layers):
#             output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            
#             if i == 0:
#                 # Predict and integrate indirect treatment (t1)
#                 if mask is not None:
#                     emb = output[torch.arange(output.size(0)), val_len - 1]
#                 else:
#                     val_mask = torch.arange(output.size(1))[None, :].to(output.device) < val_len[:, None]
#                     emb = (output * val_mask.unsqueeze(-1).float()).sum(1) / val_mask.sum(1).unsqueeze(-1).float()
#                 t1_pred = self.x2t1(emb)
#                 t1 = intervene_t[1] if intervene_t is not None and intervene_t[0] == 't1' else t1_pred
#                 output = output + t1.unsqueeze(-1) * output
            
#             elif i == 1:
#                 # Predict and integrate direct treatment (t2)
#                 if mask is not None:
#                     emb = output[torch.arange(output.size(0)), val_len - 1]
#                 else:
#                     val_mask = torch.arange(output.size(1))[None, :].to(output.device) < val_len[:, None]
#                     emb = (output * val_mask.unsqueeze(-1).float()).sum(1) / val_mask.sum(1).unsqueeze(-1).float()
                
#                 t2_pred = torch.sigmoid(self.xt12t2(emb))
#                 t2 = intervene_t[1] if intervene_t is not None and intervene_t[0] == 't2' else t2_pred
#                 output = output + t2.unsqueeze(-1) * output

#         if self.norm is not None:
#             output = self.norm(output)

#         # Predict outcome
#         if mask is not None:
#             final_emb = output[torch.arange(output.size(0)), val_len - 1]
#         else:
#             val_mask = torch.arange(output.size(1))[None, :].to(output.device) < val_len[:, None]
#             final_emb = (output * val_mask.unsqueeze(-1).float()).sum(1) / val_mask.sum(1).unsqueeze(-1).float()
        
#         y = self.xt2yd(final_emb)

#         return output, (t1_pred, t2_pred), y

# class CEVT(nn.Module):
#     def __init__(self, input_dim, d_model, nhead, num_layers, seq_len):
#         super(CEVT, self).__init__()
#         self.input_dim = input_dim
#         self.d_model = d_model
#         self.seq_len = seq_len
        
#         self.embedding = nn.Linear(input_dim, d_model)
#         self.positional_encoding = PositionalEncoding(d_model, max_len=seq_len)
        
#         encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
#         self.transformer_encoder = customTransformerEncoder(encoder_layer, num_layers, d_model, seq_len)
        
#         self.fc_mu = nn.Linear(d_model, d_model)
#         self.fc_logvar = nn.Linear(d_model, d_model)
        
#         # Decoders for reconstruction
#         self.decoder_x = nn.Linear(d_model, input_dim)
#         self.decoder_y = nn.Linear(d_model, 1)  # Assuming y is a single value per timestep
#         self.decoder_t1 = nn.Linear(d_model, 1)  # For indirect treatment
#         self.decoder_t2 = nn.Linear(d_model, 1)  # For direct treatment

#     def forward(self, x, val_len, intervene_t=None):
#         batch_size, seq_len, _ = x.shape
        
#         x_embedded = self.embedding(x)
#         x_embedded = self.positional_encoding(x_embedded)
        
#         output, (t1, t2), y = self.transformer_encoder(x_embedded, val_len=val_len, intervene_t=intervene_t)
        
#         # Get the last valid output for each sequence
#         if self.transformer_encoder.layers[-1].self_attn is not None:
#             z = output[torch.arange(batch_size), val_len - 1]
#         else:
#             val_mask = torch.arange(seq_len)[None, :].to(x.device) < val_len[:, None]
#             z = (output * val_mask.unsqueeze(-1).float()).sum(1) / val_mask.sum(1).unsqueeze(-1).float()
        
#         z_mu = self.fc_mu(z)
#         z_logvar = self.fc_logvar(z)
        
#         z = self.reparameterize(z_mu, z_logvar)
        
#         # Reconstruct x, y, t1, and t2 for all timesteps
#         x_recon = self.decoder_x(output)
#         y_recon = self.decoder_y(output).squeeze(-1)
#         t1_recon = self.decoder_t1(output).squeeze(-1)
#         t2_recon = self.decoder_t2(output).squeeze(-1)
        
#         return x_recon, (y_recon+y)/2, (t1_recon+t1)/2, (t2_recon+t2)/2, z_mu, z_logvar
#         # return x_recon, y, t1, t2, z_mu, z_logvar

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

# def train_cevt(model, X, t_indirect, t, y, epochs=100, batch_size=32):
#     optimizer = torch.optim.Adam(model.parameters())
#     mse_loss = nn.MSELoss(reduction='none')
#     bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
#     for epoch in range(epochs):
#         model.train()
#         for i in range(0, len(X), batch_size):
#             batch_X = torch.FloatTensor(X[i:i+batch_size]).to(device)
#             batch_t_indirect = torch.FloatTensor(t_indirect[i:i+batch_size]).to(device)
#             batch_t = torch.FloatTensor(t[i:i+batch_size]).to(device)
#             batch_y = torch.FloatTensor(y[i:i+batch_size]).to(device)
            
#             val_len = torch.sum(~torch.isnan(batch_X).any(dim=-1), dim=1)
            
#             optimizer.zero_grad()
#             x_recon, y_recon, t1_recon, t2_recon, mu, logvar = model(batch_X, val_len)
            
#             # Compute losses
#             recon_loss_x = mse_loss(x_recon, batch_X).mean(dim=-1)
#             recon_loss_y = mse_loss(y_recon, batch_y)
#             recon_loss_t1 = mse_loss(t1_recon, batch_t_indirect)
#             recon_loss_t2 = bce_loss(t2_recon, batch_t)
            
#             # Create mask for valid timesteps
#             mask = torch.arange(batch_X.size(1)).unsqueeze(0).to(device) < val_len.unsqueeze(1)
            
#             # Apply mask to losses
#             recon_loss = (recon_loss_x + recon_loss_y + recon_loss_t1 + recon_loss_t2) * mask.float()
#             recon_loss = recon_loss.sum() / mask.sum()
            
#             # KL divergence
#             kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_X.size(0)
            
#             # Total loss
#             loss = recon_loss + 0.1 * kl_loss
            
#             loss.backward()
#             optimizer.step()

# def compute_ite_cevt(model, X, t_indirect):
#     model.eval()
#     with torch.no_grad():
#         X_tensor = torch.FloatTensor(X).to(device)
#         t_indirect_tensor = torch.FloatTensor(t_indirect).to(device)
#         val_len = torch.sum(~torch.isnan(X_tensor).any(dim=-1), dim=1)
        
#         # Compute outcome for t=0
#         _, y_pred_0, _, _, _, _ = model(X_tensor, val_len, intervene_t=('t2', torch.zeros(t_indirect_tensor.shape[0]).unsqueeze(-1).cuda()))
        
#         # Compute outcome for t=1
#         _, y_pred_1, _, _, _, _ = model(X_tensor, val_len, intervene_t=('t2', torch.ones(t_indirect_tensor.shape[0]).unsqueeze(-1).cuda()))
        
#         # Compute ITE for each timestep
#         ite = y_pred_1 - y_pred_0
        
#         # Average ITE across valid timesteps
#         mask = torch.arange(X.shape[1]).unsqueeze(0).to(device) < val_len.unsqueeze(1)
#         ite = (ite * mask.float()).sum(dim=1) / val_len.float()
    
#     return ite.cpu().numpy()

class CEVT(nn.Module):
    def __init__(self, dim_x, seq_len, dim_z=10, d_model=16, nhead=4, num_layers=2):
        super().__init__()
        self.dim_z = dim_z
        self.seq_len = seq_len
        
        # Encoder networks
        self.encoder = TransformerEncoder(dim_x, d_model, nhead, num_layers)
        self.q_t_indirect = nn.Linear(d_model * seq_len, 1)
        self.q_t = nn.Linear(d_model * seq_len + 1, 1)
        self.q_y = nn.Linear(d_model * seq_len + 2, 1)
        self.q_z = nn.Linear(d_model * seq_len + 3, 2 * dim_z)
        
        # Decoder networks
        self.p_x = nn.Sequential(nn.Linear(dim_z, d_model * seq_len), nn.ReLU(), nn.Linear(d_model * seq_len, 2 * dim_x * seq_len))
        self.p_t_indirect = nn.Sequential(nn.Linear(dim_z, d_model), nn.ReLU(), nn.Linear(d_model, seq_len))
        self.p_t = nn.Sequential(nn.Linear(dim_z + seq_len, d_model), nn.ReLU(), nn.Linear(d_model, seq_len))
        self.p_y_t0 = nn.Sequential(nn.Linear(dim_z + seq_len, d_model), nn.ReLU(), nn.Linear(d_model, seq_len))
        self.p_y_t1 = nn.Sequential(nn.Linear(dim_z + seq_len, d_model), nn.ReLU(), nn.Linear(d_model, seq_len))

    def encode(self, x, t_indirect=None, t=None, y=None):
        h = self.encoder(x)  
        h = h.transpose(0, 1).reshape(-1, self.seq_len * h.shape[-1])  # shape: (batch_size, seq_len * d_model)
        
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
            batch_X = torch.FloatTensor(X[i:i+batch_size]).to(device)  # (batch_size, seq_len, input_dim)
            batch_t_indirect = torch.FloatTensor(t_indirect[i:i+batch_size]).to(device)
            batch_t = torch.FloatTensor(t[i:i+batch_size]).to(device)
            batch_y = torch.FloatTensor(y[i:i+batch_size]).to(device)
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

def compute_ite(model, X):
    if isinstance(model, (TARNet, DragonNet)):
        model.eval()
        with torch.no_grad():
            if isinstance(model, DragonNet):
                y0, y1, _ = model(torch.FloatTensor(X).to(device))
            else:
                y0, y1 = model(torch.FloatTensor(X).to(device))
        return (y1 - y0).cpu().numpy().flatten()
    elif isinstance(model, CEVAE):
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            results_t0 = model(X_tensor, torch.zeros(X.shape[0], 1).cuda())
            results_t1 = model(X_tensor, torch.ones(X.shape[0], 1).cuda())
            ite = results_t1['y_t1'] - results_t0['y_t0']
        return ite.cpu().numpy().flatten()
    elif isinstance(model, (LinearRegression, Ridge, MLPRegressor)):
        t_0 = np.zeros((X.shape[0], 1))
        t_1 = np.ones((X.shape[0], 1))
        y_0 = model.predict(np.c_[X, t_0])
        y_1 = model.predict(np.c_[X, t_1])
        return y_1 - y_0
    
def compute_ite_cevt(model, X, t_indirect):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        t_indirect_tensor = torch.FloatTensor(t_indirect).to(device)
        
        results_t0 = model(X_tensor, t_indirect_tensor, torch.zeros_like(t_indirect_tensor).cuda())
        y_t0 = results_t0['y_t0']
        
        results_t1 = model(X_tensor, t_indirect_tensor, torch.ones_like(t_indirect_tensor).cuda())
        y_t1 = results_t1['y_t1']
        
        ite = y_t1 - y_t0
        
    return ite.mean(dim=1).cpu().numpy() 

def compute_pehe(ite_true, ite_pred):
    return np.sqrt(np.mean((ite_true - ite_pred)**2))

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
    X_ts = df[['X1', 'X2', 'X3', 'X4']].values.reshape(-1, seq_len, 4)  
    t_indirect_ts = df['Ti'].values.reshape(-1, seq_len)
    t_ts = df['Td'].values.reshape(-1, seq_len)
    y_ts = df['Y'].values.reshape(-1, seq_len)
    treatment_effect_ts = df['treatment_effect'].values.reshape(-1,seq_len)
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
        'Ridge': Ridge(alpha=10),
        # 'MLP': MLPRegressor(hidden_layer_sizes=(200, 200), max_iter=global_epochs, alpha=1),
        # 'Transformer': Transformer(
        #     input_dim=X_ts_train.shape[2], 
        #     d_model=200, 
        #     nhead=2, 
        #     num_layers=1, 
        #     seq_len=X_ts_train.shape[1],
        #     num_outputs=1
        # ).to(device),        
        # 'iTransformer': iTransformer(
        #     input_dim=X_ts_train.shape[2], 
        #     d_model=200, 
        #     nhead=2, 
        #     num_layers=1, 
        #     seq_len=X_ts_train.shape[1]
        # ).to(device),
        # 'TARNet': TARNet(X_train.shape[1]).to(device),
        # 'DragonNet': DragonNet(X_train.shape[1]).to(device),
        # 'CEVAE': CEVAE(X_train.shape[1]).to(device),
        # 'CEVT': CEVT(
        #     X_ts_train.shape[2], 
        #     d_model=32, 
        #     nhead=2, 
        #     num_layers=3, 
        #     seq_len=X_ts_train.shape[1]
        # ).to(device)
    }
    results = {}
    
    for name, model in models.items():
        best_val_pehe = float('inf')
        best_model = None
        
        if name in ['iTransformer','Transformer', 'TARNet', 'DragonNet', 'CEVAE', 'CEVT']:
            for epoch in range(global_epochs):
                if name in ['TARNet', 'DragonNet']:
                    model.train()
                    optimizer = optim.Adam(model.parameters(), lr=1e-6, weight_decay=0.01)
                    criterion = nn.MSELoss()
                    
                    if name == 'DragonNet':
                        y0, y1, prop = model(torch.FloatTensor(X_train).to(device))
                        loss = 0
                        if (t_train == 0).any():
                            loss += criterion(y0[t_train==0].squeeze(), torch.FloatTensor(y_train[t_train==0]).to(device))
                        if (t_train == 1).any():
                            loss += criterion(y1[t_train==1].squeeze(), torch.FloatTensor(y_train[t_train==1]).to(device))
                        loss += nn.BCELoss()(prop.squeeze(), torch.FloatTensor(t_train).to(device))
                    else:
                        y0, y1 = model(torch.FloatTensor(X_train).to(device))
                        loss = 0
                        if (t_train == 0).any():
                            loss += criterion(y0[t_train==0].squeeze(), torch.FloatTensor(y_train[t_train==0]).to(device))
                        if (t_train == 1).any():
                            loss += criterion(y1[t_train==1].squeeze(), torch.FloatTensor(y_train[t_train==1]).to(device))
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                elif name == 'CEVAE':
                    train_cevae(model, X_train, t_train, y_train, epochs=1)
                elif name == 'CEVT':
                    train_cevt(model, X_ts_train, t_indirect_ts_train, t_ts_train, y_ts_train, epochs=1)
                elif name == 'Transformer':
                    train_transformer(model, X_ts_train, t_ts_train, y_ts_train.mean(axis=1), epochs=1)
                elif name == 'iTransformer':
                    train_transformer(model, X_ts_train, t_ts_train, y_ts_train.mean(axis=1), epochs=1)
                
                # Validation
                if name == 'CEVT':
                    ite_pred_val = compute_ite_cevt(model, X_ts_val, t_indirect_ts_val)
                elif name == 'Transformer':
                    ite_pred_val = compute_ite_transformer(model, X_ts_val)
                elif name == 'iTransformer':
                    ite_pred_val = compute_ite_transformer(model, X_ts_val)
                else:
                    ite_pred_val = compute_ite(model, X_val)
                
                val_pehe = compute_pehe(treatment_effect_val, ite_pred_val)
                
                if val_pehe < best_val_pehe:
                    best_val_pehe = val_pehe
                    best_model = copy.deepcopy(model)
        else:
            model.fit(np.c_[X_train, t_train], y_train)
            best_model = model
        
        # Test with best model
        if name == 'CEVT':
            ite_pred_test = compute_ite_cevt(best_model, X_ts_test, t_indirect_ts_test)
        elif name == 'Transformer':
            ite_pred_test = compute_ite_transformer(best_model, X_ts_test)
        elif name == 'iTransformer':
            ite_pred_test = compute_ite_transformer(best_model, X_ts_test)
        else:
            ite_pred_test = compute_ite(best_model, X_test)
        
        results[name] = {
            'PEHE': compute_pehe(treatment_effect_test, ite_pred_test),
            'ATE': compute_ate(treatment_effect_test, ite_pred_test)
        }
    
    return results

# Update main execution code
df = pd.read_csv('./syn_ts_data_with_treatment_effect.csv')  # Load your data
X, t_indirect, t, y, treatment_effect, X_ts, t_indirect_ts, t_ts, y_ts, treatment_effect_ts = prepare_data(df)

# Train and evaluate models
results = train_and_evaluate(X, t_indirect, t, y, treatment_effect, X_ts, t_indirect_ts, t_ts, y_ts, treatment_effect_ts)

# Print results
for model_name, metrics in results.items():
    print(f"{model_name}:")
    print(f"  PEHE: {metrics['PEHE']:.4f}")
    print(f"  |ATE|: {metrics['ATE']:.4f}")