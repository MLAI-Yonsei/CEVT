import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd

import pickle
import random
import math
from torch.utils.data import Dataset
from torch.distributions import Normal
from collections import defaultdict
from prettytable import PrettyTable
## Data----------------------------------------------------------------------------------------
class Tabledata(Dataset):
    def __init__(self, args, data, scale='minmax', binary_t=False):
        self.use_treatment = args.use_treatment
        # padding tensors
        self.diff_tensor = torch.zeros([124,1])
        if args.use_treatment:
            if not args.single_treatment:
                self.cont_tensor = torch.zeros([124,3])
            else:
                self.cont_tensor = torch.zeros([124,4])
        else:
            self.cont_tensor = torch.zeros([124,5])

        self.cat_tensor = torch.zeros([124,7])
        yd=[]
        for _, group in data.groupby('cluster'):
            yd.append(group[['y', 'd']].tail(1))
        yd = pd.concat(yd)

        # Normalize continuous data
        for c in ["age", "dis", "danger", "CT_R", "CT_E"]:
            # dis : 0~6
            # danger : 3~11
            minmax_col(data, c) if scale == 'minmax' else meanvar_col(data, c)
        if scale == 'minmax':
            self.a_y, self.b_y = minmax_col(yd,"y")
            self.a_d, self.b_d = minmax_col(yd,"d")
        elif scale =='meanvar':
            self.a_y, self.b_y = meanvar_col(yd, "y")
            self.a_d, self.b_d = meanvar_col(yd, "d")

        ## Classify and store data by characteristics ##
        self.cluster = data.iloc[:,0].values.astype('float32')
        
        if not binary_t:
            self.treatment = data[['dis', 'danger']].values.astype('float32') if not args.single_treatment else data['danger'].values.astype('float32')
        else:
            raise('do not use binary t')
            print("use binary t")
            self.treatment = (data['dis'].values >= 0.5).astype('float32')
            
        if args.use_treatment:
            drop_col = ['dis'] if args.single_treatment else ['dis', 'danger']
            self.cont_X = data.iloc[:, 1:6].drop(columns=drop_col).values.astype('float32')
        else:
            self.cont_X = data.iloc[:, 1:6].values.astype('float32')
        
        self.cat_X = data.iloc[:, 6:13].astype('category')
        self.diff_days = data.iloc[:, 13].values.astype('float32')

        # y label tukey transformation
        # self.y = yd.values.astype('float32')
        y = torch.tensor(yd['y'].values.astype('float32'))
        d = torch.tensor(yd['d'].values.astype('float32'))
        if args.tukey:
            y = tukey_transformation(y, args)
            d = tukey_transformation(d, args)
        
        self.yd = torch.stack([y, d], dim=1)
        
        # Sort and store discrete data
        self.cat_cols = self.cat_X.columns
        self.cat_map = {col: {cat: i for i, cat in enumerate(self.cat_X[col].cat.categories)} for col in self.cat_cols}
        self.cat_X = self.cat_X.apply(lambda x: x.cat.codes)
        self.cat_X = torch.from_numpy(self.cat_X.to_numpy()).long()
    def __len__(self):
        return len(np.unique(self.cluster))

    def __getitem__(self, index):
        '''
            [batch x padding x embedding]
            cont_tensor_p: Padded patient-related continuous data 
            cont_tensor_c: Padded cluster-related continuous data 
            cat_tensor_p: Padded patient-related discrete data 
            cat_tensor_c: Padded cluster-related discrete data 
            data_len: Returns the number of valid patients per cluster 
            y: Correct answer label 
            diff_tensor: Returns the valid date per cluster
        '''
        diff_days = torch.from_numpy(self.diff_days[self.cluster == index]).unsqueeze(1)
        diff_tensor = self.diff_tensor.clone()
        diff_tensor[:diff_days.shape[0]] = diff_days
        cont_X = torch.from_numpy(self.cont_X[self.cluster == index])
        data_len = cont_X.shape[0]
        cont_tensor = self.cont_tensor.clone()
        cont_tensor[:cont_X.shape[0],] = cont_X
        cat_X = self.cat_X[self.cluster == index]
        cat_tensor = self.cat_tensor.clone()
        cat_tensor[:cat_X.shape[0],] = cat_X
        cat_tensor_p = cat_tensor[:, :5]
        cat_tensor_c = cat_tensor[:, 5:]
        cont_tensor_p = cont_tensor[:, :3]
        cont_tensor_c = cont_tensor[:, 3:]
        yd = self.yd[index]
        
        treatment = torch.mean(torch.tensor(self.treatment[self.cluster == index]), dim=0) # t1: dis|t2: danger
        return cont_tensor_p, cont_tensor_c, cat_tensor_p, cat_tensor_c, data_len, yd, diff_tensor, treatment
## MinMax Scaling Functions ------------------------------------
def minmax_col(data, name):
    minval , maxval = data[name].min(), data[name].max()
    data[name]=(data[name]-data[name].min())/(data[name].max()-data[name].min())
    return minval, maxval

def minmax_tensor(tensor):
    minvals = tensor.min()
    maxvals = tensor.max()
    
    normalized = (tensor - minvals) / (maxvals - minvals)
    return normalized, minvals, maxvals

def restore_minmax(data, minv, maxv):
    minv=0 if minv==None else minv
    maxv=0 if maxv==None else maxv
    data = (data * (maxv - minv)) + minv
    return data

## MinMax Scaling Functions ------------------------------------
def minmax_col(data, name):
    minval , maxval = data[name].min(), data[name].max()
    data[name]=(data[name]-data[name].min())/(data[name].max()-data[name].min())
    return minval, maxval

def minmax_tensor(tensor):
    minvals = tensor.min()
    maxvals = tensor.max()
    
    normalized = (tensor - minvals) / (maxvals - minvals)
    return normalized, minvals, maxvals

def restore_minmax(data, minv, maxv):
    minv=0 if minv==None else minv
    maxv=0 if maxv==None else maxv
    data = (data * (maxv - minv)) + minv
    return data
# ---------------------------------------------------------------

## Normalization Scaling Functions ---------------------------------
def meanvar_col(data, name):
    mean_val = data[name].mean()
    std_val = data[name].var()
    data[name]=(data[name]-data[name].mean())/data[name].var()
    return mean_val, std_val

def restore_meanvar(data, mean, var):
    data = data * var + mean
    return data
# ----------------------------------------------------------------



## Loss ----------------------------------------------------------------------------------------
class RMSELoss(nn.Module):
    def __init__(self, reduction):
        super(RMSELoss,self).__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        self.eps = 1e-12

    def forward(self, target, pred):
        x = torch.sqrt(self.mse(target, pred) + self.eps)
        return x
# ---------------------------------------------------------------------------------------------


## Train --------------------------------------------------------------------------------------
def train(args, data, model, optimizer, criterion, epoch, warmup_iter=0, lamb=0.0, aux_criterion=None, use_treatment=False, eval_criterion = None, scaling="minmax",a_y=None, b_y=None, a_d=None, b_d=None, pred_model="enc", binary_t=False, lambdas=[1,1,1]):
    eval_loss_y = None; eval_loss_d=None
    model.train()
    optimizer.zero_grad()
    batch_num, cont_p, cont_c, cat_p, cat_c, len, y, diff_days, *t = data_load(data)
    out = model(cont_p, cont_c, cat_p, cat_c, len, diff_days)
    eval_loss_t1 = None; eval_loss_t2 = None; pred_loss=0; kl_loss=0; recon_loss=0
    if use_treatment:
        gt_t = t[0] # t1: dis|t2: danger
        x, x_reconstructed, (enc_yd_pred, enc_t_pred), (dec_yd_pred, dec_t_pred), (z_mu, z_logvar) = out
        
        if args.model == 'cevt':
            loss, *ind_losses = cevt_loss(x_reconstructed, x, enc_t_pred, enc_yd_pred[:, 0], enc_yd_pred[:, 1], dec_t_pred, dec_yd_pred[:, 0], dec_yd_pred[:, 1], z_mu, z_logvar, gt_t, y[:,0] , y[:,1], criterion, lambdas, val_len=len)
        elif args.model == 'cevae':
            loss, *ind_losses = cevae_loss(x_reconstructed, x, enc_t_pred, enc_yd_pred[:, 0], enc_yd_pred[:, 1], dec_t_pred, dec_yd_pred[:, 0], dec_yd_pred[:, 1], z_mu, z_logvar, gt_t, y[:,0] , y[:,1], criterion, lambdas, val_len=len)
        (enc_loss_y, enc_loss_d), (dec_loss_y, dec_loss_d), (enc_loss_t, dec_loss_t), (pred_loss, kl_loss, recon_loss) = ind_losses

    else:
        loss_d = criterion(out[:,0], y[:,0])
        loss_y = criterion(out[:,1], y[:,1])    
        loss = loss_d + loss_y

    if eval_criterion != None:
        if use_treatment:
            if args.model=='cevt' or 'cevae':
                # enc loss
                enc_pred_y, enc_pred_d, gt_y, gt_d = reverse_scaling(scaling, enc_yd_pred, y, a_y, b_y, a_d, b_d)
                enc_eval_loss_y = eval_criterion(enc_pred_y, gt_y)
                enc_eval_loss_d = eval_criterion(enc_pred_d, gt_d)
                
                if args.single_treatment:
                    enc_eval_loss_t2 = eval_criterion(enc_t_pred[:,0].squeeze(), gt_t)
                    enc_eval_loss_t1 = torch.zeros_like(enc_eval_loss_t2)
                else:
                    enc_eval_loss_t1 = eval_criterion(enc_t_pred[:,0].squeeze(), gt_t[:,0])
                    enc_eval_loss_t2 = eval_criterion(enc_t_pred[:,1].squeeze(), gt_t[:,1])
                
                # dec loss
                dec_pred_y, dec_pred_d, gt_y, gt_d = reverse_scaling(scaling, dec_yd_pred, y, a_y, b_y, a_d, b_d)
                dec_eval_loss_y = eval_criterion(dec_pred_y, gt_y)
                dec_eval_loss_d = eval_criterion(dec_pred_d, gt_d)
                if args.single_treatment:
                    dec_eval_loss_t2 = eval_criterion(dec_t_pred[:,0].squeeze(), gt_t)
                    dec_eval_loss_t1 = torch.zeros_like(dec_eval_loss_t2)
                else:
                    dec_eval_loss_t1 = eval_criterion(dec_t_pred[:,0].squeeze(), gt_t[:,0])
                    dec_eval_loss_t2 = eval_criterion(dec_t_pred[:,1].squeeze(), gt_t[:,1])
                if enc_eval_loss_y + enc_eval_loss_d > dec_eval_loss_y + dec_eval_loss_d:
                    eval_loss_y, eval_loss_d = dec_eval_loss_y, dec_eval_loss_d
                    out = dec_yd_pred
                    loss_y = dec_loss_y
                    loss_d = dec_loss_d
                    eval_loss_t1 = dec_eval_loss_t1
                    eval_loss_t2 = dec_eval_loss_t2
                    eval_model = "Decoder"
                else:
                    eval_loss_y, eval_loss_d = enc_eval_loss_y, enc_eval_loss_d
                    out = enc_yd_pred
                    loss_y = enc_loss_y
                    loss_d = enc_loss_d
                    eval_loss_t1 = enc_eval_loss_t1
                    eval_loss_t2 = enc_eval_loss_t2
                    eval_model = "Encoder"
            elif args.model == 'iTransformer':
                pred_y, pred_d, gt_y, gt_d = reverse_scaling(scaling, out, y, a_y, b_y, a_d, b_d)
                eval_loss_y = eval_criterion(pred_y, gt_y)
                eval_loss_d = eval_criterion(pred_d, gt_d)
                eval_model = "nan"
        else:
            pred_y, pred_d, gt_y, gt_d = reverse_scaling(scaling, out, y, a_y, b_y, a_d, b_d)
            eval_loss_y = eval_criterion(pred_y, gt_y)
            eval_loss_d = eval_criterion(pred_d, gt_d)
            eval_model = "nan"
    # Add Penalty term for ridge regression
    if lamb != 0.0:
        loss += lamb * torch.norm(model.linear1.weight, p=2)
    if not torch.isnan(loss):
        loss.backward()
        optimizer.step()
        return loss_d.item(), loss_y.item(), batch_num, out, y, eval_loss_y, eval_loss_d, eval_model, (eval_loss_t1, eval_loss_t2), (pred_loss, kl_loss, recon_loss)
    else:
        # return 0, batch_num, out, y
        raise ValueError("Loss raised nan.")

## Validation --------------------------------------------------------------------------------
@torch.no_grad()
def valid(args, data, model, eval_criterion, scaling, a_y, b_y, a_d, b_d, use_treatment=False, MC_sample=1):
    model.eval()
    
    batch_num, cont_p, cont_c, cat_p, cat_c, len, y, diff_days, *rest = data_load(data)
    accumulated_outputs = [0] * 6  # (x, x_reconstructed, enc_yd_pred, enc_t_pred, dec_yd_pred, dec_t_pred)

    if use_treatment:
        if args.model =='cevt' or 'cevae':
            gt_t = rest[0]
            out = model(cont_p, cont_c, cat_p, cat_c, len, diff_days, is_MAP=True)
            x, x_reconstructed, (enc_yd_pred, enc_t_pred), (dec_yd_pred, dec_t_pred), (z_mu, z_logvar) = out
            
            # enc loss
            enc_pred_y, enc_pred_d, gt_y, gt_d = reverse_scaling(scaling, enc_yd_pred, y, a_y, b_y, a_d, b_d)
            enc_loss_y = eval_criterion(enc_pred_y, gt_y)
            enc_loss_d = eval_criterion(enc_pred_d, gt_d)
            
            if args.single_treatment:
                enc_loss_t2 = eval_criterion(enc_t_pred[:,0].squeeze(), gt_t)
                enc_loss_t1 = torch.zeros_like(enc_loss_t2)
            else:
                enc_loss_t1 = eval_criterion(enc_t_pred[:,0].squeeze(), gt_t[:,0])
                enc_loss_t2 = eval_criterion(enc_t_pred[:,1].squeeze(), gt_t[:,1])
            
            # dec loss
            dec_pred_y, dec_pred_d, gt_y, gt_d = reverse_scaling(scaling, dec_yd_pred, y, a_y, b_y, a_d, b_d)
            dec_loss_y = eval_criterion(dec_pred_y, gt_y)
            dec_loss_d = eval_criterion(dec_pred_d, gt_d)
            # dec_loss_t = eval_criterion(dec_t_pred.squeeze(), gt_t)
            
            if args.single_treatment:
                dec_loss_t2 = eval_criterion(dec_t_pred[:,0].squeeze(), gt_t)
                dec_loss_t1 = torch.zeros_like(dec_loss_t2)
            else:
                dec_loss_t1 = eval_criterion(dec_t_pred[:,0].squeeze(), gt_t[:,0])
                dec_loss_t2 = eval_criterion(dec_t_pred[:,1].squeeze(), gt_t[:,1])

            if enc_loss_y + enc_loss_d > dec_loss_y + dec_loss_d:
                loss_y, loss_d, loss_t1, loss_t2 = dec_loss_y, dec_loss_d, dec_loss_t1, dec_loss_t2
                out = dec_yd_pred
                eval_model = "Decoder"
            else:
                loss_y, loss_d, loss_t1, loss_t2 = enc_loss_y, enc_loss_d, enc_loss_t1, enc_loss_t2
                out = enc_yd_pred
                eval_model = "Encoder"
        elif args.model=='iTransformer':
            out = model(cont_p, cont_c, cat_p, cat_c, len, diff_days)
            
            pred_y, pred_d, gt_y, gt_d = reverse_scaling(scaling, out, y, a_y, b_y, a_d, b_d)
            loss_y = eval_criterion(pred_y, gt_y)
            loss_d = eval_criterion(pred_d, gt_d)
            
            loss = loss_y + loss_d
            
            if not torch.isnan(loss):
                return loss_d.item(), loss_y.item(), batch_num, out, y
            else:
                return 0, batch_num, out, y
    else:
        out = model(cont_p, cont_c, cat_p, cat_c, len, diff_days)
        pred_y, pred_d, gt_y, gt_d = reverse_scaling(scaling, out, y, a_y, b_y, a_d, b_d)
        loss_y = eval_criterion(pred_y, gt_y)
        loss_d = eval_criterion(pred_d, gt_d)
        eval_model = "nan"
        
    loss = loss_y + loss_d
    if not torch.isnan(loss):
        if use_treatment:
            return loss_d.item(), loss_y.item(), batch_num, out, y, eval_model, loss_t1, loss_t2
        else:
            return loss_d.item(), loss_y.item(), batch_num, out, y, eval_model
    else:
        return 0, batch_num, out, y

## Test ----------------------------------------------------------------------------------------
@torch.no_grad()
def test(args, data, model, scaling, a_y, b_y, a_d, b_d, use_treatment=False, MC_sample=1):
    
    criterion_mae = nn.L1Loss(reduction="sum")
    criterion_rmse = nn.MSELoss(reduction="sum")
    
    model.eval()

    batch_num, cont_p, cont_c, cat_p, cat_c, len, y, diff_days, *rest = data_load(data)
    out = model(cont_p, cont_c, cat_p, cat_c, len, diff_days)
    max_unique_tensor = torch.tensor([batch.unique().max() for batch in diff_days], device='cuda:0') + 1

    accumulated_outputs = [0] * 6  # (x, x_reconstructed, enc_yd_pred, enc_t_pred, dec_yd_pred, dec_t_pred)
    
    if use_treatment:
        gt_t = rest[0]
        if args.model=='cevt' or 'cevae':
            for i in range(MC_sample):
                out = model(cont_p, cont_c, cat_p, cat_c, len, diff_days)
                x, x_reconstructed, (enc_yd_pred, enc_t_pred), (dec_yd_pred, dec_t_pred), (z_mu, z_logvar) = out
                
                # accumulate predictions
                outputs = [x, x_reconstructed, enc_yd_pred, enc_t_pred, dec_yd_pred, dec_t_pred]
                accumulated_outputs = [accumulated + output for accumulated, output in zip(accumulated_outputs, outputs)]
            
            # calculate average
            avg_outputs = [accumulated / MC_sample for accumulated in accumulated_outputs]
            x, x_reconstructed, enc_yd_pred, enc_t_pred, dec_yd_pred, dec_t_pred = avg_outputs
            
            
            # enc loss
            enc_pred_y, enc_pred_d, gt_y, gt_d = reverse_scaling(scaling, enc_yd_pred, y, a_y, b_y, a_d, b_d)
            enc_loss_y = criterion_mae(enc_pred_y, gt_y)
            enc_loss_d = criterion_mae(enc_pred_d, gt_d)
            if args.single_treatment:
                enc_loss_t2 = criterion_mae(enc_t_pred[:,0].squeeze(), gt_t)
                enc_loss_t1 = torch.zeros_like(enc_loss_t2)
            else:
                enc_loss_t1 = criterion_mae(enc_t_pred[:,0].squeeze(), gt_t[:,0])
                enc_loss_t2 = criterion_mae(enc_t_pred[:,1].squeeze(), gt_t[:,1])
            
            # dec loss
            dec_pred_y, dec_pred_d, gt_y, gt_d = reverse_scaling(scaling, dec_yd_pred, y, a_y, b_y, a_d, b_d)
            dec_loss_y = criterion_mae(dec_pred_y, gt_y)
            dec_loss_d = criterion_mae(dec_pred_d, gt_d)
            # dec_loss_t = criterion_mae(dec_t_pred.squeeze(), gt_t)
            if args.single_treatment:
                dec_loss_t2 = criterion_mae(dec_t_pred[:,0].squeeze(), gt_t)
                dec_loss_t1 = torch.zeros_like(dec_loss_t2)
            else:
                dec_loss_t1 = criterion_mae(dec_t_pred[:,0].squeeze(), gt_t[:,0])
                dec_loss_t2 = criterion_mae(dec_t_pred[:,1].squeeze(), gt_t[:,1])

            if enc_loss_y + enc_loss_d > dec_loss_y + dec_loss_d:
                mae_y, mae_d, loss_t1, loss_t2 = dec_loss_y, dec_loss_d, dec_loss_t1, dec_loss_t2
                rmse_y, rmse_d = criterion_rmse(dec_pred_y, gt_y), criterion_rmse(dec_pred_d, gt_d)
                out = dec_yd_pred
                eval_model = "Decoder"
            else:
                mae_y, mae_d, loss_t1, loss_t2 = enc_loss_y, enc_loss_d, enc_loss_t1, enc_loss_t2
                rmse_y, rmse_d = criterion_rmse(enc_pred_y, gt_y), criterion_rmse(enc_pred_d, gt_d)
                out = enc_yd_pred
                eval_model = "Encoder"
            mae = mae_y + mae_d
            rmse = rmse_y + rmse_d
        elif args.model == 'iTransformer':
            yd_pred = model(cont_p, cont_c, cat_p, cat_c, len, diff_days)

            pred_y, pred_d, gt_y, gt_d = reverse_scaling(scaling, yd_pred, yd_true, a_y, b_y, a_d, b_d)
            
            # MAE
            mae_y = criterion_mae(pred_y, gt_y)
            mae_d = criterion_mae(pred_d, gt_d)
            mae = mae_y + mae_d
            
            # RMSE
            rmse_y = criterion_rmse(pred_y, gt_y)
            rmse_d = criterion_rmse(pred_d, gt_d)
            rmse = rmse_y + rmse_d
            
            if not torch.isnan(mae) and not torch.isnan(rmse):
                return mae_d.item(), mae_y.item(), rmse_d.item(), rmse_y.item(), batch_num, yd_pred, y
            else:
                return 0, batch_num, yd_pred, y
    else:
        out = model(cont_p, cont_c, cat_p, cat_c, len, diff_days)
        if out.shape == torch.Size([2]):
            out = out.unsqueeze(0)
        pred_y, pred_d, gt_y, gt_d = reverse_scaling(scaling, out, y, a_y, b_y, a_d, b_d)
        # MAE
        mae_y = criterion_mae(pred_y, gt_y)
        mae_d = criterion_mae(pred_d, gt_d)
        mae = mae_y + mae_d
        
        # RMSE
        rmse_y = criterion_rmse(pred_y, gt_y)
        rmse_d = criterion_rmse(pred_d, gt_d)
        rmse = rmse_y + rmse_d
        eval_model = "nan"
    
    if not torch.isnan(mae) and not torch.isnan(rmse):
        if use_treatment:
            return mae_d.item(), mae_y.item(), rmse_d.item(), rmse_y.item(), batch_num, out, y, loss_t1, loss_t2
        else:
            return mae_d.item(), mae_y.item(), rmse_d.item(), rmse_y.item(), batch_num, out, y
    else:
        return 0, batch_num, out, y

@torch.no_grad()
def CE(args, model, dataloader, intervene_var):
    model.eval()  
    data_points_y = []; data_points_d=[]
    
    for data in dataloader:
        _, cont_p, cont_c, cat_p, cat_c, val_len, y, diff_days, *rest = data_load(data)
        gt_t = rest[0]
        
        if args.use_treatment:
            if args.model == 'cevt':
                (x, diff_days, _), _ = model.embedding(cont_p, cont_c, cat_p, cat_c, val_len, diff_days)
                src_key_padding_mask = ~(torch.arange(x.size(1)).expand(x.size(0), -1).cuda() < val_len.unsqueeze(1)).cuda()
                src_mask = model.generate_square_subsequent_mask(x.size(1)).cuda() if model.unidir else None

                # use ground truth t instead of x2t_pred
                original_t = gt_t[:,0].unsqueeze(1) if intervene_var == 't1' else gt_t[:,1].unsqueeze(1)
                _, _, original_enc_yd = model.transformer_encoder(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask, val_len=val_len, intervene_t=(intervene_var,original_t))
                
                saved_original_t = original_t.clone()
                saved_original_enc_yd = original_enc_yd.clone()
                
                intervene_t_value_range = range(0, 61) if intervene_var == 't1' else range(30, 111)
                for intervene_t_value in [x * 0.1 for x in intervene_t_value_range]:
                    original_t=saved_original_t.clone()
                    original_enc_yd=saved_original_enc_yd.clone()
                    
                    
                    if intervene_var == 't1':
                        intervene_t_value = intervene_t_value / 6  # t1 norm [0, 6]
                    elif intervene_var == 't2':
                        intervene_t_value = (intervene_t_value - 3) / 8  # t2 norm [3, 11] 
                    
                    intervene_t = torch.full((x.size(0),), intervene_t_value, dtype=torch.float).unsqueeze(1).cuda()
                    _, _, intervene_enc_yd = model.transformer_encoder(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask, val_len=val_len, intervene_t=(intervene_var,intervene_t))
                    
                    delta_y = original_enc_yd - intervene_enc_yd
                    delta_t = (original_t - intervene_t)
                    
                    if intervene_var == 't1':
                        delta_t = delta_t*6  # denormalize 
                    elif intervene_var == 't2':
                        delta_t = delta_t*8  # +3 denormalize
                    
                        
                    delta_y, delta_d, _, _ = reverse_scaling(args.scaling, delta_y, y, dataloader.dataset.dataset.a_y, dataloader.dataset.dataset.b_y, dataloader.dataset.dataset.a_d, dataloader.dataset.dataset.b_d)
            
                    for i in range(delta_y.size(0)):
                        data_points_y.append((delta_t[i].item(), delta_y[i].item()))
                        data_points_d.append((delta_t[i].item(), delta_d[i].item()))
            if args.model == 'cevae': 
                x = model.embedding(cont_p, cont_c, cat_p, cat_c, val_len, diff_days)

                original_t = gt_t
                _, _, original_enc_yd, _ = model.encoder(x, t_gt=original_t)
                
                saved_original_t = original_t.clone()
                saved_original_enc_yd = original_enc_yd.clone()
                
                intervene_t_value_range = range(0, 61) if intervene_var == 't1' else range(30, 111)
                for intervene_t_value in [x * 0.1 for x in intervene_t_value_range]:
                    original_t=saved_original_t.clone()
                    original_enc_yd=saved_original_enc_yd.clone()
                    
                    if intervene_var == 't1':
                        intervene_t_value = intervene_t_value / 6  # t1 norm [0, 6]
                    elif intervene_var == 't2':
                        intervene_t_value = (intervene_t_value - 3) / 8  # t2 norm [3, 11] 

                    
                    intervene_t = torch.full((x.size(0),), intervene_t_value, dtype=torch.float).cuda()
                    _, _, intervene_enc_yd, _ = model.encoder(x, t_gt=intervene_t)
                    
                    delta_y = original_enc_yd - intervene_enc_yd
                    if intervene_var == 't1':
                        delta_t = (original_t - intervene_t)*6  # denormalize 
                    elif intervene_var == 't2':
                        delta_t = (original_t - intervene_t)*8  # +3 denormalize 
                    delta_y, delta_d, _, _ = reverse_scaling(args.scaling, delta_y, y, dataloader.dataset.dataset.a_y, dataloader.dataset.dataset.b_y, dataloader.dataset.dataset.a_d, dataloader.dataset.dataset.b_d)
            
                    for i in range(delta_y.size(0)):
                        data_points_y.append((delta_t[i].item(), delta_y[i].item()))
                        data_points_d.append((delta_t[i].item(), delta_d[i].item()))
        else:
            original_t = cont_c[:,:,0].clone() if intervene_var=='t1' else cont_c[:,:,1].clone()
            if args.model == 'cevt':
                _, _, (original_yd, _), (_, _), (_, _) = model(cont_p, cont_c, cat_p, cat_c, val_len, diff_days)
            else:
                original_yd = model(cont_p, cont_c, cat_p, cat_c, val_len, diff_days)
            original_yd = torch.clamp(original_yd, 0, 1)
            
            intervene_t_value_range = range(0, 61) if intervene_var == 't1' else range(30, 111)
            for intervene_t_value in [x * 0.1 for x in intervene_t_value_range]:
                
                if intervene_var == 't1':
                    intervene_t_value = intervene_t_value / 6  # t1 norm [0, 6]
                    cont_c[:,:,0] = intervene_t_value 
                elif intervene_var == 't2':
                    intervene_t_value = (intervene_t_value - 3) / 8  # t2 norm [3, 11] 
                    cont_c[:,:,1] = intervene_t_value 
        
                 
                intervene_yd = model(cont_p, cont_c, cat_p, cat_c, val_len, diff_days)
                
                # for fair comaparison
                intervene_yd = torch.clamp(intervene_yd, 0, 1)
                
                delta_y = original_yd - intervene_yd
                # delta_t = (original_t[:,0] - intervene_t_value)*6  # denormalize 
                if intervene_var == 't1':
                    delta_t = (original_t[:,0] - intervene_t_value)*6  # denormalize 
                elif intervene_var == 't2':
                    delta_t = (original_t[:,0] - intervene_t_value)*8  # +3 denormalize  
                
                delta_y, delta_d, _, _ = reverse_scaling(args.scaling, delta_y, y, dataloader.dataset.dataset.a_y, dataloader.dataset.dataset.b_y, dataloader.dataset.dataset.a_d, dataloader.dataset.dataset.b_d)
        
                for i in range(delta_y.size(0)):
                    data_points_y.append((delta_t[i].item(), delta_y[i].item()))
                    data_points_d.append((delta_t[i].item(), delta_d[i].item()))
    
    def calculate_gradients_and_effect(data_points, method='coef'):
        del_t = data_points[:, 0]  # delta_t
        del_var = data_points[:, 1]   # delta_y or delta_d

        non_zero_indices = del_t != 0
        del_t = del_t[non_zero_indices]
        del_var = del_var[non_zero_indices]
        
         
        gradients = del_var / del_t
        negative_acc = np.sum(gradients < 0) / len(gradients)
        
        treatment_effect = np.mean(gradients)
            
        return negative_acc, treatment_effect

    negative_acc_y, ce_y = calculate_gradients_and_effect(np.array(data_points_y), method = 'mean')
    negative_acc_d, ce_d = calculate_gradients_and_effect(np.array(data_points_d), method = 'mean')
    
    print(f"CE y : {ce_y:.3f}, CE d : {ce_d:.3f}")
    print(f"CACC y : {negative_acc_y:.3f}, CACC d : {negative_acc_d:.3f}")

    return negative_acc_y, negative_acc_d, ce_y, ce_d


def data_load(data):
    # Move all tensors in the data tuple to GPU at once
    data = tuple(tensor.cuda() for tensor in data)
    
    cont_p, cont_c, cat_p, cat_c, len, y, diff_days, *t = data
    return cont_p.shape[0], cont_p, cont_c, cat_p, cat_c, len, y, diff_days, t[0]

def reverse_scaling(scaling, out, y, a_y, b_y, a_d, b_d):
    '''
    a_y : min_y or mean_y
    b_y : max_y or var_y
    a_d : min_d or min_d
    b_d : max_d or var_d
    '''
    if scaling=="minmax":
        pred_y = restore_minmax(out[:, 0], a_y, b_y)
        pred_d = restore_minmax(out[:, 1], a_d, b_d)
        gt_y = restore_minmax(y[:,0], a_y, b_y)
        gt_d = restore_minmax(y[:,1], a_d, b_d)
        
    elif scaling == "normalization":
        pred_y = restore_meanvar(out[:, 0], a_y, b_y)
        pred_d = restore_meanvar(out[:, 1], a_d, b_d)
        gt_y = restore_meanvar(y[:,0], a_y, b_y)
        gt_d = restore_meanvar(y[:,1], a_d, b_d)
    return pred_y, pred_d, gt_y, gt_d

def set_seed(random_seed=1000):
    '''
    Set Seed for Reproduction
    '''
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def save_checkpoint(file_path, epoch, **kwargs):
    '''
    Save Checkpoint
    '''
    state = {"epoch": epoch}
    state.update(kwargs)
    torch.save(state, file_path)

def data_split_num(dataset, tr=0.8, val=0.1, te=0.1):
    train_length = int(tr * len(dataset))
    val_length = int(val * len(dataset))
    test_length = len(dataset) - train_length - val_length

    return train_length, val_length, test_length

def patient_seq_to_date_seq(non_padded_cluster, non_padded_days ):
    days_uniq=non_padded_days.unique()
    result = torch.zeros(days_uniq.size()[0], non_padded_cluster.shape[-1])  

    for i, value in enumerate(days_uniq):
        indices = torch.where(non_padded_days == value)[0].unsqueeze(1)
        mean_value = torch.mean(non_padded_cluster[indices], dim=0)  
        result[i] = mean_value

    return result, days_uniq.size()[0]

def reduction_cluster(x, diff_days, len, reduction):
    cluster = []
    for i in range(x.shape[0]):
        pad_tensor = torch.zeros([5,x.shape[-1]]).cuda()
        m = len[i].item()
        non_padded_cluster = x[i, :m, :]  
        ## Cluster-wise average ##
        if reduction == "mean":
            non_padded_cluster = torch.mean(non_padded_cluster, dim=0)
        ## Date-wise average ##
        elif reduction == "date":
            non_padded_days = diff_days[i, :m, :]
            non_padded_cluster, new_len = patient_seq_to_date_seq(non_padded_cluster, non_padded_days)
            len[i]=new_len
            pad_tensor[:non_padded_cluster.shape[0]] = non_padded_cluster
            non_padded_cluster=pad_tensor
        cluster.append(non_padded_cluster)

    return torch.stack(cluster, dim=0)

### Tukey transformation
def tukey_transformation(data, args):
    epsilon = 1e-8  
    
    if args.tukey:
        data[data == 0] = epsilon
        if args.beta != 0:
            data = torch.pow(data, args.beta)
        elif args.beta == 0:
            data = torch.log(data)
        else:
            data = (-1) * torch.pow(data, args.beta)
        data[torch.isnan(data)] = 0.0
        
    return data

def inverse_tukey_transformation(data, args):
    epsilon = 1e-8
    
    if args.tukey:
        # Handle NaNs (these would have been zeros in the original data)
        data[torch.isnan(data)] = 0.0

        # Inverse transform based on beta
        if args.beta != 0:
            data = torch.pow(data, 1 / args.beta)
        elif args.beta == 0:
            data = torch.exp(data)
        
        # Restore zeros (these were converted to epsilon in the original data)
        data[torch.abs(data - epsilon) < 1e-8] = 0.0
    
    return data

## for VAE
def reparametrize(mu, logvar):
    # Calculate standard deviation
    std = torch.exp(0.5 * logvar)
    
    # Create a standard normal distribution
    epsilon = Normal(torch.zeros_like(mu), torch.ones_like(std)).rsample()
    
    # Reparametrization trick
    z = mu + epsilon * std
    return z

def nan_filtered_loss(pred, target, criterion):
    valid_indices = torch.where(~torch.isnan(pred))[0]
    return criterion(pred[valid_indices], target[valid_indices])

def cevt_loss(x_reconstructed, x,   
            enc_t_pred, enc_y_pred, enc_d_pred,
            dec_t_pred, dec_y_pred, dec_d_pred,
            z_mu, z_logvar,
            t, y , d,
            criterion,
            lambdas,
            t_loss=True,
            val_len=None):
    
    # Encoder Prediction Loss
    enc_y_loss = criterion(enc_y_pred, y)
    enc_d_loss = criterion(enc_d_pred, d)
    if t_loss: # t1: dis|t2: danger
        # enc_t_loss = criterion(enc_t_pred, t)
        enc_t1_loss = criterion(enc_t_pred[:, 0], t[:, 0])
        enc_t2_loss = criterion(enc_t_pred[:, 1], t[:, 1])
        enc_loss = enc_y_loss + enc_d_loss + enc_t1_loss + enc_t2_loss # + enc_t_loss
    else:
        enc_t_loss = None
        enc_loss = enc_y_loss + enc_d_loss

    # Decoder Prediction Loss
    dec_y_loss = criterion(dec_y_pred, y)
    dec_d_loss = criterion(dec_d_pred, d)
    if t_loss: # t1: dis|t2: danger
        # dec_t_loss = criterion(dec_t_pred, t)
        dec_t1_loss = criterion(dec_t_pred[:, 0], t[:, 0])
        dec_t2_loss = criterion(dec_t_pred[:, 1], t[:, 1])
        dec_loss = dec_y_loss + dec_d_loss + dec_t1_loss + dec_t2_loss # + dec_t_loss
    else:
        dec_t_loss = None
        dec_loss = dec_y_loss + dec_d_loss

    pred_loss = enc_loss + dec_loss
    
    # KLD loss
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())

    # Reconstruction Loss
    recon_loss = criterion(x_reconstructed, x)

    # # Reconstruction Loss
    # recon_loss = nan_filtered_loss(x_reconstructed, x, criterion)
    total_loss = lambdas[0]*pred_loss + lambdas[1]*kl_loss + lambdas[2]*recon_loss
    # total_loss = lambdas[0]*enc_y_loss + lambdas[1]*enc_d_loss #+ lambdas[2]*recon_loss
    return total_loss, (enc_y_loss, enc_d_loss), (dec_y_loss, dec_d_loss), ((enc_t1_loss, enc_t2_loss), (dec_t1_loss, dec_t2_loss)), (pred_loss, kl_loss, recon_loss)


def cevae_loss(x_reconstructed, x,   
            enc_t_pred, enc_y_pred, enc_d_pred,
            dec_t_pred, dec_y_pred, dec_d_pred,
            z_mu, z_logvar,
            t, y , d,
            criterion,
            lambdas,
            t_loss=True,
            val_len=None):
    
    # Encoder Prediction Loss
    enc_y_loss = criterion(enc_y_pred, y)
    enc_d_loss = criterion(enc_d_pred, d)
    if t_loss: # t1: dis|t2: danger
        # enc_t_loss = criterion(enc_t_pred, t)
        enc_t2_loss = criterion(enc_t_pred[:, 0], t)
        # enc_t2_loss = criterion(enc_t_pred[:, 1], t[:, 1])
        enc_loss = enc_y_loss + enc_d_loss + enc_t2_loss # + enc_t_loss
    else:
        enc_t_loss = None
        enc_loss = enc_y_loss + enc_d_loss

    # Decoder Prediction Loss
    dec_y_loss = criterion(dec_y_pred, y)
    dec_d_loss = criterion(dec_d_pred, d)
    if t_loss: # t1: dis|t2: danger
        # dec_t_loss = criterion(dec_t_pred, t)
        dec_t2_loss = criterion(dec_t_pred[:, 0], t)
        dec_loss = dec_y_loss + dec_d_loss + dec_t2_loss # + dec_t_loss
    else:
        dec_t_loss = None
        dec_loss = dec_y_loss + dec_d_loss

    pred_loss = enc_loss + dec_loss
    
    # KLD loss
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())

    # Reconstruction Loss
    recon_loss = criterion(x_reconstructed, x)

    total_loss = lambdas[0]*pred_loss + lambdas[1]*kl_loss + lambdas[2]*recon_loss
    # total_loss = lambdas[0]*enc_y_loss + lambdas[1]*enc_d_loss #+ lambdas[2]*recon_loss
    return total_loss, (enc_y_loss, enc_d_loss), (dec_y_loss, dec_d_loss), ((torch.zeros_like(enc_t2_loss), enc_t2_loss), (torch.zeros_like(dec_t2_loss), dec_t2_loss)), (pred_loss, kl_loss, recon_loss)


def sigmoid_annealing(epoch, total_epochs, k=1.0, x0=0.5):
    """Calculate the sigmoid annealing value for lambda.
    
    Args:
    - epoch (int): Current epoch.
    - total_epochs (int): Total number of epochs.
    - k (float): Steepness of the curve.
    - x0 (float): Midpoint of the sigmoid.
    
    Returns:
    - float: Sigmoid annealed value for the lambda.
    """
    x = (epoch / total_epochs) - x0
    return 1 / (1 + np.exp(-k * x))


##############################################################
## iTransformer
##############################################################
def train_iTrans(args, data, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    batch_num, cont_p, cont_c, cat_p, cat_c, len, yd_true, diff_days, *t = data_load(data)
    
    t_true = t[0]
    yd_pred = model(cont_p, cont_c, cat_p, cat_c, len, diff_days)
    
    loss_d = criterion(yd_pred[:,0], yd_true[:,0])
    loss_y = criterion(yd_pred[:,1], yd_true[:,1])    
    loss = loss_d + loss_y
                
    if not torch.isnan(loss):
        loss.backward()
        optimizer.step()
        return loss_d.item(), loss_y.item(), batch_num, yd_pred, yd_true
    else:
        # return 0, batch_num, out, y
        raise ValueError("Loss raised nan.")
    
    
@torch.no_grad()
def valid_iTrans(args, data, model, eval_criterion, scaling, a_y, b_y, a_d, b_d):
    model.eval()
    batch_num, cont_p, cont_c, cat_p, cat_c, len, yd_true, diff_days, *t = data_load(data)

    t_true = t[0]
    yd_pred = model(cont_p, cont_c, cat_p, cat_c, len, diff_days)
    
    pred_y, pred_d, gt_y, gt_d = reverse_scaling(scaling, yd_pred, yd_true, a_y, b_y, a_d, b_d)
    loss_y = eval_criterion(pred_y, gt_y)
    loss_d = eval_criterion(pred_d, gt_d)
    
    loss = loss_y + loss_d
    
    if not torch.isnan(loss):
        return loss_d.item(), loss_y.item(), batch_num, yd_pred, yd_true
    else:
        return 0, batch_num, yd_pred, yd_true
    
    
@torch.no_grad()
def test_iTrans(args, data, model, scaling, a_y, b_y, a_d, b_d):
    
    criterion_mae = nn.L1Loss(reduction="sum")
    criterion_rmse = nn.MSELoss(reduction="sum")
    
    model.eval()

    batch_num, cont_p, cont_c, cat_p, cat_c, len, yd_true, diff_days, *t = data_load(data)
    
    t_true = t[0]
    yd_pred = model(cont_p, cont_c, cat_p, cat_c, len, diff_days)

    pred_y, pred_d, gt_y, gt_d = reverse_scaling(scaling, yd_pred, yd_true, a_y, b_y, a_d, b_d)
    
    # MAE
    mae_y = criterion_mae(pred_y, gt_y)
    mae_d = criterion_mae(pred_d, gt_d)
    mae = mae_y + mae_d
    
    # RMSE
    rmse_y = criterion_rmse(pred_y, gt_y)
    rmse_d = criterion_rmse(pred_d, gt_d)
    rmse = rmse_y + rmse_d
    
    if not torch.isnan(mae) and not torch.isnan(rmse):
        return mae_d.item(), mae_y.item(), rmse_d.item(), rmse_y.item(), batch_num, yd_pred, yd_true
    else:
        return 0, batch_num, yd_pred, yd_true
    
    

@torch.no_grad()
def iTrans_CE(args, model, dataloader, intervene_var):
    model.eval()  
    data_points_y = []; data_points_d=[]
    
    for data in dataloader:
        _, cont_p, cont_c, cat_p, cat_c, val_len, y, diff_days, *rest = data_load(data)
        gt_t = rest[0]
        
        if args.use_treatment:
            if args.model == 'cevt':
                (x, diff_days, _), _ = model.embedding(cont_p, cont_c, cat_p, cat_c, val_len, diff_days)
                src_key_padding_mask = ~(torch.arange(x.size(1)).expand(x.size(0), -1).cuda() < val_len.unsqueeze(1)).cuda()
                src_mask = model.generate_square_subsequent_mask(x.size(1)).cuda() if model.unidir else None

                # use ground truth t instead of x2t_pred
                original_t = gt_t[:,0].unsqueeze(1) if intervene_var == 't1' else gt_t[:,1].unsqueeze(1)
                _, _, original_enc_yd = model.transformer_encoder(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask, val_len=val_len, intervene_t=("t1",original_t))
                
                saved_original_t = original_t.clone()
                saved_original_enc_yd = original_enc_yd.clone()
                
                
                intervene_t_value_range = range(0, 61) if intervene_var == 't1' else range(30, 111)
                for intervene_t_value in [x * 0.1 for x in intervene_t_value_range]:
                    original_t=saved_original_t.clone()
                    original_enc_yd=saved_original_enc_yd.clone()
                    
                    if intervene_var == 't1':
                        intervene_t_value = intervene_t_value / 6  # t1 norm [0, 6]
                    elif intervene_var == 't2':
                        intervene_t_value = (intervene_t_value - 3) / 8  # t2 norm [3, 11] 

                    
                    intervene_t = torch.full((x.size(0),), intervene_t_value, dtype=torch.float).unsqueeze(1).cuda()
                    
                    _, _, intervene_enc_yd = model.transformer_encoder(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask, val_len=val_len, intervene_t=("t1",intervene_t))
                    
                        
                    delta_y = original_enc_yd - intervene_enc_yd
                    delta_t = (original_t - intervene_t)
                    # delta_t = (original_t[:,0] - intervene_t_value)*6  # denormalize 
                    
                    if intervene_var == 't1':
                        delta_t = delta_t*6  # denormalize 
                    elif intervene_var == 't2':
                        delta_t = delta_t*8  # +3 denormalize
                    for i in range(delta_y.size(0)):
                        data_points_y.append((delta_t[i].item(), delta_y[i].item()))
                        data_points_d.append((delta_t[i].item(), delta_d[i].item()))
            
            elif args.model in ['tarnet', 'dragonnet']:
                original_t = cont_c[:,:,0].clone()
                
                yd0_pred, yd1_pred, yd2_pred, yd3_pred, yd4_pred, yd5_pred, yd6_pred, t_pred, epsilons = model(cont_p, cont_c, cat_p, cat_c, val_len, diff_days)
                t_pred = F.softmax(t_pred, dim=-1).argmax(dim=1).to(int)
                mask_t0 = torch.zeros_like(t_pred); mask_t0[t_pred == 0] = True; mask_t0 = mask_t0.unsqueeze(1).expand(-1, 2)
                mask_t1 = torch.zeros_like(t_pred); mask_t1[t_pred == 1] = True; mask_t1 = mask_t1.unsqueeze(1).expand(-1, 2)
                mask_t2 = torch.zeros_like(t_pred); mask_t2[t_pred == 2] = True; mask_t2 = mask_t2.unsqueeze(1).expand(-1, 2)
                mask_t3 = torch.zeros_like(t_pred); mask_t3[t_pred == 3] = True; mask_t3 = mask_t3.unsqueeze(1).expand(-1, 2)
                mask_t4 = torch.zeros_like(t_pred); mask_t4[t_pred == 4] = True; mask_t4 = mask_t4.unsqueeze(1).expand(-1, 2)
                mask_t5 = torch.zeros_like(t_pred); mask_t5[t_pred == 5] = True; mask_t5 = mask_t5.unsqueeze(1).expand(-1, 2)
                mask_t6 = torch.zeros_like(t_pred); mask_t6[t_pred == 6] = True; mask_t6 = mask_t6.unsqueeze(1).expand(-1, 2)
                original_yd_pred = mask_t0 * yd0_pred + mask_t1 * yd1_pred + mask_t2 * yd2_pred + mask_t3 * yd3_pred + mask_t4 * yd4_pred + mask_t5 * yd5_pred + mask_t6 * yd6_pred
                original_yd_pred = torch.clamp(original_yd_pred, 0, 1)
                
                saved_original_t = original_t.clone()
                saved_original_yd_pred = original_yd_pred.clone()
                
                intervene_t_value_range = range(0, 61) if intervene_var == 't1' else range(30, 111)
                for intervene_t_value in [x * 0.1 for x in intervene_t_value_range]:
                    
                    # for 't2'
                    intervene_t_value = (intervene_t_value - 3) / 8  # t2 norm [3, 11] 
                    cont_c[:,:,0] = intervene_t_value 
                    
                    yd0_pred, yd1_pred, yd2_pred, yd3_pred, yd4_pred, yd5_pred, yd6_pred, t_pred, epsilons = model(cont_p, cont_c, cat_p, cat_c, val_len, diff_days)
                    t_pred = F.softmax(t_pred, dim=-1).argmax(dim=1).to(int)
                    mask_t0 = torch.zeros_like(t_pred); mask_t0[t_pred == 0] = True; mask_t0 = mask_t0.unsqueeze(1).expand(-1, 2)
                    mask_t1 = torch.zeros_like(t_pred); mask_t1[t_pred == 1] = True; mask_t1 = mask_t1.unsqueeze(1).expand(-1, 2)
                    mask_t2 = torch.zeros_like(t_pred); mask_t2[t_pred == 2] = True; mask_t2 = mask_t2.unsqueeze(1).expand(-1, 2)
                    mask_t3 = torch.zeros_like(t_pred); mask_t3[t_pred == 3] = True; mask_t3 = mask_t3.unsqueeze(1).expand(-1, 2)
                    mask_t4 = torch.zeros_like(t_pred); mask_t4[t_pred == 4] = True; mask_t4 = mask_t4.unsqueeze(1).expand(-1, 2)
                    mask_t5 = torch.zeros_like(t_pred); mask_t5[t_pred == 5] = True; mask_t5 = mask_t5.unsqueeze(1).expand(-1, 2)
                    mask_t6 = torch.zeros_like(t_pred); mask_t6[t_pred == 6] = True; mask_t6 = mask_t6.unsqueeze(1).expand(-1, 2)
                    intervene_yd_pred = mask_t0 * yd0_pred + mask_t1 * yd1_pred + mask_t2 * yd2_pred + mask_t3 * yd3_pred + mask_t4 * yd4_pred + mask_t5 * yd5_pred + mask_t6 * yd6_pred
                        
                    # for fair comaparison
                    intervene_yd_pred = torch.clamp(intervene_yd_pred, 0, 1)
                    
                    delta_y = original_yd_pred - intervene_yd_pred
                    # delta_t = (original_t[:,0] - intervene_t_value)*8  # +3 denormalize      
                    delta_t = (original_t[:,0] - intervene_t_value)
                    # delta_t = (original_t[:,0] - intervene_t_value)*6  # denormalize 
                
                    if intervene_var == 't1':
                        delta_t = delta_t*6  # denormalize 
                    elif intervene_var == 't2':
                        delta_t = delta_t*8  # +3 denormalize
                    delta_y, delta_d, _, _ = reverse_scaling(args.scaling, delta_y, y, dataloader.dataset.dataset.a_y, dataloader.dataset.dataset.b_y, dataloader.dataset.dataset.a_d, dataloader.dataset.dataset.b_d)

                    for i in range(delta_y.size(0)):
                        data_points_y.append((delta_t[i].item(), delta_y[i].item()))
                        data_points_d.append((delta_t[i].item(), delta_d[i].item()))
                        
                        
            elif args.model=='iTransformer':
                original_t = cont_c[:,:,0].clone()
                original_yd = model(cont_p, cont_c, cat_p, cat_c, val_len, diff_days)
                original_yd = torch.clamp(original_yd, 0, 1)
                
                
                intervene_t_value_range = range(0, 61) if intervene_var == 't1' else range(30, 111)
                for intervene_t_value in [x * 0.1 for x in intervene_t_value_range]:
                    intervene_t_value = (intervene_t_value - 3) / 8  # t2 norm [3, 11] 
                    cont_c[:,:,0] = intervene_t_value 
            
                     
                    intervene_yd = model(cont_p, cont_c, cat_p, cat_c, val_len, diff_days)
                    intervene_yd = torch.clamp(intervene_yd, 0, 1)
                    
                    delta_y = original_yd - intervene_yd
                    # delta_t = (original_t[:,0] - intervene_t_value)*8  # +3 denormalize  
                    delta_t = (original_t[:,0] - intervene_t_value)
                    # delta_t = (original_t[:,0] - intervene_t_value)*6  # denormalize 
                    
                    if intervene_var == 't1':
                        delta_t = delta_t*6  # denormalize 
                    elif intervene_var == 't2':
                        delta_t = delta_t*8  # +3 denormalize
                    
                    
                    delta_y, delta_d, _, _ = reverse_scaling(args.scaling, delta_y, y, dataloader.dataset.dataset.a_y, dataloader.dataset.dataset.b_y, dataloader.dataset.dataset.a_d, dataloader.dataset.dataset.b_d)
            
                    for i in range(delta_y.size(0)):
                        data_points_y.append((delta_t[i].item(), delta_y[i].item()))
                        data_points_d.append((delta_t[i].item(), delta_d[i].item()))
                
        else:
            original_t = cont_c[:,:,0].clone() if intervene_var=='t1' else cont_c[:,:,1].clone()
            if args.model == 'cevt':
                _, _, (original_yd, _), (_, _), (_, _) = model(cont_p, cont_c, cat_p, cat_c, val_len, diff_days)
            else:
                original_yd = model(cont_p, cont_c, cat_p, cat_c, val_len, diff_days)
            
            original_yd = torch.clamp(original_yd, 0, 1)
            
            intervene_t_value_range = range(0, 61) if intervene_var == 't1' else range(30, 111)
            for intervene_t_value in [x * 0.1 for x in intervene_t_value_range]:
                
                if intervene_var == 't1':
                    intervene_t_value = intervene_t_value / 6  # t1 norm [0, 6]
                    cont_c[:,:,0] = intervene_t_value 
                elif intervene_var == 't2':
                    intervene_t_value = (intervene_t_value - 3) / 8  # t2 norm [3, 11] 
                    cont_c[:,:,1] = intervene_t_value 
        
                 
                intervene_yd = model(cont_p, cont_c, cat_p, cat_c, val_len, diff_days)
                
                # for fair comaparison
                intervene_yd = torch.clamp(intervene_yd, 0, 1)
                
                delta_y = original_yd - intervene_yd               
                delta_t = (original_t[:,0] - intervene_t_value)
                if intervene_var == 't1':
                    delta_t = delta_t*6  # denormalize 
                elif intervene_var == 't2':
                    delta_t = delta_t*8  # +3 denormalize 
                
                delta_y, delta_d, _, _ = reverse_scaling(args.scaling, delta_y, y, dataloader.dataset.dataset.a_y, dataloader.dataset.dataset.b_y, dataloader.dataset.dataset.a_d, dataloader.dataset.dataset.b_d)
        
                for i in range(delta_y.size(0)):
                    data_points_y.append((delta_t[i].item(), delta_y[i].item()))
                    data_points_d.append((delta_t[i].item(), delta_d[i].item()))
    
    def iTrans_calculate_gradients_and_effect(data_points, method='coef'):
        del_t = data_points[:, 0]  # delta_t
        del_var = data_points[:, 1]   # delta_y or delta_d

        non_zero_indices = del_t != 0
        del_t = del_t[non_zero_indices]
        del_var = del_var[non_zero_indices]
        
         
        gradients = del_var / del_t
        negative_acc = np.sum(gradients < 0) / len(gradients)
        
        treatment_effect = np.mean(gradients)
            
        return negative_acc, treatment_effect

    negative_acc_y, ce_y = iTrans_calculate_gradients_and_effect(np.array(data_points_y), method = 'mean')
    negative_acc_d, ce_d = iTrans_calculate_gradients_and_effect(np.array(data_points_d), method = 'mean')
    
    print(f"CE y : {ce_y:.3f}, CE d : {ce_d:.3f}")
    print(f"CACC y : {negative_acc_y:.3f}, CACC d : {negative_acc_d:.3f}")

    return negative_acc_y, negative_acc_d, ce_y, ce_d
    

##############################################################
## DragonNet, TarNet
##############################################################
"""
Ref
[1] https://github.com/kochbj/Deep-Learning-for-Causal-Inference/issues/4
[2] https://github.com/claudiashi57/dragonnet/issues/4
"""


def causal_yd_loss(yd0_pred, yd1_pred, yd2_pred, yd3_pred, yd4_pred, yd5_pred, yd6_pred, yd_true, t_pred, criterion):
    t_pred = F.softmax(t_pred, dim=-1).argmax(dim=1).to(int)
    
    # hard-coded masking
    mask_t0 = torch.zeros_like(t_pred); mask_t0[t_pred == 0] = True; mask_t0 = mask_t0.unsqueeze(1).expand(-1, 2)
    mask_t1 = torch.zeros_like(t_pred); mask_t1[t_pred == 1] = True; mask_t1 = mask_t1.unsqueeze(1).expand(-1, 2)
    mask_t2 = torch.zeros_like(t_pred); mask_t2[t_pred == 2] = True; mask_t2 = mask_t2.unsqueeze(1).expand(-1, 2)
    mask_t3 = torch.zeros_like(t_pred); mask_t3[t_pred == 3] = True; mask_t3 = mask_t3.unsqueeze(1).expand(-1, 2)
    mask_t4 = torch.zeros_like(t_pred); mask_t4[t_pred == 4] = True; mask_t4 = mask_t4.unsqueeze(1).expand(-1, 2)
    mask_t5 = torch.zeros_like(t_pred); mask_t5[t_pred == 5] = True; mask_t5 = mask_t5.unsqueeze(1).expand(-1, 2)
    mask_t6 = torch.zeros_like(t_pred); mask_t6[t_pred == 6] = True; mask_t6 = mask_t6.unsqueeze(1).expand(-1, 2)
    yd_pred = mask_t0 * yd0_pred + mask_t1 * yd1_pred + mask_t2 * yd2_pred + mask_t3 * yd3_pred + mask_t4 * yd4_pred + mask_t5 * yd5_pred + mask_t6 * yd6_pred
    
    loss_y = criterion(yd_pred[:,0], yd_true[:,0])
    loss_d = criterion(yd_pred[:,1], yd_true[:,1])
    
    return loss_y, loss_d, yd_pred



def causal_t_loss(t_pred, t_true):
    t_pred = 6 * t_pred
    t_true = (6 * t_true).to(dtype=torch.long)
    ce = nn.CrossEntropyLoss()
    return ce(t_pred, t_true)



def train_causal_model(args, data, model, optimizer, criterion):
    
    eval_loss_y = None; eval_loss_d=None
    model.train()
    optimizer.zero_grad()
    batch_num, cont_p, cont_c, cat_p, cat_c, len, yd_true, diff_days, *t = data_load(data)
    
    t_true = t[0]
    yd0_pred, yd1_pred, yd2_pred, yd3_pred, yd4_pred, yd5_pred, yd6_pred, t_pred, epsilons = model(cont_p, cont_c, cat_p, cat_c, len, diff_days)
    
    loss_y, loss_d, yd_pred = causal_yd_loss(yd0_pred, yd1_pred, yd2_pred, yd3_pred, yd4_pred, yd5_pred, yd6_pred, yd_true, t_pred, criterion)
    yd_pred_loss = loss_y + loss_d
    t_loss = causal_t_loss(t_pred, t_true)
    total_loss = yd_pred_loss + args.alpha * t_loss
                
    if not torch.isnan(total_loss):
        total_loss.backward()
        optimizer.step()
        return loss_d.item(), loss_y.item(), t_loss.item(), batch_num, yd_pred, yd_true
    else:
        # return 0, batch_num, out, y
        raise ValueError("Loss raised nan.")


@torch.no_grad()
def valid_causal_model(args, data, model, eval_criterion, scaling, a_y, b_y, a_d, b_d):
    model.eval()
    batch_num, cont_p, cont_c, cat_p, cat_c, len, yd_true, diff_days, *t = data_load(data)

    t_true = t[0]
    yd0_pred, yd1_pred, yd2_pred, yd3_pred, yd4_pred, yd5_pred, yd6_pred, t_pred, epsilons = model(cont_p, cont_c, cat_p, cat_c, len, diff_days)
    
    _, _, yd_pred = causal_yd_loss(yd0_pred, yd1_pred, yd2_pred, yd3_pred, yd4_pred, yd5_pred, yd6_pred, yd_true, t_pred, eval_criterion)
    pred_y, pred_d, gt_y, gt_d = reverse_scaling(scaling, yd_pred, yd_true, a_y, b_y, a_d, b_d)
    loss_y = eval_criterion(pred_y, gt_y)
    loss_d = eval_criterion(pred_d, gt_d)
    t_loss = causal_t_loss(t_pred, t_true)
    
    total_loss = loss_y + loss_d + args.alpha * t_loss
    
    if not torch.isnan(total_loss):
        return loss_d.item(), loss_y.item(), t_loss.item(), batch_num, yd_pred, yd_true
    else:
        return 0, batch_num, yd_pred, yd_true
    
@torch.no_grad()
def test_causal_model(args, data, model, scaling, a_y, b_y, a_d, b_d):
    
    criterion_mae = nn.L1Loss(reduction="sum")
    criterion_rmse = nn.MSELoss(reduction="sum")
    
    model.eval()

    batch_num, cont_p, cont_c, cat_p, cat_c, len, yd_true, diff_days, *t = data_load(data)
    
    t_true = t[0]
    yd0_pred, yd1_pred, yd2_pred, yd3_pred, yd4_pred, yd5_pred, yd6_pred, t_pred, epsilons = model(cont_p, cont_c, cat_p, cat_c, len, diff_days)

    # if out.shape == torch.Size([2]):
    #     out = out.unsqueeze(0)
    _, _, yd_pred = causal_yd_loss(yd0_pred, yd1_pred, yd2_pred, yd3_pred, yd4_pred, yd5_pred, yd6_pred, yd_true, t_pred, criterion_mae)
    pred_y, pred_d, gt_y, gt_d = reverse_scaling(scaling, yd_pred, yd_true, a_y, b_y, a_d, b_d)
    
    # MAE
    mae_y = criterion_mae(pred_y, gt_y)
    mae_d = criterion_mae(pred_d, gt_d)
    mae = mae_y + mae_d
    
    # RMSE
    rmse_y = criterion_rmse(pred_y, gt_y)
    rmse_d = criterion_rmse(pred_d, gt_d)
    rmse = rmse_y + rmse_d
    
    t_loss = causal_t_loss(t_pred, t_true)
    
    if not torch.isnan(mae) and not torch.isnan(rmse):
        return mae_d.item(), mae_y.item(), rmse_d.item(), rmse_y.item(), t_loss.item(), batch_num, yd_pred, yd_true
    else:
        return 0, batch_num, yd_pred, yd_true
  