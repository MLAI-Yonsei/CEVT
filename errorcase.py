import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle 

import os, time
import math

import argparse
import tabulate

import utils, models, ml_algorithm
import wandb
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset, TensorDataset
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

## Argparse ----------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Cluster Medical-AI")

parser.add_argument("--seed", type=int, default=1000, help="random seed (default: 1000)")

parser.add_argument("--eval_model", type=str, default=None,
    help="path to load saved model to evaluate model (default: None)",)

parser.add_argument("--ignore_wandb", action='store_true',
        help = "Stop using wandb (Default : False)")

parser.add_argument("--run_group", type=str, default="default")

parser.add_argument("--save_pred", action='store_true',
        help = "Save ground truth and prediction as csv (Default : False)")
parser.add_argument(
    "--table_idx",
    type=int, default=0, choices=[0, 1, 2, 3, 4, 5],
    help="Cluster Date print date (Default : 0) if 0, use concated dataset"
)

parser.add_argument("--filter_out_clip", action='store_true',
        help = "Filter out clamped data points when calculate causal effect (Default : False)")

# Data ---------------------------------------------------------
parser.add_argument(
    "--data_path",
    type=str,
    default='./data/',
    help="path to datasets location",)

# parser.add_argument("--tr_ratio", type=float, default=0.8,
#           help="Ratio of train data (Default : 0.8)")

# parser.add_argument("--val_ratio", type=float, default=0.1,
#           help="Ratio of validation data (Default : 0.1)")

# parser.add_argument("--te_ratio", type=float, default=0.1,
#           help="Ratio of test data (Default : 0.1)")

parser.add_argument(
    "--batch_size",
    type=int, default=32,
    help="Batch Size (default : 32)"
)

parser.add_argument(
    "--scaling",
    type=str,
    default='minmax',
    choices=['minmax', 'meanvar']
)

parser.add_argument('--tukey', action='store_true', help='Use tukey transformation to get divergence')

parser.add_argument(
    "--beta",
    type=float, default=0.5
)

parser.add_argument(
    "--use_treatment", action='store_true', help='If True, use treatment as x feature'
)

# parser.add_argument("--treatment_var", type=str, default='danger', choices=['dis, danger'], help="treatment variable")

parser.add_argument('--single_treatment', action='store_true', help='use only <dis> variable as treatment (default false)')

parser.add_argument('--shift', action='store_true', help='do not use treatment as feature (default false)')

parser.add_argument(
    "--MC_sample",
    type=int, default=30,
    help="Counts of Monte Carlo resampling"
)

parser.add_argument("--cutoff_dataset",
                    type=int, default=0,
                    help="If 0, uses as combined augmented dataset")

#----------------------------------------------------------------


# Model ---------------------------------------------------------
parser.add_argument(
    "--model",
    type=str, default='transformer',
    choices=["cet", "cevae", "transformer", "linear", "ridge", "mlp", "svr", "rfr"],
    help="model name (default : transformer)")

parser.add_argument("--save_path",
            type=str, default="./best_models/",
            help="Path to save best model dict")

parser.add_argument(
    "--num_features",
    type=int, default=128,
    help="feature size (default : 128)"
)

parser.add_argument(
    "--hidden_dim",
    type=int, default=128,
    help="DL model hidden size (default : 128)"
)

parser.add_argument(
    "--num_layers",
    type=int, default=1,
    help="DL model layer num (default : 1)"
)

parser.add_argument(
    "--cet_transformer_layers",
    type=int, default=4,
    help="It has to be over 3 layers (default : 4)"
)

parser.add_argument(
    "--num_heads",
    type=int, default=2,
    help="Transformer model head num (default : 2)"
)

parser.add_argument(
    "--output_size",
    type=int, default=2,
    help="Output size (default : 2)"
)

parser.add_argument(
    "--drop_out",
    type=float, default=0.0,
    help="Dropout Rate (Default : 0)"
)

parser.add_argument("--disable_embedding", action='store_true',
        help = "Disable embedding to use raw data (Default : False)")

parser.add_argument("--unidir", action='store_true',
        help = "Unidirectional attention to transformer encoder (Default : False)")

parser.add_argument("--variational", action='store_true',
        help = "variational z sampling (Default : False)")

parser.add_argument("--residual_t", action='store_true',
        help = "residual connection with t to yd (Default : False)")

parser.add_argument("--residual_x", action='store_true',
        help = "residual connection with x to tyd (Default : False)")

#----------------------------------------------------------------

# Criterion -----------------------------------------------------
parser.add_argument(
    "--criterion",
    type=str, default='MSE', choices=["MSE", "RMSE"],
    help="Criterion for training (default : MSE)")

parser.add_argument(
    "--eval_criterion",
    type=str, default='MAE', choices=["MAE", "RMSE"],
    help="Criterion for training (default : MAE)")

#----------------------------------------------------------------

# Learning Hyperparameter --------------------------------------
parser.add_argument("--lr_init", type=float, default=0.01,
                help="learning rate (Default : 0.01)")

parser.add_argument("--optim", type=str, default="adam",
                    choices=["sgd", "adam", "radam", "adamw"],
                    help="Optimization options")

parser.add_argument("--momentum", type=float, default=0.9,
                help="momentum (Default : 0.9)")

parser.add_argument("--epochs", type=int, default=200, metavar="N",
    help="number epochs to train (Default : 200)")

parser.add_argument("--wd", type=float, default=5e-4, help="weight decay (Default: 5e-4)")

parser.add_argument("--scheduler", type=str, default='cos_anneal', choices=['constant', "cos_anneal"])

parser.add_argument("--t_max", type=int, default=200,
                help="T_max for Cosine Annealing Learning Rate Scheduler (Default : 200)")

parser.add_argument("--lambdas", nargs='+', type=float, default=[1.0, 1.0, 1.0], help='pred loss + kld loss + recon loss')

parser.add_argument("--sig_x0", type=float, default=0.75)
#----------------------------------------------------------------

parser.add_argument("--lamb", type=float, default=0.0,
                help="Penalty term for Ridge Regression (Default : 0)")

parser.add_argument(
    "--intervene_var",
    type=str, default='t1', choices=["t1", "t2"],
    help="Intervention variable for Causal Effect Estimation (default : t1)")

parser.add_argument('--is_synthetic', action='store_true', help='use synthetic dataset (default false)')

args = parser.parse_args()
## ----------------------------------------------------------------------------------------------------


## Set seed and device ----------------------------------------------------------------
utils.set_seed(args.seed)

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {args.device}")
#-------------------------------------------------------------------------------------

## Set wandb ---------------------------------------------------------------------------
if args.ignore_wandb == False:
    wandb.init(entity="mlai_medical_ai", project="cluster-regression", group=args.run_group)
    wandb.config.update(args)
    if args.disable_embedding:
        wandb.run.name = f"raw_{args.model}({args.hidden_dim})-{args.optim}-{args.lr_init}-{args.wd}-{args.drop_out}"
    else:
        wandb.run.name = f"embed_{args.model}({args.hidden_dim})-{args.optim}-{args.lr_init}-{args.wd}-{args.drop_out}"
       
## Load Data --------------------------------------------------------------------------------
### ./data/data_mod.ipynb 에서 기본적인 데이터 전처리  ###
if args.is_synthetic:
    with open('./data/synthetic/synthetic_dowhy.pkl', 'rb') as f:
        data = pickle.load(f)
    dataset = utils.SyntheticTimeSeriesDataset(args, data)
else:
    dataset = utils.Tabledata(args, pd.read_csv(args.data_path+f"data_cut_{args.cutoff_dataset}.csv"), args.scaling)

train_dataset, val_dataset, test_dataset = random_split(dataset, utils.data_split_num(dataset))
tr_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
print(f"Number of training Clusters : {len(train_dataset)}")

val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

print(f"use treatment as feature : {not args.use_treatment}")
print("Successfully load data!")
#-------------------------------------------------------------------------------------


## Model ------------------------------------------------------------------------------------
if args.model == 'transformer':
    model = models.Transformer(args).to(args.device)
    
if args.model == 'cet':
    assert(args.use_treatment == True)
    model = models.CETransformer(args).to(args.device) 

if args.model == 'cevae':
    assert(args.use_treatment == True)
    assert(args.single_treatment == True)
    model = models.CEVAE(args).to(args.device) 
    
elif args.model == "mlp":
    model = models.MLPRegressor(args=args).to(args.device)

elif args.model in ["linear", "ridge"]:
    model = models.LinearRegression(args=args).to(args.device)

elif args.model in ["svr", "rfr"]:
    args.device = torch.device("cpu")
    ml_algorithm.fit(args.data_path, args.model, args.ignore_wandb, cutdates_num, args.table_idx)

print(f"Successfully prepared {args.model} model")
# ---------------------------------------------------------------------------------------------


## Criterion ------------------------------------------------------------------------------
# Train Criterion
if args.criterion in ['MSE', 'RMSE']:
    criterion = nn.MSELoss(reduction="sum") 

# Validation Criterion
if args.eval_criterion == 'MAE':
    eval_criterion = nn.L1Loss(reduction="sum")

elif args.eval_criterion == "RMSE":
    eval_criterion = nn.MSELoss(reduction="sum")
    

# ---------------------------------------------------------------------------------------------

## Optimizer and Scheduler --------------------------------------------------------------------
if args.optim == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init, weight_decay=args.wd)
elif args.optim == "radam":
    optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr_init, weight_decay=args.wd)
elif args.optim == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_init, weight_decay=args.wd)
elif args.optim == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_init, momentum=args.momentum, weight_decay=args.wd)
else:
    raise NotImplementedError

if args.scheduler  == "cos_anneal":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max)    
else:
    scheduler = None
# ---------------------------------------------------------------------------------------------

## Training Phase -----------------------------------------------------------------------------
columns = ["ep", "lr", f"tr_loss_d({args.eval_criterion})", f"tr_loss_y({args.eval_criterion})", f"val_loss_d({args.eval_criterion})", f"val_loss_y({args.eval_criterion})",
           "te_loss_d(MAE)", "te_loss_y(MAE)", "te_loss_d(RMSE)", "te_loss_y(RMSE)", "eval_model"]
## print table index, 0 = cocnated data
cutdates_num=0
best_epochs=[0] * (cutdates_num+1) 
best_val_loss_d = [9999] * (cutdates_num+1); best_val_loss_y = [9999] * (cutdates_num+1); best_val_loss_t1 = [9999] * (cutdates_num+1); best_val_loss_t2 = [9999] * (cutdates_num+1)
best_val_models = [""] * (cutdates_num+1); best_tr_models = [""] * (cutdates_num+1)
best_test_losses = [[9999 for j in range(4)] for i in range(cutdates_num+1)]
tr_eval_model=None; tr_eval_loss_d=None; tr_eval_loss_y=None; tr_eval_loss_t=None
if args.eval_model != None:
    args.epochs = 1
    tr_loss_d=0; tr_loss_y=0
    model.load_state_dict(torch.load(args.eval_model)['state_dict'])
    best_model=model

lambda0 = args.lambdas[1]
for epoch in range(1, args.epochs + 1):
    lr = optimizer.param_groups[0]['lr']
    tr_epoch_eval_loss_d = 0; tr_epoch_eval_loss_y = 0; tr_epoch_eval_loss_t1 = 0; tr_epoch_eval_loss_t2 = 0
    tr_epoch_loss_d = 0; tr_epoch_loss_y = 0
    val_epoch_loss_d = 0; val_epoch_loss_y = 0; val_epoch_loss_t1 = 0; val_epoch_loss_t2 = 0
    te_mae_epoch_loss_d = 0; te_mae_epoch_loss_y = 0; te_mae_epoch_loss_t1 = 0; te_mae_epoch_loss_t2 = 0
    te_mse_epoch_loss_d = 0; te_mse_epoch_loss_y = 0; te_mse_epoch_loss_t1 = 0; te_mse_epoch_loss_t2 = 0
    tr_epoch_pred_loss = 0; tr_epoch_kl_loss = 0; tr_epoch_recon_loss = 0
    
    concat_tr_num_data = 0; concat_val_num_data = 0; concat_te_num_data = 0

    error_analysis_list = []

    # Test Phase ----------------------------------------------------------------------
    for itr, batch_data in enumerate(test_dataloader):
        te_mae_batch_loss_d, te_mae_batch_loss_y, te_mse_batch_loss_d, te_mse_batch_loss_y, te_num_data, te_predicted, te_ground_truth, *t_loss = utils.test(args, batch_data, model,
                                                                            args.scaling, test_dataset.dataset.a_y, test_dataset.dataset.b_y,
                                                                            test_dataset.dataset.a_d, test_dataset.dataset.b_d, use_treatment=args.use_treatment, MC_sample=args.MC_sample)
        te_mae_epoch_loss_d += te_mae_batch_loss_d
        te_mae_epoch_loss_y += te_mae_batch_loss_y
        if args.use_treatment:
            te_loss_t1, te_loss_t2 = t_loss[0], t_loss[1]      
            te_mae_epoch_loss_t1 += te_loss_t1     
            te_mae_epoch_loss_t2 += te_loss_t2
            
        te_mse_epoch_loss_d += te_mse_batch_loss_d
        te_mse_epoch_loss_y += te_mse_batch_loss_y
        concat_te_num_data += te_num_data

        for i in range(te_num_data):
            # Extract data point from batch_data list (assuming the first element is the data tensor)
            cont_tensor_p, cont_tensor_c, cat_tensor_p, cat_tensor_c, data_len, yd, diff_tensor, treatment = batch_data
            
            
            mae_d = te_mae_batch_loss_d 
            mae_y = te_mae_batch_loss_y 
            
            # Create a dictionary for the current data point
            data_entry = {
                'cont_tensor_p': cont_tensor_p,
                'cont_tensor_c': cont_tensor_c,
                'cat_tensor_p': cat_tensor_p,
                'cat_tensor_c': cat_tensor_c,
                'data_len': data_len,
                'yd': yd,
                'diff_tensor': diff_tensor,
                'treatment': treatment,
                'mae_d': mae_d,
                'mae_y': mae_y
            }
            
            # Append the dictionary to the list
            error_analysis_list.append(data_entry)

        # Save the list containing tensors using torch.save()
        torch.save(error_analysis_list, f'./errorcase/{args.model}.pt')

