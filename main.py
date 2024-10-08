import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle 

import os, time
import math

import argparse
import tabulate

import utils, models
import wandb
from torch.utils.data import DataLoader, random_split
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
 
# Data ---------------------------------------------------------
parser.add_argument(
    "--data_path",
    type=str,
    default='./data/',
    help="path to datasets location",)

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
    type=str, default='cevt',
    choices=["cevt", "cevae", "transformer", "linear", "ridge", "mlp", 'tarnet', 'dragonnet', 'iTransformer'],
    help="model name (default : cet)")

parser.add_argument("--save_path",
            type=str, default="./best_model/",
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

args = parser.parse_args()
## ----------------------------------------------------------------------------------------------------


## Set seed and device ----------------------------------------------------------------
utils.set_seed(args.seed)

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {args.device}")
#-------------------------------------------------------------------------------------

## Set wandb ---------------------------------------------------------------------------
if args.ignore_wandb == False:
    wandb.init(entity="your_entity", project="your_project", group=args.run_group)
    wandb.config.update(args)
    if args.disable_embedding:
        wandb.run.name = f"raw_{args.model}({args.hidden_dim})-{args.optim}-{args.lr_init}-{args.wd}-{args.drop_out}"
    else:
        wandb.run.name = f"embed_{args.model}({args.hidden_dim})-{args.optim}-{args.lr_init}-{args.wd}-{args.drop_out}"
       
## Load Data --------------------------------------------------------------------------------
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
    
if args.model == 'cevt':
    assert(args.use_treatment == True)
    model = models.CEVT(args).to(args.device) 
    
if args.model == 'cevae':
    assert(args.use_treatment == True)
    assert(args.single_treatment == True)
    model = models.CEVAE(args).to(args.device) 

if args.model == "mlp":
    model = models.MLPRegressor(args=args).to(args.device)

if args.model in ["linear", "ridge"]:
    model = models.LinearRegression(args=args).to(args.device)

if args.model == 'tarnet':
    raise('use run_causal.py')
    
if args.model == 'dragonnet':
    raise('use run_causal.py')

if args.model == 'iTransformer':
    raise('use run_itransformer.py')

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

lambda0 = args.lambdas[1]
for epoch in range(1, args.epochs + 1):
    lr = optimizer.param_groups[0]['lr']
    tr_epoch_eval_loss_d=0; tr_epoch_eval_loss_y=0; tr_epoch_eval_loss_t1=0; tr_epoch_eval_loss_t2=0; tr_epoch_loss_d = 0; tr_epoch_loss_y = 0; val_epoch_loss_d = 0; val_epoch_loss_y = 0; val_epoch_loss_t1 = 0; val_epoch_loss_t2 = 0; te_mae_epoch_loss_d = 0; te_mae_epoch_loss_y = 0; te_mae_epoch_loss_t1 = 0; te_mae_epoch_loss_t2 = 0; te_mse_epoch_loss_d = 0; te_mse_epoch_loss_y = 0; te_mse_epoch_loss_t1 = 0; te_mse_epoch_loss_t2 = 0
    tr_epoch_pred_loss=0; tr_epoch_kl_loss=0; tr_epoch_recon_loss=0
    
    concat_tr_num_data = 0; concat_val_num_data = 0; concat_te_num_data = 0

    tr_gt_y_list = []; val_gt_y_list = []; te_gt_y_list = []
    tr_pred_y_list = []; val_pred_y_list = []; te_pred_y_list = []
    
    tr_gt_d_list = []; val_gt_d_list = []; te_gt_d_list = []
    tr_pred_d_list = []; val_pred_d_list = []; te_pred_d_list = []
    # kld loss sigmoid scheduling 
    args.lambdas[1] = lambda0*utils.sigmoid_annealing(epoch, args.epochs, k=15, x0=args.sig_x0)
    if args.eval_model == None:
        for itr, data in enumerate(tr_dataloader):
            ## Training phase
            tr_batch_loss_d, tr_batch_loss_y, tr_num_data, tr_predicted, tr_ground_truth, tr_eval_loss_y, tr_eval_loss_d, tr_eval_model, *indv_loss = utils.train(args, data, model, optimizer, criterion, epoch, lamb=args.lamb, eval_criterion=eval_criterion,
                                                                                                                                        a_y=train_dataset.dataset.a_y, a_d=train_dataset.dataset.a_d, b_y=train_dataset.dataset.b_y, b_d=train_dataset.dataset.b_d,
                                                                                                                                        use_treatment=args.use_treatment, lambdas=args.lambdas)
            tr_epoch_loss_d += tr_batch_loss_d
            tr_epoch_loss_y += tr_batch_loss_y
            tr_epoch_eval_loss_d += tr_eval_loss_d
            tr_epoch_eval_loss_y += tr_eval_loss_y
            if args.use_treatment:     
                tr_eval_loss_t1, tr_eval_loss_t2 = indv_loss[0]       
                tr_epoch_eval_loss_t1 += tr_eval_loss_t1     
                tr_epoch_eval_loss_t2 += tr_eval_loss_t2
            
            tr_epoch_pred_loss += indv_loss[1][0]
            tr_epoch_kl_loss += indv_loss[1][1]
            tr_epoch_recon_loss += indv_loss[1][2]
            
            concat_tr_num_data += tr_num_data

            tr_pred_y_list += list(tr_predicted[:,0].cpu().detach().numpy())
            tr_gt_y_list += list(tr_ground_truth[:,0].cpu().detach().numpy())
            tr_pred_d_list += list(tr_predicted[:,1].cpu().detach().numpy())
            tr_gt_d_list += list(tr_ground_truth[:,1].cpu().detach().numpy())

        # Calculate Epoch loss
        tr_loss_d = tr_epoch_loss_d / concat_tr_num_data
        tr_loss_y = tr_epoch_loss_y / concat_tr_num_data
        tr_eval_loss_d = tr_epoch_eval_loss_d / concat_tr_num_data
        tr_eval_loss_y = tr_epoch_eval_loss_y / concat_tr_num_data
        tr_eval_loss_t1 = tr_epoch_eval_loss_t1 * 6 / concat_tr_num_data       # denorm
        tr_eval_loss_t2 = (tr_epoch_eval_loss_t2 * 8 + 3) / concat_tr_num_data # denorm
        
        tr_pred_loss = tr_epoch_pred_loss / concat_tr_num_data
        tr_kl_loss = tr_epoch_kl_loss / concat_tr_num_data
        tr_recon_loss = tr_epoch_recon_loss / concat_tr_num_data
        
        if args.criterion == "RMSE":
            tr_loss_d = math.sqrt(tr_loss_d)
            tr_loss_y = math.sqrt(tr_loss_y)
        # ---------------------------------------------------------------------------------------
    
    val_output=[]; test_output=[]
    val_loss_d_list = []; val_loss_y_list = [] ; val_loss_t1_list = []; val_loss_t2_list = []
    test_mae_d_list = []; test_mae_y_list = [] ; test_mae_t1_list = [] ; test_mae_t2_list = [] ; test_rmse_d_list = []; test_rmse_y_list = []
    ## Validation Phase ----------------------------------------------------------------------
    for itr, data in enumerate(val_dataloader):
        val_batch_loss_d, val_batch_loss_y, val_num_data, val_predicted, val_ground_truth, eval_model, *t_loss = utils.valid(args, data, model, eval_criterion,
                                                                            args.scaling, val_dataset.dataset.a_y, val_dataset.dataset.b_y,
                                                                            val_dataset.dataset.a_d, val_dataset.dataset.b_d, use_treatment=args.use_treatment, MC_sample=args.MC_sample)
        val_epoch_loss_d += val_batch_loss_d
        val_epoch_loss_y += val_batch_loss_y
        if args.use_treatment:            
            val_loss_t1, val_loss_t2 = t_loss[0], t_loss[1]       
            val_epoch_loss_t1 += val_loss_t1     
            val_epoch_loss_t2 += val_loss_t2
        concat_val_num_data += val_num_data

    # Calculate Epoch loss
    val_loss_d = val_epoch_loss_d / concat_val_num_data
    val_loss_y = val_epoch_loss_y / concat_val_num_data
    val_loss_t1 = val_epoch_loss_t1 * 6 / concat_val_num_data
    val_loss_t2 = (val_epoch_loss_t2 * 8 + 3) / concat_val_num_data
    if args.eval_criterion == "RMSE":
        val_loss_d = math.sqrt(val_loss_d)
        val_loss_y = math.sqrt(val_loss_y)

    # save list for all cut-off dates
    val_loss_d_list.append(val_loss_d)
    val_loss_y_list.append(val_loss_y)
    val_loss_t1_list.append(val_loss_t1)
    val_loss_t2_list.append(val_loss_t2)
    
    # ---------------------------------------------------------------------------------------

    ## Test Phase ----------------------------------------------------------------------
    for itr, data in enumerate(test_dataloader):
        te_mae_batch_loss_d, te_mae_batch_loss_y, te_mse_batch_loss_d, te_mse_batch_loss_y, te_num_data, te_predicted, te_ground_truth, *t_loss = utils.test(args, data, model,
                                                                            args.scaling, test_dataset.dataset.a_y, test_dataset.dataset.b_y,
                                                                            test_dataset.dataset.a_d, test_dataset.dataset.b_d, use_treatment=args.use_treatment, MC_sample=args.MC_sample)
        te_mae_epoch_loss_d += te_mae_batch_loss_d
        te_mae_epoch_loss_y += te_mae_batch_loss_y
        if args.use_treatment:
            # te_mae_epoch_loss_t += t_loss[0]
            te_loss_t1, te_loss_t2 = t_loss[0], t_loss[1]      
            te_mae_epoch_loss_t1 += te_loss_t1     
            te_mae_epoch_loss_t2 += te_loss_t2
            
        te_mse_epoch_loss_d += te_mse_batch_loss_d
        te_mse_epoch_loss_y += te_mse_batch_loss_y
        concat_te_num_data += te_num_data

    # Calculate Epoch loss
    te_mae_loss_d = te_mae_epoch_loss_d / concat_te_num_data
    te_mae_loss_y = te_mae_epoch_loss_y / concat_te_num_data
    te_mae_loss_t1 = te_mae_epoch_loss_t1 * 6 / concat_te_num_data
    te_mae_loss_t2 = (te_mae_epoch_loss_t2 * 8 + 3) / concat_te_num_data
    te_rmse_loss_d = math.sqrt(te_mse_epoch_loss_d / concat_te_num_data)
    te_rmse_loss_y = math.sqrt(te_mse_epoch_loss_y / concat_te_num_data)

    # save list for all cut-off dates
    test_mae_d_list.append(te_mae_loss_d);test_mae_y_list.append(te_mae_loss_y); test_mae_t1_list.append(te_mae_loss_t1); test_mae_t2_list.append(te_mae_loss_t2)
    test_rmse_d_list.append(te_rmse_loss_d); test_rmse_y_list.append(te_rmse_loss_y)

    # ---------------------------------------------------------------------------------------
    
    # Save Best Model (Early Stopping)
    i=0
    if val_loss_d + val_loss_y < best_val_loss_d[i] + best_val_loss_y[i]:
            
        best_epochs[i] = epoch
        best_val_loss_d[i] = val_loss_d
        best_val_loss_y[i] = val_loss_y
        best_val_loss_t1[i] = val_loss_t1
        best_val_loss_t2[i] = val_loss_t2

        best_test_losses[i][0] = te_mae_loss_d
        best_test_losses[i][1] = te_mae_loss_y
        best_test_losses[i][2] = te_rmse_loss_d
        best_test_losses[i][3] = te_rmse_loss_y

        best_val_models[i] = eval_model
        best_tr_models[i] = tr_eval_model
        
        # best_model = model
        best_model_weights = {key: value.clone() for key, value in model.state_dict().items()}
        # save state_dict
        os.makedirs(args.save_path, exist_ok=True)
        utils.save_checkpoint(file_path = f"{args.save_path}/best_{args.model}-{args.optim}-{args.lr_init}-{args.wd}-{args.drop_out}-{args.seed}-date{i}_best_val.pt",
                            epoch = epoch,
                            state_dict = model.state_dict(),
                            optimizer = optimizer.state_dict(),
                            )
        
        if args.save_pred:
            # save prediction and ground truth as csv
            val_df = pd.DataFrame({'val_pred_y':val_pred_y_list,
                            'val_ground_truth_y':val_gt_y_list,
                            'val_pred_d' : val_pred_d_list,
                            'val_ground_truth_d' : val_gt_d_list})
            val_df.to_csv(f"{args.save_path}/{args.model}-{args.optim}-{args.lr_init}-{args.wd}-{args.drop_out}_best_val_pred.csv", index=False)

            te_df = pd.DataFrame({'te_pred_y':te_pred_y_list,
                            'te_ground_truth_y':te_gt_y_list,
                            'te_pred_d' : te_pred_d_list,
                            'te_ground_truth_d' : te_gt_d_list})                
            te_df.to_csv(f"{args.save_path}/{args.model}-{args.optim}-{args.lr_init}-{args.wd}-{args.drop_out}_best_te_pred.csv", index=False)

    # print values
    values = [epoch, lr, tr_eval_loss_d, tr_eval_loss_y, val_loss_d_list[args.table_idx], val_loss_y_list[args.table_idx], test_mae_d_list[args.table_idx], test_mae_y_list[args.table_idx], test_rmse_d_list[args.table_idx], test_rmse_y_list[args.table_idx], eval_model]    
    
    table = tabulate.tabulate([values], headers=columns, tablefmt="simple", floatfmt="8.4f")
    if epoch % 20 == 0 or epoch == 1:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)
    
    # step scheduler
    if args.scheduler == 'cos_anneal':
        scheduler.step()
    
    # update wandb
    if not args.ignore_wandb:
        wandb_log = {
        "train/d": tr_loss_d,
        "train/y": tr_loss_y,
        "train_eval/d": tr_eval_loss_d,
        "train_eval/y": tr_eval_loss_y,
        "train_eval/t ": tr_eval_loss_t,
        "train_pred_loss": tr_pred_loss,
        "train_kld_loss" : tr_kl_loss,
        "train_reconstruction_loss": tr_recon_loss, 
        "concat/valid_d": val_loss_d_list[0],
        "concat/valid_y": val_loss_y_list[0],
        "concat/valid_t1 ": val_loss_t1_list[0],
        "concat/valid_t2 ": val_loss_t2_list[0],
        "concat/test_total (mae)": test_mae_d_list[0] + test_mae_y_list[0],
        "concat/test_d (mae)": test_mae_d_list[0],
        "concat/test_y (mae)": test_mae_y_list[0],
        "concat/test_t1 (mae)": test_mae_t1_list[0],
        "concat/test_t2 (mae)": test_mae_t2_list[0],
        "concat/test_total (rmse)": test_rmse_d_list[0] + test_rmse_y_list[0],
        "concat/test_d (rmse)": test_rmse_d_list[0],
        "concat/test_y (rmse)": test_rmse_y_list[0],
        "setting/lr": lr,
        "setting/kld_lambda": args.lambdas[1]
    }

        for i in range(1, cutdates_num+1):
            wandb_log.update({
                f"valid_d": val_loss_d_list[i],
                f"valid_y": val_loss_y_list[i],
                f"valid_t1 ": val_loss_t1_list[i],
                f"valid_t2 ": val_loss_t2_list[i],
                f"test_total (mae)": test_mae_d_list[i] + test_mae_y_list[i],
                f"test_d (mae)": test_mae_d_list[i],
                f"test_y (mae)": test_mae_y_list[i],
                f"test_t1 (mae)": test_mae_t1_list[i],
                f"test_t2 (mae)": test_mae_t2_list[i],
                f"test_total (rmse)": test_rmse_d_list[i] + test_rmse_y_list[i],
                f"test_d (rmse)": test_rmse_d_list[i],
                f"test_y (rmse)": test_rmse_y_list[i],
            })

        wandb.log(wandb_log)
# ---------------------------------------------------------------------------------------------

model.load_state_dict(best_model_weights)
# Estimate Population average treatment effects
negative_acc_y_t1, negative_acc_d_t1, ce_y_t1, ce_d_t1 = utils.CE(args, model, val_dataloader, 't1')
negative_acc_y_t2, negative_acc_d_t2, ce_y_t2, ce_d_t2 = utils.CE(args, model, val_dataloader, 't2')

## Print Best Model ---------------------------------------------------------------------------
print(f"Best {args.model} achieved [d:{best_test_losses[args.table_idx][0]}, y:{best_test_losses[args.table_idx][1]}] on {best_epochs[args.table_idx]} epoch!!")
print(f"The model saved as '{args.save_path}{args.model}-{args.optim}-{args.lr_init}-{args.wd}-{args.drop_out}_best_val.pt'!!")
if args.ignore_wandb == False:
    for i in range(cutdates_num+1):
        date_key = ""
        wandb.run.summary[f"best_epoch {date_key}"] = best_epochs[i]
        wandb.run.summary[f"best_tr_models {date_key}"] = best_tr_models[i]
        wandb.run.summary[f"best_val_models {date_key}"] = best_val_models[i]
        wandb.run.summary[f"best_val_loss (d) {date_key}"] = best_val_loss_d[i]
        wandb.run.summary[f"best_val_loss (y) {date_key}"] = best_val_loss_y[i]
        wandb.run.summary[f"best_val_loss (t1) {date_key}"] = best_val_loss_t1[i]
        wandb.run.summary[f"best_val_loss (t2) {date_key}"] = best_val_loss_t2[i]
        wandb.run.summary[f"best_val_loss {date_key}"] = best_val_loss_d[i] + best_val_loss_y[i]
        wandb.run.summary[f"best_test_mae_loss (d)"] = best_test_losses[i][0]
        wandb.run.summary[f"best_test_mae_loss (y)"] = best_test_losses[i][1]
        wandb.run.summary[f"best_test_mae_loss"] = best_test_losses[i][0] + best_test_losses[i][1]
        wandb.run.summary[f"best_test_rmse_loss (d)"] = best_test_losses[i][2]
        wandb.run.summary[f"best_test_rmse_loss (y)"] = best_test_losses[i][3]
        wandb.run.summary[f"best_test_rmse_loss"] = best_test_losses[i][2] + best_test_losses[i][3]
    
    wandb.run.summary["CE_y (t1)"] = ce_y_t1
    wandb.run.summary["CE_y (t2)"] = ce_y_t2
    wandb.run.summary["CE_d (t1)"] = ce_d_t1
    wandb.run.summary["CE_d (t2)"] = ce_d_t2
    wandb.run.summary["CACC_y (t1)"] = negative_acc_y_t1
    wandb.run.summary["CACC_y (t2)"] = negative_acc_y_t2
    wandb.run.summary["CACC_d (t1)"] = negative_acc_d_t1
    wandb.run.summary["CACC_d (t2)"] = negative_acc_d_t2
    wandb.run.summary["CACC_avg (t1)"] = (negative_acc_y_t1 + negative_acc_d_t1)/2
    wandb.run.summary["CACC_avg (t2)"] = (negative_acc_y_t2 + negative_acc_d_t2)/2
# ---------------------------------------------------------------------------------------------
