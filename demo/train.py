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
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset, TensorDataset
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

## Argparse ----------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Cluster Medical-AI")

parser.add_argument("--seed", type=int, default=1000, help="random seed (default: 1000)")

parser.add_argument("--eval_model", type=str, default=None,
    help="path to load saved model to evaluate model (default: None)",)

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

parser.add_argument("--alpha", type=float, default=1.0,
                help="cross entropy for t predction on dragonnet (Default : 1.0)")
# Data ---------------------------------------------------------
parser.add_argument(
    "--data_path",
    type=str,
    default='./demo_dataset/',
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

# parser.add_argument("--treatment_var", type=str, default='danger', choices=['dis, danger'], help="treatment variable")

parser.add_argument('--single_treatment', action='store_true', help='use only <dis> variable as treatment (default false)')

parser.add_argument('--shift', action='store_true', help='do not use treatment as feature (default false)')

parser.add_argument(
    "--MC_sample",
    type=int, default=1,
    help="Counts of Monte Carlo resampling"
)

parser.add_argument("--cutoff_dataset",
                    type=int, default=0,
                    help="If 0, uses as combined augmented dataset")

#----------------------------------------------------------------


# Model ---------------------------------------------------------
parser.add_argument(
    "--model",
    type=str, default='cet',
    choices=["cet", "cevae", "transformer", "linear", "ridge", "mlp", "svr", "rfr", 'tarnet', 'dragonnet', 'iTransformer'],
    help="model name (default : cet)")

parser.add_argument("--save_path",
            type=str, default="./best_models/",
            help="Path to save best model dict")

parser.add_argument(
    "--num_features",
    type=int, default=64,
    help="feature size (default : 64)"
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
    type=int, default=8,
    help="Transformer model head num (default : 8)"
)

parser.add_argument(
    "--output_size",
    type=int, default=2,
    help="Output size (default : 2)"
)

parser.add_argument(
    "--drop_out",
    type=float, default=0.1,
    help="Dropout Rate (Default : 0.1)"
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
parser.add_argument("--lr_init", type=float, default=1e-4,
                help="learning rate (Default : 0.01)")

parser.add_argument("--optim", type=str, default="adam",
                    choices=["sgd", "adam", "radam", "adamw"],
                    help="Optimization options")

parser.add_argument("--momentum", type=float, default=0.9,
                help="momentum (Default : 0.9)")

parser.add_argument("--epochs", type=int, default=300, metavar="N",
    help="number epochs to train (Default : 300)")

parser.add_argument("--wd", type=float, default=5e-3, help="weight decay (Default: 5e-3)")

parser.add_argument("--scheduler", type=str, default='cos_anneal', choices=['constant', "cos_anneal"])

parser.add_argument("--t_max", type=int, default=300,
                help="T_max for Cosine Annealing Learning Rate Scheduler (Default : 300)")

parser.add_argument("--lambdas", nargs='+', type=float, default=[1.0, 1e-6, 1e-6], help='pred loss + kld loss + recon loss')

parser.add_argument("--sig_x0", type=float, default=0.75)
#----------------------------------------------------------------

parser.add_argument("--lamb", type=float, default=0.0,
                help="Penalty term for Ridge Regression (Default : 0)")

parser.add_argument(
    "--intervene_var",
    type=str, default='t1', choices=["t1", "t2"],
    help="Intervention variable for Causal Effect Estimation (default : t1)")

parser.add_argument("--eval_per_cutoffs", action='store_true')

parser.add_argument('--is_synthetic', action='store_true', help='use synthetic dataset (default false)')

args = parser.parse_args()
args.t_max = args.epochs
## ----------------------------------------------------------------------------------------------------


## Set seed and device ----------------------------------------------------------------
utils.set_seed(args.seed)

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Device : {args.device}")
#-------------------------------------------------------------------------------------
       
## Load Data --------------------------------------------------------------------------------
### ./data/data_mod.ipynb 에서 기본적인 데이터 전처리  ###
if args.is_synthetic:
    # with open('./data/synthetic/synthetic_ts.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # dataset = utils.SyntheticTimeSeriesDataset(args, data)
    
    print('using synthetic data')
    with open('./data/synthetic/synthetic_dowhy.pkl', 'rb') as f:
        data = pickle.load(f)
    dataset = utils.SyntheticDataset(args, data)
else:
    dataset = utils.Tabledata(args, pd.read_csv(args.data_path+f"demo_data.csv"), args.scaling)

train_dataset, val_dataset, test_dataset = random_split(dataset, utils.data_split_num(dataset))
tr_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
print(f"Number of Data Clusters | train: {len(train_dataset)} | valid: {len(val_dataset)}")

val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
# print(len(train_dataset), len(val_dataset), len(test_dataset))
if args.eval_per_cutoffs:
    all_diff_tensors = []

    # 데이터를 순회하며 diff_tensor 수집
    for *features, diff_tensor, treatment in test_dataloader:
        all_diff_tensors.append(diff_tensor)

    # 모든 diff_tensor를 하나로 결합
    all_diff_tensors = torch.cat(all_diff_tensors)

    # 최대 unique diff_tensor 값을 구함
    max_diff = int(max(all_diff_tensors.unique()).item())
    # 데이터 분할을 위한 구조 초기화
    grouped_data = defaultdict(list)

    # 데이터를 다시 순회하며 분류
    for *features, diff_tensor, treatment in test_dataloader:
        for i in range(diff_tensor.size(0)):
            key = int(max(diff_tensor.squeeze()[i].unique()))
            grouped_data[key].append(tuple(feature[i] for feature in features) + (diff_tensor[i],) + (treatment[i],))

    # 각 diff_tensor 값별로 새로운 DataLoader 생성
    grouped_dataloaders = {}
    for key in range(max_diff + 1):
        if key in grouped_data:
            group_data = list(zip(*grouped_data[key]))
            tensor_datasets = [torch.stack(items) for items in group_data]
            dataset = TensorDataset(*tensor_datasets)
            if key == 0:
                test_dataloader = DataLoader(dataset, batch_size=len(dataset))
                print("use cut-off 0 testloader")
            grouped_dataloaders[key] = DataLoader(dataset, batch_size=len(dataset))

    # 생성된 각 DataLoader 정보 출력
    # for key, loader in grouped_dataloaders.items():
    #     print(f'DataLoader for group {key}: {len(loader.dataset)} items')

# print(f"use treatment as feature : {not args.use_treatment}")
# print("Successfully load data!")
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

if args.model == "mlp":
    model = models.MLPRegressor(args=args).to(args.device)

if args.model in ["linear", "ridge"]:
    model = models.LinearRegression(args=args).to(args.device)

if args.model == 'tarnet':
    model = models.TarNet(args = args,
                        input_size = args.num_features,
                        hidden_size = args.hidden_dim,
                        output_size = args.output_size,
                        disable_embedding = args.disable_embedding).to(args.device)
    
if args.model == 'dragonnet':
    model = models.DragonNet(args=args,
                        input_size = args.num_features,
                        hidden_size = args.hidden_dim,
                        output_size = args.output_size,
                        disable_embedding = args.disable_embedding).to(args.device)

if args.model == 'iTransformer':
    model = models.iTransformer(args=args,
                               input_size=args.num_features, 
                               hidden_size=args.hidden_dim, 
                               output_size=args.output_size, 
                               num_layers=args.num_layers, 
                               num_heads=args.num_heads, 
                               drop_out=args.drop_out,).to(args.device)

elif args.model in ["svr", "rfr"]:
    args.device = torch.device("cpu")
    ml_algorithm.fit(args.data_path, args.model, 0, args.table_idx)

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


if args.model =='tarnet':
    columns = ["ep", "lr", f"tr_loss_d", f"tr_loss_y", f"val_loss_d({args.eval_criterion})", f"val_loss_y({args.eval_criterion})",
           "te_loss_d(MAE)", "te_loss_y(MAE)", "te_loss_d(RMSE)", "te_loss_y(RMSE)"]
    ## print table index, 0 = cocnated data
    cutdates_num=0
    best_epochs=[0] * (cutdates_num+1) 
    best_val_loss_d = [9999] * (cutdates_num+1); best_val_loss_y = [9999] * (cutdates_num+1); best_val_loss_t = [9999] * (cutdates_num+1)
    best_val_models = [""] * (cutdates_num+1); best_tr_models = [""] * (cutdates_num+1)
    best_test_losses = [[9999 for j in range(4)] for i in range(cutdates_num+1)]
    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[0]['lr']
        tr_epoch_loss_d = 0; tr_epoch_loss_y = 0; tr_epoch_loss_t = 0
        val_epoch_loss_d = 0; val_epoch_loss_y = 0; val_epoch_loss_t = 0
        te_epoch_loss_t = 0
        te_mae_epoch_loss_d = 0; te_mae_epoch_loss_y = 0; te_mae_epoch_loss_t = 0
        te_mse_epoch_loss_d = 0; te_mse_epoch_loss_y = 0; te_mse_epoch_loss_t = 0
        
        concat_tr_num_data = 0; concat_val_num_data = 0; concat_te_num_data = 0

        tr_gt_y_list = []; val_gt_y_list = []; te_gt_y_list = []
        tr_pred_y_list = []; val_pred_y_list = []; te_pred_y_list = []
        
        tr_gt_d_list = []; val_gt_d_list = []; te_gt_d_list = []
        tr_pred_d_list = []; val_pred_d_list = []; te_pred_d_list = []
        
        ## Training Phase -----------------------------------------------------------------------------
        for itr, data in enumerate(tr_dataloader):
            ## Training phase
            tr_batch_loss_d, tr_batch_loss_y, tr_batch_loss_t, tr_num_data, tr_predicted, tr_ground_truth = utils.train_causal_model(args, data, model, optimizer, criterion)
                
            tr_epoch_loss_d += tr_batch_loss_d
            tr_epoch_loss_y += tr_batch_loss_y
            tr_epoch_loss_t += tr_batch_loss_t
            
            concat_tr_num_data += tr_num_data

            tr_pred_y_list += list(tr_predicted[:,0].cpu().detach().numpy())
            tr_gt_y_list += list(tr_ground_truth[:,0].cpu().detach().numpy())
            tr_pred_d_list += list(tr_predicted[:,1].cpu().detach().numpy())
            tr_gt_d_list += list(tr_ground_truth[:,1].cpu().detach().numpy())

        # Calculate Epoch loss
        tr_loss_d = tr_epoch_loss_d / concat_tr_num_data
        tr_loss_y = tr_epoch_loss_y / concat_tr_num_data
        
        if args.criterion == "RMSE":
            tr_loss_d = math.sqrt(tr_loss_d)
            tr_loss_y = math.sqrt(tr_loss_y)
        # ---------------------------------------------------------------------------------------

        ## Validation Phase ----------------------------------------------------------------------
        val_output=[]; test_output=[]
        val_loss_d_list = []; val_loss_y_list = [] ; val_loss_t_list = []
        test_mae_d_list = []; test_mae_y_list = [] ; test_rmse_d_list = []; test_rmse_y_list = []; test_t_list = []
        for i in range(cutdates_num+1):
            for itr, data in enumerate(val_dataloader):
                val_batch_loss_d, val_batch_loss_y, val_batch_loss_t, val_num_data, val_predicted, val_ground_truth = utils.valid_causal_model(args, data, model, eval_criterion,
                                                                                    args.scaling, val_dataset.dataset.a_y, val_dataset.dataset.b_y,
                                                                                    val_dataset.dataset.a_d, val_dataset.dataset.b_d)
                val_epoch_loss_d += val_batch_loss_d
                val_epoch_loss_y += val_batch_loss_y
                val_epoch_loss_t += val_batch_loss_t
                
                concat_val_num_data += val_num_data

            # Calculate Epoch loss
            val_loss_d = val_epoch_loss_d / concat_val_num_data
            val_loss_y = val_epoch_loss_y / concat_val_num_data
            val_loss_t = val_epoch_loss_t / concat_val_num_data
            if args.eval_criterion == "RMSE":
                val_loss_d = math.sqrt(val_loss_d)
                val_loss_y = math.sqrt(val_loss_y)

            # save list for all cut-off dates
            val_loss_d_list.append(val_loss_d)
            val_loss_y_list.append(val_loss_y)
            val_loss_t_list.append(val_loss_t)
            # ---------------------------------------------------------------------------------------

            ## Test Phase ----------------------------------------------------------------------
            for itr, data in enumerate(test_dataloader):
                te_mae_batch_loss_d, te_mae_batch_loss_y, te_mse_batch_loss_d, te_mse_batch_loss_y, te_batch_loss_t, te_num_data, te_predicted, te_ground_truth = utils.test_causal_model(args, data, model, args.scaling,
                                                                                                                                                    test_dataset.dataset.a_y, test_dataset.dataset.b_y,
                                                                                                                                                    test_dataset.dataset.a_d, test_dataset.dataset.b_d)

                te_mae_epoch_loss_d += te_mae_batch_loss_d
                te_mae_epoch_loss_y += te_mae_batch_loss_y
                te_mse_epoch_loss_d += te_mse_batch_loss_d
                te_mse_epoch_loss_y += te_mse_batch_loss_y
                
                te_epoch_loss_t += te_batch_loss_t
                
                concat_te_num_data += te_num_data

            # Calculate Epoch loss
            te_mae_loss_d = te_mae_epoch_loss_d / concat_te_num_data
            te_mae_loss_y = te_mae_epoch_loss_y / concat_te_num_data
            te_rmse_loss_d = math.sqrt(te_mse_epoch_loss_d / concat_te_num_data)
            te_rmse_loss_y = math.sqrt(te_mse_epoch_loss_y / concat_te_num_data)
            
            te_loss_t = te_epoch_loss_t / concat_te_num_data

            # save list for all cut-off dates
            test_mae_d_list.append(te_mae_loss_d);test_mae_y_list.append(te_mae_loss_y)
            test_rmse_d_list.append(te_rmse_loss_d); test_rmse_y_list.append(te_rmse_loss_y)
            test_t_list.append(te_loss_t)
            # ---------------------------------------------------------------------------------------
            
            # Save Best Model (Early Stopping)
            if val_loss_d + val_loss_y < best_val_loss_d[i] + best_val_loss_y[i]:
                best_epochs[i] = epoch
                best_val_loss_d[i] = val_loss_d
                best_val_loss_y[i] = val_loss_y
                best_val_loss_t[i] = val_loss_t

                best_test_losses[i][0] = te_mae_loss_d
                best_test_losses[i][1] = te_mae_loss_y
                best_test_losses[i][2] = te_rmse_loss_d
                best_test_losses[i][3] = te_rmse_loss_y
                
                best_model_weights = {key: value.clone() for key, value in model.state_dict().items()}
                # save state_dict
                # os.makedirs(args.save_path, exist_ok=True)
                os.makedirs("./best_models/seed_1000/", exist_ok=True)
                utils.save_checkpoint(file_path = f"./best_models/seed_1000/best_{args.model}-{args.optim}-{args.lr_init}-{args.wd}-{args.alpha}-{args.seed}-date{i}_best_val.pt",
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
                    val_df.to_csv(f"{args.save_path}/{args.model}-{args.optim}-{args.lr_init}-{args.wd}-{args.alpha}_best_val_pred.csv", index=False)

                    te_df = pd.DataFrame({'te_pred_y':te_pred_y_list,
                                    'te_ground_truth_y':te_gt_y_list,
                                    'te_pred_d' : te_pred_d_list,
                                    'te_ground_truth_d' : te_gt_d_list})                
                    te_df.to_csv(f"{args.save_path}/{args.model}-{args.optim}-{args.lr_init}-{args.wd}-{args.alpha}_best_te_pred.csv", index=False)

        # print values
        values = [epoch, lr, tr_loss_d, tr_loss_y, val_loss_d_list[args.table_idx], val_loss_y_list[args.table_idx], test_mae_d_list[args.table_idx], test_mae_y_list[args.table_idx], test_rmse_d_list[args.table_idx], test_rmse_y_list[args.table_idx]]    
        
        table = tabulate.tabulate([values], headers=columns, tablefmt="simple", floatfmt="8.4f")
        if epoch % 20 == 0 or epoch == 1:
            table = table.split("\n")
            table = "\n".join([table[1]] + table)
        else:
            table = table.split("\n")[2]
        # print(table)
        
        # step scheduler
        if args.scheduler == 'cos_anneal':
            scheduler.step()
else:
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
                if args.model == 'tarnet':
                    tr_batch_loss_d, tr_batch_loss_y, tr_batch_loss_t, tr_num_data, tr_predicted, tr_ground_truth = utils.train_causal_model(args, data, model, optimizer, criterion)
                else:
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
                # val_epoch_loss_t += t_loss[0]
            concat_val_num_data += val_num_data

            # val_pred_y_list += list(val_predicted[:,0].cpu().detach().numpy())
            # val_gt_y_list += list(val_ground_truth[:,0].cpu().detach().numpy())
            # val_pred_d_list += list(val_predicted[:,1].cpu().detach().numpy())
            # val_gt_d_list += list(val_ground_truth[:,1].cpu().detach().numpy())

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
        # val_loss_t_list.append(val_loss_t)
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

            # Restore Prediction and Ground Truth
            # te_pred_y, te_pred_d, te_gt_y, te_gt_d= utils.reverse_scaling(args.scaling, te_predicted, te_ground_truth, test_dataset.dataset.a_y, test_dataset.dataset.b_y, test_dataset.dataset.a_d, test_dataset.dataset.b_d)

            # te_pred_y_list += list(te_pred_y.cpu().detach().numpy())
            # te_gt_y_list += list(te_gt_y.cpu().detach().numpy())
            # te_pred_d_list += list(te_pred_d.cpu().detach().numpy())
            # te_gt_d_list += list(te_gt_d.cpu().detach().numpy())

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
            # if args.eval_per_cutoffs:
            #     best_cutoffs_mae_d={}
            #     best_cutoffs_mae_y={}
            #     best_cutoffs_mae_t1={}
            #     best_cutoffs_mae_t2={}
            #     best_cutoffs_rmse_d={}
            #     best_cutoffs_rmse_y={}
                
            #     for eval_key, gtest_dataloader in grouped_dataloaders.items():
            #         for _, data in enumerate(gtest_dataloader):
            #             cuteval_te_mae_batch_loss_d, cuteval_te_mae_batch_loss_y, cuteval_te_mse_batch_loss_d, cuteval_te_mse_batch_loss_y, cuteval_te_num_data, cuteval_te_predicted, cuteval_te_ground_truth, *cuteval_t_loss = utils.test(args, data, model,
            #                                                                     args.scaling, test_dataset.dataset.a_y, test_dataset.dataset.b_y,
            #                                                                     test_dataset.dataset.a_d, test_dataset.dataset.b_d, use_treatment=args.use_treatment, MC_sample=args.MC_sample)
            #             cuteval_te_mae_epoch_loss_d = cuteval_te_mae_batch_loss_d
            #             cuteval_te_mae_epoch_loss_y = cuteval_te_mae_batch_loss_y
            #             if args.use_treatment:
            #                 # te_mae_epoch_loss_t += t_loss[0]
            #                 cuteval_te_loss_t1, cuteval_te_loss_t2 = cuteval_t_loss[0], cuteval_t_loss[1]      
            #                 cuteval_te_mae_epoch_loss_t1 = cuteval_te_loss_t1     
            #                 cuteval_te_mae_epoch_loss_t2 = cuteval_te_loss_t2
                            
            #             cuteval_te_mse_epoch_loss_d = cuteval_te_mse_batch_loss_d
            #             cuteval_te_mse_epoch_loss_y = cuteval_te_mse_batch_loss_y
            #             cuteval_concat_te_num_data = cuteval_te_num_data

            #         # Calculate Epoch loss
            #         cuteval_te_mae_loss_d = cuteval_te_mae_epoch_loss_d / cuteval_concat_te_num_data
            #         cuteval_te_mae_loss_y = cuteval_te_mae_epoch_loss_y / cuteval_concat_te_num_data
            #         cuteval_te_mae_loss_t1 = cuteval_te_mae_epoch_loss_t1 * 6 / cuteval_concat_te_num_data
            #         cuteval_te_mae_loss_t2 = (cuteval_te_mae_epoch_loss_t2 * 8 + 3) / cuteval_concat_te_num_data
            #         cuteval_te_rmse_loss_d = math.sqrt(cuteval_te_mse_epoch_loss_d / cuteval_concat_te_num_data)
            #         cuteval_te_rmse_loss_y = math.sqrt(cuteval_te_mse_epoch_loss_y / cuteval_concat_te_num_data)
                    
            #         best_cutoffs_mae_d[eval_key]=cuteval_te_mae_loss_d
            #         best_cutoffs_mae_y[eval_key]=cuteval_te_mae_loss_y
            #         best_cutoffs_mae_t1[eval_key]=cuteval_te_mae_loss_t1
            #         best_cutoffs_mae_t2[eval_key]=cuteval_te_mae_loss_t2
            #         best_cutoffs_rmse_d[eval_key]=cuteval_te_rmse_loss_d
            #         best_cutoffs_rmse_y[eval_key]=cuteval_te_rmse_loss_y
                
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
            save_path=f"./best_models/seed_{args.seed}" if not args.is_synthetic else f"./best_syn_model/seed_{args.seed}"
            os.makedirs(save_path, exist_ok=True)
            utils.save_checkpoint(file_path = f"{save_path}/best_{args.model}-{args.optim}-{args.lr_init}-{args.wd}-{args.drop_out}-{args.seed}-date{i}_best_val.pt",
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
        # print(table)
        
        # step scheduler
        if args.scheduler == 'cos_anneal':
            scheduler.step()

    # ---------------------------------------------------------------------------------------------

model.load_state_dict(best_model_weights)
# Estimate Population average treatment effects
# negative_acc_y_t1, negative_acc_d_t1, ce_y_t1, ce_d_t1 = utils.CE(args, model, val_dataloader, 't1')
if args.model == 'tarnet':
    negative_acc_y_t2, negative_acc_d_t2, ce_y_t2, ce_d_t2 = utils.iTrans_CE(args, model, val_dataloader, 't2')
else:
    negative_acc_y_t2, negative_acc_d_t2, ce_y_t2, ce_d_t2 = utils.CE(args, model, val_dataloader, 't2')
# pehe_y, ate_error_y = utils.PEHE(args, model, val_dataloader, 't2')

# utils.save_posterior(tr_dataloader, model, f'/data1/bubble3jh/cluster-regression/data/synthetic/synthetic_dowhy_posterior_{args.model}.pkl')

## Print Best Model ---------------------------------------------------------------------------
# print(f"Best {args.model} achieved [d:{best_test_losses[args.table_idx][0]}, y:{best_test_losses[args.table_idx][1]}] on {best_epochs[args.table_idx]} epoch!!")
print(f"Training completed.\nThe model saved as '{args.save_path}{args.model}-{args.optim}-{args.lr_init}-{args.wd}-{args.drop_out}.pt'!!")
    # ---------------------------------------------------------------------------------------------
