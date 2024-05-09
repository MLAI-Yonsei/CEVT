import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from utils import reduction_cluster, reparametrize
import pdb
import warnings
from torch.nn.modules.transformer import _get_seq_len, _detect_is_causal_mask

warnings.filterwarnings("ignore", "Converting mask without torch.bool dtype to bool")

class MLPRegressor(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_size=args.num_features
        hidden_size=args.hidden_dim
        disable_embedding=args.disable_embedding
        self.num_layers = args.num_layers
        
        if disable_embedding:
            input_size = 12
        if args.is_synthetic:
            self.embedding = SyntheticEmbedding(args, output_size=args.num_features, disable_embedding = False, disable_pe=True, reduction="mean", shift= args.shift, use_treatment=args.use_treatment)
        else: 
            self.embedding = TableEmbedding(input_size, disable_embedding = disable_embedding, disable_pe=True, reduction="mean",  use_treatment=args.use_treatment)
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size, bias=True)])
        for _ in range(args.num_layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size, bias=True))
        self.layers.append(nn.Linear(hidden_size, args.output_size, bias=True))
        self.dropout = nn.Dropout(args.drop_out)
        
    def forward(self, cont_p, cont_c, cat_p, cat_c, len, diff_days):
        x = self.embedding(cont_p, cont_c, cat_p, cat_c, len, diff_days)
        for i, layer in enumerate(self.layers):
            if i == self.num_layers - 1:
                x = layer(x)  
            else:
                x = self.dropout(F.relu(layer(x)))
        return x

class LinearRegression(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        input_size=args.num_features

        if args.disable_embedding:
            input_size = 12
        if args.is_synthetic:
            self.embedding = SyntheticEmbedding(args, output_size=args.num_features, disable_embedding = False, disable_pe=True, reduction="mean", shift= args.shift, use_treatment=args.use_treatment)
        else: 
            self.embedding = TableEmbedding(input_size, disable_embedding = args.disable_embedding, disable_pe=True, reduction="mean",  use_treatment=args.use_treatment)
        self.linear1 = torch.nn.Linear(input_size, args.output_size)

    def forward(self, cont_p, cont_c, cat_p, cat_c, len, diff_days):
        x = self.embedding(cont_p, cont_c, cat_p, cat_c, len, diff_days)
        x = self.linear1(x)
        return x

class Transformer(nn.Module):
    '''
        input_size : TableEmbedding 크기
        hidden_size : Transformer Encoder 크기
        output_size : y, d (2)
        num_layers : Transformer Encoder Layer 개수
        num_heads : Multi Head Attention Head 개수
        drop_out : DropOut 정도
        disable_embedding : 연속 데이터 embedding 여부
    '''
    def __init__(self, args):
        super(Transformer, self).__init__()
        
        if args.is_synthetic:
            self.embedding = SyntheticEmbedding(args, output_size=args.num_features, disable_embedding = False, disable_pe=False, reduction="none", shift= args.shift, use_treatment=args.use_treatment)
        else:        
            self.embedding = TableEmbedding(output_size=args.num_features, disable_embedding = args.disable_embedding, disable_pe=False, reduction="none", use_treatment=args.use_treatment) #reduction="date")
        self.cls_token = nn.Parameter(torch.randn(1, 1, args.num_features))
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=args.num_features,
            nhead=args.num_heads,
            dim_feedforward=args.hidden_dim, 
            dropout=args.drop_out,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = TransformerEncoder(self.transformer_layer, args.num_layers)
        self.fc = nn.Linear(args.num_features, args.output_size)  

        self.init_weights()

    ## Transformer Weight 초기화 방법 ##
    def init_weights(self) -> None:
        initrange = 0.1
        for module in self.embedding.modules():
                if isinstance(module, nn.Linear) :
                    module.weight.data.uniform_(-initrange, initrange)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.Embedding):
                    module.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, cont_p, cont_c, cat_p, cat_c, val_len, diff_days):
        if self.embedding.reduction != "none":
            embedded, cls_token_pe = self.embedding(cont_p, cont_c, cat_p, cat_c, val_len, diff_days)
        else:
            (embedded, diff_days, _), cls_token_pe = self.embedding(cont_p, cont_c, cat_p, cat_c, val_len, diff_days) # embedded:(32, 124, 128)
        
        cls_token = self.cls_token.expand(embedded.size(0), -1, -1) + cls_token_pe.unsqueeze(0).expand(embedded.size(0), -1, -1)
        input_with_cls = torch.cat([cls_token, embedded], dim=1)
        mask = ~(torch.arange(input_with_cls.size(1)).expand(input_with_cls.size(0), -1).cuda() < (val_len+1).unsqueeze(1)).cuda() # val_len + 1 ?
        
        output = self.transformer_encoder(input_with_cls, src_key_padding_mask=mask.bool())  
        cls_output = output[:, 0, :] 
        regression_output = self.fc(cls_output) 
        
        return regression_output
    
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(0), :]
    
class TableEmbedding(torch.nn.Module):
    '''
        output_size : embedding output의 크기
        disable_embedding : 연속 데이터의 임베딩 유무
        disable_pe : transformer의 sequance 기준 positional encoding add 유무
        reduction : "mean" : cluster 별 평균으로 reduction
                    "date" : cluster 내 date 평균으로 reduction
    '''
    def __init__(self, output_size=128, disable_embedding=False, disable_pe=True, reduction="mean", use_treatment=False):
        super().__init__()
        self.reduction = reduction
        if self.reduction == "none":
            print("do not reduce cluster")
        self.disable_embedding = disable_embedding
        self.disable_pe = disable_pe
        if not disable_embedding:
            print("Embedding applied to data")
            nn_dim = emb_hidden_dim = emb_dim_c = emb_dim_p = output_size//4
            self.cont_p_NN = nn.Sequential(nn.Linear(3, emb_hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(emb_hidden_dim, nn_dim))
            self.cont_c_NN = nn.Sequential(nn.Linear(1 if use_treatment else 2, emb_hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(emb_hidden_dim, nn_dim))
        else:
            emb_dim_p = 5
            emb_dim_c = 2
        self.lookup_gender  = nn.Embedding(2, emb_dim_p)
        self.lookup_korean  = nn.Embedding(2, emb_dim_p)
        self.lookup_primary  = nn.Embedding(2, emb_dim_p)
        self.lookup_job  = nn.Embedding(11, emb_dim_p)
        self.lookup_rep  = nn.Embedding(34, emb_dim_p)
        self.lookup_place  = nn.Embedding(19, emb_dim_c)
        self.lookup_add  = nn.Embedding(31, emb_dim_c)
        if not disable_pe:
            self.positional_embedding  = nn.Embedding(6, output_size)

    def forward(self, cont_p, cont_c, cat_p, cat_c, val_len, diff_days):
        if not self.disable_embedding:
            cont_p_emb = self.cont_p_NN(cont_p)
            cont_c_emb = self.cont_c_NN(cont_c)
        a1_embs = self.lookup_gender(cat_p[:,:,0].to(torch.int))
        a2_embs = self.lookup_korean(cat_p[:,:,1].to(torch.int))
        a3_embs = self.lookup_primary(cat_p[:,:,2].to(torch.int))
        a4_embs = self.lookup_job(cat_p[:,:,3].to(torch.int))
        a5_embs = self.lookup_rep(cat_p[:,:,4].to(torch.int))
        a6_embs = self.lookup_place(cat_c[:,:,0].to(torch.int))
        a7_embs = self.lookup_add(cat_c[:,:,1].to(torch.int))
        
        cat_p_emb = torch.mean(torch.stack([a1_embs, a2_embs, a3_embs, a4_embs, a5_embs]), axis=0)
        cat_c_emb = torch.mean(torch.stack([a6_embs, a7_embs]), axis=0)

        if not self.disable_embedding:
            x = torch.cat((cat_p_emb, cat_c_emb, cont_p_emb, cont_c_emb), dim=2)
        else:
            x = torch.cat((cat_p_emb, cat_c_emb, cont_p, cont_c), dim=2)
            
        if not self.disable_pe:
            x = x + self.positional_embedding(diff_days.int().squeeze(2))
            # import pdb;pdb.set_trace()
        if self.reduction == "none":   
            return (x, diff_days, val_len), self.positional_embedding(torch.tensor([5]).cuda())
        elif not self.disable_pe:
            return reduction_cluster(x, diff_days, val_len, self.reduction), self.positional_embedding(torch.tensor([5]).cuda())
        else:
            return reduction_cluster(x, diff_days, val_len, self.reduction)

class CEVAEEmbedding(torch.nn.Module):
    '''
        output_size : embedding output의 크기
        disable_embedding : 연속 데이터의 임베딩 유무
        disable_pe : transformer의 sequance 기준 positional encoding add 유무
        reduction : "mean" : cluster 별 평균으로 reduction
                    "date" : cluster 내 date 평균으로 reduction
    '''
    def __init__(self, args, output_size=128, disable_embedding=False, disable_pe=True, reduction="date", shift=False, use_treatment = False):
        super().__init__()
        self.shift = shift
        self.reduction = reduction
        self.disable_embedding = disable_embedding
        self.disable_pe = disable_pe
        activation = nn.ELU()
        if not disable_embedding:
            print("Embedding applied to data")
            nn_dim = emb_hidden_dim = emb_dim = output_size//4
            if args.single_treatment:
                self.cont_c_NN = nn.Sequential(nn.Linear(1 if use_treatment else 2, emb_hidden_dim),
                                    activation,
                                    nn.Linear(emb_hidden_dim, nn_dim))
            else:
                nn_dim = nn_dim * 2
                self.cont_c_NN = None
            self.cont_p_NN = nn.Sequential(nn.Linear(3 , emb_hidden_dim),
                                        activation,
                                        nn.Linear(emb_hidden_dim, nn_dim))
        else:
            emb_dim_p = 5
            emb_dim_c = 2
        self.lookup_gender  = nn.Embedding(2, emb_dim)
        self.lookup_korean  = nn.Embedding(2, emb_dim)
        self.lookup_primary  = nn.Embedding(2, emb_dim)
        self.lookup_job  = nn.Embedding(11, emb_dim)
        self.lookup_rep  = nn.Embedding(34, emb_dim)
        self.lookup_place  = nn.Embedding(19, emb_dim)
        self.lookup_add  = nn.Embedding(31, emb_dim)
        if not disable_pe:
            if shift:
                self.positional_embedding  = nn.Embedding(6, output_size)
            else:
                self.positional_embedding  = nn.Embedding(5, output_size)
            # self.positional_embedding = SinusoidalPositionalEncoding(output_size)

    def forward(self, cont_p, cont_c, cat_p, cat_c, val_len, diff_days):
        if not self.disable_embedding:
            cont_p_emb = self.cont_p_NN(cont_p)
            cont_c_emb = self.cont_c_NN(cont_c) if self.cont_c_NN != None else None
                
        a1_embs = self.lookup_gender(cat_p[:,:,0].to(torch.int))
        a2_embs = self.lookup_korean(cat_p[:,:,1].to(torch.int))
        a3_embs = self.lookup_primary(cat_p[:,:,2].to(torch.int))
        a4_embs = self.lookup_job(cat_p[:,:,3].to(torch.int))
        a5_embs = self.lookup_rep(cat_p[:,:,4].to(torch.int))
        a6_embs = self.lookup_place(cat_c[:,:,0].to(torch.int))
        a7_embs = self.lookup_add(cat_c[:,:,1].to(torch.int))
        
        cat_p_emb = torch.mean(torch.stack([a1_embs, a2_embs, a3_embs, a4_embs, a5_embs]), axis=0)
        cat_c_emb = torch.mean(torch.stack([a6_embs, a7_embs]), axis=0)

        if not self.disable_embedding:
            tensors_to_concat = [tensor for tensor in [cat_p_emb, cat_c_emb, cont_p_emb, cont_c_emb] if tensor is not None]
            x = torch.cat(tensors_to_concat, dim=2)
            # x = torch.cat((cat_p_emb, cat_c_emb, cont_p_emb, cont_c_emb), dim=2)
        else:
            x = torch.cat((cat_p_emb, cat_c_emb, cont_p, cont_c), dim=2)
            
        if not self.disable_pe:
            x = x + self.positional_embedding(diff_days.int().squeeze(2))
        # return reduction_cluster(x, diff_days, val_len, self.reduction)
        if self.reduction == "none":   
            if self.shift:
                return (x, diff_days, val_len), self.positional_embedding(torch.tensor([5]).cuda())
            else:
                return (x, diff_days, val_len), None
        else:
            return reduction_cluster(x, diff_days, val_len, self.reduction)
        
class SyntheticEmbedding(torch.nn.Module):
    '''
        output_size : embedding output의 크기
        disable_embedding : 연속 데이터의 임베딩 유무
        disable_pe : transformer의 sequance 기준 positional encoding add 유무
        reduction : "mean" : cluster 별 평균으로 reduction
                    "date" : cluster 내 date 평균으로 reduction
    '''
    def __init__(self, args, output_size=128, disable_embedding=False, disable_pe=True, reduction="date", shift=False, use_treatment = False):
        super().__init__()
        self.shift = shift
        self.reduction = reduction
        self.disable_embedding = disable_embedding
        self.disable_pe = disable_pe
        self.use_treatment = args.use_treatment
        activation = nn.ELU()
        
        print("Embedding applied to data")
        nn_dim = emb_hidden_dim = emb_dim = output_size//4
            
        self.x1_NN = nn.Sequential(nn.Linear(1 if self.use_treatment else 2, emb_hidden_dim),
                            activation,
                            nn.Linear(emb_hidden_dim, nn_dim))
        self.x2_NN = nn.Sequential(nn.Linear(1 if self.use_treatment else 2, emb_hidden_dim),
                                    activation,
                                    nn.Linear(emb_hidden_dim, nn_dim))
        self.x3_NN = nn.Sequential(nn.Linear(1, emb_hidden_dim),
                                        activation,
                                        nn.Linear(emb_hidden_dim, nn_dim))
        self.x4_NN = nn.Sequential(nn.Linear(1, emb_hidden_dim),
                                        activation,
                                        nn.Linear(emb_hidden_dim, nn_dim))
        
        if not disable_pe:
            self.positional_embedding  = nn.Embedding(6, output_size)

    def forward(self, x1, x2, x3, x4, val_len, diff_days):
        x1_emb = self.x1_NN(x1)
        x2_emb = self.x2_NN(x2) 
        x3_emb = self.x3_NN(x3) 
        x4_emb = self.x4_NN(x4) 

        tensors_to_concat = [tensor for tensor in [x1_emb, x2_emb, x3_emb, x4_emb] if tensor is not None]
        x = torch.cat(tensors_to_concat, dim=2)
            
        if not self.disable_pe:
            x = x + self.positional_embedding(diff_days.int())
            return (x, diff_days, val_len), self.positional_embedding(torch.tensor([5]).cuda())
        else:        
            return reduction_cluster(x, diff_days, val_len, self.reduction)
    

class CEVAETransformer(nn.Module):
    '''
        input_size : TableEmbedding 크기
        hidden_size : Transformer Encoder 크기
        output_size : y, d (2)
        num_layers : Transformer Encoder Layer 개수
        num_heads : Multi Head Attention Head 개수
        drop_out : DropOut 정도
        disable_embedding : 연속 데이터 embedding 여부
    '''
    def __init__(self, input_size, hidden_size, num_layers, num_heads, drop_out):
        super(CEVAETransformer, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=num_heads,
            dim_feedforward=hidden_size, 
            dropout=drop_out,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(self.transformer_layer, num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_size))

    def forward(self, x, val_len):
        cls_token = self.cls_token.expand(x.size(0), -1, -1) 
        input_with_cls = torch.cat([cls_token, x], dim=1)
        mask = (torch.arange(input_with_cls.size(1)).expand(input_with_cls.size(0), -1).cuda() < val_len.unsqueeze(1)).cuda()
        output = self.transformer_encoder(input_with_cls, src_key_padding_mask=mask.bool())  
        cls_emb = output[:, 0, :]  
        return cls_emb
    
# ------------------------ for yd only --embedding_dim 64 --shared_layers 1 --pred_layers 1 -n 300
# class CEVAE_Encoder(nn.Module):
#     def __init__(self, input_dim, latent_dim, hidden_dim=128, shared_layers=3, pred_layers=3):
#         super(CEVAE_Encoder, self).__init__()
                
#         # Shared layers
#         layers = []
#         for _ in range(shared_layers):
#             layers.append(nn.Linear(input_dim if len(layers) == 0 else hidden_dim, hidden_dim))
#             layers.append(nn.ReLU())
#         self.fc_shared = nn.Sequential(*layers)
        
#         # Latent variable z distribution parameters (mean and log variance)
#         self.fc_mu = nn.Linear(hidden_dim, latent_dim)
#         self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
#         # Predict y, d with MLP
#         yd_layers = []
#         for _ in range(pred_layers):
#             yd_layers.append(nn.Linear(hidden_dim if len(yd_layers) == 0 else hidden_dim, hidden_dim))
#             yd_layers.append(nn.ReLU())
#         yd_layers.append(nn.Linear(hidden_dim, 2))
#         self.fc_yd = nn.Sequential(*yd_layers)
        
#         # Predict t with MLP
#         t_layers = []
#         for _ in range(pred_layers):
#             t_layers.append(nn.Linear(hidden_dim if len(t_layers) == 0 else hidden_dim, hidden_dim))
#             t_layers.append(nn.ReLU())
#         t_layers.append(nn.Linear(hidden_dim, 7))
#         self.fc_t = nn.Sequential(*t_layers)
    
#     def forward(self, x):
#         h_shared = self.fc_shared(x)
        
#         mu = self.fc_mu(h_shared)
#         logvar = self.fc_logvar(h_shared)
        
#         yd_pred = self.fc_yd(h_shared)
#         t_pred = self.fc_t(h_shared)
        
#         return mu, logvar, yd_pred, t_pred
    
# class CEVAE_Encoder(nn.Module): # -- [train all, divided by t]
#     def __init__(self, input_dim, latent_dim, hidden_dim=128, shared_layers=3, pred_layers=3, t_classes=7):
#         super(CEVAE_Encoder, self).__init__()
#         # Warm up layer
#         self.warm_up = nn.Linear(input_dim, 2) # predict only y and d

#         # Shared layers
#         layers = []
#         for _ in range(shared_layers):
#             layers.append(nn.Linear(input_dim if len(layers) == 0 else hidden_dim, hidden_dim))
#             layers.append(nn.ReLU())
#         self.fc_shared = nn.Sequential(*layers)
        
#         # Predict t with MLP
#         t_layers = []
#         for _ in range(pred_layers):
#             t_layers.append(nn.Linear(hidden_dim if len(t_layers) == 0 else hidden_dim, hidden_dim))
#             t_layers.append(nn.ReLU())
#         t_layers.append(nn.Linear(hidden_dim, t_classes))
#         self.fc_t = nn.Sequential(*t_layers)
        
#         # Predict yd based on t
#         self.yd_nns = nn.ModuleList([
#             self._build_yd_predictor(hidden_dim, pred_layers) for _ in range(t_classes)
#         ])
        
#         # Calculate z (mu and logvar) based on t, yd, and x
#         self.z_nns = nn.ModuleList([
#             self._build_z_predictor(hidden_dim + 2, latent_dim, hidden_dim, pred_layers) for _ in range(t_classes)
#         ])

#     def _build_yd_predictor(self, hidden_dim, pred_layers):
#         yd_layers = []
#         for _ in range(pred_layers):
#             yd_layers.append(nn.Linear(hidden_dim, hidden_dim))
#             yd_layers.append(nn.ReLU())
#         yd_layers.append(nn.Linear(hidden_dim, 2))
#         return nn.Sequential(*yd_layers)
    
#     def _build_z_predictor(self, input_dim, latent_dim, hidden_dim, pred_layers):
#         z_layers = []
#         for _ in range(pred_layers):
#             z_layers.append(nn.Linear(input_dim if len(z_layers) == 0 else hidden_dim, hidden_dim))
#             z_layers.append(nn.ReLU())
#         z_layers.extend([nn.Linear(hidden_dim, latent_dim), nn.Linear(hidden_dim, latent_dim)])
#         return nn.ModuleDict({
#             'mu': nn.Sequential(*z_layers[:-1]),      # pred layers hidden 공유, final layer 분리
#             'logvar': nn.Sequential(*(z_layers[:-2] + [z_layers[-1]]))  
#         })

#     def forward(self, x, t_gt=None):
#         h_shared = self.fc_shared(x)
#         warm_yd = self.warm_up(x)
#         if t_gt==None:
#             t_pred = self.fc_t(h_shared)
#             t_class = t_pred.argmax(dim=1)
#         else:
#             t_class = t_gt
#             t_pred = None
#         yd_preds = [yd_nn(h_shared) for yd_nn in self.yd_nns]
#         yd_pred = torch.stack([yd_preds[i][idx] for idx, i in enumerate(t_class)], dim=0)

#         # Calculate mu and logvar based on t, yd, and x
#         h_combined = torch.cat([h_shared, yd_pred], dim=1)
#         z_preds = [z_nn['mu'](h_combined) for z_nn in self.z_nns]
#         mu = torch.stack([z_preds[i][idx] for idx, i in enumerate(t_class)], dim=0)
        
#         z_preds_logvar = [z_nn['logvar'](h_combined) for z_nn in self.z_nns]
#         logvar = torch.stack([z_preds_logvar[i][idx] for idx, i in enumerate(t_class)], dim=0)

#         return mu, logvar, yd_pred, t_pred, warm_yd


class CEVAE_Encoder(nn.Module): # -- [train all, conditioned by t]
    def __init__(self, input_dim, latent_dim, hidden_dim=128, shared_layers=3, pred_layers=3, t_pred_layers=3, t_embed_dim=8, yd_embed_dim=8, drop_out=0, t_classes=None, skip_hidden=False):
        super(CEVAE_Encoder, self).__init__()
        # Embedding for continuous t
        # Warm up layer
        self.warm_up = nn.Linear(input_dim, 2) # predict only y and d
        self.t_embedding = nn.Linear(1, t_embed_dim)
        self.yd_embedding = nn.Linear(2, yd_embed_dim)
        activation = nn.ELU()
        self.skip_hidden = skip_hidden
        # Predict t with MLP
        t_layers = []
        for _ in range(t_pred_layers):
            t_layers.append(nn.Linear(hidden_dim if len(t_layers) == 0 else hidden_dim, hidden_dim))
            t_layers.append(activation)
            t_layers.append(nn.Dropout(drop_out))
        t_layers.append(nn.Linear(hidden_dim, 1))
        self.fc_t = nn.Sequential(*t_layers)

        if not skip_hidden:
            # Shared layers
            layers = []
            for _ in range(shared_layers):
                layers.append(nn.Linear(input_dim + t_embed_dim if len(layers) == 0 else hidden_dim, hidden_dim))
                layers.append(activation)
                layers.append(nn.Dropout(drop_out))
            self.fc_shared = nn.Sequential(*layers)

        # Latent variable z distribution parameters (mean and log variance)
        self.fc_mu = nn.Linear(hidden_dim + t_embed_dim + yd_embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim + t_embed_dim + yd_embed_dim, latent_dim)
        
        # Predict y, d with MLP (now conditioned on t)
        yd_layers = []
        for _ in range(pred_layers):
            yd_layers.append(nn.Linear(hidden_dim + t_embed_dim if len(yd_layers) == 0 else hidden_dim, hidden_dim))
            yd_layers.append(activation)
            yd_layers.append(nn.Dropout(drop_out))
        yd_layers.append(nn.Linear(hidden_dim, 2))
        self.fc_yd = nn.Sequential(*yd_layers)
    
    def forward(self, x, t_gt=None):
        warm_yd = self.warm_up(x)

        # Embed the continuous t
        t_pred = self.fc_t(x) if t_gt == None else t_gt.float().unsqueeze(1)
        t_embed = self.t_embedding(t_pred)

        # Concatenate input x and embedded t
        h_shared = x if self.skip_hidden else self.fc_shared(torch.cat([x, t_embed], dim=1))

        # Predict y, d conditioned on t
        yd_pred = self.fc_yd(torch.cat([h_shared, t_embed], dim=1))
        yd_embed = self.yd_embedding(yd_pred)

        # Concatenate shared features and embedded t for mu and logvar
        h_tyd = torch.cat([h_shared, t_embed, yd_embed], dim=1)

        # Pred mu, logvar of Z
        mu = self.fc_mu(h_tyd)
        logvar = self.fc_logvar(h_tyd)
        
        return mu, logvar, yd_pred, t_pred.squeeze(), warm_yd

# class CEVAE_Decoder(nn.Module): [train seperated yd]
#     def __init__(self, latent_dim, output_dim, hidden_dim=128, num_layers=2, t_classes=7):
#         super(CEVAE_Decoder, self).__init__()
        
#         # Predict t from z
#         t_layers = []
#         for _ in range(num_layers):
#             t_layers.append(nn.Linear(latent_dim if len(t_layers) == 0 else hidden_dim, hidden_dim))
#             t_layers.append(nn.ReLU())
#         t_layers.append(nn.Linear(hidden_dim, t_classes))
#         self.fc_t = nn.Sequential(*t_layers)
        
#         # Predict y,d based on z and t
#         self.yd_nns = nn.ModuleList([
#             self._build_yd_predictor(latent_dim, hidden_dim, num_layers) for _ in range(t_classes)
#         ])
        
#         # Directly predict x from z
#         x_layers = []
#         for _ in range(num_layers):
#             x_layers.append(nn.Linear(latent_dim if len(x_layers) == 0 else hidden_dim, hidden_dim))
#             x_layers.append(nn.ReLU())
#         x_layers.append(nn.Linear(hidden_dim, output_dim))
#         self.fc_x = nn.Sequential(*x_layers)
    
#     def _build_yd_predictor(self, latent_dim, hidden_dim, num_layers):
#         yd_layers = []
#         for _ in range(num_layers):
#             yd_layers.append(nn.Linear(latent_dim if len(yd_layers) == 0 else hidden_dim, hidden_dim))
#             yd_layers.append(nn.ReLU())
#         yd_layers.append(nn.Linear(hidden_dim, 2))  # Assuming y,d output is of size 2
#         return nn.Sequential(*yd_layers)
    
#     def forward(self, z, t_gt=None):
#         # Predict t from z
#         if t_gt==None:
#             t_pred = self.fc_t(z)
#             t_class = t_pred.argmax(dim=1)
#         else:
#             t_class = t_gt
#             t_pred = None
#         yd_preds = [yd_nn(z) for yd_nn in self.yd_nns]
#         yd_pred = torch.stack([yd_preds[i][idx] for idx, i in enumerate(t_class)], dim=0)
        
#         # Directly predict x from z
#         x_pred = self.fc_x(z)
        
#         return t_pred, yd_pred, x_pred

class CEVAE_Decoder(nn.Module): #  [conditioned t, train overall yd]
    def __init__(self, latent_dim, output_dim, hidden_dim=128, t_pred_layers=2, shared_layers=2, t_embed_dim=16, drop_out=0, t_classes=7, skip_hidden=False):
        super(CEVAE_Decoder, self).__init__()
        self.skip_hidden = skip_hidden
        self.t_embedding = nn.Linear(1, t_embed_dim)
        activation = nn.ELU()
        # Predict t from z
        t_layers = []
        for _ in range(t_pred_layers):
            t_layers.append(nn.Linear(latent_dim if len(t_layers) == 0 else hidden_dim, hidden_dim))
            t_layers.append(activation)
            t_layers.append(nn.Dropout(drop_out))
        t_layers.append(nn.Linear(hidden_dim, 1))
        self.fc_t = nn.Sequential(*t_layers)
        
        if not skip_hidden:
            # Shared layers
            layers = []
            for _ in range(shared_layers):
                layers.append(nn.Linear(latent_dim + t_embed_dim if len(layers) == 0 else hidden_dim, hidden_dim))
                layers.append(activation)
                layers.append(nn.Dropout(drop_out))
            self.fc_shared = nn.Sequential(*layers)

        self.x_head = nn.Linear(latent_dim + t_embed_dim if skip_hidden else hidden_dim + t_embed_dim, output_dim)
        self.yd_head = nn.Linear(latent_dim + t_embed_dim if skip_hidden else hidden_dim + t_embed_dim, 2)
    
    def forward(self, z, t_gt=None):
        # Predict t from z
        t_pred = self.fc_t(z) if t_gt == None else t_gt.float().unsqueeze(1)
        
        t_embed = self.t_embedding(t_pred)
        h = z if self.skip_hidden else self.fc_shared(torch.cat([z, t_embed], dim=1))

        # Directly predict x from z
        x_pred = self.x_head(torch.cat([h, t_embed], dim=1))
        yd_pred = self.yd_head(torch.cat([h, t_embed], dim=1))
        
        return t_pred.squeeze(), yd_pred, x_pred

class CEVAE_det(nn.Module):
    def __init__(self, embedding_dim, latent_dim=64, encoder_hidden_dim=128, encoder_shared_layers=3, encoder_pred_layers=1, transformer_layers=3, drop_out=0.0, num_heads=2, t_pred_layers=3, t_classes=7, t_embed_dim=16, yd_embed_dim=16, skip_hidden=False):
        super(CEVAE_det, self).__init__()
        self.x_emb = CEVAEEmbedding(output_size=embedding_dim)
        self.transformer_encoder = CEVAETransformer(input_size=embedding_dim, hidden_size=latent_dim, num_layers=transformer_layers, num_heads=num_heads, drop_out=drop_out)
        self.encoder = CEVAE_Encoder(input_dim=embedding_dim, latent_dim=latent_dim, hidden_dim=encoder_hidden_dim, shared_layers=encoder_shared_layers, t_pred_layers=t_pred_layers , pred_layers=encoder_pred_layers, t_classes=t_classes, t_embed_dim=t_embed_dim, yd_embed_dim=yd_embed_dim, skip_hidden=skip_hidden)
        self.decoder = CEVAE_Decoder(latent_dim=latent_dim, output_dim=embedding_dim, hidden_dim=encoder_hidden_dim, t_pred_layers=t_pred_layers, shared_layers=encoder_shared_layers, t_classes=t_classes, t_embed_dim=t_embed_dim, skip_hidden=skip_hidden)

    def forward(self, cont_p, cont_c, cat_p, cat_c, _len, diff, t_gt=None):
        x = self.x_emb(cont_p, cont_c, cat_p, cat_c, _len, diff)
        x_transformed = self.transformer_encoder(x, _len)
        z_mu, z_logvar, enc_yd_pred, enc_t_pred, warm_yd = self.encoder(x_transformed, t_gt)
        
        # Sample z using reparametrization trick
        z = reparametrize(z_mu, z_logvar)
        
        # Decode z to get the reconstruction of x
        dec_t_pred, dec_yd_pred, x_reconstructed = self.decoder(z, t_gt)
        
        return x_transformed, x_reconstructed, z_mu, z_logvar, (enc_yd_pred, enc_t_pred), (dec_yd_pred, dec_t_pred), warm_yd
    

class CEVAE_debug(nn.Module):
    def __init__(self, embedding_dim, latent_dim=64, encoder_hidden_dim=128, encoder_shared_layers=3, encoder_pred_layers=1, transformer_layers=3, drop_out=0.0, num_heads=2, t_pred_layers=3, t_classes=7, t_embed_dim=16, yd_embed_dim=16, skip_hidden=False):
        super(CEVAE_debug, self).__init__()
        activation=nn.ELU()
        self.x_emb = CEVAEEmbedding(output_size=embedding_dim)
        self.transformer_encoder = CEVAETransformer(input_size=embedding_dim, hidden_size=latent_dim, num_layers=transformer_layers, num_heads=num_heads, drop_out=drop_out)
        self.fc = nn.Linear(embedding_dim,2)
        
        fc_layers = []
        for _ in range(3):
            fc_layers.append(nn.Linear(embedding_dim if len(fc_layers) == 0 else latent_dim, latent_dim))
            fc_layers.append(activation)
            fc_layers.append(nn.Dropout(drop_out))
        fc_layers.append(nn.Linear(latent_dim,2))
        self.fc_mlp = nn.Sequential(*fc_layers)
        
        fc_layers_d = []
        for _ in range(3):
            fc_layers_d.append(nn.Linear(embedding_dim if len(fc_layers_d) == 0 else latent_dim, latent_dim))
            fc_layers_d.append(activation)
            fc_layers_d.append(nn.Dropout(drop_out))
        fc_layers_d.append(nn.Linear(latent_dim,1))
        self.fc_mlp_d = nn.Sequential(*fc_layers_d)

    def forward(self, cont_p, cont_c, cat_p, cat_c, _len, diff, t_gt=None):
        x = self.x_emb(cont_p, cont_c, cat_p, cat_c, _len, diff)
        x_transformed = self.transformer_encoder(x, _len)
        # yd = self.fc(x_transformed)
        yd = self.fc_mlp(x_transformed)
        # d = self.fc_mlp_d(x_transformed)
        return yd
        # return torch.stack([y,d],dim=1).squeeze()
    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate=0.5):
        super(MLP, self).__init__()
        layers = []

        if num_layers == 1:
            # 단일 층인 경우
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            # 입력층
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

            # 은닉층
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

            # 출력층
            layers.append(nn.Linear(hidden_dim, output_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class customTransformerEncoder(TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, d_model, pred_layers=1, norm=None, enable_nested_tensor=True, mask_check=True, residual_t=False, residual_x = False):
        super().__init__(encoder_layer, num_layers, norm, enable_nested_tensor, mask_check)
        self.x2t1 = MLP(d_model,d_model//2, 1, num_layers=pred_layers) # Linear
        self.xt12t2 = MLP(d_model,d_model//2, 1, num_layers=pred_layers) # Linear
        self.t1_emb = MLP(1,d_model//2, d_model, num_layers=pred_layers) # Linear
        self.t2_emb = MLP(1,d_model//2, d_model, num_layers=pred_layers) # Linear
        self.xt2yd = MLP(d_model,d_model//2, 2, num_layers=pred_layers) # Linear
        self.yd_emb = MLP(2,d_model//2, d_model, num_layers=pred_layers) # Linear
        self.residual_t = residual_t
        self.residual_x = residual_x

    def forward(self, src: Tensor, mask: Tensor | None = None, src_key_padding_mask: Tensor | None = None, is_causal: bool | None = None, val_len: Tensor | None = None, intervene_t: Tensor | None = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``is_causal`` provides a hint that ``mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in Transformer class.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )
        output = src
        convert_to_nested = False
        val_idx = val_len - 1
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ''
        str_first_layer = "self.layers[0]"
        batch_first = first_layer.self_attn.batch_first
        # if not hasattr(self, "use_nested_tensor"):
        #     why_not_sparsity_fast_path = "use_nested_tensor attribute not present"
        # elif not self.use_nested_tensor:
        #     why_not_sparsity_fast_path = "self.use_nested_tensor (set in init) was not True"
        # elif first_layer.training:
        #     why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        # elif not src.dim() == 3:
        #     why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        # elif src_key_padding_mask is None:
        #     why_not_sparsity_fast_path = "src_key_padding_mask was None"
        # elif (((not hasattr(self, "mask_check")) or self.mask_check)
        #         and not torch._nested_tensor_from_mask_left_aligned(src, src_key_padding_mask.logical_not())):
        #     why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        # elif output.is_nested:
        #     why_not_sparsity_fast_path = "NestedTensor input is not supported"
        # elif mask is not None:
        #     why_not_sparsity_fast_path = "src_key_padding_mask and mask were both supplied"
        # elif torch.is_autocast_enabled():
        #     why_not_sparsity_fast_path = "autocast is enabled"

        # if not why_not_sparsity_fast_path:
        #     tensor_args = (
        #         src,
        #         first_layer.self_attn.in_proj_weight,
        #         first_layer.self_attn.in_proj_bias,
        #         first_layer.self_attn.out_proj.weight,
        #         first_layer.self_attn.out_proj.bias,
        #         first_layer.norm1.weight,
        #         first_layer.norm1.bias,
        #         first_layer.norm2.weight,
        #         first_layer.norm2.bias,
        #         first_layer.linear1.weight,
        #         first_layer.linear1.bias,
        #         first_layer.linear2.weight,
        #         first_layer.linear2.bias,
        #     )
        #     _supported_device_type = ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
        #     if torch.overrides.has_torch_function(tensor_args):
        #         why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
        #     elif src.device.type not in _supported_device_type:
        #         why_not_sparsity_fast_path = f"src device is neither one of {_supported_device_type}"
        #     elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
        #         why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
        #                                       "input/output projection weights or biases requires_grad")

        #     if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
        #         convert_to_nested = True
        #         output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not(), mask_check=False)
        #         src_key_padding_mask_for_layers = None

        seq_len = _get_seq_len(src, batch_first)
        is_causal = _detect_is_causal_mask(mask, is_causal, seq_len)

        for idx, mod in enumerate(self.layers):
            output = mod(output, src_mask=mask, is_causal=is_causal, src_key_padding_mask=src_key_padding_mask_for_layers)
            if idx == 0:
                if mask is not None:
                    output_emb = output[torch.arange(output.size(0)), val_idx] # uni dir last
                else:
                    val_mask = torch.arange(output.size(1))[None, :].cuda() < val_len[:, None]
                    output_emb = (output * val_mask.unsqueeze(-1).float()).sum(1) / val_mask.sum(1).unsqueeze(-1).float()
                    # output_emb = torch.mean(output, dim=1) # average
                t1_pred = self.x2t1(output_emb) 
                t1_pred = torch.clamp(t1_pred, 0, 1) # min-max normalized
                t1 = intervene_t[1] if intervene_t != None and intervene_t[0]=='t1' else t1_pred

                t1_emb = self.t1_emb(t1)
                t1_res = t1_emb.clone()
                x1_res = output_emb.clone()
                output = output + t1_emb.unsqueeze(1)
            elif idx == 1:
                # output = output + t_emb.unsqueeze(1)
                if mask is not None:
                    output_emb = output[torch.arange(output.size(0)), val_idx] # uni dir last
                else:
                    val_mask = torch.arange(output.size(1))[None, :].cuda() < val_len[:, None]
                    output_emb = (output * val_mask.unsqueeze(-1).float()).sum(1) / val_mask.sum(1).unsqueeze(-1).float()
                    # output_emb = torch.mean(output, dim=1) # average
                x2_res = output_emb.clone()
                if self.residual_t:
                    output_emb = output_emb + t1_res
                if self.residual_x:
                    output_emb = output_emb + x1_res
                t2_pred = self.xt12t2(output_emb)
                t2_pred = torch.clamp(t2_pred, 0, 1)  # min-max normalized
                t2 = intervene_t[1] if intervene_t != None and intervene_t[0]=='t2' else t2_pred
                t2_emb = self.t2_emb(t2)
                t_res = t1_res + t2_emb.clone() # USE T1+T2 EMB AS RESIDUAL T EMB
                output = output + t2_emb.unsqueeze(1)
            elif idx == 2:
                # output = output + t_emb.unsqueeze(1)
                if mask is not None:
                    output_emb = output[torch.arange(output.size(0)), val_idx] # uni dir last
                else:
                    val_mask = torch.arange(output.size(1))[None, :].cuda() < val_len[:, None]
                    output_emb = (output * val_mask.unsqueeze(-1).float()).sum(1) / val_mask.sum(1).unsqueeze(-1).float()
                    # output_emb = torch.mean(output, dim=1) # average
                x3_res = output_emb.clone()
                if self.residual_t:
                    output_emb = output_emb + t_res
                if self.residual_x:
                    output_emb = output_emb + x2_res
                yd = self.xt2yd(output_emb)
                yd = torch.clamp(yd, 0, 1)  # min-max normalized
                yd_emb = self.yd_emb(yd)
                output = output + yd_emb.unsqueeze(1)
            elif idx == 3:
                if self.residual_x:
                    output = output + x3_res.unsqueeze(1)

        if convert_to_nested:
            output = output.to_padded_tensor(0., src.size())

        if self.norm is not None:
            output = self.norm(output)
        
        return output, (t1, t2), yd

class CETransformer(nn.Module):
    def __init__(self, args):
        super(CETransformer, self).__init__()
        d_model=args.num_features
        nhead=args.num_heads
        d_hid=args.hidden_dim
        nlayers=args.cet_transformer_layers
        dropout=args.drop_out
        pred_layers=args.num_layers
        self.shift = args.shift
        self.unidir = args.unidir
        self.is_variational = args.variational
        
        if args.variational:
            print("variational z sampling")
        else:
            print("determinant z ")
            
        if args.unidir:
            print("unidirectional attention applied")
        else:
            print("maxpool applied")
        if not args.is_synthetic:
            self.embedding = CEVAEEmbedding(args, output_size=d_model, disable_embedding = False, disable_pe=False, reduction="none", shift= args.shift, use_treatment=args.use_treatment)
        else:
            self.embedding = SyntheticEmbedding(args, output_size=d_model, disable_embedding = False, disable_pe=False, reduction="none", shift= args.shift, use_treatment=args.use_treatment)
        # self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True, norm_first=True)
        self.transformer_encoder = customTransformerEncoder(encoder_layers, nlayers, d_model, pred_layers=pred_layers, residual_t=args.residual_t, residual_x=args.residual_x)
        # self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # Vairatioanl Z
        self.fc_mu = nn.Linear(d_model, d_model)
        self.fc_logvar = nn.Linear(d_model, d_model)

        decoder_layers = TransformerDecoderLayer(d_model, nhead, d_hid, dropout, batch_first=True, norm_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        self.max_pool = nn.MaxPool1d(kernel_size=124, stride=1)

        self.d_model = d_model
        
        self.z2t = MLP(d_model, d_model//2, 1, num_layers=pred_layers)
        self.t1_emb = MLP(1, d_model//2, d_model, num_layers=pred_layers)
        self.t2_emb = MLP(1, d_model//2, d_model, num_layers=pred_layers)
        self.zt12t2 = MLP(d_model, d_model//2, 1, num_layers=pred_layers)
        self.zt2yd = MLP(d_model, d_model//2, 2, num_layers=pred_layers)

        self.linear_decoder = MLP(d_model, d_model, d_model, num_layers=1) # Linear
        # self.init_weights(1)

    # def generate_square_subsequent_mask(self, sz):
    #     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    #     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #     return mask
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.masked_fill(mask == 0, True).masked_fill(mask == 1, False)
        return mask

    def init_weights(self, c):
        initrange = 0.1
        # For embedding layers
        if hasattr(self.embedding, 'weight'):
            self.embedding.weight.data.uniform_(-initrange, initrange)
        
        # For transformer encoder and decoder
        for module in [self.transformer_encoder, self.transformer_decoder]:
            for param in module.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
        
        # For MLP layers
        for mlp in [self.z2t, self.t_emb, self.zt2yd]:
            for layer in mlp.layers:
                if isinstance(layer, nn.Linear):
                    layer.weight.data.uniform_(-initrange, initrange)
                    if layer.bias is not None:
                        layer.bias.data.zero_()
        
        # For variational layers
        # for fc in [self.fc_mu]:
        #     for layer in fc.layers:
        #         if isinstance(layer, nn.Linear):
        #             layer.weight.data.zero_() 
        #             if layer.bias is not None:
        #                 layer.bias.data.zero_()

        # # For variational layers
        # for fc in [self.fc_logvar]:
        #     for layer in fc.layers:
        #         if isinstance(layer, nn.Linear):
        #             layer.weight.data = (torch.log(c) * torch.ones_like(layer.weight.data)).requires_grad_(True)
        #             if layer.bias is not None:
        #                 layer.bias.data.zero_()

    def forward(self, cont_p, cont_c, cat_p, cat_c, val_len, diff_days, is_MAP=False):
        # Encoder
        if self.embedding.reduction != "none":
            x = self.embedding(cont_p, cont_c, cat_p, cat_c, val_len, diff_days)
        else:
            (x, diff_days, _), start_tok = self.embedding(cont_p, cont_c, cat_p, cat_c, val_len, diff_days) # embedded:(32, 124, 128)
        index_tensor = torch.arange(x.size(1), device=x.device)[None, :, None]
        # x = self.embedding(cont_p, cont_c, cat_p, cat_c, val_len, diff_days) #* math.sqrt(self.d_model)
        # src_mask = (torch.arange(x.size(1)).expand(x.size(0), -1).cuda() < val_len.unsqueeze(1)).cuda()
        src_key_padding_mask = ~(torch.arange(x.size(1)).expand(x.size(0), -1).cuda() < val_len.unsqueeze(1)).cuda()
        src_mask = self.generate_square_subsequent_mask(x.size(1)).cuda() if self.unidir else None
        
        # Z ------
        # CETransformer encoder
        z, (enc_t1, enc_t2), enc_yd = self.transformer_encoder(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask, val_len=val_len)
        if self.unidir:
            idx = val_len - 1
            z = z[torch.arange(z.size(0)), idx] # padding 이 아닌값에 해당하는 seq중 마지막 값 사용
        else:
            val_mask = torch.arange(z.size(1))[None, :].cuda() < val_len[:, None]
            valid_z = z * val_mask[:, :, None].float().cuda()
            z = valid_z.max(dim=1)[0] # padding 이 아닌값에 해당하는 seq 들 max pool
        
        # z_mu, z_logvar = self.fc_mu(z), self.fc_logvar(z)
        z_mu, z_logvar = z, self.fc_logvar(z)
            
        if is_MAP:
            z=z_mu
        elif self.is_variational:
            z = reparametrize(z_mu, z_logvar)
        else:
            z_logvar = torch.full_like(z_mu, -100.0).cuda()
            z = reparametrize(z_mu, z_logvar)
        # Baseline Transformer encoder
        # z = self.transformer_encoder(x, src_key_padding_mask=src_mask)
        
        dec_t1 = self.z2t(z.squeeze())
        t1_emb = self.t1_emb(dec_t1)
        dec_t2 = self.zt12t2(z.squeeze()+t1_emb)
        t2_emb = self.t2_emb(dec_t2)
        
        # Linear Decoder
        dec_yd = self.zt2yd(z.squeeze() + t1_emb + t2_emb)
        
        pos_embeddings = self.embedding.positional_embedding(diff_days.squeeze().long())

        # 입력 z에 위치 임베딩 추가
        z_expanded = z.unsqueeze(1) + pos_embeddings  # [batch_size, 124, hidden_dim]
        z_expanded = torch.where(index_tensor < val_len[:, None, None], z_expanded, torch.zeros_like(z_expanded))
        
        # linear_decoder 적용
        z_flat = z_expanded.view(-1, z.shape[-1])  # [batch_size * 5, hidden_dim]
        x_recon_flat = self.linear_decoder(z_flat)  # [batch_size * 5, hidden_dim]

        # 최종 x_recon을 원하는 형태로 재구성
        x_recon = x_recon_flat.view(z_expanded.shape)  # [batch_size, 5, hidden_dim]
        
        # Reconstruction w/ Transformer Decoder
        # if self.shift:
        #     start_tok = start_tok.repeat(x.size(0), 1).unsqueeze(1)
        #     x_in = torch.cat([start_tok, x], dim=1)
        #     tgt_key_padding_mask = ~(torch.arange(x.size(1)).expand(x.size(0), -1).cuda() < (val_len + 1).unsqueeze(1)).cuda()
        # else :
        #     x_in = x
        #     tgt_key_padding_mask = ~(torch.arange(x.size(1)).expand(x.size(0), -1).cuda() < val_len.unsqueeze(1)).cuda()
        # x_recon = self.transformer_decoder(tgt = x_in, memory = z, tgt_mask = tgt_mask, memory_mask = None, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=src_mask)
        # return x, x_recon, (enc_yd, enc_t), (dec_yd, dec_t)
        
        # Reconstruction w/ Linear Decoder
        
        x = torch.where(index_tensor < val_len[:, None, None], x, torch.zeros_like(x))
        x_recon = torch.where(index_tensor < val_len[:, None, None], x_recon, torch.zeros_like(x_recon))

        # x = torch.where(torch.arange(x.size(1), device=x.device)[None, :] < val_len[:, None], x, torch.zeros_like(x))
        # x_recon = torch.where(torch.arange(x_recon.size(1), device=x_recon.device)[None, :] < val_len[:, None], x_recon, torch.zeros_like(x))
        
        return x, x_recon, (enc_yd, torch.cat([enc_t1, enc_t2], dim=1)), (dec_yd, torch.cat([dec_t1, dec_t2], dim=1)), (z_mu, z_logvar)
    
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.5, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0)]
#         return self.dropout(x)