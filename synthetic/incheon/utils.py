import torch
import torch.nn as nn
import torch.nn.functional as F

from fancyimpute import IterativeImputer
from mdlp.discretization import MDLP
from sklearn.metrics import mean_absolute_error, mean_squared_error

import numpy as np
import pandas as pd

import pdb
import pickle
import random

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


## Data---------------------------------------------------------------------------------------
def restore_min_max(scaled_dat, min = 0, max = 37):
    '''
    Restore Min-Max Scaled Data
    '''
    dat = (max - min) * scaled_dat + min
    return dat



def full_load_data(data_path = './data/20221122_raw_data_utf-8.csv',
                num_features = 17,
                target_order = 3,
                train_ratio = 0.2,
                classification = True,
                device = 'cpu',
                model_name = 'MagNet',
                args = None):
    '''
    Description:
        Load Data from bottom
    Input :
        - data_path : Path to load data
        - num_features : Number of features on one node
        - target_order : The order which we want to predict
        - train_ratio : Ratio of train data
        - device : Running device (cuda / cpu)
    Output :
        - final_data_list : the data we use to learning
        - y_min : minimum value of y label 
        - y_max : maximum value of y label
    '''
    df_dat, unique_group = load_data(data_path = data_path, classification = classification, args=args)

    # aggregate group (for each graph)
    group_df = df_dat.groupby("transmission_route").agg(list)

    data_x_list, data_edge_list, data_y_list, data_order_list, num_graph = split_graphs(group_df = group_df, unique_group = unique_group, num_features = num_features, model_name = model_name, args=args)

    new_data_x_list, new_data_edge_list, new_data_y_list, new_data_order_list, new_data_tr_mask_list, new_data_val_mask_list, new_data_te_mask_list = masking(data_x_list = data_x_list,
                                                                                                                            data_edge_list = data_edge_list,
                                                                                                                            data_y_list = data_y_list,
                                                                                                                            data_order_list = data_order_list,
                                                                                                                            target_order = target_order,
                                                                                                                            train_ratio = train_ratio)

    final_data_x_list, final_data_edge_list, final_data_y_list, final_data_order_list, final_data_tr_mask_list, final_data_val_mask_list, final_data_te_mask_list = del_zero_edge(new_data_x_list = new_data_x_list,
                                                                                                                            new_data_edge_list = new_data_edge_list,
                                                                                                                            new_data_y_list = new_data_y_list,
                                                                                                                            new_data_order_list = new_data_order_list,
                                                                                                                            new_data_tr_mask_list = new_data_tr_mask_list,
                                                                                                                            new_data_val_mask_list = new_data_val_mask_list,
                                                                                                                            new_data_te_mask_list = new_data_te_mask_list,
                                                                                                                            num_graph = num_graph)

    final_data_x_list, num_birthyear_disc = birthyear_disc(final_data_x_list, final_data_y_list, final_data_tr_mask_list, final_data_val_mask_list, args)
    final_data_y_list, y_min, y_max = scaling_y(final_data_y_list, final_data_tr_mask_list, final_data_val_mask_list)
    
    check_graphs(final_data_x_list = final_data_x_list,
                final_data_edge_list = final_data_edge_list,
                final_data_y_list = final_data_y_list,
                final_data_order_list = final_data_order_list, 
                final_data_tr_mask_list = final_data_tr_mask_list,
                final_data_val_mask_list = final_data_val_mask_list,
                final_data_te_mask_list = final_data_te_mask_list)

    final_data_list = to_tensor(final_data_x_list = final_data_x_list,
                            final_data_edge_list = final_data_edge_list,
                            final_data_y_list = final_data_y_list,
                            final_data_order_list = final_data_order_list,
                            final_data_tr_mask_list = final_data_tr_mask_list,
                            final_data_val_mask_list = final_data_val_mask_list,
                            final_data_te_mask_list = final_data_te_mask_list,
                            device = device)

    return final_data_list, y_min, y_max, num_birthyear_disc



def load_data(data_path='./data/20221122_raw_data_utf-8.csv', classification = True, args=None):
    # load data
    df = pd.read_csv(data_path, index_col=0)       # [8844, 37]

    # Making y label
    cdc_count = df['index_id_cdc'].value_counts()
    cdc_count = cdc_count.to_frame().reset_index()
    cdc_count.columns = ['id_inch', 'index_id_cdc_num']
    if classification:
        over = cdc_count['index_id_cdc_num']>=6
        cdc_count['index_id_cdc_num'][over] = 6

    df = pd.merge(df, cdc_count, how='left', on='id_inch')      # left outer join
    df['index_id_cdc_num'] = df['index_id_cdc_num'].fillna(0)   # 결측값 0으로 대치

    if args.replace_missing_value == 'replace':
        for i in range(1, 16):
            col_name = f'a{i}'
            df.loc[:, [col_name]] = df.loc[:, [col_name]].fillna(2)
    elif args.replace_missing_value == 'binomial':
        pass
    elif args.replace_missing_value == 'mice':
        def custom_transform(value):
            if value < 0:
                return 0
            elif value <= 0.5:
                return 0
            else:
                return 1
        imputer = IterativeImputer(random_state=args.seed)
        selected_columns = [f'a{i}' for i in range(1, 16)]
        selected_df = df[selected_columns]
        completed_selected_df = imputer.fit_transform(selected_df)
        df[selected_columns] = completed_selected_df
        for col in selected_columns:
            df[col] = df[col].apply(custom_transform)
            
    else:   
        df.dropna(how='any', subset=['contact_n'], inplace=True)    # 차수가 없는것 삭제
        df = df.drop(df[df.contact_n == '?'].index)

        df.dropna(how='any', subset=['sex'], inplace=True)          # 성별 없는것 삭제

        df.dropna(how='any', subset=['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10','a11','a12','a13','a14','a15'], inplace=True)   # 증상 (a1-15) 없는것 삭제

    # index reset
    df = df.reset_index(drop=True)                        # [1154, 38]
    unique_group = pd.unique(df['transmission_route'])    # 155
    return df, unique_group



def split_graphs(group_df, unique_group, num_features, model_name, args):
    num_graph = len(unique_group)                       # number of graph (155)
    data_x_list = []                                    # node별 feature list (17)
    data_y_list = []                                    # node별 y (바이러스를 전파시킬 인원수)
    data_edge_list = []                                 # edge list
    data_direct_list = []                               # 선행확진자 (0, 1)
    data_order_list = []                                # 차수 # 1,2,3,4

    for itr in range(num_graph):
        temp_data = group_df.iloc[itr]
        num_person = len(temp_data['id_inch'])
        # Birth: 0 if birth<1982 1 else
        # Gender: 0 if male 1 else 

        ##### Edge ----------------------------------------------------------------------------|
        ## Assign indices per patient on each graph
        cnt = 0
        p2i_dict = {}
        for person_itr in range(num_person):
            p2i_dict[temp_data['id_inch'][person_itr]] = cnt
            cnt += 1

        ## Assign edge on each graph
        edge1_list = []
        edge2_list = []
        for person_itr in range(num_person):
            pre_patient = temp_data['index_id_cdc'][person_itr]   # pre_patient : 지금 환자에게 전파한 환자 (선행확진자)

            # graph 내에 선행확진자(pre_patient)가 존재하면 edge 설정
            if pre_patient in p2i_dict:
                post_patient = temp_data['id_inch'][person_itr]

                ## edge1 -> edge2 로 나타나는 edge (i.e. edge1_list[0] -> edge2_list[0])
                edge1_list.append(p2i_dict[pre_patient])
                edge2_list.append(p2i_dict[post_patient])
                if model_name not in ['MagNet']:
                    edge1_list.append(p2i_dict[post_patient])
                    edge2_list.append(p2i_dict[pre_patient])
        data_edge = [edge1_list, edge2_list]
        data_edge_list.append(data_edge)
        #-----------------------------------------------------------------------------------|

        ##### order ------------------------------------------------------------------------|
        ## Assign order of nodes on each graph
        data_order = np.zeros((num_person,))
        for person_itr in range(num_person):
            order = int(temp_data['contact_n'][person_itr][0])
            data_order[person_itr] = order
        data_order_list.append(data_order)
        #-----------------------------------------------------------------------------------|

        ##### y ----------------------------------------------------------------------------|
        ## Assign y label of nodes on each graph
        data_y = np.zeros((num_person,))
        for person_itr in range(num_person):
            y = temp_data['index_id_cdc_num'][person_itr]
            data_y[person_itr] = y
        data_y_list.append(data_y)
        #-----------------------------------------------------------------------------------|

        ##### NODE -------------------------------------------------------------------------|
        ## Assign x features of nodes on each graph
        data_x = np.zeros((num_person, num_features))
        for person_itr in range(num_person):
            # birthyear
            data_x[person_itr, 0] = temp_data['birthyear'][person_itr]
            # Gender
            if temp_data['sex'][person_itr] == '남':
                data_x[person_itr, 1] = 0
            else:
                data_x[person_itr, 1] = 1
            for i, a_itr in enumerate(range(2, num_features)):
                data_x[person_itr,a_itr] = temp_data['a'+str(i + 1)][person_itr]

        data_x_list.append(data_x)
        #-----------------------------------------------------------------------------------|

    assert len(data_x_list) == len(data_edge_list) == len(data_y_list) == len(data_order_list)
    print(f"Exist {len(data_x_list)} graphs after first step....(Split graphs)")
    return data_x_list, data_edge_list, data_y_list, data_order_list, num_graph



def birthyear_disc(final_data_x_list, final_data_y_list, final_data_tr_mask_list, final_data_val_mask_list, args):
    if args.birthyear_rule == 'mdlp':
        # train val masking
        total_mask_list = [tr_mask + val_mask for tr_mask, val_mask in zip(final_data_tr_mask_list, final_data_val_mask_list)]
        yss = [np.array(final_data_y)[total_mask.astype(bool)] for final_data_y, total_mask in zip(final_data_y_list, total_mask_list)]
        Xs = [np.array(final_data_x)[total_mask.astype(bool)] for final_data_x, total_mask in zip(final_data_x_list, total_mask_list)]
        
        labels = []
        birthyears = []
        for ys in yss:
            for y in ys:
                labels.append(y)
        for X in Xs:
            for data_x in X:
                birthyear = data_x[0]
                birthyears.append(int(birthyear))

        # fitting to trian/val set
        transformer = MDLP()
        transformer.fit(np.array(birthyears).reshape(-1, 1), labels)
        
        # transform all data
        birthyears = []
        for X in final_data_x_list:
            for data_x in X:
                birthyear = data_x[0]
                birthyears.append(int(birthyear))   
        # transform / birthyear to one-hot
        X_disc = transformer.transform(np.array(birthyears).reshape(-1, 1))
        num_disc = int(max(X_disc))
        birthyear_onehot = np.eye(num_disc + 1)[X_disc.squeeze()]
        
        # remove first col and concat birthyear one-hot
        processed_final_data_x_list = list()
        cnt = 0
        for data_x in final_data_x_list:
            new_data_x = list()
            for x in data_x:
                new_array = birthyear_onehot[cnt,:]
                processed_x = np.concatenate((new_array, x[1:]))
                new_data_x.append(processed_x)
                cnt += 1
            processed_final_data_x_list.append(new_data_x)
        num_birthyear_disc = num_disc + 1
    else:
        def heuristic_birthyear(birthyears):
            new_birthyears = list()
            for birthyear in birthyears:
                if birthyear >= 1990:
                    value = 0
                
                elif 1960 <= birthyear < 1990:
                    value = 1

                elif 1930 <= birthyear < 1960:
                    value = 2

                elif birthyear < 1930:
                    value = 3

                new_birthyears.append(value)
            return new_birthyears 
        processed_final_data_x_list = list()
        birthyears = []
        for X in final_data_x_list:
            for data_x in X:
                birthyear = data_x[0]
                birthyears.append(int(birthyear)) 

        birthyears = np.array(heuristic_birthyear(birthyears))
        birthyear_onehot = np.eye(4)[birthyears.squeeze()] 
        processed_final_data_x_list = list()
        cnt = 0
        for data_x in final_data_x_list:
            new_data_x = list()
            for x in data_x:
                new_array = birthyear_onehot[cnt,:]
                processed_x = np.concatenate((new_array, x[1:]))
                new_data_x.append(processed_x)
                cnt += 1
            processed_final_data_x_list.append(new_data_x)
        num_birthyear_disc = 4
    return processed_final_data_x_list, num_birthyear_disc



def min_max_scale(data_y_list, y_min, y_max, per_data=False):
    final_data_y_list = list()
    for data_y in data_y_list:
        if not per_data:
            y_in_graph = list()
            for y in data_y:
                scaled_y = (y - y_min) / (y_max - y_min)
                y_in_graph.append(scaled_y)
            final_data_y_list.append(y_in_graph)
        else:
            scaled_y = (data_y - y_min) / (y_max - y_min)
            final_data_y_list.append(scaled_y)
    return final_data_y_list



def scaling_y(final_data_y_list, final_data_tr_mask_list, final_data_val_mask_list):
    total_mask_list = [tr_mask + val_mask for tr_mask, val_mask in zip(final_data_tr_mask_list, final_data_val_mask_list)]
    ys = [np.array(final_data_y) * total_mask.astype(np.int64) for final_data_y, total_mask in zip(final_data_y_list, total_mask_list)]
    y_min = min([min(y) for y in ys])
    y_max = max([max(y) for y in ys])
    final_data_y_list = min_max_scale(final_data_y_list, y_min, y_max)
    return final_data_y_list, y_min, y_max



def masking(data_x_list, data_edge_list, data_y_list, data_order_list, target_order, train_ratio=0.2):
    te_val_ratio = 1 - train_ratio

    new_data_x_list = []
    new_data_edge_list = []
    new_data_y_list = []
    new_data_order_list = []
    new_data_tr_mask_list = []
    new_data_val_mask_list = []
    new_data_te_mask_list = []

    for data_x, data_edge, data_y, data_order in zip(data_x_list, data_edge_list, data_y_list, data_order_list):
        mask_node_idx_list = np.argwhere(np.asarray(data_order) > target_order) # one-hop 보장
        
        # data node / label / order remove
        if not len(mask_node_idx_list) == 0:
            data_x = list(np.delete(list(data_x), mask_node_idx_list, axis=0))
            data_y = list(np.delete(data_y, mask_node_idx_list, axis=0))
            data_order = list(np.delete(data_order, mask_node_idx_list, axis=0))

        # edge remove
        edge_node_idx = []
        for mask_node_idx in mask_node_idx_list:
            match_value1 = np.argwhere(np.asarray(data_edge[0]) == mask_node_idx)
            match_value2 = np.argwhere(np.asarray(data_edge[1]) == mask_node_idx)
            edge_node_idx.extend(match_value1)
            edge_node_idx.extend(match_value2)
        edge_node_idx = np.unique(edge_node_idx)
        if not len(edge_node_idx) == 0:
            data_edge = [list(np.delete(data_edge[0], edge_node_idx)), list(np.delete(data_edge[1], edge_node_idx))]

        unique_node = list(set(data_edge[0] + data_edge[1]))
        unique_node = sorted(unique_node)
        old_new_dict = {}
        cnt = 0
        for node in unique_node:
            old_new_dict[node] = cnt
            cnt += 1
        
        l1 = []
        l2 = []
        for l1_ele, l2_ele in zip(data_edge[0], data_edge[1]):
            l1.append(old_new_dict[l1_ele])
            l2.append(old_new_dict[l2_ele])
        data_edge = [l1,l2]

        tr_data_mask = np.ones((len(data_x),)) # train mask 
        val_data_mask = np.zeros((len(data_x),)) # validation mask
        te_data_mask = np.zeros((len(data_x),)) # test mask
        mask_node_idx_list = np.argwhere(np.asarray(data_order) == target_order)

        if len(mask_node_idx_list) == 1:
            mask_node_idx_list = [int(mask_node_idx_list[0])]
        else:
            mask_node_idx_list = mask_node_idx_list.squeeze()


        if not len(mask_node_idx_list) == 0:
            # train에 사용하는 mask
            tr_mask_node_idx_list = np.random.choice(np.asarray(mask_node_idx_list),
                                            size=int(len(mask_node_idx_list)*te_val_ratio),
                                            replace=False)
            tr_data_mask[tr_mask_node_idx_list] = 0
            
            # train 하고 남은 것 중, 절반은 val, 절반은 test
            val_mask_node_idx_list = np.random.choice(np.asarray(tr_mask_node_idx_list),
                                            size=int(len(tr_mask_node_idx_list)*0.5),
                                            replace=False)
            if len(val_mask_node_idx_list) > 0:
                val_data_mask[val_mask_node_idx_list] = 1
            te_mask_node_idx_list = np.asarray(list(set(tr_mask_node_idx_list) - set(val_mask_node_idx_list)))
            if len(te_mask_node_idx_list) > 0:
                te_data_mask[te_mask_node_idx_list] = 1

            assert len(data_x) == np.sum(tr_data_mask) + np.sum(val_data_mask) + np.sum(te_data_mask)

        new_data_x_list.append(data_x)
        new_data_edge_list.append(data_edge)
        new_data_y_list.append(data_y)
        new_data_order_list.append(data_order)
        new_data_tr_mask_list.append(tr_data_mask)
        new_data_val_mask_list.append(val_data_mask)
        new_data_te_mask_list.append(te_data_mask)

    assert len(new_data_x_list) == len(new_data_edge_list) == len(new_data_y_list) == len(new_data_order_list) == len(new_data_tr_mask_list)
    print(f"Exist {len(new_data_x_list)} graphs after second step....(Masking)")

    return new_data_x_list, new_data_edge_list, new_data_y_list, new_data_order_list, new_data_tr_mask_list, new_data_val_mask_list, new_data_te_mask_list



def del_zero_edge(new_data_x_list, new_data_edge_list, new_data_y_list, new_data_order_list,
                new_data_tr_mask_list, new_data_val_mask_list, new_data_te_mask_list,
                num_graph):
    final_data_x_list = []
    final_data_edge_list = []
    final_data_y_list = []
    final_data_order_list = []
    final_data_tr_mask_list = []
    final_data_val_mask_list = []
    final_data_te_mask_list = []
    for itr in range(num_graph):
        if len(new_data_edge_list[itr][0]) >= 1:                            # direct : 1 / undriect : 2
            final_data_x_list.append(new_data_x_list[itr])
            final_data_edge_list.append(new_data_edge_list[itr])
            final_data_y_list.append(new_data_y_list[itr])
            final_data_order_list.append(new_data_order_list[itr])
            final_data_tr_mask_list.append(new_data_tr_mask_list[itr])
            final_data_val_mask_list.append(new_data_val_mask_list[itr])
            final_data_te_mask_list.append(new_data_te_mask_list[itr])

    
    assert len(final_data_x_list)==len(final_data_edge_list)==len(final_data_y_list)==len(final_data_order_list)==len(final_data_tr_mask_list)
    print(f"Exist {len(final_data_x_list)} graphs after third step....(Deleting zero edge graphs)")
    return final_data_x_list, final_data_edge_list, final_data_y_list, final_data_order_list, final_data_tr_mask_list, final_data_val_mask_list, final_data_te_mask_list



def check_graphs(final_data_x_list, final_data_edge_list, final_data_y_list, final_data_order_list,
                final_data_tr_mask_list, final_data_val_mask_list, final_data_te_mask_list):
    for itr in range(len(final_data_x_list)):
        num_node = len(final_data_x_list[itr])
        num_uniqe_node_edge = len(set(final_data_edge_list[itr][0]))
        num_y = len(final_data_y_list[itr])
        num_order = len(final_data_order_list[itr])
        num_tr_mask = len(final_data_tr_mask_list[itr])
        num_val_mask = len(final_data_val_mask_list[itr])
        num_te_mask = len(final_data_te_mask_list[itr])

        assert num_node == num_order == num_tr_mask == num_val_mask == num_te_mask == num_y
        assert num_uniqe_node_edge >= 1



def to_tensor(final_data_x_list, final_data_edge_list, final_data_y_list, final_data_order_list,
                        final_data_tr_mask_list, final_data_val_mask_list, final_data_te_mask_list,
                        device):
    final_data_list = []
    for data in zip(final_data_x_list, final_data_edge_list, final_data_y_list, 
                    final_data_tr_mask_list, final_data_val_mask_list, final_data_te_mask_list,
                    final_data_order_list):
        data_x = torch.FloatTensor(data[0]).to(device)
        data_edge = torch.LongTensor(data[1]).to(device)
        data_y = torch.FloatTensor(data[2]).squeeze().to(device)
        # data_y = torch.FloatTensor(data[2]).to(device)

        data_tr_mask = torch.LongTensor(data[3]).to(device)
        data_tr_mask = data_tr_mask.type(torch.bool)

        data_val_mask = torch.LongTensor(data[4]).to(device)
        data_val_mask = data_val_mask.type(torch.bool)

        data_te_mask = torch.LongTensor(data[5]).to(device)
        data_te_mask = data_te_mask.type(torch.bool)

        data_order = data[-1]
        final_data_list.append([data_x, data_edge, data_y, 
                        data_tr_mask, data_val_mask, data_te_mask, data_order])
    
    print(f"Finally exist {len(final_data_list)} graphs after all steps!!")

    return final_data_list
# ---------------------------------------------------------------------------------------------



## Loss ----------------------------------------------------------------------------------------
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()
        self.mse = nn.MSELoss()
        self.eps = 1e-12

    def forward(self, target, pred):
        return torch.sqrt(self.mse(target, pred) + self.eps)
# ---------------------------------------------------------------------------------------------




## Train --------------------------------------------------------------------------------------
def train(data, model, model_type, optimizer, criterion, y_min=0, y_max=37, lamb=1.0):
    model.train()
    data_x, data_edge, data_y, data_tr_mask, _, _, _ = data
    num_data = torch.sum(data_tr_mask)

    optimizer.zero_grad()

    if model_type in ['ridge', 'non_graph']:
        out = model(data_x).squeeze()
    elif model_type == 'graph':
        out = model(x=data_x, edge_index=data_edge).squeeze()

    loss = criterion(out[data_tr_mask], data_y[data_tr_mask])
    if model_type == 'ridge':
        loss += lamb*torch.norm(model.linear1.weight, p=2)

    num_missing_value = torch.sum(data_x[data_tr_mask] == 2, axis=1).tolist()
    if not torch.isnan(loss):
      loss.backward()
      optimizer.step()
      return loss.item(), num_data, torch.clamp(out[data_tr_mask], min=y_min, max=y_max), data_y[data_tr_mask], num_missing_value
    else:
      return 0, num_data, torch.clamp(out[data_tr_mask], min=y_min, max=y_max), data_y[data_tr_mask], num_missing_value


## Validation --------------------------------------------------------------------------------
@torch.no_grad()
def valid(data, model, model_type, eval_criterion, y_min=0, y_max=37, lamb=1.0):
    model.eval()
    data_x, data_edge, data_y, data_tr_mask, data_val_mask, data_te_mask, _ = data
    num_data = torch.sum(data_val_mask)

    if model_type in ['ridge', 'non_graph']:
        out = model(data_x).squeeze()
    elif model_type == 'graph':
        out = model(data_x, data_edge).squeeze()

    out, data_y = restore_min_max(out, y_min, y_max), restore_min_max(data_y, y_min, y_max)
    loss = eval_criterion(out[data_val_mask], data_y[data_val_mask])
    if model_type == 'ridge':
        loss += lamb*torch.norm(model.linear1.weight, p=2)

    # 정성분석
    num_missing_value = torch.sum(data_x[data_val_mask] == 2, axis=1).tolist()
    num_tr = sum(data_tr_mask).item()
    num_val = sum(data_val_mask).item()
    num_te = sum(data_te_mask).item()
    # if not torch.isnan(mae_loss):
    #     return mae_loss.item(), rmse_loss.item(), num_data, torch.clamp(out[data_te_mask], min=y_min, max=y_max), data_y[data_te_mask], num_missing_value, num_tr, num_val, num_te
    # else:
    #     return 0, 0, num_data, torch.clamp(out[data_te_mask], min=y_min, max=y_max), data_y[data_te_mask], num_missing_value, num_tr, num_val, num_te

    if not torch.isnan(loss):
        return loss.item(), num_data, torch.clamp(out[data_val_mask], min=y_min, max=y_max), data_y[data_val_mask], num_missing_value
    else:
        return 0, num_data, torch.clamp(out[data_val_mask], min=y_min, max=y_max), data_y[data_val_mask], num_missing_value
    

def cal_group_loss(ground_truth_list, predicted_list, per_graph=False, num_sample=50):
    mae_group_loss_dict = {} ; rmse_group_loss_dict = {}
    mae_mean_group_loss = 0 ; rmse_mean_group_loss = 0
    if per_graph:
        mae_mean_group_loss = np.zeros(num_sample) ; rmse_mean_group_loss = np.zeros(num_sample)
        for i in np.unique(ground_truth_list):
            mask = (ground_truth_list == i)
            mae_group_loss = mean_absolute_error(ground_truth_list[mask].reshape((-1, num_sample)), predicted_list[mask].reshape((-1, num_sample)), multioutput='raw_values')
            rmse_group_loss = mean_squared_error(ground_truth_list[mask].reshape((-1, num_sample)), predicted_list[mask].reshape((-1, num_sample)), multioutput='raw_values')
            rmse_group_loss = np.sqrt(rmse_group_loss)

            mae_group_loss_dict[i] = mae_group_loss
            rmse_group_loss_dict[i] = rmse_group_loss

            mae_mean_group_loss += mae_group_loss
            rmse_mean_group_loss += rmse_group_loss

        mae_mean_group_loss /= len(mae_group_loss_dict)
        rmse_mean_group_loss /= len(rmse_group_loss_dict)
        min_mae_idx = np.argmin(mae_mean_group_loss)
        min_mae = mae_mean_group_loss[min_mae_idx]
        min_rmse = rmse_mean_group_loss[min_mae_idx]

        mae_group_loss_dict = {k: v[min_mae_idx] for k, v in mae_group_loss_dict.items()}
        rmse_group_loss_dict = {k: v[min_mae_idx] for k, v in rmse_group_loss_dict.items()}
        
        return mae_group_loss_dict, min_mae, rmse_group_loss_dict, min_rmse, min_mae_idx

    for i in np.unique(ground_truth_list):
        mask = (ground_truth_list == i)

        
        mae_group_loss = mean_absolute_error(ground_truth_list[mask], predicted_list[mask])
    
        rmse_group_loss = mean_squared_error(ground_truth_list[mask], predicted_list[mask])
        rmse_group_loss = np.sqrt(rmse_group_loss)

        mae_mean_group_loss += mae_group_loss
        rmse_mean_group_loss += rmse_group_loss
        
        mae_group_loss_dict[i] = mae_group_loss
        rmse_group_loss_dict[i] = rmse_group_loss

    mae_mean_group_loss /= len(mae_group_loss_dict)
    rmse_mean_group_loss /= len(rmse_group_loss_dict)

    return mae_group_loss_dict, mae_mean_group_loss, rmse_group_loss_dict, rmse_mean_group_loss


## Test ----------------------------------------------------------------------------------------
@torch.no_grad()
def test(data, model, model_type, mae, rmse, y_min=0, y_max=6, lamb=1.0):
    model.eval()
    data_x, data_edge, data_y, data_tr_mask, data_val_mask, data_te_mask, _ = data
    num_data = torch.sum(data_te_mask)

    if model_type in ['ridge', 'non_graph']:
        out = model(data_x).squeeze()
    elif model_type == 'graph':
        out = model(data_x, data_edge).squeeze()
    out, data_y = restore_min_max(out, y_min, y_max), restore_min_max(data_y, y_min, y_max)
    mae_loss = mae(out[data_te_mask], data_y[data_te_mask])
    rmse_loss = rmse(out[data_te_mask], data_y[data_te_mask])
    if model_type == 'ridge':
        mae_loss += lamb*torch.norm(model.linear1.weight, p=2)
        rmse_loss += lamb*torch.norm(model.linear1.weight, p=2)

    # 정성분석
    num_missing_value = torch.sum(data_x[data_te_mask] == 2, axis=1).tolist()
    num_tr = sum(data_tr_mask).item()
    num_val = sum(data_val_mask).item()
    num_te = sum(data_te_mask).item()
    if not torch.isnan(mae_loss):
        return mae_loss.item(), rmse_loss.item(), num_data, torch.clamp(out[data_te_mask], min=y_min, max=y_max), data_y[data_te_mask], num_missing_value, num_tr, num_val, num_te
    else:
        return 0, 0, num_data, torch.clamp(out[data_te_mask], min=y_min, max=y_max), data_y[data_te_mask], num_missing_value, num_tr, num_val, num_te





############################################################################################################
############################################################################################################
############################################################################################################
## 추가된 부분
############################################################################################################
############################################################################################################
############################################################################################################
def full_load_data_cv(data_path = './data/20221122_raw_data_utf-8.csv',
                num_features = 17,
                target_order = 3,
                num_folds = 3,
                tr_fold = 0,
                classification = True,
                device = 'cpu',
                model_name = 'MagNet',
                args = None):
    '''
    Description:
        Load Data from bottom
    Input :
        - data_path : Path to load data
        - num_features : Number of features on one node
        - target_order : The order which we want to predict
        - train_ratio : Ratio of train data
        - device : Running device (cuda / cpu)
    Output :
        - final_data_list : the data we use to learning
        - y_min : minimum value of y label 
        - y_max : maximum value of y label
    '''
    df_dat, unique_group = load_data(data_path = data_path, classification = classification, args=args)

    # aggregate group (for each graph)
    group_df = df_dat.groupby("transmission_route").agg(list)

    data_x_list, data_edge_list, data_y_list, data_order_list, num_graph = split_graphs(group_df = group_df, unique_group = unique_group, num_features = num_features, model_name = model_name, args=args)

    new_data_x_list, new_data_edge_list, new_data_y_list, new_data_order_list, new_data_tr_mask_list, new_data_val_mask_list, new_data_te_mask_list = masking_cv(data_x_list = data_x_list,
                                                                                                                            data_edge_list = data_edge_list,
                                                                                                                            data_y_list = data_y_list,
                                                                                                                            data_order_list = data_order_list,
                                                                                                                            target_order = target_order,
                                                                                                                            num_folds = num_folds,
                                                                                                                            tr_fold = tr_fold)

    final_data_x_list, final_data_edge_list, final_data_y_list, final_data_order_list, final_data_tr_mask_list, final_data_val_mask_list, final_data_te_mask_list = del_zero_edge(new_data_x_list = new_data_x_list,
                                                                                                                            new_data_edge_list = new_data_edge_list,
                                                                                                                            new_data_y_list = new_data_y_list,
                                                                                                                            new_data_order_list = new_data_order_list,
                                                                                                                            new_data_tr_mask_list = new_data_tr_mask_list,
                                                                                                                            new_data_val_mask_list = new_data_val_mask_list,
                                                                                                                            new_data_te_mask_list = new_data_te_mask_list,
                                                                                                                            num_graph = num_graph)

    final_data_x_list, num_birthyear_disc = birthyear_disc(final_data_x_list, final_data_y_list, final_data_tr_mask_list, final_data_val_mask_list, args)
    final_data_y_list, y_min, y_max = scaling_y(final_data_y_list, final_data_tr_mask_list, final_data_val_mask_list)

    check_graphs(final_data_x_list = final_data_x_list,
                final_data_edge_list = final_data_edge_list,
                final_data_y_list = final_data_y_list,
                final_data_order_list = final_data_order_list, 
                final_data_tr_mask_list = final_data_tr_mask_list,
                final_data_val_mask_list = final_data_val_mask_list,
                final_data_te_mask_list = final_data_te_mask_list)

    final_data_list = to_tensor(final_data_x_list = final_data_x_list,
                            final_data_edge_list = final_data_edge_list,
                            final_data_y_list = final_data_y_list,
                            final_data_order_list = final_data_order_list,
                            final_data_tr_mask_list = final_data_tr_mask_list,
                            final_data_val_mask_list = final_data_val_mask_list,
                            final_data_te_mask_list = final_data_te_mask_list,
                            device = device)

    return final_data_list, y_min, y_max, num_birthyear_disc



def masking_cv(data_x_list, data_edge_list, data_y_list, data_order_list, target_order, num_folds=3, tr_fold=0, te_ratio=0.4):
    new_data_x_list = []
    new_data_edge_list = []
    new_data_y_list = []
    new_data_order_list = []
    new_data_tr_mask_list = []
    new_data_val_mask_list = []
    new_data_te_mask_list = []

    for data_x, data_edge, data_y, data_order in zip(data_x_list, data_edge_list, data_y_list, data_order_list):
        mask_node_idx_list = np.argwhere(np.asarray(data_order) > target_order) # one-hop 보장
        
        # data node / label / order remove
        if not len(mask_node_idx_list) == 0:
            data_x = list(np.delete(list(data_x), mask_node_idx_list, axis=0))
            data_y = list(np.delete(data_y, mask_node_idx_list, axis=0))
            data_order = list(np.delete(data_order, mask_node_idx_list, axis=0))

        # edge remove
        edge_node_idx = []
        for mask_node_idx in mask_node_idx_list:
            match_value1 = np.argwhere(np.asarray(data_edge[0]) == mask_node_idx)
            match_value2 = np.argwhere(np.asarray(data_edge[1]) == mask_node_idx)
            edge_node_idx.extend(match_value1)
            edge_node_idx.extend(match_value2)
        edge_node_idx = np.unique(edge_node_idx)
        if not len(edge_node_idx) == 0:
            data_edge = [list(np.delete(data_edge[0], edge_node_idx)), list(np.delete(data_edge[1], edge_node_idx))]

        unique_node = list(set(data_edge[0] + data_edge[1]))
        unique_node = sorted(unique_node)
        old_new_dict = {}
        cnt = 0
        for node in unique_node:
            old_new_dict[node] = cnt
            cnt += 1
        
        l1 = []
        l2 = []
        for l1_ele, l2_ele in zip(data_edge[0], data_edge[1]):
            l1.append(old_new_dict[l1_ele])
            l2.append(old_new_dict[l2_ele])
        data_edge = [l1,l2]
        ####################
        mask_node_idx_list = np.argwhere(np.asarray(data_order) == target_order) 
        
        tr_data_mask_ = np.ones((len(data_x),)) # train mask lower than target order
        tr_data_mask_[np.asarray(mask_node_idx_list)] = 0
        # num_pure_tr = int(np.sum(tr_data_mask_))
        tr_val_data_mask = np.zeros((len(data_x),num_folds))
        te_data_mask = np.zeros((len(data_x),)) # test mask

        if len(mask_node_idx_list) == 1:
            mask_node_idx_list = [int(mask_node_idx_list[0])]
        else:
            mask_node_idx_list = mask_node_idx_list.squeeze()

        if not len(mask_node_idx_list) == 0:
            # Split Test 
            te_mask_node_idx_list = np.random.choice(np.asarray(mask_node_idx_list),
                                            size=int(len(mask_node_idx_list)*te_ratio),
                                            replace=False)
            te_data_mask[te_mask_node_idx_list] = 1
            
            # get whole folds of data (except te data)
            tr_val_mask_node_idx_list = list()
            for idx in mask_node_idx_list:
                if idx not in te_mask_node_idx_list:
                    tr_val_mask_node_idx_list.append(idx)
            random.shuffle(tr_val_mask_node_idx_list)

            # split into folds
            for fold ,idx in enumerate(tr_val_mask_node_idx_list):
                num_fold = fold % num_folds
                tr_val_data_mask[idx, num_fold] = 1
            
            # assign train data
            tr_data_mask = np.asarray(tr_data_mask_) + tr_val_data_mask[:, tr_fold]
            val_data_mask = np.delete(tr_val_data_mask, tr_fold, axis=1)
            
            # assign validation data
            val_data_mask = np.sum(val_data_mask, axis=1)
        else:
            tr_data_mask = tr_data_mask_
            val_data_mask = np.zeros((len(data_x, )))

        new_data_x_list.append(data_x)
        new_data_edge_list.append(data_edge)
        new_data_y_list.append(data_y)
        new_data_order_list.append(data_order)
        new_data_tr_mask_list.append(tr_data_mask)
        new_data_val_mask_list.append(val_data_mask)
        new_data_te_mask_list.append(te_data_mask)

    assert len(new_data_x_list) == len(new_data_edge_list) == len(new_data_y_list) == len(new_data_order_list) == len(new_data_tr_mask_list)
    print(f"Exist {len(new_data_x_list)} graphs after second step....(Masking)")

    return new_data_x_list, new_data_edge_list, new_data_y_list, new_data_order_list, new_data_tr_mask_list, new_data_val_mask_list, new_data_te_mask_list
