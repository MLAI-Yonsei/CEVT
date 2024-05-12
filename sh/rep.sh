#!/bin/bash

seeds=(998 999 1001 1002 2000 3000)

for seed in "${seeds[@]}"; do
    echo "Running experiments with seed $seed"

    # CET model
    CUDA_VISIBLE_DEVICES=6 python main.py --model=cet --cutoff_dataset=0 --hidden_dim=128 --optim=adam --lr_init=1e-2 --wd=1e-2 --epochs=300 --scheduler=cos_anneal --t_max=300 --drop_out=0.1 --num_layers=1 --cet_transformer_layers=5 --num_features=64 --num_heads=4 --lambdas 1 1e-1 1e-6 --run_group=rep --sig_x0=0.75 --residual_x --use_treatment --MC_sample=1 --seed=$seed &

    # Transformer model
    CUDA_VISIBLE_DEVICES=5 python main.py --model=transformer --hidden_dim=128 --optim=adam --lr_init=1e-2 --wd=1e-4 --epochs=200 --scheduler=cos_anneal --t_max=200 --drop_out=0.0 --num_layers=2 --num_heads=8 --run_group=rep --cutoff_dataset=0 --seed=$seed &

    # MLP model
    CUDA_VISIBLE_DEVICES=4 python main.py --model=mlp --hidden_dim=128 --optim=adam --lr_init=1e-3 --wd=1e-3 --epochs=200 --scheduler=cos_anneal --t_max=200 --drop_out=0.1 --num_layers=2 --num_features=128 --run_group=rep --cutoff_dataset=0 --seed=$seed &

    # Linear model
    CUDA_VISIBLE_DEVICES=3 python main.py --model=linear --hidden_dim=128 --optim=adam --lr_init=1e-4 --wd=1e-3 --epochs=200 --scheduler=cos_anneal --t_max=200 --drop_out=0.1 --num_layers=2 --num_features=128 --run_group=rep --cutoff_dataset=0 --seed=$seed &

    # Ridge regression model
    CUDA_VISIBLE_DEVICES=2 python main.py --model=ridge --hidden_dim=256 --optim=adam --lr_init=1e-2 --wd=1e-4 --epochs=200 --scheduler=cos_anneal --t_max=200 --drop_out=0.1 --num_layers=3 --num_features=256 --lamb=10 --run_group=rep --cutoff_dataset=0 --seed=$seed &

    wait
done