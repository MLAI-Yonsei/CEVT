#!/bin/bash

run_group="synthetic_baselines"
seed=1000
# 사용 가능한 GPU ID 목록
GPU_IDS=(2 3 4 5 6 7)

# 현재 할당할 GPU의 인덱스
IDX=0

models=("transformer" "mlp" "linear" "ridge")

for model in "${models[@]}"; do
    for lr_init in 1e-3 1e-4 1e-2; do
        for wd in 1e-3 1e-4; do
            for drop_out in 0.0 0.1; do
                for hidden_dim in 32 64; do
                    for num_features in 32 64; do
                        for num_layers in 2 3; do
                            for num_heads in 2 4; do
                                for cutoff_dataset in 0; do 
                                # 현재 할당할 GPU ID
                                GPU_ID=${GPU_IDS[$IDX]}

                                if [[ "$model" == "transformer" ]]; then
                                    CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --model=$model --hidden_dim=$hidden_dim --optim=adam --lr_init=$lr_init --wd=$wd --epochs=200 --scheduler=cos_anneal --t_max=200 --drop_out=$drop_out --num_layers=$num_layers --num_heads=$num_heads --run_group=${run_group} --seed=${seed} --cutoff_dataset=${cutoff_dataset} --is_synthetic &
                                elif [[ "$model" == "ridge" ]]; then
                                    # Ridge 모델일 때 --lamb 10 추가
                                    CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --model=$model --hidden_dim=$hidden_dim --optim=adam --lr_init=$lr_init --wd=$wd --epochs=200 --scheduler=cos_anneal --t_max=200 --drop_out=$drop_out --num_layers=$num_layers --num_features=$num_features --lamb=10 --run_group=${run_group} --seed=${seed} --cutoff_dataset=${cutoff_dataset} --is_synthetic &
                                else
                                    # MLP, Linear 등 나머지 모델들
                                    CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --model=$model --hidden_dim=$hidden_dim --optim=adam --lr_init=$lr_init --wd=$wd --epochs=200 --scheduler=cos_anneal --t_max=200 --drop_out=$drop_out --num_layers=$num_layers --num_features=$num_features --run_group=${run_group} --seed=${seed} --cutoff_dataset=${cutoff_dataset} --is_synthetic &
                                fi

                                # GPU ID 업데이트
                                IDX=$(( ($IDX + 1) % ${#GPU_IDS[@]} ))

                                # 모든 GPU가 사용 중이면 기다림
                                if [ $IDX -eq 0 ]; then
                                    wait
                                fi
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

# 마지막에 남은 작업이 있다면 완료될 때까지 대기
wait
