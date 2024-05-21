for lr_init in 1e-3 1e-4
do
for wd in 1e-3 1e-4
do
for hidden_dim in 128 256
do
for num_layers in 1 2
do
for num_heads in 2 4
do
drop_out=0.0
CUDA_VISIBLE_DEVICES=2 /mlainas/bubble3jh/anaconda3/envs/cluster/bin/python3 run_itransformer.py \
--model=iTransformer --use_treatment --single_treatment --run_group=synthetic_5415_1789 \
--hidden_dim=${hidden_dim} --num_layers=${num_layers} --num_heads=${num_heads} \
--lr_init=${lr_init} --optim=adam --wd=${wd} --drop_out=${drop_out} &

drop_out=0.2
CUDA_VISIBLE_DEVICES=3 /mlainas/bubble3jh/anaconda3/envs/cluster/bin/python3 run_itransformer.py \
--model=iTransformer --use_treatment --single_treatment --run_group=synthetic_5415_1789 \
--hidden_dim=${hidden_dim} --num_layers=${num_layers} --num_heads=${num_heads} \
--lr_init=${lr_init} --optim=adam --wd=${wd} --drop_out=${drop_out} &

wait
done
done
done
done
done
# for lr_init in 1e-3 1e-4
# do
# for wd in 1e-3 1e-4
# do
# for alpha in 10 1 0.1 
# do
# for optim in 'adam'
# do
# CUDA_VISIBLE_DEVICES=6 /mlainas/bubble3jh/anaconda3/envs/cluster/bin/python3 run_causal.py \
# --model=dragonnet --use_treatment --single_treatment --run_group=synthetic_5415_1789 \
# --lr_init=${lr_init} --optim=${optim} --wd=${wd} --alpha=${alpha} --is_synthetic &

# CUDA_VISIBLE_DEVICES=7 /mlainas/bubble3jh/anaconda3/envs/cluster/bin/python3 run_causal.py \
# --model=tarnet --use_treatment --single_treatment --run_group=synthetic_5415_1789 \
# --lr_init=${lr_init} --optim=${optim} --wd=${wd} --alpha=${alpha} --is_synthetic &

# wait 
# done
# done
# done
# done
