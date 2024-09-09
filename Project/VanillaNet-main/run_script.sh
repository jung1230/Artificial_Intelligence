#!/bin/bash

py -3.9 -m torch.distributed.launch --nproc_per_node=1 main.py \
--model vanillanet_5 \
--data_path ../imagenet \
--batch_size 32 --update_freq 1 --epochs 10 --decay_epochs 100 \
--lr 3.5e-1 --weight_decay 0.35 --drop 0.05 \
--opt lamb --aa rand-m2-mstd0.5-inc1 --mixup 0.0 --bce_loss \
--output_dir ../output \
--model_ema true --model_ema_eval true --model_ema_decay 0.99996 \
--use_amp true
# true -> 12 days. false -> 12 days

