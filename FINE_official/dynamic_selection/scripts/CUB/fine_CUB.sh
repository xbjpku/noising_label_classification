#!/bin/bash

python main.py -d 1 --asym false --percent 0.6 --lr_scheduler multistep --arch rn34 --loss_fn cce --dataset CUB --traintools robustloss --no_wandb --dynamic --distill_mode fine-gmm --seed 123 --warmup 40