CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 train.py --name 40_noise_peer --dataset CUB \
 --model_type ViT-B_16 --pretrained_dir /home/xbj/lable_noising/pretrained/ViT-B_16.npz --img_size 448 --resize_size 600 \
 --train_batch_size 16 --learning_rate 0.02 --num_steps 10000 --fp16 --eval_every 200 --feature_fusion \
 --data_root /home/xbj/lable_noising/release --noise_rate 0.8 --decay_type linear --peer