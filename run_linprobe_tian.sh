#!/bin/bash

# export MASTER_ADDR=localhost
# export MASTER_PORT=29500
# export WORLD_SIZE=1
# export LOCAL_RANK=-1
# export RANK=0

datasets=(
    "semi-aves" 
    # "flowers102" 
    # "fgvc-aircraft" 
    # "eurosat" 
    # "dtd" 
    # "oxford_pets" 
    # "food101" 
    # "stanford_cars" 
    # "imagenet"
)

# echo "Starting MAE Pretraining on datasets: ${datasets[*]}"

for dataset in "${datasets[@]}"; do
    echo "MAE Pretraining on $dataset"

    # OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 \
    
    python \
    main_pretrain_tian.py \
    --output_dir output/scratch-slurm \
    --batch_size 512 \
    --accum_iter 8 \
    --model mae_vit_base_patch32 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --warmup_epochs 10 \
    --epochs 300 \
    --blr 1.5e-4 \
    --num_workers 8 \
    --weight_decay 0.05 \
    --dataset $dataset \
    --unlabeled_split u_train_in.txt

done

# effective batchsize = batch_size (per gpu) * nodes * 8 (gpus per node) * accum_iter