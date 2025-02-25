#!/bin/bash

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
    --output_dir output/scratch-aves-vitb32-bs256-dec2b \
    --batch_size 256 \
    --accum_iter 1 \
    --model mae_vit_base_patch32_d2b \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --warmup_epochs 15 \
    --epochs 800 \
    --blr 1.5e-4 \
    --num_workers 8 \
    --weight_decay 0.05 \
    --dataset $dataset \
    --unlabeled_split u_train_in.txt

done

# effective batchsize = batch_size (per gpu) * nodes * 8 (gpus per node) * accum_iter