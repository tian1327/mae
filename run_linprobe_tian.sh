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

for dataset in "${datasets[@]}"; do
    echo "MAE linear probing on $dataset"

    python \
    main_linprobe_tian.py \
    --finetune output/scratch-aves-vitb32-bs256-dec2b/checkpoint-299.pth \
    --model vit_base_patch32 \
    --nb_classes 200 \
    --output_dir output/scratch-aves-vitb32-bs25-dec2b/linearprobe \
    --cls_token \
    --batch_size 32 \
    --epochs 10 \
    --blr 0.1 \
    --weight_decay 0.0 \
    --dataset $dataset \
    --shot 16 \
    --fewshot_seed 1
    

    # try imagenet pretrained checkpoint vitb16

    # python \
    # main_linprobe_tian.py \
    # --finetune imagenet_pretrained_checkpoint/mae_pretrain_vit_base.pth \
    # --model vit_base_patch16 \
    # --nb_classes 1000 \
    # --output_dir output/imagenet-full-linearprobe \
    # --cls_token \
    # --batch_size 512 \
    # --accum_iter 32 \
    # --epochs 90 \
    # --blr 0.1 \
    # --weight_decay 0.0 \
    # --dataset $dataset \
    # --shot 16 \
    # --fewshot_seed 1 \
    # --imgnet_pretrained

done