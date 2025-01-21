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
    --finetune output/scratch/checkpoint-200.pth \
    --model vit_base_patch32 \
    --nb_classes 200 \
    --output_dir output/scratch-linearprobe \
    --cls_token \
    --batch_size 32 \
    --epochs 100 \
    --blr 0.1 \
    --weight_decay 0.0 \
    --dataset $dataset \
    --shot 16 \
    --fewshot_seed 1
    
done