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
    echo "MAE few-shot finetuning on $dataset"

    # python \
    # main_finetune_tian.py \
    # --finetune maws_checkpoints/mae_vit_b16.pt \
    # --model vit_base_patch16 \
    # --nb_classes 200 \
    # --output_dir output/maws_mae_vit_b16/FSFT \
    # --cls_token \
    # --batch_size 32 \
    # --epochs 100 \
    # --blr 5e-4 \
    # --layer_decay 0.65 \
    # --weight_decay 0.05 \
    # --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    # --dataset $dataset \
    # --shot 16 \
    # --fewshot_seed 1
    

    python \
    main_finetune_tian.py \
    --finetune maws_checkpoints/maws_vit_b16.pt \
    --model vit_base_patch16 \
    --nb_classes 200 \
    --output_dir output/maws_vit_b16/FSFT \
    --cls_token \
    --batch_size 32 \
    --epochs 100 \
    --blr 5e-4 \
    --layer_decay 0.65 \
    --weight_decay 0.05 \
    --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dataset $dataset \
    --shot 16 \
    --fewshot_seed 1   


done