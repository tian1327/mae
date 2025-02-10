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

    # python \
    # main_linprobe_tian.py \
    # --finetune maws_checkpoints/mae_vit_b16.pt \
    # --model vit_base_patch16 \
    # --nb_classes 200 \
    # --output_dir output/maws_mae_vit_b16/linearprobe \
    # --cls_token \
    # --batch_size 32 \
    # --epochs 100 \
    # --blr 0.1 \
    # --weight_decay 0.0 \
    # --dataset $dataset \
    # --shot 16 \
    # --fewshot_seed 1
    


    python \
    main_linprobe_anwesha.py \
    --finetune output/imagenet-vitb16-scratch-semi-aves/checkpoint-500.pth \
    --model vit_base_patch16 \
    --nb_classes 200 \
    --output_dir output/imagenet-vitb16-scratch-semi-aves/vit_probing \
    --cls_token \
    --batch_size 32 \
    --epochs 100 \
    --blr 1e-3 \
    --weight_decay 0.2 \
    --dataset $dataset \
    --shot 16 \
    --fewshot_seed 1 \
    --head_type vit_head

done