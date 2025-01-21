20250120 Mon

1. Pretrain MAE on semi-Aves unlabeled dataset
```bash
conda activate mae

# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py \
# --output_dir pt300_mae --log_dir pt300_mae \
# --batch_size 512 \
# --model mae_vit_base_patch16 \
# --norm_pix_loss \
# --mask_ratio 0.75 \
# --warmup_epochs 10 --epochs 300 \
# --blr 1.5e-4 --weight_decay 0.05 \
# --data_path /data/datasets/ILSVRC2012

# use the run_pretrain_tian.sh script to run this
bash run_pretrain_tian.sh

```
