
Follow instructions in [DeiT](https://github.com/facebookresearch/deit/blob/main/README_deit.md)

```bash
conda create -n mae python=3.8 -y

conda activate mae

conda install -c pytorch pytorch torchvision python

pip install tensorboard

pip install timm==0.3.2

# fix the torch._six issue

# step 1: 
# follow instructions in https://github.com/huggingface/pytorch-image-models/issues/420#issuecomment-776459842
# add the code to timm/models/layers/helpers.py

# step 2:
# https://github.com/facebookresearch/mae/issues/172
# fix the inf code

```