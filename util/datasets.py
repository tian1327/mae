# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from torch.utils.data import Dataset
import pathlib
from torchvision.datasets import folder as dataset_parser


class MyUnlabeledDataset(Dataset):
    def __init__(self, dataset_root, split, transform,
                 loader=dataset_parser.default_loader):
        
        self.dataset_root = pathlib.Path(dataset_root)
        self.loader = loader

        file_list = split[0]
        path_list = split[1]

        lines = []
        for file, path in zip(file_list, path_list):
            with open(os.path.join(self.dataset_root, file), 'r') as f:
                line = f.readlines()
                # prepend the path to the each line !!!
                line = [os.path.join(path, l) for l in line]
            lines.extend(line)

        self.data = []
        self.labels = []
        for line in lines:
            path, id, is_fewshot = line.strip('\n').split(' ')
            file_path = path
            self.data.append((file_path, int(id), int(is_fewshot)))
            self.labels.append(int(id))
        
        self.targets = self.labels  # Sampler needs to use targets

        self.transform = transform
        print(f'# of images in {split}: {len(self.data)}')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        
        img = self.loader(self.data[i][0])
        label = self.data[i][1]
        # source = self.data[i][2] # 0 for retrived data, 1 for fewshot data
        img = self.transform(img) # this will return weak aug and strong aug
        # tokenized_text = torch.zeros(1, 1).long() # dummy tokenized text

        return img, label


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
