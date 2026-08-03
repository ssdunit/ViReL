# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
import torch

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

class HFStreamingImageDataset(torch.utils.data.IterableDataset):
    def __init__(self, hf_iterable_dataset, transform=None, image_key='image', label_key='label'):
        self.dataset = hf_iterable_dataset
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key

    def __iter__(self):
        for item in self.dataset:
            img = item[self.image_key]
            if hasattr(img, 'convert'):
                img = img.convert('RGB')
            label = item[self.label_key]
            if self.transform is not None:
                img = self.transform(img)
            yield img, label

class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


class HFImageClassificationDataset:
    """
    Thin torch-Dataset-compatible wrapper around a Hugging Face `datasets` split,
    so it drops into the same (image, label) + torchvision-transform pipeline
    used by the local ImageFolder/CIFAR/iNat paths below.
    """
    def __init__(self, hf_dataset, transform=None, image_key='image', label_key='label'):
        self.dataset = hf_dataset
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item[self.image_key]
        if hasattr(img, 'convert'):
            img = img.convert('RGB')
        label = item[self.label_key]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def _parse_data_pct(data_pct: str) -> str:
    """'20%' / '20' -> '20' for use inside a HF split slice expression like 'train[:20%]'."""
    return str(data_pct).strip().rstrip('%')


def build_hf_dataset(is_train, args):
    # Imported lazily and only here: importing `datasets` (the Hugging Face pip
    # package) at module level in this file used to be impossible because this
    # file itself was named datasets.py, which shadowed the real package for
    # anything under deit/. Now that this file is data.py, the import below
    # correctly resolves to the pip package instead of itself.
    import datasets as _hf_datasets_pkg
    if not hasattr(_hf_datasets_pkg, 'load_dataset'):
        raise ImportError(
            f"Imported 'datasets' from {getattr(_hf_datasets_pkg, '__file__', '?')}, "
            "which has no load_dataset() - this is not the Hugging Face `datasets` pip "
            "package. Something in this project (likely a leftover local datasets.py or "
            "datasets/ folder, or a stale __pycache__ entry) is shadowing it. Delete the "
            "stray file/folder and try again."
        )
    load_dataset = _hf_datasets_pkg.load_dataset
    """
    pct = _parse_data_pct(args.data_pct)
    split_name = 'train' if is_train else 'validation'
    split_expr = f"{split_name}[:{args.data_pct}%]" if pct != '100' else split_name

    hf_dataset = load_dataset(args.dataset, split=split_name,streaming=True,token=True)


    transform = build_transform(is_train, args)
    """
    split_name = 'train' if is_train else 'validation'
    
    hf_dataset = load_dataset(args.dataset, split=split_name, streaming=True, token=True)
    
    max_samples = getattr(args, 'max_train_samples' if is_train else 'max_val_samples', None)
    if max_samples is not None and max_samples > 0:
        shuffle_buffer = getattr(args, 'shuffle_buffer_size', 0)
        if shuffle_buffer and shuffle_buffer > 0:
            hf_dataset = hf_dataset.shuffle(buffer_size=shuffle_buffer, seed=getattr(args, 'seed', 0))
        hf_dataset = hf_dataset.take(max_samples)
    
    transform = build_transform(is_train, args)
    if hasattr(hf_dataset, '__len__'):
        dataset = HFImageClassificationDataset(hf_dataset, transform=transform)
    else:
        dataset = HFStreamingImageDataset(hf_dataset, transform=transform)
    #dataset = HFImageClassificationDataset(hf_dataset, transform=transform)

    label_feature = hf_dataset.features.get('label')
    nb_classes = getattr(label_feature, 'num_classes', None)
    if nb_classes is None:
        import itertools
        peek_limit = getattr(args, 'nb_classes_peek_limit', 10000)
        seen_labels = {ex['label'] for ex in itertools.islice(hf_dataset, peek_limit)}
        nb_classes = len(seen_labels)

    return dataset, nb_classes


def build_dataset(is_train, args):
    if getattr(args, 'dataset', None):
        return build_hf_dataset(is_train, args)

    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
