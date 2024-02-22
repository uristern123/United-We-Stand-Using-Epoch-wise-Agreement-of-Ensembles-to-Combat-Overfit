from torchvision.datasets import ImageFolder
from torchvision.datasets.imagenet import ImageNet
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import pandas as pd
import torch
import os
import numpy as np
# faster init for imagenet


class MyImageNet(ImageFolder):
    def __init__(self,
                 root: str = '/cs/labs/daphna/data/imagenet/',
                 split='train',
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 loader: Callable[[str], Any] = default_loader,
                 is_valid_file: Optional[Callable[[str], bool]] = None,
                 test_transform=None,
                 files_list: str = '/cs/labs/daphna/data/imagenet/imagenet_{}_files.txt',
                 num_classes = 50, only_features=False
                 ):
        self.root = os.path.join(root, 'ILSVRC2012_img_{}_preprocessed/'.format(split))
        self.split = split
        self.no_aug = False
        self.test_transform = test_transform
        cls_to_name, val_labels = torch.load('/cs/labs/daphna/daphna/data/imagenet/meta.bin')
        self.orig_classes = sorted(list(cls_to_name.keys()))
        self.class_to_idx = {k:i for i,k in enumerate(self.orig_classes)}
        #fname = '/cs/labs/daphna/avihu.dekel/dino/runs/{}feat.pth'.format('train' if split == 'train' else 'test')
        #self.all_features = torch.load(fname).numpy()
        # instead of iterating over the whole folder, load saved files - much faster
        if split == 'train':
            df = pd.read_csv(files_list.format(split), names=['_', 'class', 'sample'], sep='/')
            self.orig_imgs = (self.root + df['class'] + '/' + df['sample'])
            self.orig_targets = np.array([self.class_to_idx[s] for s in df['class']])
        else:
            ser = pd.read_csv(files_list.format(split), header=None)
            self.orig_imgs = (self.root + ser)
            self.orig_targets = np.array([self.class_to_idx[s] for s in val_labels])

        # filter to relevant classes - according to SCAN split
        if num_classes < 1000:
            classes_fname = f'/cs/labs/daphna/archive/avihu.dekel/Unsupervised-Classification/data/imagenet_subsets/imagenet_{num_classes}.txt'
            #classes_df = pd.read_csv(classes_fname, header=None, squeeze=True, sep='\t')
            classes_df = pd.read_csv(classes_fname, header=None, sep='\t')
            self.classes = [class_name[0].split(' ')[0] for class_name in classes_df.values]
            self.relevant_ids = [self.class_to_idx[n] for n in self.classes]
            # contains the indices from the original ordering
            self.rel_indices = np.isin(self.orig_targets, self.relevant_ids).nonzero()[0]
            #self.features = self.all_features[self.rel_indices]
            if split == 'train':
                self.data = list(self.orig_imgs[self.rel_indices])
            else:
                self.data = list(self.orig_imgs[0][self.rel_indices])
            self.filtered_targets = self.orig_targets[self.rel_indices]
            # re-orders targets to 0,..,K-1
            self.targets = np.zeros_like(self.filtered_targets)
            for i, val in enumerate(np.unique(self.filtered_targets)):
                self.targets[self.filtered_targets == val] = i
        else:
            self.data = np.asarray(list(self.orig_imgs))
            self.targets = self.orig_targets
            self.classes = range(1000)


        self.loader = loader
        self.extensions = IMG_EXTENSIONS
        self.transform = transform
        self.target_transform = target_transform
        self.is_valid_file = is_valid_file
        self.samples = list(zip(self.data, self.targets))
        self.only_features = only_features

    def set_targets(self, targets):
        self.targets = targets
        self.samples = list(zip(self.data, self.targets))
    def set_targets_and_data(self, targets, data):
        self.targets = targets
        self.data = data
        self.samples = list(zip(self.data, self.targets))
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        if not self.only_features:
            sample = self.loader(path)

            if self.no_aug:
                if self.test_transform is not None:
                    sample = self.test_transform(sample)
            else:
                if self.transform is not None:
                    sample = self.transform(sample)
        else:
            sample = self.features[index]
        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

