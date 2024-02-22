from __future__ import print_function
import os
import os.path
import errno
import numpy as np
import sys
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pickle
from torchvision.datasets.utils import download_url, check_integrity


class Animal10n(Dataset):
    def __init__(self, root='/cs/labs/daphna/daniels44/BiModal/datasets/raw_image/', train=True,
                 transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.classes = ['cat','lynx','wolf','coyote','cheetah','jaguer','chimpanzee','orangutan','hamster','guinea']
        self.class_to_idx={0: 'cat'       ,
        1: 'lynx'       ,
        2: 'wolf'       ,
        3: 'coyote'     ,
        4: 'cheetah'    ,
        5: 'jaguer'     ,
        6: 'chimpanzee' ,
        7: 'orangutan'  ,
        8: 'hamster'    ,
        9: 'guinea'}
        if self.train:
            dirname = 'training/'
        else:
            dirname = 'testing/'
        filenames = os.listdir(root+dirname)
        self.data, self.targets = [],[]
        for filename in filenames:
            self.data.append(Image.open(self.root+dirname+filename).convert('RGB'))
            self.targets.append(int(filename[0]))


    def __getitem__(self, index):
        target = self.targets[index]
        image = self.data[index]
        img = self.transform(image)
        return img,target

    def __len__(self):
        return len(self.targets)
