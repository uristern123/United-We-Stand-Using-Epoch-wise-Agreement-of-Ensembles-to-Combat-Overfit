from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
from PIL import ImageFile
import torch
import os

class imagenet_dataset(Dataset):
    def __init__(self, root_dir, transform, num_class):
        self.root = root_dir + 'imagenet/val/'
        self.transform = transform
        self.val_data = []
        for c in range(num_class):
            imgs = os.listdir(self.root + str(c))
            for img in imgs:
                self.val_data.append([c, os.path.join(self.root, str(c), img)])

    def __getitem__(self, index):
        data = self.val_data[index]
        target = data[0]
        image = Image.open(data[1]).convert('RGB')
        img = self.transform(image)
        return img, target

    def __len__(self):
        return len(self.val_data)


class webvision_dataset(Dataset):
    def __init__(self, root_dir, transform, mode, num_class):
        self.root = root_dir
        self.transform = transform
        self.mode = mode
        if self.mode == 'test':
            with open(self.root + 'info/val_filelist.txt') as f:
                lines = f.readlines()
            self.val_imgs = []
            # self.val_labels = {}
            self.targets = []
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target < num_class:
                    self.val_imgs.append(img)
                    self.targets.append(target)
                    # self.val_labels[img] = target
            self.data = self.val_imgs
        else:
            with open(self.root + 'info/train_filelist_google.txt') as f:
                lines = f.readlines()
            train_imgs = []
            self.targets = []
            # self.train_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target < num_class:
                    train_imgs.append(img)
                    # self.train_labels[img] = target
                    self.targets.append(target)
            self.data = train_imgs

    def __getitem__(self, index):
        if self.mode == 'train':
            img_path = self.data[index]
            # target = self.train_labels[img_path]
            target = self.targets[index]
            try:
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                image = Image.open(self.root + img_path).convert('RGB')
            except:
                print("image: "+self.root + img_path+" is broken")
            img = self.transform(image)
            return img, target
        elif self.mode == 'test':
            img_path = self.data[index]
            target = self.targets[index]
            # target = self.val_labels[img_path]
            image = Image.open(self.root + 'val_images/' + img_path).convert('RGB')
            img = self.transform(image)
            return img, target

    def __len__(self):
        # if self.mode != 'test':
        return len(self.data)
        # else:
        #     return len(self.val_imgs)


class webvision_dataloader():
    def __init__(self, batch_size, num_class, num_workers, root_dir):
        self.batch_size = batch_size
        self.num_class = num_class
        self.num_workers = num_workers
        self.root_dir = root_dir

        self.transform_train = transforms.Compose([
            transforms.Resize(320),
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.transform_imagenet = transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(299),
            transforms.ToTensor(),            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def run(self, mode):
        if mode == 'train':
            all_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="train",
                                            num_class=self.num_class)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)
            return trainloader
        elif mode == 'test':
            test_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transform_test, mode='test',
                                             num_class=self.num_class)
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size * 5,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            return test_loader
        elif mode == 'eval_train':
            eval_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transform_test, mode='all',
                                             num_class=self.num_class)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size * 5,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            return eval_loader

        elif mode == 'imagenet':
            imagenet_val = imagenet_dataset(root_dir=self.root_dir, transform=self.transform_imagenet,
                                            num_class=self.num_class)
            imagenet_loader = DataLoader(
                dataset=imagenet_val,
                batch_size=self.batch_size * 5,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            return imagenet_loader

if __name__ =='__main__':
    import matplotlib.pyplot as plt
    loader = webvision_dataloader(batch_size=128, num_workers=5, root_dir='/cs/labs/daphna/daniels44/BiModal/datasets/webvision/', num_class=50)
    trainloader = loader.run('train')
    print(len(trainloader)*128)
    testloader = loader.run('test')
    print(len(testloader)*128*20)
    b=1
    for batch_idx, (img, target, index) in enumerate(trainloader):
        print(batch_idx,target)
        # plt.imshow(img.numpy()[0].transpose([1, 2, 0]))
        # plt.show()
        # plt.imshow(img)
        # plt.imshow(img)
        # plt.show()
