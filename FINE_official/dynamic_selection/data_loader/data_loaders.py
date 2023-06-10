import sys

from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.cifar10 import get_cifar10
from data_loader.cifar100 import get_cifar100
from data_loader.clothing1m import get_clothing1m
from data_loader.webvision import get_webvision
from utils.parse_config import ConfigParser
from PIL import Image
import os
import numpy as np

class CUB():
    def __init__(self, root, is_train=True, transform=None, label_file = "training_labels_noise_00.txt", teacher_idx=None):
        self.root = root 
        self.is_train = is_train
        self.transform = transform
        if is_train:
            img_path = os.path.join(self.root, "train")
            label_path = os.path.join(self.root, label_file)
            print(label_file)
        else:
            img_path = os.path.join(self.root, "test")
            label_path = os.path.join(self.root, "test_labels.txt")
        print("Preparing dataset")
        img_name_list = os.listdir(img_path)
        if teacher_idx is not None:
            img_name_list = [img_name_list[i] for i in teacher_idx]
        label_list = {}
        gt_label_list = {}
        
        with open(os.path.join(self.root, "training_labels_noise_00.txt")) as gt_labels:
            for line in gt_labels:
                img_id, lable = tuple(line[:-1].split(' '))
                gt_label_list[int(img_id)] = (int(lable) - 1)
        
        
        with open(label_path) as labels:
            for line in labels:
                img_id, lable = tuple(line[:-1].split(' '))
                label_list[int(img_id)] = (int(lable) - 1)

        # print(img_name_list)
        if self.is_train:
            self.train_img = [os.path.join(img_path, train_file) for train_file in
                              img_name_list]
            self.train_labels = np.array([label_list[int(os.path.splitext(i)[0])] for i in img_name_list])
            self.train_labels_gt = np.array([gt_label_list[int(os.path.splitext(i)[0])] for i in img_name_list])
            # print(self.train_labels == self.train_labels_gt)
            self.train_imgname = [int(os.path.splitext(x)[0]) for x in img_name_list]
        if not self.is_train:
            # self.test_img = [transform(Image.open(os.path.join(img_path, test_file)).convert('RGB')) for test_file in
                            #  img_name_list[:data_len]]
            self.test_img = [os.path.join(img_path, test_file) for test_file in
                             img_name_list]
            self.test_labels = [label_list[int(os.path.splitext(i)[0])] for i in img_name_list]
            self.test_imgname = [x for x in img_name_list]
            
            
    def __getitem__(self, index):
        if self.is_train:
            img, target, imgname = self.train_img[index], self.train_labels[index], self.train_imgname[index]
            # if len(img.shape) == 2:
            #     img = np.stack([img] * 3, 2)
            # if self.transform is not None:
            #     img = self.transform(img)
            return self.transform(Image.open(img).convert('RGB')), target, index, self.train_labels_gt[index]
        else:
            img, target, imgname = self.test_img[index], self.test_labels[index], self.test_imgname[index]
            # if len(img.shape) == 2:
            #     img = np.stack([img] * 3, 2)
            # if self.transform is not None:
            #     img = self.transform(img)
            return self.transform(Image.open(img).convert('RGB')), target 

    def __len__(self):
        if self.is_train:
            return len(self.train_labels)
        else:
            # print(len(self.test_label))
            return len(self.test_labels)


class CUBDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_batches=0,  training=True, num_workers=4,  pin_memory=True, config=None, teacher_idx=None, seed=888):
        if config == None:
            config = ConfigParser.get_instance()
        cfg_trainer = config['trainer']
        
        train_transform=transforms.Compose([
            transforms.Resize((600, 600), Image.BILINEAR),
            transforms.RandomCrop((448, 448)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), # my add
            transforms.RandomHorizontalFlip(),
            
            transforms.ToTensor(),
            #transforms.Normalize([0.8416, 0.867, 0.8233], [0.2852, 0.246, 0.3262])])
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ])
        test_transform=transforms.Compose([
            transforms.Resize((600, 600), Image.BILINEAR),
            transforms.CenterCrop((448, 448)),
            transforms.ToTensor(),
            #transforms.Normalize([0.8416, 0.867, 0.8233], [0.2852, 0.246, 0.3262])])
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.data_dir = data_dir

        noise_file=f"training_labels_noise_{str(int(cfg_trainer['percent'] * 100)).format('%02d')}.txt"
        
        
        self.train_dataset = CUB(root=data_dir, is_train=True, transform=train_transform, label_file = noise_file, teacher_idx=teacher_idx)
        self.val_dataset = CUB(root=data_dir, is_train=False, transform = test_transform)

        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         val_dataset = self.val_dataset)
        
    def run_loader(self, batch_size, shuffle, validation_split, num_workers, pin_memory):
        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         val_dataset = self.val_dataset)

class CIFAR10DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_batches=0,  training=True, num_workers=4,  pin_memory=True, config=None, teacher_idx=None, seed=888):
        if config == None:
            config = ConfigParser.get_instance()
        cfg_trainer = config['trainer']
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.data_dir = data_dir

        noise_file='%sCIFAR10_%.1f_Asym_%s.json'%(config['data_loader']['args']['data_dir'],cfg_trainer['percent'],cfg_trainer['asym'])
        
        self.train_dataset, self.val_dataset = get_cifar10(config['data_loader']['args']['data_dir'], cfg_trainer, train=training, transform_train=transform_train, 
                                                           transform_val=transform_val, noise_file=noise_file, teacher_idx=teacher_idx, seed=seed)

        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         val_dataset = self.val_dataset)
        
    def run_loader(self, batch_size, shuffle, validation_split, num_workers, pin_memory):
        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         val_dataset = self.val_dataset)



class CIFAR100DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_batches=0, training=True, num_workers=4,  pin_memory=True, config=None, teacher_idx=None, seed=888):
        
        if config is None:
            config = ConfigParser.get_instance()
        cfg_trainer = config['trainer']
        
        transform_train = transforms.Compose([
                #transforms.ColorJitter(brightness= 0.4, contrast= 0.4, saturation= 0.4, hue= 0.1),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        self.data_dir = data_dir
#         cfg_trainer = config['trainer']

        noise_file='%sCIFAR100_%.1f_Asym_%s.json'%(config['data_loader']['args']['data_dir'],cfg_trainer['percent'],cfg_trainer['asym'])

        self.train_dataset, self.val_dataset = get_cifar100(config['data_loader']['args']['data_dir'], cfg_trainer, train=training, transform_train=transform_train, 
                                                            transform_val=transform_val, noise_file = noise_file, teacher_idx=teacher_idx, seed=seed)

        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         val_dataset = self.val_dataset)
    def run_loader(self, batch_size, shuffle, validation_split, num_workers, pin_memory):
        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         val_dataset = self.val_dataset)
        
class Clothing1MDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_batches=0, training=True, num_workers=4, pin_memory=True, config=None, teacher_idx=None, seed=8888):

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_batches = num_batches
        self.training = training

        self.transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),                
                transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),                     
            ]) 
        self.transform_val = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
            ])     

        self.data_dir = data_dir
        if config == None:
            config = ConfigParser.get_instance()
        cfg_trainer = config['trainer']
        self.train_dataset, self.val_dataset = get_clothing1m(config['data_loader']['args']['data_dir'], cfg_trainer, num_samples=self.num_batches*self.batch_size, train=training,
#         self.train_dataset, self.val_dataset = get_clothing1m(config['data_loader']['args']['data_dir'], cfg_trainer, num_samples=260000, train=training,
                transform_train=self.transform_train, transform_val=self.transform_val, teacher_idx=teacher_idx, seed=seed)

        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         val_dataset = self.val_dataset)
        
        
class WebvisionDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_batches=0, training=True, num_workers=4, pin_memory=True, num_class=50, teacher_idx=None):

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_batches = num_batches
        self.training = training

        self.transform_train = transforms.Compose([
                transforms.RandomCrop(227),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ]) 
        self.transform_val = transforms.Compose([
                transforms.CenterCrop(227),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ])  
        self.transform_imagenet = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(227),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ])         

        self.data_dir = data_dir
        config = ConfigParser.get_instance()
        cfg_trainer = config['trainer']
        self.train_dataset, self.val_dataset = get_webvision(config['data_loader']['args']['data_dir'], cfg_trainer, num_samples=self.num_batches*self.batch_size, train=training,
                transform_train=self.transform_train, transform_val=self.transform_val, num_class=num_class, teacher_idx=teacher_idx)

        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         val_dataset = self.val_dataset)

