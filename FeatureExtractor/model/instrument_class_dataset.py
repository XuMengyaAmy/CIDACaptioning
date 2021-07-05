'''
Paper : Incremental Domain Adaptation with Smoothing and Calibration for Surgical Report Generation
Note  : Dataloader for incremental learning
'''
import os
import sys
import random
import numpy as np
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET



class SurgicalClassDataset18_incremental(Dataset):
    def __init__(self, classes, memory=None, fine_tune_size=None,is_train=None): 
       
        self.is_train = is_train

        if self.is_train: self.dir_root_gt = '/media/mmlab/dataset/global_dataset/Classification_dataset/train/'
        else: self.dir_root_gt = '/media/mmlab/dataset/global_dataset/Classification_dataset/test/'
        
        self.xml_dir_list = []

        self.img_dir_list = []
        self.classes = classes
        if memory is not None: self.memory=memory
        else: self.memory=[]
        
        xml_dir_temp = self.dir_root_gt + '*.png' 
        self.xml_dir_list = self.xml_dir_list + glob(xml_dir_temp) 

        
        for _img_dir in self.xml_dir_list:
            _target = int(_img_dir[:-4].split('_')[-1:][0])  
            if _target in self.classes: 
                self.img_dir_list.append(_img_dir)
        random.shuffle(self.img_dir_list) 
        
        if fine_tune_size is not None:
            self.img_dir_list = self.memory + self.img_dir_list[0: fine_tune_size] 
        else:
            self.img_dir_list = self.img_dir_list + self.memory 

    def __len__(self):
        return len(self.img_dir_list)

    def __getitem__(self, index):
        _img_dir = self.img_dir_list[index]
        _img = Image.open(_img_dir).convert('RGB')
        
        _img = np.asarray(_img, np.float32) / 255
        _img = torch.from_numpy(np.array(_img).transpose(2, 0, 1,)).float()

        _target = int(_img_dir[:-4].split('_')[-1:][0])  
        _target = torch.from_numpy(np.array(_target)).long()
        return _img, _target


class SurgicalClassDataset18_incremental_transform(Dataset): 
    def __init__(self, classes, memory=None, fine_tune_size=None, transform=None, is_train=None): 
       
        self.is_train = is_train
        if self.is_train: self.dir_root_gt = '/media/mmlab/dataset/global_dataset/Classification_dataset/train/'
        else: self.dir_root_gt = '/media/mmlab/dataset/global_dataset/Classification_dataset/test/' 
        self.xml_dir_list = []
        self.img_dir_list = []
        self.classes = classes
        if memory is not None: self.memory=memory
        else: self.memory=[]
        self.transform = transform
        
        xml_dir_temp = self.dir_root_gt + '*.png' 
        self.xml_dir_list = self.xml_dir_list + glob(xml_dir_temp) 

        for _img_dir in self.xml_dir_list:
            _target = int(_img_dir[:-4].split('_')[-1:][0])  
            if _target in self.classes: 
                self.img_dir_list.append(_img_dir)
        random.shuffle(self.img_dir_list) 
        
        if fine_tune_size is not None:
            self.img_dir_list = self.memory + self.img_dir_list[0: fine_tune_size] 
        else:
            self.img_dir_list = self.img_dir_list + self.memory 

    def __len__(self):
        return len(self.img_dir_list)

    def __getitem__(self, index):
        _img_dir = self.img_dir_list[index]
        _img = Image.open(_img_dir).convert('RGB')

        if self.transform:
            _img = self.transform(_img)
            
        _target = int(_img_dir[:-4].split('_')[-1:][0])  
        _target = torch.from_numpy(np.array(_target)).long()
        return _img, _target

def memory_managment(classes, fine_tune_size): 
    
    dir_root_gt = '/media/mmlab/dataset/global_dataset/Classification_dataset/train/'
    xml_dir_list = []
    img_dir_list = []
    
    xml_dir_temp = dir_root_gt + '*.png' 
    xml_dir_list = xml_dir_list + glob(xml_dir_temp) 

    for _img_dir in xml_dir_list:
        _target = int(_img_dir[:-4].split('_')[-1:][0])  
        if _target in classes:
            img_dir_list.append(_img_dir)

    random.shuffle(img_dir_list) 

    '------------------------------------------------------------'
    '''
    for new_added_memory:
    period=0, choose some samples from class0-9 to form the memory_0; 
    period=1, choose some samples from class10-11 to form the memory_1;
    ''' 
    new_added_memory = img_dir_list[0: fine_tune_size]  
    return new_added_memory


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)] 


        
