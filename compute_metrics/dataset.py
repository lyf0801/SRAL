from torchvision import models, transforms
from torch.utils import data
from PIL import Image
import os
import numpy as np

    
class ORSSD(data.Dataset):
    def __init__(self, root, mode, aug=False):
        self.mode = mode 
        self.aug = aug 
        self.prefixes = [line.strip() for line in open(os.path.join(root, mode+'.txt'))]
        self.image_paths = [os.path.join(root, 'images', prefix + '.jpg') for prefix in self.prefixes]
        self.label_paths = [os.path.join(root, 'labels', prefix + '.png') for prefix in self.prefixes]
        self.edge_paths = [os.path.join(root, 'edges', prefix + '.png') for prefix in self.prefixes]
        self.image_transformation = transforms.Compose([transforms.Resize((448,448),Image.BILINEAR),transforms.ToTensor()])
        self.label_transformation = transforms.Compose([transforms.Resize((448,448),Image.BILINEAR),transforms.ToTensor()])

    def __getitem__(self, index):
        
        if self.mode == "test": 
            image = self.image_transformation(Image.open(self.image_paths[index]).convert('RGB'))
            label = self.label_transformation(Image.open(self.label_paths[index]).convert('L'))
            edge = self.label_transformation(Image.open(self.edge_paths[index]))
        name = self.prefixes[index]
        
        return image, label, edge, name
        

    def __len__(self):
        return len(self.prefixes)
    
    
class EORSSD(data.Dataset):
    def __init__(self, root, mode, aug=False):
        self.mode = mode 
        self.aug = aug 
        self.prefixes = [line.strip() for line in open(os.path.join(root, mode+'.txt'))]
        self.image_paths = [os.path.join(root, 'images', prefix + '.jpg') for prefix in self.prefixes]
        self.label_paths = [os.path.join(root, 'labels', prefix + '.png') for prefix in self.prefixes]
        self.edge_paths = [os.path.join(root, 'edges', prefix + '.png') for prefix in self.prefixes]
        self.image_transformation = transforms.Compose([transforms.Resize((448,448),Image.BILINEAR),transforms.ToTensor()])
        self.label_transformation = transforms.Compose([transforms.Resize((448,448),Image.BILINEAR),transforms.ToTensor()])

    def __getitem__(self, index):
        
        if self.mode == "test": 
            image = self.image_transformation(Image.open(self.image_paths[index]).convert('RGB'))
            label = self.label_transformation(Image.open(self.label_paths[index]).convert('L'))
            edge = self.label_transformation(Image.open(self.edge_paths[index]))
        name = self.prefixes[index]
        
        return image, label, edge, name
        

    def __len__(self):
        return len(self.prefixes)

    
class ORS_4199(data.Dataset):
    def __init__(self, root, mode, aug=False):
        self.mode = mode 
        self.aug = aug
        self.prefixes = [line.strip() for line in open(os.path.join(root, mode+'.txt'))]
        self.image_paths = [os.path.join(root, 'images', prefix + '.jpg') for prefix in self.prefixes]
        self.label_paths = [os.path.join(root, 'labels', prefix + '.png') for prefix in self.prefixes]
        self.edge_paths = [os.path.join(root, 'edges', prefix + '.png') for prefix in self.prefixes]
        self.image_transformation = transforms.Compose([transforms.Resize((448,448),Image.BILINEAR),transforms.ToTensor()])
        self.label_transformation = transforms.Compose([transforms.Resize((448,448),Image.BILINEAR),transforms.ToTensor()])

    def __getitem__(self, index):
        
        if self.mode == "test": 
            image = self.image_transformation(Image.open(self.image_paths[index]).convert('RGB'))
            label = self.label_transformation(Image.open(self.label_paths[index]).convert('L'))
            edge = self.label_transformation(Image.open(self.edge_paths[index]))
        name = self.prefixes[index]
        
        return image, label, edge, name
        

    def __len__(self):
        return len(self.prefixes)