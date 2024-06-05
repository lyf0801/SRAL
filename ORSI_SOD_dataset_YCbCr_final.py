from torchvision import models, transforms
from torch.utils import data
from PIL import Image
import os
import numpy as np


def dataset_info(dt):   
    assert dt in ['EORSSD']
    if dt == 'EORSSD':
        dt_mean = [0.3412, 0.3798, 0.3583]
        dt_std = [0.1148, 0.1042, 0.0990]
    return dt_mean, dt_std

def random_aug_transform(): 
    flip_h = transforms.RandomHorizontalFlip(p=1)
    flip_v = transforms.RandomVerticalFlip(p=1)
    angles = [0, 90, 180, 270]
    rot_angle = angles[np.random.choice(4)]
    rotate = transforms.RandomRotation((rot_angle, rot_angle))
    r = np.random.random()
    if r <= 0.25:
        flip_rot = transforms.Compose([flip_h, flip_v, rotate])
    elif r <= 0.5:
        flip_rot = transforms.Compose([flip_h, rotate])
    elif r <= 0.75:
        flip_rot = transforms.Compose([flip_v, flip_h, rotate])  
    else:
        flip_rot = transforms.Compose([flip_v, rotate])
    return flip_rot



class ORSI_SOD_dataset(data.Dataset):
    def __init__(self, root, size, mode, aug=False):
        self.mode = mode 
        self.aug = aug 
        self.dt_mean, self.dt_std = dataset_info('EORSSD')
        self.prefixes = [line.strip() for line in open(os.path.join(root, mode+'.txt'))]
        self.image_paths = [os.path.join(root, 'images', prefix + '.jpg') for prefix in self.prefixes]
        self.label_paths = [os.path.join(root, 'labels', prefix + '.png') for prefix in self.prefixes]

        self.image_lr_transformation = transforms.Compose([transforms.Resize((size, size),Image.BILINEAR),transforms.ToTensor()])
        self.image_sr_transformation = transforms.Compose([transforms.Resize((size*2, size*2),Image.BILINEAR),transforms.ToTensor()])

        self.image_lr_transformation_rgb = transforms.Compose([transforms.Resize((size, size),Image.BILINEAR),transforms.ToTensor(), transforms.Normalize(self.dt_mean, self.dt_std)])
        self.image_sr_transformation_rgb = transforms.Compose([transforms.Resize((size*2, size*2),Image.BILINEAR),transforms.ToTensor(), transforms.Normalize(self.dt_mean, self.dt_std)])

        self.label_transformation = transforms.Compose([transforms.Resize((size*2, size*2),Image.BILINEAR),transforms.ToTensor()])

    def __getitem__(self, index):
        if self.mode == "train": 
            if self.aug:
                flip_rot = random_aug_transform()
                """
                change color space
                """
                image_YCbCr = Image.open(self.image_paths[index]).convert('YCbCr')

                image_YCbCr_lr = self.image_lr_transformation(flip_rot(image_YCbCr)) ##low-resolution YCbCr image
                image_YCbCr_sr = self.image_sr_transformation(flip_rot(image_YCbCr)) ##super-resolution YCbCr image

                image_lr_rgb = self.image_lr_transformation_rgb((flip_rot(Image.open(self.image_paths[index]).convert('RGB'))))
                image_sr_rgb = self.image_sr_transformation_rgb((flip_rot(Image.open(self.image_paths[index]).convert('RGB'))))
                label = self.label_transformation(flip_rot(Image.open(self.label_paths[index]).convert('L')))
                
        elif self.mode == "test": 
            """
            change color space
            """
            image_YCbCr = Image.open(self.image_paths[index]).convert('YCbCr')

            image_YCbCr_lr = self.image_lr_transformation(image_YCbCr) ##low-resolution YCbCr image
            image_YCbCr_sr = self.image_sr_transformation(image_YCbCr) ##super-resolution YCbCr image

            image_lr_rgb = self.image_lr_transformation_rgb(Image.open(self.image_paths[index]).convert('RGB'))
            image_sr_rgb = self.image_sr_transformation_rgb(Image.open(self.image_paths[index]).convert('RGB'))
            label = self.label_transformation(Image.open(self.label_paths[index]).convert('L'))
            
        name = self.prefixes[index]
        
        return image_lr_rgb, image_YCbCr_lr, image_YCbCr_sr,  label, name

    def __len__(self):
        return len(self.prefixes)

