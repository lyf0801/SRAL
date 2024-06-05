
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

from ORSI_SOD_dataset_YCbCr_final import ORSI_SOD_dataset
from src.baseline123_original_YCbCr import net as Net 
from evaluator_SR_YCbCr import Eval_thread



os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def unload(x):
    y = x.squeeze().cpu().data.numpy()
    return y
def convert2img(x):
    return Image.fromarray(x*255).convert('L')
def min_max_normalization(x):
    x_normed = (x - np.min(x)) / (np.max(x)-np.min(x))
    return x_normed
def save_smap(smap, path, negative_threshold=0.25):
    # smap: [1, H, W]
    if torch.max(smap) <= negative_threshold:
        smap[smap<negative_threshold] = 0
        smap = convert2img(unload(smap))
    else:
        smap = convert2img(min_max_normalization(unload(smap)))
    
    smap.save(path)


def getsmaps(dataset_name):
    ##define test dataset and test dataloader
    dataset_root  = "/data/iopen/lyf/SaliencyOD_in_RSIs/" + dataset_name + " dataset/"

    test_set = ORSI_SOD_dataset(root = dataset_root, size = 224, mode = "test", aug = False)  
    test_loader = DataLoader(test_set, batch_size = 1, shuffle = False, num_workers = 1)

    
    
    # define model
    net = Net().cuda().eval()  
    if dataset_name == "ORSSD":
        net.load_state_dict(torch.load("./data/weights/SRAL_ORSSD.pth", map_location='cuda:0')) 
    elif dataset_name == "EORSSD":
        net.load_state_dict(torch.load("./data/weights/SRAL_EORSSD.pth", map_location='cuda:0'))
    elif dataset_name == "ORS_4199":
        net.load_state_dict(torch.load("./data/weights/SRAL_ORS_4199.pth", map_location='cuda:0'))
        

    thread = Eval_thread(epoch = 0, model = net.eval(), loader = test_loader, method = "ASRNet_YCbCr_final", dataset = dataset_name, output_dir = "./data/output", cuda=True)  
    logg, fm = thread.run()
    print(logg)
    infer_time = 0
    for image_rgb_lr, image_YCbCr_lr, image_YCbCr_sr, label, name in tqdm(test_loader):  #image_lr, image_sr, label, name  
        input1 = image_rgb_lr.cuda()  ###input YCbCr space
        input2 = image_YCbCr_lr.cuda()
        net.eval()
        with torch.no_grad():
            smaps = net(input1, input2)  
            path = "./data/output/predict_smaps_ASRNet_" + dataset_name + "_YCbCr_final/"
            if not os.path.exists(path):
                os.makedirs(path)
            save_smap(smaps, os.path.join(path, name[0]+"_ASRNet_YCbCr_final.png"))


if __name__ == "__main__":
        

    dataset = ["ORSSD", "EORSSD", "ORS_4199"]
    for datseti in dataset:
        getsmaps(datseti)
    