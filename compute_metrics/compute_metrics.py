from pickle import FALSE
import numpy as np
import os
from PIL import Image
from evaluator import Eval_thread
from dataset import ORSSD, EORSSD, ORS_4199
from torch.utils.data import DataLoader
import os
import time
from torch import nn
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import cv2
from PIL import Image

if __name__ == '__main__':  
    method_names = ['ASRNet_YCbCr_final']
    print("一共" + str(len(method_names)) + "种对比算法")
    dataset_name = ["ORSSD", "EORSSD", "ORS_4199"]
    for method_name in method_names:
        for dataseti in dataset_name:

            root = r"D:/optical_RSIs_SOD/" + dataseti + " dataset"
            smap_path = "F:/服务器代码/SRAL/data/predict_smaps_" + method_name + "_" + dataseti + "/"
            prefixes = [line.strip() for line in open(os.path.join(root, 'test.txt'))]

            if dataseti == "ORSSD":
                test_set = ORSSD(root = root, mode = "test", aug = False)
            elif dataseti == "EORSSD":
                test_set = EORSSD(root = root, mode = "test", aug = False)
            elif dataseti == "ORS_4199":
                test_set = ORS_4199(root = root, mode = "test", aug = False)

            test_loader = DataLoader(test_set, batch_size = 1, num_workers = 1)  

            thread = Eval_thread(smap_path=smap_path, loader = test_loader, method = method_name, dataset = dataseti, output_dir = "F:/服务器代码/HFANet_test/data/", cuda=False)#backbone_ablation/
            logg, fm = thread.run()
            print(logg)

