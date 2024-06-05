#!/usr/bin/python3
#coding=utf-8
import sys
sys.path.insert(0, '../')
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim


os.environ['CUDA_VISIBLE_DEVICES'] = '0'



    



from misc import AvgMeter, check_mkdir
from ORSI_SOD_dataset_YCbCr_final import  ORSI_SOD_dataset  
from src.baseline123_YCbCr import net as Net  
from src.TFGM import AuxiliaryLoss
from evaluator_SR_YCbCr import Eval_thread


loss_enhance = AuxiliaryLoss()


args = {
    'iter_num': 7500,
    'epoch': 100,
    'train_batch_size': 2,
    'last_iter': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'snapshot': ''
}

torch.manual_seed(2021)

### saliency loss function
"""
  smaps : BCE + wIOU
  edges: BCE
"""
def structure_loss(pred, mask):
    #mask = mask.detach()
    wbce  = F.binary_cross_entropy_with_logits(pred, mask)
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return wbce.mean()+wiou.mean()#


#define dataset and dataloader
dataset = "ORS_4199"  #or "ORSSD"  or "EORSSD"
input_size = 224
train_set = ORSI_SOD_dataset(root = '/data/iopen/lyf/SaliencyOD_in_RSIs/'+ dataset +' dataset/', size = input_size, mode = "train", aug = True)
train_loader = DataLoader(train_set, batch_size = args['train_batch_size'], shuffle = True, num_workers = args['train_batch_size'])
test_set = ORSI_SOD_dataset(root = '/data/iopen/lyf/SaliencyOD_in_RSIs/'+ dataset +' dataset/', size = input_size, mode = "test", aug = False)
test_loader =  DataLoader(test_set, batch_size = 1, shuffle = False, num_workers = 1)
args['iter_num'] = args["epoch"] * len(train_loader)  





def main():
   
    model = Net()
    net = model.cuda().train()
    
    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr':  args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])
    
    train(net, optimizer)



def train(net, optimizer):
    curr_iter = args['last_iter']

    for epoch in range(0, args['epoch']): # total 100 epoches
        total_loss_record, t1_record, t2_record, t3_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        net.train() 
        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num'] #
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']# 
                                                            ) ** args['lr_decay']
            
            image_lr_rgb, img_YCbCr_lr, img_YCbCr_sr,  label, name = data
            label = label.cuda()
            img_YCbCr_sr = img_YCbCr_sr.cuda()
            input1 = image_lr_rgb.cuda() 
            input2 = img_YCbCr_lr.cuda()
            optimizer.zero_grad()
            smaps, pred_sr, sod_fea, sr_fea = net(input1, input2)
            smap1, smap2, smap3, smap4, smap5 = smaps
            
            ########## compute loss #############
            loss1_1 = structure_loss(smap1, label)
            loss1_2 = structure_loss(smap2, label)
            loss1_3 = structure_loss(smap3, label)
            loss1_4 = structure_loss(smap4, label)
            loss1_5 = structure_loss(smap5, label)

            t1 = loss1_1 + loss1_2 + (loss1_3 / 2) + (loss1_4 / 4) + (loss1_5 / 8)

            t2 = nn.MSELoss()(pred_sr, img_YCbCr_sr) ##input YCbCr，supervise YCbcr

            t3 = loss_enhance(sod_fea*sr_fea, img_YCbCr_sr * torch.cat((label, label, label), dim=1))
            
            if epoch == 0:
                # warm up
                total_loss = t1 + t2 + t3
            else:
                total_loss = t1 + 100*t2 + t3
            
            total_loss.backward()

            optimizer.step()
            t1_record.update(t1.item(), args['train_batch_size'])
            t2_record.update(t2.item(), args['train_batch_size'])
            t3_record.update(t3.item(), args['train_batch_size'])
            total_loss_record.update(total_loss.item(), args['train_batch_size'])


            #############log###############
            if curr_iter % 125 == 0:
                log = '[epoch: %03d] [iter: %05d] [total loss %.5f] [loss1 %.8f] [loss2 %.8f] [loss3 %.8f] [lr %.13f] ' % \
                    (epoch, curr_iter, total_loss_record.avg,  t1_record.avg,  t2_record.avg, t3_record.avg, optimizer.param_groups[1]['lr'])
                print(log)
            
            curr_iter += 1



        if epoch % 10 == 0 or (epoch >= 60 and epoch %2 ==0):
            thread = Eval_thread(epoch = epoch, model = net.eval(), loader = test_loader, method = "baseline123_YCbCr_final_", dataset = dataset, output_dir = "./data/output", cuda=True)
            logg, fm = thread.run()
            print(logg)
            torch.save(net.state_dict(), './data/model_224*224_bs=8_baseline123_YCbCr_final_'+ dataset +'/epoch_{}_{}.pth'.format(epoch,fm))
           
        
    ##      #############end###############


if __name__ == '__main__':
    main()


