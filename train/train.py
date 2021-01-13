#!/usr/bin/python3
#coding=utf-8

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import sys
import datetime
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import dataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from net  import DSGGN


def print_network(model, name):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(name)
    print("The number of parameters: {}".format(num_params))

def iou_loss(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou  = 1-(inter+1)/(union-inter+1)
    return iou.mean()


def train(Dataset, Network):
    ## dataset
    cfg    = Dataset.Config(datapath='/home/gaosy/DATA/DUTS/DUTS-TR', savepath='./out', mode='train', batch=16, lr=0.05, momen=0.9, decay=5e-4, epoch=45)
    data   = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, pin_memory=True, num_workers=8)  #
    ## network
    net    = Network(cfg)
    #net = nn.DataParallel(net)
    net.train(True)
    net.cuda()
    ## parameter
    base, head = [], []

    for name, param in net.named_parameters():
        if 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer      = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    sw             = SummaryWriter(cfg.savepath)
    global_step    = 0

    print_network(net, 'DSGGN')

    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.1
        optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr

        for step, (image, mask, edge) in enumerate(loader):
            image, mask, edge = image.type(torch.FloatTensor).cuda(), mask.type(torch.FloatTensor).cuda(), edge.type(torch.FloatTensor).cuda()

            outb1, outd1, out1 = net(image)
            
            lossb1 = F.binary_cross_entropy_with_logits(outb1, mask)
            lossd1 = F.binary_cross_entropy_with_logits(outd1, edge)
            loss1  = F.binary_cross_entropy_with_logits(out1, mask) + iou_loss(out1, mask)

            loss   = lossb1/2 + lossd1/2 + loss1/2

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            global_step += 1
            sw.add_scalar('lr'   , optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', {'lossb1':lossb1.item(), 'lossd1':lossd1.item(), 'loss1':loss1.item()}, global_step=global_step)
            if step%10 == 0:
                print('%s | step:%d/%d/%d | lr=%.6f | lossb1=%.6f | lossd1=%.6f | loss1=%.6f '
                    %(datetime.datetime.now(),  global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], lossb1.item(), lossd1.item(), loss1.item()))

        if epoch > cfg.epoch*1/2:
            torch.save(net.state_dict(), cfg.savepath+'/model-'+str(epoch+1))


if __name__=='__main__':
    train(dataset, DSGGN)