#!/usr/bin/python3
#coding=utf-8

import os
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import torch
import dataset
from torch.utils.data import DataLoader
from net import DSGGN  # todo 注意使用的是哪个模型


class Test(object):
    def __init__(self, Dataset, Network, Path):
        ## dataset
        self.cfg    = Dataset.Config(datapath=Path, snapshot='./out2/model-41', mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net    = Network(self.cfg)
        self.net.cuda()
        self.net.load_state_dict(torch.load(self.cfg.snapshot))
        self.net.train(False)


    def save(self):
        with torch.no_grad():
            for image, (H, W), name in self.loader:
                image, shape  = image.cuda().float(), (H, W)
                outb2, outd2, out2 = self.net(image, shape)
                out  = out2
                pred = torch.sigmoid(out[0,0]).cpu().numpy()*255
                pred = cv2.resize(pred, dsize=(W,H), interpolation=cv2.INTER_LINEAR)
                head = '../eval/maps/Real_TimeSOD/'+ self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head+'/'+name[0]+'.png', np.round(pred))



if __name__=='__main__':
    for path in ['/home/gaosy/DATA/GT/ECSSD', '/home/gaosy/DATA/GT/PASCAL_S', '/home/gaosy/DATA/GT/DUTS_test', '/home/gaosy/DATA/GT/HKU_IS', '/home/gaosy/DATA/GT/DUT_O']: #
        t = Test(dataset, DSGGN, path)
        t.save()
