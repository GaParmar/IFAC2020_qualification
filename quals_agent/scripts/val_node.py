#!/usr/bin/env python3
import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan

import os, sys, time, json, pdb
import numpy as np
from controllers.physical_controller import *

import torch
import torch.nn as nn
from collections import OrderedDict

p = "/home/gparmar/Desktop/robotics/github_gaparmar/F1tenth_gparmar/training"
if p not in sys.path: sys.path.append(p)
from dataset import F110Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

class LinearPolicy(nn.Module):
    def __init__(self):
        super(LinearPolicy, self).__init__()
        self.fc_model = nn.Sequential(OrderedDict([
            ('fc_1', nn.Linear(in_features=1080, out_features=256, bias=True)),
            # ('fc_1_bn', nn.BatchNorm1d(num_features=256)),
            ('LeakyRelu_1', nn.LeakyReLU(negative_slope=0.2, inplace=True)),

            ('fc_2', nn.Linear(in_features=256, out_features=256, bias=True)),
            # ('fc_2_bn', nn.BatchNorm1d(num_features=256)),
            ('LeakyRelu_2', nn.LeakyReLU(negative_slope=0.2, inplace=True)),

            ('fc_3', nn.Linear(in_features=256, out_features=2, bias=True))
        ]))

    def forward(self, x):
        y = self.fc_model(x)
        angle, throttle = y[:,0], y[:,1]
        return angle, throttle

class ValidationAgent(object):
    def __init__(self):
        roots= [ "/home/gparmar/Desktop/robotics/github_gaparmar/F1tenth_gparmar/data_train/berlin_0"]
        ds = F110Dataset(roots)
        print(len(ds))
        dl = DataLoader(ds, batch_size=1,
                    shuffle=True, pin_memory=True, 
                    num_workers=4)
        
        self.policy = LinearPolicy().cuda()
        sd = "/home/gparmar/Desktop/robotics/github_gaparmar/F1tenth_gparmar/training/saved_models/model_350.sd"
        self.policy.load_state_dict(torch.load(sd))
        self.policy.train()
        train_loss = 0.0
        for idx, batch in enumerate(dl, 1):
            bs = 249
            for b in range(1, bs+1):
                x = batch[0,b:b+1,0:1080].cuda().float()
                x = torch.tensor(x, requires_grad=True).float().cuda()
                gt_angle = batch[0,b:b+1,1081].cuda().float()
                gt_throttle = (batch[0,b:b+1,1082].cuda().float()-1.5)/2.0 # revert the transform applied
                pred_angle, pred_throttle = self.policy(x)
                mse_loss = F.mse_loss(pred_throttle.view(-1), gt_throttle)
                mse_loss += F.mse_loss(pred_angle.view(-1),gt_angle)
                train_loss += mse_loss.item()
        print("total loss ", train_loss)

if __name__ == '__main__':
    rospy.init_node('validation_agent')
    dummy_agent = ValidationAgent()
    rospy.spin()