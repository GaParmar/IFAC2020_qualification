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
class LinearPolicyBN(nn.Module):
    def __init__(self, dims=[128, 64, 32]):
        super(LinearPolicyBN, self).__init__()
        self.fc_model = nn.Sequential(OrderedDict([
            ('fc_1', nn.Linear(in_features=1080, out_features=dims[0], bias=True)),
            ('fc_1_bn', nn.BatchNorm1d(num_features=dims[0])),
            ('LeakyRelu_1', nn.LeakyReLU(negative_slope=0.2, inplace=True)),

            ('fc_2', nn.Linear(in_features=dims[0], out_features=dims[1], bias=True)),
            ('fc_2_bn', nn.BatchNorm1d(num_features=dims[1])),
            ('LeakyRelu_2', nn.LeakyReLU(negative_slope=0.2, inplace=True)),

            ('fc_2b', nn.Linear(in_features=dims[1], out_features=dims[2], bias=True)),
            ('fc_2b_bn', nn.BatchNorm1d(num_features=dims[2])),
            ('LeakyRelu_2b', nn.LeakyReLU(negative_slope=0.2, inplace=True)),

            ('fc_3', nn.Linear(in_features=dims[2], out_features=2, bias=True))
        ]))

    def forward(self, x):
        y = self.fc_model(x)
        angle, throttle = y[:,0], y[:,1]
        return angle, throttle

class EvalAgent(object):
    def __init__(self):
        
        self.policy = LinearPolicyBN().cuda()
        sd = sys.argv[1]
        self.policy.load_state_dict(torch.load(sd))
        self.policy.eval().cpu()
        self.total = 0
        self.total_ts = 0.0
        """
        Initialize ROS stuff for interfacing with simulator
        """
        self.drive_pub = rospy.Publisher('/drive', AckermannDriveStamped, queue_size=1)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback, queue_size=1)

    def scan_callback(self, scan_msg):
        data = np.array(scan_msg.ranges).reshape(-1)
        s = time.time()
        with torch.no_grad():
            
            x = torch.tensor(data, requires_grad=True).float().cpu().view((1,-1))
            pred_angle, pred_throttle = self.policy(x)
            pred_throttle = (pred_throttle).cpu().detach().item()
            pred_angle = pred_angle.cpu().detach().item()
            if self.total<30:
                pred_throttle += 3.0
            print(pred_throttle, pred_angle)
        drive = AckermannDriveStamped()
        drive.drive.speed = pred_throttle*1.0
        drive.drive.steering_angle = pred_angle
        self.drive_pub.publish(drive)
        d = time.time()-s
        self.total+=1
        self.total_ts+=d


if __name__ == '__main__':
    rospy.init_node('eval_agent')
    dummy_agent = EvalAgent()
    rospy.spin()