#!/usr/bin/env python3
import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan

import os, sys, time, json, pdb
import numpy as np
from controllers.physical_controller import *


class DataAgent(object):
    def __init__(self, mod="", override=True):
        """
        Initialize a new thread for controller
        """
        cont = XboxOneJoystickController()
        cont_thread = Thread(target=cont.update, args=())
        cont_thread.daemon = True
        cont_thread.start()
        self.cont = cont

        """
        Initialize ROS stuff for interfacing with simulator
        """
        self.drive_pub = rospy.Publisher('/drive', AckermannDriveStamped, queue_size=1)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback, queue_size=1)
        """
        Initialize files for logging
        """
        base = "/home/gparmar/Desktop/robotics/github_gaparmar/F1tenth_gparmar/data_train"
        self.output_folder = output_folder = os.path.join(base, mod)
        if override and os.path.exists(output_folder):
            os.system(f"rm -r {output_folder}")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        self.buffer = np.zeros((1,1080+3))#[lidar(1081), ctr, angle, throttle]
        self.ctr = 0
        self.start_ts = time.time()


    def scan_callback(self, scan_msg):
        angle, throttle, _, recording = self.cont.run_threaded(img_arr=None)
        # throttle convert [-1,+1] to [-5.0,5.0]
        throttle = throttle*-5.0
        angle *= -0.5
        # print('got scan, now plan')
        drive = AckermannDriveStamped()
        drive.drive.speed = throttle
        drive.drive.steering_angle = angle
        self.drive_pub.publish(drive)
        data = np.array(scan_msg.ranges).reshape(-1)
        if recording:
            curr = np.concatenate([data, [self.ctr, angle, throttle]]).reshape((1,-1))
            self.buffer = np.concatenate((self.buffer,curr), axis=0)
            self.ctr += 1
            if self.buffer.shape[0]==250:
                outf = os.path.join(self.output_folder, f"log_{self.ctr}")
                np.save(outf, self.buffer)
                self.buffer = np.zeros((1,1080+3))
        else:
            self.buffer = np.zeros((1,1080+3))
        print(self.buffer.shape, angle, throttle, recording)
        


if __name__ == '__main__':
    rospy.init_node('data_agent')
    dummy_agent = DataAgent(override=True, mod="berlin_quals_B_1c")
    rospy.spin()