#! /usr/bin/env python
import rospy
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64, UInt8, Bool
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from math import radians
import numpy as np
import copy
import time
import math
import os, sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

class Env(object):
    def __init__(self):
        rospy.init_node('gazebo_env_node')
        rate = rospy.Rate(1)
        self.nagents=3
        initPos=[[-0.030240,-3.015927],[0.172365,-2.434824],[0.142683,-1.522063]]
        lane_flag=[1,0,0]
        self.eneities=[Entity('/robot'+str(i+5),initPos[i],lane_flag[i]) for i in range(self.nagents)]
        self.lastPos=[]
        self.rw_scale=20

    def reset(self):
        print('enter reset')
        reset_simulation = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        reset_simulation()
        time.sleep(0.5)
        obs = [self.eneities[i].getObs() for i in range(self.nagents)]
        self.lastPos=[]
        for entity in self.eneities:
            self.lastPos.append(entity.getPos())
            entity.reset()
        return np.array([obs])

    def step(self,actions):
        actions=actions[0]
        actions=[np.argmax(action) for action in actions]
        for entity, action in zip(self.eneities,actions):
            entity.step(action)
        obs = []
        rewards=[]
        done_flag=0
        # calculate reward
        for i, entity in zip(range(self.nagents),self.eneities):
            ob=entity.getObs()
            scan_msg=entity.scan_data
            #print(scan_data)
            front_data= [scan_msg[i] for i in range(-30,30)]
            front_dist = np.min(front_data)
            #print(entity.name, 'front_dist: ', front_dist)
            if front_dist < 0.2:
                reward=-100
                done_flag=1
            else:
                lastP = self.lastPos[i]
                cur_pos = entity.pos
                reward = self.rw_scale * math.sqrt((lastP[0] - cur_pos[0]) ** 2 + (lastP[1] - cur_pos[1]) ** 2)
                self.lastPos[i]=cur_pos
            rewards.append(reward)
            obs.append(ob)
        dones=np.full((1, self.nagents), done_flag)
        return np.array([obs]),np.array([rewards]),dones

class Entity(object):

    def __init__(self,name,pos,lane_flag):
        self.counter = 1
        self.sub_scan = rospy.Subscriber(name + '/scan', LaserScan, self.scanCallback, queue_size=1)
        self.sub_odom = rospy.Subscriber(name + '/odom', Odometry, self.getOdometry)
        self.sub_speed = rospy.Subscriber(name + '/cmd_vel', Twist, self.speedCallBack, queue_size=1)
        self.pub_lane_behavior = rospy.Publisher(name + '/lane_behavior', UInt8, queue_size=1)
        self.pub_reSet = rospy.Publisher(name + '/reset', Bool, queue_size=1)
        self.scan_data = [3.5]*360
        self.name=name
        self.speed_x=0.06
        self.pos=pos
        self.init_lane_flag=lane_flag
        self.lane_flag=self.init_lane_flag

    def step(self,action):
        behavior_msg = UInt8()
        behavior_msg.data = np.uint8(action)
        self.pub_lane_behavior.publish(behavior_msg)
        if action == 3:
            self.lane_flag=1

    def reset(self):
        self.pub_reSet.publish(Bool(data=True))

    def getObs(self):
        obs=copy.deepcopy(self.scan_data)
        print('')
        obs.append(self.speed_x)
        obs.append(self.pos[0])
        obs.append(self.pos[1])
        obs.append(self.lane_flag)
        return np.array(obs)

    def scanCallback(self,data):
        if self.counter % 3 != 0:
            self.counter += 1
            return
        else:
            self.counter = 1
        #print('enter scanCallback')
        scan = data
        scan_range = []
        # print('scan_data_lenth: ',len(scan.ranges))
        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])
        self.scan_data=scan_range

    def speedCallBack(self, msg):
        self.speed_x = msg.linear.x

    def getOdometry(self, odom):
        self.pos = [odom.pose.pose.position.x,odom.pose.pose.position.y]

    def getPos(self):
        return self.pos