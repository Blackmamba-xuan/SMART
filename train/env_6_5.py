import random

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
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
import os, sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

class Env(object):
    def __init__(self):
        rospy.init_node('gazebo_env_node')
        rate = rospy.Rate(1)
        self.nagents=3
        #initPos=[[-0.030240,-3.015927],[0.172365,-2.434824],[0.142683,-1.522063]]
        #initPos=[[-0.030240,-3.015927],[0.172365,-2.434824],[-0.030240,-3.015927],]
        robot_pos1=[[-0.574424,0.633950]]
        robot_pos2=[[-0.805519,0.284848],[-0.796213,0.177235]]
        robot_pos3=[[-0.559980,0.018121],[-0.553760,-0.099075],[-0.557075,-0.248758]]
        self.initPos=[robot_pos1,robot_pos2,robot_pos3]
        lane_flag=[0,1,0]
        self.eneities=[Entity('/robot'+str(i+5),self.initPos[i][0],lane_flag[i]) for i in range(self.nagents)]
        self.lastPos=[]
        self.rw_scale=20

    def reset(self):
        print('enter reset')
        reset_simulation = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        reset_simulation()
        idx1=np.random.randint(2)
        robot6_initPos=self.initPos[1][idx1]
        print(idx1)
        idx2=np.random.randint(3)
        robot7_initPos=self.initPos[2][idx2]
        print(idx2)
        print('enter set model service')
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            state_msg = ModelState()
            state_msg.model_name = 'robot6'
            state_msg.pose.position.x = robot6_initPos[0]
            state_msg.pose.position.y = robot6_initPos[1]
            state_msg.pose.position.z = -0.001003
            state_msg.pose.orientation.x=0.002640941576348912
            state_msg.pose.orientation.y=0.002830898435420755
            state_msg.pose.orientation.z=-0.6842180448216592
            state_msg.pose.orientation.w=0.7292672202848998

            resp = set_state(state_msg)
            state_msg2 = ModelState()
            state_msg2.model_name = 'robot7'
            state_msg2.pose.position.x = robot7_initPos[0]
            state_msg2.pose.position.y = robot7_initPos[1]
            state_msg2.pose.position.z = -0.001003
            state_msg2.pose.orientation.x=0.00264318220817986
            state_msg2.pose.orientation.y=0.002815092351699225
            state_msg2.pose.orientation.z=-0.6858120802965438
            state_msg2.pose.orientation.w=0.7277684242684571
            resp2 = set_state(state_msg2)
        except:
            print("set model state Service call failed: %s")

        time.sleep(0.2)
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
            print('agent ', i, ' action: ', actions[i])
            ob=entity.getObs()
            scan_msg=entity.scan_data
            #print(scan_data)
            front_data= [scan_msg[i] for i in range(-30,30)]
            front_dist = np.min(front_data)
            #print(entity.name, 'front_dist: ', front_dist)
            if front_dist < 0.2:
                reward=-100
                done_flag=1
            elif entity.lane_flag==1 and actions[i]==3:
                reward=-2
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
        self.pub_reSet = rospy.Publisher(name + '/reset', Float64, queue_size=1)
        self.sub_lane_flag = rospy.Subscriber(name+'detect/lane_flag', UInt8, self.setLaneFlag, queue_size=1)
        self.scan_data = [3.5]*360
        self.name=name
        self.speed_x=0.06
        self.pos=pos
        self.lidar_frames=[[3.5]*360for i in range(3)]
        self.init_lane_flag=lane_flag
        self.lane_flag=self.init_lane_flag

    def setLaneFlag(self, lane_flag_msg):
        self.lane_flag=lane_flag_msg.data

    def step(self,action):
        behavior_msg = UInt8()
        behavior_msg.data = np.uint8(action)
        self.pub_lane_behavior.publish(behavior_msg)
        if action == 3:
            self.lane_flag=0

    def reset(self):
        speed_x=random.uniform(0.8,1.3)
        msg=Float64()
        msg.data=speed_x
        self.pub_reSet.publish(msg)
        self.speed_x=speed_x

    def getObs(self):
        obs=copy.deepcopy(self.scan_data)
        print('')
        obs = copy.deepcopy(self.lidar_frames)
        obs=np.array(obs).reshape(-1)
        obs=np.append(obs,self.speed_x)
        # obs.append(self.pos[0])
        # obs.append(self.pos[1])
        obs=np.append(obs,self.lane_flag)
        return obs

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
            elif scan.ranges[i] == float('inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])
        self.scan_data=scan_range
        self.lidar_frames.pop(0)
        self.lidar_frames.append(self.scan_data)



    def speedCallBack(self, msg):
        self.speed_x = msg.linear.x

    def getOdometry(self, odom):
        self.pos = [odom.pose.pose.position.x,odom.pose.pose.position.y]

    def getPos(self):
        return self.pos