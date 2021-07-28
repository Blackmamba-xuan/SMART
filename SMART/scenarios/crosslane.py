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
import os, sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

class Scenario(object):
    def __init__(self):
        print('creating crosslane')
        rospy.init_node('gazebo_env_crossLane')
        rate = rospy.Rate(1)
        self.nagents = 4
        initPos = [[2.078582, 0.112576], [-0.107081, 1.551970], [-1.785433, -0.116726], [0.107544, -0.116726]]
        self.eneities = [Entity('/robot' + str(i + 5), initPos[i]) for i in range(self.nagents)]
        self.lastPos = []
        self.rw_scale = 5

    def reset(self):
        print('enter reset')
        reset_simulation = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        reset_simulation()
        time.sleep(0.2)
        obs = [self.eneities[i].getObs() for i in range(self.nagents)]
        self.lastPos = []
        for entity in self.eneities:
            self.lastPos.append(entity.getPos())
            entity.reset()
        return np.array([obs])


    def step(self, actions):
        actions = actions[0]
        actions = [np.argmax(action) for action in actions]
        for entity, action in zip(self.eneities, actions):
            entity.step(action)
        obs = []
        rewards = []
        done_flag = 0
        # calculate reward
        for i, entity in zip(range(self.nagents), self.eneities):
            ob = entity.getObs()
            scan_msg = entity.scan_data
            # print(scan_data)
            front_data = [scan_msg[i] for i in range(-30, 30)]
            front_dist = np.min(front_data)
            # print(entity.name, 'front_dist: ', front_dist)
            if front_dist < 0.2:
                reward = -5
                done_flag = 1
            else:
                last_y = self.lastPos[i][1]
                cur_pos = entity.pos
                reward = self.rw_scale * (cur_pos[1] - last_y)
                self.lastPos[i] = cur_pos
            rewards.append(reward)
            obs.append(ob)
        dones = np.full((1, self.nagents), done_flag)
        return np.array([obs]), np.array([rewards]), dones

class Entity(object):

    def __init__(self,name,pos,lane_flag):
        self.counter = 1
        self.sub_scan = rospy.Subscriber(name + '/scan', LaserScan, self.scanCallback, queue_size=1)
        self.sub_odom = rospy.Subscriber(name + '/odom', Odometry, self.getOdometry)
        self.sub_speed = rospy.Subscriber(name + '/cmd_vel', Twist, self.speedCallBack, queue_size=1)
        self.pub_reSet = rospy.Publisher(name + '/reset', Bool, queue_size=1)
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.scan_data = [3.5]*360
        self.name=name
        self.speed_x=0.06
        self.vel_step = 0.01
        self.pos=pos

    def step(self,action):
        if action == 3:
            self.lane_flag=1

    def reset(self):
        self.pub_reSet.publish(Bool(data=True))

    def getObs(self):
        obs=copy.deepcopy(self.scan_data)
        #print('')
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
        print('enter scanCallback')
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