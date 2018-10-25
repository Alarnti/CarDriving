#!/usr/bin/env python
# -*- coding: utf-8 -*-

import roslib
# roslib.load_manifest('my_package')
import sys
import rospy
import cv2
import math
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from std_msgs.msg import Int16, UInt8

from gradients import get_edges

import tensorflow as tf

from collections import deque
from keras.models import Sequential, load_model
MODEL_PATH = 'actionValue.model'

np.set_printoptions(precision=2, suppress=True)

rospy.init_node('image_converter_foobar')

FRAMES = 4

COUNT = 0

from typing import Tuple

from keras.layers import Activation, MaxPooling2D, Dropout, Convolution2D, Flatten, Dense
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras import regularizers
from keras.layers import merge, Input
from keras import backend as K
from keras.optimizers import Adam

class LaneTracker():
    def __init__(self):
        self.frames = deque(maxlen=FRAMES)

        #y = 30
        self.main_hor_line = (0, 1, -30)
    def add_frame():
        return 0

    def intersection_of_lines(self, line_one, line_two):
        # 0, 1, 2 = A, B, C
        
        d = (line_one[0] * line_two[1] - line_two[0] * line_one[1]) + 1e-6

        x = - (line_one[2] * line_two[1] - line_two[2] * line_one[1])/ \
            d
        y = - (line_one[0] * line_two[2] - line_two[0] * line_one[2])/ \
            d
        return y, x

    def process_frame(self, frame):
        # print(frame.shape)
        # frame_ = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny((255*frame).astype(np.uint8),100,200)
        edges[:30,:] = 0
        edges[75:,:] = 0
        edges[60:75, 30:70] = 0

        a = self.coord_line(edges)
        #print(a)

        l  = self.intersection_of_lines(self.main_hor_line, a[0])
        r  = self.intersection_of_lines(self.main_hor_line, a[1])
        #print(l, r)

        im_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        img_out = cv2.line(im_bgr, (0,30), (100,30), (0,0,255),2)
        #line_im = cv2.line(im,(a[0][1], a[0][0]),(a[1][1], a[1][0]),(255))

        #middle - red
        cv2.circle(img_out,(50,30), 2, (255,0,0), -1)

        #right - green
        cv2.circle(img_out,(int(r[1]),int(r[0])), 5, (0,255,0), -1)

        #left - yellow
        # cv2.circle(img_out,(int(l[1]),int(l[0])), 2, (255,255,0), -1)

    
        return img_out

    def process_frame_(self, frame):
        edges = cv2.Canny(frame,100,200)
        edges[:30,:] = 0
        edges[75:,:] = 0
        edges[60:75, 30:70] = 0

        return self.coord_line(edges)


    def coord_line(self, edges):
        points_right = []
        points_left = []
        
        filt = np.array([[1,1,1],[1,0,1],[1,1,1]])
        
        for i in range(int(len(edges)/3.5),len(edges), 5):
            
            found_left = False
            found_right = False
            
            for j in range(1, len(edges[0]) - 1):           
                if edges[i,j] > 0:
                    # print(edges[i-1:i+2,j-1:j+2])
                    if np.sum(edges[i-1:i+2,j-1:j+2] * filt) > 0:
                        if j < len(edges[0])/2 and not found_left:
                            points_left.append([i,j])
                            found_left = True
                        # elif j > len(edges[0])/2 and not found_right:
                        #     points_right.append([i,j])
                        #     found_right = True
                    if found_left:# and found_right:
                        break

            for j in range(len(edges[0]) - 2, 1, -1):           
                if edges[i,j] > 0:
                    # print(edges[i-1:i+2,j-1:j+2])
                    if np.sum(edges[i-1:i+2,j-1:j+2] * filt) > 0:

                        # if j < len(edges[0])/2 and not found_left:
                        #     points_left.append([i,j])
                        #     found_left = True
                        if j > len(edges[0])/2 and not found_right:
                            points_right.append([i,j])
                            found_right = True
                    if found_right:
                        break
                    
        found_right_line = False
        found_left_line = False

        A_right = 0
        B_right = 0
        C_right = 0

        A_left = 0
        B_left = 0
        C_left = 0

        if len(points_right) > 1:
            found_right = True
            A_right = points_right[0][0] - points_right[-1][0]
            B_right = points_right[-1][1] - points_right[0][1]
            C_right = points_right[0][1] * points_right[-1][0] - \
            points_right[-1][1] * points_right[0][0]
        
        if len(points_left) > 1:
            found_left_line = True
            A_left = points_left[0][0] - points_left[-1][0]
            B_left = points_left[-1][1] - points_left[0][1]
            C_left = points_left[0][1] * points_left[-1][0] - \
            points_left[-1][1] * points_left[0][0]

        return (A_left, B_left, C_left),(A_right, B_right, C_right)#points_right[0], points_right[1]

tracker = LaneTracker()



#TUple changed
def create_atari_model(input_shape, output_units):
#def create_atari_model(input_shape: Tuple[ int, int, int], output_units: int) -> Model:
    model = Sequential()
    # model.add(Convolution2D(16, 5, 5, activation='relu', border_mode='same',
    #                         input_shape=input_shape, subsample=(3, 3), init ='glorot_uniform', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)))
    # model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', subsample=(1, 1), init='glorot_uniform', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)))
    # model.add(Flatten())
    # model.add(BatchNormalization())
    # model.add(Dense(256, activation='relu', init='glorot_uniform', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
    # model.add(Dense(output_units, activation='linear'))
    # model.compile(optimizer=optimizers.Nadam(), loss='mean_squared_error', metrics=['mean_squared_error'])

    # model.add(Convolution2D(16, 8, 8, activation='relu', border_mode='same',
    #                           input_shape=input_shape, subsample=(4, 4)))
    # model.add(Convolution2D(32, 4, 4, activation='relu', border_mode='same', subsample=(2, 2)))
    # model.add(Flatten())
    # model.add(Dense(256, input_shape=(2,), activation='relu'))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(output_units, activation='linear'))
    # model.compile(optimizer='RMSprop', loss='logcosh', metrics=['mean_squared_error'])

    # NUM_ACTIONS = output_units
    # input_layer = Input(shape = input_shape)
    # conv1 = Convolution2D(16, 5, 5, subsample=(2, 2), activation='relu')(input_layer)
    # pool = MaxPooling2D(pool_size=(2, 2), padding='valid')(conv1)
    # conv2 = Convolution2D(32, 3, 3, subsample=(1, 1), activation='relu')(pool)
    # #conv3 = Convolution2D(64, 3, 3, activation = 'relu')(conv2)
    # flatten = Flatten()(conv2)
    # fc1 = Dense(128)(flatten)
    # advantage = Dense(NUM_ACTIONS)(fc1)
    # fc2 = Dense(64)(flatten)
    # value = Dense(1)(fc2)
    # policy = merge([advantage, value], mode = lambda x: x[0]-K.mean(x[0])+x[1], output_shape = (NUM_ACTIONS,))

    # model = Model(input=[input_layer], output=[policy])
    # model.compile(optimizer=Adam(lr=0.0001), loss='logcosh', metrics=['mean_squared_error'])
    # return model



    # NUM_ACTIONS = output_units
    # input_layer = Input(shape = input_shape)
    # conv1 = Convolution2D(8, 11, 11, subsample=(2, 2), activation='relu')(input_layer)
    # #pool = MaxPooling2D(pool_size=(2, 2), padding='valid')(conv1)
    # #conv2 = Convolution2D(16, 3, 3, subsample=(1, 1), activation='relu')(conv1)
    # #conv3 = Convolution2D(64, 3, 3, activation = 'relu')(conv2)
    # flatten = Flatten()(conv1)
    # fc1 = Dense(64, activation='relu')(flatten)
    # advantage = Dense(NUM_ACTIONS)(fc1)
    # fc2 = Dense(32, activation='relu')(flatten)
    # value = Dense(1)(fc2)
    # policy = merge([advantage, value], mode = lambda x: x[0]-K.mean(x[0])+x[1], output_shape = (NUM_ACTIONS,))

    # model = Model(input=[input_layer], output=[policy])
    # model.compile(optimizer=Adam(lr=0.0001), loss='logcosh', metrics=['mean_squared_error'])
    # return model

    NUM_ACTIONS = output_units
    input_layer = Input(shape = input_shape)
    conv1 = Convolution2D(8, 5, 5, subsample=(2, 2), activation='relu')(input_layer)
    conv2 = Convolution2D(16, 3, 3, subsample=(2, 2), activation='relu')(conv1)
    pool = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)
    # conv3 = Convolution2D(32, 3, 3, activation = 'relu')(pool)
    flatten = Flatten()(pool)
    fc1 = Dense(128, activation='relu')(flatten)
    advantage = Dense(NUM_ACTIONS)(fc1)
    fc2 = Dense(64, activation='relu')(flatten)
    value = Dense(1)(fc2)
    policy = merge([advantage, value], mode = lambda x: x[0]-K.mean(x[0])+x[1], output_shape = (NUM_ACTIONS,))

    model = Model(input=[input_layer], output=[policy])
    model.compile(optimizer=Adam(lr=0.0001), loss='mse', metrics=['mean_squared_error'])
    return model

def softmax(x):

   # print x - np.max(x)
    e_x = np.exp(x - np.max(x))
    return e_x/e_x.sum()



class Filter:

    def __init__(self, length):
        self.predictionsMaxLength = length
        self.lastPredictionsArray = np.zeros(self.predictionsMaxLength)
        self.currentPosition = 0
        self.firstlyFilled = False

    def add_prediction(self, prediction):
        if not self.firstlyFilled and self.currentPosition + 1 < self.predictionsMaxLength:
            self.firstlyFilled = True

        self.currentPosition = (self.currentPosition + 1) % self.predictionsMaxLength
        self.lastPredictionsArray[self.currentPosition] = prediction
    def get_filtered_prediction(self):
        if self.firstlyFilled:
            print self.lastPredictionsArray.astype(int)
            return np.argmax(np.bincount(self.lastPredictionsArray.astype(int)))
        return 1


class image_converter:

  def __init__(self):

    self.pubSpeed = rospy.Publisher('/manual_control/speed', Int16, queue_size=1, latch=True)
    self.pubSteering = rospy.Publisher('/steering', UInt8, queue_size=1, latch=True)

    # self.model = load_model(MODEL_PATH)

    self.FRAME_COUNT = 4
    self.im_input_shape = (100,100)
    self.model = create_atari_model((self.im_input_shape[0], self.im_input_shape[1], self.FRAME_COUNT),3)
    self.model.load_weights(MODEL_PATH) 
    self.model._make_predict_function()
    self.graph = tf.get_default_graph()


    self.frames = []
    self.index = 0
    self.max_frames = self.FRAME_COUNT

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/app/camera/rgb/image_color/compressed",CompressedImage,self.callback, queue_size=1)



    self.frame_nums = 0
    self.cache = deque(maxlen=self.FRAME_COUNT) 

    self.frame_count = 0

    self.filter = Filter(5)

    self.current_speed = 130
    self.current_steering = 90

    self.last_command_time = rospy.get_time()

    self.prev_command = 1



    self.mask_right = np.ones((480,640))
    # for i in range(0,len(self.mask_right)):
    #     for j in range(0,len(self.mask_right[0])):
    #         if i < 2.5 * j - 25:
    #             self.mask_right[i,j] = 0

  def softmax(self, x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)



  def add_frame(self, image):
    self.cache.append(image)
    while len(self.cache) < self.FRAME_COUNT:
        self.cache.append(image)
    
    # print len(self.cache)
    self.frames = np.stack(self.cache, axis=2)
    # print 'ads',self.frames.shape
    self.index = (self.index + 1) % 4
    self.frame_nums += 1

  def predict(self, image=None):
    with self.graph.as_default():

        x = np.expand_dims(self.frames, axis = 0)
        # print x.shape
        pred = self.model.predict(x)[0]

        softmax = self.softmax(pred)
        print pred
        print softmax
        # print softmax[np.argmax(softmax)]
        # if softmax[np.argmax(softmax)] > 0.30:
        return np.argmax(softmax)
        # else:
        #     return None
        # return np.argmax(pred) 



  def callback(self,data):
    try:
      frame = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
      pass
    except CvBridgeError as e:
      print(e)

    # print('AAAAAAAAA')

    self.frame_count += 1

    # frame = cv2.flip(frame, 1)

    frame = cv2.resize(frame, self.im_input_shape)

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



    lower = np.array([0, 175, 0])
    upper = np.array([180,255,255])

    # resized = cv2.resize(frame, self.im_input_shape)

    frame[:int(len(frame)/3),:] = 0

    # frame[int(len(frame)/1.8):, :int(len(frame[0])/5)] = 0

    # frame[:int(len(frame)*3/4), int(len(frame[0])*3/4):] = 0

    mask = cv2.inRange(frame_hsv, lower, upper)

    res = cv2.bitwise_and(frame,frame, mask= mask)

    # res = cv2.bitwise_and(res,res, mask=self.mask_right.astype(np.uint8))


    # kernel = np.ones((7,7))
    # res = cv2.dilate(res,kernel,5)


    resized = res#<cv2.resize(res, self.im_input_shape)


    ret, resized = cv2.threshold(resized,150,255,cv2.THRESH_BINARY)
    # resized[:int(len(resized)/),:] = 0

    edges = get_edges(resized)

    resized = edges.astype(np.float32) #resized.astype(np.float32)

    # resized = np.clip(resized + np.mean(resized), 0, 1)

    # print(np.mean(np.clip(resized + np.mean(resized), 0, 1)))
    resized = (resized) * 255 #/ 255

    # print(np.max(resized))


    self.add_frame(resized)



    # self.filter.add_prediction(command)

    # filtered_command = self.filter.get_filtered_prediction()

    # command = filtered_command

    command = None

    current_time = rospy.get_time()
    if current_time - self.last_command_time > 0.3:
        command = self.predict()

        if command is None:
            command = self.prev_command
            # print(current_time - self.last_command_time)
        self.last_command_time = current_time
        self.current_steering = 90 *(2 -command)#(2 - command)#(self.filter.get_filtered_prediction())# self.filter.get_filtered_prediction())

        # if self.prev_command != self.current_steering:
        #     steering_now = 90*(2 - self.prev_command) 
        print self.current_steering
        # print(current_time)

    self.pubSpeed.publish(self.current_speed)
    self.pubSteering.publish(self.current_steering)


    if command is None:
        command = self.prev_command

    self.prev_command = command

    #global COUNT
    #cv2.imwrite('real_ims/' + str(COUNT) + '.png', res)
    #COUNT += 1
    cv2.imshow('edges',255 *edges)
    # cv2.imshow('resized',cv2.putText(cv2.resize(resized, (frame.shape[1], frame.shape[0])), str(command), (10, len(res[0])//2), cv2.FONT_HERSHEY_SIMPLEX, 3, 255))
    # cv2.imshow('frame',cv2.putText(frame, str(command), (10, len(res[0])//2), cv2.FONT_HERSHEY_SIMPLEX, 3, 255))
    # cv2.imshow('hist',hist_im)
    cv2.waitKey(1)


ic = image_converter()
try:
    rospy.spin()
except KeyboardInterrupt:
    print("Shutting down")
cv2.destroyAllWindows()

#600/400 -> 35.3/23.5
# 45/ 18/ 17 cm
# 2.65/1.05/1

#Un 7/5/5