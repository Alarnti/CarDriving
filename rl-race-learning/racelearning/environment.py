import math
import random
import socket
from collections import deque
from struct import pack, unpack
import typing

import numpy as np

import cv2

def preprocess_image(camera_image: np.ndarray) -> np.ndarray:

    #print('camera image shape', camera_image.shape)

    frame = camera_image
    frame = cv2.flip(frame, 0)
    #frame = cv2.flip(frame, 1)

    # ignore pixels too far away
    #frame[int(len(frame)/1.5):,:] = 0

    lower = np.array([160])#np.array([0,0,150])
    upper = np.array([255])#np.array([255,255,255])
    # Threshold the HSV image to get only blue colors

    #test = cv2.GaussianBlur(rgb, (3, 3), 0)
    mask = cv2.inRange(frame, lower, upper)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    return res

class Action:

    COUNT = 9

    def __init__(self, vertical: int, horizontal: int):
        if vertical > 1 or vertical < -1:
            raise ValueError('Vertical must be -1, 0 or 1')
        if horizontal > 1 or horizontal < -1:
            raise ValueError('Horizontal must be -1, 0 or 1')
        self.vertical = vertical
        self.horizontal = horizontal

    @classmethod
    def from_code(cls, code: int):
        return cls(code // 3 - 1, code % 3 - 1)

    @classmethod
    def random(cls):
        return cls(random.randrange(3) - 1, random.randrange(3) - 1)

    def get_code(self) -> int:
        return (self.vertical + 1) * 3 + self.horizontal + 1


class LeftRightAction(Action):

    COUNT = 3

    def __init__(self, horizontal: int):
        super().__init__(1, horizontal)

    @classmethod
    def from_code(cls, code: int):
        return cls(code - 1)

    @classmethod
    def random(cls):
        return cls(random.randrange(3) - 1)

    def get_code(self) -> int:
        return self.horizontal + 1


class State:

    def __init__(self, data: np.ndarray, is_terminal: bool):

        
        # image = data[:,:,:-1]
        #cv2.imwrite('b.png', data[:,:,:-1])
        data.setflags(write=1)

        for i in range(0,4):
            image = data[:,:,i]
            preprocessed_image = preprocess_image(image)
            data[:,:,i] = preprocessed_image
        # cv2.imwrite('0.png', data[:,:,0])
        # cv2.imwrite('1.png', data[:,:,1])
        # cv2.imwrite('2.png', data[:,:,2])
        # cv2.imwrite('3.png', data[:,:,3])
        data.setflags(write=0)
        #cv2.imshow('b', data[:,:,:-1])
        #cv2.waitKey(0);
        #self.data =self.data.astype(np.float32) / 255
        #self.data = (self.data - np.mean(self.data)) / np.std(self.data, ddof=1)
        # Convert to 0 - 1 ranges
        self.data = (data.astype(np.float32) - 128) / 128
        # Z-normalize
        #self.data = (self.data - np.mean(self.data)) / np.std(self.data, ddof=1)
        # Add channel dimension
        if len(self.data.shape) < 3:
            self.data = np.expand_dims(self.data, axis=2)
        self.is_terminal = is_terminal

    def _preprocess_image(self, camera_image: np.ndarray) -> np.ndarray:

        #print('camera image shape', camera_image.shape)

        frame = camera_image
        #frame = cv2.flip(frame, 0)
        frame = cv2.flip(frame, 1)
        cv2.imshow('b',frame)
        cv2.waitKey(0)

        # ignore pixels too far away
        #frame[int(len(frame)/1.5):,:] = 0

        lower = np.array([160])#np.array([0,0,150])
        upper = np.array([255])#np.array([255,255,255])
        # Threshold the HSV image to get only blue colors

        #test = cv2.GaussianBlur(rgb, (3, 3), 0)
        mask = cv2.inRange(frame, lower, upper)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame,frame, mask= mask)

        cv2.imshow('a',res)
        cv2.waitKey(0)

        return res

class StateBare:
    def __init__(self, data, is_terminal):
        self.data = data
        self.is_terminal = is_terminal

def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


class StateAssembler:

    FRAME_COUNT = 4

    def __init__(self):
        self.cache = deque(maxlen=self.FRAME_COUNT)

    def assemble_next(self, camera_image: np.ndarray, is_terminal: bool) -> State:
        self.cache.append(camera_image)
        # If cache is still empty, put this image in there multiple times
        while len(self.cache) < self.FRAME_COUNT:
            self.cache.append(camera_image)
        images = np.stack(self.cache, axis=2)
        return State(images, is_terminal)


class EnvironmentInterface:

    REQUEST_READ_SENSORS = 1
    REQUEST_WRITE_ACTION = 2

    def __init__(self, host: str, port: int):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))
        self.assembler = StateAssembler()

    @staticmethod
    def _calc_reward(disqualified: bool, finished: bool, velocity: float) -> float:
        print(disqualified, finished, velocity)
        if disqualified:
            return -10
        return velocity

    def read_sensors(self, width: int, height: int) -> (State, int):
        request = pack('!bii', self.REQUEST_READ_SENSORS, width, height)
        self.socket.sendall(request)

        # Response size: disqualified, finished, velocity, camera_image
        response_size = width * height + 2 + 4
        response_buffer = bytes()
        while len(response_buffer) < response_size:
            response_buffer += self.socket.recv(response_size - len(response_buffer))

        disqualified, finished, velocity = unpack('!??i', response_buffer[:6])
        # Velocity is encoded as x * 2^64
        velocity /= 0xffffff
        camera_image = np.frombuffer(response_buffer[6:], dtype=np.uint8)
        camera_image = np.reshape(camera_image, (height, width), order='C')

        #print('cam SHAPE', camera_image.shape)

        reward = self._calc_reward(disqualified, finished, velocity)
        ##### PLACE
        camera_image.setflags(write=1)
        state = self.assembler.assemble_next(preprocess_image(camera_image), disqualified or finished)

        #cv2.imwrite('b.png', preprocess_image(camera_image))

        return state, reward

    def write_action(self, action: Action):
        request = pack('!bii', self.REQUEST_WRITE_ACTION, action.vertical, action.horizontal)
        self.socket.sendall(request)

    def _preprocess_image(self, camera_image: np.ndarray) -> np.ndarray:

        #print('camera image shape', camera_image.shape)

        frame = camera_image
        #frame = cv2.flip(frame, 0)
        frame = cv2.flip(frame, 1)

        # ignore pixels too far away
        #frame[int(len(frame)/1.5):,:] = 0

        lower = np.array([160])#np.array([0,0,150])
        upper = np.array([255])#np.array([255,255,255])
        # Threshold the HSV image to get only blue colors

        #test = cv2.GaussianBlur(rgb, (3, 3), 0)
        mask = cv2.inRange(frame, lower, upper)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame,frame, mask= mask)

        return res

        # print('camera image shape', camera_image.shape)

        # lower_blue = np.array([200])
        # upper_blue = np.array([255])

        # rows,cols = camera_image.shape
        # M = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
        # camera_image = cv2.warpAffine(camera_image,M,(cols,rows))

        # return camera_image

        # mask = cv2.inRange(camera_image, lower_blue, upper_blue)
        # res = cv2.bitwise_and(camera_image,camera_image, mask= mask)

        # kernel = np.ones((3,3),np.uint8)
        # dilation = cv2.dilate(res,kernel,iterations = 1)
        # dilation = cv2.GaussianBlur(dilation, (3, 3), 5)

        # edges = cv2.Canny(dilation,50,150)

        # lines = cv2.HoughLines(edges,1,np.pi/180,25)

        # sum_rho = 0
        # sum_theta = 0
        # if lines is not None:
        #     for rho,theta in lines[0]:
        #         sum_rho += rho
        #         sum_theta += theta
        # else:
        #     return np.array([0,0])

        # sum_rho /= len(lines[0])
        # sum_theta /= len(lines[0])

        # return np.array([sum_rho, sum_theta])