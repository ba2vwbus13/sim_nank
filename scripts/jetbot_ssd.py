#!/usr/bin/env python3

#iiyama = python2
#jetbot = python3

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import rospy
from PIL import Image as PIL_Image
from conversion_utils import imgmsg_to_pil, pil_to_cv, closest_detection, detection_center
import cv2
import time

GST_STR = 'nvarguscamerasrc \
    ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=(fraction)10/1 \
    ! nvvidconv ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx \
    ! videoconvert \
    ! appsink'
WINDOW_NAME = 'Camera Test'

class Controller:

    def __init__(self):
        from jetbot import ObjectDetector
        following_model_path = "/home/jetbot/catkin_ws/src/sim_nank/weights/ssd_mobilenet_v2_coco.engine"
        self.detection_model = ObjectDetector(following_model_path)
        print("detection object make success")    
        self.cap = cv2.VideoCapture(GST_STR, cv2.CAP_GSTREAMER)
        self.target_label = 1
        self.display = True

    def start(self):
        while True:
            self._publish_loop()
            time.sleep(0.1)
     
    def _object_detection(self):
        ret, image = self.cap.read()
        if ret != True:
            return None
        image = cv2.resize(image, dsize=(300,300))
        (height, width, channel) = image.shape
        #detections = self.detection_model.predictor(image) # draw all detections on image
        detections = self.detection_model(image) # draw all detections on image
        for det in detections[0]:
            bbox = det['bbox']
            cv2.rectangle(image, (int(width * bbox[0]), int(height * bbox[1])), 
                          (int(width * bbox[2]), int(height * bbox[3])), (255, 0, 0), 2)
    
        # select detections that match selected class label
        matching_detections = [d for d in detections[0] if d['label'] == self.target_label]

        # get detection closest to center of field of view and draw it
        det = closest_detection(matching_detections)
        if det is not None:
            bbox = det['bbox']
            cv2.rectangle(image, (int(width * bbox[0]), int(height * bbox[1])), 
                          (int(width * bbox[2]), int(height * bbox[3])), (0, 255, 0), 5)

        if self.display:
            cv2.imshow("Image", image)
            cv2.waitKey(3)

        return det

    def _decide_motor_value(self):
        det = self._object_detection()

        if det is None:
            forward = 5.0
            angle = 0.0      
        # otherwsie steer towards target
        else:
            # move robot forward and steer proportional target's x-distance from center
            center = detection_center(det)
            forward = 5.0
            angle = 5.0 * center[0]

        print("det:{}".format(det))

        return forward, angle

    def _publish_loop(self):
        forward, angle = self._decide_motor_value()
        print("linear_x: {}, angular_z: {}".format(forward, angle))

if __name__ == '__main__':
    oc = Controller()
    oc.start()
