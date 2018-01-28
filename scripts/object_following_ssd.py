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

class ObjectFollowingController:

    def __init__(self):
        rospy.init_node('object_following')
        rospy.Subscriber('/image', Image, self._image_callback)
        self._cmd_vel_pub = rospy.Publisher('/jetbot/cmd_vel', Twist, queue_size=1)
        self.device = torch.device('cuda')
        self.pil_image = None
        self.cv_image = None
        self.collision_model = None
        self.detection_model = None
        avoidance_model_path = rospy.get_param('~avoidance_model')
        following_model_path = rospy.get_param('~following_model_path')
        self.following_model = rospy.get_param('~following_model')
        if self.following_model == 'ssd':
            from ssd import ObjectDetector
        else:
            from jetbot import ObjectDetector
        self.detection_model = ObjectDetector(following_model_path)       
        collision_model = torchvision.models.resnet18(pretrained=False)
        collision_model.fc = torch.nn.Linear(512, 2)
        collision_model.load_state_dict(torch.load(avoidance_model_path))
        collision_model = collision_model.to(self.device)
        self.collision_model = collision_model.eval().half()
        self.speed = rospy.get_param('~speed')
        self.turn_block = rospy.get_param('~turn_block')
        self.turn_gain = rospy.get_param('~turn_gain')
        self.target_label = rospy.get_param('~target_label')
        self.display = rospy.get_param('~display')

    def _decide_blocked(self):
        x = self._preprocess(self.pil_image)
        y = self.collision_model(x)
        y = F.softmax(y, dim=1)
        prob_blocked = float(y.flatten()[0])  
        return prob_blocked
     
    def _object_detection(self):
        image = self.cv_image.copy()
        (height, width, channel) = image.shape
        detections = self.detection_model(image) # draw all detections on image
        for det in detections[0]:
            bbox = det['bbox']
            cv2.rectangle(image, (int(width * bbox[0]), int(height * bbox[1])), 
                          (int(width * bbox[2]), int(height * bbox[3])), (255, 0, 0), 2)
    
        # select detections that match selected class label
        matching_detections = [d for d in detections[0] if d['label'] == self.target_label]
    
        matching_detections = self._reject_small(matching_detections)

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

    #small objects are rejected by k. nakahira
    def _reject_small(self, dets):
        rdets = []
        for det in dets:
            bbox = det['bbox']
            if not (abs(bbox[2] - bbox[0]) < 0.05 and abs(bbox[3] - bbox[1]) < 0.05):
                rdets.append(det)
        return rdets

    def _decide_motor_value(self):
        prob_blocked = self._decide_blocked()
        det = self._object_detection()

        # otherwise go forward if no target detected
        if prob_blocked > 0.3:
            forward = 0.0
            angle = self.turn_block
        elif det is None:
            forward = self.speed
            angle = 0.0      
        # otherwsie steer towards target
        else:
            # move robot forward and steer proportional target's x-distance from center
            center = detection_center(det)
            forward = self.speed
            angle = self.turn_gain * center[0]

        rospy.loginfo("blocked: {},  det:{}".format(prob_blocked, det))

        return forward, angle

    def publish_loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.pil_image is None:
                rospy.logwarn("No Image subscride.")
                continue
            forward, angle = self._decide_motor_value()
            twist = Twist()
            twist.linear.x = forward
            twist.angular.z = -angle
            rospy.loginfo("linear_x: {}, angular_z: {}".format(twist.linear.x, twist.angular.z))
            self._cmd_vel_pub.publish(twist)
            rate.sleep()

    def _preprocess(self, image):
        mean = torch.Tensor([0.485, 0.456, 0.406]).to(self.device).half()
        stdev = torch.Tensor([0.229, 0.224, 0.225]).to(self.device).half()
        image = image.resize((224,224))
        image = transforms.functional.to_tensor(image).to(self.device).half()
        image.sub_(mean[:, None, None]).div_(stdev[:, None, None])
        return image[None, ...]        

    def _image_callback(self, msg):
        self.pil_image = imgmsg_to_pil(msg)
        self.cv_image = cv2.resize(pil_to_cv(self.pil_image), dsize=(300,300))

if __name__ == '__main__':
    oc = ObjectFollowingController()
    oc.publish_loop()
