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
from PIL import ImageOps
from PIL import Image as PIL_Image
from conversion_utils import (imgmsg_to_pil, pil_to_cv, closest_detection, detection_center, 
bbox_to_rbbox, bbox_to_roi, roi_to_bbox, plot_one_box)
import cv2
import time

class ObjectFollowingController:

    def __init__(self):
        rospy.init_node('object_following')
        rospy.Subscriber('/image', Image, self._image_callback)
        self._cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        if rospy.has_param('avoidance_model'):
            avoidance_model_path = rospy.get_param('avoidance_model')
            following_model_path = rospy.get_param('following_model_path')
            self.following_model = rospy.get_param('following_model')            
            self.speed = rospy.get_param('speed')
            self.turn_block = rospy.get_param('turn_block')
            self.turn_gain = rospy.get_param('turn_gain')
            self.target_label = rospy.get_param('target_label')
            self.display_flip = rospy.get_param('display_flip')
        else:
            avoidance_model_path = "/home/nakahira/catkin_ws/src/sim_nank/weights/best_model_resnet18.pth"
            following_model_path = '/home/nakahira/catkin_ws/src/sim_nank/weights/yolov5m.pt'
            self.following_model = 'yolo'        
            self.speed = 0.7
            self.turn_block = 0.5
            self.turn_gain = 1.0
            self.target_label = 0
            self.display_flip = False

        self.device = torch.device('cuda')
        self.pil_image = None
        self.cv_image = None
        self.collision_model = None
        self.detection_model = None
        if self.following_model == 'ssd':
            from ssd import ObjectDetector
        elif self.following_model == 'yolo':
            from ObjectDetector import ObjectDetector
        else:
            from jetbot import ObjectDetector
        self.detection_model = ObjectDetector(following_model_path)
        collision_model = torchvision.models.resnet18(pretrained=False)
        collision_model.fc = torch.nn.Linear(512, 2)
        collision_model.load_state_dict(torch.load(avoidance_model_path))
        collision_model = collision_model.to(self.device)
        self.collision_model = collision_model.eval().half()
        self.detections = None
        self.matching_detections = None
        self.collision_state = True
        self.frame_id = 0
        self.is_tracking = None
        self.get_bottle = False

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
        self.detections = detections[0]
        self.detections = self._reject_small(self.detections)

        # select detections that match selected class label
        matching_detections = [d for d in self.detections if d['label'] == self.target_label]

        return closest_detection(matching_detections)

    def mk_image_pet(self):
        image = self.cv_image.copy()
        (height, width, channel) = image.shape
        cv2.putText(image, 'Getting Petbot!', (int(width*0.1), int(height*0.9)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        image = cv2.resize(image, (int(width*2), int(height*2)))
        cv2.imshow('jet_camera', image)

    def mk_image(self):
        image = self.cv_image.copy()
        (height, width, channel) = image.shape


        if self.collision_state:
            image2 = np.full((height, width,  3), 128, dtype=np.uint8)
            cv2.rectangle(image2, (0,0), (width, height), (0, 0, 255), thickness=-1)
            image = cv2.addWeighted(image, 0.5, image2, 0.5, 2.2)
            cv2.rectangle(image, (0,0), (width, height), (0, 0, 255), thickness=10)

        for det in self.detections:
            plot_one_box(det['bbox'], image, label=det['label_name'], color=det['color'], line_thickness=3)

        if self.target_detection is not None:
            if self.is_tracking:
                plot_one_box(self.target_detection['bbox'], image, label='pet bottle!!!', color=(0, 255, 0), line_thickness=5)
            else:
                plot_one_box(self.target_detection['bbox'], image, label='pet bottle!!!', color=(255, 255, 255), line_thickness=10)

        if self.get_bottle:
            cv2.putText(image, 'Getting Petbot!', (int(width*0.1), int(height*0.9)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        image = cv2.resize(image, (int(width*2), int(height*2)))
        cv2.imshow('jet_camera', image)
    
    #small objects are rejected by k. nakahira
    def _reject_small(self, dets):
        rdets = []
        for det in dets:
            bbox = det['bbox']
            if abs(bbox[2] - bbox[0]) > 2 or abs(bbox[3] - bbox[1]) > 2:
                rdets.append(det)
        return rdets

    def update_target_detection_with_tracking(self):
        if self.frame_id % 1 == 0 or not self.is_tracking:
            self.is_tracking = None
            self.target_detection = self._object_detection()
            print('target detections :{}'.format(self.target_detection))
            if self.target_detection is not None:
                self.tracker = cv2.TrackerKCF_create()
                self.tracker.init(self.cv_image, bbox_to_roi(self.target_detection['bbox']))

        if self.target_detection is not None:
            self.is_tracking, roi = self.tracker.update(self.cv_image)
            self.target_detection['bbox'] = roi_to_bbox(roi)
            self.target_detection['rbbox'] = bbox_to_rbbox(self.target_detection['bbox'], self.cv_image)

    def update_target_detection(self):
        self.target_detection = self._object_detection()
        print('target detections :{}'.format(self.target_detection)) 
        if self.target_detection is not None:
            self.target_detection['rbbox'] = bbox_to_rbbox(self.target_detection['bbox'], self.cv_image)


    def decide_motor_value(self):
        
        prob = self._decide_blocked()
        self.collision_state = prob > 0.5
        self.update_target_detection()
        self.get_bottle = False

        if self.target_detection:
            rbbox = self.target_detection['rbbox']
            closer = (rbbox[2]-rbbox[0])*(rbbox[3]-rbbox[1])
        else:
            closer = 0.0

        # otherwise go forward if no target detected
        if self.collision_state:
            forward = 0.0
            angle = self.turn_block
        elif self.target_detection is None:
            forward = self.speed
            angle = 0.0      
        # otherwsie steer towards target
        else:
            if closer > 0.2:
                forward = 0.0
                angle = 0.0
                #angle = self.turn_block
                self.get_bottle = True
            else:
                # move robot forward and steer proportional target's x-distance from center
                center = detection_center(self.target_detection)
                forward = self.speed *0.8
                angle = 0.001 * (150-center[0])
               #angle = self.turn_gain * center[0]
                rospy.loginfo("center: {}".format(center))

        rospy.loginfo("blocked: {},  det:{}, closer:{}, forward:{}, angle:{}".format(prob, self.target_detection, closer, forward, angle))

        return forward, angle, self.get_bottle

    def publish_loop(self, forward, angle):
        twist = Twist()
        twist.linear.x = forward
        twist.angular.z = angle
        #rospy.loginfo("linear_x: {}, angular_z: {}".format(twist.linear.x, twist.angular.z))
        self._cmd_vel_pub.publish(twist)

    def _preprocess(self, image):
        mean = torch.Tensor([0.485, 0.456, 0.406]).to(self.device).half()
        stdev = torch.Tensor([0.229, 0.224, 0.225]).to(self.device).half()
        image = image.resize((224,224))
        image = transforms.functional.to_tensor(image).to(self.device).half()
        image.sub_(mean[:, None, None]).div_(stdev[:, None, None])
        return image[None, ...]        

    def _image_callback(self, msg):
        self.pil_image = imgmsg_to_pil(msg)
        if hasattr(self, 'display_flip') and self.display_flip:
            self.pil_image = ImageOps.flip(self.pil_image)
            self.pil_image = ImageOps.mirror(self.pil_image)
        self.cv_image = cv2.resize(pil_to_cv(self.pil_image), dsize=(300,300))


if __name__ == '__main__':
    oc = ObjectFollowingController()
    rate = rospy.Rate(5)
    wloop=0
    get_bottle = False
    bottle_picking = False
    while not rospy.is_shutdown():
        if bottle_picking:
            wloop+=1
            oc.mk_image_pet()
            oc.publish_loop(0.0, 0.0)
            if wloop > 50:
                bottle_picking = False
                wloop=0
                oc.publish_loop(0.0, oc.turn_block*1.5)
                rospy.sleep(0.5)
                rospy.loginfo("Bottle picked!")
        else:
            if oc.pil_image is None:
                rospy.logwarn("No Image subscride.")
                continue
            forward, angle, get_bottle = oc.decide_motor_value()
            if not get_bottle:
                oc.mk_image()
                oc.publish_loop(forward, angle)
            else:
                rospy.loginfo("Get bottle!")
                bottle_picking = True

        if cv2.waitKey(10) == 27: # ESC 
            break
        rate.sleep()
