#!/usr/bin/env python3
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import rospy
from PIL import Image as PIL_Image
from conversion_utils import imgmsg_to_pil, pil_to_cv
from jetbot import ObjectDetector
import cv2
from cv_bridge import CvBridge

class ObjectFollowingController:

    def __init__(self):
        rospy.Subscriber('/jetbot_camera/raw', Image, self._image_callback)
        self._cmd_vel_pub = rospy.Publisher('/jetbot/cmd_vel', Twist, queue_size=1)
        self.image_pub = rospy.Publisher('image_topic', Image, queue_size=1) 
        rospy.init_node('object_following')
        self.avoidance_model_path = rospy.get_param('~avoidance_model')
        self.following_model_path = rospy.get_param('~following_model')
        self.device = torch.device('cuda')
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).to(self.device).half()
        self.stdev = torch.Tensor([0.229, 0.224, 0.225]).to(self.device).half()
        self.pil_image = None
        self.cv_image = None
        self.collision_model = None
        self.detection_model = None
        self.speed = rospy.get_param('~speed')
        self.turn_gain = rospy.get_param('~turn_gain')
        self.target_label = rospy.get_param('~target_label')

    def start(self):
        rate = rospy.Rate(10)
        self._publish_loop()

    def detection_center(detection):
        """Computes the center x, y coordinates of the object"""
        bbox = detection['bbox']
        center_x = (bbox[0] + bbox[2]) / 2.0 - 0.5
        center_y = (bbox[1] + bbox[3]) / 2.0 - 0.5
        return (center_x, center_y)   
    def norm(vec):
        """Computes the length of the 2D vector"""
        return np.sqrt(vec[0]**2 + vec[1]**2)
    def closest_detection(detections):
        """Finds the detection closest to the image center"""
        closest_detection = None
        for det in detections:
            center = detection_center(det)
            if closest_detection is None:
                closest_detection = det
            elif norm(detection_center(det)) < norm(detection_center(closest_detection)):
                closest_detection = det
        return closest_detection


    def _decide_motor_value(self):
        x = self._preprocess(self.pil_image)
        y = self.collision_model(x)
        y = F.softmax(y, dim=1)
        prob_blocked = float(y.flatten()[0])
        if prob_blocked > 0.3:
            forward = 0.0
            angle = 0.3
            return forward, angle
        
        image = self.cv_image.copy()
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
        
        bridge = CvBridge()
        try:
            image_message = bridge.cv2_to_imgmsg(image, encoding='bgr8')
            self.image_pub.publish(image_message)
        except CvBridgeError as e:
            rospy.loginfo("CvBridgeError: {}".format(e))
        # otherwise go forward if no target detected
        if det is None:
            foward = 0.3
            angle = 0.0
            return forward, angle        
        # otherwsie steer towards target
        else:
            # move robot forward and steer proportional target's x-distance from center
            center = detection_center(det)
            forward = 0.3
            angle = self.turn_gain * center[0]
            return forward, angle

    def _publish_loop(self):
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

    def initialize_inferance(self):
        self.detection_model = ObjectDetector(self.following_model_path)
        collision_model = torchvision.models.resnet18(pretrained=False)
        collision_model.fc = torch.nn.Linear(512, 2)
        collision_model.load_state_dict(torch.load(self.avoidance_model_path))
        collision_model = collision_model.to(self.device)
        self.collision_model = collision_model.eval().half()

    def _preprocess(self, image):
        image = image.resize((224,224))
        image = transforms.functional.to_tensor(image).to(self.device).half()
        image.sub_(self.mean[:, None, None]).div_(self.stdev[:, None, None])
        return image[None, ...]        

    def _image_callback(self, msg):
        self.pil_image = imgmsg_to_pil(msg)
        self.cv2_image = pil_to_cv(self.pil_image)

if __name__ == '__main__':
    oc = ObjectFollowingController()
    oc.initialize_inferance()
    oc.start()
