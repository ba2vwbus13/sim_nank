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
from conversion_utils import imgmsg_to_pil, pil_to_cv
import cv2

class RoadFollowingController:

    def __init__(self):
        self._cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        rospy.init_node('road_following')
        self.model_path = rospy.get_param('~model')
        self.device = torch.device('cuda')
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).to(self.device).half()
        self.stdev = torch.Tensor([0.229, 0.224, 0.225]).to(self.device).half()
        self.pil_image = None
        self.cv_image = None
        self.angle_last = 0.0
        self.speed = rospy.get_param('~speed')
        self.steering_gain = rospy.get_param('~steering_gain')
        self.steering_bias = rospy.get_param('~steering_bias')
        self.steering_dgain = rospy.get_param('~steering_dgain')
        self.display_flip = rospy.get_param('~display_flip')
        rospy.Subscriber('/image', Image, self._image_callback)

    def publish_loop(self):
        if self.pil_image is None:
            rospy.logwarn("No Image subscride.")
        steering = self._decide_motor_value()
        twist = Twist()
        twist.linear.x = self.speed
        twist.angular.z = -steering
        rospy.loginfo("linear_x: {}, angular_z: {}".format(twist.linear.x, twist.angular.z))
        self._cmd_vel_pub.publish(twist)

    def _decide_motor_value(self):
        image = self._preprocess()
        self.xy = self.model(image).detach().float().cpu().numpy().flatten()
        x = self.xy[0]
        y = (0.5 - self.xy[1]) / 2.0
        angle = np.arctan2(x,y)
        pid = angle * self.steering_gain + (angle - self.angle_last) * self.steering_dgain
        self.angle_last = angle
        steering = pid + self.steering_bias
        left_motor = max(min(self.speed + steering, 1.0), 0.0)
        right_motor = max(min(self.speed - steering, 1.0), 0.0)
        return steering

    def mk_image(self):
        image = self.cv_image.copy()
        (height, width, channel) = image.shape
        #print("x:{} y:{}".format(self.xy[0], self.xy[1]))
        x = int(width/2*(1+self.xy[0]))
        y = int(height/2*(1+self.xy[1]))
        cv2.arrowedLine(image, (int(width/2),int(height/2)), (x,y), (255,255,255), 2, tipLength=0.5)
        image = cv2.resize(image, (int(width), int(height)))
        return image

    def initialize_inferance(self):
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(512, 2)
        model.load_state_dict(torch.load(self.model_path))
        model = model.to(self.device)
        self.model = model.eval().half()

    def _preprocess(self):
        image = self.pil_image
        image = transforms.functional.to_tensor(image).to(self.device).half()
        image.sub_(self.mean[:, None, None]).div_(self.stdev[:, None, None])
        image = image[None, ...]
        return image

    def _image_callback(self, msg):
        self.pil_image = imgmsg_to_pil(msg)
        self.pil_image = self.pil_image.resize((224, 224))
        if self.display_flip:
            self.pil_image = ImageOps.flip(self.pil_image)
            self.pil_image = ImageOps.mirror(self.pil_image)
        self.cv_image = pil_to_cv(self.pil_image)

if __name__ == '__main__':
    rc = RoadFollowingController()
    rc.initialize_inferance()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        if rc.pil_image is None:
            rospy.logwarn("No Image subscride.")
            continue
        rc.publish_loop()
        img = rc.mk_image()
        cv2.imshow('jet_camera', img)
        key = cv2.waitKey(10)
        if key == 27: # ESC 
            break
        rate.sleep()
