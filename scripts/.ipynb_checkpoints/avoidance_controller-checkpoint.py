#!/usr/bin/env python3
import cv2

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import rospy
from cv_bridge import CvBridge
from PIL import Image as PIL_Image
from conversion_utils import imgmsg_to_pil

class AvoidanceController:

    def __init__(self):
        rospy.Subscriber('/jetbot_camera/raw', Image, self._image_callback)
        self._cmd_vel_pub = rospy.Publisher('/jetbot/avoidance/cmd_vel', Twist, queue_size=1)
        rospy.init_node('avoidance_controller')
        self.model_path = rospy.get_param('~model')
        self.device = torch.device('cuda')
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).to(self.device).half()
        self.stdev = torch.Tensor([0.229, 0.224, 0.225]).to(self.device).half()
        self.image = None

    def start(self):
        rate = rospy.Rate(10)
        self._publish_loop()

    def _publish_loop(self):

        while not rospy.is_shutdown():
            if self.image is None:
                rospy.logwarn("No Image subscride.")
                continue
            coli_deci = self._decide_collision()
            twist = Twist()
            if coli_deci:
                twist.angular.z = 0.5
            else:
                twist.linear.x = 0.5
            rospy.loginfo("linear_x: {}, angular_z: {}".format(twist.linear.x, twist.angular.z))
            self._cmd_vel_pub.publish(twist)

    def _decide_collision(self):
        image = self._preprocess()
        y = self.model(image)
        y = F.softmax(y, dim=1)
        prob_blocked = float(y.flatten()[0])
        rospy.loginfo("prob_blocked: {}".format(prob_blocked))
        return prob_blocked > 0.5

    def initialize_inferance(self):
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(512, 2)
        model.load_state_dict(torch.load(self.model_path))
        model = model.to(self.device)
        self.model = model.eval().half()

    def _preprocess(self):
        image = self.image
        image = transforms.functional.to_tensor(image).to(self.device).half()
        image.sub_(self.mean[:, None, None]).div_(self.stdev[:, None, None])
        image = image[None, ...]
        return image


    def _image_callback(self, msg):
        self.image = imgmsg_to_pil(msg)
        #self.image = imgmsg_to_pil(msg, 'bgr8')
        #image = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
        #self.image = PIL_Image.fromstring("L", cv2.GetSize(image), image.tostring())


if __name__ == '__main__':
    ac = AvoidanceController()
    ac.initialize_inferance()
    ac.start()
