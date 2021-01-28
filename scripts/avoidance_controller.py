#!/usr/bin/env python3
import cv2

import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import rospy
from cv_bridge import CvBridge
from PIL import Image as PIL_Image

class AvoidanceController:

    _bridge = CvBridge()
    image = None
    device = torch.device('cuda')
    mean = 255.0 * np.array([0.485, 0.456, 0.406])
    stdev = 255.0 * np.array([0.229, 0.224, 0.225])

    def __init__(self):
        rospy.Subscriber('/jetbot_camera/raw', Image, self._image_callback)
        self._cmd_vel_pub = rospy.Publisher('/jetbot/avoidance/cmd_vel', Twist, queue_size=1)
        rospy.init_node('avoidance_controller')
        self.model_path = rospy.get_param('~model')

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
        return prob_blocked > 0.5

    def initialize_inferance(self):
        model = torchvision.models.resnet18(pretraind=False)
        model.fc = torch.nn.Linear(512, 2)
        model.load_state_dict(torch.load(self.model_path))
        model = model.to(self.device)
        self.model = model.eval().half()

    def _preprocess(self):
        image = self.image
        image = transforms.functional.to_tensor(image).to(device).half()
        image.sub_(mean[:, None, None]).dev_(std[:, None, None])
        image = image[None, ...]
        return image


    def _image_callback(self, msg):
        image = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.image = PIL_Image.fromstring("L", cv2.GetSize(image), image.tostring())


if __name__ == '__main__':
    ac = AvoidanceController()
    ac.initialize_inferance()
    ac.start()