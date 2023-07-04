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

class AvoidanceController:

    def __init__(self):
        self._cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        rospy.init_node('avoidance_controller')
        self.model_path = rospy.get_param('~model')
        self.display_flip = rospy.get_param('~display_flip')
        self.device = torch.device('cuda')
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).to(self.device).half()
        self.stdev = torch.Tensor([0.229, 0.224, 0.225]).to(self.device).half()
        self.pil_image = None
        self.cv_image = None
        self.collision_state = None
        rospy.Subscriber('/image', Image, self._image_callback)

    def publish_loop(self):
        if self.pil_image is None:
            rospy.logwarn("No Image subscride.")
            return
        self._decide_collision()
        twist = Twist()
        if self.collision_state:
            twist.angular.z = 0.5
        else:
            twist.linear.x = 0.5
        rospy.loginfo("linear_x: {}, angular_z: {}".format(twist.linear.x, twist.angular.z))
        self._cmd_vel_pub.publish(twist)

    def _decide_collision(self):
        image = self._preprocess(self.pil_image)
        y = self.model(image)
        y = F.softmax(y, dim=1)
        prob_blocked = float(y.flatten()[0])
        rospy.loginfo("prob_blocked: {}".format(prob_blocked))
        self.collision_state = (prob_blocked > 0.5)

    def mk_image(self):
        image = self.cv_image.copy()
        (height, width, channel) = image.shape
        if self.collision_state:
            image2 = np.full((height, width, 3), 128, dtype=np.uint8)
            cv2.rectangle(image2, (0,0), (width, height), (0, 0, 255), thickness=-1)
            image = cv2.addWeighted(image, 0.5, image2, 0.5, 2.2)
            cv2.rectangle(image, (0,0), (width, height), (0, 0, 255), thickness=10)

        #image = cv2.resize(image, (int(width*2), int(height*2)))
        return image

    def initialize_inferance(self):
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(512, 2)
        model.load_state_dict(torch.load(self.model_path))
        model = model.to(self.device)
        self.model = model.eval().half()

    def _preprocess(self, image):
        image = transforms.functional.to_tensor(image).to(self.device).half()
        image.sub_(self.mean[:, None, None]).div_(self.stdev[:, None, None])
        image = image[None, ...]
        return image

    def _image_callback(self, msg):
        if msg is not None:
            self.pil_image = imgmsg_to_pil(msg)
            if self.display_flip:
                self.pil_image = ImageOps.flip(self.pil_image)
            #self.cv_image = cv2.resize(pil_to_cv(self.pil_image), dsize=(1280, 720))
            self.cv_image = pil_to_cv(self.pil_image)

if __name__ == '__main__':
    ac = AvoidanceController()
    ac.initialize_inferance()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        if ac.pil_image is None:
            rospy.logwarn("No Image subscride.")
            continue
        ac.publish_loop()
        img = ac.mk_image()
        cv2.imshow('jet_camera', img)
        key = cv2.waitKey(10)
        if key == 27: # ESC 
            break
        rate.sleep()
