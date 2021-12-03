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
from conversion_utils import imgmsg_to_pil

class RoadFollowingController:

    def __init__(self):
        rospy.Subscriber('/jetbot_camera/raw', Image, self._image_callback)
        self._cmd_vel_pub = rospy.Publisher('/jetbot/cmd_vel', Twist, queue_size=1)
        rospy.init_node('road_following')
        self.model_path = rospy.get_param('~model')
        self.device = torch.device('cuda')
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).to(self.device).half()
        self.stdev = torch.Tensor([0.229, 0.224, 0.225]).to(self.device).half()
        self.image = None
        self.angle_last = 0.0
        self.speed = rospy.get_param('~speed')
        self.steering_gain = rospy.get_param('~steering_gain')
        self.steering_bias = rospy.get_param('~steering_bias')
        self.steering_dgain = rospy.get_param('~steering_dgain')

    def start(self):
        rate = rospy.Rate(10)
        self._publish_loop()

    def _publish_loop(self):

        while not rospy.is_shutdown():
            if self.image is None:
                rospy.logwarn("No Image subscride.")
                continue
            #lmotor, rmotor = self._decide_motor_value()
            steering = self._decide_motor_value()
            twist = Twist()
            twist.linear.x = self.speed
            twist.angular.z = -steering

            #forward_hz = 80000.0*message.linear.x/(9*math.pi)
            #rot_hz = 400.0*message.angular.z/math.pi
            #self.set_raw_freq(forward_hz-rot_hz, forward_hz+rot_hz)
            #if coli_deci:
            #    twist.angular.z = 0.5
            #else:
            #    twist.linear.x = 0.5
            rospy.loginfo("linear_x: {}, angular_z: {}".format(twist.linear.x, twist.angular.z))
            self._cmd_vel_pub.publish(twist)

    def _decide_motor_value(self):
        image = self._preprocess()
        xy = self.model(image).detach().float().cpu().numpy().flatten()
        x = xy[0]
        y = (0.5-xy[1])/2.0
        angle = np.arctan2(x,y)
        pid = angle * self.steering_gain + (angle - self.angle_last) * self.steering_dgain
        self.angle_last = angle
        steering = pid + self.steering_bias
        left_motor = max(min(self.speed + steering, 1.0), 0.0)
        right_motor = max(min(self.speed - steering, 1.0), 0.0)
        #return left_motor, right_motor
        return steering

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

if __name__ == '__main__':
    rc = RoadFollowingController()
    rc.initialize_inferance()
    rc.start()
