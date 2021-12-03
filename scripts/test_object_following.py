#!/usr/bin/env python3
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import rospy
from PIL import Image as PIL_Image
from conversion_utils import imgmsg_to_pil, pil_to_cv
import cv2
from cv_bridge import CvBridge, CvBridgeError

class ObjectFollowingController:

    def __init__(self):
        rospy.Subscriber('/jetbot_camera/raw', Image, self._image_callback)
        self._cmd_vel_pub = rospy.Publisher('/jetbot/cmd_vel', Twist, queue_size=1)
        self.image_pub = rospy.Publisher('image_topic', Image) 
        rospy.init_node('object_following')
        self.pil_image = None
        self.cv_image = None
        self.collision_model = None
        self.detection_model = None

    def start(self):
        rate = rospy.Rate(10)
        self._publish_loop()

    def _decide_motor_value(self):
        
        image = self.cv2_image.copy()
        #image = self.cv_image
        (rows,cols,channels) = image.shape
        if cols > 60 and rows > 60 :
            cv2.circle(image, (50,50), 10, 255)
            
        cv2.imshow("Image window", image)
        cv2.waitKey(3)            
        
        #bridge = CvBridge()
        #try:
        #    image_message = bridge.cv2_to_imgmsg(image, encoding='bgr8')
        #    self.image_pub.publish(image_message)
        #except CvBridgeError as e:
        #    rospy.loginfo("CvBridgeError: {}".format(e))
        return 1.0, 1.0
            
    def _publish_loop(self):
        while not rospy.is_shutdown():
            if self.pil_image is None:
                rospy.logwarn("No Image subscride.")
                continue
            forward, angle = self._decide_motor_value()

    def _image_callback(self, msg):
        self.pil_image = imgmsg_to_pil(msg)
        self.cv2_image = cv2.resize(pil_to_cv(self.pil_image), dsize=(300,300))

if __name__ == '__main__':
    oc = ObjectFollowingController()
    oc.start()
