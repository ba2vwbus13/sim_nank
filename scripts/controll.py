#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from jetbot import Robot #nedd python3
#from JetbotDriver import JetbotDriver
import time
import threading

class JetbotController:
    def __init__(self):
        self.robot = Robot()
        #self.robot = JetbotDriver().robot
        self.left_v = 0.0
        self.right_v = 0.0
        self.loop = True
        self.controll_thread = threading.Thread(target=self._controll_loop)
        rospy.init_node('jetbot_cmd_vel_controller')
        rospy.Subscriber("/cmd_vel", Twist, self._callback)

    def start(self):
        self.controll_thread = threading.Thread(target=self._controll_loop)
        self.controll_thread.start()
        rospy.spin()
        
    def _callback(self, msg):
        speed = msg.linear.x
        radius = msg.angular.z
        self.right_v = (speed + radius)*0.5
        self.left_v = (speed - radius)*0.5
        
    def _controll_loop(self):
        while self.loop:
            self.robot.set_motors(self.left_v, self.right_v)
            time.sleep(0.1)

    def stop(self):
        self.loop = False
        self.robot.set_motors(0.0, 0.0)
        
def main():
    controll = JetbotController()
    controll.controll_thread.start()
    rospy.spin()    
    if rospy.is_shutdown():
        controll.stop()
        controll.controll_thread.join()

if __name__ == '__main__':
    main()
