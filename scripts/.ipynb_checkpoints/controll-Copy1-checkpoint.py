#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from mobile import MobileController
from jetbot import Robot
import time

class JetbotController:

    def __init__(self):
        robot = Robot()
        self.mobile = MobileController(10, robot)
        rospy.Subscriber("/jetbot/cmd_vel", Twist, self.cmd_vel_callback)

    def start(self):
        rospy.init_node('jetson_controller')
        self.mobile.run()
        rospy.spin()

    def _input_disc(self,input):
        negative = -1.0 if input < 0.0 else 1.0
        input = abs(input)
        if 0.0 < input <= 0.3:
            return negative * 0.3
        elif 0.3 < input <= 0.5:
            return negative * 0.5
        elif 0.5 < input <= 0.7:
            return negative * 0.7
        elif 0.7 < input <= 1.0:
            return negative * 1.0
        else:
            return 0.0

    def cmd_vel_callback(self, msg):
        self.mobile.controll(msg.linear.x, msg.angular.z)

    def joy_stick_callback(self, msg):
        slottle = self._input_disc(msg.axes[1])
        handle = msg.axes[2]
        #buttons.5,7
        rospy.loginfo("joystick event Slottle: {}, Argument: {}".format(slottle, handle))
        self.mobile.controll(slottle, -1.0*handle)

def main():
    jetbot_controller = JetbotController()
    jetbot_controller.start()
    
    try:
        while True:
                time.sleep(0.1)
    except 
    
if __name__ == '__main__':
    main()
    #jetbot_controller = JetbotController()
    #jetbot_controller.start()

