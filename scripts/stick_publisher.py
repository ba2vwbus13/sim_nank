#!/usr/bin/env python2
import rospy
import time
import numpy as np
from std_msgs.msg import String
from std_msgs.msg import UInt16
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from sensor_msgs.msg import MagneticField

class NancleController(object):
    def __init__(self):
        self.a_x = 0
        self.a_y = 0
        self.a_z = 0
        self.m_x = 0
        self.m_y = 0
        self.m_z = 0
        self.cmd_data = Twist()
        rospy.Subscriber('imu/data_raw', Imu, self._callback_imu, queue_size=1)
        rospy.Subscriber('imu/mag', MagneticField, self._callback_mag, queue_size=1)
        self.cmd_vel = rospy.Publisher('/stick/cmd_vel', Twist, queue_size=1)
    def _callback_imu(self, data):
        self.a_x = data.linear_acceleration.x
        self.a_y = data.linear_acceleration.y
        self.a_z = data.linear_acceleration.z
    def _callback_mag(self, data):
        self.m_x = data.magnetic_field.x*1E6
        self.m_y = data.magnetic_field.y*1E6
        self.m_z = data.magnetic_field.z*1E6

    def accel_update(self):
        #print('mx:{} my:{} mz:{}'.format(self.m_x, self.m_y, self.m_z))
        #print('ax:{} ay:{} az:{}'.format(self.a_x, self.a_y, self.a_z))
        mm = 0.4
        mx = 0.80
        grav = 9.8

        if abs(self.a_x) < mm and abs(self.a_y) < mm and self.a_z > mx:
            rospy.loginfo('command => dance')
            dancing_patterns = np.concatenate([np.repeat('forward', 3), np.repeat('stop', 1), np.repeat('backward', 3), np.repeat('stop', 1), np.repeat('right', 2), np.repeat('stop', 1), np.repeat('left', 2)])
            self.str_move(dancing_patterns)
            self.cmd_data.linear.x = 0
            self.cmd_data.linear.y = 0
            self.cmd_data.angular.z = 0
        elif abs(self.a_x) < mm and abs(self.a_y) < mm:
            self.cmd_data.linear.x = 0
            self.cmd_data.linear.y = 0
            self.cmd_data.angular.z = 0
        else:
            #rospy.loginfo('command => normal')
            self.cmd_data.linear.x = -self.a_y / grav # *2
            self.cmd_data.linear.y = 0
            self.cmd_data.angular.z = self.a_x / grav # *3

        self.cmd_vel.publish(self.cmd_data)    
    def str_move(self, dancing_patterns):
        r = rospy.Rate(1)
        for dancing_pattern in dancing_patterns:
            self._set(dancing_pattern)
            self.cmd_vel.publish(self.cmd_data)
            r.sleep()

    def _set(self, pattern):
        self.cmd_data.linear.x = 0
        self.cmd_data.linear.y = 0
        self.cmd_data.angular.z = 0
    
        if pattern == 'forward':
            self.cmd_data.linear.x = 1.0
        elif pattern == 'backward':
            self.cmd_data.linear.x = -1.0
        elif pattern == 'right':
            self.cmd_data.angular.z = -1.0
        elif pattern == 'left':
            self.cmd_data.angular.z = 1.0
        elif pattern == 'stop':
            pass
        else:
            pass

# initialization
if __name__ == '__main__':
    rospy.init_node('jetbot_imu_motors')
    controller = NancleController()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        controller.accel_update()
        rate.sleep()
    controller.str_move('stop')
