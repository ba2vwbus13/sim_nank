#!/usr/bin/env python
import numpy as np
import rospy
from std_msgs.msg import UInt16
from std_msgs.msg import Float64
from std_msgs.msg import String
from sensor_msgs.msg import Imu
from sensor_msgs.msg import MagneticField

class DirectionDetecter(object):
    def __init__(self):
        self.a_x = None
        self.a_y = None
        self.a_z = None
        self.prev_a_x = None
        self.prev_a_y = None
        self.prev_a_z = None

        self.m_x = None
        self.m_y = None
        self.m_z = None
        self.stop_m_x = None
        self.stop_m_y = None
        self.stop_m_z = None
        self.forward_m_x = None
        self.forward_m_y = None
        self.forward_m_z = None        
        self.back_m_x = None
        self.back_m_y = None
        self.back_m_z = None
        self.right_m_x = None
        self.right_m_y = None
        self.right_m_z = None
        self.left_m_x = None
        self.left_m_y = None
        self.left_m_z = None
        self.dance_m_x = None
        self.dance_m_y = None
        self.dance_m_z = None

        self.command = 'stop'
        rospy.Subscriber('imu/data_raw', Imu, self._callback_imu, queue_size=1)
        rospy.Subscriber('imu/mag', MagneticField, self._callback_mag, queue_size=1)
        #self.com_pub = rospy.Publisher('jet_command', String, queue_size=1)
        self.mortor = rospy.Publisher('/jetbot_motors/cmd_str', String, queue_size=1)

    def _callback_imu(self, data):
        self.a_x = data.linear_acceleration.x
        self.a_y = data.linear_acceleration.y
        self.a_z = data.linear_acceleration.z

    def _callback_mag(self, data):
        self.m_x = data.magnetic_field.x*1E6
        self.m_y = data.magnetic_field.y*1E6
        self.m_z = data.magnetic_field.z*1E6

    def mag_init(self):
        while True:
            key = input('stop (y/n): ')
            if key == 'y':
                print('mx:{} my:{} mz:{}'.format(self.m_x, self.m_y, self.m_z))
                self.stop_m_x = self.m_x
                self.stop_m_y = self.m_y
                self.stop_m_z = self.m_z                
                break

        while True:
            key = input('forward  (y/n): ')
            if key == 'y':
                print('mx:{} my:{} mz:{}'.format(self.m_x, self.m_y, self.m_z))
                self.forward_m_x = self.m_x
                self.forward_m_y = self.m_y
                self.forward_m_z = self.m_z                
                break

        while True:          
            key = input('back  (y/n): ')
            if key == 'y':
                print('mx:{} my:{} mz:{}'.format(self.m_x, self.m_y, self.m_z))
                self.back_m_x = self.m_x
                self.back_m_y = self.m_y
                self.back_m_z = self.m_z  
                break

        while True:
            key = input('right  (y/n): ')
            if key == 'y':
                print('mx:{} my:{} mz:{}'.format(self.m_x, self.m_y, self.m_z))
                self.right_m_x = self.m_x
                self.right_m_y = self.m_y
                self.right_m_z = self.m_z
                break

        while True:         
            key = input('left  (y/n): ')
            if key == 'y':
                print('mx:{} my:{} mz:{}'.format(self.m_x, self.m_y, self.m_z))
                self.left_m_x = self.m_x
                self.left_m_y = self.m_y
                self.left_m_z = self.m_z                   
                break

        while True:         
            key = input('dance  (y/n): ')
            if key == 'y':
                print('mx:{} my:{} mz:{}'.format(self.m_x, self.m_y, self.m_z))
                self.dance_m_x = self.m_x
                self.dance_m_y = self.m_y
                self.dance_m_z = self.m_z                   
                break

    def mag_update(self):
        print('mx:{} my:{} mz:{}'.format(self.m_x, self.m_y, self.m_z))
        #print('ax:{} ay:{} az:{}'.format(self.a_x, self.a_y, self.a_z))
        mm = 10
        if self.m_x == None or self.m_y == None or self.m_z == None:
            self.command = 'stop'
        else:
            if abs(self.forward_m_x - self.m_x) < mm and abs(self.forward_m_y - self.m_y) < mm and abs(self.forward_m_z - self.m_z) < mm:
                self.command = 'forward'
            elif abs(self.back_m_x - self.m_x) < mm and abs(self.back_m_y - self.m_y) < mm and abs(self.back_m_z - self.m_z) < mm:
                self.command = 'backward' 
            elif abs(self.right_m_x - self.m_x) < mm and abs(self.right_m_y - self.m_y) < mm and abs(self.right_m_z - self.m_z) < mm:
                self.command = 'right'
            elif abs(self.left_m_x - self.m_x) < mm and abs(self.left_m_y - self.m_y) < mm and abs(self.left_m_z - self.m_z) < mm:
                self.command = 'left'
            elif abs(self.dance_m_x - self.m_x) < mm and abs(self.dance_m_y - self.m_y) < mm and abs(self.dance_m_z - self.m_z) < mm:
                self._dancing()
                self.command = 'stop'
            else:
                self.command = 'stop'
        self.mortor.publish(self.command)

    def _dancing(self):
        dancing_pattern = np.concatenate(
            [np.repeat('forward', 3), np.repeat('stop', 1), 
            np.repeat('backward', 3), np.repeat('stop', 1), 
            np.repeat('right', 2), np.repeat('stop', 1), 
            np.repeat('left', 2)])
        print('command => dance')
        r = rospy.Rate(1)
        for com in dancing_pattern:
            print('command => {}'.format(com))
            self.mortor.publish(com)
            r.sleep()

def main():
    rospy.init_node('direction_detect')

    detecter = DirectionDetecter()
    detecter.mag_init()
    rospy.sleep(2)
    r = rospy.Rate(1)
    while not rospy.is_shutdown():
        detecter.mag_update()
        print('command => {}'.format(detecter.command))
        r.sleep()

if __name__ == '__main__':
    main()