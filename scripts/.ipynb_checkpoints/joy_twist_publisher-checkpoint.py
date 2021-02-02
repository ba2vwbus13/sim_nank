import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

class JoyController:
    def __init__(self):
        rospy.init_node('joy_twist_publisher')
        self.cmd_vel = Twist()
        self.pub = rospy.Publisher('/jetbot/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/joy', Joy, self._callback)
      
    def _callback(self, joy_msg):
        self.cmd_vel.linear.x = joy_msg.axes[1]
        self.cmd_vel.linear.y = joy_msg.axes[0]
        self.cmd_vel.angular.z = joy_msg.axes[3]
        rospy.loginfo("ax0 :{}  ax1 :{}  ax2 :{}".format(joy_msg.axes[0], joy_msg.axes[1], joy_msg.axes[2]))
        
    def pub(self):
        self.pub.publish(self.cmd_vel)

if __name__ == '__main__':
    jc = JoyController()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        jc.pub()
        rate.sleep()
