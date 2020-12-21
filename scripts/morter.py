#!/usr/bin/env python3
import rospy
import time
from std_msgs.msg import String
from geometry_msgs.msg import Twist

cmd_data = Twist()

# simple string commands (left/right/forward/backward/stop)
def on_cmd_str(msg):
	rospy.loginfo(rospy.get_caller_id() + ' cmd_str=%s', msg.data)

	speed=0.1
	ang=0.1
	if msg.data.lower() == "left":
		cmd_data.linear.x = 0
		cmd_data.angular.z = ang 
	elif msg.data.lower() == "right":
		cmd_data.linear.x = 0
		cmd_data.angular.z = -ang 
	elif msg.data.lower() == "forward":
		cmd_data.linear.x = speed
		cmd_data.angular.z = 0
	elif msg.data.lower() == "backward":
		cmd_data.linear.x = -speed
		cmd_data.angular.z = ang 
	elif msg.data.lower() == "stop":
		cmd_data.linear.x = 0
		cmd_data.angular.z = 0 
	else:
		rospy.logerror(rospy.get_caller_id() + ' invalid cmd_str=%s', msg.data)


# initialization
if __name__ == '__main__':

	# setup ros node
	rospy.init_node('jetbot_motors')
	
	rospy.Subscriber('/jetbot_mortors/cmd_str', String, on_cmd_str)
	cmd_vel = rospy.Publisher('/dtw_robot/diff_drive_controller/cmd_vel', Twist, queue_size=1)
	# start running
	rate = rospy.Rate(10)
	while not rospy.is_shutdown():
		cmd_vel.publish(cmd_data)
		rate.sleep()

	# stop motors before exiting
	cmd_data.linear.x = 0
	cmd_data.angular.z = 0 
	cmd_vel.publish(cmd_data)
