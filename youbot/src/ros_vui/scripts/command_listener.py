#!/usr/bin/env python
from typing import Tuple
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist

FORWARD = "forward"
BACKWARD = "backward"
LEFT = "left"
RIGHT = "right"
STOP = "stop"


def try_get_velocity(command: str) -> Tuple[bool, Twist]:
    twist = Twist()

    if command == FORWARD:
        twist.linear.x = 0.2
    elif command == BACKWARD:
        twist.linear.x = -0.2
    elif command == LEFT:
        twist.angular.z = -0.2
    elif command == RIGHT:
        twist.angular.z = 0.2
    elif command == STOP:
        pass
    else:
        return False, twist

    return True, twist


def main():
    try:
        rospy.init_node('command_listener_node', anonymous=True)
        velocity_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        def command_callback(command: String):
            success, twist = try_get_velocity(command.data)
            if success:
                velocity_pub.publish(twist)

        rospy.Subscriber("/text_command", String, command_callback)
        rospy.spin()

    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
