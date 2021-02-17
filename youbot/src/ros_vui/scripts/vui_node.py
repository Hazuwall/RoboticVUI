#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import vui


def main():
    try:
        rospy.init_node('vui_node', anonymous=True)
        pub = rospy.Publisher('/text_command', String, queue_size=10)

        def handle_word(word, weight):
            pub.publish(word)
            rospy.loginfo("{}, {:.2f}".format(word, weight))

        vui.run(handle_word)

    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
