cmake_minimum_required(VERSION 3.0.2)
project(ros_vui)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_package(
  CATKIN_DEPENDS
)

catkin_install_python(PROGRAMS scripts/vui_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
