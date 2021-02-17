cmake_minimum_required(VERSION 3.0.2)
project(ros_vui)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_package(
  CATKIN_DEPENDS
)

catkin_install_python(PROGRAMS
  scripts/speech_to_command.py
  scripts/command_listener.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
