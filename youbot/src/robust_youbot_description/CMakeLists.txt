cmake_minimum_required(VERSION 3.0.2)
project(robust_youbot_description)

find_package(catkin REQUIRED)

catkin_package(
  CATKIN_DEPENDS
)

install(DIRECTORY sdf
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})