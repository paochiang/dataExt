cmake_minimum_required(VERSION 2.8)  
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
project(kinectDataExc)

#OPENCV
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS} )
link_directories(${OPENCV_LIBRARY_DIRS})
add_definitions(${OPENCV_DEFINITIONS})

include_directories("C:/Program Files/Microsoft SDKs/Kinect/v2.0_1409/inc")
link_directories("C:/Program Files/Microsoft SDKs/Kinect/v2.0_1409/Lib/x64")

add_executable(kinectDataExc main.cpp)

# Allow the developer to select if Dynamic or Static libraries are built
OPTION (BUILD_SHARED_LIBS "Build Shared Libraries" OFF)
# Set the LIB_TYPE variable to STATIC
SET (LIB_TYPE SHARED)
IF (BUILD_SHARED_LIBS)
  # User wants to build Dynamic Libraries, so change the LIB_TYPE variable to CMake keyword 'SHARED'
  SET (LIB_TYPE SHARED)
ENDIF (BUILD_SHARED_LIBS)

# Create a target for the library
#ADD_LIBRARY(MyLibrary ${LIB_TYPE} main.cpp)

TARGET_LINK_LIBRARIES(kinectDataExc ${OpenCV_LIBS} Kinect20.Fusion.lib Kinect20.lib)
#ddddddd