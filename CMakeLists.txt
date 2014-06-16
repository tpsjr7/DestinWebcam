cmake_minimum_required(VERSION 2.8)
project( DestinWebcam )

find_package( OpenCV REQUIRED )

include_directories("/home/dickson/naoqi/naoqi-sdk-1.14.5-linux32/include")
link_directories("/home/dickson/naoqi/naoqi-sdk-1.14.5-linux32/lib")

link_directories("/home/dickson/Desktop/destin/Destin/Common")
link_directories("/home/dickson/Desktop/destin/Destin/DavisDestin")
include_directories("/home/dickson/Desktop/destin/Destin/DavisDestin/include")
include_directories("/home/dickson/Desktop/destin/Destin/Common")


add_executable( DestinWebcam DestinWebcam.cpp )
target_link_libraries( DestinWebcam ${OpenCV_LIBS} )

target_link_libraries (DestinWebcam "alproxies")
target_link_libraries (DestinWebcam "alvalue")
target_link_libraries (DestinWebcam "alvision")
target_link_libraries (DestinWebcam "alerror")

#TARGET_LINK_LIBRARiES( DestinWebcam destinalt ${OpenCV_LIBS} ${avutil_lib} common)
target_link_libraries (DestinWebcam "common")
target_link_libraries (DestinWebcam "destinalt")
