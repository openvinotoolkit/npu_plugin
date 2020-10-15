cmake_minimum_required(VERSION 3.5.0)
SET(PROJECT_NAME "HDE")
set(CMAKE_BUILD_TYPE Release)
 
include_directories(include)
 
file(GLOB SOURCES "src/*.cpp")
 
add_library(hde SHARED ${SOURCES})
