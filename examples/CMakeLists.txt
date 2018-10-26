cmake_minimum_required(VERSION 3.5.0)
project(mcmComplier VERSION 0.0.1)

add_executable(graph_basic graph_basic.cpp)

add_executable(graph_conjoined graph_conjoined.cpp)

add_executable(cm_linear cm_linear.cpp)
target_link_libraries(cm_linear cm)

add_executable(cm_two_branch cm_two_branch.cpp)
target_link_libraries(cm_two_branch cm)

add_executable(cm_resnet18 cm_resnet18.cpp)
target_link_libraries(cm_resnet18 cm)

add_executable(cm_resnet50 cm_resnet50.cpp)
target_link_libraries(cm_resnet50 cm)

add_executable(cm_allocate_resources cm_allocate_resources.cpp)
target_link_libraries(cm_allocate_resources cm)

add_executable(test_f test.cpp)
target_link_libraries(test_f cm)


include_directories("/opt/movidius/caffe/include")
include_directories("/opt/movidius/caffe/build/include")
include_directories("/opt/movidius/caffe/build/include/caffe/proto")
#link_directories("/opt/movidius/caffe/build/lib")

#include_directories("/usr/local/include/google")
#include(FindProtobuf)
#find_package(Protobuf REQUIRED)
#include_directories(${PROTOBUF_INCLUDE_DIR})

#find_package(Boost COMPONENTS regex system filesystem REQUIRED)

add_executable(prototxt_test prototxt_test.cpp)
target_link_libraries(prototxt_test cm_validation ${Caffe_LIBRARIES} ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_REGEX_LIBRARY} ${PROTOBUF_LIBRARY}) 
target_compile_definitions(prototxt_test PRIVATE CPU_ONLY=1)
