cmake_minimum_required(VERSION 3.5.0)
project(mcmComplier VERSION 0.0.1)

add_executable(graph_basic graph_basic.cpp)

add_executable(graph_conjoined graph_conjoined.cpp)

add_executable(cm_linear cm_linear.cpp)
target_link_libraries(cm_linear cm)

add_executable(cm_two_branch cm_two_branch.cpp)
target_link_libraries(cm_two_branch cm)

add_executable(cm_allocate_resources cm_allocate_resources.cpp)
target_link_libraries(cm_allocate_resources cm)

add_executable(cm_resnet50 cm_resnet50.cpp)
target_link_libraries(cm_resnet50 cm)

add_executable(cm_resnet18 cm_resnet18.cpp)
target_link_libraries(cm_resnet18 cm)

add_executable(test_f test.cpp)
target_link_libraries(test_f cm)

if(VALIDATION)
add_executable(caffe_files_generator caffe_files_generator.cpp)
target_link_libraries(caffe_files_generator cm ${Caffe_LIBRARIES} ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_REGEX_LIBRARY} ${PROTOBUF_LIBRARY}) 
target_compile_definitions(caffe_files_generator PRIVATE CPU_ONLY=1)
endif(VALIDATION)
