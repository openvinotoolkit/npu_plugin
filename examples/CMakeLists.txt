cmake_minimum_required(VERSION 3.5.0)
project(mcmComplier VERSION 0.0.1)

#[[add_executable(graph_basic graph_basic.cpp)
target_sources(graph_basic PUBLIC ${allocator})

add_executable(graph_conjoined graph_conjoined.cpp)
target_sources(graph_conjoined PUBLIC ${allocator})

add_executable(cm_linear cm_linear.cpp)
target_sources(cm_linear PUBLIC ${base} ${allocator} ${computation_model} ${ops} ${logger} ${tensor})

add_executable(cm_two_branch cm_two_branch.cpp)
target_sources(cm_two_branch PUBLIC ${base} ${allocator} ${computation_model} ${ops} ${logger} ${tensor})

add_executable(cm_resnet18 cm_resnet18.cpp)
target_sources(cm_resnet18 PUBLIC ${base} ${allocator} ${computation_model} ${ops} ${logger} ${pass} ${tensor} ${compiler})
#[[
add_executable(cm_resnet50 cm_resnet50.cpp)
target_sources(cm_resnet50 PUBLIC ${base} ${allocator} ${computation_model} ${ops} ${logger} ${pass} ${tensor} ${compiler})

add_executable(cm_allocate_resources cm_allocate_resources.cpp)
target_sources(cm_allocate_resources PUBLIC ${base} ${allocator} ${computation_model} ${ops} ${logger} ${pass} ${tensor} ${compiler})]]

