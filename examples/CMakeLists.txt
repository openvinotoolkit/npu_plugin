cmake_minimum_required(VERSION 3.5.0)
project(fathom VERSION 1.0.0)

add_executable(graph_container graph_container.cpp)
target_sources(graph_container PUBLIC ${allocator})

add_executable(conjoined_graph conjoined_graph.cpp)
target_sources(conjoined_graph PUBLIC ${allocator})

add_executable(cm_linear cm_linear.cpp)
target_sources(cm_linear PUBLIC ${base} ${allocator} ${computation_model} ${ops} ${logger} ${tensor})

add_executable(cm_two_branch cm_two_branch.cpp)
target_sources(cm_two_branch PUBLIC ${base} ${allocator} ${computation_model} ${ops} ${logger} ${pass} ${tensor})

add_executable(cm_batchnorm_fuse cm_batchnorm_fuse.cpp)
target_sources(cm_batchnorm_fuse PUBLIC ${base} ${allocator} ${computation_model} ${ops} ${logger} ${pass} ${tensor} ${compiler})
target_link_libraries(cm_batchnorm_fuse m)

add_executable(cm_resnet18 cm_resnet18.cpp)
target_sources(cm_resnet18 PUBLIC ${base} ${allocator} ${computation_model} ${ops} ${logger} ${pass} ${tensor} ${compiler})

add_executable(cm_allocate_resources cm_allocate_resources.cpp)
target_sources(cm_allocate_resources PUBLIC ${base} ${allocator} ${computation_model} ${ops} ${logger} ${pass} ${tensor})