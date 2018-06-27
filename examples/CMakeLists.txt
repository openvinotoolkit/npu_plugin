cmake_minimum_required(VERSION 3.5.0)
project(fathom VERSION 1.0.0)

add_executable(graph_container graph_container.cpp)
target_sources(graph_container PUBLIC ${allocator})

add_executable(computation_model computation_model.cpp)
target_sources(computation_model PUBLIC ${allocator} ${computation_model} ${ops} ${logger})

add_executable(cm_two_branch cm_two_branch.cpp)
target_sources(cm_two_branch PUBLIC ${allocator} ${computation_model} ${ops} ${logger} ${pass})

add_executable(cm_allocate_resources cm_allocate_resources.cpp)
target_sources(cm_allocate_resources PUBLIC ${allocator} ${computation_model} ${ops} ${logger} ${pass})

add_executable(cm_resnet18 cm_resnet18.cpp)
target_sources(cm_resnet18 PUBLIC ${allocator} ${computation_model} ${ops} ${logger} ${pass})

add_executable(cm_batchnorm_fuse cm_batchnorm_fuse.cpp)
target_sources(cm_batchnorm_fuse PUBLIC ${allocator} ${computation_model} ${ops} ${logger} ${pass})
target_link_libraries(cm_batchnorm_fuse m)

add_executable(cm_bias_fuse cm_bias_fuse.cpp)
target_sources(cm_bias_fuse PUBLIC ${allocator} ${computation_model} ${ops} ${logger} ${pass})
target_link_libraries(cm_bias_fuse m)

add_executable(conjoined_graph conjoined_graph.cpp)
target_sources(conjoined_graph PUBLIC ${allocator})