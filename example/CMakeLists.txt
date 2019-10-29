cmake_minimum_required(VERSION 3.5.0)
project(mcmComplier VERSION 0.0.1)

if (MSVC)

    add_executable(templateExample templateExample.cpp $<TARGET_OBJECTS:cm> $<TARGET_OBJECTS:corebase> $<TARGET_OBJECTS:metamodel> $<TARGET_OBJECTS:model>)
    add_executable(z_major_convolution z_major_convolution.cpp $<TARGET_OBJECTS:cm> $<TARGET_OBJECTS:corebase> $<TARGET_OBJECTS:metamodel> $<TARGET_OBJECTS:model>)

else()
    add_executable(normalize_only normalize_only/normalize_only.cpp)
    target_link_libraries(normalize_only cm)

    add_executable(permute_only permute_only/permute_only.cpp)
    target_link_libraries(permute_only cm)

    add_executable(proposal_only proposal_only/proposal_only.cpp)
    target_link_libraries(proposal_only cm)

    add_executable(regionyolo_only regionyolo_only/regionyolo_only.cpp)
    target_link_libraries(regionyolo_only cm)

    add_executable(reorgyolo_only reorgyolo_only/reorgyolo_only.cpp)
    target_link_libraries(reorgyolo_only cm)

    add_executable(reshape_only reshape_only/reshape_only.cpp)
    target_link_libraries(reshape_only cm)

    add_executable(ROIpooling_only ROIpooling_only/ROIpooling_only.cpp)
    target_link_libraries(ROIpooling_only cm)

    add_executable(softmax_only softmax_only/softmax_only.cpp)
    target_link_libraries(softmax_only cm)

    if(VALIDATION)
    add_executable(caffe_files_generator caffe_files_generator.cpp)
    target_link_libraries(caffe_files_generator cm ${Caffe_LIBRARIES} ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_REGEX_LIBRARY} ${PROTOBUF_LIBRARY})
    target_compile_definitions(caffe_files_generator PRIVATE CPU_ONLY=1)
    add_executable(deployer_example deployer_example.cpp)
    target_link_libraries(deployer_example cm dep)
    add_executable(dep_graph_file_example dep_graph_file_example.cpp)
    target_link_libraries(dep_graph_file_example cm dep)
    endif(VALIDATION)

endif(MSVC)
