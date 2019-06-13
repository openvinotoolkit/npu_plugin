cmake_minimum_required(VERSION 3.5.0)
project(mcmComplier VERSION 0.0.1)

add_executable(graph_basic graph_basic.cpp)

add_executable(graph_conjoined graph_conjoined.cpp)

if (MSVC)

    add_executable(average_to_depthwise average_to_depthwise.cpp $<TARGET_OBJECTS:cm> $<TARGET_OBJECTS:corebase> $<TARGET_OBJECTS:metamodel> $<TARGET_OBJECTS:model>)

    add_executable(elementwise_to_dpu.cpp $<TARGET_OBJECTS:cm> $<TARGET_OBJECTS:corebase> $<TARGET_OBJECTS:metamodel> $<TARGET_OBJECTS:model>)

    add_executable(cm_linear cm_linear.cpp $<TARGET_OBJECTS:cm> $<TARGET_OBJECTS:corebase> $<TARGET_OBJECTS:metamodel> $<TARGET_OBJECTS:model>)

    add_executable(cm_two_branch cm_two_branch.cpp $<TARGET_OBJECTS:cm> $<TARGET_OBJECTS:corebase> $<TARGET_OBJECTS:metamodel> $<TARGET_OBJECTS:model>)

    add_executable(cm_allocate_resources cm_allocate_resources.cpp $<TARGET_OBJECTS:cm> $<TARGET_OBJECTS:corebase> $<TARGET_OBJECTS:metamodel> $<TARGET_OBJECTS:model>)

    add_executable(cm_resnet50 cm_resnet50.cpp $<TARGET_OBJECTS:cm> $<TARGET_OBJECTS:corebase> $<TARGET_OBJECTS:metamodel> $<TARGET_OBJECTS:model>)

    add_executable(cm_resnet18 cm_resnet18.cpp $<TARGET_OBJECTS:cm> $<TARGET_OBJECTS:corebase> $<TARGET_OBJECTS:metamodel> $<TARGET_OBJECTS:model>)

    add_executable(standalone_softmax standalone_softmax.cpp $<TARGET_OBJECTS:cm> $<TARGET_OBJECTS:corebase> $<TARGET_OBJECTS:metamodel> $<TARGET_OBJECTS:model>)

    add_executable(standalone_tanh standalone_tanh.cpp $<TARGET_OBJECTS:cm> $<TARGET_OBJECTS:corebase> $<TARGET_OBJECTS:metamodel> $<TARGET_OBJECTS:model>)

    add_executable(standalone_sigmoid standalone_sigmoid.cpp $<TARGET_OBJECTS:cm> $<TARGET_OBJECTS:corebase> $<TARGET_OBJECTS:metamodel> $<TARGET_OBJECTS:model>)

    add_executable(standalone_relu standalone_relu.cpp $<TARGET_OBJECTS:cm> $<TARGET_OBJECTS:corebase> $<TARGET_OBJECTS:metamodel> $<TARGET_OBJECTS:model>)

    add_executable(standalone_leaky_relu standalone_leaky_relu.cpp $<TARGET_OBJECTS:cm> $<TARGET_OBJECTS:corebase> $<TARGET_OBJECTS:metamodel> $<TARGET_OBJECTS:model>)

    add_executable(standalone_elu standalone_elu.cpp $<TARGET_OBJECTS:cm> $<TARGET_OBJECTS:corebase> $<TARGET_OBJECTS:metamodel> $<TARGET_OBJECTS:model>)

    add_executable(standalone_prelu standalone_prelu.cpp $<TARGET_OBJECTS:cm> $<TARGET_OBJECTS:corebase> $<TARGET_OBJECTS:metamodel> $<TARGET_OBJECTS:model>)

    add_executable(hw_convolution hw_convolution.cpp $<TARGET_OBJECTS:cm> $<TARGET_OBJECTS:corebase> $<TARGET_OBJECTS:metamodel> $<TARGET_OBJECTS:model>)

    add_executable(z_major_convolution z_major_convolution.cpp $<TARGET_OBJECTS:cm> $<TARGET_OBJECTS:corebase> $<TARGET_OBJECTS:metamodel> $<TARGET_OBJECTS:model>)

    add_executable(dpu_task_example dpu_task_example.cpp $<TARGET_OBJECTS:cm> $<TARGET_OBJECTS:corebase> $<TARGET_OBJECTS:metamodel> $<TARGET_OBJECTS:model>)

    add_executable(barrier_task_api barrier_task_api.cpp $<TARGET_OBJECTS:cm> $<TARGET_OBJECTS:corebase> $<TARGET_OBJECTS:metamodel> $<TARGET_OBJECTS:model>)

    add_executable(depthwise_convolution depthwise_convolution.cpp $<TARGET_OBJECTS:cm> $<TARGET_OBJECTS:corebase> $<TARGET_OBJECTS:metamodel> $<TARGET_OBJECTS:model>)

    add_executable(fully_connected fully_connected.cpp $<TARGET_OBJECTS:cm> $<TARGET_OBJECTS:corebase> $<TARGET_OBJECTS:metamodel> $<TARGET_OBJECTS:model>)

    add_executable(test_f test.cpp $<TARGET_OBJECTS:cm> $<TARGET_OBJECTS:corebase> $<TARGET_OBJECTS:metamodel> $<TARGET_OBJECTS:model>)

    add_executable(dilationExample dilatedExample.cpp $<TARGET_OBJECTS:cm> $<TARGET_OBJECTS:corebase> $<TARGET_OBJECTS:metamodel> $<TARGET_OBJECTS:model>)

    add_executable(barrier_dependency_example barrier_dependency_example.cpp $<TARGET_OBJECTS:cm> $<TARGET_OBJECTS:corebase> $<TARGET_OBJECTS:metamodel> $<TARGET_OBJECTS:model>)

    add_executable(slice_conv slice_conv.cpp  $<TARGET_OBJECTS:cm> $<TARGET_OBJECTS:corebase> $<TARGET_OBJECTS:metamodel> $<TARGET_OBJECTS:model>)

    add_executable(slice_concat slice_concat.cpp  $<TARGET_OBJECTS:cm> $<TARGET_OBJECTS:corebase> $<TARGET_OBJECTS:metamodel> $<TARGET_OBJECTS:model>)

    add_executable(conv_multiSplit conv_multiSplit.cpp  $<TARGET_OBJECTS:cm> $<TARGET_OBJECTS:corebase> $<TARGET_OBJECTS:metamodel> $<TARGET_OBJECTS:model>)

    add_executable(manual_group_conv manual_group_conv.cpp  $<TARGET_OBJECTS:cm> $<TARGET_OBJECTS:corebase> $<TARGET_OBJECTS:metamodel> $<TARGET_OBJECTS:model>)

else()

    add_executable(average_to_depthwise average_to_depthwise.cpp)
    target_link_libraries(average_to_depthwise cm)

    add_executable(elementwise_to_dpu elementwise_to_dpu.cpp)
    target_link_libraries(elementwise_to_dpu cm)

    add_executable(cm_linear cm_linear.cpp)
    target_link_libraries(cm_linear cm)

    add_executable(cm_two_branch cm_two_branch.cpp)
    target_link_libraries(cm_two_branch cm)

    add_executable(cm_allocate_resources cm_allocate_resources.cpp)
    target_link_libraries(cm_allocate_resources cm)

    add_executable(cm_resnet50 cm_resnet50.cpp)
    target_link_libraries(cm_resnet50 cm)

    add_executable(cm_resnet50_keembay cm_resnet50_keembay.cpp)
    target_link_libraries(cm_resnet50_keembay cm)

    add_executable(cm_resnet18 cm_resnet18.cpp)
    target_link_libraries(cm_resnet18 cm)

    add_executable(standalone_softmax standalone_softmax.cpp)
    target_link_libraries(standalone_softmax cm)

    add_executable(standalone_tanh standalone_tanh.cpp)
    target_link_libraries(standalone_tanh cm)

    add_executable(standalone_sigmoid standalone_sigmoid.cpp)
    target_link_libraries(standalone_sigmoid cm)

    add_executable(standalone_relu standalone_relu.cpp)
    target_link_libraries(standalone_relu cm)

    add_executable(standalone_leaky_relu standalone_leaky_relu.cpp)
    target_link_libraries(standalone_leaky_relu cm)

    add_executable(standalone_elu standalone_elu.cpp)
    target_link_libraries(standalone_elu cm)

    add_executable(standalone_prelu standalone_prelu.cpp)
    target_link_libraries(standalone_prelu cm)

    add_executable(hw_convolution_mx hw_convolution_mx.cpp)
    target_link_libraries(hw_convolution_mx cm)

    add_executable(z_major_convolution z_major_convolution.cpp)
    target_link_libraries(z_major_convolution cm)

    add_executable(channel_major_convolution channel_major_convolution.cpp)
    target_link_libraries(channel_major_convolution cm)

    add_executable(manual_yolo_example manual_yolo_example.cpp)
    target_link_libraries(manual_yolo_example cm)

    add_executable(dpu_task_example dpu_task_example.cpp)
    target_link_libraries(dpu_task_example cm)

    add_executable(dpu_conv_3_conv dpu_conv_3_conv.cpp)
    target_link_libraries(dpu_conv_3_conv cm)

    add_executable(fault_dpu_conv_3_conv fault_dpu_conv_3_conv.cpp)
    target_link_libraries(fault_dpu_conv_3_conv cm)

    add_executable(dpu_conv_2_conv dpu_conv_2_conv.cpp)
    target_link_libraries(dpu_conv_2_conv cm)

    add_executable(barrier_task_api barrier_task_api.cpp)
    target_link_libraries(barrier_task_api cm)

    add_executable(barrier_dependency_example barrier_dependency_example.cpp)
    target_link_libraries(barrier_dependency_example cm)

    add_executable(hw_pooling hw_pooling.cpp)
    target_link_libraries(hw_pooling cm)

    add_executable(hw_fullyconnected_convolution hw_fullyconnected_convolution.cpp)
    target_link_libraries(hw_fullyconnected_convolution cm)

    add_executable(depthwise_convolution depthwise_convolution.cpp)
    target_link_libraries(depthwise_convolution cm)

    add_executable(fully_connected fully_connected.cpp)
    target_link_libraries(fully_connected cm)

    add_executable(test_f test.cpp)
    target_link_libraries(test_f cm)

    add_executable(dilationExample dilatedExample.cpp)
    target_link_libraries(dilationExample cm)

    add_executable(slice_conv slice_conv.cpp)
    target_link_libraries(slice_conv cm)

    add_executable(slice_concat slice_concat.cpp)
    target_link_libraries(slice_concat cm)

    add_executable(conv_multiSplit conv_multiSplit.cpp)
    target_link_libraries(conv_multiSplit cm)

    add_executable(manual_group_conv manual_group_conv.cpp)
    target_link_libraries(manual_group_conv cm)

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
