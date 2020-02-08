cmake_minimum_required(VERSION 3.5.0)
project(mcmComplier VERSION 0.0.1)

if (MSVC)

    # TODO - implement

else()

    add_executable(single_conv_hde single_conv_hde/single_conv_hde.cpp)
    target_link_libraries(single_conv_hde cm)

    add_executable(argmax_noaxis argmax_noaxis/argmax_noaxis.cpp)
    target_link_libraries(argmax_noaxis cm)

    add_executable(argmax_axis argmax_axis/argmax_axis.cpp)
    target_link_libraries(argmax_axis cm)

    add_executable(conv_clamp conv_clamp/conv_clamp.cpp)
    target_link_libraries(conv_clamp cm)

    add_executable(conv_elu conv_elu/conv_elu.cpp)
    target_link_libraries(conv_elu cm)

    add_executable(conv_fp16output conv_fp16output/conv_fp16output.cpp)
    target_link_libraries(conv_fp16output cm)

    add_executable(conv_fp16input conv_fp16input/conv_fp16input.cpp)
    target_link_libraries(conv_fp16input cm)

    add_executable(conv_with_precision conv_with_precision/conv_with_precision.cpp)
    target_link_libraries(conv_with_precision cm)

    add_executable(conv_leaky_relu conv_leaky_relu/conv_leaky_relu.cpp)
    target_link_libraries(conv_leaky_relu cm)

    add_executable(conv_relu conv_relu/conv_relu.cpp)
    target_link_libraries(conv_relu cm)

    add_executable(unfusable_scale unfusable_scale/unfusable_scale.cpp)
    target_link_libraries(unfusable_scale cm)

    add_executable(conv_relu_channel_scaling conv_relu_channel_scaling/conv_relu_channel_scaling.cpp)
    target_link_libraries(conv_relu_channel_scaling cm)

    add_executable(conv_sigmoid conv_sigmoid/conv_sigmoid.cpp)
    target_link_libraries(conv_sigmoid cm)

    add_executable(conv_softmax conv_softmax/conv_softmax.cpp)
    target_link_libraries(conv_softmax cm)

    add_executable(detection_output detection_output/detection_output.cpp)
    target_link_libraries(detection_output cm)

    add_executable(depthwise_channel_scaling depthwise_channel_scaling/depthwise_channel_scaling.cpp)
    target_link_libraries(depthwise_channel_scaling cm)
    
    add_executable(flatten flatten/flatten.cpp)
    target_link_libraries(flatten cm)

    add_executable(fully_connected_channel_scaling fully_connected_channel_scaling/fully_connected_channel_scaling.cpp)
    target_link_libraries(fully_connected_channel_scaling cm)

    add_executable(normalize normalize/normalize.cpp)
    target_link_libraries(normalize cm)

    add_executable(permute permute/permute.cpp)
    target_link_libraries(permute cm)

    add_executable(priorbox priorbox/priorbox.cpp)
    target_link_libraries(priorbox cm)

    add_executable(proposal_TF proposal_TF/proposal_TF.cpp)
    target_link_libraries(proposal_TF cm)

    add_executable(proposal_caffe proposal_caffe/proposal_caffe.cpp)
    target_link_libraries(proposal_caffe cm)

    add_executable(regionyolo regionyolo/regionyolo.cpp)
    target_link_libraries(regionyolo cm)

    add_executable(reorgyolo reorgyolo/reorgyolo.cpp)
    target_link_libraries(reorgyolo cm)

    add_executable(resample resample/resample.cpp)
    target_link_libraries(resample cm)

    add_executable(reshape reshape/reshape.cpp)
    target_link_libraries(reshape cm)

    add_executable(roi_pooling roi_pooling/roi_pooling.cpp)
    target_link_libraries(roi_pooling cm)

    add_executable(slice slice/slice.cpp)
    target_link_libraries(slice cm)

    add_executable(slice_single slice/slice_single.cpp)
    target_link_libraries(slice_single cm)

    add_executable(softmax softmax/softmax.cpp)
    target_link_libraries(softmax cm)

    add_executable(eltwise_fp16 eltwise_fp16/eltwise_fp16.cpp)
    target_link_libraries(eltwise_fp16 cm)

    add_executable(eltwise_replacement eltwise_replacement/eltwise_replacement.cpp)
    target_link_libraries(eltwise_replacement cm)

    add_executable(interp interp/interp.cpp)
    target_link_libraries(interp cm)

    add_executable(interp_noalign interp_noalign/interp_noalign.cpp)
    target_link_libraries(interp_noalign cm)

    add_executable(norm norm/norm.cpp)
    target_link_libraries(norm cm)
endif(MSVC)
