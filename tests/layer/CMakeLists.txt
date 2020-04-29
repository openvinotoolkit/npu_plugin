cmake_minimum_required(VERSION 3.5.0)
project(mcmComplier VERSION 0.0.1)

if (MSVC)

    # TODO - implement

else()
    add_subdirectory(test_runner)

    add_executable(avgpool_asymmetric_1d_large_kernel avgpool_asymmetric_1d_large_kernel/avgpool_asymmetric_1d_large_kernel.cpp)
    target_link_libraries(avgpool_asymmetric_1d_large_kernel cm)

    add_executable(avgpool_asymmetric_2d_large_kernel avgpool_asymmetric_2d_large_kernel/avgpool_asymmetric_2d_large_kernel.cpp)
    target_link_libraries(avgpool_asymmetric_2d_large_kernel cm)

    add_executable(single_conv_hde_SOK single_conv_hde_SOK/single_conv_hde_SOK.cpp)
    target_link_libraries(single_conv_hde_SOK cm)

    add_executable(single_conv_hde_SOH single_conv_hde_SOH/single_conv_hde_SOH.cpp)
    target_link_libraries(single_conv_hde_SOH cm)

    add_executable(argmax_noaxis argmax_noaxis/argmax_noaxis.cpp)
    target_link_libraries(argmax_noaxis cm)

    add_executable(argmax_axis argmax_axis/argmax_axis.cpp)
    target_link_libraries(argmax_axis cm)

    add_executable(conv_clamp conv_clamp/conv_clamp.cpp)
    target_link_libraries(conv_clamp cm)

    add_executable(conv_elu conv_elu/conv_elu.cpp)
    target_link_libraries(conv_elu cm)

    add_executable(conv_fakequantize conv_fakequantize/conv_fakequantize.cpp)
    target_link_libraries(conv_fakequantize cm)

    add_executable(multiple_output_same_size multiple_output_same_size/multiple_output_same_size.cpp)
    target_link_libraries(multiple_output_same_size cm)

    add_executable(multiple_output_different_sizes multiple_output_different_sizes/multiple_output_different_sizes.cpp)
    target_link_libraries(multiple_output_different_sizes cm)

    add_executable(conv_fp16output conv_fp16output/conv_fp16output.cpp)
    target_link_libraries(conv_fp16output cm)

    add_executable(conv_int32_output conv_int32_output/int32_output_conv.cpp)
    target_link_libraries(conv_int32_output cm)

    add_executable(conv_output_quant conv_output_quant/conv_output_quant.cpp)
    target_link_libraries(conv_output_quant cm)
    add_executable(conv_fp16input conv_fp16input/conv_fp16input.cpp)
    target_link_libraries(conv_fp16input cm)

    add_executable(conv_with_precision conv_with_precision/conv_with_precision.cpp)
    target_link_libraries(conv_with_precision cm)

    add_executable(conv_leaky_relu conv_leaky_relu/conv_leaky_relu.cpp)
    target_link_libraries(conv_leaky_relu cm)

    add_executable(group_convolution group_convolution/group_convolution.cpp)
    target_link_libraries(group_convolution cm)

    add_executable(mixed_mode_eltwise mixed_mode_eltwise/mixed_mode_eltwise.cpp)
    target_link_libraries(mixed_mode_eltwise cm)

    add_executable(conv_sparsity conv_sparsity/conv_sparsity.cpp)
    target_link_libraries(conv_sparsity cm)

    add_executable(conv_relu conv_relu/conv_relu.cpp)
    target_link_libraries(conv_relu cm)

    add_executable(conv_batch conv_batch/conv_batch.cpp)
    target_link_libraries(conv_batch cm)

    add_executable(conv_batch_fasterRCNN conv_batch_fasterRCNN/conv_batch_fasterRCNN.cpp)
    target_link_libraries(conv_batch_fasterRCNN cm)

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

    add_executable(trailing_upatasks trailing_upatasks/trailing_upatasks.cpp)
    target_link_libraries(trailing_upatasks cm)

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
