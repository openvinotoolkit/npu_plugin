//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --optimize-concate-slice-to-slice-concat  %s | FileCheck %s


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @ConcatSliceOptimization
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x16x360x640xf16, {order = #NHWC}>
  func @ConcatSliceOptimization(
            %arg0: tensor<1x16x360x640xf16, {order = #NHWC}>)
                 -> tensor<1x1x360x640xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00>
              : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 16, 52, 640] : tensor<1x16x360x640xf16, {order = #NHWC}>
         to tensor<1x16x52x640xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %weights, %weights_table) {
          multiClusterStrategy = "SplitOverHeight",
          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
          rawFilterShape = [16, 16, 1, 1],
          strides = [1, 1],
          tilingStrategy = [1, 1, 7, 1]} -> tensor<1x16x52x640xf16, {order = #NHWC}> 
    
    %2 = VPU.Slice %arg0 [0, 0, 52, 0] [1, 16, 52, 640] : tensor<1x16x360x640xf16, {order = #NHWC}>
         to tensor<1x16x52x640xf16, {order = #NHWC}>
    %3 = VPU.NCE.Convolution(%2, %weights, %weights_table) {
          multiClusterStrategy = "SplitOverHeight",
          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
          rawFilterShape = [16, 16, 1, 1],
          strides = [1, 1],
          tilingStrategy = [1, 1, 7, 1]} -> tensor<1x16x52x640xf16, {order = #NHWC}> 
    
    %4 = VPU.Slice %arg0 [0, 0, 104, 0] [1, 16, 52, 640] : tensor<1x16x360x640xf16, {order = #NHWC}>
         to tensor<1x16x52x640xf16, {order = #NHWC}>
    %5 = VPU.NCE.Convolution(%4, %weights, %weights_table) {
          multiClusterStrategy = "SplitOverHeight",
          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
          rawFilterShape = [16, 16, 1, 1],
          strides = [1, 1],
          tilingStrategy = [1, 1, 7, 1]} -> tensor<1x16x52x640xf16, {order = #NHWC}> 
    
    %6 = VPU.Slice %arg0 [0, 0, 156, 0] [1, 16, 51, 640] : tensor<1x16x360x640xf16, {order = #NHWC}>
         to tensor<1x16x51x640xf16, {order = #NHWC}>
    %7 = VPU.NCE.Convolution(%6, %weights, %weights_table) {
          multiClusterStrategy = "SplitOverHeight",
          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
          rawFilterShape = [16, 16, 1, 1],
          strides = [1, 1],
          tilingStrategy = [1, 1, 7, 1]} -> tensor<1x16x51x640xf16, {order = #NHWC}> 
    
    %8 = VPU.Slice %arg0 [0, 0, 207, 0] [1, 16, 51, 640] : tensor<1x16x360x640xf16, {order = #NHWC}>
         to tensor<1x16x51x640xf16, {order = #NHWC}>
    %9 = VPU.NCE.Convolution(%8, %weights, %weights_table) {
          multiClusterStrategy = "SplitOverHeight",
          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
          rawFilterShape = [16, 16, 1, 1],
          strides = [1, 1],
          tilingStrategy = [1, 1, 7, 1]} -> tensor<1x16x51x640xf16, {order = #NHWC}> 
    
    %10 = VPU.Slice %arg0 [0, 0, 258, 0] [1, 16, 51, 640] : tensor<1x16x360x640xf16, {order = #NHWC}>
          to tensor<1x16x51x640xf16, {order = #NHWC}>
    %11 = VPU.NCE.Convolution(%10, %weights, %weights_table) {
            multiClusterStrategy = "SplitOverHeight",
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1],
            tilingStrategy = [1, 1, 7, 1]} -> tensor<1x16x51x640xf16, {order = #NHWC}> 
    
    %12 = VPU.Slice %arg0 [0, 0, 309, 0] [1, 16, 51, 640] : tensor<1x16x360x640xf16, {order = #NHWC}>
          to tensor<1x16x51x640xf16, {order = #NHWC}>
    %13 = VPU.NCE.Convolution(%12, %weights, %weights_table) {
            multiClusterStrategy = "SplitOverHeight",
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1], tilingStrategy = [1, 1, 7, 1]} -> tensor<1x16x51x640xf16, {order = #NHWC}> 

    %14 = VPU.Concat(%1, %3, %5, %7, %9, %11, %13) {
            static_offsets = [[0, 0, 0, 0], [0, 0, 52, 0], [0, 0, 104, 0], [0, 0, 156, 0], [0, 0, 207, 0], [0, 0, 258, 0],[0, 0, 309, 0]]}
            : tensor<1x16x52x640xf16, {order = #NHWC}>, tensor<1x16x52x640xf16, {order = #NHWC}>, tensor<1x16x52x640xf16, {order = #NHWC}>,
            tensor<1x16x51x640xf16, {order = #NHWC}>, tensor<1x16x51x640xf16, {order = #NHWC}>, tensor<1x16x51x640xf16, {order = #NHWC}>,
            tensor<1x16x51x640xf16, {order = #NHWC}> -> tensor<1x16x360x640xf16, {order = #NHWC}>    
    
    %output = VPU.Slice %14 [0, 0, 0, 0] [1, 1, 360, 640] : tensor<1x16x360x640xf16, {order = #NHWC}> to tensor<1x1x360x640xf16, {order = #NHWC}>
    return %output : tensor<1x1x360x640xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>     
    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> 

    // Tile 0
    
    // CHECK:       [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 52, 640]
    // CHECK-SAME:         tensor<1x16x360x640xf16, {order = #NHWC}> to tensor<1x16x52x640xf16, {order = #NHWC}>

    // CHECK:       [[CONV0:%.+]] = VPU.NCE.Convolution([[SLICE0]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) 
    // CHECK-SAME:          multiClusterStrategy = "SplitOverHeight"
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK-SAME:          rawFilterShape = [16, 16, 1, 1]
    // CHECK-SAME:          tensor<1x16x52x640xf16, {order = #NHWC}>
    
    // Tile 1

    // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 52, 0] [1, 16, 52, 640]
    // CHECK-SAME:         tensor<1x16x360x640xf16, {order = #NHWC}> to tensor<1x16x52x640xf16, {order = #NHWC}>

    // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[SLICE1]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) 
    // CHECK-SAME:          multiClusterStrategy = "SplitOverHeight"
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK-SAME:          rawFilterShape = [16, 16, 1, 1]
    // CHECK-SAME:          tensor<1x16x52x640xf16, {order = #NHWC}>

    // Tile 2

    // CHECK:       [[SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 104, 0] [1, 16, 52, 640]
    // CHECK-SAME:         tensor<1x16x360x640xf16, {order = #NHWC}> to tensor<1x16x52x640xf16, {order = #NHWC}>

    // CHECK:       [[CONV2:%.+]] = VPU.NCE.Convolution([[SLICE2]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) 
    // CHECK-SAME:          multiClusterStrategy = "SplitOverHeight"
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK-SAME:          rawFilterShape = [16, 16, 1, 1]
    // CHECK-SAME:          tensor<1x16x52x640xf16, {order = #NHWC}>

    // Tile 3

    // CHECK:       [[SLICE3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 156, 0] [1, 16, 51, 640]
    // CHECK-SAME:         tensor<1x16x360x640xf16, {order = #NHWC}> to tensor<1x16x51x640xf16, {order = #NHWC}>

    // CHECK:       [[CONV3:%.+]] = VPU.NCE.Convolution([[SLICE3]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) 
    // CHECK-SAME:          multiClusterStrategy = "SplitOverHeight"
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK-SAME:          rawFilterShape = [16, 16, 1, 1]
    // CHECK-SAME:          tensor<1x16x51x640xf16, {order = #NHWC}>

    // Tile 4

    // CHECK:       [[SLICE4:%.+]] = VPU.Slice [[INPUT]] [0, 0, 207, 0] [1, 16, 51, 640]
    // CHECK-SAME:         tensor<1x16x360x640xf16, {order = #NHWC}> to tensor<1x16x51x640xf16, {order = #NHWC}>

    // CHECK:       [[CONV4:%.+]] = VPU.NCE.Convolution([[SLICE4]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) 
    // CHECK-SAME:          multiClusterStrategy = "SplitOverHeight"
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK-SAME:          rawFilterShape = [16, 16, 1, 1]
    // CHECK-SAME:          tensor<1x16x51x640xf16, {order = #NHWC}>

    // Tile 5

    // CHECK:       [[SLICE5:%.+]] = VPU.Slice [[INPUT]] [0, 0, 258, 0] [1, 16, 51, 640]
    // CHECK-SAME:         tensor<1x16x360x640xf16, {order = #NHWC}> to tensor<1x16x51x640xf16, {order = #NHWC}>

    // CHECK:       [[CONV5:%.+]] = VPU.NCE.Convolution([[SLICE5]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) 
    // CHECK-SAME:          multiClusterStrategy = "SplitOverHeight"
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK-SAME:          rawFilterShape = [16, 16, 1, 1]
    // CHECK-SAME:          tensor<1x16x51x640xf16, {order = #NHWC}>

    // Tile 6

    // CHECK:       [[SLICE6:%.+]] = VPU.Slice [[INPUT]] [0, 0, 309, 0] [1, 16, 51, 640]
    // CHECK-SAME:         tensor<1x16x360x640xf16, {order = #NHWC}> to tensor<1x16x51x640xf16, {order = #NHWC}>

    // CHECK:       [[CONV6:%.+]] = VPU.NCE.Convolution([[SLICE6]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) 
    // CHECK-SAME:          multiClusterStrategy = "SplitOverHeight"
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK-SAME:          rawFilterShape = [16, 16, 1, 1]
    // CHECK-SAME:          tensor<1x16x51x640xf16, {order = #NHWC}>

    // Slice inserted

    // CHECK:       [[INSERTED_SLICE0:%.+]] = VPU.Slice [[CONV0]] [0, 0, 0, 0] [1, 1, 52, 640]
    // CHECK-SAME:         tensor<1x16x52x640xf16, {order = #NHWC}> to tensor<1x1x52x640xf16, {order = #NHWC}>
    // CHECK:       [[INSERTED_SLICE1:%.+]] = VPU.Slice [[CONV1]] [0, 0, 0, 0] [1, 1, 52, 640]
    // CHECK-SAME:         tensor<1x16x52x640xf16, {order = #NHWC}> to tensor<1x1x52x640xf16, {order = #NHWC}>
    // CHECK:       [[INSERTED_SLICE2:%.+]] = VPU.Slice [[CONV2]] [0, 0, 0, 0] [1, 1, 52, 640]
    // CHECK-SAME:         tensor<1x16x52x640xf16, {order = #NHWC}> to tensor<1x1x52x640xf16, {order = #NHWC}>
    // CHECK:       [[INSERTED_SLICE3:%.+]] = VPU.Slice [[CONV3]] [0, 0, 0, 0] [1, 1, 51, 640]
    // CHECK-SAME:         tensor<1x16x51x640xf16, {order = #NHWC}> to tensor<1x1x51x640xf16, {order = #NHWC}>
    // CHECK:       [[INSERTED_SLICE4:%.+]] = VPU.Slice [[CONV4]] [0, 0, 0, 0] [1, 1, 51, 640]
    // CHECK-SAME:         tensor<1x16x51x640xf16, {order = #NHWC}> to tensor<1x1x51x640xf16, {order = #NHWC}>
    // CHECK:       [[INSERTED_SLICE5:%.+]] = VPU.Slice [[CONV5]] [0, 0, 0, 0] [1, 1, 51, 640]
    // CHECK-SAME:         tensor<1x16x51x640xf16, {order = #NHWC}> to tensor<1x1x51x640xf16, {order = #NHWC}>
    // CHECK:       [[INSERTED_SLICE6:%.+]] = VPU.Slice [[CONV6]] [0, 0, 0, 0] [1, 1, 51, 640]
    // CHECK-SAME:         tensor<1x16x51x640xf16, {order = #NHWC}> to tensor<1x1x51x640xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[INSERTED_SLICE0]], [[INSERTED_SLICE1]], [[INSERTED_SLICE2]], [[INSERTED_SLICE3]], [[INSERTED_SLICE4]], [[INSERTED_SLICE5]], [[INSERTED_SLICE6]])

    // CHECK:       return [[CONCAT]] : tensor<1x1x360x640xf16, {order = #NHWC}>
}
