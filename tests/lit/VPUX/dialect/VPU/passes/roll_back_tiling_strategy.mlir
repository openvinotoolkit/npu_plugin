//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --roll-back-tiling-strategy %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @RollBackTOHConv
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x512x14x14xf16, {order = #NHWC}>
func.func @RollBackTOHConv(%arg0: tensor<1x512x14x14xf16, {order = #NHWC}>) -> tensor<1x512x14x14xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<512x512x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x512x1x1xf16, {order = #NHWC}>
    %cst_1 = const.Declare tensor<512x1x1x4xsi32> = dense<1> : tensor<512x1x1x4xsi32>
    %0 = VPU.NCE.Convolution(%arg0, %cst, %cst_1) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        rawFilterShape = [512, 512, 1, 1], strides = [1, 1],
        tilingStrategy = [1, 1, 2, 1]}
            -> tensor<1x512x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x512x14x14xf16, {order = #NHWC}>

     // CHECK-DAG: [[WEIGHTS_0:%.+]] = const.Declare tensor<512x512x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x512x1x1xf16, {order = #NHWC}>
     // CHECK-DAG: [[WT_0:%.+]] = const.Declare tensor<512x1x1x4xsi32> = dense<1> : tensor<512x1x1x4xsi32>

     // CHECK:     [[CONV_0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_0]], [[WT_0]]) {
     // CHECK-SAME:     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
     // CHECK-SAME:     pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
     // CHECK-SAME:     rawFilterShape = [512, 512, 1, 1], strides = [1, 1],
     // CHECK-SAME:     tilingStrategy = [1, 2, 1, 1]}
     // CHECK-SAME:         -> tensor<1x512x14x14xf16, {order = #NHWC}>

     // CHECK:      return [[CONV_0]] : tensor<1x512x14x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: func.func @NotRollBackVF
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x512x14x14xf16, {order = #NHWC}>
func.func @NotRollBackVF(%arg0: tensor<1x512x14x14xf16, {order = #NHWC}>) -> tensor<1x512x14x14xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<512x512x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x512x1x1xf16, {order = #NHWC}>
    %cst_1 = const.Declare tensor<512x1x1x4xsi32> = dense<1> : tensor<512x1x1x4xsi32>
    %0 = VPU.VerticalFusion (
                %arg0 as %arg1:  tensor<1x512x14x14xf16, {order = #NHWC}>,
                %cst_0 as %arg2: tensor<512x512x1x1xf16, {order = #NHWC}>,
                %cst_1 as %arg3: tensor<512x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x512x14x14xf16, {order = #NHWC}> {
        %1 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [512, 512, 1, 1], strides = [1, 1]}
                -> tensor<1x512x14x14xf16, {order = #NHWC}>
        VPU.Yield %1
    }
    return %0 : tensor<1x512x14x14xf16, {order = #NHWC}>

    // CHECK-DAG: [[WEIGHTS_0:%.+]] = const.Declare tensor<512x512x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x512x1x1xf16, {order = #NHWC}>
    // CHECK-DAG: [[WT_0:%.+]] = const.Declare tensor<512x1x1x4xsi32> = dense<1> : tensor<512x1x1x4xsi32>

    // CHECK:     [[VF:%.+]] = VPU.VerticalFusion
    // CHECK-SAME:      tilingStrategy = [1, 1, 2, 1]
    // CHECK:     [[CONV_0:%.+]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3)

    // CHECK:      return [[VF]] : tensor<1x512x14x14xf16, {order = #NHWC}>
}
