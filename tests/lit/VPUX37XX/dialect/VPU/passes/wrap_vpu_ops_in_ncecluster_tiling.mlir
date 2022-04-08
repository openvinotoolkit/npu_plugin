//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX compilation-mode=DefaultHW" --wrap-vpu-ops-in-ncecluster-tiling %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToNCEClusterTilingSOH
func @ConvToNCEClusterTilingSOH(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    %cst_0 = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = "SplitOverHeight", pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]} -> tensor<1x80x28x28xf16, {order = #NHWC}>
    return %0 : tensor<1x80x28x28xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x64x28x28xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x64x28x28xf16, {order = #NHWC}> -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<80x64x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<80x64x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<80x64x3x3xf16, {order = #NHWC}> -> tensor<80x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<80x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<80x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<80x1x1x4xsi32> -> tensor<80x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_CMX]] as %arg2: tensor<80x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg3: tensor<80x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    //CHECK-SAME:                 rawFilterShape = [80, 64, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x80x28x28xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x80x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToNCEClusterTilingSOK
func @ConvToNCEClusterTilingSOK(%arg0: tensor<1x128x28x28xf16, {order = #NHWC}>) -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    %cst_0 = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = "SplitOverKernel", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, rawFilterShape = [64, 128, 1, 1], strides = [1, 1]} -> tensor<1x64x28x28xf16, {order = #NHWC}>
    return %0 : tensor<1x64x28x28xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x128x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x128x28x28xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x128x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x128x28x28xf16, {order = #NHWC}> -> tensor<1x128x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<64x128x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<64x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<64x128x1x1xf16, {order = #NHWC}> -> tensor<64x128x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<64x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<64x1x1x4xsi32> -> tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x128x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_CMX]] as %arg2: tensor<64x128x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg3: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    //CHECK-SAME:                 rawFilterShape = [64, 128, 1, 1], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x64x28x28xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToNCEClusterTilingClustering
func @ConvToNCEClusterTilingClustering(%arg0: tensor<1x64x14x14xf16, {order = #NHWC}>) -> tensor<1x48x14x14xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<48x1x1x4xsi32> = dense<10> : tensor<48x1x1x4xsi32>
    %cst_0 = const.Declare tensor<48x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = "Clustering", pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [48, 64, 3, 3], strides = [1, 1]} -> tensor<1x48x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x48x14x14xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<48x1x1x4xsi32> = dense<10> : tensor<48x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<48x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x64x14x14xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x64x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x64x14x14xf16, {order = #NHWC}> -> tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<48x64x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<48x64x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<48x64x3x3xf16, {order = #NHWC}> -> tensor<48x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<48x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<48x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<48x1x1x4xsi32> -> tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTS_CMX]] as %arg2: tensor<48x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg3: tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x48x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES2:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    //CHECK-SAME:                 rawFilterShape = [48, 64, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x48x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x48x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x48x14x14xf16, {order = #NHWC}> {
    //CHECK:            [[RES3:%.*]] = VPU.Copy(%arg1) : tensor<1x48x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x48x14x14xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x48x14x14xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvToNCEClusterTilingSOH
func @DepthConvToNCEClusterTilingSOH(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_1 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0, %cst) {multiClusterStrategy = "SplitOverHeight", activation_window_channel_length = 18 : i64, pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[ACTIVATION_WINDOW:%.*]] = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x32x112x112xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x112x112xf16, {order = #NHWC}> -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<32x16x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x16x1x1xf16, {order = #NHWC}> -> tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<32x1x1x4xsi32>) -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32> -> tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[ACTIVATION_WINDOW_CMX:%.*]] = VPU.NCE.ClusterTiling ([[ACTIVATION_WINDOW]] as %arg1: tensor<1x1x1x16xui8>) -> !VPU.DistributedTensor<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES3:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x1x1x16xui8> -> tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTS_CMX]] as %arg2: tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg3: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
    //CHECK-SAME:             [[ACTIVATION_WINDOW_CMX]] as %arg4: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[RES4:%.*]] = VPU.NCE.DepthConvolution(%arg1, %arg2, %arg3, %arg4) {
    //CHECK-SAME:               activation_window_channel_length = 18 : i64, pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling (%4 as %arg1: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    //CHECK:            [[RES5:%.*]] = VPU.Copy(%arg1) : tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x112x112xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES5:%.*]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

// -----


#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvToNCEClusterTilingSOHWithAlign
func @DepthConvToNCEClusterTilingSOHWithAlign(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_1 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0, %cst) {multiClusterStrategy = "SplitOverHeight", activation_window_channel_length = 18 : i64, pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        [[ACTIVATION_WINDOW:%.*]] = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x32x14x14xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x14x14xf16, {order = #NHWC}> -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<32x16x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x16x1x1xf16, {order = #NHWC}> -> tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<32x1x1x4xsi32>) -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32> -> tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[ACTIVATION_WINDOW_CMX:%.*]] = VPU.NCE.ClusterTiling ([[ACTIVATION_WINDOW]] as %arg1: tensor<1x1x1x16xui8>) -> !VPU.DistributedTensor<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES3:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x1x1x16xui8> -> tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTS_CMX]] as %arg2: tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg3: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
    //CHECK-SAME:             [[ACTIVATION_WINDOW_CMX]] as %arg4: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[RES4:%.*]] = VPU.NCE.DepthConvolution(%arg1, %arg2, %arg3, %arg4) {
    //CHECK-SAME:               activation_window_channel_length = 18 : i64, pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    //CHECK:            [[RES5:%.*]] = VPU.Copy(%arg1) : tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x14x14xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES5:%.*]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x14x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @DepthConvToNCEClusterTilingSOK
func @DepthConvToNCEClusterTilingSOK(%arg0: tensor<1x128x56x56xf16, {order = #NHWC}>) -> tensor<1x128x56x56xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    %cst_0 = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>
    %cst_1 = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0, %cst) {multiClusterStrategy = "SplitOverKernel", activation_window_channel_length = 18 : i64, pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [128, 1, 3, 3], strides = [1, 1]} -> tensor<1x128x56x56xf16, {order = #NHWC}>
    return %0 : tensor<1x128x56x56xf16, {order = #NHWC}>

    //CHECK:        [[ACTIVATION_WINDOW:%.*]] = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<128x16x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x128x56x56xf16, {order = #NHWC}>)
    //CHECK-SAME:   !VPU.DistributedTensor<1x128x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x128x56x56xf16, {order = #NHWC}> -> tensor<1x128x56x56xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<128x16x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<128x16x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<128x16x1x1xf16, {order = #NHWC}> -> tensor<128x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<128x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<128x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<128x1x1x4xsi32> -> tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[ACTIVATION_WINDOW_CMX:%.*]] = VPU.NCE.ClusterTiling ([[ACTIVATION_WINDOW]] as %arg1: tensor<1x1x1x16xui8>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES3:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x1x1x16xui8> -> tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x128x56x56xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTS_CMX]] as %arg2: tensor<128x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg3: tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK-SAME:             [[ACTIVATION_WINDOW_CMX]] as %arg4: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x128x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES4:%.*]] = VPU.NCE.DepthConvolution(%arg1, %arg2, %arg3, %arg4) {
    //CHECK-SAME:                 activation_window_channel_length = 18 : i64, pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    //CHECK-SAME:                 rawFilterShape = [128, 1, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x128x56x56xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x128x56x56xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x128x56x56xf16, {order = #NHWC}> {
    //CHECK:            [[RES5:%.*]] = VPU.Copy(%arg1) : tensor<1x128x56x56xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x128x56x56xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES5]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x128x56x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @DepthConvToNCEClusterTilingClustering
func @DepthConvToNCEClusterTilingClustering(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_1 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0, %cst) {multiClusterStrategy = "Clustering", activation_window_channel_length = 18 : i64, pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        [[ACTIVATION_WINDOW:%.*]] = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x32x14x14xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x14x14xf16, {order = #NHWC}> -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<32x16x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x16x1x1xf16, {order = #NHWC}> -> tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<32x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32> -> tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[ACTIVATION_WINDOW_CMX:%.*]] = VPU.NCE.ClusterTiling ([[ACTIVATION_WINDOW]] as %arg1: tensor<1x1x1x16xui8>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES3:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x1x1x16xui8> -> tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTS_CMX]] as %arg2: tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg3: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK-SAME:             [[ACTIVATION_WINDOW_CMX]] as %arg4: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES4:%.*]] = VPU.NCE.DepthConvolution(%arg1, %arg2, %arg3, %arg4) {
    //CHECK-SAME:                 activation_window_channel_length = 18 : i64, pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    //CHECK-SAME:                 rawFilterShape = [32, 1, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    //CHECK:            [[RES5:%.*]] = VPU.Copy(%arg1) : tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x14x14xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES5]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x14x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolToNCEClusterTilingSOH
func @MaxPoolToNCEClusterTilingSOH(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %0 = VPU.NCE.MaxPool(%arg0, %cst_0, %cst) {
            multiClusterStrategy = "SplitOverHeight",
            activation_window_channel_length = 4 : i64,
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            strides = [1, 1],
            kernel_size = [1, 1]
         } -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[ACTIVATION_WINDOW:%.*]] = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x32x112x112xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:        [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x112x112xf16, {order = #NHWC}> -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        VPU.Yield [[RES0]]

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<32x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32> -> tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[ACTIVATION_WINDOW_CMX:%.*]] = VPU.NCE.ClusterTiling ([[ACTIVATION_WINDOW]] as %arg1: tensor<1x1x1x16xui8>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x1x1x16xui8> -> tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg2: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK-SAME:             [[ACTIVATION_WINDOW_CMX]] as %arg3: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[RES3:%.*]] = VPU.NCE.MaxPool(%arg1, %arg2, %arg3) {activation_window_channel_length = 4 : i64, kernel_size = [1, 1], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, strides = [1, 1]} -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x112x112xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolToNCEClusterTilingSOHWithAlign
func @MaxPoolToNCEClusterTilingSOHWithAlign(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %0 = VPU.NCE.MaxPool(%arg0, %cst_0, %cst) {
            multiClusterStrategy = "SplitOverHeight",
            activation_window_channel_length = 4 : i64,
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            strides = [1, 1],
            kernel_size = [1, 1]
         } -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        [[ACTIVATION_WINDOW:%.*]] = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x32x14x14xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}> {
    //CHECK:        [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x14x14xf16, {order = #NHWC}> -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        VPU.Yield [[RES0]]

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<32x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32> -> tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[ACTIVATION_WINDOW_CMX:%.*]] = VPU.NCE.ClusterTiling ([[ACTIVATION_WINDOW]] as %arg1: tensor<1x1x1x16xui8>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x1x1x16xui8> -> tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg2: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK-SAME:             [[ACTIVATION_WINDOW_CMX]] as %arg3: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[RES3:%.*]] = VPU.NCE.MaxPool(%arg1, %arg2, %arg3) {activation_window_channel_length = 4 : i64, kernel_size = [1, 1], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, strides = [1, 1]} -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x14x14xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x14x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolToNCEClusterTilingClustering
func @MaxPoolToNCEClusterTilingClustering(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %0 = VPU.NCE.MaxPool(%arg0, %cst_0, %cst) {
            multiClusterStrategy = "Clustering",
            activation_window_channel_length = 4 : i64,
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            strides = [1, 1],
            kernel_size = [1, 1]
         } -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        [[ACTIVATION_WINDOW:%.*]] = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x32x14x14xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:        [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x14x14xf16, {order = #NHWC}> -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        VPU.Yield [[RES0]]

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<32x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32> -> tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[ACTIVATION_WINDOW_CMX:%.*]] = VPU.NCE.ClusterTiling ([[ACTIVATION_WINDOW]] as %arg1: tensor<1x1x1x16xui8>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x1x1x16xui8> -> tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg2: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK-SAME:             [[ACTIVATION_WINDOW_CMX]] as %arg3: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES3:%.*]] = VPU.NCE.MaxPool(%arg1, %arg2, %arg3) {activation_window_channel_length = 4 : i64, kernel_size = [1, 1], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, strides = [1, 1]} -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x14x14xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x14x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAddToNCEClusterTilingSOH
func @EltwiseAddToNCEClusterTilingSOH(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>, %arg1: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) { multiClusterStrategy = "SplitOverHeight", op_type = "ADD" } :
         tensor<1x32x112x112xf16, {order = #NHWC}>, tensor<1x32x112x112xf16, {order = #NHWC}>
         -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0: tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[INPUT0_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg2: tensor<1x32x112x112xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x112x112xf16, {order = #NHWC}> -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[INPUT1_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg1 as %arg2: tensor<1x32x112x112xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x112x112xf16, {order = #NHWC}> -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0_CMX]] as %arg2: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[INPUT1_CMX]] as %arg3: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[RES2:%.*]] = VPU.NCE.Eltwise(%arg2, %arg3) {op_type = "ADD"} -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg2: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    //CHECK:            [[RES3:%.*]] = VPU.Copy(%arg2) : tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x112x112xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }
    //CHECK:        return [[OUT]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAddToNCEClusterTilingClustering
func @EltwiseAddToNCEClusterTilingClustering(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>, %arg1: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) { multiClusterStrategy = "Clustering", op_type = "ADD" } :
         tensor<1x32x14x14xf16, {order = #NHWC}>, tensor<1x32x14x14xf16, {order = #NHWC}>
         -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0: tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        [[INPUT0_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg2: tensor<1x32x14x14xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x14x14xf16, {order = #NHWC}> -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[INPUT1_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg1 as %arg2: tensor<1x32x14x14xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x14x14xf16, {order = #NHWC}> -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0_CMX]] as %arg2: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[INPUT1_CMX]] as %arg3: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    //CHECK:            [[RES2:%.*]] = VPU.NCE.Eltwise(%arg2, %arg3) {op_type = "ADD"} -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg2: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    //CHECK:            [[RES3:%.*]] = VPU.Copy(%arg2) : tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x14x14xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x14x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AvgPoolToNCEClusterTilingSOH
func @AvgPoolToNCEClusterTilingSOH(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {
            multiClusterStrategy = "SplitOverHeight",
            pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
            strides = [1, 1],
            kernel_size = [3, 3]
         } -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x32x112x112xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:        [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x112x112xf16, {order = #NHWC}> -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        VPU.Yield [[RES0]]

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[RES3:%.*]] = VPU.NCE.AveragePool(%arg1) {kernel_size = [3, 3], pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, strides = [1, 1]} -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x112x112xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AvgPoolToNCEClusterTilingSOHWithAlign
func @AvgPoolToNCEClusterTilingSOHWithAlign(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {
            multiClusterStrategy = "SplitOverHeight",
            pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
            strides = [1, 1],
            kernel_size = [3, 3]
         } -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x32x14x14xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}> {
    //CHECK:        [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x14x14xf16, {order = #NHWC}> -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        VPU.Yield [[RES0]]

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[RES3:%.*]] = VPU.NCE.AveragePool(%arg1) {kernel_size = [3, 3], pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, strides = [1, 1]} -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x14x14xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x14x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SparseConvToNCEClusterTilingSOH
func @SparseConvToNCEClusterTilingSOH(%arg0 : tensor<1x64x28x28xf16, {order = #NHWC}>, %arg1 : tensor<1x64x28x28xi1, {order = #NHWC}>)
        -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {

    %input_sparse = VPU.GroupSparseTensor(%arg0, %arg1)
        -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>

    %weights = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<80x1x1x640xi1> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {is_weights}
        -> !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>,
                             sparsity_map=tensor<80x1x1x640xi1>, is_weights>

    %weights_table = const.Declare tensor<80x1x1x4xsi32> = dense<1> : tensor<80x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%input_sparse, %weights_sparse, %weights_table) {
            multiClusterStrategy = "SplitOverHeight",
            pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 01 : i64},
            rawFilterShape = [80, 64, 3, 3],
            strides = [1, 1]
        } -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                               sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {
            VPU.DPU.Workload [0, 0, 0, 0] [1, 32, 16, 16] {bottom = 0, left = 0, right = 0, top = 0} "VECTOR_FP16"
        }

    return %0 : !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                                  sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>>

    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, %arg1)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>

    // CHECK:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK:       [[CST_WEIGHTS_SM:%.+]] = const.Declare tensor<80x1x1x640xi1> = dense<1.000000e+00>
    // CHECK:       [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]]) {is_weights}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<80x1x1x640xi1>, is_weights>

    // CHECK:       [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<80x1x1x4xsi32>

    // CHECK:       [[INPUT_SPARSE_CMX:%.+]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:      ([[INPUT_SPARSE]] as [[INPUT_SPARSE_ARG:%.+]]: !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                                                                       sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                           sparsity_map=!VPU.DistributedTensor<1x64x28x28xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>> {
    // CHECK:         [[VAR0:%.+]] = VPU.Copy([[INPUT_SPARSE_ARG]]) {out_mem_space = @CMX_NN}
    // CHECK:         VPU.Yield [[VAR0]]
    // CHECK:       }

    // CHECK:       [[WEIGHTS_SPARSE_CMX:%.+]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:      ([[WEIGHTS_SPARSE]] as [[WEIGHTS_SPARSE_ARG:%.+]]: !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                                                                           sparsity_map=tensor<80x1x1x640xi1>, is_weights>)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=!VPU.DistributedTensor<80x64x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
    // CHECK-SAME:                           sparsity_map=!VPU.DistributedTensor<80x1x1x640xi1, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
    // CHECK-SAME:                           is_weights> {
    // CHECK:         [[VAR1:%.+]] = VPU.Copy([[WEIGHTS_SPARSE_ARG]]) {out_mem_space = @CMX_NN}
    // CHECK:         VPU.Yield [[VAR1]]
    // CHECK:       }

    // CHECK:       [[WEIGHTS_TABLE_CMX:%.+]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:      ([[CST_WEIGHTS_TABLE]] as [[WEIGHTS_TABLE_ARG:%.+]]: tensor<80x1x1x4xsi32>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<80x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK:         [[VAR2:%.+]] = VPU.Copy([[WEIGHTS_TABLE_ARG]]) {out_mem_space = @CMX_NN}
    // CHECK:         VPU.Yield [[VAR2]]
    // CHECK:       }

    // CHECK:       [[OUT_CMX:%.+]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:      [[INPUT_SPARSE_CMX]] as [[INPUT_SPARSE_CMX_ARG:[^:]+]]:
    // CHECK-SAME:          !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                            sparsity_map=tensor<1x64x28x28xi1, {mem_space = @CMX_NN, order = #NHWC}>>,
    // CHECK-SAME:      [[WEIGHTS_SPARSE_CMX]] as [[WEIGHTS_SPARSE_CMX_ARG:[^:]+]]:
    // CHECK-SAME:          !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                            sparsity_map=tensor<80x1x1x640xi1, {mem_space = @CMX_NN, order = #NCHW}>, is_weights>,
    // CHECK-SAME:      [[WEIGHTS_TABLE_CMX]] as [[WEIGHTS_TABLE_CMX_ARG:[^:]+]]: tensor<80x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                           sparsity_map=!VPU.DistributedTensor<1x80x28x28xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>> {
    // CHECK:         [[VAR3:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE_CMX_ARG]], [[WEIGHTS_SPARSE_CMX_ARG]], [[WEIGHTS_TABLE_CMX_ARG]])
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x80x28x28xi1, {mem_space = @CMX_NN, order = #NHWC}>>
    // CHECK:         VPU.Yield [[VAR3]]
    // CHECK:       }

    // CHECK:       [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[OUT_CMX_ARG:[^:]+]]:
    // CHECK-SAME:      !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                        sparsity_map=tensor<1x80x28x28xi1, {mem_space = @CMX_NN, order = #NHWC}>>)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>, sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {
    // CHECK:         [[VAR4:%.+]] = VPU.Copy([[OUT_CMX_ARG]])
    // CHECK:         VPU.Yield [[VAR4]]
    // CHECK:       }

    // CHECK:       return [[OUT]] : !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                                     sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSOHAlignmentForEltwiseCase1
func @OptimizeSOHAlignmentForEltwiseCase1(%arg0: tensor<1x16x22x22xf16, {order = #NHWC}>, %arg1: tensor<1x16x22x22xf16, {order = #NHWC}>) -> tensor<1x16x22x22xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    %cst_0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = "SplitOverHeight", pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [16, 16, 3, 3], strides = [1, 1]} -> tensor<1x16x22x22xf16, {order = #NHWC}> 
    %1 = VPU.NCE.Eltwise(%arg0, %arg1) {multiClusterStrategy = "SplitOverHeight", op_type = "ADD"} -> tensor<1x16x22x22xf16, {order = #NHWC}>
    %2 = VPU.NCE.Eltwise(%0, %1) {op_type = "ADD"} -> tensor<1x16x22x22xf16, {order = #NHWC}>
    return %2 : tensor<1x16x22x22xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX_0:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg2: tensor<1x16x22x22xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x16x22x22xf16, {order = #NHWC}> -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg2: tensor<16x16x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<16x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<16x16x3x3xf16, {order = #NHWC}> -> tensor<16x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg2: tensor<16x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<16x1x1x4xsi32> -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX_0:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX_0]] as %arg2: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_CMX]] as %arg3: tensor<16x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg4: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg2, %arg3, %arg4) {
    //CHECK-SAME:                 pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    //CHECK-SAME:                 rawFilterShape = [16, 16, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_0:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX_0]] as %arg2: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x16x22x22xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg2) : tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x22x22xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[INPUT0_CMX_1:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg2: tensor<1x16x22x22xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x16x22x22xf16, {order = #NHWC}> -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[INPUT1_CMX_1:%.*]] = VPU.NCE.ClusterTiling (%arg1 as %arg2: tensor<1x16x22x22xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x16x22x22xf16, {order = #NHWC}> -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX_1:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0_CMX_1]] as %arg2: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[INPUT1_CMX_1]] as %arg3: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}> {
    //CHECK:            [[RES2:%.*]] = VPU.NCE.Eltwise(%arg2, %arg3) {op_type = "ADD"} -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_1:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX_1]] as %arg2: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x16x22x22xf16, {order = #NHWC}> {
    //CHECK:            [[RES3:%.*]] = VPU.Copy(%arg2) : tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x22x22xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_2:%.*]] = VPU.NCE.Eltwise([[OUT_0]], [[OUT_1]]) {op_type = "ADD"} -> tensor<1x16x22x22xf16, {order = #NHWC}>

    //CHECK:        return [[OUT_2]] : tensor<1x16x22x22xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSOHAlignmentForEltwiseCase2
func @OptimizeSOHAlignmentForEltwiseCase2(%arg0: tensor<1x16x22x22xf16, {order = #NHWC}>, %arg1: tensor<1x16x22x22xf16, {order = #NHWC}>) -> tensor<1x16x22x22xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {multiClusterStrategy = "SplitOverHeight", op_type = "ADD"} -> tensor<1x16x22x22xf16, {order = #NHWC}>
    %1 = VPU.NCE.Eltwise(%0, %arg1) {multiClusterStrategy = "SplitOverHeight", op_type = "ADD"} -> tensor<1x16x22x22xf16, {order = #NHWC}>
    %cst = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    %cst_0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %2 = VPU.NCE.Convolution(%1, %cst_0, %cst) {multiClusterStrategy = "SplitOverHeight", pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [16, 16, 3, 3], strides = [1, 1]} -> tensor<1x16x22x22xf16, {order = #NHWC}> 
    return %2 : tensor<1x16x22x22xf16, {order = #NHWC}>

    //CHECK:        [[INPUT0_CMX_0:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg2: tensor<1x16x22x22xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x16x22x22xf16, {order = #NHWC}> -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[INPUT1_CMX_0:%.*]] = VPU.NCE.ClusterTiling (%arg1 as %arg2: tensor<1x16x22x22xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x16x22x22xf16, {order = #NHWC}> -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        } 

    //CHECK:        [[OUT_CMX_0:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0_CMX_0]] as %arg2: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[INPUT1_CMX_0]] as %arg3: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}> {
    //CHECK:            [[RES2:%.*]] = VPU.NCE.Eltwise(%arg2, %arg3) {op_type = "ADD"} -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_0:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX_0]] as %arg2: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x16x22x22xf16, {order = #NHWC}> {
    //CHECK:            [[RES3:%.*]] = VPU.Copy(%arg2) : tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x22x22xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[INPUT0_CMX_1:%.*]] = VPU.NCE.ClusterTiling ([[OUT_0]] as %arg2: tensor<1x16x22x22xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x16x22x22xf16, {order = #NHWC}> -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[INPUT1_CMX_1:%.*]] = VPU.NCE.ClusterTiling (%arg1 as %arg2: tensor<1x16x22x22xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x16x22x22xf16, {order = #NHWC}> -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        } 

    //CHECK:        [[OUT_CMX_1:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0_CMX_1]] as %arg2: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[INPUT1_CMX_1]] as %arg3: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}> {
    //CHECK:            [[RES2:%.*]] = VPU.NCE.Eltwise(%arg2, %arg3) {op_type = "ADD"} -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_1:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX_1]] as %arg2: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x16x22x22xf16, {order = #NHWC}> {
    //CHECK:            [[RES3:%.*]] = VPU.Copy(%arg2) : tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x22x22xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX_2:%.*]] = VPU.NCE.ClusterTiling ([[OUT_1]] as %arg2: tensor<1x16x22x22xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x16x22x22xf16, {order = #NHWC}> -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg2: tensor<16x16x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<16x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<16x16x3x3xf16, {order = #NHWC}> -> tensor<16x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg2: tensor<16x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<16x1x1x4xsi32> -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX_2:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX_2]] as %arg2: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_CMX]] as %arg3: tensor<16x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg4: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg2, %arg3, %arg4) {
    //CHECK-SAME:                 pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    //CHECK-SAME:                 rawFilterShape = [16, 16, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_2:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX_2]] as %arg2: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x16x22x22xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg2) : tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x22x22xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT_2]] : tensor<1x16x22x22xf16, {order = #NHWC}>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @MVNToNCEClusterTilingDuplicateBuffer
func @MVNToNCEClusterTilingDuplicateBuffer(%arg0: tensor<1x4x512x1xf16, {order = #NCWH}>) -> tensor<1x4x512x1xf16, {order = #NCWH}> {

    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, multiClusterStrategy = "Clustering", normalize_variance = true} : tensor<1x4x512x1xf16, {order = #NCWH}> -> tensor<1x4x512x1xf16, {order = #NCWH}>

    return %0: tensor<1x4x512x1xf16, {order = #NCWH}>

    //CHECK: [[ClusterCopy:%.*]] = VPU.NCE.ClusterTiling (%arg0 as [[ARG1:%.*]]: tensor<1x4x512x1xf16, {order = #NCWH}>) -> !VPU.DistributedTensor<1x4x512x1xf16, #NCWH, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK: [[RCopy:%*]] = VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x4x512x1xf16, {order = #NCWH}> -> tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}>
    //CHECK: VPU.Yield [[RCopy]]

    //CHECK: [[RClusterMVN:%.*]] = VPU.NCE.ClusterTiling ([[ClusterCopy]] as [[ARG2:%.*]]: tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}>) -> !VPU.DistributedTensor<1x4x512x1xf16, #NCWH, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK: [[RMVN:%*]] = VPU.MVN([[ARG2]]) {across_channels = false, eps = 1.0013580322265625E-5 : f64, normalize_variance = true} : tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}> -> tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}>
    //CHECK: VPU.Yield [[RMVN]]

    //CHECK: [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[RClusterMVN]] as [[ARG3:%.*]]: tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}>) -> tensor<1x4x512x1xf16, {order = #NCWH}> {
    //CHECK: [[RCopy1:%*]] =  VPU.Copy([[ARG3]]) : tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}> -> tensor<1x4x512x1xf16, {order = #NCWH}>
    //CHECK: VPU.Yield [[RCopy1]]

    //CHECK: return [[OUT]] : tensor<1x4x512x1xf16, {order = #NCWH}>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @MVNToNCEClusterTilingSegmentedBuffer
func @MVNToNCEClusterTilingSegmentedBuffer(%arg0: tensor<1x4x512x1xf16, {order = #NCWH}>) -> tensor<1x4x512x1xf16, {order = #NCWH}> {

    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, multiClusterStrategy = "SplitOverKernel", normalize_variance = true} : tensor<1x4x512x1xf16, {order = #NCWH}> -> tensor<1x4x512x1xf16, {order = #NCWH}>

    return %0: tensor<1x4x512x1xf16, {order = #NCWH}>

    //CHECK: [[ClusterCopy:%.*]] = VPU.NCE.ClusterTiling (%arg0 as [[ARG1:%.*]]: tensor<1x4x512x1xf16, {order = #NCWH}>) -> !VPU.DistributedTensor<1x4x512x1xf16, #NCWH, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK: [[RCopy:%*]] = VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x4x512x1xf16, {order = #NCWH}> -> tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}>
    //CHECK: VPU.Yield [[RCopy]]

    //CHECK: [[RClusterMVN:%.*]] = VPU.NCE.ClusterTiling ([[ClusterCopy]] as [[ARG2:%.*]]: tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}>) -> !VPU.DistributedTensor<1x4x512x1xf16, #NCWH, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    //CHECK: [[RMVN:%*]] = VPU.MVN([[ARG2]]) {across_channels = false, eps = 1.0013580322265625E-5 : f64, normalize_variance = true} : tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}> -> tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}>
    //CHECK: VPU.Yield [[RMVN]]

    //CHECK: [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[RClusterMVN]] as [[ARG3:%.*]]: tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}>) -> tensor<1x4x512x1xf16, {order = #NCWH}> {
    //CHECK: [[RCopy1:%*]] =  VPU.Copy([[ARG3]]) : tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}> -> tensor<1x4x512x1xf16, {order = #NCWH}>
    //CHECK: VPU.Yield [[RCopy1]]

    //CHECK: return [[OUT]] : tensor<1x4x512x1xf16, {order = #NCWH}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @UnrollSOKConvOutputSegmented
func @UnrollSOKConvOutputSegmented(%input: tensor<1x64x64x64xf16, {order = #NHWC}>) -> tensor<1x64x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) {
            multiClusterStrategy = "SplitOverKernel",
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [64, 64, 1, 1], strides = [1, 1]}
                -> tensor<1x64x64x64xf16, {order = #NHWC}>
    %mvn = VPU.MVN(%conv) {
            across_channels = false, eps = 1.0E-4 : f64,
            multiClusterStrategy = "SplitOverKernel", normalize_variance = true
            } : tensor<1x64x64x64xf16, {order = #NHWC}>
                -> tensor<1x64x64x64xf16, {order = #NHWC}>

    return %mvn : tensor<1x64x64x64xf16, {order = #NHWC}>

    // (DUP) CONV (SEG) -> (SEG) MVN (SEG)

    //CHECK:        [[CONV_IN:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x64x64x64xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {

    //CHECK:        [[SOK_CONV:%.*]] = VPU.NCE.ClusterTiling ([[CONV_IN]] as %arg1: tensor<1x64x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>, %1 as %arg2: tensor<64x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>, %2 as %arg3: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[MVN_IN:%.*]] = VPU.NCE.ClusterTiling (%4 as %arg1: tensor<1x64x64x64xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {

    //CHECK:        [[SOK_MVN:%.*]] = VPU.NCE.ClusterTiling (%5 as %arg1: tensor<1x64x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @UnrollSOKDWConvInputOutputSegmented
func @UnrollSOKDWConvInputOutputSegmented(%input: tensor<1x64x64x64xf16, {order = #NHWC}>) -> tensor<1x64x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    %act_win = const.Declare tensor<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>

    %mvn = VPU.MVN(%input) {across_channels = false, eps = 1.0E-4 : f64, multiClusterStrategy = "SplitOverKernel", normalize_variance = true}
            : tensor<1x64x64x64xf16, {order = #NHWC}> -> tensor<1x64x64x64xf16, {order = #NHWC}>
    %dwconv = VPU.NCE.DepthConvolution(%mvn, %weights, %weights_table, %act_win) {
            activation_window_channel_length = 4 : i64, multiClusterStrategy = "SplitOverKernel",
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [64, 1, 1, 1], strides = [1, 1]}
                -> tensor<1x64x64x64xf16, {order = #NHWC}>

    return %dwconv : tensor<1x64x64x64xf16, {order = #NHWC}>

    // (SEG) MVN (SEG) -> (SEG) DWCONV (SEG)

    //CHECK:        [[MVN_IN:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x64x64x64xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {

    //CHECK:        [[SOK_MVN:%.*]] = VPU.NCE.ClusterTiling ([[MVN_IN]] as %arg1: tensor<1x64x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {

    //CHECK:        [[DWCONV_IN:%.*]] = VPU.NCE.ClusterTiling (%2 as %arg1: tensor<1x64x64x64xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {

    //CHECK:        [[SOK_DWCONV:%.*]] = VPU.NCE.ClusterTiling (%3 as %arg1: tensor<1x64x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>, %4 as %arg2: tensor<64x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>, %5 as %arg3: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>, %6 as %arg4: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {

}
