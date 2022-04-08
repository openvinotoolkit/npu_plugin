//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX compilation-mode=DefaultHW" --wrap-vpu-ops-in-ncecluster-tiling %s | FileCheck %s

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MVNInputAlignForSegmentedInput
func @MVNInputAlignForSegmentedInput(%input: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x32x64x64xf16, {order = #NWHC}> {
    %weights = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) {
            multiClusterStrategy = "SplitOverKernel",
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [32, 32, 1, 1], strides = [1, 1]}
                -> tensor<1x32x64x64xf16, {order = #NHWC}>
    %permute_cast = VPU.PermuteCast(%conv) {
            dst_order = #NWHC,
            mem_perm = #NCHW}
            : tensor<1x32x64x64xf16, {order = #NHWC}>
                -> tensor<1x32x64x64xf16, {order = #NWHC}>
    %mvn = VPU.MVN(%permute_cast) {
            across_channels = false, eps = 1.0E-4 : f64,
            multiClusterStrategy = "Clustering", normalize_variance = true
            } : tensor<1x32x64x64xf16, {order = #NWHC}>
                -> tensor<1x32x64x64xf16, {order = #NWHC}>

    return %mvn : tensor<1x32x64x64xf16, {order = #NWHC}>

    //CHECK:        [[CLUSTERED_CONV:%.*]] = VPU.NCE.ClusterTiling (%0 as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>, %1 as %arg2: tensor<32x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>, %2 as %arg3: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[CLUSTERED_COPY_TO_MVN:%.*]] = VPU.NCE.ClusterTiling (%5 as %arg1: tensor<1x32x64x64xf16, {order = #NWHC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x64x64xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[CLUSTERED_MVN:%.*]] = VPU.NCE.ClusterTiling (%6 as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NWHC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x64x64xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SOKMVNInputAlignForSegmentedInput
func @SOKMVNInputAlignForSegmentedInput(%input: tensor<1x64x64x64xf16, {order = #NHWC}>) -> tensor<1x64x64x64xf16, {order = #NWHC}> {
    %weights = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) {
            multiClusterStrategy = "SplitOverKernel",
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [64, 64, 1, 1], strides = [1, 1]}
                -> tensor<1x64x64x64xf16, {order = #NHWC}>
    %permute_cast = VPU.PermuteCast(%conv) {
            dst_order = #NWHC,
            mem_perm = #NCHW}
            : tensor<1x64x64x64xf16, {order = #NHWC}>
                -> tensor<1x64x64x64xf16, {order = #NWHC}>
    %mvn = VPU.MVN(%permute_cast) {
            across_channels = false, eps = 1.0E-4 : f64,
            multiClusterStrategy = "SplitOverKernel", normalize_variance = true
            } : tensor<1x64x64x64xf16, {order = #NWHC}>
                -> tensor<1x64x64x64xf16, {order = #NWHC}>

    return %mvn : tensor<1x64x64x64xf16, {order = #NWHC}>

    //CHECK:        [[CLUSTERED_CONV:%.*]] = VPU.NCE.ClusterTiling (%0 as %arg1: tensor<1x64x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>, %1 as %arg2: tensor<64x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>, %2 as %arg3: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[CLUSTERED_COPY_TO_MVN:%.*]] = VPU.NCE.ClusterTiling (%5 as %arg1: tensor<1x64x64x64xf16, {order = #NWHC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NWHC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[CLUSTERED_MVN:%.*]] = VPU.NCE.ClusterTiling (%6 as %arg1: tensor<1x64x64x64xf16, {mem_space = @CMX_NN, order = #NWHC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NWHC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @NoSOKMVNInputAlignForSegmentedInput
func @NoSOKMVNInputAlignForSegmentedInput(%input: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x32x64x64xf16, {order = #NWHC}> {
    %weights = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) {
            multiClusterStrategy = "SplitOverKernel",
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [32, 32, 1, 1], strides = [1, 1]}
                -> tensor<1x32x64x64xf16, {order = #NHWC}>
    %permute_cast = VPU.PermuteCast(%conv) {
            dst_order = #NWHC,
            mem_perm = #NCHW}
            : tensor<1x32x64x64xf16, {order = #NHWC}>
                -> tensor<1x32x64x64xf16, {order = #NWHC}>
    %mvn = VPU.MVN(%permute_cast) {
            across_channels = false, eps = 1.0E-4 : f64,
            multiClusterStrategy = "SplitOverKernel", normalize_variance = true
            } : tensor<1x32x64x64xf16, {order = #NWHC}>
                -> tensor<1x32x64x64xf16, {order = #NWHC}>

    return %mvn : tensor<1x32x64x64xf16, {order = #NWHC}>

    //CHECK:        [[CLUSTERED_CONV:%.*]] = VPU.NCE.ClusterTiling (%0 as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>, %1 as %arg2: tensor<32x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>, %2 as %arg3: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {

    //CHECK:        [[CLUSTERED_COPY_TO_MVN:%.*]] = VPU.NCE.ClusterTiling (%5 as %arg1: tensor<1x32x64x64xf16, {order = #NWHC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x64x64xf16, #NWHC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[CLUSTERED_MVN:%.*]] = VPU.NCE.ClusterTiling (%6 as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NWHC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x64x64xf16, #NWHC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @TanhInputAlignForSegmentedInput
func @TanhInputAlignForSegmentedInput(%input: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x32x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) {
            multiClusterStrategy = "SplitOverKernel",
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [32, 32, 1, 1], strides = [1, 1]}
                -> tensor<1x32x64x64xf16, {order = #NHWC}>
    %tanh = VPU.Tanh(%conv) {multiClusterStrategy = "Clustering"} : tensor<1x32x64x64xf16, {order = #NHWC}> -> tensor<1x32x64x64xf16, {order = #NHWC}>

    return %tanh : tensor<1x32x64x64xf16, {order = #NHWC}>

    //CHECK:        [[CLUSTERED_CONV:%.*]] = VPU.NCE.ClusterTiling (%0 as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>, %1 as %arg2: tensor<32x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>, %2 as %arg3: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[CLUSTERED_COPY_TO_TANH:%.*]] = VPU.NCE.ClusterTiling (%4 as %arg1: tensor<1x32x64x64xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[CLUSTERED_TANH:%.*]] = VPU.NCE.ClusterTiling (%5 as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SOKTanhInputAlignForSegmentedInput
func @SOKTanhInputAlignForSegmentedInput(%input: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x32x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) {
            multiClusterStrategy = "SplitOverKernel",
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [32, 32, 1, 1], strides = [1, 1]}
                -> tensor<1x32x64x64xf16, {order = #NHWC}>
    %tanh = VPU.Tanh(%conv) {multiClusterStrategy = "SplitOverKernel"} : tensor<1x32x64x64xf16, {order = #NHWC}> -> tensor<1x32x64x64xf16, {order = #NHWC}>

    return %tanh : tensor<1x32x64x64xf16, {order = #NHWC}>

    //CHECK:        [[CLUSTERED_CONV:%.*]] = VPU.NCE.ClusterTiling (%0 as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>, %1 as %arg2: tensor<32x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>, %2 as %arg3: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[CLUSTERED_COPY_TO_TANH:%.*]] = VPU.NCE.ClusterTiling (%4 as %arg1: tensor<1x32x64x64xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[CLUSTERED_TANH:%.*]] = VPU.NCE.ClusterTiling (%5 as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MVNOutputAlignForSegmentedOutput
func @MVNOutputAlignForSegmentedOutput(%input: tensor<1x64x64x64xf16, {order = #NWHC}>) -> tensor<1x64x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    %act_win = const.Declare tensor<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>

    %mvn = VPU.MVN(%input) {across_channels = false, eps = 1.0E-4 : f64, multiClusterStrategy = "SplitOverKernel", normalize_variance = true}
            : tensor<1x64x64x64xf16, {order = #NWHC}> -> tensor<1x64x64x64xf16, {order = #NWHC}>
    %permute_cast = VPU.PermuteCast(%mvn) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x64x64x64xf16, {order = #NWHC}> -> tensor<1x64x64x64xf16, {order = #NHWC}>
    %dwconv = VPU.NCE.DepthConvolution(%permute_cast, %weights, %weights_table, %act_win) {
            activation_window_channel_length = 4 : i64, multiClusterStrategy = "SplitOverKernel",
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [64, 1, 1, 1], strides = [1, 1]}
                -> tensor<1x64x64x64xf16, {order = #NHWC}>

    return %dwconv : tensor<1x64x64x64xf16, {order = #NHWC}>

    //CHECK:        [[CLUSTERED_MVN:%.*]] = VPU.NCE.ClusterTiling (%0 as %arg1: tensor<1x64x64x64xf16, {mem_space = @CMX_NN, order = #NWHC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NWHC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[CLUSTERED_DW_CONV:%.*]] = VPU.NCE.ClusterTiling (%4 as %arg1: tensor<1x64x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>, %5 as %arg2: tensor<64x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>, %6 as %arg3: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>, %7 as %arg4: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @NoMVNOutputAlignForSegmentedOutput
func @NoMVNOutputAlignForSegmentedOutput(%input: tensor<1x32x64x64xf16, {order = #NWHC}>) -> tensor<1x32x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %act_win = const.Declare tensor<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>

    %mvn = VPU.MVN(%input) {across_channels = false, eps = 1.0E-4 : f64, multiClusterStrategy = "SplitOverKernel", normalize_variance = true}
            : tensor<1x32x64x64xf16, {order = #NWHC}> -> tensor<1x32x64x64xf16, {order = #NWHC}>
    %permute_cast = VPU.PermuteCast(%mvn) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x32x64x64xf16, {order = #NWHC}> -> tensor<1x32x64x64xf16, {order = #NHWC}>
    %dwconv = VPU.NCE.DepthConvolution(%permute_cast, %weights, %weights_table, %act_win) {
            activation_window_channel_length = 4 : i64, multiClusterStrategy = "SplitOverKernel",
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [32, 1, 1, 1], strides = [1, 1]}
                -> tensor<1x32x64x64xf16, {order = #NHWC}>

    return %dwconv : tensor<1x32x64x64xf16, {order = #NHWC}>


    //CHECK:        [[CLUSTERED_MVN:%.*]] = VPU.NCE.ClusterTiling (%0 as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NWHC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x64x64xf16, #NWHC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[CLUSTERED_DW_CONV:%.*]] = VPU.NCE.ClusterTiling (%4 as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>, %5 as %arg2: tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>, %6 as %arg3: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>, %7 as %arg4: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @DoNotAlignForSOHInput
func @DoNotAlignForSOHInput(%input: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x32x64x64xf16, {order = #NWHC}> {
    %weights = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) {
            multiClusterStrategy = "SplitOverHeight",
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [32, 32, 1, 1], strides = [1, 1]}
                -> tensor<1x32x64x64xf16, {order = #NHWC}>
    %permute_cast = VPU.PermuteCast(%conv) {
            dst_order = #NWHC,
            mem_perm = #NCHW}
            : tensor<1x32x64x64xf16, {order = #NHWC}>
                -> tensor<1x32x64x64xf16, {order = #NWHC}>
    %mvn = VPU.MVN(%permute_cast) {
            across_channels = false, eps = 1.0E-4 : f64,
            multiClusterStrategy = "SplitOverKernel", normalize_variance = true
            } : tensor<1x32x64x64xf16, {order = #NWHC}>
                -> tensor<1x32x64x64xf16, {order = #NWHC}>

    return %mvn : tensor<1x32x64x64xf16, {order = #NWHC}>

    // Don't add alignment when the input is SOH
    // since SOH and channel split cannot be compatible
    //CHECK:        [[CLUSTERED_CONV:%.*]] = VPU.NCE.ClusterTiling (%0 as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>, %1 as %arg2: tensor<32x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>, %2 as %arg3: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[CLUSTERED_COPY_TO_MVN:%.*]] = VPU.NCE.ClusterTiling (%5 as %arg1: tensor<1x32x64x64xf16, {order = #NWHC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x64x64xf16, #NWHC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[CLUSTERED_MVN:%.*]] = VPU.NCE.ClusterTiling (%6 as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NWHC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x64x64xf16, #NWHC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @DoNotAlignForUnalignedChannel
func @DoNotAlignForUnalignedChannel(%input: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x3x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) {
            multiClusterStrategy = "SplitOverKernel",
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [32, 32, 1, 1], strides = [1, 1]}
                -> tensor<1x32x64x64xf16, {order = #NHWC}>
    %slice = VPU.Slice %conv [0, 0, 0, 0] [1, 3, 64, 64] : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x3x64x64xf16, {order = #NHWC}>
    %tanh = VPU.Tanh(%slice) {multiClusterStrategy = "Clustering"} : tensor<1x3x64x64xf16, {order = #NHWC}> -> tensor<1x3x64x64xf16, {order = #NHWC}>

    return %tanh : tensor<1x3x64x64xf16, {order = #NHWC}>

    //CHECK:        [[CLUSTERED_CONV:%.*]] = VPU.NCE.ClusterTiling (%0 as %arg1: tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>, %1 as %arg2: tensor<32x32x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>, %2 as %arg3: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:        [[CLUSTERED_COPY_TO_TANH:%.*]] = VPU.NCE.ClusterTiling (%5 as %arg1: tensor<1x3x64x64xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x3x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[CLUSTERED_TANH:%.*]] = VPU.NCE.ClusterTiling ([[CLUSTERED_COPY_TO_TANH]] as %arg1: tensor<1x3x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x3x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
}
