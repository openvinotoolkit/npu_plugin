//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX compilation-mode=DefaultHW" --split-NCE-ops-onto-workloads %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvRewriter
func @ConvRewriter(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}> =
        dense<10> : tensor<16x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x16x16x16xf16, {order = #NHWC}>
        -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<16x16x1x1xf16, {order = #NHWC}>
        -> tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %2 = VPU.Copy(%wt) {out_mem_space = @CMX_NN} : tensor<16x1x1x4xsi32, {order = #NHWC}>
        -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
    %3 = VPU.NCE.Convolution(%0, %1, %2) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %4 = VPU.Copy(%3) : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %4 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK:       [[CST0:%.+]] = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = VPU.Copy([[CST]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL2:%.+]] = VPU.Copy([[CST0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL3:%.+]] = VPU.NCE.Convolution([[VAL0]], [[VAL1]], [[VAL2]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK:               DPU.Workload [0, 0, 0, 0] [1, 16, 16, 16] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "CUBOID_16x16"
    // CHECK:           }

    // CHECK:       [[VAL4:%.+]] = VPU.Copy([[VAL3]])
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[VAL4]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvRewriter
func @DepthConvRewriter(%arg0: tensor<1x16x40x80xf16, {order = #NHWC}>) -> tensor<1x16x37x73xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<16x1x4x8xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x1x4x8xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}> =
        dense<10> : tensor<16x1x1x4xsi32>, [#const.Reorder<#NHWC>]
    %aw = const.Declare tensor<1x1x1x16xui8, {order = #NHWC}> =
        dense<1> : tensor<1x1x1x16xui8>, [#const.Reorder<#NHWC>]

    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x16x40x80xf16, {order = #NHWC}>
        -> tensor<1x16x40x80xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<16x1x4x8xf16, {order = #NHWC}>
        -> tensor<16x1x4x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %2 = VPU.Copy(%wt) {out_mem_space = @CMX_NN} : tensor<16x1x1x4xsi32, {order = #NHWC}>
        -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
    %3 = VPU.Copy(%aw) {out_mem_space = @CMX_NN} : tensor<1x1x1x16xui8, {order = #NHWC}>
        -> tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NHWC}>

    %4 = VPU.NCE.DepthConvolution(%0, %1, %2, %3) {
            activation_window_channel_length = 44 : i64,
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [16, 1, 4, 8],
            strides = [1, 1]
        } -> tensor<1x16x37x73xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %5 = VPU.Copy(%4) : tensor<1x16x37x73xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x16x37x73xf16, {order = #NHWC}>

    return %5 : tensor<1x16x37x73xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x1x4x8xf16, {order = #NHWC}>
    // CHECK:       [[CST0:%.+]] = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}>
    // CHECK:       [[CST1:%.+]] = const.Declare tensor<1x1x1x16xui8, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x16x40x80xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = VPU.Copy([[CST]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<16x1x4x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL2:%.+]] = VPU.Copy([[CST0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL3:%.+]] = VPU.Copy([[CST1]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL4:%.+]] = VPU.NCE.DepthConvolution([[VAL0]], [[VAL1]], [[VAL2]], [[VAL3]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x16x37x73xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK:               DPU.Workload [0, 0, 0, 0] [1, 16, 37, 73] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "CUBOID_16x16"
    // CHECK:           }

    // CHECK:       [[VAL5:%.+]] = VPU.Copy([[VAL4]])
    // CHECK-SAME:      -> tensor<1x16x37x73xf16, {order = #NHWC}>

    // CHECK:       return [[VAL5]] : tensor<1x16x37x73xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolRewriter
func @MaxPoolRewriter(%arg0: tensor<1x16x1x4xf16, {order = #NHWC}>) -> tensor<1x16x1x4xf16, {order = #NHWC}> {
    %wt = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}> =
        dense<10> : tensor<16x1x1x4xsi32>, [#const.Reorder<#NHWC>]
    %aw = const.Declare tensor<1x1x1x16xui8, {order = #NHWC}> =
        dense<1> : tensor<1x1x1x16xui8>, [#const.Reorder<#NHWC>]

    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x16x1x4xf16, {order = #NHWC}>
        -> tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = VPU.Copy(%wt) {out_mem_space = @CMX_NN} : tensor<16x1x1x4xsi32, {order = #NHWC}>
        -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
    %2 = VPU.Copy(%aw) {out_mem_space = @CMX_NN} : tensor<1x1x1x16xui8, {order = #NHWC}>
        -> tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NHWC}>

    %3 = VPU.NCE.MaxPool(%0, %1, %2) {
            activation_window_channel_length = 4 : i64,
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            strides = [1, 1],
            kernel_size = [1, 1]
        } -> tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %4 = VPU.Copy(%3) : tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x16x1x4xf16, {order = #NHWC}>

    return %4 : tensor<1x16x1x4xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}>
    // CHECK:       [[CST0:%.+]] = const.Declare tensor<1x1x1x16xui8, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = VPU.Copy([[CST]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL2:%.+]] = VPU.Copy([[CST0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL3:%.+]] = VPU.NCE.MaxPool([[VAL0]], [[VAL1]], [[VAL2]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK:               DPU.Workload [0, 0, 0, 0] [1, 16, 1, 4] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "CUBOID_16x16"
    // CHECK:           }

    // CHECK:       [[VAL4:%.+]] = VPU.Copy([[VAL3]])
    // CHECK-SAME:      -> tensor<1x16x1x4xf16, {order = #NHWC}>

    // CHECK:       return [[VAL4]] : tensor<1x16x1x4xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAddRewriter
func @EltwiseAddRewriter(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>, %arg1: tensor<1x64x28x28xf16, {order = #NHWC}>)
        -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x64x28x28xf16, {order = #NHWC}>
        -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x64x28x28xf16, {order = #NHWC}>
        -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %2 = VPU.NCE.Eltwise(%0, %1) { op_type = "ADD" } :
        tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %3 = VPU.Copy(%2) : tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x64x28x28xf16, {order = #NHWC}>

    return %3 : tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL2:%.+]] = VPU.NCE.Eltwise([[VAL0]], [[VAL1]]) {minimumHardwareExecutionCost = 4620 : i64, op_type = "ADD"}
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK:               DPU.Workload [0, 0, 0, 0] [1, 64, 28, 28] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "CUBOID_8x16"
    // CHECK:           }

    // CHECK:       [[VAL3:%.+]] = VPU.Copy([[VAL2]])
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       return [[VAL3]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = type !VPU.DistributedTensor<
    1x32x16x16xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    kernel = [3, 3],
    pads = {bottom = 1, left = 1, right = 1, top = 1},
    strides = [1, 1],
    num_clusters = 4
}>

!WeightsDistributed = type !VPU.DistributedTensor<
    64x32x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4
}>

!WeightsTableDistributed = type !VPU.DistributedTensor<
    64x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4
}>

!OutputDistributed = type !VPU.DistributedTensor<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4
}>

!Input_DDR = type tensor<1x32x16x16xf16, {mem_space = @DDR, order = #NHWC}>
!Weights_DDR = type tensor<64x32x3x3xf16, {mem_space = @DDR, order = #NHWC}>
!WeightsTable_DDR = type tensor<64x1x1x4xsi32, {mem_space = @DDR, order = #NCHW}>
!Output_DDR = type tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>

!InputStub_CMX = type tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsStub_CMX = type tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsTableStub_CMX = type tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
!OutputStub_CMX = type tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>


func @ConvolutionWithDistributedTensor(%arg0: !Input_DDR) -> !Output_DDR {
    %weights = const.Declare tensor<64x32x3x3xf16, {mem_space = @DDR, order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x3x3xf16, {mem_space = @DDR}>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> = dense<10> : tensor<64x1x1x4xsi32, {mem_space = @CMX_NN}>

    %input_cmx = VPU.NCE.ClusterTiling(%arg0 as %arg1: !Input_DDR) -> !InputDistributed {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !Input_DDR -> !InputStub_CMX
        VPU.Yield %0
    }

    %weights_cmx = VPU.NCE.ClusterTiling(%weights as %arg1: !Weights_DDR) -> !WeightsDistributed {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !Weights_DDR -> !WeightsStub_CMX
        VPU.Yield %0
    }

    %wt_cmx = VPU.NCE.ClusterTiling(%wt as %arg1: !WeightsTable_DDR) -> !WeightsTableDistributed {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !WeightsTable_DDR -> !WeightsTableStub_CMX
        VPU.Yield %0
    }

    %output_cmx = VPU.NCE.ClusterTiling (
              %input_cmx as %arg1: !InputStub_CMX,
              %weights_cmx as %arg2: !WeightsStub_CMX,
              %wt_cmx as %arg3: !WeightsTableStub_CMX)
              -> !OutputDistributed {
        // Generate different workloads due to different pads on each cluster
        %0 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
                pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
                rawFilterShape = [64, 32, 3, 3],
                strides = [1, 1]
            } -> !OutputStub_CMX
        VPU.Yield %0
    }
    %output = VPU.NCE.ClusterTiling(%output_cmx as %arg1: !OutputStub_CMX) -> !Output_DDR {
        %0 = VPU.Copy(%arg1) { out_mem_space = @DDR } : !OutputStub_CMX -> !Output_DDR
        VPU.Yield %0
    }

    return %output: !Output_DDR

    // CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<64x32x3x3xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK-SAME:       = dense<1.000000e+00> : tensor<64x32x3x3xf16, {mem_space = @DDR}>, [#const.Reorder<#NHWC>]
    // CHECK:        [[WEIGHTS_TABLE:%.*]] = const.Declare tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    // CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x32x16x16xf16, {mem_space = @DDR, order = #NHWC}>)
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x32x16x16xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:              {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], kernel = [3, 3],
    // CHECK-SAME:               pads = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, strides = [1, 1], num_clusters = 4 : i64}> {
    // CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK-SAME:           -> tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:            VPU.Yield [[RES0]]
    // CHECK:        }

    // CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<64x32x3x3xf16, {mem_space = @DDR, order = #NHWC}>)
    // CHECK-SAME:           -> !VPU.DistributedTensor<64x32x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    // CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<64x32x3x3xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK-SAME:           -> tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:            VPU.Yield [[RES1]]
    // CHECK:        }

    // CHECK:        [[WEIGHTS_TABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_TABLE]] as %arg1: tensor<64x1x1x4xsi32, {mem_space = @DDR, order = #NCHW}>)
    // CHECK-SAME:           -> !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    // CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<64x1x1x4xsi32, {mem_space = @DDR, order = #NCHW}>
    // CHECK-SAME:           -> tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:            VPU.Yield [[RES2]]
    // CHECK:        }

    // CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:             [[WEIGHTS_CMX]] as %arg2: tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:             [[WEIGHTS_TABLE_CMX]] as %arg3: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:             -> !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
    // CHECK:                [[RES4:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
    // CHECK-SAME:                            pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    // CHECK-SAME:                            strides = [1, 1]
    // CHECK-SAME:             } -> tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:                    DPU.Workload [0, 0, 0, 0] [1, 64, 4, 16] {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64} "CUBOID_16x16" attributes {cluster_id = 0 : i64}
    // CHECK:                    DPU.Workload [0, 0, 4, 0] [1, 64, 4, 16] {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64} "CUBOID_16x16" attributes {cluster_id = 1 : i64}
    // CHECK:                    DPU.Workload [0, 0, 8, 0] [1, 64, 4, 16] {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64} "CUBOID_16x16" attributes {cluster_id = 2 : i64}
    // CHECK:                    DPU.Workload [0, 0, 12, 0] [1, 64, 4, 16] {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64} "CUBOID_16x16" attributes {cluster_id = 3 : i64}
    // CHECK:            VPU.Yield [[RES4]]
    // CHECK:        }

    // CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:           -> tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}> {
    // CHECK:            [[RES5:%.*]] = VPU.Copy(%arg1) {out_mem_space = @DDR} : tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:           -> tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:            VPU.Yield [[RES5]]
    // CHECK:        }

    // CHECK:        return [[OUT]] : tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvRewriterCUBOID_16x16
func @ConvRewriterCUBOID_16x16(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}> =
        dense<10> : tensor<16x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x16x16x16xf16, {order = #NHWC}>
        -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<16x16x1x1xf16, {order = #NHWC}>
        -> tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %2 = VPU.Copy(%wt) {out_mem_space = @CMX_NN} : tensor<16x1x1x4xsi32, {order = #NHWC}>
        -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
    %3 = VPU.NCE.Convolution(%0, %1, %2) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %4 = VPU.Copy(%3) : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %4 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK:       [[CST0:%.+]] = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = VPU.Copy([[CST]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL2:%.+]] = VPU.Copy([[CST0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL3:%.+]] = VPU.NCE.Convolution([[VAL0]], [[VAL1]], [[VAL2]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK:               DPU.Workload [0, 0, 0, 0] [1, 16, 16, 16] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "CUBOID_16x16"
    // CHECK:           }

    // CHECK:       [[VAL4:%.+]] = VPU.Copy([[VAL3]])
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[VAL4]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----



#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvRewriterCUBOID_8x16
func @ConvRewriterCUBOID_8x16(%arg0: tensor<1x128x56x28xf16, {order = #NHWC}>) -> tensor<1x128x56x28xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<128x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<128x1x1x4xsi32, {order = #NHWC}> =
        dense<10> : tensor<128x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x128x56x28xf16, {order = #NHWC}>
        -> tensor<1x128x56x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<128x128x1x1xf16, {order = #NHWC}>
        -> tensor<128x128x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %2 = VPU.Copy(%wt) {out_mem_space = @CMX_NN} : tensor<128x1x1x4xsi32, {order = #NHWC}>
        -> tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
    %3 = VPU.NCE.Convolution(%0, %1, %2) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [128, 128, 1, 1],
            strides = [1, 1]
        } : tensor<1x128x56x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            tensor<128x128x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x128x56x28xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %4 = VPU.Copy(%3) : tensor<1x128x56x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x128x56x28xf16, {order = #NHWC}>

    return %4 : tensor<1x128x56x28xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}>
    // CHECK:       [[CST0:%.+]] = const.Declare tensor<128x1x1x4xsi32, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x128x56x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = VPU.Copy([[CST]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<128x128x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL2:%.+]] = VPU.Copy([[CST0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL3:%.+]] = VPU.NCE.Convolution([[VAL0]], [[VAL1]], [[VAL2]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x128x56x28xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK:               DPU.Workload [0, 0, 0, 0] [1, 128, 56, 28] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "CUBOID_8x16"
    // CHECK:           }

    // CHECK:       [[VAL4:%.+]] = VPU.Copy([[VAL3]])
    // CHECK-SAME:      -> tensor<1x128x56x28xf16, {order = #NHWC}>

    // CHECK:       return [[VAL4]] : tensor<1x128x56x28xf16, {order = #NHWC}>
}

// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvRewriterCUBOID_4x16
func @ConvRewriterCUBOID_4x16(%arg0: tensor<1x256x8x8xf16, {order = #NHWC}>) -> tensor<1x256x8x8xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<256x256x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<256x256x1x1xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<256x1x1x4xsi32, {order = #NHWC}> =
        dense<10> : tensor<256x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x256x8x8xf16, {order = #NHWC}>
        -> tensor<1x256x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<256x256x1x1xf16, {order = #NHWC}>
        -> tensor<256x256x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %2 = VPU.Copy(%wt) {out_mem_space = @CMX_NN} : tensor<256x1x1x4xsi32, {order = #NHWC}>
        -> tensor<256x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
    %3 = VPU.NCE.Convolution(%0, %1, %2) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [256, 256, 1, 1],
            strides = [1, 1]
        } : tensor<1x256x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            tensor<256x256x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            tensor<256x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x256x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %4 = VPU.Copy(%3) : tensor<1x256x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x256x8x8xf16, {order = #NHWC}>

    return %4 : tensor<1x256x8x8xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<256x256x1x1xf16, {order = #NHWC}>
    // CHECK:       [[CST0:%.+]] = const.Declare tensor<256x1x1x4xsi32, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x256x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = VPU.Copy([[CST]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<256x256x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL2:%.+]] = VPU.Copy([[CST0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<256x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL3:%.+]] = VPU.NCE.Convolution([[VAL0]], [[VAL1]], [[VAL2]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x256x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK:               DPU.Workload [0, 0, 0, 0] [1, 256, 8, 8] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "CUBOID_4x16"
    // CHECK:           }

    // CHECK:       [[VAL4:%.+]] = VPU.Copy([[VAL3]])
    // CHECK-SAME:      -> tensor<1x256x8x8xf16, {order = #NHWC}>

    // CHECK:       return [[VAL4]] : tensor<1x256x8x8xf16, {order = #NHWC}>
}

// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = type !VPU.SparseTensor<
    data=!VPU.DistributedTensor<1x16x224x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, 
    sparsity_map=!VPU.DistributedTensor<1x16x224x224xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    >

!WeightsDistributed = type !VPU.DistributedTensor<
    32x16x7x7xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!WeightsTableDistributed = type !VPU.DistributedTensor<
    32x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!OutputDistributed = type !VPU.SparseTensor<
    data=!VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    sparsity_map=!VPU.DistributedTensor<1x32x112x112xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    >

!Input_DDR = type !VPU.SparseTensor<data=tensor<1x16x224x224xf16, {mem_space = @DDR, order = #NHWC}>, sparsity_map=tensor<1x16x224x224xi1, {mem_space = @DDR, order = #NHWC}>>
!Weights_DDR = type tensor<32x16x7x7xf16, {mem_space = @DDR, order = #NHWC}>
!WeightsTable_DDR = type tensor<32x1x1x4xsi32, {mem_space = @DDR, order = #NCHW}>
!Output_DDR = type !VPU.SparseTensor<
    data=tensor<1x32x112x112xf16, {mem_space = @DDR, order = #NHWC}>,
    sparsity_map=tensor<1x32x112x112xi1, {mem_space = @DDR, order = #NHWC}>
    >

!InputStub_CMX = type !VPU.SparseTensor<data=tensor<1x16x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x16x224x224xi1, {mem_space = @CMX_NN, order = #NHWC}>> 
!WeightsStub_CMX = type tensor<32x16x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsTableStub_CMX = type tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
!OutputStub_CMX = type !VPU.SparseTensor<
    data=tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    sparsity_map=tensor<1x32x112x112xi1, {mem_space = @CMX_NN, order = #NHWC}>
    >


func @ConvolutionWithSparseDistributedTensor(%arg0: !Input_DDR) -> !Output_DDR {
    %weights = const.Declare !Weights_DDR = dense<1.000000e+00> : tensor<32x16x7x7xf16, {mem_space = @DDR}>, [#const.Reorder<#NHWC>]
    %wt = const.Declare !WeightsTable_DDR = dense<10> : tensor<32x1x1x4xsi32, {mem_space = @CMX_NN}>

    // copy input DDR -> CMX
    %input_cmx = VPU.NCE.ClusterTiling (%arg0 as %arg1: !Input_DDR) -> !InputDistributed {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !Input_DDR -> !InputStub_CMX
        VPU.Yield %0
    }

    // copy dense weights DDR->CMX
    %weights_cmx = VPU.NCE.ClusterTiling(%weights as %arg1: !Weights_DDR) -> !WeightsDistributed {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !Weights_DDR -> !WeightsStub_CMX
        VPU.Yield %0
    }

    // copy weights table DDR->CMX
    %wt_cmx = VPU.NCE.ClusterTiling(%wt as %arg1: !WeightsTable_DDR) -> !WeightsTableDistributed {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !WeightsTable_DDR -> !WeightsTableStub_CMX
        VPU.Yield %0
    }

    %output_cmx = VPU.NCE.ClusterTiling (
              %input_cmx as %arg1: !InputStub_CMX,
              %weights_cmx as %arg2: !WeightsStub_CMX,
              %wt_cmx as %arg3: !WeightsTableStub_CMX)
              -> !OutputDistributed {
        // Generate different workloads due to different pads on each cluster
        %0 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
            pad = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64}, 
            ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "LRELU"}, 
            rawFilterShape = [32, 16, 7, 7], 
            strides = [2, 2], 
            tilingStrategy = [1, 2, 1, 1]
            } -> !OutputStub_CMX
        VPU.Yield %0
    }
    %output = VPU.NCE.ClusterTiling(%output_cmx as %arg1: !OutputStub_CMX) -> !Output_DDR {
        %0 = VPU.Copy(%arg1) { out_mem_space = @DDR } : !OutputStub_CMX -> !Output_DDR
        VPU.Yield %0
    }

    return %output: !Output_DDR

    // CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<32x16x7x7xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK-SAME:       = dense<1.000000e+00> : tensor<32x16x7x7xf16, {mem_space = @DDR}>, [#const.Reorder<#NHWC>]
    // CHECK:        [[WEIGHTS_TABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32, {mem_space = @DDR, order = #NCHW}>
    // CHECK-SAME:       = dense<10> : tensor<32x1x1x4xsi32, {mem_space = @CMX_NN}>
    // CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling 
    // CHECK-SAME:                       (%arg0 as %arg1: !VPU.SparseTensor<data=tensor<1x16x224x224xf16, {mem_space = @DDR, order = #NHWC}>, 
    // CHECK-SAME:                                                          sparsity_map=tensor<1x16x224x224xi1, {mem_space = @DDR, order = #NHWC}>>)
    // CHECK-SAME:           -> !VPU.SparseTensor<
    // CHECK-SAME:              data=!VPU.DistributedTensor<1x16x224x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, 
    // CHECK-SAME:              sparsity_map=!VPU.DistributedTensor<1x16x224x224xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>> {
    // CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : !VPU.SparseTensor
    // CHECK-SAME:           -> !VPU.SparseTensor
    // CHECK:            VPU.Yield [[RES0]]
    // CHECK:        }

    // CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<32x16x7x7xf16, {mem_space = @DDR, order = #NHWC}>)
    // CHECK-SAME:           -> !VPU.DistributedTensor<32x16x7x7xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x16x7x7xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK-SAME:           -> tensor<32x16x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:            VPU.Yield [[RES1]]
    // CHECK:        }

    // CHECK:        [[WEIGHTS_TABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_TABLE]] as %arg1: tensor<32x1x1x4xsi32, {mem_space = @DDR, order = #NCHW}>)
    // CHECK-SAME:           -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32, {mem_space = @DDR, order = #NCHW}>
    // CHECK-SAME:           -> tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:            VPU.Yield [[RES2]]
    // CHECK:        }

    // CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:             [[INPUT_CMX]] as %arg1: !VPU.SparseTensor<data=tensor<1x16x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                                       sparsity_map=tensor<1x16x224x224xi1, {mem_space = @CMX_NN, order = #NHWC}>>,
    // CHECK-SAME:             [[WEIGHTS_CMX]] as %arg2: tensor<32x16x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:             [[WEIGHTS_TABLE_CMX]] as %arg3: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:             -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                  sparsity_map=!VPU.DistributedTensor<1x32x112x112xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>> {
    // CHECK:                [[RES4:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
    // CHECK-SAME:                            pad = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64},
    // CHECK-SAME:                            ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "LRELU"},
    // CHECK-SAME:                            rawFilterShape = [32, 16, 7, 7], strides = [2, 2], tilingStrategy = [1, 2, 1, 1]}
    // CHECK-SAME:             -> !VPU.SparseTensor<data=tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                  sparsity_map=tensor<1x32x112x112xi1, {mem_space = @CMX_NN, order = #NHWC}>> {
    // CHECK:                    VPU.DPU.Workload [0, 0, 0, 0] [1, 32, 56, 112] {bottom = 0 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64} "CUBOID_16x16" attributes {cluster_id = 0 : i64}
    // CHECK:                    VPU.DPU.Workload [0, 0, 56, 0] [1, 32, 56, 112] {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 0 : i64} "CUBOID_16x16" attributes {cluster_id = 1 : i64}
    // CHECK:            VPU.Yield [[RES4]]
    // CHECK:        }

    // CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: !VPU.SparseTensor<data=tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>, 
    // CHECK-SAME:                                                                                sparsity_map=tensor<1x32x112x112xi1, {mem_space = @CMX_NN, order = #NHWC}>>)
    // CHECK-SAME:           -> !VPU.SparseTensor<data=tensor<1x32x112x112xf16, {mem_space = @DDR, order = #NHWC}>, 
    // CHECK-SAME:                                sparsity_map=tensor<1x32x112x112xi1, {mem_space = @DDR, order = #NHWC}>> {
    // CHECK:            [[RES5:%.*]] = VPU.Copy(%arg1) {out_mem_space = @DDR} : !VPU.SparseTensor<data=tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                                                                 sparsity_map=tensor<1x32x112x112xi1, {mem_space = @CMX_NN, order = #NHWC}>>
    // CHECK-SAME:           -> !VPU.SparseTensor<data=tensor<1x32x112x112xf16, {mem_space = @DDR, order = #NHWC}>, 
    // CHECK-SAME:                                sparsity_map=tensor<1x32x112x112xi1, {mem_space = @DDR, order = #NHWC}>>
    // CHECK:            VPU.Yield [[RES5]]
    // CHECK:        }

    // CHECK:        return [[OUT]] : !VPU.SparseTensor<data=tensor<1x32x112x112xf16, {mem_space = @DDR, order = #NHWC}>,
    // CHECK-SAME:                                      sparsity_map=tensor<1x32x112x112xi1, {mem_space = @DDR, order = #NHWC}>>
}
