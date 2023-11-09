//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX allow-custom-values=true" --correct-NCE-workloads %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!Input_CMX = !VPU.DistributedTensor<
    1x224x3x224xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 1, 2],
    kernel = [7, 7],
    pads = #VPU.Padding<left = 3 , right = 2, top = 3, bottom = 2>,
    strides = [2, 2],
    num_clusters = 2
}>

!Output_CMX = !VPU.DistributedTensor<
    1x224x4x224xf16, #NWCH, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 1, 2],
    kernel = [7, 7],
    pads = #VPU.Padding<left = 3 , right = 2, top = 3, bottom = 2>,
    strides = [2, 2],
    num_clusters = 2,
    equal_memory_and_compute_view
}>

!Input_DDR = tensor<1x224x3x224xf16, {order = #NHWC}>
!InputStub_CMX = tensor<1x224x3x224xf16, {mem_space = @CMX_NN, order = #NHWC}>
!OutputStub_CMX = tensor<1x224x4x224xf16, {mem_space = @CMX_NN, order = #NWCH}>

module @PermuteQuantize attributes {VPU.arch = #VPU.arch_kind<VPUX37XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {

// CHECK-LABEL: @PermuteQuantizeClustered
func.func @PermuteQuantizeClustered(%arg0: !Input_DDR) -> !Output_CMX {
    %input_cmx = VPU.NCE.ClusterTiling(%arg0 as %arg1: !Input_DDR) -> !Input_CMX {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !Input_DDR -> !InputStub_CMX
        VPU.Yield %0
    }

    %output = VPU.NCE.ClusterTiling (%input_cmx as %arg2: !InputStub_CMX) -> !Output_CMX {
        %0 = VPU.NCE.PermuteQuantize(%arg2) {
                dstElemType = !quant.uniform<u8:f16, 1.000000e+00>,
                dstOrder = #NWCH,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>
            } -> !OutputStub_CMX {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 224, 4, 57] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 0 : i64}
                VPU.DPU.Workload outOffsets [0, 0, 0, 57] outSizes [1, 224, 4, 57] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 0 : i64}
                VPU.DPU.Workload outOffsets [0, 0, 0, 109] outSizes [1, 224, 4, 58] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 1 : i64}
                VPU.DPU.Workload outOffsets [0, 0, 0, 167] outSizes [1, 224, 4, 57] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 1 : i64}
            }
        VPU.Yield %0
    }

    return %output : !Output_CMX

    // CHECK:       VPU.NCE.PermuteQuantize
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>
    // CHECK-SAME:      -> tensor<1x224x4x224xf16, {mem_space = @CMX_NN, order = #NWCH}> {

    // CHECK:       VPU.DPU.Workload
    // CHECK-SAME:      outOffsets [0, 0, 0, 0] outSizes [1, 224, 3, 57]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      cluster_id = 0

    // CHECK:       VPU.DPU.Workload
    // CHECK-SAME:      outOffsets [0, 0, 0, 57] outSizes [1, 224, 3, 57]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      cluster_id = 0

    // CHECK:       VPU.DPU.Workload
    // CHECK-SAME:      outOffsets [0, 0, 0, 0] outSizes [1, 224, 3, 58]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      cluster_id = 1

    // CHECK:       VPU.DPU.Workload
    // CHECK-SAME:      outOffsets [0, 0, 0, 58] outSizes [1, 224, 3, 57]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      cluster_id = 1

}
}
