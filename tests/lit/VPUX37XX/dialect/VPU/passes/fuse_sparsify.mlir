//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --fuse-sparsity-ops="fuse-sparsify=true" %s | FileCheck %s

!qElemType = type !quant.uniform<u8:f16, 0.034255280214197492:128>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

module @PermuteQuantize attributes {VPU.arch = "VPUX37XX", VPU.compilationMode = "DefaultHW"} {
  IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}

// CHECK-LABEL: @DoNotFusePermuteQuantize
func @DoNotFusePermuteQuantize(%arg0: tensor<1x32x3x1568xf16, {order = #NHWC}>) -> tensor<1x32x4x1568x!qElemType, {order = #NWCH}> {
    %0 = VPU.NCE.PermuteQuantize(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NWCH,
        pad = {
            bottom = 1 : i64,
            left = 0 : i64,
            right = 0 : i64,
            top = 0 : i64
        },
        ppe = {
            clamp_high = 255 : i64,
            clamp_low = 0 : i64,
            fp_prelu_alpha = 29.19257926940918 : f64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = "NOOP"
        }
    } -> tensor<1x32x4x1568x!qElemType, {order = #NWCH}>

    %1 = VPU.Sparsify(%0) : tensor<1x32x4x1568x!qElemType, {order = #NWCH}>
        -> !VPU.SparseTensor<data=tensor<1x32x4x1568x!qElemType, {order = #NWCH}>>

    %2 = VPU.Desparsify(%1) : !VPU.SparseTensor<data=tensor<1x32x4x1568x!qElemType, {order = #NWCH}>>
        -> tensor<1x32x4x1568x!qElemType, {order = #NWCH}>

    return %2 : tensor<1x32x4x1568x!qElemType, {order = #NWCH}>

    // CHECK:       [[PERMUTE_QUANTIZE:%.+]] = VPU.NCE.PermuteQuantize(%arg0)
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x32x4x1568x!qElemType, {order = #NWCH}>

    // CHECK:       [[SPARSIFY:%.+]] = VPU.Sparsify([[PERMUTE_QUANTIZE]])
    // CHECK:       [[DESPARSIFY:%.+]] = VPU.Desparsify([[SPARSIFY]])
    // CHECK:       return [[DESPARSIFY]]
}

}
