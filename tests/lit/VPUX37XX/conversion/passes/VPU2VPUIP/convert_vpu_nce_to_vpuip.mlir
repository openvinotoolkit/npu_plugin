//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --convert-vpu-nce-to-vpuip --canonicalize %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = type !quant.uniform<u8:f16, 1.000000e+00>

module @PermuteQuantize attributes {VPU.arch = "VPUX37XX", VPU.compilationMode = "DefaultHW"} {
  IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}

// CHECK-LABEL: @NCEPermuteQuantize
func @NCEPermuteQuantize(%arg0: memref<1x32x3x1568xf16, #NHWC, @CMX_NN>) -> memref<1x32x4x1568xf16, #NWCH, @CMX_NN> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x32x3x1568xf16, #NHWC, @CMX_NN>
        to tensor<1x32x3x1568xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %1 = VPU.NCE.PermuteQuantize(%0) {
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
            fp_prelu_alpha = 1.000000e+00 : f64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = "NOOP"
        }
    } -> tensor<1x32x4x1568x!qElemType, {mem_space = @CMX_NN, order = #NWCH}> {
        VPU.DPU.Workload [0, 0, 0, 0] [1, 32, 3, 1568] {
            bottom = 0 : i64,
            left = 0 : i64,
            right = 0 : i64,
            top = 0 : i64
        } "CUBOID_16x16"
    }

    %2 = builtin.unrealized_conversion_cast %1 : tensor<1x32x4x1568x!qElemType, {mem_space = @CMX_NN, order = #NWCH}>
        to memref<1x32x4x1568xf16, #NWCH, @CMX_NN>

    return %2 : memref<1x32x4x1568xf16, #NWCH, @CMX_NN>

    // CHECK-NOT:   VPU.NCE.PermuteQuantize

    // CHECK:       [[ALLOC:%.*]] = memref.alloc() : memref<1x32x4x1568x!qElemType, #NWCH, @CMX_NN>

    // CHECK:       VPUIP.NCEClusterTask {
    // CHECK-SAME:      activation_window_channel_length = 0 : i64,
    // CHECK-SAME:      is_superdense,
    // CHECK-SAME:      task_type = "ELTWISE"
    // CHECK-SAME:  }
    // CHECK-SAME:  input(%arg0 : memref<1x32x3x1568xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  weights(%arg0 : memref<1x32x3x1568xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  parent_input(%arg0 : memref<1x32x3x1568xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  parent_output(%0 : memref<1x32x4x1568x!qElemType, #NWCH, @CMX_NN>)
    // CHECK-SAME:  outputs(%0 : memref<1x32x4x1568x!qElemType, #NWCH, @CMX_NN>)
    // CHECK-SAME:  -> memref<1x32x4x1568x!qElemType, #NWCH, @CMX_NN>
    // CHECK-SAME:  variants : {
    // CHECK:           DPUTask {
    // CHECK-SAME:          mpe_mode = "CUBOID_16x16",
    // CHECK-SAME:          outEnd = [1567, 2, 31],
    // CHECK-SAME:          outStart = [0, 0, 0],
    // CHECK-SAME:          pad = {
    // CHECK-SAME:              bottom = 0 : i64,
    // CHECK-SAME:              left = 0 : i64,
    // CHECK-SAME:              right = 0 : i64,
    // CHECK-SAME:              top = 0 : i64
    // CHECK-SAME:          }
    // CHECK:           } PPE : {
    // CHECK:               PPETask "ADD" {
    // CHECK-SAME:              clamp_high = 255 : i64,
    // CHECK-SAME:              clamp_low = 0 : i64,
    // CHECK-SAME:              lrelu_mult = 1 : i64,
    // CHECK-SAME:              lrelu_shift = 0 : i64,
    // CHECK-SAME:              quant_scale = [5.000000e-01]
    // CHECK-SAME:          }
    // CHECK:           }
    // CHECK:       }
}

}
