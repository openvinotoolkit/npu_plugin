//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --convert-VPUIP-to-VPUIPRegMapped %s | FileCheck %s

module @mainModule {

  IE.CNNNetwork entryPoint : @singleEltwise inputsInfo : {
    DataInfo "input_0" : tensor<1x32x56x56xui8, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
    DataInfo "input_1" : tensor<1x32x56x56xui8, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x32x56x56xui8, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  }

func @singleEltwise(%arg0: memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>, %arg1: memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>, %arg2: memref<1x32x56x56x!quant.uniform<u8:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>) -> memref<1x32x56x56x!quant.uniform<u8:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR> {
  %0 = VPURT.DeclareBuffer "CMX_NN" [0] <100352> -> memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %1 = VPURT.DeclareBuffer "CMX_NN" [0] <200704> -> memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %2 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x32x56x56x!quant.uniform<u8:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %3 = VPURT.DeclareBuffer "CMX_NN" [0] <100352> -> memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %4 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x32x56x56x!quant.uniform<u8:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %5 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
  %6 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
  VPURT.Task updates(%5 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    %7 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>) outputs(%0 : memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) -> memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  }
  VPURT.Task updates(%5 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    %7 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg1 : memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>) outputs(%1 : memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) -> memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  }
  VPURT.Task waits(%6 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    %7 = VPUIP.NNDMA {port = 0 : i64} inputs(%2 : memref<1x32x56x56x!quant.uniform<u8:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%arg2 : memref<1x32x56x56x!quant.uniform<u8:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>) -> memref<1x32x56x56x!quant.uniform<u8:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
  }
  VPURT.Task waits(%5 : !VPURT.Barrier) updates(%6 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    %7 = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i32, task_type = "ELTWISE"} input(%0 : memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) weights(%1 : memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) parent_input(%3 : memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) parent_output(%4 : memref<1x32x56x56x!quant.uniform<u8:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%2 : memref<1x32x56x56x!quant.uniform<u8:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) -> memref<1x32x56x56x!quant.uniform<u8:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]> variants : {
      DPUTask {outEnd = [55, 55, 31], mpe_mode = "CUBOID_8x16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0]}
    } PPE : {
      PPETask "ADD" {clamp_high = 255 : i64, clamp_low = 0 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [16384], quant_post_shift = 0 : i64, quant_scale = [1.000000e+00], quant_shift = [14]}
    }
  }
  return %arg2 : memref<1x32x56x56x!quant.uniform<u8:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
}
}

// CHECK: func @singleEltwise
// CHECK: VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 2 : ui8}<0, -1> -> !VPUIPRegMapped.Index<0>
// CHECK: VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPUIPRegMapped.Index<1>
// CHECK-NOT: VPURT.Task
// CHECK-NEXT: VPUIPRegMapped.NNDMA
// CHECK-NOT: VPURT.Task
// CHECK-NEXT: VPUIPRegMapped.NNDMA
// CHECK-NOT: VPURT.Task
// CHECK-NEXT: VPUIPRegMapped.NNDMA
// CHECK-NOT: VPURT.Task
// CHECK-NEXT: VPUIPRegMapped.DPUInvariant
// CHECK: VPUIPRegMapped.DPUVariant
