//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --convert-VPUIPRegMapped-to-ELF %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType0 = type !quant.uniform<u8<0:3>:f32, 1.000000e+00>
!qElemType1 = type !quant.uniform<u8:f32, 1.000000e+00>
module @mainModule {
  IE.CNNNetwork entryPoint : @singleEltwise inputsInfo : {
    DataInfo "input_0" : tensor<1x32x56x56xui8, {order = #NHWC}>
    DataInfo "input_1" : tensor<1x32x56x56xui8, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x32x56x56xui8, {order = #NHWC}>
  }
  func @singleEltwise(%arg0: memref<1x32x56x56x!qElemType0, #NHWC, @DDR>, %arg1: memref<1x32x56x56x!qElemType0, #NHWC, @DDR>, %arg2: memref<1x32x56x56x!qElemType1, #NHWC, @DDR>) -> memref<1x32x56x56x!qElemType1, #NHWC, @DDR> {
    %0 = VPURT.DeclareBuffer "CMX_NN" [0] <100352> -> memref<1x32x56x56x!qElemType0, #NHWC, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer "CMX_NN" [0] <200704> -> memref<1x32x56x56x!qElemType0, #NHWC, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x32x56x56x!qElemType1, #NHWC, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer "CMX_NN" [0] <100352> -> memref<1x32x56x56x!qElemType0, #NHWC, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x32x56x56x!qElemType1, #NHWC, [@CMX_NN, 0]>
    %5 = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 2 : ui8}<0, -1> -> !VPUIPRegMapped.Index<0>
    %6 = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPUIPRegMapped.Index<1>
    %7 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x32x56x56x!qElemType0, #NHWC, @DDR>) outputs(%0 : memref<1x32x56x56x!qElemType0, #NHWC, [@CMX_NN, 0]>) updates(%5 : !VPUIPRegMapped.Index<0>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<0>
    %8 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg1 : memref<1x32x56x56x!qElemType0, #NHWC, @DDR>) outputs(%1 : memref<1x32x56x56x!qElemType0, #NHWC, [@CMX_NN, 0]>) previousDMA(%7 : !VPUIPRegMapped.Index<0>) updates(%5 : !VPUIPRegMapped.Index<0>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<1>
    %9 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%2 : memref<1x32x56x56x!qElemType1, #NHWC, [@CMX_NN, 0]>) outputs(%arg2 : memref<1x32x56x56x!qElemType1, #NHWC, @DDR>) previousDMA(%8 : !VPUIPRegMapped.Index<1>) waits(%6 : !VPUIPRegMapped.Index<1>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<2>
    %10 = VPUIPRegMapped.DPUInvariant {activation_window_channel_length = 0 : i32, clean_after = 0 : ui64, mpe_frequent_mode = "CUBOID_8x16", start_after = 0 : ui64, task_type = "ELTWISE"} input(%0 : memref<1x32x56x56x!qElemType0, #NHWC, [@CMX_NN, 0]>) weights(%1 : memref<1x32x56x56x!qElemType0, #NHWC, [@CMX_NN, 0]>) parent_input(%3 : memref<1x32x56x56x!qElemType0, #NHWC, [@CMX_NN, 0]>) parent_output(%4 : memref<1x32x56x56x!qElemType1, #NHWC, [@CMX_NN, 0]>) outputs(%2 : memref<1x32x56x56x!qElemType1, #NHWC, [@CMX_NN, 0]>) waits(%5 : !VPUIPRegMapped.Index<0>) updates(%6 : !VPUIPRegMapped.Index<1>) -> <0> PPE : {
      VPUIP.PPETask "ADD" {clamp_high = 255 : i64, clamp_low = 0 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [16384], quant_post_shift = 0 : i64, quant_scale = [1.000000e+00], quant_shift = [14]}
    }
    %11 = "VPUIPRegMapped.DPUVariant"(%10) {end = [55, 55, 31], mpe_mode = "CUBOID_8x16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]} : (!VPUIPRegMapped.Index<0>) -> !VPUIPRegMapped.Index<0>
    %12 = VPUIPRegMapped.MappedInference dmas(%7 : !VPUIPRegMapped.Index<0>) invariants(%10 : !VPUIPRegMapped.Index<0>) variants(%11 : !VPUIPRegMapped.Index<0>) barriers(%5 : !VPUIPRegMapped.Index<0>) dmaCount([3, 0]) invariantCount(1) variantCount(1) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(2) -> !VPUIPRegMapped.Index<0>
    return %arg2 : memref<1x32x56x56x!qElemType1, #NHWC, @DDR>
  }
}

// CHECK: func @singleEltwise
// CHECK: %[[VAL0:.*]] = VPURT.DeclareBuffer "CMX_NN"
// CHECK-NEXT: %[[VAL1:.*]] = VPURT.DeclareBuffer "CMX_NN"
// CHECK-NEXT: %[[VAL2:.*]] = VPURT.DeclareBuffer "CMX_NN"
// CHECK-NEXT: %[[VAL3:.*]] = VPURT.DeclareBuffer "CMX_NN"
// CHECK-NEXT: %[[VAL4:.*]] = VPURT.DeclareBuffer "CMX_NN"
// CHECK-NEXT: %[[VAL5:.*]] = VPUIPRegMapped.ConfigureBarrier
// CHECK-NEXT: %[[VAL6:.*]] = VPUIPRegMapped.ConfigureBarrier
// CHECK-NEXT: %[[VAL7:.*]] = VPUIPRegMapped.NNDMA
// CHECK-NEXT: %[[VAL8:.*]] = VPUIPRegMapped.NNDMA
// CHECK-NEXT: %[[VAL9:.*]] = VPUIPRegMapped.NNDMA
// CHECK-NEXT: %[[VAL10:.*]] = VPUIPRegMapped.DPUInvariant
// CHECK: %[[VAL11:.*]] = "VPUIPRegMapped.DPUVariant"
// CHECK-NEXT: %[[VAL12:.*]] = VPUIPRegMapped.MappedInference

// CHECK-DAG: ELF.CreateSection {{.*}} secName = ".text.dmaTasks0"
// CHECK-NEXT: ELF.PutOpInSection %[[VAL7]]
// CHECK-NEXT: ELF.PutOpInSection %[[VAL8]]
// CHECK-NEXT: ELF.PutOpInSection %[[VAL9]]

// CHECK-DAG: ELF.CreateSection {{.*}} secName = ".text.BarrierConfigs"
// CHECK-NEXT: ELF.PutOpInSection %[[VAL5]]
// CHECK-NEXT: ELF.PutOpInSection %[[VAL6]]

// CHECK-DAG: ELF.CreateSection {{.*}} secName = ".text.MappedInference"
// CHECK-NEXT: ELF.PutOpInSection %[[VAL12]]

// CHECK-DAG: ELF.CreateSection {{.*}} secName = ".text.DPUInvariants"
// CHECK-NEXT: ELF.PutOpInSection %[[VAL10]]

// CHECK-DAG: ELF.CreateSection {{.*}} secName = ".text.DPUVariants"
// CHECK-NEXT: ELF.PutOpInSection %[[VAL11]]
