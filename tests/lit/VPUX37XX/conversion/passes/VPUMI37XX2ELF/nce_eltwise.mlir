//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --convert-VPUMI37XX-to-ELF %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType0 = !quant.uniform<u8<0:3>:f32, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f32, 1.000000e+00>
module @mainModule {
  IE.CNNNetwork entryPoint : @singleEltwise inputsInfo : {
    DataInfo "input_0" : tensor<1x32x56x56xui8>
    DataInfo "input_1" : tensor<1x32x56x56xui8>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x32x56x56xui8>
  }
  func.func @singleEltwise(%arg0: memref<1x32x56x56x!qElemType0, #NHWC, @DDR>, %arg1: memref<1x32x56x56x!qElemType0, #NHWC, @DDR>, %arg2: memref<1x32x56x56x!qElemType1, #NHWC, @DDR>) -> memref<1x32x56x56x!qElemType1, #NHWC, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <100352> -> memref<1x32x56x56x!qElemType0, #NHWC, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <200704> -> memref<1x32x56x56x!qElemType0, #NHWC, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x32x56x56x!qElemType1, #NHWC, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <100352> -> memref<1x32x56x56x!qElemType0, #NHWC, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x32x56x56x!qElemType1, #NHWC, [@CMX_NN, 0]>
    %5 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 2 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:0>
    %6 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPURegMapped.Index<0:0:1>
    %7 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x32x56x56x!qElemType0, #NHWC, @DDR>) outputs(%0 : memref<1x32x56x56x!qElemType0, #NHWC, [@CMX_NN, 0]>) updates(%5 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
    %8 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg1 : memref<1x32x56x56x!qElemType0, #NHWC, @DDR>) outputs(%1 : memref<1x32x56x56x!qElemType0, #NHWC, [@CMX_NN, 0]>) previousDMA(%7 : !VPURegMapped.Index<0:0:0>) updates(%5 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>
    %9 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%2 : memref<1x32x56x56x!qElemType1, #NHWC, [@CMX_NN, 0]>) outputs(%arg2 : memref<1x32x56x56x!qElemType1, #NHWC, @DDR>) previousDMA(%8 : !VPURegMapped.Index<0:0:1>) waits(%6 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:2>
    %10 = VPUMI37XX.DPUInvariant {activation_window_channel_length = 0 : i32, clean_after = 0 : ui64, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_8x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<ELTWISE>} input(%0 : memref<1x32x56x56x!qElemType0, #NHWC, [@CMX_NN, 0]>) weights(%1 : memref<1x32x56x56x!qElemType0, #NHWC, [@CMX_NN, 0]>) parent_input(%3 : memref<1x32x56x56x!qElemType0, #NHWC, [@CMX_NN, 0]>) parent_output(%4 : memref<1x32x56x56x!qElemType1, #NHWC, [@CMX_NN, 0]>) outputs(%2 : memref<1x32x56x56x!qElemType1, #NHWC, [@CMX_NN, 0]>) waits(%5 : !VPURegMapped.Index<0:0:0>) updates(%6 : !VPURegMapped.Index<0:0:1>) -> <0:0:0> PPE : {
      VPUIP.PPETask <ADD> {clamp_high = 255 : i64, clamp_low = 0 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [16384], quant_post_shift = 0 : i64, quant_scale = [1.000000e+00], quant_shift = [14]}
    }
    %11 = "VPUMI37XX.DPUVariant"(%10) {end = [55, 55, 31], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} : (!VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:0>
    %12 = VPUMI37XX.MappedInference dmas(%7 : !VPURegMapped.Index<0:0:0>) invariants(%10 : !VPURegMapped.Index<0:0:0>) variants(%11 : !VPURegMapped.Index<0:0:0>) barriers(%5 : !VPURegMapped.Index<0:0:0>) dmaCount([3, 0]) invariantCount(1) variantCount(1) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(2) -> !VPURegMapped.Index<0:0:0>
    return %arg2 : memref<1x32x56x56x!qElemType1, #NHWC, @DDR>
  }
}

// CHECK: func.func @singleEltwise
// CHECK: %[[VAL0:.*]] = VPURT.DeclareBuffer <CMX_NN>
// CHECK-NEXT: %[[VAL1:.*]] = VPURT.DeclareBuffer <CMX_NN>
// CHECK-NEXT: %[[VAL2:.*]] = VPURT.DeclareBuffer <CMX_NN>
// CHECK-NEXT: %[[VAL3:.*]] = VPURT.DeclareBuffer <CMX_NN>
// CHECK-NEXT: %[[VAL4:.*]] = VPURT.DeclareBuffer <CMX_NN>
// CHECK-NEXT: %[[VAL5:.*]] = VPUMI37XX.ConfigureBarrier
// CHECK-NEXT: %[[VAL6:.*]] = VPUMI37XX.ConfigureBarrier
// CHECK-NEXT: %[[VAL7:.*]] = VPUMI37XX.NNDMA
// CHECK-NEXT: %[[VAL8:.*]] = VPUMI37XX.NNDMA
// CHECK-NEXT: %[[VAL9:.*]] = VPUMI37XX.NNDMA
// CHECK-NEXT: %[[VAL10:.*]] = VPUMI37XX.DPUInvariant
// CHECK: %[[VAL11:.*]] = "VPUMI37XX.DPUVariant"
// CHECK-NEXT: %[[VAL12:.*]] = VPUMI37XX.MappedInference

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
