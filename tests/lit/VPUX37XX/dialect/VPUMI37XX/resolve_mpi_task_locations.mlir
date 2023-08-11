//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --resolve-mapped-inference-task-locations %s | FileCheck %s

func.func @oneDma() {
  %0 = VPURT.DeclareBuffer "NetworkInput" [0] <0> {swizzlingKey = 0 : i64} -> memref<1x2x3x4xf16, @DDR>
  %1 = VPURT.DeclareBuffer "NetworkOutput" [0] <0> {swizzlingKey = 0 : i64} -> memref<1x2x3x4xf16, @DDR>
  %2 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x2x3x4xf16, @DDR>) outputs(%1 : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
  return
}

//CHECK func.func @oneDma
//CHECK: [[TB0:%.*]] = VPURegMapped.DeclareTaskBuffer "DMA" -> !VPURegMapped.Index<0:0:0>
//CHECK: VPUMI37XX.NNDMA
//CHECK-SAME: taskLocation([[TB0]] : !VPURegMapped.Index<0:0:0>)

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @multiDMA() {
  %0 = VPURT.DeclareBuffer "NetworkInput" [0] <0> {swizzlingKey = 0 : i64} -> memref<1x16x16x16xf16, #NHWC, @DDR>
  %1 = VPURT.DeclareBuffer "NetworkOutput" [0] <0> {swizzlingKey = 0 : i64} -> memref<1x16x16x16xf16, #NHWC, @DDR>
  %2 = VPURT.DeclareBuffer "NetworkOutput" [1] <0> {swizzlingKey = 0 : i64} -> memref<1x16x16x16xf16, #NHWC, @DDR>

  %3 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
  %4 = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>

  %7 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%3 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
  %8 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%3 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%7 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>
  %9 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%3 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x16x16x16xf16, #NHWC, @DDR>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:1:0>

  %10 = VPUMI37XX.NNDMA {port = 1 : i64} inputs(%0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%4 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:0:0>
  %11 = VPUMI37XX.NNDMA {port = 1 : i64} inputs(%0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%4 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) previousDMA(%10 : !VPURegMapped.Index<1:0:0>) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:0:1>
  %12 = VPUMI37XX.NNDMA {port = 1 : i64} inputs(%4 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) outputs(%2 : memref<1x16x16x16xf16, #NHWC, @DDR>) previousDMA(%11 : !VPURegMapped.Index<1:0:1>) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:1:0>

  return
}

// CHECK: func.func @multiDMA()
//CHECK-DAG: [[TB000:%.*]] = VPURegMapped.DeclareTaskBuffer "DMA" -> !VPURegMapped.Index<0:0:0>
//CHECK-DAG: [[TB001:%.*]] = VPURegMapped.DeclareTaskBuffer "DMA" -> !VPURegMapped.Index<0:0:1>
//CHECK-DAG: [[TB010:%.*]] = VPURegMapped.DeclareTaskBuffer "DMA" -> !VPURegMapped.Index<0:1:0>
//CHECK-DAG: [[TB100:%.*]] = VPURegMapped.DeclareTaskBuffer "DMA" -> !VPURegMapped.Index<1:0:0>
//CHECK-DAG: [[TB101:%.*]] = VPURegMapped.DeclareTaskBuffer "DMA" -> !VPURegMapped.Index<1:0:1>
//CHECK-DAG: [[TB110:%.*]] = VPURegMapped.DeclareTaskBuffer "DMA" -> !VPURegMapped.Index<1:1:0>

//CHECK: VPUMI37XX.NNDMA
    //CHECK-SAME: taskLocation([[TB000]] : !VPURegMapped.Index<0:0:0>)
//CHECK: VPUMI37XX.NNDMA
    //CHECK-SAME: taskLocation([[TB001]] : !VPURegMapped.Index<0:0:1>)
//CHECK: VPUMI37XX.NNDMA
    //CHECK-SAME: taskLocation([[TB010]] : !VPURegMapped.Index<0:1:0>)

//CHECK: VPUMI37XX.NNDMA
    //CHECK-SAME: taskLocation([[TB100]] : !VPURegMapped.Index<1:0:0>)
//CHECK: VPUMI37XX.NNDMA
    //CHECK-SAME: taskLocation([[TB101]] : !VPURegMapped.Index<1:0:1>)
//CHECK: VPUMI37XX.NNDMA
    //CHECK-SAME: taskLocation([[TB110]] : !VPURegMapped.Index<1:1:0>)

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @maxpool_f16_f16() {
  %2 = VPURT.DeclareBuffer "CMX_NN" [0] <8192> -> memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>
  %3 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>
  %7 = VPURT.DeclareBuffer "CMX_NN" [0] <40976> -> memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>

  %14 = VPUMI37XX.DPUInvariant {activation_window_channel_length = 16 : i32, clean_after = 0 : ui64, kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = "CUBOID_16x16", start_after = 0 : ui64, task_type = "MAXPOOL"} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) parent_input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:0> PPE : {
    VPUIP.PPETask "NOOP" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
  }

  %15 = "VPUMI37XX.DPUVariant"(%14) {end = [7, 7, 63], mpe_mode = "CUBOID_16x16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]} : (!VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:0>

  return
}

//CHECK: func.func @maxpool_f16_f16

//CHECK-DAG: [[TB0:%.*]] = VPURegMapped.DeclareTaskBuffer "DPUVariant" -> !VPURegMapped.Index<0:0:0>
//CHECK-DAG: [[TB1:%.*]] = VPURegMapped.DeclareTaskBuffer "DPUInvariant" -> !VPURegMapped.Index<0:0:0>

//CHECK: [[IVAR0:%.*]] = VPUMI37XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TB1]] : !VPURegMapped.Index<0:0:0>)
//CHECK: VPUMI37XX.DPUVariant
    //CHECK-SAME: ([[TB0]], [[IVAR0]])
    //CHECK-SAME: (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>)

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @multiple_clusters_dpu_soh_f16_f16_f16() {
  %6 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
  %7 = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 1]>

  %8 = VPURT.DeclareBuffer "CMX_NN" [0, 1] <4096> -> !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments}>
  %9 = VPURT.DeclareBuffer "CMX_NN" [0] <4096> -> memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 0]>
  %10 = VPURT.DeclareBuffer "CMX_NN" [1] <4096> -> memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 1]>

  %11 = VPURT.DeclareBuffer "CMX_NN" [0, 1] <69632> -> !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1], uniform_distributed_segments}>

  %12 = VPURT.DeclareBuffer "CMX_NN" [0] <69632> -> memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 0]>
  %13 = VPURT.DeclareBuffer "CMX_NN" [1] <69632> -> memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 1]>

  %14 = VPURT.DeclareBuffer "CMX_NN" [0] <102400> -> memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
  %15 = VPURT.DeclareBuffer "CMX_NN" [1] <102400> -> memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 1]>

  %31 = VPUMI37XX.DPUInvariant {clean_after = 0 : ui64, is_segmented, kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_frequent_mode = "CUBOID_16x16", start_after = 0 : ui64, task_type = "CONV"} input(%12 : memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 0]>) weights(%6 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%14 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) parent_input(%11 : !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1], uniform_distributed_segments}>) parent_output(%8 : !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments}>) outputs(%9 : memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:0> PPE : {
  }
  %32 = VPUMI37XX.DPUInvariant {clean_after = 0 : ui64, is_segmented, kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_frequent_mode = "CUBOID_16x16", start_after = 0 : ui64, task_type = "CONV"} input(%13 : memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 1]>) weights(%7 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 1]>) weight_table(%15 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 1]>) parent_input(%11 : !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1], uniform_distributed_segments}>) parent_output(%8 : !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments}>) outputs(%10 : memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 1]>) -> <0:0:1> PPE : {
  }
  %33 = "VPUMI37XX.DPUVariant"(%31) {end = [31, 15, 63], mpe_mode = "CUBOID_16x16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]} : (!VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:0>
  %34 = "VPUMI37XX.DPUVariant"(%32) {end = [31, 31, 63], mpe_mode = "CUBOID_16x16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 16, 0]} : (!VPURegMapped.Index<0:0:1>) -> !VPURegMapped.Index<0:0:1>
  return
}

//CHECK: func.func @multiple_clusters_dpu_soh_f16_f16_f16
//CHECK-DAG: [[TB0:%.*]] = VPURegMapped.DeclareTaskBuffer "DPUVariant" -> !VPURegMapped.Index<0:0:0>
//CHECK-DAG: [[TB1:%.*]] = VPURegMapped.DeclareTaskBuffer "DPUVariant" -> !VPURegMapped.Index<0:0:1>
//CHECK-DAG: [[TB2:%.*]] = VPURegMapped.DeclareTaskBuffer "DPUInvariant" -> !VPURegMapped.Index<0:0:0>
//CHECK-DAG: [[TB3:%.*]] = VPURegMapped.DeclareTaskBuffer "DPUInvariant" -> !VPURegMapped.Index<0:0:1>

//CHECK: [[IVAR0:%.*]] = VPUMI37XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TB2]] : !VPURegMapped.Index<0:0:0>)
//CHECK: [[IVAR1:%.*]] = VPUMI37XX.DPUInvariant
    //CHECK-SAME: taskLocation([[TB3]] : !VPURegMapped.Index<0:0:1>)

//CHECK: VPUMI37XX.DPUVariant
    //CHECK-SAME: ([[TB0]], [[IVAR0]])
    //CHECK-SAME: (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>)
//CHECK: VPUMI37XX.DPUVariant
    //CHECK-SAME: ([[TB1]], [[IVAR1]])
    //CHECK-SAME: (!VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:1>)

// -----

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096]
module @VPU.SW {
  func.func private @builtin_hswish(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "hswish_fp16.cpp", VPU.kernel_entry = "hswish_fp16"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @single_hswish() {
  %2 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
  %3 = VPURT.DeclareBuffer "CMX_NN" [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
  %4 = VPUMI37XX.DeclareKernelText kernel_path("hswish_fp16") -> !VPURegMapped.Index<0:0:0>
  %5 = VPUMI37XX.DeclareKernelEntry kernel_path("hswish_fp16") -> !VPURegMapped.Index<0:0:0>
  %6 = VPUMI37XX.DeclareKernelArgs kernel_path("hswish_fp16") -> !VPURegMapped.Index<0:0:0>
  %7 = VPUMI37XX.KernelParams inputs(%2 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("hswish_fp16") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]> : vector<72xui8>) -> !VPURegMapped.Index<0:0:0>
  %12 = VPUMI37XX.ActKernelRange kernel_text_index(%4 : <0:0:0>) kernel_args_index(%6 : <0:0:0>) kernel_entry_index(%5 : <0:0:0>) -> !VPURegMapped.Index<0:0:0>
  %13 = VPUMI37XX.ActKernelInvocation range_index(%12 : <0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
  return
}

//CHECK: func.func @single_hswish

//CHECK-DAG: [[TB0:%.*]] = VPURegMapped.DeclareTaskBuffer "ActKernelRange" -> !VPURegMapped.Index<0:0:0>
//CHECK-DAG: [[TB1:%.*]] = VPURegMapped.DeclareTaskBuffer "ActKernelInvocation" -> !VPURegMapped.Index<0:0:0>

//CHECK: VPUMI37XX.ActKernelRange
    //CHECK-SAME: taskLocation([[TB0]] : !VPURegMapped.Index<0:0:0>)

//CHECK: VPUMI37XX.ActKernelInvocation
    //CHECK-SAME: taskLocation([[TB1]] : !VPURegMapped.Index<0:0:0>)
