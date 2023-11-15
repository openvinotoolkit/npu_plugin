//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --flatten-sparse-weights-types %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SparseConvWeights(%arg0: memref<1x32x3x3xf16, #NHWC, [@CMX_NN, 0]>, %arg1: memref<1x64x3x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x3x3xf16, #NHWC, [@CMX_NN, 0]> {
  %cst_weights = const.Declare memref<64x32x1x1xf16, {compressionScheme = #VPUIP.CompressionSchemeAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>, order = #NHWC}>
    = dense<1.0> : tensor<64x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false, dense<32> : tensor<64xi64>>]
  %cst_weights_sm = const.Declare memref<64x1x1x128xi1> = dense<1.0> : tensor<64x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
  %cst_weights_table = const.Declare memref<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

  %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

  %1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x32x3x3xf16, #NHWC, [@CMX_NN, 0]>
  %2 = VPURT.DeclareBuffer <CMX_NN> [0] <6688> -> memref<1x64x3x3xf16, #NHWC, [@CMX_NN, 0]>
  %3 = VPURT.DeclareBuffer <CMX_NN> [0] <576>-> memref<64x32x1x1xf16, {compressionScheme = #VPUIP.CompressionSchemeAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>, order = #NHWC}, [@CMX_NN, 0]>
  %4 = VPURT.DeclareBuffer <CMX_NN> [0] <4672>-> memref<64x1x1x128xi1, [@CMX_NN, 0]>
  %5 = VPURT.DeclareBuffer <CMX_NN> [0] <5696> -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>

  VPURT.Task updates(%0 : !VPURT.Barrier) attributes {cycleBegin = 0 : i64, cycleEnd = 100 : i64, isTrailingSWLayer = false} {
    %6 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_weights : memref<64x32x1x1xf16, {compressionScheme = #VPUIP.CompressionSchemeAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>, order = #NHWC}>)
                                      outputs(%3 : memref<64x32x1x1xf16, {compressionScheme = #VPUIP.CompressionSchemeAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>, order = #NHWC}, [@CMX_NN, 0]>)
            -> memref<64x32x1x1xf16, {compressionScheme = #VPUIP.CompressionSchemeAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>, order = #NHWC}, [@CMX_NN, 0]>
  }
  VPURT.Task updates(%0 : !VPURT.Barrier) attributes {cycleBegin = 100 : i64, cycleEnd = 200 : i64, isTrailingSWLayer = false} {
    %6 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_weights_sm : memref<64x1x1x128xi1>) outputs(%4 : memref<64x1x1x128xi1, [@CMX_NN, 0]>) -> memref<64x1x1x128xi1, [@CMX_NN, 0]>
  }
  VPURT.Task updates(%0 : !VPURT.Barrier) attributes {cycleBegin = 200 : i64, cycleEnd = 300 : i64, isTrailingSWLayer = false} {
    %6 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_weights_table : memref<64x1x1x4xsi32>) outputs(%5 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
  }

  VPURT.Task waits(%0 : !VPURT.Barrier) attributes {cycleBegin = 300 : i64, cycleEnd = 400 : i64, isTrailingSWLayer = false} {
    %6 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
            input(%1 : memref<1x32x3x3xf16, #NHWC, [@CMX_NN, 0]>)
            weights(%3 : memref<64x32x1x1xf16, {compressionScheme = #VPUIP.CompressionSchemeAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>, order = #NHWC}, [@CMX_NN, 0]>)
            weights_sparsity_map(%4 : memref<64x1x1x128xi1, [@CMX_NN, 0]>)
            weight_table(%5 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%1 : memref<1x32x3x3xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%2 : memref<1x64x3x3xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%2 : memref<1x64x3x3xf16, #NHWC, [@CMX_NN, 0]>)
    -> memref<1x64x3x3xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 2, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask <NOOP> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
    }
  }

  return %arg1 : memref<1x64x3x3xf16, #NHWC, [@CMX_NN, 0]>

  // CHECK:       [[CST_WEIGHTS:%.+]] = const.Declare memref<4096x1x1x1xui8, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x1x1xf16>,
  // CHECK-SAME:      [#const.Reorder<#NHWC>, #const.Sparsify<true, dense<32> : tensor<64xi64>>]

  // CHECK:       [[WEIGHTS_CMX_DENSE:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<64x32x1x1xf16,
  // CHECK-SAME:      {compressionScheme = #VPUIP.CompressionSchemeAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>, order = #NHWC}, [@CMX_NN, 0]>
  // CHECK:       [[WEIGHTS_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<4096x1x1x1xui8, {order = #NHWC}, [@CMX_NN, 0]>

  // CHECK:       VPURT.Task
  // CHECK:           VPUIP.NNDMA {port = 0 : i64} inputs([[CST_WEIGHTS]] : memref<4096x1x1x1xui8, {order = #NHWC}>)
  // CHECK-SAME:                                   outputs([[WEIGHTS_CMX]] : memref<4096x1x1x1xui8, {order = #NHWC}, [@CMX_NN, 0]>)
  // CHECK-SAME:        -> memref<4096x1x1x1xui8, {order = #NHWC}, [@CMX_NN, 0]>

  // CHECK:       VPURT.Task
  // CHECK:           VPUIP.NCEClusterTask
  // CHECK-SAME:          weights([[WEIGHTS_CMX_DENSE]] : memref<64x32x1x1xf16, {compressionScheme = #VPUIP.CompressionSchemeAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>, order = #NHWC}, [@CMX_NN, 0]>)
}
