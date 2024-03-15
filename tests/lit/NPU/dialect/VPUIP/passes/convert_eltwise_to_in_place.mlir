//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-eltwise-to-in-place --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0047491008160161037:146>

!qTypeCMX = memref<1x32x103x512x!qElemType, #NHWC, @CMX_NN>
!qTypeDDR = memref<1x32x103x512x!qElemType, #NHWC>

!DistributedType1 = !VPUIP.DistributedBuffer<
    1x32x103x512x!qElemType,
    #NHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [1, 1, 2, 1],
        num_clusters = 2 : i64
}>

// CHECK-LABEL: InplaceEltwiseToEltwise
func.func @InplaceEltwiseToEltwise(%in1: !qTypeDDR, %in2: !qTypeDDR, %in3: !qTypeDDR) -> (!DistributedType1) {
    %eltwise_1_cmx_input_buf_1 = VPURT.AllocDistributed -> !DistributedType1
    %eltwise_1_cmx_input_1 = VPUIP.NCEClusterTiling inputs(%in1 as %arg4: !qTypeDDR) outputs(%eltwise_1_cmx_input_buf_1 as %arg5: !qTypeCMX) -> !DistributedType1 {
      %copy_to_cmx_1 = VPUIP.Copy inputs(%arg4 : !qTypeDDR) outputs(%arg5 : !qTypeCMX) -> !qTypeCMX
    }

    %eltwise_1_cmx_input_buf_2 = VPURT.AllocDistributed -> !DistributedType1
    %eltwise_1_cmx_input_2 = VPUIP.NCEClusterTiling inputs(%in2 as %arg4: !qTypeDDR) outputs(%eltwise_1_cmx_input_buf_2 as %arg5: !qTypeCMX) -> !DistributedType1 {
      %copy_to_cmx_2 = VPUIP.Copy inputs(%arg4 : !qTypeDDR) outputs(%arg5 : !qTypeCMX) -> !qTypeCMX
    }

    %eltwise_1_cmx_out_buf = VPURT.AllocDistributed -> !DistributedType1
    %eltwise_1 = VPUIP.NCEClusterTiling inputs(%eltwise_1_cmx_input_1 as %arg4: !qTypeCMX, %eltwise_1_cmx_input_2 as %arg5: !qTypeCMX) outputs(%eltwise_1_cmx_out_buf as %arg6: !qTypeCMX) -> !DistributedType1 {
      %eltwise_1_inner = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i64, is_inplace = true, minimumHardwareExecutionCost = 21125 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%arg4 : !qTypeCMX) weights(%arg5 : !qTypeCMX) parent_input(%arg4 : !qTypeCMX) parent_output(%arg6 : !qTypeCMX) outputs(%arg6 : !qTypeCMX) -> !qTypeCMX
        variants : {
          DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [511, 51, 31], outStart = [0, 0, 0], pad = #VPU.Padding<bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64>}
          DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [511, 102, 31], outStart = [0, 52, 0], pad = #VPU.Padding<bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64>}
        } PPE : {
          PPETask <NOOP> {clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, in1_quant_mult = [19919], in2_quant_mult = [7511], lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [27275], quant_post_shift = 0 : i64, quant_shift = [29]}
        }
    }

    %eltwise_2_cmx_input_buf_1 = VPURT.AllocDistributed -> !DistributedType1
    %eltwise_2_cmx_input_1 = VPUIP.NCEClusterTiling inputs(%in3 as %arg4: !qTypeDDR) outputs(%eltwise_2_cmx_input_buf_1 as %arg5: !qTypeCMX) -> !DistributedType1 {
      %copy_to_cmx_2 = VPUIP.Copy inputs(%arg4 : !qTypeDDR) outputs(%arg5 : !qTypeCMX) -> !qTypeCMX
    }

    %eltwise_2_cmx_out_buf = VPURT.AllocDistributed -> !DistributedType1
    %eltwise_2 = VPUIP.NCEClusterTiling inputs(%eltwise_1 as %arg4: !qTypeCMX, %eltwise_2_cmx_input_1 as %arg5: !qTypeCMX) outputs(%eltwise_2_cmx_out_buf as %arg6: !qTypeCMX) -> !DistributedType1 {
      %eltwise_2_inner = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i64, is_inplace = true, minimumHardwareExecutionCost = 21125 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%arg4 : !qTypeCMX) weights(%arg5 : !qTypeCMX) parent_input(%arg4 : !qTypeCMX) parent_output(%arg6 : !qTypeCMX) outputs(%arg6 : !qTypeCMX) -> !qTypeCMX
        variants : {
          DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [511, 51, 31], outStart = [0, 0, 0], pad = #VPU.Padding<bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64>}
          DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [511, 102, 31], outStart = [0, 52, 0], pad = #VPU.Padding<bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64>}
        } PPE : {
          PPETask <NOOP> {clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, in1_quant_mult = [19919], in2_quant_mult = [7511], lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [27275], quant_post_shift = 0 : i64, quant_shift = [29]}
       }
    }

    return %eltwise_2 : !DistributedType1

  // CHECK: [[ELTWISE_1_INPUT_BUF_1:%.+]] = VPURT.AllocDistributed
  // CHECK: [[ELTWISE_1_INPUT_1:%.+]] = VPUIP.NCEClusterTiling
  // CHECK: [[ELTWISE_1_INPUT_BUF_2:%.+]] = VPURT.AllocDistributed 
  // CHECK: [[ELTWISE_1_INPUT_2:%.+]] = VPUIP.NCEClusterTiling
  // CHECK: [[ELTWISE_1:%.+]] = VPUIP.NCEClusterTiling inputs(
  // CHECK-SAME:  [[ELTWISE_1_INPUT_1]] as %arg3: memref<1x32x103x512x!qElemType, #NHWC, @CMX_NN>,
  // CHECK-SAME:  [[ELTWISE_1_INPUT_2]] as %arg4: memref<1x32x103x512x!qElemType, #NHWC, @CMX_NN>)
  // CHECK-SAME:  outputs([[ELTWISE_1_INPUT_BUF_1]] as %arg5: memref<1x32x103x512x!qElemType, #NHWC, @CMX_NN>)
  
  // CHECK: [[ELTWISE_2_INPUT_BUF_1:%.+]] = VPURT.AllocDistributed 
  // CHECK: [[ELTWISE_2_INPUT_1:%.+]] = VPUIP.NCEClusterTiling
  // CHECK: [[ELTWISE_2:%.+]] = VPUIP.NCEClusterTiling inputs(
  // CHECK-SAME:  [[ELTWISE_1]] as %arg3: memref<1x32x103x512x!qElemType, #NHWC, @CMX_NN>,
  // CHECK-SAME:  [[ELTWISE_2_INPUT_1]] as %arg4: memref<1x32x103x512x!qElemType, #NHWC, @CMX_NN>)
  // CHECK-SAME:  outputs([[ELTWISE_1_INPUT_BUF_1]] as %arg5: memref<1x32x103x512x!qElemType, #NHWC, @CMX_NN>)
 
  // CHECK: return [[ELTWISE_2]] : !VPUIP.DistributedBuffer<1x32x103x512x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> 
}
