//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --copy-op-hoisting %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CopyFromBlockArgumentNoChange
func.func @CopyFromBlockArgumentNoChange(%arg0: memref<1x16x1x1xf16, @CMX_NN>, %arg1: memref<1x16x1x1xf16>, %arg2: memref<1x16x1x1xf16>)
        -> (memref<1x16x1x1xf16>, memref<1x16x1x1xf16>) {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x16x1x1xf16, @CMX_NN>) outputs(%arg1 : memref<1x16x1x1xf16>) -> memref<1x16x1x1xf16>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x1x1xf16, @CMX_NN>) outputs(%arg2 : memref<1x16x1x1xf16>) -> memref<1x16x1x1xf16>
    return %0, %1 : memref<1x16x1x1xf16>, memref<1x16x1x1xf16>

    // CHECK: [[VAR0:%.+]] = VPUIP.Copy inputs(%arg0 : memref<1x16x1x1xf16, @CMX_NN>) outputs(%arg1 : memref<1x16x1x1xf16>)
    // CHECK: [[VAR1:%.+]] = VPUIP.Copy inputs(%arg0 : memref<1x16x1x1xf16, @CMX_NN>) outputs(%arg2 : memref<1x16x1x1xf16>)
    // CHECK: return [[VAR0]], [[VAR1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CopyToBlockArgumentNoChange
func.func @CopyToBlockArgumentNoChange(%arg0: memref<1x16x1x1xf16, #NHWC, @CMX_NN>, %arg1: memref<1x16x1x1xf16>) -> memref<1x16x1x1xf16> {
    %wt = const.Declare memref<16x1x1x4xsi32, @CMX_NN> = dense<1> : tensor<16x1x1x4xsi32>
    %act_win = const.Declare memref<1x1x1x16xui8, @CMX_NN> = dense<1> : tensor<1x1x1x16xui8>
    %0 = memref.alloc() : memref<1x16x1x1xf16, #NHWC, @CMX_NN>
    %1 = VPUIP.NCEClusterTask {
            activation_window_channel_length = 27 : i64,
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = "MAXPOOL"
        }
        input(%arg0 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
        weight_table(%wt : memref<16x1x1x4xsi32, @CMX_NN>)
        activation_window(%act_win : memref<1x1x1x16xui8, @CMX_NN>)
        parent_input(%arg0 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
        parent_output(%0 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
        outputs(%0 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>) -> memref<1x16x1x1xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask { outEnd = [16, 1, 1], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
        }
        PPE : {
        }

    %2 = VPUIP.Copy inputs(%1 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>) outputs(%arg1 : memref<1x16x1x1xf16>) -> memref<1x16x1x1xf16>
    return %2 : memref<1x16x1x1xf16>

    // CHECK: [[VAR0:%.+]] = memref.alloc() : memref<1x16x1x1xf16, #NHWC, @CMX_NN>
    // CHECK: [[VAR1:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME: input(%arg0 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME: outputs([[VAR0]] : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK: [[VAR2:%.+]] = VPUIP.Copy inputs([[VAR1]] : memref<1x16x1x1xf16, #NHWC, @CMX_NN>) outputs(%arg1 : memref<1x16x1x1xf16>)
    // CHECK: return [[VAR2]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CopyToBlockArgumentMoveCopyOnly
func.func @CopyToBlockArgumentMoveCopyOnly(%arg0: memref<1x16x1x1xf16, #NHWC, @CMX_NN>, %arg1: memref<1x16x1x1xf16>, %arg2: memref<1x16x1x1xf16>) -> (memref<1x16x1x1xf16>, memref<1x16x1x1xf16>) {
    %wt = const.Declare memref<16x1x1x4xsi32, @CMX_NN> = dense<1> : tensor<16x1x1x4xsi32>
    %act_win = const.Declare memref<1x1x1x16xui8, @CMX_NN> = dense<1> : tensor<1x1x1x16xui8>
    %0 = memref.alloc() : memref<1x16x1x1xf16, #NHWC, @CMX_NN>
    %1 = memref.alloc() : memref<1x16x1x1xf16, #NHWC, @CMX_NN>
    %2 = VPUIP.NCEClusterTask {
            activation_window_channel_length = 27 : i64,
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = "MAXPOOL"
        }
        input(%arg0 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
        weight_table(%wt : memref<16x1x1x4xsi32, @CMX_NN>)
        activation_window(%act_win : memref<1x1x1x16xui8, @CMX_NN>)
        parent_input(%arg0 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
        parent_output(%0 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
        outputs(%0 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>) -> memref<1x16x1x1xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask { outEnd = [16, 1, 1], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
        }
        PPE : {
        }
    %3 = VPUIP.NCEClusterTask {
            activation_window_channel_length = 27 : i64,
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = "MAXPOOL"
        }
        input(%arg0 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
        weight_table(%wt : memref<16x1x1x4xsi32, @CMX_NN>)
        activation_window(%act_win : memref<1x1x1x16xui8, @CMX_NN>)
        parent_input(%arg0 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
        parent_output(%1 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
        outputs(%1 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>) -> memref<1x16x1x1xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask { outEnd = [16, 1, 1], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
        }
        PPE : {
        }
    %4 = VPUIP.Copy inputs(%2 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>) outputs(%arg1 : memref<1x16x1x1xf16>) -> memref<1x16x1x1xf16>
    %5 = VPUIP.Copy inputs(%3 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x16x1x1xf16>) -> memref<1x16x1x1xf16>
    return %4, %5 : memref<1x16x1x1xf16>, memref<1x16x1x1xf16>

    // CHECK: [[VAR0:%.+]] = memref.alloc() : memref<1x16x1x1xf16, #NHWC, @CMX_NN>
    // CHECK: [[VAR1:%.+]] = memref.alloc() : memref<1x16x1x1xf16, #NHWC, @CMX_NN>
    // CHECK: [[VAR2:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME: input(%arg0 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME: outputs([[VAR0]] : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK: [[VAR3:%.+]] = VPUIP.Copy inputs([[VAR2]] : memref<1x16x1x1xf16, #NHWC, @CMX_NN>) outputs(%arg1 : memref<1x16x1x1xf16>)
    // CHECK: [[VAR4:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME: input(%arg0 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME: outputs([[VAR1]] : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK: [[VAR5:%.+]] = VPUIP.Copy inputs([[VAR4]] : memref<1x16x1x1xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x16x1x1xf16>)
    // CHECK: return [[VAR3]], [[VAR5]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CopyToBlockArgumentSubView
func.func @CopyToBlockArgumentSubView(%arg0: memref<1x8x1x1xf16, #NHWC, @CMX_NN>, %arg1: memref<16x1x1xf16>) -> memref<16x1x1xf16> {
    %wt = const.Declare memref<8x1x1x4xsi32, @CMX_NN> = dense<1> : tensor<8x1x1x4xsi32>
    %act_win = const.Declare memref<1x1x1x8xui8, @CMX_NN> = dense<1> : tensor<1x1x1x8xui8>
    %0 = memref.alloc() : memref<1x8x1x1xf16, #NHWC, @CMX_NN>
    %1 = memref.alloc() : memref<1x8x1x1xf16, #NHWC, @CMX_NN>
    %2 = VPUIP.NCEClusterTask {
            activation_window_channel_length = 27 : i64,
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = "MAXPOOL"
        }
        input(%arg0 : memref<1x8x1x1xf16, #NHWC, @CMX_NN>)
        weight_table(%wt : memref<8x1x1x4xsi32, @CMX_NN>)
        activation_window(%act_win : memref<1x1x1x8xui8, @CMX_NN>)
        parent_input(%arg0 : memref<1x8x1x1xf16, #NHWC, @CMX_NN>)
        parent_output(%0 : memref<1x8x1x1xf16, #NHWC, @CMX_NN>)
        outputs(%0 : memref<1x8x1x1xf16, #NHWC, @CMX_NN>) -> memref<1x8x1x1xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask { outEnd = [16, 1, 1], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
        }
        PPE : {
        }
    %3 = VPUIP.NCEClusterTask {
            activation_window_channel_length = 27 : i64,
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = "MAXPOOL"
        }
        input(%arg0 : memref<1x8x1x1xf16, #NHWC, @CMX_NN>)
        weight_table(%wt : memref<8x1x1x4xsi32, @CMX_NN>)
        activation_window(%act_win : memref<1x1x1x8xui8, @CMX_NN>)
        parent_input(%arg0 : memref<1x8x1x1xf16, #NHWC, @CMX_NN>)
        parent_output(%1 : memref<1x8x1x1xf16, #NHWC, @CMX_NN>)
        outputs(%1 : memref<1x8x1x1xf16, #NHWC, @CMX_NN>) -> memref<1x8x1x1xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask { outEnd = [16, 1, 1], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
        }
        PPE : {
        }
    %4 = VPUIP.SubView %arg1 [0, 0, 0] [8, 1, 1] : memref<16x1x1xf16> to memref<8x1x1xf16>
    %5 = VPUIP.Copy inputs(%2 : memref<1x8x1x1xf16, #NHWC, @CMX_NN>) outputs(%4 : memref<8x1x1xf16>) -> memref<8x1x1xf16>
    %6 = VPUIP.SubView %arg1 [8, 0, 0] [8, 1, 1] : memref<16x1x1xf16> to memref<8x1x1xf16>
    %7 = VPUIP.Copy inputs(%3 : memref<1x8x1x1xf16, #NHWC, @CMX_NN>) outputs(%6 : memref<8x1x1xf16>) -> memref<8x1x1xf16>
    %8 = VPUIP.ConcatView inputs(%5, %7 : memref<8x1x1xf16>, memref<8x1x1xf16>) outputs(%arg1 : memref<16x1x1xf16>) -> memref<16x1x1xf16>
    return %8 : memref<16x1x1xf16>

    // CHECK: [[VAR0:%.+]] = memref.alloc() : memref<1x8x1x1xf16, #NHWC, @CMX_NN>
    // CHECK: [[VAR1:%.+]] = memref.alloc() : memref<1x8x1x1xf16, #NHWC, @CMX_NN>
    // CHECK: [[VAR2:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME: input(%arg0 : memref<1x8x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME: outputs([[VAR0]] : memref<1x8x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK: [[VAR3:%.+]] = VPUIP.SubView %arg1 [0, 0, 0] [8, 1, 1]
    // CHECK: [[VAR4:%.+]] = VPUIP.Copy inputs([[VAR2]] : memref<1x8x1x1xf16, #NHWC, @CMX_NN>) outputs([[VAR3]] : memref<8x1x1xf16>)
    // CHECK: [[VAR5:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME: input(%arg0 : memref<1x8x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME: outputs([[VAR1]] : memref<1x8x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK: [[VAR6:%.+]] = VPUIP.SubView %arg1 [8, 0, 0] [8, 1, 1]
    // CHECK: [[VAR7:%.+]] = VPUIP.Copy inputs([[VAR5]] : memref<1x8x1x1xf16, #NHWC, @CMX_NN>) outputs([[VAR6]] : memref<8x1x1xf16>)
    // CHECK: [[VAR8:%.+]] = VPUIP.ConcatView inputs([[VAR4]], [[VAR7]] : memref<8x1x1xf16>, memref<8x1x1xf16>) outputs(%arg1 : memref<16x1x1xf16>)
    // CHECK: return [[VAR8]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!SparseInputCMXBufferType = !VPUIP.SparseBuffer<
    data=memref<1x16x1x1xf16, @CMX_NN>,
    sparsity_map=memref<1x16x1x1xi1, @CMX_NN>
>

!SparseInputDDRBufferType = !VPUIP.SparseBuffer<
    data=memref<1x16x1x1xf16, @DDR>,
    sparsity_map=memref<1x16x1x1xi1, @DDR>
>

// CHECK-LABEL: @CopyFromBlockArgumentNoChangeSparse
func.func @CopyFromBlockArgumentNoChangeSparse(%arg0: !SparseInputCMXBufferType, %arg1: !SparseInputDDRBufferType, %arg2: !SparseInputDDRBufferType)
        -> (!SparseInputDDRBufferType, !SparseInputDDRBufferType) {
    %0 = VPUIP.Copy inputs(%arg0 : !SparseInputCMXBufferType) outputs(%arg1 : !SparseInputDDRBufferType) -> !SparseInputDDRBufferType
    %1 = VPUIP.Copy inputs(%arg0 : !SparseInputCMXBufferType) outputs(%arg2 : !SparseInputDDRBufferType) -> !SparseInputDDRBufferType
    return %0, %1 : !SparseInputDDRBufferType, !SparseInputDDRBufferType

    // CHECK:       [[VAR0:%.+]] = VPUIP.Copy inputs(%arg0 : !VPUIP.SparseBuffer<data=memref<1x16x1x1xf16, @CMX_NN>, sparsity_map=memref<1x16x1x1xi1, @CMX_NN>>)
    // CHECK-SAME:                      outputs(%arg1 : !VPUIP.SparseBuffer<data=memref<1x16x1x1xf16, @DDR>, sparsity_map=memref<1x16x1x1xi1, @DDR>>)
    // CHECK:       [[VAR1:%.+]] = VPUIP.Copy inputs(%arg0 : !VPUIP.SparseBuffer<data=memref<1x16x1x1xf16, @CMX_NN>, sparsity_map=memref<1x16x1x1xi1, @CMX_NN>>)
    // CHECK-SAME:                      outputs(%arg2 : !VPUIP.SparseBuffer<data=memref<1x16x1x1xf16, @DDR>, sparsity_map=memref<1x16x1x1xi1, @DDR>>)
    // CHECK: return [[VAR0]], [[VAR1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IOZMajorDataType = memref<1x16x1x1xf16, #NHWC, @CMX_NN>
!IOZMajorSMType = memref<1x16x1x1xi1, #NHWC, @CMX_NN>

!IOZMajorSparseType = !VPUIP.SparseBuffer<
    data=!IOZMajorDataType,
    sparsity_map=!IOZMajorSMType
>

!SparseOutputDDRBufferType = !VPUIP.SparseBuffer<
    data=memref<1x16x1x1xf16>,
    sparsity_map=memref<1x16x1x1xi1>
>

// CHECK-LABEL: @CopyToBlockArgumentNoChangeSparse
func.func @CopyToBlockArgumentNoChangeSparse(%arg0: !IOZMajorSparseType, %arg1: !SparseOutputDDRBufferType) -> !SparseOutputDDRBufferType {
    %wt = const.Declare memref<16x1x1x4xsi32, @CMX_NN> = dense<1> : tensor<16x1x1x4xsi32>
    %act_win = const.Declare memref<1x1x1x16xui8, @CMX_NN> = dense<1> : tensor<1x1x1x16xui8>
    %0 = memref.alloc() : !IOZMajorDataType
    %1 = memref.alloc() : !IOZMajorSMType
    %2 = VPUIP.GroupSparseBuffer(%0, %1) -> !IOZMajorSparseType

    %in_data, %in_sm = VPUIP.UngroupSparseBuffer(%arg0) {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>}
        -> !IOZMajorDataType, !IOZMajorSMType
    %out_data, %out_sm = VPUIP.UngroupSparseBuffer(%2) {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>}
        -> !IOZMajorDataType, !IOZMajorSMType

    %3:2 = VPUIP.NCEClusterTask {
            activation_window_channel_length = 27 : i64,
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = "MAXPOOL"
        }
        input(%in_data : !IOZMajorDataType)
        input_sparsity_map(%in_sm : !IOZMajorSMType)
        weight_table(%wt : memref<16x1x1x4xsi32, @CMX_NN>)
        activation_window(%act_win : memref<1x1x1x16xui8, @CMX_NN>)
        parent_input(%in_data : !IOZMajorDataType)
        parent_input_sparsity_map(%in_sm : !IOZMajorSMType)
        parent_output(%out_data : !IOZMajorDataType)
        parent_output_sparsity_map(%out_sm : !IOZMajorSMType)
        outputs(%out_data : !IOZMajorDataType)
        output_sparsity_map(%out_sm : !IOZMajorSMType) -> !IOZMajorDataType, !IOZMajorSMType
        variants :
        {
            DPUTask { outEnd = [16, 1, 1], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
        }
        PPE : {
        }
    %4 = VPUIP.GroupSparseBuffer(%3#0, %3#1) -> !IOZMajorSparseType

    %5 = VPUIP.Copy inputs(%4 : !IOZMajorSparseType) outputs(%arg1 : !SparseOutputDDRBufferType) -> !SparseOutputDDRBufferType
    return %5 : !SparseOutputDDRBufferType

    // CHECK:       [[BUFF_0_DATA:%.+]] = memref.alloc() : memref<1x16x1x1xf16, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x16x1x1xi1, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_0:%.*]] = VPUIP.GroupSparseBuffer([[BUFF_0_DATA]], [[BUFF_0_SM]])

    // CHECK:       [[DATA_0:%.*]], [[SM_0:%.*]] = VPUIP.UngroupSparseBuffer(%arg0)
    // CHECK:       [[DATA_1:%.*]], [[SM_1:%.*]] = VPUIP.UngroupSparseBuffer([[BUFF_0]])

    // CHECK:       [[NCE0:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[DATA_0]] : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      input_sparsity_map([[SM_0]] : memref<1x16x1x1xi1, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[DATA_1]] : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      output_sparsity_map([[SM_1]] : memref<1x16x1x1xi1, #NHWC, @CMX_NN>)

    // CHECK:       [[VAR1:%.*]] = VPUIP.GroupSparseBuffer([[NCE0]]#0, [[NCE0]]#1)

    // CHECK:       [[VAR2:%.+]] = VPUIP.Copy inputs([[VAR1]] : !VPUIP.SparseBuffer<data=memref<1x16x1x1xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x16x1x1xi1, #NHWC, @CMX_NN>>)
    // CHECK-SAME:      outputs(%arg1 : !VPUIP.SparseBuffer<data=memref<1x16x1x1xf16>, sparsity_map=memref<1x16x1x1xi1>>)
    // CHECK:       return [[VAR2]]
}

// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IOZMajorDataType = memref<1x16x1x1xf16, #NHWC, @CMX_NN>
!IOZMajorSMType = memref<1x16x1x1xi1, #NHWC, @CMX_NN>

!IOZMajorSparseType = !VPUIP.SparseBuffer<
    data=!IOZMajorDataType,
    sparsity_map=!IOZMajorSMType
>

!SparseOutputDDRBufferType = !VPUIP.SparseBuffer<
    data=memref<1x16x1x1xf16>,
    sparsity_map=memref<1x16x1x1xi1>
>
// CHECK-LABEL: @CopyToBlockArgumentMoveCopyOnlySparse
func.func @CopyToBlockArgumentMoveCopyOnlySparse(%arg0: !IOZMajorSparseType, %arg1: !SparseOutputDDRBufferType, %arg2: !SparseOutputDDRBufferType) -> (!SparseOutputDDRBufferType, !SparseOutputDDRBufferType) {
    %wt = const.Declare memref<16x1x1x4xsi32, @CMX_NN> = dense<1> : tensor<16x1x1x4xsi32>
    %act_win = const.Declare memref<1x1x1x16xui8, @CMX_NN> = dense<1> : tensor<1x1x1x16xui8>

    %0 = memref.alloc() : !IOZMajorDataType
    %1 = memref.alloc() : !IOZMajorSMType
    %2 = VPUIP.GroupSparseBuffer(%0, %1) -> !IOZMajorSparseType

    %3 = memref.alloc() : !IOZMajorDataType
    %4 = memref.alloc() : !IOZMajorSMType
    %5 = VPUIP.GroupSparseBuffer(%3, %4) -> !IOZMajorSparseType

    %in_data, %in_sm = VPUIP.UngroupSparseBuffer(%arg0) {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>}
        -> !IOZMajorDataType, !IOZMajorSMType
    %out_data_0, %out_sm_0 = VPUIP.UngroupSparseBuffer(%2) {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>}
        -> !IOZMajorDataType, !IOZMajorSMType
    %out_data_1, %out_sm_1 = VPUIP.UngroupSparseBuffer(%5) {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>}
        -> !IOZMajorDataType, !IOZMajorSMType

    %6:2 = VPUIP.NCEClusterTask {
            activation_window_channel_length = 27 : i64,
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = "MAXPOOL"
        }
        input(%in_data : !IOZMajorDataType)
        input_sparsity_map(%in_sm : !IOZMajorSMType)
        weight_table(%wt : memref<16x1x1x4xsi32, @CMX_NN>)
        activation_window(%act_win : memref<1x1x1x16xui8, @CMX_NN>)
        parent_input(%in_data : !IOZMajorDataType)
        parent_input_sparsity_map(%in_sm : !IOZMajorSMType)
        parent_output(%out_data_0 : !IOZMajorDataType)
        parent_output_sparsity_map(%out_sm_0 : !IOZMajorSMType)
        outputs(%out_data_0 : !IOZMajorDataType)
        output_sparsity_map(%out_sm_0 : !IOZMajorSMType) -> !IOZMajorDataType, !IOZMajorSMType
        variants :
        {
            DPUTask { outEnd = [16, 1, 1], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
        }
        PPE : {
        }
    %7 = VPUIP.GroupSparseBuffer(%6#0, %6#1) -> !IOZMajorSparseType

    %8:2 = VPUIP.NCEClusterTask {
            activation_window_channel_length = 27 : i64,
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = "MAXPOOL"
        }
        input(%in_data : !IOZMajorDataType)
        input_sparsity_map(%in_sm : !IOZMajorSMType)
        weight_table(%wt : memref<16x1x1x4xsi32, @CMX_NN>)
        activation_window(%act_win : memref<1x1x1x16xui8, @CMX_NN>)
        parent_input(%in_data : !IOZMajorDataType)
        parent_input_sparsity_map(%in_sm : !IOZMajorSMType)
        parent_output(%out_data_1 : !IOZMajorDataType)
        parent_output_sparsity_map(%out_sm_1 : !IOZMajorSMType)
        outputs(%out_data_1 : !IOZMajorDataType)
        output_sparsity_map(%out_sm_1 : !IOZMajorSMType) -> !IOZMajorDataType, !IOZMajorSMType
        variants :
        {
            DPUTask { outEnd = [16, 1, 1], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
        }
        PPE : {
        }
    %9 = VPUIP.GroupSparseBuffer(%8#0, %8#1) -> !IOZMajorSparseType

    %10 = VPUIP.Copy inputs(%7 : !IOZMajorSparseType) outputs(%arg1 : !SparseOutputDDRBufferType) -> !SparseOutputDDRBufferType
    %11 = VPUIP.Copy inputs(%9 : !IOZMajorSparseType) outputs(%arg2 : !SparseOutputDDRBufferType) -> !SparseOutputDDRBufferType
    return %10, %11 : !SparseOutputDDRBufferType, !SparseOutputDDRBufferType

    // CHECK:       [[BUFF_0_DATA:%.+]] = memref.alloc() : memref<1x16x1x1xf16, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x16x1x1xi1, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_0:%.*]] = VPUIP.GroupSparseBuffer([[BUFF_0_DATA]], [[BUFF_0_SM]])

    // CHECK:       [[BUFF_1_DATA:%.+]] = memref.alloc() : memref<1x16x1x1xf16, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_1_SM:%.+]] = memref.alloc() : memref<1x16x1x1xi1, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_1:%.*]] = VPUIP.GroupSparseBuffer([[BUFF_1_DATA]], [[BUFF_1_SM]])

    // CHECK:       [[DATA_0:%.*]], [[SM_0:%.*]] = VPUIP.UngroupSparseBuffer(%arg0)
    // CHECK:       [[DATA_1:%.*]], [[SM_1:%.*]] = VPUIP.UngroupSparseBuffer([[BUFF_0]])
    // CHECK:       [[DATA_2:%.*]], [[SM_2:%.*]] = VPUIP.UngroupSparseBuffer([[BUFF_1]])

    // CHECK:       [[NCE0:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[DATA_0]] : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      input_sparsity_map([[SM_0]] : memref<1x16x1x1xi1, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[DATA_1]] : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      output_sparsity_map([[SM_1]] : memref<1x16x1x1xi1, #NHWC, @CMX_NN>)

    // CHECK:       [[VAR1:%.*]] = VPUIP.GroupSparseBuffer([[NCE0]]#0, [[NCE0]]#1)
    // CHECK:       [[VAR2:%.+]] = VPUIP.Copy inputs([[VAR1]] : !VPUIP.SparseBuffer<data=memref<1x16x1x1xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x16x1x1xi1, #NHWC, @CMX_NN>>)
    // CHECK-SAME:      outputs(%arg1 : !VPUIP.SparseBuffer<data=memref<1x16x1x1xf16>, sparsity_map=memref<1x16x1x1xi1>>)


    // CHECK:       [[NCE1:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[DATA_0]] : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      input_sparsity_map([[SM_0]] : memref<1x16x1x1xi1, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[DATA_2]] : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      output_sparsity_map([[SM_2]] : memref<1x16x1x1xi1, #NHWC, @CMX_NN>)

    // CHECK:       [[VAR3:%.*]] = VPUIP.GroupSparseBuffer([[NCE1]]#0, [[NCE1]]#1)
    // CHECK:       [[VAR4:%.+]] = VPUIP.Copy inputs([[VAR3]] : !VPUIP.SparseBuffer<data=memref<1x16x1x1xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x16x1x1xi1, #NHWC, @CMX_NN>>)
    // CHECK-SAME:      outputs(%arg2 : !VPUIP.SparseBuffer<data=memref<1x16x1x1xf16>, sparsity_map=memref<1x16x1x1xi1>>)

    // CHECK: return [[VAR2]], [[VAR4]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IOZMajorDataType = memref<1x8x1x1xf16, #NHWC, @CMX_NN>
!IOZMajorSMType = memref<1x8x1x1xi1, #NHWC, @CMX_NN>

!IOZMajorSparseType = !VPUIP.SparseBuffer<
    data=!IOZMajorDataType,
    sparsity_map=!IOZMajorSMType
>

!SparseOutputDDRBufferType = !VPUIP.SparseBuffer<
    data=memref<1x16x1x1xf16>,
    sparsity_map=memref<1x16x1x1xi1>
>

!SparseHalfOutputDDRBufferType = !VPUIP.SparseBuffer<
    data=memref<1x8x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [16, 1, 1, 1]}>,
    sparsity_map=memref<1x8x1x1xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [16, 1, 1, 1]}>
>

// CHECK-LABEL: @CopyToBlockArgumentSubViewSparse
func.func @CopyToBlockArgumentSubViewSparse(%arg0: !IOZMajorSparseType, %arg1: !SparseOutputDDRBufferType) -> !SparseOutputDDRBufferType {
    %wt = const.Declare memref<8x1x1x4xsi32, @CMX_NN> = dense<1> : tensor<8x1x1x4xsi32>
    %act_win = const.Declare memref<1x1x1x8xui8, @CMX_NN> = dense<1> : tensor<1x1x1x8xui8>

    %0 = memref.alloc() : !IOZMajorDataType
    %1 = memref.alloc() : !IOZMajorSMType
    %2 = VPUIP.GroupSparseBuffer(%0, %1) -> !IOZMajorSparseType

    %3 = memref.alloc() : !IOZMajorDataType
    %4 = memref.alloc() : !IOZMajorSMType
    %5 = VPUIP.GroupSparseBuffer(%3, %4) -> !IOZMajorSparseType

    %in_data, %in_sm = VPUIP.UngroupSparseBuffer(%arg0) {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>}
        -> !IOZMajorDataType, !IOZMajorSMType
    %out_data_0, %out_sm_0 = VPUIP.UngroupSparseBuffer(%2) {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>}
        -> !IOZMajorDataType, !IOZMajorSMType
    %out_data_1, %out_sm_1 = VPUIP.UngroupSparseBuffer(%5) {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>}
        -> !IOZMajorDataType, !IOZMajorSMType

    %6:2 = VPUIP.NCEClusterTask {
            activation_window_channel_length = 27 : i64,
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = "MAXPOOL"
        }
        input(%in_data : !IOZMajorDataType)
        input_sparsity_map(%in_sm : !IOZMajorSMType)
        weight_table(%wt : memref<8x1x1x4xsi32, @CMX_NN>)
        activation_window(%act_win : memref<1x1x1x8xui8, @CMX_NN>)
        parent_input(%in_data : !IOZMajorDataType)
        parent_input_sparsity_map(%in_sm : !IOZMajorSMType)
        parent_output(%out_data_0 : !IOZMajorDataType)
        parent_output_sparsity_map(%out_sm_0 : !IOZMajorSMType)
        outputs(%out_data_0 : !IOZMajorDataType)
        output_sparsity_map(%out_sm_0 : !IOZMajorSMType) -> !IOZMajorDataType, !IOZMajorSMType
        variants :
        {
            DPUTask { outEnd = [16, 1, 1], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
        }
        PPE : {
        }
    %7 = VPUIP.GroupSparseBuffer(%6#0, %6#1) -> !IOZMajorSparseType

    %8:2 = VPUIP.NCEClusterTask {
            activation_window_channel_length = 27 : i64,
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = "MAXPOOL"
        }
        input(%in_data : !IOZMajorDataType)
        input_sparsity_map(%in_sm : !IOZMajorSMType)
        weight_table(%wt : memref<8x1x1x4xsi32, @CMX_NN>)
        activation_window(%act_win : memref<1x1x1x8xui8, @CMX_NN>)
        parent_input(%in_data : !IOZMajorDataType)
        parent_input_sparsity_map(%in_sm : !IOZMajorSMType)
        parent_output(%out_data_1 : !IOZMajorDataType)
        parent_output_sparsity_map(%out_sm_1 : !IOZMajorSMType)
        outputs(%out_data_1 : !IOZMajorDataType)
        output_sparsity_map(%out_sm_1 : !IOZMajorSMType) -> !IOZMajorDataType, !IOZMajorSMType
        variants :
        {
            DPUTask { outEnd = [16, 1, 1], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
        }
        PPE : {
        }
    %9 = VPUIP.GroupSparseBuffer(%8#0, %8#1) -> !IOZMajorSparseType

    %10 = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 8, 1, 1] : !SparseOutputDDRBufferType to !SparseHalfOutputDDRBufferType
    %11 = VPUIP.Copy inputs(%7 : !IOZMajorSparseType) outputs(%10 : !SparseHalfOutputDDRBufferType) -> !SparseHalfOutputDDRBufferType
    %12 = VPUIP.SubView %arg1 [0, 8, 0, 0] [1, 8, 1, 1] : !SparseOutputDDRBufferType to !SparseHalfOutputDDRBufferType
    %13 = VPUIP.Copy inputs(%9 : !IOZMajorSparseType) outputs(%12 : !SparseHalfOutputDDRBufferType) -> !SparseHalfOutputDDRBufferType
    %14 = VPUIP.ConcatView inputs(%11, %13 : !SparseHalfOutputDDRBufferType, !SparseHalfOutputDDRBufferType) outputs(%arg1 : !SparseOutputDDRBufferType) -> !SparseOutputDDRBufferType


    return %14 : !SparseOutputDDRBufferType

    // CHECK:       [[BUFF_0_DATA:%.+]] = memref.alloc() : memref<1x8x1x1xf16, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x8x1x1xi1, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_0:%.*]] = VPUIP.GroupSparseBuffer([[BUFF_0_DATA]], [[BUFF_0_SM]])

    // CHECK:       [[BUFF_1_DATA:%.+]] = memref.alloc() : memref<1x8x1x1xf16, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_1_SM:%.+]] = memref.alloc() : memref<1x8x1x1xi1, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_1:%.*]] = VPUIP.GroupSparseBuffer([[BUFF_1_DATA]], [[BUFF_1_SM]])

    // CHECK:       [[DATA_0:%.*]], [[SM_0:%.*]] = VPUIP.UngroupSparseBuffer(%arg0)
    // CHECK:       [[DATA_1:%.*]], [[SM_1:%.*]] = VPUIP.UngroupSparseBuffer([[BUFF_0]])
    // CHECK:       [[DATA_2:%.*]], [[SM_2:%.*]] = VPUIP.UngroupSparseBuffer([[BUFF_1]])

    // CHECK:       [[NCE0:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[DATA_0]] : memref<1x8x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      input_sparsity_map([[SM_0]] : memref<1x8x1x1xi1, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[DATA_1]] : memref<1x8x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      output_sparsity_map([[SM_1]] : memref<1x8x1x1xi1, #NHWC, @CMX_NN>)

    // CHECK:       [[VAR1:%.*]] = VPUIP.GroupSparseBuffer([[NCE0]]#0, [[NCE0]]#1)
    // CHECK:       [[VAR2:%.*]] = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 8, 1, 1]
    // CHECK:       [[VAR3:%.+]] = VPUIP.Copy inputs([[VAR1]] : !VPUIP.SparseBuffer<data=memref<1x8x1x1xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x8x1x1xi1, #NHWC, @CMX_NN>>)
    // CHECK-SAME:      outputs([[VAR2]] : !VPUIP.SparseBuffer<data=memref<1x8x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [16, 1, 1, 1]}>,
    // CHECK-SAME:                                             sparsity_map=memref<1x8x1x1xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [16, 1, 1, 1]}>>)


    // CHECK:       [[NCE1:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[DATA_0]] : memref<1x8x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      input_sparsity_map([[SM_0]] : memref<1x8x1x1xi1, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[DATA_2]] : memref<1x8x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      output_sparsity_map([[SM_2]] : memref<1x8x1x1xi1, #NHWC, @CMX_NN>)

    // CHECK:       [[VAR4:%.*]] = VPUIP.GroupSparseBuffer([[NCE1]]#0, [[NCE1]]#1)
    // CHECK:       [[VAR5:%.*]] = VPUIP.SubView %arg1 [0, 8, 0, 0] [1, 8, 1, 1]
    // CHECK:       [[VAR6:%.+]] = VPUIP.Copy inputs([[VAR4]] : !VPUIP.SparseBuffer<data=memref<1x8x1x1xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x8x1x1xi1, #NHWC, @CMX_NN>>)
    // CHECK-SAME:      outputs([[VAR5]] : !VPUIP.SparseBuffer<data=memref<1x8x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [16, 1, 1, 1]}>,
    // CHECK-SAME:                                             sparsity_map=memref<1x8x1x1xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [16, 1, 1, 1]}>>)

    // CHECK:       [[VAR7:%.+]] = VPUIP.ConcatView inputs([[VAR3]], [[VAR6]]
    // CHECK-SAME:      outputs(%arg1 : !VPUIP.SparseBuffer<data=memref<1x16x1x1xf16>, sparsity_map=memref<1x16x1x1xi1>>) -> !VPUIP.SparseBuffer<data=memref<1x16x1x1xf16>, sparsity_map=memref<1x16x1x1xi1>>

    // CHECK: return [[VAR7]]
}
