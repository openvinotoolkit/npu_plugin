//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --compute-se-base-ptrs %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x3x3xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!InputSMDistributed = !VPUIP.DistributedBuffer<
    1x16x3x3xi1, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!InputSEDistributed = !VPUIP.DistributedBuffer<
    1x1x3x3xi32, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x3x3xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    16x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsSMDistributed = !VPUIP.DistributedBuffer<
    16x1x1x128xi1, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = memref<1x16x3x3xf16, #NHWC>
!InputSM_DDR = memref<1x16x3x3xi1, #NHWC>
!InputSE_DDR = memref<1x1x3x3xi32, #NHWC>
!Weights_DDR = memref<16x16x1x1xf16, #NHWC>
!WeightsSM_DDR = memref<16x1x1x128xi1>
!WeightsTable_DDR = memref<16x1x1x4xsi32>
!Output_DDR = memref<1x16x6x6xf16, #NHWC>

!Input_CMX = memref<1x16x3x3xf16, #NHWC, @CMX_NN>
!InputSM_CMX = memref<1x16x3x3xi1, #NHWC, @CMX_NN>
!InputSE_CMX = memref<1x1x3x3xi32, #NHWC, @CMX_NN>
!Weights_CMX = memref<16x16x1x1xf16, #NHWC, @CMX_NN>
!WeightsSM_CMX = memref<16x1x1x128xi1, @CMX_NN>
!WeightsTable_CMX = memref<16x1x1x4xsi32, @CMX_NN>
!Output_CMX = memref<1x16x6x6xf16, #NHWC, @CMX_NN>

// CHECK:       func @SETable([[ARG0:%.+]]: memref<1x16x3x3xf16, #NHWC>, [[ARG1:%.+]]: memref<1x16x3x3xi1, #NHWC>,
// CHECK-SAME:                    [[ARG2:%.+]]: memref<1x16x6x6xf16, #NHWC>)
// CHECK-SAME:      -> memref<1x16x6x6xf16, #NHWC>
func.func @SETable(%arg0: !Input_DDR, %arg1: !InputSM_DDR, %arg2: !Output_DDR) -> !Output_DDR {
    %input_se = VPUIP.StorageElementTable {
                dataShape = [1, 16, 3, 3], dataElemType = f16,
                seSize = 16, seDepth = 1
            } -> !InputSE_DDR
    %input_sparse = VPUIP.GroupSparseBuffer (%arg0, %arg1, %input_se)
        -> !VPUIP.SparseBuffer<data=!Input_DDR, sparsity_map=!InputSM_DDR, storage_element_table=!InputSE_DDR>

    %input_data_cmx = VPURT.AllocDistributed -> !InputDistributed
    %input_sm_cmx = VPURT.AllocDistributed -> !InputSMDistributed
    %input_se_cmx = VPURT.AllocDistributed -> !InputSEDistributed
    %input_sparse_cmx = VPUIP.GroupSparseBuffer (%input_data_cmx, %input_sm_cmx, %input_se_cmx)
        -> !VPUIP.SparseBuffer<data=!InputDistributed, sparsity_map=!InputSMDistributed, storage_element_table=!InputSEDistributed>

    %input = VPUIP.NCEClusterTiling inputs(%input_sparse as %arg3: !VPUIP.SparseBuffer<data=!Input_DDR, sparsity_map=!InputSM_DDR, storage_element_table=!InputSE_DDR>)
                               outputs(%input_sparse_cmx as %arg4: !VPUIP.SparseBuffer<data=!Input_CMX, sparsity_map=!InputSM_CMX, storage_element_table=!InputSE_CMX>)
            -> !VPUIP.SparseBuffer<data=!InputDistributed, sparsity_map=!InputSMDistributed, storage_element_table=!InputSEDistributed> {
        %0 = VPUIP.Copy inputs(%arg3: !VPUIP.SparseBuffer<data=!Input_DDR, sparsity_map=!InputSM_DDR, storage_element_table=!InputSE_DDR>)
                       outputs(%arg4: !VPUIP.SparseBuffer<data=!Input_CMX, sparsity_map=!InputSM_CMX, storage_element_table=!InputSE_CMX>)
              -> !VPUIP.SparseBuffer<data=!Input_CMX, sparsity_map=!InputSM_CMX, storage_element_table=!InputSE_CMX>
    }

    %cst_weights = const.Declare !Weights_DDR = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %cst_weights_sm = const.Declare !WeightsSM_DDR = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPUIP.GroupSparseBuffer (%cst_weights, %cst_weights_sm) {is_weights}
        -> !VPUIP.SparseBuffer<data=!Weights_DDR, sparsity_map=!WeightsSM_DDR, is_weights>

    %weights_data_cmx = VPURT.AllocDistributed -> !WeightsDistributed
    %weights_sm_cmx = VPURT.AllocDistributed -> !WeightsSMDistributed
    %weights_sparse_cmx = VPUIP.GroupSparseBuffer (%weights_data_cmx, %weights_sm_cmx) {is_weights}
        -> !VPUIP.SparseBuffer<data=!WeightsDistributed, sparsity_map=!WeightsSMDistributed, is_weights>

    %weights = VPUIP.NCEClusterTiling inputs(%input_sparse as %arg3: !VPUIP.SparseBuffer<data=!Weights_DDR, sparsity_map=memref<!WeightsSM_DDR>, is_weights>)
                                 outputs(%input_sparse_cmx as %arg4: !VPUIP.SparseBuffer<data=!Weights_CMX, sparsity_map=memref<!WeightsSM_CMX>, is_weights>)
            -> !VPUIP.SparseBuffer<data=!WeightsDistributed, sparsity_map=!WeightsSMDistributed, is_weights> {
        %0 = VPUIP.Copy inputs(%arg3: !VPUIP.SparseBuffer<data=!Weights_DDR, sparsity_map=memref<!WeightsSM_DDR>, is_weights>)
                       outputs(%arg4: !VPUIP.SparseBuffer<data=!Weights_CMX, sparsity_map=memref<!WeightsSM_CMX>, is_weights>)
              -> !VPUIP.SparseBuffer<data=!Weights_CMX, sparsity_map=memref<!WeightsSM_CMX>, is_weights>
    }

    %cst_weights_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<16x1x1x4xsi32>
    %weights_table_cmx = VPURT.AllocDistributed -> !WeightsTableDistributed

    %weights_table = VPUIP.NCEClusterTiling inputs(%cst_weights_table as %arg3: !WeightsTable_DDR)
                                           outputs(%weights_table_cmx as %arg4: !WeightsTable_CMX)
            -> !WeightsTableDistributed {
        %0 = VPUIP.Copy inputs(%arg3: !WeightsTable_DDR) outputs(%arg4: !WeightsTable_CMX) -> !WeightsTable_CMX
    }

    %in_data, %in_sm, %in_se = VPUIP.UngroupSparseBuffer(%input) {result_segment_sizes = dense<[1, 1, 1]> : vector<3xi32>}
        -> !InputDistributed, !InputSMDistributed, !InputSEDistributed
    %w_data, %w_sm = VPUIP.UngroupSparseBuffer(%weights)  {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>}
        -> !WeightsDistributed, !WeightsSMDistributed
    %output = VPURT.AllocDistributed -> !OutputDistributed

    %conv_out = VPUIP.NCEClusterTiling inputs(%in_data as %arg3: !Input_CMX,
                                              %in_sm as %arg4: !InputSM_CMX,
                                              %in_se as %arg5: !InputSE_CMX,
                                              %w_data as %arg6: !Weights_CMX,
                                              %w_sm as %arg7: !WeightsSM_CMX,
                                              %weights_table as %arg8: !WeightsTable_CMX)
                                      outputs(%output as %arg9: !Output_CMX)
            -> !OutputDistributed {
        %0 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 27 : i64,
                kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = "CONV"
            }
            input(%arg3 : !Input_CMX)
            input_sparsity_map(%arg4 : !InputSM_CMX)
            input_storage_element_table(%arg5 : !InputSE_CMX)
            weights(%arg6 : !Weights_CMX)
            weights_sparsity_map(%arg7 : !WeightsSM_CMX)
            weight_table(%arg8 : !WeightsTable_CMX)
            parent_input(%arg3 : !Input_CMX)
            parent_input_sparsity_map(%arg4 : !InputSM_CMX)
            parent_input_storage_element_table(%arg5 : !InputSE_CMX)
            parent_output(%arg9 : !Output_CMX)
            outputs(%arg9 : !Output_CMX)
                -> !Output_CMX
            variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [16, 6, 6], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0]}
            } PPE :  {
            }
    }

    %output_ddr = VPUIP.NCEClusterTiling inputs(%conv_out as %arg3: !Output_CMX)
                                        outputs(%arg2 as %arg4: !Output_DDR)
            -> !Output_DDR {
        %0 = VPUIP.Copy inputs(%arg3: !Output_CMX) outputs(%arg4: !Output_DDR) -> !Output_DDR
    }

    return %output_ddr : !Output_DDR

    // CHECK:       VPUIP.StorageElementTable {
    // CHECK-SAME:    basePtrs = dense<[0, 0, 0,
    // CHECK-SAME:                      0, 0, 0,
    // CHECK-SAME:                      1, 1, 1]> : tensor<9xi32>,
    // CHECK-SAME:    dataElemType = f16, dataShape = [1, 16, 3, 3],
    // CHECK-SAME:    seDepth = 1 : i64, seSize = 16 : i64
    // CHECK-SAME:  } -> memref<1x1x3x3xi32, #NHWC>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x3x3xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!InputSMDistributed = !VPUIP.DistributedBuffer<
    1x16x3x3xi1, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!InputSEDistributed = !VPUIP.DistributedBuffer<
    1x1x6x6xi32, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x3x3xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    16x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsSMDistributed = !VPUIP.DistributedBuffer<
    16x1x1x128xi1, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = memref<1x16x3x3xf16, #NHWC>
!InputSM_DDR = memref<1x16x3x3xi1, #NHWC>
!InputSE_DDR = memref<1x1x6x6xi32, #NHWC>
!Weights_DDR = memref<16x16x1x1xf16, #NHWC>
!WeightsSM_DDR = memref<16x1x1x128xi1>
!WeightsTable_DDR = memref<16x1x1x4xsi32>
!Output_DDR = memref<1x16x6x6xf16, #NHWC>

!Input_CMX = memref<1x16x3x3xf16, #NHWC, @CMX_NN>
!InputSM_CMX = memref<1x16x3x3xi1, #NHWC, @CMX_NN>
!InputSE_CMX = memref<1x1x6x6xi32, #NHWC, @CMX_NN>
!Weights_CMX = memref<16x16x1x1xf16, #NHWC, @CMX_NN>
!WeightsSM_CMX = memref<16x1x1x128xi1, @CMX_NN>
!WeightsTable_CMX = memref<16x1x1x4xsi32, @CMX_NN>
!Output_CMX = memref<1x16x6x6xf16, #NHWC, @CMX_NN>

// CHECK:       func @Interpolate([[ARG0:%.+]]: memref<1x16x3x3xf16, #NHWC>, [[ARG1:%.+]]: memref<1x16x3x3xi1, #NHWC>,
// CHECK-SAME:                    [[ARG2:%.+]]: memref<1x16x6x6xf16, #NHWC>)
// CHECK-SAME:      -> memref<1x16x6x6xf16, #NHWC>
func.func @Interpolate(%arg0: !Input_DDR, %arg1: !InputSM_DDR, %arg2: !Output_DDR) -> !Output_DDR {
    %input_se = VPUIP.StorageElementTable {
                dataShape = [1, 16, 3, 3], dataElemType = f16,
                seAttr = #VPU.SEInterpolate<mode = "NEAREST", nearest_mode = "FLOOR", coordinate_transformation_mode = "ASYMMETRIC", scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>,
                seSize = 16, seDepth = 1
            } -> !InputSE_DDR
    %input_sparse = VPUIP.GroupSparseBuffer (%arg0, %arg1, %input_se) {
            seAttr = #VPU.SEInterpolate<mode = "NEAREST", nearest_mode = "FLOOR", coordinate_transformation_mode = "ASYMMETRIC", scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>
        } -> !VPUIP.SparseBuffer<data=!Input_DDR, sparsity_map=!InputSM_DDR, storage_element_table=!InputSE_DDR,
                                 #VPU.SEInterpolate<mode = "NEAREST", nearest_mode = "FLOOR", coordinate_transformation_mode = "ASYMMETRIC", scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>>

    %input_data_cmx = VPURT.AllocDistributed -> !InputDistributed
    %input_sm_cmx = VPURT.AllocDistributed -> !InputSMDistributed
    %input_se_cmx = VPURT.AllocDistributed -> !InputSEDistributed
    %input_sparse_cmx = VPUIP.GroupSparseBuffer (%input_data_cmx, %input_sm_cmx, %input_se_cmx) {
            seAttr = #VPU.SEInterpolate<mode = "NEAREST", nearest_mode = "FLOOR", coordinate_transformation_mode = "ASYMMETRIC", scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>
        } -> !VPUIP.SparseBuffer<data=!InputDistributed, sparsity_map=!InputSMDistributed, storage_element_table=!InputSEDistributed,
                                 #VPU.SEInterpolate<mode = "NEAREST", nearest_mode = "FLOOR", coordinate_transformation_mode = "ASYMMETRIC", scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>>

    %input = VPUIP.NCEClusterTiling inputs(%input_sparse as %arg3: !VPUIP.SparseBuffer<data=!Input_DDR, sparsity_map=!InputSM_DDR, storage_element_table=!InputSE_DDR>)
                               outputs(%input_sparse_cmx as %arg4: !VPUIP.SparseBuffer<data=!Input_CMX, sparsity_map=!InputSM_CMX, storage_element_table=!InputSE_CMX>)
            -> !VPUIP.SparseBuffer<data=!InputDistributed, sparsity_map=!InputSMDistributed, storage_element_table=!InputSEDistributed,
                                   #VPU.SEInterpolate<mode = "NEAREST", nearest_mode = "FLOOR", coordinate_transformation_mode = "ASYMMETRIC", scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>> {
        %0 = VPUIP.Copy inputs(%arg3: !VPUIP.SparseBuffer<data=!Input_DDR, sparsity_map=!InputSM_DDR, storage_element_table=!InputSE_DDR>)
                       outputs(%arg4: !VPUIP.SparseBuffer<data=!Input_CMX, sparsity_map=!InputSM_CMX, storage_element_table=!InputSE_CMX>)
              -> !VPUIP.SparseBuffer<data=!Input_CMX, sparsity_map=!InputSM_CMX, storage_element_table=!InputSE_CMX>
    }

    %cst_weights = const.Declare !Weights_DDR = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %cst_weights_sm = const.Declare !WeightsSM_DDR = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPUIP.GroupSparseBuffer (%cst_weights, %cst_weights_sm) {is_weights}
        -> !VPUIP.SparseBuffer<data=!Weights_DDR, sparsity_map=!WeightsSM_DDR, is_weights>

    %weights_data_cmx = VPURT.AllocDistributed -> !WeightsDistributed
    %weights_sm_cmx = VPURT.AllocDistributed -> !WeightsSMDistributed
    %weights_sparse_cmx = VPUIP.GroupSparseBuffer (%weights_data_cmx, %weights_sm_cmx) {is_weights}
        -> !VPUIP.SparseBuffer<data=!WeightsDistributed, sparsity_map=!WeightsSMDistributed, is_weights>

    %weights = VPUIP.NCEClusterTiling inputs(%input_sparse as %arg3: !VPUIP.SparseBuffer<data=!Weights_DDR, sparsity_map=memref<!WeightsSM_DDR>, is_weights>)
                                 outputs(%input_sparse_cmx as %arg4: !VPUIP.SparseBuffer<data=!Weights_CMX, sparsity_map=memref<!WeightsSM_CMX>, is_weights>)
            -> !VPUIP.SparseBuffer<data=!WeightsDistributed, sparsity_map=!WeightsSMDistributed, is_weights> {
        %0 = VPUIP.Copy inputs(%arg3: !VPUIP.SparseBuffer<data=!Weights_DDR, sparsity_map=memref<!WeightsSM_DDR>, is_weights>)
                       outputs(%arg4: !VPUIP.SparseBuffer<data=!Weights_CMX, sparsity_map=memref<!WeightsSM_CMX>, is_weights>)
              -> !VPUIP.SparseBuffer<data=!Weights_CMX, sparsity_map=memref<!WeightsSM_CMX>, is_weights>
    }

    %cst_weights_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<16x1x1x4xsi32>
    %weights_table_cmx = VPURT.AllocDistributed -> !WeightsTableDistributed

    %weights_table = VPUIP.NCEClusterTiling inputs(%cst_weights_table as %arg3: !WeightsTable_DDR)
                                           outputs(%weights_table_cmx as %arg4: !WeightsTable_CMX)
            -> !WeightsTableDistributed {
        %0 = VPUIP.Copy inputs(%arg3: !WeightsTable_DDR) outputs(%arg4: !WeightsTable_CMX) -> !WeightsTable_CMX
    }

    %in_data, %in_sm, %in_se = VPUIP.UngroupSparseBuffer(%input) {result_segment_sizes = dense<[1, 1, 1]> : vector<3xi32>}
        -> !InputDistributed, !InputSMDistributed, !InputSEDistributed
    %w_data, %w_sm = VPUIP.UngroupSparseBuffer(%weights)  {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>}
        -> !WeightsDistributed, !WeightsSMDistributed
    %output = VPURT.AllocDistributed -> !OutputDistributed

    %conv_out = VPUIP.NCEClusterTiling inputs(%in_data as %arg3: !Input_CMX,
                                              %in_sm as %arg4: !InputSM_CMX,
                                              %in_se as %arg5: !InputSE_CMX,
                                              %w_data as %arg6: !Weights_CMX,
                                              %w_sm as %arg7: !WeightsSM_CMX,
                                              %weights_table as %arg8: !WeightsTable_CMX)
                                      outputs(%output as %arg9: !Output_CMX)
            -> !OutputDistributed {
        %0 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 27 : i64,
                kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = "CONV"
            }
            input(%arg3 : !Input_CMX)
            input_sparsity_map(%arg4 : !InputSM_CMX)
            input_storage_element_table(%arg5 : !InputSE_CMX)
            weights(%arg6 : !Weights_CMX)
            weights_sparsity_map(%arg7 : !WeightsSM_CMX)
            weight_table(%arg8 : !WeightsTable_CMX)
            parent_input(%arg3 : !Input_CMX)
            parent_input_sparsity_map(%arg4 : !InputSM_CMX)
            parent_input_storage_element_table(%arg5 : !InputSE_CMX)
            parent_output(%arg9 : !Output_CMX)
            outputs(%arg9 : !Output_CMX)
                -> !Output_CMX
            variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [16, 6, 6], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0]}
            } PPE :  {
            }
    }

    %output_ddr = VPUIP.NCEClusterTiling inputs(%conv_out as %arg3: !Output_CMX)
                                        outputs(%arg2 as %arg4: !Output_DDR)
            -> !Output_DDR {
        %0 = VPUIP.Copy inputs(%arg3: !Output_CMX) outputs(%arg4: !Output_DDR) -> !Output_DDR
    }

    return %output_ddr : !Output_DDR

    // CHECK:       VPUIP.StorageElementTable {
    // CHECK-SAME:    basePtrs = dense<[0, 0, 0, 0, 0, 0,
    // CHECK-SAME:                      0, 0, 0, 0, 0, 0,
    // CHECK-SAME:                      0, 0, 0, 0, 0, 0,
    // CHECK-SAME:                      0, 0, 0, 0, 0, 0,
    // CHECK-SAME:                      1, 1, 1, 1, 1, 1,
    // CHECK-SAME:                      1, 1, 1, 1, 1, 1]> : tensor<36xi32>,
    // CHECK-SAME:    dataElemType = f16, dataShape = [1, 16, 3, 3],
    // CHECK-SAME:    seAttr = #VPU.SEInterpolate<mode = "NEAREST", nearest_mode = "FLOOR", coordinate_transformation_mode = "ASYMMETRIC",
    // CHECK-SAME:                                scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                                offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>,
    // CHECK-SAME:    seDepth = 1 : i64, seSize = 16 : i64
    // CHECK-SAME:  } -> memref<1x1x6x6xi32, #NHWC>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x32x3x3xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!InputSMDistributed = !VPUIP.DistributedBuffer<
    1x32x3x3xi1, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!InputSEDistributed = !VPUIP.DistributedBuffer<
    1x2x6x6xi32, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x32x3x3xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    32x32x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsSMDistributed = !VPUIP.DistributedBuffer<
    32x1x1x128xi1, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    32x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = memref<1x32x3x3xf16, #NHWC>
!InputSM_DDR = memref<1x32x3x3xi1, #NHWC>
!InputSE_DDR = memref<1x2x6x6xi32, #NHWC>
!Weights_DDR = memref<32x32x1x1xf16, #NHWC>
!WeightsSM_DDR = memref<32x1x1x128xi1>
!WeightsTable_DDR = memref<32x1x1x4xsi32>
!Output_DDR = memref<1x32x6x6xf16, #NHWC>

!Input_CMX = memref<1x32x3x3xf16, #NHWC, @CMX_NN>
!InputSM_CMX = memref<1x32x3x3xi1, #NHWC, @CMX_NN>
!InputSE_CMX = memref<1x2x6x6xi32, #NHWC, @CMX_NN>
!Weights_CMX = memref<32x32x1x1xf16, #NHWC, @CMX_NN>
!WeightsSM_CMX = memref<32x1x1x128xi1, @CMX_NN>
!WeightsTable_CMX = memref<32x1x1x4xsi32, @CMX_NN>
!Output_CMX = memref<1x32x6x6xf16, #NHWC, @CMX_NN>

// CHECK:       func @InterpolateSESize([[ARG0:%.+]]: memref<1x32x3x3xf16, #NHWC>, [[ARG1:%.+]]: memref<1x32x3x3xi1, #NHWC>,
// CHECK-SAME:                    [[ARG2:%.+]]: memref<1x32x6x6xf16, #NHWC>)
// CHECK-SAME:      -> memref<1x32x6x6xf16, #NHWC>
func.func @InterpolateSESize(%arg0: !Input_DDR, %arg1: !InputSM_DDR, %arg2: !Output_DDR) -> !Output_DDR {
    %input_se = VPUIP.StorageElementTable {
                dataShape = [1, 32, 3, 3], dataElemType = f16,
                seAttr = #VPU.SEInterpolate<mode = "NEAREST", nearest_mode = "FLOOR", coordinate_transformation_mode = "ASYMMETRIC", scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>,
                seSize = 16, seDepth = 2
            } -> !InputSE_DDR
    %input_sparse = VPUIP.GroupSparseBuffer (%arg0, %arg1, %input_se) {
            seAttr = #VPU.SEInterpolate<mode = "NEAREST", nearest_mode = "FLOOR", coordinate_transformation_mode = "ASYMMETRIC", scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>
        } -> !VPUIP.SparseBuffer<data=!Input_DDR, sparsity_map=!InputSM_DDR, storage_element_table=!InputSE_DDR,
                                 #VPU.SEInterpolate<mode = "NEAREST", nearest_mode = "FLOOR", coordinate_transformation_mode = "ASYMMETRIC", scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>>

    %input_data_cmx = VPURT.AllocDistributed -> !InputDistributed
    %input_sm_cmx = VPURT.AllocDistributed -> !InputSMDistributed
    %input_se_cmx = VPURT.AllocDistributed -> !InputSEDistributed
    %input_sparse_cmx = VPUIP.GroupSparseBuffer (%input_data_cmx, %input_sm_cmx, %input_se_cmx) {
            seAttr = #VPU.SEInterpolate<mode = "NEAREST", nearest_mode = "FLOOR", coordinate_transformation_mode = "ASYMMETRIC", scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>
        } -> !VPUIP.SparseBuffer<data=!InputDistributed, sparsity_map=!InputSMDistributed, storage_element_table=!InputSEDistributed,
                                 #VPU.SEInterpolate<mode = "NEAREST", nearest_mode = "FLOOR", coordinate_transformation_mode = "ASYMMETRIC", scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>>

    %input = VPUIP.NCEClusterTiling inputs(%input_sparse as %arg3: !VPUIP.SparseBuffer<data=!Input_DDR, sparsity_map=!InputSM_DDR, storage_element_table=!InputSE_DDR,
                                                                                       #VPU.SEInterpolate<mode = "NEAREST", nearest_mode = "FLOOR", coordinate_transformation_mode = "ASYMMETRIC", scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>>)
                               outputs(%input_sparse_cmx as %arg4: !VPUIP.SparseBuffer<data=!Input_CMX, sparsity_map=!InputSM_CMX, storage_element_table=!InputSE_CMX,
                                                                                       #VPU.SEInterpolate<mode = "NEAREST", nearest_mode = "FLOOR", coordinate_transformation_mode = "ASYMMETRIC", scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>>)
            -> !VPUIP.SparseBuffer<data=!InputDistributed, sparsity_map=!InputSMDistributed, storage_element_table=!InputSEDistributed,
                                   #VPU.SEInterpolate<mode = "NEAREST", nearest_mode = "FLOOR", coordinate_transformation_mode = "ASYMMETRIC", scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>> {
        %0 = VPUIP.Copy inputs(%arg3: !VPUIP.SparseBuffer<data=!Input_DDR, sparsity_map=!InputSM_DDR, storage_element_table=!InputSE_DDR,
                                                          #VPU.SEInterpolate<mode = "NEAREST", nearest_mode = "FLOOR", coordinate_transformation_mode = "ASYMMETRIC", scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>>)
                       outputs(%arg4: !VPUIP.SparseBuffer<data=!Input_CMX, sparsity_map=!InputSM_CMX, storage_element_table=!InputSE_CMX,
                                                          #VPU.SEInterpolate<mode = "NEAREST", nearest_mode = "FLOOR", coordinate_transformation_mode = "ASYMMETRIC", scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>>)
              -> !VPUIP.SparseBuffer<data=!Input_CMX, sparsity_map=!InputSM_CMX, storage_element_table=!InputSE_CMX,
                                     #VPU.SEInterpolate<mode = "NEAREST", nearest_mode = "FLOOR", coordinate_transformation_mode = "ASYMMETRIC", scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>>
    }

    %cst_weights = const.Declare !Weights_DDR = dense<1.0> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %cst_weights_sm = const.Declare !WeightsSM_DDR = dense<1.0> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPUIP.GroupSparseBuffer (%cst_weights, %cst_weights_sm) {is_weights}
        -> !VPUIP.SparseBuffer<data=!Weights_DDR, sparsity_map=!WeightsSM_DDR, is_weights>

    %weights_data_cmx = VPURT.AllocDistributed -> !WeightsDistributed
    %weights_sm_cmx = VPURT.AllocDistributed -> !WeightsSMDistributed
    %weights_sparse_cmx = VPUIP.GroupSparseBuffer (%weights_data_cmx, %weights_sm_cmx) {is_weights}
        -> !VPUIP.SparseBuffer<data=!WeightsDistributed, sparsity_map=!WeightsSMDistributed, is_weights>

    %weights = VPUIP.NCEClusterTiling inputs(%input_sparse as %arg3: !VPUIP.SparseBuffer<data=!Weights_DDR, sparsity_map=memref<!WeightsSM_DDR>, is_weights>)
                                 outputs(%input_sparse_cmx as %arg4: !VPUIP.SparseBuffer<data=!Weights_CMX, sparsity_map=memref<!WeightsSM_CMX>, is_weights>)
            -> !VPUIP.SparseBuffer<data=!WeightsDistributed, sparsity_map=!WeightsSMDistributed, is_weights> {
        %0 = VPUIP.Copy inputs(%arg3: !VPUIP.SparseBuffer<data=!Weights_DDR, sparsity_map=memref<!WeightsSM_DDR>, is_weights>)
                       outputs(%arg4: !VPUIP.SparseBuffer<data=!Weights_CMX, sparsity_map=memref<!WeightsSM_CMX>, is_weights>)
              -> !VPUIP.SparseBuffer<data=!Weights_CMX, sparsity_map=memref<!WeightsSM_CMX>, is_weights>
    }

    %cst_weights_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<32x1x1x4xsi32>
    %weights_table_cmx = VPURT.AllocDistributed -> !WeightsTableDistributed

    %weights_table = VPUIP.NCEClusterTiling inputs(%cst_weights_table as %arg3: !WeightsTable_DDR)
                                           outputs(%weights_table_cmx as %arg4: !WeightsTable_CMX)
            -> !WeightsTableDistributed {
        %0 = VPUIP.Copy inputs(%arg3: !WeightsTable_DDR) outputs(%arg4: !WeightsTable_CMX) -> !WeightsTable_CMX
    }

    %in_data, %in_sm, %in_se = VPUIP.UngroupSparseBuffer(%input) {result_segment_sizes = dense<[1, 1, 1]> : vector<3xi32>}
        -> !InputDistributed, !InputSMDistributed, !InputSEDistributed
    %w_data, %w_sm = VPUIP.UngroupSparseBuffer(%weights)  {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>}
        -> !WeightsDistributed, !WeightsSMDistributed
    %output = VPURT.AllocDistributed -> !OutputDistributed

    %conv_out = VPUIP.NCEClusterTiling inputs(%in_data as %arg3: !Input_CMX,
                                              %in_sm as %arg4: !InputSM_CMX,
                                              %in_se as %arg5: !InputSE_CMX,
                                              %w_data as %arg6: !Weights_CMX,
                                              %w_sm as %arg7: !WeightsSM_CMX,
                                              %weights_table as %arg8: !WeightsTable_CMX)
                                      outputs(%output as %arg9: !Output_CMX)
            -> !OutputDistributed {
        %0 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 27 : i64,
                kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = "CONV"
            }
            input(%arg3 : !Input_CMX)
            input_sparsity_map(%arg4 : !InputSM_CMX)
            input_storage_element_table(%arg5 : !InputSE_CMX)
            weights(%arg6 : !Weights_CMX)
            weights_sparsity_map(%arg7 : !WeightsSM_CMX)
            weight_table(%arg8 : !WeightsTable_CMX)
            parent_input(%arg3 : !Input_CMX)
            parent_input_sparsity_map(%arg4 : !InputSM_CMX)
            parent_input_storage_element_table(%arg5 : !InputSE_CMX)
            parent_output(%arg9 : !Output_CMX)
            outputs(%arg9 : !Output_CMX)
                -> !Output_CMX
            variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [16, 6, 6], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0]}
            } PPE :  {
            }
    }

    %output_ddr = VPUIP.NCEClusterTiling inputs(%conv_out as %arg3: !Output_CMX)
                                        outputs(%arg2 as %arg4: !Output_DDR)
            -> !Output_DDR {
        %0 = VPUIP.Copy inputs(%arg3: !Output_CMX) outputs(%arg4: !Output_DDR) -> !Output_DDR
    }

    return %output_ddr : !Output_DDR

    // CHECK:       VPUIP.StorageElementTable {
    // CHECK-SAME:    basePtrs = dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // CHECK-SAME:                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // CHECK-SAME:                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // CHECK-SAME:                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // CHECK-SAME:                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    // CHECK-SAME:                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]> : tensor<72xi32>,
    // CHECK-SAME:    dataElemType = f16, dataShape = [1, 32, 3, 3],
    // CHECK-SAME:    seAttr = #VPU.SEInterpolate<mode = "NEAREST", nearest_mode = "FLOOR", coordinate_transformation_mode = "ASYMMETRIC",
    // CHECK-SAME:                                scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                                offsets = [0, 0, 0, 0], sizes = [1, 32, 6, 6]>,
    // CHECK-SAME:    seDepth = 2 : i64, seSize = 16 : i64
    // CHECK-SAME:  } -> memref<1x2x6x6xi32, #NHWC>
}
