//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --unroll-cluster-tiling  %s | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ParentInputDistributed = type !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!ParentOutputDistributed = type !VPUIP.DistributedBuffer<
    1x32x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!WeightsDistributed = type !VPUIP.DistributedBuffer<
    32x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2
}>

!OutputDistributed = type !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = type memref<1x16x32x32xf16, #NHWC, @DDR>
!Output_DDR = type memref<1x32x32x32xf16, #NHWC, @DDR>
!Weights_DDR = type memref<32x16x1x1xf16, #NHWC, @DDR>

!InputStub_CMX = type memref<1x16x32x32xf16, #NHWC, @CMX_NN>
!InputStub_CMX0 = type memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>
!InputStub_CMX1 = type memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 1]>
!OutputStub_CMX = type memref<1x32x32x32xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = type memref<32x16x1x1xf16, #NHWC, @CMX_NN>

//CHECK-LABEL: @UnrollNNDMA
func @UnrollNNDMA(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare memref<32x16x1x1xf16, #NHWC> =
        dense<1.0> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer "NetworkInput" <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer "NetworkOutput" <0> -> !Output_DDR

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer "CMX_NN" <0> -> !ParentInputDistributed
    %input1 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> !InputStub_CMX0
    %input2 = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> !InputStub_CMX1

    %weights = VPURT.DeclareBuffer "CMX_NN" <32768> -> !WeightsDistributed

    %parent_out_cmx = VPURT.DeclareBuffer "CMX_NN" <33280> -> !ParentOutputDistributed
    %out_cmx1 = VPURT.DeclareBuffer "CMX_NN" [0, 1] <33280> -> !OutputDistributed
    %out_cmx2 = VPURT.DeclareBuffer "CMX_NN" [0, 1] <66048> -> !OutputDistributed

    // Upload input
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%parent_in as %arg0: !Input_DDR)
                outputs(%parent_input_cmx as %arg1: !InputStub_CMX) -> !ParentInputDistributed {
             VPUIP.NNDMA inputs(%arg0: !Input_DDR) outputs(%arg1: !InputStub_CMX) -> !InputStub_CMX
         }
    }

    // Upload weights
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%weights_cst as %arg0: !Weights_DDR)
                outputs(%weights as %arg1: !WeightsStub_CMX) -> !WeightsDistributed {
             VPUIP.NNDMA inputs(%arg0: !Weights_DDR) outputs(%arg1: !WeightsStub_CMX) -> !WeightsStub_CMX
         }
    }

    // Simulate 1st task
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
        VPUIP.NNDMA {port = 0 : i64} inputs(%input1: !InputStub_CMX0) outputs(%out_cmx1 : !OutputDistributed) -> !OutputDistributed
    }

    // Simulate 2st task
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
        VPUIP.NNDMA {port = 0 : i64} inputs(%input2: !InputStub_CMX1) outputs(%out_cmx2 : !OutputDistributed) -> !OutputDistributed
    }

    // Copyback output
    VPURT.Task waits(%bar1: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%parent_out_cmx as %arg0: !OutputStub_CMX)
                outputs(%parent_out as %arg1: !Output_DDR) -> !Output_DDR {
             VPUIP.NNDMA inputs(%arg0: !OutputStub_CMX) outputs(%arg1: !Output_DDR) -> !Output_DDR
         }
    }

    return %output: !Output_DDR

    //CHECK:        [[WEIGHTS1_CST:%.*]] = const.Declare memref<16x16x1x1xf16, #NHWC> =
    //CHECK-SAME:       dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [16, 16, 1, 1]>]
    //CHECK:        [[WEIGHTS2_CST:%.*]] = const.Declare memref<16x16x1x1xf16, #NHWC> =
    //CHECK-SAME:       dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.SubView<[16, 0, 0, 0], [16, 16, 1, 1]>]

    //CHECK:        [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:        [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:        [[IN_DDR:%.*]] = VPURT.DeclareBuffer "NetworkInput" <0> -> memref<1x16x32x32xf16, #NHWC, @DDR>
    //CHECK:        [[OUT_DDR:%.*]] = VPURT.DeclareBuffer "NetworkOutput" <0> -> memref<1x32x32x32xf16, #NHWC, @DDR>
    //CHECK:        [[IN_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <0> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[IN1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[IN2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 1]>

    //CHECK:        [[WEIGHTS1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[WEIGHTS2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:        [[OUT_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <33280> -> memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 0]>

    // Upload input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[IN_DDR]] : memref<1x16x32x32xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }

    // Upload 1st part of weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS1_CST]] : memref<16x16x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS1_CMX]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 1 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS2_CST]] : memref<16x16x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS2_CMX]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK:        }

    // Copyback output
    //CHECK:        VPURT.Task waits(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[OUT_CMX:%.*]] : memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT_DDR]] : memref<1x32x32x32xf16, #NHWC, @DDR>)
    //CHECK:        }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ParentInputDistributed = type !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!ParentOutputDistributed = type !VPUIP.DistributedBuffer<
    1x32x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!WeightsDistributed = type !VPUIP.DistributedBuffer<
    32x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2
}>

!WeightsTableDistributed = type !VPUIP.DistributedBuffer<
    32x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2
}>

!Input_DDR = type memref<1x16x32x32xf16, #NHWC, @DDR>
!Output_DDR = type memref<1x32x32x32xf16, #NHWC, @DDR>
!Weights_DDR = type memref<32x16x1x1xf16, #NHWC, @DDR>
!WeightsTable_DDR = type memref<32x1x1x4xsi32, #NCHW, @DDR>

!InputStub_CMX = type memref<1x16x32x32xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = type memref<1x32x32x32xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = type memref<32x16x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = type memref<32x1x1x4xsi32, #NCHW, @CMX_NN>


//CHECK-LABEL: @UnrollNCE
func @UnrollNCE(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare memref<32x16x1x1xf16, #NHWC> =
        dense<1.0> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table_cst = const.Declare memref<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer "NetworkInput" <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer "NetworkOutput" <0> -> !Output_DDR

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer "CMX_NN" <0> -> !ParentInputDistributed
    %weights = VPURT.DeclareBuffer "CMX_NN" <32768> -> !WeightsDistributed
    %weights_table = VPURT.DeclareBuffer "CMX_NN" <33280> -> !WeightsTableDistributed
    %parent_out_cmx = VPURT.DeclareBuffer "CMX_NN" <33536> -> !ParentOutputDistributed

    // Upload input
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%parent_in as %arg0: !Input_DDR)
                outputs(%parent_input_cmx as %arg1: !InputStub_CMX) -> !ParentInputDistributed {
             VPUIP.NNDMA inputs(%arg0: !Input_DDR) outputs(%arg1: !InputStub_CMX) -> !InputStub_CMX
         }
    }

    // Upload weights
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%weights_cst as %arg0: !Weights_DDR)
                outputs(%weights as %arg1: !WeightsStub_CMX) -> !WeightsDistributed {
             VPUIP.NNDMA inputs(%arg0: !Weights_DDR) outputs(%arg1: !WeightsStub_CMX) -> !WeightsStub_CMX
         }
    }

    // Upload weights table
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%weights_table_cst as %arg0: !WeightsTable_DDR)
                outputs(%weights_table as %arg1: !WeightsTableStub_CMX) -> !WeightsTableDistributed {
             VPUIP.NNDMA inputs(%arg0: !WeightsTable_DDR) outputs(%arg1: !WeightsTableStub_CMX) -> !WeightsTableStub_CMX
         }
    }

    // Cluster tiling
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling
                 inputs(%parent_input_cmx as %arg0: !InputStub_CMX,
                         %weights as %arg1: !WeightsStub_CMX,
                         %weights_table as %arg2: !WeightsTableStub_CMX)
                 outputs(%parent_out_cmx as %arg3: !OutputStub_CMX)
                     -> !OutputStub_CMX {

               %1 = VPUIP.NCEClusterTask {
                         kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                         kernel_size = [1, 1],
                         kernel_strides = [1, 1],
                         task_type = "CONV"
                     }  input(%arg0 : !InputStub_CMX)
                         weights(%arg1 : !WeightsStub_CMX)
                         weight_table(%arg2 : !WeightsTableStub_CMX)
                         parent_input(%arg0 : !InputStub_CMX)
                         parent_output(%arg3 : !OutputStub_CMX)
                         outputs(%arg3 : !OutputStub_CMX)
                             -> !OutputStub_CMX variants :  {
                            DPUTask {
                                outStart = [0, 0, 0], outEnd = [31, 31, 15],
                                mpe_mode = "VECTOR_FP16",
                                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                                cluster_id = 0 : i64
                            }
                            DPUTask {
                                outStart = [0, 0, 16], outEnd = [31, 31, 31],
                                mpe_mode = "VECTOR_FP16",
                                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                                cluster_id = 1 : i64
                            }
                         } PPE :  {
                         }
        }
    }

    // Copyback output
    VPURT.Task waits(%bar1: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%parent_out_cmx as %arg0: !OutputStub_CMX)
                outputs(%parent_out as %arg1: !Output_DDR) -> !Output_DDR {
             VPUIP.NNDMA inputs(%arg0: !OutputStub_CMX) outputs(%arg1: !Output_DDR) -> !Output_DDR
         }
    }

    return %output: !Output_DDR

    //CHECK:        [[WEIGHTS_TABLE1_CST:%.*]] = const.Declare memref<16x1x1x4xsi32> =
    //CHECK-SAME:       dense<1> : tensor<32x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [16, 1, 1, 4]>]
    //CHECK:        [[WEIGHTS_TABLE2_CST:%.*]] = const.Declare memref<16x1x1x4xsi32> =
    //CHECK-SAME:       dense<1> : tensor<32x1x1x4xsi32>, [#const.SubView<[16, 0, 0, 0], [16, 1, 1, 4]>]

    //CHECK:        [[WEIGHTS1_CST:%.*]] = const.Declare memref<16x16x1x1xf16, #NHWC> =
    //CHECK-SAME:       dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [16, 16, 1, 1]>]
    //CHECK:        [[WEIGHTS2_CST:%.*]] = const.Declare memref<16x16x1x1xf16, #NHWC> =
    //CHECK-SAME:       dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.SubView<[16, 0, 0, 0], [16, 16, 1, 1]>]

    //CHECK:        [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:        [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:        [[IN_DDR:%.*]] = VPURT.DeclareBuffer "NetworkInput" <0> -> memref<1x16x32x32xf16, #NHWC, @DDR>
    //CHECK:        [[OUT_DDR:%.*]] = VPURT.DeclareBuffer "NetworkOutput" <0> -> memref<1x32x32x32xf16, #NHWC, @DDR>

    //CHECK:        [[PARENT_IN_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:        [[IN1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[IN2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:        [[IN_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <0> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:        [[WEIGHTS1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[WEIGHTS2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:        [[WEIGHTS1_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[WEIGHTS2_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:        [[WEIGHTS_TABLE1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:        [[WEIGHTS_TABLE2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:        [[WEIGHTS_TABLE1_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:        [[WEIGHTS_TABLE2_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:        [[PARENT_OUT_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" <33536> -> !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:        [[OUT_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <33536> -> memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[OUT1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <33536> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    //CHECK:        [[OUT2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <33536> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // Upload input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[IN_DDR]] : memref<1x16x32x32xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }

    // Upload 1st part of weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS1_CST]] : memref<16x16x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS1_CMX_COPY]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of weight    %12 = VPURT.DeclareBuffer "CMX_NN" [0] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 1 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS2_CST]] : memref<16x16x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS2_CMX_COPY]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK:        }

    // Upload 1st part of weights table
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE1_CST]] : memref<16x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE1_CMX_COPY]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of weights table
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 1 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE2_CST]] : memref<16x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE2_CMX_COPY]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK:        }

    // 1st task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], out_channel_offset = 0 : i64, task_type = "CONV"
    //CHECK-SAME:       } input([[IN1_CMX]] : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX]] : !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT1_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = "VECTOR_FP16", outEnd = [31, 31, 15], outStart = [0, 0, 0],
    //CHECK-SAME:               pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // 2nd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], out_channel_offset = 16 : i64, task_type = "CONV"
    //CHECK-SAME:       } input([[IN2_CMX]] : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX]] : !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT2_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = "VECTOR_FP16", outEnd = [31, 31, 31], outStart = [0, 0, 16],
    //CHECK-SAME:               pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // Copyback output
    //CHECK:        VPURT.Task waits(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[OUT_CMX:%.*]] : memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT_DDR]] : memref<1x32x32x32xf16, #NHWC, @DDR>)
    //CHECK:        }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ParentInputDistributed_1 = type !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!ParentOutputDistributed_1 = type !VPUIP.DistributedBuffer<
    1x32x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!WeightsDistributed_1 = type !VPUIP.DistributedBuffer<
    32x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2
}>

!WeightsTableDistributed_1 = type !VPUIP.DistributedBuffer<
    32x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2
}>

!ParentInputDistributed_2 = type !VPUIP.DistributedBuffer<
    1x32x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!ParentOutputDistributed_2 = type !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!WeightsDistributed_2 = type !VPUIP.DistributedBuffer<
    16x32x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2
}>

!WeightsTableDistributed_2 = type !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2
}>

!Input_DDR = type memref<1x16x32x32xf16, #NHWC, @DDR>
!Output_DDR = type memref<1x16x32x32xf16, #NHWC, @DDR>

!Weights1_DDR = type memref<32x16x1x1xf16, #NHWC, @DDR>
!WeightsTable1_DDR = type memref<32x1x1x4xsi32, #NCHW, @DDR>

!InputStub1_CMX = type memref<1x16x32x32xf16, #NHWC, @CMX_NN>
!OutputStub1_CMX = type memref<1x32x32x32xf16, #NHWC, @CMX_NN>
!WeightsStub1_CMX = type memref<32x16x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub1_CMX = type memref<32x1x1x4xsi32, #NCHW, @CMX_NN>

!Weights2_DDR = type memref<16x32x1x1xf16, #NHWC, @DDR>
!WeightsTable2_DDR = type memref<16x1x1x4xsi32, #NCHW, @DDR>

!InputStub2_CMX = type memref<1x32x32x32xf16, #NHWC, @CMX_NN>
!OutputStub2_CMX = type memref<1x16x32x32xf16, #NHWC, @CMX_NN>
!WeightsStub2_CMX = type memref<16x32x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub2_CMX = type memref<16x1x1x4xsi32, #NCHW, @CMX_NN>


//CHECK-LABEL: @UnrollNCESequence
func @UnrollNCESequence(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights1_cst = const.Declare memref<32x16x1x1xf16, #NHWC> =
        dense<1.0> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table1_cst = const.Declare memref<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    %weights2_cst = const.Declare memref<16x32x1x1xf16, #NHWC> =
        dense<1.0> : tensor<16x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table2_cst = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer "NetworkInput" <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer "NetworkOutput" <0> -> !Output_DDR

    // CMX buffers/ 1st task
    %parent_input1_cmx = VPURT.DeclareBuffer "CMX_NN" <0> -> !ParentInputDistributed_1
    %weights1 = VPURT.DeclareBuffer "CMX_NN" <32768> -> !WeightsDistributed_1
    %weights_table1 = VPURT.DeclareBuffer "CMX_NN" <33280> -> !WeightsTableDistributed_1
    %parent_out1_cmx = VPURT.DeclareBuffer "CMX_NN" <33536> -> !ParentOutputDistributed_1

    // CMX buffers/ 2nd task
    %parent_input2_cmx = VPURT.DeclareBuffer "CMX_NN" <33536> -> !ParentInputDistributed_2
    %weights2 = VPURT.DeclareBuffer "CMX_NN" <99072> -> !WeightsDistributed_2
    %weights_table2 = VPURT.DeclareBuffer "CMX_NN" <99584> -> !WeightsTableDistributed_2
    %parent_out2_cmx = VPURT.DeclareBuffer "CMX_NN" <0> -> !ParentOutputDistributed_2

    // Upload input
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%parent_in as %arg0: !Input_DDR)
                outputs(%parent_input1_cmx as %arg1: !InputStub1_CMX) -> !ParentInputDistributed_1 {
             VPUIP.NNDMA inputs(%arg0: !Input_DDR) outputs(%arg1: !InputStub1_CMX) -> !InputStub1_CMX
         }
    }

    // Upload weights/ 1st task
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%weights1_cst as %arg0: !Weights1_DDR)
                outputs(%weights1 as %arg1: !WeightsStub1_CMX) -> !WeightsDistributed_1 {
             VPUIP.NNDMA inputs(%arg0: !Weights1_DDR) outputs(%arg1: !WeightsStub1_CMX) -> !WeightsStub1_CMX
         }
    }

    // Upload weights table/ 1st task
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%weights_table1_cst as %arg0: !WeightsTable1_DDR)
                outputs(%weights_table1 as %arg1: !WeightsTableStub1_CMX) -> !WeightsTableDistributed_1 {
             VPUIP.NNDMA inputs(%arg0: !WeightsTable1_DDR) outputs(%arg1: !WeightsTableStub1_CMX) -> !WeightsTableStub1_CMX
         }
    }

    // Cluster tiling/ 1st task
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling
                 inputs(%parent_input1_cmx as %arg0: !InputStub1_CMX,
                         %weights1 as %arg1: !WeightsStub1_CMX,
                         %weights_table1 as %arg2: !WeightsTableStub1_CMX)
                 outputs(%parent_out1_cmx as %arg3: !OutputStub1_CMX)
                     -> !OutputStub1_CMX {

               %1 = VPUIP.NCEClusterTask {
                         kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                         kernel_size = [1, 1],
                         kernel_strides = [1, 1],
                         task_type = "CONV"
                     }  input(%arg0 : !InputStub1_CMX)
                         weights(%arg1 : !WeightsStub1_CMX)
                         weight_table(%arg2 : !WeightsTableStub1_CMX)
                         parent_input(%arg0 : !InputStub1_CMX)
                         parent_output(%arg3 : !OutputStub1_CMX)
                         outputs(%arg3 : !OutputStub1_CMX)
                             -> !OutputStub1_CMX variants :  {
                            DPUTask {
                                outStart = [0, 0, 0], outEnd = [31, 31, 15],
                                mpe_mode = "VECTOR_FP16",
                                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                                cluster_id = 0 : i64
                            }
                            DPUTask {
                                outStart = [0, 0, 16], outEnd = [31, 31, 31],
                                mpe_mode = "VECTOR_FP16",
                                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                                cluster_id = 1 : i64
                            }
                         } PPE :  {
                         }
        }
    }

    // Upload weights/ 2nd task
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%weights2_cst as %arg0: !Weights2_DDR)
                outputs(%weights2 as %arg1: !WeightsStub2_CMX) -> !WeightsDistributed_2 {
             VPUIP.NNDMA inputs(%arg0: !Weights2_DDR) outputs(%arg1: !WeightsStub2_CMX) -> !WeightsStub2_CMX
         }
    }

    // Upload weights table/ 2nd task
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%weights_table2_cst as %arg0: !WeightsTable2_DDR)
                outputs(%weights_table2 as %arg1: !WeightsTableStub2_CMX) -> !WeightsTableDistributed_2 {
             VPUIP.NNDMA inputs(%arg0: !WeightsTable2_DDR) outputs(%arg1: !WeightsTableStub2_CMX) -> !WeightsTableStub2_CMX
         }
    }

    // Cluster tiling/ 2nd task
    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling
                 inputs(%parent_input2_cmx as %arg0: !InputStub2_CMX,
                         %weights2 as %arg1: !WeightsStub2_CMX,
                         %weights_table2 as %arg2: !WeightsTableStub2_CMX)
                 outputs(%parent_out2_cmx as %arg3: !OutputStub2_CMX)
                     -> !OutputStub2_CMX {

               %1 = VPUIP.NCEClusterTask {
                         kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                         kernel_size = [1, 1],
                         kernel_strides = [1, 1],
                         task_type = "CONV"
                     }  input(%arg0 : !InputStub2_CMX)
                         weights(%arg1 : !WeightsStub2_CMX)
                         weight_table(%arg2 : !WeightsTableStub2_CMX)
                         parent_input(%arg0 : !InputStub2_CMX)
                         parent_output(%arg3 : !OutputStub2_CMX)
                         outputs(%arg3 : !OutputStub2_CMX)
                             -> !OutputStub2_CMX variants :  {
                            DPUTask {
                                outStart = [0, 0, 0], outEnd = [31, 31, 7],
                                mpe_mode = "VECTOR_FP16",
                                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                                cluster_id = 0 : i64
                            }
                            DPUTask {
                                outStart = [0, 0, 8], outEnd = [31, 31, 15],
                                mpe_mode = "VECTOR_FP16",
                                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                                cluster_id = 1 : i64
                            }
                         } PPE :  {
                         }
        }
    }

    // Copyback output
    VPURT.Task waits(%bar2: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%parent_out2_cmx as %arg0: !OutputStub2_CMX)
                outputs(%parent_out as %arg1: !Output_DDR) -> !Output_DDR {
             VPUIP.NNDMA inputs(%arg0: !OutputStub2_CMX) outputs(%arg1: !Output_DDR) -> !Output_DDR
         }
    }

    return %output: !Output_DDR

    //CHECK:        [[WEIGHTS_TABLE1_CST_2ND_TASK:%.*]] = const.Declare memref<8x1x1x4xsi32> =
    //CHECK-SAME:       dense<1> : tensor<16x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [8, 1, 1, 4]>]
    //CHECK:        [[WEIGHTS_TABLE2_CST_2ND_TASK:%.*]] = const.Declare memref<8x1x1x4xsi32> =
    //CHECK-SAME:       dense<1> : tensor<16x1x1x4xsi32>, [#const.SubView<[8, 0, 0, 0], [8, 1, 1, 4]>]

    //CHECK:        [[WEIGHTS1_CST_2ND_TASK:%.*]] = const.Declare memref<8x32x1x1xf16, #NHWC> =
    //CHECK-SAME:       dense<1.000000e+00> : tensor<16x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [8, 32, 1, 1]>]
    //CHECK:        [[WEIGHTS2_CST_2ND_TASK:%.*]] = const.Declare memref<8x32x1x1xf16, #NHWC> =
    //CHECK-SAME:       dense<1.000000e+00> : tensor<16x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.SubView<[8, 0, 0, 0], [8, 32, 1, 1]>]

    //CHECK:        [[WEIGHTS_TABLE1_CST_1ST_TASK:%.*]] = const.Declare memref<16x1x1x4xsi32> =
    //CHECK-SAME:       dense<1> : tensor<32x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [16, 1, 1, 4]>]
    //CHECK:        [[WEIGHTS_TABLE2_CST_1ST_TASK:%.*]] = const.Declare memref<16x1x1x4xsi32> =
    //CHECK-SAME:       dense<1> : tensor<32x1x1x4xsi32>, [#const.SubView<[16, 0, 0, 0], [16, 1, 1, 4]>]

    //CHECK:        [[WEIGHTS1_CST_1ST_TASK:%.*]] = const.Declare memref<16x16x1x1xf16, #NHWC> =
    //CHECK-SAME:       dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [16, 16, 1, 1]>]
    //CHECK:        [[WEIGHTS2_CST_1ST_TASK:%.*]] = const.Declare memref<16x16x1x1xf16, #NHWC> =
    //CHECK-SAME:       dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.SubView<[16, 0, 0, 0], [16, 16, 1, 1]>]

    //CHECK:        [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:        [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:        [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:        [[IN_DDR:%.*]] = VPURT.DeclareBuffer "NetworkInput" <0> -> memref<1x16x32x32xf16, #NHWC, @DDR>
    //CHECK:        [[OUT_DDR:%.*]] = VPURT.DeclareBuffer "NetworkOutput" <0> -> memref<1x16x32x32xf16, #NHWC, @DDR>
    //CHECK:        [[PARENT_IN_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:    [[IN1_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[IN_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <0> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:    [[WEIGHTS1_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS2_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS1_CMX_COPY_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS2_CMX_COPY_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS_TABLE1_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS_TABLE2_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS_TABLE1_CMX_COPY_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS_TABLE2_CMX_COPY_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:    [[PARENT_OUT_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" <33536> -> !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUT1_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <33536> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUT2_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <33536> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>


    //CHECK:    [[PARENT_IN_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" <33536> -> !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}
    //CHECK:    [[IN1_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <33536> -> memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <33536> -> memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS1_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <99072> -> memref<8x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS2_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <99072> -> memref<8x32x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS1_CMX_COPY_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <99072> -> memref<8x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS2_CMX_COPY_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <99072> -> memref<8x32x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS_TABLE1_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <99584> -> memref<8x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS_TABLE2_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <99584> -> memref<8x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS_TABLE1_CMX_COPY_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <99584> -> memref<8x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS_TABLE2_CMX_COPY_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <99584> -> memref<8x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:    [[PARENT_OUT_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:    [[OUT_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUT1_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <0> -> !VPUIP.DistributedBuffer<1x8x32x32xf16, {order = #NHWC, strides = [16384, 1, 512, 16]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUT2_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <0> -> !VPUIP.DistributedBuffer<1x8x32x32xf16, {order = #NHWC, strides = [16384, 1, 512, 16]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // Upload input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[IN_DDR]] : memref<1x16x32x32xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }

    // Upload 1st part of weights/ 1st task
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS1_CST_1ST_TASK]] : memref<16x16x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS1_CMX_COPY_1ST_TASK]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of weights/ 1st task
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 1 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS2_CST_1ST_TASK]] : memref<16x16x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS2_CMX_COPY_1ST_TASK]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK:        }

    // Upload 1st part of weights table/ 1st task
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE1_CST_1ST_TASK]] : memref<16x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE1_CMX_COPY_1ST_TASK]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of weights table/ 1st task
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 1 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE2_CST_1ST_TASK]] : memref<16x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE2_CMX_COPY_1ST_TASK]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK:        }

    // 1st task/ 1st subtask
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], out_channel_offset = 0 : i64, task_type = "CONV"
    //CHECK-SAME:       } input([[IN1_CMX_1ST_TASK]] : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX_1ST_TASK]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX_1ST_TASK]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT1_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = "VECTOR_FP16", outEnd = [31, 31, 15], outStart = [0, 0, 0],
    //CHECK-SAME:               pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // 1st task/ 2nd subtask
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], out_channel_offset = 16 : i64, task_type = "CONV"
    //CHECK-SAME:       } input([[IN2_CMX_1ST_TASK]] : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX_1ST_TASK]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX_1ST_TASK]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT2_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = "VECTOR_FP16", outEnd = [31, 31, 31], outStart = [0, 0, 16],
    //CHECK-SAME:               pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // Upload 1st part of weights/ 2nd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS1_CST_2ND_TASK]] : memref<8x32x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS1_CMX_COPY_2ND_TASK]] : memref<8x32x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of weights/ 2nd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 1 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS2_CST_2ND_TASK]] : memref<8x32x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS2_CMX_COPY_2ND_TASK]] : memref<8x32x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK:        }

    // Upload 1st part of weights table/ 2nd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE1_CST_2ND_TASK]] : memref<8x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE1_CMX_COPY_2ND_TASK]] : memref<8x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of weights table/ 2nd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 1 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE2_CST_2ND_TASK]] : memref<8x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE2_CMX_COPY_2ND_TASK]] : memref<8x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK:        }


    // 2nd task/ 1st subtask
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], out_channel_offset = 0 : i64, task_type = "CONV"
    //CHECK-SAME:       } input([[IN1_CMX_2ND_TASK]] : memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX_2ND_TASK]] : memref<8x32x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX_2ND_TASK]] : memref<8x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX_2ND_TASK]] : !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX_2ND_TASK]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT1_CMX_2ND_TASK]] : !VPUIP.DistributedBuffer<1x8x32x32xf16, {order = #NHWC, strides = [16384, 1, 512, 16]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = "VECTOR_FP16", outEnd = [31, 31, 7], outStart = [0, 0, 0],
    //CHECK-SAME:               pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // 2nd task/ 2nd subtask
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], out_channel_offset = 8 : i64, task_type = "CONV"
    //CHECK-SAME:       } input([[IN2_CMX_2ND_TASK]] : memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX_2ND_TASK]] : memref<8x32x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX_2ND_TASK]] : memref<8x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX_2ND_TASK]] : !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX_2ND_TASK]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT2_CMX_2ND_TASK]] : !VPUIP.DistributedBuffer<1x8x32x32xf16, {order = #NHWC, strides = [16384, 1, 512, 16]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = "VECTOR_FP16", outEnd = [31, 31, 15], outStart = [0, 0, 8],
    //CHECK-SAME:               pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // Copyback output
    //CHECK:        VPURT.Task waits([[BAR2]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[OUT_CMX:%.*]] : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT_DDR]] : memref<1x16x32x32xf16, #NHWC, @DDR>)
    //CHECK:        }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ParentInputDistributed = type !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

// Define non contiguous output using strides. Below is an example case
// that would happen in case of CMX concat along "C" axis in NHWC order
// Full output tensor is 1x32x32x32xf16 but given NCE task produces 
// only half of it - 1x16x32x32xf16 
!ParentOutputDistributed = type !VPUIP.DistributedBuffer<
    1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!WeightsDistributed = type !VPUIP.DistributedBuffer<
    16x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2
}>

!WeightsTableDistributed = type !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2
}>

!Input_DDR = type memref<1x16x32x32xf16, #NHWC, @DDR>
!Output_DDR = type memref<1x16x32x32xf16, #NHWC, @DDR>
!Weights_DDR = type memref<16x16x1x1xf16, #NHWC, @DDR>
!WeightsTable_DDR = type memref<16x1x1x4xsi32, #NCHW, @DDR>

!InputStub_CMX = type memref<1x16x32x32xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = type memref<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN>
!WeightsStub_CMX = type memref<16x16x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = type memref<16x1x1x4xsi32, #NCHW, @CMX_NN>

func @UnrollNCENonContiguousOutput(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare memref<32x16x1x1xf16, #NHWC> =
        dense<1.0> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table_cst = const.Declare memref<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer "NetworkInput" <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer "NetworkOutput" <0> -> !Output_DDR

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer "CMX_NN" <0> -> !ParentInputDistributed
    %weights = VPURT.DeclareBuffer "CMX_NN" <32768> -> !WeightsDistributed
    %weights_table = VPURT.DeclareBuffer "CMX_NN" <33280> -> !WeightsTableDistributed
    %parent_out_cmx = VPURT.DeclareBuffer "CMX_NN" <33536> -> !ParentOutputDistributed

    // Upload input
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%parent_in as %arg0: !Input_DDR)
                outputs(%parent_input_cmx as %arg1: !InputStub_CMX) -> !ParentInputDistributed {
             VPUIP.NNDMA inputs(%arg0: !Input_DDR) outputs(%arg1: !InputStub_CMX) -> !InputStub_CMX
         }
    }

    // Upload weights
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%weights_cst as %arg0: !Weights_DDR)
                outputs(%weights as %arg1: !WeightsStub_CMX) -> !WeightsDistributed {
             VPUIP.NNDMA inputs(%arg0: !Weights_DDR) outputs(%arg1: !WeightsStub_CMX) -> !WeightsStub_CMX
         }
    }

    // Upload weights table
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%weights_table_cst as %arg0: !WeightsTable_DDR)
                outputs(%weights_table as %arg1: !WeightsTableStub_CMX) -> !WeightsTableDistributed {
             VPUIP.NNDMA inputs(%arg0: !WeightsTable_DDR) outputs(%arg1: !WeightsTableStub_CMX) -> !WeightsTableStub_CMX
         }
    }

    // Cluster tiling
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling
                 inputs(%parent_input_cmx as %arg0: !InputStub_CMX,
                         %weights as %arg1: !WeightsStub_CMX,
                         %weights_table as %arg2: !WeightsTableStub_CMX)
                 outputs(%parent_out_cmx as %arg3: !OutputStub_CMX)
                     -> !OutputStub_CMX {

               %1 = VPUIP.NCEClusterTask {
                         kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                         kernel_size = [1, 1],
                         kernel_strides = [1, 1],
                         task_type = "CONV"
                     }  input(%arg0 : !InputStub_CMX)
                         weights(%arg1 : !WeightsStub_CMX)
                         weight_table(%arg2 : !WeightsTableStub_CMX)
                         parent_input(%arg0 : !InputStub_CMX)
                         parent_output(%arg3 : !OutputStub_CMX)
                         outputs(%arg3 : !OutputStub_CMX)
                             -> !OutputStub_CMX variants :  {
                            DPUTask {
                                outStart = [0, 0, 0], outEnd = [15, 31, 15],
                                mpe_mode = "VECTOR_FP16",
                                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                                cluster_id = 0 : i64
                            }
                            DPUTask {
                                outStart = [0, 0, 16], outEnd = [15, 31, 31],
                                mpe_mode = "VECTOR_FP16",
                                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                                cluster_id = 1 : i64
                            }
                         } PPE :  {
                         }
        }
    }

    // Copyback output
    VPURT.Task waits(%bar1: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%parent_out_cmx as %arg0: !OutputStub_CMX)
                outputs(%parent_out as %arg1: !Output_DDR) -> !Output_DDR {
             VPUIP.NNDMA inputs(%arg0: !OutputStub_CMX) outputs(%arg1: !Output_DDR) -> !Output_DDR
         }
    }

    return %output: !Output_DDR

    //CHECK:        [[WEIGHTS_TABLE1_CST:%.*]] = const.Declare memref<8x1x1x4xsi32> =
    //CHECK-SAME:       dense<1> : tensor<32x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [8, 1, 1, 4]>]
    //CHECK:        [[WEIGHTS_TABLE2_CST:%.*]] = const.Declare memref<8x1x1x4xsi32> =
    //CHECK-SAME:       dense<1> : tensor<32x1x1x4xsi32>, [#const.SubView<[8, 0, 0, 0], [8, 1, 1, 4]>]

    //CHECK:        [[WEIGHTS1_CST:%.*]] = const.Declare memref<8x16x1x1xf16, #NHWC> =
    //CHECK-SAME:       dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [8, 16, 1, 1]>]
    //CHECK:        [[WEIGHTS2_CST:%.*]] = const.Declare memref<8x16x1x1xf16, #NHWC> =
    //CHECK-SAME:       dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.SubView<[8, 0, 0, 0], [8, 16, 1, 1]>]

    //CHECK:        [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:        [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:        [[IN_DDR:%.*]] = VPURT.DeclareBuffer "NetworkInput" <0> -> memref<1x16x32x32xf16, #NHWC, @DDR>
    //CHECK:        [[OUT_DDR:%.*]] = VPURT.DeclareBuffer "NetworkOutput" <0> -> memref<1x16x32x32xf16, #NHWC, @DDR>

    //CHECK:        [[PARENT_IN_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:    [[IN1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:        [[IN_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <0> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:    [[WEIGHTS1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <32768> -> memref<8x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <32768> -> memref<8x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:        [[WEIGHTS1_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <32768> -> memref<8x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[WEIGHTS2_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <32768> -> memref<8x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS_TABLE1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <33280> -> memref<8x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS_TABLE2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <33280> -> memref<8x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:        [[WEIGHTS_TABLE1_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <33280> -> memref<8x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:        [[WEIGHTS_TABLE2_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <33280> -> memref<8x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:        [[PARENT_OUT_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" <33536> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK:    [[OUT_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <33536> -> memref<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, [@CMX_NN, 0]>
    //CHECK:    [[OUT1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <33536> -> !VPUIP.DistributedBuffer<1x8x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUT2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <33536> -> !VPUIP.DistributedBuffer<1x8x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // Upload input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[IN_DDR]] : memref<1x16x32x32xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }

    // Upload 1st part of weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS1_CST]] : memref<8x16x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS1_CMX_COPY]] : memref<8x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 1 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS2_CST]] : memref<8x16x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS2_CMX_COPY]] : memref<8x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK:        }

    // Upload 1st part of weights table
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE1_CST]] : memref<8x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE1_CMX_COPY]] : memref<8x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of weights table
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 1 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE2_CST]] : memref<8x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE2_CMX_COPY]] : memref<8x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK:        }


    // 1st task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], out_channel_offset = 0 : i64, task_type = "CONV"
    //CHECK-SAME:       } input([[IN1_CMX]] : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX]] : memref<8x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX]] : memref<8x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT1_CMX]] : !VPUIP.DistributedBuffer<1x8x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = "VECTOR_FP16", outEnd = [15, 31, 15], outStart = [0, 0, 0],
    //CHECK-SAME:               pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // 2nd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], out_channel_offset = 8 : i64, task_type = "CONV"
    //CHECK-SAME:       } input([[IN2_CMX]] : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX]] : memref<8x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX]] : memref<8x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT2_CMX]] : !VPUIP.DistributedBuffer<1x8x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = "VECTOR_FP16", outEnd = [15, 31, 31], outStart = [0, 0, 16],
    //CHECK-SAME:               pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // Copyback output
    //CHECK:        VPURT.Task waits(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[OUT_CMX:%.*]] : memref<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT_DDR]] : memref<1x16x32x32xf16, #NHWC, @DDR>)
    //CHECK:        }
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

!typeCmxDistributed = type !VPUIP.DistributedBuffer<
    1x4x512x1xf16, #NCWH, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!type_CMX_memref = type memref<1x4x512x1xf16, #NCWH, @CMX_NN>


!Input_DDR  = type memref<1x4x512x1xf16, #NCWH, @DDR>
!Output_DDR = type memref<1x4x512x1xf16, #NCWH, @DDR>


VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW {
    func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "singleShaveMVN.cpp", VPU.kernel_entry = "singleShaveMVN"}
    func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @UnrollSWOpInterface
func @UnrollSWOpInterface(%input0: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %395 = VPURT.DeclareBuffer "DDR" <4096> -> !Input_DDR
    %302 = VPURT.DeclareBuffer "DDR" <0> -> !Output_DDR

    %300 = VPURT.DeclareBuffer "CMX_NN" <0> -> !typeCmxDistributed
    %301 = VPURT.DeclareBuffer "CMX_NN" <2048> -> !typeCmxDistributed


    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {cycleBegin = 1907728 : i64, cycleEnd = 1907899 : i64, isTrailingSWLayer = false} {
      %398 = VPUIP.NCEClusterTiling inputs(%395 as %arg2: !Input_DDR) outputs(%300 as %arg3: !type_CMX_memref) -> !typeCmxDistributed {
        %399 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg2 : !Input_DDR) outputs(%arg3 : !type_CMX_memref) -> !type_CMX_memref
      }
    }
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {cycleBegin = 1907899 : i64, cycleEnd = 1907901 : i64, isTrailingSWLayer = false} {
      %398 = VPUIP.NCEClusterTiling inputs(%300 as %arg2: !type_CMX_memref) outputs(%301 as %arg3: !type_CMX_memref) -> !typeCmxDistributed {
        %results = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>}
          @VPU.SW::@builtin_MVN inputs(%arg2 as %arg4: !type_CMX_memref) outputs(%arg3 as %arg5: !type_CMX_memref) on tile 0 -> !type_CMX_memref {
          VPUIP.SW.Kernel.run {
            attrs = [false, true, 1.0013580322265625E-5]}(%arg4, %arg5) : !type_CMX_memref, !type_CMX_memref
        }
      }
    }
    VPURT.Task waits(%bar1 : !VPURT.Barrier)  attributes {cycleBegin = 1907901 : i64, cycleEnd = 1908072 : i64, isTrailingSWLayer = false} {
      %398 = VPUIP.NCEClusterTiling inputs(%301 as %arg2 : !type_CMX_memref) outputs(%302 as %arg3: !Output_DDR) -> !Output_DDR {
        %399 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg2 : !type_CMX_memref) outputs(%arg3 : !Output_DDR) -> !Output_DDR
      }
    }


    return %output: !Output_DDR

    //CHECK:        [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:        [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[IN1_DDR:%.*]] = VPURT.DeclareBuffer "DDR" <4096> -> memref<1x2x512x1xf16, #NCWH, @DDR>
    //CHECK:    [[IN2_DDR:%.*]] = VPURT.DeclareBuffer "DDR" <6144> -> memref<1x2x512x1xf16, #NCWH, @DDR>
    //CHECK:    [[OUT1_DDR:%.*]] = VPURT.DeclareBuffer "DDR" <0> -> memref<1x2x512x1xf16, #NCWH, @DDR>
    //CHECK:    [[OUT2_DDR:%.*]] = VPURT.DeclareBuffer "DDR" <2048> -> memref<1x2x512x1xf16, #NCWH, @DDR>

    //CHECK:    [[IN1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x2x512x1xf16, #NCWH, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x2x512x1xf16, #NCWH, [@CMX_NN, 1]>
    //CHECK:    [[IN1_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x2x512x1xf16, #NCWH, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x2x512x1xf16, #NCWH, [@CMX_NN, 1]>

    //CHECK:    [[OUT1_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <2048> -> memref<1x2x512x1xf16, #NCWH, [@CMX_NN, 0]>
    //CHECK:    [[OUT2_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <2048> -> memref<1x2x512x1xf16, #NCWH, [@CMX_NN, 1]>
    //CHECK:    [[OUT1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <2048> -> memref<1x2x512x1xf16, #NCWH, [@CMX_NN, 0]>
    //CHECK:    [[OUT2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <2048> -> memref<1x2x512x1xf16, #NCWH, [@CMX_NN, 1]>

    // Upload 1st part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {cycleBegin = 1907728 : i64, cycleEnd = 1907899 : i64, isTrailingSWLayer = false} {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[IN1_DDR]] : memref<1x2x512x1xf16, #NCWH, @DDR>)
    //CHECK-SAME:       outputs([[IN1_CMX_COPY]] : memref<1x2x512x1xf16, #NCWH, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {cycleBegin = 1907728 : i64, cycleEnd = 1907899 : i64, isTrailingSWLayer = false} {
    //CHECK:          VPUIP.NNDMA {port = 1 : i64}
    //CHECK-SAME:       inputs([[IN2_DDR]] : memref<1x2x512x1xf16, #NCWH, @DDR>)
    //CHECK-SAME:       outputs([[IN2_CMX_COPY]] : memref<1x2x512x1xf16, #NCWH, [@CMX_NN, 1]>)
    //CHECK:        }

    // sw tasks
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {cycleBegin = 1907899 : i64, cycleEnd = 1907901 : i64, isTrailingSWLayer = false} {
    //CHECK:          VPUIP.SW.Kernel
    //CHECK-SAME:       inputs([[IN1_CMX]] as %arg2: memref<1x2x512x1xf16, #NCWH, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT1_CMX]] as %arg3: memref<1x2x512x1xf16, #NCWH, [@CMX_NN, 0]>)
    //CHECK:        }

    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {cycleBegin = 1907899 : i64, cycleEnd = 1907901 : i64, isTrailingSWLayer = false} {
    //CHECK:          VPUIP.SW.Kernel
    //CHECK-SAME:       inputs([[IN2_CMX]] as %arg2: memref<1x2x512x1xf16, #NCWH, [@CMX_NN, 1]>)
    //CHECK-SAME:       outputs([[OUT2_CMX]] as %arg3: memref<1x2x512x1xf16, #NCWH, [@CMX_NN, 1]>)
    //CHECK:        }

    // Copyback 1st part of output
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) attributes {cycleBegin = 1907901 : i64, cycleEnd = 1908072 : i64, isTrailingSWLayer = false} {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[OUT1_CMX_COPY]] : memref<1x2x512x1xf16, #NCWH, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT1_DDR]] : memref<1x2x512x1xf16, #NCWH, @DDR>)
    //CHECK:        }

    // Copyback 2nd part of output
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) attributes {cycleBegin = 1907901 : i64, cycleEnd = 1908072 : i64, isTrailingSWLayer = false} {
    //CHECK:          VPUIP.NNDMA {port = 1 : i64}
    //CHECK-SAME:       inputs([[OUT2_CMX_COPY]] : memref<1x2x512x1xf16, #NCWH, [@CMX_NN, 1]>)
    //CHECK-SAME:       outputs([[OUT2_DDR]] : memref<1x2x512x1xf16, #NCWH, @DDR>)
    //CHECK:        }
}
