// RUN: vpux-opt --split-input-file --unroll-cluster-tiling  %s | FileCheck %s

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
!OutputStub_CMX = type memref<1x32x32x32xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = type memref<32x16x1x1xf16, #NHWC, @CMX_NN>

func @UnrollNNDMA(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare memref<32x16x1x1xf16, #NHWC> =
        #const.Content<dense<1.0> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]>

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer "NetworkInput" <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer "NetworkOutput" <0> -> !Output_DDR

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer "CMX_NN" <0> -> !ParentInputDistributed
    %input1 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> !InputStub_CMX
    %input2 = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> !InputStub_CMX

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
        VPUIP.NNDMA {port = 0 : i64} inputs(%input1: memref<1x16x32x32xf16, #NHWC, @CMX_NN>) outputs(%out_cmx1 : !OutputDistributed) -> !OutputDistributed
    }

    // Simulate 2st task
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
        VPUIP.NNDMA {port = 0 : i64} inputs(%input2: memref<1x16x32x32xf16, #NHWC, @CMX_NN>) outputs(%out_cmx2 : !OutputDistributed) -> !OutputDistributed
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
    //CHECK-SAME:       #const.Content<dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [16, 16, 1, 1]>]>
    //CHECK:        [[WEIGHTS2_CST:%.*]] = const.Declare memref<16x16x1x1xf16, #NHWC> =
    //CHECK-SAME:       #const.Content<dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.SubView<[16, 0, 0, 0], [16, 16, 1, 1]>]>

    //CHECK:        [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:        [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:        [[IN_DDR:%.*]] = VPURT.DeclareBuffer "NetworkInput" <0> -> memref<1x16x32x32xf16, #NHWC, @DDR>
    //CHECK:        [[OUT_DDR:%.*]] = VPURT.DeclareBuffer "NetworkOutput" <0> -> memref<1x32x32x32xf16, #NHWC, @DDR>

    //CHECK:        [[IN1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x32x32xf16, #NHWC, @CMX_NN>
    //CHECK:        [[IN2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x32x32xf16, #NHWC, @CMX_NN>

    // Upload input
    //CHECK:        [[IN_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <0> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[IN_DDR]] : memref<1x16x32x32xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>)
    //CHECK:        }

    // Upload 1st part of weights
    //CHECK:        [[WEIGHTS1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS1_CST]] : memref<16x16x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS1_CMX]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of weights
    //CHECK:        [[WEIGHTS2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS2_CST]] : memref<16x16x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS2_CMX]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK:        }

    // Copyback output
    //CHECK:        [[OUT_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <33280> -> memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 0]>
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

func @UnrollNCE(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare memref<32x16x1x1xf16, #NHWC> =
        #const.Content<dense<1.0> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]>
    %weights_table_cst = const.Declare memref<32x1x1x4xsi32> = #const.Content<dense<1> : tensor<32x1x1x4xsi32>>

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
                                start = [0, 0, 0], end = [31, 31, 15],
                                mpe_mode = "VECTOR_FP16",
                                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                                cluster_id = 0 : i64
                            }
                            DPUTask {
                                start = [0, 0, 16], end = [31, 31, 31],
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
    //CHECK-SAME:       #const.Content<dense<1> : tensor<32x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [16, 1, 1, 4]>]>
    //CHECK:        [[WEIGHTS_TABLE2_CST:%.*]] = const.Declare memref<16x1x1x4xsi32> =
    //CHECK-SAME:       #const.Content<dense<1> : tensor<32x1x1x4xsi32>, [#const.SubView<[16, 0, 0, 0], [16, 1, 1, 4]>]>

    //CHECK:        [[WEIGHTS1_CST:%.*]] = const.Declare memref<16x16x1x1xf16, #NHWC> =
    //CHECK-SAME:       #const.Content<dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [16, 16, 1, 1]>]>
    //CHECK:        [[WEIGHTS2_CST:%.*]] = const.Declare memref<16x16x1x1xf16, #NHWC> =
    //CHECK-SAME:       #const.Content<dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.SubView<[16, 0, 0, 0], [16, 16, 1, 1]>]>

    //CHECK:        [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:        [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:        [[IN_DDR:%.*]] = VPURT.DeclareBuffer "NetworkInput" <0> -> memref<1x16x32x32xf16, #NHWC, @DDR>
    //CHECK:        [[OUT_DDR:%.*]] = VPURT.DeclareBuffer "NetworkOutput" <0> -> memref<1x32x32x32xf16, #NHWC, @DDR>

    //CHECK:        [[PARENT_IN_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>
    //CHECK:        [[PARENT_OUT_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" <33536> -> !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = DUPLICATED|SEGMENTED, num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // Upload input
    //CHECK:        [[IN_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <0> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[IN_DDR]] : memref<1x16x32x32xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>)
    //CHECK:        }

    // Upload 1st part of weights
    //CHECK:        [[WEIGHTS1_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS1_CST]] : memref<16x16x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS1_CMX_COPY]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of weights
    //CHECK:        [[WEIGHTS2_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS2_CST]] : memref<16x16x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS2_CMX_COPY]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK:        }

    // Upload 1st part of weights table
    //CHECK:        [[WEIGHTS_TABLE1_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE1_CST]] : memref<16x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE1_CMX_COPY]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of weights table
    //CHECK:        [[WEIGHTS_TABLE2_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE2_CST]] : memref<16x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE2_CMX_COPY]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK:        }

    //CHECK-DAG:    [[IN1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[WEIGHTS1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[WEIGHTS_TABLE1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUT1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <33536> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = DUPLICATED|SEGMENTED, num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK-DAG:    [[IN2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK-DAG:    [[WEIGHTS2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK-DAG:    [[WEIGHTS_TABLE2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK-DAG:    [[OUT2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <33536> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = DUPLICATED|SEGMENTED, num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // 1st task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"
    //CHECK-SAME:       } input([[IN1_CMX]] : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX]] : !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = DUPLICATED|SEGMENTED, num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT1_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = DUPLICATED|SEGMENTED, num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, end = [31, 31, 15], mpe_mode = "VECTOR_FP16",
    //CHECK-SAME:               pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // 2nd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"
    //CHECK-SAME:       } input([[IN2_CMX]] : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX]] : !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = DUPLICATED|SEGMENTED, num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT2_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = DUPLICATED|SEGMENTED, num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, end = [31, 31, 31], mpe_mode = "VECTOR_FP16",
    //CHECK-SAME:               pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 16]}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // Copyback output
    //CHECK:        [[OUT_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <33536> -> memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 0]>
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

func @UnrollNCESequence(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights1_cst = const.Declare memref<32x16x1x1xf16, #NHWC> =
        #const.Content<dense<1.0> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]>
    %weights_table1_cst = const.Declare memref<32x1x1x4xsi32> = #const.Content<dense<1> : tensor<32x1x1x4xsi32>>

    %weights2_cst = const.Declare memref<16x32x1x1xf16, #NHWC> =
        #const.Content<dense<1.0> : tensor<16x32x1x1xf16>, [#const.Reorder<#NHWC>]>
    %weights_table2_cst = const.Declare memref<16x1x1x4xsi32> = #const.Content<dense<1> : tensor<16x1x1x4xsi32>>

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
                                start = [0, 0, 0], end = [31, 31, 15],
                                mpe_mode = "VECTOR_FP16",
                                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                                cluster_id = 0 : i64
                            }
                            DPUTask {
                                start = [0, 0, 16], end = [31, 31, 31],
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
                                start = [0, 0, 0], end = [31, 31, 7],
                                mpe_mode = "VECTOR_FP16",
                                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                                cluster_id = 0 : i64
                            }
                            DPUTask {
                                start = [0, 0, 8], end = [31, 31, 15],
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
    //CHECK-SAME:       #const.Content<dense<1> : tensor<16x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [8, 1, 1, 4]>]>
    //CHECK:        [[WEIGHTS_TABLE2_CST_2ND_TASK:%.*]] = const.Declare memref<8x1x1x4xsi32> =
    //CHECK-SAME:       #const.Content<dense<1> : tensor<16x1x1x4xsi32>, [#const.SubView<[8, 0, 0, 0], [8, 1, 1, 4]>]>

    //CHECK:        [[WEIGHTS1_CST_2ND_TASK:%.*]] = const.Declare memref<8x32x1x1xf16, #NHWC> =
    //CHECK-SAME:       #const.Content<dense<1.000000e+00> : tensor<16x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [8, 32, 1, 1]>]>
    //CHECK:        [[WEIGHTS2_CST_2ND_TASK:%.*]] = const.Declare memref<8x32x1x1xf16, #NHWC> =
    //CHECK-SAME:       #const.Content<dense<1.000000e+00> : tensor<16x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.SubView<[8, 0, 0, 0], [8, 32, 1, 1]>]>

    //CHECK:        [[WEIGHTS_TABLE1_CST_1ST_TASK:%.*]] = const.Declare memref<16x1x1x4xsi32> =
    //CHECK-SAME:       #const.Content<dense<1> : tensor<32x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [16, 1, 1, 4]>]>
    //CHECK:        [[WEIGHTS_TABLE2_CST_1ST_TASK:%.*]] = const.Declare memref<16x1x1x4xsi32> =
    //CHECK-SAME:       #const.Content<dense<1> : tensor<32x1x1x4xsi32>, [#const.SubView<[16, 0, 0, 0], [16, 1, 1, 4]>]>

    //CHECK:        [[WEIGHTS1_CST_1ST_TASK:%.*]] = const.Declare memref<16x16x1x1xf16, #NHWC> =
    //CHECK-SAME:       #const.Content<dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [16, 16, 1, 1]>]>
    //CHECK:        [[WEIGHTS2_CST_1ST_TASK:%.*]] = const.Declare memref<16x16x1x1xf16, #NHWC> =
    //CHECK-SAME:       #const.Content<dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.SubView<[16, 0, 0, 0], [16, 16, 1, 1]>]>

    //CHECK:        [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:        [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:        [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:        [[IN_DDR:%.*]] = VPURT.DeclareBuffer "NetworkInput" <0> -> memref<1x16x32x32xf16, #NHWC, @DDR>
    //CHECK:        [[OUT_DDR:%.*]] = VPURT.DeclareBuffer "NetworkOutput" <0> -> memref<1x16x32x32xf16, #NHWC, @DDR>

    //CHECK-DAG:    [[PARENT_IN_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>
    //CHECK-DAG:    [[PARENT_OUT_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" <33536> -> !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = DUPLICATED|SEGMENTED, num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK-DAG:    [[PARENT_IN_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" <33536> -> !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}
    //CHECK-DAG:    [[PARENT_OUT_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = DUPLICATED|SEGMENTED, num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // Upload input
    //CHECK:        [[IN_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <0> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[IN_DDR]] : memref<1x16x32x32xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>)
    //CHECK:        }

    // Upload 1st part of weights/ 1st task
    //CHECK:        [[WEIGHTS1_CMX_COPY_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS1_CST_1ST_TASK]] : memref<16x16x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS1_CMX_COPY_1ST_TASK]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of weights/ 1st task
    //CHECK:        [[WEIGHTS2_CMX_COPY_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS2_CST_1ST_TASK]] : memref<16x16x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS2_CMX_COPY_1ST_TASK]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK:        }

    // Upload 1st part of weights table/ 1st task
    //CHECK:        [[WEIGHTS_TABLE1_CMX_COPY_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE1_CST_1ST_TASK]] : memref<16x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE1_CMX_COPY_1ST_TASK]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of weights table/ 1st task
    //CHECK:        [[WEIGHTS_TABLE2_CMX_COPY_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE2_CST_1ST_TASK]] : memref<16x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE2_CMX_COPY_1ST_TASK]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK:        }

    //CHECK-DAG:    [[IN1_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[WEIGHTS1_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[WEIGHTS_TABLE1_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUT1_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <33536> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = DUPLICATED|SEGMENTED, num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK-DAG:    [[IN2_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK-DAG:    [[WEIGHTS2_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK-DAG:    [[WEIGHTS_TABLE2_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK-DAG:    [[OUT2_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <33536> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = DUPLICATED|SEGMENTED, num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // 1st task/ 1st subtask
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"
    //CHECK-SAME:       } input([[IN1_CMX_1ST_TASK]] : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX_1ST_TASK]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX_1ST_TASK]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = DUPLICATED|SEGMENTED, num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT1_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = DUPLICATED|SEGMENTED, num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, end = [31, 31, 15], mpe_mode = "VECTOR_FP16",
    //CHECK-SAME:               pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // 1st task/ 2nd subtask
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"
    //CHECK-SAME:       } input([[IN2_CMX_1ST_TASK]] : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX_1ST_TASK]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX_1ST_TASK]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = DUPLICATED|SEGMENTED, num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT2_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = DUPLICATED|SEGMENTED, num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, end = [31, 31, 31], mpe_mode = "VECTOR_FP16",
    //CHECK-SAME:               pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 16]}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // Upload 1st part of weights/ 2nd task
    //CHECK:        [[WEIGHTS1_CMX_COPY_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <99072> -> memref<8x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS1_CST_2ND_TASK]] : memref<8x32x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS1_CMX_COPY_2ND_TASK]] : memref<8x32x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of weights/ 2nd task
    //CHECK:        [[WEIGHTS2_CMX_COPY_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <99072> -> memref<8x32x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS2_CST_2ND_TASK]] : memref<8x32x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS2_CMX_COPY_2ND_TASK]] : memref<8x32x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK:        }

    // Upload 1st part of weights table/ 2nd task
    //CHECK:        [[WEIGHTS_TABLE1_CMX_COPY_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <99584> -> memref<8x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE1_CST_2ND_TASK]] : memref<8x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE1_CMX_COPY_2ND_TASK]] : memref<8x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of weights table/ 2nd task
    //CHECK:        [[WEIGHTS_TABLE2_CMX_COPY_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <99584> -> memref<8x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE2_CST_2ND_TASK]] : memref<8x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE2_CMX_COPY_2ND_TASK]] : memref<8x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK:        }

    //CHECK-DAG:    [[IN1_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <33536> -> memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[WEIGHTS1_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <99072> -> memref<8x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[WEIGHTS_TABLE1_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <99584> -> memref<8x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUT1_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <0> -> !VPUIP.DistributedBuffer<1x8x32x32xf16, {order = #NHWC, strides = [16384, 1, 512, 16]}, @CMX_NN, {mode = DUPLICATED|SEGMENTED, num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    //CHECK-DAG:    [[IN2_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <33536> -> memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK-DAG:    [[WEIGHTS2_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <99072> -> memref<8x32x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK-DAG:    [[WEIGHTS_TABLE2_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <99584> -> memref<8x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK-DAG:    [[OUT2_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <0> -> !VPUIP.DistributedBuffer<1x8x32x32xf16, {order = #NHWC, strides = [16384, 1, 512, 16]}, @CMX_NN, {mode = DUPLICATED|SEGMENTED, num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // 2nd task/ 1st subtask
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"
    //CHECK-SAME:       } input([[IN1_CMX_2ND_TASK]] : memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX_2ND_TASK]] : memref<8x32x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX_2ND_TASK]] : memref<8x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX_2ND_TASK]] : !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX_2ND_TASK]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = DUPLICATED|SEGMENTED, num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT1_CMX_2ND_TASK]] : !VPUIP.DistributedBuffer<1x8x32x32xf16, {order = #NHWC, strides = [16384, 1, 512, 16]}, @CMX_NN, {mode = DUPLICATED|SEGMENTED, num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, end = [31, 31, 7], mpe_mode = "VECTOR_FP16",
    //CHECK-SAME:               pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // 2nd task/ 2nd subtask
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"
    //CHECK-SAME:       } input([[IN2_CMX_2ND_TASK]] : memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX_2ND_TASK]] : memref<8x32x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX_2ND_TASK]] : memref<8x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX_2ND_TASK]] : !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX_2ND_TASK]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = DUPLICATED|SEGMENTED, num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT2_CMX_2ND_TASK]] : !VPUIP.DistributedBuffer<1x8x32x32xf16, {order = #NHWC, strides = [16384, 1, 512, 16]}, @CMX_NN, {mode = DUPLICATED|SEGMENTED, num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, end = [31, 31, 15], mpe_mode = "VECTOR_FP16",
    //CHECK-SAME:               pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 8]}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // Copyback output
    //CHECK:        [[OUT_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        VPURT.Task waits([[BAR2]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[OUT_CMX:%.*]] : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT_DDR]] : memref<1x16x32x32xf16, #NHWC, @DDR>)
    //CHECK:        }
}
