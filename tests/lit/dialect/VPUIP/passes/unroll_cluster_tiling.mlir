// RUN: vpux-opt --split-input-file --unroll-cluster-tiling  %s | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x16x33x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!OutputDistributed = type !VPUIP.DistributedBuffer<
    1x16x33x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!WeightsDistributed = type !VPUIP.DistributedBuffer<
    16x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = type memref<1x16x33x32xf16, #NHWC, @DDR>
!Output_DDR = type memref<1x16x33x32xf16, #NHWC, @DDR>
!Weights_DDR = type memref<16x16x1x1xf16, #NHWC, @DDR>

!InputStub_CMX = type memref<1x16x33x32xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = type memref<1x16x33x32xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = type memref<16x16x1x1xf16, #NHWC, @CMX_NN>

!Buffer0_CMX = type memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
!Buffer1_CMX = type memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>

func @UnrollNNDMA(%input: memref<1x16x33x32xf16>, %output: memref<1x16x33x32xf16>) -> memref<1x16x33x32xf16> {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare memref<16x16x1x1xf16, #NHWC> =
        #const.Content<dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]>

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer "DDR" <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer "DDR" <33792> -> !Output_DDR

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer "CMX_NN" <0> -> !InputDistributed
    %input1 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> !Buffer0_CMX
    %input2 = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> !Buffer1_CMX

    %parent_out_cmx = VPURT.DeclareBuffer "CMX_NN" <17408> -> !OutputDistributed
    %output1 = VPURT.DeclareBuffer "CMX_NN" [0] <17408> -> !Buffer0_CMX
    %output2 = VPURT.DeclareBuffer "CMX_NN" [1] <17408> -> !Buffer1_CMX

    %weights = VPURT.DeclareBuffer "CMX_NN" <34816> -> !WeightsDistributed

    // Reorder input

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.PermuteUPA {order_value = #NHWC}
            inputs(%input: memref<1x16x33x32xf16>)
            outputs(%parent_in: !Input_DDR)
            -> !Input_DDR
    }

    // Upload weights
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%weights_cst as %arg0: !Weights_DDR)
                outputs(%weights as %arg1: !WeightsStub_CMX) -> !WeightsDistributed {
             VPUIP.NNDMA inputs(%arg0: !Weights_DDR) outputs(%arg1: !WeightsStub_CMX) -> !WeightsStub_CMX
         }
    }

    // Upload input
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%parent_in as %arg0: !Input_DDR)
                outputs(%parent_input_cmx as %arg1: !InputStub_CMX) -> !InputDistributed {
             VPUIP.NNDMA inputs(%arg0: !Input_DDR) outputs(%arg1: !InputStub_CMX) -> !InputStub_CMX
         }
    }

    // Simulate 1st task
    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%input1: !Buffer0_CMX) outputs(%output1: !Buffer0_CMX) -> !Buffer0_CMX
    }

    // Simulate 2nd task
    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%input2: !Buffer1_CMX) outputs(%output2: !Buffer1_CMX) -> !Buffer1_CMX
    }

    // Copyback output
    VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%parent_out_cmx as %arg0: !OutputStub_CMX)
                outputs(%parent_out as %arg1: !Output_DDR) -> !Output_DDR {
             VPUIP.NNDMA inputs(%arg0: !OutputStub_CMX) outputs(%arg1: !Output_DDR) -> !Output_DDR
         }
    }

    // Reorder output

    VPURT.Task waits(%bar3: !VPURT.Barrier) {
        VPUIP.PermuteUPA {order_value = #NCHW}
            inputs(%parent_out: !Output_DDR)
            outputs(%output: memref<1x16x33x32xf16>)
            -> memref<1x16x33x32xf16>
    }

    return %output: memref<1x16x33x32xf16>

    //CHECK:    [[WEIGHTS_CST:%.*]] = const.Declare memref<16x16x1x1xf16, #NHWC>

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR3:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[IN1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[OUT1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <17408> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUT2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <17408> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>

    // Upload weights
    //CHECK:        [[WEIGHTS_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <34816> -> !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS_CST]] : memref<16x16x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS_CMX_COPY]] : !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>)
    //CHECK:        }

    // Upload 1st part of input
    //CHECK:        [[IN1_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[IN1_DDR:%.*]] = VPURT.DeclareBuffer "DDR" <0> -> memref<1x16x17x32xf16, #NHWC, @DDR>
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[IN1_DDR]] : memref<1x16x17x32xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[IN1_CMX_COPY]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of input
    //CHECK:        [[IN2_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:        [[IN2_DDR:%.*]] = VPURT.DeclareBuffer "DDR" <17408> -> memref<1x16x16x32xf16, #NHWC, @DDR>
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[IN2_DDR]] : memref<1x16x16x32xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[IN2_CMX_COPY]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK:        }

    // Simulate tasks
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)  {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN1_CMX]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT1_CMX]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:        }
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)  {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN2_CMX]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:       outputs([[OUT2_CMX]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK:        }

    // Copyback 1st part of output
    //CHECK:        [[OUT1_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <17408> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[OUT1_DDR:%.*]] = VPURT.DeclareBuffer "DDR" <33792> -> memref<1x16x17x32xf16, #NHWC, @DDR>
    //CHECK:        VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[OUT1_CMX_COPY]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT1_DDR]] : memref<1x16x17x32xf16, #NHWC, @DDR>)
    //CHECK:        }

    // Copyback 2nd part of output
    //CHECK:        [[OUT2_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <17408> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:        [[OUT2_DDR:%.*]] = VPURT.DeclareBuffer "DDR" <51200> -> memref<1x16x16x32xf16, #NHWC, @DDR>
    //CHECK:        VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[OUT2_CMX_COPY]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:       outputs([[OUT2_DDR]] : memref<1x16x16x32xf16, #NHWC, @DDR>)
    //CHECK:        }

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x16x33x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!OutputDistributed = type !VPUIP.DistributedBuffer<
    1x16x33x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!WeightsDistributed = type !VPUIP.DistributedBuffer<
    16x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsTableDistributed = type !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = type memref<1x16x33x32xf16, #NHWC>
!Output_DDR = type memref<1x16x33x32xf16, #NHWC>
!Weights_DDR = type memref<16x16x1x1xf16, #NHWC>
!WeightsTable_DDR = type memref<16x1x1x4xsi32>

!InputStub_CMX = type memref<1x16x33x32xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = type memref<1x16x33x32xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = type memref<16x16x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = type memref<16x1x1x4xsi32, @CMX_NN>

func @UnrollNCE(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare memref<16x16x1x1xf16, #NHWC> =
        #const.Content<dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]>
    %weights_table_cst = const.Declare memref<16x1x1x4xsi32> = #const.Content<dense<1> : tensor<16x1x1x4xsi32>>

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer "NetworkOutput" [0] <0> -> !Output_DDR

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer "CMX_NN" <0> -> !InputDistributed
    %parent_out_cmx = VPURT.DeclareBuffer "CMX_NN" <17408> -> !OutputDistributed
    %weights = VPURT.DeclareBuffer "CMX_NN" [0, 1] <34816> -> !WeightsDistributed
    %weights_table = VPURT.DeclareBuffer "CMX_NN" [0, 1] <35328> -> !WeightsTableDistributed

    // Upload input
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%parent_in as %arg0: !Input_DDR)
                outputs(%parent_input_cmx as %arg1: !InputStub_CMX) -> !InputDistributed {
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
                                start = [0, 0, 0], end = [31, 16, 31],
                                mpe_mode = "VECTOR_FP16",
                                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                                cluster_id = 0 : i64
                            }
                            DPUTask {
                                start = [0, 17, 0], end = [31, 32, 31],
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

    //CHECK:    [[WEIGHTS_TABLE_CST:%.*]] = const.Declare memref<16x1x1x4xsi32>
    //CHECK:    [[WEIGHTS_CST:%.*]] = const.Declare memref<16x16x1x1xf16, #NHWC>

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[PARENT_IN_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = SEGMENTED, num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[PARENT_OUT_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" <17408> -> !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = SEGMENTED, num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // Upload 1st part of input
    //CHECK:        [[IN1_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[IN1_DDR:%.*]] = VPURT.DeclareBuffer "DDR" <0> -> memref<1x16x17x32xf16, #NHWC>
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[IN1_DDR]] : memref<1x16x17x32xf16, #NHWC>)
    //CHECK-SAME:       outputs([[IN1_CMX_COPY]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2st part of input
    //CHECK:        [[IN2_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:        [[IN2_DDR:%.*]] = VPURT.DeclareBuffer "DDR" <17408> -> memref<1x16x16x32xf16, #NHWC>
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[IN2_DDR]] : memref<1x16x16x32xf16, #NHWC>)
    //CHECK-SAME:       outputs([[IN2_CMX_COPY]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK:        }

    // Upload weights
    //CHECK:        [[WEIGHTS_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <34816> -> !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS_CST]] : memref<16x16x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS_CMX_COPY]] : !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>)
    //CHECK:        }

    // Upload weights table
    //CHECK:        [[WEIGHTS_TABLE_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <35328> -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE_CST]] : memref<16x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE_CMX_COPY]] : !VPUIP.DistributedBuffer<16x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>)
    //CHECK:        }

    //CHECK-DAG:    [[IN1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[WEIGHTS1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <34816> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[WEIGHTS_TABLE1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <35328> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUT1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <17408> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK-DAG:    [[IN2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK-DAG:    [[WEIGHTS2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <34816> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK-DAG:    [[WEIGHTS_TABLE2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <35328> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK-DAG:    [[OUT2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <17408> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>

    // 1st task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NCEClusterTask
    //CHECK-SAME:           {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"}
    //CHECK-SAME:       input([[IN1_CMX]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = SEGMENTED, num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = SEGMENTED, num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT1_CMX]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, end = [31, 16, 31], mpe_mode = "VECTOR_FP16",
    //CHECK-SAME:               pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // 2nd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NCEClusterTask
    //CHECK-SAME:           {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"}
    //CHECK-SAME:       input([[IN2_CMX]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = SEGMENTED, num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = SEGMENTED, num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT2_CMX]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, end = [31, 32, 31], mpe_mode = "VECTOR_FP16",
    //CHECK-SAME:               pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 17, 0]}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    //CHECK:        [[OUT1_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <17408> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[OUT1_DDR:%.*]] = VPURT.DeclareBuffer "DDR" <0> -> memref<1x16x17x32xf16, #NHWC>
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[OUT1_CMX_COPY]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT1_DDR]] : memref<1x16x17x32xf16, #NHWC>)
    //CHECK:        }

    //CHECK:        [[OUT2_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <17408> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:        [[OUT2_DDR:%.*]] = VPURT.DeclareBuffer "DDR" <17408> -> memref<1x16x16x32xf16, #NHWC>
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[OUT2_CMX_COPY]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:       outputs([[OUT2_DDR]] : memref<1x16x16x32xf16, #NHWC>)
    //CHECK:        }

    //CHECK:    return %arg1 : memref<1x16x33x32xf16, #NHWC>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x16x33x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!OutputDistributed = type !VPUIP.DistributedBuffer<
    1x16x33x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!WeightsDistributed = type !VPUIP.DistributedBuffer<
    16x16x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsTableDistributed = type !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = type memref<1x16x33x32xf16, #NHWC>
!Output_DDR = type memref<1x16x33x32xf16, #NHWC>
!Weights_DDR = type memref<16x16x3x3xf16, #NHWC>
!WeightsTable_DDR = type memref<16x1x1x4xsi32>

!InputStub_CMX = type memref<1x16x33x32xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = type memref<1x16x33x32xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = type memref<16x16x3x3xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = type memref<16x1x1x4xsi32, @CMX_NN>

func @UnrollNCESegmentedConv(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare memref<16x16x3x3xf16, #NHWC> =
        #const.Content<dense<1.0> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]>
    %weights_table_cst = const.Declare memref<16x1x1x4xsi32> = #const.Content<dense<1> : tensor<16x1x1x4xsi32>>

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer "NetworkOutput" [0] <0> -> !Output_DDR

    // CMX buffers
    %parent_input_cmx    = VPURT.DeclareBuffer "CMX_NN" <0> -> !InputDistributed
    %parent_out_cmx = VPURT.DeclareBuffer "CMX_NN" <17408> -> !OutputDistributed
    %weights = VPURT.DeclareBuffer "CMX_NN" [0, 1] <34816> -> !WeightsDistributed
    %weights_table = VPURT.DeclareBuffer "CMX_NN" [0, 1] <39424> -> !WeightsTableDistributed

    // Upload input
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%parent_in as %arg0: !Input_DDR)
                outputs(%parent_input_cmx as %arg1: !InputStub_CMX) -> !InputDistributed {
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
                         kernel_padding = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
                         kernel_size = [3, 3],
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
                                start = [0, 0, 0], end = [31, 16, 31],
                                mpe_mode = "VECTOR_FP16",
                                pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
                                cluster_id = 0 : i64
                            }
                            DPUTask {
                                start = [0, 17, 0], end = [31, 32, 31],
                                mpe_mode = "VECTOR_FP16",
                                pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64},
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

    //CHECK:    [[WEIGHTS_TABLE_CST:%.*]] = const.Declare memref<16x1x1x4xsi32>
    //CHECK:    [[WEIGHTS_CST:%.*]] = const.Declare memref<16x16x3x3xf16, #NHWC>

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[PARENT_IN_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = SEGMENTED, num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[PARENT_OUT_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" <17408> -> !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = SEGMENTED, num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // Upload 1st part of input
    //CHECK:        [[IN1_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[IN1_DDR:%.*]] = VPURT.DeclareBuffer "DDR" <0> -> memref<1x16x17x32xf16, #NHWC>
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[IN1_DDR]] : memref<1x16x17x32xf16, #NHWC>)
    //CHECK-SAME:       outputs([[IN1_CMX_COPY]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2st part of input
    //CHECK:        [[IN2_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:        [[IN2_DDR:%.*]] = VPURT.DeclareBuffer "DDR" <17408> -> memref<1x16x16x32xf16, #NHWC>
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[IN2_DDR]] : memref<1x16x16x32xf16, #NHWC>)
    //CHECK-SAME:       outputs([[IN2_CMX_COPY]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK:        }

    // Upload weights
    //CHECK:        [[WEIGHTS_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <34816> -> !VPUIP.DistributedBuffer<16x16x3x3xf16, #NHWC, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS_CST]] : memref<16x16x3x3xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS_CMX_COPY]] : !VPUIP.DistributedBuffer<16x16x3x3xf16, #NHWC, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>)
    //CHECK:        }

    // Upload weights table
    //CHECK:        [[WEIGHTS_TABLE_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <39424> -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE_CST]] : memref<16x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE_CMX_COPY]] : !VPUIP.DistributedBuffer<16x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>)
    //CHECK:        }

    //CHECK-DAG:    [[IN1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[WEIGHTS1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <34816> -> memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[WEIGHTS_TABLE1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <39424> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUT1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <17408> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK-DAG:    [[IN2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK-DAG:    [[WEIGHTS2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <34816> -> memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK-DAG:    [[WEIGHTS_TABLE2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <39424> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK-DAG:    [[OUT2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <17408> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>

    // 1st task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NCEClusterTask
    //CHECK-SAME:           {is_segmented, kernel_padding = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    //CHECK-SAME:           kernel_size = [3, 3], kernel_strides = [1, 1], task_type = "CONV"}
    //CHECK-SAME:       input([[IN1_CMX]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX]] : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = SEGMENTED, num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = SEGMENTED, num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT1_CMX]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, end = [31, 16, 31], mpe_mode = "VECTOR_FP16",
    //CHECK-SAME:               pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, start = [0, 0, 0]}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // 2nd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NCEClusterTask
    //CHECK-SAME:           {is_segmented, kernel_padding = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    //CHECK-SAME:           kernel_size = [3, 3], kernel_strides = [1, 1], task_type = "CONV"}
    //CHECK-SAME:       input([[IN2_CMX]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX]] : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = SEGMENTED, num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = SEGMENTED, num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT2_CMX]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, end = [31, 32, 31], mpe_mode = "VECTOR_FP16",
    //CHECK-SAME:               pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, start = [0, 17, 0]}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    //CHECK:        [[OUT1_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <17408> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[OUT1_DDR:%.*]] = VPURT.DeclareBuffer "DDR" <0> -> memref<1x16x17x32xf16, #NHWC>
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[OUT1_CMX_COPY]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT1_DDR]] : memref<1x16x17x32xf16, #NHWC>)
    //CHECK:        }

    //CHECK:        [[OUT2_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <17408> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:        [[OUT2_DDR:%.*]] = VPURT.DeclareBuffer "DDR" <17408> -> memref<1x16x16x32xf16, #NHWC>
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[OUT2_CMX_COPY]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:       outputs([[OUT2_DDR]] : memref<1x16x16x32xf16, #NHWC>)
    //CHECK:        }

    //CHECK:    return %arg1 : memref<1x16x33x32xf16, #NHWC>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x16x33x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!OutputDistributed = type !VPUIP.DistributedBuffer<
    1x16x33x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!WeightsDistributed = type !VPUIP.DistributedBuffer<
    16x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsTableDistributed = type !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = type memref<1x16x33x32xf16, #NHWC>
!Output_DDR = type memref<1x16x33x32xf16, #NHWC>
!Weights_DDR = type memref<16x16x1x1xf16, #NHWC>
!WeightsTable_DDR = type memref<16x1x1x4xsi32>

!InputStub_CMX = type memref<1x16x33x32xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = type memref<1x16x33x32xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = type memref<16x16x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = type memref<16x1x1x4xsi32, @CMX_NN>

func @UnrollNCESequence(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare memref<16x16x1x1xf16, #NHWC> =
        #const.Content<dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]>
    %weights_table_cst = const.Declare memref<16x1x1x4xsi32> = #const.Content<dense<1> : tensor<16x1x1x4xsi32>>

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer "NetworkOutput" [0] <0> -> !Output_DDR

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer "CMX_NN" <0> -> !InputDistributed
    %parent_out_cmx = VPURT.DeclareBuffer "CMX_NN" <17408> -> !OutputDistributed
    %weights = VPURT.DeclareBuffer "CMX_NN" [0, 1] <34816> -> !WeightsDistributed
    %weights_table = VPURT.DeclareBuffer "CMX_NN" [0, 1] <35328> -> !WeightsTableDistributed

    %parent_input_cmx_2 = VPURT.DeclareBuffer "CMX_NN" <17408> -> !InputDistributed
    %parent_out_cmx_2 = VPURT.DeclareBuffer "CMX_NN" <0> -> !OutputDistributed

    // Upload input
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%parent_in as %arg0: !Input_DDR)
                outputs(%parent_input_cmx as %arg1: !InputStub_CMX) -> !InputDistributed {
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
                                start = [0, 0, 0], end = [31, 16, 31],
                                mpe_mode = "VECTOR_FP16",
                                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                                cluster_id = 0 : i64
                            }
                            DPUTask {
                                start = [0, 17, 0], end = [31, 32, 31],
                                mpe_mode = "VECTOR_FP16",
                                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                                cluster_id = 1 : i64
                            }
                         } PPE :  {
                         }
        }
    }

    // Cluster tiling
    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling
                 inputs(%parent_input_cmx_2 as %arg0: !InputStub_CMX,
                         %weights as %arg1: !WeightsStub_CMX,
                         %weights_table as %arg2: !WeightsTableStub_CMX)
                 outputs(%parent_out_cmx_2 as %arg3: !OutputStub_CMX)
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
                                start = [0, 0, 0], end = [31, 16, 31],
                                mpe_mode = "VECTOR_FP16",
                                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                                cluster_id = 0 : i64
                            }
                            DPUTask {
                                start = [0, 17, 0], end = [31, 32, 31],
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
         %0 = VPUIP.NCEClusterTiling inputs(%parent_out_cmx_2 as %arg0: !OutputStub_CMX)
                outputs(%parent_out as %arg1: !Output_DDR) -> !Output_DDR {
             VPUIP.NNDMA inputs(%arg0: !OutputStub_CMX) outputs(%arg1: !Output_DDR) -> !Output_DDR
         }
    }

    return %output: !Output_DDR

    //CHECK:    [[WEIGHTS_TABLE_CST:%.*]] = const.Declare memref<16x1x1x4xsi32>
    //CHECK:    [[WEIGHTS_CST:%.*]] = const.Declare memref<16x16x1x1xf16, #NHWC>

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[PARENT_IN_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = SEGMENTED, num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[PARENT_OUT_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" <17408> -> !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = SEGMENTED, num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:    [[PARENT_IN_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" <17408> -> !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = SEGMENTED, num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[PARENT_OUT_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = SEGMENTED, num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // Upload 1st part of input
    //CHECK:        [[IN1_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[IN1_DDR:%.*]] = VPURT.DeclareBuffer "DDR" <0> -> memref<1x16x17x32xf16, #NHWC>
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[IN1_DDR]] : memref<1x16x17x32xf16, #NHWC>)
    //CHECK-SAME:       outputs([[IN1_CMX_COPY]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2st part of input
    //CHECK:        [[IN2_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:        [[IN2_DDR:%.*]] = VPURT.DeclareBuffer "DDR" <17408> -> memref<1x16x16x32xf16, #NHWC>
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[IN2_DDR]] : memref<1x16x16x32xf16, #NHWC>)
    //CHECK-SAME:       outputs([[IN2_CMX_COPY]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK:        }

    // Upload weights
    //CHECK:        [[WEIGHTS_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <34816> -> !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS_CST]] : memref<16x16x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS_CMX_COPY]] : !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>)
    //CHECK:        }

    // Upload weights table
    //CHECK:        [[WEIGHTS_TABLE_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <35328> -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE_CST]] : memref<16x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE_CMX_COPY]] : !VPUIP.DistributedBuffer<16x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>)
    //CHECK:        }

    //CHECK-DAG:    [[IN1_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[WEIGHTS1_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <34816> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[WEIGHTS_TABLE1_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <35328> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUT1_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <17408> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK-DAG:    [[IN2_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK-DAG:    [[WEIGHTS2_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <34816> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK-DAG:    [[WEIGHTS_TABLE2_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <35328> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK-DAG:    [[OUT2_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <17408> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>

    // 1st task/ 1st subtask
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NCEClusterTask
    //CHECK-SAME:           {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"}
    //CHECK-SAME:       input([[IN1_CMX_1ST_TASK]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX_1ST_TASK]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX_1ST_TASK]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = SEGMENTED, num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = SEGMENTED, num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT1_CMX_1ST_TASK]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, end = [31, 16, 31], mpe_mode = "VECTOR_FP16",
    //CHECK-SAME:               pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // 1st task/ 2nd subtask
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NCEClusterTask
    //CHECK-SAME:           {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"}
    //CHECK-SAME:       input([[IN2_CMX_1ST_TASK]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX_1ST_TASK]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX_1ST_TASK]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = SEGMENTED, num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = SEGMENTED, num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT2_CMX_1ST_TASK]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, end = [31, 32, 31], mpe_mode = "VECTOR_FP16",
    //CHECK-SAME:               pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 17, 0]}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    //CHECK-DAG:    [[IN1_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <17408> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[WEIGHTS1_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <34816> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[WEIGHTS_TABLE1_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <35328> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUT1_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK-DAG:    [[IN2_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <17408> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK-DAG:    [[WEIGHTS2_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <34816> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK-DAG:    [[WEIGHTS_TABLE2_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <35328> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK-DAG:    [[OUT2_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>

    // 2nd task/ 1st subtask
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NCEClusterTask
    //CHECK-SAME:           {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"}
    //CHECK-SAME:       input([[IN1_CMX_2ND_TASK]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX_2ND_TASK]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX_2ND_TASK]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX_2ND_TASK]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = SEGMENTED, num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX_2ND_TASK]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = SEGMENTED, num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT1_CMX_2ND_TASK]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, end = [31, 16, 31], mpe_mode = "VECTOR_FP16",
    //CHECK-SAME:               pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // 2nd task/ 2nd subtask
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NCEClusterTask
    //CHECK-SAME:           {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"}
    //CHECK-SAME:       input([[IN2_CMX_2ND_TASK]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX_2ND_TASK]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX_2ND_TASK]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX_2ND_TASK]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = SEGMENTED, num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX_2ND_TASK]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = SEGMENTED, num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT2_CMX_2ND_TASK]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, end = [31, 32, 31], mpe_mode = "VECTOR_FP16",
    //CHECK-SAME:               pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 17, 0]}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    //CHECK:        [[OUT1_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[OUT1_DDR:%.*]] = VPURT.DeclareBuffer "DDR" <0> -> memref<1x16x17x32xf16, #NHWC>
    //CHECK:        VPURT.Task waits([[BAR2]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[OUT1_CMX_COPY]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT1_DDR]] : memref<1x16x17x32xf16, #NHWC>)
    //CHECK:        }

    //CHECK:        [[OUT2_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:        [[OUT2_DDR:%.*]] = VPURT.DeclareBuffer "DDR" <17408> -> memref<1x16x16x32xf16, #NHWC>
    //CHECK:        VPURT.Task waits([[BAR2]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[OUT2_CMX_COPY]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:       outputs([[OUT2_DDR]] : memref<1x16x16x32xf16, #NHWC>)
    //CHECK:        }

    //CHECK:    return %arg1 : memref<1x16x33x32xf16, #NHWC>
}
