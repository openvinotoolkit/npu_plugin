// RUN: vpux-opt --split-input-file --unroll-cluster-tiling  %s | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// In this example DPU task in SOHoverlapped mode with input 1x16x32x32
// will be split into 2 clusters where each input will be 1x16x17x32
!InputDistributed = type !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [3, 3],
    pads = {bottom = 0, left = 0, right = 0, top = 0},
    strides = [1, 1],
    num_clusters = 2
}>

!OutputDistributed = type !VPUIP.DistributedBuffer<
    1x16x30x30xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!WeightsDistributed = type !VPUIP.DistributedBuffer<
    16x16x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = type memref<1x16x32x32xf16, #NHWC, @DDR>
!Output_DDR = type memref<1x16x30x30xf16, #NHWC, @DDR>
!Weights_DDR = type memref<16x16x3x3xf16, #NHWC, @DDR>

!InputStub_CMX = type memref<1x16x32x32xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = type memref<1x16x30x30xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = type memref<16x16x3x3xf16, #NHWC, @CMX_NN>

// Below dimensions are not correct but this is because related NCETask
// is just simulated using NNDMA
!Buffer0_CMX = type memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 0]>
!Buffer1_CMX = type memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 1]>

func @UnrollNNDMA(%input: memref<1x16x32x32xf16>, %output: memref<1x16x30x30xf16>) -> memref<1x16x30x30xf16> {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare memref<16x16x3x3xf16, #NHWC> =
        #const.Content<dense<1.0> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]>

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer "DDR" <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer "DDR" <32768> -> !Output_DDR

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer "CMX_NN" <0> -> !InputDistributed
    %input1 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> !Buffer0_CMX
    %input2 = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> !Buffer1_CMX

    %parent_out_cmx = VPURT.DeclareBuffer "CMX_NN" <17408> -> !OutputDistributed
    %output1 = VPURT.DeclareBuffer "CMX_NN" [0] <17408> -> !Buffer0_CMX
    %output2 = VPURT.DeclareBuffer "CMX_NN" [1] <17408> -> !Buffer1_CMX

    %weights = VPURT.DeclareBuffer "CMX_NN" <31808> -> !WeightsDistributed

    // Reorder input

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.PermuteUPA {order_value = #NHWC}
            inputs(%input: memref<1x16x32x32xf16>)
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
            outputs(%output: memref<1x16x30x30xf16>)
            -> memref<1x16x30x30xf16>
    }

    return %output: memref<1x16x30x30xf16>

    //CHECK:    [[WEIGHTS_CST:%.*]] = const.Declare memref<16x16x3x3xf16, #NHWC>

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR3:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[IN1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[OUT1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <17408> -> memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUT2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <17408> -> memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 1]>

    // Upload weights
    //CHECK:        [[WEIGHTS_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <31808> -> !VPUIP.DistributedBuffer<16x16x3x3xf16, #NHWC, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS_CST]] : memref<16x16x3x3xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS_CMX_COPY]] : !VPUIP.DistributedBuffer<16x16x3x3xf16, #NHWC, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>)
    //CHECK:        }

    // Upload 1st part of input
    //CHECK-DAG:    [[IN1_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[IN1_DDR:%.*]] = VPURT.DeclareBuffer "DDR" <0> -> memref<1x16x17x32xf16, #NHWC, @DDR>
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[IN1_DDR]] : memref<1x16x17x32xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[IN1_CMX_COPY]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of input
    //CHECK-DAG:    [[IN2_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK-DAG:    [[IN2_DDR:%.*]] = VPURT.DeclareBuffer "DDR" <15360> -> memref<1x16x17x32xf16, #NHWC, @DDR>
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[IN2_DDR]] : memref<1x16x17x32xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[IN2_CMX_COPY]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK:        }

    // Simulate tasks
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)  {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN1_CMX]] : memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT1_CMX]] : memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:        }
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)  {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN2_CMX]] : memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:       outputs([[OUT2_CMX]] : memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK:        }

    // Copyback 1st part of output
    //CHECK-DAG:    [[OUT1_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <17408> -> memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUT1_DDR:%.*]] = VPURT.DeclareBuffer "DDR" <32768> -> memref<1x16x15x30xf16, #NHWC, @DDR>
    //CHECK:        VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[OUT1_CMX_COPY]] : memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT1_DDR]] : memref<1x16x15x30xf16, #NHWC, @DDR>)
    //CHECK:        }

    // Copyback 2nd part of output
    //CHECK-DAG:    [[OUT2_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <17408> -> memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK-DAG:    [[OUT2_DDR:%.*]] = VPURT.DeclareBuffer "DDR" <47168> -> memref<1x16x15x30xf16, #NHWC, @DDR>
    //CHECK:        VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[OUT2_CMX_COPY]] : memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:       outputs([[OUT2_DDR]] : memref<1x16x15x30xf16, #NHWC, @DDR>)
    //CHECK:        }

}
