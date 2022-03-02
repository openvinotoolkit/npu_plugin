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
