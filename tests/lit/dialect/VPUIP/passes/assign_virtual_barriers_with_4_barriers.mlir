// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=KMB" --assign-virtual-barriers="num-barriers=4 num-slots-per-barrier=1" %s | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ParallelGraph
func @ParallelGraph(%arg0: memref<1x16x32x32xf16, #NHWC>, %arg1: memref<1x16x32x32xf16>) -> memref<1x16x32x32xf16> {
    %cst0 = const.Declare memref<16x16x1x1xf16, #NHWC> =
        #const.Content<dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]>
    %cst1 = const.Declare memref<16x1x1x4xsi32> = #const.Content<dense<1> : tensor<16x1x1x4xsi32>>

    // input buffers for SOH tiling
    %buf0 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x16x32x32xf16, #NHWC>
    %buf1 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x16x8x32xf16, #NHWC>
    %buf2 = VPURT.DeclareBuffer "DDR" <8192> -> memref<1x16x8x32xf16, #NHWC>
    %buf3 = VPURT.DeclareBuffer "DDR" <16384> -> memref<1x16x8x32xf16, #NHWC>
    %buf4 = VPURT.DeclareBuffer "DDR" <24576> -> memref<1x16x8x32xf16, #NHWC>

    // output buffers for SOH tiling
    %buf5 = VPURT.DeclareBuffer "DDR" <32768> -> memref<1x16x32x32xf16, #NHWC>
    %buf6 = VPURT.DeclareBuffer "DDR" <32768> -> memref<1x16x8x32xf16, #NHWC>
    %buf7 = VPURT.DeclareBuffer "DDR" <40960> -> memref<1x16x8x32xf16, #NHWC>
    %buf8 = VPURT.DeclareBuffer "DDR" <49152> -> memref<1x16x8x32xf16, #NHWC>
    %buf9 = VPURT.DeclareBuffer "DDR" <57344> -> memref<1x16x8x32xf16, #NHWC>

    // CMX buffers
    %buf10 = VPURT.DeclareBuffer "CMX_NN" <0> -> memref<1x16x8x32xf16, #NHWC, @CMX_NN>
    %buf11 = VPURT.DeclareBuffer "CMX_NN" <8192> -> memref<1x16x8x32xf16, #NHWC, @CMX_NN>
    %buf12 = VPURT.DeclareBuffer "CMX_NN" <16384> -> memref<1x16x8x32xf16, #NHWC, @CMX_NN>
    %buf13 = VPURT.DeclareBuffer "CMX_NN" <24576> -> memref<1x16x8x32xf16, #NHWC, @CMX_NN>
    %buf14 = VPURT.DeclareBuffer "CMX_NN" <32768> -> memref<16x16x1x1xf16, #NHWC, @CMX_NN>
    %buf15 = VPURT.DeclareBuffer "CMX_NN" <33280> -> memref<16x1x1x4xsi32, @CMX_NN>

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar5 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar6 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar7 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar8 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar9 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar10 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar11 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar12 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar13 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // Upload weights and weights table

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%cst0: memref<16x16x1x1xf16, #NHWC>)
            outputs(%buf14: memref<16x16x1x1xf16, #NHWC, @CMX_NN>)
            -> memref<16x16x1x1xf16, #NHWC, @CMX_NN>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%cst1: memref<16x1x1x4xsi32>)
            outputs(%buf15: memref<16x1x1x4xsi32, @CMX_NN>)
            -> memref<16x1x1x4xsi32, @CMX_NN>
    }

    // Copy input

    VPURT.Task updates(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA
            inputs(%arg0: memref<1x16x32x32xf16, #NHWC>)
            outputs(%buf0: memref<1x16x32x32xf16, #NHWC>)
            -> memref<1x16x32x32xf16, #NHWC>
    }

    // Upload 1st input tile

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf1: memref<1x16x8x32xf16, #NHWC>)
            outputs(%buf10: memref<1x16x8x32xf16, #NHWC, @CMX_NN>)
            -> memref<1x16x8x32xf16, #NHWC, @CMX_NN>
    }

    // 1st tile

    VPURT.Task waits(%bar0, %bar2: !VPURT.Barrier, !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) {
        VPUIP.NCEClusterTask {
                kernel_padding = {bottom = 0, left = 0, right = 0, top = 0},
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = "CONV"
            }
            input(%buf10: memref<1x16x8x32xf16, #NHWC, @CMX_NN>)
            weights(%buf14: memref<16x16x1x1xf16, #NHWC, @CMX_NN>)
            weight_table(%buf15: memref<16x1x1x4xsi32, @CMX_NN>)
            parent_input(%buf10: memref<1x16x8x32xf16, #NHWC, @CMX_NN>)
            parent_output(%buf11: memref<1x16x8x32xf16, #NHWC, @CMX_NN>)
            outputs(%buf11: memref<1x16x8x32xf16, #NHWC, @CMX_NN>)
            -> memref<1x16x8x32xf16, #NHWC, @CMX_NN>
            variants : {
                DPUTask {
                    start = [0, 0, 0],
                    end = [31, 7, 15],
                    pad = {bottom = 0, left = 0, right = 0, top = 0},
                    mpe_mode = "VECTOR_FP16"
                }
            } PPE : {
            }
    }

    // Copyback 1st result tile

    VPURT.Task waits(%bar3: !VPURT.Barrier) updates(%bar4: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf11: memref<1x16x8x32xf16, #NHWC, @CMX_NN>)
            outputs(%buf6: memref<1x16x8x32xf16, #NHWC>)
            -> memref<1x16x8x32xf16, #NHWC>
    }

    // Upload 2st input tile

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar5: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf2: memref<1x16x8x32xf16, #NHWC>)
            outputs(%buf12: memref<1x16x8x32xf16, #NHWC, @CMX_NN>)
            -> memref<1x16x8x32xf16, #NHWC, @CMX_NN>
    }

    // 2nd tile

    VPURT.Task waits(%bar0, %bar5: !VPURT.Barrier, !VPURT.Barrier) updates(%bar6: !VPURT.Barrier) {
        VPUIP.NCEClusterTask {
                kernel_padding = {bottom = 0, left = 0, right = 0, top = 0},
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = "CONV"
            }
            input(%buf12: memref<1x16x8x32xf16, #NHWC, @CMX_NN>)
            weights(%buf14: memref<16x16x1x1xf16, #NHWC, @CMX_NN>)
            weight_table(%buf15: memref<16x1x1x4xsi32, @CMX_NN>)
            parent_input(%buf12: memref<1x16x8x32xf16, #NHWC, @CMX_NN>)
            parent_output(%buf13: memref<1x16x8x32xf16, #NHWC, @CMX_NN>)
            outputs(%buf13: memref<1x16x8x32xf16, #NHWC, @CMX_NN>)
            -> memref<1x16x8x32xf16, #NHWC, @CMX_NN>
            variants : {
                DPUTask {
                    start = [0, 0, 0],
                    end = [31, 7, 15],
                    pad = {bottom = 0, left = 0, right = 0, top = 0},
                    mpe_mode = "VECTOR_FP16"
                }
            } PPE : {
            }
    }

    // Copyback 2nd result tile

    VPURT.Task waits(%bar6: !VPURT.Barrier) updates(%bar7: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf13: memref<1x16x8x32xf16, #NHWC, @CMX_NN>)
            outputs(%buf7: memref<1x16x8x32xf16, #NHWC>)
            -> memref<1x16x8x32xf16, #NHWC>
    }

    // Upload 3st input tile

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar8: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf3: memref<1x16x8x32xf16, #NHWC>)
            outputs(%buf10: memref<1x16x8x32xf16, #NHWC, @CMX_NN>)
            -> memref<1x16x8x32xf16, #NHWC, @CMX_NN>
    }

    // 3rd tile

    VPURT.Task waits(%bar0, %bar8: !VPURT.Barrier, !VPURT.Barrier) updates(%bar9: !VPURT.Barrier) {
        VPUIP.NCEClusterTask {
                kernel_padding = {bottom = 0, left = 0, right = 0, top = 0},
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = "CONV"
            }
            input(%buf10: memref<1x16x8x32xf16, #NHWC, @CMX_NN>)
            weights(%buf14: memref<16x16x1x1xf16, #NHWC, @CMX_NN>)
            weight_table(%buf15: memref<16x1x1x4xsi32, @CMX_NN>)
            parent_input(%buf10: memref<1x16x8x32xf16, #NHWC, @CMX_NN>)
            parent_output(%buf11: memref<1x16x8x32xf16, #NHWC, @CMX_NN>)
            outputs(%buf11: memref<1x16x8x32xf16, #NHWC, @CMX_NN>)
            -> memref<1x16x8x32xf16, #NHWC, @CMX_NN>
            variants : {
                DPUTask {
                    start = [0, 0, 0],
                    end = [31, 7, 15],
                    pad = {bottom = 0, left = 0, right = 0, top = 0},
                    mpe_mode = "VECTOR_FP16"
                }
            } PPE : {
            }
    }

    // Copyback 3rd result tile

    VPURT.Task waits(%bar9: !VPURT.Barrier) updates(%bar10: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf11: memref<1x16x8x32xf16, #NHWC, @CMX_NN>)
            outputs(%buf8: memref<1x16x8x32xf16, #NHWC>)
            -> memref<1x16x8x32xf16, #NHWC>
    }

    // Upload 4st input tile

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar11: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf4: memref<1x16x8x32xf16, #NHWC>)
            outputs(%buf12: memref<1x16x8x32xf16, #NHWC, @CMX_NN>)
            -> memref<1x16x8x32xf16, #NHWC, @CMX_NN>
    }

    // 4th tile

    VPURT.Task waits(%bar0, %bar11: !VPURT.Barrier, !VPURT.Barrier) updates(%bar12: !VPURT.Barrier) {
        VPUIP.NCEClusterTask {
                kernel_padding = {bottom = 0, left = 0, right = 0, top = 0},
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = "CONV"
            }
            input(%buf12: memref<1x16x8x32xf16, #NHWC, @CMX_NN>)
            weights(%buf14: memref<16x16x1x1xf16, #NHWC, @CMX_NN>)
            weight_table(%buf15: memref<16x1x1x4xsi32, @CMX_NN>)
            parent_input(%buf12: memref<1x16x8x32xf16, #NHWC, @CMX_NN>)
            parent_output(%buf13: memref<1x16x8x32xf16, #NHWC, @CMX_NN>)
            outputs(%buf13: memref<1x16x8x32xf16, #NHWC, @CMX_NN>)
            -> memref<1x16x8x32xf16, #NHWC, @CMX_NN>
            variants : {
                DPUTask {
                    start = [0, 0, 0],
                    end = [31, 7, 15],
                    pad = {bottom = 0, left = 0, right = 0, top = 0},
                    mpe_mode = "VECTOR_FP16"
                }
            } PPE : {
            }
    }

    // Copyback 4th result tile

    VPURT.Task waits(%bar12: !VPURT.Barrier) updates(%bar13: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf13: memref<1x16x8x32xf16, #NHWC, @CMX_NN>)
            outputs(%buf9: memref<1x16x8x32xf16, #NHWC>)
            -> memref<1x16x8x32xf16, #NHWC>
    }

    // Reorder output

    VPURT.Task waits(%bar4, %bar7, %bar10, %bar13: !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier) {
        VPUIP.PermuteUPA {order_value = #NCHW}
            inputs(%buf5: memref<1x16x32x32xf16, #NHWC>)
            outputs(%arg1: memref<1x16x32x32xf16>)
            -> memref<1x16x32x32xf16>
    }

    return %arg1 : memref<1x16x32x32xf16>

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR3:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR4:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR5:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR6:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR7:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR8:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR9:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR10:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR11:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR12:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR13:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR14:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       VPURT.Task
    // CHECK-SAME:       updates([[BAR0]] : !VPURT.Barrier)
    // CHECK:       VPURT.Task
    // CHECK-SAME:       updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:       VPURT.Task
    // CHECK-SAME:       waits([[BAR0]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)
    // CHECK:       VPURT.Task
    // CHECK-SAME:       waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)
    // CHECK:       VPURT.Task
    // CHECK-SAME:       waits([[BAR1]], [[BAR2]] : !VPURT.Barrier, !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier)
    // CHECK:       VPURT.Task
    // CHECK-SAME:       waits([[BAR3]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier)
    // CHECK:       VPURT.Task
    // CHECK-SAME:       waits([[BAR4]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier)
    // CHECK:       VPURT.Task
    // CHECK-SAME:       waits([[BAR1]], [[BAR5]] : !VPURT.Barrier, !VPURT.Barrier) updates([[BAR7]] : !VPURT.Barrier)
    // CHECK:       VPURT.Task
    // CHECK-SAME:       waits([[BAR6]] : !VPURT.Barrier) updates([[BAR8]] : !VPURT.Barrier)
    // CHECK:       VPURT.Task
    // CHECK-SAME:       waits([[BAR7]] : !VPURT.Barrier) updates([[BAR9]] : !VPURT.Barrier)
    // CHECK:       VPURT.Task
    // CHECK-SAME:       waits([[BAR8]] : !VPURT.Barrier) updates([[BAR10]] : !VPURT.Barrier)
    // CHECK:       VPURT.Task
    // CHECK-SAME:       waits([[BAR9]] : !VPURT.Barrier) updates([[BAR11]] : !VPURT.Barrier)
    // CHECK:       VPURT.Task
    // CHECK-SAME:       waits([[BAR10]] : !VPURT.Barrier) updates([[BAR12]] : !VPURT.Barrier)
    // CHECK:       VPURT.Task
    // CHECK-SAME:       waits([[BAR11]] : !VPURT.Barrier) updates([[BAR13]] : !VPURT.Barrier)
    // CHECK:       VPURT.Task
    // CHECK-SAME:       waits([[BAR12]] : !VPURT.Barrier) updates([[BAR14]] : !VPURT.Barrier)
    // CHECK:       VPURT.Task
    // CHECK-SAME:       waits([[BAR13]], [[BAR14]] : !VPURT.Barrier, !VPURT.Barrier)
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x16x8x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4
}>

!WeightsDistributed = type !VPUIP.DistributedBuffer<
    16x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4
}>

!WeightsTableDistributed = type !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4
}>

!OutputDistributed = type !VPUIP.DistributedBuffer<
    1x16x8x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4
}>

!Input_DDR = type memref<1x16x8x32xf16, #NHWC, @DDR>
!Weights_DDR = type memref<16x16x1x1xf16, #NHWC, @DDR>
!WeightsTable_DDR = type memref<16x1x1x4xsi32, #NHWC, @DDR>
!Output_DDR = type memref<1x16x8x32xf16, #NHWC, @DDR>

!InputStub_CMX = type memref<1x16x8x32xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = type memref<16x16x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = type memref<16x1x1x4xsi32, #NHWC, @CMX_NN>
!OutputStub_CMX = type memref<1x16x8x32xf16, #NHWC, @CMX_NN>

// CHECK-LABEL: @ParallelGraphMultiCluster
func @ParallelGraphMultiCluster(%input: memref<1x16x32x32xf16, #NHWC, @DDR>, %output: memref<1x16x32x32xf16, @DDR>) -> memref<1x16x32x32xf16, @DDR> {
    %cst0 = const.Declare memref<16x16x1x1xf16, #NHWC, @DDR> =
        #const.Content<dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]>
    %cst1 = const.Declare memref<16x1x1x4xsi32, #NHWC, @DDR> = #const.Content<dense<1> : tensor<16x1x1x4xsi32>, [#const.Reorder<#NHWC>]>

    // input buffers for SOH tiling
    %buf0 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x16x32x32xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer "DDR" <0> -> !Input_DDR
    %buf2 = VPURT.DeclareBuffer "DDR" <8192> -> !Input_DDR
    %buf3 = VPURT.DeclareBuffer "DDR" <16384> -> !Input_DDR
    %buf4 = VPURT.DeclareBuffer "DDR" <24576> -> !Input_DDR

    // output buffers for SOH tiling
    %buf5 = VPURT.DeclareBuffer "DDR" <32768> -> memref<1x16x32x32xf16, #NHWC, @DDR>
    %buf6 = VPURT.DeclareBuffer "DDR" <32768> -> !Output_DDR
    %buf7 = VPURT.DeclareBuffer "DDR" <40960> -> !Output_DDR
    %buf8 = VPURT.DeclareBuffer "DDR" <49152> -> !Output_DDR
    %buf9 = VPURT.DeclareBuffer "DDR" <57344> -> !Output_DDR

    // CMX buffers
    %buf10 = VPURT.DeclareBuffer "CMX_NN" <0> -> !InputDistributed
    %buf11 = VPURT.DeclareBuffer "CMX_NN" <8192> -> !OutputDistributed
    %buf12 = VPURT.DeclareBuffer "CMX_NN" <16384> -> !InputDistributed
    %buf13 = VPURT.DeclareBuffer "CMX_NN" <24576> -> !OutputDistributed
    %buf14 = VPURT.DeclareBuffer "CMX_NN" <32768> -> !InputDistributed
    %buf15 = VPURT.DeclareBuffer "CMX_NN" <40960> -> !OutputDistributed
    %buf16 = VPURT.DeclareBuffer "CMX_NN" <49152> -> !InputDistributed
    %buf17 = VPURT.DeclareBuffer "CMX_NN" <57344> -> !OutputDistributed
    %buf18 = VPURT.DeclareBuffer "CMX_NN" <65536> -> !WeightsDistributed
    %buf19 = VPURT.DeclareBuffer "CMX_NN" <66048> -> !WeightsTableDistributed

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar5 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar6 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar7 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar8 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar9 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar10 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar11 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar12 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar13 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // Upload weights and weights table

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        %0 = VPUIP.NCEClusterTiling inputs(%cst0 as %arg0: !Weights_DDR) outputs(%buf18 as %arg1: !WeightsStub_CMX) -> !WeightsDistributed {
            VPUIP.NNDMA
            inputs(%arg0: !Weights_DDR)
            outputs(%arg1: !WeightsStub_CMX)
            -> !WeightsStub_CMX
        }
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        %0 = VPUIP.NCEClusterTiling inputs(%cst1 as %arg0: !WeightsTable_DDR) outputs(%buf19 as %arg1: !WeightsTableStub_CMX) -> !WeightsTableDistributed {
            VPUIP.NNDMA
            inputs(%arg0: !WeightsTable_DDR)
            outputs(%arg1: !WeightsTableStub_CMX)
            -> !WeightsTableStub_CMX
        }
    }

    // Copy input

    VPURT.Task updates(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA
            inputs(%input: memref<1x16x32x32xf16, #NHWC, @DDR>)
            outputs(%buf0: memref<1x16x32x32xf16, #NHWC, @DDR>)
            -> memref<1x16x32x32xf16, #NHWC, @DDR>
    }

    // Upload 1st input tile

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
        %0 = VPUIP.NCEClusterTiling inputs(%buf1 as %arg0: !Input_DDR) outputs(%buf10 as %arg1: !InputStub_CMX) -> !InputDistributed {
            VPUIP.NNDMA
            inputs(%arg0: !Input_DDR)
            outputs(%arg1: !InputStub_CMX)
            -> !InputStub_CMX
        }
    }

    // 1st tile

    VPURT.Task waits(%bar0, %bar2: !VPURT.Barrier, !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) {
        %0 = VPUIP.NCEClusterTiling
                    inputs(%buf10 as %arg0: !InputStub_CMX,
                            %buf18 as %arg1: !WeightsStub_CMX,
                            %buf19 as %arg2: !WeightsTableStub_CMX)
                    outputs(%buf11 as %arg3: !OutputStub_CMX)
                        -> !OutputStub_CMX {

            VPUIP.NCEClusterTask {
                    kernel_padding = {bottom = 0, left = 0, right = 0, top = 0},
                    kernel_size = [1, 1],
                    kernel_strides = [1, 1],
                    task_type = "CONV"
                }
                input(%arg0: !InputStub_CMX)
                weights(%arg1: !WeightsStub_CMX)
                weight_table(%arg2: !WeightsTableStub_CMX)
                parent_input(%arg0: !InputStub_CMX)
                parent_output(%arg3: !OutputStub_CMX)
                outputs(%arg3: !OutputStub_CMX)
                -> memref<1x16x8x32xf16, #NHWC, @CMX_NN>
                variants : {
                    DPUTask {
                        start = [0, 0, 0],
                        end = [31, 7, 15],
                        pad = {bottom = 0, left = 0, right = 0, top = 0},
                        mpe_mode = "VECTOR_FP16"
                    }
                } PPE : {
                }
        }
    }

    // Copyback 1st result tile

    VPURT.Task waits(%bar3: !VPURT.Barrier) updates(%bar4: !VPURT.Barrier) {
        %0 = VPUIP.NCEClusterTiling inputs(%buf11 as %arg0: !Output_DDR) outputs(%buf6 as %arg1: !OutputStub_CMX) -> !OutputDistributed {
            VPUIP.NNDMA
            inputs(%arg0: !Output_DDR)
            outputs(%arg1: !OutputStub_CMX)
            -> !OutputStub_CMX
        }
    }

    // Upload 2st input tile

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar5: !VPURT.Barrier) {
        %0 = VPUIP.NCEClusterTiling inputs(%buf2 as %arg0: !Input_DDR) outputs(%buf12 as %arg1: !InputStub_CMX) -> !InputDistributed {
            VPUIP.NNDMA
            inputs(%arg0: !Input_DDR)
            outputs(%arg1: !InputStub_CMX)
            -> !InputStub_CMX
        }
    }

    // 2nd tile

    VPURT.Task waits(%bar0, %bar5: !VPURT.Barrier, !VPURT.Barrier) updates(%bar6: !VPURT.Barrier) {
        %0 = VPUIP.NCEClusterTiling
                    inputs(%buf12 as %arg0: !InputStub_CMX,
                            %buf18 as %arg1: !WeightsStub_CMX,
                            %buf19 as %arg2: !WeightsTableStub_CMX)
                    outputs(%buf13 as %arg3: !OutputStub_CMX)
                        -> !OutputStub_CMX {

            VPUIP.NCEClusterTask {
                    kernel_padding = {bottom = 0, left = 0, right = 0, top = 0},
                    kernel_size = [1, 1],
                    kernel_strides = [1, 1],
                    task_type = "CONV"
                }
                input(%arg0: !InputStub_CMX)
                weights(%arg1: !WeightsStub_CMX)
                weight_table(%arg2: !WeightsTableStub_CMX)
                parent_input(%arg0: !InputStub_CMX)
                parent_output(%arg3: !OutputStub_CMX)
                outputs(%arg3: !OutputStub_CMX)
                -> memref<1x16x8x32xf16, #NHWC, @CMX_NN>
                variants : {
                    DPUTask {
                        start = [0, 0, 0],
                        end = [31, 7, 15],
                        pad = {bottom = 0, left = 0, right = 0, top = 0},
                        mpe_mode = "VECTOR_FP16"
                    }
                } PPE : {
                }
        }
    }

    // Copyback 2nd result tile

    VPURT.Task waits(%bar6: !VPURT.Barrier) updates(%bar7: !VPURT.Barrier) {
        %0 = VPUIP.NCEClusterTiling inputs(%buf13 as %arg0: !Output_DDR) outputs(%buf7 as %arg1: !OutputStub_CMX) -> !OutputDistributed {
            VPUIP.NNDMA
            inputs(%arg0: !Output_DDR)
            outputs(%arg1: !OutputStub_CMX)
            -> !OutputStub_CMX
        }
    }

    // Upload 3st input tile

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar8: !VPURT.Barrier) {
        %0 = VPUIP.NCEClusterTiling inputs(%buf3 as %arg0: !Input_DDR) outputs(%buf14 as %arg1: !InputStub_CMX) -> !InputDistributed {
            VPUIP.NNDMA
            inputs(%arg0: !Input_DDR)
            outputs(%arg1: !InputStub_CMX)
            -> !InputStub_CMX
        }
    }

    // 3rd tile

    VPURT.Task waits(%bar0, %bar8: !VPURT.Barrier, !VPURT.Barrier) updates(%bar9: !VPURT.Barrier) {
        %0 = VPUIP.NCEClusterTiling
                    inputs(%buf14 as %arg0: !InputStub_CMX,
                            %buf18 as %arg1: !WeightsStub_CMX,
                            %buf19 as %arg2: !WeightsTableStub_CMX)
                    outputs(%buf15 as %arg3: !OutputStub_CMX)
                        -> !OutputStub_CMX {

            VPUIP.NCEClusterTask {
                    kernel_padding = {bottom = 0, left = 0, right = 0, top = 0},
                    kernel_size = [1, 1],
                    kernel_strides = [1, 1],
                    task_type = "CONV"
                }
                input(%arg0: !InputStub_CMX)
                weights(%arg1: !WeightsStub_CMX)
                weight_table(%arg2: !WeightsTableStub_CMX)
                parent_input(%arg0: !InputStub_CMX)
                parent_output(%arg3: !OutputStub_CMX)
                outputs(%arg3: !OutputStub_CMX)
                -> memref<1x16x8x32xf16, #NHWC, @CMX_NN>
                variants : {
                    DPUTask {
                        start = [0, 0, 0],
                        end = [31, 7, 15],
                        pad = {bottom = 0, left = 0, right = 0, top = 0},
                        mpe_mode = "VECTOR_FP16"
                    }
                } PPE : {
                }
        }
    }

    // Copyback 3rd result tile

    VPURT.Task waits(%bar9: !VPURT.Barrier) updates(%bar10: !VPURT.Barrier) {
        %0 = VPUIP.NCEClusterTiling inputs(%buf15 as %arg0: !Output_DDR) outputs(%buf8 as %arg1: !OutputStub_CMX) -> !OutputDistributed {
            VPUIP.NNDMA
            inputs(%arg0: !Output_DDR)
            outputs(%arg1: !OutputStub_CMX)
            -> !OutputStub_CMX
        }
    }

    // Upload 4st input tile

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar11: !VPURT.Barrier) {
        %0 = VPUIP.NCEClusterTiling inputs(%buf4 as %arg0: !Input_DDR) outputs(%buf16 as %arg1: !InputStub_CMX) -> !InputDistributed {
            VPUIP.NNDMA
            inputs(%arg0: !Input_DDR)
            outputs(%arg1: !InputStub_CMX)
            -> !InputStub_CMX
        }
    }

    // 4th tile

    VPURT.Task waits(%bar0, %bar11: !VPURT.Barrier, !VPURT.Barrier) updates(%bar12: !VPURT.Barrier) {
        %0 = VPUIP.NCEClusterTiling
                    inputs(%buf16 as %arg0: !InputStub_CMX,
                            %buf18 as %arg1: !WeightsStub_CMX,
                            %buf19 as %arg2: !WeightsTableStub_CMX)
                    outputs(%buf17 as %arg3: !OutputStub_CMX)
                        -> !OutputStub_CMX {

            VPUIP.NCEClusterTask {
                    kernel_padding = {bottom = 0, left = 0, right = 0, top = 0},
                    kernel_size = [1, 1],
                    kernel_strides = [1, 1],
                    task_type = "CONV"
                }
                input(%arg0: !InputStub_CMX)
                weights(%arg1: !WeightsStub_CMX)
                weight_table(%arg2: !WeightsTableStub_CMX)
                parent_input(%arg0: !InputStub_CMX)
                parent_output(%arg3: !OutputStub_CMX)
                outputs(%arg3: !OutputStub_CMX)
                -> memref<1x16x8x32xf16, #NHWC, @CMX_NN>
                variants : {
                    DPUTask {
                        start = [0, 0, 0],
                        end = [31, 7, 15],
                        pad = {bottom = 0, left = 0, right = 0, top = 0},
                        mpe_mode = "VECTOR_FP16"
                    }
                } PPE : {
                }
        }
    }

    // Copyback 4th result tile

    VPURT.Task waits(%bar12: !VPURT.Barrier) updates(%bar13: !VPURT.Barrier) {
        %0 = VPUIP.NCEClusterTiling inputs(%buf17 as %arg0: !Output_DDR) outputs(%buf9 as %arg1: !OutputStub_CMX) -> !OutputDistributed {
            VPUIP.NNDMA
            inputs(%arg0: !Output_DDR)
            outputs(%arg1: !OutputStub_CMX)
            -> !OutputStub_CMX
        }
    }

    // Reorder output

    VPURT.Task waits(%bar4, %bar7, %bar10, %bar13: !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier) {
        VPUIP.PermuteUPA {order_value = #NCHW}
            inputs(%buf5: memref<1x16x32x32xf16, #NHWC, @DDR>)
            outputs(%output: memref<1x16x32x32xf16, @DDR>)
            -> memref<1x16x32x32xf16, @DDR>
    }

    return %output : memref<1x16x32x32xf16, @DDR>

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR3:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR4:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR5:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR6:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR7:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR8:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR9:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR10:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR11:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR12:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR13:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR14:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       VPURT.Task
    // CHECK-SAME:       updates([[BAR0]] : !VPURT.Barrier)
    // CHECK:       VPURT.Task
    // CHECK-SAME:       updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:       VPURT.Task
    // CHECK-SAME:       waits([[BAR0]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)
    // CHECK:       VPURT.Task
    // CHECK-SAME:       waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)
    // CHECK:       VPURT.Task
    // CHECK-SAME:       waits([[BAR1]], [[BAR2]] : !VPURT.Barrier, !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier)
    // CHECK:       VPURT.Task
    // CHECK-SAME:       waits([[BAR3]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier)
    // CHECK:       VPURT.Task
    // CHECK-SAME:       waits([[BAR4]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier)
    // CHECK:       VPURT.Task
    // CHECK-SAME:       waits([[BAR1]], [[BAR5]] : !VPURT.Barrier, !VPURT.Barrier) updates([[BAR7]] : !VPURT.Barrier)
    // CHECK:       VPURT.Task
    // CHECK-SAME:       waits([[BAR6]] : !VPURT.Barrier) updates([[BAR8]] : !VPURT.Barrier)
    // CHECK:       VPURT.Task
    // CHECK-SAME:       waits([[BAR7]] : !VPURT.Barrier) updates([[BAR9]] : !VPURT.Barrier)
    // CHECK:       VPURT.Task
    // CHECK-SAME:       waits([[BAR8]] : !VPURT.Barrier) updates([[BAR10]] : !VPURT.Barrier)
    // CHECK:       VPURT.Task
    // CHECK-SAME:       waits([[BAR9]] : !VPURT.Barrier) updates([[BAR11]] : !VPURT.Barrier)
    // CHECK:       VPURT.Task
    // CHECK-SAME:       waits([[BAR10]] : !VPURT.Barrier) updates([[BAR12]] : !VPURT.Barrier)
    // CHECK:       VPURT.Task
    // CHECK-SAME:       waits([[BAR11]] : !VPURT.Barrier) updates([[BAR13]] : !VPURT.Barrier)
    // CHECK:       VPURT.Task
    // CHECK-SAME:       waits([[BAR12]] : !VPURT.Barrier) updates([[BAR14]] : !VPURT.Barrier)
    // CHECK:       VPURT.Task
    // CHECK-SAME:       waits([[BAR13]], [[BAR14]] : !VPURT.Barrier, !VPURT.Barrier)
}
