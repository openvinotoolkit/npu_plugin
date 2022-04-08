//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --mlir-elide-elementsattrs-if-larger 8  --default-hw-mode="vpu-arch=VPUX37XX" %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Convolution
module @Convolution {
    // CHECK:   module @UsedMemory
    // CHECK-DAG:       IE.MemoryResource {{[0-9]+}} bytes of @CMX_NN
    // CHECK-DAG:       IE.MemoryResource {{[0-9]+}} bytes of @DDR

    // CHECK-DAG:       IE.ExecutorResource 2 of @DMA_NN
    // CHECK-DAG:       IE.ExecutorResource 2 of @SHAVE_ACT
    // CHECK-DAG:       IE.ExecutorResource 1 of @SHAVE_NN
    // CHECK-DAG:       IE.ExecutorResource 2 of @NCE at 1.300000e+03 MHz
    // CHECK-DAG:           IE.ExecutorResource 1 of @DPU

    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xf16, {order = #NHWC}>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf16, {order = #NHWC}>
    }

    // CHECK:       func @main(
    // CHECK-SAME:      [[ARG0:%.+]]: memref<1x3x62x62xf16, #NHWC, @DDR>,
    // CHECK-SAME:      [[ARG1:%.+]]: memref<1x48x60x60xf16, #NHWC, @DDR>) -> memref<1x48x60x60xf16, #NHWC, @DDR> {
    func @main(%arg: tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32> {
        %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.0> : tensor<48x3x3x3xf32>
        %1 = IE.Convolution(%arg, %cst) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
        return %1 : tensor<1x48x60x60xf32>

        // CHECK-DAG:   const.Declare memref<48x3x3x3xf16, #NHWC>
        // CHECK-DAG:   const.Declare memref<48x1x1x4xsi32>

        // CHECK:       VPURT.Task waits([[barrier_0:%.*]] : !VPURT.Barrier) updates([[barrier_1:%.*]] : !VPURT.Barrier)
        // CHECK:       VPUIP.NCEClusterTask
        // CHECK-SAME:          task_type = "CONV"
        // CHECK-SAME:      [[input_0:%.*]] : memref<1x16x32x62xf16, #NHWC, [@CMX_NN, 0]>)
        // CHECK-SAME:      [[weight_0:%.*]] : memref<48x16x3x3xf16, #NHWC, [@CMX_NN, 0]>)
        // CHECK-SAME:      [[weight_table_0:%.*]] : memref<48x1x1x4xsi32, [@CMX_NN, 0]>)
        // CHECK-SAME:      [[parent_input:%.*]] : !VPUIP.DistributedBuffer<1x16x62x62xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>)
        // CHECK-SAME:      [[parent_output:%.*]] : !VPUIP.DistributedBuffer<1x48x60x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        // CHECK-SAME:      [[output_0:%.*]] : memref<1x48x30x60xf16, #NHWC, [@CMX_NN, 0]>)
        // CHECK:               DPUTask

        // CHECK:       VPURT.Task waits([[barrier_0:%.*]] : !VPURT.Barrier) updates([[barrier_1:%.*]] : !VPURT.Barrier)
        // CHECK:       VPUIP.NCEClusterTask
        // CHECK-SAME:          task_type = "CONV"
        // CHECK-SAME:      [[input_0:%.*]] : memref<1x16x30x62xf16, #NHWC, [@CMX_NN, 1]>)
        // CHECK-SAME:      [[weight_0:%.*]] : memref<48x16x3x3xf16, #NHWC, [@CMX_NN, 1]>)
        // CHECK-SAME:      [[weight_table_0:%.*]] : memref<48x1x1x4xsi32, [@CMX_NN, 1]>)
        // CHECK-SAME:      [[parent_input:%.*]] : !VPUIP.DistributedBuffer<1x16x62x62xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>)
        // CHECK-SAME:      [[parent_output:%.*]] : !VPUIP.DistributedBuffer<1x48x60x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        // CHECK-SAME:      [[output_0:%.*]] : memref<1x48x30x60xf16, #NHWC, [@CMX_NN, 1]>)
        // CHECK:               DPUTask
  }
}
