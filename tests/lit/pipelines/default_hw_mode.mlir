// RUN: vpux-opt --split-input-file --default-hw-mode="vpu-arch=KMB" %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Convolution
module @Convolution {
    // CHECK:   module @UsedMemory
    // CHECK-DAG:       IE.MemoryResource {{[0-9]+}} bytes of @CMX_NN
    // CHECK-DAG:       IE.MemoryResource {{[0-9]+}} bytes of @DDR

    // CHECK-DAG:       IE.ExecutorResource 1 of @DMA_NN
    // CHECK-DAG:       IE.ExecutorResource 16 of @SHAVE_UPA
    // CHECK-DAG:       IE.ExecutorResource {VPU.processorFrequency = 7.000000e+02 : f64} 4 of @NCE
    // CHECK-DAG:           IE.ExecutorResource 5 of @DPU

    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xf16, {order = #NHWC}>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf16, {order = #NHWC}>
    }

    // CHECK:       func @main(
    // CHECK-SAME:      [[ARG0:%.+]]: memref<1x3x62x62xf16>,
    // CHECK-SAME:      [[ARG1:%.+]]: memref<1x48x60x60xf16>) -> memref<1x48x60x60xf16> {
    func @main(%arg: tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32> {
        %cst = const.Declare tensor<48x3x3x3xf32> = #const.Content<dense<1.0> : tensor<48x3x3x3xf32>>
        %1 = IE.Convolution(%arg, %cst) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
        return %1 : tensor<1x48x60x60xf32>

        // CHECK-DAG:   const.Declare memref<48x16x3x3xf16, #NHWC>
        // CHECK-DAG:   const.Declare memref<48x1x1x4xsi32>

        // CHECK:       VPURT.Task
        // CHECK-SAME:      updates(%{{[0-9]+}} : !VPURT.Barrier)
        // CHECK:       VPUIP.NCEClusterTask
        // CHECK-SAME:          task_type = "CONV"
        // CHECK-SAME:      input(%{{[0-9]+}} : memref<1x16x62x62xf16, #NHWC, @CMX_NN>)
        // CHECK-SAME:      weights(%{{[0-9]+}} : memref<48x16x3x3xf16, #NHWC, @CMX_NN>)
        // CHECK-SAME:      weight_table(%{{[0-9]+}} : memref<48x1x1x4xsi32, @CMX_NN>)
        // CHECK-SAME:      parent_input(%{{[0-9]+}} : memref<1x16x62x62xf16, #NHWC, @CMX_NN>)
        // CHECK-SAME:      parent_output(%{{[0-9]+}} : memref<1x48x60x60xf16, #NHWC, @CMX_NN>)
        // CHECK-SAME:      outputs(%{{[0-9]+}} : memref<1x48x60x60xf16, #NHWC, @CMX_NN>)
        // CHECK:               DPUTask
  }
}
