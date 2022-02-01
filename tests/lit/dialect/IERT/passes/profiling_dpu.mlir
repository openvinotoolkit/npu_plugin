// RUN: vpux-opt --convert-IE-to-VPU-NCE --adjust-memory-space --split-NCE-ops-onto-workloads --lower-IE-to-IERT --convert-vpu-to-vpuip --dpu-profiling %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DpuProfiling
module @DpuProfiling attributes {VPU.arch = "KMB", VPU.compilationMode = "DefaultHW"}  {

IE.MemoryResource 524288000 bytes of @DDR {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64}
IE.MemoryResource 917504 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}

IE.ExecutorResource 1 of @DMA_NN
IE.ExecutorResource 16 of @SHAVE_UPA
IE.ExecutorResource {VPU.processorFrequency = 7.000000e+02 : f64} 4 of  @NCE {
    IE.ExecutorResource 5 of @DPU
}

IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x3x62x62xf16, {order = #NHWC}>
} outputsInfo :  {
    DataInfo "output" : tensor<1x48x60x60xf16, {order = #NHWC}>
} profilingOutputsInfo :  {
}

func @main(%arg0: tensor<1x3x62x62xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<48x16x3x3xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<48x3x3x3xf32>, [#const.ConvertElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>, #const.Reorder<#NHWC>]>
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x62x62xf16, {order = #NHWC}> -> tensor<1x16x62x62xf16, {order = #NHWC}>
    %1 = IE.Convolution(%0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x62x62xf16, {order = #NHWC}>, tensor<48x16x3x3xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16, {order = #NHWC}>
    return %1 : tensor<1x48x60x60xf16, {order = #NHWC}>
}

    //CHECK:        profilingOutputsInfo
    //CHECK-NEXT:   DataInfo "dpu" : tensor<2xui64>
    //CHECK:        func @main(%arg0: memref<1x3x62x62xf16, #NHWC>, %arg1: memref<1x48x60x60xf16, #NHWC>, %arg2: memref<2xui64>) -> (memref<1x48x60x60xf16, #NHWC>, memref<2xui64>)
    //CHECK:        [[VAR0:%.+]] = memref.alloc() : memref<2xui64, @CMX_NN>

    //CHECK:        [[VAR1:%.+]] = IERT.SubView [[VAR0]] [0] [2] : memref<2xui64, @CMX_NN>
    //CHECK:        [[VAR2:%[0-9]+]]:2 = VPUIP.NCEClusterTask
    //CHECK-SAME:   profiling_data([[VAR1]] : memref<2xui64, @CMX_NN>)

    //CHECK:        [[VAR3:%.*]] = IERT.SubView %arg2 [0] [2] : memref<2xui64>
    //CHECK:        [[VAR4:%.*]] = IERT.ConcatView inputs([[VAR2]]#1 : memref<2xui64, @CMX_NN>) outputs([[VAR0]] : memref<2xui64, @CMX_NN>)
    //CHECK:        [[VAR5:%.*]] = IERT.Copy inputs([[VAR4]] : memref<2xui64, @CMX_NN>) outputs([[VAR3]] : memref<2xui64>)
    //CHECK:        [[VAR6:%.*]] = IERT.ConcatView inputs([[VAR5]] : memref<2xui64>) outputs(%arg2 : memref<2xui64>)
    //CHECK:        return
    //CHECK-SAME:   [[VAR6]]
    //CHECK-SAME:   memref<2xui64>
}
