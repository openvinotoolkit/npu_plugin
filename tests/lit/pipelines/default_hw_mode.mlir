// RUN: vpux-opt --split-input-file --default-hw-mode="vpu-arch=KMB" %s | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ExpandTest
module @ExpandTest {

// CHECK:       VPUIP.Graph
// CHECK-SAME:      options : "NONE"

// CHECK:   IERT.RunTimeResources
// CHECK:       usedMemory
// CHECK:           MemoryResource 256 bytes of "DDR"

// CHECK: IE.CNNNetwork
IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        // CHECK: DataInfo "input" : tensor<1x3x4x4xf16>
        DataInfo "input" : tensor<1x3x4x4xf16>
    }
    outputsInfo : {
        // CHECK: DataInfo "output" : tensor<1x8x4x4xf16>
        DataInfo "output" : tensor<1x8x4x4xf16>
    }

// CHECK:       func @main(
// CHECK-SAME:      [[ARG0:%.+]]: memref<1x3x4x4xf16>,
// CHECK-SAME:      [[ARG1:%.+]]: memref<1x8x4x4xf16>) -> memref<1x8x4x4xf16> {
func @main(%arg0: tensor<1x3x4x4xf16>) -> tensor<1x8x4x4xf16> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 5, 0, 0]} : tensor<1x3x4x4xf16> -> tensor<1x8x4x4xf16>
    return %0 : tensor<1x8x4x4xf16>

    // CHECK-NEXT:  [[VAR0:%.+]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0>
    // CHECK-NEXT:  [[VAR1:%.+]] = VPUIP.ConfigureBarrier {virtualId = 0 : i64}<0>
    // CHECK-NEXT:  [[VAR2:%.+]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0>
    // CHECK-NEXT:  [[VAR3:%.+]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <96>
    // CHECK-NEXT:  [[VAR4:%.+]] = VPUIP.DeclareTensor "ProgrammableInput" [0] <0>
    // CHECK-NEXT:  [[VAR5:%.+]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <192>

    // CHECK-NEXT:  [[VAR6:%.+]] = VPUIP.NNDMA
    // CHECK-SAME:                  inputs([[ARG0]] : memref<1x3x4x4xf16>)
    // CHECK-SAME:                  outputs([[VAR2]] : memref<1x3x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}, "DDR">)
    // CHECK-SAME:                  updates([[VAR1]] : !VPUIP.Barrier)
    
    // CHECK-NEXT:  [[VAR7:%.+]] = VPUIP.NNDMA
    // CHECK-SAME:                  inputs([[ARG0]] : memref<1x3x4x4xf16>)
    // CHECK-SAME:                  outputs([[VAR3]] : memref<1x3x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}, "DDR">)
    // CHECK-SAME:                  updates([[VAR1]] : !VPUIP.Barrier)
    
    // CHECK-NEXT:  [[VAR8:%.+]] = VPUIP.NNDMA 
    // CHECK-SAME:                  inputs([[VAR4]] : memref<1x2x4x4xf16, {order = #NCHW, strides = [48, 16, 4, 1]}>)
    // CHECK-SAME:                  outputs([[VAR5]] : memref<1x2x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}, "DDR">)
    // CHECK-SAME:                  updates([[VAR1]] : !VPUIP.Barrier)
    
    // CHECK-NEXT:  [[VAR9:%.+]] = VPUIP.NNDMA
    // CHECK-SAME:                  inputs([[VAR0]] : memref<1x8x4x4xf16, "DDR">)
    // CHECK-SAME:                  outputs([[ARG1]] : memref<1x8x4x4xf16>)
    // CHECK-SAME:                  waits([[VAR1]] : !VPUIP.Barrier)

    // CHECK-NEXT:  return [[VAR9]] : memref<1x8x4x4xf16>
}

}

// -----// IR Dump After ConvertPrecisionToFP16 //----- //

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
!qElemType0 = type !quant.uniform<u8<0:254>:f16, 0.003937007874015748>
!qElemType1 = type !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @KmbQuantizedConv
module @KmbQuantizedConv {
    // CHECK:       VPUIP.Graph
    // CHECK-SAME:      options : "NONE"

    // CHECK:   IERT.RunTimeResources
    // CHECK:       usedMemory
    // CHECK:           MemoryResource 1382400 bytes of "DDR"

    // CHECK: IE.CNNNetwork
    IE.CNNNetwork entryPoint : @main inputsInfo :  {
        // CHECK: DataInfo "Parameter_994" : tensor<1x3x62x62xf32>
        DataInfo "Parameter_994" : tensor<1x3x62x62xf32>
    } outputsInfo :  {
        // CHECK: DataInfo "FakeQuantize_1012" : tensor<1x48x60x60xf32>
        DataInfo "FakeQuantize_1012" : tensor<1x48x60x60xf32>
    }

    // CHECK:       func @main(
    // CHECK-SAME:      [[ARG0:%.+]]: memref<1x3x62x62xf32>,
    // CHECK-SAME:      [[ARG1:%.+]]: memref<1x48x60x60xf32>) -> memref<1x48x60x60xf32> {
    func @main(%arg0: tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16> {
        %0 = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf32>
        %cst = const.Declare tensor<f32> = #const.Content<dense<2.550000e+02> : tensor<f32>>
        %cst_0 = const.Declare tensor<f32> = #const.Content<dense<0.000000e+00> : tensor<f32>>
        %cst_1 = const.Declare tensor<48x1x1x1xf32> = #const.Content<dense<1.000000e+00> : tensor<48x1x1x1xf32>>
        %cst_2 = const.Declare tensor<48x1x1x1xf32> = #const.Content<dense<0.000000e+00> : tensor<48x1x1x1xf32>>
        %cst_3 = const.Declare tensor<1xf32> = #const.Content<dense<2.540000e+02> : tensor<1xf32>>
        %cst_4 = const.Declare tensor<1xf32> = #const.Content<dense<0.000000e+00> : tensor<1xf32>>
        %cst_5 = const.Declare tensor<48x3x3x3xf32> = #const.Content<dense<0.000000e+00> : tensor<48x3x3x3xf32>>
        %1 = IE.Convert(%0) {dstElemType = f16} : tensor<1x3x62x62xf32> -> tensor<1x3x62x62xf16>
        %cst_6 = const.Declare tensor<f16> = #const.Content<dense<0.000000e+00> : tensor<f32>, [#const.ConvertElemType<f16>]>
        %cst_7 = const.Declare tensor<f16> = #const.Content<dense<2.550000e+02> : tensor<f32>, [#const.ConvertElemType<f16>]>
        %cst_8 = const.Declare tensor<f16> = #const.Content<dense<0.000000e+00> : tensor<f32>, [#const.ConvertElemType<f16>]>
        %cst_9 = const.Declare tensor<f16> = #const.Content<dense<2.550000e+02> : tensor<f32>, [#const.ConvertElemType<f16>]>
        %2 = IE.FakeQuantize(%1, %cst_8, %cst_9, %cst_8, %cst_9) {auto_broadcast = "NUMPY", levels = 256 : i64} : tensor<1x3x62x62xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<1x3x62x62xf16>
        %cst_10 = const.Declare tensor<48x3x3x3xf16> = #const.Content<dense<0.000000e+00> : tensor<48x3x3x3xf32>, [#const.ConvertElemType<f16>]>
        %cst_11 = const.Declare tensor<1xf16> = #const.Content<dense<0.000000e+00> : tensor<1xf32>, [#const.ConvertElemType<f16>]>
        %cst_12 = const.Declare tensor<1xf16> = #const.Content<dense<2.540000e+02> : tensor<1xf32>, [#const.ConvertElemType<f16>]>
        %cst_13 = const.Declare tensor<48x1x1x1xf16> = #const.Content<dense<0.000000e+00> : tensor<48x1x1x1xf32>, [#const.ConvertElemType<f16>]>
        %cst_14 = const.Declare tensor<48x1x1x1xf16> = #const.Content<dense<1.000000e+00> : tensor<48x1x1x1xf32>, [#const.ConvertElemType<f16>]>
        %3 = IE.FakeQuantize(%cst_10, %cst_11, %cst_12, %cst_13, %cst_14) {auto_broadcast = "NUMPY", levels = 255 : i64} : tensor<48x3x3x3xf16>, tensor<1xf16>, tensor<1xf16>, tensor<48x1x1x1xf16>, tensor<48x1x1x1xf16> -> tensor<48x3x3x3xf16>
        %4 = IE.Convolution(%2, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf16>, tensor<48x3x3x3xf16> -> tensor<1x48x60x60xf16>
        %cst_15 = const.Declare tensor<f16> = #const.Content<dense<0.000000e+00> : tensor<f32>, [#const.ConvertElemType<f16>]>
        %cst_16 = const.Declare tensor<f16> = #const.Content<dense<2.550000e+02> : tensor<f32>, [#const.ConvertElemType<f16>]>
        %cst_17 = const.Declare tensor<f16> = #const.Content<dense<0.000000e+00> : tensor<f32>, [#const.ConvertElemType<f16>]>
        %cst_18 = const.Declare tensor<f16> = #const.Content<dense<2.550000e+02> : tensor<f32>, [#const.ConvertElemType<f16>]>
        %5 = IE.FakeQuantize(%4, %cst_17, %cst_18, %cst_17, %cst_18) {auto_broadcast = "NUMPY", levels = 256 : i64} : tensor<1x48x60x60xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<1x48x60x60xf16>
        return %5 : tensor<1x48x60x60xf16>

        // CHECK-DAG:   %[[CST:.*]] = const.Declare memref<48x16x3x3x!qElemType0, #NHWC>
        // CHECK-DAG:   %[[CST0:.*]] = const.Declare memref<48x1x1x4xsi32>
       
        // CHECK:       [[VAR1:%.+]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <23104>
        // CHECK-NEXT:  [[VAR2:%.+]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <146112>
        // CHECK-NEXT:  [[VAR3:%.+]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0>
        // CHECK-NEXT:  [[VAR4:%.+]] = VPUIP.DeclareTensor "VPU_CMX_NN" [0] <0>
        // CHECK-NEXT:  [[VAR5:%.+]] = VPUIP.DeclareTensor "VPU_CMX_NN" [0] <123008>
        // CHECK-NEXT:  [[VAR6:%.+]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0>
        // CHECK-NEXT:  [[VAR7:%.+]] = VPUIP.DeclareTensor "VPU_CMX_NN" [0] <172800>
        // CHECK-NEXT:  [[VAR8:%.+]] = VPUIP.DeclareTensor "VPU_CMX_NN" [0] <234304>
        // CHECK-NEXT:  [[VAR9:%.+]] = VPUIP.DeclareTensor "VPU_CMX_NN" [0] <0>
        // CHECK-NEXT:  [[VAR10:%.+]] = VPUIP.DeclareTensor "VPU_CMX_NN" [0] <241216>
        // CHECK-NEXT:  [[VAR11:%.+]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0>
        // CHECK-NEXT:  [[VAR12:%.+]] = VPUIP.DeclareTensor "VPU_CMX_NN" [0] <345600>
        // CHECK-NEXT:  [[VAR13:%.+]] = VPUIP.DeclareTensor "VPU_CMX_NN" [0] <0>
        // CHECK-NEXT:  [[VAR14:%.+]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <345600>
        // CHECK:       [[VAR15:%.+]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <691200>
        // CHECK:       [[VAR36:%.+]] = VPUIP.ConfigureBarrier {virtualId = 13 : i64}<13>

        // CHECK:       [[VAR44:%.+]] = VPUIP.PermuteUPA
        // CHECK-SAME:                  inputs([[VAR1]] : memref<1x16x62x62xf16, "DDR">)
        // CHECK-SAME:                  outputs([[VAR2]] : memref<1x16x62x62xf16, #NHWC, "DDR">)

        // CHECK-NEXT:  [[VAR45:%.+]] = VPUIP.NNDMA
        // CHECK-SAME:                  inputs([[VAR44]] : memref<1x16x62x62xf16, #NHWC, "DDR">)
        // CHECK-SAME:                  outputs([[VAR4]] : memref<1x16x62x62xf16, #NHWC, "CMX_NN">)

        // CHECK-NEXT:  [[VAR46:%.+]] = VPUIP.NCEClusterTask 
        // CHECK-SAME:                  input([[VAR45]] : memref<1x16x62x62xf16, #NHWC, "CMX_NN">)
        // CHECK-SAME:                  weights([[VAR45]] : memref<1x16x62x62xf16, #NHWC, "CMX_NN">)
        // CHECK-SAME:                  outputs([[VAR5]] : memref<1x16x62x62x!qElemType1, #NHWC, "CMX_NN">)
        // CHECK:                       PPETask "AND"

        // CHECK:       [[VAR47:%.+]] = VPUIP.NNDMA
        // CHECK-SAME:                  inputs([[VAR46]] : memref<1x16x62x62x!qElemType1, #NHWC, "CMX_NN">)
        // CHECK-SAME:                  outputs([[VAR3]] : memref<1x16x62x62x!qElemType1, #NHWC, "DDR">)
        // CHECK-NEXT:  [[VAR48:%.+]] = VPUIP.NNDMA
        // CHECK-SAME:                  inputs([[VAR47]] : memref<1x16x62x62x!qElemType1, #NHWC, "DDR">)
        // CHECK-SAME:                  outputs([[VAR7]] : memref<1x16x62x62x!qElemType1, #NHWC, "CMX_NN">)
        // CHECK-NEXT:  [[VAR49:%.+]] = VPUIP.NNDMA
        // CHECK-SAME:                  inputs(%[[CST0]] : memref<48x1x1x4xsi32>)
        // CHECK-SAME:                  outputs([[VAR10]] : memref<48x1x1x4xsi32, "CMX_NN">)
        // CHECK-NEXT:  [[VAR50:%.+]] = VPUIP.NNDMA
        // CHECK-SAME:                  inputs(%[[CST]] : memref<48x16x3x3x!qElemType0, #NHWC>)
        // CHECK-SAME:                  outputs([[VAR8]] : memref<48x16x3x3x!qElemType0, #NHWC, "CMX_NN">)

        // CHECK-NEXT:  [[VAR51:%.+]] = VPUIP.NCEClusterTask
        // CHECK-SAME:                  input([[VAR48]] : memref<1x16x62x62x!qElemType1, #NHWC, "CMX_NN">)
        // CHECK-SAME:                  weights([[VAR50]] : memref<48x16x3x3x!qElemType0, #NHWC, "CMX_NN">)
        // CHECK-SAME:                  weight_table([[VAR49]] : memref<48x1x1x4xsi32, "CMX_NN">)
        // CHECK-SAME:                  outputs([[VAR9]] : memref<1x48x60x60x!qElemType1, #NHWC, "CMX_NN">)

        // CHECK:       [[VAR52:%.+]] = VPUIP.NNDMA
        // CHECK-SAME:                  inputs([[VAR51]] : memref<1x48x60x60x!qElemType1, #NHWC, "CMX_NN">)
        // CHECK-SAME:                  outputs([[VAR6]] : memref<1x48x60x60x!qElemType1, #NHWC, "DDR">)
        // CHECK-NEXT:  [[VAR53:%.+]] = VPUIP.NNDMA
        // CHECK-SAME:                  inputs([[VAR52]] : memref<1x48x60x60x!qElemType1, #NHWC, "DDR">)
        // CHECK-SAME:                  outputs([[VAR12]] : memref<1x48x60x60x!qElemType1, #NHWC, "CMX_NN">)
        // CHECK-NEXT:  [[VAR54:%.+]] = VPUIP.NCEClusterTask
        // CHECK-SAME:                  input([[VAR53]] : memref<1x48x60x60x!qElemType1, #NHWC, "CMX_NN">)
        // CHECK-SAME:                  weights([[VAR53]] : memref<1x48x60x60x!qElemType1, #NHWC, "CMX_NN">)
        // CHECK-SAME:                  outputs([[VAR13]] : memref<1x48x60x60xf16, #NHWC, "CMX_NN">)
        // CHECK:                       PPETask "AND"
        
        // CHECK:       [[VAR55:%.+]] = VPUIP.NNDMA
        // CHECK-SAME:                  inputs([[VAR54]] : memref<1x48x60x60xf16, #NHWC, "CMX_NN">)
        // CHECK-SAME:                  outputs([[VAR11]] : memref<1x48x60x60xf16, #NHWC, "DDR">)

        // CHECK-NEXT:  [[VAR56:%.+]] = VPUIP.PermuteUPA
        // CHECK-SAME:                  inputs([[VAR55]] : memref<1x48x60x60xf16, #NHWC, "DDR">)
        // CHECK-SAME:                  outputs([[VAR14]] : memref<1x48x60x60xf16, "DDR">)

        // CHECK-NEXT:  [[VAR57:%.+]] = VPUIP.ConvertUPA
        // CHECK-SAME:                  inputs([[VAR56]] : memref<1x48x60x60xf16, "DDR">)
        // CHECK-SAME:                  outputs([[VAR15]] : memref<1x48x60x60xf32, "DDR">)

        // CHECK-NEXT:  [[VAR58:%.+]] = VPUIP.NNDMA
        // CHECK-SAME:                  inputs([[VAR57]] : memref<1x48x60x60xf32, "DDR">)
        // CHECK-SAME:                  outputs([[ARG1]] : memref<1x48x60x60xf32>)
        // CHECK-SAME:                  waits([[VAR36]] : !VPUIP.Barrier)

        // CHECK-NEXT:  return [[VAR58]] : memref<1x48x60x60xf32>
  }
}
