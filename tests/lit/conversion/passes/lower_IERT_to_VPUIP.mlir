// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=MA2490" --lower-IERT-to-VPUIP %s | FileCheck %s

// CHECK-LABEL: @SingleLayer
module @SingleLayer {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "input" : memref<1x1x1x1000xf16>
    }
    outputsInfo : {
        IE.DataInfo "output" : memref<1x1x1x1000xf16>
    }

func @main(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) {
    IERT.SoftMax(%arg0, %arg1) {axisInd = 3 : i32} : memref<1x1x1x1000xf16>, memref<1x1x1x1000xf16>
    return

    // CHECK:       VPUIP.SoftMaxUPA
    // CHECK-SAME:      axisInd = 3
    // CHECK-SAME:      inputs(%arg0 : memref<1x1x1x1000xf16>)
    // CHECK-SAME:      outputs(%arg1 : memref<1x1x1x1000xf16>)
}

}

// -----

// CHECK-LABEL: @ConstantLayer
module @ConstantLayer {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
    }
    outputsInfo :  {
        IE.DataInfo "output" : memref<1x2x2x2xf16>
    }

func @main(%arg0: memref<1x2x2x2xf16>) {
    %0 = IERT.Constant memref<1x2x2x2xf16> = dense<1.0> : tensor<1x2x2x2xf16>
    linalg.copy(%0, %arg0) : memref<1x2x2x2xf16>, memref<1x2x2x2xf16>
    return

    // CHECK:       [[VAR0:%.*]] = VPUIP.DeclareConstantTensor memref<1x2x2x2xf16>
    // CHECK-SAME:      = dense<1.000000e+00> : tensor<1x2x2x2xf16>

    // CHECK:       VPUIP.NNDMA
    // CHECK-SAME:      inputs([[VAR0]] : memref<1x2x2x2xf16>)
    // CHECK-SAME:      outputs(%arg0 : memref<1x2x2x2xf16>)
}

}

// -----

// CHECK-LABEL: @StaticAlloc
module @StaticAlloc {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "input" : memref<1x1x1x1000xf16>
    }
    outputsInfo : {
        IE.DataInfo "output" : memref<1x1x1x1000xf16>
    }

func @main(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) {
    %0 = IERT.StaticAlloc<0> -> memref<1x1x1x1000xf16, "DDR">
    IERT.SoftMax(%arg0, %0) {axisInd = 3 : i32} : memref<1x1x1x1000xf16>, memref<1x1x1x1000xf16, "DDR">

    %1 = IERT.StaticAlloc<2048> -> memref<1x1x1x1000xf16, "DDR">
    IERT.SoftMax(%0, %1) {axisInd = 3 : i32} : memref<1x1x1x1000xf16, "DDR">, memref<1x1x1x1000xf16, "DDR">

    IERT.SoftMax(%1, %arg1) {axisInd = 3 : i32} : memref<1x1x1x1000xf16, "DDR">, memref<1x1x1x1000xf16>

    return

    // CHECK:       [[VAR0:%.*]] = VPUIP.DeclareTensor "VPU_DDR_Heap" <0> -> memref<1x1x1x1000xf16, "DDR">

    // CHECK:       VPUIP.SoftMaxUPA
    // CHECK-SAME:      axisInd = 3
    // CHECK-SAME:      inputs(%arg0 : memref<1x1x1x1000xf16>)
    // CHECK-SAME:      outputs([[VAR0]] : memref<1x1x1x1000xf16, "DDR">)

    // CHECK:       [[VAR1:%.*]] = VPUIP.DeclareTensor "VPU_DDR_Heap" <2048> -> memref<1x1x1x1000xf16, "DDR">

    // CHECK:       VPUIP.SoftMaxUPA
    // CHECK-SAME:      axisInd = 3
    // CHECK-SAME:      inputs([[VAR0]] : memref<1x1x1x1000xf16, "DDR">)
    // CHECK-SAME:      outputs([[VAR1]] : memref<1x1x1x1000xf16, "DDR">)

    // CHECK:       VPUIP.SoftMaxUPA
    // CHECK-SAME:      axisInd = 3
    // CHECK-SAME:      inputs([[VAR1]] : memref<1x1x1x1000xf16, "DDR">)
    // CHECK-SAME:      outputs(%arg1 : memref<1x1x1x1000xf16>)
}

}

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>

// CHECK-LABEL: @ReshapeInGraph
module @ReshapeInGraph {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "input" : memref<1x512xf16>
    }
    outputsInfo : {
        IE.DataInfo "output" : memref<1x512xf16>
    }

func @main(%arg0: memref<1x512xf16>, %arg1: memref<1x512xf16>) {
    %0 = linalg.reshape %arg0 [#map0, #map1] : memref<1x512xf16> into memref<1x512x1x1xf16>
    %1 = IERT.StaticAlloc<0> -> memref<1x512x1x1xf16, "DDR">
    IERT.SoftMax(%0, %1) {axisInd = 1 : i32} : memref<1x512x1x1xf16>, memref<1x512x1x1xf16, "DDR">
    %2 = linalg.reshape %1 [#map0, #map1] : memref<1x512x1x1xf16, "DDR"> into memref<1x512xf16, "DDR">
    linalg.copy(%2, %arg1) : memref<1x512xf16, "DDR">, memref<1x512xf16>
    return

    // CHECK:       [[VAR0:%.*]] = VPUIP.DeclareTensor "ProgrammableInput" [0] <0> -> memref<1x512x1x1xf16>

    // CHECK:       [[VAR1:%.*]] = VPUIP.DeclareTensor "VPU_DDR_Heap" <0> -> memref<1x512x1x1xf16, "DDR">

    // CHECK:       VPUIP.SoftMaxUPA
    // CHECK-SAME:      axisInd = 1
    // CHECK-SAME:      inputs([[VAR0]] : memref<1x512x1x1xf16>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<1x512x1x1xf16, "DDR">)

    // CHECK:       [[VAR2:%.*]] = VPUIP.DeclareTensor "VPU_DDR_Heap" <0> -> memref<1x512xf16, "DDR">

    // CHECK:       VPUIP.NNDMA
    // CHECK-SAME:      inputs([[VAR2]] : memref<1x512xf16, "DDR">)
    // CHECK-SAME:      outputs(%arg1 : memref<1x512xf16>)
}

}
