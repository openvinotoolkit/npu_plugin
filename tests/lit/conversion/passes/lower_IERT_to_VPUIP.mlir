// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=VPU3400_A0" --lower-IERT-to-VPUIP %s | FileCheck %s

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

func @main(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    %0 = IERT.SoftMax {axisInd = 3 : i32} inputs(%arg0 : memref<1x1x1x1000xf16>) outputs(%arg1 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
    return %0: memref<1x1x1x1000xf16>

    // CHECK: [[VAR0:%.*]] = VPUIP.SoftMaxUPA
    // CHECK-SAME:      axisInd = 3
    // CHECK-SAME:      inputs(%arg0 : memref<1x1x1x1000xf16>)
    // CHECK-SAME:      outputs(%arg1 : memref<1x1x1x1000xf16>)

    // CHECK: return [[VAR0]] : memref<1x1x1x1000xf16>
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

func @main(%arg0: memref<1x2x2x2xf16>) -> memref<1x2x2x2xf16> {
    %0 = IERT.Constant memref<1x2x2x2xf16> = dense<1.0> : tensor<1x2x2x2xf16>
    %1 = IERT.Copy inputs(%0 : memref<1x2x2x2xf16>) outputs(%arg0 : memref<1x2x2x2xf16>) -> memref<1x2x2x2xf16>
    return %1: memref<1x2x2x2xf16>

    // CHECK:       [[VAR0:%.*]] = VPUIP.DeclareConstantTensor memref<1x2x2x2xf16>
    // CHECK-SAME:      = dense<1.000000e+00> : tensor<1x2x2x2xf16>

    // CHECK:       [[VAR1:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs([[VAR0]] : memref<1x2x2x2xf16>)
    // CHECK-SAME:      outputs(%arg0 : memref<1x2x2x2xf16>) -> memref<1x2x2x2xf16>
    // CHECK: return [[VAR1]] : memref<1x2x2x2xf16>
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

func @main(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    %0 = IERT.StaticAlloc<0> -> memref<1x1x1x1000xf16, "DDR">
    %1 = IERT.SoftMax {axisInd = 3 : i32} inputs(%arg0 : memref<1x1x1x1000xf16>) outputs(%0 : memref<1x1x1x1000xf16, "DDR">) -> memref<1x1x1x1000xf16, "DDR">

    %2 = IERT.StaticAlloc<2048> -> memref<1x1x1x1000xf16, "DDR">
    %3 = IERT.SoftMax {axisInd = 3 : i32} inputs(%1 : memref<1x1x1x1000xf16, "DDR">) outputs(%2 : memref<1x1x1x1000xf16, "DDR">) -> memref<1x1x1x1000xf16, "DDR">
    %4 = IERT.SoftMax {axisInd = 3 : i32} inputs(%3 : memref<1x1x1x1000xf16, "DDR">) outputs(%arg1 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>

    return %4: memref<1x1x1x1000xf16>

    // CHECK:       [[VAR0:%.*]] = VPUIP.DeclareTensor "VPU_DDR_Heap" <0> -> memref<1x1x1x1000xf16, "DDR">

    // CHECK:       [[VAR1:%.*]] = VPUIP.SoftMaxUPA
    // CHECK-SAME:      axisInd = 3
    // CHECK-SAME:      inputs(%arg0 : memref<1x1x1x1000xf16>)
    // CHECK-SAME:      outputs([[VAR0]] : memref<1x1x1x1000xf16, "DDR">)

    // CHECK:       [[VAR2:%.*]] = VPUIP.DeclareTensor "VPU_DDR_Heap" <2048> -> memref<1x1x1x1000xf16, "DDR">

    // CHECK:       [[VAR3:%.*]] = VPUIP.SoftMaxUPA
    // CHECK-SAME:      axisInd = 3
    // CHECK-SAME:      inputs([[VAR1]] : memref<1x1x1x1000xf16, "DDR">)
    // CHECK-SAME:      outputs([[VAR2]] : memref<1x1x1x1000xf16, "DDR">)

    // CHECK:       [[VAR4:%.*]] = VPUIP.SoftMaxUPA
    // CHECK-SAME:      axisInd = 3
    // CHECK-SAME:      inputs([[VAR3]] : memref<1x1x1x1000xf16, "DDR">)
    // CHECK-SAME:      outputs(%arg1 : memref<1x1x1x1000xf16>)

    // CHECK: return [[VAR4]] : memref<1x1x1x1000xf16>
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

func @main(%arg0: memref<1x512xf16>, %arg1: memref<1x512xf16>) -> memref<1x512xf16> {
    %0 = linalg.reshape %arg0 [#map0, #map1] : memref<1x512xf16> into memref<1x512x1x1xf16>
    %1 = IERT.StaticAlloc<0> -> memref<1x512x1x1xf16, "DDR">
    %2 = IERT.SoftMax {axisInd = 1 : i32} inputs(%0 : memref<1x512x1x1xf16>) outputs(%1 : memref<1x512x1x1xf16, "DDR">) -> memref<1x512x1x1xf16, "DDR">
    %3 = linalg.reshape %2 [#map0, #map1] : memref<1x512x1x1xf16, "DDR"> into memref<1x512xf16, "DDR">
    %4 = IERT.Copy inputs(%3 : memref<1x512xf16, "DDR">) outputs(%arg1 : memref<1x512xf16>) -> memref<1x512xf16>
    return %4: memref<1x512xf16>

    // CHECK:       [[VAR0:%.*]] = VPUIP.DeclareTensor "ProgrammableInput" [0] <0> -> memref<1x512x1x1xf16>

    // CHECK:       [[VAR1:%.*]] = VPUIP.DeclareTensor "VPU_DDR_Heap" <0> -> memref<1x512x1x1xf16, "DDR">

    // CHECK:       [[VAR2:%.*]] = VPUIP.SoftMaxUPA
    // CHECK-SAME:      axisInd = 1
    // CHECK-SAME:      inputs([[VAR0]] : memref<1x512x1x1xf16>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<1x512x1x1xf16, "DDR">)

    // CHECK:       [[VAR3:%.*]] = VPUIP.DeclareTensor "VPU_DDR_Heap" <0> -> memref<1x512xf16, "DDR">

    // CHECK:       [[VAR4:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs([[VAR3]] : memref<1x512xf16, "DDR">)
    // CHECK-SAME:      outputs(%arg1 : memref<1x512xf16>) -> memref<1x512xf16>

    // CHECK: return [[VAR4]] : memref<1x512xf16>
}

}
