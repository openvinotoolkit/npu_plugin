// RUN: vpux-opt --split-input-file --convert-declarations-to-VPUIP %s | FileCheck %s

// CHECK-LABEL: @ConstantLayer
func @ConstantLayer(%arg0: memref<1x2x2x2xf16>) -> memref<1x2x2x2xf16> {
    %0 = const.Declare memref<1x2x2x2xf16> = #const.Content<dense<1.0> : tensor<1x2x2x2xf16>>
    %1 = VPUIP.NNDMA inputs(%0 : memref<1x2x2x2xf16>) outputs(%arg0 : memref<1x2x2x2xf16>) -> memref<1x2x2x2xf16>
    return %1: memref<1x2x2x2xf16>

    // CHECK:       [[VAR0:%.*]] = const.Declare memref<1x2x2x2xf16>
    // CHECK-SAME:      = #const.Content<dense<1.000000e+00> : tensor<1x2x2x2xf16>>

    // CHECK:       [[VAR1:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs([[VAR0]] : memref<1x2x2x2xf16>)
    // CHECK-SAME:      outputs(%arg0 : memref<1x2x2x2xf16>) -> memref<1x2x2x2xf16>

    // CHECK: return [[VAR1]] : memref<1x2x2x2xf16>
}

// -----

// CHECK-LABEL: @StaticAlloc
func @StaticAlloc(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    %0 = IERT.StaticAlloc<0> -> memref<1x1x1x1000xf16, "DDR">
    %1 = VPUIP.SoftMaxUPA {axisInd = 3} inputs(%arg0 : memref<1x1x1x1000xf16>) outputs(%0 : memref<1x1x1x1000xf16, "DDR">) -> memref<1x1x1x1000xf16, "DDR">

    %2 = IERT.StaticAlloc<2048> -> memref<1x1x1x1000xf16, "DDR">
    %3 = VPUIP.SoftMaxUPA {axisInd = 3} inputs(%1 : memref<1x1x1x1000xf16, "DDR">) outputs(%2 : memref<1x1x1x1000xf16, "DDR">) -> memref<1x1x1x1000xf16, "DDR">
    %4 = VPUIP.SoftMaxUPA {axisInd = 3} inputs(%3 : memref<1x1x1x1000xf16, "DDR">) outputs(%arg1 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>

    return %4: memref<1x1x1x1000xf16>

    // CHECK:       [[VAR0:%.*]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<1x1x1x1000xf16, "DDR">

    // CHECK:       [[VAR1:%.*]] = VPUIP.SoftMaxUPA
    // CHECK-SAME:      axisInd = 3
    // CHECK-SAME:      inputs(%arg0 : memref<1x1x1x1000xf16>)
    // CHECK-SAME:      outputs([[VAR0]] : memref<1x1x1x1000xf16, "DDR">)

    // CHECK:       [[VAR2:%.*]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <2048> -> memref<1x1x1x1000xf16, "DDR">

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

// -----

// CHECK-LABEL: @ReshapeInGraph
func @ReshapeInGraph(%arg0: memref<1x512xf16>, %arg1: memref<1x512xf16>) -> memref<1x512xf16> {
    %0 = IERT.GenericReshape inputs(%arg0 : memref<1x512xf16>) -> memref<1x512x1x1xf16>
    %1 = IERT.StaticAlloc<0> -> memref<1x512x1x1xf16, "DDR">
    %2 = VPUIP.SoftMaxUPA {axisInd = 1} inputs(%0 : memref<1x512x1x1xf16>) outputs(%1 : memref<1x512x1x1xf16, "DDR">) -> memref<1x512x1x1xf16, "DDR">
    %3 = IERT.GenericReshape inputs(%2 : memref<1x512x1x1xf16, "DDR">) -> memref<1x512xf16, "DDR">
    %4 = VPUIP.NNDMA inputs(%3 : memref<1x512xf16, "DDR">) outputs(%arg1 : memref<1x512xf16>) -> memref<1x512xf16>
    return %4: memref<1x512xf16>

    // CHECK:       [[VAR0:%.*]] = IERT.GenericReshape inputs(%arg0 : memref<1x512xf16>) -> memref<1x512x1x1xf16>

    // CHECK:       [[VAR1:%.*]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<1x512x1x1xf16, "DDR">

    // CHECK:       [[VAR2:%.*]] = VPUIP.SoftMaxUPA
    // CHECK-SAME:      axisInd = 1
    // CHECK-SAME:      inputs([[VAR0]] : memref<1x512x1x1xf16>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<1x512x1x1xf16, "DDR">)

    // CHECK:       [[VAR3:%.*]] = IERT.GenericReshape inputs([[VAR2]] : memref<1x512x1x1xf16, "DDR">) -> memref<1x512xf16, "DDR">

    // CHECK:       [[VAR4:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs([[VAR3]] : memref<1x512xf16, "DDR">)
    // CHECK-SAME:      outputs(%arg1 : memref<1x512xf16>) -> memref<1x512xf16>

    // CHECK: return [[VAR4]] : memref<1x512xf16>
}

// -----

// CHECK-LABEL: @WithAsyncRegions
func @WithAsyncRegions(%arg0: memref<10xf16>, %arg1: memref<10xf16>) -> memref<10xf16> {
    %0 = IERT.StaticAlloc<0> -> memref<10xf16, "DDR">

    %t1, %f1 = async.execute -> !async.value<memref<10xf16, "DDR">> {
        %1 = VPUIP.NNDMA inputs(%arg0 : memref<10xf16>) outputs(%0 : memref<10xf16, "DDR">) -> memref<10xf16, "DDR">
        async.yield %1 : memref<10xf16, "DDR">
    }

    %t2, %f2 = async.execute [%t1] (%f1 as %1: !async.value<memref<10xf16, "DDR">>) -> !async.value<memref<10xf16>> {
        %2 = VPUIP.NNDMA inputs(%1 : memref<10xf16, "DDR">) outputs(%arg1 : memref<10xf16>) -> memref<10xf16>
        async.yield %2 : memref<10xf16>
    }

    %2 = async.await %f2 : !async.value<memref<10xf16>>
    return %2 : memref<10xf16>

    // CHECK:       [[VAR0:%.*]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<10xf16, "DDR">

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK:           [[VAR1:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:          inputs(%arg0 : memref<10xf16>)
    // CHECK-SAME:          outputs([[VAR0]] : memref<10xf16, "DDR">
    // CHECK:           async.yield [[VAR1]] : memref<10xf16, "DDR">

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute
    // CHECK-SAME:          [[T1]]
    // CHECK-SAME:          ([[F1]] as [[VAR1:%.+]]: !async.value<memref<10xf16, "DDR">>)
    // CHECK:           [[VAR2:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:          inputs([[VAR1]] : memref<10xf16, "DDR">)
    // CHECK-SAME:          outputs(%arg1 : memref<10xf16>)
    // CHECK:           async.yield [[VAR2]] : memref<10xf16>

    // CHECK:       [[VAR2:%.+]] = async.await [[F2]] : !async.value<memref<10xf16>>
    // CHECK:       return [[VAR2]] : memref<10xf16>
}
