// RUN: vpux-opt --split-input-file --convert-view-ops-to-VPUIP %s | FileCheck %s

// CHECK-LABEL: @Reshape
func @Reshape(%arg0: memref<1x512xf16>, %arg1: memref<1x512xf16>) -> memref<1x512xf16> {
    %0 = IERT.GenericReshape inputs(%arg0 : memref<1x512xf16>) -> memref<1x512x1x1xf16>
    %1 = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<1x512x1x1xf16, "DDR">
    %2 = VPUIP.SoftMaxUPA {axisInd = 1 : i32} inputs(%0 : memref<1x512x1x1xf16>) outputs(%1 : memref<1x512x1x1xf16, "DDR">) -> memref<1x512x1x1xf16, "DDR">
    %3 = IERT.GenericReshape inputs(%2 : memref<1x512x1x1xf16, "DDR">) -> memref<1x512xf16, "DDR">
    %4 = VPUIP.NNDMA inputs(%3 : memref<1x512xf16, "DDR">) outputs(%arg1 : memref<1x512xf16>) -> memref<1x512xf16>
    return %4 : memref<1x512xf16>

    // CHECK:       [[VAR0:%.*]] = VPUIP.DeclareTensor "ProgrammableInput" [0] <0> -> memref<1x512x1x1xf16>

    // CHECK:       [[VAR1:%.*]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<1x512x1x1xf16, "DDR">

    // CHECK:       [[VAR2:%.*]] = VPUIP.SoftMaxUPA
    // CHECK-SAME:      axisInd = 1
    // CHECK-SAME:      inputs([[VAR0]] : memref<1x512x1x1xf16>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<1x512x1x1xf16, "DDR">)

    // CHECK:       [[VAR3:%.*]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<1x512xf16, "DDR">

    // CHECK:       [[VAR4:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs([[VAR3]] : memref<1x512xf16, "DDR">)
    // CHECK-SAME:      outputs(%arg1 : memref<1x512xf16>) -> memref<1x512xf16>

    // CHECK: return [[VAR4]] : memref<1x512xf16>
}

// -----

#map0 = affine_map<(d0, d1) -> (d0 * 4 + d1)>
#map1 = affine_map<(d0, d1) -> (d0 * 4 + d1 + 8)>

// CHECK-LABEL: @SubView
func @SubView(%arg0: memref<4x4xf16>, %arg1: memref<4x4xf16>) -> memref<4x4xf16> {
    %0 = memref.subview %arg0[0, 0][2, 4][1, 1] : memref<4x4xf16> to memref<2x4xf16, #map0>
    %1 = memref.subview %arg1[0, 0][2, 4][1, 1] : memref<4x4xf16> to memref<2x4xf16, #map0>
    %2 = VPUIP.NNDMA inputs(%0 : memref<2x4xf16, #map0>) outputs(%1 : memref<2x4xf16, #map0>) -> memref<2x4xf16, #map0>

    %3 = memref.subview %arg0[2, 0][2, 4][1, 1] : memref<4x4xf16> to memref<2x4xf16, #map1>
    %4 = memref.subview %arg1[2, 0][2, 4][1, 1] : memref<4x4xf16> to memref<2x4xf16, #map1>
    %5 = VPUIP.NNDMA inputs(%3 : memref<2x4xf16, #map1>) outputs(%4 : memref<2x4xf16, #map1>) -> memref<2x4xf16, #map1>

    return %arg1 : memref<4x4xf16>

    // CHECK:       [[VAR0:%.*]] = VPUIP.DeclareTensor "ProgrammableInput" [0] <0> -> memref<2x4xf16, #map0>
    // CHECK:       [[VAR1:%.*]] = VPUIP.DeclareTensor "ProgrammableOutput" [0] <0> -> memref<2x4xf16, #map0>
    // CHECK:       [[VAR2:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs([[VAR0]] : memref<2x4xf16, #map0>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<2x4xf16, #map0>) -> memref<2x4xf16, #map0>

    // CHECK:       [[VAR3:%.*]] = VPUIP.DeclareTensor "ProgrammableInput" [0] <16> -> memref<2x4xf16, #map1>
    // CHECK:       [[VAR4:%.*]] = VPUIP.DeclareTensor "ProgrammableOutput" [0] <16> -> memref<2x4xf16, #map1>
    // CHECK:       [[VAR5:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs([[VAR3]] : memref<2x4xf16, #map1>)
    // CHECK-SAME:      outputs([[VAR4]] : memref<2x4xf16, #map1>) -> memref<2x4xf16, #map1>

    // CHECK:       return %arg1 : memref<4x4xf16>
}

// -----

#map = affine_map<(d0) -> (d0 + 256)>

// CHECK-LABEL: @Broadcasting
func @Broadcasting(%arg0: memref<256xf16>) -> memref<256xf16> {
    %0 = VPUIP.DeclareTensor "VPU_CMX_NN" [0, 1, 2, 3] <0> -> memref<512xf16, "CMX_NN">

    %1 = memref.subview %0[0][256][1] : memref<512xf16, "CMX_NN"> to memref<256xf16, "CMX_NN">
    %2 = VPUIP.NNDMA inputs(%arg0 : memref<256xf16>) outputs(%1 : memref<256xf16, "CMX_NN">) -> memref<256xf16, "CMX_NN">

    %3 = memref.subview %0[256][256][1] : memref<512xf16, "CMX_NN"> to memref<256xf16, #map, "CMX_NN">
    %4 = VPUIP.NNDMA inputs(%arg0 : memref<256xf16>) outputs(%3 : memref<256xf16, #map, "CMX_NN">) -> memref<256xf16, #map, "CMX_NN">

    return %arg0 : memref<256xf16>

    // CHECK:       [[VAR1:%.*]] = VPUIP.DeclareTensor "VPU_CMX_NN" [0, 1, 2, 3] <0> -> memref<256xf16, "CMX_NN">
    // CHECK:       [[VAR2:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs(%arg0 : memref<256xf16>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<256xf16, "CMX_NN">) -> memref<256xf16, "CMX_NN">

    // CHECK:       [[VAR3:%.*]] = VPUIP.DeclareTensor "VPU_CMX_NN" [0, 1, 2, 3] <512> -> memref<256xf16, #map, "CMX_NN">
    // CHECK:       [[VAR4:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs(%arg0 : memref<256xf16>)
    // CHECK-SAME:      outputs([[VAR3]] : memref<256xf16, #map, "CMX_NN">) -> memref<256xf16, #map, "CMX_NN">

    // CHECK:       return %arg0 : memref<256xf16>
}

// -----

// CHECK-LABEL: @WithAsyncRegions
func @WithAsyncRegions(%arg0: memref<1x512xf16>, %arg1: memref<1x512xf16>) -> memref<1x512xf16> {
    %0 = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<1x512x1x1xf16, "DDR">

    %t2, %f2 = async.execute -> !async.value<memref<1x512x1x1xf16, "DDR">> {
        %1 = IERT.GenericReshape inputs(%arg0 : memref<1x512xf16>) -> memref<1x512x1x1xf16>
        %2 = VPUIP.SoftMaxUPA {axisInd = 1 : i32}
            inputs(%1 : memref<1x512x1x1xf16>)
            outputs(%0 : memref<1x512x1x1xf16, "DDR">)
            -> memref<1x512x1x1xf16, "DDR">
        async.yield %2 : memref<1x512x1x1xf16, "DDR">
    }

    %t4, %f4 = async.execute [%t2] (%f2 as %2: !async.value<memref<1x512x1x1xf16, "DDR">>) -> !async.value<memref<1x512xf16>> {
        %3 = IERT.GenericReshape inputs(%2 : memref<1x512x1x1xf16, "DDR">) -> memref<1x512xf16, "DDR">
        %4 = VPUIP.NNDMA inputs(%3 : memref<1x512xf16, "DDR">) outputs(%arg1 : memref<1x512xf16>) -> memref<1x512xf16>
        async.yield %4 : memref<1x512xf16>
    }

    %4 = async.await %f4 : !async.value<memref<1x512xf16>>
    return %4 : memref<1x512xf16>

    // CHECK:       [[VAR0:%.*]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<1x512x1x1xf16, "DDR">

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute
    // CHECK:           [[VAR1:%.*]] = VPUIP.DeclareTensor "ProgrammableInput" [0] <0> -> memref<1x512x1x1xf16>
    // CHECK:           [[VAR2:%.*]] = VPUIP.SoftMaxUPA
    // CHECK-SAME:          inputs([[VAR1]] : memref<1x512x1x1xf16>)
    // CHECK-SAME:          outputs([[VAR0]] : memref<1x512x1x1xf16, "DDR">)
    // CHECK:           async.yield [[VAR2]] : memref<1x512x1x1xf16, "DDR">

    // CHECK:       [[T4:%.+]], [[F4:%.+]] = async.execute
    // CHECK-SAME:          [[T2]]
    // CHECK-SAME:          ([[F2]] as [[VAR2:%.+]]: !async.value<memref<1x512x1x1xf16, "DDR">>)
    // CHECK:           [[VAR3:%.*]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<1x512xf16, "DDR">
    // CHECK:           [[VAR4:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:          inputs([[VAR3]] : memref<1x512xf16, "DDR">)
    // CHECK-SAME:          outputs(%arg1 : memref<1x512xf16>)
    // CHECK:           async.yield [[VAR4]] : memref<1x512xf16>

    // CHECK:       [[VAR4:%.+]] = async.await [[F4]] : !async.value<memref<1x512xf16>>
    // CHECK:       return [[VAR4]] : memref<1x512xf16>
}
