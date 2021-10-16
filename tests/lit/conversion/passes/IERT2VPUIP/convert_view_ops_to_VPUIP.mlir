// RUN: vpux-opt --split-input-file --convert-view-ops-to-VPUIP %s | FileCheck %s

// CHECK-LABEL: @Reshape
func @Reshape(%arg0: memref<1x512xf16>, %arg1: memref<1x512xf16>) -> memref<1x512xf16> {
    %0 = IERT.GenericReshape inputs(%arg0 : memref<1x512xf16>) -> memref<1x512x1x1xf16>
    %1 = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<1x512x1x1xf16, "DDR">
    %2 = VPUIP.SoftMaxUPA {axisInd = 1} inputs(%0 : memref<1x512x1x1xf16>) outputs(%1 : memref<1x512x1x1xf16, "DDR">) -> memref<1x512x1x1xf16, "DDR">
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

#map = affine_map<(d0, d1) -> (d0 * 4 + d1)>

// CHECK-LABEL: @SubView
func @SubView(%arg0: memref<4x4xf16>, %arg1: memref<4x4xf16>) -> memref<4x4xf16> {
    %0 = IERT.SubView %arg0 [0, 0][2, 4] : memref<4x4xf16> to memref<2x4xf16, #map>
    %1 = IERT.SubView %arg1 [0, 0][2, 4] : memref<4x4xf16> to memref<2x4xf16, #map>
    %2 = VPUIP.NNDMA inputs(%0 : memref<2x4xf16, #map>) outputs(%1 : memref<2x4xf16, #map>) -> memref<2x4xf16, #map>

    %3 = IERT.SubView %arg0 [2, 0][2, 4] : memref<4x4xf16> to memref<2x4xf16, #map>
    %4 = IERT.SubView %arg1 [2, 0][2, 4] : memref<4x4xf16> to memref<2x4xf16, #map>
    %5 = VPUIP.NNDMA inputs(%3 : memref<2x4xf16, #map>) outputs(%4 : memref<2x4xf16, #map>) -> memref<2x4xf16, #map>

    return %arg1 : memref<4x4xf16>

    // CHECK:       [[VAR0:%.*]] = VPUIP.DeclareTensor "ProgrammableInput" [0] <0> -> memref<2x4xf16, #map>
    // CHECK:       [[VAR1:%.*]] = VPUIP.DeclareTensor "ProgrammableOutput" [0] <0> -> memref<2x4xf16, #map>
    // CHECK:       [[VAR2:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs([[VAR0]] : memref<2x4xf16, #map>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<2x4xf16, #map>) -> memref<2x4xf16, #map>

    // CHECK:       [[VAR3:%.*]] = VPUIP.DeclareTensor "ProgrammableInput" [0] <16> -> memref<2x4xf16, #map>
    // CHECK:       [[VAR4:%.*]] = VPUIP.DeclareTensor "ProgrammableOutput" [0] <16> -> memref<2x4xf16, #map>
    // CHECK:       [[VAR5:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs([[VAR3]] : memref<2x4xf16, #map>)
    // CHECK-SAME:      outputs([[VAR4]] : memref<2x4xf16, #map>) -> memref<2x4xf16, #map>

    // CHECK:       return %arg1 : memref<4x4xf16>
}

// -----

// CHECK-LABEL: @Broadcasting
func @Broadcasting(%arg0: memref<256xf16>) -> memref<256xf16> {
    %0 = VPUIP.DeclareTensor "VPU_CMX_NN" [0, 1, 2, 3] <0> -> memref<512xf16, "CMX_NN">

    %1 = IERT.SubView %0[0][256] : memref<512xf16, "CMX_NN"> to memref<256xf16, "CMX_NN">
    %2 = VPUIP.NNDMA inputs(%arg0 : memref<256xf16>) outputs(%1 : memref<256xf16, "CMX_NN">) -> memref<256xf16, "CMX_NN">

    %3 = IERT.SubView %0[256][256] : memref<512xf16, "CMX_NN"> to memref<256xf16, "CMX_NN">
    %4 = VPUIP.NNDMA inputs(%arg0 : memref<256xf16>) outputs(%3 : memref<256xf16, "CMX_NN">) -> memref<256xf16, "CMX_NN">

    return %arg0 : memref<256xf16>

    // CHECK:       [[VAR1:%.*]] = VPUIP.DeclareTensor "VPU_CMX_NN" [0, 1, 2, 3] <0> -> memref<256xf16, "CMX_NN">
    // CHECK:       [[VAR2:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs(%arg0 : memref<256xf16>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<256xf16, "CMX_NN">) -> memref<256xf16, "CMX_NN">

    // CHECK:       [[VAR3:%.*]] = VPUIP.DeclareTensor "VPU_CMX_NN" [0, 1, 2, 3] <512> -> memref<256xf16, "CMX_NN">
    // CHECK:       [[VAR4:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs(%arg0 : memref<256xf16>)
    // CHECK-SAME:      outputs([[VAR3]] : memref<256xf16, "CMX_NN">) -> memref<256xf16, "CMX_NN">

    // CHECK:       return %arg0 : memref<256xf16>
}

// -----

// CHECK-LABEL: @WithAsyncRegions
func @WithAsyncRegions(%arg0: memref<1x1x1x512xf32>, %arg1: memref<1x1x1x512xf32>) -> memref<1x1x1x512xf32> {
    %t0, %f0 = async.execute -> !async.value<memref<1x1x1x512xf16, "DDR">> {
        %0 = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<1x1x1x512xf16, "DDR">
        %1 = VPUIP.ConvertUPA inputs(%arg0 : memref<1x1x1x512xf32>) outputs(%0 : memref<1x1x1x512xf16, "DDR">) -> memref<1x1x1x512xf16, "DDR">
        async.yield %1 : memref<1x1x1x512xf16, "DDR">
    }

    %t1, %f1 = async.execute [%t0] (%f0 as %1: !async.value<memref<1x1x1x512xf16, "DDR">>) -> !async.value<memref<1x512xf16, "DDR">> {
        %2 = IERT.GenericReshape inputs(%1 : memref<1x1x1x512xf16, "DDR">) -> memref<1x512xf16, "DDR">
        %3 = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <1024> -> memref<1x512xf16, "DDR">
        %4 = VPUIP.SoftMaxUPA {axisInd = 1} inputs(%2 : memref<1x512xf16, "DDR">) outputs(%3 : memref<1x512xf16, "DDR">) -> memref<1x512xf16, "DDR">
        async.yield %4 : memref<1x512xf16, "DDR">
    }

    %t2, %f2 = async.execute [%t1] (%f1 as %4: !async.value<memref<1x512xf16, "DDR">>) -> !async.value<memref<1x1x1x512xf32>> {
        %5 = IERT.GenericReshape inputs(%4 : memref<1x512xf16, "DDR">) -> memref<1x1x1x512xf16, "DDR">
        %6 = VPUIP.ConvertUPA inputs(%5 : memref<1x1x1x512xf16, "DDR">) outputs(%arg1 : memref<1x1x1x512xf32>) -> memref<1x1x1x512xf32>
        async.yield %6 : memref<1x1x1x512xf32>
    }

    %6 = async.await %f2 : !async.value<memref<1x1x1x512xf32>>
    return %6 : memref<1x1x1x512xf32>

    // CHECK:       [[T0:%.+]], [[F0:%.+]] = async.execute
    // CHECK-SAME:          -> !async.value<memref<1x1x1x512xf16, "DDR">>
    // CHECK:           [[VAR0:%.*]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<1x1x1x512xf16, "DDR">
    // CHECK:           [[VAR1:%.*]] = VPUIP.ConvertUPA
    // CHECK-SAME:          inputs(%arg0 : memref<1x1x1x512xf32>)
    // CHECK-SAME:          outputs([[VAR0]] : memref<1x1x1x512xf16, "DDR">)
    // CHECK:           async.yield [[VAR1]] : memref<1x1x1x512xf16, "DDR">

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK-SAME:          [[T0]]
    // CHECK-SAME:          ([[F0]] as [[VAR1:%.+]]: !async.value<memref<1x1x1x512xf16, "DDR">>)
    // CHECK-SAME:          -> !async.value<memref<1x512xf16, "DDR">>
    // CHECK:           [[VAR2:%.*]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<1x512xf16, "DDR">
    // CHECK:           [[VAR3:%.*]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <1024> -> memref<1x512xf16, "DDR">
    // CHECK:           [[VAR4:%.*]] = VPUIP.SoftMaxUPA
    // CHECK-SAME:          inputs([[VAR2]] : memref<1x512xf16, "DDR">)
    // CHECK-SAME:          outputs([[VAR3]] : memref<1x512xf16, "DDR">)
    // CHECK:           async.yield [[VAR4]] : memref<1x512xf16, "DDR">

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute
    // CHECK-SAME:          [[T1]]
    // CHECK-SAME:          ([[F1]] as [[VAR4:%.+]]: !async.value<memref<1x512xf16, "DDR">>)
    // CHECK-SAME:          -> !async.value<memref<1x1x1x512xf32>>
    // CHECK:           [[VAR5:%.*]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <1024> -> memref<1x1x1x512xf16, "DDR">
    // CHECK:           [[VAR6:%.*]] = VPUIP.ConvertUPA
    // CHECK-SAME:          inputs([[VAR5]] : memref<1x1x1x512xf16, "DDR">)
    // CHECK-SAME:          outputs(%arg1 : memref<1x1x1x512xf32>)
    // CHECK:           async.yield [[VAR6]] : memref<1x1x1x512xf32>

    // CHECK:       [[VAR6:%.+]] = async.await [[F2]] : !async.value<memref<1x1x1x512xf32>>
    // CHECK:       return [[VAR6]] : memref<1x1x1x512xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d0 * 3072 + d1 * 192 + d2 * 12 + d3)>

// CHECK-LABEL: @PermuteCast
func @PermuteCast(%arg0: memref<1x12x16x16xf16, #NHWC, #map>, %arg1: memref<1x16x16x12xf16>) -> memref<1x16x16x12xf16> {
    %0 = IERT.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW}
        inputs(%arg0 : memref<1x12x16x16xf16, #NHWC, #map>)
        -> memref<1x16x16x12xf16>

    %1 = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <2000> -> memref<1x16x16x12xf16, "DDR">
    %2 = VPUIP.SoftMaxUPA {axisInd = 1}
        inputs(%0 : memref<1x16x16x12xf16>)
        outputs(%1 : memref<1x16x16x12xf16, "DDR">) -> memref<1x16x16x12xf16, "DDR">
    %3 = VPUIP.NNDMA
        inputs(%2 : memref<1x16x16x12xf16, "DDR">)
        outputs(%arg1 : memref<1x16x16x12xf16>) -> memref<1x16x16x12xf16>
    return %3 : memref<1x16x16x12xf16>

    //CHECK:        [[VAR0:%.*]] = VPUIP.DeclareTensor "ProgrammableInput" [0] <0> -> memref<1x16x16x12xf16>
    //CHECK:        [[VAR1:%.*]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <2000> -> memref<1x16x16x12xf16, "DDR">
    //CHECK:        [[VAR2:%.*]] = VPUIP.SoftMaxUPA {axisInd = 1 : i64}
    //CHECK-SAME:       inputs([[VAR0]] : memref<1x16x16x12xf16>)
    //CHECK-SAME:       outputs([[VAR1]] : memref<1x16x16x12xf16, "DDR">) -> memref<1x16x16x12xf16, "DDR">
    //CHECK:        [[VAR3:%.*]] = VPUIP.NNDMA
    //CHECK-SAME:       inputs([[VAR2]] : memref<1x16x16x12xf16, "DDR">)
    //CHECK-SAME:       outputs(%arg1 : memref<1x16x16x12xf16>) -> memref<1x16x16x12xf16>
    //CHECK:        return [[VAR3]] : memref<1x16x16x12xf16>
}
