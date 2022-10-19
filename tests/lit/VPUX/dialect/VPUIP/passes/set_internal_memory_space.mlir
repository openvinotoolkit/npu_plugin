// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --set-internal-memory-space="memory-space=DDR" %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

func @MultipleAllocs(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) -> memref<1x1000xf16> {
    %0 = memref.alloc() : memref<1x1000xf16>
    %1 = IERT.SoftMax {axisInd = 1} inputs(%arg0 : memref<1x1000xf16>) outputs(%0 : memref<1x1000xf16>) -> memref<1x1000xf16>
    %2 = memref.alloc() : memref<1x1000xf16>
    %3 = IERT.SoftMax {axisInd = 1} inputs(%1 : memref<1x1000xf16>) outputs(%2 : memref<1x1000xf16>) -> memref<1x1000xf16>
    %4 = IERT.SoftMax {axisInd = 1} inputs(%3 : memref<1x1000xf16>) outputs(%arg1 : memref<1x1000xf16>) -> memref<1x1000xf16>
    return %4 : memref<1x1000xf16>

    // CHECK: [[VAR0:%.+]] = memref.alloc() : memref<1x1000xf16, @DDR>
    // CHECK: [[VAR1:%.+]] = IERT.SoftMax {axisInd = 1 : i64} inputs(%arg0 : memref<1x1000xf16>) outputs([[VAR0]] : memref<1x1000xf16, @DDR>) -> memref<1x1000xf16, @DDR>
    // CHECK: [[VAR2:%.+]] = memref.alloc() : memref<1x1000xf16, @DDR>
    // CHECK: [[VAR3:%.+]] = IERT.SoftMax {axisInd = 1 : i64} inputs([[VAR1]] : memref<1x1000xf16, @DDR>) outputs([[VAR2]] : memref<1x1000xf16, @DDR>) -> memref<1x1000xf16, @DDR>
    // CHECK: [[VAR4:%.+]] = IERT.SoftMax {axisInd = 1 : i64} inputs([[VAR3]] : memref<1x1000xf16, @DDR>) outputs(%arg1 : memref<1x1000xf16>) -> memref<1x1000xf16>
    // CHECK: return [[VAR4]] : memref<1x1000xf16>
}

// -----

func @ReshapeInGraph(%arg0: memref<1x512x1x1xf32>, %arg1: memref<1x512x1x1xf32>) -> memref<1x512x1x1xf32> {
    %0 = VPUIP.GenericReshape inputs(%arg0 : memref<1x512x1x1xf32>) -> memref<1x512xf32>
    %1 = memref.alloc() : memref<1x512xf32>
    %2 = IERT.SoftMax {axisInd = 1} inputs(%0 : memref<1x512xf32>) outputs(%1 : memref<1x512xf32>) -> memref<1x512xf32>
    %3 = VPUIP.GenericReshape inputs(%2 : memref<1x512xf32>) -> memref<1x512x1x1xf32>
    %4 = VPUIP.Copy inputs(%3 : memref<1x512x1x1xf32>) outputs(%arg1 : memref<1x512x1x1xf32>) -> memref<1x512x1x1xf32>
    return %4 : memref<1x512x1x1xf32>

    // CHECK: [[VAR0:%.+]] =  VPUIP.GenericReshape inputs(%arg0 : memref<1x512x1x1xf32>) -> memref<1x512xf32>
    // CHECK: [[VAR1:%.+]] =  memref.alloc() : memref<1x512xf32, @DDR>
    // CHECK: [[VAR2:%.+]] =  IERT.SoftMax {axisInd = 1 : i64} inputs([[VAR0]] : memref<1x512xf32>) outputs([[VAR1]] : memref<1x512xf32, @DDR>) -> memref<1x512xf32, @DDR>
    // CHECK: [[VAR3:%.+]] =  VPUIP.GenericReshape inputs([[VAR2]] : memref<1x512xf32, @DDR>) -> memref<1x512x1x1xf32, @DDR>
    // CHECK: [[VAR4:%.+]] =  VPUIP.Copy inputs([[VAR3]] : memref<1x512x1x1xf32, @DDR>) outputs(%arg1 : memref<1x512x1x1xf32>) -> memref<1x512x1x1xf32>
    // CHECK: return [[VAR4]] : memref<1x512x1x1xf32>
}

// -----

func @Async(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) -> memref<1x1000xf16> {
    %0 = memref.alloc() : memref<1x1000xf16>

    %t1, %f1 = async.execute -> !async.value<memref<1x1000xf16>> {
        %1 = IERT.ReLU inputs(%arg0 : memref<1x1000xf16>) outputs(%0 : memref<1x1000xf16>) -> memref<1x1000xf16>
        async.yield %1 : memref<1x1000xf16>
    }

    %t2, %f2 = async.execute [%t1] (%f1 as %1 : !async.value<memref<1x1000xf16>>) -> !async.value<memref<1x1000xf16>> {
        %2 = VPUIP.Copy inputs(%1 : memref<1x1000xf16>) outputs(%arg1 : memref<1x1000xf16>) -> memref<1x1000xf16>
        async.yield %2 : memref<1x1000xf16>
    }

    %2 = async.await %f2 : !async.value<memref<1x1000xf16>>
    return %2 : memref<1x1000xf16>

    // CHECK:       [[VAR0:%.+]] = memref.alloc() : memref<1x1000xf16, @DDR>

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute -> !async.value<memref<1x1000xf16, @DDR>>
    // CHECK:           [[VAR1:%.+]] = IERT.ReLU
    // CHECK-SAME:          inputs(%arg0 : memref<1x1000xf16>)
    // CHECK-SAME:          outputs([[VAR0]] : memref<1x1000xf16, @DDR>)
    // CHECK-SAME:          -> memref<1x1000xf16, @DDR>
    // CHECK:           async.yield [[VAR1]] : memref<1x1000xf16, @DDR>

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute
    // CHECK-SAME:          [[T1]]
    // CHECK-SAME:          [[F1]] as [[VAR1:%.+]]: !async.value<memref<1x1000xf16, @DDR>>
    // CHECK-SAME:          -> !async.value<memref<1x1000xf16>>
    // CHECK:           [[VAR2:%.+]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[VAR1]] : memref<1x1000xf16, @DDR>)
    // CHECK-SAME:          outputs(%arg1 : memref<1x1000xf16>)
    // CHECK-SAME:          -> memref<1x1000xf16>
    // CHECK:           async.yield [[VAR2]] : memref<1x1000xf16>

    // CHECK:       [[VAR2:%.+]] = async.await [[F2]] : !async.value<memref<1x1000xf16>>
    // CHECK:       return [[VAR2]] : memref<1x1000xf16>
}
