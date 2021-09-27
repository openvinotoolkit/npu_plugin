// RUN: vpux-opt --move-wait-result-to-async-block-args %s | FileCheck %s

// CHECK-LABEL: @LinearCase
func @LinearCase(%arg0: memref<10xf16>, %arg1: memref<10xf16>) -> memref<10xf16> {
    %buf0 = memref.alloc() : memref<10xf16>
    %buf1 = memref.alloc() : memref<10xf16>

    %t1, %f1 = async.execute -> !async.value<memref<10xf16>> {
        %1 = IERT.ReLU inputs(%arg0 : memref<10xf16>) outputs(%buf0 : memref<10xf16>) -> memref<10xf16>
        async.yield %1 : memref<10xf16>
    }
    %1 = async.await %f1 : !async.value<memref<10xf16>>

    %t2, %f2 = async.execute -> !async.value<memref<10xf16>> {
        %2 = IERT.ReLU inputs(%1 : memref<10xf16>) outputs(%buf1 : memref<10xf16>) -> memref<10xf16>
        async.yield %2 : memref<10xf16>
    }
    %2 = async.await %f2 : !async.value<memref<10xf16>>

    %t3, %f3 = async.execute -> !async.value<memref<10xf16>> {
        %3 = IERT.ReLU inputs(%2 : memref<10xf16>) outputs(%arg1 : memref<10xf16>) -> memref<10xf16>
        async.yield %3 : memref<10xf16>
    }
    %3 = async.await %f3 : !async.value<memref<10xf16>>

    return %3 : memref<10xf16>

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK-NOT:   async.await

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute
    // CHECK-SAME:          [[T1]]
    // CHECK-SAME:          ([[F1]] as [[VAL1:%.+]]: !async.value<memref<10xf16>>)
    // CHECK:           IERT.ReLU inputs([[VAL1]] : memref<10xf16>)
    // CHECK-NOT:   async.await

    // CHECK:       [[T3:%.+]], [[F3:%.+]] = async.execute
    // CHECK-NOT:           [[T1]]
    // CHECK-SAME:          [[T2]]
    // CHECK-SAME:          ([[F2]] as [[VAL2:%.+]]: !async.value<memref<10xf16>>)
    // CHECK:           IERT.ReLU inputs([[VAL2]] : memref<10xf16>)

    // CHECK:       [[VAL3:%.+]] = async.await [[F3]]
    // CHECK:       return [[VAL3]]
}

// CHECK-LABEL: @MultipleUsesInOneRegion
func @MultipleUsesInOneRegion(%arg0: memref<10xf16>, %arg1: memref<10xf16>) -> memref<10xf16> {
    %buf0 = memref.alloc() : memref<10xf16>

    %t1, %f1 = async.execute -> !async.value<memref<10xf16>> {
        %1 = IERT.ReLU inputs(%arg0 : memref<10xf16>) outputs(%buf0 : memref<10xf16>) -> memref<10xf16>
        async.yield %1 : memref<10xf16>
    }
    %1 = async.await %f1 : !async.value<memref<10xf16>>

    %t2, %f2 = async.execute -> !async.value<memref<10xf16>> {
        %2 = IERT.Add inputs(%1 : memref<10xf16>, %1 : memref<10xf16>) outputs(%arg1 : memref<10xf16>) -> memref<10xf16>
        async.yield %2 : memref<10xf16>
    }
    %2 = async.await %f2 : !async.value<memref<10xf16>>

    return %2 : memref<10xf16>

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK-NOT:   async.await

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute
    // CHECK-SAME:          [[T1]]
    // CHECK-SAME:          ([[F1]] as [[VAL1:%.+]]: !async.value<memref<10xf16>>)
    // CHECK:           IERT.Add inputs([[VAL1]] : memref<10xf16>, [[VAL1]] : memref<10xf16>)

    // CHECK:       [[VAL2:%.+]] = async.await [[F2]]
    // CHECK:       return [[VAL2]]
}

// CHECK-LABEL: @UsesFromMultipleWaits
func @UsesFromMultipleWaits(%arg0: memref<10xf16>, %arg1: memref<10xf16>) -> memref<10xf16> {
    %buf0 = memref.alloc() : memref<10xf16>
    %buf1 = memref.alloc() : memref<10xf16>

    %t1, %f1 = async.execute -> !async.value<memref<10xf16>> {
        %1 = IERT.ReLU inputs(%arg0 : memref<10xf16>) outputs(%buf0 : memref<10xf16>) -> memref<10xf16>
        async.yield %1 : memref<10xf16>
    }
    %1 = async.await %f1 : !async.value<memref<10xf16>>

    %t2, %f2 = async.execute -> !async.value<memref<10xf16>> {
        %2 = IERT.ReLU inputs(%arg0 : memref<10xf16>) outputs(%buf1 : memref<10xf16>) -> memref<10xf16>
        async.yield %2 : memref<10xf16>
    }
    %2 = async.await %f2 : !async.value<memref<10xf16>>

    %t3, %f3 = async.execute -> !async.value<memref<10xf16>> {
        %3 = IERT.Add inputs(%1 : memref<10xf16>, %2 : memref<10xf16>) outputs(%arg1 : memref<10xf16>) -> memref<10xf16>
        async.yield %3 : memref<10xf16>
    }
    %3 = async.await %f3 : !async.value<memref<10xf16>>

    return %3 : memref<10xf16>

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK-NOT:   async.await

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute
    // CHECK-NOT:   async.await

    // CHECK:       [[T3:%.+]], [[F3:%.+]] = async.execute
    // CHECK-SAME:          [[T1]]
    // CHECK-SAME:          [[T2]]
    // CHECK-SAME:          ([[F1]] as [[VAL1:%.+]]: !async.value<memref<10xf16>>,
    // CHECK-SAME:           [[F2]] as [[VAL2:%.+]]: !async.value<memref<10xf16>>)
    // CHECK:           IERT.Add inputs([[VAL1]] : memref<10xf16>, [[VAL2]] : memref<10xf16>)

    // CHECK:       [[VAL3:%.+]] = async.await [[F3]]
    // CHECK:       return [[VAL3]]
}

// -----

// CHECK-LABEL: @TwoOutputs
func @TwoOutputs(%arg0: memref<2xf16>, %arg1: memref<2xf16>, %arg2: memref<2xf16>) -> (memref<2xf16>, memref<2xf16>) {
    %cst = const.Declare memref<2xf16> = #const.Content<dense<1.0> : tensor<2xf16>>

    %buf1 = memref.alloc() : memref<2xf16>
    %buf2 = memref.alloc() : memref<2xf16>

    %t1, %f1 = async.execute -> !async.value<memref<2xf16>> {
        %1 = IERT.ReLU inputs(%arg0 : memref<2xf16>) outputs(%buf1 : memref<2xf16>) -> memref<2xf16>
        async.yield %1 : memref<2xf16>
    }
    %1 = async.await %f1 : !async.value<memref<2xf16>>


    %t2, %f2 = async.execute -> !async.value<memref<2xf16>> {
        %2 = IERT.ReLU inputs(%cst : memref<2xf16>) outputs(%buf2 : memref<2xf16>) -> memref<2xf16>
        async.yield %2 : memref<2xf16>
    }
    %2 = async.await %f2 : !async.value<memref<2xf16>>

    %t3, %f3 = async.execute -> !async.value<memref<2xf16>> {
        %3 = IERT.Copy inputs(%1 : memref<2xf16>) outputs(%arg1 : memref<2xf16>) -> memref<2xf16>
        async.yield %3 : memref<2xf16>
    }
    %3 = async.await %f3 : !async.value<memref<2xf16>>

    %t4, %f4 = async.execute -> !async.value<memref<2xf16>> {
        %4 = IERT.Copy inputs(%2 : memref<2xf16>) outputs(%arg2 : memref<2xf16>) -> memref<2xf16>
        async.yield %4 : memref<2xf16>
    }
    %4 = async.await %f4 : !async.value<memref<2xf16>>

    return %3, %4 : memref<2xf16>, memref<2xf16>

    // CHECK:       [[CST:%.+]] = const.Declare

    // CHECK:       [[BUF1:%.+]] = memref.alloc() : memref<2xf16>
    // CHECK:       [[BUF2:%.+]] = memref.alloc() : memref<2xf16>

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK:           IERT.ReLU inputs(%arg0 : memref<2xf16>) outputs([[BUF1]] : memref<2xf16>)

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute
    // CHECK:           IERT.ReLU inputs([[CST]] : memref<2xf16>) outputs([[BUF2]] : memref<2xf16>)

    // CHECK:       [[T3:%.+]], [[F3:%.+]] = async.execute
    // CHECK-SAME:          [[T1]]
    // CHECK-NOT:           [[T2]]
    // CHECK-SAME:          [[F1]] as [[VAL1:%.+]]: !async.value<memref<2xf16>>
    // CHECK:           IERT.Copy inputs([[VAL1]] : memref<2xf16>) outputs(%arg1 : memref<2xf16>)

    // CHECK:       [[T4:%.+]], [[F4:%.+]] = async.execute
    // CHECK-NOT:           [[T1]]
    // CHECK-SAME:          [[T2]]
    // CHECK-NOT:           [[T3]]
    // CHECK-SAME:          [[F2]] as [[VAL2:%.+]]: !async.value<memref<2xf16>>
    // CHECK:           IERT.Copy inputs([[VAL2]] : memref<2xf16>) outputs(%arg2 : memref<2xf16>)

    // CHECK:       [[VAL3:%.+]] = async.await [[F3]]
    // CHECK:       [[VAL4:%.+]] = async.await [[F4]]
    // CHECK:       return [[VAL3]], [[VAL4]]
}
