// RUN: vpux-opt --move-wait-result-to-async-block-args %s | FileCheck %s

// CHECK-LABEL: @LinearCase
func @LinearCase(%arg0: memref<10xf16>, %arg1: memref<10xf16>) -> memref<10xf16> {
    %buf0 = IERT.StaticAlloc<0> -> memref<10xf16>
    %buf1 = IERT.StaticAlloc<20> -> memref<10xf16>

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
    // CHECK:       async.await [[T1]]

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute
    // CHECK-SAME:          ([[F1]] as [[VAL1:%.+]]: !async.value<memref<10xf16>>)
    // CHECK:           IERT.ReLU inputs([[VAL1]] : memref<10xf16>)
    // CHECK:       async.await [[T2]]

    // CHECK:       [[T3:%.+]], [[F3:%.+]] = async.execute
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
    // CHECK:       async.await [[T1]]

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute
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
    // CHECK:       async.await [[T1]]

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute
    // CHECK:       async.await [[T2]]

    // CHECK:       [[T3:%.+]], [[F3:%.+]] = async.execute
    // CHECK-SAME:          ([[F1]] as [[VAL1:%.+]]: !async.value<memref<10xf16>>,
    // CHECK-SAME:           [[F2]] as [[VAL2:%.+]]: !async.value<memref<10xf16>>)
    // CHECK:           IERT.Add inputs([[VAL1]] : memref<10xf16>, [[VAL2]] : memref<10xf16>)

    // CHECK:       [[VAL3:%.+]] = async.await [[F3]]
    // CHECK:       return [[VAL3]]
}
