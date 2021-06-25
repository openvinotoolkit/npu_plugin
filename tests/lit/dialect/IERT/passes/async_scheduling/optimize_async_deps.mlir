// RUN: vpux-opt --split-input-file --optimize-async-deps %s | FileCheck %s

// CHECK-LABEL: @LinearGraph
func @LinearGraph(%arg0: memref<10xf16>, %arg1: memref<10xf16>) -> memref<10xf16> {
    %buf0 = IERT.StaticAlloc<0> -> memref<10xf16>
    %buf1 = IERT.StaticAlloc<20> -> memref<10xf16>

    %t1, %f1 = async.execute -> !async.value<memref<10xf16>> {
        %1 = IERT.ReLU inputs(%arg0 : memref<10xf16>) outputs(%buf0 : memref<10xf16>) -> memref<10xf16>
        async.yield %1 : memref<10xf16>
    }

    async.await %t1 : !async.token

    %t2, %f2 = async.execute(%f1 as %1 : !async.value<memref<10xf16>>) -> !async.value<memref<10xf16>> {
        %2 = IERT.ReLU inputs(%1 : memref<10xf16>) outputs(%buf1 : memref<10xf16>) -> memref<10xf16>
        async.yield %2 : memref<10xf16>
    }

    async.await %t2 : !async.token

    %t3, %f3 = async.execute(%f2 as %2 : !async.value<memref<10xf16>>) -> !async.value<memref<10xf16>> {
        %3 = IERT.ReLU inputs(%2 : memref<10xf16>) outputs(%arg1 : memref<10xf16>) -> memref<10xf16>
        async.yield %3 : memref<10xf16>
    }

    %3 = async.await %f3 : !async.value<memref<10xf16>>
    return %3 : memref<10xf16>

    // CHECK:       [[BUF0:%.+]] = IERT.StaticAlloc<0>
    // CHECK:       [[BUF1:%.+]] = IERT.StaticAlloc<20>

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute
    // CHECK-SAME:          [[T1]]
    // CHECK-SAME:          ([[F1]] as [[VAL1:%.+]]: !async.value<memref<10xf16>>)
    // CHECK-SAME:          -> !async.value<memref<10xf16>> {
    // CHECK:           {{%.+}} = IERT.ReLU inputs([[VAL1]] : memref<10xf16>) outputs([[BUF1]] : memref<10xf16>) -> memref<10xf16>

    // CHECK:       [[T3:%.+]], [[F3:%.+]] = async.execute
    // CHECK-SAME:          [[T2]]
    // CHECK-SAME:          ([[F2]] as [[VAL2:%.+]]: !async.value<memref<10xf16>>)
    // CHECK-SAME:          -> !async.value<memref<10xf16>> {
    // CHECK:           {{%.+}} = IERT.ReLU inputs([[VAL2]] : memref<10xf16>) outputs(%arg1 : memref<10xf16>) -> memref<10xf16>

    // CHECK:       [[VAL3:%.+]] = async.await [[F3]] : !async.value<memref<10xf16>>
    // CHECK:       return [[VAL3]]
}

// -----

#map = affine_map<(d0) -> (d0 + 10)>

// CHECK-LABEL: @IndependentBranchesLinearSched
func @IndependentBranchesLinearSched(%arg0: memref<10xf16>, %arg1: memref<10xf16>, %arg2: memref<20xf16>) -> memref<20xf16> {
    %buf = IERT.StaticAlloc<0> -> memref<20xf16>

    %t0, %f0 = async.execute -> !async.value<memref<10xf16>> {
        %buf0 = memref.subview %buf[0][10][1] : memref<20xf16> to memref<10xf16>
        %0 = IERT.ReLU inputs(%arg0 : memref<10xf16>) outputs(%buf0 : memref<10xf16>) -> memref<10xf16>
        async.yield %0 : memref<10xf16>
    }

    async.await %t0 : !async.token

    %t1, %f1 = async.execute -> !async.value<memref<10xf16, #map>> {
        %buf1 = memref.subview %buf[10][10][1] : memref<20xf16> to memref<10xf16, #map>
        %1 = IERT.ReLU inputs(%arg1 : memref<10xf16>) outputs(%buf1 : memref<10xf16, #map>) -> memref<10xf16, #map>
        async.yield %1 : memref<10xf16, #map>
    }

    async.await %t1 : !async.token

    %t3, %f3 = async.execute(
                %f0 as %0 : !async.value<memref<10xf16>>,
                %f1 as %1 : !async.value<memref<10xf16, #map>>
            ) -> !async.value<memref<20xf16>> {
        %2 = IERT.ConcatView inputs(%0, %1 : memref<10xf16>, memref<10xf16, #map>) outputs(%buf : memref<20xf16>) -> memref<20xf16>
        %3 = IERT.Copy inputs(%2 : memref<20xf16>) outputs(%arg2 : memref<20xf16>) -> memref<20xf16>
        async.yield %3 : memref<20xf16>
    }

    %3 = async.await %f3 : !async.value<memref<20xf16>>
    return %3 : memref<20xf16>

    // CHECK:       [[T0:%.+]], [[F0:%.+]] = async.execute
    // CHECK-SAME:          -> !async.value<memref<10xf16>> {
    // CHECK:           {{%.+}} = IERT.ReLU inputs(%arg0 : memref<10xf16>)

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK-SAME:          [[T0]]
    // CHECK-SAME:          -> !async.value<memref<10xf16, #map>> {
    // CHECK:           {{%.+}} = IERT.ReLU inputs(%arg1 : memref<10xf16>)

    // CHECK:       [[T3:%.+]], [[F3:%.+]] = async.execute
    // CHECK-SAME:          [[T1]]
    // CHECK-SAME:          [[F0]] as [[VAL0:%.+]]: !async.value<memref<10xf16>>
    // CHECK-SAME:          [[F1]] as [[VAL1:%.+]]: !async.value<memref<10xf16, #map>>
    // CHECK-SAME:          -> !async.value<memref<20xf16>> {
    // CHECK:           {{%.+}} = IERT.ConcatView
    // CHECK-SAME:          inputs([[VAL0]], [[VAL1]] : memref<10xf16>, memref<10xf16, #map>)

    // CHECK:       [[VAL3:%.+]] = async.await [[F3]] : !async.value<memref<20xf16>>
    // CHECK:       return [[VAL3]]
}

// -----

#map = affine_map<(d0) -> (d0 + 10)>

// CHECK-LABEL: @IndependentBranchesParallelSched
func @IndependentBranchesParallelSched(%arg0: memref<10xf16>, %arg1: memref<10xf16>, %arg2: memref<20xf16>) -> memref<20xf16> {
    %buf = IERT.StaticAlloc<0> -> memref<20xf16>

    %t0, %f0 = async.execute -> !async.value<memref<10xf16>> {
        %buf0 = memref.subview %buf[0][10][1] : memref<20xf16> to memref<10xf16>
        %0 = IERT.ReLU inputs(%arg0 : memref<10xf16>) outputs(%buf0 : memref<10xf16>) -> memref<10xf16>
        async.yield %0 : memref<10xf16>
    }

    %t1, %f1 = async.execute -> !async.value<memref<10xf16, #map>> {
        %buf1 = memref.subview %buf[10][10][1] : memref<20xf16> to memref<10xf16, #map>
        %1 = IERT.ReLU inputs(%arg1 : memref<10xf16>) outputs(%buf1 : memref<10xf16, #map>) -> memref<10xf16, #map>
        async.yield %1 : memref<10xf16, #map>
    }

    %t3, %f3 = async.execute(
                %f0 as %0 : !async.value<memref<10xf16>>,
                %f1 as %1 : !async.value<memref<10xf16, #map>>
            ) -> !async.value<memref<20xf16>> {
        %2 = IERT.ConcatView inputs(%0, %1 : memref<10xf16>, memref<10xf16, #map>) outputs(%buf : memref<20xf16>) -> memref<20xf16>
        %3 = IERT.Copy inputs(%2 : memref<20xf16>) outputs(%arg2 : memref<20xf16>) -> memref<20xf16>
        async.yield %3 : memref<20xf16>
    }

    %3 = async.await %f3 : !async.value<memref<20xf16>>
    return %3 : memref<20xf16>

    // CHECK:       [[T0:%.+]], [[F0:%.+]] = async.execute
    // CHECK-SAME:          -> !async.value<memref<10xf16>> {
    // CHECK:           {{%.+}} = IERT.ReLU inputs(%arg0 : memref<10xf16>)

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK-SAME:          -> !async.value<memref<10xf16, #map>> {
    // CHECK:           {{%.+}} = IERT.ReLU inputs(%arg1 : memref<10xf16>)

    // CHECK:       [[T3:%.+]], [[F3:%.+]] = async.execute
    // CHECK-SAME:          [[T0]], [[T1]]
    // CHECK-SAME:          [[F0]] as [[VAL0:%.+]]: !async.value<memref<10xf16>>
    // CHECK-SAME:          [[F1]] as [[VAL1:%.+]]: !async.value<memref<10xf16, #map>>
    // CHECK-SAME:          -> !async.value<memref<20xf16>> {
    // CHECK:           {{%.+}} = IERT.ConcatView
    // CHECK-SAME:          inputs([[VAL0]], [[VAL1]] : memref<10xf16>, memref<10xf16, #map>)

    // CHECK:       [[VAL3:%.+]] = async.await [[F3]] : !async.value<memref<20xf16>>
    // CHECK:       return [[VAL3]]
}

// -----

// CHECK-LABEL: @TwoOutputs
func @TwoOutputs(%arg0: memref<2xf16>, %arg1: memref<2xf16>, %arg2: memref<2xf16>) -> (memref<2xf16>, memref<2xf16>) {
    %0 = IERT.Constant memref<2xf16> = dense<1.0> : tensor<2xf32>

    %t1, %f1 = async.execute -> !async.value<memref<2xf16>> {
        %buf1 = IERT.StaticAlloc<0> -> memref<2xf16>
        %1 = IERT.ReLU inputs(%arg0 : memref<2xf16>) outputs(%buf1 : memref<2xf16>) -> memref<2xf16>
        async.yield %1 : memref<2xf16>
    }

    async.await %t1 : !async.token

    %t2, %f2 = async.execute -> !async.value<memref<2xf16>> {
        %buf2 = IERT.StaticAlloc<4> -> memref<2xf16>
        %2 = IERT.ReLU inputs(%0 : memref<2xf16>) outputs(%buf2 : memref<2xf16>) -> memref<2xf16>
        async.yield %2 : memref<2xf16>
    }

    async.await %t2 : !async.token

    %t3, %f3 = async.execute(%f1 as %1 : !async.value<memref<2xf16>>) -> !async.value<memref<2xf16>> {
        %3 = IERT.Copy inputs(%1 : memref<2xf16>) outputs(%arg1 : memref<2xf16>) -> memref<2xf16>
        async.yield %3 : memref<2xf16>
    }

    %3 = async.await %f3 : !async.value<memref<2xf16>>

    %t4, %f4 = async.execute(%f2 as %2 : !async.value<memref<2xf16>>) -> !async.value<memref<2xf16>> {
        %4 = IERT.Copy inputs(%2 : memref<2xf16>) outputs(%arg2 : memref<2xf16>) -> memref<2xf16>
        async.yield %4 : memref<2xf16>
    }

    %4 = async.await %f4 : !async.value<memref<2xf16>>

    return %3, %4 : memref<2xf16>, memref<2xf16>

    // CHECK:       [[CST:%.+]] = IERT.Constant

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK:           {{%.+}} = IERT.StaticAlloc<0>

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute
    // CHECK-SAME:          [[T1]]
    // CHECK:           {{%.+}} = IERT.StaticAlloc<4>

    // CHECK:       [[T3:%.+]], [[F3:%.+]] = async.execute
    // CHECK-SAME:          [[T2]]
    // CHECK-SAME:          [[F1]] as [[VAL1:%.+]]: !async.value<memref<2xf16>>
    // CHECK:           {{%.+}} = IERT.Copy inputs([[VAL1]] : memref<2xf16>) outputs(%arg1 : memref<2xf16>)

    // CHECK:       [[T4:%.+]], [[F4:%.+]] = async.execute
    // CHECK-SAME:          [[T3]]
    // CHECK-SAME:          [[F2]] as [[VAL2:%.+]]: !async.value<memref<2xf16>>
    // CHECK:           {{%.+}} = IERT.Copy inputs([[VAL2]] : memref<2xf16>) outputs(%arg2 : memref<2xf16>)

    // CHECK:       [[VAL3:%.+]] = async.await [[F3]]
    // CHECK:       [[VAL4:%.+]] = async.await [[F4]]
    // CHECK:       return [[VAL3]], [[VAL4]]
}
