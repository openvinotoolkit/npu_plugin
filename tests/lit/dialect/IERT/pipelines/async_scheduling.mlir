// RUN: vpux-opt --set-compile-params="vpu-arch=VPU3700" --async-scheduling %s | FileCheck %s

func @main(%arg0: memref<1x100xf16>, %arg1: memref<100xf16>) -> memref<100xf16> {
    %0 = IERT.StaticAlloc<0> -> memref<1x100xf16>
    %1 = IERT.ReLU inputs(%arg0 : memref<1x100xf16>) outputs(%0 : memref<1x100xf16>) -> memref<1x100xf16>
    %2 = IERT.StaticAlloc<200> -> memref<1x100xf16>
    %3 = IERT.ReLU inputs(%1 : memref<1x100xf16>) outputs(%2 : memref<1x100xf16>) -> memref<1x100xf16>
    %4 = IERT.GenericReshape inputs(%3 : memref<1x100xf16>) -> memref<100xf16>
    %5 = IERT.Copy inputs(%4 : memref<100xf16>) outputs(%arg1 : memref<100xf16>) -> memref<100xf16>
    return %5: memref<100xf16>

    // CHECK:       [[VAL0:%.+]] = IERT.StaticAlloc<0>

    // CHECK:       [[TOKEN1:%.+]], [[FUTURE1:%.+]] = async.execute
    // CHECK-SAME:          IERT.executor = "SHAVE_UPA"
    // CHECK-SAME:          IERT.num_units = 16
    // CHECK:           [[INNER1:%.+]] = IERT.ReLU inputs(%arg0 : memref<1x100xf16>) outputs([[VAL0]] : memref<1x100xf16>)
    // CHECK:           async.yield [[INNER1]]

    // CHECK:       [[VAL2:%.+]] = IERT.StaticAlloc<200>

    // CHECK:       [[TOKEN3:%.+]], [[FUTURE3:%.+]] = async.execute
    // CHECK-SAME:          [[TOKEN1]]
    // CHECK-SAME:          ([[FUTURE1]] as [[VAL1:%.+]]: !async.value<memref<1x100xf16>>)
    // CHECK-SAME:          IERT.executor = "SHAVE_UPA"
    // CHECK-SAME:          IERT.num_units = 16
    // CHECK:           [[INNER3:%.+]] = IERT.ReLU inputs([[VAL1]] : memref<1x100xf16>) outputs([[VAL2]] : memref<1x100xf16>)
    // CHECK:           async.yield [[INNER3]]

    // CHECK:       [[TOKEN5:%.+]], [[FUTURE5:%.+]] = async.execute
    // CHECK-SAME:          [[TOKEN3]]
    // CHECK-SAME:          ([[FUTURE3]] as [[VAL3:%.+]]: !async.value<memref<1x100xf16>>)
    // CHECK-SAME:          IERT.executor = "DMA_NN"
    // CHECK-SAME:          IERT.num_units = 1
    // CHECK:           [[VAL4:%.+]] = IERT.GenericReshape inputs([[VAL3]] : memref<1x100xf16>) -> memref<100xf16>
    // CHECK:           [[INNER5:%.+]] = IERT.Copy inputs([[VAL4]] : memref<100xf16>) outputs(%arg1 : memref<100xf16>)
    // CHECK:           async.yield [[INNER5]]

    // CHECK:       [[VAL5:%.+]] = async.await [[FUTURE5]]
    // CHECK:       return [[VAL5]]
}
