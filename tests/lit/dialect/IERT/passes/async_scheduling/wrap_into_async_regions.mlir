// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=VPU3700" --wrap-into-async-regions %s | FileCheck %s

// CHECK-LABEL: @LinearGraph
func @LinearGraph(%arg0: memref<100xf16>, %arg1: memref<100xf16>) -> memref<100xf16> {
    %0 = IERT.StaticAlloc<0> -> memref<100xf16>
    %1 = IERT.ReLU inputs(%arg0 : memref<100xf16>) outputs(%0 : memref<100xf16>) -> memref<100xf16>
    %2 = IERT.Copy inputs(%1 : memref<100xf16>) outputs(%arg1 : memref<100xf16>) -> memref<100xf16>
    return %2 : memref<100xf16>

    // CHECK:       [[VAR0:%.+]] = IERT.StaticAlloc<0>

    // CHECK:       [[TOKEN1:%.+]], [[FUTURE1:%.+]] = async.execute -> !async.value<memref<100xf16>>
    // CHECK-SAME:          IERT.executor = "SHAVE_UPA"
    // CHECK-SAME:          IERT.num_units = 16
    // CHECK:           [[INNER_VAR1:%.+]] = IERT.ReLU inputs(%arg0 : memref<100xf16>) outputs([[VAR0]] : memref<100xf16>)
    // CHECK:           async.yield [[INNER_VAR1]]

    // CHECK:       [[VAR1:%.+]] = async.await [[FUTURE1]]

    // CHECK:       [[TOKEN2:%.+]], [[FUTURE2:%.+]] = async.execute -> !async.value<memref<100xf16>>
    // CHECK-SAME:          IERT.executor = "DMA_NN"
    // CHECK-SAME:          IERT.num_units = 1
    // CHECK:           [[INNER_VAR2:%.+]] = IERT.Copy inputs([[VAR1]] : memref<100xf16>) outputs(%arg1 : memref<100xf16>)
    // CHECK:           async.yield [[INNER_VAR2]]

    // CHECK:       [[VAR2:%.+]] = async.await [[FUTURE2]]

    // CHECK:       return [[VAR2]]
}

// -----

#map = affine_map<(d0) -> (d0 + 50)>

// CHECK-LABEL: @ConcatView
func @ConcatView(%arg0: memref<50xf16>, %arg1: memref<100xf16>) -> memref<100xf16> {
    %0 = memref.subview %arg1[0] [50] [1] : memref<100xf16> to memref<50xf16>
    %1 = IERT.ReLU inputs(%arg0 : memref<50xf16>) outputs(%0 : memref<50xf16>) -> memref<50xf16>

    %2 = memref.subview %arg1[50] [50] [1] : memref<100xf16> to memref<50xf16, #map>
    %3 = IERT.Copy inputs(%arg0 : memref<50xf16>) outputs(%2 : memref<50xf16, #map>) -> memref<50xf16, #map>

    %4 = IERT.ConcatView inputs(%1, %3 : memref<50xf16>, memref<50xf16, #map>) outputs(%arg1 : memref<100xf16>) -> memref<100xf16>
    return %4 : memref<100xf16>

    // CHECK:       [[VAR0:%.+]] = memref.subview %arg1[0] [50] [1]
    // CHECK:       [[TOKEN1:%.+]], [[FUTURE1:%.+]] = async.execute -> !async.value<memref<50xf16>>
    // CHECK-SAME:          IERT.executor = "SHAVE_UPA"
    // CHECK-SAME:          IERT.num_units = 16
    // CHECK:           [[INNER_VAR1:%.+]] = IERT.ReLU inputs(%arg0 : memref<50xf16>) outputs([[VAR0]] : memref<50xf16>)
    // CHECK:           async.yield [[INNER_VAR1]]
    // CHECK:       [[VAR1:%.+]] = async.await [[FUTURE1]]

    // CHECK:       [[VAR2:%.+]] = memref.subview %arg1[50] [50] [1]
    // CHECK:       [[TOKEN3:%.+]], [[FUTURE3:%.+]] = async.execute -> !async.value<memref<50xf16, #map>>
    // CHECK-SAME:          IERT.executor = "DMA_NN"
    // CHECK-SAME:          IERT.num_units = 1
    // CHECK:           [[INNER_VAR3:%.+]] = IERT.Copy inputs(%arg0 : memref<50xf16>) outputs([[VAR2]] : memref<50xf16, #map>)
    // CHECK:           async.yield [[INNER_VAR3]]
    // CHECK:       [[VAR3:%.+]] = async.await [[FUTURE3]]

    // CHECK:       [[VAR4:%.+]] = IERT.ConcatView inputs([[VAR1]], [[VAR3]] : memref<50xf16>, memref<50xf16, #map>) outputs(%arg1 : memref<100xf16>)

    // CHECK:       return [[VAR4]] : memref<100xf16>
}
