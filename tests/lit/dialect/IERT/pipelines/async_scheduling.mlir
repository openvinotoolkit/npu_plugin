// RUN: vpux-opt --init-compiler="vpu-arch=VPUX30XX" --async-scheduling %s | FileCheck %s

func @main(%arg0: memref<1x100xf16>, %arg1: memref<100xf16>) -> memref<100xf16> {
    %buf0 = memref.alloc() : memref<1x100xf16>
    %buf1 = memref.alloc() : memref<1x100xf16>

    %0 = IERT.ReLU inputs(%arg0 : memref<1x100xf16>) outputs(%buf0 : memref<1x100xf16>) -> memref<1x100xf16>
    %1 = IERT.ReLU inputs(%0 : memref<1x100xf16>) outputs(%buf1 : memref<1x100xf16>) -> memref<1x100xf16>
    %2 = IERT.GenericReshape inputs(%1 : memref<1x100xf16>) -> memref<100xf16>
    %3 = IERT.Copy inputs(%2 : memref<100xf16>) outputs(%arg1 : memref<100xf16>) -> memref<100xf16>

    return %3: memref<100xf16>

    // CHECK:       [[BUF0:%.+]] = memref.alloc() : memref<1x100xf16>
    // CHECK:       [[BUF1:%.+]] = memref.alloc() : memref<1x100xf16>

    // CHECK:       [[T0:%.+]], [[F0:%.+]] = async.execute
    // CHECK-SAME:          IERT.executor = @SHAVE_UPA
    // CHECK:           [[VAR0:%.+]] = IERT.ReLU inputs(%arg0 : memref<1x100xf16>) outputs([[BUF0]] : memref<1x100xf16>)
    // CHECK:           async.yield [[VAR0]]

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK-SAME:          [[T0]]
    // CHECK-SAME:          ([[F0]] as [[VAR0:%.+]]: !async.value<memref<1x100xf16>>)
    // CHECK-SAME:          IERT.executor = @SHAVE_UPA
    // CHECK:           [[VAR1:%.+]] = IERT.ReLU inputs([[VAR0]] : memref<1x100xf16>) outputs([[BUF1]] : memref<1x100xf16>)
    // CHECK:           async.yield [[VAR1]]

    // CHECK:       [[T3:%.+]], [[F3:%.+]] = async.execute
    // CHECK-SAME:          [[T1]]
    // CHECK-SAME:          ([[F1]] as [[VAR1:%.+]]: !async.value<memref<1x100xf16>>)
    // CHECK-SAME:          IERT.executor = @DMA_NN
    // CHECK:           [[VAR2:%.+]] = IERT.GenericReshape inputs([[VAR1]] : memref<1x100xf16>) -> memref<100xf16>
    // CHECK:           [[VAR3:%.+]] = IERT.Copy inputs([[VAR2]] : memref<100xf16>) outputs(%arg1 : memref<100xf16>)
    // CHECK:           async.yield [[VAR3]]

    // CHECK:       [[VAR3:%.+]] = async.await [[F3]]
    // CHECK:       return [[VAR3]]
}
