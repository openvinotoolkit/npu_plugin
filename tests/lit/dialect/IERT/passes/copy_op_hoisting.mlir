// RUN: vpux-opt --copy-op-hoisting %s | FileCheck %s

// CHECK-LABEL: @CopyFromBlockArgumentNoChange
func @CopyFromBlockArgumentNoChange(%arg0: memref<8xf16, @CMX_NN>, %arg1: memref<8xf16>, %arg2: memref<8xf16>)
        -> (memref<8xf16>, memref<8xf16>) {
    %0 = IERT.Copy inputs(%arg0 : memref<8xf16, @CMX_NN>) outputs(%arg1 : memref<8xf16>) -> memref<8xf16>
    %1 = IERT.Copy inputs(%arg0 : memref<8xf16, @CMX_NN>) outputs(%arg2 : memref<8xf16>) -> memref<8xf16>
    return %0, %1 : memref<8xf16>, memref<8xf16>

    // CHECK: [[VAR0:%.+]] = IERT.Copy inputs(%arg0 : memref<8xf16, @CMX_NN>) outputs(%arg1 : memref<8xf16>)
    // CHECK: [[VAR1:%.+]] = IERT.Copy inputs(%arg0 : memref<8xf16, @CMX_NN>) outputs(%arg2 : memref<8xf16>)
    // CHECK: return [[VAR0]], [[VAR1]]
}

// CHECK-LABEL: @CopyToBlockArgumentNoChange
func @CopyToBlockArgumentNoChange(%arg0: memref<8xf16>, %arg1: memref<8xf16>) -> memref<8xf16> {
    %0 = memref.alloc() : memref<8xf16, @CMX_NN>
    %1 = IERT.ReLU inputs(%arg0 : memref<8xf16>) outputs(%0 : memref<8xf16, @CMX_NN>) -> memref<8xf16, @CMX_NN>
    %2 = IERT.Copy inputs(%1 : memref<8xf16, @CMX_NN>) outputs(%arg1 : memref<8xf16>) -> memref<8xf16>
    return %2 : memref<8xf16>

    // CHECK: [[VAR0:%.+]] = memref.alloc() : memref<8xf16, @CMX_NN>
    // CHECK: [[VAR1:%.+]] = IERT.ReLU inputs(%arg0 : memref<8xf16>) outputs([[VAR0]] : memref<8xf16, @CMX_NN>)
    // CHECK: [[VAR2:%.+]] = IERT.Copy inputs([[VAR1]] : memref<8xf16, @CMX_NN>) outputs(%arg1 : memref<8xf16>)
    // CHECK: return [[VAR2]]
}

// CHECK-LABEL: @CopyToTempBufNoChange
func @CopyToTempBufNoChange(%arg0: memref<8xf16>, %arg1: memref<8xf16>) -> memref<8xf16> {
    %0 = memref.alloc() : memref<8xf16, @CMX_NN>
    %1 = IERT.ReLU inputs(%arg0 : memref<8xf16>) outputs(%0 : memref<8xf16, @CMX_NN>) -> memref<8xf16, @CMX_NN>
    %2 = memref.alloc() : memref<8xf16>
    %3 = IERT.Copy inputs(%1 : memref<8xf16, @CMX_NN>) outputs(%2 : memref<8xf16>) -> memref<8xf16>
    %4 = IERT.ReLU inputs(%3 : memref<8xf16>) outputs(%arg1 : memref<8xf16>) -> memref<8xf16>
    return %4 : memref<8xf16>

    // CHECK: [[VAR0:%.+]] = memref.alloc() : memref<8xf16, @CMX_NN>
    // CHECK: [[VAR1:%.+]] = IERT.ReLU inputs(%arg0 : memref<8xf16>) outputs([[VAR0]] : memref<8xf16, @CMX_NN>)
    // CHECK: [[VAR2:%.+]] = memref.alloc() : memref<8xf16>
    // CHECK: [[VAR3:%.+]] = IERT.Copy inputs([[VAR1]] : memref<8xf16, @CMX_NN>) outputs([[VAR2]] : memref<8xf16>)
    // CHECK: [[VAR4:%.+]] = IERT.ReLU inputs([[VAR3]] : memref<8xf16>) outputs(%arg1 : memref<8xf16>)
    // CHECK: return [[VAR4]]
}

// CHECK-LABEL: @CopyToBlockArgumentMoveCopyOnly
func @CopyToBlockArgumentMoveCopyOnly(%arg0: memref<8xf16>, %arg1: memref<8xf16>, %arg2: memref<8xf16>) -> (memref<8xf16>, memref<8xf16>) {
    %0 = memref.alloc() : memref<8xf16, @CMX_NN>
    %1 = memref.alloc() : memref<8xf16, @CMX_NN>
    %2 = IERT.ReLU inputs(%arg0 : memref<8xf16>) outputs(%0 : memref<8xf16, @CMX_NN>) -> memref<8xf16, @CMX_NN>
    %3 = IERT.ReLU inputs(%arg0 : memref<8xf16>) outputs(%1 : memref<8xf16, @CMX_NN>) -> memref<8xf16, @CMX_NN>
    %4 = IERT.Copy inputs(%2 : memref<8xf16, @CMX_NN>) outputs(%arg1 : memref<8xf16>) -> memref<8xf16>
    %5 = IERT.Copy inputs(%3 : memref<8xf16, @CMX_NN>) outputs(%arg2 : memref<8xf16>) -> memref<8xf16>
    return %4, %5 : memref<8xf16>, memref<8xf16>

    // CHECK: [[VAR0:%.+]] = memref.alloc() : memref<8xf16, @CMX_NN>
    // CHECK: [[VAR1:%.+]] = memref.alloc() : memref<8xf16, @CMX_NN>
    // CHECK: [[VAR2:%.+]] = IERT.ReLU inputs(%arg0 : memref<8xf16>) outputs([[VAR0]] : memref<8xf16, @CMX_NN>)
    // CHECK: [[VAR3:%.+]] = IERT.Copy inputs([[VAR2]] : memref<8xf16, @CMX_NN>) outputs(%arg1 : memref<8xf16>)
    // CHECK: [[VAR4:%.+]] = IERT.ReLU inputs(%arg0 : memref<8xf16>) outputs([[VAR1]] : memref<8xf16, @CMX_NN>)
    // CHECK: [[VAR5:%.+]] = IERT.Copy inputs([[VAR4]] : memref<8xf16, @CMX_NN>) outputs(%arg2 : memref<8xf16>)
    // CHECK: return [[VAR3]], [[VAR5]]
}

// CHECK-LABEL: @CopyToTempBufMoveCopyOnly
func @CopyToTempBufMoveCopyOnly(%arg0: memref<8xf16>, %arg1: memref<8xf16>, %arg2: memref<8xf16>) -> (memref<8xf16>, memref<8xf16>) {
    %0 = memref.alloc() : memref<8xf16, @CMX_NN>
    %1 = memref.alloc() : memref<8xf16, @CMX_NN>
    %2 = memref.alloc() : memref<8xf16>
    %3 = memref.alloc() : memref<8xf16>
    %4 = IERT.ReLU inputs(%arg0 : memref<8xf16>) outputs(%0 : memref<8xf16, @CMX_NN>) -> memref<8xf16, @CMX_NN>
    %5 = IERT.ReLU inputs(%arg0 : memref<8xf16>) outputs(%1 : memref<8xf16, @CMX_NN>) -> memref<8xf16, @CMX_NN>
    %6 = IERT.Copy inputs(%4 : memref<8xf16, @CMX_NN>) outputs(%2 : memref<8xf16>) -> memref<8xf16>
    %7 = IERT.Copy inputs(%5 : memref<8xf16, @CMX_NN>) outputs(%3 : memref<8xf16>) -> memref<8xf16>
    %8 = IERT.ReLU inputs(%6 : memref<8xf16>) outputs(%arg1 : memref<8xf16>) -> memref<8xf16>
    %9 = IERT.ReLU inputs(%7 : memref<8xf16>) outputs(%arg2 : memref<8xf16>) -> memref<8xf16>
    return %8, %9 : memref<8xf16>, memref<8xf16>

    // CHECK: [[VAR0:%.+]] = memref.alloc() : memref<8xf16, @CMX_NN>
    // CHECK: [[VAR1:%.+]] = memref.alloc() : memref<8xf16, @CMX_NN>
    // CHECK: [[VAR2:%.+]] = memref.alloc() : memref<8xf16>
    // CHECK: [[VAR3:%.+]] = memref.alloc() : memref<8xf16>
    // CHECK: [[VAR4:%.+]] = IERT.ReLU inputs(%arg0 : memref<8xf16>) outputs([[VAR0]] : memref<8xf16, @CMX_NN>)
    // CHECK: [[VAR5:%.+]] = IERT.Copy inputs([[VAR4]] : memref<8xf16, @CMX_NN>) outputs([[VAR2]] : memref<8xf16>)
    // CHECK: [[VAR6:%.+]] = IERT.ReLU inputs(%arg0 : memref<8xf16>) outputs([[VAR1]] : memref<8xf16, @CMX_NN>)
    // CHECK: [[VAR7:%.+]] = IERT.Copy inputs([[VAR6]] : memref<8xf16, @CMX_NN>) outputs([[VAR3]] : memref<8xf16>)
    // CHECK: [[VAR8:%.+]] = IERT.ReLU inputs([[VAR5]] : memref<8xf16>) outputs(%arg1 : memref<8xf16>)
    // CHECK: [[VAR9:%.+]] = IERT.ReLU inputs([[VAR7]] : memref<8xf16>) outputs(%arg2 : memref<8xf16>)
    // CHECK: return [[VAR8]], [[VAR9]]
}

// CHECK-LABEL: @CopyToTempBufMoveWithAlloc
func @CopyToTempBufMoveWithAlloc(%arg0: memref<8xf16>, %arg1: memref<8xf16>, %arg2: memref<8xf16>) -> (memref<8xf16>, memref<8xf16>) {
    %0 = memref.alloc() : memref<8xf16, @CMX_NN>
    %1 = IERT.ReLU inputs(%arg0 : memref<8xf16>) outputs(%0 : memref<8xf16, @CMX_NN>) -> memref<8xf16, @CMX_NN>
    %2 = memref.alloc() : memref<8xf16, @CMX_NN>
    %3 = IERT.ReLU inputs(%arg0 : memref<8xf16>) outputs(%2 : memref<8xf16, @CMX_NN>) -> memref<8xf16, @CMX_NN>
    %4 = memref.alloc() : memref<8xf16>
    %5 = IERT.Copy inputs(%1 : memref<8xf16, @CMX_NN>) outputs(%4 : memref<8xf16>) -> memref<8xf16>
    %6 = memref.alloc() : memref<8xf16>
    %7 = IERT.Copy inputs(%3 : memref<8xf16, @CMX_NN>) outputs(%6 : memref<8xf16>) -> memref<8xf16>
    %8 = IERT.ReLU inputs(%5 : memref<8xf16>) outputs(%arg1 : memref<8xf16>) -> memref<8xf16>
    %9 = IERT.ReLU inputs(%7 : memref<8xf16>) outputs(%arg2 : memref<8xf16>) -> memref<8xf16>
    return %8, %9 : memref<8xf16>, memref<8xf16>

    // CHECK: [[VAR0:%.+]] = memref.alloc() : memref<8xf16, @CMX_NN>
    // CHECK: [[VAR1:%.+]] = IERT.ReLU inputs(%arg0 : memref<8xf16>) outputs([[VAR0]] : memref<8xf16, @CMX_NN>)
    // CHECK: [[VAR2:%.+]] = memref.alloc() : memref<8xf16>
    // CHECK: [[VAR3:%.+]] = IERT.Copy inputs([[VAR1]] : memref<8xf16, @CMX_NN>) outputs([[VAR2]] : memref<8xf16>)
    // CHECK: [[VAR4:%.+]] = memref.alloc() : memref<8xf16, @CMX_NN>
    // CHECK: [[VAR5:%.+]] = IERT.ReLU inputs(%arg0 : memref<8xf16>) outputs([[VAR4]] : memref<8xf16, @CMX_NN>)
    // CHECK: [[VAR6:%.+]] = memref.alloc() : memref<8xf16>
    // CHECK: [[VAR7:%.+]] = IERT.Copy inputs([[VAR5]] : memref<8xf16, @CMX_NN>) outputs([[VAR6]] : memref<8xf16>)
    // CHECK: [[VAR8:%.+]] = IERT.ReLU inputs([[VAR3]] : memref<8xf16>) outputs(%arg1 : memref<8xf16>)
    // CHECK: [[VAR9:%.+]] = IERT.ReLU inputs([[VAR7]] : memref<8xf16>) outputs(%arg2 : memref<8xf16>)
    // CHECK: return [[VAR8]], [[VAR9]]
}

// CHECK-LABEL: @CopyToBlockArgumentSubView
func @CopyToBlockArgumentSubView(%arg0: memref<8xf16>, %arg1: memref<16xf16>) -> memref<16xf16> {
    %0 = memref.alloc() : memref<8xf16, @CMX_NN>
    %1 = memref.alloc() : memref<8xf16, @CMX_NN>
    %2 = IERT.ReLU inputs(%arg0 : memref<8xf16>) outputs(%0 : memref<8xf16, @CMX_NN>) -> memref<8xf16, @CMX_NN>
    %3 = IERT.ReLU inputs(%arg0 : memref<8xf16>) outputs(%1 : memref<8xf16, @CMX_NN>) -> memref<8xf16, @CMX_NN>
    %4 = IERT.SubView %arg1 [0] [8] : memref<16xf16> to memref<8xf16>
    %5 = IERT.Copy inputs(%2 : memref<8xf16, @CMX_NN>) outputs(%4 : memref<8xf16>) -> memref<8xf16>
    %6 = IERT.SubView %arg1 [8] [8] : memref<16xf16> to memref<8xf16>
    %7 = IERT.Copy inputs(%3 : memref<8xf16, @CMX_NN>) outputs(%6 : memref<8xf16>) -> memref<8xf16>
    %8 = IERT.ConcatView inputs(%5, %7 : memref<8xf16>, memref<8xf16>) outputs(%arg1 : memref<16xf16>) -> memref<16xf16>
    return %8 : memref<16xf16>

    // CHECK: [[VAR0:%.+]] = memref.alloc() : memref<8xf16, @CMX_NN>
    // CHECK: [[VAR1:%.+]] = memref.alloc() : memref<8xf16, @CMX_NN>
    // CHECK: [[VAR2:%.+]] = IERT.ReLU inputs(%arg0 : memref<8xf16>) outputs([[VAR0]] : memref<8xf16, @CMX_NN>)
    // CHECK: [[VAR3:%.+]] = IERT.SubView %arg1 [0] [8]
    // CHECK: [[VAR4:%.+]] = IERT.Copy inputs([[VAR2]] : memref<8xf16, @CMX_NN>) outputs([[VAR3]] : memref<8xf16>)
    // CHECK: [[VAR5:%.+]] = IERT.ReLU inputs(%arg0 : memref<8xf16>) outputs([[VAR1]] : memref<8xf16, @CMX_NN>)
    // CHECK: [[VAR6:%.+]] = IERT.SubView %arg1 [8] [8]
    // CHECK: [[VAR7:%.+]] = IERT.Copy inputs([[VAR5]] : memref<8xf16, @CMX_NN>) outputs([[VAR6]] : memref<8xf16>)
    // CHECK: [[VAR8:%.+]] = IERT.ConcatView inputs([[VAR4]], [[VAR7]] : memref<8xf16>, memref<8xf16>) outputs(%arg1 : memref<16xf16>)
    // CHECK: return [[VAR8]]
}

// CHECK-LABEL: @CopyToTempBufSubViewMoveCopyOnly
func @CopyToTempBufSubViewMoveCopyOnly(%arg0: memref<8xf16>, %arg1: memref<16xf16>) -> memref<16xf16> {
    %0 = memref.alloc() : memref<8xf16, @CMX_NN>
    %1 = memref.alloc() : memref<8xf16, @CMX_NN>
    %2 = memref.alloc() : memref<16xf16>
    %3 = IERT.SubView %2 [0] [8] : memref<16xf16> to memref<8xf16>
    %4 = IERT.SubView %2 [8] [8] : memref<16xf16> to memref<8xf16>
    %5 = IERT.ReLU inputs(%arg0 : memref<8xf16>) outputs(%0 : memref<8xf16, @CMX_NN>) -> memref<8xf16, @CMX_NN>
    %6 = IERT.ReLU inputs(%arg0 : memref<8xf16>) outputs(%1 : memref<8xf16, @CMX_NN>) -> memref<8xf16, @CMX_NN>
    %7 = IERT.Copy inputs(%5 : memref<8xf16, @CMX_NN>) outputs(%3 : memref<8xf16>) -> memref<8xf16>
    %8 = IERT.Copy inputs(%6 : memref<8xf16, @CMX_NN>) outputs(%4 : memref<8xf16>) -> memref<8xf16>
    %9 = IERT.ConcatView inputs(%7, %8 : memref<8xf16>, memref<8xf16>) outputs(%2 : memref<16xf16>) -> memref<16xf16>
    %10 = IERT.ReLU inputs(%9 : memref<16xf16>) outputs(%arg1 : memref<16xf16>) -> memref<16xf16>
    return %10 : memref<16xf16>

    // CHECK: [[VAR0:%.+]] = memref.alloc() : memref<8xf16, @CMX_NN>
    // CHECK: [[VAR1:%.+]] = memref.alloc() : memref<8xf16, @CMX_NN>
    // CHECK: [[VAR2:%.+]] = memref.alloc() : memref<16xf16>
    // CHECK: [[VAR3:%.+]] = IERT.SubView [[VAR2]] [0] [8]
    // CHECK: [[VAR4:%.+]] = IERT.SubView [[VAR2]] [8] [8]
    // CHECK: [[VAR5:%.+]] = IERT.ReLU inputs(%arg0 : memref<8xf16>) outputs([[VAR0]] : memref<8xf16, @CMX_NN>)
    // CHECK: [[VAR6:%.+]] = IERT.Copy inputs([[VAR5]] : memref<8xf16, @CMX_NN>) outputs([[VAR3]] : memref<8xf16>)
    // CHECK: [[VAR7:%.+]] = IERT.ReLU inputs(%arg0 : memref<8xf16>) outputs([[VAR1]] : memref<8xf16, @CMX_NN>)
    // CHECK: [[VAR8:%.+]] = IERT.Copy inputs([[VAR7]] : memref<8xf16, @CMX_NN>) outputs([[VAR4]] : memref<8xf16>)
    // CHECK: [[VAR9:%.+]] = IERT.ConcatView inputs([[VAR6]], [[VAR8]] : memref<8xf16>, memref<8xf16>) outputs([[VAR2]] : memref<16xf16>)
    // CHECK: [[VAR10:%.+]] = IERT.ReLU inputs([[VAR9]] : memref<16xf16>) outputs(%arg1 : memref<16xf16>)
    // CHECK: return [[VAR10]]
}

// CHECK-LABEL: @CopyToTempBufSubViewMoveWithAlloc
func @CopyToTempBufSubViewMoveWithAlloc(%arg0: memref<8xf16>, %arg1: memref<16xf16>) -> memref<16xf16> {
    %0 = memref.alloc() : memref<8xf16, @CMX_NN>
    %1 = IERT.ReLU inputs(%arg0 : memref<8xf16>) outputs(%0 : memref<8xf16, @CMX_NN>) -> memref<8xf16, @CMX_NN>
    %2 = memref.alloc() : memref<8xf16, @CMX_NN>
    %3 = IERT.ReLU inputs(%arg0 : memref<8xf16>) outputs(%2 : memref<8xf16, @CMX_NN>) -> memref<8xf16, @CMX_NN>
    %4 = memref.alloc() : memref<16xf16>
    %5 = IERT.SubView %4 [0] [8] : memref<16xf16> to memref<8xf16>
    %6 = IERT.Copy inputs(%1 : memref<8xf16, @CMX_NN>) outputs(%5 : memref<8xf16>) -> memref<8xf16>
    %7 = IERT.SubView %4 [8] [8] : memref<16xf16> to memref<8xf16>
    %8 = IERT.Copy inputs(%3 : memref<8xf16, @CMX_NN>) outputs(%7 : memref<8xf16>) -> memref<8xf16>
    %9 = IERT.ConcatView inputs(%6, %8 : memref<8xf16>, memref<8xf16>) outputs(%4 : memref<16xf16>) -> memref<16xf16>
    %10 = IERT.ReLU inputs(%9 : memref<16xf16>) outputs(%arg1 : memref<16xf16>) -> memref<16xf16>
    return %10 : memref<16xf16>

    // CHECK: [[VAR0:%.+]] = memref.alloc() : memref<8xf16, @CMX_NN>
    // CHECK: [[VAR1:%.+]] = IERT.ReLU inputs(%arg0 : memref<8xf16>) outputs([[VAR0]] : memref<8xf16, @CMX_NN>)
    // CHECK: [[VAR2:%.+]] = memref.alloc() : memref<16xf16>
    // CHECK: [[VAR3:%.+]] = IERT.SubView [[VAR2]] [0] [8]
    // CHECK: [[VAR4:%.+]] = IERT.Copy inputs([[VAR1]] : memref<8xf16, @CMX_NN>) outputs([[VAR3]] : memref<8xf16>)
    // CHECK: [[VAR5:%.+]] = memref.alloc() : memref<8xf16, @CMX_NN>
    // CHECK: [[VAR6:%.+]] = IERT.ReLU inputs(%arg0 : memref<8xf16>) outputs([[VAR5]] : memref<8xf16, @CMX_NN>)
    // CHECK: [[VAR7:%.+]] = IERT.SubView [[VAR2]] [8] [8]
    // CHECK: [[VAR8:%.+]] = IERT.Copy inputs([[VAR6]] : memref<8xf16, @CMX_NN>) outputs([[VAR7]] : memref<8xf16>)
    // CHECK: [[VAR9:%.+]] = IERT.ConcatView inputs([[VAR4]], [[VAR8]] : memref<8xf16>, memref<8xf16>) outputs([[VAR2]] : memref<16xf16>)
    // CHECK: [[VAR10:%.+]] = IERT.ReLU inputs([[VAR9]] : memref<16xf16>) outputs(%arg1 : memref<16xf16>)
    // CHECK: return [[VAR10]]
}
