// RUN: vpux-opt %s --split-input-file --verify-diagnostics

func @main(%arg0: memref<97x257x673xui8>, %arg1: memref<97x257x673xui8>) -> memref<97x257x673xui8> {
// expected-error@+1 {{The size of the transaction 16777217 Byte is greater than the limit 16777216 Byte}}
    %0 = VPUIP.NNDMA inputs(%arg0 : memref<97x257x673xui8>) outputs(%arg1 : memref<97x257x673xui8>) -> memref<97x257x673xui8>
    return %0 : memref<97x257x673xui8>
}

// -----

func @main(%arg0: memref<97x257x673xui8>, %arg1: memref<97x257x673xui8>) -> memref<97x257x673xui8> {
// expected-error@+1 {{The size of the transaction 16777217 Byte is greater than the limit 16777216 Byte}}
    %0 = VPUIP.UPADMA inputs(%arg0 : memref<97x257x673xui8>) outputs(%arg1 : memref<97x257x673xui8>) -> memref<97x257x673xui8>
    return %0 : memref<97x257x673xui8>
}
