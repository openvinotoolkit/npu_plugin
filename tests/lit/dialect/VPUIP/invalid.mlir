// RUN: vpux-opt %s --split-input-file --verify-diagnostics

func @verifyNNDMA(%arg0: memref<97x257x673xui8>, %arg1: memref<97x257x673xui8>) -> memref<97x257x673xui8> {
// expected-error@+1 {{The size of the DMA transaction 16777217 Byte for a [97, 257, 673] tensor is greater than the limit 16777216 Byte}}
    %0 = VPUIP.NNDMA inputs(%arg0 : memref<97x257x673xui8>) outputs(%arg1 : memref<97x257x673xui8>) -> memref<97x257x673xui8>
    return %arg0 : memref<97x257x673xui8>
}

// -----

func @main() {
// expected-error@+1 {{BufferSection 'NetworkInput' is not compatible with memory space '@CMX_NN'}}
    %buf0 = VPURT.DeclareBuffer "NetworkInput" <0> -> memref<10xf16, @CMX_NN>
}

// -----

func @main() {
// expected-error@+1 {{Output type must have DDR memory space}}
    %buf0 = VPURT.DeclareBuffer "DDR" <0> -> memref<10xf16>
}

// -----

func @main() {
// expected-error@+1 {{Wrong section index value for DDR memory space: '1'}}
    %buf0 = VPURT.DeclareBuffer "DDR" [1] <0> -> memref<10xf16, @DDR>
}

// -----

func @main() {
// expected-error@+1 {{Array of section indexes is supported for DDR memory space}}
    %buf0 = VPURT.DeclareBuffer "DDR" [0, 1] <0> -> memref<10xf16, @DDR>
}

// -----

func @main() {
// expected-error@+1 {{Section index is missing}}
    %buf0 = VPURT.DeclareBuffer "CMX_NN" <0> -> memref<10xf16, @CMX_NN>
}

// -----

func @main() {
// expected-error@+1 {{Array of section indexes is supported only for distributed buffer type}}
    %buf0 = VPURT.DeclareBuffer "CMX_NN" [0, 1] <0> -> memref<10xf16, @CMX_NN>
}

// -----

func @main() {
// expected-error@+1 {{Section index '0' and memory space index '1' mismatch}}
    %buf0 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<10xf16, [@CMX_NN, 1]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x16x33x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

func @main() {
// expected-error@+1 {{Empty section index is not supported}}
    %buf0 = VPURT.DeclareBuffer "CMX_NN" [] <0> -> !InputDistributed
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x16x33x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

func @main() {
// expected-error@+1 {{Number of clusters '2' and section indexes '1' mismatch}}
    %buf0 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> !InputDistributed
}
