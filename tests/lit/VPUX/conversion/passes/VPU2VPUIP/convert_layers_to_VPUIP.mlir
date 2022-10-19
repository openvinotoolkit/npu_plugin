// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-layers-to-VPUIP --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @ConstantLayer
func @ConstantLayer() -> memref<1x2x2x2xf16, @CMX_NN> {
    %0 = const.Declare tensor<1x2x2x2xf16> = #const.Content<dense<1.0> : tensor<1x2x2x2xf16>>
    %1 = VPU.Copy(%0) : tensor<1x2x2x2xf16> -> tensor<1x2x2x2xf16, {mem_space = @CMX_NN}>
    %2 = builtin.unrealized_conversion_cast %1 : tensor<1x2x2x2xf16, {mem_space = @CMX_NN}> to memref<1x2x2x2xf16, @CMX_NN>
    return %2: memref<1x2x2x2xf16, @CMX_NN>
    // CHECK:       [[VAR0:%.*]] = const.Declare memref<1x2x2x2xf16>
    // CHECK-SAME:      = #const.Content<dense<1.000000e+00> : tensor<1x2x2x2xf16>>

    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<1x2x2x2xf16, @CMX_NN>
    // CHECK:       [[VAR2:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[VAR0]] : memref<1x2x2x2xf16>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<1x2x2x2xf16, @CMX_NN>) -> memref<1x2x2x2xf16, @CMX_NN>

    // CHECK: return [[VAR2]] : memref<1x2x2x2xf16, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributedTensor = type !VPU.DistributedTensor<
    1x128x16x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4
}>

!OutputDistributedTensor = type !VPU.DistributedTensor<
    1x128x16x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4
}>

// CHECK-LABEL: @DistributedCast
func @DistributedCast(%arg0: memref<1x128x16x16xf16, #NHWC, @CMX_NN>) -> memref<1x128x16x16xf16, #NHWC, @CMX_NN> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x128x16x16xf16, #NHWC, @CMX_NN> to !InputDistributedTensor
    %1 = VPU.DistributedCast(%0 : !InputDistributedTensor) -> !OutputDistributedTensor
    %2 = builtin.unrealized_conversion_cast %1 : !OutputDistributedTensor to memref<1x128x16x16xf16, #NHWC, @CMX_NN>
    return %2 : memref<1x128x16x16xf16, #NHWC, @CMX_NN>

    // CHECK:       VPUIP.DistributedCast inputs(
    // CHECK-SAME:         !VPUIP.DistributedBuffer<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputBufferDistributed = type !VPUIP.DistributedBuffer<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4
}>

!OutputBufferDistributed = type !VPUIP.DistributedBuffer<
    1x64x8x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4
}>

!InputTensorDistributed = type !VPU.DistributedTensor<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4
}>

!OutputTensorDistributed = type !VPU.DistributedTensor<
    1x64x8x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4
}>

// CHECK-LABEL: @Slice
func @Slice(%arg0: !InputBufferDistributed) -> !OutputBufferDistributed {
    %0 = builtin.unrealized_conversion_cast %arg0 : !InputBufferDistributed to !InputTensorDistributed
    %1 = VPU.Slice %0 [0, 0, 0, 0] [1, 64, 8, 16]: !InputTensorDistributed to !OutputTensorDistributed
    %2 = builtin.unrealized_conversion_cast %1 : !OutputTensorDistributed to !OutputBufferDistributed
    return %2 : !OutputBufferDistributed

    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 64, 8, 16] : !VPUIP.DistributedBuffer<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}> to !VPUIP.DistributedBuffer<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>

    // CHECK:       [[ALLOC_DISTRIBUTED:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x8x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>

    // CHECK:       [[CLUSTER_COPY:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:       inputs([[SUBVIEW]] as [[ARG1:%.+]]: memref<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN>)
    // CHECK-SAME:       outputs([[ALLOC_DISTRIBUTED]] as [[ARG2:%.+]]: memref<1x64x8x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x64x8x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
    // CHECK:       VPUIP.Copy
    // CHECK-SAME:       inputs([[ARG1]] : memref<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN>)
    // CHECK-SAME:      outputs([[ARG2]] : memref<1x64x8x16xf16, #NHWC, @CMX_NN>)

    // CHECK:       return [[CLUSTER_COPY]] :
    // CHECK-SAME:       !VPUIP.DistributedBuffer<1x64x8x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputBufferDistributed = type !VPUIP.DistributedBuffer<
    1x64x8x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4
}>

!OutputBufferDistributed = type !VPUIP.DistributedBuffer<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4
}>

!InputTensorDistributed = type !VPU.DistributedTensor<
    1x64x8x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4
}>

!OutputTensorDistributed = type !VPU.DistributedTensor<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4
}>

// CHECK-LABEL: @Concat
func @Concat(%arg0: !InputBufferDistributed, %arg1: !InputBufferDistributed) -> !OutputBufferDistributed {
    %0 = builtin.unrealized_conversion_cast %arg0 : !InputBufferDistributed to !InputTensorDistributed
    %1 = builtin.unrealized_conversion_cast %arg1 : !InputBufferDistributed to !InputTensorDistributed
    %2 = VPU.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]]}: !InputTensorDistributed, !InputTensorDistributed -> !OutputTensorDistributed
    %3 = builtin.unrealized_conversion_cast %2 : !OutputTensorDistributed to !OutputBufferDistributed
    return %3 : !OutputBufferDistributed

    // CHECK:       [[ALLOC_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>

    // CHECK:       [[SUBVIEW0:%.+]] = VPUIP.SubView [[ALLOC_BUF]] [0, 0, 0, 0] [1, 64, 8, 16] : !VPUIP.DistributedBuffer<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}> to !VPUIP.DistributedBuffer<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
    // CHECK:       [[CLUSTER_COPY0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:       inputs(%arg0 as [[ARG1:%.+]]: memref<1x64x8x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:       outputs([[SUBVIEW0]] as [[ARG2:%.+]]: memref<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN>)
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
    // CHECK:       VPUIP.Copy
    // CHECK-SAME:       inputs([[ARG1]] : memref<1x64x8x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[ARG2]] : memref<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN>)


    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[ALLOC_BUF]] [0, 0, 8, 0] [1, 64, 8, 16] : !VPUIP.DistributedBuffer<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}> to !VPUIP.DistributedBuffer<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
    // CHECK:       [[CLUSTER_COPY1:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:       inputs(%arg1 as [[ARG3:%.+]]: memref<1x64x8x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:       outputs([[SUBVIEW1]] as [[ARG4:%.+]]: memref<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN>)
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
    // CHECK:       VPUIP.Copy
    // CHECK-SAME:       inputs([[ARG3]] : memref<1x64x8x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[ARG4]] : memref<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN>)


    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:       inputs([[CLUSTER_COPY0]], [[CLUSTER_COPY1]] : !VPUIP.DistributedBuffer<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>, !VPUIP.DistributedBuffer<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:       outputs([[ALLOC_BUF]] : !VPUIP.DistributedBuffer<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>) -> !VPUIP.DistributedBuffer<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>

    // CHECK:       return [[CONCAT]] :
    // CHECK-SAME:       !VPUIP.DistributedBuffer<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
}
