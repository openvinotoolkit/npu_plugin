//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-layers-to-VPUIP --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @ConstantLayer
func @ConstantLayer() -> memref<1x2x2x2xf16, @CMX_NN> {
    %0 = const.Declare tensor<1x2x2x2xf16> = dense<1.0> : tensor<1x2x2x2xf16>
    %1 = VPU.Copy(%0) {out_mem_space = @CMX_NN} : tensor<1x2x2x2xf16> -> tensor<1x2x2x2xf16, {mem_space = @CMX_NN}>
    %2 = builtin.unrealized_conversion_cast %1 : tensor<1x2x2x2xf16, {mem_space = @CMX_NN}> to memref<1x2x2x2xf16, @CMX_NN>
    return %2: memref<1x2x2x2xf16, @CMX_NN>
    // CHECK:       [[VAR0:%.*]] = const.Declare memref<1x2x2x2xf16>
    // CHECK-SAME:      = dense<1.000000e+00> : tensor<1x2x2x2xf16>

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
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputSparseBuffer = type !VPUIP.SparseBuffer<
    data=memref<1x32x16x16xf16, #NHWC, @CMX_NN>,
    sparsity_map=memref<1x32x16x16xi1, #NHWC, @CMX_NN>,
    is_weights
>

!InputSparseTensor = type !VPU.SparseTensor<
    data=tensor<1x32x16x16xf16, {order = #NHWC, mem_space = @CMX_NN}>,
    sparsity_map=tensor<1x32x16x16xi1, {order = #NHWC, mem_space = @CMX_NN}>,
    is_weights
>

!OutputSparseBuffer = type !VPUIP.SparseBuffer<
    data=memref<1x4x8x16xf16, #NHWC, @CMX_NN>,
    sparsity_map=memref<1x1x1x512xi1, #NHWC, @CMX_NN>,
    is_weights
>

!OutputSparseTensor = type !VPU.SparseTensor<
    data=tensor<1x4x8x16xf16, {order = #NHWC, mem_space = @CMX_NN}>,
    sparsity_map=tensor<1x1x1x512xi1, {order = #NHWC, mem_space = @CMX_NN}>,
    is_weights
>

// CHECK-LABEL: @SliceSparseMemRefBuf
func @SliceSparseMemRefBuf(%arg0: memref<1x32x16x16xf16, #NHWC, @CMX_NN>, %arg1: memref<1x32x16x16xi1, #NHWC, @CMX_NN>) -> !OutputSparseBuffer {
    %sb = VPUIP.GroupSparseBuffer(%arg0, %arg1) {is_weights} -> !InputSparseBuffer
    %st = builtin.unrealized_conversion_cast %sb : !InputSparseBuffer to !InputSparseTensor
    %slice_res = VPU.Slice %st [0, 0, 0, 0] [1, 4, 8, 16]: !InputSparseTensor to !OutputSparseTensor
    %res = builtin.unrealized_conversion_cast %slice_res : !OutputSparseTensor to !OutputSparseBuffer
    return %res: !OutputSparseBuffer

    // CHECK: [[SUBVIEW_RES:%.+]] = VPUIP.SubView {{%.+}} [0, 0, 0, 0] [1, 4, 8, 16] : !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x32x16x16xi1, #NHWC, @CMX_NN>, is_weights>
    // CHECK-SAME:                                                                      to
    // CHECK-SAME:                                                                     !VPUIP.SparseBuffer<data=memref<1x4x8x16xf16, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                                                                         sparsity_map=memref<1x1x1x512xi1, #NHWC, @CMX_NN>, is_weights>

    // CHECK: [[ALLOC_DATA_BUF:%.+]] = memref.alloc() : memref<1x4x8x16xf16, #NHWC, @CMX_NN>
    // CHECK: [[ALLOC_SM_BUF:%.+]] = memref.alloc() : memref<1x1x1x512xi1, #NHWC, @CMX_NN>
    // CHECK: [[ALLOC_SPARSE_BUF:%.+]] = VPUIP.GroupSparseBuffer([[ALLOC_DATA_BUF]], [[ALLOC_SM_BUF]]) {is_weights} -> !VPUIP.SparseBuffer<data=memref<1x4x8x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x1x1x512xi1, #NHWC, @CMX_NN>, is_weights>

    // CHECK: {{%.+}} = VPUIP.Copy inputs([[SUBVIEW_RES]] : !VPUIP.SparseBuffer<data=memref<1x4x8x16xf16, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                                              sparsity_map=memref<1x1x1x512xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:                 outputs([[ALLOC_SPARSE_BUF]] : !VPUIP.SparseBuffer<data=memref<1x4x8x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x1x1x512xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:                 ->
    // CHECK-SAME:                 !VPUIP.SparseBuffer<data=memref<1x4x8x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x1x1x512xi1, #NHWC, @CMX_NN>, is_weights>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputBufferDistributed = type !VPUIP.DistributedBuffer<
    1x32x16x16xf16, #NHWC, @CMX_NN, { mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2}
>

!InputSMBufferDistributed = type !VPUIP.DistributedBuffer<
    1x32x16x16xi1, #NHWC, @CMX_NN, { mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2}
>

!InputSparseBufferDistributed = type !VPUIP.SparseBuffer<
    data=!InputBufferDistributed, sparsity_map=!InputSMBufferDistributed, is_weights
>

!OutputBufferDistributed = type !VPUIP.DistributedBuffer<
    1x4x8x16xf16, #NHWC, @CMX_NN, { mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64 }
>

!OutputSMBufferDistributed = type !VPUIP.DistributedBuffer<
    1x1x1x512xi1, #NHWC, @CMX_NN, { mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64 }
>

!OutputSparseBufferDistributed = type !VPUIP.SparseBuffer<
    data=!OutputBufferDistributed, sparsity_map=!OutputSMBufferDistributed, is_weights
>

!InputTensorDistributed = type !VPU.DistributedTensor<
    1x32x16x16xf16, #NHWC, @CMX_NN, { mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2}
>

!InputSMTensorDistributed = type !VPU.DistributedTensor<
    1x32x16x16xi1, #NHWC, @CMX_NN, { mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2}
>

!InputSparseTensorDistributed = type !VPU.SparseTensor<
    data=!InputTensorDistributed, sparsity_map=!InputSMTensorDistributed, is_weights
>

!OutputTensorDistributed = type !VPU.DistributedTensor<
    1x4x8x16xf16, #NHWC, @CMX_NN, { mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64 }
>

!OutputSMTensorDistributed = type !VPU.DistributedTensor<
    1x1x1x512xi1, #NHWC, @CMX_NN, { mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64 }
>

!OutputSparseTensorDistributed = type !VPU.SparseTensor<
    data=!OutputTensorDistributed, sparsity_map=!OutputSMTensorDistributed, is_weights
>

// CHECK-LABEL: @SliceSparseDistributedBuf
func @SliceSparseDistributedBuf(%arg0: !InputBufferDistributed, %arg1: !InputSMBufferDistributed) -> !OutputSparseBufferDistributed {
    %sb = VPUIP.GroupSparseBuffer(%arg0, %arg1) {is_weights} -> !VPUIP.SparseBuffer<data=!InputBufferDistributed, sparsity_map=!InputSMBufferDistributed, is_weights>
    %st = builtin.unrealized_conversion_cast %sb : !InputSparseBufferDistributed to !InputSparseTensorDistributed
    %slice_res = VPU.Slice %st [0, 0, 0, 0] [1, 4, 8, 16]: !InputSparseTensorDistributed to !OutputSparseTensorDistributed
    %res = builtin.unrealized_conversion_cast %slice_res : !OutputSparseTensorDistributed to !OutputSparseBufferDistributed
    return %res: !OutputSparseBufferDistributed

    // CHECK:       [[SUBVIEW_RES:%.+]] = VPUIP.SubView {{%.+}} [0, 0, 0, 0] [1, 4, 8, 16] :
    // CHECK-SAME:                           !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x32x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                              sparsity_map=!VPUIP.DistributedBuffer<1x32x16x16xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                              is_weights> to
    // CHECK-SAME:                           !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x4x8x16xf16, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                               sparsity_map=!VPUIP.DistributedBuffer<1x1x1x512xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                               is_weights>

    // CHECK:       [[ALLOC_DATA_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x4x8x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[ALLOC_SM_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x512xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[ALLOC_SPARSE_BUF:%.+]] = VPUIP.GroupSparseBuffer([[ALLOC_DATA_BUF]], [[ALLOC_SM_BUF]]) {is_weights} -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x4x8x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                                                                                                               sparsity_map=!VPUIP.DistributedBuffer<1x1x1x512xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                                                                                                               is_weights>

    // CHECK:       %{{.+}} = VPUIP.NCEClusterTiling inputs([[SUBVIEW_RES]] as [[ARG2:%.+]]: !VPUIP.SparseBuffer<data=memref<1x4x8x16xf16, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                                                                               sparsity_map=memref<1x1x1x512xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:                                   outputs([[ALLOC_SPARSE_BUF]] as [[ARG3:%.+]]: !VPUIP.SparseBuffer<data=memref<1x4x8x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x1x1x512xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:            -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x4x8x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                   sparsity_map=!VPUIP.DistributedBuffer<1x1x1x512xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                   is_weights>
    // CHECK-SAME:   {
    // CHECK:           {{%.+}} = VPUIP.Copy inputs([[ARG2]] : !VPUIP.SparseBuffer<data=memref<1x4x8x16xf16, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                                                 sparsity_map=memref<1x1x1x512xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:                           outputs([[ARG3]] : !VPUIP.SparseBuffer<data=memref<1x4x8x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x1x1x512xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:                           -> !VPUIP.SparseBuffer<data=memref<1x4x8x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x1x1x512xi1, #NHWC, @CMX_NN>, is_weights>
    // CHECK:        }
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

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputSparseBuffer = type !VPUIP.SparseBuffer<
    data=memref<1x32x16x16xf16, #NHWC, @CMX_NN>,
    sparsity_map=memref<1x32x16x16xi1, #NHWC, @CMX_NN>,
    is_weights
>

!InputSparseTensor = type !VPU.SparseTensor<
    data=tensor<1x32x16x16xf16, {order = #NHWC, mem_space = @CMX_NN}>,
    sparsity_map=tensor<1x32x16x16xi1, {order = #NHWC, mem_space = @CMX_NN}>,
    is_weights
>

!OutputSparseBuffer = type !VPUIP.SparseBuffer<
    data=memref<1x32x24x16xf16, #NHWC, @CMX_NN>,
    sparsity_map=memref<1x32x24x16xi1, #NHWC, @CMX_NN>,
    is_weights
>

!OutputSparseTensor = type !VPU.SparseTensor<
    data=tensor<1x32x24x16xf16, {order = #NHWC, mem_space = @CMX_NN}>,
    sparsity_map=tensor<1x32x24x16xi1, {order = #NHWC, mem_space = @CMX_NN}>,
    is_weights
>

// CHECK-LABEL: @ConcatSparseMemRefBuf
func @ConcatSparseMemRefBuf(%arg0: memref<1x32x16x16xf16, #NHWC, @CMX_NN>, %arg1: memref<1x32x16x16xi1, #NHWC, @CMX_NN>,
                            %arg2: memref<1x32x16x16xf16, #NHWC, @CMX_NN>, %arg3: memref<1x32x16x16xi1, #NHWC, @CMX_NN>) -> !OutputSparseBuffer {
    %sb1 = VPUIP.GroupSparseBuffer(%arg0, %arg1) {is_weights} -> !InputSparseBuffer
    %sb2 = VPUIP.GroupSparseBuffer(%arg2, %arg3) {is_weights} -> !InputSparseBuffer

    %st1 = builtin.unrealized_conversion_cast %sb1 : !InputSparseBuffer to !InputSparseTensor
    %st2 = builtin.unrealized_conversion_cast %sb2 : !InputSparseBuffer to !InputSparseTensor

    %conc_res = VPU.Concat(%st1, %st2) {static_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]]}: !InputSparseTensor, !InputSparseTensor -> !OutputSparseTensor
    %res = builtin.unrealized_conversion_cast %conc_res : !OutputSparseTensor to !OutputSparseBuffer
    return %res : !OutputSparseBuffer

    // CHECK:       [[ALLOC_DATA_BUF:%.+]] = memref.alloc() : memref<1x32x24x16xf16, #NHWC, @CMX_NN>
    // CHECK:       [[ALLOC_SM_BUF:%.+]] = memref.alloc() : memref<1x32x24x16xi1, #NHWC, @CMX_NN>
    // CHECK:       [[ALLOC_SPARSE_BUF:%.+]] = VPUIP.GroupSparseBuffer([[ALLOC_DATA_BUF]], [[ALLOC_SM_BUF]]) {is_weights} ->
    // CHECK-SAME:                             !VPUIP.SparseBuffer<data=memref<1x32x24x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x32x24x16xi1, #NHWC, @CMX_NN>, is_weights>

    // CHECK:       [[SUBVIEW_1_RES:%.+]] = VPUIP.SubView [[ALLOC_SPARSE_BUF]] [0, 0, 0, 0] [1, 32, 16, 16] :
    // CHECK-SAME:                              !VPUIP.SparseBuffer<data=memref<1x32x24x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x32x24x16xi1, #NHWC, @CMX_NN>, is_weights>
    // CHECK-SAME:                              to
    // CHECK-SAME:                              !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, {order = #NHWC, strides = [12288, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                                  sparsity_map=memref<1x1x1x8192xi1, #NHWC, @CMX_NN>, is_weights>

    // CHECK:       [[COPY_1_RES:%.+]] = VPUIP.Copy inputs({{%.+}} : !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x32x16x16xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:                                  outputs([[SUBVIEW_1_RES]] : !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, {order = #NHWC, strides = [12288, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                                                                  sparsity_map=memref<1x1x1x8192xi1, #NHWC, @CMX_NN>,  is_weights>)
    // CHECK-SAME:                                  -> !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, {order = #NHWC, strides = [12288, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                                         sparsity_map=memref<1x1x1x8192xi1, #NHWC, @CMX_NN>, is_weights>

    // CHECK:       [[SUBVIEW_2_RES:%.+]] = VPUIP.SubView [[ALLOC_SPARSE_BUF]] [0, 0, 8, 0] [1, 32, 16, 16] :
    // CHECK-SAME:                              !VPUIP.SparseBuffer<data=memref<1x32x24x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x32x24x16xi1, #NHWC, @CMX_NN>, is_weights>
    // CHECK-SAME:                              to
    // CHECK-SAME:                              !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, {order = #NHWC, strides = [12288, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                                  sparsity_map=memref<1x1x1x8192xi1, #NHWC, @CMX_NN>, is_weights>

    // CHECK:       [[COPY_2_RES:%.+]] = VPUIP.Copy inputs({{%.+}} : !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x32x16x16xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:                                  outputs([[SUBVIEW_2_RES]] : !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, {order = #NHWC, strides = [12288, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                                                                  sparsity_map=memref<1x1x1x8192xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:                                  -> !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, {order = #NHWC, strides = [12288, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                                         sparsity_map=memref<1x1x1x8192xi1, #NHWC, @CMX_NN>, is_weights>
        
    // CHECK:       {{%.+}} = VPUIP.ConcatView inputs([[COPY_1_RES]], [[COPY_2_RES]] :
    // CHECK-SAME:                                    !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, {order = #NHWC, strides = [12288, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                                        sparsity_map=memref<1x1x1x8192xi1, #NHWC, @CMX_NN>, is_weights>,
    // CHECK-SAME:                                    !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, {order = #NHWC, strides = [12288, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                                        sparsity_map=memref<1x1x1x8192xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:                             outputs([[ALLOC_SPARSE_BUF]] : !VPUIP.SparseBuffer<data=memref<1x32x24x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x32x24x16xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:                             -> !VPUIP.SparseBuffer<data=memref<1x32x24x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x32x24x16xi1, #NHWC, @CMX_NN>, is_weights>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputBufferDistributed = type !VPUIP.DistributedBuffer<
    1x32x16x16xf16, #NHWC, @CMX_NN, { mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 }
>

!InputSMBufferDistributed = type !VPUIP.DistributedBuffer<
    1x32x16x16xi1, #NHWC, @CMX_NN, { mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 }
>

!InputSparseBufferDistributed = type !VPUIP.SparseBuffer<
    data=!InputBufferDistributed,
    sparsity_map=!InputSMBufferDistributed,
    is_weights
>

!OutputBufferDistributed = type !VPUIP.DistributedBuffer<
    1x32x24x16xf16, #NHWC, @CMX_NN, { mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64 }
>

!OutputSMBufferDistributed = type !VPUIP.DistributedBuffer<
    1x32x24x16xi1, #NHWC, @CMX_NN, { mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64 }
>

!OutputSparseBufferDistributed = type !VPUIP.SparseBuffer<
    data=!OutputBufferDistributed,
    sparsity_map=!OutputSMBufferDistributed,
    is_weights
>

!InputTensorDistributed = type !VPU.DistributedTensor<
    1x32x16x16xf16, #NHWC, @CMX_NN, { mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 }
>

!InputSMTensorDistributed = type !VPU.DistributedTensor<
    1x32x16x16xi1, #NHWC, @CMX_NN, { mode = "SEGMENTED",  num_tiles = [1, 1, 2, 1], num_clusters = 2}
>

!InputSparseTensorDistributed = type !VPU.SparseTensor<
    data=!InputTensorDistributed,
    sparsity_map=!InputSMTensorDistributed,
    is_weights
>

!OutputTensorDistributed = type !VPU.DistributedTensor<
    1x32x24x16xf16, #NHWC, @CMX_NN, { mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64 }
>

!OutputSMTensorDistributed = type !VPU.DistributedTensor<
    1x32x24x16xi1, #NHWC, @CMX_NN, { mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64 }
>

!OutputSparseTensorDistributed = type !VPU.SparseTensor<
    data=!OutputTensorDistributed,
    sparsity_map=!OutputSMTensorDistributed,
    is_weights
>

// CHECK-LABEL: @ConcatSparseDistributedBuf
func @ConcatSparseDistributedBuf(%arg0: !InputBufferDistributed, %arg1: !InputSMBufferDistributed,
                                 %arg2: !InputBufferDistributed, %arg3: !InputSMBufferDistributed) -> !OutputSparseBufferDistributed {
    %sb1 = VPUIP.GroupSparseBuffer(%arg0, %arg1) {is_weights} -> !VPUIP.SparseBuffer<data=!InputBufferDistributed, sparsity_map=!InputSMBufferDistributed, is_weights>
    %sb2 = VPUIP.GroupSparseBuffer(%arg2, %arg3) {is_weights} -> !VPUIP.SparseBuffer<data=!InputBufferDistributed, sparsity_map=!InputSMBufferDistributed, is_weights>

    %st1 = builtin.unrealized_conversion_cast %sb1 : !InputSparseBufferDistributed to !InputSparseTensorDistributed
    %st2 = builtin.unrealized_conversion_cast %sb2 : !InputSparseBufferDistributed to !InputSparseTensorDistributed

    %conc_res = VPU.Concat(%st1, %st2) {static_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]]}: !InputSparseTensorDistributed, !InputSparseTensorDistributed -> !OutputSparseTensorDistributed
    %res = builtin.unrealized_conversion_cast %conc_res : !OutputSparseTensorDistributed to !OutputSparseBufferDistributed
    return %res : !OutputSparseBufferDistributed

    // CHECK:       [[ALLOC_DATA_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x24x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[ALLOC_SM_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x24x16xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[ALLOC_SPARSE_BUF:%.+]] = VPUIP.GroupSparseBuffer([[ALLOC_DATA_BUF]], [[ALLOC_SM_BUF]]) {is_weights} -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x32x24x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                                                                                                               sparsity_map=!VPUIP.DistributedBuffer<1x32x24x16xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                                                                                                               is_weights>

    // CHECK:       [[SUBVIEW_1_RES:%.+]] = VPUIP.SubView [[ALLOC_SPARSE_BUF]] [0, 0, 0, 0] [1, 32, 16, 16] :
    // CHECK-SAME:                              !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x32x24x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                                  sparsity_map=!VPUIP.DistributedBuffer<1x32x24x16xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                                  is_weights>
    // CHECK-SAME:                               to !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [12288, 1, 512, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                                      sparsity_map=!VPUIP.DistributedBuffer<1x1x1x8192xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                                      is_weights>

    // CHECK:       [[COPY_1_RES:%.+]] = VPUIP.NCEClusterTiling inputs({{%.+}} as [[ARG2:%.+]]: !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x32x16x16xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:                                              outputs([[SUBVIEW_1_RES]] as [[ARG3:%.+]]: !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, {order = #NHWC, strides = [12288, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                                                                                             sparsity_map=memref<1x1x1x8192xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:                          -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [12288, 1, 512, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                                 sparsity_map=!VPUIP.DistributedBuffer<1x1x1x8192xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                                 is_weights>
    // CHECK-SAME:                 {
    // CHECK:                          {{%.+}} = VPUIP.Copy inputs([[ARG2]] : !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x32x16x16xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:                                          outputs([[ARG3]] : !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, {order = #NHWC, strides = [12288, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                                                                 sparsity_map=memref<1x1x1x8192xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:                                          -> !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, {order = #NHWC, strides = [12288, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                                                 sparsity_map=memref<1x1x1x8192xi1, #NHWC, @CMX_NN>, is_weights>
    // CHECK:                       }

    // CHECK:       [[SUBVIEW_2_RES:%.+]] = VPUIP.SubView [[ALLOC_SPARSE_BUF]] [0, 0, 8, 0] [1, 32, 16, 16] :
    // CHECK-SAME:                              !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x32x24x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                                  sparsity_map=!VPUIP.DistributedBuffer<1x32x24x16xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                                  is_weights>
    // CHECK-SAME:                               to !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [12288, 1, 512, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                                      sparsity_map=!VPUIP.DistributedBuffer<1x1x1x8192xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                                      is_weights>

    // CHECK:       [[COPY_2_RES:%.+]] = VPUIP.NCEClusterTiling inputs({{%.+}} as [[ARG2:%.+]]: !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x32x16x16xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:                                              outputs([[SUBVIEW_2_RES]] as [[ARG3:%.+]]: !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, {order = #NHWC, strides = [12288, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                                                                                             sparsity_map=memref<1x1x1x8192xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:                          -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [12288, 1, 512, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                                 sparsity_map=!VPUIP.DistributedBuffer<1x1x1x8192xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                                 is_weights>
    // CHECK-SAME:                 {
    // CHECK:                          {{%.+}} = VPUIP.Copy inputs([[ARG2]] : !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x32x16x16xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:                                          outputs([[ARG3]] : !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, {order = #NHWC, strides = [12288, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                                                                 sparsity_map=memref<1x1x1x8192xi1, #NHWC, @CMX_NN>, is_weights>)
    // CHECK-SAME:                                          -> !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, {order = #NHWC, strides = [12288, 1, 512, 32]}, @CMX_NN>,
    // CHECK-SAME:                                                                 sparsity_map=memref<1x1x1x8192xi1, #NHWC, @CMX_NN>, is_weights>
    // CHECK:                      }

    // CHECK:       {{%.+}} = VPUIP.ConcatView inputs([[COPY_1_RES]], [[COPY_2_RES]] : !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [12288, 1, 512, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                                                                         sparsity_map=!VPUIP.DistributedBuffer<1x1x1x8192xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                                                                         is_weights>,
    // CHECK-SAME:                                     !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [12288, 1, 512, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                                         sparsity_map=!VPUIP.DistributedBuffer<1x1x1x8192xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                                         is_weights>)
    // CHECK-SAME:                             outputs([[ALLOC_SPARSE_BUF]] : !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x32x24x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                                                                sparsity_map=!VPUIP.DistributedBuffer<1x32x24x16xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                                                                is_weights>)
    // CHECK-SAME:            -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x32x24x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                   sparsity_map=!VPUIP.DistributedBuffer<1x32x24x16xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                   is_weights>
}    

// -----

// CHECK-LABEL: @EmbeddingBagOffsetsSum
    func @EmbeddingBagOffsetsSum(%arg0: memref<5x6x4xsi32>, %arg1: memref<4x6x4xsi32>) -> memref<4x6x4xsi32> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<5x6x4xsi32> to tensor<5x6x4xsi32>
    %1 = VPU.EmbeddingBagOffsetsSum(%0) {default_index_value = 4 : i32, indices_value = [0, 1, 2, 2, 3], offsets_value = [0, 0, 2, 2], weights_value = [1.000000e+00, 5.000000e+00, 1.000000e+01, 8.000000e+00, 1.000000e+01]} : tensor<5x6x4xsi32> -> tensor<4x6x4xsi32>
    %2 = builtin.unrealized_conversion_cast %1 : tensor<4x6x4xsi32> to memref<4x6x4xsi32>
    %3 = VPUIP.Copy inputs(%2 : memref<4x6x4xsi32>) outputs(%arg1 : memref<4x6x4xsi32>) -> memref<4x6x4xsi32>
    return %3 : memref<4x6x4xsi32>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<4x6x4xsi32>
    // CHECK:       [[VAR1:%.*]] = VPUIP.EmbeddingBagOffsetsSumUPA {default_index_value = 4 : i32, indices_value = [0, 1, 2, 2, 3], offsets_value = [0, 0, 2, 2], weights_value = [1.000000e+00, 5.000000e+00, 1.000000e+01, 8.000000e+00, 1.000000e+01]} inputs(%arg0 : memref<5x6x4xsi32>) outputs(%0 : memref<4x6x4xsi32>) -> memref<4x6x4xsi32>
    // CHECK:       [[VAR2:%.*]] = VPUIP.Copy inputs(%1 : memref<4x6x4xsi32>) outputs(%arg1 : memref<4x6x4xsi32>) -> memref<4x6x4xsi32>
    // CHECK:       return [[VAR2]] : memref<4x6x4xsi32>
}

// -----
// CHECK-LABEL: @EmbeddingSegmentsSum
func @EmbeddingSegmentsSum(%arg0: memref<5x6x4xsi32>, %arg1: memref<7x6x4xsi32>) -> memref<7x6x4xsi32> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<5x6x4xsi32> to tensor<5x6x4xsi32>
    %1 = VPU.EmbeddingSegmentsSum(%0) {default_index_value = 4 : si32, indices_value = [0, 1, 2, 2, 3], num_segments_value = 7 : si32,
        operand_segment_sizes = dense<[1, 0, 0, 0, 0, 0]> : vector<6xi32>,
        per_sample_weights_value = [1.000000e+00, 5.000000e+00, 1.000000e+01, 8.000000e+00, 1.000000e+01], segment_ids_value = [0, 1, 2, 3, 4]} : tensor<5x6x4xsi32> -> tensor<7x6x4xsi32>
    %2 = builtin.unrealized_conversion_cast %1 : tensor<7x6x4xsi32> to memref<7x6x4xsi32>
    return %2 : memref<7x6x4xsi32>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<7x6x4xsi32>
    // CHECK:       [[VAR1:%.*]] = VPUIP.EmbeddingSegmentsSumUPA {default_index_value = 4 : si32, indices_value = [0, 1, 2, 2, 3], num_segments_value = 7 : si32, per_sample_weights_value = [1.000000e+00, 5.000000e+00, 1.000000e+01, 8.000000e+00, 1.000000e+01], segment_ids_value = [0, 1, 2, 3, 4]} inputs(%arg0 : memref<5x6x4xsi32>) ouputs([[VAR0]] : memref<7x6x4xsi32>) -> memref<7x6x4xsi32>
    // CHECK:       return [[VAR1]] : memref<7x6x4xsi32>
}

// -----

// CHECK-LABEL: @CumSum
func @CumSum(%arg0: memref<1x9xf16>) -> memref<1x9xf16> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x9xf16> to tensor<1x9xf16>
    %1 = VPU.CumSum(%0) {axis_value = 1 : i64 , exclusive, reverse} : tensor<1x9xf16> -> tensor<1x9xf16>
    %2 = builtin.unrealized_conversion_cast %1 : tensor<1x9xf16> to memref<1x9xf16>
    return %2 : memref<1x9xf16>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x9xf16>
    // CHECK:       [[VAR1:%.*]] = VPUIP.CumSumUPA {axis_value = 1 : i64, exclusive, reverse} inputs(%arg0 : memref<1x9xf16>) outputs(%0 : memref<1x9xf16>) -> memref<1x9xf16>
    // CHECK:       return [[VAR1]] : memref<1x9xf16>
}

// -----

// CHECK-LABEL: @DeformablePSROIPooling
 func @DeformablePSROIPooling(%arg0: memref<1x441x8x8xf16>, %arg1: memref<30x5xf16>) -> memref<30x49x3x3xf16> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x441x8x8xf16> to tensor<1x441x8x8xf16>
    %1 = builtin.unrealized_conversion_cast %arg1 : memref<30x5xf16> to tensor<30x5xf16>
    %2 = VPU.DeformablePSROIPooling(%0, %1) {group_size = 3 : i64, mode = "BILINEAR_DEFORMABLE", output_dim = 49 : i64, part_size = 3 : i64, spatial_bins_x = 1 : i64, spatial_bins_y = 1 : i64, spatial_scale = 6.250000e-02 : f64, trans_std = 0.000000e+00 : f64} : tensor<1x441x8x8xf16>, tensor<30x5xf16> -> tensor<30x49x3x3xf16>
    %3 = builtin.unrealized_conversion_cast %2 : tensor<30x49x3x3xf16> to memref<30x49x3x3xf16>
    return %3 : memref<30x49x3x3xf16>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<30x49x3x3xf16>
    // CHECK:       [[VAR1:%.*]] = VPUIP.DeformablePSROIPoolingUPA {group_size = 3 : i64, mode = "BILINEAR_DEFORMABLE", output_dim = 49 : i64, part_size = 3 : i64, spatial_bins_x = 1 : i64, spatial_bins_y = 1 : i64, spatial_scale = 6.250000e-02 : f64, trans_std = 0.000000e+00 : f64} inputs(%arg0 : memref<1x441x8x8xf16>, %arg1 : memref<30x5xf16>) outputs(%0 : memref<30x49x3x3xf16>) -> memref<30x49x3x3xf16>
    // CHECK:       return [[VAR1]] : memref<30x49x3x3xf16>
}

// -----

func @Expand(%arg0: tensor<1x3x4x4xf16>) -> tensor<1x8x4x4xf16> {
    %0 = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 5, 0, 0]} : tensor<1x3x4x4xf16> -> tensor<1x8x4x4xf16>
    return %0 : tensor<1x8x4x4xf16>

    // CHECK:       [[VAR0:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<1x3x4x4xf16> to memref<1x3x4x4xf16>

    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<1x8x4x4xf16>

    // CHECK:       [[OUT:%.*]] = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 5, 0, 0]}
    // CHECK-SAME:      inputs([[VAR0]] : memref<1x3x4x4xf16>) outputs([[VAR1]] : memref<1x8x4x4xf16>) -> memref<1x8x4x4xf16>

    // CHECK:       [[VAR2:%.*]] = builtin.unrealized_conversion_cast [[OUT]] : memref<1x8x4x4xf16> to tensor<1x8x4x4xf16>
    // CHECK:       return [[VAR2]] : tensor<1x8x4x4xf16>
}

// -----

func @ExpandToSubviewWithoutTail(%arg0: tensor<1x4x4x4xf16>) -> tensor<1x8x4x4xf16> {
    %0 = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 4, 0, 0]} : tensor<1x4x4x4xf16> -> tensor<1x8x4x4xf16>
    return %0 : tensor<1x8x4x4xf16>

    // CHECK:       [[VAR0:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<1x4x4x4xf16> to memref<1x4x4x4xf16>

    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<1x8x4x4xf16>

    // CHECK:       [[OUT:%.*]] = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 4, 0, 0]}
    // CHECK-SAME:      inputs([[VAR0]] : memref<1x4x4x4xf16>) outputs([[VAR1]] : memref<1x8x4x4xf16>) -> memref<1x8x4x4xf16>

    // CHECK:       [[VAR2:%.*]] = builtin.unrealized_conversion_cast [[OUT]] : memref<1x8x4x4xf16> to tensor<1x8x4x4xf16>
    // CHECK:       return [[VAR2]] : tensor<1x8x4x4xf16>
}

// -----

func @ExpandToSubviewOnlyWithTail(%arg0: tensor<1x5x4x4xf16>) -> tensor<1x8x4x4xf16> {
    %0 = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 3, 0, 0]} : tensor<1x5x4x4xf16> -> tensor<1x8x4x4xf16>
    return %0 : tensor<1x8x4x4xf16>

    // CHECK:       [[VAR0:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<1x5x4x4xf16> to memref<1x5x4x4xf16>

    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<1x8x4x4xf16>

    // CHECK:       [[OUT:%.*]] = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 3, 0, 0]}
    // CHECK-SAME:     inputs([[VAR0]] : memref<1x5x4x4xf16>) outputs([[VAR1]] : memref<1x8x4x4xf16>) -> memref<1x8x4x4xf16>

    // CHECK:       [[VAR2:%.*]] = builtin.unrealized_conversion_cast [[OUT]] : memref<1x8x4x4xf16> to tensor<1x8x4x4xf16>
    // CHECK:       return [[VAR2]] : tensor<1x8x4x4xf16>
}

// -----

func @ExpandOverWidth(%arg0: tensor<1x3x4x4xf16>) -> tensor<1x3x4x9xf16> {
    %0 = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 5]} : tensor<1x3x4x4xf16> -> tensor<1x3x4x9xf16>
    return %0 : tensor<1x3x4x9xf16>

    // CHECK:       [[VAR0:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<1x3x4x4xf16> to memref<1x3x4x4xf16>

    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<1x3x4x9xf16>

    // CHECK:       [[OUT:%.*]] = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 5]}
    // CHECK-SAME:     inputs([[VAR0]] : memref<1x3x4x4xf16>) outputs([[VAR1]] : memref<1x3x4x9xf16>) -> memref<1x3x4x9xf16>
    // CHECK:       [[VAR2:%.*]] = builtin.unrealized_conversion_cast [[OUT]] : memref<1x3x4x9xf16> to tensor<1x3x4x9xf16>
    // CHECK:       return [[VAR2]] : tensor<1x3x4x9xf16>
}

// -----

func @ExpandOverHeight(%arg0: tensor<1x3x4x4xf16>) -> tensor<1x3x9x4xf16> {
    %0 = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 5, 0]} : tensor<1x3x4x4xf16> -> tensor<1x3x9x4xf16>
    return %0 : tensor<1x3x9x4xf16>

    // CHECK:       [[VAR0:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<1x3x4x4xf16> to memref<1x3x4x4xf16>

    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<1x3x9x4xf16>

    // CHECK:       [[OUT:%.*]] = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 5, 0]}
    // CHECK-SAME:     inputs([[VAR0]] : memref<1x3x4x4xf16>) outputs([[VAR1]] : memref<1x3x9x4xf16>) -> memref<1x3x9x4xf16>

    // CHECK:       [[VAR2:%.*]] = builtin.unrealized_conversion_cast [[OUT]] : memref<1x3x9x4xf16> to tensor<1x3x9x4xf16>
    // CHECK:       return [[VAR2]] : tensor<1x3x9x4xf16>
}

// -----

func @ExpandPadsBeginFullCopy(%arg0: tensor<1x3x4x4xf16>) -> tensor<1x6x4x4xf16> {
    %0 = VPU.Expand(%arg0) {
        pads_begin = [0, 3, 0, 0],
        pads_end = [0, 0, 0, 0]
    } : tensor<1x3x4x4xf16> -> tensor<1x6x4x4xf16>

    return %0 : tensor<1x6x4x4xf16>

    // CHECK:       [[VAR0:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<1x3x4x4xf16> to memref<1x3x4x4xf16>

    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<1x6x4x4xf16>

    // CHECK:       [[OUT:%.*]] = VPUIP.Expand {pads_begin = [0, 3, 0, 0], pads_end = [0, 0, 0, 0]}
    // CHECK-SAME:     inputs([[VAR0]] : memref<1x3x4x4xf16>) outputs([[VAR1]] : memref<1x6x4x4xf16>) -> memref<1x6x4x4xf16>

    // CHECK:       [[VAR2:%.*]] = builtin.unrealized_conversion_cast [[OUT]] : memref<1x6x4x4xf16> to tensor<1x6x4x4xf16>
    // CHECK:       return [[VAR2]] : tensor<1x6x4x4xf16>
}

// -----

func @ExpandPadsBeginSliceCopy(%arg0: tensor<1x3x4x4xf16>) -> tensor<1x5x4x4xf16> {
    %0 = VPU.Expand(%arg0) {
        pads_begin = [0, 2, 0, 0],
        pads_end = [0, 0, 0, 0]
    } : tensor<1x3x4x4xf16> -> tensor<1x5x4x4xf16>

    return %0 : tensor<1x5x4x4xf16>

    // CHECK:       [[VAR0:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<1x3x4x4xf16> to memref<1x3x4x4xf16>

    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<1x5x4x4xf16>

    // CHECK:       [[OUT:%.*]] = VPUIP.Expand {pads_begin = [0, 2, 0, 0], pads_end = [0, 0, 0, 0]}
    // CHECK-SAME:     inputs([[VAR0]] : memref<1x3x4x4xf16>) outputs([[VAR1]] : memref<1x5x4x4xf16>) -> memref<1x5x4x4xf16>

    // CHECK:       [[VAR2:%.*]] = builtin.unrealized_conversion_cast [[OUT]] : memref<1x5x4x4xf16> to tensor<1x5x4x4xf16>
    // CHECK:       return [[VAR2]] : tensor<1x5x4x4xf16>
}

// -----

func @ExpandPadsBeginCopiesWithTail(%arg0: tensor<1x3x4x4xf16>) -> tensor<1x11x4x4xf16> {
    %0 = VPU.Expand(%arg0) {
        pads_begin = [0, 8, 0, 0],
        pads_end = [0, 0, 0, 0]
    } : tensor<1x3x4x4xf16> -> tensor<1x11x4x4xf16>

    return %0 : tensor<1x11x4x4xf16>

    // CHECK:       [[VAR0:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<1x3x4x4xf16> to memref<1x3x4x4xf16>

    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<1x11x4x4xf16>

    // CHECK:       [[OUT:%.*]] = VPUIP.Expand {pads_begin = [0, 8, 0, 0], pads_end = [0, 0, 0, 0]}
    // CHECK-SAME:     inputs([[VAR0]] : memref<1x3x4x4xf16>) outputs([[VAR1]] : memref<1x11x4x4xf16>) -> memref<1x11x4x4xf16>

    // CHECK:       [[VAR2:%.*]] = builtin.unrealized_conversion_cast [[OUT]] : memref<1x11x4x4xf16> to tensor<1x11x4x4xf16>
    // CHECK:       return [[VAR2]] : tensor<1x11x4x4xf16>
}

// -----

func @ExpandBeginPadsWithEndPads(%arg0: tensor<1x3x4x4xf16>) -> tensor<1x9x4x4xf16> {
    %0 = VPU.Expand(%arg0) {
        pads_begin = [0, 3, 0, 0],
        pads_end = [0, 3, 0, 0]
    } : tensor<1x3x4x4xf16> -> tensor<1x9x4x4xf16>

    return %0 : tensor<1x9x4x4xf16>

    // CHECK:       [[VAR0:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<1x3x4x4xf16> to memref<1x3x4x4xf16>

    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<1x9x4x4xf16>

    // CHECK:       [[OUT:%.*]] = VPUIP.Expand {pads_begin = [0, 3, 0, 0], pads_end = [0, 3, 0, 0]}
    // CHECK-SAME:     inputs([[VAR0]] : memref<1x3x4x4xf16>) outputs([[VAR1]] : memref<1x9x4x4xf16>) -> memref<1x9x4x4xf16>

    // CHECK:       [[VAR2:%.*]] = builtin.unrealized_conversion_cast [[OUT]] : memref<1x9x4x4xf16> to tensor<1x9x4x4xf16>
    // CHECK:       return [[VAR2]] : tensor<1x9x4x4xf16>
}

// -----

func @StorageElementTable() -> tensor<1x1x16x16xi32> {
    %0 = VPU.StorageElementTable{seDepth = 1, seSize = 32, height = 16, width = 16} -> tensor<1x1x16x16xi32>
    return %0 : tensor<1x1x16x16xi32>

    // CHECK:       [[SETable:%.*]] = VPUIP.StorageElementTable {height = 16 : i64, seDepth = 1 : i64, seSize = 32 : i64, width = 16 : i64} -> memref<1x1x16x16xi32>
    // CHECK:       [[RESULT:%.*]] = builtin.unrealized_conversion_cast [[SETable]] : memref<1x1x16x16xi32> to tensor<1x1x16x16xi32>
    // CHECK:       return [[RESULT]] : tensor<1x1x16x16xi32>

}

// -----

func @ShapeCast(%arg0: tensor<1x3x32x32xf16>) -> tensor<1x16x16x12xf16> {
    %0 = VPU.ShapeCast {shape = [1, 16, 16, 12]} inputs(%arg0 : tensor<1x3x32x32xf16>) -> tensor<1x16x16x12xf16>
    return %0 : tensor<1x16x16x12xf16>

    // CHECK:       [[VAR0:%.+]] = builtin.unrealized_conversion_cast %arg0 : tensor<1x3x32x32xf16> to memref<1x3x32x32xf16>
    // CHECK:       [[VPUIP_SHAPE_CAST:%.+]] = VPUIP.ShapeCast {shape = [1, 16, 16, 12]} inputs([[VAR0]] : memref<1x3x32x32xf16>) -> memref<1x16x16x12xf16>
    // CHECK:       [[VAR1:%.+]] = builtin.unrealized_conversion_cast [[VPUIP_SHAPE_CAST]] : memref<1x16x16x12xf16> to tensor<1x16x16x12xf16>
    // CHECK:       return [[VAR1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @LayoutCast
func @LayoutCast(%arg0: tensor<1x3x32x32xf16, {order = #NWCH}>) -> tensor<1x3x32x32xf16, {order = #NHWC}> {
    %0 = VPU.LayoutCast(%arg0) {
        dst_order = #NHWC
    } : tensor<1x3x32x32xf16, {order = #NWCH}> -> tensor<1x3x32x32xf16, {order = #NHWC}>

    return %0 : tensor<1x3x32x32xf16, {order = #NHWC}>

    // CHECK:       [[TYPE_IN_CAST:%.+]] = builtin.unrealized_conversion_cast %arg0 :
    // CHECK-SAME:      tensor<1x3x32x32xf16, {order = #NWCH}> to memref<1x3x32x32xf16, #NWCH>

    // CHECK:       [[LAYOUT_CAST:%.+]] = VPUIP.PermuteCast {
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NHWC
    // CHECK-SAME:  } inputs([[TYPE_IN_CAST]] : memref<1x3x32x32xf16, #NWCH>) -> memref<1x3x32x32xf16, #NHWC>

    // CHECK:       [[TYPE_OUT_CAST:%.+]] = builtin.unrealized_conversion_cast [[LAYOUT_CAST]] :
    // CHECK-SAME:      memref<1x3x32x32xf16, #NHWC> to tensor<1x3x32x32xf16, {order = #NHWC}>
    // CHECK:       return [[TYPE_OUT_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputBufferDistributed = type !VPUIP.DistributedBuffer<
    1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4}
>

!InputSMBufferDistributed = type !VPUIP.DistributedBuffer<
    1x128x16x16xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4}
>

!InputSparseBufferDistributed = type !VPUIP.SparseBuffer<
    data=!InputBufferDistributed, sparsity_map=!InputSMBufferDistributed
>

!OutputBufferDistributed = type !VPUIP.DistributedBuffer<
    1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4}
>

!OutputSMBufferDistributed = type !VPUIP.DistributedBuffer<
    1x128x16x16xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4}
>

!OutputSparseBufferDistributed = type !VPUIP.SparseBuffer<
    data=!OutputBufferDistributed, sparsity_map=!OutputSMBufferDistributed
>

!InputTensorDistributed = type !VPU.DistributedTensor<
    1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4}
>

!InputSMTensorDistributed = type !VPU.DistributedTensor<
    1x128x16x16xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4}
>

!InputSparseTensorDistributed = type !VPU.SparseTensor<
    data=!InputTensorDistributed, sparsity_map=!InputSMTensorDistributed
>

!OutputTensorDistributed = type !VPU.DistributedTensor<
    1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4}
>

!OutputSMTensorDistributed = type !VPU.DistributedTensor<
    1x128x16x16xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4}
>

!OutputSparseTensorDistributed = type !VPU.SparseTensor<
    data=!OutputTensorDistributed, sparsity_map=!OutputSMTensorDistributed
>

// CHECK-LABEL: @DistributedCastSparseDistributedBuf
func @DistributedCastSparseDistributedBuf(%arg0: !InputSparseBufferDistributed) -> !OutputSparseBufferDistributed {
    %0 = builtin.unrealized_conversion_cast %arg0 : !InputSparseBufferDistributed to !InputSparseTensorDistributed
    %1 = VPU.DistributedCast(%0 : !InputSparseTensorDistributed) -> !OutputSparseTensorDistributed
    %2 = builtin.unrealized_conversion_cast %1 : !OutputSparseTensorDistributed to !OutputSparseBufferDistributed
    return %2 : !OutputSparseBufferDistributed

    // CHECK:      [[RES:%.+]] = VPUIP.DistributedCast inputs(
    // CHECK-SAME:   : !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x128x16x16xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>,
    // CHECK-SAME:                               sparsity_map=!VPUIP.DistributedBuffer<1x128x16x16xi1, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>>)
    // CHECK-SAME:   -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x128x16x16xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>,
    // CHECK-SAME:                          sparsity_map=!VPUIP.DistributedBuffer<1x128x16x16xi1, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>>
    // CHECK:      return [[RES]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputSparseBuffer = type !VPUIP.SparseBuffer<
    data=memref<1x32x16x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x32x16x16xi1, #NHWC, @CMX_NN>
>

!InputSparseTensor = type !VPU.SparseTensor<
    data=tensor<1x32x16x16xf16, {order = #NHWC, mem_space = @CMX_NN}>, sparsity_map=tensor<1x32x16x16xi1, {order = #NHWC, mem_space = @CMX_NN}>
>

!OutputSparseBuffer = type !VPUIP.SparseBuffer<
    data=memref<1x16x16x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x1x1x4096xi1, #NHWC, @CMX_NN>
>

!OutputSparseTensor = type !VPU.SparseTensor<
    data=tensor<1x16x16x16xf16, {order = #NHWC, mem_space = @CMX_NN}>, sparsity_map=tensor<1x1x1x4096xi1, {order = #NHWC, mem_space = @CMX_NN}>
>

// CHECK-LABEL: @SplitSparseMemRefBuf
func @SplitSparseMemRefBuf(%arg0: memref<1x32x16x16xf16, #NHWC, @CMX_NN>, %arg1: memref<1x32x16x16xi1, #NHWC, @CMX_NN>) -> (!OutputSparseBuffer, !OutputSparseBuffer) {
    %sb = VPUIP.GroupSparseBuffer(%arg0, %arg1) -> !InputSparseBuffer
    %st = builtin.unrealized_conversion_cast %sb : !InputSparseBuffer to !InputSparseTensor
    %parts:2 = VPU.Split(%st) {num_splits = 2, axis_value = 1} : !InputSparseTensor -> !OutputSparseTensor, !OutputSparseTensor
    %res_1 = builtin.unrealized_conversion_cast %parts#0 : !OutputSparseTensor to !OutputSparseBuffer
    %res_2 = builtin.unrealized_conversion_cast %parts#1 : !OutputSparseTensor to !OutputSparseBuffer
    return %res_1, %res_2 : !OutputSparseBuffer, !OutputSparseBuffer

    // CHECK:      [[BUF_1_PART:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, @CMX_NN>
    // CHECK:      [[MAP_FOR_1_PART:%.+]] = memref.alloc() : memref<1x1x1x4096xi1, #NHWC, @CMX_NN>
    // CHECK:      [[SPARSE_BUF_FOR_1_PART:%.+]] = VPUIP.GroupSparseBuffer([[BUF_1_PART]], [[MAP_FOR_1_PART]]) -> !VPUIP.SparseBuffer<data=memref<1x16x16x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x1x1x4096xi1, #NHWC, @CMX_NN>>

    // CHECK:      [[BUF_FOR_2_PART:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, @CMX_NN>
    // CHECK:      [[MAP_FOR_2_PART:%.+]] = memref.alloc() : memref<1x1x1x4096xi1, #NHWC, @CMX_NN>
    // CHECK:      [[SPARSE_BUF_FOR_2_PART:%.+]] = VPUIP.GroupSparseBuffer([[BUF_FOR_2_PART]], [[MAP_FOR_2_PART]]) -> !VPUIP.SparseBuffer<data=memref<1x16x16x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x1x1x4096xi1, #NHWC, @CMX_NN>>

    // CHECK:      [[SUBVIEW_1:%.+]] = VPUIP.SubView {{%.+}} [0, 0, 0, 0] [1, 16, 16, 16]
    // CHECK-SAME:    : !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x32x16x16xi1, #NHWC, @CMX_NN>>
    // CHECK-SAME:    to !VPUIP.SparseBuffer<data=memref<1x16x16x16xf16, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN>, sparsity_map=memref<1x16x16x16xi1, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN>>
    // CHECK:      [[RES_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:    inputs([[SUBVIEW_1]] : !VPUIP.SparseBuffer<data=memref<1x16x16x16xf16, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN>, sparsity_map=memref<1x16x16x16xi1, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN>>)
    // CHECK-SAME:    outputs([[SPARSE_BUF_FOR_1_PART]] : !VPUIP.SparseBuffer<data=memref<1x16x16x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x1x1x4096xi1, #NHWC, @CMX_NN>>)
    // CHECK-SAME:    -> !VPUIP.SparseBuffer<data=memref<1x16x16x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x1x1x4096xi1, #NHWC, @CMX_NN>>

    // CHECK:      [[SUBVIEW_2:%.+]] = VPUIP.SubView {{%.+}} [0, 16, 0, 0] [1, 16, 16, 16]
    // CHECK-SAME:    : !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x32x16x16xi1, #NHWC, @CMX_NN>>
    // CHECK-SAME:    to !VPUIP.SparseBuffer<data=memref<1x16x16x16xf16, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN>, sparsity_map=memref<1x16x16x16xi1, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN>>
    // CHECK:      [[RES_2:%.+]] = VPUIP.Copy
    // CHECK-SAME:    inputs([[SUBVIEW_2]] : !VPUIP.SparseBuffer<data=memref<1x16x16x16xf16, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN>, sparsity_map=memref<1x16x16x16xi1, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN>>)
    // CHECK-SAME:    outputs([[SPARSE_BUF_FOR_2_PART]] : !VPUIP.SparseBuffer<data=memref<1x16x16x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x1x1x4096xi1, #NHWC, @CMX_NN>>)
    // CHECK-SAME:    -> !VPUIP.SparseBuffer<data=memref<1x16x16x16xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x1x1x4096xi1, #NHWC, @CMX_NN>>

    // CHECK:      return [[RES_1]], [[RES_2]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputSparseBuffer = type !VPUIP.SparseBuffer<
    data=memref<1x32x16x16xf16, #NCHW, @CMX_NN>, sparsity_map=memref<1x32x16x16xi1, #NCHW, @CMX_NN>
>

!InputSparseTensor = type !VPU.SparseTensor<
    data=tensor<1x32x16x16xf16, {order = #NCHW, mem_space = @CMX_NN}>, sparsity_map=tensor<1x32x16x16xi1, {order = #NCHW, mem_space = @CMX_NN}>
>

!OutputSparseBuffer = type !VPUIP.SparseBuffer<
    data=memref<1x16x32x16xf16, #NCHW, @CMX_NN>, sparsity_map=memref<1x16x32x16xi1, #NCHW, @CMX_NN>
>

!OutputSparseTensor = type !VPU.SparseTensor<
    data=tensor<1x16x32x16xf16, {order = #NCHW, mem_space = @CMX_NN}>, sparsity_map=tensor<1x16x32x16xi1, {order = #NCHW, mem_space = @CMX_NN}>
>

// CHECK-LABEL: @ReshapeSparseMemRefBuf
func @ReshapeSparseMemRefBuf(%arg0: memref<1x32x16x16xf16, #NCHW, @CMX_NN>, %arg1: memref<1x32x16x16xi1, #NCHW, @CMX_NN>) -> !OutputSparseBuffer {
    %sb = VPUIP.GroupSparseBuffer(%arg0, %arg1) -> !InputSparseBuffer
    %st = builtin.unrealized_conversion_cast %sb : !InputSparseBuffer to !InputSparseTensor
    %resh = VPU.Reshape(%st) {shape_value = [1, 16, 32, 16]} : !InputSparseTensor -> !OutputSparseTensor
    %res = builtin.unrealized_conversion_cast %resh : !OutputSparseTensor to !OutputSparseBuffer
    return %res : !OutputSparseBuffer

    // CHECK:      [[RES:%.+]] = VPUIP.GenericReshape inputs({{%[0-9a-zA-Z]+}}
    // CHECK-SAME:   : !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, @CMX_NN>, sparsity_map=memref<1x32x16x16xi1, @CMX_NN>>)
    // CHECK-SAME:   -> !VPUIP.SparseBuffer<data=memref<1x16x32x16xf16, @CMX_NN>, sparsity_map=memref<1x16x32x16xi1, @CMX_NN>>
    // CHECK:      return [[RES]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputSparseBuffer = type !VPUIP.SparseBuffer<
    data=memref<32x16x16x1xf16, #NCHW, @CMX_NN>, sparsity_map=memref<32x16x16x1xi1, #NCHW, @CMX_NN>
>

!InputSparseTensor = type !VPU.SparseTensor<
    data=tensor<32x16x16x1xf16, {order = #NCHW, mem_space = @CMX_NN}>, sparsity_map=tensor<32x16x16x1xi1, {order = #NCHW, mem_space = @CMX_NN}>
>

!OutputSparseBuffer = type !VPUIP.SparseBuffer<
    data=memref<1x32x16x16xf16, #NCHW, @CMX_NN>, sparsity_map=memref<1x32x16x16xi1, #NCHW, @CMX_NN>
>

!OutputSparseTensor = type !VPU.SparseTensor<
    data=tensor<1x32x16x16xf16, {order = #NCHW, mem_space = @CMX_NN}>, sparsity_map=tensor<1x32x16x16xi1, {order = #NCHW, mem_space = @CMX_NN}>
>

// CHECK-LABEL: @AffineReshapeSparseMemRefBuf
func @AffineReshapeSparseMemRefBuf(%arg0: memref<32x16x16x1xf16, #NCHW, @CMX_NN>, %arg1: memref<32x16x16x1xi1, #NCHW, @CMX_NN>) -> !OutputSparseBuffer {
    %sb = VPUIP.GroupSparseBuffer(%arg0, %arg1) -> !InputSparseBuffer
    %st = builtin.unrealized_conversion_cast %sb : !InputSparseBuffer to !InputSparseTensor
    %resh = VPU.AffineReshape(%st) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 32, 16, 16]} : !InputSparseTensor -> !OutputSparseTensor
    %res = builtin.unrealized_conversion_cast %resh : !OutputSparseTensor to !OutputSparseBuffer
    return %res : !OutputSparseBuffer

    // CHECK:      [[RES:%.+]] = VPUIP.GenericReshape inputs({{%[0-9a-zA-Z]+}}
    // CHECK-SAME:   : !VPUIP.SparseBuffer<data=memref<32x16x16x1xf16, @CMX_NN>, sparsity_map=memref<32x16x16x1xi1, @CMX_NN>>)
    // CHECK-SAME:   -> !VPUIP.SparseBuffer<data=memref<1x32x16x16xf16, @CMX_NN>, sparsity_map=memref<1x32x16x16xi1, @CMX_NN>>
    // CHECK:      return [[RES]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputSparseBuffer = type !VPUIP.SparseBuffer<
    data=memref<1x3x32x32xf16, #NCHW, @CMX_NN>, sparsity_map=memref<1x3x32x32xi1, #NCHW, @CMX_NN>
>

!InputSparseTensor = type !VPU.SparseTensor<
    data=tensor<1x3x32x32xf16, {order = #NCHW, mem_space = @CMX_NN}>, sparsity_map=tensor<1x3x32x32xi1, {order = #NCHW, mem_space = @CMX_NN}>
>

!OutputSparseBuffer = type !VPUIP.SparseBuffer<
    data=memref<1x16x16x12xf16, #NCHW, @CMX_NN>, sparsity_map=memref<1x16x16x12xi1, @CMX_NN>
>

!OutputSparseTensor = type !VPU.SparseTensor<
    data=tensor<1x16x16x12xf16, {order = #NCHW, mem_space = @CMX_NN}>, sparsity_map=tensor<1x16x16x12xi1, {order = #NCHW, mem_space = @CMX_NN}>
>

// CHECK-LABEL: @ShapeCastSparseMemRefBuf
func @ShapeCastSparseMemRefBuf(%arg0: memref<1x3x32x32xf16, #NCHW, @CMX_NN>, %arg1: memref<1x3x32x32xi1, #NCHW, @CMX_NN>) -> !OutputSparseBuffer {
    %sb = VPUIP.GroupSparseBuffer(%arg0, %arg1) -> !InputSparseBuffer
    %st = builtin.unrealized_conversion_cast %sb : !InputSparseBuffer to !InputSparseTensor
    %resh = VPU.ShapeCast {shape = [1, 16, 16, 12]} inputs(%st : !InputSparseTensor) -> !OutputSparseTensor
    %res = builtin.unrealized_conversion_cast %resh : !OutputSparseTensor to !OutputSparseBuffer
    return %res : !OutputSparseBuffer

    // CHECK:      [[RES:%.+]] = VPUIP.ShapeCast {shape = [1, 16, 16, 12]} inputs({{%[0-9a-zA-Z]+}}
    // CHECK-SAME:   : !VPUIP.SparseBuffer<data=memref<1x3x32x32xf16, @CMX_NN>, sparsity_map=memref<1x3x32x32xi1, @CMX_NN>>)
    // CHECK-SAME:   -> !VPUIP.SparseBuffer<data=memref<1x16x16x12xf16, @CMX_NN>, sparsity_map=memref<1x16x16x12xi1, @CMX_NN>>
    // CHECK:      return [[RES]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputSparseBuffer = type !VPUIP.SparseBuffer<
    data=memref<1x16x32x32xf16, #NCHW, @CMX_NN>, sparsity_map=memref<1x16x32x32xi1, #NCHW, @CMX_NN>
>

!InputSparseTensor = type !VPU.SparseTensor<
    data=tensor<1x16x32x32xf16, {order = #NCHW, mem_space = @CMX_NN}>, sparsity_map=tensor<1x16x32x32xi1, {order = #NCHW, mem_space = @CMX_NN}>
>

!OutputSparseBuffer = type !VPUIP.SparseBuffer<
    data=memref<1x16x32x32xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x16x32x32xi1, #NHWC, @CMX_NN>
>

!OutputSparseTensor = type !VPU.SparseTensor<
    data=tensor<1x16x32x32xf16, {order = #NHWC, mem_space = @CMX_NN}>, sparsity_map=tensor<1x16x32x32xi1, {order = #NHWC, mem_space = @CMX_NN}>
>

// CHECK-LABEL: @LayoutCastSparseMemRefBuf
func @LayoutCastSparseMemRefBuf(%arg0: memref<1x16x32x32xf16, #NCHW, @CMX_NN>, %arg1: memref<1x16x32x32xi1, #NCHW, @CMX_NN>) -> !OutputSparseBuffer {
    %sb = VPUIP.GroupSparseBuffer(%arg0, %arg1) -> !InputSparseBuffer
    %st = builtin.unrealized_conversion_cast %sb : !InputSparseBuffer to !InputSparseTensor
    %resh = VPU.LayoutCast(%st) {dst_order = #NHWC} : !InputSparseTensor -> !OutputSparseTensor
    %res = builtin.unrealized_conversion_cast %resh : !OutputSparseTensor to !OutputSparseBuffer
    return %res : !OutputSparseBuffer

    // CHECK:      [[RES:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs({{%[0-9a-zA-Z]+}}
    // CHECK-SAME:   : !VPUIP.SparseBuffer<data=memref<1x16x32x32xf16, @CMX_NN>, sparsity_map=memref<1x16x32x32xi1, @CMX_NN>>)
    // CHECK-SAME:   -> !VPUIP.SparseBuffer<data=memref<1x16x32x32xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x16x32x32xi1, #NHWC, @CMX_NN>>
    // CHECK:      return [[RES]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#CHW = affine_map<(d1, d2, d3) -> (d1, d2, d3)>

!InputSparseBuffer = type !VPUIP.SparseBuffer<
    data=memref<1x16x32x32xf16, #NCHW, @CMX_NN>, sparsity_map=memref<1x16x32x32xi1, #NCHW, @CMX_NN>
>

!InputSparseTensor = type !VPU.SparseTensor<
    data=tensor<1x16x32x32xf16, {order = #NCHW, mem_space = @CMX_NN}>, sparsity_map=tensor<1x16x32x32xi1, {order = #NCHW, mem_space = @CMX_NN}>
>

!OutputSparseBuffer = type !VPUIP.SparseBuffer<
    data=memref<16x32x32xf16, #CHW, @CMX_NN>, sparsity_map=memref<16x32x32xi1, #CHW, @CMX_NN>
>

!OutputSparseTensor = type !VPU.SparseTensor<
    data=tensor<16x32x32xf16, {order = #CHW, mem_space = @CMX_NN}>, sparsity_map=tensor<16x32x32xi1, {order = #CHW, mem_space = @CMX_NN}>
>

// CHECK-LABEL: @SqueezeSparseMemRefBuf
func @SqueezeSparseMemRefBuf(%arg0: memref<1x16x32x32xf16, #NCHW, @CMX_NN>, %arg1: memref<1x16x32x32xi1, #NCHW, @CMX_NN>) -> !OutputSparseBuffer {
    %sb = VPUIP.GroupSparseBuffer(%arg0, %arg1) -> !InputSparseBuffer
    %st = builtin.unrealized_conversion_cast %sb : !InputSparseBuffer to !InputSparseTensor
    %squeezed = VPU.Squeeze(%st) { axes_value = [0] } : !InputSparseTensor -> !OutputSparseTensor
    %res = builtin.unrealized_conversion_cast %squeezed : !OutputSparseTensor to !OutputSparseBuffer
    return %res : !OutputSparseBuffer

    // CHECK:      [[RES:%.+]] = VPUIP.GenericReshape inputs({{%[0-9a-zA-Z]+}}
    // CHECK-SAME:   : !VPUIP.SparseBuffer<data=memref<1x16x32x32xf16, @CMX_NN>, sparsity_map=memref<1x16x32x32xi1, @CMX_NN>>)
    // CHECK-SAME:   -> !VPUIP.SparseBuffer<data=memref<16x32x32xf16, @CMX_NN>, sparsity_map=memref<16x32x32xi1, @CMX_NN>>
    // CHECK:      return [[RES]]
}

// -----

#CHW = affine_map<(d1, d2, d3) -> (d1, d2, d3)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputSparseBuffer = type !VPUIP.SparseBuffer<
    data=memref<16x32x32xf16, #CHW, @CMX_NN>, sparsity_map=memref<16x32x32xi1, #CHW, @CMX_NN>
>

!InputSparseTensor = type !VPU.SparseTensor<
    data=tensor<16x32x32xf16, {order = #CHW, mem_space = @CMX_NN}>, sparsity_map=tensor<16x32x32xi1, {order = #CHW, mem_space = @CMX_NN}>
>

!OutputSparseBuffer = type !VPUIP.SparseBuffer<
    data=memref<1x16x32x32xf16, #NCHW, @CMX_NN>, sparsity_map=memref<1x16x32x32xi1, #NCHW, @CMX_NN>
>

!OutputSparseTensor = type !VPU.SparseTensor<
    data=tensor<1x16x32x32xf16, {order = #NCHW, mem_space = @CMX_NN}>, sparsity_map=tensor<1x16x32x32xi1, {order = #NCHW, mem_space = @CMX_NN}>
>

// CHECK-LABEL: @UnsqueezeSparseMemRefBuf
func @UnsqueezeSparseMemRefBuf(%arg0: memref<16x32x32xf16, #CHW, @CMX_NN>, %arg1: memref<16x32x32xi1, #CHW, @CMX_NN>) -> !OutputSparseBuffer {
    %sb = VPUIP.GroupSparseBuffer(%arg0, %arg1) -> !InputSparseBuffer
    %st = builtin.unrealized_conversion_cast %sb : !InputSparseBuffer to !InputSparseTensor
    %unsqueezed = VPU.Unsqueeze(%st) { axes_value = [0] } : !InputSparseTensor -> !OutputSparseTensor
    %res = builtin.unrealized_conversion_cast %unsqueezed : !OutputSparseTensor to !OutputSparseBuffer
    return %res : !OutputSparseBuffer

    // CHECK:      [[RES:%.+]] = VPUIP.GenericReshape inputs({{%[0-9a-zA-Z]+}}
    // CHECK-SAME:   : !VPUIP.SparseBuffer<data=memref<16x32x32xf16, @CMX_NN>, sparsity_map=memref<16x32x32xi1, @CMX_NN>>)
    // CHECK-SAME:   -> !VPUIP.SparseBuffer<data=memref<1x16x32x32xf16, @CMX_NN>, sparsity_map=memref<1x16x32x32xi1, @CMX_NN>>
    // CHECK:      return [[RES]]
}
