//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-concat-view-copies %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x57x512xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    alignment = [1, 16, 1, 1]
}>

func.func @AvoidConcatExtraChannel(
        %arg0: !InputDistributed,
        %arg1: !InputDistributed,
        %arg2: memref<1x3x110x512xf16, #NHWC, @DDR>,
        %arg3: memref<1x3x4x512xf16, #NHWC, @DDR>)
         -> (memref<1x3x110x512xf16, #NHWC, @DDR>, memref<1x3x4x512xf16, #NHWC, @DDR>){
    %buffer = memref.alloc() : memref<1x16x114x512xf16, #NHWC, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 16, 57, 512] : memref<1x16x114x512xf16, #NHWC, @DDR> to memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    %nceTilingCopy0 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg4: memref<1x16x57x512xf16, #NHWC, @CMX_NN>) outputs(%subview0 as %arg5: memref<1x16x57x512xf16, #NHWC>) -> memref<1x16x57x512xf16, {order = #NHWC}, @DDR> {
      %0 = VPUIP.Copy inputs(%arg4 : memref<1x16x57x512xf16, #NHWC, @CMX_NN>) outputs(%arg5 : memref<1x16x57x512xf16, #NHWC>) -> memref<1x16x57x512xf16, #NHWC>
    }
    %subview1 = VPUIP.SubView %buffer [0, 0, 57, 0] [1, 16, 57, 512] : memref<1x16x114x512xf16, #NHWC, @DDR> to memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    %nceTilingCopy1 = VPUIP.NCEClusterTiling inputs(%arg1 as %arg4: memref<1x16x57x512xf16, #NHWC, @CMX_NN>) outputs(%subview1 as %arg5: memref<1x16x57x512xf16, #NHWC>) -> memref<1x16x57x512xf16, {order = #NHWC}, @DDR> {
      %0 = VPUIP.Copy inputs(%arg4 : memref<1x16x57x512xf16, #NHWC, @CMX_NN>) outputs(%arg5 : memref<1x16x57x512xf16, #NHWC>) -> memref<1x16x57x512xf16, #NHWC>
    }
    %concat = VPUIP.ConcatView inputs(%nceTilingCopy0, %nceTilingCopy1 : memref<1x16x57x512xf16, {order = #NHWC}, @DDR>, memref<1x16x57x512xf16, {order = #NHWC}, @DDR>) outputs(%buffer : memref<1x16x114x512xf16, #NHWC, @DDR>) -> memref<1x16x114x512xf16, #NHWC, @DDR>
    %subview2 = VPUIP.SubView %concat [0, 0, 0, 0] [1, 3, 110, 512] : memref<1x16x114x512xf16, #NHWC, @DDR> to memref<1x3x110x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    %copy0 = VPUIP.Copy
        inputs(%subview2 : memref<1x3x110x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>)
        outputs(%arg2 : memref<1x3x110x512xf16, #NHWC, @DDR>)
        -> memref<1x3x110x512xf16, #NHWC, @DDR>
    %subview3 = VPUIP.SubView %concat [0, 0, 110, 0] [1, 3, 4, 512] : memref<1x16x114x512xf16, #NHWC, @DDR> to memref<1x3x4x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    %copy1 = VPUIP.Copy
        inputs(%subview3 : memref<1x3x4x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>)
        outputs(%arg3 : memref<1x3x4x512xf16, #NHWC, @DDR>)
        -> memref<1x3x4x512xf16, #NHWC, @DDR>
    return %copy0, %copy1 : memref<1x3x110x512xf16, #NHWC, @DDR>, memref<1x3x4x512xf16, #NHWC, @DDR>

    // CHECK-NOT: memref.alloc() : memref<1x16x114x512xf16, #NHWC, @DDR>
    // CHECK: [[NEW_BUFFER:%.+]] = memref.alloc() : memref<1x3x114x512xf16, #NHWC, @DDR>
    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView
    // CHECK-SAME:  [0, 0, 0, 0] [1, 3, 57, 512] : !VPUIP.DistributedBuffer<1x16x57x512xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView [[NEW_BUFFER]]
    // CHECK-SAME:  [0, 0, 0, 0] [1, 3, 57, 512] : memref<1x3x114x512xf16, #NHWC, @DDR> to memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>
    // CHECK: [[TILING_COPY0:%.+]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW0]] as %arg4: memref<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>) outputs([[SUBVIEW1]] as %arg5: memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) -> memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR> {
    // CHECK:  VPUIP.Copy inputs(%arg4 : memref<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>) outputs(%arg5 : memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) -> memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>

    // CHECK: [[SUBVIEW2:%.+]] = VPUIP.SubView
    // CHECK-SAME:   [0, 0, 0, 0] [1, 3, 57, 512] : !VPUIP.DistributedBuffer<1x16x57x512xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK: [[SUBVIEW3:%.+]] = VPUIP.SubView [[NEW_BUFFER]] [0, 0, 57, 0] [1, 3, 57, 512] : memref<1x3x114x512xf16, #NHWC, @DDR> to memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>
    // CHECK: [[TILING_COPY1:%.+]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW2]] as %arg4: memref<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>) outputs([[SUBVIEW3]] as %arg5: memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) -> memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR> {
    // CHECK: VPUIP.Copy inputs(%arg4 : memref<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>) outputs(%arg5 : memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) -> memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>

    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[TILING_COPY0]], [[TILING_COPY1]] : memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) outputs([[NEW_BUFFER]] : memref<1x3x114x512xf16, #NHWC, @DDR>) -> memref<1x3x114x512xf16, #NHWC, @DDR>
    // CHECK: [[SUBVIEW2:%.+]] = VPUIP.SubView [[CONCAT]] [0, 0, 0, 0] [1, 3, 110, 512] : memref<1x3x114x512xf16, #NHWC, @DDR> to memref<1x3x110x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>
    // CHECK: [[LAST_COPY0:%.+]] = VPUIP.Copy inputs([[SUBVIEW2]] : memref<1x3x110x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) outputs(%arg2 : memref<1x3x110x512xf16, #NHWC, @DDR>) -> memref<1x3x110x512xf16, #NHWC, @DDR>
    // CHECK: [[SUBVIEW3:%.+]] = VPUIP.SubView [[CONCAT]] [0, 0, 110, 0] [1, 3, 4, 512] : memref<1x3x114x512xf16, #NHWC, @DDR> to memref<1x3x4x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>
    // CHECK: [[LAST_COPY1:%.+]] = VPUIP.Copy inputs([[SUBVIEW3]] : memref<1x3x4x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) outputs(%arg3 : memref<1x3x4x512xf16, #NHWC, @DDR>) -> memref<1x3x4x512xf16, #NHWC, @DDR>
    // CHECK: return [[LAST_COPY0]], [[LAST_COPY1]] : memref<1x3x110x512xf16, #NHWC, @DDR>, memref<1x3x4x512xf16, #NHWC, @DDR>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x57x512xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    alignment = [1, 16, 1, 1]
}>

func.func @AvoidConcatExtraChannelAndChannelOffsetNotEqualZero(
        %arg0: !InputDistributed,
        %arg1: !InputDistributed,
        %arg2: memref<1x3x110x512xf16, #NHWC, @DDR>,
        %arg3: memref<1x3x4x512xf16, #NHWC, @DDR>)
         -> (memref<1x3x110x512xf16, #NHWC, @DDR>, memref<1x3x4x512xf16, #NHWC, @DDR>){
    %buffer = memref.alloc() : memref<1x16x114x512xf16, #NHWC, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 16, 57, 512] : memref<1x16x114x512xf16, #NHWC, @DDR> to memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    %nceTilingCopy0 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg4: memref<1x16x57x512xf16, #NHWC, @CMX_NN>) outputs(%subview0 as %arg5: memref<1x16x57x512xf16, #NHWC>) -> memref<1x16x57x512xf16, {order = #NHWC}, @DDR> {
      %0 = VPUIP.Copy inputs(%arg4 : memref<1x16x57x512xf16, #NHWC, @CMX_NN>) outputs(%arg5 : memref<1x16x57x512xf16, #NHWC>) -> memref<1x16x57x512xf16, #NHWC>
    }
    %subview1 = VPUIP.SubView %buffer [0, 0, 57, 0] [1, 16, 57, 512] : memref<1x16x114x512xf16, #NHWC, @DDR> to memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    %nceTilingCopy1 = VPUIP.NCEClusterTiling inputs(%arg1 as %arg4: memref<1x16x57x512xf16, #NHWC, @CMX_NN>) outputs(%subview1 as %arg5: memref<1x16x57x512xf16, #NHWC>) -> memref<1x16x57x512xf16, {order = #NHWC}, @DDR> {
      %0 = VPUIP.Copy inputs(%arg4 : memref<1x16x57x512xf16, #NHWC, @CMX_NN>) outputs(%arg5 : memref<1x16x57x512xf16, #NHWC>) -> memref<1x16x57x512xf16, #NHWC>
    }
    %concat = VPUIP.ConcatView inputs(%nceTilingCopy0, %nceTilingCopy1 : memref<1x16x57x512xf16, {order = #NHWC}, @DDR>, memref<1x16x57x512xf16, {order = #NHWC}, @DDR>) outputs(%buffer : memref<1x16x114x512xf16, #NHWC, @DDR>) -> memref<1x16x114x512xf16, #NHWC, @DDR>
    %subview2 = VPUIP.SubView %concat [0, 3, 0, 0] [1, 3, 110, 512] : memref<1x16x114x512xf16, #NHWC, @DDR> to memref<1x3x110x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    %copy0 = VPUIP.Copy
        inputs(%subview2 : memref<1x3x110x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>)
        outputs(%arg2 : memref<1x3x110x512xf16, #NHWC, @DDR>)
        -> memref<1x3x110x512xf16, #NHWC, @DDR>
    %subview3 = VPUIP.SubView %concat [0, 3, 110, 0] [1, 3, 4, 512] : memref<1x16x114x512xf16, #NHWC, @DDR> to memref<1x3x4x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    %copy1 = VPUIP.Copy
        inputs(%subview3 : memref<1x3x4x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>)
        outputs(%arg3 : memref<1x3x4x512xf16, #NHWC, @DDR>)
        -> memref<1x3x4x512xf16, #NHWC, @DDR>
    return %copy0, %copy1 : memref<1x3x110x512xf16, #NHWC, @DDR>, memref<1x3x4x512xf16, #NHWC, @DDR>

    // CHECK-NOT: memref.alloc() : memref<1x16x114x512xf16, #NHWC, @DDR>
    // CHECK: [[NEW_BUFFER:%.+]] = memref.alloc() : memref<1x3x114x512xf16, #NHWC, @DDR>
    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView
    // CHECK-SAME:  [0, 3, 0, 0] [1, 3, 57, 512] : !VPUIP.DistributedBuffer<1x16x57x512xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView [[NEW_BUFFER]]
    // CHECK-SAME:  [0, 0, 0, 0] [1, 3, 57, 512] : memref<1x3x114x512xf16, #NHWC, @DDR> to memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>
    // CHECK: [[TILING_COPY0:%.+]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW0]] as %arg4: memref<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>) outputs([[SUBVIEW1]] as %arg5: memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) -> memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR> {
    // CHECK:  VPUIP.Copy inputs(%arg4 : memref<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>) outputs(%arg5 : memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) -> memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>

    // CHECK: [[SUBVIEW2:%.+]] = VPUIP.SubView
    // CHECK-SAME:  [0, 3, 0, 0] [1, 3, 57, 512] : !VPUIP.DistributedBuffer<1x16x57x512xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK: [[SUBVIEW3:%.+]] = VPUIP.SubView [[NEW_BUFFER]]
    // CHECK-SAME:  [0, 0, 57, 0] [1, 3, 57, 512] : memref<1x3x114x512xf16, #NHWC, @DDR> to memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>
    // CHECK: [[TILING_COPY1:%.+]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW2]] as %arg4: memref<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>) outputs([[SUBVIEW3]] as %arg5: memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) -> memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR> {
    // CHECK: VPUIP.Copy inputs(%arg4 : memref<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>) outputs(%arg5 : memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) -> memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>

    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[TILING_COPY0]], [[TILING_COPY1]] : memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) outputs([[NEW_BUFFER]] : memref<1x3x114x512xf16, #NHWC, @DDR>) -> memref<1x3x114x512xf16, #NHWC, @DDR>
    // CHECK: [[SUBVIEW2:%.+]] = VPUIP.SubView [[CONCAT]] [0, 0, 0, 0] [1, 3, 110, 512] : memref<1x3x114x512xf16, #NHWC, @DDR> to memref<1x3x110x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>
    // CHECK: [[LAST_COPY0:%.+]] = VPUIP.Copy inputs([[SUBVIEW2]] : memref<1x3x110x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) outputs(%arg2 : memref<1x3x110x512xf16, #NHWC, @DDR>) -> memref<1x3x110x512xf16, #NHWC, @DDR>
    // CHECK: [[SUBVIEW3:%.+]] = VPUIP.SubView [[CONCAT]] [0, 0, 110, 0] [1, 3, 4, 512] : memref<1x3x114x512xf16, #NHWC, @DDR> to memref<1x3x4x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>
    // CHECK: [[LAST_COPY1:%.+]] = VPUIP.Copy inputs([[SUBVIEW3]] : memref<1x3x4x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) outputs(%arg3 : memref<1x3x4x512xf16, #NHWC, @DDR>) -> memref<1x3x4x512xf16, #NHWC, @DDR>
    // CHECK: return [[LAST_COPY0]], [[LAST_COPY1]] : memref<1x3x110x512xf16, #NHWC, @DDR>, memref<1x3x4x512xf16, #NHWC, @DDR>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IODDRData0 = memref<1x16x57x512xf16, {order = #NHWC}, @DDR>
!IODDRSM0 = memref<1x16x57x512xi1, {order = #NHWC}, @DDR>
!IODDRSparse0 = !VPUIP.SparseBuffer<
    data=!IODDRData0,
    sparsity_map=!IODDRSM0
>

!IODDRSparse1 = !VPUIP.SparseBuffer<
    data=memref<1x3x110x512xf16, #NHWC, @DDR>,
    sparsity_map=memref<1x3x110x512xi1, #NHWC, @DDR>
>
!IODistrCMXSparse0 = !VPUIP.SparseBuffer<

    data=!VPUIP.DistributedBuffer<
    1x16x57x512xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64, alignment = [1, 16, 1, 1]
}>,
    sparsity_map=!VPUIP.DistributedBuffer<
    1x16x57x512xi1, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64, alignment = [1, 16, 1, 1]
}>
>

!IODDRData2 = memref<1x16x57x512xf16, #NHWC>
!IODDRSM2 = memref<1x16x57x512xi1, #NHWC>
!IODDRSparse2 = !VPUIP.SparseBuffer<
    data=!IODDRData2,
    sparsity_map=!IODDRSM2
>

!IODDRSparse3 = !VPUIP.SparseBuffer<
    data=memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>,
    sparsity_map=memref<1x16x57x512xi1, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
>
!IOCMXData0 = memref<1x16x57x512xf16, #NHWC, @CMX_NN>
!IOCMXSM0 = memref<1x16x57x512xi1, #NHWC, @CMX_NN>
!IOCMXSparse0 = !VPUIP.SparseBuffer<
    data=!IOCMXData0,
    sparsity_map=!IOCMXSM0
>

!IODDRData4 = memref<1x16x114x512xf16, #NHWC, @DDR>
!IODDRSM4 = memref<1x16x114x512xi1, #NHWC, @DDR>
!IODDRSparse4 = !VPUIP.SparseBuffer<
    data=!IODDRData4,
    sparsity_map=!IODDRSM4
>

!IODDRSparse5 = !VPUIP.SparseBuffer<
    data=memref<1x3x4x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>,
    sparsity_map=memref<1x3x4x512xi1, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
>
!IODDRSparse6 = !VPUIP.SparseBuffer<
    data=memref<1x3x110x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>,
    sparsity_map=memref<1x3x110x512xi1, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
>
!IODDRSparse7 = !VPUIP.SparseBuffer<
    data=memref<1x3x4x512xf16, #NHWC, @DDR>,
    sparsity_map=memref<1x3x4x512xi1, #NHWC, @DDR>
>

// CHECK-LABEL: @AvoidConcatExtraChannelSparse
func.func @AvoidConcatExtraChannelSparse(%arg0: !IODistrCMXSparse0, %arg1: !IODistrCMXSparse0, %arg2: !IODDRSparse1, %arg3: !IODDRSparse7) -> (!IODDRSparse1, !IODDRSparse7) {
    %0 = memref.alloc() : !IODDRData4
    %1 = memref.alloc() : !IODDRSM4
    %2 = VPUIP.GroupSparseBuffer(%0, %1) -> !IODDRSparse4

    %3 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 16, 57, 512] : !IODDRSparse4 to !IODDRSparse3
    %4 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg4: !IOCMXSparse0) outputs(%3 as %arg5: !IODDRSparse2) -> !IODDRSparse0 {
      %12 = VPUIP.Copy inputs(%arg4 : !IOCMXSparse0) outputs(%arg5 : !IODDRSparse2) -> !IODDRSparse2
    }
    %5 = VPUIP.SubView %2 [0, 0, 57, 0] [1, 16, 57, 512] : !IODDRSparse4 to !IODDRSparse3
    %6 = VPUIP.NCEClusterTiling inputs(%arg1 as %arg4: !IOCMXSparse0) outputs(%5 as %arg5: !IODDRSparse2) -> !IODDRSparse0 {
      %12 = VPUIP.Copy inputs(%arg4 : !IOCMXSparse0) outputs(%arg5 : !IODDRSparse2) -> !IODDRSparse2
    }
    %7 = VPUIP.ConcatView inputs(%4, %6 : !IODDRSparse0, !IODDRSparse0) outputs(%2 : !IODDRSparse4) -> !IODDRSparse4
    %8 = VPUIP.SubView %7 [0, 0, 0, 0] [1, 3, 110, 512] : !IODDRSparse4 to !IODDRSparse6
    %9 = VPUIP.Copy inputs(%8 : !IODDRSparse6) outputs(%arg2 : !IODDRSparse1) -> !IODDRSparse1
    %10 = VPUIP.SubView %7 [0, 0, 110, 0] [1, 3, 4, 512] : !IODDRSparse4 to !IODDRSparse5
    %11 = VPUIP.Copy inputs(%10 : !IODDRSparse5) outputs(%arg3 : !IODDRSparse7) -> !IODDRSparse7
    return %9, %11 : !IODDRSparse1, !IODDRSparse7

    // CHECK-NOT: memref.alloc() : memref<1x16x114x512xf16, #NHWC, @DDR>
    // CHECK-NOT: memref.alloc() : memref<1x16x114x512xi1, #NHWC, @DDR>

    // CHECK:       [[BUFF_0_DATA:%.+]] = memref.alloc() : memref<1x3x114x512xf16, #NHWC, @DDR>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x3x114x512xi1, #NHWC, @DDR>
    // CHECK:       [[BUFF_0:%.+]] = VPUIP.GroupSparseBuffer([[BUFF_0_DATA]], [[BUFF_0_SM]])
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>

    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 3, 57, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x16x57x512xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x16x57x512xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x3x57x512xi1, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 0] [1, 3, 57, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>

    // CHECK:       [[COPY_0:%.+]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW_0]] as %arg4: !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>>)
    // CHECK-SAME:         outputs([[SUBVIEW_1]] as %arg5: !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>> {
    // CHECK:       [[inner_0:%.+]] = VPUIP.Copy inputs(%arg4 : !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>>)
    // CHECK-SAME:         outputs(%arg5 : !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>

    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 3, 57, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x16x57x512xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x16x57x512xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x3x57x512xi1, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // CHECK:       [[SUBVIEW_3:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 0, 57, 0] [1, 3, 57, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>

    // CHECK:       [[COPY_1:%.+]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW_2]] as %arg4: !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>>)
    // CHECK-SAME:         outputs([[SUBVIEW_3]] as %arg5: !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>> {
    // CHECK:       [[inner_1:%.+]] = VPUIP.Copy inputs(%arg4 : !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>>)
    // CHECK-SAME:         outputs(%arg5 : !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>

    // CHECK:       [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]] : !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>, !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>)
    // CHECK-SAME:         outputs([[BUFF_0]] : !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>

    // CHECK:       [[SUBVIEW_4:%.+]] = VPUIP.SubView [[CONCATVIEW_0]] [0, 0, 0, 0] [1, 3, 110, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x3x110x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x110x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>

    // CHECK:       [[COPY_2:%.+]] = VPUIP.Copy inputs([[SUBVIEW_4]] : !VPUIP.SparseBuffer<data=memref<1x3x110x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x110x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>)
    // CHECK-SAME:         outputs(%arg2 : !VPUIP.SparseBuffer<data=memref<1x3x110x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x110x512xi1, #NHWC, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x110x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x110x512xi1, #NHWC, @DDR>>

    // CHECK:       [[SUBVIEW_5:%.+]] = VPUIP.SubView [[CONCATVIEW_0]] [0, 0, 110, 0] [1, 3, 4, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x3x4x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x4x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>

    // CHECK:       [[COPY_3:%.+]] = VPUIP.Copy inputs([[SUBVIEW_5]] : !VPUIP.SparseBuffer<data=memref<1x3x4x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x4x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>)
    // CHECK-SAME:         outputs(%arg3 : !VPUIP.SparseBuffer<data=memref<1x3x4x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x4x512xi1, #NHWC, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x4x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x4x512xi1, #NHWC, @DDR>>

    // CHECK:       return [[COPY_2]], [[COPY_3]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IODDRData0 = memref<1x16x57x512xf16, {order = #NHWC}, @DDR>
!IODDRSM0 = memref<1x16x57x512xi1, {order = #NHWC}, @DDR>
!IODDRSparse0 = !VPUIP.SparseBuffer<
    data=!IODDRData0,
    sparsity_map=!IODDRSM0
>

!IODDRSparse1 = !VPUIP.SparseBuffer<
    data=memref<1x3x110x512xf16, #NHWC, @DDR>,
    sparsity_map=memref<1x3x110x512xi1, #NHWC, @DDR>
>
!IODistrCMXSparse0 = !VPUIP.SparseBuffer<

    data=!VPUIP.DistributedBuffer<
    1x16x57x512xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64, alignment = [1, 16, 1, 1]
}>,
    sparsity_map=!VPUIP.DistributedBuffer<
    1x16x57x512xi1, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64, alignment = [1, 16, 1, 1]
}>
>

!IODDRData2 = memref<1x16x57x512xf16, #NHWC>
!IODDRSM2 = memref<1x16x57x512xi1, #NHWC>
!IODDRSparse2 = !VPUIP.SparseBuffer<
    data=!IODDRData2,
    sparsity_map=!IODDRSM2
>

!IODDRSparse3 = !VPUIP.SparseBuffer<
    data=memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>,
    sparsity_map=memref<1x16x57x512xi1, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
>
!IOCMXData0 = memref<1x16x57x512xf16, #NHWC, @CMX_NN>
!IOCMXSM0 = memref<1x16x57x512xi1, #NHWC, @CMX_NN>
!IOCMXSparse0 = !VPUIP.SparseBuffer<
    data=!IOCMXData0,
    sparsity_map=!IOCMXSM0
>

!IODDRData4 = memref<1x16x114x512xf16, #NHWC, @DDR>
!IODDRSM4 = memref<1x16x114x512xi1, #NHWC, @DDR>
!IODDRSparse4 = !VPUIP.SparseBuffer<
    data=!IODDRData4,
    sparsity_map=!IODDRSM4
>

!IODDRSparse5 = !VPUIP.SparseBuffer<
    data=memref<1x3x4x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>,
    sparsity_map=memref<1x3x4x512xi1, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
>
!IODDRSparse6 = !VPUIP.SparseBuffer<
    data=memref<1x3x110x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>,
    sparsity_map=memref<1x3x110x512xi1, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
>
!IODDRSparse7 = !VPUIP.SparseBuffer<
    data=memref<1x3x4x512xf16, #NHWC, @DDR>,
    sparsity_map=memref<1x3x4x512xi1, #NHWC, @DDR>
>

// CHECK-LABEL: @AvoidConcatExtraChannelSparseAndChannelOffsetNotEqualZero
func.func @AvoidConcatExtraChannelSparseAndChannelOffsetNotEqualZero(%arg0: !IODistrCMXSparse0, %arg1: !IODistrCMXSparse0, %arg2: !IODDRSparse1, %arg3: !IODDRSparse7) -> (!IODDRSparse1, !IODDRSparse7) {
    %0 = memref.alloc() : !IODDRData4
    %1 = memref.alloc() : !IODDRSM4
    %2 = VPUIP.GroupSparseBuffer(%0, %1) -> !IODDRSparse4

    %3 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 16, 57, 512] : !IODDRSparse4 to !IODDRSparse3
    %4 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg4: !IOCMXSparse0) outputs(%3 as %arg5: !IODDRSparse2) -> !IODDRSparse0 {
      %12 = VPUIP.Copy inputs(%arg4 : !IOCMXSparse0) outputs(%arg5 : !IODDRSparse2) -> !IODDRSparse2
    }
    %5 = VPUIP.SubView %2 [0, 0, 57, 0] [1, 16, 57, 512] : !IODDRSparse4 to !IODDRSparse3
    %6 = VPUIP.NCEClusterTiling inputs(%arg1 as %arg4: !IOCMXSparse0) outputs(%5 as %arg5: !IODDRSparse2) -> !IODDRSparse0 {
      %12 = VPUIP.Copy inputs(%arg4 : !IOCMXSparse0) outputs(%arg5 : !IODDRSparse2) -> !IODDRSparse2
    }
    %7 = VPUIP.ConcatView inputs(%4, %6 : !IODDRSparse0, !IODDRSparse0) outputs(%2 : !IODDRSparse4) -> !IODDRSparse4
    %8 = VPUIP.SubView %7 [0, 3, 0, 0] [1, 3, 110, 512] : !IODDRSparse4 to !IODDRSparse6
    %9 = VPUIP.Copy inputs(%8 : !IODDRSparse6) outputs(%arg2 : !IODDRSparse1) -> !IODDRSparse1
    %10 = VPUIP.SubView %7 [0, 3, 110, 0] [1, 3, 4, 512] : !IODDRSparse4 to !IODDRSparse5
    %11 = VPUIP.Copy inputs(%10 : !IODDRSparse5) outputs(%arg3 : !IODDRSparse7) -> !IODDRSparse7
    return %9, %11 : !IODDRSparse1, !IODDRSparse7

    // CHECK-NOT: memref.alloc() : memref<1x16x114x512xf16, #NHWC, @DDR>
    // CHECK-NOT: memref.alloc() : memref<1x16x114x512xi1, #NHWC, @DDR>

    // CHECK:       [[BUFF_0_DATA:%.+]] = memref.alloc() : memref<1x3x114x512xf16, #NHWC, @DDR>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x3x114x512xi1, #NHWC, @DDR>
    // CHECK:       [[BUFF_0:%.+]] = VPUIP.GroupSparseBuffer([[BUFF_0_DATA]], [[BUFF_0_SM]])
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>

    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView %arg0 [0, 3, 0, 0] [1, 3, 57, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x16x57x512xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x16x57x512xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x3x57x512xi1, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 0] [1, 3, 57, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>

    // CHECK:       [[COPY_0:%.+]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW_0]] as %arg4: !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>>)
    // CHECK-SAME:         outputs([[SUBVIEW_1]] as %arg5: !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>> {
    // CHECK:       [[inner_0:%.+]] = VPUIP.Copy inputs(%arg4 : !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>>)
    // CHECK-SAME:         outputs(%arg5 : !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>

    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView %arg1 [0, 3, 0, 0] [1, 3, 57, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x16x57x512xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x16x57x512xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x3x57x512xi1, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // CHECK:       [[SUBVIEW_3:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 0, 57, 0] [1, 3, 57, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>

    // CHECK:       [[COPY_1:%.+]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW_2]] as %arg4: !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>>)
    // CHECK-SAME:         outputs([[SUBVIEW_3]] as %arg5: !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>> {
    // CHECK:       [[inner_1:%.+]] = VPUIP.Copy inputs(%arg4 : !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN>>)
    // CHECK-SAME:         outputs(%arg5 : !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>

    // CHECK:       [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]] : !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>, !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>)
    // CHECK-SAME:         outputs([[BUFF_0]] : !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>

    // CHECK:       [[SUBVIEW_4:%.+]] = VPUIP.SubView [[CONCATVIEW_0]] [0, 0, 0, 0] [1, 3, 110, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x3x110x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x110x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>

    // CHECK:       [[COPY_2:%.+]] = VPUIP.Copy inputs([[SUBVIEW_4]] : !VPUIP.SparseBuffer<data=memref<1x3x110x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x110x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>)
    // CHECK-SAME:         outputs(%arg2 : !VPUIP.SparseBuffer<data=memref<1x3x110x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x110x512xi1, #NHWC, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x110x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x110x512xi1, #NHWC, @DDR>>

    // CHECK:       [[SUBVIEW_5:%.+]] = VPUIP.SubView [[CONCATVIEW_0]] [0, 0, 110, 0] [1, 3, 4, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x3x4x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x4x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>

    // CHECK:       [[COPY_3:%.+]] = VPUIP.Copy inputs([[SUBVIEW_5]] : !VPUIP.SparseBuffer<data=memref<1x3x4x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x4x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>)
    // CHECK-SAME:         outputs(%arg3 : !VPUIP.SparseBuffer<data=memref<1x3x4x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x4x512xi1, #NHWC, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x4x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x4x512xi1, #NHWC, @DDR>>

    // CHECK:       return [[COPY_2]], [[COPY_3]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x72x256xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

func.func @FuseConcatViewOps(
        %arg0: memref<1x8x144x256xf16, #NHWC, @DDR>)
         -> memref<1x24x144x256xf16, #NHWC, @DDR> {
    %input0 = VPURT.AllocDistributed -> !InputDistributed
    %input1 = VPURT.AllocDistributed -> !InputDistributed

    %0 = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @DDR>
    %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 16, 72, 256] : memref<1x16x144x256xf16, #NHWC, @DDR> to memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
    %2 = VPUIP.NCEClusterTiling inputs(%input0 as %arg1: memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%1 as %arg2: memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR> {
        VPUIP.Copy inputs(%arg1 : memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>
    }
    %3 = VPUIP.SubView %0 [0, 0, 72, 0] [1, 16, 72, 256] : memref<1x16x144x256xf16, #NHWC, @DDR> to memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
    %4 = VPUIP.NCEClusterTiling inputs(%input1 as %arg1: memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%3 as %arg2: memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR> {
        VPUIP.Copy inputs(%arg1 : memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>
    }
    %5 = VPUIP.ConcatView inputs(%2, %4 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>, memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>) outputs(%0 : memref<1x16x144x256xf16, #NHWC, @DDR>) -> memref<1x16x144x256xf16, #NHWC, @DDR>

    %6 = memref.alloc() : memref<1x24x144x256xf16, #NHWC, @DDR>
    %7 = VPUIP.SubView %6 [0, 0, 0, 0] [1, 16, 144, 256] : memref<1x24x144x256xf16, #NHWC, @DDR> to memref<1x16x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
    %8 = VPUIP.Copy inputs(%5 : memref<1x16x144x256xf16, #NHWC, @DDR>) outputs(%7 : memref<1x16x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>) -> memref<1x16x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
    %9 = VPUIP.SubView %6 [0, 16, 0, 0] [1, 8, 144, 256] : memref<1x24x144x256xf16, #NHWC, @DDR> to memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
    %10 = VPUIP.Copy inputs(%arg0 : memref<1x8x144x256xf16, #NHWC, @DDR>) outputs(%9 : memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>) -> memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
    %11 = VPUIP.ConcatView inputs(%8, %10 : memref<1x16x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>, memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>) outputs(%6 : memref<1x24x144x256xf16, #NHWC, @DDR>) -> memref<1x24x144x256xf16, #NHWC, @DDR>

    return %11 : memref<1x24x144x256xf16, #NHWC, @DDR>


    // CHECK:       [[INPUT_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x72x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[INPUT_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x72x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[OUTPUT_BUFF:%.+]] = memref.alloc() : memref<1x24x144x256xf16, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 0, 0, 0] [1, 16, 72, 256]
    // CHECK-SAME:          memref<1x24x144x256xf16, #NHWC, @DDR> to memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
    // CHECK:       [[COPY_0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[INPUT_0]] as %arg1: memref<1x16x72x256xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[SUBVIEW_0]] as %arg2: memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>)
    // CHECK-SAME:          -> memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR> {
    // CHECK:       [[COPY_0_INNER:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg1 : memref<1x16x72x256xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs(%arg2 : memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>)
    // CHECK-SAME:          -> memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 0, 72, 0] [1, 16, 72, 256]
    // CHECK-SAME:          memref<1x24x144x256xf16, #NHWC, @DDR> to memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
    // CHECK:       [[COPY_1:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[INPUT_1]] as %arg1: memref<1x16x72x256xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[SUBVIEW_1]] as %arg2: memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>)
    // CHECK-SAME:          -> memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR> {
    // CHECK:       [[COPY_1_INNER:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg1 : memref<1x16x72x256xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs(%arg2 : memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>)
    // CHECK-SAME:          -> memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>

    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 16, 0, 0] [1, 8, 144, 256]
    // CHECK-SAME:          memref<1x24x144x256xf16, #NHWC, @DDR> to memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
    // CHECK:       [[COPY_2:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg0 : memref<1x8x144x256xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_2]] : memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>)
    // CHECK-SAME:          -> memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>

    // CHECK:       [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]], [[COPY_2]]
    // CHECK-SAME:          memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
    // CHECK-SAME:          memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
    // CHECK-SAME:          memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>)
    // CHECK-SAME:          outputs([[OUTPUT_BUFF]] : memref<1x24x144x256xf16, #NHWC, @DDR>) -> memref<1x24x144x256xf16, #NHWC, @DDR>

    // CHECK:       return [[CONCATVIEW]] : memref<1x24x144x256xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x72x256xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>
!InputSMDistributed = !VPUIP.DistributedBuffer<
    1x16x72x256xi1, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!Data0 = memref<1x16x144x256xf16, {order = #NHWC}, @DDR>
!SM0 = memref<1x16x144x256xi1, {order = #NHWC}, @DDR>
!Sparse0 = !VPUIP.SparseBuffer<data=!Data0, sparsity_map=!SM0>

!Data0Strided = memref<1x16x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
!SM0Strided = memref<1x16x144x256xi1, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
!Sparse0Strided = !VPUIP.SparseBuffer<data=!Data0Strided, sparsity_map=!SM0Strided>

!Data1CMX = memref<1x16x72x256xf16, {order = #NHWC}, @CMX_NN>
!SM1CMX = memref<1x16x72x256xi1, {order = #NHWC}, @CMX_NN>
!Sparse1CMX = !VPUIP.SparseBuffer<data=!Data1CMX, sparsity_map=!SM1CMX>

!Data1 = memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
!SM1 = memref<1x16x72x256xi1, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
!Sparse1Strided = !VPUIP.SparseBuffer<data=!Data1, sparsity_map=!SM1>

!Data2 = memref<1x8x144x256xf16, {order = #NHWC}, @DDR>
!SM2 = memref<1x8x144x256xi1, {order = #NHWC}, @DDR>
!Sparse2 = !VPUIP.SparseBuffer<data=!Data2, sparsity_map=!SM2>

!Data2Strided = memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
!SM2Strided = memref<1x8x144x256xi1, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
!Sparse2Strided = !VPUIP.SparseBuffer<data=!Data2Strided, sparsity_map=!SM2Strided>

!Data3 = memref<1x24x144x256xf16, {order = #NHWC}, @DDR>
!SM3 = memref<1x24x144x256xi1, {order = #NHWC}, @DDR>
!Sparse3 = !VPUIP.SparseBuffer<data=!Data3, sparsity_map=!SM3>

func.func @FuseConcatViewOpsSparse(%arg0: !Sparse2) -> !Sparse3 {
    %input0Data = VPURT.AllocDistributed -> !InputDistributed
    %input0SM = VPURT.AllocDistributed -> !InputSMDistributed
    %input0Sparse = VPUIP.GroupSparseBuffer(%input0Data, %input0SM) -> !VPUIP.SparseBuffer<data=!InputDistributed, sparsity_map=!InputSMDistributed>

    %input1Data = VPURT.AllocDistributed -> !InputDistributed
    %input1SM = VPURT.AllocDistributed -> !InputSMDistributed
    %input1Sparse = VPUIP.GroupSparseBuffer(%input1Data, %input1SM) -> !VPUIP.SparseBuffer<data=!InputDistributed, sparsity_map=!InputSMDistributed>

    %0 = memref.alloc() : !Data0
    %1 = memref.alloc() : !SM0
    %2 = VPUIP.GroupSparseBuffer(%0, %1) -> !Sparse0
    %3 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 16, 72, 256] : !Sparse0 to !Sparse1Strided
    %4 = VPUIP.NCEClusterTiling inputs(%input0Sparse as %arg1: !Sparse1CMX) outputs(%3 as %arg2: !Sparse1Strided) -> !Sparse1Strided {
        VPUIP.Copy inputs(%arg1 : !Sparse1CMX) outputs(%arg2 : !Sparse1Strided) -> !Sparse1Strided
    }
    %5 = VPUIP.SubView %2 [0, 0, 72, 0] [1, 16, 72, 256] : !Sparse0 to !Sparse1Strided
    %6 = VPUIP.NCEClusterTiling inputs(%input1Sparse as %arg1: !Sparse1CMX) outputs(%5 as %arg2: !Sparse1Strided) -> !Sparse1Strided {
        VPUIP.Copy inputs(%arg1 : !Sparse1CMX) outputs(%arg2 : !Sparse1Strided) -> !Sparse1Strided
    }
    %7 = VPUIP.ConcatView inputs(%4, %6 : !Sparse1Strided, !Sparse1Strided) outputs(%2 : !Sparse0) -> !Sparse0

    %8 = memref.alloc() : !Data3
    %9 = memref.alloc() : !SM3
    %10 = VPUIP.GroupSparseBuffer(%8, %9) -> !Sparse3
    %11 = VPUIP.SubView %10 [0, 0, 0, 0] [1, 16, 144, 256] : !Sparse3 to !Sparse0Strided
    %12 = VPUIP.Copy inputs(%7 : !Sparse0) outputs(%11 : !Sparse0Strided) -> !Sparse0Strided
    %13 = VPUIP.SubView %10 [0, 16, 0, 0] [1, 8, 144, 256] : !Sparse3 to !Sparse2Strided
    %14 = VPUIP.Copy inputs(%arg0 : !Sparse2) outputs(%13 : !Sparse2Strided) -> !Sparse2Strided
    %15 = VPUIP.ConcatView inputs(%12, %14 : !Sparse0Strided, !Sparse2Strided) outputs(%10 : !Sparse3) -> !Sparse3

    return %15 : !Sparse3

    // CHECK:       [[INPUT_0_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x72x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[INPUT_0_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x72x256xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[INPUT_0_SPARSE:%.+]] = VPUIP.GroupSparseBuffer([[INPUT_0_DATA]], [[INPUT_0_SM]])

    // CHECK:       [[INPUT_1_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x72x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[INPUT_1_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x72x256xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[INPUT_1_SPARSE:%.+]] = VPUIP.GroupSparseBuffer([[INPUT_1_DATA]], [[INPUT_1_SM]])

    // CHECK:       [[OUTPUT_DATA:%.+]] = memref.alloc() : memref<1x24x144x256xf16, {order = #NHWC}, @DDR>
    // CHECK:       [[OUTPUT_SM:%.+]] = memref.alloc() : memref<1x24x144x256xi1, {order = #NHWC}, @DDR>
    // CHECK:       [[OUTPUT_SPARSE:%.+]] = VPUIP.GroupSparseBuffer([[OUTPUT_DATA]], [[OUTPUT_SM]])

    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[OUTPUT_SPARSE]] [0, 0, 0, 0] [1, 16, 72, 256]
    // CHECK-SAME:          !VPUIP.SparseBuffer<data=memref<1x24x144x256xf16, {order = #NHWC}, @DDR>, sparsity_map=memref<1x24x144x256xi1, {order = #NHWC}, @DDR>> to
    // CHECK-SAME:          !VPUIP.SparseBuffer<data=memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>, sparsity_map=memref<1x16x72x256xi1, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>>
    // CHECK:       [[COPY_0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[INPUT_0_SPARSE]] as %arg1: !VPUIP.SparseBuffer<data=memref<1x16x72x256xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x16x72x256xi1, #NHWC, @CMX_NN>>)
    // CHECK-SAME:      outputs([[SUBVIEW_0]] as %arg2: !VPUIP.SparseBuffer<data=memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>, sparsity_map=memref<1x16x72x256xi1, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>, sparsity_map=memref<1x16x72x256xi1, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>> {
    // CHECK:       [[COPY_0_INNER:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg1
    // CHECK-SAME:      outputs(%arg2

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUTPUT_SPARSE]] [0, 0, 72, 0] [1, 16, 72, 256]
    // CHECK-SAME:          !VPUIP.SparseBuffer<data=memref<1x24x144x256xf16, {order = #NHWC}, @DDR>, sparsity_map=memref<1x24x144x256xi1, {order = #NHWC}, @DDR>> to
    // CHECK-SAME:          !VPUIP.SparseBuffer<data=memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>, sparsity_map=memref<1x16x72x256xi1, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>>
    // CHECK:       [[COPY_1:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[INPUT_1_SPARSE]] as %arg1: !VPUIP.SparseBuffer<data=memref<1x16x72x256xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x16x72x256xi1, #NHWC, @CMX_NN>>)
    // CHECK-SAME:      outputs([[SUBVIEW_1]] as %arg2: !VPUIP.SparseBuffer<data=memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>, sparsity_map=memref<1x16x72x256xi1, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>, sparsity_map=memref<1x16x72x256xi1, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>> {
    // CHECK:       [[COPY_1_INNER:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg1
    // CHECK-SAME:      outputs(%arg2

    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView [[OUTPUT_SPARSE]] [0, 16, 0, 0] [1, 8, 144, 256]
    // CHECK-SAME:          !VPUIP.SparseBuffer<data=memref<1x24x144x256xf16, {order = #NHWC}, @DDR>, sparsity_map=memref<1x24x144x256xi1, {order = #NHWC}, @DDR>> to
    // CHECK-SAME:          !VPUIP.SparseBuffer<data=memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>, sparsity_map=memref<1x8x144x256xi1, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>>
    // CHECK:       [[COPY_2:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg0
    // CHECK-SAME:      outputs([[SUBVIEW_2]]

    // CHECK:       [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]], [[COPY_2]]
    // CHECK-SAME:          !VPUIP.SparseBuffer<data=memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>, sparsity_map=memref<1x16x72x256xi1, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>>
    // CHECK-SAME:          !VPUIP.SparseBuffer<data=memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>, sparsity_map=memref<1x16x72x256xi1, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>>
    // CHECK-SAME:          !VPUIP.SparseBuffer<data=memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>, sparsity_map=memref<1x8x144x256xi1, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>>)
    // CHECK-SAME:          outputs([[OUTPUT_SPARSE]] : !VPUIP.SparseBuffer<data=memref<1x24x144x256xf16, {order = #NHWC}, @DDR>, sparsity_map=memref<1x24x144x256xi1, {order = #NHWC}, @DDR>>)
    // CHECK-SAME:      -> !VPUIP.SparseBuffer<data=memref<1x24x144x256xf16, {order = #NHWC}, @DDR>, sparsity_map=memref<1x24x144x256xi1, {order = #NHWC}, @DDR>>

    // CHECK:       return [[CONCATVIEW]] : !VPUIP.SparseBuffer<data=memref<1x24x144x256xf16, {order = #NHWC}, @DDR>, sparsity_map=memref<1x24x144x256xi1, {order = #NHWC}, @DDR>>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x32x96x336xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

func.func @NotFuseConcatViewOpsWithStrideLevelIs3( ) -> memref<1x32x384x672xf16, #NHWC, @DDR> {
    %0 = VPURT.AllocDistributed -> !InputDistributed
    %1 = VPURT.AllocDistributed -> !InputDistributed
    %2 = VPURT.AllocDistributed -> !InputDistributed
    %3 = VPURT.AllocDistributed -> !InputDistributed

    %4 = memref.alloc() : memref<1x32x192x672xf16, #NHWC, @DDR>
    %5 = VPUIP.SubView %4 [0, 0, 0, 0] [1, 32, 96, 336] [1, 1, 1, 2]
            : memref<1x32x192x672xf16, #NHWC, @DDR> to memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>
    %6 = VPUIP.NCEClusterTiling
            inputs(%0 as %arg2: memref<1x32x96x336xf16, #NHWC, @CMX_NN>)
            outputs(%5 as %arg3: memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>) -> memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR> {
            VPUIP.Copy inputs(%arg2 : memref<1x32x96x336xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>) -> memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>
    }

    %7 = VPUIP.SubView %4 [0, 0, 96, 0] [1, 32, 96, 336] [1, 1, 1, 2]
            : memref<1x32x192x672xf16, #NHWC, @DDR> to memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>
    %8 = VPUIP.NCEClusterTiling
            inputs(%1 as %arg2: memref<1x32x96x336xf16, #NHWC, @CMX_NN>)
            outputs(%7 as %arg3: memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>) -> memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR> {
            VPUIP.Copy inputs(%arg2 : memref<1x32x96x336xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>) -> memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>
    }

    %9 = VPUIP.SubView %4 [0, 0, 0, 1] [1, 32, 96, 336] [1, 1, 1, 2]
            : memref<1x32x192x672xf16, #NHWC, @DDR> to memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>
    %10 = VPUIP.NCEClusterTiling
            inputs(%2 as %arg2: memref<1x32x96x336xf16, #NHWC, @CMX_NN>)
            outputs(%9 as %arg3: memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>) -> memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR> {
            VPUIP.Copy inputs(%arg2 : memref<1x32x96x336xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>) -> memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>
    }

    %11 = VPUIP.SubView %4 [0, 0, 96, 1] [1, 32, 96, 336] [1, 1, 1, 2]
            : memref<1x32x192x672xf16, #NHWC, @DDR> to memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>
    %12 = VPUIP.NCEClusterTiling
            inputs(%3 as %arg2: memref<1x32x96x336xf16, #NHWC, @CMX_NN>)
            outputs(%11 as %arg3: memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>) -> memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR> {
            VPUIP.Copy inputs(%arg2 : memref<1x32x96x336xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>) -> memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>
    }

    %13 = VPUIP.ConcatView inputs(%6, %8, %10, %12 :
                memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>,
                memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>,
                memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>,
                memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>)
                outputs(%4 : memref<1x32x192x672xf16, #NHWC, @DDR>)
                    -> memref<1x32x192x672xf16, #NHWC, @DDR>

    %14 = memref.alloc() : memref<1x32x384x672xf16, #NHWC, @DDR>
    %15 = VPUIP.SubView %14 [0, 0, 0, 0] [1, 32, 192, 672] [1, 1, 2, 1]
            : memref<1x32x384x672xf16, #NHWC, @DDR> to memref<1x32x192x672xf16, {order = #NHWC, strides = [8257536, 1, 43008, 32]}, @DDR>
    %16 = memref.alloc() : memref<1x32x192x672xf16, #NHWC, @DDR>
    %17 = VPUIP.Copy inputs(%16 : memref<1x32x192x672xf16, #NHWC, @DDR>) outputs(%15 : memref<1x32x192x672xf16, {order = #NHWC, strides = [8257536, 1, 43008, 32]}, @DDR>) -> memref<1x32x192x672xf16, {order = #NHWC, strides = [8257536, 1, 43008, 32]}, @DDR>

    %18 = VPUIP.SubView %14 [0, 0, 1, 0] [1, 32, 192, 672] [1, 1, 2, 1]
            : memref<1x32x384x672xf16, #NHWC, @DDR> to memref<1x32x192x672xf16, {order = #NHWC, strides = [8257536, 1, 43008, 32]}, @DDR>
    %19 = VPUIP.Copy inputs(%13 : memref<1x32x192x672xf16, #NHWC, @DDR>) outputs(%18 : memref<1x32x192x672xf16, {order = #NHWC, strides = [8257536, 1, 43008, 32]}, @DDR>) -> memref<1x32x192x672xf16, {order = #NHWC, strides = [8257536, 1, 43008, 32]}, @DDR>

    %20 = VPUIP.ConcatView inputs(%17, %19 :
                memref<1x32x192x672xf16, {order = #NHWC, strides = [8257536, 1, 43008, 32]}, @DDR>,
                memref<1x32x192x672xf16, {order = #NHWC, strides = [8257536, 1, 43008, 32]}, @DDR>)
                outputs(%14 : memref<1x32x384x672xf16, #NHWC, @DDR>) -> memref<1x32x384x672xf16, #NHWC, @DDR>

    return %20 : memref<1x32x384x672xf16, #NHWC, @DDR>


    // CHECK:       [[INPUT_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer
    // CHECK:       [[INPUT_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer
    // CHECK:       [[INPUT_2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer
    // CHECK:       [[INPUT_3:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer

    // CHECK:       [[OUTPUT_BUFF_0:%.+]] = memref.alloc() : memref<1x32x192x672xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_0]] [0, 0, 0, 0] [1, 32, 96, 336] [1, 1, 1, 2]
    // CHECK:       [[COPY_0:%.+]] = VPUIP.NCEClusterTiling inputs([[INPUT_0]] as %arg0: memref<1x32x96x336xf16, #NHWC, @CMX_NN>) outputs([[SUBVIEW_0]] as %arg1: memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>)
    // CHECK:       [[COPY_0_INNER:%.+]] = VPUIP.Copy inputs(%arg0 : memref<1x32x96x336xf16, #NHWC, @CMX_NN>) outputs(%arg1 : memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>)

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_0]] [0, 0, 96, 0] [1, 32, 96, 336] [1, 1, 1, 2]
    // CHECK:       [[COPY_1:%.+]] = VPUIP.NCEClusterTiling inputs([[INPUT_1]] as %arg0: memref<1x32x96x336xf16, #NHWC, @CMX_NN>) outputs([[SUBVIEW_1]] as %arg1: memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>)
    // CHECK:       [[COPY_1_INNER:%.+]] = VPUIP.Copy inputs(%arg0 : memref<1x32x96x336xf16, #NHWC, @CMX_NN>) outputs(%arg1 : memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>)

    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_0]] [0, 0, 0, 1] [1, 32, 96, 336] [1, 1, 1, 2]
    // CHECK:       [[COPY_2:%.+]] = VPUIP.NCEClusterTiling inputs([[INPUT_2]] as %arg0: memref<1x32x96x336xf16, #NHWC, @CMX_NN>) outputs([[SUBVIEW_2]] as %arg1: memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>)
    // CHECK:       [[COPY_2_INNER:%.+]] = VPUIP.Copy inputs(%arg0 : memref<1x32x96x336xf16, #NHWC, @CMX_NN>) outputs(%arg1 : memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>)

    // CHECK:       [[SUBVIEW_3:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_0]] [0, 0, 96, 1] [1, 32, 96, 336] [1, 1, 1, 2]
    // CHECK:       [[COPY_3:%.+]] = VPUIP.NCEClusterTiling inputs([[INPUT_3]] as %arg0: memref<1x32x96x336xf16, #NHWC, @CMX_NN>) outputs([[SUBVIEW_3]] as %arg1: memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>)
    // CHECK:       [[COPY_3_INNER:%.+]] = VPUIP.Copy inputs(%arg0 : memref<1x32x96x336xf16, #NHWC, @CMX_NN>) outputs(%arg1 : memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>)

    // CHECK:       [[CONCAT_0:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]], [[COPY_2]], [[COPY_3]]

    // CHECK:       [[OUTPUT_BUFF_1:%.+]] = memref.alloc() : memref<1x32x384x672xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_4:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_1]] [0, 0, 0, 0] [1, 32, 192, 672] [1, 1, 2, 1]
    // CHECK:       [[INPUT_4:%.+]] = memref.alloc() : memref<1x32x192x672xf16, #NHWC, @DDR>
    // CHECK:       [[COPY_4:%.+]] = VPUIP.Copy inputs([[INPUT_4]] : memref<1x32x192x672xf16, #NHWC, @DDR>) outputs([[SUBVIEW_4]] : memref<1x32x192x672xf16, {order = #NHWC, strides = [8257536, 1, 43008, 32]}, @DDR>)

    // CHECK:       [[SUBVIEW_5:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_1]] [0, 0, 1, 0] [1, 32, 192, 672] [1, 1, 2, 1]
    // CHECK:       [[COPY_5:%.+]] = VPUIP.Copy inputs([[CONCAT_0]] : memref<1x32x192x672xf16, #NHWC, @DDR>) outputs([[SUBVIEW_5]] : memref<1x32x192x672xf16, {order = #NHWC, strides = [8257536, 1, 43008, 32]}, @DDR>)

    // CHECK:       [[CONCAT_1:%.+]] = VPUIP.ConcatView inputs([[COPY_4]], [[COPY_5]]

    // CHECK:       return [[CONCAT_1]] : memref<1x32x384x672xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x72x256xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

func.func @NotFuseWhenMoreThanOneCopyBetweenConcatView(
        %arg0: memref<1x8x144x256xf16, #NHWC, @DDR>)
         -> memref<1x40x144x256xf16, #NHWC, @DDR> {
    %input0 = VPURT.AllocDistributed -> !InputDistributed
    %input1 = VPURT.AllocDistributed -> !InputDistributed

    %0 = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @DDR>
    %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 16, 72, 256] : memref<1x16x144x256xf16, #NHWC, @DDR> to memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
    %2 = VPUIP.NCEClusterTiling inputs(%input0 as %arg1: memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%1 as %arg2: memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR> {
        VPUIP.Copy inputs(%arg1 : memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>
    }
    %3 = VPUIP.SubView %0 [0, 0, 72, 0] [1, 16, 72, 256] : memref<1x16x144x256xf16, #NHWC, @DDR> to memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
    %4 = VPUIP.NCEClusterTiling inputs(%input1 as %arg1: memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%3 as %arg2: memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR> {
        VPUIP.Copy inputs(%arg1 : memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>
    }
    %5 = VPUIP.ConcatView inputs(%2, %4 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>, memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>) outputs(%0 : memref<1x16x144x256xf16, #NHWC, @DDR>) -> memref<1x16x144x256xf16, #NHWC, @DDR>

    %6 = memref.alloc() : memref<1x40x144x256xf16, #NHWC, @DDR>
    %7 = VPUIP.SubView %6 [0, 0, 0, 0] [1, 16, 144, 256] : memref<1x40x144x256xf16, #NHWC, @DDR> to memref<1x16x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>
    %8 = VPUIP.Copy inputs(%5 : memref<1x16x144x256xf16, #NHWC, @DDR>) outputs(%7 : memref<1x16x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>) -> memref<1x16x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>

    %9 = VPUIP.SubView %6 [0, 16, 0, 0] [1, 16, 144, 256] : memref<1x40x144x256xf16, #NHWC, @DDR> to memref<1x16x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>
    %10 = VPUIP.Copy inputs(%5 : memref<1x16x144x256xf16, #NHWC, @DDR>) outputs(%9 : memref<1x16x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>) -> memref<1x16x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>

    %11 = VPUIP.SubView %6 [0, 32, 0, 0] [1, 8, 144, 256] : memref<1x40x144x256xf16, #NHWC, @DDR> to memref<1x8x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>
    %12 = VPUIP.Copy inputs(%arg0 : memref<1x8x144x256xf16, #NHWC, @DDR>) outputs(%11 : memref<1x8x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>) -> memref<1x8x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>
    %13 = VPUIP.ConcatView inputs(%8, %10, %12 :
                memref<1x16x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>,
                memref<1x16x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>,
                memref<1x8x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>)
                outputs(%6 : memref<1x40x144x256xf16, #NHWC, @DDR>) -> memref<1x40x144x256xf16, #NHWC, @DDR>

    return %13 : memref<1x40x144x256xf16, #NHWC, @DDR>

    // CHECK:       [[INPUT_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer
    // CHECK:       [[INPUT_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer

    // CHECK:       [[OUTPUT_BUFF_0:%.+]] = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_0]] [0, 0, 0, 0] [1, 16, 72, 256]
    // CHECK:       [[COPY_0:%.+]] = VPUIP.NCEClusterTiling inputs([[INPUT_0]] as %arg1: memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs([[SUBVIEW_0]] as %arg2: memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>)
    // CHECK:       [[COPY_0_INNER:%.+]] = VPUIP.Copy inputs(%arg1 : memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>)

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_0]] [0, 0, 72, 0] [1, 16, 72, 256]
    // CHECK:       [[COPY_1:%.+]] = VPUIP.NCEClusterTiling inputs([[INPUT_1]] as %arg1: memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs([[SUBVIEW_1]] as %arg2: memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>)
    // CHECK:       [[COPY_1_INNER:%.+]] = VPUIP.Copy inputs(%arg1 : memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>)

    // CHECK:       [[CONCAT_0:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]]

    // CHECK:       [[OUTPUT_BUFF_1:%.+]] = memref.alloc() : memref<1x40x144x256xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_4:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_1]] [0, 0, 0, 0] [1, 16, 144, 256]
    // CHECK:       [[COPY_4:%.+]] = VPUIP.Copy inputs([[CONCAT_0]] : memref<1x16x144x256xf16, #NHWC, @DDR>) outputs([[SUBVIEW_4]] : memref<1x16x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>)

    // CHECK:       [[SUBVIEW_5:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_1]] [0, 16, 0, 0] [1, 16, 144, 256]
    // CHECK:       [[COPY_5:%.+]] = VPUIP.Copy inputs([[CONCAT_0]] : memref<1x16x144x256xf16, #NHWC, @DDR>) outputs([[SUBVIEW_5]] : memref<1x16x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>)

    // CHECK:       [[SUBVIEW_6:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_1]] [0, 32, 0, 0] [1, 8, 144, 256]
    // CHECK:       [[COPY_6:%.+]] = VPUIP.Copy inputs(%arg0 : memref<1x8x144x256xf16, #NHWC, @DDR>) outputs([[SUBVIEW_6]] : memref<1x8x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>)

    // CHECK:       [[CONCAT_1:%.+]] = VPUIP.ConcatView inputs([[COPY_4]], [[COPY_5]], [[COPY_6]]

    // CHECK:       return [[CONCAT_1]] : memref<1x40x144x256xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
	1x16x72x256xf16, #NHWC, @CMX_NN, {
	mode = "SEGMENTED",
	num_tiles = [1, 1, 2, 1],
	num_clusters = 2
}>

func.func @OneCopyAfterConcatViewHasNoUser(
		%arg0: memref<1x8x144x256xf16, #NHWC, @DDR>,
        %arg1: memref<1x16x144x256xf16, #NHWC, @DDR>)
		-> memref<1x16x144x256xf16, #NHWC, @DDR> {
	%input0 = VPURT.AllocDistributed -> !InputDistributed
	%input1 = VPURT.AllocDistributed -> !InputDistributed

	%0 = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @DDR>
	%1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 16, 72, 256] : memref<1x16x144x256xf16, #NHWC, @DDR> to memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
	%2 = VPUIP.NCEClusterTiling inputs(%input0 as %arg2: memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%1 as %arg3: memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR> {
		VPUIP.Copy inputs(%arg2 : memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>
	}
	%3 = VPUIP.SubView %0 [0, 0, 72, 0] [1, 16, 72, 256] : memref<1x16x144x256xf16, #NHWC, @DDR> to memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
	%4 = VPUIP.NCEClusterTiling inputs(%input1 as %arg2: memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%3 as %arg3: memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR> {
		VPUIP.Copy inputs(%arg2 : memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>
	}
	%5 = VPUIP.ConcatView inputs(%2, %4 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>, memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>) outputs(%0 : memref<1x16x144x256xf16, #NHWC, @DDR>) -> memref<1x16x144x256xf16, #NHWC, @DDR>

	%7 = VPUIP.Copy inputs(%5 : memref<1x16x144x256xf16, #NHWC, @DDR>) outputs(%arg1 : memref<1x16x144x256xf16, #NHWC, @DDR>) -> memref<1x16x144x256xf16, #NHWC, @DDR>

	return %arg1 : memref<1x16x144x256xf16, #NHWC, @DDR>

	// CHECK: [[INPUT_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer
	// CHECK: [[INPUT_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer

	// CHECK: [[OUTPUT_BUFF_0:%.+]] = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @DDR>
	// CHECK: [[SUBVIEW_0:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_0]] [0, 0, 0, 0] [1, 16, 72, 256]
	// CHECK: [[COPY_0:%.+]] = VPUIP.NCEClusterTiling inputs([[INPUT_0]] as %arg2: memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs([[SUBVIEW_0]] as %arg3: memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>)
	// CHECK: [[COPY_0_INNER:%.+]] = VPUIP.Copy inputs(%arg2 : memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>)

	// CHECK: [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_0]] [0, 0, 72, 0] [1, 16, 72, 256]
	// CHECK: [[COPY_1:%.+]] = VPUIP.NCEClusterTiling inputs([[INPUT_1]] as %arg2: memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs([[SUBVIEW_1]] as %arg3: memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>)
	// CHECK: [[COPY_1_INNER:%.+]] = VPUIP.Copy inputs(%arg2 : memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>)

	// CHECK: [[CONCAT_0:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]]
	// CHECK: [[COPY_4:%.+]] = VPUIP.Copy inputs([[CONCAT_0]] : memref<1x16x144x256xf16, #NHWC, @DDR>) outputs(%arg1 : memref<1x16x144x256xf16, #NHWC, @DDR>)

	// CHECK: return %arg1 : memref<1x16x144x256xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
	1x16x72x256xf16, #NHWC, @CMX_NN, {
	mode = "SEGMENTED",
	num_tiles = [1, 1, 2, 1],
	num_clusters = 2
}>

func.func @OneCopyAfterConcatViewHasMultiUser(
		%arg0: memref<1x8x144x256xf16, #NHWC, @DDR>)
		-> (memref<1x16x144x256xf16, #NHWC, @DDR>, memref<1x16x144x256xf16, #NHWC, @CMX_NN>) {
	%input0 = VPURT.AllocDistributed -> !InputDistributed
	%input1 = VPURT.AllocDistributed -> !InputDistributed

	%0 = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @DDR>
	%1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 16, 72, 256] : memref<1x16x144x256xf16, #NHWC, @DDR> to memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
	%2 = VPUIP.NCEClusterTiling inputs(%input0 as %arg1: memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%1 as %arg2: memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR> {
		VPUIP.Copy inputs(%arg1 : memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>
	}
	%3 = VPUIP.SubView %0 [0, 0, 72, 0] [1, 16, 72, 256] : memref<1x16x144x256xf16, #NHWC, @DDR> to memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
	%4 = VPUIP.NCEClusterTiling inputs(%input1 as %arg1: memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%3 as %arg2: memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR> {
		VPUIP.Copy inputs(%arg1 : memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>
	}
	%5 = VPUIP.ConcatView inputs(%2, %4 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>, memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>) outputs(%0 : memref<1x16x144x256xf16, #NHWC, @DDR>) -> memref<1x16x144x256xf16, #NHWC, @DDR>

    %6 = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @DDR>
	%7 = VPUIP.Copy inputs(%5 : memref<1x16x144x256xf16, #NHWC, @DDR>) outputs(%6 : memref<1x16x144x256xf16, #NHWC, @DDR>) -> memref<1x16x144x256xf16, #NHWC, @DDR>

	%8 = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @CMX_NN>
    %9 = VPUIP.Copy inputs(%7 : memref<1x16x144x256xf16, #NHWC, @DDR>) outputs(%8 : memref<1x16x144x256xf16, #NHWC, @CMX_NN>) -> memref<1x16x144x256xf16, #NHWC, @CMX_NN>

	return %7, %9 : memref<1x16x144x256xf16, #NHWC, @DDR>, memref<1x16x144x256xf16, #NHWC, @CMX_NN>

	// CHECK: [[INPUT_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer
	// CHECK: [[INPUT_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer

	// CHECK: [[OUTPUT_BUFF_0:%.+]] = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @DDR>
	// CHECK: [[SUBVIEW_0:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_0]] [0, 0, 0, 0] [1, 16, 72, 256]
	// CHECK: [[COPY_0:%.+]] = VPUIP.NCEClusterTiling inputs([[INPUT_0]] as %arg1: memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs([[SUBVIEW_0]] as %arg2: memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>)
	// CHECK: [[COPY_0_INNER:%.+]] = VPUIP.Copy inputs(%arg1 : memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>)

	// CHECK: [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_0]] [0, 0, 72, 0] [1, 16, 72, 256]
	// CHECK: [[COPY_1:%.+]] = VPUIP.NCEClusterTiling inputs([[INPUT_1]] as %arg1: memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs([[SUBVIEW_1]] as %arg2: memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>)
	// CHECK: [[COPY_1_INNER:%.+]] = VPUIP.Copy inputs(%arg1 : memref<1x16x72x256xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}>)

	// CHECK: [[CONCAT_0:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]]

    // CHECK: [[OUTPUT_BUFF_0:%.+]] = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @DDR>
	// CHECK: [[COPY_4:%.+]] = VPUIP.Copy inputs([[CONCAT_0]] : memref<1x16x144x256xf16, #NHWC, @DDR>) outputs([[OUTPUT_BUFF_0]] : memref<1x16x144x256xf16, #NHWC, @DDR>)

    // CHECK: [[OUTPUT_BUFF_1:%.+]] = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @CMX_NN>
	// CHECK: [[COPY_5:%.+]] = VPUIP.Copy inputs([[COPY_4]] : memref<1x16x144x256xf16, #NHWC, @DDR>) outputs([[OUTPUT_BUFF_1]] : memref<1x16x144x256xf16, #NHWC, @CMX_NN>)

	// CHECK: return [[COPY_4]], [[COPY_5]] : memref<1x16x144x256xf16, #NHWC, @DDR>, memref<1x16x144x256xf16, #NHWC, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @AvoidConcatExtraChannelToReduceDataMovement(
        %arg0: memref<1x32x360x640xf16, #NHWC, @DDR>,
        %arg1: memref<1x1x90x640xf16, #NHWC, @DDR>)
         -> memref<1x1x90x640xf16, #NHWC, @DDR>{
    %cst_0= const.Declare memref<16x32x1x1xf16, #NHWC> = dense<1.0> : tensor<16x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    %0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 32, 30, 640] : memref<1x32x360x640xf16, #NHWC, @DDR> to memref<1x32x30x640xf16, {order = #NHWC, strides = [7372800, 1, 20480, 32]}, @DDR>
    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %2 = VPUIP.NCEClusterTiling inputs(%0 as %arg2: memref<1x32x30x640xf16, #NHWC>) outputs(%1 as %arg3: memref<1x32x30x640xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
        %38 = VPUIP.Copy inputs(%arg2 : memref<1x32x30x640xf16, #NHWC>) outputs(%arg3 : memref<1x32x30x640xf16, #NHWC, @CMX_NN>) -> memref<1x32x30x640xf16, #NHWC, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %4 = VPUIP.NCEClusterTiling inputs(%cst_0 as %arg2: memref<16x32x1x1xf16, #NHWC>) outputs(%3 as %arg3: memref<16x32x1x1xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
        %38 = VPUIP.Copy inputs(%arg2 : memref<16x32x1x1xf16, #NHWC>) outputs(%arg3 : memref<16x32x1x1xf16, #NHWC, @CMX_NN>) -> memref<16x32x1x1xf16, #NHWC, @CMX_NN>
    }
    %5 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %6 = VPUIP.NCEClusterTiling inputs(%cst_1 as %arg2: memref<16x1x1x4xsi32>) outputs(%5 as %arg3: memref<16x1x1x4xsi32, @CMX_NN>) -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
        %38 = VPUIP.Copy inputs(%arg2 : memref<16x1x1x4xsi32>) outputs(%arg3 : memref<16x1x1x4xsi32, @CMX_NN>) -> memref<16x1x1x4xsi32, @CMX_NN>
    }
    %7 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %8 = VPUIP.NCEClusterTiling inputs(%2 as %arg2: memref<1x32x30x640xf16, #NHWC, @CMX_NN>, %4 as %arg3: memref<16x32x1x1xf16, #NHWC, @CMX_NN>, %6 as %arg4: memref<16x1x1x4xsi32, @CMX_NN>) outputs(%7 as %arg5: memref<1x16x30x640xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
        %38 = VPUIP.NCEClusterTask {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 11628 : i64, task_type = "CONV"} input(%arg2 : memref<1x32x30x640xf16, #NHWC, @CMX_NN>) weights(%arg3 : memref<16x32x1x1xf16, #NHWC, @CMX_NN>) weight_table(%arg4 : memref<16x1x1x4xsi32, @CMX_NN>) parent_input(%arg2 : memref<1x32x30x640xf16, #NHWC, @CMX_NN>) parent_output(%arg5 : memref<1x16x30x640xf16, #NHWC, @CMX_NN>) outputs(%arg5 : memref<1x16x30x640xf16, #NHWC, @CMX_NN>) -> memref<1x16x30x640xf16, #NHWC, @CMX_NN> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_16x16", outEnd = [639, 14, 15], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
        DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_16x16", outEnd = [639, 29, 15], outStart = [0, 15, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
        } PPE : {
        PPETask "NOOP" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
        }
    }
    %9 = memref.alloc() : memref<1x16x90x640xf16, #NHWC, @DDR>
    %10 = VPUIP.SubView %9 [0, 0, 0, 0] [1, 16, 30, 640] : memref<1x16x90x640xf16, #NHWC, @DDR> to memref<1x16x30x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR>
    %11 = VPUIP.NCEClusterTiling inputs(%8 as %arg2: memref<1x16x30x640xf16, #NHWC, @CMX_NN>) outputs(%10 as %arg3: memref<1x16x30x640xf16, #NHWC>) -> memref<1x16x30x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR> {
        %38 = VPUIP.Copy inputs(%arg2 : memref<1x16x30x640xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x16x30x640xf16, #NHWC>) -> memref<1x16x30x640xf16, #NHWC>
    }

    %12 = VPUIP.SubView %arg0 [0, 0, 30, 0] [1, 32, 30, 640] : memref<1x32x360x640xf16, #NHWC, @DDR> to memref<1x32x30x640xf16, {order = #NHWC, strides = [7372800, 1, 20480, 32]}, @DDR>
    %13 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %14 = VPUIP.NCEClusterTiling inputs(%12 as %arg2: memref<1x32x30x640xf16, #NHWC>) outputs(%13 as %arg3: memref<1x32x30x640xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
        %38 = VPUIP.Copy inputs(%arg2 : memref<1x32x30x640xf16, #NHWC>) outputs(%arg3 : memref<1x32x30x640xf16, #NHWC, @CMX_NN>) -> memref<1x32x30x640xf16, #NHWC, @CMX_NN>
    }
    %15 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %16 = VPUIP.NCEClusterTiling inputs(%cst_0 as %arg2: memref<16x32x1x1xf16, #NHWC>) outputs(%15 as %arg3: memref<16x32x1x1xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
        %38 = VPUIP.Copy inputs(%arg2 : memref<16x32x1x1xf16, #NHWC>) outputs(%arg3 : memref<16x32x1x1xf16, #NHWC, @CMX_NN>) -> memref<16x32x1x1xf16, #NHWC, @CMX_NN>
    }
    %17 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %18 = VPUIP.NCEClusterTiling inputs(%cst_1 as %arg2: memref<16x1x1x4xsi32>) outputs(%17 as %arg3: memref<16x1x1x4xsi32, @CMX_NN>) -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
        %38 = VPUIP.Copy inputs(%arg2 : memref<16x1x1x4xsi32>) outputs(%arg3 : memref<16x1x1x4xsi32, @CMX_NN>) -> memref<16x1x1x4xsi32, @CMX_NN>
    }
    %19 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %20 = VPUIP.NCEClusterTiling inputs(%14 as %arg2: memref<1x32x30x640xf16, #NHWC, @CMX_NN>, %16 as %arg3: memref<16x32x1x1xf16, #NHWC, @CMX_NN>, %18 as %arg4: memref<16x1x1x4xsi32, @CMX_NN>) outputs(%19 as %arg5: memref<1x16x30x640xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
        %38 = VPUIP.NCEClusterTask {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 11628 : i64, task_type = "CONV"} input(%arg2 : memref<1x32x30x640xf16, #NHWC, @CMX_NN>) weights(%arg3 : memref<16x32x1x1xf16, #NHWC, @CMX_NN>) weight_table(%arg4 : memref<16x1x1x4xsi32, @CMX_NN>) parent_input(%arg2 : memref<1x32x30x640xf16, #NHWC, @CMX_NN>) parent_output(%arg5 : memref<1x16x30x640xf16, #NHWC, @CMX_NN>) outputs(%arg5 : memref<1x16x30x640xf16, #NHWC, @CMX_NN>) -> memref<1x16x30x640xf16, #NHWC, @CMX_NN> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_16x16", outEnd = [639, 14, 15], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
        DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_16x16", outEnd = [639, 29, 15], outStart = [0, 15, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
        } PPE : {
        PPETask "NOOP" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
        }
    }
    %21 = VPUIP.SubView %9 [0, 0, 30, 0] [1, 16, 30, 640] : memref<1x16x90x640xf16, #NHWC, @DDR> to memref<1x16x30x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR>
    %22 = VPUIP.NCEClusterTiling inputs(%20 as %arg2: memref<1x16x30x640xf16, #NHWC, @CMX_NN>) outputs(%21 as %arg3: memref<1x16x30x640xf16, #NHWC>) -> memref<1x16x30x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR> {
        %38 = VPUIP.Copy inputs(%arg2 : memref<1x16x30x640xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x16x30x640xf16, #NHWC>) -> memref<1x16x30x640xf16, #NHWC>
    }

    %23 = VPUIP.SubView %arg0 [0, 0, 60, 0] [1, 32, 30, 640] : memref<1x32x360x640xf16, #NHWC, @DDR> to memref<1x32x30x640xf16, {order = #NHWC, strides = [7372800, 1, 20480, 32]}, @DDR>
    %24 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %25 = VPUIP.NCEClusterTiling inputs(%23 as %arg2: memref<1x32x30x640xf16, #NHWC>) outputs(%24 as %arg3: memref<1x32x30x640xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
        %38 = VPUIP.Copy inputs(%arg2 : memref<1x32x30x640xf16, #NHWC>) outputs(%arg3 : memref<1x32x30x640xf16, #NHWC, @CMX_NN>) -> memref<1x32x30x640xf16, #NHWC, @CMX_NN>
    }
    %26 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %27 = VPUIP.NCEClusterTiling inputs(%cst_0 as %arg2: memref<16x32x1x1xf16, #NHWC>) outputs(%26 as %arg3: memref<16x32x1x1xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
        %38 = VPUIP.Copy inputs(%arg2 : memref<16x32x1x1xf16, #NHWC>) outputs(%arg3 : memref<16x32x1x1xf16, #NHWC, @CMX_NN>) -> memref<16x32x1x1xf16, #NHWC, @CMX_NN>
    }
    %28 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %29 = VPUIP.NCEClusterTiling inputs(%cst_1 as %arg2: memref<16x1x1x4xsi32>) outputs(%28 as %arg3: memref<16x1x1x4xsi32, @CMX_NN>) -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
        %38 = VPUIP.Copy inputs(%arg2 : memref<16x1x1x4xsi32>) outputs(%arg3 : memref<16x1x1x4xsi32, @CMX_NN>) -> memref<16x1x1x4xsi32, @CMX_NN>
    }
    %30 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %31 = VPUIP.NCEClusterTiling inputs(%25 as %arg2: memref<1x32x30x640xf16, #NHWC, @CMX_NN>, %27 as %arg3: memref<16x32x1x1xf16, #NHWC, @CMX_NN>, %29 as %arg4: memref<16x1x1x4xsi32, @CMX_NN>) outputs(%30 as %arg5: memref<1x16x30x640xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
        %38 = VPUIP.NCEClusterTask {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 11628 : i64, task_type = "CONV"} input(%arg2 : memref<1x32x30x640xf16, #NHWC, @CMX_NN>) weights(%arg3 : memref<16x32x1x1xf16, #NHWC, @CMX_NN>) weight_table(%arg4 : memref<16x1x1x4xsi32, @CMX_NN>) parent_input(%arg2 : memref<1x32x30x640xf16, #NHWC, @CMX_NN>) parent_output(%arg5 : memref<1x16x30x640xf16, #NHWC, @CMX_NN>) outputs(%arg5 : memref<1x16x30x640xf16, #NHWC, @CMX_NN>) -> memref<1x16x30x640xf16, #NHWC, @CMX_NN> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_16x16", outEnd = [639, 14, 15], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
        DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_16x16", outEnd = [639, 29, 15], outStart = [0, 15, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
        } PPE : {
        PPETask "NOOP" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
        }
    }
    %32 = VPUIP.SubView %9 [0, 0, 60, 0] [1, 16, 30, 640] : memref<1x16x90x640xf16, #NHWC, @DDR> to memref<1x16x30x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR>
    %33 = VPUIP.NCEClusterTiling inputs(%31 as %arg2: memref<1x16x30x640xf16, #NHWC, @CMX_NN>) outputs(%32 as %arg3: memref<1x16x30x640xf16, #NHWC>) -> memref<1x16x30x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR> {
        %38 = VPUIP.Copy inputs(%arg2 : memref<1x16x30x640xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x16x30x640xf16, #NHWC>) -> memref<1x16x30x640xf16, #NHWC>
    }

    %34 = VPUIP.ConcatView inputs(%11, %22, %33 : memref<1x16x30x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR>, memref<1x16x30x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR>, memref<1x16x30x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR>) outputs(%9 : memref<1x16x90x640xf16, #NHWC, @DDR>) -> memref<1x16x90x640xf16, #NHWC, @DDR>
    %35 = VPUIP.SubView %34 [0, 0, 0, 0] [1, 1, 90, 640] : memref<1x16x90x640xf16, #NHWC, @DDR> to memref<1x1x90x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR>
    %37 = VPUIP.Copy inputs(%35 : memref<1x1x90x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR>) outputs(%arg1 : memref<1x1x90x640xf16, #NHWC, @DDR>) -> memref<1x1x90x640xf16, #NHWC, @DDR>

    return %37 : memref<1x1x90x640xf16, #NHWC, @DDR>

    // CHECK: [[FILTER:%.+]] = const.Declare memref<16x32x1x1xf16, #NHWC> = dense<1.000000e+00> : tensor<16x32x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK: [[TABLE:%.+]] = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // Tile idx 0:
    // CHECK: [[SUBVIEW_0:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 32, 30, 640] : memref<1x32x360x640xf16, #NHWC, @DDR> to memref<1x32x30x640xf16, {order = #NHWC, strides = [7372800, 1, 20480, 32]}, @DDR>
    // CHECK: [[ACTIVATION_BUF_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[ACTIVATION_COPY_IN_0:%.+]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW_0]] as %arg2: memref<1x32x30x640xf16, #NHWC>)
    // CHECK:                                                       outputs([[ACTIVATION_BUF_0]] as %arg3: memref<1x32x30x640xf16, #NHWC, @CMX_NN>)

    // CHECK: [[FILTER_BUF_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[FILTER_COPY_IN_0:%.+]] = VPUIP.NCEClusterTiling inputs([[FILTER]] as %arg2: memref<16x32x1x1xf16, #NHWC>)
    // CHECK:                                                   outputs([[FILTER_BUF_0]] as %arg3: memref<16x32x1x1xf16, #NHWC, @CMX_NN>)

    // CHECK: [[TABLE_BUF_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[TABLE_COPY_IN_0:%.+]] = VPUIP.NCEClusterTiling inputs([[TABLE]] as %arg2: memref<16x1x1x4xsi32>)
    // CHECK:                                                  outputs([[TABLE_BUF_0]] as %arg3: memref<16x1x1x4xsi32, @CMX_NN>)

    // CHECK: [[CONV_RESULT_BUF_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[CONV_0:%.+]] = VPUIP.NCEClusterTiling inputs([[ACTIVATION_COPY_IN_0]] as %arg2: memref<1x32x30x640xf16, #NHWC, @CMX_NN>,
    // CHECK:                                                [[FILTER_COPY_IN_0]] as %arg3: memref<16x32x1x1xf16, #NHWC, @CMX_NN>,
    // CHECK:                                                [[TABLE_COPY_IN_0]] as %arg4: memref<16x1x1x4xsi32, @CMX_NN>)
    // CHECK:                                         outputs([[CONV_RESULT_BUF_0]] as %arg5: memref<1x16x30x640xf16, #NHWC, @CMX_NN>)

    // Tile idx 1:
    // CHECK: [[SUBVIEW_1:%.+]] = VPUIP.SubView %arg0 [0, 0, 30, 0] [1, 32, 30, 640] : memref<1x32x360x640xf16, #NHWC, @DDR> to memref<1x32x30x640xf16, {order = #NHWC, strides = [7372800, 1, 20480, 32]}, @DDR>
    // CHECK: [[ACTIVATION_BUF_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[ACTIVATION_COPY_IN_1:%.+]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW_1]] as %arg2: memref<1x32x30x640xf16, #NHWC>)
    // CHECK:                                                       outputs([[ACTIVATION_BUF_1]] as %arg3: memref<1x32x30x640xf16, #NHWC, @CMX_NN>)

    // CHECK: [[FILTER_BUF_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[FILTER_COPY_IN_1:%.+]] = VPUIP.NCEClusterTiling inputs([[FILTER]] as %arg2: memref<16x32x1x1xf16, #NHWC>)
    // CHECK:                                                   outputs([[FILTER_BUF_1]] as %arg3: memref<16x32x1x1xf16, #NHWC, @CMX_NN>)

    // CHECK: [[TABLE_BUF_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[TABLE_COPY_IN_1:%.+]] = VPUIP.NCEClusterTiling inputs([[TABLE]] as %arg2: memref<16x1x1x4xsi32>)
    // CHECK:                                                  outputs([[TABLE_BUF_1]] as %arg3: memref<16x1x1x4xsi32, @CMX_NN>)

    // CHECK: [[CONV_RESULT_BUF_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[CONV_1:%.+]] = VPUIP.NCEClusterTiling inputs([[ACTIVATION_COPY_IN_1]] as %arg2: memref<1x32x30x640xf16, #NHWC, @CMX_NN>,
    // CHECK:                                                [[FILTER_COPY_IN_1]] as %arg3: memref<16x32x1x1xf16, #NHWC, @CMX_NN>,
    // CHECK:                                                [[TABLE_COPY_IN_1]] as %arg4: memref<16x1x1x4xsi32, @CMX_NN>)
    // CHECK:                                         outputs([[CONV_RESULT_BUF_1]] as %arg5: memref<1x16x30x640xf16, #NHWC, @CMX_NN>)

    // Tile idx 2:
    // CHECK: [[SUBVIEW_2:%.+]] = VPUIP.SubView %arg0 [0, 0, 60, 0] [1, 32, 30, 640] : memref<1x32x360x640xf16, #NHWC, @DDR> to memref<1x32x30x640xf16, {order = #NHWC, strides = [7372800, 1, 20480, 32]}, @DDR>
    // CHECK: [[ACTIVATION_BUF_2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[ACTIVATION_COPY_IN_2:%.+]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW_2]] as %arg2: memref<1x32x30x640xf16, #NHWC>)
    // CHECK:                                                       outputs([[ACTIVATION_BUF_2]] as %arg3: memref<1x32x30x640xf16, #NHWC, @CMX_NN>)

    // CHECK: [[FILTER_BUF_2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[FILTER_COPY_IN_2:%.+]] = VPUIP.NCEClusterTiling inputs([[FILTER]] as %arg2: memref<16x32x1x1xf16, #NHWC>)
    // CHECK:                                                   outputs([[FILTER_BUF_2]] as %arg3: memref<16x32x1x1xf16, #NHWC, @CMX_NN>)

    // CHECK: [[TABLE_BUF_2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[TABLE_COPY_IN_2:%.+]] = VPUIP.NCEClusterTiling inputs([[TABLE]] as %arg2: memref<16x1x1x4xsi32>)
    // CHECK:                                                  outputs([[TABLE_BUF_2]] as %arg3: memref<16x1x1x4xsi32, @CMX_NN>)

    // CHECK: [[CONV_RESULT_BUF_2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[CONV_2:%.+]] = VPUIP.NCEClusterTiling inputs([[ACTIVATION_COPY_IN_2]] as %arg2: memref<1x32x30x640xf16, #NHWC, @CMX_NN>,
    // CHECK:                                                [[FILTER_COPY_IN_2]] as %arg3: memref<16x32x1x1xf16, #NHWC, @CMX_NN>,
    // CHECK:                                                [[TABLE_COPY_IN_2]] as %arg4: memref<16x1x1x4xsi32, @CMX_NN>)
    // CHECK:                                         outputs([[CONV_RESULT_BUF_2]] as %arg5: memref<1x16x30x640xf16, #NHWC, @CMX_NN>)

    // Slice Conv result at channel and concat result
    // CHECK: [[OUTPUT:%.+]] = memref.alloc() : memref<1x1x90x640xf16, #NHWC, @DDR>
    // CHECK: [[CONV_0_SLICE_CHANNEL:%.+]] = VPUIP.SubView [[CONV_0]] [0, 0, 0, 0] [1, 1, 30, 640] : !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x30x640xf16, {order = #NHWC, strides = [307200, 1, 10240, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[OUTPUT_SUB_0:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 0, 0, 0] [1, 1, 30, 640] : memref<1x1x90x640xf16, #NHWC, @DDR> to memref<1x1x30x640xf16, {order = #NHWC, strides = [57600, 1, 640, 1]}, @DDR>
    // CHECK: [[OUTPUT_COPY_0:%.+]] = VPUIP.NCEClusterTiling inputs([[CONV_0_SLICE_CHANNEL]] as %arg2: memref<1x1x30x640xf16, {order = #NHWC, strides = [307200, 1, 10240, 16]}, @CMX_NN>)
    // CHECK:                                                outputs([[OUTPUT_SUB_0]] as %arg3: memref<1x1x30x640xf16, {order = #NHWC, strides = [57600, 1, 640, 1]}, @DDR>)

    // CHECK: [[CONV_1_SLICE_CHANNEL:%.+]] = VPUIP.SubView [[CONV_1]] [0, 0, 0, 0] [1, 1, 30, 640] : !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x30x640xf16, {order = #NHWC, strides = [307200, 1, 10240, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[OUTPUT_SUB_1:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 0, 30, 0] [1, 1, 30, 640] : memref<1x1x90x640xf16, #NHWC, @DDR> to memref<1x1x30x640xf16, {order = #NHWC, strides = [57600, 1, 640, 1]}, @DDR>
    // CHECK: [[OUTPUT_COPY_1:%.+]] = VPUIP.NCEClusterTiling inputs([[CONV_1_SLICE_CHANNEL]] as %arg2: memref<1x1x30x640xf16, {order = #NHWC, strides = [307200, 1, 10240, 16]}, @CMX_NN>)
    // CHECK:                                                outputs([[OUTPUT_SUB_1]] as %arg3: memref<1x1x30x640xf16, {order = #NHWC, strides = [57600, 1, 640, 1]}, @DDR>) -> memref<1x1x30x640xf16, {order = #NHWC, strides = [57600, 1, 640, 1]}, @DDR> {

    // CHECK: [[CONV_2_SLICE_CHANNEL:%.+]] = VPUIP.SubView [[CONV_2]] [0, 0, 0, 0] [1, 1, 30, 640] : !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x30x640xf16, {order = #NHWC, strides = [307200, 1, 10240, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[OUTPUT_SUB_2:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 0, 60, 0] [1, 1, 30, 640] : memref<1x1x90x640xf16, #NHWC, @DDR> to memref<1x1x30x640xf16, {order = #NHWC, strides = [57600, 1, 640, 1]}, @DDR>
    // CHECK: [[OUTPUT_COPY_2:%.+]] = VPUIP.NCEClusterTiling inputs([[CONV_2_SLICE_CHANNEL]] as %arg2: memref<1x1x30x640xf16, {order = #NHWC, strides = [307200, 1, 10240, 16]}, @CMX_NN>)
    // CHECK:                                                outputs([[OUTPUT_SUB_2]] as %arg3: memref<1x1x30x640xf16, {order = #NHWC, strides = [57600, 1, 640, 1]}, @DDR>) -> memref<1x1x30x640xf16, {order = #NHWC, strides = [57600, 1, 640, 1]}, @DDR> {

    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[OUTPUT_COPY_0]], [[OUTPUT_COPY_1]], [[OUTPUT_COPY_2]]
    // CHECK:                   memref<1x1x30x640xf16, {order = #NHWC, strides = [57600, 1, 640, 1]}, @DDR>,
    // CHECK:                   memref<1x1x30x640xf16, {order = #NHWC, strides = [57600, 1, 640, 1]}, @DDR>,
    // CHECK:                   memref<1x1x30x640xf16, {order = #NHWC, strides = [57600, 1, 640, 1]}, @DDR>)
    // CHECK:                   outputs([[OUTPUT]] : memref<1x1x90x640xf16, #NHWC, @DDR>) -> memref<1x1x90x640xf16, #NHWC, @DDR>

    // CHECK-NOT: VPUIP.SubView
    // CHECK: [[RESULT_COPY:%.+]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x1x90x640xf16, #NHWC, @DDR>) outputs(%arg1 : memref<1x1x90x640xf16, #NHWC, @DDR>) -> memref<1x1x90x640xf16, #NHWC, @DDR>
	// CHECK: return [[RESULT_COPY]] : memref<1x1x90x640xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x256x20x40xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    alignment = [1, 16, 1, 1]
}>

func.func @RemoveDDRToDDRCopyAfterConcatThroughPureView(
        %arg0: !InputDistributed,
        %arg1: !InputDistributed,
        %arg2: memref<1x256x40x40xf16, #NHWC, @DDR>)
         -> (memref<1x40x256x40xf16, #NCHW, @DDR>){
    %buffer = memref.alloc() : memref<1x256x40x40xf16, #NHWC, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    %nceTilingCopy0 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg3: memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%subview0 as %arg4: memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, {order = #NHWC}, @DDR> {
      %0 = VPUIP.Copy inputs(%arg3 : memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%arg4 : memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, #NHWC>
    }
    %subview1 = VPUIP.SubView %buffer [0, 0, 20, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    %nceTilingCopy1 = VPUIP.NCEClusterTiling inputs(%arg1 as %arg3: memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%subview1 as %arg4: memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, {order = #NHWC}, @DDR> {
      %0 = VPUIP.Copy inputs(%arg3 : memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%arg4 : memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, #NHWC>
    }
    %concat = VPUIP.ConcatView inputs(%nceTilingCopy0, %nceTilingCopy1 : memref<1x256x20x40xf16, {order = #NHWC}, @DDR>, memref<1x256x20x40xf16, {order = #NHWC}, @DDR>) outputs(%buffer : memref<1x256x40x40xf16, #NHWC, @DDR>) -> memref<1x256x40x40xf16, #NHWC, @DDR>
    %permuteCast = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW} inputs(%concat : memref<1x256x40x40xf16, #NHWC, @DDR>) -> memref<1x40x256x40xf16, #NCHW, @DDR>
    %buffer1 = memref.alloc() : memref<1x40x256x40xf16, #NCHW, @DDR>
    %copy0 = VPUIP.Copy inputs(%permuteCast : memref<1x40x256x40xf16, #NCHW, @DDR>) outputs(%buffer1 : memref<1x40x256x40xf16, #NCHW, @DDR>) -> memref<1x40x256x40xf16, #NCHW, @DDR>
    return %copy0 : memref<1x40x256x40xf16, #NCHW, @DDR>

    // CHECK: [[BUFFER0:%.+]] = memref.alloc() : memref<1x256x40x40xf16, #NHWC, @DDR>
    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView [[BUFFER0]]
    // CHECK-SAME:  [0, 0, 0, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    // CHECK: [[TILING_COPY0:%.+]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg3: memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs([[SUBVIEW0]] as %arg4: memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, {order = #NHWC}, @DDR> {
    // CHECK:  VPUIP.Copy inputs(%arg3 : memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%arg4 : memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, #NHWC>
    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView [[BUFFER0]]
    // CHECK-SAME:  [0, 0, 20, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    // CHECK: [[TILING_COPY1:%.+]] = VPUIP.NCEClusterTiling inputs(%arg1 as %arg3: memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs([[SUBVIEW1]] as %arg4: memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, {order = #NHWC}, @DDR> {
    // CHECK:  VPUIP.Copy inputs(%arg3 : memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%arg4 : memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, #NHWC>
    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[TILING_COPY0]], [[TILING_COPY1]] : memref<1x256x20x40xf16, {order = #NHWC}, @DDR>, memref<1x256x20x40xf16, {order = #NHWC}, @DDR>) outputs([[BUFFER0]] : memref<1x256x40x40xf16, #NHWC, @DDR>) -> memref<1x256x40x40xf16, #NHWC, @DDR>
    // CHECK: [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW} inputs([[CONCAT]] : memref<1x256x40x40xf16, #NHWC, @DDR>) -> memref<1x40x256x40xf16, @DDR>
    // CHECK-NOT: memref.alloc() : memref<1x256x40x40xf16, #NCHW, @DDR>
    // CHECK-NOT: VPUIP.Copy
    // CHECK: return [[PERMUTECAST]] : memref<1x40x256x40xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x256x20x40xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    alignment = [1, 16, 1, 1]
}>

func.func @RemoveDDRToDDRCopyAfterConcatView(
        %arg0: !InputDistributed,
        %arg1: !InputDistributed,
        %arg2: memref<1x256x40x40xf16, #NHWC, @DDR>)
         -> (memref<1x256x40x40xf16, #NHWC, @DDR>){
    %buffer = memref.alloc() : memref<1x256x40x40xf16, #NHWC, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    %nceTilingCopy0 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg3: memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%subview0 as %arg4: memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, {order = #NHWC}, @DDR> {
      %0 = VPUIP.Copy inputs(%arg3 : memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%arg4 : memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, #NHWC>
    }
    %subview1 = VPUIP.SubView %buffer [0, 0, 20, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    %nceTilingCopy1 = VPUIP.NCEClusterTiling inputs(%arg1 as %arg3: memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%subview1 as %arg4: memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, {order = #NHWC}, @DDR> {
      %0 = VPUIP.Copy inputs(%arg3 : memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%arg4 : memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, #NHWC>
    }
    %concat = VPUIP.ConcatView inputs(%nceTilingCopy0, %nceTilingCopy1 : memref<1x256x20x40xf16, {order = #NHWC}, @DDR>, memref<1x256x20x40xf16, {order = #NHWC}, @DDR>) outputs(%buffer : memref<1x256x40x40xf16, #NHWC, @DDR>) -> memref<1x256x40x40xf16, #NHWC, @DDR>
    %buffer1 = memref.alloc() : memref<1x256x40x40xf16, #NHWC, @DDR>
    %copy0 = VPUIP.Copy inputs(%concat : memref<1x256x40x40xf16,  #NHWC, @DDR>) outputs(%buffer1 : memref<1x256x40x40xf16, #NHWC, @DDR>) -> memref<1x256x40x40xf16, #NHWC, @DDR>
    return %copy0 : memref<1x256x40x40xf16, #NHWC, @DDR>

    // CHECK: [[BUFFER0:%.+]] = memref.alloc() : memref<1x256x40x40xf16, #NHWC, @DDR>
    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView [[BUFFER0]]
    // CHECK-SAME:  [0, 0, 0, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    // CHECK: [[TILING_COPY0:%.+]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg3: memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs([[SUBVIEW0]] as %arg4: memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, {order = #NHWC}, @DDR> {
    // CHECK:  VPUIP.Copy inputs(%arg3 : memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%arg4 : memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, #NHWC>
    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView [[BUFFER0]]
    // CHECK-SAME:  [0, 0, 20, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    // CHECK: [[TILING_COPY1:%.+]] = VPUIP.NCEClusterTiling inputs(%arg1 as %arg3: memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs([[SUBVIEW1]] as %arg4: memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, {order = #NHWC}, @DDR> {
    // CHECK:  VPUIP.Copy inputs(%arg3 : memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%arg4 : memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, #NHWC>
    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[TILING_COPY0]], [[TILING_COPY1]] : memref<1x256x20x40xf16, {order = #NHWC}, @DDR>, memref<1x256x20x40xf16, {order = #NHWC}, @DDR>) outputs([[BUFFER0]] : memref<1x256x40x40xf16, #NHWC, @DDR>) -> memref<1x256x40x40xf16, #NHWC, @DDR>
    // CHECK-NOT: memref.alloc() : memref<1x256x40x40xf16, #NHWC, @DDR>
    // CHECK-NOT: VPUIP.Copy
    // CHECK: return [[CONCAT]] : memref<1x256x40x40xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x256x20x40xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    alignment = [1, 16, 1, 1]
}>

func.func @RemoveDDRToDDRCopyAfterConcatThroughPureView(
        %arg0: !InputDistributed,
        %arg1: !InputDistributed,
        %arg2: memref<1x256x40x40xf16, #NHWC, @DDR>)
         -> (memref<1x40x256x40xf16, #NCHW, @DDR>){
    %buffer = memref.alloc() : memref<1x256x40x40xf16, #NHWC, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    %nceTilingCopy0 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg3: memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%subview0 as %arg4: memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, {order = #NHWC}, @DDR> {
      %0 = VPUIP.Copy inputs(%arg3 : memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%arg4 : memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, #NHWC>
    }
    %subview1 = VPUIP.SubView %buffer [0, 0, 20, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    %nceTilingCopy1 = VPUIP.NCEClusterTiling inputs(%arg1 as %arg3: memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%subview1 as %arg4: memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, {order = #NHWC}, @DDR> {
      %0 = VPUIP.Copy inputs(%arg3 : memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%arg4 : memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, #NHWC>
    }
    %concat = VPUIP.ConcatView inputs(%nceTilingCopy0, %nceTilingCopy1 : memref<1x256x20x40xf16, {order = #NHWC}, @DDR>, memref<1x256x20x40xf16, {order = #NHWC}, @DDR>) outputs(%buffer : memref<1x256x40x40xf16, #NHWC, @DDR>) -> memref<1x256x40x40xf16, #NHWC, @DDR>
    %permuteCast = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW} inputs(%concat : memref<1x256x40x40xf16, #NHWC, @DDR>) -> memref<1x40x256x40xf16, #NCHW, @DDR>
    %buffer1 = memref.alloc() : memref<1x40x256x40xf16, #NCHW, @DDR>
    %copy0 = VPUIP.Copy inputs(%permuteCast : memref<1x40x256x40xf16, #NCHW, @DDR>) outputs(%buffer1 : memref<1x40x256x40xf16, #NCHW, @DDR>) -> memref<1x40x256x40xf16, #NCHW, @DDR>
    return %copy0 : memref<1x40x256x40xf16, #NCHW, @DDR>

    // CHECK: [[BUFFER0:%.+]] = memref.alloc() : memref<1x256x40x40xf16, #NHWC, @DDR>
    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView [[BUFFER0]]
    // CHECK-SAME:  [0, 0, 0, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    // CHECK: [[TILING_COPY0:%.+]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg3: memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs([[SUBVIEW0]] as %arg4: memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, {order = #NHWC}, @DDR> {
    // CHECK:  VPUIP.Copy inputs(%arg3 : memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%arg4 : memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, #NHWC>
    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView [[BUFFER0]]
    // CHECK-SAME:  [0, 0, 20, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    // CHECK: [[TILING_COPY1:%.+]] = VPUIP.NCEClusterTiling inputs(%arg1 as %arg3: memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs([[SUBVIEW1]] as %arg4: memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, {order = #NHWC}, @DDR> {
    // CHECK:  VPUIP.Copy inputs(%arg3 : memref<1x256x20x40xf16, #NHWC, @CMX_NN>) outputs(%arg4 : memref<1x256x20x40xf16, #NHWC>) -> memref<1x256x20x40xf16, #NHWC>
    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[TILING_COPY0]], [[TILING_COPY1]] : memref<1x256x20x40xf16, {order = #NHWC}, @DDR>, memref<1x256x20x40xf16, {order = #NHWC}, @DDR>) outputs([[BUFFER0]] : memref<1x256x40x40xf16, #NHWC, @DDR>) -> memref<1x256x40x40xf16, #NHWC, @DDR>
    // CHECK: [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW} inputs([[CONCAT]] : memref<1x256x40x40xf16, #NHWC, @DDR>) -> memref<1x40x256x40xf16, @DDR>
    // CHECK-NOT: memref.alloc() : memref<1x256x40x40xf16, #NCHW, @DDR>
    // CHECK-NOT: VPUIP.Copy
    // CHECK: return [[PERMUTECAST]] : memref<1x40x256x40xf16, @DDR>
}
