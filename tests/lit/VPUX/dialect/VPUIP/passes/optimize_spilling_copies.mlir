//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-spilling-copies %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x16x57x512xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    alignment = [1, 16, 1, 1]
}>

func @OptimizeSpillingCopies(
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

!IODDRData0 = type memref<1x16x57x512xf16, {order = #NHWC}, @DDR>
!IODDRSM0 = type memref<1x16x57x512xi1, {order = #NHWC}, @DDR>
!IODDRSparse0 = type !VPUIP.SparseBuffer<
    data=!IODDRData0,
    sparsity_map=!IODDRSM0
>

!IODDRSparse1 = type !VPUIP.SparseBuffer<
    data=memref<1x3x110x512xf16, #NHWC, @DDR>,
    sparsity_map=memref<1x3x110x512xi1, #NHWC, @DDR>
>
!IODistrCMXSparse0 = type !VPUIP.SparseBuffer<

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

!IODDRData2 = type memref<1x16x57x512xf16, #NHWC>
!IODDRSM2 = type memref<1x16x57x512xi1, #NHWC>
!IODDRSparse2 = type !VPUIP.SparseBuffer<
    data=!IODDRData2,
    sparsity_map=!IODDRSM2
>

!IODDRSparse3 = type !VPUIP.SparseBuffer<
    data=memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>,
    sparsity_map=memref<1x16x57x512xi1, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
>
!IOCMXData0 = type memref<1x16x57x512xf16, #NHWC, @CMX_NN>
!IOCMXSM0 = type memref<1x16x57x512xi1, #NHWC, @CMX_NN>
!IOCMXSparse0 = type !VPUIP.SparseBuffer<
    data=!IOCMXData0,
    sparsity_map=!IOCMXSM0
>

!IODDRData4 = type memref<1x16x114x512xf16, #NHWC, @DDR>
!IODDRSM4 = type memref<1x16x114x512xi1, #NHWC, @DDR>
!IODDRSparse4 = type !VPUIP.SparseBuffer<
    data=!IODDRData4,
    sparsity_map=!IODDRSM4
>

!IODDRSparse5 = type !VPUIP.SparseBuffer<
    data=memref<1x3x4x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>,
    sparsity_map=memref<1x3x4x512xi1, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
>
!IODDRSparse6 = type !VPUIP.SparseBuffer<
    data=memref<1x3x110x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>,
    sparsity_map=memref<1x3x110x512xi1, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
>
!IODDRSparse7 = type !VPUIP.SparseBuffer<
    data=memref<1x3x4x512xf16, #NHWC, @DDR>,
    sparsity_map=memref<1x3x4x512xi1, #NHWC, @DDR>
>

// CHECK-LABEL: @OptimizeSpillingCopiesSparse
func @OptimizeSpillingCopiesSparse(%arg0: !IODistrCMXSparse0, %arg1: !IODistrCMXSparse0, %arg2: !IODDRSparse1, %arg3: !IODDRSparse7) -> (!IODDRSparse1, !IODDRSparse7) {
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
