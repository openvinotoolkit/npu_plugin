//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-copies %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IHalfDataCMXType = memref<1x16x112x112xf16, #NHWC, @CMX_NN>
!IHalfSMCMXType = memref<1x16x112x112xi1, #NHWC, @CMX_NN>
!IHalfSparseCMXType = !VPUIP.SparseBuffer<data=!IHalfDataCMXType, sparsity_map=!IHalfSMCMXType>

!IHalfDataDDRType = memref<1x16x112x112xf16, #NHWC, @DDR>
!IHalfSMDDRType = memref<1x16x112x112xi1, #NHWC, @DDR>
!IHalfSparseDDRType = !VPUIP.SparseBuffer<data=!IHalfDataDDRType, sparsity_map=!IHalfSMDDRType>

!ISubviewedSparseDDRType = !VPUIP.SparseBuffer<
    data=memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>,
    sparsity_map=memref<1x16x112x112xi1, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>
>

!ODataDDRType = memref<1x32x112x112xf16, #NHWC, @DDR>
!OSMDDRType = memref<1x32x112x112xi1, #NHWC, @DDR>
!OSparseDDRType = !VPUIP.SparseBuffer<data=!ODataDDRType, sparsity_map=!OSMDDRType>

// CHECK-LABEL: @OptimizeCopySparse
func.func @OptimizeCopySparse(
        %arg0: !IHalfSparseCMXType,
        %arg1: !IHalfSparseCMXType,
        %arg2: !OSparseDDRType)
        -> !OSparseDDRType {
    %0 = memref.alloc() : !ODataDDRType
    %1 = memref.alloc() : !OSMDDRType
    %2 = VPUIP.GroupSparseBuffer(%0, %1) -> !OSparseDDRType

    %3 = memref.alloc() : !IHalfDataDDRType
    %4 = memref.alloc() : !IHalfSMDDRType
    %5 = VPUIP.GroupSparseBuffer(%3, %4) -> !IHalfSparseDDRType

    %6 = VPUIP.Copy inputs(%arg0 : !IHalfSparseCMXType)
        outputs(%5 : !IHalfSparseDDRType)
        -> !IHalfSparseDDRType

    %7 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 16, 112, 112] :
        !OSparseDDRType to !ISubviewedSparseDDRType
    %8 = VPUIP.Copy inputs(%6 : !IHalfSparseDDRType)
        outputs(%7 : !ISubviewedSparseDDRType)
        -> !ISubviewedSparseDDRType

    %9 = memref.alloc() : !IHalfDataDDRType
    %10 = memref.alloc() : !IHalfSMDDRType
    %11 = VPUIP.GroupSparseBuffer(%9, %10) -> !IHalfSparseDDRType

    %12 = VPUIP.Copy inputs(%arg1 : !IHalfSparseCMXType)
        outputs(%11 : !IHalfSparseDDRType)
        -> !IHalfSparseDDRType

    %13 = VPUIP.SubView %2 [0, 16, 0, 0] [1, 16, 112, 112] :
        !OSparseDDRType to !ISubviewedSparseDDRType
    %14 = VPUIP.Copy inputs(%12 : !IHalfSparseDDRType)
        outputs(%13 : !ISubviewedSparseDDRType)
        -> !ISubviewedSparseDDRType

    %15 = VPUIP.ConcatView
        inputs(%7, %14 :
            !ISubviewedSparseDDRType,
            !ISubviewedSparseDDRType
        )
        outputs(%2 : !OSparseDDRType)
        -> !OSparseDDRType

    %16 = VPUIP.Copy inputs(%15 : !OSparseDDRType)
        outputs(%arg2 : !OSparseDDRType)
        -> !OSparseDDRType

    return %16 : !OSparseDDRType

    // CHECK:       [[BUFF_0_DATA:%.+]] = memref.alloc() : memref<1x32x112x112xf16, #NHWC, @DDR>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x32x112x112xi1, #NHWC, @DDR>
    // CHECK-NOT:   VPUIP.GroupSparseBuffer

    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView %arg2 [0, 0, 0, 0] [1, 16, 112, 112]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x32x112x112xf16, #NHWC, @DDR>, sparsity_map=memref<1x32x112x112xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>, sparsity_map=memref<1x16x112x112xi1, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>>
    // CHECK:       [[COPY_0:%.+]] = VPUIP.Copy inputs(%arg0 : !VPUIP.SparseBuffer<data=memref<1x16x112x112xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x16x112x112xi1, #NHWC, @CMX_NN>>) 
    // CHECK-SAME:         outputs([[SUBVIEW_0]] : !VPUIP.SparseBuffer<data=memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>, sparsity_map=memref<1x16x112x112xi1, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>, sparsity_map=memref<1x16x112x112xi1, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>>
    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView %arg2 [0, 16, 0, 0] [1, 16, 112, 112]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x32x112x112xf16, #NHWC, @DDR>, sparsity_map=memref<1x32x112x112xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>, sparsity_map=memref<1x16x112x112xi1, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>>
    // CHECK:       [[COPY_1:%.+]] = VPUIP.Copy inputs(%arg1 : !VPUIP.SparseBuffer<data=memref<1x16x112x112xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x16x112x112xi1, #NHWC, @CMX_NN>>) 
    // CHECK-SAME:         outputs([[SUBVIEW_1]] : !VPUIP.SparseBuffer<data=memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>, sparsity_map=memref<1x16x112x112xi1, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>, sparsity_map=memref<1x16x112x112xi1, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>>
    // CHECK:       [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[SUBVIEW_0]], [[COPY_1]] : !VPUIP.SparseBuffer<data=memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>, sparsity_map=memref<1x16x112x112xi1, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>>, !VPUIP.SparseBuffer<data=memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>, sparsity_map=memref<1x16x112x112xi1, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>>) 
    // CHECK-SAME:         outputs(%arg2 : !VPUIP.SparseBuffer<data=memref<1x32x112x112xf16, #NHWC, @DDR>, sparsity_map=memref<1x32x112x112xi1, #NHWC, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x32x112x112xf16, #NHWC, @DDR>, sparsity_map=memref<1x32x112x112xi1, #NHWC, @DDR>>
    // CHECK:       return [[CONCATVIEW_0]]
}

// -----

!IODDRSparse0 = !VPUIP.SparseBuffer<data=memref<1x16x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>, sparsity_map=memref<1x16x2x2xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>>
!IODDRData1 = memref<1x64x2x2xf16>
!IODDRSM1 = memref<1x64x2x2xi1>
!IODDRSparse1 = !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>

!IODDRSparse2 = !VPUIP.SparseBuffer<
    data=memref<1x32x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>, 
    sparsity_map=memref<1x32x2x2xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>
>
!IODDRSparse3 = !VPUIP.SparseBuffer<data=memref<1x16x2x2xf16>, sparsity_map=memref<1x16x2x2xi1>>

// CHECK-LABEL: @OptimizeLastCopyForPureViewOpsSparse
func.func @OptimizeLastCopyForPureViewOpsSparse(%arg0: !IODDRSparse3, %arg1: !IODDRSparse3, %arg2: !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) -> !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1> {
    %0 = memref.alloc() : !IODDRData1
    %1 = memref.alloc() : !IODDRSM1
    %2 = VPUIP.GroupSparseBuffer(%0, %1) -> !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>

    %3 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 16, 2, 2] : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1> to !IODDRSparse0
    %4 = VPUIP.Copy inputs(%arg0 : !IODDRSparse3) outputs(%3 : !IODDRSparse0) -> !IODDRSparse0
    %5 = VPUIP.SubView %2 [0, 16, 0, 0] [1, 16, 2, 2] : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1> to !IODDRSparse0
    %6 = VPUIP.Copy inputs(%arg1 : !IODDRSparse3) outputs(%5 : !IODDRSparse0) -> !IODDRSparse0
    %7 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 32, 2, 2] : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1> to !IODDRSparse2
    %8 = VPUIP.ConcatView inputs(%4, %6 : !IODDRSparse0, !IODDRSparse0) outputs(%7 : !IODDRSparse2) -> !IODDRSparse2
    %9 = VPUIP.SubView %2 [0, 32, 0, 0] [1, 16, 2, 2] : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1> to !IODDRSparse0
    %10 = VPUIP.Copy inputs(%arg0 : !IODDRSparse3) outputs(%9 : !IODDRSparse0) -> !IODDRSparse0
    %11 = VPUIP.SubView %2 [0, 48, 0, 0] [1, 16, 2, 2] : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1> to !IODDRSparse0
    %12 = VPUIP.Copy inputs(%arg1 : !IODDRSparse3) outputs(%11 : !IODDRSparse0) -> !IODDRSparse0
    %13 = VPUIP.SubView %2 [0, 32, 0, 0] [1, 32, 2, 2] : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1> to !IODDRSparse2
    %14 = VPUIP.ConcatView inputs(%10, %12 : !IODDRSparse0, !IODDRSparse0) outputs(%13 : !IODDRSparse2) -> !IODDRSparse2
    %15 = VPUIP.ConcatView inputs(%8, %14 : !IODDRSparse2, !IODDRSparse2) outputs(%2 : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) -> !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>
    %16 = VPUIP.Copy inputs(%15 : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) outputs(%arg2 : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) -> !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>
    return %16 : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>

    // CHECK-NOT: VPUIP.GroupSparseBuffer

    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView %arg2 [0, 0, 0, 0] [1, 16, 2, 2]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x64x2x2xf16>, sparsity_map=memref<1x64x2x2xi1>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x16x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>, sparsity_map=memref<1x16x2x2xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>>
    // CHECK:       [[COPY_0:%.+]] = VPUIP.Copy inputs(%arg0 : !VPUIP.SparseBuffer<data=memref<1x16x2x2xf16>, sparsity_map=memref<1x16x2x2xi1>>) 
    // CHECK-SAME:         outputs([[SUBVIEW_0]] : !VPUIP.SparseBuffer<data=memref<1x16x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>, sparsity_map=memref<1x16x2x2xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x16x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>, sparsity_map=memref<1x16x2x2xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>>
    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView %arg2 [0, 16, 0, 0] [1, 16, 2, 2]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x64x2x2xf16>, sparsity_map=memref<1x64x2x2xi1>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x16x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>, sparsity_map=memref<1x16x2x2xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>>
    // CHECK:       [[COPY_1:%.+]] = VPUIP.Copy inputs(%arg1 : !VPUIP.SparseBuffer<data=memref<1x16x2x2xf16>, sparsity_map=memref<1x16x2x2xi1>>) 
    // CHECK-SAME:         outputs([[SUBVIEW_1]] : !VPUIP.SparseBuffer<data=memref<1x16x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>, sparsity_map=memref<1x16x2x2xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x16x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>, sparsity_map=memref<1x16x2x2xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>>
    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView %arg2 [0, 0, 0, 0] [1, 32, 2, 2]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x64x2x2xf16>, sparsity_map=memref<1x64x2x2xi1>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x32x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>, sparsity_map=memref<1x32x2x2xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>>
    // CHECK:       [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]] : !VPUIP.SparseBuffer<data=memref<1x16x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>, sparsity_map=memref<1x16x2x2xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>>, !VPUIP.SparseBuffer<data=memref<1x16x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>, sparsity_map=memref<1x16x2x2xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>>) 
    // CHECK-SAME:         outputs([[SUBVIEW_2]] : !VPUIP.SparseBuffer<data=memref<1x32x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>, sparsity_map=memref<1x32x2x2xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x32x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>, sparsity_map=memref<1x32x2x2xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>>
    // CHECK:       [[SUBVIEW_3:%.+]] = VPUIP.SubView %arg2 [0, 32, 0, 0] [1, 16, 2, 2]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x64x2x2xf16>, sparsity_map=memref<1x64x2x2xi1>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x16x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>, sparsity_map=memref<1x16x2x2xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>>
    // CHECK:       [[COPY_2:%.+]] = VPUIP.Copy inputs(%arg0 : !VPUIP.SparseBuffer<data=memref<1x16x2x2xf16>, sparsity_map=memref<1x16x2x2xi1>>) 
    // CHECK-SAME:         outputs([[SUBVIEW_3]] : !VPUIP.SparseBuffer<data=memref<1x16x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>, sparsity_map=memref<1x16x2x2xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x16x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>, sparsity_map=memref<1x16x2x2xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>>
    // CHECK:       [[SUBVIEW_4:%.+]] = VPUIP.SubView %arg2 [0, 48, 0, 0] [1, 16, 2, 2]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x64x2x2xf16>, sparsity_map=memref<1x64x2x2xi1>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x16x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>, sparsity_map=memref<1x16x2x2xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>>
    // CHECK:       [[COPY_3:%.+]] = VPUIP.Copy inputs(%arg1 : !VPUIP.SparseBuffer<data=memref<1x16x2x2xf16>, sparsity_map=memref<1x16x2x2xi1>>) 
    // CHECK-SAME:         outputs([[SUBVIEW_4]] : !VPUIP.SparseBuffer<data=memref<1x16x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>, sparsity_map=memref<1x16x2x2xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x16x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>, sparsity_map=memref<1x16x2x2xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>>
    // CHECK:       [[SUBVIEW_5:%.+]] = VPUIP.SubView %arg2 [0, 32, 0, 0] [1, 32, 2, 2]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x64x2x2xf16>, sparsity_map=memref<1x64x2x2xi1>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x32x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>, sparsity_map=memref<1x32x2x2xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>>
    // CHECK:       [[CONCATVIEW_1:%.+]] = VPUIP.ConcatView inputs([[COPY_2]], [[COPY_3]] : !VPUIP.SparseBuffer<data=memref<1x16x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>, sparsity_map=memref<1x16x2x2xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>>, !VPUIP.SparseBuffer<data=memref<1x16x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>, sparsity_map=memref<1x16x2x2xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>>) 
    // CHECK-SAME:         outputs([[SUBVIEW_5]] : !VPUIP.SparseBuffer<data=memref<1x32x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>, sparsity_map=memref<1x32x2x2xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x32x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>, sparsity_map=memref<1x32x2x2xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>>
    // CHECK:       [[CONCATVIEW_2:%.+]] = VPUIP.ConcatView inputs([[CONCATVIEW_0]], [[CONCATVIEW_1]] : !VPUIP.SparseBuffer<data=memref<1x32x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>, sparsity_map=memref<1x32x2x2xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>>, !VPUIP.SparseBuffer<data=memref<1x32x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>, sparsity_map=memref<1x32x2x2xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>>) 
    // CHECK-SAME:         outputs(%arg2 : !VPUIP.SparseBuffer<data=memref<1x64x2x2xf16>, sparsity_map=memref<1x64x2x2xi1>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x64x2x2xf16>, sparsity_map=memref<1x64x2x2xi1>>
    
    // CHECK-NOT: VPUIP.Copy

    // CHECK:       return [[CONCATVIEW_2]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ISparseDDRType = !VPUIP.SparseBuffer<
    data=memref<1x16x4x4xf16, #NHWC>, 
    sparsity_map=memref<1x16x4x4xi1, #NHWC>
>

!IDataCMXType = memref<1x16x4x4xf16, #NHWC, @CMX_NN>
!ISMCMXType = memref<1x16x4x4xi1, #NHWC, @CMX_NN>
!ISparseCMXType = !VPUIP.SparseBuffer<
    data=!IDataCMXType,
    sparsity_map=!ISMCMXType
>

// CHECK-LABEL: @NoChangesDifferentMemSpaceSparse
func.func @NoChangesDifferentMemSpaceSparse(%arg0: !ISparseDDRType, %arg1 : memref<16x1x1x4xsi32, @CMX_NN>,
                                 %arg2 : memref<16x1x1x16xui8, @CMX_NN>, %arg3: !ISparseDDRType) -> !ISparseDDRType {
    %data_buff = memref.alloc() : !IDataCMXType
    %sm_buff = memref.alloc() : !ISMCMXType
    %0 = VPUIP.GroupSparseBuffer(%data_buff, %sm_buff) -> !ISparseCMXType
    %1 = VPUIP.Copy inputs(%arg0 : !ISparseDDRType) outputs(%0 : !ISparseCMXType) -> !ISparseCMXType

    %data_buff_0 = memref.alloc() : !IDataCMXType
    %sm_buff_0 = memref.alloc() : !ISMCMXType
    %2 = VPUIP.GroupSparseBuffer(%data_buff, %sm_buff) -> !ISparseCMXType

    %in_data_0, %in_sm_0 = VPUIP.UngroupSparseBuffer(%1) {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>}
        -> !IDataCMXType, !ISMCMXType
    %out_data_0, %out_sm_0 = VPUIP.UngroupSparseBuffer(%2) {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>}
        -> !IDataCMXType, !ISMCMXType


    %mp:2 = VPUIP.NCEClusterTask {
            activation_window_channel_length = 27 : i64,
            kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>,
            kernel_size = [2, 2],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }
        input(%in_data_0 : !IDataCMXType)
        input_sparsity_map(%in_sm_0 : !ISMCMXType)
        weight_table(%arg1 : memref<16x1x1x4xsi32, @CMX_NN>)
        activation_window(%arg2 : memref<16x1x1x16xui8, @CMX_NN>)
        parent_input(%in_data_0 : !IDataCMXType)
        parent_input_sparsity_map(%in_sm_0 : !ISMCMXType)
        parent_output(%out_data_0 : !IDataCMXType)
        parent_output_sparsity_map(%in_sm_0 : !ISMCMXType)
        outputs(%out_data_0 : !IDataCMXType)
        output_sparsity_map(%in_sm_0 : !ISMCMXType) -> !IDataCMXType, !ISMCMXType
        variants :
        {
            DPUTask { outEnd = [16, 2, 2], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
        }
        PPE : {
        }
    %3 = VPUIP.GroupSparseBuffer(%mp#0, %mp#1) -> !ISparseCMXType

    %4 = VPUIP.Copy inputs(%3 : !ISparseCMXType) outputs(%arg3 : !ISparseDDRType) -> !ISparseDDRType
    return %4 : !ISparseDDRType

    // CHECK:       [[BUFF_0_DATA:%.+]] = memref.alloc() : memref<1x16x4x4xf16, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x16x4x4xi1, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_0:%.+]] = VPUIP.GroupSparseBuffer([[BUFF_0_DATA]], [[BUFF_0_SM]])
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x16x4x4xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x16x4x4xi1, #NHWC, @CMX_NN>>
      
    // CHECK:       [[COPY_0:%.+]] = VPUIP.Copy inputs(%arg0 : !VPUIP.SparseBuffer<data=memref<1x16x4x4xf16, #NHWC>, sparsity_map=memref<1x16x4x4xi1, #NHWC>>) 
    // CHECK-SAME:         outputs([[BUFF_0]] : !VPUIP.SparseBuffer<data=memref<1x16x4x4xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x16x4x4xi1, #NHWC, @CMX_NN>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x16x4x4xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x16x4x4xi1, #NHWC, @CMX_NN>>
    // CHECK:       [[BUFF_1:%.+]] = VPUIP.GroupSparseBuffer([[BUFF_0_DATA]], [[BUFF_0_SM]])
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x16x4x4xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x16x4x4xi1, #NHWC, @CMX_NN>>
      
    // CHECK:       [[DATA_0:%.+]], [[SM_0:%.+]] = VPUIP.UngroupSparseBuffer([[COPY_0]]) {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} -> memref<1x16x4x4xf16, #NHWC, @CMX_NN>, memref<1x16x4x4xi1, #NHWC, @CMX_NN>
    // CHECK:       [[DATA_1:%.+]], [[SM_1:%.+]] = VPUIP.UngroupSparseBuffer([[BUFF_1]]) {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} -> memref<1x16x4x4xf16, #NHWC, @CMX_NN>, memref<1x16x4x4xi1, #NHWC, @CMX_NN>
    // CHECK:       [[NCE_0:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:          input([[DATA_0]] : memref<1x16x4x4xf16, #NHWC, @CMX_NN>) 
    // CHECK-SAME:          weight_table(%arg1 : memref<16x1x1x4xsi32, @CMX_NN>) 
    // CHECK-SAME:          parent_input([[DATA_0]] : memref<1x16x4x4xf16, #NHWC, @CMX_NN>) 
    // CHECK-SAME:          parent_output([[DATA_0]]_0 : memref<1x16x4x4xf16, #NHWC, @CMX_NN>) 
    // CHECK-SAME:          outputs([[DATA_0]]_0 : memref<1x16x4x4xf16, #NHWC, @CMX_NN>) 
    // CHECK:       [[BUFF_2:%.+]] = VPUIP.GroupSparseBuffer([[NCE_0]]#0, [[NCE_0]]#1)
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x16x4x4xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x16x4x4xi1, #NHWC, @CMX_NN>>
      
    // CHECK:       [[COPY_1:%.+]] = VPUIP.Copy inputs([[BUFF_2]] : !VPUIP.SparseBuffer<data=memref<1x16x4x4xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x16x4x4xi1, #NHWC, @CMX_NN>>) 
    // CHECK-SAME:         outputs(%arg3 : !VPUIP.SparseBuffer<data=memref<1x16x4x4xf16, #NHWC>, sparsity_map=memref<1x16x4x4xi1, #NHWC>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x16x4x4xf16, #NHWC>, sparsity_map=memref<1x16x4x4xi1, #NHWC>>
    // CHECK:       return [[COPY_1]] : !VPUIP.SparseBuffer<data=memref<1x16x4x4xf16, #NHWC>, sparsity_map=memref<1x16x4x4xi1, #NHWC>>
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IODDRSparse0 = !VPUIP.SparseBuffer<data=memref<1x70x28x27xf16, {order = #NHWC, strides = [62720, 1, 2240, 80]}, @DDR>, sparsity_map=memref<1x70x28x27xi1, {order = #NHWC, strides = [62720, 1, 2240, 80]}, @DDR>>
!IODDRSparse1 = !VPUIP.SparseBuffer<data=memref<1x70x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>, sparsity_map=memref<1x70x28x27xi1, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>>
!IODDRData2 = memref<1x80x28x27xf16, #NHWC, @DDR>
!IODDRSM2 = memref<1x80x28x27xi1, #NHWC, @DDR>
!IODDRSparse2 = !VPUIP.SparseBuffer<data=!IODDRData2, sparsity_map=!IODDRSM2>

!IODDRData3 = memref<1x70x28x27xf16, #NHWC, @DDR>
!IODDRSM3 = memref<1x70x28x27xi1, {order = #NHWC, strides = [62720, 1, 2160, 80]}, @DDR>
!IODDRSparse3 = !VPUIP.SparseBuffer<data=!IODDRData3, sparsity_map=!IODDRSM3>

!IODDRSparse4 = !VPUIP.SparseBuffer<data=memref<1x10x28x27xf16, {order = #NHWC, strides = [52920, 1, 1890, 70]}, @DDR>, sparsity_map=memref<1x10x28x27xi1, {order = #NHWC, strides = [62720, 1, 2160, 80]}, @DDR>>
!IODDRSparse5 = !VPUIP.SparseBuffer<data=memref<1x80x28x28xf16, #NHWC, @DDR>, sparsity_map=memref<1x80x28x28xi1, #NHWC, @DDR>>
!IODDRSparse6 = !VPUIP.SparseBuffer<data=memref<1x10x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>, sparsity_map=memref<1x10x28x27xi1, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>>

// CHECK-LABEL: @CopiesWithSubViewOpsSparse
func.func @CopiesWithSubViewOpsSparse(%arg0: !IODDRSparse5) -> !VPUIP.SparseBuffer<data=!IODDRData2, sparsity_map=!IODDRSM2> {
    %0 = memref.alloc() : !IODDRData3
    %1 = memref.alloc() : !IODDRSM3
    %2 = VPUIP.GroupSparseBuffer(%0, %1) -> !VPUIP.SparseBuffer<data=!IODDRData3, sparsity_map=!IODDRSM3>

    %3 = memref.alloc() : !IODDRData2
    %4 = memref.alloc() : !IODDRSM2
    %5 = VPUIP.GroupSparseBuffer(%3, %4) -> !VPUIP.SparseBuffer<data=!IODDRData2, sparsity_map=!IODDRSM2>

    %6 = VPUIP.SubView %arg0 [0, 0, 0, 1] [1, 70, 28, 27] : !IODDRSparse5 to !IODDRSparse0
    %7 = VPUIP.Copy inputs(%6 : !IODDRSparse0) outputs(%2 : !VPUIP.SparseBuffer<data=!IODDRData3, sparsity_map=!IODDRSM3>) -> !VPUIP.SparseBuffer<data=!IODDRData3, sparsity_map=!IODDRSM3>
    %8 = VPUIP.SubView %5 [0, 0, 0, 0] [1, 70, 28, 27] : !VPUIP.SparseBuffer<data=!IODDRData2, sparsity_map=!IODDRSM2> to !IODDRSparse1
    %9 = VPUIP.Copy inputs(%7 : !VPUIP.SparseBuffer<data=!IODDRData3, sparsity_map=!IODDRSM3>) outputs(%8 : !IODDRSparse1) -> !IODDRSparse1
    %10 = VPUIP.SubView %7 [0, 0, 0, 0] [1, 10, 28, 27] : !VPUIP.SparseBuffer<data=!IODDRData3, sparsity_map=!IODDRSM3> to !IODDRSparse4
    %11 = VPUIP.SubView %5 [0, 70, 0, 0] [1, 10, 28, 27] : !VPUIP.SparseBuffer<data=!IODDRData2, sparsity_map=!IODDRSM2> to !IODDRSparse6
    %12 = VPUIP.Copy inputs(%10 : !IODDRSparse4) outputs(%11 : !IODDRSparse6) -> !IODDRSparse6
    %13 = VPUIP.ConcatView inputs(%9, %12 : !IODDRSparse1, !IODDRSparse6) outputs(%5 : !VPUIP.SparseBuffer<data=!IODDRData2, sparsity_map=!IODDRSM2>) -> !VPUIP.SparseBuffer<data=!IODDRData2, sparsity_map=!IODDRSM2>
    return %13 : !VPUIP.SparseBuffer<data=!IODDRData2, sparsity_map=!IODDRSM2>

    // CHECK:       [[BUFF_0_DATA:%.+]] = memref.alloc() : memref<1x70x28x27xf16, #NHWC, @DDR>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x70x28x27xi1, {order = #NHWC, strides = [62720, 1, 2160, 80]}, @DDR>
    // CHECK:       [[BUFF_0:%.+]] = VPUIP.GroupSparseBuffer([[BUFF_0_DATA]], [[BUFF_0_SM]])
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x70x28x27xf16, #NHWC, @DDR>, sparsity_map=memref<1x70x28x27xi1, {order = #NHWC, strides = [62720, 1, 2160, 80]}, @DDR>>
      
    // CHECK:       [[BUFF_1_DATA:%.+]] = memref.alloc() : memref<1x80x28x27xf16, #NHWC, @DDR>
    // CHECK:       [[BUFF_1_SM:%.+]] = memref.alloc() : memref<1x80x28x27xi1, #NHWC, @DDR>
    // CHECK:       [[BUFF_1:%.+]] = VPUIP.GroupSparseBuffer([[BUFF_1_DATA]], [[BUFF_1_SM]])
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x80x28x27xf16, #NHWC, @DDR>, sparsity_map=memref<1x80x28x27xi1, #NHWC, @DDR>>
      
    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 1] [1, 70, 28, 27]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x80x28x28xf16, #NHWC, @DDR>, sparsity_map=memref<1x80x28x28xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x70x28x27xf16, {order = #NHWC, strides = [62720, 1, 2240, 80]}, @DDR>, sparsity_map=memref<1x70x28x27xi1, {order = #NHWC, strides = [62720, 1, 2240, 80]}, @DDR>>
    // CHECK:       [[COPY_0:%.+]] = VPUIP.Copy inputs([[SUBVIEW_0]] : !VPUIP.SparseBuffer<data=memref<1x70x28x27xf16, {order = #NHWC, strides = [62720, 1, 2240, 80]}, @DDR>, sparsity_map=memref<1x70x28x27xi1, {order = #NHWC, strides = [62720, 1, 2240, 80]}, @DDR>>) 
    // CHECK-SAME:         outputs([[BUFF_0]] : !VPUIP.SparseBuffer<data=memref<1x70x28x27xf16, #NHWC, @DDR>, sparsity_map=memref<1x70x28x27xi1, {order = #NHWC, strides = [62720, 1, 2160, 80]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x70x28x27xf16, #NHWC, @DDR>, sparsity_map=memref<1x70x28x27xi1, {order = #NHWC, strides = [62720, 1, 2160, 80]}, @DDR>>
    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[BUFF_1]] [0, 0, 0, 0] [1, 70, 28, 27]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x80x28x27xf16, #NHWC, @DDR>, sparsity_map=memref<1x80x28x27xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x70x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>, sparsity_map=memref<1x70x28x27xi1, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>>
    // CHECK:       [[COPY_1:%.+]] = VPUIP.Copy inputs([[COPY_0]] : !VPUIP.SparseBuffer<data=memref<1x70x28x27xf16, #NHWC, @DDR>, sparsity_map=memref<1x70x28x27xi1, {order = #NHWC, strides = [62720, 1, 2160, 80]}, @DDR>>) 
    // CHECK-SAME:         outputs([[SUBVIEW_1]] : !VPUIP.SparseBuffer<data=memref<1x70x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>, sparsity_map=memref<1x70x28x27xi1, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x70x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>, sparsity_map=memref<1x70x28x27xi1, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>>
    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView [[COPY_0]] [0, 0, 0, 0] [1, 10, 28, 27]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x70x28x27xf16, #NHWC, @DDR>, sparsity_map=memref<1x70x28x27xi1, {order = #NHWC, strides = [62720, 1, 2160, 80]}, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x10x28x27xf16, {order = #NHWC, strides = [52920, 1, 1890, 70]}, @DDR>, sparsity_map=memref<1x10x28x27xi1, {order = #NHWC, strides = [62720, 1, 2160, 80]}, @DDR>>
    // CHECK:       [[SUBVIEW_3:%.+]] = VPUIP.SubView [[BUFF_1]] [0, 70, 0, 0] [1, 10, 28, 27]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x80x28x27xf16, #NHWC, @DDR>, sparsity_map=memref<1x80x28x27xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x10x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>, sparsity_map=memref<1x10x28x27xi1, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>>
    // CHECK:       [[COPY_2:%.+]] = VPUIP.Copy inputs([[SUBVIEW_2]] : !VPUIP.SparseBuffer<data=memref<1x10x28x27xf16, {order = #NHWC, strides = [52920, 1, 1890, 70]}, @DDR>, sparsity_map=memref<1x10x28x27xi1, {order = #NHWC, strides = [62720, 1, 2160, 80]}, @DDR>>) 
    // CHECK-SAME:         outputs([[SUBVIEW_3]] : !VPUIP.SparseBuffer<data=memref<1x10x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>, sparsity_map=memref<1x10x28x27xi1, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x10x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>, sparsity_map=memref<1x10x28x27xi1, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>>
    // CHECK:       [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[COPY_1]], [[COPY_2]] : !VPUIP.SparseBuffer<data=memref<1x70x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>, sparsity_map=memref<1x70x28x27xi1, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>>, !VPUIP.SparseBuffer<data=memref<1x10x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>, sparsity_map=memref<1x10x28x27xi1, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>>) 
    // CHECK-SAME:         outputs([[BUFF_1]] : !VPUIP.SparseBuffer<data=memref<1x80x28x27xf16, #NHWC, @DDR>, sparsity_map=memref<1x80x28x27xi1, #NHWC, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x80x28x27xf16, #NHWC, @DDR>, sparsity_map=memref<1x80x28x27xi1, #NHWC, @DDR>>
    // CHECK:       return [[CONCATVIEW_0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IODDRSM0 = memref<1x8x2x2xi1, @DDR>
!IODDRSparse0 = !VPUIP.SparseBuffer<data=memref<1x8x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [64, 4, 2, 1]}, @DDR>, sparsity_map=!IODDRSM0>

!IODDRData1 = memref<1x8x2x2xf16, @DDR>
!IODDRSM1 = memref<1x8x2x2xi1, @DDR>
!IODDRSparse1 = !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>

// CHECK-LABEL: @FuseLastCopiesChainWithNCEClusterTilingSparse
func.func @FuseLastCopiesChainWithNCEClusterTilingSparse(%arg0: !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>, %arg1: !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) -> !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1> {
    %0 = memref.alloc() : !IODDRData1
    %1 = memref.alloc() : !IODDRSM1
    %2 = VPUIP.GroupSparseBuffer(%0, %1) -> !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>

    %3 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg2: !VPUIP.SparseBuffer<data=memref<1x8x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [64, 4, 2, 1]}, @DDR>, sparsity_map=!IODDRSM1>) outputs(%2 as %arg3: !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) -> !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1> {
      %9 = VPUIP.Copy inputs(%arg2 : !VPUIP.SparseBuffer<data=memref<1x8x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [64, 4, 2, 1]}, @DDR>, sparsity_map=!IODDRSM1>) outputs(%arg3 : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) -> !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>
    }
    %4 = memref.alloc() : !IODDRData1
    %5 = memref.alloc() : !IODDRSM1
    %6 = VPUIP.GroupSparseBuffer(%4, %5) -> !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>

    %7 = VPUIP.Copy inputs(%3 : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) outputs(%6 : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) -> !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>
    %8 = VPUIP.Copy inputs(%7 : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) outputs(%arg1 : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) -> !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>
    return %8 : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>

    // CHECK:       [[BUFF_0_DATA:%.+]] = memref.alloc() : memref<1x8x2x2xf16, @DDR>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x8x2x2xi1, @DDR>
    // CHECK:       [[NCECLUSTERTILING_0:%.+]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg2: !VPUIP.SparseBuffer<data=memref<1x8x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [64, 4, 2, 1]}, @DDR>, sparsity_map=memref<1x8x2x2xi1, @DDR>>) 
    // CHECK-SAME:         outputs(%arg1 as %arg3: !VPUIP.SparseBuffer<data=memref<1x8x2x2xf16, @DDR>, sparsity_map=memref<1x8x2x2xi1, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x8x2x2xf16, @DDR>, sparsity_map=memref<1x8x2x2xi1, @DDR>> {
    // CHECK:       [[COPY_0:%.+]] = VPUIP.Copy inputs(%arg2 : !VPUIP.SparseBuffer<data=memref<1x8x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [64, 4, 2, 1]}, @DDR>, sparsity_map=memref<1x8x2x2xi1, @DDR>>) 
    // CHECK-SAME:         outputs(%arg3 : !VPUIP.SparseBuffer<data=memref<1x8x2x2xf16, @DDR>, sparsity_map=memref<1x8x2x2xi1, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x8x2x2xf16, @DDR>, sparsity_map=memref<1x8x2x2xi1, @DDR>>
    // CHECK:       }
    // CHECK:       return [[NCECLUSTERTILING_0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IODDRData0 = memref<1x8x2x2xf16, @DDR>
!IODDRSM0 = memref<1x8x2x2xi1, @DDR>
!IODDRSparse0 = !VPUIP.SparseBuffer<data=!IODDRData0, sparsity_map=!IODDRSM0>

!IODDRData1 = memref<1x32x2x2xf16, @DDR>
!IODDRSM1 = memref<1x32x2x2xi1, @DDR>
!IODDRSparse1 = !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>

// CHECK-LABEL: @FuseTwoLastCopiesSparse
func.func @FuseTwoLastCopiesSparse(%arg0: !VPUIP.SparseBuffer<data=!IODDRData0, sparsity_map=!IODDRSM0>, %arg1: !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>, %arg2: !VPUIP.SparseBuffer<data=!IODDRData0, sparsity_map=!IODDRSM0>, %arg3: !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) -> (!VPUIP.SparseBuffer<data=!IODDRData0, sparsity_map=!IODDRSM0>, !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) {
    %0 = memref.alloc() : !IODDRData0
    %1 = memref.alloc() : !IODDRSM0
    %2 = VPUIP.GroupSparseBuffer(%0, %1) -> !VPUIP.SparseBuffer<data=!IODDRData0, sparsity_map=!IODDRSM0>

    %3 = VPUIP.Copy inputs(%arg0 : !VPUIP.SparseBuffer<data=!IODDRData0, sparsity_map=!IODDRSM0>) outputs(%2 : !VPUIP.SparseBuffer<data=!IODDRData0, sparsity_map=!IODDRSM0>) -> !VPUIP.SparseBuffer<data=!IODDRData0, sparsity_map=!IODDRSM0>
    %4 = VPUIP.Copy inputs(%3 : !VPUIP.SparseBuffer<data=!IODDRData0, sparsity_map=!IODDRSM0>) outputs(%arg2 : !VPUIP.SparseBuffer<data=!IODDRData0, sparsity_map=!IODDRSM0>) -> !VPUIP.SparseBuffer<data=!IODDRData0, sparsity_map=!IODDRSM0>
    %5 = memref.alloc() : !IODDRData1
    %6 = memref.alloc() : !IODDRSM1
    %7 = VPUIP.GroupSparseBuffer(%5, %6) -> !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>

    %8 = VPUIP.Copy inputs(%arg1 : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) outputs(%7 : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) -> !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>
    %9 = VPUIP.Copy inputs(%8 : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) outputs(%arg3 : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) -> !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>
    return %4, %9 : !VPUIP.SparseBuffer<data=!IODDRData0, sparsity_map=!IODDRSM0>, !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>

    // CHECK:       [[COPY_0:%.+]] = VPUIP.Copy inputs(%arg0 : !VPUIP.SparseBuffer<data=memref<1x8x2x2xf16, @DDR>, sparsity_map=memref<1x8x2x2xi1, @DDR>>) 
    // CHECK-SAME:         outputs(%arg2 : !VPUIP.SparseBuffer<data=memref<1x8x2x2xf16, @DDR>, sparsity_map=memref<1x8x2x2xi1, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x8x2x2xf16, @DDR>, sparsity_map=memref<1x8x2x2xi1, @DDR>>
    // CHECK:       [[COPY_1:%.+]] = VPUIP.Copy inputs(%arg1 : !VPUIP.SparseBuffer<data=memref<1x32x2x2xf16, @DDR>, sparsity_map=memref<1x32x2x2xi1, @DDR>>) 
    // CHECK-SAME:         outputs(%arg3 : !VPUIP.SparseBuffer<data=memref<1x32x2x2xf16, @DDR>, sparsity_map=memref<1x32x2x2xi1, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x32x2x2xf16, @DDR>, sparsity_map=memref<1x32x2x2xi1, @DDR>>
    // CHECK:       return [[COPY_0]], [[COPY_1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IODDRSparse0 = !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>, sparsity_map=memref<1x144x64x128xi1, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>>
!IODDRData1 = memref<1x144x64x128xf16, #NHWC>
!IODDRSM1 = memref<1x144x64x128xi1, #NHWC>
!IODDRSparse1 = !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>

!IODDRSparse2 = !VPUIP.SparseBuffer<data=memref<1x144x128x128xf16, #NHWC, @DDR>, sparsity_map=memref<1x144x128x128xi1, #NHWC, @DDR>>
!IODDRData3 = memref<1x144x64x128xf16, #NHWC, @DDR>
!IODDRSM3 = memref<1x144x64x128xi1, #NHWC, @DDR>
!IODDRSparse3 = !VPUIP.SparseBuffer<data=!IODDRData3, sparsity_map=!IODDRSM3>

!IODistrCMXData0 = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED", 
    num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64
}>

!IODistrCMXSM0 = !VPUIP.DistributedBuffer<
    1x144x64x128xi1, #NHWC, @CMX_NN, {
    mode = "SEGMENTED", 
    num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64
}>

!IODistrCMXSparse0 = !VPUIP.SparseBuffer<
    data=!IODistrCMXData0, sparsity_map=!IODistrCMXSM0>

!IOCMXData0 = memref<1x144x64x128xf16, #NHWC, @CMX_NN>
!IOCMXSM0 = memref<1x144x64x128xi1, #NHWC, @CMX_NN>
!IOCMXSparse0 = !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>

!Weights0 = memref<32x144x1x1xf16, #NHWC, @CMX_NN>
!SMCMX0 = memref<1x32x64x128xi1, #NHWC, @CMX_NN>
!WeightsTable0 = memref<32x1x1x4xsi32, @CMX_NN>
!IOCMX1 = memref<1x32x64x128xf16, #NHWC, @CMX_NN>

// CHECK-LABEL: @DDR2DDRCopyInputSparse
func.func @DDR2DDRCopyInputSparse(%arg0: !IODDRSparse2, %arg1: !Weights0, %arg2: memref<144x1x1x4xsi32, @CMX_NN>) -> !IODistrCMXSparse0 {
    %0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 144, 64, 128] : !IODDRSparse2 to !IODDRSparse0
    %1 = memref.alloc() : !IODDRData3
    %2 = memref.alloc() : !IODDRSM3
    %3 = VPUIP.GroupSparseBuffer(%1, %2) -> !VPUIP.SparseBuffer<data=!IODDRData3, sparsity_map=!IODDRSM3>

    %4 = VPUIP.Copy inputs(%0 : !IODDRSparse0) outputs(%3 : !VPUIP.SparseBuffer<data=!IODDRData3, sparsity_map=!IODDRSM3>) -> !VPUIP.SparseBuffer<data=!IODDRData3, sparsity_map=!IODDRSM3>
    %5 = VPURT.AllocDistributed -> !IODistrCMXData0
    %6 = VPURT.AllocDistributed -> !IODistrCMXSM0
    %7 = VPUIP.GroupSparseBuffer(%5, %6) -> !IODistrCMXSparse0

    %8 = VPUIP.NCEClusterTiling inputs(%4 as %arg3: !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) outputs(%7 as %arg4: !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>) -> !IODistrCMXSparse0 {
      %14 = VPUIP.Copy inputs(%arg3 : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) outputs(%arg4 : !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>) -> !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>
    }
    %9 = VPURT.AllocDistributed -> !IODistrCMXData0
    %10 = VPURT.AllocDistributed -> !IODistrCMXSM0
    %11 = VPUIP.GroupSparseBuffer(%9, %10) -> !IODistrCMXSparse0

    %data, %sparsityMap = VPUIP.UngroupSparseBuffer(%8) {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} -> !IODistrCMXData0, !IODistrCMXSM0
    %data_0, %sparsityMap_1 = VPUIP.UngroupSparseBuffer(%11) {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} -> !IODistrCMXData0, !IODistrCMXSM0
    %12:2 = VPUIP.NCEClusterTiling inputs(%data as %arg3: !IOCMXData0, %sparsityMap as %arg4: !IOCMXSM0, %arg1 as %arg5: !Weights0, %arg2 as %arg6: !WeightsTable0) 
                                   outputs(%data_0 as %arg7: !IOCMX1, %sparsityMap_1 as %arg8: !SMCMX0) -> (!IODistrCMXData0, !IODistrCMXSM0) {
      %14:2 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1],
                                   kernel_strides = [1, 1],
                                   minimumHardwareExecutionCost = 9240 : i64, task_type = #VPUIP.nce_task_type<CONV>}
         input(%arg3 : !IOCMXData0)
         input_sparsity_map(%arg4 : !IOCMXSM0)
         weights(%arg5 : !Weights0)
         weight_table(%arg6 : !WeightsTable0)
         parent_input(%arg3 : !IOCMXData0)
         parent_input_sparsity_map(%arg4 : !IOCMXSM0)
         parent_output(%arg7 : !IOCMX1)
         parent_output_sparsity_map(%arg8 : !SMCMX0)
         outputs(%arg7 : !IOCMX1)
         output_sparsity_map(%arg8 : !SMCMX0)
         -> !IOCMX1, !SMCMX0 variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [15, 5, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
      }
    }
    %13 = VPUIP.GroupSparseBuffer(%12#0, %12#1) -> !IODistrCMXSparse0

    return %13 : !IODistrCMXSparse0

    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 144, 64, 128]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x144x128x128xf16, #NHWC, @DDR>, sparsity_map=memref<1x144x128x128xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>, sparsity_map=memref<1x144x64x128xi1, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>>
    // CHECK:       [[BUFF_0_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_0_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_0:%.+]] = VPUIP.GroupSparseBuffer([[BUFF_0_DATA]], [[BUFF_0_SM]])
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>>
      
    // CHECK:       [[COPY_0:%.+]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW_0]] as %arg3: !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, #NHWC>, sparsity_map=memref<1x144x64x128xi1, #NHWC>>) 
    // CHECK-SAME:         outputs([[BUFF_0]] as %arg4: !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x144x64x128xi1, #NHWC, @CMX_NN>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>> {
    // CHECK:       [[inner_0:%.+]] = VPUIP.Copy inputs(%arg3 : !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, #NHWC>, sparsity_map=memref<1x144x64x128xi1, #NHWC>>) 
    // CHECK-SAME:         outputs(%arg4 : !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x144x64x128xi1, #NHWC, @CMX_NN>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x144x64x128xi1, #NHWC, @CMX_NN>>
      
    
    // CHECK:       [[BUFF_1_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_1_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_1:%.+]] = VPUIP.GroupSparseBuffer([[BUFF_1_DATA]], [[BUFF_1_SM]])
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>>
      
    // CHECK:       [[DATA_0:%.+]], [[SM_0:%.+]] = VPUIP.UngroupSparseBuffer([[COPY_0]]) {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>, !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:       [[DATA_1:%.+]], [[SM_1:%.+]] = VPUIP.UngroupSparseBuffer([[BUFF_1]]) {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>, !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:       [[NCE_0:%.+]]:2 = VPUIP.NCEClusterTiling inputs([[DATA_0]] as %arg3: memref<1x144x64x128xf16, #NHWC, @CMX_NN>, [[SM_0]] as %arg4: memref<1x144x64x128xi1, #NHWC, @CMX_NN>, %arg1 as %arg5: memref<32x144x1x1xf16, #NHWC, @CMX_NN>, %arg2 as %arg6: memref<32x1x1x4xsi32, @CMX_NN>) 
    // CHECK-SAME:         outputs([[DATA_0]]_0 as %arg7: memref<1x32x64x128xf16, #NHWC, @CMX_NN>, [[SM_0]]_1 as %arg8: memref<1x32x64x128xi1, #NHWC, @CMX_NN>)
    // CHECK-SAME:          -> (!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>, !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>) {
    // CHECK:       [[inner_1:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:          input(%arg3 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>) 
    // CHECK-SAME:          weights(%arg5 : memref<32x144x1x1xf16, #NHWC, @CMX_NN>) 
    // CHECK-SAME:          parent_input(%arg3 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>) 
    // CHECK-SAME:          parent_output(%arg7 : memref<1x32x64x128xf16, #NHWC, @CMX_NN>) 
    // CHECK-SAME:          outputs(%arg7 : memref<1x32x64x128xf16, #NHWC, @CMX_NN>) 
    
    // CHECK:       [[BUFF_2:%.+]] = VPUIP.GroupSparseBuffer([[NCE_0]]#0, [[NCE_0]]#1)
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>>
      
    // CHECK:       return [[BUFF_2]]
}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IODDRSparse0 = !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>, sparsity_map=memref<1x144x64x128xi1, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>>
!IODDRData1 = memref<1x144x64x128xf16, #NHWC>
!IODDRSM1 = memref<1x144x64x128xi1, #NHWC>
!IODDRSparse1 = !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>

!IODDRSparse2 = !VPUIP.SparseBuffer<data=memref<1x144x128x128xf16, #NHWC, @DDR>, sparsity_map=memref<1x144x128x128xi1, #NHWC, @DDR>>
!IODDRData3 = memref<1x144x64x128xf16, #NHWC, @DDR>
!IODDRSM3 = memref<1x144x64x128xi1, #NHWC, @DDR>
!IODDRSparse3 = !VPUIP.SparseBuffer<data=!IODDRData3, sparsity_map=!IODDRSM3>

!IODistrCMXData0 = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED", 
    num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64
}>

!IODistrCMXSM0 = !VPUIP.DistributedBuffer<
    1x144x64x128xi1, #NHWC, @CMX_NN, {
    mode = "SEGMENTED", 
    num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64
}>

!IODistrCMXSparse0 = !VPUIP.SparseBuffer<
    data=!IODistrCMXData0, sparsity_map=!IODistrCMXSM0>

!IOCMXData0 = memref<1x144x64x128xf16, #NHWC, @CMX_NN>
!IOCMXSM0 = memref<1x144x64x128xi1, #NHWC, @CMX_NN>
!IOCMXSparse0 = !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>

!Weights0 = memref<32x144x1x1xf16, #NHWC, @CMX_NN>
!SMCMX0 = memref<1x32x64x128xi1, #NHWC, @CMX_NN>
!WeightsTable0 = memref<32x1x1x4xsi32, @CMX_NN>
!IOCMX1 = memref<1x32x64x128xf16, #NHWC, @CMX_NN>

// CHECK-LABEL: @DDR2DDRCopyInputDistrSparse
func.func @DDR2DDRCopyInputDistrSparse(%arg0: !IODDRSparse2, %arg1: !Weights0, %arg2: memref<144x1x1x4xsi32, @CMX_NN>) -> !IODistrCMXSparse0 {
    %0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 144, 64, 128] : !IODDRSparse2 to !IODDRSparse0
    %1 = memref.alloc() : !IODDRData3
    %2 = memref.alloc() : !IODDRSM3
    %3 = VPUIP.GroupSparseBuffer(%1, %2) -> !VPUIP.SparseBuffer<data=!IODDRData3, sparsity_map=!IODDRSM3>

    %4 = VPUIP.Copy inputs(%0 : !IODDRSparse0) outputs(%3 : !VPUIP.SparseBuffer<data=!IODDRData3, sparsity_map=!IODDRSM3>) -> !VPUIP.SparseBuffer<data=!IODDRData3, sparsity_map=!IODDRSM3>
    %5 = VPURT.AllocDistributed -> !IODistrCMXData0
    %6 = VPURT.AllocDistributed -> !IODistrCMXSM0
    %7 = VPUIP.GroupSparseBuffer(%5, %6) -> !IODistrCMXSparse0

    %8 = VPUIP.NCEClusterTiling inputs(%4 as %arg3: !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) outputs(%7 as %arg4: !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>) -> !IODistrCMXSparse0 {
      %14 = VPUIP.Copy inputs(%arg3 : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) outputs(%arg4 : !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>) -> !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>
    }
    %9 = VPURT.AllocDistributed -> !IODistrCMXData0
    %10 = VPURT.AllocDistributed -> !IODistrCMXSM0
    %11 = VPUIP.GroupSparseBuffer(%9, %10) -> !IODistrCMXSparse0

    %data, %sparsityMap = VPUIP.UngroupSparseBuffer(%8) {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} -> !IODistrCMXData0, !IODistrCMXSM0
    %data_0, %sparsityMap_1 = VPUIP.UngroupSparseBuffer(%11) {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} -> !IODistrCMXData0, !IODistrCMXSM0
    %12:2 = VPUIP.NCEClusterTiling inputs(%data as %arg3: !IOCMXData0, %sparsityMap as %arg4: !IOCMXSM0, %arg1 as %arg5: !Weights0, %arg2 as %arg6: !WeightsTable0) 
                                   outputs(%data_0 as %arg7: !IOCMX1, %sparsityMap_1 as %arg8: !SMCMX0) -> (!IODistrCMXData0, !IODistrCMXSM0) {
      %14:2 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1],
                                   kernel_strides = [1, 1],
                                   minimumHardwareExecutionCost = 9240 : i64, task_type = #VPUIP.nce_task_type<CONV>}
         input(%arg3 : !IOCMXData0)
         input_sparsity_map(%arg4 : !IOCMXSM0)
         weights(%arg5 : !Weights0)
         weight_table(%arg6 : !WeightsTable0)
         parent_input(%arg3 : !IOCMXData0)
         parent_input_sparsity_map(%arg4 : !IOCMXSM0)
         parent_output(%arg7 : !IOCMX1)
         parent_output_sparsity_map(%arg8 : !SMCMX0)
         outputs(%arg7 : !IOCMX1)
         output_sparsity_map(%arg8 : !SMCMX0)
         -> !IOCMX1, !SMCMX0 variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [15, 5, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
      }
    }
    %13 = VPUIP.GroupSparseBuffer(%12#0, %12#1) -> !IODistrCMXSparse0

    return %13 : !IODistrCMXSparse0

    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 144, 64, 128]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x144x128x128xf16, #NHWC, @DDR>, sparsity_map=memref<1x144x128x128xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>, sparsity_map=memref<1x144x64x128xi1, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>>
    // CHECK:       [[BUFF_0_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_0_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_0:%.+]] = VPUIP.GroupSparseBuffer([[BUFF_0_DATA]], [[BUFF_0_SM]])
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>>
      
    // CHECK:       [[COPY_0:%.+]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW_0]] as %arg3: !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, #NHWC>, sparsity_map=memref<1x144x64x128xi1, #NHWC>>) 
    // CHECK-SAME:         outputs([[BUFF_0]] as %arg4: !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x144x64x128xi1, #NHWC, @CMX_NN>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>> {
    // CHECK:       [[inner_0:%.+]] = VPUIP.Copy inputs(%arg3 : !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, #NHWC>, sparsity_map=memref<1x144x64x128xi1, #NHWC>>) 
    // CHECK-SAME:         outputs(%arg4 : !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x144x64x128xi1, #NHWC, @CMX_NN>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x144x64x128xi1, #NHWC, @CMX_NN>>

    // CHECK:       [[BUFF_1_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_1_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_1:%.+]] = VPUIP.GroupSparseBuffer([[BUFF_1_DATA]], [[BUFF_1_SM]])
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>>
      
    // CHECK:       [[DATA_0:%.+]], [[SM_0:%.+]] = VPUIP.UngroupSparseBuffer([[COPY_0]]) {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>, !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:       [[DATA_1:%.+]], [[SM_1:%.+]] = VPUIP.UngroupSparseBuffer([[BUFF_1]]) {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>, !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:       [[NCE_0:%.+]]:2 = VPUIP.NCEClusterTiling inputs([[DATA_0]] as %arg3: memref<1x144x64x128xf16, #NHWC, @CMX_NN>, [[SM_0]] as %arg4: memref<1x144x64x128xi1, #NHWC, @CMX_NN>, %arg1 as %arg5: memref<32x144x1x1xf16, #NHWC, @CMX_NN>, %arg2 as %arg6: memref<32x1x1x4xsi32, @CMX_NN>) 
    // CHECK-SAME:         outputs([[DATA_0]]_0 as %arg7: memref<1x32x64x128xf16, #NHWC, @CMX_NN>, [[SM_0]]_1 as %arg8: memref<1x32x64x128xi1, #NHWC, @CMX_NN>)
    // CHECK-SAME:          -> (!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>, !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>) {
    // CHECK:       [[inner_1:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:          input(%arg3 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>) 
    // CHECK-SAME:          weights(%arg5 : memref<32x144x1x1xf16, #NHWC, @CMX_NN>) 
    // CHECK-SAME:          parent_input(%arg3 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>) 
    // CHECK-SAME:          parent_output(%arg7 : memref<1x32x64x128xf16, #NHWC, @CMX_NN>) 
    // CHECK-SAME:          outputs(%arg7 : memref<1x32x64x128xf16, #NHWC, @CMX_NN>) 

    // CHECK:       [[BUFF_2:%.+]] = VPUIP.GroupSparseBuffer([[NCE_0]]#0, [[NCE_0]]#1)
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>>
      
    // CHECK:       return [[BUFF_2]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.99909667968750004:124>

!IODDRSparse0 = !VPUIP.SparseBuffer<data=memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x1x33xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>
!IODDRData1 = memref<1x512x18x33x!qElemType, #NHWC, @DDR>
!IODDRSM1 = memref<1x512x18x33xi1, #NHWC, @DDR>
!IODDRSparse1 = !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>

!IODDRSparse2 = !VPUIP.SparseBuffer<data=memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x1xi1, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>>
!IODDRSparse3 = !VPUIP.SparseBuffer<data=memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x32xi1, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>>
!IODDRSparse4 = !VPUIP.SparseBuffer<data=memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>, sparsity_map=memref<1x512x18x1xi1, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>>
!IODDRSparse5 = !VPUIP.SparseBuffer<data=memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x33xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>
!IODDRData6 = memref<1x512x19x33x!qElemType, #NHWC, @DDR>
!IODDRSM6 = memref<1x512x19x33xi1, #NHWC, @DDR>
!IODDRSparse6 = !VPUIP.SparseBuffer<data=!IODDRData6, sparsity_map=!IODDRSM6>

!IODDRSparse7 = !VPUIP.SparseBuffer<data=memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x1x33xi1, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>>
!IODDRSparse8 = !VPUIP.SparseBuffer<data=memref<1x512x18x32x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x18x32xi1, #NHWC, @DDR>>

// CHECK-LABEL: @DDR2DDROfConcatInputSparse
func.func @DDR2DDROfConcatInputSparse(%arg0: !IODDRSparse8) -> !VPUIP.SparseBuffer<data=!IODDRData6, sparsity_map=!IODDRSM6> {
    %0 = VPUIP.SubView %arg0 [0, 0, 0, 31] [1, 512, 18, 1] : !IODDRSparse8 to !IODDRSparse4
    %1 = memref.alloc() : !IODDRData1
    %2 = memref.alloc() : !IODDRSM1
    %3 = VPUIP.GroupSparseBuffer(%1, %2) -> !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>

    %4 = VPUIP.SubView %3 [0, 0, 0, 0] [1, 512, 18, 32] : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1> to !IODDRSparse3
    %5 = VPUIP.Copy inputs(%arg0 : !IODDRSparse8) outputs(%4 : !IODDRSparse3) -> !IODDRSparse3
    %6 = VPUIP.SubView %3 [0, 0, 0, 32] [1, 512, 18, 1] : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1> to !IODDRSparse2
    %7 = VPUIP.Copy inputs(%0 : !IODDRSparse4) outputs(%6 : !IODDRSparse2) -> !IODDRSparse2
    %8 = VPUIP.ConcatView inputs(%5, %7 : !IODDRSparse3, !IODDRSparse2) outputs(%3 : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) -> !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>
    %9 = VPUIP.SubView %8 [0, 0, 17, 0] [1, 512, 1, 33] : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1> to !IODDRSparse7
    %10 = memref.alloc() : !IODDRData6
    %11 = memref.alloc() : !IODDRSM6
    %12 = VPUIP.GroupSparseBuffer(%10, %11) -> !VPUIP.SparseBuffer<data=!IODDRData6, sparsity_map=!IODDRSM6>

    %13 = VPUIP.SubView %12 [0, 0, 0, 0] [1, 512, 18, 33] : !VPUIP.SparseBuffer<data=!IODDRData6, sparsity_map=!IODDRSM6> to !IODDRSparse5
    %14 = VPUIP.Copy inputs(%8 : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) outputs(%13 : !IODDRSparse5) -> !IODDRSparse5
    %15 = VPUIP.SubView %12 [0, 0, 18, 0] [1, 512, 1, 33] : !VPUIP.SparseBuffer<data=!IODDRData6, sparsity_map=!IODDRSM6> to !IODDRSparse0
    %16 = VPUIP.Copy inputs(%9 : !IODDRSparse7) outputs(%15 : !IODDRSparse0) -> !IODDRSparse0
    %17 = VPUIP.ConcatView inputs(%14, %16 : !IODDRSparse5, !IODDRSparse0) outputs(%12 : !VPUIP.SparseBuffer<data=!IODDRData6, sparsity_map=!IODDRSM6>) -> !VPUIP.SparseBuffer<data=!IODDRData6, sparsity_map=!IODDRSM6>
    return %17 : !VPUIP.SparseBuffer<data=!IODDRData6, sparsity_map=!IODDRSM6>

    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 31] [1, 512, 18, 1]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x512x18x32x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x18x32xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>, sparsity_map=memref<1x512x18x1xi1, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>>
    // CHECK:       [[BUFF_0_DATA:%.+]] = memref.alloc() : memref<1x512x19x33x!qElemType, #NHWC, @DDR>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x512x19x33xi1, #NHWC, @DDR>
    // CHECK:       [[BUFF_0:%.+]] = VPUIP.GroupSparseBuffer([[BUFF_0_DATA]], [[BUFF_0_SM]])
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x512x19x33x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x19x33xi1, #NHWC, @DDR>>
      
    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 0] [1, 512, 18, 33]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x512x19x33x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x19x33xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x33xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>
    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView [[SUBVIEW_1]] [0, 0, 0, 0] [1, 512, 18, 32]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x33xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x32xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>
    // CHECK:       [[COPY_0:%.+]] = VPUIP.Copy inputs(%arg0 : !VPUIP.SparseBuffer<data=memref<1x512x18x32x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x18x32xi1, #NHWC, @DDR>>) 
    // CHECK-SAME:         outputs([[SUBVIEW_2]] : !VPUIP.SparseBuffer<data=memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x32xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x32xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>
    // CHECK:       [[SUBVIEW_3:%.+]] = VPUIP.SubView [[SUBVIEW_1]] [0, 0, 0, 32] [1, 512, 18, 1]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x33xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x1xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>
    // CHECK:       [[COPY_1:%.+]] = VPUIP.Copy inputs([[SUBVIEW_0]] : !VPUIP.SparseBuffer<data=memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>, sparsity_map=memref<1x512x18x1xi1, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>>) 
    // CHECK-SAME:         outputs([[SUBVIEW_3]] : !VPUIP.SparseBuffer<data=memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x1xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x1xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>
    // CHECK:       [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]] : !VPUIP.SparseBuffer<data=memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x32xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>, !VPUIP.SparseBuffer<data=memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x1xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>) 
    // CHECK-SAME:         outputs([[SUBVIEW_1]] : !VPUIP.SparseBuffer<data=memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x33xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x33xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>
    // CHECK:       [[SUBVIEW_4:%.+]] = VPUIP.SubView [[CONCATVIEW_0]] [0, 0, 17, 0] [1, 512, 1, 33]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x33xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x1x33xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>
    // CHECK:       [[SUBVIEW_5:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 0, 18, 0] [1, 512, 1, 33]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x512x19x33x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x19x33xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x1x33xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>
    // CHECK:       [[COPY_2:%.+]] = VPUIP.Copy inputs([[SUBVIEW_4]] : !VPUIP.SparseBuffer<data=memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x1x33xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>) 
    // CHECK-SAME:         outputs([[SUBVIEW_5]] : !VPUIP.SparseBuffer<data=memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x1x33xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x1x33xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>
    // CHECK:       [[CONCATVIEW_1:%.+]] = VPUIP.ConcatView inputs([[CONCATVIEW_0]], [[COPY_2]] : !VPUIP.SparseBuffer<data=memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x33xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>, !VPUIP.SparseBuffer<data=memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x1x33xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>) 
    // CHECK-SAME:         outputs([[BUFF_0]] : !VPUIP.SparseBuffer<data=memref<1x512x19x33x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x19x33xi1, #NHWC, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x512x19x33x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x19x33xi1, #NHWC, @DDR>>
    // CHECK:       return [[CONCATVIEW_1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.99909667968750004:124>

!IODDRSparse0 = !VPUIP.SparseBuffer<data=memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x1x33xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>
!IODDRData1 = memref<1x512x18x33x!qElemType, #NHWC, @DDR>
!IODDRSM1 = memref<1x512x18x33xi1, #NHWC, @DDR>
!IODDRSparse1 = !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>

!IODDRSparse2 = !VPUIP.SparseBuffer<data=memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x1xi1, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>>
!IODDRSparse3 = !VPUIP.SparseBuffer<data=memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x32xi1, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>>
!IODDRSparse4 = !VPUIP.SparseBuffer<data=memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>, sparsity_map=memref<1x512x18x1xi1, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>>
!IODDRSparse5 = !VPUIP.SparseBuffer<data=memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x33xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>
!IODDRData6 = memref<1x512x19x33x!qElemType, #NHWC, @DDR>
!IODDRSM6 = memref<1x512x19x33xi1, #NHWC, @DDR>
!IODDRSparse6 = !VPUIP.SparseBuffer<data=!IODDRData6, sparsity_map=!IODDRSM6>

!IODDRSparse7 = !VPUIP.SparseBuffer<data=memref<1x512x1x33x!qElemType, {order = #NHWC}>, sparsity_map=memref<1x512x1x33xi1>>
!IODDRSparse8 = !VPUIP.SparseBuffer<data=memref<1x512x18x32x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x18x32xi1, #NHWC, @DDR>>

// CHECK-LABEL: @DDR2DDROfConcatWithConstInputSparse
func.func @DDR2DDROfConcatWithConstInputSparse(%arg0: !IODDRSparse8) -> !VPUIP.SparseBuffer<data=!IODDRData6, sparsity_map=!IODDRSM6> {
    %cst = const.Declare memref<1x512x1x33x!qElemType, {order = #NHWC}> = dense<0.0> : tensor<1x512x1x33xf16>,
        [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>, #const.Reorder<#NHWC>]
    %cst_0 = const.Declare memref<1x512x1x33xi1> = dense<false> : tensor<1x512x1x33xi1>
    %0 = VPUIP.SubView %arg0 [0, 0, 0, 31] [1, 512, 18, 1] : !IODDRSparse8 to !IODDRSparse4
    %1 = memref.alloc() : !IODDRData1
    %2 = memref.alloc() : !IODDRSM1
    %3 = VPUIP.GroupSparseBuffer(%1, %2) -> !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>

    %4 = VPUIP.SubView %3 [0, 0, 0, 0] [1, 512, 18, 32] : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1> to !IODDRSparse3
    %5 = VPUIP.Copy inputs(%arg0 : !IODDRSparse8) outputs(%4 : !IODDRSparse3) -> !IODDRSparse3
    %6 = VPUIP.SubView %3 [0, 0, 0, 32] [1, 512, 18, 1] : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1> to !IODDRSparse2
    %7 = VPUIP.Copy inputs(%0 : !IODDRSparse4) outputs(%6 : !IODDRSparse2) -> !IODDRSparse2
    %8 = VPUIP.ConcatView inputs(%5, %7 : !IODDRSparse3, !IODDRSparse2) outputs(%3 : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) -> !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>
    %9 = VPUIP.GroupSparseBuffer(%cst, %cst_0) -> !IODDRSparse7
    %10 = memref.alloc() : !IODDRData6
    %11 = memref.alloc() : !IODDRSM6
    %12 = VPUIP.GroupSparseBuffer(%10, %11) -> !VPUIP.SparseBuffer<data=!IODDRData6, sparsity_map=!IODDRSM6>

    %13 = VPUIP.SubView %12 [0, 0, 0, 0] [1, 512, 18, 33] : !VPUIP.SparseBuffer<data=!IODDRData6, sparsity_map=!IODDRSM6> to !IODDRSparse5
    %14 = VPUIP.Copy inputs(%8 : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) outputs(%13 : !IODDRSparse5) -> !IODDRSparse5
    %15 = VPUIP.SubView %12 [0, 0, 18, 0] [1, 512, 1, 33] : !VPUIP.SparseBuffer<data=!IODDRData6, sparsity_map=!IODDRSM6> to !IODDRSparse0
    %16 = VPUIP.Copy inputs(%9 : !IODDRSparse7) outputs(%15 : !IODDRSparse0) -> !IODDRSparse0
    %17 = VPUIP.ConcatView inputs(%14, %16 : !IODDRSparse5, !IODDRSparse0) outputs(%12 : !VPUIP.SparseBuffer<data=!IODDRData6, sparsity_map=!IODDRSM6>) -> !VPUIP.SparseBuffer<data=!IODDRData6, sparsity_map=!IODDRSM6>
    return %17 : !VPUIP.SparseBuffer<data=!IODDRData6, sparsity_map=!IODDRSM6>

    // CHECK:    [[CONST_DATA:%.+]] = const.Declare memref<1x512x1x33x!qElemType, {order = #NHWC}>
    // CHECK-SAME:      dense<0.000000e+00> : tensor<1x512x1x33xf16>
    // CHECK-SAME:      [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>, #const.Reorder<#NHWC>]

    // CHECK:   [[CONST_MAP:%.+]] = const.Declare memref<1x512x1x33xi1> = dense<false> : tensor<1x512x1x33xi1>
    
    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 31] [1, 512, 18, 1]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x512x18x32x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x18x32xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>, sparsity_map=memref<1x512x18x1xi1, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>>
    // CHECK:       [[BUFF_0_DATA:%.+]] = memref.alloc() : memref<1x512x19x33x!qElemType, #NHWC, @DDR>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x512x19x33xi1, #NHWC, @DDR>
    // CHECK:       [[BUFF_0:%.+]] = VPUIP.GroupSparseBuffer([[BUFF_0_DATA]], [[BUFF_0_SM]])
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x512x19x33x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x19x33xi1, #NHWC, @DDR>>
      
    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 0] [1, 512, 18, 33]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x512x19x33x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x19x33xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x33xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>
    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView [[SUBVIEW_1]] [0, 0, 0, 0] [1, 512, 18, 32]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x33xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x32xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>
    // CHECK:       [[COPY_0:%.+]] = VPUIP.Copy inputs(%arg0 : !VPUIP.SparseBuffer<data=memref<1x512x18x32x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x18x32xi1, #NHWC, @DDR>>) 
    // CHECK-SAME:         outputs([[SUBVIEW_2]] : !VPUIP.SparseBuffer<data=memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x32xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x32xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>
    // CHECK:       [[SUBVIEW_3:%.+]] = VPUIP.SubView [[SUBVIEW_1]] [0, 0, 0, 32] [1, 512, 18, 1]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x33xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x1xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>
    // CHECK:       [[COPY_1:%.+]] = VPUIP.Copy inputs([[SUBVIEW_0]] : !VPUIP.SparseBuffer<data=memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>, sparsity_map=memref<1x512x18x1xi1, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>>) 
    // CHECK-SAME:         outputs([[SUBVIEW_3]] : !VPUIP.SparseBuffer<data=memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x1xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x1xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>
    // CHECK:       [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]] : !VPUIP.SparseBuffer<data=memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x32xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>, !VPUIP.SparseBuffer<data=memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x1xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>) 
    // CHECK-SAME:         outputs([[SUBVIEW_1]] : !VPUIP.SparseBuffer<data=memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x33xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x33xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>
    // CHECK:       [[CONST_SPASEBUFF:%.+]] = VPUIP.GroupSparseBuffer([[CONST_DATA]], [[CONST_MAP]])
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x512x1x33x!qElemType, {order = #NHWC}>, sparsity_map=memref<1x512x1x33xi1>>

    // CHECK:       [[SUBVIEW_5:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 0, 18, 0] [1, 512, 1, 33]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x512x19x33x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x19x33xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x1x33xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>
    // CHECK:       [[COPY_2:%.+]] = VPUIP.Copy inputs([[CONST_SPASEBUFF]] : !VPUIP.SparseBuffer<data=memref<1x512x1x33x!qElemType, {order = #NHWC}>, sparsity_map=memref<1x512x1x33xi1>>)
    // CHECK-SAME:         outputs([[SUBVIEW_5]] : !VPUIP.SparseBuffer<data=memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x1x33xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x1x33xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>
    // CHECK:       [[CONCATVIEW_1:%.+]] = VPUIP.ConcatView inputs([[CONCATVIEW_0]], [[COPY_2]] : !VPUIP.SparseBuffer<data=memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x33xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>, !VPUIP.SparseBuffer<data=memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x1x33xi1, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>>) 
    // CHECK-SAME:         outputs([[BUFF_0]] : !VPUIP.SparseBuffer<data=memref<1x512x19x33x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x19x33xi1, #NHWC, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x512x19x33x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x19x33xi1, #NHWC, @DDR>>
    // CHECK:       return [[CONCATVIEW_1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.99909667968750004:124>

!IODDRSparse0 = !VPUIP.SparseBuffer<data=memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x1xi1, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>>
!IODDRData1 = memref<1x512x18x33x!qElemType, #NHWC, @DDR>
!IODDRSM1 = memref<1x512x18x33xi1, #NHWC, @DDR>
!IODDRSparse1 = !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>

!IODDRSparse2 = !VPUIP.SparseBuffer<data=memref<1x512x9x32x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x9x32xi1, #NHWC, @DDR>>
!IODDRSparse3 = !VPUIP.SparseBuffer<data=memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>, sparsity_map=memref<1x512x18x1xi1, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>>
!IODDRData4 = memref<1x512x18x32x!qElemType, #NHWC, @DDR>
!IODDRSM4 = memref<1x512x18x32xi1, #NHWC, @DDR>
!IODDRSparse4 = !VPUIP.SparseBuffer<data=!IODDRData4, sparsity_map=!IODDRSM4>

!IODDRSparse5 = !VPUIP.SparseBuffer<data=memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>, sparsity_map=memref<1x512x9x32xi1, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>>
!IODDRSparse6 = !VPUIP.SparseBuffer<data=memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x32xi1, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>>

// CHECK-LABEL: @DDR2DDROfConcatInputStrideCopySparse
func.func @DDR2DDROfConcatInputStrideCopySparse(%arg0: !IODDRSparse2) -> !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1> {
    %0 = memref.alloc() : !IODDRData4
    %1 = memref.alloc() : !IODDRSM4
    %2 = VPUIP.GroupSparseBuffer(%0, %1) -> !VPUIP.SparseBuffer<data=!IODDRData4, sparsity_map=!IODDRSM4>

    %3 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 512, 9, 32] [1, 1, 2, 1] : !VPUIP.SparseBuffer<data=!IODDRData4, sparsity_map=!IODDRSM4> to !IODDRSparse5
    %4 = VPUIP.Copy inputs(%arg0 : !IODDRSparse2) outputs(%3 : !IODDRSparse5) -> !IODDRSparse5
    %5 = VPUIP.SubView %2 [0, 0, 1, 0] [1, 512, 9, 32] [1, 1, 2, 1] : !VPUIP.SparseBuffer<data=!IODDRData4, sparsity_map=!IODDRSM4> to !IODDRSparse5
    %6 = VPUIP.Copy inputs(%arg0 : !IODDRSparse2) outputs(%5 : !IODDRSparse5) -> !IODDRSparse5
    %7 = VPUIP.ConcatView inputs(%4, %6 : !IODDRSparse5, !IODDRSparse5) outputs(%2 : !VPUIP.SparseBuffer<data=!IODDRData4, sparsity_map=!IODDRSM4>) -> !VPUIP.SparseBuffer<data=!IODDRData4, sparsity_map=!IODDRSM4>
    %8 = VPUIP.SubView %7 [0, 0, 0, 31] [1, 512, 18, 1] : !VPUIP.SparseBuffer<data=!IODDRData4, sparsity_map=!IODDRSM4> to !IODDRSparse3
    %9 = memref.alloc() : !IODDRData1
    %10 = memref.alloc() : !IODDRSM1
    %11 = VPUIP.GroupSparseBuffer(%9, %10) -> !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>

    %12 = VPUIP.SubView %11 [0, 0, 0, 0] [1, 512, 18, 32] : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1> to !IODDRSparse6
    %13 = VPUIP.Copy inputs(%7 : !VPUIP.SparseBuffer<data=!IODDRData4, sparsity_map=!IODDRSM4>) outputs(%12 : !IODDRSparse6) -> !IODDRSparse6
    %14 = VPUIP.SubView %11 [0, 0, 0, 32] [1, 512, 18, 1] : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1> to !IODDRSparse0
    %15 = VPUIP.Copy inputs(%8 : !IODDRSparse3) outputs(%14 : !IODDRSparse0) -> !IODDRSparse0
    %16 = VPUIP.ConcatView inputs(%13, %15 : !IODDRSparse6, !IODDRSparse0) outputs(%11 : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) -> !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>
    return %16 : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>

    // CHECK:       [[BUFF_0_DATA:%.+]] = memref.alloc() : memref<1x512x18x32x!qElemType, #NHWC, @DDR>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x512x18x32xi1, #NHWC, @DDR>
    // CHECK:       [[BUFF_0:%.+]] = VPUIP.GroupSparseBuffer([[BUFF_0_DATA]], [[BUFF_0_SM]])
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x512x18x32x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x18x32xi1, #NHWC, @DDR>>
      
    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 0] [1, 512, 9, 32] [1, 1, 2, 1]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x512x18x32x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x18x32xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>, sparsity_map=memref<1x512x9x32xi1, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>>
    // CHECK:       [[COPY_0:%.+]] = VPUIP.Copy inputs(%arg0 : !VPUIP.SparseBuffer<data=memref<1x512x9x32x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x9x32xi1, #NHWC, @DDR>>) 
    // CHECK-SAME:         outputs([[SUBVIEW_0]] : !VPUIP.SparseBuffer<data=memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>, sparsity_map=memref<1x512x9x32xi1, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>, sparsity_map=memref<1x512x9x32xi1, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>>
    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 0, 1, 0] [1, 512, 9, 32] [1, 1, 2, 1]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x512x18x32x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x18x32xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>, sparsity_map=memref<1x512x9x32xi1, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>>
    // CHECK:       [[COPY_1:%.+]] = VPUIP.Copy inputs(%arg0 : !VPUIP.SparseBuffer<data=memref<1x512x9x32x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x9x32xi1, #NHWC, @DDR>>) 
    // CHECK-SAME:         outputs([[SUBVIEW_1]] : !VPUIP.SparseBuffer<data=memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>, sparsity_map=memref<1x512x9x32xi1, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>, sparsity_map=memref<1x512x9x32xi1, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>>
    // CHECK:       [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]] : !VPUIP.SparseBuffer<data=memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>, sparsity_map=memref<1x512x9x32xi1, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>>, !VPUIP.SparseBuffer<data=memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>, sparsity_map=memref<1x512x9x32xi1, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>>) 
    // CHECK-SAME:         outputs([[BUFF_0]] : !VPUIP.SparseBuffer<data=memref<1x512x18x32x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x18x32xi1, #NHWC, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x512x18x32x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x18x32xi1, #NHWC, @DDR>>
    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView [[CONCATVIEW_0]] [0, 0, 0, 31] [1, 512, 18, 1]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x512x18x32x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x18x32xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>, sparsity_map=memref<1x512x18x1xi1, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>>
    // CHECK:       [[BUFF_1_DATA:%.+]] = memref.alloc() : memref<1x512x18x33x!qElemType, #NHWC, @DDR>
    // CHECK:       [[BUFF_1_SM:%.+]] = memref.alloc() : memref<1x512x18x33xi1, #NHWC, @DDR>
    // CHECK:       [[BUFF_1:%.+]] = VPUIP.GroupSparseBuffer([[BUFF_1_DATA]], [[BUFF_1_SM]])
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x512x18x33x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x18x33xi1, #NHWC, @DDR>>
      
    // CHECK:       [[SUBVIEW_3:%.+]] = VPUIP.SubView [[BUFF_1]] [0, 0, 0, 0] [1, 512, 18, 32]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x512x18x33x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x18x33xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x32xi1, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>>
    // CHECK:       [[COPY_2:%.+]] = VPUIP.Copy inputs([[CONCATVIEW_0]] : !VPUIP.SparseBuffer<data=memref<1x512x18x32x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x18x32xi1, #NHWC, @DDR>>) 
    // CHECK-SAME:         outputs([[SUBVIEW_3]] : !VPUIP.SparseBuffer<data=memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x32xi1, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x32xi1, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>>
    // CHECK:       [[SUBVIEW_4:%.+]] = VPUIP.SubView [[BUFF_1]] [0, 0, 0, 32] [1, 512, 18, 1]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x512x18x33x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x18x33xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x1xi1, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>>
    // CHECK:       [[COPY_3:%.+]] = VPUIP.Copy inputs([[SUBVIEW_2]] : !VPUIP.SparseBuffer<data=memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>, sparsity_map=memref<1x512x18x1xi1, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>>) 
    // CHECK-SAME:         outputs([[SUBVIEW_4]] : !VPUIP.SparseBuffer<data=memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x1xi1, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x1xi1, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>>
    // CHECK:       [[CONCATVIEW_1:%.+]] = VPUIP.ConcatView inputs([[COPY_2]], [[COPY_3]] : !VPUIP.SparseBuffer<data=memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x32xi1, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>>, !VPUIP.SparseBuffer<data=memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>, sparsity_map=memref<1x512x18x1xi1, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>>) 
    // CHECK-SAME:         outputs([[BUFF_1]] : !VPUIP.SparseBuffer<data=memref<1x512x18x33x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x18x33xi1, #NHWC, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x512x18x33x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x18x33xi1, #NHWC, @DDR>>
    // CHECK:       return [[CONCATVIEW_1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 5.7832517137714463:123>

!IODDRSM0 = memref<1x512x1x17xi1, #NHWC, @DDR>
!IODDRSparse0 = !VPUIP.SparseBuffer<data=memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=!IODDRSM0>

!IODDRData1 = memref<1x512x1x17x!qElemType, #NHWC, @DDR>
!IODDRSM1 = memref<1x512x1x17xi1, #NHWC, @DDR>
!IODDRSparse1 = !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>

!IOCMXData0 = memref<1x512x9x17x!qElemType, #NHWC, @CMX_NN>
!IOCMXSM0 = memref<1x512x9x17xi1, #NHWC, @CMX_NN>
!IOCMXSparse0 = !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>

!IODDRSparse2 = !VPUIP.SparseBuffer<data=memref<1x512x9x17x!qElemType, #NHWC>, sparsity_map=memref<1x512x9x17xi1, #NHWC>>
!IODDRSM3 = memref<1x512x1x17xi1, #NHWC, @DDR>
!IODDRSparse3 = !VPUIP.SparseBuffer<data=memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [78336, 1, 8704, 512]}, @DDR>, sparsity_map=!IODDRSM3>

!IODDRData4 = memref<1x512x10x17x!qElemType, #NHWC, @DDR>
!IODDRSM4 = memref<1x512x10x17xi1, #NHWC, @DDR>
!IODDRSparse4 = !VPUIP.SparseBuffer<data=!IODDRData4, sparsity_map=!IODDRSM4>

!IODDRData5 = memref<1x512x9x17x!qElemType, #NHWC, @DDR>
!IODDRSM5 = memref<1x512x9x17xi1, #NHWC, @DDR>
!IODDRSparse5 = !VPUIP.SparseBuffer<data=!IODDRData5, sparsity_map=!IODDRSM5>

!IODDRSM6 = memref<1x512x9x17xi1, #NHWC, @DDR>
!IODDRSparse6 = !VPUIP.SparseBuffer<data=memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=!IODDRSM6>

// CHECK-LABEL: @ParallelDDR2DDRCopyOutputWithSliceSparse
func.func @ParallelDDR2DDRCopyOutputWithSliceSparse(%arg0: !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>, %arg1: !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>) -> !VPUIP.SparseBuffer<data=!IODDRData4, sparsity_map=!IODDRSM4> {
    %0 = memref.alloc() : !IOCMXData0
    %1 = memref.alloc() : !IOCMXSM0
    %2 = VPUIP.GroupSparseBuffer(%0, %1) -> !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>

    %3 = memref.alloc() : !IODDRData5
    %4 = memref.alloc() : !IODDRSM6
    %5 = VPUIP.GroupSparseBuffer(%3, %4) -> !VPUIP.SparseBuffer<data=!IODDRData5, sparsity_map=!IODDRSM6>

    %6 = VPUIP.NCEClusterTiling inputs(%2 as %arg2: !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>) outputs(%5 as %arg3: !IODDRSparse2) -> !VPUIP.SparseBuffer<data=!IODDRData5, sparsity_map=!IODDRSM6> {
      %20 = VPUIP.Copy inputs(%arg2 : !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>) outputs(%arg3 : !IODDRSparse2) -> !IODDRSparse2
    }
    %7 = VPUIP.SubView %6 [0, 0, 8, 0] [1, 512, 1, 17] : !VPUIP.SparseBuffer<data=!IODDRData5, sparsity_map=!IODDRSM6> to !VPUIP.SparseBuffer<data=memref<1x512x1x17x!quant.uniform<u8:f16, 5.7832517137714463:123>, {order = #NHWC, strides = [78336, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x1x17xi1, {order = #NHWC, strides = [78336, 1, 8704, 512]}, @DDR>>
    %8 = memref.alloc() : !IODDRData1
    %9 = memref.alloc() : !IODDRSM3
    %10 = VPUIP.GroupSparseBuffer(%8, %9) -> !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM3>

    %11 = VPUIP.Copy inputs(%7 : !VPUIP.SparseBuffer<data=memref<1x512x1x17x!quant.uniform<u8:f16, 5.7832517137714463:123>, {order = #NHWC, strides = [78336, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x1x17xi1, {order = #NHWC, strides = [78336, 1, 8704, 512]}, @DDR>>) outputs(%10 : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM3>) -> !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM3>
    %12 = memref.alloc() : !IODDRData4
    %13 = memref.alloc() : !IODDRSM4
    %14 = VPUIP.GroupSparseBuffer(%12, %13) -> !VPUIP.SparseBuffer<data=!IODDRData4, sparsity_map=!IODDRSM4>

    %15 = VPUIP.SubView %14 [0, 0, 0, 0] [1, 512, 9, 17] : !VPUIP.SparseBuffer<data=!IODDRData4, sparsity_map=!IODDRSM4> to !VPUIP.SparseBuffer<data=memref<1x512x9x17x!quant.uniform<u8:f16, 5.7832517137714463:123>, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x9x17xi1, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>>
    %16 = VPUIP.Copy inputs(%6 : !VPUIP.SparseBuffer<data=!IODDRData5, sparsity_map=!IODDRSM6>) outputs(%15 : !VPUIP.SparseBuffer<data=memref<1x512x9x17x!quant.uniform<u8:f16, 5.7832517137714463:123>, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x9x17xi1, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>>) -> !VPUIP.SparseBuffer<data=memref<1x512x9x17x!quant.uniform<u8:f16, 5.7832517137714463:123>, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x9x17xi1, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>>
    %17 = VPUIP.SubView %14 [0, 0, 9, 0] [1, 512, 1, 17] : !VPUIP.SparseBuffer<data=!IODDRData4, sparsity_map=!IODDRSM4> to !VPUIP.SparseBuffer<data=memref<1x512x1x17x!quant.uniform<u8:f16, 5.7832517137714463:123>, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x1x17xi1, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>>
    %18 = VPUIP.Copy inputs(%11 : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM3>) outputs(%17 : !VPUIP.SparseBuffer<data=memref<1x512x1x17x!quant.uniform<u8:f16, 5.7832517137714463:123>, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x1x17xi1, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>>) -> !VPUIP.SparseBuffer<data=memref<1x512x1x17x!quant.uniform<u8:f16, 5.7832517137714463:123>, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x1x17xi1, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>>
    %19 = VPUIP.ConcatView inputs(%16, %18 : !VPUIP.SparseBuffer<data=memref<1x512x9x17x!quant.uniform<u8:f16, 5.7832517137714463:123>, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x9x17xi1, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>>, !VPUIP.SparseBuffer<data=memref<1x512x1x17x!quant.uniform<u8:f16, 5.7832517137714463:123>, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x1x17xi1, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>>) outputs(%14 : !VPUIP.SparseBuffer<data=!IODDRData4, sparsity_map=!IODDRSM4>) -> !VPUIP.SparseBuffer<data=!IODDRData4, sparsity_map=!IODDRSM4>
    return %19 : !VPUIP.SparseBuffer<data=!IODDRData4, sparsity_map=!IODDRSM4>

    // CHECK:       [[BUFF_0_DATA:%.+]] = memref.alloc() : memref<1x512x9x17x!qElemType, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x512x9x17xi1, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_0:%.+]] = VPUIP.GroupSparseBuffer([[BUFF_0_DATA]], [[BUFF_0_SM]])
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x512x9x17x!qElemType, #NHWC, @CMX_NN>, sparsity_map=memref<1x512x9x17xi1, #NHWC, @CMX_NN>>
      
    // CHECK:       [[BUFF_1_DATA:%.+]] = memref.alloc() : memref<1x512x10x17x!qElemType, #NHWC, @DDR>
    // CHECK:       [[BUFF_1_SM:%.+]] = memref.alloc() : memref<1x512x10x17xi1, #NHWC, @DDR>
    // CHECK:       [[BUFF_1:%.+]] = VPUIP.GroupSparseBuffer([[BUFF_1_DATA]], [[BUFF_1_SM]])
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x512x10x17x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x10x17xi1, #NHWC, @DDR>>
      
    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[BUFF_1]] [0, 0, 0, 0] [1, 512, 9, 17]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x512x10x17x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x10x17xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x9x17xi1, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>>
    // CHECK:       [[COPY_0:%.+]] = VPUIP.NCEClusterTiling inputs([[BUFF_0]] as %arg2: !VPUIP.SparseBuffer<data=memref<1x512x9x17x!qElemType, #NHWC, @CMX_NN>, sparsity_map=memref<1x512x9x17xi1, #NHWC, @CMX_NN>>) 
    // CHECK-SAME:         outputs([[SUBVIEW_0]] as %arg3: !VPUIP.SparseBuffer<data=memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x9x17xi1, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x9x17xi1, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>> {
    // CHECK:       [[inner_0:%.+]] = VPUIP.Copy inputs(%arg2 : !VPUIP.SparseBuffer<data=memref<1x512x9x17x!qElemType, #NHWC, @CMX_NN>, sparsity_map=memref<1x512x9x17xi1, #NHWC, @CMX_NN>>) 
    // CHECK-SAME:         outputs(%arg3 : !VPUIP.SparseBuffer<data=memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x9x17xi1, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x9x17xi1, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>>
      
    // CHECK:       }
    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[BUFF_1]] [0, 0, 9, 0] [1, 512, 1, 17]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x512x10x17x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x10x17xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x1x17xi1, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>>
    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 0, 8, 0] [1, 512, 1, 17]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x512x9x17x!qElemType, #NHWC, @CMX_NN>, sparsity_map=memref<1x512x9x17xi1, #NHWC, @CMX_NN>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [78336, 1, 8704, 512]}, @CMX_NN>, sparsity_map=memref<1x512x1x17xi1, {order = #NHWC, strides = [78336, 1, 8704, 512]}, @CMX_NN>>
    // CHECK:       [[COPY_1:%.+]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW_2]] as %arg2: !VPUIP.SparseBuffer<data=memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [78336, 1, 8704, 512]}, @CMX_NN>, sparsity_map=memref<1x512x1x17xi1, {order = #NHWC, strides = [78336, 1, 8704, 512]}, @CMX_NN>>) 
    // CHECK-SAME:         outputs([[SUBVIEW_1]] as %arg3: !VPUIP.SparseBuffer<data=memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x1x17xi1, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x1x17xi1, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>> {
    // CHECK:       [[inner_1:%.+]] = VPUIP.Copy inputs(%arg2 : !VPUIP.SparseBuffer<data=memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [78336, 1, 8704, 512]}, @CMX_NN>, sparsity_map=memref<1x512x1x17xi1, {order = #NHWC, strides = [78336, 1, 8704, 512]}, @CMX_NN>>) 
    // CHECK-SAME:         outputs(%arg3 : !VPUIP.SparseBuffer<data=memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x1x17xi1, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x1x17xi1, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>>
      
    // CHECK:       }
    // CHECK:       [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]] : !VPUIP.SparseBuffer<data=memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x9x17xi1, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>>, !VPUIP.SparseBuffer<data=memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x1x17xi1, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>>) 
    // CHECK-SAME:         outputs([[BUFF_1]] : !VPUIP.SparseBuffer<data=memref<1x512x10x17x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x10x17xi1, #NHWC, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x512x10x17x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x10x17xi1, #NHWC, @DDR>>
    // CHECK:       return [[CONCATVIEW_0]]

}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 5.7832517137714463:123>

!IODDRData0 = memref<1x512x9x17x!qElemType, #NHWC>
!IODDRSM0 = memref<1x512x9x17xi1, #NHWC>
!IODDRSparse0 = !VPUIP.SparseBuffer<data=!IODDRData0, sparsity_map=!IODDRSM0>

!IODDRSparse1 = !VPUIP.SparseBuffer<data=memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x9x17xi1, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>>
!IODDRData2 = memref<1x512x10x17x!qElemType, #NHWC, @DDR>
!IODDRSM2 = memref<1x512x10x17xi1, #NHWC, @DDR>
!IODDRSparse2 = !VPUIP.SparseBuffer<data=!IODDRData2, sparsity_map=!IODDRSM2>

!IODDRSparse3 = !VPUIP.SparseBuffer<data=memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x1x17xi1, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>>
!IOCMXData0 = memref<1x512x9x17x!qElemType, #NHWC, @CMX_NN>
!IOCMXSM0 = memref<1x512x9x17xi1, #NHWC, @CMX_NN>
!IOCMXSparse0 = !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>

!IODDRData4 = memref<1x512x9x17x!qElemType, #NHWC, @DDR>
!IODDRSM4 = memref<1x512x9x17xi1, #NHWC, @DDR>
!IODDRSparse4 = !VPUIP.SparseBuffer<data=!IODDRData4, sparsity_map=!IODDRSM4>

!IODDRSparse5 = !VPUIP.SparseBuffer<data=memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [78336, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x1x17xi1, {order = #NHWC, strides = [78336, 1, 8704, 512]}, @DDR>>

// CHECK-LABEL: @ParallelDDR2DDRCopyOutputWithSubviewSparse
func.func @ParallelDDR2DDRCopyOutputWithSubviewSparse(%arg0: !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>, %arg1: !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>) -> !VPUIP.SparseBuffer<data=!IODDRData2, sparsity_map=!IODDRSM2> {
    %0 = memref.alloc() : !IOCMXData0
    %1 = memref.alloc() : !IOCMXSM0
    %2 = VPUIP.GroupSparseBuffer(%0, %1) -> !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>

    %3 = memref.alloc() : !IODDRData4
    %4 = memref.alloc() : !IODDRSM4
    %5 = VPUIP.GroupSparseBuffer(%3, %4) -> !VPUIP.SparseBuffer<data=!IODDRData4, sparsity_map=!IODDRSM4>

    %6 = VPUIP.NCEClusterTiling inputs(%2 as %arg2: !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>) outputs(%5 as %arg3: !VPUIP.SparseBuffer<data=!IODDRData0, sparsity_map=!IODDRSM0>) -> !VPUIP.SparseBuffer<data=!IODDRData4, sparsity_map=!IODDRSM4> {
      %16 = VPUIP.Copy inputs(%arg2 : !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>) outputs(%arg3 : !VPUIP.SparseBuffer<data=!IODDRData0, sparsity_map=!IODDRSM0>) -> !VPUIP.SparseBuffer<data=!IODDRData0, sparsity_map=!IODDRSM0>
    }
    %7 = VPUIP.SubView %6 [0, 0, 8, 0] [1, 512, 1, 17] : !VPUIP.SparseBuffer<data=!IODDRData4, sparsity_map=!IODDRSM4> to !IODDRSparse5
    %8 = memref.alloc() : !IODDRData2
    %9 = memref.alloc() : !IODDRSM2
    %10 = VPUIP.GroupSparseBuffer(%8, %9) -> !VPUIP.SparseBuffer<data=!IODDRData2, sparsity_map=!IODDRSM2>

    %11 = VPUIP.SubView %10 [0, 0, 0, 0] [1, 512, 9, 17] : !VPUIP.SparseBuffer<data=!IODDRData2, sparsity_map=!IODDRSM2> to !IODDRSparse1
    %12 = VPUIP.Copy inputs(%6 : !VPUIP.SparseBuffer<data=!IODDRData4, sparsity_map=!IODDRSM4>) outputs(%11 : !IODDRSparse1) -> !IODDRSparse1
    %13 = VPUIP.SubView %10 [0, 0, 9, 0] [1, 512, 1, 17] : !VPUIP.SparseBuffer<data=!IODDRData2, sparsity_map=!IODDRSM2> to !IODDRSparse3
    %14 = VPUIP.Copy inputs(%7 : !IODDRSparse5) outputs(%13 : !IODDRSparse3) -> !IODDRSparse3
    %15 = VPUIP.ConcatView inputs(%12, %14 : !IODDRSparse1, !IODDRSparse3) outputs(%10 : !VPUIP.SparseBuffer<data=!IODDRData2, sparsity_map=!IODDRSM2>) -> !VPUIP.SparseBuffer<data=!IODDRData2, sparsity_map=!IODDRSM2>
    return %15 : !VPUIP.SparseBuffer<data=!IODDRData2, sparsity_map=!IODDRSM2>

    // CHECK:       [[BUFF_0_DATA:%.+]] = memref.alloc() : memref<1x512x9x17x!qElemType, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x512x9x17xi1, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_0:%.+]] = VPUIP.GroupSparseBuffer([[BUFF_0_DATA]], [[BUFF_0_SM]])
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x512x9x17x!qElemType, #NHWC, @CMX_NN>, sparsity_map=memref<1x512x9x17xi1, #NHWC, @CMX_NN>>
      
    // CHECK:       [[BUFF_1_DATA:%.+]] = memref.alloc() : memref<1x512x10x17x!qElemType, #NHWC, @DDR>
    // CHECK:       [[BUFF_1_SM:%.+]] = memref.alloc() : memref<1x512x10x17xi1, #NHWC, @DDR>
    // CHECK:       [[BUFF_1:%.+]] = VPUIP.GroupSparseBuffer([[BUFF_1_DATA]], [[BUFF_1_SM]])
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x512x10x17x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x10x17xi1, #NHWC, @DDR>>
      
    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[BUFF_1]] [0, 0, 0, 0] [1, 512, 9, 17]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x512x10x17x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x10x17xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x9x17xi1, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>>
    // CHECK:       [[COPY_0:%.+]] = VPUIP.NCEClusterTiling inputs([[BUFF_0]] as %arg2: !VPUIP.SparseBuffer<data=memref<1x512x9x17x!qElemType, #NHWC, @CMX_NN>, sparsity_map=memref<1x512x9x17xi1, #NHWC, @CMX_NN>>) 
    // CHECK-SAME:         outputs([[SUBVIEW_0]] as %arg3: !VPUIP.SparseBuffer<data=memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x9x17xi1, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x9x17xi1, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>> {
    // CHECK:       [[inner_0:%.+]] = VPUIP.Copy inputs(%arg2 : !VPUIP.SparseBuffer<data=memref<1x512x9x17x!qElemType, #NHWC, @CMX_NN>, sparsity_map=memref<1x512x9x17xi1, #NHWC, @CMX_NN>>) 
    // CHECK-SAME:         outputs(%arg3 : !VPUIP.SparseBuffer<data=memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x9x17xi1, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x9x17xi1, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>>
    
    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[BUFF_1]] [0, 0, 9, 0] [1, 512, 1, 17]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x512x10x17x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x10x17xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x1x17xi1, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>>
    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 0, 8, 0] [1, 512, 1, 17]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x512x9x17x!qElemType, #NHWC, @CMX_NN>, sparsity_map=memref<1x512x9x17xi1, #NHWC, @CMX_NN>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [78336, 1, 8704, 512]}, @CMX_NN>, sparsity_map=memref<1x512x1x17xi1, {order = #NHWC, strides = [78336, 1, 8704, 512]}, @CMX_NN>>
    // CHECK:       [[COPY_1:%.+]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW_2]] as %arg2: !VPUIP.SparseBuffer<data=memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [78336, 1, 8704, 512]}, @CMX_NN>, sparsity_map=memref<1x512x1x17xi1, {order = #NHWC, strides = [78336, 1, 8704, 512]}, @CMX_NN>>) 
    // CHECK-SAME:         outputs([[SUBVIEW_1]] as %arg3: !VPUIP.SparseBuffer<data=memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x1x17xi1, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x1x17xi1, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>> {
    // CHECK:       [[inner_1:%.+]] = VPUIP.Copy inputs(%arg2 : !VPUIP.SparseBuffer<data=memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [78336, 1, 8704, 512]}, @CMX_NN>, sparsity_map=memref<1x512x1x17xi1, {order = #NHWC, strides = [78336, 1, 8704, 512]}, @CMX_NN>>) 
    // CHECK-SAME:         outputs(%arg3 : !VPUIP.SparseBuffer<data=memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x1x17xi1, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x1x17xi1, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>>
      
    // CHECK:       [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]] : !VPUIP.SparseBuffer<data=memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x9x17xi1, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>>, !VPUIP.SparseBuffer<data=memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, sparsity_map=memref<1x512x1x17xi1, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>>) 
    // CHECK-SAME:         outputs([[BUFF_1]] : !VPUIP.SparseBuffer<data=memref<1x512x10x17x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x10x17xi1, #NHWC, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x512x10x17x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x512x10x17xi1, #NHWC, @DDR>>
    // CHECK:       return [[CONCATVIEW_0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IODDRSparse0 = !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, #NHWC, @DDR>, sparsity_map=memref<1x144x64x128xi1, #NHWC, @DDR>>
!IODistrCMXData0 = !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
!IODistrCMXSM0 = !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
!IODistrCMXSparse0 = !VPUIP.SparseBuffer<data=!IODistrCMXData0, sparsity_map=!IODistrCMXSM0>

!IODDRData1 = memref<1x144x64x128xf16, #NHWC>
!IODDRSM1 = memref<1x144x64x128xi1, #NHWC>
!IODDRSparse1 = !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>

!IOCMXData0 = memref<1x144x64x128xf16, #NHWC, @CMX_NN>
!IOCMXSM0 = memref<1x144x64x128xi1, #NHWC, @CMX_NN>
!IOCMXSparse0 = !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>


func.func @NCEClusterCopyOpSequenceSparse() -> !IODistrCMXSparse0 {
    %0 = VPURT.AllocDistributed -> !IODistrCMXData0
    %1 = VPURT.AllocDistributed -> !IODistrCMXSM0
    %2 = VPUIP.GroupSparseBuffer(%0, %1) -> !IODistrCMXSparse0

    %3 = memref.alloc() : memref<1x144x64x128xf16, #NHWC, @DDR>
    %4 = memref.alloc() : memref<1x144x64x128xi1, #NHWC, @DDR>
    %5 = VPUIP.GroupSparseBuffer(%3, %4) -> !IODDRSparse0

    %6 = VPUIP.NCEClusterTiling inputs(%2 as %arg0: !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>) outputs(%5 as %arg1: !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) -> !IODDRSparse0 {
      %11 = VPUIP.Copy inputs(%arg0 : !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>) outputs(%arg1 : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) -> !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>
    }
    %7 = VPURT.AllocDistributed -> !IODistrCMXData0
    %8 = VPURT.AllocDistributed -> !IODistrCMXSM0
    %9 = VPUIP.GroupSparseBuffer(%7, %8) -> !IODistrCMXSparse0

    %10 = VPUIP.NCEClusterTiling inputs(%6 as %arg0: !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) outputs(%9 as %arg1: !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>) -> !IODistrCMXSparse0 {
      %11 = VPUIP.Copy inputs(%arg0 : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) outputs(%arg1 : !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>) -> !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>
    }
    return %10 : !IODistrCMXSparse0


    // CHECK:       [[BUFF_0_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_0_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_0:%.+]] = VPUIP.GroupSparseBuffer([[BUFF_0_DATA]], [[BUFF_0_SM]])
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>>
    
    // CHECK-NOT:   memref.alloc()
    // CHECK-NOT:   VPUIP.NCEClusterTiling
    // CHECK-NOT:   VPURT.AllocDistributed
    // CHECK-NOT:   VPUIP.NCEClusterTiling

    // CHECK:       return [[BUFF_0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IODistrCMXSparse0 = !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>>
!IODistrCMXData1 = !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
!IODistrCMXSM1 = !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
!IODistrCMXSparse1 = !VPUIP.SparseBuffer<data=!IODistrCMXData1, sparsity_map=!IODistrCMXSM1>

!IOCMXData0 = memref<1x144x64x128xf16, #NHWC, @CMX_NN>
!IOCMXSM0 = memref<1x144x64x128xi1, #NHWC, @CMX_NN>
!IOCMXSparse0 = !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>

!IODDRData0 = memref<1x144x64x128xf16, #NHWC>
!IODDRSM0 = memref<1x144x64x128xi1, #NHWC>
!IODDRSparse0 = !VPUIP.SparseBuffer<data=!IODDRData0, sparsity_map=!IODDRSM0>

!IODDRSparse1 = !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, #NHWC, @DDR>, sparsity_map=memref<1x144x64x128xi1, #NHWC, @DDR>>

func.func @NCEClusterCopyOpSequenceWithCastSparse() -> !IODistrCMXSparse1 {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %2 = VPUIP.GroupSparseBuffer(%0, %1) -> !IODistrCMXSparse0

    %3 = memref.alloc() : memref<1x144x64x128xf16, #NHWC, @DDR>
    %4 = memref.alloc() : memref<1x144x64x128xi1, #NHWC, @DDR>
    %5 = VPUIP.GroupSparseBuffer(%3, %4) -> !IODDRSparse1

    %6 = VPUIP.NCEClusterTiling inputs(%2 as %arg0: !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>) outputs(%5 as %arg1: !VPUIP.SparseBuffer<data=!IODDRData0, sparsity_map=!IODDRSM0>) -> !IODDRSparse1 {
      %11 = VPUIP.Copy inputs(%arg0 : !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>) outputs(%arg1 : !VPUIP.SparseBuffer<data=!IODDRData0, sparsity_map=!IODDRSM0>) -> !VPUIP.SparseBuffer<data=!IODDRData0, sparsity_map=!IODDRSM0>
    }
    %7 = VPURT.AllocDistributed -> !IODistrCMXData1
    %8 = VPURT.AllocDistributed -> !IODistrCMXSM1
    %9 = VPUIP.GroupSparseBuffer(%7, %8) -> !IODistrCMXSparse1

    %10 = VPUIP.NCEClusterTiling inputs(%6 as %arg0: !VPUIP.SparseBuffer<data=!IODDRData0, sparsity_map=!IODDRSM0>) outputs(%9 as %arg1: !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>) -> !IODistrCMXSparse1 {
      %11 = VPUIP.Copy inputs(%arg0 : !VPUIP.SparseBuffer<data=!IODDRData0, sparsity_map=!IODDRSM0>) outputs(%arg1 : !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>) -> !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>
    }
    return %10 : !IODistrCMXSparse1


    // CHECK:       [[BUFF_0_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_0_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_0:%.+]] = VPUIP.GroupSparseBuffer([[BUFF_0_DATA]], [[BUFF_0_SM]])
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>>
    
    // CHECK-NOT:   memref.alloc()
    // CHECK-NOT:   VPUIP.NCEClusterTiling
    // CHECK-NOT:   VPURT.AllocDistributed
    // CHECK-NOT:   VPUIP.NCEClusterTiling

    // CHECK:       [[DISTRIBUTEDCAST_0:%.+]] = VPUIP.DistributedCast inputs([[BUFF_0]] : !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>>) -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>>
    // CHECK:       return [[DISTRIBUTEDCAST_0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IODDRSparse0 = !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, #NHWC, @DDR>, sparsity_map=memref<1x144x64x128xi1, #NHWC, @DDR>>
!IODDRData1 = memref<1x144x64x128xf16, #NHWC>
!IODDRSM1 = memref<1x144x64x128xi1, #NHWC>
!IODDRSparse1 = !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>

!IODistrCMXSparse0 = !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>>
!IODistrDDRData0 = !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @DDR, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
!IODistrDDRSM0 = !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @DDR, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
!IODistrDDRSparse0 = !VPUIP.SparseBuffer<data=!IODistrDDRData0, sparsity_map=!IODistrDDRSM0>

!IOCMXData0 = memref<1x144x64x128xf16, #NHWC, @CMX_NN>
!IOCMXSM0 = memref<1x144x64x128xi1, #NHWC, @CMX_NN>
!IOCMXSparse0 = !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>


func.func @NCEClusterCopyOpSequenceNoChangeSparse() -> !IODistrDDRSparse0 {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %2 = VPUIP.GroupSparseBuffer(%0, %1) -> !IODistrCMXSparse0

    %3 = memref.alloc() : memref<1x144x64x128xf16, #NHWC, @DDR>
    %4 = memref.alloc() : memref<1x144x64x128xi1, #NHWC, @DDR>
    %5 = VPUIP.GroupSparseBuffer(%3, %4) -> !IODDRSparse0

    %6 = VPUIP.NCEClusterTiling inputs(%2 as %arg0: !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>) outputs(%5 as %arg1: !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) -> !IODDRSparse0 {
      %11 = VPUIP.Copy inputs(%arg0 : !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>) outputs(%arg1 : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) -> !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>
    }
    %7 = VPURT.AllocDistributed -> !IODistrDDRData0
    %8 = VPURT.AllocDistributed -> !IODistrDDRSM0
    %9 = VPUIP.GroupSparseBuffer(%7, %8) -> !IODistrDDRSparse0

    %10 = VPUIP.NCEClusterTiling inputs(%6 as %arg0: !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) outputs(%9 as %arg1: !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>) -> !IODistrDDRSparse0 {
      %11 = VPUIP.Copy inputs(%arg0 : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) outputs(%arg1 : !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>) -> !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>
    }
    return %10 : !IODistrDDRSparse0


    // CHECK:       [[BUFF_0_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_0_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_0:%.+]] = VPUIP.GroupSparseBuffer([[BUFF_0_DATA]], [[BUFF_0_SM]])
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>>
      
    // CHECK:       [[BUFF_1_DATA:%.+]] = memref.alloc() : memref<1x144x64x128xf16, #NHWC, @DDR>
    // CHECK:       [[BUFF_1_SM:%.+]] = memref.alloc() : memref<1x144x64x128xi1, #NHWC, @DDR>
    // CHECK:       [[BUFF_1:%.+]] = VPUIP.GroupSparseBuffer([[BUFF_1_DATA]], [[BUFF_1_SM]])
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, #NHWC, @DDR>, sparsity_map=memref<1x144x64x128xi1, #NHWC, @DDR>>
      
    // CHECK:       [[NCECLUSTERTILING_0:%.+]] = VPUIP.NCEClusterTiling inputs([[BUFF_0]] as %arg0: !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x144x64x128xi1, #NHWC, @CMX_NN>>) 
    // CHECK-SAME:         outputs([[BUFF_1]] as %arg1: !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, #NHWC>, sparsity_map=memref<1x144x64x128xi1, #NHWC>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, #NHWC, @DDR>, sparsity_map=memref<1x144x64x128xi1, #NHWC, @DDR>> {
    // CHECK:       [[COPY_0:%.+]] = VPUIP.Copy inputs(%arg0 : !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x144x64x128xi1, #NHWC, @CMX_NN>>) 
    // CHECK-SAME:         outputs(%arg1 : !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, #NHWC>, sparsity_map=memref<1x144x64x128xi1, #NHWC>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, #NHWC>, sparsity_map=memref<1x144x64x128xi1, #NHWC>>
    // CHECK:       }
    // CHECK:       [[BUFF_2_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @DDR, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_2_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @DDR, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_2:%.+]] = VPUIP.GroupSparseBuffer([[BUFF_2_DATA]], [[BUFF_2_SM]])
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @DDR, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @DDR, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>>
      
    // CHECK:       [[NCECLUSTERTILING_1:%.+]] = VPUIP.NCEClusterTiling inputs([[NCECLUSTERTILING_0]] as %arg0: !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, #NHWC>, sparsity_map=memref<1x144x64x128xi1, #NHWC>>) 
    // CHECK-SAME:         outputs([[BUFF_2]] as %arg1: !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x144x64x128xi1, #NHWC, @CMX_NN>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @DDR, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @DDR, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>> {
    // CHECK:       [[COPY_1:%.+]] = VPUIP.Copy inputs(%arg0 : !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, #NHWC>, sparsity_map=memref<1x144x64x128xi1, #NHWC>>) 
    // CHECK-SAME:         outputs(%arg1 : !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x144x64x128xi1, #NHWC, @CMX_NN>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x144x64x128xi1, #NHWC, @CMX_NN>>
    // CHECK:       }
    // CHECK:       return [[NCECLUSTERTILING_1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IODistrCMXSparse0 = !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>>
!IODDRSparse0 = !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, #NHWC>, sparsity_map=memref<1x144x64x128xi1, #NHWC>>
!IOCMXSparse0 = !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x144x64x128xi1, #NHWC, @CMX_NN>>
!IODDRData1 = memref<1x144x64x128xf16, #NHWC, @DDR>
!IODDRSM1 = memref<1x144x64x128xi1, #NHWC, @DDR>
!IODDRSparse1 = !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>


func.func @NCEClusterCopyOpDDRCopy() -> !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %2 = VPUIP.GroupSparseBuffer(%0, %1) -> !IODistrCMXSparse0

    %3 = memref.alloc() : !IODDRData1
    %4 = memref.alloc() : !IODDRSM1
    %5 = VPUIP.GroupSparseBuffer(%3, %4) -> !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>

    %6 = VPUIP.NCEClusterTiling inputs(%2 as %arg0: !IOCMXSparse0) outputs(%5 as %arg1: !IODDRSparse0) -> !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1> {
      %11 = VPUIP.Copy inputs(%arg0 : !IOCMXSparse0) outputs(%arg1 : !IODDRSparse0) -> !IODDRSparse0
    }
    %7 = memref.alloc() : !IODDRData1
    %8 = memref.alloc() : !IODDRSM1
    %9 = VPUIP.GroupSparseBuffer(%7, %8) -> !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>

    %10 = VPUIP.Copy inputs(%6 : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) outputs(%9 : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) -> !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>
    return %10 : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>


    // CHECK:       [[BUFF_0_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_0_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_0:%.+]] = VPUIP.GroupSparseBuffer([[BUFF_0_DATA]], [[BUFF_0_SM]])
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>>
      
    // CHECK:       [[BUFF_1_DATA:%.+]] = memref.alloc() : memref<1x144x64x128xf16, #NHWC, @DDR>
    // CHECK:       [[BUFF_1_SM:%.+]] = memref.alloc() : memref<1x144x64x128xi1, #NHWC, @DDR>
    // CHECK:       [[BUFF_1:%.+]] = VPUIP.GroupSparseBuffer([[BUFF_1_DATA]], [[BUFF_1_SM]])
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, #NHWC, @DDR>, sparsity_map=memref<1x144x64x128xi1, #NHWC, @DDR>>
      
    // CHECK:       [[NCECLUSTERTILING_0:%.+]] = VPUIP.NCEClusterTiling inputs([[BUFF_0]] as %arg0: !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x144x64x128xi1, #NHWC, @CMX_NN>>) 
    // CHECK-SAME:         outputs([[BUFF_1]] as %arg1: !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, #NHWC, @DDR>, sparsity_map=memref<1x144x64x128xi1, #NHWC, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, #NHWC, @DDR>, sparsity_map=memref<1x144x64x128xi1, #NHWC, @DDR>> {
    // CHECK:       [[COPY_0:%.+]] = VPUIP.Copy inputs(%arg0 : !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x144x64x128xi1, #NHWC, @CMX_NN>>) 
    // CHECK-SAME:         outputs(%arg1 : !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, #NHWC, @DDR>, sparsity_map=memref<1x144x64x128xi1, #NHWC, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, #NHWC, @DDR>, sparsity_map=memref<1x144x64x128xi1, #NHWC, @DDR>>
    // CHECK:       }
    // CHECK:       return [[NCECLUSTERTILING_0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IOCMXData0 = memref<1x16x128x128xf16, #NHWC, @CMX_NN>
!IOCMXSM0 = memref<1x16x128x128xi1, #NHWC, @CMX_NN>
!IOCMXSparse0 = !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>

!IODDRSparse0 = !VPUIP.SparseBuffer<data=memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>, sparsity_map=memref<1x13x128x128xi1, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>>
!IODDRSparse1 = !VPUIP.SparseBuffer<data=memref<1x3x128x128xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x128x128xi1, #NHWC, @DDR>>
!IODDRData2 = memref<1x16x128x128xf16, #NHWC, @DDR>
!IODDRSM2 = memref<1x16x128x128xi1, #NHWC, @DDR>
!IODDRSparse2 = !VPUIP.SparseBuffer<data=!IODDRData2, sparsity_map=!IODDRSM2>

!IODDRSparse3 = !VPUIP.SparseBuffer<data=memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>, sparsity_map=memref<1x3x128x128xi1, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>>
!IODDRSparse4 = !VPUIP.SparseBuffer<data=memref<1x13x128x128xf16, #NHWC>, sparsity_map=memref<1x13x128x128xi1, #NHWC>>

// CHECK-LABEL: @DDRToCMXCopyWithConcatViewWithCopySparse
func.func @DDRToCMXCopyWithConcatViewWithCopySparse(%arg0: !IODDRSparse4, %arg1: !IODDRSparse1) -> !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0> {
    %0 = memref.alloc() : !IODDRData2
    %1 = memref.alloc() : !IODDRSM2
    %2 = VPUIP.GroupSparseBuffer(%0, %1) -> !VPUIP.SparseBuffer<data=!IODDRData2, sparsity_map=!IODDRSM2>

    %3 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 3, 128, 128] : !VPUIP.SparseBuffer<data=!IODDRData2, sparsity_map=!IODDRSM2> to !IODDRSparse3
    %4 = VPUIP.Copy inputs(%arg1 : !IODDRSparse1) outputs(%3 : !IODDRSparse3) -> !IODDRSparse3
    %5 = VPUIP.SubView %2 [0, 3, 0, 0] [1, 13, 128, 128] : !VPUIP.SparseBuffer<data=!IODDRData2, sparsity_map=!IODDRSM2> to !IODDRSparse0
    %6 = VPUIP.Copy inputs(%arg0 : !IODDRSparse4) outputs(%5 : !IODDRSparse0) -> !IODDRSparse0
    %7 = VPUIP.ConcatView inputs(%4, %6 : !IODDRSparse3, !IODDRSparse0) outputs(%2 : !VPUIP.SparseBuffer<data=!IODDRData2, sparsity_map=!IODDRSM2>) -> !VPUIP.SparseBuffer<data=!IODDRData2, sparsity_map=!IODDRSM2>
    %8 = memref.alloc() : !IOCMXData0
    %9 = memref.alloc() : !IOCMXSM0
    %10 = VPUIP.GroupSparseBuffer(%8, %9) -> !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>

    %11 = VPUIP.Copy inputs(%7 : !VPUIP.SparseBuffer<data=!IODDRData2, sparsity_map=!IODDRSM2>) outputs(%10 : !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>) -> !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>
    return %11 : !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>

    // CHECK:       [[BUFF_0_DATA:%.+]] = memref.alloc() : memref<1x16x128x128xf16, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x16x128x128xi1, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_0:%.+]] = VPUIP.GroupSparseBuffer([[BUFF_0_DATA]], [[BUFF_0_SM]])
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x16x128x128xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x16x128x128xi1, #NHWC, @CMX_NN>>
      
    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 0] [1, 3, 128, 128]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x16x128x128xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x16x128x128xi1, #NHWC, @CMX_NN>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>, sparsity_map=memref<1x3x128x128xi1, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>>
    // CHECK:       [[COPY_0:%.+]] = VPUIP.Copy inputs(%arg1 : !VPUIP.SparseBuffer<data=memref<1x3x128x128xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x128x128xi1, #NHWC, @DDR>>) 
    // CHECK-SAME:         outputs([[SUBVIEW_0]] : !VPUIP.SparseBuffer<data=memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>, sparsity_map=memref<1x3x128x128xi1, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>, sparsity_map=memref<1x3x128x128xi1, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>>
    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 3, 0, 0] [1, 13, 128, 128]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x16x128x128xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x16x128x128xi1, #NHWC, @CMX_NN>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>, sparsity_map=memref<1x13x128x128xi1, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>>
    // CHECK:       [[COPY_1:%.+]] = VPUIP.Copy inputs(%arg0 : !VPUIP.SparseBuffer<data=memref<1x13x128x128xf16, #NHWC>, sparsity_map=memref<1x13x128x128xi1, #NHWC>>) 
    // CHECK-SAME:         outputs([[SUBVIEW_1]] : !VPUIP.SparseBuffer<data=memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>, sparsity_map=memref<1x13x128x128xi1, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>, sparsity_map=memref<1x13x128x128xi1, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>>
    // CHECK:       [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]] : !VPUIP.SparseBuffer<data=memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>, sparsity_map=memref<1x3x128x128xi1, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>>, !VPUIP.SparseBuffer<data=memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>, sparsity_map=memref<1x13x128x128xi1, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>>) 
    // CHECK-SAME:         outputs([[BUFF_0]] : !VPUIP.SparseBuffer<data=memref<1x16x128x128xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x16x128x128xi1, #NHWC, @CMX_NN>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x16x128x128xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x16x128x128xi1, #NHWC, @CMX_NN>>
    // CHECK:       return [[CONCATVIEW_0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IOCMXData0 = memref<1x16x128x128xf16, #NHWC, @CMX_NN>
!IOCMXSM0 = memref<1x16x128x128xi1, #NHWC, @CMX_NN>
!IOCMXSparse0 = !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>

!IODDRData0 = memref<1x16x128x128xf16, #NHWC, @DDR>
!IODDRSM0 = memref<1x16x128x128xi1, #NHWC, @DDR>
!IODDRSparse0 = !VPUIP.SparseBuffer<data=!IODDRData0, sparsity_map=!IODDRSM0>

!IODDRSparse1 = !VPUIP.SparseBuffer<data=memref<1x3x128x128xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x128x128xi1, #NHWC, @DDR>>
!IODDRSparse2 = !VPUIP.SparseBuffer<data=memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>, sparsity_map=memref<1x3x128x128xi1, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>>
!IODDRSparse3 = !VPUIP.SparseBuffer<data=memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>, sparsity_map=memref<1x13x128x128xi1, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>>
!IODDRSparse4 = !VPUIP.SparseBuffer<data=memref<1x13x128x128xf16, #NHWC>, sparsity_map=memref<1x13x128x128xi1, #NHWC>>

// CHECK-LABEL: @DDRToCMXCopyWithConcatViewWithMultiCopySparse
func.func @DDRToCMXCopyWithConcatViewWithMultiCopySparse(%arg0: !IODDRSparse4, %arg1: !IODDRSparse1) -> (!VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>, !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>) {
    %0 = memref.alloc() : !IODDRData0
    %1 = memref.alloc() : !IODDRSM0
    %2 = VPUIP.GroupSparseBuffer(%0, %1) -> !VPUIP.SparseBuffer<data=!IODDRData0, sparsity_map=!IODDRSM0>

    %3 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 3, 128, 128] : !VPUIP.SparseBuffer<data=!IODDRData0, sparsity_map=!IODDRSM0> to !IODDRSparse2
    %4 = VPUIP.Copy inputs(%arg1 : !IODDRSparse1) outputs(%3 : !IODDRSparse2) -> !IODDRSparse2
    %5 = VPUIP.SubView %2 [0, 3, 0, 0] [1, 13, 128, 128] : !VPUIP.SparseBuffer<data=!IODDRData0, sparsity_map=!IODDRSM0> to !IODDRSparse3
    %6 = VPUIP.Copy inputs(%arg0 : !IODDRSparse4) outputs(%5 : !IODDRSparse3) -> !IODDRSparse3
    %7 = VPUIP.ConcatView inputs(%4, %6 : !IODDRSparse2, !IODDRSparse3) outputs(%2 : !VPUIP.SparseBuffer<data=!IODDRData0, sparsity_map=!IODDRSM0>) -> !VPUIP.SparseBuffer<data=!IODDRData0, sparsity_map=!IODDRSM0>
    %8 = memref.alloc() : !IOCMXData0
    %9 = memref.alloc() : !IOCMXSM0
    %10 = VPUIP.GroupSparseBuffer(%8, %9) -> !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>

    %11 = VPUIP.Copy inputs(%7 : !VPUIP.SparseBuffer<data=!IODDRData0, sparsity_map=!IODDRSM0>) outputs(%10 : !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>) -> !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>
    %12 = memref.alloc() : !IOCMXData0
    %13 = memref.alloc() : !IOCMXSM0
    %14 = VPUIP.GroupSparseBuffer(%12, %13) -> !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>

    %15 = VPUIP.Copy inputs(%7 : !VPUIP.SparseBuffer<data=!IODDRData0, sparsity_map=!IODDRSM0>) outputs(%14 : !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>) -> !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>
    return %11, %15 : !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>, !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>

    // CHECK:       [[BUFF_0_DATA:%.+]] = memref.alloc() : memref<1x16x128x128xf16, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x16x128x128xi1, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_0:%.+]] = VPUIP.GroupSparseBuffer([[BUFF_0_DATA]], [[BUFF_0_SM]])
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x16x128x128xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x16x128x128xi1, #NHWC, @CMX_NN>>
      
    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 0] [1, 3, 128, 128]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x16x128x128xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x16x128x128xi1, #NHWC, @CMX_NN>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>, sparsity_map=memref<1x3x128x128xi1, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>>
    // CHECK:       [[COPY_0:%.+]] = VPUIP.Copy inputs(%arg1 : !VPUIP.SparseBuffer<data=memref<1x3x128x128xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x128x128xi1, #NHWC, @DDR>>) 
    // CHECK-SAME:         outputs([[SUBVIEW_0]] : !VPUIP.SparseBuffer<data=memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>, sparsity_map=memref<1x3x128x128xi1, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>, sparsity_map=memref<1x3x128x128xi1, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>>
    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 3, 0, 0] [1, 13, 128, 128]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x16x128x128xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x16x128x128xi1, #NHWC, @CMX_NN>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>, sparsity_map=memref<1x13x128x128xi1, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>>
    // CHECK:       [[COPY_1:%.+]] = VPUIP.Copy inputs(%arg0 : !VPUIP.SparseBuffer<data=memref<1x13x128x128xf16, #NHWC>, sparsity_map=memref<1x13x128x128xi1, #NHWC>>) 
    // CHECK-SAME:         outputs([[SUBVIEW_1]] : !VPUIP.SparseBuffer<data=memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>, sparsity_map=memref<1x13x128x128xi1, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>, sparsity_map=memref<1x13x128x128xi1, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>>
    // CHECK:       [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]] : !VPUIP.SparseBuffer<data=memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>, sparsity_map=memref<1x3x128x128xi1, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>>, !VPUIP.SparseBuffer<data=memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>, sparsity_map=memref<1x13x128x128xi1, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>>) 
    // CHECK-SAME:         outputs([[BUFF_0]] : !VPUIP.SparseBuffer<data=memref<1x16x128x128xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x16x128x128xi1, #NHWC, @CMX_NN>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x16x128x128xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x16x128x128xi1, #NHWC, @CMX_NN>>
    // CHECK:       return [[CONCATVIEW_0]], [[CONCATVIEW_0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IOCMXData0 = memref<1x16x128x128xf16, #NHWC, @CMX_NN>
!IOCMXSM0 = memref<1x16x128x128xi1, #NHWC, @CMX_NN>
!IOCMXSparse0 = !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>

!IODDRData0 = memref<1x16x128x128xf16, #NHWC>
!IODDRSM0 = memref<1x16x128x128xi1, #NHWC>
!IODDRSparse0 = !VPUIP.SparseBuffer<data=!IODDRData0, sparsity_map=!IODDRSM0>

!IODistrCMXData0 = !VPUIP.DistributedBuffer<
    1x16x128x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED", 
    num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
}>

!IODistrCMXSM0 = !VPUIP.DistributedBuffer<
    1x16x128x128xi1, #NHWC, @CMX_NN, {
    mode = "SEGMENTED", 
    num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
}>

!IODistrCMXSparse0 = !VPUIP.SparseBuffer<
    data=!IODistrCMXData0, sparsity_map=!IODistrCMXSM0>

!IODDRData1 = memref<1x16x128x128xf16, #NHWC, @DDR>
!IODDRSM1 = memref<1x16x128x128xi1, #NHWC, @DDR>
!IODDRSparse1 = !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>

!IODDRSparse2 = !VPUIP.SparseBuffer<data=memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>, sparsity_map=memref<1x13x128x128xi1, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>>
!IODDRSparse3 = !VPUIP.SparseBuffer<data=memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>, sparsity_map=memref<1x3x128x128xi1, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>>
!IODDRSparse4 = !VPUIP.SparseBuffer<data=memref<1x13x128x128xf16, #NHWC>, sparsity_map=memref<1x13x128x128xi1, #NHWC>>
!IODDRSparse5 = !VPUIP.SparseBuffer<data=memref<1x3x128x128xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x128x128xi1, #NHWC, @DDR>>

// CHECK-LABEL: @DDRToCMXCopyWithConcatViewWithClusterCopySparse
func.func @DDRToCMXCopyWithConcatViewWithClusterCopySparse(%arg0: !IODDRSparse4, %arg1: !IODDRSparse5) -> !IODistrCMXSparse0 {
    %0 = memref.alloc() : !IODDRData1
    %1 = memref.alloc() : !IODDRSM1
    %2 = VPUIP.GroupSparseBuffer(%0, %1) -> !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>

    %3 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 3, 128, 128] : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1> to !IODDRSparse3
    %4 = VPUIP.Copy inputs(%arg1 : !IODDRSparse5) outputs(%3 : !IODDRSparse3) -> !IODDRSparse3
    %5 = VPUIP.SubView %2 [0, 3, 0, 0] [1, 13, 128, 128] : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1> to !IODDRSparse2
    %6 = VPUIP.Copy inputs(%arg0 : !IODDRSparse4) outputs(%5 : !IODDRSparse2) -> !IODDRSparse2
    %7 = VPUIP.ConcatView inputs(%4, %6 : !IODDRSparse3, !IODDRSparse2) outputs(%2 : !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>) -> !VPUIP.SparseBuffer<data=!IODDRData1, sparsity_map=!IODDRSM1>
    %8 = VPURT.AllocDistributed -> !IODistrCMXData0
    %9 = VPURT.AllocDistributed -> !IODistrCMXSM0
    %10 = VPUIP.GroupSparseBuffer(%8, %9) -> !IODistrCMXSparse0

    %11 = VPUIP.NCEClusterTiling inputs(%7 as %arg2: !VPUIP.SparseBuffer<data=!IODDRData0, sparsity_map=!IODDRSM0>) outputs(%10 as %arg3: !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>) -> !IODistrCMXSparse0 {
      %12 = VPUIP.Copy inputs(%arg2 : !VPUIP.SparseBuffer<data=!IODDRData0, sparsity_map=!IODDRSM0>) outputs(%arg3 : !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>) -> !VPUIP.SparseBuffer<data=!IOCMXData0, sparsity_map=!IOCMXSM0>
    }
    return %11 : !IODistrCMXSparse0

    // CHECK:       [[BUFF_0_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_0_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x128x128xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_0:%.+]] = VPUIP.GroupSparseBuffer([[BUFF_0_DATA]], [[BUFF_0_SM]])
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x16x128x128xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>>
      
    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 0] [1, 3, 128, 128]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x16x128x128xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x3x128x128xi1, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>>
    // CHECK:       [[COPY_0:%.+]] = VPUIP.NCEClusterTiling inputs(%arg1 as %arg2: !VPUIP.SparseBuffer<data=memref<1x3x128x128xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x128x128xi1, #NHWC, @DDR>>) 
    // CHECK-SAME:         outputs([[SUBVIEW_0]] as %arg3: !VPUIP.SparseBuffer<data=memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>, sparsity_map=memref<1x3x128x128xi1, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x3x128x128xi1, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>> {
    // CHECK:       [[inner_0:%.+]] = VPUIP.Copy inputs(%arg2 : !VPUIP.SparseBuffer<data=memref<1x3x128x128xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x128x128xi1, #NHWC, @DDR>>) 
    // CHECK-SAME:         outputs(%arg3 : !VPUIP.SparseBuffer<data=memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>, sparsity_map=memref<1x3x128x128xi1, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>, sparsity_map=memref<1x3x128x128xi1, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>>
      
    // CHECK:       }
    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 3, 0, 0] [1, 13, 128, 128]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x16x128x128xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x13x128x128xi1, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>>
    // CHECK:       [[COPY_1:%.+]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg2: !VPUIP.SparseBuffer<data=memref<1x13x128x128xf16, #NHWC>, sparsity_map=memref<1x13x128x128xi1, #NHWC>>) 
    // CHECK-SAME:         outputs([[SUBVIEW_1]] as %arg3: !VPUIP.SparseBuffer<data=memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>, sparsity_map=memref<1x13x128x128xi1, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x13x128x128xi1, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>> {
    // CHECK:       [[inner_1:%.+]] = VPUIP.Copy inputs(%arg2 : !VPUIP.SparseBuffer<data=memref<1x13x128x128xf16, #NHWC>, sparsity_map=memref<1x13x128x128xi1, #NHWC>>) 
    // CHECK-SAME:         outputs(%arg3 : !VPUIP.SparseBuffer<data=memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>, sparsity_map=memref<1x13x128x128xi1, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>, sparsity_map=memref<1x13x128x128xi1, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>>
      
    // CHECK:       }
    // CHECK:       [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]] : !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x3x128x128xi1, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>>, !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x13x128x128xi1, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>>) 
    // CHECK-SAME:         outputs([[BUFF_0]] : !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x16x128x128xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x16x128x128xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>>
    // CHECK:       return [[CONCATVIEW_0]]
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InDataType = memref<1x256x28x28xf16, #NHWC, @CMX_NN>
!InSMType = memref<1x256x28x28xi1, #NHWC, @CMX_NN>
!ConvWeightsType = memref<128x256x3x3xf16, #NHWC, @CMX_NN>
!ConvWeightsTableType = memref<128x1x1x4xsi32, @CMX_NN>

!OutDataBufferType = !VPUIP.DistributedBuffer<1x256x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
!OutSMBufferType = !VPUIP.DistributedBuffer<1x256x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

!ResultSparseBufferType = !VPUIP.SparseBuffer<data=memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>, sparsity_map=memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>>

!ResultSparseDistrBufferType = !VPUIP.SparseBuffer<
    data=!VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    sparsity_map=!VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
>

// CopyOp is wrapped into ClusterTilingOp, types are distributed
func.func @CMX2CMXSparseDistributedTypeStrides (
    %inData : !InDataType,
    %inSparsityMap : !InSMType,
    %inWeights : !ConvWeightsType,
    %inWeightsTable : !ConvWeightsTableType)
    -> !ResultSparseDistrBufferType
{
    // alloc for Conv data out
    %0 = memref.alloc() : memref<1x128x14x14xf16, #NHWC, @CMX_NN>
    // alloc for Conv sparsity map out
    %1 = memref.alloc() : memref<1x128x14x14xi1, #NHWC, @CMX_NN>

    // Input 1: Convolution
    %2:2 = VPUIP.NCEClusterTiling
    inputs(%inData as %arg2: !InDataType,
           %inSparsityMap as %arg3: !InSMType,
           %inWeights as %arg4: !ConvWeightsType,
           %inWeightsTable as %arg5: !ConvWeightsTableType)
    outputs(
        %0 as %arg6: memref<1x128x14x14xf16, #NHWC, @CMX_NN>,
        %1 as %arg7: memref<1x128x14x14xi1, #NHWC, @CMX_NN>)
        -> (!VPUIP.DistributedBuffer<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
            !VPUIP.DistributedBuffer<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) {
        %1409:2 = VPUIP.NCEClusterTask
            {kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], minimumHardwareExecutionCost = 34660 : i64, task_type = #VPUIP.nce_task_type<CONV>} 
        input(%arg2 : !InDataType)
        input_sparsity_map(%arg3 : !InSMType)
        weights(%arg4 : !ConvWeightsType)
        weight_table(%arg5 : !ConvWeightsTableType)
        parent_input(%arg2 : !InDataType)
        parent_input_sparsity_map(%arg3 : !InSMType)
        parent_output(%arg6 : memref<1x128x14x14xf16, #NHWC, @CMX_NN>)
        parent_output_sparsity_map(%arg7 : memref<1x128x14x14xi1, #NHWC, @CMX_NN>)
        outputs(%arg6 : memref<1x128x14x14xf16, #NHWC, @CMX_NN>)
        output_sparsity_map(%arg7 : memref<1x128x14x14xi1, #NHWC, @CMX_NN>)
        -> memref<1x128x14x14xf16, #NHWC, @CMX_NN>, memref<1x128x14x14xi1, #NHWC, @CMX_NN>
        variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 13, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 13, 127], outStart = [0, 0, 64], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        } PPE : {
            PPETask <LRELU> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
        }
    }

    %3 = VPUIP.GroupSparseBuffer(%2#0, %2#1)
        -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
                               sparsity_map=!VPUIP.DistributedBuffer<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // Input 2: Allocated buffer for grouped op output
    %4 = VPURT.AllocDistributed -> !OutDataBufferType
    %5 = VPURT.AllocDistributed -> !OutSMBufferType
    %6 = VPUIP.GroupSparseBuffer(%4, %5) -> !VPUIP.SparseBuffer<data=!OutDataBufferType, sparsity_map=!OutSMBufferType>

    %7 = VPUIP.SubView %6 [0, 0, 0, 0] [1, 128, 14, 14] : !VPUIP.SparseBuffer<data=!OutDataBufferType, sparsity_map=!OutSMBufferType>
        to !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
                               sparsity_map=!VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // CMX->CMX copy with two distributed operands
    %8 = VPUIP.NCEClusterTiling
    inputs(%3 as %arg2: !VPUIP.SparseBuffer<data=memref<1x128x14x14xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x128x14x14xi1, #NHWC, @CMX_NN>>)
    outputs(%7 as %arg3: !ResultSparseBufferType) -> !ResultSparseDistrBufferType {
        %9 = VPUIP.Copy
            inputs(%arg2 : !VPUIP.SparseBuffer<data=memref<1x128x14x14xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x128x14x14xi1, #NHWC, @CMX_NN>>)
            outputs(%arg3 : !ResultSparseBufferType) -> !ResultSparseBufferType
    }

    return %8 : !ResultSparseDistrBufferType

    // CHECK:       [[BUFF_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:       [[BUFF_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:       [[SPARSE_BUFF:%.+]] = VPUIP.GroupSparseBuffer([[BUFF_DATA]], [[BUFF_SM]]) -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x256x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x256x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>
    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView [[SPARSE_BUFF]] [0, 0, 0, 0] [1, 128, 14, 14] : !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x256x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x256x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>> to !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // Ungroup subviewd buffer to have individual buffers of proper type to write
    // CHECK:       [[BUFF_INTERM_DATA:%.+]], [[BUFF_INTERM_SM:%.+]] = VPUIP.UngroupSparseBuffer([[SUBVIEW]]) {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} -> !VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, !VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK:       [[CLUST_TILING:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME:         inputs(%arg0 as %arg4: memref<1x256x28x28xf16, #NHWC, @CMX_NN>, %arg1 as %arg5: memref<1x256x28x28xi1, #NHWC, @CMX_NN>,
    // CHECK-SAME:                %arg2 as %arg6: memref<128x256x3x3xf16, #NHWC, @CMX_NN>, %arg3 as %arg7: memref<128x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:         outputs([[BUFF_INTERM_DATA]] as %arg8: memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>,
    // CHECK-SAME:                 [[BUFF_INTERM_SM]] as %arg9: memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>)
    // CHECK-SAME:         -> (!VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, !VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) {
    // CHECK:           [[CLUST_TASK:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:              {kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], minimumHardwareExecutionCost = 34660 : i64, task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK-SAME:              input(%arg4 : memref<1x256x28x28xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:              input_sparsity_map(%arg5 : memref<1x256x28x28xi1, #NHWC, @CMX_NN>)
    // CHECK-SAME:              weights(%arg6 : memref<128x256x3x3xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:              weight_table(%arg7 : memref<128x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:              parent_input(%arg4 : memref<1x256x28x28xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:              parent_input_sparsity_map(%arg5 : memref<1x256x28x28xi1, #NHWC, @CMX_NN>)
    // CHECK-SAME:              parent_output(%arg8 : memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>)
    // CHECK-SAME:              parent_output_sparsity_map(%arg9 : memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>)
    // CHECK-SAME:              outputs(%arg8 : memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>)
    // CHECK-SAME:              output_sparsity_map(%arg9 : memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>)
    // CHECK-SAME:              -> memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>, memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>
    // CHECK-SAME:              variants : {
    // CHECK:                         DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 13, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
    // CHECK:                         DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 13, 127], outStart = [0, 0, 64], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
    // CHECK:                   } PPE : {
    // CHECK:                         PPETask <LRELU> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
    // CHECK:                   }
    // CHECK:                }

    // CHECK:       [[GROUP_OP_CONV_OUT:%.+]] = VPUIP.GroupSparseBuffer([[CLUST_TILING]]#0, [[CLUST_TILING]]#1) -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // CHECK:       return [[GROUP_OP_CONV_OUT]] : !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InDataType = memref<1x256x28x28xf16, #NHWC, @CMX_NN>
!InSMType = memref<1x256x28x28xi1, #NHWC, @CMX_NN>
!ConvWeightsType = memref<128x256x3x3xf16, #NHWC, @CMX_NN>
!ConvWeightsTableType = memref<128x1x1x4xsi32, @CMX_NN>

!OutDataBufferType = memref<1x256x14x14xf16, #NHWC, @CMX_NN>
!OutSMBufferType = memref<1x256x14x14xi1, #NHWC, @CMX_NN>

!ResultSparseBufferType = !VPUIP.SparseBuffer<data=memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>, sparsity_map=memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>>

!ResultSparseDistrBufferType = !VPUIP.SparseBuffer<
    data=memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>,
    sparsity_map=memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>
>
// CopyOp is not wrapped into ClusterTilingOp, types are not distributed
func.func @CMX2CMXSparseTypeStrides (
    %inData : !InDataType,
    %inSparsityMap : !InSMType,
    %inWeights : !ConvWeightsType,
    %inWeightsTable : !ConvWeightsTableType)
    -> !ResultSparseDistrBufferType
{
    // alloc for Conv data out
    %0 = memref.alloc() : memref<1x128x14x14xf16, #NHWC, @CMX_NN>
    // alloc for Conv sparsity map out
    %1 = memref.alloc() : memref<1x128x14x14xi1, #NHWC, @CMX_NN>

    // Input 1: Convolution
    %2:2 = VPUIP.NCEClusterTask
            {kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], minimumHardwareExecutionCost = 34660 : i64, task_type = #VPUIP.nce_task_type<CONV>} 
        input(%inData : !InDataType)
        input_sparsity_map(%inSparsityMap : !InSMType)
        weights(%inWeights : !ConvWeightsType)
        weight_table(%inWeightsTable : !ConvWeightsTableType)
        parent_input(%inData : !InDataType)
        parent_input_sparsity_map(%inSparsityMap : !InSMType)
        parent_output(%0 : memref<1x128x14x14xf16, #NHWC, @CMX_NN>)
        parent_output_sparsity_map(%1 : memref<1x128x14x14xi1, #NHWC, @CMX_NN>)
        outputs(%0 : memref<1x128x14x14xf16, #NHWC, @CMX_NN>)
        output_sparsity_map(%1 : memref<1x128x14x14xi1, #NHWC, @CMX_NN>)
        -> memref<1x128x14x14xf16, #NHWC, @CMX_NN>, memref<1x128x14x14xi1, #NHWC, @CMX_NN>
        variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 13, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 13, 127], outStart = [0, 0, 64], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        } PPE : {
            PPETask <LRELU> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
        }

    %3 = VPUIP.GroupSparseBuffer(%2#0, %2#1)
        -> !VPUIP.SparseBuffer<data=memref<1x128x14x14xf16, #NHWC, @CMX_NN>,
                               sparsity_map=memref<1x128x14x14xi1, #NHWC, @CMX_NN>>

    // Input 2: Allocated buffer for grouped op output
    %4 = memref.alloc() : !OutDataBufferType
    %5 = memref.alloc() : !OutSMBufferType
    %6 = VPUIP.GroupSparseBuffer(%4, %5) -> !VPUIP.SparseBuffer<data=!OutDataBufferType, sparsity_map=!OutSMBufferType>

    %7 = VPUIP.SubView %6 [0, 0, 0, 0] [1, 128, 14, 14] : !VPUIP.SparseBuffer<data=!OutDataBufferType, sparsity_map=!OutSMBufferType>
        to !VPUIP.SparseBuffer<data=memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>,
                               sparsity_map=memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>>

    // CMX->CMX copy with two distributed operands
    %8  = VPUIP.Copy inputs(%3 : !VPUIP.SparseBuffer<data=memref<1x128x14x14xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x128x14x14xi1, #NHWC, @CMX_NN>>)
            outputs(%7 : !ResultSparseBufferType) -> !ResultSparseBufferType

    return %8 : !ResultSparseDistrBufferType

    // CHECK:       [[BUFF_DATA:%.+]] = memref.alloc() : memref<1x256x14x14xf16, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_SM:%.+]] = memref.alloc() : memref<1x256x14x14xi1, #NHWC, @CMX_NN>
    // CHECK:       [[SPARSE_BUFF:%.+]] = VPUIP.GroupSparseBuffer([[BUFF_DATA]], [[BUFF_SM]]) -> !VPUIP.SparseBuffer<data=memref<1x256x14x14xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x256x14x14xi1, #NHWC, @CMX_NN>>
    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView [[SPARSE_BUFF]] [0, 0, 0, 0] [1, 128, 14, 14] : !VPUIP.SparseBuffer<data=memref<1x256x14x14xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x256x14x14xi1, #NHWC, @CMX_NN>> to !VPUIP.SparseBuffer<data=memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>, sparsity_map=memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>>
    // CHECK:       [[BUFF_INTERM_DATA:%.+]], [[BUFF_INTERM_SM:%.+]] = VPUIP.UngroupSparseBuffer([[SUBVIEW]]) {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} -> memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>, memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>

    // CHECK:       [[CLUST_TASK:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:      {kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], minimumHardwareExecutionCost = 34660 : i64, task_type = #VPUIP.nce_task_type<CONV>} 
    // CHECK-SAME:      input(%arg0 : memref<1x256x28x28xf16, #NHWC, @CMX_NN>) input_sparsity_map(%arg1 : memref<1x256x28x28xi1, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weights(%arg2 : memref<128x256x3x3xf16, #NHWC, @CMX_NN>) weight_table(%arg3 : memref<128x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      parent_input(%arg0 : memref<1x256x28x28xf16, #NHWC, @CMX_NN>) parent_input_sparsity_map(%arg1 : memref<1x256x28x28xi1, #NHWC, @CMX_NN>)
    // CHECK-SAME:      parent_output([[BUFF_INTERM_DATA]] : memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>)
    // CHECK-SAME:      parent_output_sparsity_map([[BUFF_INTERM_SM]] : memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUFF_INTERM_DATA]] : memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>)
    // CHECK-SAME:      output_sparsity_map([[BUFF_INTERM_SM]] : memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>)
    // CHECK-SAME:          -> memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>, memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>
    // CHECK-SAME:  variants : {
    // CHECK:           DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 13, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
    // CHECK:           DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 13, 127], outStart = [0, 0, 64], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
    // CHECK:       } PPE : {
    // CHECK:           PPETask <LRELU> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
    // CHECK:       }

    // CHECK:       [[GROUP_OP_CONV_OUT:%.+]] = VPUIP.GroupSparseBuffer([[CLUST_TASK]]#0, [[CLUST_TASK]]#1)
    // CHECK-SAME:      -> !VPUIP.SparseBuffer<data=memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>, sparsity_map=memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>>

    // CHECK:       return [[GROUP_OP_CONV_OUT]] : !VPUIP.SparseBuffer<data=memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>, sparsity_map=memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InDataType = memref<1x256x28x28xf16, #NHWC, @CMX_NN>
!InSMType = memref<1x256x28x28xi1, #NHWC, @CMX_NN>
!ConvWeightsType = memref<128x256x3x3xf16, #NHWC, @CMX_NN>
!ConvWeightsTableType = memref<128x1x1x4xsi32, @CMX_NN>

!OutDataBufferType = !VPUIP.DistributedBuffer<1x256x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
!OutSMBufferType = !VPUIP.DistributedBuffer<1x256x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

!ResultSparseBufferType = !VPUIP.SparseBuffer<data=memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>, sparsity_map=memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>>

!ResultSparseDistrBufferType = !VPUIP.SparseBuffer<
    data=!VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    sparsity_map=!VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
>

!ConcatOutSparseBufferType = !VPUIP.SparseBuffer<
    data=!VPUIP.DistributedBuffer<1x256x14x14xf16, {order = #NHWC}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    sparsity_map=!VPUIP.DistributedBuffer<1x256x14x14xi1, {order = #NHWC}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
>

// CopyOp is wrapped into ClusterTilingOp, types are distributed
func.func @SiblingTilingCopyOptimizationWithSparsity (
    %inData : !InDataType,
    %inSparsityMap : !InSMType,
    %inWeights : !ConvWeightsType,
    %inWeightsTable : !ConvWeightsTableType)
    -> (!ConcatOutSparseBufferType, memref<1x128x14x14xf16, #NHWC, @DDR>)
{
    // alloc for Conv data out
    %0 = memref.alloc() : memref<1x128x14x14xf16, #NHWC, @CMX_NN>
    // alloc for Conv sparsity map out
    %1 = memref.alloc() : memref<1x128x14x14xi1, #NHWC, @CMX_NN>

    // Input 1: Convolution
    %2:2 = VPUIP.NCEClusterTiling
    inputs(%inData as %arg2: !InDataType,
           %inSparsityMap as %arg3: !InSMType,
           %inWeights as %arg4: !ConvWeightsType,
           %inWeightsTable as %arg5: !ConvWeightsTableType)
    outputs(
        %0 as %arg6: memref<1x128x14x14xf16, #NHWC, @CMX_NN>,
        %1 as %arg7: memref<1x128x14x14xi1, #NHWC, @CMX_NN>)
        -> (!VPUIP.DistributedBuffer<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
            !VPUIP.DistributedBuffer<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) {
        %1409:2 = VPUIP.NCEClusterTask
            {kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], minimumHardwareExecutionCost = 34660 : i64, task_type = #VPUIP.nce_task_type<CONV>}
        input(%arg2 : !InDataType)
        input_sparsity_map(%arg3 : !InSMType)
        weights(%arg4 : !ConvWeightsType)
        weight_table(%arg5 : !ConvWeightsTableType)
        parent_input(%arg2 : !InDataType)
        parent_input_sparsity_map(%arg3 : !InSMType)
        parent_output(%arg6 : memref<1x128x14x14xf16, #NHWC, @CMX_NN>)
        parent_output_sparsity_map(%arg7 : memref<1x128x14x14xi1, #NHWC, @CMX_NN>)
        outputs(%arg6 : memref<1x128x14x14xf16, #NHWC, @CMX_NN>)
        output_sparsity_map(%arg7 : memref<1x128x14x14xi1, #NHWC, @CMX_NN>)
        -> memref<1x128x14x14xf16, #NHWC, @CMX_NN>, memref<1x128x14x14xi1, #NHWC, @CMX_NN>
        variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 13, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 13, 127], outStart = [0, 0, 64], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        } PPE : {
            PPETask <LRELU> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
        }
    }

    %3 = VPUIP.GroupSparseBuffer(%2#0, %2#1)
        -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
                               sparsity_map=!VPUIP.DistributedBuffer<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // Input 2: Allocated buffer for grouped op output
    %4 = VPURT.AllocDistributed -> !OutDataBufferType
    %5 = VPURT.AllocDistributed -> !OutSMBufferType
    %6 = VPUIP.GroupSparseBuffer(%4, %5) -> !VPUIP.SparseBuffer<data=!OutDataBufferType, sparsity_map=!OutSMBufferType>

    %7 = VPUIP.SubView %6 [0, 0, 0, 0] [1, 128, 14, 14] : !VPUIP.SparseBuffer<data=!OutDataBufferType, sparsity_map=!OutSMBufferType>
        to !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
                               sparsity_map=!VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // CMX->CMX copy with two distributed operands
    %8 = VPUIP.NCEClusterTiling
    inputs(%3 as %arg2: !VPUIP.SparseBuffer<data=memref<1x128x14x14xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x128x14x14xi1, #NHWC, @CMX_NN>>)
    outputs(%7 as %arg3: !ResultSparseBufferType) -> !ResultSparseDistrBufferType {
        %9 = VPUIP.Copy
            inputs(%arg2 : !VPUIP.SparseBuffer<data=memref<1x128x14x14xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x128x14x14xi1, #NHWC, @CMX_NN>>)
            outputs(%arg3 : !ResultSparseBufferType) -> !ResultSparseBufferType
    }

    // alloc for Conv data out
    %10 = memref.alloc() : memref<1x128x14x14xf16, #NHWC, @CMX_NN>
    // alloc for Conv sparsity map out
    %11 = memref.alloc() : memref<1x128x14x14xi1, #NHWC, @CMX_NN>

    // Input 1: Convolution
    %12:2 = VPUIP.NCEClusterTiling
    inputs(%inData as %arg2: !InDataType,
           %inSparsityMap as %arg3: !InSMType,
           %inWeights as %arg4: !ConvWeightsType,
           %inWeightsTable as %arg5: !ConvWeightsTableType)
    outputs(
        %10 as %arg6: memref<1x128x14x14xf16, #NHWC, @CMX_NN>,
        %11 as %arg7: memref<1x128x14x14xi1, #NHWC, @CMX_NN>)
        -> (!VPUIP.DistributedBuffer<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
            !VPUIP.DistributedBuffer<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) {
        %1409:2 = VPUIP.NCEClusterTask
            {kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], minimumHardwareExecutionCost = 34660 : i64, task_type = #VPUIP.nce_task_type<CONV>}
        input(%arg2 : !InDataType)
        input_sparsity_map(%arg3 : !InSMType)
        weights(%arg4 : !ConvWeightsType)
        weight_table(%arg5 : !ConvWeightsTableType)
        parent_input(%arg2 : !InDataType)
        parent_input_sparsity_map(%arg3 : !InSMType)
        parent_output(%arg6 : memref<1x128x14x14xf16, #NHWC, @CMX_NN>)
        parent_output_sparsity_map(%arg7 : memref<1x128x14x14xi1, #NHWC, @CMX_NN>)
        outputs(%arg6 : memref<1x128x14x14xf16, #NHWC, @CMX_NN>)
        output_sparsity_map(%arg7 : memref<1x128x14x14xi1, #NHWC, @CMX_NN>)
        -> memref<1x128x14x14xf16, #NHWC, @CMX_NN>, memref<1x128x14x14xi1, #NHWC, @CMX_NN>
        variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 13, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 13, 127], outStart = [0, 0, 64], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        } PPE : {
            PPETask <LRELU> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
        }
    }

    %13 = VPUIP.GroupSparseBuffer(%12#0, %12#1)
        -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
                               sparsity_map=!VPUIP.DistributedBuffer<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // Input 2: Allocated buffer for grouped op output
    %14 = VPURT.AllocDistributed -> !OutDataBufferType
    %15 = VPURT.AllocDistributed -> !OutSMBufferType
    %16 = VPUIP.GroupSparseBuffer(%14, %15) -> !VPUIP.SparseBuffer<data=!OutDataBufferType, sparsity_map=!OutSMBufferType>

    %17 = VPUIP.SubView %16 [0, 128, 0, 0] [1, 128, 14, 14] : !VPUIP.SparseBuffer<data=!OutDataBufferType, sparsity_map=!OutSMBufferType>
        to !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
                               sparsity_map=!VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // CMX->CMX copy with two distributed operands
    %18 = VPUIP.NCEClusterTiling
    inputs(%13 as %arg2: !VPUIP.SparseBuffer<data=memref<1x128x14x14xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x128x14x14xi1, #NHWC, @CMX_NN>>)
    outputs(%17 as %arg3: !ResultSparseBufferType) -> !ResultSparseDistrBufferType {
        %9 = VPUIP.Copy
            inputs(%arg2 : !VPUIP.SparseBuffer<data=memref<1x128x14x14xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x128x14x14xi1, #NHWC, @CMX_NN>>)
            outputs(%arg3 : !ResultSparseBufferType) -> !ResultSparseBufferType
    }

    // alloc for concat data out
    %19 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x14x14xf16, {order = #NHWC}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // alloc for concat sparsity map out
    %20 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x14x14xi1, {order = #NHWC}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %21 = VPUIP.GroupSparseBuffer(%19, %20)
        -> !ConcatOutSparseBufferType

    %22 = VPUIP.ConcatView
    inputs(%8, %18 :
        !ResultSparseDistrBufferType,
        !ResultSparseDistrBufferType
    )
    outputs(%21 : !ConcatOutSparseBufferType)
    -> !ConcatOutSparseBufferType

    %23 = memref.alloc() : memref<1x128x14x14xf16, #NHWC, @DDR>
    %24 = VPUIP.NCEClusterTiling
    inputs(%12 as %arg2: memref<1x128x14x14xf16, #NHWC, @CMX_NN>)
    outputs(%23 as %arg3: memref<1x128x14x14xf16, #NHWC, @DDR>) -> memref<1x128x14x14xf16, #NHWC, @DDR>{
        %9 = VPUIP.Copy
            inputs(%arg2 : memref<1x128x14x14xf16, #NHWC, @CMX_NN>)
            outputs(%arg3 : memref<1x128x14x14xf16, #NHWC, @DDR>) -> memref<1x128x14x14xf16, #NHWC, @DDR>}

    return %22, %24 : !ConcatOutSparseBufferType, memref<1x128x14x14xf16, #NHWC, @DDR>

    // CHECK:      [[BUFF_0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:      [[BUFF_1:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:      [[SPARSE:%.*]] = VPUIP.GroupSparseBuffer([[BUFF_0]], [[BUFF_1]]) -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x256x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x256x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>
    // CHECK:      [[SUBVIEW_0:%.*]] = VPUIP.SubView [[SPARSE]] [0, 0, 0, 0] [1, 128, 14, 14] : !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x256x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x256x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>> to !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>
    // CHECK:      [[DATA_0:%.*]], [[SPARSITYMAP_0:%.*]] = VPUIP.UngroupSparseBuffer([[SUBVIEW_0]]) {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} -> !VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, !VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK:      [[NCE_0:%.*]]:2 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg4: memref<1x256x28x28xf16, #NHWC, @CMX_NN>, %arg1 as %arg5: memref<1x256x28x28xi1, #NHWC, @CMX_NN>, %arg2 as %arg6: memref<128x256x3x3xf16, #NHWC, @CMX_NN>, %arg3 as %arg7: memref<128x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      outputs([[DATA_0]] as %arg8: memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>, [[SPARSITYMAP_0]] as %arg9: memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, !VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) {
    // CHECK:           [[INNER_0:%.*]]:2 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], minimumHardwareExecutionCost = 34660 : i64, task_type = #VPUIP.nce_task_type<CONV>} input(%arg4 : memref<1x256x28x28xf16, #NHWC, @CMX_NN>) input_sparsity_map(%arg5 : memref<1x256x28x28xi1, #NHWC, @CMX_NN>) weights(%arg6 : memref<128x256x3x3xf16, #NHWC, @CMX_NN>) weight_table(%arg7 : memref<128x1x1x4xsi32, @CMX_NN>) parent_input(%arg4 : memref<1x256x28x28xf16, #NHWC, @CMX_NN>) parent_input_sparsity_map(%arg5 : memref<1x256x28x28xi1, #NHWC, @CMX_NN>) parent_output(%arg8 : memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>) parent_output_sparsity_map(%arg9 : memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>) outputs(%arg8 : memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>) output_sparsity_map(%arg9 : memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>) -> memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>, memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN> variants : {
    // CHECK:           DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 13, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
    // CHECK:           DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 13, 127], outStart = [0, 0, 64], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
    // CHECK:           } PPE : {
    // CHECK:           PPETask <LRELU> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
    // CHECK:           }
    // CHECK:      }
    // CHECK:      [[SPARSE_0:%.*]] = VPUIP.GroupSparseBuffer([[NCE_0]]#0, [[NCE_0]]#1) -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // CHECK:      [[BUFF_3:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:      [[BUFF_4:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:      [[SPARSE_1:%.*]] = VPUIP.GroupSparseBuffer([[BUFF_3]], [[BUFF_4]]) -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x256x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x256x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>
    // CHECK:      [[SUBVIEW_1:%.*]] = VPUIP.SubView [[SPARSE_1]] [0, 128, 0, 0] [1, 128, 14, 14] : !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x256x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x256x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>> to !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>
    // CHECK:      [[DATA_1:%.*]], [[SPARSITYMAP_1:%.*]] = VPUIP.UngroupSparseBuffer([[SUBVIEW_1]]) {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} -> !VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, !VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK:      [[NCE_1:%.*]]:2 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg4: memref<1x256x28x28xf16, #NHWC, @CMX_NN>, %arg1 as %arg5: memref<1x256x28x28xi1, #NHWC, @CMX_NN>, %arg2 as %arg6: memref<128x256x3x3xf16, #NHWC, @CMX_NN>, %arg3 as %arg7: memref<128x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      outputs([[DATA_1]] as %arg8: memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>, [[SPARSITYMAP_1]] as %arg9: memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, !VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) {
    // CHECK:           [[INNER_1:%.*]]:2 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], minimumHardwareExecutionCost = 34660 : i64, task_type = #VPUIP.nce_task_type<CONV>} input(%arg4 : memref<1x256x28x28xf16, #NHWC, @CMX_NN>) input_sparsity_map(%arg5 : memref<1x256x28x28xi1, #NHWC, @CMX_NN>) weights(%arg6 : memref<128x256x3x3xf16, #NHWC, @CMX_NN>) weight_table(%arg7 : memref<128x1x1x4xsi32, @CMX_NN>) parent_input(%arg4 : memref<1x256x28x28xf16, #NHWC, @CMX_NN>) parent_input_sparsity_map(%arg5 : memref<1x256x28x28xi1, #NHWC, @CMX_NN>) parent_output(%arg8 : memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>) parent_output_sparsity_map(%arg9 : memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>) outputs(%arg8 : memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>) output_sparsity_map(%arg9 : memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>) -> memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>, memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN> variants : {
    // CHECK:           DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 13, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
    // CHECK:           DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 13, 127], outStart = [0, 0, 64], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
    // CHECK:           } PPE : {
    // CHECK:           PPETask <LRELU> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
    // CHECK:           }
    // CHECK:      }
    // CHECK:      [[SPARSE_1:%.*]] = VPUIP.GroupSparseBuffer([[NCE_1]]#0, [[NCE_1]]#1) -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // CHECK:      [[BUFF_5:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x14x14xf16, {order = #NHWC}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:      [[BUFF_6:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x14x14xi1, {order = #NHWC}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:      [[SPARSE_2:%.*]] = VPUIP.GroupSparseBuffer([[BUFF_5]], [[BUFF_6]]) -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x256x14x14xf16, {order = #NHWC}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x256x14x14xi1, {order = #NHWC}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>
    // CHECK:      [[CONCAT:%.*]] =  VPUIP.ConcatView inputs([[SPARSE_0]], [[SPARSE_1]] : !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>, !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>) outputs(%14 : !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x256x14x14xf16, {order = #NHWC}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x256x14x14xi1, {order = #NHWC}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>) -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x256x14x14xf16, {order = #NHWC}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x256x14x14xi1, {order = #NHWC}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // CHECK:      [[COPYBUFF:%.*]] = memref.alloc() : memref<1x128x14x14xf16, #NHWC, @DDR>
    // CHECK:      [[COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[NCE_1]]#0 as %arg4: memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>) outputs([[COPYBUFF]] as %arg5: memref<1x128x14x14xf16, #NHWC, @DDR>) -> memref<1x128x14x14xf16, #NHWC, @DDR> {
    // CHECK:       %18 = VPUIP.Copy inputs(%arg4 : memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>) outputs(%arg5 : memref<1x128x14x14xf16, #NHWC, @DDR>) -> memref<1x128x14x14xf16, #NHWC, @DDR>
    // CHECK:       }

    // CHECK:       return [[CONCAT]], [[COPY]] : !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x256x14x14xf16, {order = #NHWC}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x256x14x14xi1, {order = #NHWC}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>, memref<1x128x14x14xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InDataType = memref<1x256x28x28xf16, #NHWC, @CMX_NN>
!InSMType = memref<1x256x28x28xi1, #NHWC, @CMX_NN>
!ConvWeightsType = memref<128x256x3x3xf16, #NHWC, @CMX_NN>
!ConvWeightsTableType = memref<128x1x1x4xsi32, @CMX_NN>

!ConvOutDataBufferType = memref<1x128x14x14xf16, #NHWC, @DDR>
!ConvOutSMBufferType = memref<1x128x14x14xi1, #NHWC, @DDR>
!ConvOutSparseBufferType = !VPUIP.SparseBuffer<data=!ConvOutDataBufferType, sparsity_map=!ConvOutSMBufferType>

!ConvOutSubview1SparseBufferTypeWithStrides = !VPUIP.SparseBuffer<
    data=memref<1x1x14x14xf16, {order = #NHWC, strides = [25088, 1, 1792, 128]}, @DDR>,
    sparsity_map=memref<1x1x14x14xi1, {order = #NHWC, strides = [25088, 1, 1792, 128]}, @DDR>
>

!ConcatOutDataBufferType = memref<1x130x14x14xf16, #NHWC, @DDR>
!ConcatOutSMBufferType = memref<1x130x14x14xi1, #NHWC, @DDR>
!ConcatOutSparseBufferType = !VPUIP.SparseBuffer<data=!ConcatOutDataBufferType, sparsity_map=!ConcatOutSMBufferType>
!ConcatOutSparseBufferTypeWithStrides = !VPUIP.SparseBuffer<
    data=memref<1x128x14x14xf16, {order = #NHWC, strides = [25480, 1, 1820, 130]}, @DDR>,
    sparsity_map=memref<1x128x14x14xi1, {order = #NHWC, strides = [25480, 1, 1820, 130]}, @DDR>
>

!ConcatOutSubview1SparseBufferTypeWithStrides = !VPUIP.SparseBuffer<
    data=memref<1x1x14x14xf16, {order = #NHWC, strides = [25480, 1, 1820, 130]}, @DDR>,
    sparsity_map=memref<1x1x14x14xi1, {order = #NHWC, strides = [25480, 1, 1820, 130]}, @DDR>
>

func.func @ParallelDDR2DDRCopyOutputNoChangeToFixAccuracyWithSparsity (
    %inData : !InDataType,
    %inSparsityMap : !InSMType,
    %inWeights : !ConvWeightsType,
    %inWeightsTable : !ConvWeightsTableType)
    -> (!ConcatOutSparseBufferType)
{
    // alloc for Conv data out
    %0 = memref.alloc() : memref<1x128x14x14xf16, #NHWC, @CMX_NN>
    // alloc for Conv sparsity map out
    %1 = memref.alloc() : memref<1x128x14x14xi1, #NHWC, @CMX_NN>

    // Input 1: Convolution
    %2:2 = VPUIP.NCEClusterTiling
    inputs(%inData as %arg2: !InDataType,
           %inSparsityMap as %arg3: !InSMType,
           %inWeights as %arg4: !ConvWeightsType,
           %inWeightsTable as %arg5: !ConvWeightsTableType)
    outputs(
        %0 as %arg6: memref<1x128x14x14xf16, #NHWC, @CMX_NN>,
        %1 as %arg7: memref<1x128x14x14xi1, #NHWC, @CMX_NN>)
        -> (!VPUIP.DistributedBuffer<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
            !VPUIP.DistributedBuffer<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) {
        %1409:2 = VPUIP.NCEClusterTask
            {kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], minimumHardwareExecutionCost = 34660 : i64, task_type = #VPUIP.nce_task_type<CONV>}
        input(%arg2 : !InDataType)
        input_sparsity_map(%arg3 : !InSMType)
        weights(%arg4 : !ConvWeightsType)
        weight_table(%arg5 : !ConvWeightsTableType)
        parent_input(%arg2 : !InDataType)
        parent_input_sparsity_map(%arg3 : !InSMType)
        parent_output(%arg6 : memref<1x128x14x14xf16, #NHWC, @CMX_NN>)
        parent_output_sparsity_map(%arg7 : memref<1x128x14x14xi1, #NHWC, @CMX_NN>)
        outputs(%arg6 : memref<1x128x14x14xf16, #NHWC, @CMX_NN>)
        output_sparsity_map(%arg7 : memref<1x128x14x14xi1, #NHWC, @CMX_NN>)
        -> memref<1x128x14x14xf16, #NHWC, @CMX_NN>, memref<1x128x14x14xi1, #NHWC, @CMX_NN>
        variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 13, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 13, 127], outStart = [0, 0, 64], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        } PPE : {
            PPETask <LRELU> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
        }
    }

    %3 = VPUIP.GroupSparseBuffer(%2#0, %2#1)
        -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
                               sparsity_map=!VPUIP.DistributedBuffer<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // Input 2: Allocated buffer for grouped op output
    %4 = memref.alloc() : memref<1x128x14x14xf16, #NHWC, @DDR>
    %5 = memref.alloc() : memref<1x128x14x14xi1, #NHWC, @DDR>
    %6 = VPUIP.GroupSparseBuffer(%4, %5) -> !ConvOutSparseBufferType

    // CMX->DDR copy with two distributed operands
    %7 = VPUIP.NCEClusterTiling
    inputs(%3 as %arg2: !VPUIP.SparseBuffer<data=memref<1x128x14x14xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x128x14x14xi1, #NHWC, @CMX_NN>>)
    outputs(%6 as %arg3: !ConvOutSparseBufferType) -> !ConvOutSparseBufferType {
        %1500 = VPUIP.Copy
            inputs(%arg2 : !VPUIP.SparseBuffer<data=memref<1x128x14x14xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x128x14x14xi1, #NHWC, @CMX_NN>>)
            outputs(%arg3 : !ConvOutSparseBufferType) -> !ConvOutSparseBufferType
    }

    // 2 x Subview
    %8 = VPUIP.SubView %7 [0, 1, 0, 0] [1, 1, 14, 14] : !ConvOutSparseBufferType to !ConvOutSubview1SparseBufferTypeWithStrides
    %9 = VPUIP.SubView %7 [0, 126, 0, 0] [1, 1, 14, 14] : !ConvOutSparseBufferType to !ConvOutSubview1SparseBufferTypeWithStrides

    // Output: Allocated buffer for grouped op output
    %10 = memref.alloc() : memref<1x130x14x14xf16, #NHWC, @DDR>
    %11 = memref.alloc() : memref<1x130x14x14xi1, #NHWC, @DDR>
    %12 = VPUIP.GroupSparseBuffer(%10, %11) -> !VPUIP.SparseBuffer<data=!ConcatOutDataBufferType, sparsity_map=!ConcatOutSMBufferType>

    %13 = VPUIP.SubView %12 [0, 0, 0, 0] [1, 1, 14, 14] : !VPUIP.SparseBuffer<data=!ConcatOutDataBufferType, sparsity_map=!ConcatOutSMBufferType> to !ConcatOutSubview1SparseBufferTypeWithStrides
    %14 = VPUIP.Copy inputs(%8 : !ConvOutSubview1SparseBufferTypeWithStrides) outputs(%13 : !ConcatOutSubview1SparseBufferTypeWithStrides) -> !ConcatOutSubview1SparseBufferTypeWithStrides

    %15 = VPUIP.SubView %12 [0, 1, 0, 0] [1, 128, 14, 14] : !VPUIP.SparseBuffer<data=!ConcatOutDataBufferType, sparsity_map=!ConcatOutSMBufferType> to !ConcatOutSparseBufferTypeWithStrides
    %16 = VPUIP.Copy inputs(%7 : !ConvOutSparseBufferType) outputs(%15 : !ConcatOutSparseBufferTypeWithStrides) -> !ConcatOutSparseBufferTypeWithStrides

    %17 = VPUIP.SubView %12 [0, 129, 0, 0] [1, 1, 14, 14] : !VPUIP.SparseBuffer<data=!ConcatOutDataBufferType, sparsity_map=!ConcatOutSMBufferType> to !ConcatOutSubview1SparseBufferTypeWithStrides
    %18 = VPUIP.Copy inputs(%9 : !ConvOutSubview1SparseBufferTypeWithStrides) outputs(%17 : !ConcatOutSubview1SparseBufferTypeWithStrides) -> !ConcatOutSubview1SparseBufferTypeWithStrides

    // Concat for output
    %19 = VPUIP.ConcatView inputs(%14, %16, %18 : !ConcatOutSubview1SparseBufferTypeWithStrides, !ConcatOutSparseBufferTypeWithStrides, !ConcatOutSubview1SparseBufferTypeWithStrides) outputs(%12 : !ConcatOutSparseBufferType) -> !ConcatOutSparseBufferType

    return %19 : !ConcatOutSparseBufferType

    // CHECK:      [[BUFF_0:%.*]] = memref.alloc() : memref<1x128x14x14xf16, #NHWC, @CMX_NN>
    // CHECK:      [[BUFF_1:%.*]] = memref.alloc() : memref<1x128x14x14xi1, #NHWC, @CMX_NN>
    // CHECK:      [[NCE_0:%.*]]:2 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg4: memref<1x256x28x28xf16, #NHWC, @CMX_NN>, %arg1 as %arg5: memref<1x256x28x28xi1, #NHWC, @CMX_NN>, %arg2 as %arg6: memref<128x256x3x3xf16, #NHWC, @CMX_NN>, %arg3 as %arg7: memref<128x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUFF_0]] as %arg8: memref<1x128x14x14xf16, #NHWC, @CMX_NN>, [[BUFF_1]] as %arg9: memref<1x128x14x14xi1, #NHWC, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, !VPUIP.DistributedBuffer<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) {
    // CHECK:           [[INNER_0:%.*]]:2 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], minimumHardwareExecutionCost = 34660 : i64, task_type = #VPUIP.nce_task_type<CONV>} input(%arg4 : memref<1x256x28x28xf16, #NHWC, @CMX_NN>) input_sparsity_map(%arg5 : memref<1x256x28x28xi1, #NHWC, @CMX_NN>) weights(%arg6 : memref<128x256x3x3xf16, #NHWC, @CMX_NN>) weight_table(%arg7 : memref<128x1x1x4xsi32, @CMX_NN>) parent_input(%arg4 : memref<1x256x28x28xf16, #NHWC, @CMX_NN>) parent_input_sparsity_map(%arg5 : memref<1x256x28x28xi1, #NHWC, @CMX_NN>) parent_output(%arg8 : memref<1x128x14x14xf16, #NHWC, @CMX_NN>) parent_output_sparsity_map(%arg9 : memref<1x128x14x14xi1, #NHWC, @CMX_NN>) outputs(%arg8 : memref<1x128x14x14xf16, #NHWC, @CMX_NN>) output_sparsity_map(%arg9 : memref<1x128x14x14xi1, #NHWC, @CMX_NN>) -> memref<1x128x14x14xf16, #NHWC, @CMX_NN>, memref<1x128x14x14xi1, #NHWC, @CMX_NN> variants : {
    // CHECK:           DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 13, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
    // CHECK:           DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 13, 127], outStart = [0, 0, 64], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
    // CHECK:           } PPE : {
    // CHECK:           PPETask <LRELU> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
    // CHECK:           }
    // CHECK:      }
    // CHECK:      [[SPARSE_0:%.*]] = VPUIP.GroupSparseBuffer([[NCE_0]]#0, [[NCE_0]]#1) -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // CHECK:      [[BUFF_2:%.*]] = memref.alloc() : memref<1x128x14x14xf16, #NHWC, @DDR>
    // CHECK:      [[BUFF_3:%.*]] = memref.alloc() : memref<1x128x14x14xi1, #NHWC, @DDR>
    // CHECK:      [[SPARSE_1:%.*]] = VPUIP.GroupSparseBuffer([[BUFF_2]], [[BUFF_3]]) -> !VPUIP.SparseBuffer<data=memref<1x128x14x14xf16, #NHWC, @DDR>, sparsity_map=memref<1x128x14x14xi1, #NHWC, @DDR>>

    // CHECK:      [[COPY_0:%.*]] = VPUIP.NCEClusterTiling inputs([[SPARSE_0]] as %arg4: !VPUIP.SparseBuffer<data=memref<1x128x14x14xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x128x14x14xi1, #NHWC, @CMX_NN>>) outputs([[SPARSE_1]] as %arg5: !VPUIP.SparseBuffer<data=memref<1x128x14x14xf16, #NHWC, @DDR>, sparsity_map=memref<1x128x14x14xi1, #NHWC, @DDR>>) -> !VPUIP.SparseBuffer<data=memref<1x128x14x14xf16, #NHWC, @DDR>, sparsity_map=memref<1x128x14x14xi1, #NHWC, @DDR>> {
    // CHECK:           [[COPY_0_INNER:%.*]] = VPUIP.Copy inputs(%arg4 : !VPUIP.SparseBuffer<data=memref<1x128x14x14xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x128x14x14xi1, #NHWC, @CMX_NN>>) outputs(%arg5 : !VPUIP.SparseBuffer<data=memref<1x128x14x14xf16, #NHWC, @DDR>, sparsity_map=memref<1x128x14x14xi1, #NHWC, @DDR>>) -> !VPUIP.SparseBuffer<data=memref<1x128x14x14xf16, #NHWC, @DDR>, sparsity_map=memref<1x128x14x14xi1, #NHWC, @DDR>>
    // CHECK:      }
    // CHECK:      [[SUBVIEW_0:%.*]] = VPUIP.SubView [[COPY_0]] [0, 1, 0, 0] [1, 1, 14, 14] : !VPUIP.SparseBuffer<data=memref<1x128x14x14xf16, #NHWC, @DDR>, sparsity_map=memref<1x128x14x14xi1, #NHWC, @DDR>> to !VPUIP.SparseBuffer<data=memref<1x1x14x14xf16, {order = #NHWC, strides = [25088, 1, 1792, 128]}, @DDR>, sparsity_map=memref<1x1x14x14xi1, {order = #NHWC, strides = [25088, 1, 1792, 128]}, @DDR>>
    // CHECK:      [[SUBVIEW_1:%.*]] = VPUIP.SubView [[COPY_0]] [0, 126, 0, 0] [1, 1, 14, 14] : !VPUIP.SparseBuffer<data=memref<1x128x14x14xf16, #NHWC, @DDR>, sparsity_map=memref<1x128x14x14xi1, #NHWC, @DDR>> to !VPUIP.SparseBuffer<data=memref<1x1x14x14xf16, {order = #NHWC, strides = [25088, 1, 1792, 128]}, @DDR>, sparsity_map=memref<1x1x14x14xi1, {order = #NHWC, strides = [25088, 1, 1792, 128]}, @DDR>>

    // CHECK:      [[BUFF_4:%.*]] = memref.alloc() : memref<1x130x14x14xf16, #NHWC, @DDR>
    // CHECK:      [[BUFF_5:%.*]] = memref.alloc() : memref<1x130x14x14xi1, #NHWC, @DDR>
    // CHECK:      [[SPARSE_2:%.*]] = VPUIP.GroupSparseBuffer([[BUFF_4]], [[BUFF_5]]) -> !VPUIP.SparseBuffer<data=memref<1x130x14x14xf16, #NHWC, @DDR>, sparsity_map=memref<1x130x14x14xi1, #NHWC, @DDR>>

    // CHECK:      [[SUBVIEW_2:%.*]] = VPUIP.SubView [[SPARSE_2]] [0, 0, 0, 0] [1, 1, 14, 14] : !VPUIP.SparseBuffer<data=memref<1x130x14x14xf16, #NHWC, @DDR>, sparsity_map=memref<1x130x14x14xi1, #NHWC, @DDR>> to !VPUIP.SparseBuffer<data=memref<1x1x14x14xf16, {order = #NHWC, strides = [25480, 1, 1820, 130]}, @DDR>, sparsity_map=memref<1x1x14x14xi1, {order = #NHWC, strides = [25480, 1, 1820, 130]}, @DDR>>
    // CHECK:      [[COPY_1:%.*]] = VPUIP.Copy inputs([[SUBVIEW_0]] : !VPUIP.SparseBuffer<data=memref<1x1x14x14xf16, {order = #NHWC, strides = [25088, 1, 1792, 128]}, @DDR>, sparsity_map=memref<1x1x14x14xi1, {order = #NHWC, strides = [25088, 1, 1792, 128]}, @DDR>>) outputs([[SUBVIEW_2]] : !VPUIP.SparseBuffer<data=memref<1x1x14x14xf16, {order = #NHWC, strides = [25480, 1, 1820, 130]}, @DDR>, sparsity_map=memref<1x1x14x14xi1, {order = #NHWC, strides = [25480, 1, 1820, 130]}, @DDR>>) -> !VPUIP.SparseBuffer<data=memref<1x1x14x14xf16, {order = #NHWC, strides = [25480, 1, 1820, 130]}, @DDR>, sparsity_map=memref<1x1x14x14xi1, {order = #NHWC, strides = [25480, 1, 1820, 130]}, @DDR>>

    // CHECK:      [[SUBVIEW_3:%.*]] = VPUIP.SubView [[SPARSE_2]] [0, 1, 0, 0] [1, 128, 14, 14] : !VPUIP.SparseBuffer<data=memref<1x130x14x14xf16, #NHWC, @DDR>, sparsity_map=memref<1x130x14x14xi1, #NHWC, @DDR>> to !VPUIP.SparseBuffer<data=memref<1x128x14x14xf16, {order = #NHWC, strides = [25480, 1, 1820, 130]}, @DDR>, sparsity_map=memref<1x128x14x14xi1, {order = #NHWC, strides = [25480, 1, 1820, 130]}, @DDR>>
    // CHECK:      [[COPY_2:%.*]] = VPUIP.Copy inputs([[COPY_0]] : !VPUIP.SparseBuffer<data=memref<1x128x14x14xf16, #NHWC, @DDR>, sparsity_map=memref<1x128x14x14xi1, #NHWC, @DDR>>) outputs([[SUBVIEW_3]] : !VPUIP.SparseBuffer<data=memref<1x128x14x14xf16, {order = #NHWC, strides = [25480, 1, 1820, 130]}, @DDR>, sparsity_map=memref<1x128x14x14xi1, {order = #NHWC, strides = [25480, 1, 1820, 130]}, @DDR>>) -> !VPUIP.SparseBuffer<data=memref<1x128x14x14xf16, {order = #NHWC, strides = [25480, 1, 1820, 130]}, @DDR>, sparsity_map=memref<1x128x14x14xi1, {order = #NHWC, strides = [25480, 1, 1820, 130]}, @DDR>>

    // CHECK:      [[SUBVIEW_4:%.*]] = VPUIP.SubView [[SPARSE_2]] [0, 129, 0, 0] [1, 1, 14, 14] : !VPUIP.SparseBuffer<data=memref<1x130x14x14xf16, #NHWC, @DDR>, sparsity_map=memref<1x130x14x14xi1, #NHWC, @DDR>> to !VPUIP.SparseBuffer<data=memref<1x1x14x14xf16, {order = #NHWC, strides = [25480, 1, 1820, 130]}, @DDR>, sparsity_map=memref<1x1x14x14xi1, {order = #NHWC, strides = [25480, 1, 1820, 130]}, @DDR>>
    // CHECK:      [[COPY_3:%.*]]  = VPUIP.Copy inputs([[SUBVIEW_1]] : !VPUIP.SparseBuffer<data=memref<1x1x14x14xf16, {order = #NHWC, strides = [25088, 1, 1792, 128]}, @DDR>, sparsity_map=memref<1x1x14x14xi1, {order = #NHWC, strides = [25088, 1, 1792, 128]}, @DDR>>) outputs([[SUBVIEW_4]] : !VPUIP.SparseBuffer<data=memref<1x1x14x14xf16, {order = #NHWC, strides = [25480, 1, 1820, 130]}, @DDR>, sparsity_map=memref<1x1x14x14xi1, {order = #NHWC, strides = [25480, 1, 1820, 130]}, @DDR>>) -> !VPUIP.SparseBuffer<data=memref<1x1x14x14xf16, {order = #NHWC, strides = [25480, 1, 1820, 130]}, @DDR>, sparsity_map=memref<1x1x14x14xi1, {order = #NHWC, strides = [25480, 1, 1820, 130]}, @DDR>>

    // CHECK:      [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[COPY_1]], [[COPY_2]], [[COPY_3]] : !VPUIP.SparseBuffer<data=memref<1x1x14x14xf16, {order = #NHWC, strides = [25480, 1, 1820, 130]}, @DDR>, sparsity_map=memref<1x1x14x14xi1, {order = #NHWC, strides = [25480, 1, 1820, 130]}, @DDR>>, !VPUIP.SparseBuffer<data=memref<1x128x14x14xf16, {order = #NHWC, strides = [25480, 1, 1820, 130]}, @DDR>, sparsity_map=memref<1x128x14x14xi1, {order = #NHWC, strides = [25480, 1, 1820, 130]}, @DDR>>, !VPUIP.SparseBuffer<data=memref<1x1x14x14xf16, {order = #NHWC, strides = [25480, 1, 1820, 130]}, @DDR>, sparsity_map=memref<1x1x14x14xi1, {order = #NHWC, strides = [25480, 1, 1820, 130]}, @DDR>>) outputs([[SPARSE_2]] : !VPUIP.SparseBuffer<data=memref<1x130x14x14xf16, #NHWC, @DDR>, sparsity_map=memref<1x130x14x14xi1, #NHWC, @DDR>>) -> !VPUIP.SparseBuffer<data=memref<1x130x14x14xf16, #NHWC, @DDR>, sparsity_map=memref<1x130x14x14xi1, #NHWC, @DDR>>
    // CHECK:       return [[CONCAT]] : !VPUIP.SparseBuffer<data=memref<1x130x14x14xf16, #NHWC, @DDR>, sparsity_map=memref<1x130x14x14xi1, #NHWC, @DDR>>
}
