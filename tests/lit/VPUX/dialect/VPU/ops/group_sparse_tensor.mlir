//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @ParsePrintGroupSparseTensorPartial
func @ParsePrintGroupSparseTensorPartial(%arg0: tensor<1x32x16x16xf16>) -> tensor<1x32x16x16xf16> {
    %0 = const.Declare tensor<1x32x16x16xi1> = dense<1> : tensor<1x32x16x16xi1>
    %1 = VPU.GroupSparseTensor(%arg0, %0)
            -> !VPU.SparseTensor<data=tensor<1x32x16x16xf16>, sparsity_map=tensor<1x32x16x16xi1>>
    %2:2 = builtin.unrealized_conversion_cast %1 : !VPU.SparseTensor<data=tensor<1x32x16x16xf16>, sparsity_map=tensor<1x32x16x16xi1>>
            to tensor<1x32x16x16xf16>, tensor<1x32x16x16xi1>
    return %2#0 : tensor<1x32x16x16xf16>

    // CHECK:       [[SM:%.*]] = const.Declare tensor<1x32x16x16xi1> = dense<true> : tensor<1x32x16x16xi1>
    // CHECK:       [[VAL0:%.*]] = VPU.GroupSparseTensor(%arg0, [[SM]])
    // CHECK-SAME:                 -> !VPU.SparseTensor<data=tensor<1x32x16x16xf16>, sparsity_map=tensor<1x32x16x16xi1>>
    // CHECK:       [[VAL1:%.*]]:2 = builtin.unrealized_conversion_cast
    // CHECK-SAME:                      [[VAL0]] : !VPU.SparseTensor<data=tensor<1x32x16x16xf16>, sparsity_map=tensor<1x32x16x16xi1>>
    // CHECK-SAME:                      to tensor<1x32x16x16xf16>, tensor<1x32x16x16xi1>
    // CHECK:       return [[VAL1]]#0 : tensor<1x32x16x16xf16>
}

// -----

// CHECK-LABEL: @ParsePrintGroupSparseTensor
func @ParsePrintGroupSparseTensor(%arg0: tensor<1x32x16x16xf16>) -> tensor<1x32x16x16xf16> {
    %0 = const.Declare tensor<1x32x16x16xi1> = dense<1> : tensor<1x32x16x16xi1>
    %1 = const.Declare tensor<1x32x1x1xi32> = dense<1> : tensor<1x32x1x1xi32>
    %2 = VPU.GroupSparseTensor(%arg0, %0, %1)
            -> !VPU.SparseTensor<data=tensor<1x32x16x16xf16>, sparsity_map=tensor<1x32x16x16xi1>, storage_element_table=tensor<1x32x1x1xi32>>
    %3:3 = builtin.unrealized_conversion_cast %2
            : !VPU.SparseTensor<data=tensor<1x32x16x16xf16>, sparsity_map=tensor<1x32x16x16xi1>, storage_element_table=tensor<1x32x1x1xi32>>
            to tensor<1x32x16x16xf16>, tensor<1x32x16x16xi1>, tensor<1x32x1x1xi32>
    return %3#0 : tensor<1x32x16x16xf16>

    // CHECK-DAG:   [[SM:%.*]] = const.Declare tensor<1x32x16x16xi1> = dense<true> : tensor<1x32x16x16xi1>
    // CHECK-DAG:   [[SE:%.*]] = const.Declare tensor<1x32x1x1xi32> = dense<1> : tensor<1x32x1x1xi32>
    // CHECK:       [[VAL0:%.*]] = VPU.GroupSparseTensor(%arg0, [[SM]], [[SE]])
    // CHECK-SAME:                 -> !VPU.SparseTensor<data=tensor<1x32x16x16xf16>, sparsity_map=tensor<1x32x16x16xi1>, storage_element_table=tensor<1x32x1x1xi32>>
    // CHECK:       [[VAL1:%.*]]:3 = builtin.unrealized_conversion_cast
    // CHECK-SAME:                      [[VAL0]] : !VPU.SparseTensor<data=tensor<1x32x16x16xf16>, sparsity_map=tensor<1x32x16x16xi1>, storage_element_table=tensor<1x32x1x1xi32>>
    // CHECK-SAME:                      to tensor<1x32x16x16xf16>, tensor<1x32x16x16xi1>, tensor<1x32x1x1xi32>
    // CHECK:       return [[VAL1]]#0 : tensor<1x32x16x16xf16>
}

// -----

// CHECK-LABEL: @ParsePrintGroupSparseTensorWeights
func @ParsePrintGroupSparseTensorWeights(%arg0: tensor<16x32x1x1xf16>)
        -> !VPU.SparseTensor<data=tensor<16x32x1x1xf16>, sparsity_map=tensor<16x1x1x128xi1>, is_weights,
                             #VPU.CompressionScheme<axis = 0 : i64, numElems = dense<0> : tensor<16xi64>, alignment = 16 : i64>> {
    %cst_sm = const.Declare tensor<16x1x1x128xi1> = dense<1> : tensor<16x1x1x128xi1>
    %2 = VPU.GroupSparseTensor(%arg0, %cst_sm) {
            compression_scheme = #VPU.CompressionScheme<axis = 0 : i64, numElems = dense<0> : tensor<16xi64>, alignment = 16 : i64>,
            is_weights
        } -> !VPU.SparseTensor<data=tensor<16x32x1x1xf16>, sparsity_map=tensor<16x1x1x128xi1>, is_weights,
                               #VPU.CompressionScheme<axis = 0 : i64, numElems = dense<0> : tensor<16xi64>, alignment = 16 : i64>>
    return %2 : !VPU.SparseTensor<data=tensor<16x32x1x1xf16>, sparsity_map=tensor<16x1x1x128xi1>, is_weights,
                                  #VPU.CompressionScheme<axis = 0 : i64, numElems = dense<0> : tensor<16xi64>, alignment = 16 : i64>>

    // CHECK-DAG:   [[CST_SM:%.*]] = const.Declare tensor<16x1x1x128xi1> = dense<true> : tensor<16x1x1x128xi1>
    // CHECK:       [[VAL0:%.*]] = VPU.GroupSparseTensor(%arg0, [[CST_SM]]) {
    // CHECK:            compression_scheme = #VPU.CompressionScheme<axis = 0 : i64, numElems = dense<0> : tensor<16xi64>, alignment = 16 : i64>,
    // CHECK:            is_weights
    // CHECK:        } -> !VPU.SparseTensor<data=tensor<16x32x1x1xf16>, sparsity_map=tensor<16x1x1x128xi1>, is_weights,
    // CHECK-SAME:                          #VPU.CompressionScheme<axis = 0 : i64, numElems = dense<0> : tensor<16xi64>, alignment = 16 : i64>>
    // CHECK:       return [[VAL0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CanonicalizeSlice
func @CanonicalizeSlice(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>)
        -> !VPU.SparseTensor<data=tensor<64x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x384xi1>, is_weights> {
    %cst = const.Declare tensor<128x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %cst_sm = const.Declare tensor<128x1x1x384xi1> = dense<1.000000e+00> : tensor<128x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %cst_sparse = VPU.GroupSparseTensor(%cst, %cst_sm) {is_weights}
        -> !VPU.SparseTensor<data=tensor<128x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<128x1x1x384xi1>, is_weights>

    %cst_sparse_slice = VPU.Slice %cst_sparse [0, 0, 0, 0] [64, 32, 3, 3] :
        !VPU.SparseTensor<data=tensor<128x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<128x1x1x384xi1>, is_weights> to
        !VPU.SparseTensor<data=tensor<64x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x384xi1>, is_weights>

    return %cst_sparse_slice : !VPU.SparseTensor<data=tensor<64x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x384xi1>, is_weights>

    // CHECK-DAG:  [[WEIGHTS:%.*]] = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> =
    // CHECK-SAME:     dense<1.000000e+00> : tensor<128x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [64, 32, 3, 3]>, #const.Sparsify<false, dense<288> : tensor<64xi64>>]
    // CHECK-DAG:  [[WEIGHTS_SM:%.*]] = const.Declare tensor<64x1x1x384xi1> =
    // CHECK-SAME:     dense<1.000000e+00> : tensor<128x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [64, 32, 3, 3]>, #const.GetSparsityMap]
    // CHECK:      [[WEIGHTS_SPARSE:%.*]] = VPU.GroupSparseTensor([[WEIGHTS]], [[WEIGHTS_SM]]) {is_weights}
    // CHECK-SAME:     -> !VPU.SparseTensor<data=tensor<64x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x384xi1>, is_weights>
    // CHECK:      return [[WEIGHTS_SPARSE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CanonicalizeMultipleSlice
func @CanonicalizeMultipleSlice(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>)
        -> (!VPU.SparseTensor<data=tensor<64x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x384xi1>, is_weights>,
            !VPU.SparseTensor<data=tensor<64x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x384xi1>, is_weights>) {
    %cst = const.Declare tensor<128x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %cst_sm = const.Declare tensor<128x1x1x384xi1> = dense<1.000000e+00> : tensor<128x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %cst_sparse = VPU.GroupSparseTensor(%cst, %cst_sm) {is_weights}
        -> !VPU.SparseTensor<data=tensor<128x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<128x1x1x384xi1>, is_weights>

    %cst_sparse_slice_0 = VPU.Slice %cst_sparse [0, 0, 0, 0] [64, 32, 3, 3] :
        !VPU.SparseTensor<data=tensor<128x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<128x1x1x384xi1>, is_weights> to
        !VPU.SparseTensor<data=tensor<64x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x384xi1>, is_weights>

    %cst_sparse_slice_1 = VPU.Slice %cst_sparse [64, 0, 0, 0] [64, 32, 3, 3] :
        !VPU.SparseTensor<data=tensor<128x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<128x1x1x384xi1>, is_weights> to
        !VPU.SparseTensor<data=tensor<64x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x384xi1>, is_weights>

    return %cst_sparse_slice_0, %cst_sparse_slice_1
        : !VPU.SparseTensor<data=tensor<64x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x384xi1>, is_weights>,
          !VPU.SparseTensor<data=tensor<64x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x384xi1>, is_weights>

    // CHECK-DAG:  [[WEIGHTS_SLICE_1:%.*]] = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> =
    // CHECK-SAME:     dense<1.000000e+00> : tensor<128x32x3x3xf16>,
    // CHECK-SAME:     [#const.Reorder<#NHWC>, #const.SubView<[64, 0, 0, 0], [64, 32, 3, 3]>, #const.Sparsify<false, dense<288> : tensor<64xi64>>]
    // CHECK-DAG:  [[WEIGHTS_SM_SLICE_1:%.*]] = const.Declare tensor<64x1x1x384xi1> =
    // CHECK-SAME:     dense<1.000000e+00> : tensor<128x32x3x3xf16>,
    // CHECK-SAME:     [#const.Reorder<#NHWC>, #const.SubView<[64, 0, 0, 0], [64, 32, 3, 3]>, #const.GetSparsityMap]

    // CHECK-DAG:  [[WEIGHTS_SLICE_0:%.*]] = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> =
    // CHECK-SAME:     dense<1.000000e+00> : tensor<128x32x3x3xf16>,
    // CHECK-SAME:     [#const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [64, 32, 3, 3]>, #const.Sparsify<false, dense<288> : tensor<64xi64>>]
    // CHECK-DAG:  [[WEIGHTS_SM_SLICE_0:%.*]] = const.Declare tensor<64x1x1x384xi1> =
    // CHECK-SAME:     dense<1.000000e+00> : tensor<128x32x3x3xf16>,
    // CHECK-SAME:     [#const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [64, 32, 3, 3]>, #const.GetSparsityMap]

    // CHECK:      [[WEIGHTS_SPARSE_SLICE_1:%.*]] = VPU.GroupSparseTensor([[WEIGHTS_SLICE_1]], [[WEIGHTS_SM_SLICE_1]]) {is_weights}
    // CHECK-SAME:     -> !VPU.SparseTensor<data=tensor<64x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x384xi1>, is_weights>

    // CHECK:      [[WEIGHTS_SPARSE_SLICE_0:%.*]] = VPU.GroupSparseTensor([[WEIGHTS_SLICE_0]], [[WEIGHTS_SM_SLICE_0]]) {is_weights}
    // CHECK-SAME:     -> !VPU.SparseTensor<data=tensor<64x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x384xi1>, is_weights>

    // CHECK:      return [[WEIGHTS_SPARSE_SLICE_0]], [[WEIGHTS_SPARSE_SLICE_1]]
}
