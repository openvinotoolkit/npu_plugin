//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --multi-cluster-strategy-assignment %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAddAssignedSOHOverlapped
func.func @EltwiseAddAssignedSOHOverlapped(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>, %arg1: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) { op_type = #VPU.eltwise_type<ADD> } :
         tensor<1x32x112x112xf16, {order = #NHWC}>, tensor<1x32x112x112xf16, {order = #NHWC}>
         -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0: tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.Eltwise(%arg0, %arg1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>} -> tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvAssignedSOHForLargeLayer
func.func @ConvAssignedSOHForLargeLayer(%arg0: tensor<1x64x608x608xf16, {order = #NHWC}>) -> tensor<1x80x608x608xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    %cst_0 = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]} -> tensor<1x80x608x608xf16, {order = #NHWC}>
    return %0 : tensor<1x80x608x608xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.+]] = VPU.NCE.Convolution(%arg0, %cst_0, %cst)
    //CHECK-SAME:    {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]}
    //CHECK-SAME:      -> tensor<1x80x608x608xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x80x608x608xf16, {order = #NHWC}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InterpolateNearestAssignedClustering
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x16x1x1xf16, {order = #NHWC}>)
func.func @InterpolateNearestAssignedClustering(%input_data: tensor<1x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x2x2xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x16x2x2xi1> = dense<1> : tensor<1x16x2x2xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = 16, dataShape = [1, 16, 1, 1],
        seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>
    } -> tensor<1x1x2x2xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%input_data, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 16, 2, 2]>
    } -> !VPU.SparseTensor<data=tensor<1x16x1x1xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x16x2x2xi1>,
                           storage_element_table=tensor<1x1x2x2xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        rawFilterShape = [16, 16, 1, 1],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<NEAREST>,
        scales_attr = [2, 2],
        ppe = #VPU.PPETask<clamp_high = 2147483647, clamp_low = 0, lrelu_mult = 1, lrelu_shift = 0, mode = <NOOP>>
    } -> tensor<1x16x2x2xf16, {order = #NHWC}>

    return %interpolate : tensor<1x16x2x2xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x16x2x2xi1> = dense<true> : tensor<1x16x2x2xi1>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      mode = #VPU.nce_interpolate_mode<NEAREST>,
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
    // CHECK-SAME:      ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
    // CHECK-SAME:      rawFilterShape = [16, 16, 1, 1],
    // CHECK-SAME:      scales_attr = [2, 2],
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x16x2x2xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]] : tensor<1x16x2x2xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InterpolateBilinearAssignedClustering
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x16x1x1xf16, {order = #NHWC}>)
func.func @InterpolateBilinearAssignedClustering(%input_data: tensor<1x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x2x2xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x2x2xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x2x2xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x16x3x3xi1> = dense<1> : tensor<1x16x3x3xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = 16, dataShape = [1, 16, 1, 1],
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], offsets = [0, 0, 0, 0], sizes = [1, 16, 3, 3]>
    } -> tensor<1x1x3x3xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%input_data, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            offsets = [0, 0, 0, 0],
            sizes = [1, 16, 3, 3]>
    } -> !VPU.SparseTensor<data=tensor<1x16x1x1xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x16x3x3xi1>,
                           storage_element_table=tensor<1x1x3x3xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], offsets = [0, 0, 0, 0], sizes = [1, 16, 3, 3]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        rawFilterShape = [16, 16, 2, 2],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        scales_attr = [2, 2],
        ppe = #VPU.PPETask<clamp_high = 2147483647, clamp_low = 0, lrelu_mult = 1, lrelu_shift = 0, mode = <NOOP>>
    } -> tensor<1x16x2x2xf16, {order = #NHWC}>

    return %interpolate : tensor<1x16x2x2xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x16x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x2x2xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x16x3x3xi1> = dense<true> : tensor<1x16x3x3xi1>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
    // CHECK-SAME:      ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>
    // CHECK-SAME:      rawFilterShape = [16, 16, 2, 2],
    // CHECK-SAME:      scales_attr = [2, 2],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x2x2xf16, {order = #NHWC}> 
    // CHECK:       return [[OUTPUT]] : tensor<1x16x2x2xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvAssignedSOKForSmallHW
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x1024x4x4xf16, {order = #NHWC}>)
func.func @ConvAssignedSOKForSmallHW(%input_data: tensor<1x1024x4x4xf16, {order = #NHWC}>) -> tensor<1x2048x2x2xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<2048x1024x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<2048x1024x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<2048x1x1x4xsi32> = dense<1> : tensor<2048x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%input_data, %weights, %weights_table) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPETask<mode = <NOOP>,
        clamp_low = 0 : i64, clamp_high = 255 : i64,
        lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        rawFilterShape = [2048, 1024, 1, 1], strides = [2, 2]
    } -> tensor<1x2048x2x2xf16, {order = #NHWC}>

    return %conv : tensor<1x2048x2x2xf16, {order = #NHWC}>

    // CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<2048x1024x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<2048x1024x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK:        [[WEIGHTS_TBL:%.+]] = const.Declare tensor<2048x1x1x4xsi32> = dense<1> : tensor<2048x1x1x4xsi32>
    // CHECK:        [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT_DATA]], [[WEIGHTS]], [[WEIGHTS_TBL]])
    // CHECK-SAME:        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
    // CHECK-SAME:        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:        ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64,
    // CHECK-SAME:        lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:        rawFilterShape = [2048, 1024, 1, 1], strides = [2, 2]} -> tensor<1x2048x2x2xf16, {order = #NHWC}>
    // CHECK:        return [[CONV]] : tensor<1x2048x2x2xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipSubGraphOptimizationForINFCost
func.func @SkipSubGraphOptimizationForINFCost(%input_data: tensor<1x128x72x72xf16, {order = #NHWC}>, %input_data2: tensor<1x64x144x144xf16, {order = #NHWC}>) -> tensor<1x64x144x144xf16, {order = #NHWC}> {
    // conv
    %convWeights = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x128x1x1xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]
    %convWeightsTable = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%input_data, %convWeights, %convWeightsTable) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPETask<mode = <LRELU>,
        clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
        lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        rawFilterShape = [64, 128, 1, 1], strides = [1, 1]
    } -> tensor<1x64x72x72xf16, {order = #NHWC}>

    // transposedConv
    %sparsityMap = const.Declare tensor<1x64x147x147xi1, {order = #NHWC}> = dense<1> : tensor<1x64x147x147xi8>, [#const.Reorder<#NHWC>, #const.ConvertElemType<i1>]
    %storageElement = VPU.StorageElementTable {
        dataElemType = f16,
        dataShape = [1, 64, 72, 72],
        seAttr = #VPU.SEUpsampling<
            factors = [1, 1],
            padding = [2, 2, 2, 2]>,
        seDepth = 1 : i64,
        seSize = 64 : i64
    } -> tensor<1x1x147x147xi32, {order = #NHWC}>
    %input = VPU.GroupSparseTensor(%conv, %sparsityMap, %storageElement) {
        seAttr = #VPU.SEUpsampling<
            factors = [1, 1],
            padding = [2, 2, 2, 2]>
    } -> !VPU.SparseTensor<
            data=tensor<1x64x72x72xf16, {order = #NHWC}>,
            sparsity_map=tensor<1x64x147x147xi1, {order = #NHWC}>,
            storage_element_table=tensor<1x1x147x147xi32, {order = #NHWC}>,
            #VPU.SEUpsampling<
                factors = [1, 1],
                padding = [2, 2, 2, 2]>>

    %weightsCst = const.Declare tensor<64x64x4x4xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x4x4xf16, {order = #NHWC}>, [#const.Sparsify<false>]
    %weightsSparsityMap = const.Declare tensor<64x1x1x1024xi1> = dense<1.0> : tensor<64x64x4x4xf16, {order = #NHWC}>, [#const.GetSparsityMap]
    %weights = VPU.GroupSparseTensor(%weightsCst, %weightsSparsityMap) {
        compression_scheme = #VPU.CompressionScheme<axis = 0 : i64, numElems = dense<16> : tensor<64xi64>, alignment = 16 : i64>, is_weights}
        -> !VPU.SparseTensor<data=tensor<64x64x4x4xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x1024xi1>, is_weights, #VPU.CompressionScheme<axis = 0 : i64, numElems = dense<16> : tensor<64xi64>, alignment = 16 : i64>>

    %weightsTable = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    %transposedConv = VPU.NCE.Convolution(%input, %weights, %weightsTable) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPETask<mode = <NOOP>,
        clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
        lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        rawFilterShape = [64, 64, 4, 4], strides = [1, 1]
    } -> tensor<1x64x144x144xf16, {order = #NHWC}>

    // add
    %add = VPU.NCE.Eltwise(%transposedConv, %input_data2) {
        is_inplace = true,
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPETask<mode = <LRELU>,
        clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
        lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>
    } -> tensor<1x64x144x144xf16, {order = #NHWC}>


    return %add : tensor<1x64x144x144xf16, {order = #NHWC}>

    // CHECK:        [[CONV_WEIGHTS:%.+]] = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x128x1x1xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:        [[CONV_WEIGHTS_TBL:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    // CHECK:        [[CONV:%.+]] = VPU.NCE.Convolution(%arg0, [[CONV_WEIGHTS]], [[CONV_WEIGHTS_TBL]])
    // CHECK-SAME:        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:        ppe = #VPU.PPETask<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
    // CHECK-SAME:        lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:        rawFilterShape = [64, 128, 1, 1], strides = [1, 1]} -> tensor<1x64x72x72xf16, {order = #NHWC}>

    // CHECK:        [[SPARSITY_MAP:%.+]] = const.Declare tensor<1x64x147x147xi1, {order = #NHWC}> = dense<1> : tensor<1x64x147x147xi8>, [#const.Reorder<#NHWC>, #const.ConvertElemType<i1>]
    // CHECK:        [[SE_TBL:%.+]] = VPU.StorageElementTable
    // CHECK-SAME:        {dataElemType = f16,
    // CHECK-SAME:        dataShape = [1, 64, 72, 72],
    // CHECK-SAME:        seAttr = #VPU.SEUpsampling<
    // CHECK-SAME:            factors = [1, 1],
    // CHECK-SAME:            padding = [2, 2, 2, 2]>,
    // CHECK-SAME:            seDepth = 1 : i64, seSize = 64 : i64} -> tensor<1x1x147x147xi32, {order = #NHWC}>
    // CHECK:        [[SPARSE_INPUT:%.+]] = VPU.GroupSparseTensor([[CONV]], [[SPARSITY_MAP]], [[SE_TBL]])
    // CHECK-SAME:        {seAttr = #VPU.SEUpsampling<
    // CHECK-SAME:            factors = [1, 1],
    // CHECK-SAME:            padding = [2, 2, 2, 2]>}
    // CHECK-SAME:        -> !VPU.SparseTensor<
    // CHECK-SAME:            data=tensor<1x64x72x72xf16, {order = #NHWC}>,
    // CHECK-SAME:            sparsity_map=tensor<1x64x147x147xi1, {order = #NHWC}>,
    // CHECK-SAME:            storage_element_table=tensor<1x1x147x147xi32, {order = #NHWC}>,
    // CHECK-SAME:        #VPU.SEUpsampling<
    // CHECK-SAME:            factors = [1, 1],
    // CHECK-SAME:            padding = [2, 2, 2, 2]>>
    // CHECK:        [[DECONV_WEIGHTS_CST:%.+]] = const.Declare tensor<64x64x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x4x4xf16, {order = #NHWC}>, [#const.Sparsify<false>]
    // CHECK:        [[DECONV_WEIGHTS_SM:%.+]] = const.Declare tensor<64x1x1x1024xi1> = dense<1.000000e+00> : tensor<64x64x4x4xf16, {order = #NHWC}>, [#const.GetSparsityMap]
    // CHECK:        [[DECONV_WEIGHTS:%.+]] = VPU.GroupSparseTensor([[DECONV_WEIGHTS_CST]], [[DECONV_WEIGHTS_SM]]) {
    // CHECK-SAME:        compression_scheme = #VPU.CompressionScheme<axis = 0 : i64, numElems = dense<16> : tensor<64xi64>, alignment = 16 : i64>, is_weights}
    // CHECK-SAME:        -> !VPU.SparseTensor<data=tensor<64x64x4x4xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x1024xi1>, is_weights, #VPU.CompressionScheme<axis = 0 : i64, numElems = dense<16> : tensor<64xi64>, alignment = 16 : i64>>
    // CHECK:        [[DECONV_WEIGHTS_TBL:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    // CHECK:        [[DECONV:%.+]] = VPU.NCE.Convolution([[SPARSE_INPUT]], [[DECONV_WEIGHTS]], [[DECONV_WEIGHTS_TBL]])
    // CHECK-SAME:        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:        ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
    // CHECK-SAME:        lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:        rawFilterShape = [64, 64, 4, 4], strides = [1, 1]} -> tensor<1x64x144x144xf16, {order = #NHWC}>

    // CHECK:        [[ADD:%.+]] = VPU.NCE.Eltwise([[DECONV]], %arg1)
    // CHECK-SAME:        {is_inplace = true,
    // CHECK-SAME:        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:        op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:        ppe = #VPU.PPETask<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
    // CHECK-SAME:        lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>}
    // CHECK-SAME:        -> tensor<1x64x144x144xf16, {order = #NHWC}>

    // CHECK:        return [[ADD]] : tensor<1x64x144x144xf16, {order = #NHWC}>
}
