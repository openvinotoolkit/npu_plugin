//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX compilation-mode=DefaultHW" --lower-ops-to-se-nce %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @InterpolateNearest([[INPUT_DATA:%.+]]: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x6x6xf16, {order = #NHWC}> {
func.func @InterpolateNearest(%arg0: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x6x6xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <ASYMMETRIC>,
                                   cube_coeff = -7.500000e-01,
                                   mode = <NEAREST>,
                                   nearest_mode = <FLOOR>,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   shape_calc_mode = <SCALES>>,
            axes_attr = [2, 3],
            scales_attr = [2.000000e+00, 2.000000e+00],
            sizes_attr = [6, 6],
            operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>
        } : tensor<1x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x6x6xf16, {order = #NHWC}>

    return %0 : tensor<1x16x6x6xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 16, 3, 3],
    // CHECK-SAME:      seAttr = #VPU.SEInterpolate<mode = "NEAREST", nearest_mode = "FLOOR", coordinate_transformation_mode = "ASYMMETRIC",
    // CHECK-SAME:               scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = <<NULL ATTRIBUTE>>, sizes = <<NULL ATTRIBUTE>>>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = 16 : i64}
    // CHECK-SAME:      -> tensor<1x1x6x6xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x16x6x6xi1, {order = #NHWC}> =
    // CHECK-SAME:      dense<1> : tensor<1x16x6x6xi8>, [#const.Reorder<#NHWC>, #const.ConvertElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      {seAttr = #VPU.SEInterpolate<mode = "NEAREST", nearest_mode = "FLOOR", coordinate_transformation_mode = "ASYMMETRIC",
    // CHECK-SAME:       scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = <<NULL ATTRIBUTE>>, sizes = <<NULL ATTRIBUTE>>>}
    // CHECK-SAME:    -> !VPU.SparseTensor<data=tensor<1x16x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x16x6x6xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x6x6xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEInterpolate<mode = "NEAREST", nearest_mode = "FLOOR", coordinate_transformation_mode = "ASYMMETRIC",
    // CHECK-SAME:                                            scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                                            offsets = <<NULL ATTRIBUTE>>, sizes = <<NULL ATTRIBUTE>>>>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<16x16x1x1xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 81921, 0]]], [[[32, 0, 81921, 0]]], [[[64, 0, 81921, 0]]], [[[96, 0, 81921, 0]]],
    // CHECK-SAME{LITERAL}:          [[[128, 0, 81921, 0]]], [[[160, 0, 81921, 0]]], [[[192, 0, 81921, 0]]], [[[224, 0, 81921, 0]]],
    // CHECK-SAME{LITERAL}:          [[[256, 0, 81921, 0]]], [[[288, 0, 81921, 0]]], [[[320, 0, 81921, 0]]], [[[352, 0, 81921, 0]]],
    // CHECK-SAME{LITERAL}:          [[[384, 0, 81921, 0]]], [[[416, 0, 81921, 0]]], [[[448, 0, 81921, 0]]], [[[480, 0, 81921, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      {mode = "NEAREST",
    // CHECK-SAME:       rawFilterShape = [16, 16, 1, 1]}
    // CHECK-SAME:      -> tensor<1x16x6x6xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @InterpolateBilinear([[INPUT_DATA:%.+]]: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x6x6xf16, {order = #NHWC}> {
func.func @InterpolateBilinear(%arg0: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x6x6xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <ASYMMETRIC>,
                                   cube_coeff = -7.500000e-01,
                                   mode = <LINEAR>,
                                   nearest_mode = <FLOOR>,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   shape_calc_mode = <SCALES>>,
            axes_attr = [2, 3],
            scales_attr = [2.000000e+00, 2.000000e+00],
            sizes_attr = [6, 6],
            operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>
        } : tensor<1x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x6x6xf16, {order = #NHWC}>

    return %0 : tensor<1x16x6x6xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 16, 3, 3],
    // CHECK-SAME:      seAttr = #VPU.SEInterpolate<mode = "BILINEAR", nearest_mode = <<NULL ATTRIBUTE>>, coordinate_transformation_mode = "ASYMMETRIC",
    // CHECK-SAME:               scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = <<NULL ATTRIBUTE>>, sizes = <<NULL ATTRIBUTE>>>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = 16 : i64}
    // CHECK-SAME:      -> tensor<1x1x7x7xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x16x7x7xi1, {order = #NHWC}> =
    // CHECK-SAME:      dense<1> : tensor<1x16x7x7xi8>, [#const.Reorder<#NHWC>, #const.ConvertElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      {seAttr = #VPU.SEInterpolate<mode = "BILINEAR", nearest_mode = <<NULL ATTRIBUTE>>, coordinate_transformation_mode = "ASYMMETRIC",
    // CHECK-SAME:       scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = <<NULL ATTRIBUTE>>, sizes = <<NULL ATTRIBUTE>>>}
    // CHECK-SAME:    -> !VPU.SparseTensor<data=tensor<1x16x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x16x7x7xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x7x7xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEInterpolate<mode = "BILINEAR", nearest_mode = <<NULL ATTRIBUTE>>, coordinate_transformation_mode = "ASYMMETRIC",
    // CHECK-SAME:                                            scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                                            offsets = <<NULL ATTRIBUTE>>, sizes = <<NULL ATTRIBUTE>>>>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x2x2xf16, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<16x16x2x2xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 81921, 0]]], [[[128, 0, 81921, 0]]], [[[256, 0, 81921, 0]]], [[[384, 0, 81921, 0]]],
    // CHECK-SAME{LITERAL}:          [[[512, 0, 81921, 0]]], [[[640, 0, 81921, 0]]], [[[768, 0, 81921, 0]]], [[[896, 0, 81921, 0]]],
    // CHECK-SAME{LITERAL}:          [[[1024, 0, 81921, 0]]], [[[1152, 0, 81921, 0]]], [[[1280, 0, 81921, 0]]], [[[1408, 0, 81921, 0]]],
    // CHECK-SAME{LITERAL}:          [[[1536, 0, 81921, 0]]], [[[1664, 0, 81921, 0]]], [[[1792, 0, 81921, 0]]], [[[1920, 0, 81921, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      {mode = "BILINEAR",
    // CHECK-SAME:       rawFilterShape = [16, 16, 2, 2]}
    // CHECK-SAME:      -> tensor<1x16x6x6xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0078407292272530352:128>
// CHECK: !qElemType0 = !quant.uniform<u8:f16, 0.0078407292272530352:128>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 2.500000e-01>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @InterpolateBilinearQuantized([[INPUT_DATA:%.+]]: tensor<1x16x3x3x!qElemType0, {order = #NHWC}>) -> tensor<1x16x6x6x!qElemType0, {order = #NHWC}> {
func.func @InterpolateBilinearQuantized(%arg0: tensor<1x16x3x3x!qElemType, {order = #NHWC}>) -> tensor<1x16x6x6x!qElemType, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <ASYMMETRIC>,
                                   cube_coeff = -7.500000e-01,
                                   mode = <LINEAR>,
                                   nearest_mode = <FLOOR>,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   shape_calc_mode = <SCALES>>,
            axes_attr = [2, 3],
            scales_attr = [2.000000e+00, 2.000000e+00],
            sizes_attr = [6, 6],
            operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>
        } : tensor<1x16x3x3x!qElemType, {order = #NHWC}> -> tensor<1x16x6x6x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x16x6x6x!qElemType, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = !qElemType0, dataShape = [1, 16, 3, 3],
    // CHECK-SAME:      seAttr = #VPU.SEInterpolate<mode = "BILINEAR", nearest_mode = <<NULL ATTRIBUTE>>, coordinate_transformation_mode = "ASYMMETRIC",
    // CHECK-SAME:               scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = <<NULL ATTRIBUTE>>, sizes = <<NULL ATTRIBUTE>>>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = 16 : i64}
    // CHECK-SAME:      -> tensor<1x1x7x7xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x16x7x7xi1, {order = #NHWC}> =
    // CHECK-SAME:      dense<1> : tensor<1x16x7x7xi8>, [#const.Reorder<#NHWC>, #const.ConvertElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      {seAttr = #VPU.SEInterpolate<mode = "BILINEAR", nearest_mode = <<NULL ATTRIBUTE>>, coordinate_transformation_mode = "ASYMMETRIC",
    // CHECK-SAME:       scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = <<NULL ATTRIBUTE>>, sizes = <<NULL ATTRIBUTE>>>}
    // CHECK-SAME:    -> !VPU.SparseTensor<data=tensor<1x16x3x3x!qElemType0, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x16x7x7xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x7x7xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEInterpolate<mode = "BILINEAR", nearest_mode = <<NULL ATTRIBUTE>>, coordinate_transformation_mode = "ASYMMETRIC",
    // CHECK-SAME:                                            scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                                            offsets = <<NULL ATTRIBUTE>>, sizes = <<NULL ATTRIBUTE>>>>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x2x2x!qElemType1, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<16x16x2x2xf32>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>, #const.Reorder<#NHWC>]
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1073762305, 0]]], [[[64, 0, 1073762305, 0]]], [[[128, 0, 1073762305, 0]]], [[[192, 0, 1073762305, 0]]],
    // CHECK-SAME{LITERAL}:          [[[256, 0, 1073762305, 0]]], [[[320, 0, 1073762305, 0]]], [[[384, 0, 1073762305, 0]]], [[[448, 0, 1073762305, 0]]],
    // CHECK-SAME{LITERAL}:          [[[512, 0, 1073762305, 0]]], [[[576, 0, 1073762305, 0]]], [[[640, 0, 1073762305, 0]]], [[[704, 0, 1073762305, 0]]],
    // CHECK-SAME{LITERAL}:          [[[768, 0, 1073762305, 0]]], [[[832, 0, 1073762305, 0]]], [[[896, 0, 1073762305, 0]]], [[[960, 0, 1073762305, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      {mode = "BILINEAR",
    // CHECK-SAME:       rawFilterShape = [16, 16, 2, 2]}
    // CHECK-SAME:      -> tensor<1x16x6x6x!qElemType0, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]]
}
