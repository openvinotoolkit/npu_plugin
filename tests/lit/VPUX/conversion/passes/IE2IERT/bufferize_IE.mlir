// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --bufferize-IE %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

func @SingleLayer(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
    %prob = IE.SoftMax(%arg0) {axisInd = 1} : tensor<1x1000xf16> -> tensor<1x1000xf16>
    return %prob : tensor<1x1000xf16>

    // CHECK: [[VAR0:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<1x1000xf16> to memref<1x1000xf16>

    // CHECK: [[VAR1:%.*]] = memref.alloc() : memref<1x1000xf16>
    // CHECK: [[VAR2:%.*]] = IERT.SoftMax
    // CHECK-SAME:      axisInd = 1
    // CHECK-SAME:      inputs([[VAR0]] : memref<1x1000xf16>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<1x1000xf16>)

    // CHECK: [[VAR3:%.*]] = builtin.unrealized_conversion_cast [[VAR2]] : memref<1x1000xf16> to tensor<1x1000xf16>
    // CHECK: return [[VAR3]] : tensor<1x1000xf16>
}

// -----

func @ConstantLayer() -> tensor<1x2x2x2xf16> {
    %0 = const.Declare tensor<1x2x2x2xf16> =
      #const.Content<dense<1.0> : tensor<1x2x2x2xf32>, [#const.ConvertElemType<f16>]>
    return %0 : tensor<1x2x2x2xf16>

    // CHECK:       [[VAR0:%.*]] = const.Declare memref<1x2x2x2xf16> =
    // CHECK-SAME:      #const.Content<dense<1.000000e+00> : tensor<1x2x2x2xf32>, [#const.ConvertElemType<f16>]>

    // CHECK:       [[VAR1:%.*]] = builtin.unrealized_conversion_cast [[VAR0]] : memref<1x2x2x2xf16> to tensor<1x2x2x2xf16>
    // CHECK:       return [[VAR1]] : tensor<1x2x2x2xf16>
}

// -----

func @Reshape(%arg0 : tensor<1x512x1x1xf32>) -> tensor<1x512xf32> {
    %0 = IE.Reshape(%arg0) { shape_value = [1, 512] } : tensor<1x512x1x1xf32> -> tensor<1x512xf32>
    return %0 : tensor<1x512xf32>

    // CHECK: [[VAR0:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<1x512x1x1xf32> to memref<1x512x1x1xf32>

    // CHECK: [[VAR1:%.*]] = IERT.GenericReshape inputs([[VAR0]] : memref<1x512x1x1xf32>) -> memref<1x512xf32>

    // CHECK: [[VAR2:%.*]] = builtin.unrealized_conversion_cast [[VAR1]] : memref<1x512xf32> to tensor<1x512xf32>
    // CHECK: return [[VAR2]] : tensor<1x512xf32>
}

// -----

func @Split(%tensor: tensor<2x6x4x2xf32>) -> (tensor<1x6x4x2xf32>, tensor<1x6x4x2xf32>) {
    %0:2 = IE.Split(%tensor) {num_splits = 2, axis_value = 0} : tensor<2x6x4x2xf32> -> tensor<1x6x4x2xf32>, tensor<1x6x4x2xf32>
    return %0#0, %0#1 : tensor<1x6x4x2xf32>, tensor<1x6x4x2xf32>

    // CHECK:       [[BUFFER:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<2x6x4x2xf32> to memref<2x6x4x2xf32>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x6x4x2xf32>
    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<1x6x4x2xf32>

    // CHECK:       [[VAR2:%.*]] = IERT.SubView [[BUFFER]] [0, 0, 0, 0] [1, 6, 4, 2]
    // CHECK-SAME:      : memref<2x6x4x2xf32> to memref<1x6x4x2xf32>
    // CHECK:       [[VAR4:%.*]] = IERT.Copy inputs([[VAR2]] : memref<1x6x4x2xf32>)
    // CHECK-SAME:      outputs([[VAR0]] : memref<1x6x4x2xf32>)
    // CHECK:       [[VAR3:%.*]] = IERT.SubView [[BUFFER]] [1, 0, 0, 0] [1, 6, 4, 2]
    // CHECK-SAME:      : memref<2x6x4x2xf32> to memref<1x6x4x2xf32>
    // CHECK:       [[VAR5:%.*]] = IERT.Copy inputs([[VAR3]] : memref<1x6x4x2xf32>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<1x6x4x2xf32>)

    // CHECK:       [[OUT0:%.*]] = builtin.unrealized_conversion_cast [[VAR4]] : memref<1x6x4x2xf32> to tensor<1x6x4x2xf32>
    // CHECK:       [[OUT1:%.*]] = builtin.unrealized_conversion_cast [[VAR5]] : memref<1x6x4x2xf32> to tensor<1x6x4x2xf32>
    // CHECK:       return [[OUT0]], [[OUT1]] : tensor<1x6x4x2xf32>, tensor<1x6x4x2xf32>
}

// -----

func @Concat(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x2x3x4xf32>) -> tensor<1x4x3x4xf32> {
    %0 = IE.Concat(%arg0, %arg1) {per_axis = {axis = 1}} :
        tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32> -> tensor<1x4x3x4xf32>
    return %0 : tensor<1x4x3x4xf32>

    // CHECK:       [[VAR0:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<1x2x3x4xf32> to memref<1x2x3x4xf32>
    // CHECK:       [[VAR1:%.*]] = builtin.unrealized_conversion_cast %arg1 : tensor<1x2x3x4xf32> to memref<1x2x3x4xf32>

    // CHECK:       [[VAR2:%.*]] = memref.alloc() : memref<1x4x3x4xf32>

    // CHECK:       [[VAR3:%.*]] = IERT.SubView [[VAR2]] [0, 0, 0, 0] [1, 2, 3, 4]
    // CHECK-SAME:      : memref<1x4x3x4xf32> to memref<1x2x3x4xf32, {order = #NCHW, strides = [48, 12, 4, 1]}>
    // CHECK:       [[VAR4:%.*]] = IERT.Copy inputs([[VAR0]] : memref<1x2x3x4xf32>)
    // CHECK-SAME:      outputs([[VAR3]] : memref<1x2x3x4xf32, {order = #NCHW, strides = [48, 12, 4, 1]}>)

    // CHECK:       [[VAR5:%.*]] = IERT.SubView [[VAR2]] [0, 2, 0, 0] [1, 2, 3, 4]
    // CHECK-SAME:      : memref<1x4x3x4xf32> to memref<1x2x3x4xf32, {order = #NCHW, strides = [48, 12, 4, 1]}>
    // CHECK:       [[VAR6:%.*]] = IERT.Copy inputs([[VAR1]] : memref<1x2x3x4xf32>)
    // CHECK-SAME:      outputs([[VAR5]] : memref<1x2x3x4xf32, {order = #NCHW, strides = [48, 12, 4, 1]}>)

    // CHECK:       [[VAR7:%.*]] = IERT.ConcatView
    // CHECK-SAME:      inputs([[VAR4]], [[VAR6]] :
    // CHECK-SAME:          memref<1x2x3x4xf32, {order = #NCHW, strides = [48, 12, 4, 1]}>,
    // CHECK-SAME:          memref<1x2x3x4xf32, {order = #NCHW,  strides = [48, 12, 4, 1]}>)
    // CHECK-SAME:      outputs([[VAR2]] : memref<1x4x3x4xf32>)

    // CHECK:       [[VAR8:%.*]] = builtin.unrealized_conversion_cast [[VAR7]] : memref<1x4x3x4xf32> to tensor<1x4x3x4xf32>
    // CHECK:       return [[VAR8]] : tensor<1x4x3x4xf32>
}

// -----

func @ConcatWithStride(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x2x3x4xf32>) -> tensor<1x4x3x4xf32> {
    %0 = IE.Concat(%arg0, %arg1) {per_axis = {axis = 1, offset = 1, stride = 2}} :
        tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32> -> tensor<1x4x3x4xf32>
    return %0 : tensor<1x4x3x4xf32>

    // CHECK:       [[VAR0:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<1x2x3x4xf32> to memref<1x2x3x4xf32>
    // CHECK:       [[VAR1:%.*]] = builtin.unrealized_conversion_cast %arg1 : tensor<1x2x3x4xf32> to memref<1x2x3x4xf32>

    // CHECK:       [[VAR2:%.*]] = memref.alloc() : memref<1x4x3x4xf32>

    // CHECK:       [[VAR3:%.*]] = IERT.SubView [[VAR2]] [0, 0, 0, 0] [1, 2, 3, 4] [1, 2, 1, 1]
    // CHECK-SAME:      : memref<1x4x3x4xf32> to memref<1x2x3x4xf32, {order = #NCHW, strides = [48, 24, 4, 1]}>
    // CHECK:       [[VAR4:%.*]] = IERT.Copy inputs([[VAR0]] : memref<1x2x3x4xf32>)
    // CHECK-SAME:      outputs([[VAR3]] : memref<1x2x3x4xf32, {order = #NCHW, strides = [48, 24, 4, 1]}>)

    // CHECK:       [[VAR5:%.*]] = IERT.SubView [[VAR2]] [0, 1, 0, 0] [1, 2, 3, 4] [1, 2, 1, 1]
    // CHECK-SAME:      : memref<1x4x3x4xf32> to memref<1x2x3x4xf32, {order = #NCHW, strides = [48, 24, 4, 1]}>
    // CHECK:       [[VAR6:%.*]] = IERT.Copy inputs([[VAR1]] : memref<1x2x3x4xf32>)
    // CHECK-SAME:      outputs([[VAR5]] : memref<1x2x3x4xf32, {order = #NCHW, strides = [48, 24, 4, 1]}>)

    // CHECK:       [[VAR7:%.*]] = IERT.ConcatView
    // CHECK-SAME:      inputs([[VAR4]], [[VAR6]] :
    // CHECK-SAME:          memref<1x2x3x4xf32, {order = #NCHW, strides = [48, 24, 4, 1]}>,
    // CHECK-SAME:          memref<1x2x3x4xf32, {order = #NCHW, strides = [48, 24, 4, 1]}>)
    // CHECK-SAME:      outputs([[VAR2]] : memref<1x4x3x4xf32>) -> memref<1x4x3x4xf32>

    // CHECK:       [[VAR8:%.*]] = builtin.unrealized_conversion_cast [[VAR7]] : memref<1x4x3x4xf32> to tensor<1x4x3x4xf32>
    // CHECK:       return [[VAR8]] : tensor<1x4x3x4xf32>
}

// -----

func @ExpandToSubview(%arg0: tensor<1x3x4x4xf16>) -> tensor<1x8x4x4xf16> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 5, 0, 0]} : tensor<1x3x4x4xf16> -> tensor<1x8x4x4xf16>
    return %0 : tensor<1x8x4x4xf16>

    // CHECK:       [[VAR0:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<1x3x4x4xf16> to memref<1x3x4x4xf16>

    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<1x8x4x4xf16>

    // CHECK:       [[VIEW1:%.*]] = IERT.SubView [[VAR1]] [0, 0, 0, 0] [1, 3, 4, 4]
    // CHECK-SAME:      : memref<1x8x4x4xf16> to memref<1x3x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>
    // CHECK:       [[COPY1:%.*]] = IERT.Copy inputs([[VAR0]] : memref<1x3x4x4xf16>)
    // CHECK-SAME:      outputs([[VIEW1]] : memref<1x3x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>)

    // CHECK:       [[VIEW2:%.*]] = IERT.SubView [[VAR1]] [0, 3, 0, 0] [1, 3, 4, 4]
    // CHECK-SAME:      : memref<1x8x4x4xf16> to memref<1x3x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>
    // CHECK:       [[COPY2:%.*]] = IERT.Copy inputs([[VAR0]] : memref<1x3x4x4xf16>)
    // CHECK-SAME:      outputs([[VIEW2]] : memref<1x3x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>)

    // CHECK:       [[VIEW_IN:%.*]] = IERT.SubView [[VAR0]] [0, 0, 0, 0] [1, 2, 4, 4]
    // CHECK-SAME:      : memref<1x3x4x4xf16> to memref<1x2x4x4xf16, {order = #NCHW, strides = [48, 16, 4, 1]}>
    // CHECK:       [[VIEW_TAIL:%.*]] = IERT.SubView [[VAR1]] [0, 6, 0, 0] [1, 2, 4, 4]
    // CHECK-SAME:      : memref<1x8x4x4xf16> to memref<1x2x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>
    // CHECK:       [[COPY3:%.*]] = IERT.Copy inputs([[VIEW_IN]] : memref<1x2x4x4xf16, {order = #NCHW, strides = [48, 16, 4, 1]}>)
    // CHECK-SAME:      outputs([[VIEW_TAIL]] : memref<1x2x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>)

    // CHECK:       [[OUT:%.*]] = IERT.ConcatView
    // CHECK-SAME:      inputs([[COPY1]], [[COPY2]], [[COPY3]] :
    // CHECK-SAME:          memref<1x3x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>,
    // CHECK-SAME:          memref<1x3x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>,
    // CHECK-SAME:          memref<1x2x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<1x8x4x4xf16>) -> memref<1x8x4x4xf16>

    // CHECK:       [[VAR10:%.*]] = builtin.unrealized_conversion_cast [[OUT]] : memref<1x8x4x4xf16> to tensor<1x8x4x4xf16>
    // CHECK:       return [[VAR10]] : tensor<1x8x4x4xf16>
}

// -----

func @ExpandToSubviewWithoutTail(%arg0: tensor<1x4x4x4xf16>) -> tensor<1x8x4x4xf16> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 4, 0, 0]} : tensor<1x4x4x4xf16> -> tensor<1x8x4x4xf16>
    return %0 : tensor<1x8x4x4xf16>

    // CHECK:       [[VAR0:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<1x4x4x4xf16> to memref<1x4x4x4xf16>

    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<1x8x4x4xf16>

    // CHECK:       [[VIEW1:%.*]] = IERT.SubView [[VAR1]] [0, 0, 0, 0] [1, 4, 4, 4]
    // CHECK-SAME:      : memref<1x8x4x4xf16> to memref<1x4x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>
    // CHECK:       [[COPY1:%.*]] = IERT.Copy inputs([[VAR0]] : memref<1x4x4x4xf16>)
    // CHECK-SAME:      outputs([[VIEW1]] : memref<1x4x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>)

    // CHECK:       [[VIEW1:%.*]] = IERT.SubView [[VAR1]] [0, 4, 0, 0] [1, 4, 4, 4]
    // CHECK-SAME:      : memref<1x8x4x4xf16> to memref<1x4x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>
    // CHECK:       [[COPY2:%.*]] = IERT.Copy inputs([[VAR0]] : memref<1x4x4x4xf16>)
    // CHECK-SAME:      outputs([[VIEW2]] : memref<1x4x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>)

    // CHECK:       [[OUT:%.*]] = IERT.ConcatView
    // CHECK-SAME:      inputs([[COPY1]], [[COPY2]] :
    // CHECK-SAME:          memref<1x4x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>,
    // CHECK-SAME:          memref<1x4x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<1x8x4x4xf16>) -> memref<1x8x4x4xf16>

    // CHECK:       [[VAR7:%.*]] = builtin.unrealized_conversion_cast [[OUT]] : memref<1x8x4x4xf16> to tensor<1x8x4x4xf16>
    // CHECK:       return [[VAR7]] : tensor<1x8x4x4xf16>
}

// -----

func @ExpandToSubviewOnlyWithTail(%arg0: tensor<1x5x4x4xf16>) -> tensor<1x8x4x4xf16> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 3, 0, 0]} : tensor<1x5x4x4xf16> -> tensor<1x8x4x4xf16>
    return %0 : tensor<1x8x4x4xf16>

    // CHECK:       [[VAR0:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<1x5x4x4xf16> to memref<1x5x4x4xf16>

    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<1x8x4x4xf16>

    // CHECK:       [[VIEW1:%.*]] = IERT.SubView [[VAR1]] [0, 0, 0, 0] [1, 5, 4, 4]
    // CHECK-SAME:      : memref<1x8x4x4xf16> to memref<1x5x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>
    // CHECK:       [[COPY1:%.*]] = IERT.Copy inputs([[VAR0]] : memref<1x5x4x4xf16>)
    // CHECK-SAME:      outputs([[VIEW1]] : memref<1x5x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>)

    // CHECK:       [[VIEW_IN:%.*]] = IERT.SubView [[VAR0]] [0, 0, 0, 0] [1, 3, 4, 4]
    // CHECK-SAME:      : memref<1x5x4x4xf16> to memref<1x3x4x4xf16, {order = #NCHW, strides = [80, 16, 4, 1]}>
    // CHECK:       [[VIEW_TAIL:%.*]] = IERT.SubView [[VAR1]] [0, 5, 0, 0] [1, 3, 4, 4]
    // CHECK-SAME:      : memref<1x8x4x4xf16> to memref<1x3x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>
    // CHECK:       [[COPY2:%.*]] = IERT.Copy inputs([[VIEW_IN]] : memref<1x3x4x4xf16, {order = #NCHW, strides = [80, 16, 4, 1]}>)
    // CHECK-SAME:      outputs([[VIEW_TAIL]] : memref<1x3x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>)

    // CHECK:       [[OUT:%.*]] = IERT.ConcatView
    // CHECK-SAME:      inputs([[COPY1]], [[COPY2]] :
    // CHECK-SAME:          memref<1x5x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>,
    // CHECK-SAME:          memref<1x3x4x4xf16, {order = #NCHW, strides = [128, 16, 4, 1]}>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<1x8x4x4xf16>) -> memref<1x8x4x4xf16>

    // CHECK:       [[VAR8:%.*]] = builtin.unrealized_conversion_cast [[OUT]] : memref<1x8x4x4xf16> to tensor<1x8x4x4xf16>
    // CHECK:       return [[VAR8]] : tensor<1x8x4x4xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @WithMemSpace(%arg0: tensor<1x2x3x4xf16>) -> tensor<1x2x3x4xf16> {
    %0 = IE.ReLU(%arg0) : tensor<1x2x3x4xf16> -> tensor<1x2x3x4xf16, {order = #NHWC, mem_space = @CMX_NN}>
    %1 = IE.Tanh(%0) : tensor<1x2x3x4xf16, {order = #NHWC, mem_space = @CMX_NN}> -> tensor<1x2x3x4xf16>
    return %1 : tensor<1x2x3x4xf16>

    // CHECK:       [[VAR0:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<1x2x3x4xf16> to memref<1x2x3x4xf16>

    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<1x2x3x4xf16, #NHWC, @CMX_NN>
    // CHECK:       [[VAR2:%.*]] = IERT.ReLU
    // CHECK-SAME:      inputs([[VAR0]] : memref<1x2x3x4xf16>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<1x2x3x4xf16, #NHWC, @CMX_NN>)

    // CHECK:       [[VAR3:%.*]] = memref.alloc() : memref<1x2x3x4xf16>
    // CHECK:       [[VAR4:%.*]] = IERT.Tanh
    // CHECK-SAME:      inputs([[VAR2]] : memref<1x2x3x4xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[VAR3]] : memref<1x2x3x4xf16>)

    // CHECK:       [[VAR5:%.*]] = builtin.unrealized_conversion_cast [[VAR4]] : memref<1x2x3x4xf16> to tensor<1x2x3x4xf16>
    // CHECK:       return [[VAR5]] : tensor<1x2x3x4xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @PermuteCast(%arg0: tensor<1x12x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x12xf16> {
    %0 = IE.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NCHW} :
        tensor<1x12x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x12xf16>
    return %0 : tensor<1x16x16x12xf16>

    //CHECK:        [[VAR0:%.*]] = builtin.unrealized_conversion_cast %arg0
    // CHECK-SAME:      : tensor<1x12x16x16xf16, {order = #NHWC}> to memref<1x12x16x16xf16, #NHWC>

    //CHECK:        [[VAR1:%.*]] = IERT.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW}
    // CHECK-SAME:      inputs([[VAR0]] : memref<1x12x16x16xf16, #NHWC>) -> memref<1x16x16x12xf16>

    //CHECK:        [[VAR2:%.*]] = builtin.unrealized_conversion_cast [[VAR1]]
    // CHECK-SAME:      : memref<1x16x16x12xf16> to tensor<1x16x16x12xf16>
    //CHECK:        return [[VAR2]] : tensor<1x16x16x12xf16>
}

// -----

// CHECK-LABEL: @CTCGreedyDecoder
func @CTCGreedyDecoder(%arg0: tensor<20x8x128xf16>, %arg1: tensor<20x8xf16>) -> tensor<8x20x1x1xf16> {
    %0 = IE.CTCGreedyDecoder(%arg0, %arg1) {mergeRepeated} : tensor<20x8x128xf16>, tensor<20x8xf16> -> tensor<8x20x1x1xf16>
    return %0 : tensor<8x20x1x1xf16>

    // CHECK:       [[VAR0:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<20x8x128xf16> to memref<20x8x128xf16>
    // CHECK:       [[VAR1:%.*]] = builtin.unrealized_conversion_cast %arg1 : tensor<20x8xf16> to memref<20x8xf16>
    // CHECK:       [[VAR2:%.*]] = memref.alloc() : memref<8x20x1x1xf16>
    // CHECK:       [[VAR3:%.*]] = IERT.CTCGreedyDecoder {mergeRepeated}
    // CHECK-SAME:      inputs([[VAR0]] : memref<20x8x128xf16>, [[VAR1]] : memref<20x8xf16>) outputs([[VAR2]] : memref<8x20x1x1xf16>) -> memref<8x20x1x1xf16>
    // CHECK:       [[VAR4:%.*]] = builtin.unrealized_conversion_cast [[VAR3]] : memref<8x20x1x1xf16> to tensor<8x20x1x1xf16>
    //CHECK:        return [[VAR4]] : tensor<8x20x1x1xf16>
}

// -----

// CHECK-LABEL: @ExtractImagePatches
func @ExtractImagePatches(%arg0: tensor<64x3x10x10xf32>) -> tensor<64x27x2x2xf32> {
    %0 = IE.ExtractImagePatches(%arg0) {sizes = [3, 3], strides = [5, 5], rates = [1, 1], autoPad = "VALID"} : tensor<64x3x10x10xf32> -> tensor<64x27x2x2xf32>
    return %0 : tensor<64x27x2x2xf32>

    // CHECK:       [[VAR0:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<64x3x10x10xf32> to memref<64x3x10x10xf32>
    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<64x27x2x2xf32>
    // CHECK:       [[VAR2:%.*]] = IERT.ExtractImagePatches {autoPad = "VALID", rates = [1, 1], sizes = [3, 3], strides = [5, 5]} 
    // CHECK-SAME:      inputs([[VAR0]] : memref<64x3x10x10xf32>) 
    // CHECK-SAME:      outputs([[VAR1]] : memref<64x27x2x2xf32>) -> memref<64x27x2x2xf32>
    // CHECK:       [[VAR3:%.*]] = builtin.unrealized_conversion_cast [[VAR2]] : memref<64x27x2x2xf32> to tensor<64x27x2x2xf32>
    // CHECK:       return [[VAR3]] : tensor<64x27x2x2xf32>
}

// -----
// CHECK-LABEL: @ReduceL2
func @ReduceL2(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x1xf16> {
    %cst = const.Declare tensor<1xsi32> = #const.Content<dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]>
    %0 = IE.ReduceL2(%arg0, %cst) {keep_dims} : tensor<1x32x112x112xf16>, tensor<1xsi32> -> tensor<1x32x112x1xf16>
    return %0 : tensor<1x32x112x1xf16>

    // CHECK:       [[VAR0:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<1x32x112x112xf16> to memref<1x32x112x112xf16>
    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<1x32x112x1xf16>
    // CHECK:       [[VAR2:%.*]] = IERT.ReduceL2 {keep_dims}
    // CHECK-SAME:      inputs([[VAR0]] : memref<1x32x112x112xf16>, %cst : memref<1xsi32>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<1x32x112x1xf16>) -> memref<1x32x112x1xf16>
    // CHECK:       [[VAR3:%.*]] = builtin.unrealized_conversion_cast [[VAR2]] : memref<1x32x112x1xf16> to tensor<1x32x112x1xf16>
    // CHECK:       return [[VAR3]] : tensor<1x32x112x1xf16>
}
