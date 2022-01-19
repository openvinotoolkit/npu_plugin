// RUN: vpux-opt --split-input-file --legalize-copies %s | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func @LegalizeCopy(
        %arg0: memref<1x64x512x512xf16, #NCHW>,
        %arg1: memref<1x64x512x512xf16, #NCHW>)
        -> memref<1x64x512x512xf16, #NCHW> {
    %0 = IERT.Copy inputs(%arg0 : memref<1x64x512x512xf16, #NCHW>)
                   outputs(%arg1 : memref<1x64x512x512xf16, #NCHW>)
                   -> memref<1x64x512x512xf16, #NCHW>

    return %0 : memref<1x64x512x512xf16, #NCHW>

    // Currently, large Copy nodes are tiled C-wise

    // Cut first tile:
    // CHECK: [[VAR0:%.*]] = IERT.SubView %arg0 [0, 0, 0, 0] [1, 32, 512, 512] :
    // CHECK-SAME:      memref<1x64x512x512xf16>
    // CHECK-SAME:   to memref<1x32x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK: [[VAR1:%.*]] = IERT.SubView %arg1 [0, 0, 0, 0] [1, 32, 512, 512] :
    // CHECK-SAME:      memref<1x64x512x512xf16>
    // CHECK-SAME:   to memref<1x32x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK: [[VAR2:%.*]] = IERT.Copy
    // CHECK-SAME:      inputs([[VAR0]] : memref<1x32x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<1x32x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:        -> memref<1x32x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // Cut the second tile:
    // CHECK: [[VAR3:%.*]] = IERT.SubView %arg0 [0, 32, 0, 0] [1, 32, 512, 512] :
    // CHECK-SAME:      memref<1x64x512x512xf16>
    // CHECK-SAME:   to memref<1x32x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK: [[VAR4:%.*]] = IERT.SubView %arg1 [0, 32, 0, 0] [1, 32, 512, 512] :
    // CHECK-SAME:      memref<1x64x512x512xf16>
    // CHECK-SAME:   to memref<1x32x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK: [[VAR5:%.*]] = IERT.Copy
    // CHECK-SAME:      inputs([[VAR3]] : memref<1x32x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:      outputs([[VAR4]] : memref<1x32x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:        -> memref<1x32x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // Concatenate the resulting output tiles:
    // CHECK: [[VAR6:%.*]] = IERT.ConcatView
    // CHECK-SAME:      inputs([[VAR2]], [[VAR5]] :
    // CHECK-SAME:        memref<1x32x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>,
    // CHECK-SAME:        memref<1x32x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:      outputs(%arg1 : memref<1x64x512x512xf16>)
    // CHECK-SAME:        -> memref<1x64x512x512xf16>
    // CHECK: return [[VAR6]] : memref<1x64x512x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func @LegalizeStridedCopy(
        %arg0: memref<1x64x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>,
        %arg1: memref<1x64x512x512xf16, #NCHW>)
        -> memref<1x64x512x512xf16, #NCHW> {
    %0 = IERT.Copy inputs(%arg0 : memref<1x64x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>)
                   outputs(%arg1 : memref<1x64x512x512xf16, #NCHW>)
                   -> memref<1x64x512x512xf16, #NCHW>

    return %0 : memref<1x64x512x512xf16, #NCHW>

    // Currently, large Copy nodes are tiled C-wise
    // If the Copy is strided, the strides should be preserved

    // Cut the first tile:
    // CHECK: [[VAR0:%.*]] = IERT.SubView %arg0 [0, 0, 0, 0] [1, 32, 512, 512] :
    // CHECK-SAME:      memref<1x64x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>
    // CHECK-SAME:   to memref<1x32x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>
    // CHECK: [[VAR1:%.*]] = IERT.SubView %arg1 [0, 0, 0, 0] [1, 32, 512, 512] :
    // CHECK-SAME:      memref<1x64x512x512xf16>
    // CHECK-SAME:   to memref<1x32x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // The Copy-tile preserves the original strides:
    // CHECK: [[VAR2:%.*]] = IERT.Copy
    // CHECK-SAME:      inputs([[VAR0]] : memref<1x32x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<1x32x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:        -> memref<1x32x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // Cut the second tile:
    // CHECK: [[VAR3:%.*]] = IERT.SubView %arg0 [0, 32, 0, 0] [1, 32, 512, 512] :
    // CHECK-SAME:      memref<1x64x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>
    // CHECK-SAME:   to memref<1x32x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>
    // CHECK: [[VAR4:%.*]] = IERT.SubView %arg1 [0, 32, 0, 0] [1, 32, 512, 512] :
    // CHECK-SAME:      memref<1x64x512x512xf16>
    // CHECK-SAME:   to memref<1x32x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // The Copy-tile preserves the original strides:
    // CHECK: [[VAR5:%.*]] = IERT.Copy
    // CHECK-SAME:      inputs([[VAR3]] : memref<1x32x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>)
    // CHECK-SAME:      outputs([[VAR4]] : memref<1x32x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:        -> memref<1x32x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // Concatenate the resulting output tiles:
    // CHECK: [[VAR6:%.*]] = IERT.ConcatView
    // CHECK-SAME:      inputs([[VAR2]], [[VAR5]] :
    // CHECK-SAME:        memref<1x32x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>,
    // CHECK-SAME:        memref<1x32x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:      outputs(%arg1 : memref<1x64x512x512xf16>)
    // CHECK-SAME:        -> memref<1x64x512x512xf16>
    // CHECK: return [[VAR6]] : memref<1x64x512x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func @DoNotLegalizeCopy(
        %arg0: memref<1x32x512x512xf16, #NCHW>,
        %arg1: memref<1x32x512x512xf16, #NCHW>)
        -> memref<1x32x512x512xf16, #NCHW> {
    %0 = IERT.Copy inputs(%arg0 : memref<1x32x512x512xf16, #NCHW>)
                   outputs(%arg1 : memref<1x32x512x512xf16, #NCHW>)
                   -> memref<1x32x512x512xf16, #NCHW>

    return %0 : memref<1x32x512x512xf16, #NCHW>

    // Small enough Copy nodes (those with transaction volume less than 16MB) should not be affected by the pass

    // CHECK: [[VAR0:%.*]] = IERT.Copy
    // CHECK-SAME:      inputs(%arg0 : memref<1x32x512x512xf16>)
    // CHECK-SAME:      outputs(%arg1 : memref<1x32x512x512xf16>)
    // CHECK-SAME:        -> memref<1x32x512x512xf16>
    // CHECK: return [[VAR0]] : memref<1x32x512x512xf16>
}
