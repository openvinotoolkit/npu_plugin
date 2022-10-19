// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-view-ops-to-declarations %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @Reshape
func @Reshape(%arg0: memref<1x512xf16>, %arg1: memref<1x512xf16>) -> memref<1x512xf16> {
    %0 = VPUIP.GenericReshape inputs(%arg0 : memref<1x512xf16>) -> memref<1x512x1x1xf16>
    %1 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x512x1x1xf16, @DDR>
    %2 = VPUIP.SoftMaxUPA {axisInd = 1} inputs(%0 : memref<1x512x1x1xf16>) outputs(%1 : memref<1x512x1x1xf16, @DDR>) -> memref<1x512x1x1xf16, @DDR>
    %3 = VPUIP.GenericReshape inputs(%2 : memref<1x512x1x1xf16, @DDR>) -> memref<1x512xf16, @DDR>
    %4 = VPUIP.NNDMA inputs(%3 : memref<1x512xf16, @DDR>) outputs(%arg1 : memref<1x512xf16>) -> memref<1x512xf16>
    return %4 : memref<1x512xf16>

    // CHECK:       [[VAR0:%.*]] = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> memref<1x512x1x1xf16>

    // CHECK:       [[VAR1:%.*]] = VPURT.DeclareBuffer "DDR" <0> -> memref<1x512x1x1xf16, @DDR>

    // CHECK:       [[VAR2:%.*]] = VPUIP.SoftMaxUPA
    // CHECK-SAME:      axisInd = 1
    // CHECK-SAME:      inputs([[VAR0]] : memref<1x512x1x1xf16>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<1x512x1x1xf16, @DDR>)

    // CHECK:       [[VAR3:%.*]] = VPURT.DeclareBuffer "DDR" <0> -> memref<1x512xf16, @DDR>

    // CHECK:       [[VAR4:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs([[VAR3]] : memref<1x512xf16, @DDR>)
    // CHECK-SAME:      outputs(%arg1 : memref<1x512xf16>) -> memref<1x512xf16>

    // CHECK: return [[VAR4]] : memref<1x512xf16>
}

// -----

// CHECK-LABEL: @SubView
func @SubView(%arg0: memref<4x4xf16>, %arg1: memref<4x4xf16>) -> memref<4x4xf16> {
    %0 = VPUIP.SubView %arg0 [0, 0][2, 4] : memref<4x4xf16> to memref<2x4xf16>
    %1 = VPUIP.SubView %arg1 [0, 0][2, 4] : memref<4x4xf16> to memref<2x4xf16>
    %2 = VPUIP.NNDMA inputs(%0 : memref<2x4xf16>) outputs(%1 : memref<2x4xf16>) -> memref<2x4xf16>

    %3 = VPUIP.SubView %arg0 [2, 0][2, 4] : memref<4x4xf16> to memref<2x4xf16>
    %4 = VPUIP.SubView %arg1 [2, 0][2, 4] : memref<4x4xf16> to memref<2x4xf16>
    %5 = VPUIP.NNDMA inputs(%3 : memref<2x4xf16>) outputs(%4 : memref<2x4xf16>) -> memref<2x4xf16>

    return %arg1 : memref<4x4xf16>

    // CHECK:       [[VAR0:%.*]] = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> memref<2x4xf16>
    // CHECK:       [[VAR1:%.*]] = VPURT.DeclareBuffer "NetworkOutput" [0] <0> -> memref<2x4xf16>
    // CHECK:       [[VAR2:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs([[VAR0]] : memref<2x4xf16>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<2x4xf16>) -> memref<2x4xf16>

    // CHECK:       [[VAR3:%.*]] = VPURT.DeclareBuffer "NetworkInput" [0] <16> -> memref<2x4xf16>
    // CHECK:       [[VAR4:%.*]] = VPURT.DeclareBuffer "NetworkOutput" [0] <16> -> memref<2x4xf16>
    // CHECK:       [[VAR5:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs([[VAR3]] : memref<2x4xf16>)
    // CHECK-SAME:      outputs([[VAR4]] : memref<2x4xf16>) -> memref<2x4xf16>

    // CHECK:       return %arg1 : memref<4x4xf16>
}

// -----

// CHECK-LABEL: @WithAsyncRegions
func @WithAsyncRegions(%arg0: memref<1x1x1x512xf32>, %arg1: memref<1x1x1x512xf32>) -> memref<1x1x1x512xf32> {
    %t0, %f0 = async.execute -> !async.value<memref<1x1x1x512xf16, @DDR>> {
        %0 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x1x1x512xf16, @DDR>
        %1 = VPUIP.ConvertUPA inputs(%arg0 : memref<1x1x1x512xf32>) outputs(%0 : memref<1x1x1x512xf16, @DDR>) -> memref<1x1x1x512xf16, @DDR>
        async.yield %1 : memref<1x1x1x512xf16, @DDR>
    }

    %t1, %f1 = async.execute [%t0] (%f0 as %1: !async.value<memref<1x1x1x512xf16, @DDR>>) -> !async.value<memref<1x512xf16, @DDR>> {
        %2 = VPUIP.GenericReshape inputs(%1 : memref<1x1x1x512xf16, @DDR>) -> memref<1x512xf16, @DDR>
        %3 = VPURT.DeclareBuffer "DDR" <1024> -> memref<1x512xf16, @DDR>
        %4 = VPUIP.SoftMaxUPA {axisInd = 1} inputs(%2 : memref<1x512xf16, @DDR>) outputs(%3 : memref<1x512xf16, @DDR>) -> memref<1x512xf16, @DDR>
        async.yield %4 : memref<1x512xf16, @DDR>
    }

    %t2, %f2 = async.execute [%t1] (%f1 as %4: !async.value<memref<1x512xf16, @DDR>>) -> !async.value<memref<1x1x1x512xf32>> {
        %5 = VPUIP.GenericReshape inputs(%4 : memref<1x512xf16, @DDR>) -> memref<1x1x1x512xf16, @DDR>
        %6 = VPUIP.ConvertUPA inputs(%5 : memref<1x1x1x512xf16, @DDR>) outputs(%arg1 : memref<1x1x1x512xf32>) -> memref<1x1x1x512xf32>
        async.yield %6 : memref<1x1x1x512xf32>
    }

    %6 = async.await %f2 : !async.value<memref<1x1x1x512xf32>>
    return %6 : memref<1x1x1x512xf32>

    // CHECK:       [[T0:%.+]], [[F0:%.+]] = async.execute
    // CHECK-SAME:          -> !async.value<memref<1x1x1x512xf16, @DDR>>
    // CHECK:           [[VAR0:%.*]] = VPURT.DeclareBuffer "DDR" <0> -> memref<1x1x1x512xf16, @DDR>
    // CHECK:           [[VAR1:%.*]] = VPUIP.ConvertUPA
    // CHECK-SAME:          inputs(%arg0 : memref<1x1x1x512xf32>)
    // CHECK-SAME:          outputs([[VAR0]] : memref<1x1x1x512xf16, @DDR>)
    // CHECK:           async.yield [[VAR1]] : memref<1x1x1x512xf16, @DDR>

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK-SAME:          [[T0]]
    // CHECK-SAME:          ([[F0]] as [[VAR1:%.+]]: !async.value<memref<1x1x1x512xf16, @DDR>>)
    // CHECK-SAME:          -> !async.value<memref<1x512xf16, @DDR>>
    // CHECK:           [[VAR2:%.*]] = VPURT.DeclareBuffer "DDR" <0> -> memref<1x512xf16, @DDR>
    // CHECK:           [[VAR3:%.*]] = VPURT.DeclareBuffer "DDR" <1024> -> memref<1x512xf16, @DDR>
    // CHECK:           [[VAR4:%.*]] = VPUIP.SoftMaxUPA
    // CHECK-SAME:          inputs([[VAR2]] : memref<1x512xf16, @DDR>)
    // CHECK-SAME:          outputs([[VAR3]] : memref<1x512xf16, @DDR>)
    // CHECK:           async.yield [[VAR4]] : memref<1x512xf16, @DDR>

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute
    // CHECK-SAME:          [[T1]]
    // CHECK-SAME:          ([[F1]] as [[VAR4:%.+]]: !async.value<memref<1x512xf16, @DDR>>)
    // CHECK-SAME:          -> !async.value<memref<1x1x1x512xf32>>
    // CHECK:           [[VAR5:%.*]] = VPURT.DeclareBuffer "DDR" <1024> -> memref<1x1x1x512xf16, @DDR>
    // CHECK:           [[VAR6:%.*]] = VPUIP.ConvertUPA
    // CHECK-SAME:          inputs([[VAR5]] : memref<1x1x1x512xf16, @DDR>)
    // CHECK-SAME:          outputs(%arg1 : memref<1x1x1x512xf32>)
    // CHECK:           async.yield [[VAR6]] : memref<1x1x1x512xf32>

    // CHECK:       [[VAR6:%.+]] = async.await [[F2]] : !async.value<memref<1x1x1x512xf32>>
    // CHECK:       return [[VAR6]] : memref<1x1x1x512xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PermuteCast
func @PermuteCast(%arg0: memref<1x12x16x16xf16, #NHWC>, %arg1: memref<1x16x16x12xf16>) -> memref<1x16x16x12xf16> {
    %0 = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW}
        inputs(%arg0 : memref<1x12x16x16xf16, #NHWC>)
        -> memref<1x16x16x12xf16>

    %1 = VPURT.DeclareBuffer "DDR" <2000> -> memref<1x16x16x12xf16, @DDR>
    %2 = VPUIP.SoftMaxUPA {axisInd = 1}
        inputs(%0 : memref<1x16x16x12xf16>)
        outputs(%1 : memref<1x16x16x12xf16, @DDR>) -> memref<1x16x16x12xf16, @DDR>
    %3 = VPUIP.NNDMA
        inputs(%2 : memref<1x16x16x12xf16, @DDR>)
        outputs(%arg1 : memref<1x16x16x12xf16>) -> memref<1x16x16x12xf16>
    return %3 : memref<1x16x16x12xf16>

    //CHECK:        [[VAR0:%.*]] = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> memref<1x16x16x12xf16>
    //CHECK:        [[VAR1:%.*]] = VPURT.DeclareBuffer "DDR" <2000> -> memref<1x16x16x12xf16, @DDR>
    //CHECK:        [[VAR2:%.*]] = VPUIP.SoftMaxUPA {axisInd = 1 : i64}
    //CHECK-SAME:       inputs([[VAR0]] : memref<1x16x16x12xf16>)
    //CHECK-SAME:       outputs([[VAR1]] : memref<1x16x16x12xf16, @DDR>) -> memref<1x16x16x12xf16, @DDR>
    //CHECK:        [[VAR3:%.*]] = VPUIP.NNDMA
    //CHECK-SAME:       inputs([[VAR2]] : memref<1x16x16x12xf16, @DDR>)
    //CHECK-SAME:       outputs(%arg1 : memref<1x16x16x12xf16>) -> memref<1x16x16x12xf16>
    //CHECK:        return [[VAR3]] : memref<1x16x16x12xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributedBuffer = type !VPUIP.DistributedBuffer<
    1x128x16x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4
}>

!OutputDistributedBuffer = type !VPUIP.DistributedBuffer<
    1x128x16x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4
}>

// CHECK-LABEL: @DistributedCast
func @DistributedCast(%arg0: memref<1x128x16x16xf16, #NHWC>) -> memref<1x128x16x16xf16, #NHWC> {
    %0 = VPURT.DeclareBuffer "CMX_NN" <0> -> !InputDistributedBuffer
    %1 = VPUIP.DistributedCast inputs(%0 : !InputDistributedBuffer) -> !OutputDistributedBuffer
    return %arg0 : memref<1x128x16x16xf16, #NHWC>

    // CHECK:       VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer<1x128x16x16xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:        {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
    // CHECK:       VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer<1x128x16x16xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:        {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
    // CHECK-NOT:   VPUIP.DistributedCast
}

// -----

// CHECK-LABEL: @VPUIPSubViewMemRef
func @VPUIPSubViewMemRef(%arg0: memref<4x4xf16>, %arg1: memref<4x4xf16>) -> memref<4x4xf16> {
    %0 = VPUIP.SubView %arg0 [0, 0][2, 4] : memref<4x4xf16> to memref<2x4xf16>
    %1 = VPUIP.SubView %arg1 [0, 0][2, 4] : memref<4x4xf16> to memref<2x4xf16>
    %2 = VPUIP.NNDMA inputs(%0 : memref<2x4xf16>) outputs(%1 : memref<2x4xf16>) -> memref<2x4xf16>

    %3 = VPUIP.SubView %arg0 [2, 0][2, 4] : memref<4x4xf16> to memref<2x4xf16>
    %4 = VPUIP.SubView %arg1 [2, 0][2, 4] : memref<4x4xf16> to memref<2x4xf16>
    %5 = VPUIP.NNDMA inputs(%3 : memref<2x4xf16>) outputs(%4 : memref<2x4xf16>) -> memref<2x4xf16>

    return %arg1 : memref<4x4xf16>

    // CHECK:       [[VAR0:%.*]] = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> memref<2x4xf16>
    // CHECK:       [[VAR1:%.*]] = VPURT.DeclareBuffer "NetworkOutput" [0] <0> -> memref<2x4xf16>
    // CHECK:       [[VAR2:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs([[VAR0]] : memref<2x4xf16>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<2x4xf16>) -> memref<2x4xf16>

    // CHECK:       [[VAR3:%.*]] = VPURT.DeclareBuffer "NetworkInput" [0] <16> -> memref<2x4xf16>
    // CHECK:       [[VAR4:%.*]] = VPURT.DeclareBuffer "NetworkOutput" [0] <16> -> memref<2x4xf16>
    // CHECK:       [[VAR5:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs([[VAR3]] : memref<2x4xf16>)
    // CHECK-SAME:      outputs([[VAR4]] : memref<2x4xf16>) -> memref<2x4xf16>

    // CHECK:       return %arg1 : memref<4x4xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributedBuffer = type !VPUIP.DistributedBuffer<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64
}>

!InputSliceDistributedBuffer = type !VPUIP.DistributedBuffer<
    1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64
}>

!OutputDistributedBuffer = type !VPUIP.DistributedBuffer<
    1x64x8x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64
}>

!InputSliceBuffer = type memref<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN>
!OutputBuffer = type memref<1x64x8x16xf16, #NHWC, @CMX_NN>

// CHECK-LABEL: @VPUIPSubViewDistributed
func @VPUIPSubViewDistributed(%arg0: !OutputBuffer) -> !OutputBuffer {

    %0 = VPURT.DeclareBuffer "CMX_NN" <0> -> !InputDistributedBuffer
    %1 = VPURT.DeclareBuffer "CMX_NN" <0> -> !OutputDistributedBuffer
    %2 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 64, 8, 16] : !InputDistributedBuffer to !InputSliceDistributedBuffer
    %3 = VPUIP.NCEClusterTiling inputs(%2 as %arg3: !InputSliceBuffer) outputs(%1 as %arg4: !OutputBuffer) -> !OutputDistributedBuffer {
      %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg3 : !InputSliceBuffer) outputs(%arg4 : !OutputBuffer) -> !OutputBuffer
    }

    return %arg0 : !OutputBuffer

    // CHECK-DAG:       [[BUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
    // CHECK-DAG:       [[BUF_OUT:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer<1x64x8x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
    // CHECK-DAG:       [[BUF_SLICE:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>

    // CHECK-NOT:   VPUIP.SubView

    // CHECK:       VPUIP.NCEClusterTiling
    // CHECK-SAME:       inputs([[BUF_SLICE]] as [[ARG1:%.+]]: memref<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN>)
    // CHECK-SAME:       outputs([[BUF_OUT]] as [[ARG2:%.+]]: memref<1x64x8x16xf16, #NHWC, @CMX_NN>)
    // CHECK:       VPUIP.NNDMA
    // CHECK-SAME:       inputs([[ARG1]] : memref<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN>)
    // CHECK-SAME:       outputs([[ARG2]] : memref<1x64x8x16xf16, #NHWC, @CMX_NN>)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributedBuffer = type !VPUIP.DistributedBuffer<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64
}>

!InputSliceDistributedBuffer = type !VPUIP.DistributedBuffer<
    1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64
}>

!OutputDistributedBuffer = type !VPUIP.DistributedBuffer<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64
}>

!InputBufferDdr = type memref<1x64x8x16xf16, #NHWC, @DDR>
!InputBuffer = type memref<1x64x8x16xf16, #NHWC, @CMX_NN>
!InputSliceBuffer = type memref<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN>
!OutputBuffer = type memref<1x64x16x16xf16, #NHWC, @CMX_NN>
!OutputBufferDdr = type memref<1x64x16x16xf16, #NHWC, @DDR>

// CHECK-LABEL: @VPUIPConcatView
func @VPUIPConcatView(%arg0: !InputBufferDdr, %arg1: !InputBufferDdr) -> !OutputBufferDdr {

    %0 = VPURT.DeclareBuffer "CMX_NN" <0> -> !InputDistributedBuffer
    %1 = VPURT.DeclareBuffer "DDR" <0> -> !OutputBufferDdr

   
    %2 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 64, 8, 16] : !InputDistributedBuffer to !InputSliceDistributedBuffer
    %3 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg2: !InputBufferDdr) outputs(%2 as %arg3: !InputSliceBuffer) -> !InputSliceDistributedBuffer {
      %8 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg2 : !InputBufferDdr) outputs(%arg3 : !InputSliceBuffer) -> !InputSliceBuffer
    }

    %4 = VPUIP.SubView %0 [0, 0, 8, 0] [1, 64, 8, 16] : !InputDistributedBuffer to !InputSliceDistributedBuffer
    %5 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg2: !InputBufferDdr) outputs(%4 as %arg3: !InputSliceBuffer) -> !InputSliceDistributedBuffer {
      %8 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg2 : !InputBufferDdr) outputs(%arg3 : !InputSliceBuffer) -> !InputSliceBuffer
    }

    %6 = VPUIP.ConcatView inputs(%3, %5 : !InputSliceDistributedBuffer, !InputSliceDistributedBuffer) outputs(%0 : !OutputDistributedBuffer) -> !OutputDistributedBuffer
    %7 = VPUIP.NCEClusterTiling inputs(%6 as %arg2: !OutputBuffer) outputs(%1 as %arg3: !OutputBufferDdr) -> !OutputBufferDdr {
      %8 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg2 : !OutputBuffer) outputs(%arg3 : !OutputBufferDdr) -> !OutputBufferDdr
    }

    return %1 : !OutputBufferDdr

    // CHECK-DAG:       [[BUF_INPUT:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
    // CHECK-DAG:       [[BUF_OUTPUT:%.*]] = VPURT.DeclareBuffer "DDR" <0> -> memref<1x64x16x16xf16, #NHWC, @DDR>

    // CHECK-NOT:   VPUIP.SubView
    // CHECK:       [[BUF_INPUT_SLICE1:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>

    // CHECK:       VPUIP.NCEClusterTiling
    // CHECK-SAME:       inputs(%arg0 as [[ARG1:%.+]]: memref<1x64x8x16xf16, #NHWC, @DDR>)
    // CHECK-SAME:       outputs([[BUF_INPUT_SLICE1]] as [[ARG2:%.+]]: memref<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN>)
    // CHECK:       VPUIP.NNDMA
    // CHECK-SAME:       inputs([[ARG1]] : memref<1x64x8x16xf16, #NHWC, @DDR>)
    // CHECK-SAME:       outputs([[ARG2]] : memref<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN>)
   
    // CHECK-NOT:   VPUIP.SubView
    // CHECK:       [[BUF_INPUT_SLICE2:%.*]] = VPURT.DeclareBuffer "CMX_NN" <16384> -> !VPUIP.DistributedBuffer<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>

    // CHECK:       VPUIP.NCEClusterTiling
    // CHECK-SAME:       inputs(%arg0 as [[ARG3:%.+]]: memref<1x64x8x16xf16, #NHWC, @DDR>)
    // CHECK-SAME:       outputs([[BUF_INPUT_SLICE2]] as [[ARG4:%.+]]: memref<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN>)
    // CHECK:       VPUIP.NNDMA
    // CHECK-SAME:       inputs([[ARG3]] : memref<1x64x8x16xf16, #NHWC, @DDR>)
    // CHECK-SAME:       outputs([[ARG4]] : memref<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN>)

    // CHECK-NOT:   VPUIP.ConcatView

    // CHECK:       VPUIP.NCEClusterTiling
    // CHECK-SAME:       inputs([[BUF_INPUT]] as [[ARG5:%.+]]: memref<1x64x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:       outputs([[BUF_OUTPUT]] as [[ARG6:%.+]]: memref<1x64x16x16xf16, #NHWC, @DDR>)
    // CHECK:       VPUIP.NNDMA
    // CHECK-SAME:       inputs([[ARG5]] : memref<1x64x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:       outputs([[ARG6]] : memref<1x64x16x16xf16, #NHWC, @DDR>)

    // CHECK:       return [[BUF_OUTPUT]] : memref<1x64x16x16xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ShapeCast
func @ShapeCast(%arg0: memref<64x3x7x7xf16, #NHWC>, %arg1: memref<1x64x112x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x112x112xf16, #NHWC, [@CMX_NN, 0]> {

    %0 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<64x3x7x7xf16, #NHWC, [@CMX_NN, 0]>
    %weights = VPUIP.NNDMA
        inputs(%arg0: memref<64x3x7x7xf16, #NHWC>)
        outputs(%0: memref<64x3x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<64x3x7x7xf16, #NHWC, [@CMX_NN, 0]>

    %weights_align = VPUIP.ShapeCast{shape = [64, 16, 7, 7]}
        inputs(%weights: memref<64x3x7x7xf16, #NHWC, [@CMX_NN, 0]>)
         -> memref<64x16x7x7xf16, #NHWC, [@CMX_NN, 0]>
    %weight_table = VPURT.DeclareBuffer "CMX_NN" [0] <1024> -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>

    %in = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x224x224xf16, #NHWC, [@CMX_NN, 0]>

    %1 = VPUIP.NCEClusterTask {
            kernel_padding = {bottom = 3 : i64, left = 3 : i64, right = 3 : i64, top = 3 : i64},
            kernel_size = [7, 7],
            kernel_strides = [2, 2],
            task_type = "CONV"
        }
        input(%in : memref<1x16x224x224xf16, #NHWC, [@CMX_NN, 0]>)
        weights(%weights_align : memref<64x16x7x7xf16, #NHWC, [@CMX_NN, 0]>)
        weight_table(%weight_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
        parent_input(%in : memref<1x16x224x224xf16, #NHWC, [@CMX_NN, 0]>)
        parent_output(%arg1 : memref<1x64x112x112xf16, #NHWC, [@CMX_NN, 0]>)
        outputs(%arg1 : memref<1x64x112x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x112x112xf16, #NHWC, [@CMX_NN, 0]>
        variants :
        {
            DPUTask {end = [111, 111, 63], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
        }
        PPE :  {
        }

    return %1 : memref<1x64x112x112xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:        [[VAR0:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<64x3x7x7xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[VAR1:%.*]] = VPUIP.NNDMA inputs(%arg0 : memref<64x3x7x7xf16, #NHWC>) outputs([[VAR0]] : memref<64x3x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<64x3x7x7xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[VAR2:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<64x16x7x7xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[VAR3:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <1024> -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:        [[VAR4:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x224x224xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[VAR5:%.*]] = VPUIP.NCEClusterTask {kernel_padding = {bottom = 3 : i64, left = 3 : i64, right = 3 : i64, top = 3 : i64}, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = "CONV"}
    //CHECK-SAME:           input([[VAR4]] : memref<1x16x224x224xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[VAR2]] : memref<64x16x7x7xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[VAR3]] : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[VAR4]] : memref<1x16x224x224xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_output(%arg1 : memref<1x64x112x112xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           outputs(%arg1 : memref<1x64x112x112xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           -> memref<1x64x112x112xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-SAME:           variants :  {
    //CHECK:       DPUTask {end = [111, 111, 63], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
    //CHECK:       return [[VAR5]] : memref<1x64x112x112xf16, #NHWC, [@CMX_NN, 0]>
}
