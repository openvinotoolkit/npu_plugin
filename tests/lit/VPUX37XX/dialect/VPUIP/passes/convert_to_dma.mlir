//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX compilation-mode=DefaultHW" --convert-to-dma --canonicalize %s | FileCheck %s

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder_fp16.cpp", VPU.kernel_entry = "reorder_fp16"}
    func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

func @ConvertMemPermute(%arg0: memref<1x16x12x12xf16, @DDR>)
        -> memref<1x16x12x12xf16, #NHWC, @DDR> {
    %0 = memref.alloc() : memref<1x16x12x12xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x12x12xf16, @DDR>) outputs(%0 : memref<1x16x12x12xf16, [@CMX_NN, 0]>) -> memref<1x16x12x12xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x16x12x12xf16, #NHWC, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_MemPermute inputs(%1 as %arg2: memref<1x16x12x12xf16, [@CMX_NN, 0]>) outputs(%2 as %arg3: memref<1x16x12x12xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x12x12xf16, #NHWC, [@CMX_NN, 0]>{
       VPUIP.SW.Kernel.run {attrs = [[2, 0, 1, 3]]}(%arg2, %arg3) : memref<1x16x12x12xf16, [@CMX_NN, 0]>, memref<1x16x12x12xf16, #NHWC, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x16x12x12xf16, #NHWC, @DDR>
    %4 = VPUIP.Copy inputs(%results : memref<1x16x12x12xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x12x12xf16, #NHWC, @DDR>) -> memref<1x16x12x12xf16, #NHWC, @DDR>
    return %4: memref<1x16x12x12xf16, #NHWC, @DDR>

    // CHECK:   [[VAR0:%.*]] = memref.alloc() : memref<1x16x12x12xf16, [@CMX_NN, 0]>
    // CHECK:   [[VAR1:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x16x12x12xf16, @DDR>) outputs([[VAR0]] : memref<1x16x12x12xf16, [@CMX_NN, 0]>) -> memref<1x16x12x12xf16, [@CMX_NN, 0]>
    // CHECK:   [[VAR2:%.*]] = memref.alloc() : memref<1x16x12x12xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR3:%.*]] = VPUIP.PermuteDMA {mem_perm = #NHWC, port = 0 : i64} inputs([[VAR1]] : memref<1x16x12x12xf16, [@CMX_NN, 0]>) outputs([[VAR2]] : memref<1x16x12x12xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x12x12xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR4:%.*]] = memref.alloc() : memref<1x16x12x12xf16, #NHWC, @DDR>
    // CHECK:   [[VAR5:%.*]]  = VPUIP.Copy inputs([[VAR3]] : memref<1x16x12x12xf16, #NHWC, [@CMX_NN, 0]>) outputs([[VAR4]] : memref<1x16x12x12xf16, #NHWC, @DDR>) -> memref<1x16x12x12xf16, #NHWC, @DDR>
    // CHECK:   return [[VAR5:%.*]] : memref<1x16x12x12xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func private @builtin_DepthToSpace(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "single_shave_depth_to_space.cpp", VPU.kernel_entry = "single_shave_depth_to_space"}
    func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @ConvertSWDepthToSpaceToDMA_BLOCKS_FIRST
func @ConvertSWDepthToSpaceToDMA_BLOCKS_FIRST(%arg0: memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]> {
    %outBuffer = memref.alloc() : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
    %depthToSpace = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_DepthToSpace
                        inputs(%arg0 as %arg1: memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>)
                        outputs(%outBuffer as %arg2: memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>{
                    VPUIP.SW.Kernel.run {attrs = [2, 0]}(%arg1, %arg2) : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>, memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
    }

    return %depthToSpace : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[OUTBUFFER:%.*]] = memref.alloc() : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[DepthToSpaceDMAOUT:%.*]] = VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = "BLOCKS_FIRST", port = 0 : i64}
    //CHECK:            inputs(%arg0 : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTBUFFER]] : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    return [[DepthToSpaceDMAOUT]] : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func private @builtin_DepthToSpace(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "single_shave_depth_to_space.cpp", VPU.kernel_entry = "single_shave_depth_to_space"}
    func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @ConvertSWDepthToSpaceToDMA_DEPTH_FIRST
func @ConvertSWDepthToSpaceToDMA_DEPTH_FIRST(%arg0: memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]> {
    %outBuffer = memref.alloc() : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
    %depthToSpace = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_DepthToSpace
                        inputs(%arg0 as %arg1: memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>)
                        outputs(%outBuffer as %arg2: memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>{
                    VPUIP.SW.Kernel.run {attrs = [2, 1]}(%arg1, %arg2) : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>, memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
    }

    return %depthToSpace : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[OUTBUFFER:%.*]] = memref.alloc() : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[DepthToSpaceDMAOUT:%.*]] = VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = "DEPTH_FIRST", port = 0 : i64}
    //CHECK:            inputs(%arg0 : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTBUFFER]] : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    return [[DepthToSpaceDMAOUT]] : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder_fp16.cpp", VPU.kernel_entry = "reorder_fp16"}
    func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

func @ConvertMemPermuteWithThreeAxis(%arg0: memref<1x16x4x128xf16, @DDR>)
        -> memref<1x4x16x128xf16, #NHWC, @DDR> {
    %0 = memref.alloc() : memref<1x16x4x128xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x4x128xf16, @DDR>) outputs(%0 : memref<1x16x4x128xf16, [@CMX_NN, 0]>) -> memref<1x16x4x128xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x4x16x128xf16, #NHWC, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_MemPermute inputs(%1 as %arg2: memref<1x16x4x128xf16, [@CMX_NN, 0]>) outputs(%2 as %arg3: memref<1x4x16x128xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x4x16x128xf16, #NHWC, [@CMX_NN, 0]>{
       VPUIP.SW.Kernel.run {attrs = [[0, 2, 1, 3]]}(%arg2, %arg3) : memref<1x16x4x128xf16, [@CMX_NN, 0]>, memref<1x4x16x128xf16, #NHWC, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x4x16x128xf16, #NHWC, @DDR>
    %4 = VPUIP.Copy inputs(%results : memref<1x4x16x128xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x4x16x128xf16, #NHWC, @DDR>) -> memref<1x4x16x128xf16, #NHWC, @DDR>
    return %4: memref<1x4x16x128xf16, #NHWC, @DDR>

    // CHECK:   [[VAR0:%.*]] = memref.alloc() : memref<1x16x4x128xf16, [@CMX_NN, 0]>
    // CHECK:   [[VAR1:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x16x4x128xf16, @DDR>) outputs([[VAR0]] : memref<1x16x4x128xf16, [@CMX_NN, 0]>) -> memref<1x16x4x128xf16, [@CMX_NN, 0]>
    // CHECK:   [[VAR2:%.*]] = memref.alloc() : memref<1x4x16x128xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR3:%.*]] = VPUIP.PermuteDMA {mem_perm = #NHCW, port = 0 : i64} inputs([[VAR1]] : memref<1x16x4x128xf16, [@CMX_NN, 0]>) outputs([[VAR2]] : memref<1x4x16x128xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x4x16x128xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR4:%.*]] = memref.alloc() : memref<1x4x16x128xf16, #NHWC, @DDR>
    // CHECK:   [[VAR5:%.*]]  = VPUIP.Copy inputs([[VAR3]] : memref<1x4x16x128xf16, #NHWC, [@CMX_NN, 0]>) outputs([[VAR4]] : memref<1x4x16x128xf16, #NHWC, @DDR>) -> memref<1x4x16x128xf16, #NHWC, @DDR>
    // CHECK:   return [[VAR5:%.*]] : memref<1x4x16x128xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map0 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder_fp16.cpp", VPU.kernel_entry = "reorder_fp16"}
    func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

func @ConvertMemPermuteHWCToWHC(%arg0: memref<1x16x4x76xf16, #map0, @DDR>)
        -> memref<1x16x4x76xf16, #NHWC, @DDR> {
    %0 = memref.alloc() : memref<1x16x4x76xf16, #map0, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x4x76xf16, #map0, @DDR>) outputs(%0 : memref<1x16x4x76xf16, #map0, [@CMX_NN, 0]>) -> memref<1x16x4x76xf16, #map0, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_MemPermute inputs(%1 as %arg2: memref<1x16x4x76xf16, #map0, [@CMX_NN, 0]>) outputs(%2 as %arg3: memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>{
       VPUIP.SW.Kernel.run {attrs = [[1, 0, 3, 2]]}(%arg2, %arg3) : memref<1x16x4x76xf16, #map0, [@CMX_NN, 0]>, memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x16x4x76xf16, #NHWC, @DDR>
    %4 = VPUIP.Copy inputs(%results : memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x4x76xf16, #NHWC, @DDR>) -> memref<1x16x4x76xf16, #NHWC, @DDR>
    return %4: memref<1x16x4x76xf16, #NHWC, @DDR>

    // CHECK:   [[VAR0:%.*]] = memref.alloc() : memref<1x16x4x76xf16, #map0, [@CMX_NN, 0]>
    // CHECK:   [[VAR1:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x16x4x76xf16, #map0, @DDR>) outputs([[VAR0]] : memref<1x16x4x76xf16, #map0, [@CMX_NN, 0]>) -> memref<1x16x4x76xf16, #map0, [@CMX_NN, 0]>
    // CHECK:   [[VAR2:%.*]] = memref.alloc() : memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR3:%.*]] = VPUIP.PermuteDMA {mem_perm = #map1, port = 0 : i64} inputs([[VAR1]] : memref<1x16x4x76xf16, #map0, [@CMX_NN, 0]>) outputs([[VAR2]] : memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR4:%.*]] = memref.alloc() : memref<1x16x4x76xf16, #NHWC, @DDR>
    // CHECK:   [[VAR5:%.*]] = VPUIP.Copy inputs([[VAR3]] : memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>) outputs([[VAR4]] : memref<1x16x4x76xf16, #NHWC, @DDR>) -> memref<1x16x4x76xf16, #NHWC, @DDR>
    // CHECK:   return [[VAR5]] : memref<1x16x4x76xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder_fp16.cpp", VPU.kernel_entry = "reorder_fp16"}
    func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

func @ConvertMemPermuteHWCToHCW(%arg0: memref<1x16x4x76xf16, @DDR>)
        -> memref<1x16x4x76xf16, #NHWC, @DDR> {
    %0 = memref.alloc() : memref<1x16x4x76xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x4x76xf16, @DDR>) outputs(%0 : memref<1x16x4x76xf16, [@CMX_NN, 0]>) -> memref<1x16x4x76xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_MemPermute inputs(%1 as %arg2: memref<1x16x4x76xf16, [@CMX_NN, 0]>) outputs(%2 as %arg3: memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>{
       VPUIP.SW.Kernel.run {attrs = [[1, 0, 3, 2]]}(%arg2, %arg3) : memref<1x16x4x76xf16, [@CMX_NN, 0]>, memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x16x4x76xf16, #NHWC, @DDR>
    %4 = VPUIP.Copy inputs(%results : memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x4x76xf16, #NHWC, @DDR>) -> memref<1x16x4x76xf16, #NHWC, @DDR>
    return %4: memref<1x16x4x76xf16, #NHWC, @DDR>

    // CHECK:   [[VAR0:%.*]] = memref.alloc() : memref<1x16x4x76xf16, [@CMX_NN, 0]>
    // CHECK:   [[VAR1:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x16x4x76xf16, @DDR>) outputs([[VAR0]] : memref<1x16x4x76xf16, [@CMX_NN, 0]>) -> memref<1x16x4x76xf16, [@CMX_NN, 0]>
    // CHECK:   [[VAR2:%.*]] = memref.alloc() : memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR3:%.*]] = VPUIP.PermuteDMA {mem_perm = #map, port = 0 : i64} inputs([[VAR1]] : memref<1x16x4x76xf16, [@CMX_NN, 0]>) outputs([[VAR2]] : memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR4:%.*]] = memref.alloc() : memref<1x16x4x76xf16, #NHWC, @DDR>
    // CHECK:   [[VAR5:%.*]] = VPUIP.Copy inputs([[VAR3]] : memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>) outputs([[VAR4]] : memref<1x16x4x76xf16, #NHWC, @DDR>) -> memref<1x16x4x76xf16, #NHWC, @DDR>
    // CHECK:   return [[VAR5]] : memref<1x16x4x76xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = type !quant.uniform<u8:f16, 0.0173492431640625:114>

// CHECK-LABEL: @WrapExpandandPermuteWithoutClusterTiling
func @WrapExpandandPermuteWithoutClusterTiling(%arg0: memref<1x3x24x24x!qElemType>) -> memref<1x16x24x24x!qElemType> {
   %0 = memref.alloc() : memref<1x16x24x24x!qElemType>
   %1 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} inputs(%arg0 : memref<1x3x24x24x!qElemType>) outputs(%0 : memref<1x16x24x24x!qElemType>) -> memref<1x16x24x24x!qElemType>

   return %1 : memref<1x16x24x24x!qElemType>

   //CHECK:   [[VAR0:%.*]] = memref.alloc() : memref<1x16x24x24x!qElemType>
   //CHECK:   [[VAR1:%.*]] = VPUIP.ExpandDMA {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0], port = 0 : i64} inputs(%arg0 : memref<1x3x24x24x!qElemType>) outputs([[VAR0]] : memref<1x16x24x24x!qElemType>) -> memref<1x16x24x24x!qElemType>
   //CHECK:   return [[VAR1]] : memref<1x16x24x24x!qElemType>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func private @builtin_Tile(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "single_shave_tile.cpp", VPU.kernel_entry = "single_shave_tile"}
    func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

func @ConvertPerAxisTileToDMA(%arg0: memref<1x1x1x1xf16, #NHWC, @DDR>)
        -> memref<1x512x1x1xf16, #NHWC, @DDR> {
    %cst_0 = const.Declare memref<4xsi32> = dense<[1, 512, 1, 1]> : tensor<4xsi32>
    %0 = memref.alloc() : memref<1x1x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x1x1x1xf16, #NHWC, @DDR>) outputs(%0 : memref<1x1x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x1x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    %3 = VPUIP.Copy inputs(%cst_0 : memref<4xsi32>) outputs(%2 : memref<4xsi32, [@CMX_NN, 0]>) -> memref<4xsi32, [@CMX_NN, 0]>
    %4 = memref.alloc() : memref<1x512x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %5 = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_Tile
          inputs(%1 as %arg3: memref<1x1x1x1xf16, #NHWC, [@CMX_NN, 0]>, %3 as %arg4: memref<4xsi32, [@CMX_NN, 0]>)
          outputs(%4 as %arg5: memref<1x512x1x1xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x512x1x1xf16, #NHWC, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : memref<1x1x1x1xf16, #NHWC, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>, memref<1x512x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }
    %6 = memref.alloc() : memref<1x512x1x1xf16, #NHWC, @DDR>
    %7 = VPUIP.Copy inputs(%5 : memref<1x512x1x1xf16, #NHWC, [@CMX_NN, 0]>) outputs(%6 : memref<1x512x1x1xf16, #NHWC, @DDR>) -> memref<1x512x1x1xf16, #NHWC, @DDR>
    return %7: memref<1x512x1x1xf16, #NHWC, @DDR>

    // CHECK:   [[VAR0:%.*]] = const.Declare memref<4xsi32> = dense<[1, 512, 1, 1]> : tensor<4xsi32>
    // CHECK:   [[VAR1:%.*]] = memref.alloc() : memref<1x1x1x1xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR2:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x1x1x1xf16, #NHWC, @DDR>) outputs([[VAR1]] : memref<1x1x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x1x1x1xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR3:%.*]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    // CHECK:   [[VAR4:%.*]] = VPUIP.Copy inputs(%cst : memref<4xsi32>) outputs([[VAR3]] : memref<4xsi32, [@CMX_NN, 0]>) -> memref<4xsi32, [@CMX_NN, 0]>
    // CHECK:   [[VAR5:%.*]] = memref.alloc() : memref<1x512x1x1xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR6:%.*]] = VPUIP.PerAxisTileDMA {axis = 1 : i64, port = 0 : i64, tiles = 512 : i64} inputs([[VAR2]] : memref<1x1x1x1xf16, #NHWC, [@CMX_NN, 0]>) outputs([[VAR5]] : memref<1x512x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x512x1x1xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR7:%.*]] = memref.alloc() : memref<1x512x1x1xf16, #NHWC, @DDR>
    // CHECK:   [[VAR8:%.*]] = VPUIP.Copy inputs([[VAR6]] : memref<1x512x1x1xf16, #NHWC, [@CMX_NN, 0]>) outputs([[VAR7]] : memref<1x512x1x1xf16, #NHWC, @DDR>) -> memref<1x512x1x1xf16, #NHWC, @DDR>
    // CHECK:   return [[VAR8]] : memref<1x512x1x1xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func private @builtin_Tile(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "single_shave_tile.cpp", VPU.kernel_entry = "single_shave_tile"}
    func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

func @ConvertTileToDMAWithThreeAxisExpansion(%arg0: memref<1x2x3x4xf16, #NHWC, @DDR>)
        -> memref<1x4x9x16xf16, #NHWC, @DDR> {
    %cst_0 = const.Declare memref<4xsi32> = dense<[1, 2, 3, 4]> : tensor<4xsi32>
    %0 = memref.alloc() : memref<1x2x3x4xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x2x3x4xf16, #NHWC, @DDR>) outputs(%0 : memref<1x2x3x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x2x3x4xf16, #NHWC, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    %3 = VPUIP.Copy inputs(%cst_0 : memref<4xsi32>) outputs(%2 : memref<4xsi32, [@CMX_NN, 0]>) -> memref<4xsi32, [@CMX_NN, 0]>
    %4 = memref.alloc() : memref<1x4x9x16xf16, #NHWC, [@CMX_NN, 0]>
    %5 = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_Tile
          inputs(%1 as %arg3: memref<1x2x3x4xf16, #NHWC, [@CMX_NN, 0]>, %3 as %arg4: memref<4xsi32, [@CMX_NN, 0]>)
          outputs(%4 as %arg5: memref<1x4x9x16xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x4x9x16xf16, #NHWC, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : memref<1x2x3x4xf16, #NHWC, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>, memref<1x4x9x16xf16, #NHWC, [@CMX_NN, 0]>
    }
    %6 = memref.alloc() : memref<1x4x9x16xf16, #NHWC, @DDR>
    %7 = VPUIP.Copy inputs(%5 : memref<1x4x9x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%6 : memref<1x4x9x16xf16, #NHWC, @DDR>) -> memref<1x4x9x16xf16, #NHWC, @DDR>
    return %7: memref<1x4x9x16xf16, #NHWC, @DDR>

    // CHECK:   [[VAR0:%.*]] = const.Declare memref<4xsi32> = dense<[1, 2, 3, 4]> : tensor<4xsi32>
    // CHECK:   [[VAR1:%.*]] = memref.alloc() : memref<1x2x3x4xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR2:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x2x3x4xf16, #NHWC, @DDR>) outputs(%0 : memref<1x2x3x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x2x3x4xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR3:%.*]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    // CHECK:   [[VAR4:%.*]] = VPUIP.Copy inputs(%cst : memref<4xsi32>) outputs(%2 : memref<4xsi32, [@CMX_NN, 0]>) -> memref<4xsi32, [@CMX_NN, 0]>

    // CHECK:   [[OUTBUFFER_0:%.*]] = memref.alloc() : memref<1x4x3x4xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[PERAXISTILE_0:%.*]] = VPUIP.PerAxisTileDMA {axis = 1 : i64, port = 0 : i64, tiles = 2 : i64}
    // CHECK:       inputs([[VAR2]] : memref<1x2x3x4xf16, #NHWC, [@CMX_NN, 0]>) outputs([[OUTBUFFER_0]] : memref<1x4x3x4xf16, #NHWC, [@CMX_NN, 0]>)

    // CHECK:   [[OUTBUFFER_1:%.*]] = memref.alloc() : memref<1x4x9x4xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[PERAXISTILE_1:%.*]] = VPUIP.PerAxisTileDMA {axis = 2 : i64, port = 0 : i64, tiles = 3 : i64}
    // CHECK:       inputs([[PERAXISTILE_0]] : memref<1x4x3x4xf16, #NHWC, [@CMX_NN, 0]>) outputs([[OUTBUFFER_1]] : memref<1x4x9x4xf16, #NHWC, [@CMX_NN, 0]>)

    // CHECK:   [[OUTBUFFER_2:%.*]] = memref.alloc() : memref<1x4x9x16xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[PERAXISTILE_2:%.*]] = VPUIP.PerAxisTileDMA {axis = 3 : i64, port = 0 : i64, tiles = 4 : i64}
    // CHECK:       inputs([[PERAXISTILE_1]] : memref<1x4x9x4xf16, #NHWC, [@CMX_NN, 0]>) outputs([[OUTBUFFER_2]] : memref<1x4x9x16xf16, #NHWC, [@CMX_NN, 0]>)

    // CHECK:   [[OUTBUFFER:%.*]] = memref.alloc() : memref<1x4x9x16xf16, #NHWC, @DDR>
    // CHECK:   [[OUTCOPY:%.*]] = VPUIP.Copy inputs([[PERAXISTILE_2]] : memref<1x4x9x16xf16, #NHWC, [@CMX_NN, 0]>) outputs([[OUTBUFFER]] : memref<1x4x9x16xf16, #NHWC, @DDR>)
    // CHECK:   return [[OUTCOPY]] : memref<1x4x9x16xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @convertUpsamping2DMACopyWithNHWC(%arg0: memref<1x256x16x32xf16, #NHWC>) -> memref<1x256x32x64xf16, #NHWC> {
    %0 = memref.alloc() : memref<1x256x32x64xf16, #NHWC>
    %1 = VPUIP.UpsamplingUPA {pad_l = [0, 0, 0], pad_r = [1, 1, 0], upsampling_factor = [2, 2, 1]} inputs(%arg0 : memref<1x256x16x32xf16, #NHWC>) outputs(%0 : memref<1x256x32x64xf16, #NHWC>) -> memref<1x256x32x64xf16, #NHWC>
    return %1 : memref<1x256x32x64xf16, #NHWC>

    // CHECK-DAG:   [[OUTPUT_BUFF:%.*]] = memref.alloc() : memref<1x256x32x64xf16, #NHWC>
    // CHECK-DAG:   [[CONST:%.*]] = const.Declare memref<1x256x32x64xf16, #NHWC> = dense<0.000000e+00> : tensor<1x256x32x64xf16, {order = #NHWC}>

    // CHECK:       [[ZERO_MEMORY:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[CONST]] : memref<1x256x32x64xf16, #NHWC>)
    // CHECK-SAME:      outputs([[OUTPUT_BUFF]] : memref<1x256x32x64xf16, #NHWC>) -> memref<1x256x32x64xf16, #NHWC>

    // CHECK:       [[RESULT:%.*]] = VPUIP.UpsamplingDMAOp {port = 0 : i64, upsampling_factor = [1, 1, 2, 2]}
    // CHECK-SAME:      inputs(%arg0 : memref<1x256x16x32xf16, #NHWC>)
    // CHECK-SAME:      outputs([[ZERO_MEMORY]] : memref<1x256x32x64xf16, #NHWC>) -> memref<1x256x32x64xf16, #NHWC>

    // CHECK:       return [[RESULT]] : memref<1x256x32x64xf16, #NHWC>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func @convertUpsamping2DMACopyWithNCHW(%arg0: memref<1x256x16x32xf16>) -> memref<1x256x32x64xf16> {
    %0 = memref.alloc() : memref<1x256x32x64xf16>
    %1 = VPUIP.UpsamplingUPA {pad_l = [0, 0, 0], pad_r = [1, 1, 0], upsampling_factor = [2, 2, 1]} inputs(%arg0 : memref<1x256x16x32xf16>) outputs(%0 : memref<1x256x32x64xf16>) -> memref<1x256x32x64xf16>
    return %1 : memref<1x256x32x64xf16>

    // CHECK-DAG:   [[OUTPUT_BUFF:%.*]] = memref.alloc() : memref<1x256x32x64xf16>
    // CHECK-DAG:   [[CONST:%.*]] = const.Declare memref<1x256x32x64xf16> = dense<0.000000e+00> : tensor<1x256x32x64xf16>

    // CHECK:       [[ZERO_MEMORY:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[CONST]] : memref<1x256x32x64xf16>)
    // CHECK-SAME:      outputs([[OUTPUT_BUFF]] : memref<1x256x32x64xf16>) -> memref<1x256x32x64xf16>

    // CHECK:       [[RESULT:%.*]] = VPUIP.UpsamplingDMAOp {port = 0 : i64, upsampling_factor = [1, 1, 2, 2]}
    // CHECK-SAME:      inputs(%arg0 : memref<1x256x16x32xf16>)
    // CHECK-SAME:      outputs([[ZERO_MEMORY]] : memref<1x256x32x64xf16>) -> memref<1x256x32x64xf16>

    // CHECK:       return [[RESULT]] : memref<1x256x32x64xf16>
}
