//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX compilation-mode=DefaultHW" --convert-to-dma %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertDepthToSpace_BLOCKS_FIRST
func @ConvertDepthToSpace_BLOCKS_FIRST(%arg0: memref<1x8x2x3xf16, #NHWC, @DDR>) -> memref<1x2x4x6xf16, #NHWC, @DDR> {
    %outBuffer = memref.alloc() : memref<1x2x4x6xf16, #NHWC, @DDR>
    %depthToSpace = VPUIP.DepthToSpaceUPA {block_size = 2 : i64, mode = "BLOCKS_FIRST"}
                inputs(%arg0 : memref<1x8x2x3xf16, #NHWC, @DDR>)
                outputs(%outBuffer : memref<1x2x4x6xf16, #NHWC, @DDR>) -> memref<1x2x4x6xf16, #NHWC, @DDR>
    
    return %depthToSpace : memref<1x2x4x6xf16, #NHWC, @DDR>

    //CHECK:    [[OUTBUFFER:%.*]] = memref.alloc() : memref<1x2x4x6xf16, #NHWC, @DDR>
    //CHECK:    [[COPYBUFFER_0:%.*]] = memref.alloc() : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[COPYOUT_0:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x8x2x3xf16, #NHWC, @DDR>)
    //CHECK:            outputs([[COPYBUFFER_0]] : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[OPBUFFER:%.*]] = memref.alloc() : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[DepthToSpaceDMAOUT:%.*]] = VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = "BLOCKS_FIRST", port = 0 : i64}
    //CHECK:            inputs([[COPYOUT_0]] : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OPBUFFER]] : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[COPYBUFFER_1:%.*]] = memref.alloc() : memref<1x2x4x6xf16, #NHWC, @DDR>
    //CHECK:    [[COPYOUT_1:%.*]] = VPUIP.Copy inputs([[DepthToSpaceDMAOUT]] : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:            outputs([[COPYBUFFER_1]] : memref<1x2x4x6xf16, #NHWC, @DDR>) -> memref<1x2x4x6xf16, #NHWC, @DDR>

    //CHECK:    return [[COPYOUT_1]] : memref<1x2x4x6xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertDepthToSpace_DEPTH_FIRST
func @ConvertDepthToSpace_DEPTH_FIRST(%arg0: memref<1x8x2x3xf16, #NHWC, @DDR>) -> memref<1x2x4x6xf16, #NHWC, @DDR> {
    %outBuffer = memref.alloc() : memref<1x2x4x6xf16, #NHWC, @DDR>
    %depthToSpace = VPUIP.DepthToSpaceUPA {block_size = 2 : i64, mode = "DEPTH_FIRST"}
                inputs(%arg0 : memref<1x8x2x3xf16, #NHWC, @DDR>)
                outputs(%outBuffer : memref<1x2x4x6xf16, #NHWC, @DDR>) -> memref<1x2x4x6xf16, #NHWC, @DDR>

    return %depthToSpace : memref<1x2x4x6xf16, #NHWC, @DDR>

    //CHECK:    [[OUTBUFFER:%.*]] = memref.alloc() : memref<1x2x4x6xf16, #NHWC, @DDR>
    //CHECK:    [[COPYBUFFER_0:%.*]] = memref.alloc() : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[COPYOUT_0:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x8x2x3xf16, #NHWC, @DDR>)
    //CHECK:            outputs([[COPYBUFFER_0]] : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[OPBUFFER:%.*]] = memref.alloc() : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[DepthToSpaceDMAOUT:%.*]] = VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = "DEPTH_FIRST", port = 0 : i64}
    //CHECK:            inputs([[COPYOUT_0]] : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OPBUFFER]] : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[COPYBUFFER_1:%.*]] = memref.alloc() : memref<1x2x4x6xf16, #NHWC, @DDR>
    //CHECK:    [[COPYOUT_1:%.*]] = VPUIP.Copy inputs([[DepthToSpaceDMAOUT]] : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:            outputs([[COPYBUFFER_1]] : memref<1x2x4x6xf16, #NHWC, @DDR>) -> memref<1x2x4x6xf16, #NHWC, @DDR>

    //CHECK:    return [[COPYOUT_1]] : memref<1x2x4x6xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

func @ConvertMemPermute(%arg0: memref<1x8x16x16xf16, #NHWC, @DDR>)
        -> memref<1x8x16x16xf16, @DDR> {
    %buf1 = memref.alloc() : memref<1x8x16x16xf16, #NCHW, @DDR>
    %0 = VPUIP.PermuteUPA {order_value = #NWCH} inputs(%arg0 : memref<1x8x16x16xf16, #NHWC, @DDR>) outputs(%buf1 : memref<1x8x16x16xf16, #NCHW, @DDR>) -> memref<1x8x16x16xf16, @DDR>
    return %0: memref<1x8x16x16xf16, @DDR>
    // CHECK:   [[VAR0:%.*]] = memref.alloc() : memref<1x8x16x16xf16, @DDR>
    // CHECK:   [[VAR1:%.*]] = memref.alloc() : memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR2:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x8x16x16xf16, #NHWC, @DDR>) outputs([[VAR1]] : memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR3:%.*]] = memref.alloc() : memref<1x8x16x16xf16, [@CMX_NN, 0]>
    // CHECK:   [[VAR4:%.*]] = VPUIP.PermuteDMA {mem_perm = #NWCH, port = 0 : i64} inputs([[VAR2]] : memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs([[VAR3]] : memref<1x8x16x16xf16, [@CMX_NN, 0]>) -> memref<1x8x16x16xf16, [@CMX_NN, 0]>
    // CHECK:   [[VAR5:%.*]] = memref.alloc() : memref<1x8x16x16xf16, @DDR>
    // CHECK:   [[VAR6:%.*]] = VPUIP.Copy inputs([[VAR4]] : memref<1x8x16x16xf16, [@CMX_NN, 0]>) outputs([[VAR5]] : memref<1x8x16x16xf16, @DDR>) -> memref<1x8x16x16xf16, @DDR>
    // CHECK:   return [[VAR6]] : memref<1x8x16x16xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

func @ConvertOutputMemPermute(%arg0: memref<1x8x16x16xf16, #NHWC, @DDR>, %arg1: memref<1x8x16x16xf16>)
        -> memref<1x8x16x16xf16> {
    %0 = VPUIP.PermuteUPA {order_value = #NWCH} inputs(%arg0 : memref<1x8x16x16xf16, #NHWC, @DDR>) outputs(%arg1 : memref<1x8x16x16xf16>) -> memref<1x8x16x16xf16>
    return %arg1: memref<1x8x16x16xf16>
    // CHECK:   [[VAR0:%.*]] = memref.alloc() : memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR1:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x8x16x16xf16, #NHWC, @DDR>) outputs([[VAR0]] : memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR2:%.*]] = memref.alloc() : memref<1x8x16x16xf16, [@CMX_NN, 0]>
    // CHECK:   [[VAR3:%.*]] = VPUIP.PermuteDMA {mem_perm = #NWCH, port = 0 : i64} inputs([[VAR1]] : memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs([[VAR2]] : memref<1x8x16x16xf16, [@CMX_NN, 0]>) -> memref<1x8x16x16xf16, [@CMX_NN, 0]>
    // CHECK:   [[VAR6:%.*]] = VPUIP.Copy inputs([[VAR3]] : memref<1x8x16x16xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x8x16x16xf16>) -> memref<1x8x16x16xf16>
    // CHECK:   return %arg1 : memref<1x8x16x16xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

func @CanNotMoveMemPermuteFromDDRToCMX(%arg0: memref<1x8x16x16xf16, #NHWC, @DDR>)
        -> memref<1x8x16x16xf16, #NCWH, @DDR> {
    %buf = memref.alloc() : memref<1x8x16x16xf16, #NCWH, @DDR>
    %0 = VPUIP.PermuteUPA {order_value = #NWHC} inputs(%arg0 : memref<1x8x16x16xf16, #NHWC, @DDR>) outputs(%buf : memref<1x8x16x16xf16, #NCWH, @DDR>) -> memref<1x8x16x16xf16, #NCWH, @DDR>
    return %0: memref<1x8x16x16xf16, #NCWH, @DDR>

    // CHECK-NOT:   VPUIP.PermuteDMA
    // CHECK:       [[VAR:%.*]] = memref.alloc() : memref<1x8x16x16xf16, #NCWH, @DDR>
    // CHECK:       [[PERMUTEUPA:%.*]] = VPUIP.PermuteUPA {order_value = #NWHC}
    // CHECK-SAME:      inputs(%arg0 : memref<1x8x16x16xf16, #NHWC, @DDR>) outputs([[VAR]] : memref<1x8x16x16xf16, #NCWH, @DDR>) -> memref<1x8x16x16xf16, #NCWH, @DDR>
    // CHECK:       return [[PERMUTEUPA]] : memref<1x8x16x16xf16, #NCWH, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

func @CanNotMoveMemPermuteFromDDRToCMXWithLargeDMANumber(%arg0: memref<1x8x256x256xf16, #NHWC, @DDR>)
        -> memref<1x8x256x256xf16, @DDR> {
    %buf = memref.alloc() : memref<1x8x256x256xf16, @DDR>
    %0 = VPUIP.PermuteUPA {order_value = #NWCH} inputs(%arg0 : memref<1x8x256x256xf16, #NHWC, @DDR>) outputs(%buf : memref<1x8x256x256xf16, @DDR>) -> memref<1x8x256x256xf16, @DDR>
    return %0: memref<1x8x256x256xf16, @DDR>

    // CHECK-NOT:   VPUIP.PermuteDMA
    // CHECK:       [[VAR:%.*]] = memref.alloc() : memref<1x8x256x256xf16, @DDR>
    // CHECK:       [[PERMUTEUPA:%.*]] = VPUIP.PermuteUPA {order_value = #NWCH}
    // CHECK-SAME:      inputs(%arg0 : memref<1x8x256x256xf16, #NHWC, @DDR>) outputs([[VAR]] : memref<1x8x256x256xf16, @DDR>) -> memref<1x8x256x256xf16, @DDR>
    // CHECK:       return [[PERMUTEUPA]] : memref<1x8x256x256xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

func @CanNotMoveMemPermuteFromDDRToCMXWithLargeLenth(%arg0: memref<1x512x32x32xf16, #NHWC, @DDR>)
        -> memref<1x512x32x32xf16, @DDR> {
    %buf = memref.alloc() : memref<1x512x32x32xf16, @DDR>
    %0 = VPUIP.PermuteUPA {order_value = #NWCH} inputs(%arg0 : memref<1x512x32x32xf16, #NHWC, @DDR>) outputs(%buf : memref<1x512x32x32xf16, @DDR>) -> memref<1x512x32x32xf16, @DDR>
    return %0: memref<1x512x32x32xf16, @DDR>

    // CHECK-NOT:   VPUIP.PermuteDMA
    // CHECK:       [[VAR:%.*]] = memref.alloc() : memref<1x512x32x32xf16, @DDR>
    // CHECK:       [[PERMUTEUPA:%.*]] = VPUIP.PermuteUPA {order_value = #NWCH}
    // CHECK-SAME:      inputs(%arg0 : memref<1x512x32x32xf16, #NHWC, @DDR>) outputs([[VAR]] : memref<1x512x32x32xf16, @DDR>) -> memref<1x512x32x32xf16, @DDR>
    // CHECK:       return [[PERMUTEUPA]] : memref<1x512x32x32xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

func @ConvertMemPermuteWithThreeAxis(%arg0: memref<1x16x4x128xf16, @DDR>)
        -> memref<1x4x16x128xf16, #NHWC, @DDR> {
    %buf1 = memref.alloc() : memref<1x4x16x128xf16, #NHWC, @DDR>
    %0 = VPUIP.PermuteUPA {order_value = #NHCW} inputs(%arg0 : memref<1x16x4x128xf16, @DDR>) outputs(%buf1 : memref<1x4x16x128xf16, #NHWC, @DDR>) -> memref<1x4x16x128xf16, #NHWC, @DDR>
    return %0: memref<1x4x16x128xf16, #NHWC, @DDR>
    // CHECK:   [[VAR0:%.*]] = memref.alloc() : memref<1x4x16x128xf16, #NHWC, @DDR>
    // CHECK:   [[VAR1:%.*]] = memref.alloc() : memref<1x16x4x128xf16, [@CMX_NN, 0]>
    // CHECK:   [[VAR2:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x16x4x128xf16, @DDR>) outputs([[VAR1]] : memref<1x16x4x128xf16, [@CMX_NN, 0]>) -> memref<1x16x4x128xf16, [@CMX_NN, 0]>
    // CHECK:   [[VAR3:%.*]] = memref.alloc() : memref<1x4x16x128xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR4:%.*]] = VPUIP.PermuteDMA {mem_perm = #NHCW, port = 0 : i64} inputs([[VAR2]] : memref<1x16x4x128xf16, [@CMX_NN, 0]>) outputs([[VAR3]] : memref<1x4x16x128xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x4x16x128xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR5:%.*]] = memref.alloc() : memref<1x4x16x128xf16, #NHWC, @DDR>
    // CHECK:   [[VAR6:%.*]] = VPUIP.Copy inputs([[VAR4]] : memref<1x4x16x128xf16, #NHWC, [@CMX_NN, 0]>) outputs([[VAR5]] : memref<1x4x16x128xf16, #NHWC, @DDR>) -> memref<1x4x16x128xf16, #NHWC, @DDR>
    // CHECK:   return [[VAR6]] : memref<1x4x16x128xf16, #NHWC, @DDR>
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
   //CHECK:   [[VAR1:%.*]] = memref.alloc() : memref<1x16x24x24x!qElemType>
   //CHECK:   [[VAR2:%.*]] = VPUIP.ExpandDMA {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0], port = 0 : i64} inputs(%arg0 : memref<1x3x24x24x!qElemType>) outputs([[VAR1]] : memref<1x16x24x24x!qElemType>) -> memref<1x16x24x24x!qElemType>
   //CHECK:   return [[VAR2]] : memref<1x16x24x24x!qElemType>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @ConvertPerAxisTileToDMA(%arg0: memref<1x1x1x1xf16, #NHWC, @DDR>)
        -> memref<1x512x1x1xf16, #NHWC, @DDR> {
    %outBuffer = memref.alloc() : memref<1x512x1x1xf16, #NHWC, @DDR>
    %0 = VPUIP.PerAxisTileUPA {axis = 1 : i64, tiles = 512 : i64} inputs(%arg0 : memref<1x1x1x1xf16, #NHWC, @DDR>) outputs(%outBuffer : memref<1x512x1x1xf16, #NHWC, @DDR>) -> memref<1x512x1x1xf16, #NHWC, @DDR>
    return %0: memref<1x512x1x1xf16, #NHWC, @DDR>

    // CHECK:   [[VAR0:%.*]] = memref.alloc() : memref<1x512x1x1xf16, #NHWC, @DDR>
    // CHECK:   [[VAR1:%.*]] = memref.alloc() : memref<1x1x1x1xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR2:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x1x1x1xf16, #NHWC, @DDR>) outputs([[VAR1]] : memref<1x1x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x1x1x1xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR3:%.*]] = memref.alloc() : memref<1x512x1x1xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR4:%.*]] = VPUIP.PerAxisTileDMA {axis = 1 : i64, port = 0 : i64, tiles = 512 : i64} inputs([[VAR2]] : memref<1x1x1x1xf16, #NHWC, [@CMX_NN, 0]>) outputs([[VAR3]] : memref<1x512x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x512x1x1xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR5:%.*]] = memref.alloc() : memref<1x512x1x1xf16, #NHWC, @DDR>
    // CHECK:   [[VAR6:%.*]] = VPUIP.Copy inputs(%4 : memref<1x512x1x1xf16, #NHWC, [@CMX_NN, 0]>) outputs(%5 : memref<1x512x1x1xf16, #NHWC, @DDR>) -> memref<1x512x1x1xf16, #NHWC, @DDR>
    // CHECK:   return [[VAR6]] : memref<1x512x1x1xf16, #NHWC, @DDR>
}
