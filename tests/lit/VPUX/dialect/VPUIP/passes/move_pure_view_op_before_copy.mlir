//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --move-pure-view-op-before-copy %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @MovePureViewOpBeforeCopyMultipleConsumers(
        %arg0: memref<1x16x112x112xf16, #NHWC, @CMX>,
        %arg1: memref<1x16x112x112xf16, #NCHW, @DDR>)
        -> (memref<1x16x56x224xf16, #NHWC, @DDR>, memref<1x16x112x112xf16, #NCHW, @DDR>) {
    %0 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x112x112xf16, #NHWC, @CMX>)
        outputs(%0 : memref<1x16x112x112xf16, #NHWC, @DDR>)
        -> memref<1x16x112x112xf16, #NHWC, @DDR>

    %2 = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>}
            inputs(%1 : memref<1x16x112x112xf16, #NHWC, @DDR>) -> memref<1x16x112x112xf16, #NCHW, @DDR>

    %3 = VPUIP.GenericReshape inputs(%1 : memref<1x16x112x112xf16, #NHWC, @DDR>) -> memref<1x16x56x224xf16, #NHWC, @DDR>

    %4 = VPUIP.Copy inputs(%2 : memref<1x16x112x112xf16, #NCHW, @DDR>)
        outputs(%arg1 : memref<1x16x112x112xf16, #NCHW, @DDR>)
        -> memref<1x16x112x112xf16, #NCHW, @DDR>

    return %3, %4 : memref<1x16x56x224xf16, #NHWC, @DDR>, memref<1x16x112x112xf16, #NCHW, @DDR>


    // CHECK: [[PERMUTECAST:%.*]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NWCH} inputs(%arg0 : memref<1x16x112x112xf16, #NHWC, @CMX>) -> memref<1x16x112x112xf16, @CMX>
    // CHECK: [[ALLOC0:%.*]] = memref.alloc() : memref<1x16x112x112xf16, @DDR>
    // CHECK: [[COPY0:%.*]] = VPUIP.Copy inputs([[PERMUTECAST]] : memref<1x16x112x112xf16, @CMX>) outputs([[ALLOC0]] : memref<1x16x112x112xf16, @DDR>) -> memref<1x16x112x112xf16, @DDR>

    // CHECK: [[GENERICRESHAPE:%.*]] = VPUIP.GenericReshape inputs(%arg0 : memref<1x16x112x112xf16, #NHWC, @CMX>) -> memref<1x16x56x224xf16, #NHWC, @CMX>
    // CHECK: [[ALLOC1:%.*]] = memref.alloc() : memref<1x16x56x224xf16, #NHWC, @DDR>
    // CHECK: [[COPY1:%.*]] = VPUIP.Copy inputs([[GENERICRESHAPE]] : memref<1x16x56x224xf16, #NHWC, @CMX>) outputs([[ALLOC1]] : memref<1x16x56x224xf16, #NHWC, @DDR>) -> memref<1x16x56x224xf16, #NHWC, @DDR>

    // CHECK: [[COPY2:%.*]] = VPUIP.Copy inputs([[COPY0]] : memref<1x16x112x112xf16, @DDR>) outputs(%arg1 : memref<1x16x112x112xf16, @DDR>) -> memref<1x16x112x112xf16, @DDR>

    // CHECK: return [[COPY1]], [[COPY2]] : memref<1x16x56x224xf16, #NHWC, @DDR>, memref<1x16x112x112xf16, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @MovePureViewOpBeforeCopy(
        %arg0: memref<1x16x112x112xf16, #NHWC, @CMX>,
        %arg1: memref<1x16x112x112xf16, #NCHW, @DDR>)
        -> memref<1x16x112x112xf16, #NCHW, @DDR> {
    %0 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x112x112xf16, #NHWC, @CMX>)
        outputs(%0 : memref<1x16x112x112xf16, #NHWC, @DDR>)
        -> memref<1x16x112x112xf16, #NHWC, @DDR>

    %2 = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>}
            inputs(%1 : memref<1x16x112x112xf16, #NHWC, @DDR>) -> memref<1x16x112x112xf16, #NCHW, @DDR>

    %3 = VPUIP.Copy inputs(%2 : memref<1x16x112x112xf16, #NCHW, @DDR>)
        outputs(%arg1 : memref<1x16x112x112xf16, #NCHW, @DDR>)
        -> memref<1x16x112x112xf16, #NCHW, @DDR>

    return %3 : memref<1x16x112x112xf16, #NCHW, @DDR>

    //CHECK:        [[VAR0:%.+]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NWCH}
    //CHECK-SAME:                   inputs(%arg0 : memref<1x16x112x112xf16, #NHWC, @CMX>) -> memref<1x16x112x112xf16, @CMX>

    //CHECK:        [[VAR1:%.+]] = memref.alloc() : memref<1x16x112x112xf16, @DDR>
    //CHECK:        [[VAR2:%.+]] = VPUIP.Copy inputs([[VAR0]] : memref<1x16x112x112xf16, @CMX>)
    //CHECK-SAME:                   outputs([[VAR1]] : memref<1x16x112x112xf16, @DDR>) -> memref<1x16x112x112xf16, @DDR>

    //CHECK:        [[VAR3:%.+]] = VPUIP.Copy inputs([[VAR2]] : memref<1x16x112x112xf16, @DDR>)
    //CHECK-SAME:                   outputs(%arg1 : memref<1x16x112x112xf16, @DDR>) -> memref<1x16x112x112xf16, @DDR>

    //CHECK:        return [[VAR3]] : memref<1x16x112x112xf16, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @MoveSeveralPureViewOpsBeforeCopy(
        %arg0: memref<1x16x112x112xf16, #NHWC, @CMX>,
        %arg1: memref<1x16x392x32xf16, @DDR>)
        -> memref<1x16x392x32xf16, @DDR> {
    %0 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x112x112xf16, #NHWC, @CMX>)
        outputs(%0 : memref<1x16x112x112xf16, #NHWC, @DDR>)
        -> memref<1x16x112x112xf16, #NHWC, @DDR>

    %2 = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>}
            inputs(%1 : memref<1x16x112x112xf16, #NHWC, @DDR>) -> memref<1x16x112x112xf16, @DDR>

    %3 = VPUIP.GenericReshape inputs(%2 : memref<1x16x112x112xf16, @DDR>) -> memref<1x16x12544xf16, @DDR>

    %4 = memref.alloc() : memref<1x16x12544xf16, @DDR>
    %5 = VPUIP.Copy inputs(%3 : memref<1x16x12544xf16, @DDR>)
        outputs(%4 : memref<1x16x12544xf16, @DDR>)
        -> memref<1x16x12544xf16, @DDR>

    %6 = VPUIP.GenericReshape inputs(%5 : memref<1x16x12544xf16, @DDR>) -> memref<1x16x392x32xf16, @DDR>

    %7 = VPUIP.Copy inputs(%6 : memref<1x16x392x32xf16, @DDR>)
        outputs(%arg1 : memref<1x16x392x32xf16, @DDR>)
        -> memref<1x16x392x32xf16, @DDR>

    return %7 : memref<1x16x392x32xf16, @DDR>

    //CHECK:        [[VAR0:%.+]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NWCH}
    //CHECK-SAME:                   inputs(%arg0 : memref<1x16x112x112xf16, #NHWC, @CMX>) -> memref<1x16x112x112xf16, @CMX>
    //CHECK:        [[VAR1:%.+]] = VPUIP.GenericReshape inputs([[VAR0]] : memref<1x16x112x112xf16, @CMX>) -> memref<1x16x12544xf16, @CMX>
    //CHECK:        [[VAR2:%.+]] = VPUIP.GenericReshape inputs([[VAR1]] : memref<1x16x12544xf16, @CMX>) -> memref<1x16x392x32xf16, @CMX>

    //CHECK:        [[VAR3:%.+]] = memref.alloc() : memref<1x16x392x32xf16, @DDR>
    //CHECK:        [[VAR4:%.+]] = VPUIP.Copy inputs([[VAR2]] : memref<1x16x392x32xf16, @CMX>) outputs([[VAR3]] : memref<1x16x392x32xf16, @DDR>)
    //CHECK:        [[VAR5:%.+]] = memref.alloc() : memref<1x16x392x32xf16, @DDR>
    //CHECK:        [[VAR6:%.+]] = VPUIP.Copy inputs([[VAR4]] : memref<1x16x392x32xf16, @DDR>) outputs([[VAR5]] : memref<1x16x392x32xf16, @DDR>)
    //CHECK:        [[VAR7:%.+]] = VPUIP.Copy inputs([[VAR6]] : memref<1x16x392x32xf16, @DDR>) outputs(%arg1 : memref<1x16x392x32xf16, @DDR>)
    //CHECK:        return [[VAR7]] : memref<1x16x392x32xf16, @DDR>
}

// -----

!qElemType0 = type !quant.uniform<u8:f16, 0.0036305147058823531>
!qElemType1 = type !quant.uniform<u8:f16, 0.0042424242424242424>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @QuantizeCastBeforeClusterTilingCopy(%arg0: memref<1x128x8x8x!qElemType0, #NHWC, @CMX_NN>) -> memref<1x128x8x8x!qElemType1, #NHWC, @CMX_NN> {
    %buf0 = memref.alloc() : memref<1x128x8x8x!qElemType0, #NHWC, @DDR>
    %0 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x8x8x!qElemType0, #NHWC, @CMX_NN>)
                                outputs(%buf0 as %arg2: memref<1x128x8x8x!qElemType0, #NHWC>) -> memref<1x128x8x8x!qElemType0, #NHWC, @DDR> {
        %3 = VPUIP.Copy inputs(%arg1 : memref<1x128x8x8x!qElemType0, #NHWC, @CMX_NN>)
                        outputs(%arg2 : memref<1x128x8x8x!qElemType0, #NHWC>) -> memref<1x128x8x8x!qElemType0, #NHWC>
    }

    %1 = VPUIP.QuantizeCast inputs(%0 : memref<1x128x8x8x!qElemType0, #NHWC, @DDR>) -> memref<1x128x8x8x!qElemType1, #NHWC, @DDR>

    %buf1 = memref.alloc() : memref<1x128x8x8x!qElemType1, #NHWC, @CMX_NN>
    %2 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x128x8x8x!qElemType1, #NHWC>)
                                outputs(%buf1 as %arg2: memref<1x128x8x8x!qElemType1, #NHWC, @CMX_NN>) -> memref<1x128x8x8x!qElemType1, #NHWC, @CMX_NN> {
        %4 = VPUIP.Copy inputs(%arg1 : memref<1x128x8x8x!qElemType1, #NHWC>)
                        outputs(%arg2 : memref<1x128x8x8x!qElemType1, #NHWC, @CMX_NN>) -> memref<1x128x8x8x!qElemType1, #NHWC, @CMX_NN>
    }

    return %2 : memref<1x128x8x8x!qElemType1, #NHWC, @CMX_NN>

    // CHECK:       [[VAR0:%.*]] = VPUIP.QuantizeCast inputs(%arg0 :
    // CHECK:       [[VAR1:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[VAR0]]
    // CHECK:       [[VAR2:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[VAR1]]
    // CHECK:       return [[VAR2]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func @MoveSubviewToTheFrontOfCopy(%arg0: memref<1x16x2x2xf16, @DDR>, %arg1: memref<1x8x2x2xf16, @DDR>) -> memref<1x8x2x2xf16, @DDR> {
    %1 = memref.alloc() : memref<1x16x2x2xf16, @DDR>
    %2 = VPUIP.Copy inputs(%arg0: memref<1x16x2x2xf16, @DDR>) outputs(%1 : memref<1x16x2x2xf16, @DDR>) -> memref<1x16x2x2xf16, @DDR>
    %3 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 8, 2, 2] : memref<1x16x2x2xf16, @DDR> to memref<1x8x2x2xf16, {order = #NCHW, strides = [64, 4, 2, 1]}, @DDR>
    %4 = VPUIP.Copy inputs(%3 : memref<1x8x2x2xf16, {order = #NCHW, strides = [64, 4, 2, 1]}, @DDR>) outputs(%arg1 : memref<1x8x2x2xf16, @DDR>) -> memref<1x8x2x2xf16, @DDR>

    return %4 : memref<1x8x2x2xf16, @DDR>

    // CHECK:       [[VAR0:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 8, 2, 2] :
    // CHECK-SAME:                      memref<1x16x2x2xf16, @DDR> to memref<1x8x2x2xf16, {order = #NCHW, strides = [64, 4, 2, 1]}, @DDR>

    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<1x8x2x2xf16, @DDR>
    // CHECK:       [[VAR2:%.*]] = VPUIP.Copy inputs([[VAR0]] : memref<1x8x2x2xf16, {order = #NCHW, strides = [64, 4, 2, 1]}, @DDR>)
    // CHECK-SAME:                      outputs([[VAR1]] : memref<1x8x2x2xf16, @DDR>)

    // CHECK:       [[VAR3:%.*]] = VPUIP.Copy inputs([[VAR2]] : memref<1x8x2x2xf16, @DDR>)
    // CHECK-SAME:                      outputs(%arg1 : memref<1x8x2x2xf16, @DDR>)

    // CHECK:       return [[VAR3]] : memref<1x8x2x2xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = type !quant.uniform<u8:f16, 5.7832517137714463:123>

 func @MoveSubviewToTheFrontOfTillingCopy(%in0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>,
                                %in1 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
                                    -> memref<1x32x48x88x!qElemType, #NHWC, @DDR> {
    %0 = memref.alloc() : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>
    %1 = VPUIP.NCEClusterTiling inputs(%in0 as %arg2: memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>,
                                       %in1 as %arg3: memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
                                outputs(%0 as %arg4: memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
                                    -> !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
        %1232 = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i64, minimumHardwareExecutionCost = 1081 : i64, task_type = "ELTWISE"}
            input(%arg2 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
            weights(%arg3 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
            parent_input(%arg2 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
            parent_output(%arg4 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
            outputs(%arg4 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
                -> memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN> variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [87, 47, 63], mpe_mode = "MATRIX", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0]}
        } PPE :  {
            PPETask "ADD" {clamp_high = 131 : i64, clamp_low = -124 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [26565], quant_post_shift = 0 : i64, quant_shift = [15]}
        }
    }

    %2 = memref.alloc() : memref<1x64x48x88x!qElemType, #NHWC, @DDR>
    %3 = VPUIP.NCEClusterTiling inputs(%1 as %arg2: memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
                                outputs(%2 as %arg3: memref<1x64x48x88x!qElemType, #NHWC>)
                                    -> memref<1x64x48x88x!qElemType, #NHWC, @DDR> {
        %1232 = VPUIP.Copy inputs(%arg2 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
                           outputs(%arg3 : memref<1x64x48x88x!qElemType, #NHWC>)
                               -> memref<1x64x48x88x!qElemType, #NHWC>
    }
    %4 = VPUIP.SubView %3 [0, 0, 0, 0] [1, 32, 48, 88] : memref<1x64x48x88x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR> to memref<1x32x48x88x!qElemType, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [270336, 1, 5632, 64]}, @DDR>
    %5 = memref.alloc() : memref<1x32x48x88x!qElemType, #NHWC, @DDR>
    %6 = VPUIP.Copy inputs(%4 : memref<1x32x48x88x!qElemType, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [270336, 1, 5632, 64]}, @DDR>) outputs(%5 : memref<1x32x48x88x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>) -> memref<1x32x48x88x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
    return %6 : memref<1x32x48x88x!qElemType, #NHWC, @DDR>

    // CHECK:       [[BUFF_0:%.*]] = memref.alloc() : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>
    // CHECK:       [[ADD_0:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs(%arg0 as %arg2: memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>,
    // CHECK-SAME:             %arg1 as %arg3: memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUFF_0]] as %arg4: memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
    // CHECK:           [[ADD_0_INNER:%.*]] = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i64, minimumHardwareExecutionCost = 1081 : i64, task_type = "ELTWISE"}
    // CHECK-SAME:          input(%arg2 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:          weights(%arg3 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:          parent_input(%arg2 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:          parent_output(%arg4 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:          outputs(%arg4 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:              -> memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN> variants :  {
    // CHECK:                   DPUTask {cluster_id = 0 : i64, mpe_mode = "MATRIX", outEnd = [87, 47, 63], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
    // CHECK:           } PPE :  {
    // CHECK:                   PPETask "ADD" {clamp_high = 131 : i64, clamp_low = -124 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [26565], quant_post_shift = 0 : i64, quant_shift = [15]}
    // CHECK:           }
    // CHECK:       }
    // CHECK:       [[SUBVIEW:%.*]] = VPUIP.SubView [[ADD_0]] [0, 0, 0, 0] [1, 32, 48, 88] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> to !VPUIP.DistributedBuffer<1x32x48x88x!qElemType, {order = #NHWC, strides = [270336, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_1:%.*]] = memref.alloc() : memref<1x32x48x88x!qElemType, #NHWC, @DDR>
    // CHECK:       [[Tilling_COPY:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[SUBVIEW]] as %arg2: memref<1x32x48x88x!qElemType, {order = #NHWC, strides = [270336, 1, 5632, 64]}, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUFF_1]] as %arg3: memref<1x32x48x88x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x32x48x88x!qElemType, #NHWC, @DDR> {
    // CHECK:           [[COPY_2_INNER:%.*]] = VPUIP.Copy
    // CHECK-SAME:          inputs(%arg2 : memref<1x32x48x88x!qElemType, {order = #NHWC, strides = [270336, 1, 5632, 64]}, @CMX_NN>)
    // CHECK-SAME:          outputs(%arg3 : memref<1x32x48x88x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:              -> memref<1x32x48x88x!qElemType, #NHWC, @DDR>
    // CHECK:       }

    // CHECK:       [[BUFF_2:%.*]] = memref.alloc() : memref<1x32x48x88x!qElemType, #NHWC, @DDR>
    // CHECK:       [[COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[Tilling_COPY]] : memref<1x32x48x88x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[BUFF_2]] : memref<1x32x48x88x!qElemType, #NHWC, @DDR>)
    // CHECK:       return [[COPY]] : memref<1x32x48x88x!qElemType, #NHWC, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @NoChangesForStridedCopy(
        %arg0: memref<1x2x16x4xf16, @DDR>)
        -> memref<16x4xf16, @DDR> {
    %0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 1, 16, 4] : memref<1x2x16x4xf16, @DDR> to memref<1x1x16x4xf16, {order = #NCHW, strides = [128, 64, 4, 1]}, @DDR>

    %1 = memref.alloc() : memref<1x1x16x4xf16, @DDR>
    %2 = VPUIP.Copy inputs(%0 : memref<1x1x16x4xf16, {order = #NCHW, strides = [128, 64, 4, 1]}, @DDR>)
        outputs(%1 : memref<1x1x16x4xf16, @DDR>)
        -> memref<1x1x16x4xf16, @DDR>

    %3 = VPUIP.GenericReshape inputs(%2 : memref<1x1x16x4xf16, @DDR>) -> memref<16x4xf16, @DDR>

    return %3 : memref<16x4xf16, @DDR>

    // CHECK:       [[SUBVIEW:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 1, 16, 4] : memref<1x2x16x4xf16, @DDR> to memref<1x1x16x4xf16, {order = #NCHW, strides = [128, 64, 4, 1]}, @DDR>

    // CHECK:       [[BUFF_0:%.*]] = memref.alloc() : memref<1x1x16x4xf16, @DDR>
    // CHECK:       [[COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW]] : memref<1x1x16x4xf16, {order = #NCHW, strides = [128, 64, 4, 1]}, @DDR>)
    // CHECK-SAME:      outputs([[BUFF_0]] : memref<1x1x16x4xf16, @DDR>)
    // CHECK:       [[RESHAPE:%.*]] = VPUIP.GenericReshape inputs([[COPY]] : memref<1x1x16x4xf16, @DDR>) -> memref<16x4xf16, @DDR>

    // CHECK:       return [[RESHAPE]] : memref<16x4xf16, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!SparseInputActBufferType = type !VPUIP.SparseBuffer<
    data=memref<1x16x112x112xf16, #NHWC, @CMX>,
    sparsity_map=memref<1x16x112x112xi1, #NHWC, @CMX>
>

!SparseOutputActBufferType = type !VPUIP.SparseBuffer<
    data=memref<1x16x112x112xf16, #NCHW, @DDR>,
    sparsity_map=memref<1x16x112x112xi1, #NCHW, @DDR>
>

!IZMajorDDRType = type memref<1x16x112x112xf16, #NHWC, @DDR>
!IZMajorSMDDRType = type memref<1x16x112x112xi1, #NHWC, @DDR>
!IZMajorSparseType = type !VPUIP.SparseBuffer<data=!IZMajorDDRType, sparsity_map=!IZMajorSMDDRType>

func @MovePureViewOpBeforeCopySparse(
        %arg0: !SparseInputActBufferType,
        %arg1: !SparseOutputActBufferType)
        -> !SparseOutputActBufferType {
    %0 = memref.alloc() : !IZMajorDDRType
    %1 = memref.alloc() : !IZMajorSMDDRType
    %2 = VPUIP.GroupSparseBuffer(%0, %1) -> !IZMajorSparseType
    %3 = VPUIP.Copy inputs(%arg0 : !SparseInputActBufferType )
        outputs(%2 : !IZMajorSparseType )
        -> !IZMajorSparseType

    %4 = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW}
            inputs(%3 : !IZMajorSparseType ) -> !SparseOutputActBufferType

    %5 = VPUIP.Copy inputs(%4 : !SparseOutputActBufferType)
        outputs(%arg1 : !SparseOutputActBufferType)
        -> !SparseOutputActBufferType

    return %5 : !SparseOutputActBufferType

    //CHECK:        [[VAR0:%.+]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW}
    //CHECK-SAME:                   inputs(%arg0 : !VPUIP.SparseBuffer<data=memref<1x16x112x112xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX>, sparsity_map=memref<1x16x112x112xi1, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX>>)
    //CHECK-SAME:                   -> !VPUIP.SparseBuffer<data=memref<1x16x112x112xf16, @CMX>, sparsity_map=memref<1x16x112x112xi1, @CMX>>

    //CHECK:        [[VAR1:%.+]] = memref.alloc() : memref<1x16x112x112xf16, @DDR>
    //CHECK:        [[VAR2:%.+]] = memref.alloc() : memref<1x16x112x112xi1, @DDR>
    //CHECK:        [[VAR3:%.+]] = VPUIP.GroupSparseBuffer([[VAR1]], [[VAR2]]) -> !VPUIP.SparseBuffer<data=memref<1x16x112x112xf16, @DDR>, sparsity_map=memref<1x16x112x112xi1, @DDR>>

    //CHECK:        [[VAR4:%.+]] = VPUIP.Copy inputs([[VAR0]] : !VPUIP.SparseBuffer<data=memref<1x16x112x112xf16, @CMX>, sparsity_map=memref<1x16x112x112xi1, @CMX>>)
    //CHECK-SAME:                   outputs([[VAR3]] : !VPUIP.SparseBuffer<data=memref<1x16x112x112xf16, @DDR>, sparsity_map=memref<1x16x112x112xi1, @DDR>>)
    //CHECK-SAME:                   -> !VPUIP.SparseBuffer<data=memref<1x16x112x112xf16, @DDR>, sparsity_map=memref<1x16x112x112xi1, @DDR>>

    //CHECK:        [[VAR5:%.+]] = VPUIP.Copy inputs([[VAR4]] : !VPUIP.SparseBuffer<data=memref<1x16x112x112xf16, @DDR>, sparsity_map=memref<1x16x112x112xi1, @DDR>>)
    //CHECK-SAME:                   outputs(%arg1 : !VPUIP.SparseBuffer<data=memref<1x16x112x112xf16, @DDR>, sparsity_map=memref<1x16x112x112xi1, @DDR>>)
    //CHECK-SAME:                   -> !VPUIP.SparseBuffer<data=memref<1x16x112x112xf16, @DDR>, sparsity_map=memref<1x16x112x112xi1, @DDR>>

    //CHECK:        return [[VAR5]] : !VPUIP.SparseBuffer<data=memref<1x16x112x112xf16, @DDR>, sparsity_map=memref<1x16x112x112xi1, @DDR>>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!SparseInputActBufferType = type !VPUIP.SparseBuffer<
    data=memref<1x16x112x112xf16, #NHWC, @CMX>,
    sparsity_map=memref<1x16x112x112xi1, #NHWC, @CMX>
>

!SparsePermOutputBufferType = type !VPUIP.SparseBuffer<
    data=memref<1x16x112x112xf16, #NCHW, @DDR>,
    sparsity_map=memref<1x16x112x112xi1, #NCHW, @DDR>
>

!SparseOutputActBufferType = type !VPUIP.SparseBuffer<
    data=memref<1x16x392x32xf16, @DDR>,
    sparsity_map=memref<1x16x392x32xi1, @DDR>
>

!IZMajorDDRType = type memref<1x16x112x112xf16, #NHWC, @DDR>
!IZMajorSMDDRType = type memref<1x16x112x112xi1, #NHWC, @DDR>
!IZMajorSparseType = type !VPUIP.SparseBuffer<data=!IZMajorDDRType, sparsity_map=!IZMajorSMDDRType>

!FlatDDRType = type memref<1x16x12544xf16, @DDR>
!FlatSMDDRType = type memref<1x16x12544xi1, @DDR>
!FlatSparseType = type !VPUIP.SparseBuffer<data=!FlatDDRType, sparsity_map=!FlatSMDDRType>

func @MoveSeveralPureViewOpsBeforeCopySparse(
        %arg0: !SparseInputActBufferType,
        %arg1: !SparseOutputActBufferType)
        -> !SparseOutputActBufferType {
    %0 = memref.alloc() : !IZMajorDDRType
    %1 = memref.alloc() : !IZMajorSMDDRType
    %2 = VPUIP.GroupSparseBuffer(%0, %1) -> !IZMajorSparseType

    %3 = VPUIP.Copy inputs(%arg0 : !SparseInputActBufferType )
        outputs(%2 : !IZMajorSparseType )
        -> !IZMajorSparseType

    %4 = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW}
            inputs(%3 : !IZMajorSparseType ) -> !SparsePermOutputBufferType

    %5 = VPUIP.GenericReshape inputs(%4 : !SparsePermOutputBufferType) -> !FlatSparseType

    %6 = memref.alloc() : !FlatDDRType
    %7 = memref.alloc() : !FlatSMDDRType
    %8 = VPUIP.GroupSparseBuffer(%6, %7) -> !FlatSparseType

    %9 = VPUIP.Copy inputs(%5 : !FlatSparseType)
        outputs(%8 : !FlatSparseType)
        -> !FlatSparseType

    %10 = VPUIP.GenericReshape inputs(%9 : !FlatSparseType) -> !SparseOutputActBufferType

    %11 = VPUIP.Copy inputs(%10 : !SparseOutputActBufferType)
        outputs(%arg1 : !SparseOutputActBufferType)
        -> !SparseOutputActBufferType

    return %11 : !SparseOutputActBufferType

    //CHECK:        [[VAR0:%.+]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW}
    //CHECK-SAME:                   inputs(%arg0 : !VPUIP.SparseBuffer<data=memref<1x16x112x112xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX>, sparsity_map=memref<1x16x112x112xi1, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX>>)
    //CHECK-SAME:                   -> !VPUIP.SparseBuffer<data=memref<1x16x112x112xf16, @CMX>, sparsity_map=memref<1x16x112x112xi1, @CMX>>

    //CHECK:        [[VAR1:%.+]] = VPUIP.GenericReshape inputs([[VAR0]] : !VPUIP.SparseBuffer<data=memref<1x16x112x112xf16, @CMX>, sparsity_map=memref<1x16x112x112xi1, @CMX>>)
    //CHECK-SAME:                   -> !VPUIP.SparseBuffer<data=memref<1x16x12544xf16, @CMX>, sparsity_map=memref<1x16x12544xi1, @CMX>>

    //CHECK:        [[VAR2:%.+]] = VPUIP.GenericReshape inputs([[VAR1]] : !VPUIP.SparseBuffer<data=memref<1x16x12544xf16, @CMX>, sparsity_map=memref<1x16x12544xi1, @CMX>>)
    //CHECK-SAME:                   -> !VPUIP.SparseBuffer<data=memref<1x16x392x32xf16, @CMX>, sparsity_map=memref<1x16x392x32xi1, @CMX>>

    //CHECK:        [[VAR3:%.+]] = memref.alloc() : memref<1x16x392x32xf16, @DDR>
    //CHECK:        [[VAR4:%.+]] = memref.alloc() : memref<1x16x392x32xi1, @DDR>
    //CHECK:        [[VAR5:%.+]] = VPUIP.GroupSparseBuffer([[VAR3]], [[VAR4]]) -> !VPUIP.SparseBuffer<data=memref<1x16x392x32xf16, @DDR>, sparsity_map=memref<1x16x392x32xi1, @DDR>>

    //CHECK:        [[VAR6:%.+]] = VPUIP.Copy inputs([[VAR2]] : !VPUIP.SparseBuffer<data=memref<1x16x392x32xf16, @CMX>, sparsity_map=memref<1x16x392x32xi1, @CMX>>)
    //CHECK-SAME:                   outputs([[VAR5]] : !VPUIP.SparseBuffer<data=memref<1x16x392x32xf16, @DDR>, sparsity_map=memref<1x16x392x32xi1, @DDR>>)

    //CHECK:        [[VAR7:%.+]] = memref.alloc() : memref<1x16x392x32xf16, @DDR>
    //CHECK:        [[VAR8:%.+]] = memref.alloc() : memref<1x16x392x32xi1, @DDR>
    //CHECK:        [[VAR9:%.+]] = VPUIP.GroupSparseBuffer([[VAR7]], [[VAR8]]) -> !VPUIP.SparseBuffer<data=memref<1x16x392x32xf16, @DDR>, sparsity_map=memref<1x16x392x32xi1, @DDR>>

    //CHECK:        [[VAR10:%.+]] = VPUIP.Copy inputs([[VAR6]] : !VPUIP.SparseBuffer<data=memref<1x16x392x32xf16, @DDR>, sparsity_map=memref<1x16x392x32xi1, @DDR>>)
    //CHECK-SAME:                   outputs([[VAR9]] : !VPUIP.SparseBuffer<data=memref<1x16x392x32xf16, @DDR>, sparsity_map=memref<1x16x392x32xi1, @DDR>>)
    //CHECK:        [[VAR11:%.+]] = VPUIP.Copy inputs([[VAR10]] : !VPUIP.SparseBuffer<data=memref<1x16x392x32xf16, @DDR>, sparsity_map=memref<1x16x392x32xi1, @DDR>>)
    //CHECK-SAME:                   outputs(%arg1 : !VPUIP.SparseBuffer<data=memref<1x16x392x32xf16, @DDR>, sparsity_map=memref<1x16x392x32xi1, @DDR>>)
    //CHECK:        return [[VAR11]] : !VPUIP.SparseBuffer<data=memref<1x16x392x32xf16, @DDR>, sparsity_map=memref<1x16x392x32xi1, @DDR>>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!SparseInputActBufferType = type !VPUIP.SparseBuffer<
    data=memref<1x16x2x2xf16, @DDR>,
    sparsity_map=memref<1x16x2x2xi1, @DDR>
>

!SparseOutputSubviewBufferType = type !VPUIP.SparseBuffer<
    data=memref<1x8x2x2xf16, {order = #NCHW, strides = [64, 4, 2, 1]}, @DDR>,
    sparsity_map=memref<1x8x2x2xi1, {order = #NCHW, strides = [64, 4, 2, 1]}, @DDR>
>

!SparseOutputActBufferType = type !VPUIP.SparseBuffer<
    data=memref<1x8x2x2xf16, @DDR>,
    sparsity_map=memref<1x8x2x2xi1, @DDR>
>

func @MoveSubviewToTheFrontOfCopySparse(%arg0: !SparseInputActBufferType, %arg1: !SparseOutputActBufferType) -> !SparseOutputActBufferType {
    %0 = memref.alloc() : memref<1x16x2x2xf16, @DDR>
    %1 = memref.alloc() : memref<1x16x2x2xi1, @DDR>
    %2 = VPUIP.GroupSparseBuffer(%0, %1) -> !SparseInputActBufferType

    %3 = VPUIP.Copy inputs(%arg0: !SparseInputActBufferType) outputs(%2 : !SparseInputActBufferType) -> !SparseInputActBufferType
    %4 = VPUIP.SubView %3 [0, 0, 0, 0] [1, 8, 2, 2] : !SparseInputActBufferType to !SparseOutputSubviewBufferType
    %5 = VPUIP.Copy inputs(%4 : !SparseOutputSubviewBufferType) outputs(%arg1 : !SparseOutputActBufferType) -> !SparseOutputActBufferType

    return %5 : !SparseOutputActBufferType

    // CHECK:       [[VAR0:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 8, 2, 2] :
    // CHECK-SAME:                      !VPUIP.SparseBuffer<data=memref<1x16x2x2xf16, @DDR>, sparsity_map=memref<1x16x2x2xi1, @DDR>> to
    // CHECK-SAME:                      !VPUIP.SparseBuffer<data=memref<1x8x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [64, 4, 2, 1]}, @DDR>,
    // CHECK-SAME:                                          sparsity_map=memref<1x8x2x2xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [64, 4, 2, 1]}, @DDR>>

    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<1x8x2x2xf16, @DDR>
    // CHECK:       [[VAR2:%.*]] = memref.alloc() : memref<1x8x2x2xi1, @DDR>
    // CHECK:       [[VAR3:%.*]] = VPUIP.GroupSparseBuffer([[VAR1]], [[VAR2]]) -> !VPUIP.SparseBuffer<data=memref<1x8x2x2xf16, @DDR>, sparsity_map=memref<1x8x2x2xi1, @DDR>>

    // CHECK:       [[VAR4:%.*]] = VPUIP.Copy inputs([[VAR0]] : !VPUIP.SparseBuffer<data=memref<1x8x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [64, 4, 2, 1]}, @DDR>,
    // CHEKC-SAME:                                                                  sparsity_map=memref<1x8x2x2xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [64, 4, 2, 1]}, @DDR>>)
    // CHECK-SAME:                      outputs([[VAR3]] : !VPUIP.SparseBuffer<data=memref<1x8x2x2xf16, @DDR>, sparsity_map=memref<1x8x2x2xi1, @DDR>>)

    // CHECK:       [[VAR5:%.*]] = VPUIP.Copy inputs([[VAR4]] : !VPUIP.SparseBuffer<data=memref<1x8x2x2xf16, @DDR>, sparsity_map=memref<1x8x2x2xi1, @DDR>>)
    // CHECK-SAME:                      outputs(%arg1 : !VPUIP.SparseBuffer<data=memref<1x8x2x2xf16, @DDR>, sparsity_map=memref<1x8x2x2xi1, @DDR>>)

    // CHECK:       return [[VAR5]] : !VPUIP.SparseBuffer<data=memref<1x8x2x2xf16, @DDR>, sparsity_map=memref<1x8x2x2xi1, @DDR>>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = type !quant.uniform<u8:f16, 5.7832517137714463:123>

!IOBufferType = type memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>
!IOSMBuffer = type memref<1x64x48x88xi1, #NHWC, @CMX_NN>

!SparseInputActBufferType = type !VPUIP.SparseBuffer<
    data=!IOBufferType,
    sparsity_map=!IOSMBuffer
>

!SparseInputActDDRBufferType = type !VPUIP.SparseBuffer<
    data=memref<1x64x48x88x!qElemType, #NHWC, @DDR>,
    sparsity_map=memref<1x64x48x88xi1, #NHWC, @DDR>
>

!SparseSubviewOutputBufferType = type !VPUIP.SparseBuffer<
    data=memref<1x32x48x88x!qElemType, {order = #NHWC, strides = [270336, 1, 5632, 64]}, @DDR>,
    sparsity_map=memref<1x32x48x88xi1, {order = #NHWC, strides = [270336, 1, 5632, 64]}, @DDR>
>

!SparseOutputActDDRBufferType = type !VPUIP.SparseBuffer<
    data=memref<1x32x48x88x!qElemType, #NHWC, @DDR>,
    sparsity_map=memref<1x32x48x88xi1, #NHWC, @DDR>
>

!DistributedDataType = type !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
!DistributedSMType = type !VPUIP.DistributedBuffer<1x64x48x88xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>

!DistributedSparseType = type !VPUIP.SparseBuffer<
    data=!DistributedDataType,
    sparsity_map=!DistributedSMType
>


func @MoveSubviewToTheFrontOfTillingCopySparse(%in0 : !SparseInputActBufferType,
                                %in1 : !SparseInputActBufferType)
                                    -> !SparseOutputActDDRBufferType {
    %0 = memref.alloc() : !IOBufferType
    %1 = memref.alloc() : !IOSMBuffer
    %2 = VPUIP.GroupSparseBuffer(%0, %1) -> !SparseInputActBufferType

    %in_data_0, %in_sm_0 = VPUIP.UngroupSparseBuffer(%in0) {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>}
        -> !IOBufferType, !IOSMBuffer
    %in_data_1, %in_sm_1 = VPUIP.UngroupSparseBuffer(%in1) {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>}
        -> !IOBufferType, !IOSMBuffer
    %out_data, %out_sm = VPUIP.UngroupSparseBuffer(%2) {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>}
        -> !IOBufferType, !IOSMBuffer

    %3:2 = VPUIP.NCEClusterTiling inputs(%in_data_0 as %arg2: !IOBufferType,
                                       %in_sm_0 as %arg3: !IOSMBuffer,
                                       %in_data_1 as %arg4: !IOBufferType,
                                       %in_sm_1 as %arg5: !IOSMBuffer)
                                outputs(%out_data as %arg6: !IOBufferType,
                                       %out_sm as %arg7: !IOSMBuffer)
                                    -> (!DistributedDataType, !DistributedSMType) {
        %1232:2 = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i64, minimumHardwareExecutionCost = 1081 : i64, task_type = "ELTWISE"}
            input(%arg2 : !IOBufferType)
            input_sparsity_map(%arg3 : !IOSMBuffer)
            weights(%arg4 : !IOBufferType)
            weights_sparsity_map(%arg5 : !IOSMBuffer)
            parent_input(%arg2 : !IOBufferType)
            parent_input_sparsity_map(%arg3 : !IOSMBuffer)
            parent_output(%arg6 : !IOBufferType)
            parent_output_sparsity_map(%arg7 : !IOSMBuffer)
            outputs(%arg6 : !IOBufferType)
            output_sparsity_map(%arg7 : !IOSMBuffer)
                -> !IOBufferType, !IOSMBuffer variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [87, 47, 63], mpe_mode = "MATRIX", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0]}
        } PPE :  {
            PPETask "ADD" {clamp_high = 131 : i64, clamp_low = -124 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [26565], quant_post_shift = 0 : i64, quant_shift = [15]}
        }
    }
    %output_sparse = VPUIP.GroupSparseBuffer(%3#0, %3#1) -> !DistributedSparseType

    %4 = memref.alloc() : memref<1x64x48x88x!qElemType, #NHWC, @DDR>
    %5 = memref.alloc() : memref<1x64x48x88xi1, #NHWC, @DDR>
    %6 = VPUIP.GroupSparseBuffer(%4, %5) -> !SparseInputActDDRBufferType

    %7 = VPUIP.NCEClusterTiling inputs(%output_sparse as %arg2: !SparseInputActBufferType)
                                outputs(%6 as %arg3: !SparseInputActDDRBufferType)
                                    -> !SparseInputActDDRBufferType {
        %1232 = VPUIP.Copy inputs(%arg2 : !SparseInputActBufferType)
                           outputs(%arg3 : !SparseInputActDDRBufferType)
                               -> !SparseInputActDDRBufferType
    }
    %8 = VPUIP.SubView %7 [0, 0, 0, 0] [1, 32, 48, 88] : !SparseInputActDDRBufferType to !SparseSubviewOutputBufferType
    %9 = memref.alloc() : memref<1x32x48x88x!qElemType, #NHWC, @DDR>
    %10 = memref.alloc() : memref<1x32x48x88xi1, #NHWC, @DDR>
    %11 = VPUIP.GroupSparseBuffer(%9, %10) -> !SparseOutputActDDRBufferType
    %12 = VPUIP.Copy inputs(%8 : !SparseSubviewOutputBufferType) outputs(%11 : !SparseOutputActDDRBufferType) -> !SparseOutputActDDRBufferType
    return %12 : !SparseOutputActDDRBufferType

    // CHECK:       [[BUFF_0_DATA:%.*]] = memref.alloc() : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_0_SM:%.*]] = memref.alloc() : memref<1x64x48x88xi1, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_0:%.*]] = VPUIP.GroupSparseBuffer([[BUFF_0_DATA]], [[BUFF_0_SM]])
    // CHECK-SAME:                 -> !VPUIP.SparseBuffer<data=memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>, sparsity_map=memref<1x64x48x88xi1, #NHWC, @CMX_NN>>

    // CHECK:       [[DATA_0:%.*]], [[SM_0:%.*]] = VPUIP.UngroupSparseBuffer(%arg0)
    // CHECK:       [[DATA_1:%.*]], [[SM_1:%.*]] = VPUIP.UngroupSparseBuffer(%arg1)
    // CHECK:       [[DATA_2:%.*]], [[SM_2:%.*]] = VPUIP.UngroupSparseBuffer([[BUFF_0]])

    // CHECK:       [[ADD:%.*]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[DATA_0]] as %arg2: memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>,
    // CHECK-SAME:             [[SM_0]] as %arg3: memref<1x64x48x88xi1, #NHWC, @CMX_NN>
    // CHECK-SAME:             [[DATA_1]] as %arg4: memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>,
    // CHECK-SAME:             [[SM_1]] as %arg5: memref<1x64x48x88xi1, #NHWC, @CMX_NN>
    // CHECK-SAME:      outputs([[DATA_2]] as %arg6: memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>,
    // CHECK-SAME:              [[SM_2]] as %arg7: memref<1x64x48x88xi1, #NHWC, @CMX_NN>)
    // CHECK-SAME:              -> (!VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>,
    // CHECK-SAME:                  !VPUIP.DistributedBuffer<1x64x48x88xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)

    // CHECK:       [[ADD_R:%.*]] = VPUIP.GroupSparseBuffer([[ADD]]#0, [[ADD]]#1)
    // CHECK-SAME:                 -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>,
    // CHECK-SAME:                  sparsity_map=!VPUIP.DistributedBuffer<1x64x48x88xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>>

    // CHECK:       [[SUBVIEW:%.*]] = VPUIP.SubView [[ADD_R]] [0, 0, 0, 0] [1, 32, 48, 88] :
    // CHECK-SAME:                      !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>,
    // CHECK-SAME:                          sparsity_map=!VPUIP.DistributedBuffer<1x64x48x88xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>> to
    // CHECK-SAME:                      !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x32x48x88x!qElemType, {order = #NHWC, strides = [270336, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>,
    // CHECK-SAME:                          sparsity_map=!VPUIP.DistributedBuffer<1x32x48x88xi1, {order = #NHWC, strides = [270336, 1, 5632, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>>

    // CHECK:       [[BUFF_1_DATA:%.*]] = memref.alloc() : memref<1x32x48x88x!qElemType, #NHWC, @DDR>
    // CHECK:       [[BUFF_1_SM:%.*]] = memref.alloc() : memref<1x32x48x88xi1, #NHWC, @DDR>
    // CHECK:       [[BUFF_1:%.*]] = VPUIP.GroupSparseBuffer([[BUFF_1_DATA]], [[BUFF_1_SM]])
    // CHECK-SAME:                 -> !VPUIP.SparseBuffer<data=memref<1x32x48x88x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x32x48x88xi1, #NHWC, @DDR>>


    // CHECK:       [[Tilling_COPY:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[SUBVIEW]] as %arg2: !VPUIP.SparseBuffer<data=memref<1x32x48x88x!qElemType, {order = #NHWC, strides = [270336, 1, 5632, 64]}, @CMX_NN>, sparsity_map=memref<1x32x48x88xi1, {order = #NHWC, strides = [270336, 1, 5632, 64]}, @CMX_NN>>)
    // CHECK-SAME:      outputs([[BUFF_1]] as %arg3: !VPUIP.SparseBuffer<data=memref<1x32x48x88x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x32x48x88xi1, #NHWC, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x32x48x88x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x32x48x88xi1, #NHWC, @DDR>> {
    // CHECK:           [[COPY_2_INNER:%.*]] = VPUIP.Copy
    // CHECK-SAME:          inputs(%arg2 : !VPUIP.SparseBuffer<data=memref<1x32x48x88x!qElemType, {order = #NHWC, strides = [270336, 1, 5632, 64]}, @CMX_NN>, sparsity_map=memref<1x32x48x88xi1, {order = #NHWC, strides = [270336, 1, 5632, 64]}, @CMX_NN>>)
    // CHECK-SAME:          outputs(%arg3 : !VPUIP.SparseBuffer<data=memref<1x32x48x88x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x32x48x88xi1, #NHWC, @DDR>>)
    // CHECK-SAME:              -> !VPUIP.SparseBuffer<data=memref<1x32x48x88x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x32x48x88xi1, #NHWC, @DDR>>
    // CHECK:       }

    // CHECK:       [[BUFF_2_DATA:%.*]] = memref.alloc() : memref<1x32x48x88x!qElemType, #NHWC, @DDR>
    // CHECK:       [[BUFF_2_SM:%.*]] = memref.alloc() : memref<1x32x48x88xi1, #NHWC, @DDR>
    // CHECK:       [[BUFF_2:%.*]] = VPUIP.GroupSparseBuffer([[BUFF_2_DATA]], [[BUFF_2_SM]])
    // CHECK-SAME:                 -> !VPUIP.SparseBuffer<data=memref<1x32x48x88x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x32x48x88xi1, #NHWC, @DDR>>

    // CHECK:       [[COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[Tilling_COPY]] : !VPUIP.SparseBuffer<data=memref<1x32x48x88x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x32x48x88xi1, #NHWC, @DDR>>)
    // CHECK-SAME:      outputs([[BUFF_2]] : !VPUIP.SparseBuffer<data=memref<1x32x48x88x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x32x48x88xi1, #NHWC, @DDR>>)
    // CHECK:       return [[COPY]] : !VPUIP.SparseBuffer<data=memref<1x32x48x88x!qElemType, #NHWC, @DDR>, sparsity_map=memref<1x32x48x88xi1, #NHWC, @DDR>>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = type !quant.uniform<u8:f16, 0.0036305147058823531>
!qElemType1 = type !quant.uniform<u8:f16, 0.0042424242424242424>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x16x32x32x!qElemType0, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!OutputStub_CMX = type memref<1x16x32x32x!qElemType0 , #NHWC, @CMX_NN>
!Output_DDR = type memref<1x16x32x32x!qElemType0, #NHWC, @DDR>

func @MoveQuantizeCastBeforeTilingCopy(%arg0: !InputDistributed) -> memref<1x16x32x32x!qElemType1, #NHWC, @DDR> {
    %out = memref.alloc() : !Output_DDR

    %0 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: !OutputStub_CMX)
           outputs(%out as %arg2: !Output_DDR) -> !Output_DDR {
        VPUIP.Copy inputs(%arg1: !OutputStub_CMX) outputs(%arg2: !Output_DDR) -> !Output_DDR
    }
    %1 = VPUIP.QuantizeCast inputs(%0 : memref<1x16x32x32x!qElemType0, #NHWC, @DDR>) -> memref<1x16x32x32x!qElemType1, #NHWC, @DDR>
    return %1 : memref<1x16x32x32x!qElemType1, #NHWC, @DDR>

    // CHECK:       [[QUANTCAST:%.*]] = VPUIP.QuantizeCast inputs(%arg0 : !VPUIP.DistributedBuffer<1x16x32x32x!qElemType0, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x16x32x32x!qElemType1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:       [[BUF_0:%.*]] = memref.alloc() : memref<1x16x32x32x!qElemType1, #NHWC, @DDR>
    // CHECK:       [[COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[QUANTCAST]] as %arg1: memref<1x16x32x32x!qElemType1, #NHWC, @CMX_NN>) outputs([[BUF_0]] as %arg2: memref<1x16x32x32x!qElemType1, #NHWC, @DDR>) -> memref<1x16x32x32x!qElemType1, #NHWC, @DDR> {
    // CHECK:                    VPUIP.Copy inputs(%arg1 : memref<1x16x32x32x!qElemType1, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x16x32x32x!qElemType1, #NHWC, @DDR>) -> memref<1x16x32x32x!qElemType1, #NHWC, @DDR>
    // CHECK:       }
    // CHECK:       return [[COPY]] : memref<1x16x32x32x!qElemType1, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x16x32x32xf16,#NCHW, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!OutputStub_CMX = type memref<1x16x32x32xf16, @CMX_NN>
!Output_DDR = type memref<1x16x32x32xf16, @DDR>

func @DoNotMoveGenericReshapeBeforeTilingCopy(
        %arg0: !InputDistributed)
        -> memref<1x16x1024xf16, @DDR> {
    %out = memref.alloc() : !Output_DDR

    %0 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: !OutputStub_CMX)
           outputs(%out as %arg2: !Output_DDR) -> !Output_DDR {
        VPUIP.Copy inputs(%arg1: !OutputStub_CMX) outputs(%arg2: !Output_DDR) -> !Output_DDR
    }
    %1 = VPUIP.GenericReshape inputs(%0 : memref<1x16x32x32xf16, @DDR>) -> memref<1x16x1024xf16, @DDR>

    return %1 : memref<1x16x1024xf16, @DDR>

    // CHECK:       [[BUFF_0:%.*]] = memref.alloc() : memref<1x16x32x32xf16, @DDR>
    // CHECK:       [[COPY:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x16x32x32xf16, @CMX_NN>) outputs([[BUFF_0]] as %arg2: memref<1x16x32x32xf16, @DDR>) -> memref<1x16x32x32xf16, @DDR> {
    // CHECK:                         VPUIP.Copy inputs(%arg1 : memref<1x16x32x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x16x32x32xf16, @DDR>) -> memref<1x16x32x32xf16, @DDR>
    // CHECK:       }
    // CHECK:       [[RESHAPE:%.*]] = VPUIP.GenericReshape inputs([[COPY]] : memref<1x16x32x32xf16, @DDR>) -> memref<1x16x1024xf16, @DDR>
    // CHECK:       return [[RESHAPE]] : memref<1x16x1024xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!OutputStub_CMX = type memref<1x16x32x32xf16, #NHWC, @CMX_NN>
!Output_DDR = type memref<1x16x32x32xf16, #NHWC, @DDR>

func @MovePermuteCastBeforeTilingCopy(
        %arg0: !InputDistributed)
        -> memref<1x16x32x32xf16, @DDR> {

    %out = memref.alloc() : !Output_DDR

    %0 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: !OutputStub_CMX)
           outputs(%out as %arg2: !Output_DDR) -> !Output_DDR {
        VPUIP.Copy inputs(%arg1: !OutputStub_CMX) outputs(%arg2: !Output_DDR) -> !Output_DDR
    }
    %1 = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>}
            inputs(%0 : memref<1x16x32x32xf16, #NHWC, @DDR>) -> memref<1x16x32x32xf16, @DDR>

    return %1 : memref<1x16x32x32xf16, @DDR>

    // CHECK:       [[PERMUTE:%.*]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NWCH} inputs(%arg0 : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_0:%.*]] = memref.alloc() : memref<1x16x32x32xf16, @DDR>
    // CHECK:       [[COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[PERMUTE]] as %arg1: memref<1x16x32x32xf16, @CMX_NN>) outputs([[BUFF_0]] as %arg2: memref<1x16x32x32xf16, @DDR>) -> memref<1x16x32x32xf16, @DDR> {
    // CHECK:                          VPUIP.Copy inputs(%arg1 : memref<1x16x32x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x16x32x32xf16, @DDR>) -> memref<1x16x32x32xf16, @DDR>
    // CHECK:       }
    // CHECK:       return [[COPY]] : memref<1x16x32x32xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

func @DoNotMovePermuteCastBeforeTilingCopySegmented(%arg0: !InputDistributed) -> memref<1x16x32x32xf16, @DDR> {
    %out = memref.alloc() : memref<1x16x32x32xf16, #NHWC, @DDR>

    %0 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x16x32x32xf16, #NHWC, @CMX_NN>) outputs(%out as %arg2: memref<1x16x32x32xf16, #NHWC, @DDR>) -> memref<1x16x32x32xf16, #NHWC, @DDR> {
        VPUIP.Copy inputs(%arg1: memref<1x16x32x32xf16, #NHWC, @CMX_NN>) outputs(%arg2: memref<1x16x32x32xf16, #NHWC, @DDR>) -> memref<1x16x32x32xf16, #NHWC, @DDR>
    }
    %1 = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} inputs(%0 : memref<1x16x32x32xf16, #NHWC, @DDR>) -> memref<1x16x32x32xf16, @DDR>

    return %1 : memref<1x16x32x32xf16, @DDR>

    // CHECK:    [[OUT_BUFF:%.*]] = memref.alloc() : memref<1x16x32x32xf16, #NHWC, @DDR>
    // CHECK:    [[COPY:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x16x32x32xf16, #NHWC, @CMX_NN>) outputs([[OUT_BUFF]] as %arg2: memref<1x16x32x32xf16, #NHWC, @DDR>) -> memref<1x16x32x32xf16, #NHWC, @DDR> {
    // CHECK:        VPUIP.Copy inputs(%arg1 : memref<1x16x32x32xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x16x32x32xf16, #NHWC, @DDR>) -> memref<1x16x32x32xf16, #NHWC, @DDR>
    // CHECK:    }
    // CHECK:    [[PERMUTE:%.*]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NWCH} inputs([[COPY]] : memref<1x16x32x32xf16, #NHWC, @DDR>) -> memref<1x16x32x32xf16, @DDR>
    // CHECK:    return [[PERMUTE]] : memref<1x16x32x32xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

func @MoveShapeCastBeforeTilingCopySegmented(%arg0: !InputDistributed) -> memref<1x4x64x64xf16, #NHWC, @DDR> {
    %out = memref.alloc() : memref<1x16x32x32xf16, #NHWC, @DDR>

    %0 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x16x32x32xf16, #NHWC, @CMX_NN>) outputs(%out as %arg2: memref<1x16x32x32xf16, #NHWC, @DDR>) -> memref<1x16x32x32xf16, #NHWC, @DDR> {
        VPUIP.Copy inputs(%arg1: memref<1x16x32x32xf16, #NHWC, @CMX_NN>) outputs(%arg2: memref<1x16x32x32xf16, #NHWC, @DDR>) -> memref<1x16x32x32xf16, #NHWC, @DDR>
    }
    %1 = VPUIP.ShapeCast {shape = [1, 4, 64, 64]} inputs(%0 : memref<1x16x32x32xf16, #NHWC, @DDR>) -> memref<1x4x64x64xf16, #NHWC, @DDR>

    return %1 : memref<1x4x64x64xf16, #NHWC, @DDR>

    //CHECK:    [[SHAPECAST:%.*]] = VPUIP.ShapeCast {shape = [1, 4, 64, 64]} inputs(%arg0 : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x4x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUTBUFF:%.*]] = memref.alloc() : memref<1x4x64x64xf16, #NHWC, @DDR>
    //CHECK:    [[COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[SHAPECAST]] as %arg1: memref<1x4x64x64xf16, #NHWC, @CMX_NN>) outputs([[OUTBUFF]] as %arg2: memref<1x4x64x64xf16, #NHWC, @DDR>) -> memref<1x4x64x64xf16, #NHWC, @DDR> {
    //CHECK:        VPUIP.Copy inputs(%arg1 : memref<1x4x64x64xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x4x64x64xf16, #NHWC, @DDR>) -> memref<1x4x64x64xf16, #NHWC, @DDR>
    //CHECK:    }
    //CHECK:    return [[COPY]] : memref<1x4x64x64xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

func @MoveGenericReshapeBeforeTilingCopySegmented(%arg0: !InputDistributed) -> memref<1x4x64x64xf16, #NHWC, @DDR> {
    %out = memref.alloc() : memref<1x16x32x32xf16, #NHWC, @DDR>

    %0 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x16x32x32xf16, #NHWC, @CMX_NN>) outputs(%out as %arg2: memref<1x16x32x32xf16, #NHWC, @DDR>) -> memref<1x16x32x32xf16, #NHWC, @DDR> {
        VPUIP.Copy inputs(%arg1: memref<1x16x32x32xf16, #NHWC, @CMX_NN>) outputs(%arg2: memref<1x16x32x32xf16, #NHWC, @DDR>) -> memref<1x16x32x32xf16, #NHWC, @DDR>
    }
    %1 = VPUIP.GenericReshape inputs(%0 : memref<1x16x32x32xf16, #NHWC, @DDR>) -> memref<1x4x64x64xf16, #NHWC, @DDR>

    return %1 : memref<1x4x64x64xf16, #NHWC, @DDR>

    //CHECK:    [[RESHAPE:%.*]] = VPUIP.GenericReshape inputs(%arg0 : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x4x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUTBUFF:%.*]] = memref.alloc() : memref<1x4x64x64xf16, #NHWC, @DDR>
    //CHECK:    [[COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[RESHAPE]] as %arg1: memref<1x4x64x64xf16, #NHWC, @CMX_NN>) outputs([[OUTBUFF]] as %arg2: memref<1x4x64x64xf16, #NHWC, @DDR>) -> memref<1x4x64x64xf16, #NHWC, @DDR> {
    //CHECK:        VPUIP.Copy inputs(%arg1 : memref<1x4x64x64xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x4x64x64xf16, #NHWC, @DDR>) -> memref<1x4x64x64xf16, #NHWC, @DDR>
    //CHECK:    }
    //CHECK:    return [[COPY]] : memref<1x4x64x64xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

func @DoNotMoveGenericReshapeBeforeTilingCopySegmented(%arg0: !InputDistributed) -> memref<1x16x1024xf16, @DDR> {
    %out = memref.alloc() : memref<1x16x32x32xf16, #NHWC, @DDR>

    %0 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x16x32x32xf16, #NHWC, @CMX_NN>) outputs(%out as %arg2: memref<1x16x32x32xf16, #NHWC, @DDR>) -> memref<1x16x32x32xf16, #NHWC, @DDR> {
        VPUIP.Copy inputs(%arg1: memref<1x16x32x32xf16, #NHWC, @CMX_NN>) outputs(%arg2: memref<1x16x32x32xf16, #NHWC, @DDR>) -> memref<1x16x32x32xf16, #NHWC, @DDR>
    }
    %1 = VPUIP.GenericReshape inputs(%0 : memref<1x16x32x32xf16, #NHWC, @DDR>) -> memref<1x16x1024xf16, @DDR>

    return %1 : memref<1x16x1024xf16, @DDR>

    // CHECK:    [[OUT_BUFF:%.*]] = memref.alloc() : memref<1x16x32x32xf16, #NHWC, @DDR>
    // CHECK:    [[COPY:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x16x32x32xf16, #NHWC, @CMX_NN>) outputs([[OUT_BUFF]] as %arg2: memref<1x16x32x32xf16, #NHWC, @DDR>) -> memref<1x16x32x32xf16, #NHWC, @DDR> {
    // CHECK:        VPUIP.Copy inputs(%arg1 : memref<1x16x32x32xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x16x32x32xf16, #NHWC, @DDR>) -> memref<1x16x32x32xf16, #NHWC, @DDR>
    // CHECK:    }
    // CHECK:    [[RESHAPE:%.*]] = VPUIP.GenericReshape inputs([[COPY]] : memref<1x16x32x32xf16, #NHWC, @DDR>) -> memref<1x16x1024xf16, @DDR>
    // CHECK:    return [[RESHAPE]] : memref<1x16x1024xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = type !quant.uniform<u8:f16, 0.0036305147058823531>
!qElemType1 = type !quant.uniform<u8:f16, 0.0042424242424242424>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x16x32x32x!qElemType0, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

func @MoveQuantizeCastBeforeTilingCopySegmented(%arg0: !InputDistributed) -> memref<1x16x32x32x!qElemType1, #NHWC, @DDR> {
    %out = memref.alloc() : memref<1x16x32x32x!qElemType0, #NHWC, @DDR>

    %0 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x16x32x32x!qElemType0, #NHWC, @CMX_NN>) outputs(%out as %arg2: memref<1x16x32x32x!qElemType0, #NHWC, @DDR>) -> memref<1x16x32x32x!qElemType0, #NHWC, @DDR> {
        VPUIP.Copy inputs(%arg1: memref<1x16x32x32x!qElemType0, #NHWC, @CMX_NN>) outputs(%arg2: memref<1x16x32x32x!qElemType0, #NHWC, @DDR>) -> memref<1x16x32x32x!qElemType0, #NHWC, @DDR>
    }
    %1 = VPUIP.QuantizeCast inputs(%0 : memref<1x16x32x32x!qElemType0, #NHWC, @DDR>) -> memref<1x16x32x32x!qElemType1, #NHWC, @DDR>

    return %1 : memref<1x16x32x32x!qElemType1, #NHWC, @DDR>

    //CHECK:    [[QUANTIZE:%.*]] = VPUIP.QuantizeCast inputs(%arg0 : !VPUIP.DistributedBuffer<1x16x32x32x!qElemType0, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x16x32x32x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUTBUFF:%.*]] = memref.alloc() : memref<1x16x32x32x!qElemType1, #NHWC, @DDR>
    //CHECK:    [[COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[QUANTIZE]] as %arg1: memref<1x16x32x32x!qElemType1, #NHWC, @CMX_NN>) outputs([[OUTBUFF]] as %arg2: memref<1x16x32x32x!qElemType1, #NHWC, @DDR>) -> memref<1x16x32x32x!qElemType1, #NHWC, @DDR> {
    //CHECK:        VPUIP.Copy inputs(%arg1 : memref<1x16x32x32x!qElemType1, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x16x32x32x!qElemType1, #NHWC, @DDR>) -> memref<1x16x32x32x!qElemType1, #NHWC, @DDR>
    //CHECK:    }
    //CHECK:    return [[COPY]] : memref<1x16x32x32x!qElemType1, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = type !quant.uniform<u8:f16, 0.0036305147058823531>
!qElemType1 = type !quant.uniform<u8:f16, 0.0042424242424242424>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x16x32x32x!qElemType0, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [1, 1],
    pads = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    strides = [1, 1],
    num_clusters = 2
}>

// CHECK-LABEL: @MoveQuantizeCastBeforeTilingCopyOverlapped
func @MoveQuantizeCastBeforeTilingCopyOverlapped(%arg0: !InputDistributed) -> memref<1x16x32x32x!qElemType1, #NHWC, @DDR> {
    %out = memref.alloc() : memref<1x16x32x32x!qElemType0, #NHWC, @DDR>

    %0 = VPUIP.NCEClusterTiling
        inputs(%arg0 as %arg1: memref<1x16x32x32x!qElemType0, #NHWC, @CMX_NN>)
        outputs(%out as %arg2: memref<1x16x32x32x!qElemType0, #NHWC, @DDR>)
        -> memref<1x16x32x32x!qElemType0, #NHWC, @DDR> {
        VPUIP.Copy
            inputs(%arg1: memref<1x16x32x32x!qElemType0, #NHWC, @CMX_NN>)
            outputs(%arg2: memref<1x16x32x32x!qElemType0, #NHWC, @DDR>)
            -> memref<1x16x32x32x!qElemType0, #NHWC, @DDR>
    }

    %1 = VPUIP.QuantizeCast
        inputs(%0 : memref<1x16x32x32x!qElemType0, #NHWC, @DDR>)
        -> memref<1x16x32x32x!qElemType1, #NHWC, @DDR>

    return %1 : memref<1x16x32x32x!qElemType1, #NHWC, @DDR>

    // CHECK:    [[QUANTIZE:%.*]] = VPUIP.QuantizeCast
    // CHECK-SAME:  inputs(%arg0 : !VPUIP.DistributedBuffer<1x16x32x32x!qElemType0, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:      kernel = [1, 1], pads = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, strides = [1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x16x32x32x!qElemType1, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:      kernel = [1, 1], pads = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, strides = [1, 1], num_clusters = 2 : i64}>

    // CHECK:    [[OUTBUFF:%.*]] = memref.alloc() : memref<1x16x32x32x!qElemType1, #NHWC, @DDR>

    // CHECK:    [[COPY:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:  inputs([[QUANTIZE]] as %arg1: memref<1x16x32x32x!qElemType1, #NHWC, @CMX_NN>)
    // CHECK-SAME:  outputs([[OUTBUFF]] as %arg2: memref<1x16x32x32x!qElemType1, #NHWC, @DDR>)
    // CHECK-SAME:  -> memref<1x16x32x32x!qElemType1, #NHWC, @DDR> {

    // CHECK:        VPUIP.Copy
    // CHECK-SAME:      inputs(%arg1 : memref<1x16x32x32x!qElemType1, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs(%arg2 : memref<1x16x32x32x!qElemType1, #NHWC, @DDR>)
    // CHECK-SAME:      -> memref<1x16x32x32x!qElemType1, #NHWC, @DDR>

    // CHECK:    }
    // CHECK:    return [[COPY]] : memref<1x16x32x32x!qElemType1, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = type !quant.uniform<u8:f16:1, {1.25, 1.5, 1.75, 2.0,
                                             1.25, 1.5, 1.75, 2.0,
                                             1.25, 1.5, 1.75, 2.0,
                                             1.25, 1.5, 1.75, 2.0}>
!qElemType1 = type !quant.uniform<u8:f16:1, {0.25, 0.5, 0.75, 1.0,
                                             0.25, 0.5, 0.75, 1.0,
                                             0.25, 0.5, 0.75, 1.0,
                                             0.25, 0.5, 0.75, 1.0}>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x16x32x32x!qElemType0, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [1, 1],
    pads = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    strides = [1, 1],
    num_clusters = 2
}>

// CHECK-LABEL: @SkipPerChannelOverlappedQuantizeCast
func @SkipPerChannelOverlappedQuantizeCast(%arg0: !InputDistributed) -> memref<1x16x32x32x!qElemType1, #NHWC, @DDR> {
    %out = memref.alloc() : memref<1x16x32x32x!qElemType0, #NHWC, @DDR>

    %0 = VPUIP.NCEClusterTiling
        inputs(%arg0 as %arg1: memref<1x16x32x32x!qElemType0, #NHWC, @CMX_NN>)
        outputs(%out as %arg2: memref<1x16x32x32x!qElemType0, #NHWC, @DDR>)
        -> memref<1x16x32x32x!qElemType0, #NHWC, @DDR> {
        VPUIP.Copy
            inputs(%arg1: memref<1x16x32x32x!qElemType0, #NHWC, @CMX_NN>)
            outputs(%arg2: memref<1x16x32x32x!qElemType0, #NHWC, @DDR>)
            -> memref<1x16x32x32x!qElemType0, #NHWC, @DDR>
    }

    %1 = VPUIP.QuantizeCast
        inputs(%0 : memref<1x16x32x32x!qElemType0, #NHWC, @DDR>)
        -> memref<1x16x32x32x!qElemType1, #NHWC, @DDR>

    return %1 : memref<1x16x32x32x!qElemType1, #NHWC, @DDR>

    // CHECK:   [[ALLOCATE:%.*]] = memref.alloc() : memref<1x16x32x32x!qElemType0, #NHWC, @DDR>

    // CHECK:   [[TILING:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:  inputs(%arg0 as %arg1: memref<1x16x32x32x!qElemType0, #NHWC, @CMX_NN>)
    // CHECK-SAME:  outputs([[ALLOCATE]] as %arg2: memref<1x16x32x32x!qElemType0, #NHWC, @DDR>)
    // CHECK-SAME:  -> memref<1x16x32x32x!qElemType0, #NHWC, @DDR> {
    // CHECK:   VPUIP.Copy
    // CHECK-SAME:  inputs(%arg1 : memref<1x16x32x32x!qElemType0, #NHWC, @CMX_NN>)
    // CHECK-SAME:  outputs(%arg2 : memref<1x16x32x32x!qElemType0, #NHWC, @DDR>)
    // CHECK-SAME:  -> memref<1x16x32x32x!qElemType0, #NHWC, @DDR>
    // CHECK:   }

    // CHECK:   [[QUANTIZE_CAST:%.*]] = VPUIP.QuantizeCast
    // CHECK-SAME:  inputs([[TILING]] : memref<1x16x32x32x!qElemType0, #NHWC, @DDR>)
    // CHECK-SAME:  -> memref<1x16x32x32x!qElemType1, #NHWC, @DDR>

    // CHECK:   return [[QUANTIZE_CAST]] : memref<1x16x32x32x!qElemType1, #NHWC, @DDR>
}
