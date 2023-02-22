//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-constants %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!Input_DDR = type memref<1x16x16x16xf16, #NHWC, @DDR>
!Output_DDR = type memref<1x16x16x16xf16, #NHWC, @DDR>
!Weights_DDR = type memref<16x1x1x4xf16, #NHWC, @DDR>
!WeightsTable_DDR = type memref<16x1x1x4xsi32, @DDR>
!ActivationWindow_DDR = type memref<1x1x1x16xui8, @DDR>

!InputStub_CMX = type memref<1x16x16x16xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = type memref<1x16x16x16xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = type memref<16x1x1x4xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = type memref<16x1x1x4xsi32, @CMX_NN>
!ActivationWindowStub_CMX = type memref<1x1x1x16xui8, @CMX_NN>

// CHECK-LABEL: @FuseConstantsConv
func @FuseConstantsConv(%in : !Input_DDR) -> !OutputStub_CMX {

    %buf0 = memref.alloc() : !InputStub_CMX
    %buf1 = memref.alloc() : !OutputStub_CMX
    %buf2 = memref.alloc() : !WeightsTableStub_CMX
    %buf3 = memref.alloc() : !WeightsStub_CMX

    %weight_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<16x1x1x4xsi32>
    %weights = const.Declare !Weights_DDR = dense<1.0> : tensor<16x1x1x4xf16>, [#const.Reorder<#NHWC>]

    %0 = VPUIP.Copy inputs(%weight_table : !WeightsTable_DDR) outputs(%buf2 : !WeightsTableStub_CMX) -> !WeightsTableStub_CMX
    %1 = VPUIP.Copy inputs(%in : !Input_DDR) outputs(%buf0 : !InputStub_CMX) -> !InputStub_CMX
    %2 = VPUIP.Copy inputs(%weights : !Weights_DDR) outputs(%buf3 : !WeightsStub_CMX) -> !WeightsStub_CMX
    %3 = VPUIP.NCEClusterTask 
        {
            activation_window_channel_length = 27 : i64, 
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, 
            kernel_size = [1, 1], 
            kernel_strides = [1, 1], 
            task_type = "CONV"
        } 
        input(%1 : !InputStub_CMX) 
        weights(%2 : !WeightsStub_CMX) 
        weight_table(%0 : !WeightsTableStub_CMX) 
        parent_input(%1 : !InputStub_CMX) 
        parent_output(%buf1 : !OutputStub_CMX) 
        outputs(%buf1 : !OutputStub_CMX) -> !OutputStub_CMX
        variants :  
        {
            DPUTask 
                {
                    outEnd = [55, 10, 15], mpe_mode = "VECTOR_FP16", 
                    pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, 
                    outStart = [0, 0, 0]
                }
        }   
        PPE :  
        {
        }
    return %3 : !OutputStub_CMX

	// CHECK:       [[FUSED_CONSTANT:%.+]] = const.Declare memref<1x1x1x384xui8>
    // CHECK-NOT:   [[WEIGHT_TABLE:%.+]] = const.Declare !WeightsTable_DDR
    // CHECK-NOT:   [[ACTIVATION_WINDOW:%.+]] = const.Declare !ActivationWindow_DDR
    // CHECK-NOT:   [[WEIGHTS:%.+]] = const.Declare !Weights_DDR

    // CHECK:       [[OUT_BUF1:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, @CMX_NN>
    // CHECK:       [[OUT_BUF2:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, @CMX_NN>
    // CHECK:       [[VAR_INPUT:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs(%0 : memref<1x16x16x16xf16, #NHWC, @CMX_NN>) -> memref<1x16x16x16xf16, #NHWC, @CMX_NN>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x1x1x384xui8, [@CMX_NN, 0]>
    // CHECK:       [[VAR1:%.*]] = VPUIP.Copy inputs([[FUSED_CONSTANT]] : memref<1x1x1x384xui8>)
    // CHECK-SAME:      outputs([[VAR0]] : memref<1x1x1x384xui8, [@CMX_NN, 0]>) -> memref<1x1x1x384xui8, [@CMX_NN, 0]>

    // CHECK:       [[VAR2:%.*]] = VPUIP.SubView [[VAR1]] [0, 0, 0, 0] [1, 1, 1, 256] : 
    // CHECK-SAME:		memref<1x1x1x384xui8, [@CMX_NN, 0]> to memref<1x1x1x256xui8, 
    // CHECK-SAME:		{order = #NCHW, strides = [384, 384, 384, 1]}, [@CMX_NN, 0]>

    // CHECK:       [[VAR3:%.*]] = VPUIP.ViewOp [[VAR2]] : memref<1x1x1x256xui8, 
    // CHECK_SAME:		{order = #NCHW, strides = [384, 384, 384, 1]}, [@CMX_NN, 0]> to memref<16x1x1x4xsi32, @CMX_NN>

    // CHECK:       [[VAR4:%.*]] = VPUIP.SubView [[VAR1]] [0, 0, 0, 256] [1, 1, 1, 128] : 
    // CHECK-SAME:		memref<1x1x1x384xui8, [@CMX_NN, 0]> to memref<1x1x1x128xui8, 
    // CHECK-SAME:		{order = #NCHW, strides = [384, 384, 384, 1]}, [@CMX_NN, 0]>

    // CHECK:       [[VAR5:%.*]] = VPUIP.ViewOp [[VAR4]] : memref<1x1x1x128xui8,  
    // CHECK_SAME:		{order = #NCHW, strides = [384, 384, 384, 1]}, [@CMX_NN, 0]> to memref<16x1x1x4xf16, #NHWC, @CMX_NN>

    // CHECK:       [[VAR6:%.*]] = VPUIP.NCEClusterTask
    // CHECK-SAME:          activation_window_channel_length = 27 : i64,
    // CHECK-SAME:          constantsFused = true,
    // CHECK-SAME:          kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:          kernel_size = [1, 1],
    // CHECK-SAME:          kernel_strides = [1, 1],
    // CHECK-SAME:          task_type = "CONV"
    // CHECK-SAME:      input([[VAR_INPUT]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weights([[VAR5]] : memref<16x1x1x4xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weight_table([[VAR3]] : memref<16x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      parent_input([[VAR_INPUT]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      parent_output([[OUT_BUF2]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[OUT_BUF2]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK:           DPUTask
    // CHECK-SAME:          mpe_mode = "VECTOR_FP16",
    // CHECK-SAME:          outEnd = [55, 10, 15],
    // CHECK-SAME:          outStart = [0, 0, 0],
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK:       return [[VAR6]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!Input_DDR = type memref<1x16x16x16xf16, #NHWC, @DDR>
!Output_DDR = type memref<1x16x16x16xf16, #NHWC, @DDR>
!Weights_DDR = type memref<16x1x1x4xf16, #NHWC, @DDR>
!WeightsTable_DDR = type memref<16x1x1x4xsi32, @DDR>
!ActivationWindow_DDR = type memref<1x1x1x16xui8, @DDR>

!InputStub_CMX = type memref<1x16x16x16xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = type memref<1x16x16x16xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = type memref<16x1x1x4xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = type memref<16x1x1x4xsi32, @CMX_NN>
!ActivationWindowStub_CMX = type memref<1x1x1x16xui8, @CMX_NN>

// CHECK-LABEL: @FuseConstantsMaxPool
func @FuseConstantsMaxPool(%in : !Input_DDR) -> !OutputStub_CMX {

    %buf0 = memref.alloc() : !InputStub_CMX
    %buf1 = memref.alloc() : !OutputStub_CMX
    %buf2 = memref.alloc() : !WeightsTableStub_CMX                   
    %buf3 = memref.alloc() : !ActivationWindowStub_CMX

    %weight_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<16x1x1x4xsi32>
    %act_wind = const.Declare !ActivationWindow_DDR = dense<1> : tensor<1x1x1x16xui8>

    %0 = VPUIP.Copy inputs(%weight_table : !WeightsTable_DDR) outputs(%buf2 : !WeightsTableStub_CMX) -> !WeightsTableStub_CMX
    %1 = VPUIP.Copy inputs(%act_wind : !ActivationWindow_DDR) outputs(%buf3 : !ActivationWindowStub_CMX) -> !ActivationWindowStub_CMX
    %2 = VPUIP.Copy inputs(%in : !Input_DDR) outputs(%buf0 : !InputStub_CMX) -> !InputStub_CMX    
    %3 = VPUIP.NCEClusterTask 
        {
            activation_window_channel_length = 27 : i64, 
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, 
            kernel_size = [1, 1], 
            kernel_strides = [1, 1], 
            task_type = "MAXPOOL"
        } 
        input(%2 : !InputStub_CMX) 
        weight_table(%0 : !WeightsTableStub_CMX) 
        activation_window(%1 : !ActivationWindowStub_CMX) 
        parent_input(%2 : !InputStub_CMX) 
        parent_output(%buf1 : !OutputStub_CMX) 
        outputs(%buf1 : !OutputStub_CMX) -> !OutputStub_CMX
        variants :  
        {
            DPUTask 
                {
                    outEnd = [55, 10, 15], mpe_mode = "VECTOR_FP16", 
                    pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, 
                    outStart = [0, 0, 0]
                }
        }   
        PPE :  
        {
        }
    return %3 : !OutputStub_CMX

    // CHECK:       [[FUSED_CONSTANT:%.+]] = const.Declare memref<1x1x1x272xui8>
    // CHECK-NOT:   [[WEIGHT_TABLE:%.+]] = const.Declare !WeightsTable_DDR
    // CHECK-NOT:   [[ACTIVATION_WINDOW:%.+]] = const.Declare !ActivationWindow_DDR

    // CHECK:       [[OUT_BUF1:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, @CMX_NN>
    // CHECK:       [[OUT_BUF2:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, @CMX_NN>
    // CHECK:       [[VAR_INPUT:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%0 : memref<1x16x16x16xf16, #NHWC, @CMX_NN>) -> memref<1x16x16x16xf16, #NHWC, @CMX_NN>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x1x1x272xui8, [@CMX_NN, 0]>
    // CHECK:       [[VAR1:%.*]] = VPUIP.Copy inputs([[FUSED_CONSTANT]] : memref<1x1x1x272xui8>)
    // CHECK-SAME:      outputs([[VAR0]] : memref<1x1x1x272xui8, [@CMX_NN, 0]>) -> memref<1x1x1x272xui8, [@CMX_NN, 0]>

    // CHECK:       [[VAR2:%.*]] = VPUIP.SubView [[VAR1]] [0, 0, 0, 0] [1, 1, 1, 256] : 
    // CHECK-SAME:		memref<1x1x1x272xui8, [@CMX_NN, 0]> to memref<1x1x1x256xui8, 
    // CHECK-SAME:		{order = #NCHW, strides = [272, 272, 272, 1]}, [@CMX_NN, 0]>

    // CHECK:       [[VAR3:%.*]] = VPUIP.ViewOp [[VAR2]] : memref<1x1x1x256xui8,   
    // CHECK_SAME:		{order = #NCHW, strides = [272, 272, 272, 1]}, [@CMX_NN, 0]> to memref<16x1x1x4xsi32, @CMX_NN>

    // CHECK:       [[VAR4:%.*]] = VPUIP.SubView [[VAR1]] [0, 0, 0, 256] [1, 1, 1, 16] : 
    // CHECK-SAME:		memref<1x1x1x272xui8, [@CMX_NN, 0]> to memref<1x1x1x16xui8, 
    // CHECK-SAME:		{order = #NCHW, strides = [272, 272, 272, 1]}, [@CMX_NN, 0]>

    // CHECK:       [[VAR5:%.*]] = VPUIP.ViewOp [[VAR4]] : memref<1x1x1x16xui8, 
    // CHECK_SAME:		{order = #NCHW, strides = [272, 272, 272, 1]}, [@CMX_NN, 0]> to memref<1x1x1x16xui8, @CMX_NN>

    // CHECK:       [[VAR6:%.*]] = VPUIP.NCEClusterTask
    // CHECK-SAME:          activation_window_channel_length = 27 : i64,
    // CHECK-SAME:          constantsFused = true,
    // CHECK-SAME:          kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:          kernel_size = [1, 1],
    // CHECK-SAME:          kernel_strides = [1, 1],
    // CHECK-SAME:          task_type = "MAXPOOL"
    // CHECK-SAME:      input([[VAR_INPUT]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weight_table([[VAR3]] : memref<16x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      activation_window([[VAR5]] : memref<1x1x1x16xui8, @CMX_NN>)
    // CHECK-SAME:      parent_input([[VAR_INPUT]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      parent_output([[OUT_BUF2]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[OUT_BUF2]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK:           DPUTask
    // CHECK-SAME:          mpe_mode = "VECTOR_FP16",
    // CHECK-SAME:          outEnd = [55, 10, 15],
    // CHECK-SAME:          outStart = [0, 0, 0],
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK:       return [[VAR6]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!Input_DDR = type memref<1x16x16x16xf16, #NCHW, @DDR>
!Output_DDR = type memref<1x16x16x16xf16, #NHWC, @DDR>
!Weights_DDR = type memref<16x1x1x4xf16, #NHWC, @DDR>
!WeightsTable_DDR = type memref<16x1x1x4xsi32, @DDR>
!ActivationWindow_DDR = type memref<1x1x1x16xui8, @DDR>

!InputStub_CMX = type memref<1x16x16x16xf16, #NCHW, @CMX_NN>
!OutputStub_CMX = type memref<1x16x16x16xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = type memref<16x1x1x4xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = type memref<16x1x1x4xsi32, @CMX_NN>
!ActivationWindowStub_CMX = type memref<1x1x1x16xui8, @CMX_NN>

// CHECK-LABEL: @FuseConstantsCMConv
func @FuseConstantsCMConv(%in : !Input_DDR) -> !OutputStub_CMX {

    %buf0 = memref.alloc() : !InputStub_CMX
    %buf1 = memref.alloc() : !OutputStub_CMX
    %buf2 = memref.alloc() : !WeightsTableStub_CMX                   
    %buf3 = memref.alloc() : !ActivationWindowStub_CMX
    %buf4 = memref.alloc() : !WeightsStub_CMX

    %weight_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<16x1x1x4xsi32>
    %act_wind = const.Declare !ActivationWindow_DDR = dense<1> : tensor<1x1x1x16xui8>
    %weights = const.Declare !Weights_DDR = dense<1.0> : tensor<16x1x1x4xf16>, [#const.Reorder<#NHWC>]

    %0 = VPUIP.Copy inputs(%weight_table : !WeightsTable_DDR) outputs(%buf2 : !WeightsTableStub_CMX) -> !WeightsTableStub_CMX
    %1 = VPUIP.Copy inputs(%act_wind : !ActivationWindow_DDR) outputs(%buf3 : !ActivationWindowStub_CMX) -> !ActivationWindowStub_CMX
    %2 = VPUIP.Copy inputs(%in : !Input_DDR) outputs(%buf0 : !InputStub_CMX) -> !InputStub_CMX    
    %3 = VPUIP.Copy inputs(%weights : !Weights_DDR) outputs(%buf4 : !WeightsStub_CMX) -> !WeightsStub_CMX

    %4 = VPUIP.NCEClusterTask 
        {
                activation_window_channel_length = 16 : i64,
                kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                kernel_size = [3, 3],
                kernel_strides = [1, 1],
                task_type = "CMCONV"
        }  
        input(%2 : !InputStub_CMX)
        weights(%3 : !WeightsStub_CMX)
        weight_table(%0 : !WeightsTableStub_CMX)
        activation_window(%1 : !ActivationWindowStub_CMX)
        parent_input(%2 : !InputStub_CMX)
        parent_output(%buf1 : !OutputStub_CMX)
        outputs(%buf1 : !OutputStub_CMX) -> !OutputStub_CMX 
        variants :  
        {
            DPUTask 
            {
                outEnd = [55, 10, 15], mpe_mode = "VECTOR_FP16", 
                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, 
                outStart = [0, 0, 0]
            }
        } 
        PPE :  
        {
        }
    return %4 : !OutputStub_CMX

    // CHECK:       [[FUSED_CONSTANT:%.+]] = const.Declare memref<1x1x1x400xui8>
    // CHECK-NOT:   [[WEIGHT_TABLE:%.+]] = const.Declare !WeightsTable_DDR
    // CHECK-NOT:   [[ACTIVATION_WINDOW:%.+]] = const.Declare !ActivationWindow_DDR
    // CHECK-NOT:   [[WEIGHTS:%.+]] = const.Declare !Weights_DDR

    // CHECK:       [[OUT_BUF1:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, @CMX_NN>
    // CHECK:       [[VAR_INPUT:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x16x16x16xf16, @DDR>)
    // CHECK-SAME:      outputs(%0 : memref<1x16x16x16xf16, @CMX_NN>) -> memref<1x16x16x16xf16, @CMX_NN>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x1x1x400xui8, [@CMX_NN, 0]>
    // CHECK:       [[VAR1:%.*]] = VPUIP.Copy inputs([[FUSED_CONSTANT]] : memref<1x1x1x400xui8>)
    // CHECK-SAME:      outputs([[VAR0]] : memref<1x1x1x400xui8, [@CMX_NN, 0]>) -> memref<1x1x1x400xui8, [@CMX_NN, 0]>

    // CHECK:       [[VAR2:%.*]] = VPUIP.SubView [[VAR1]] [0, 0, 0, 0] [1, 1, 1, 256] : 
    // CHECK-SAME:		memref<1x1x1x400xui8, [@CMX_NN, 0]> to memref<1x1x1x256xui8, 
    // CHECK-SAME:		{order = #NCHW, strides = [400, 400, 400, 1]}, [@CMX_NN, 0]>

    // CHECK:       [[VAR3:%.*]] = VPUIP.ViewOp [[VAR2]] : memref<1x1x1x256xui8, 
    // CHECK_SAME:		{order = #NCHW, strides = [400, 400, 400, 1]}, [@CMX_NN, 0]> to memref<16x1x1x4xsi32, @CMX_NN>

    // CHECK:       [[VAR4:%.*]] = VPUIP.SubView [[VAR1]] [0, 0, 0, 256] [1, 1, 1, 128] : 
    // CHECK-SAME:		memref<1x1x1x400xui8, [@CMX_NN, 0]> to memref<1x1x1x128xui8, 
    // CHECK-SAME:		{order = #NCHW, strides = [400, 400, 400, 1]}, [@CMX_NN, 0]>

    // CHECK:       [[VAR5:%.*]] = VPUIP.ViewOp [[VAR4]] : memref<1x1x1x128xui8, 
    // CHECK_SAME:		{order = #NCHW, strides = [400, 400, 400, 1]}, [@CMX_NN, 0]> to memref<16x1x1x4xf16, #NHWC, @CMX_NN>

    // CHECK:       [[VAR6:%.*]] = VPUIP.SubView [[VAR1]] [0, 0, 0, 384] [1, 1, 1, 16] : 
    // CHECK-SAME:		memref<1x1x1x400xui8, [@CMX_NN, 0]> to memref<1x1x1x16xui8, 
    // CHECK-SAME:		{order = #NCHW, strides = [400, 400, 400, 1]}, [@CMX_NN, 0]>

    // CHECK:       [[VAR7:%.*]] = VPUIP.ViewOp [[VAR6]] : memref<1x1x1x16xui8, 
    // CHECK_SAME:		{order = #NCHW, strides = [400, 400, 400, 1]}, [@CMX_NN, 0]> to memref<1x1x1x16xui8, @CMX_NN>

    // CHECK:       [[VAR8:%.*]] = VPUIP.NCEClusterTask
    // CHECK-SAME:          activation_window_channel_length = 16 : i64,
    // CHECK-SAME:          constantsFused = true,
    // CHECK-SAME:          kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:          kernel_size = [3, 3],
    // CHECK-SAME:          kernel_strides = [1, 1],
    // CHECK-SAME:          task_type = "CMCONV"
    // CHECK-SAME:      input([[VAR_INPUT]] : memref<1x16x16x16xf16, @CMX_NN>)
    // CHECK-SAME:      weights([[VAR5]] : memref<16x1x1x4xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weight_table([[VAR3]] : memref<16x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      activation_window([[VAR7]] : memref<1x1x1x16xui8, @CMX_NN>)
    // CHECK-SAME:      parent_input([[VAR_INPUT]] : memref<1x16x16x16xf16, @CMX_NN>)
    // CHECK-SAME:      parent_output([[OUT_BUF1]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[OUT_BUF1]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK:           DPUTask
    // CHECK-SAME:          mpe_mode = "VECTOR_FP16",
    // CHECK-SAME:          outEnd = [55, 10, 15],
    // CHECK-SAME:          outStart = [0, 0, 0],
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}

    // CHECK:       return [[VAR8]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input_DDR = type memref<1x16x16x16xf16, #NHWC, @DDR>
!Output_DDR = type memref<1x16x16x16xf16, #NHWC, @DDR>
!Weights_DDR = type memref<16x16x1x1xf16, {compressionScheme = #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<16> : tensor<16xi64>, alignment = 16 : i64>, order = #NHWC}, @DDR>
!WeightsSM_DDR = type memref<16x1x1x128xi1, @DDR>
!WeightsTable_DDR = type memref<16x1x1x4xsi32, @DDR>

!InputStub_CMX = type memref<1x16x16x16xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = type memref<1x16x16x16xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = type memref<16x16x1x1xf16, {compressionScheme = #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<16> : tensor<16xi64>, alignment = 16 : i64>, order = #NHWC}, @CMX_NN>
!WeightsSMStub_CMX = type memref<16x1x1x128xi1, @CMX_NN>
!WeightsTableStub_CMX = type memref<16x1x1x4xsi32, @CMX_NN>
!ActivationWindowStub_CMX = type memref<1x1x1x16xui8, @CMX_NN>

// CHECK-LABEL: @FuseConstantsConvSparseWeights
func @FuseConstantsConvSparseWeights(%in : !Input_DDR) -> !OutputStub_CMX {

    %buf0 = memref.alloc() : !InputStub_CMX
    %buf1 = memref.alloc() : !OutputStub_CMX
    %buf2 = memref.alloc() : !WeightsTableStub_CMX
    %buf3 = memref.alloc() : !WeightsStub_CMX
    %buf4 = memref.alloc() : !WeightsSMStub_CMX

    %weight_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<16x1x1x4xsi32>
    %weights = const.Declare !Weights_DDR = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false, dense<16> : tensor<16xi64>>]
    %weights_sm = const.Declare !WeightsSM_DDR = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

    %0 = VPUIP.Copy inputs(%weight_table : !WeightsTable_DDR) outputs(%buf2 : !WeightsTableStub_CMX) -> !WeightsTableStub_CMX
    %1 = VPUIP.Copy inputs(%in : !Input_DDR) outputs(%buf0 : !InputStub_CMX) -> !InputStub_CMX
    %2 = VPUIP.Copy inputs(%weights : !Weights_DDR) outputs(%buf3 : !WeightsStub_CMX) -> !WeightsStub_CMX
    %3 = VPUIP.Copy inputs(%weights_sm : !WeightsSM_DDR) outputs(%buf4 : !WeightsSMStub_CMX) -> !WeightsSMStub_CMX
    %4 = VPUIP.NCEClusterTask
        {
            activation_window_channel_length = 27 : i64,
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = "CONV"
        } 
        input(%1 : !InputStub_CMX)
        weights(%2 : !WeightsStub_CMX)
        weights_sparsity_map(%3 : !WeightsSMStub_CMX)
        weight_table(%0 : !WeightsTableStub_CMX)
        parent_input(%1 : !InputStub_CMX)
        parent_output(%buf1 : !OutputStub_CMX)
        outputs(%buf1 : !OutputStub_CMX) -> !OutputStub_CMX
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 10, 15], mpe_mode = "VECTOR_FP16",
                    pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                    outStart = [0, 0, 0]
                }
        }
        PPE :
        {
        }
    return %4 : !OutputStub_CMX

    // CHECK:       [[FUSED_CONSTANT:%.+]] = const.Declare memref<1x1x1x1024xui8>
    // CHECK-NOT:   [[WEIGHT_TABLE:%.+]] = const.Declare !WeightsTable_DDR
    // CHECK-NOT:   [[WEIGHTS:%.+]] = const.Declare !Weights_DDR
    // CHECK-NOT:   [[WEIGHTS_SM:%.+]] = const.Declare !WeightsSM_DDR

    // CHECK:       [[OUT_BUF1:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, @CMX_NN>
    // CHECK:       [[OUT_BUF2:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, @CMX_NN>
    // CHECK:       [[VAR_INPUT:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs(%0 : memref<1x16x16x16xf16, #NHWC, @CMX_NN>) -> memref<1x16x16x16xf16, #NHWC, @CMX_NN>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x1x1x1024xui8, [@CMX_NN, 0]>
    // CHECK:       [[VAR1:%.*]] = VPUIP.Copy inputs([[FUSED_CONSTANT]] : memref<1x1x1x1024xui8>)
    // CHECK-SAME:      outputs([[VAR0]] : memref<1x1x1x1024xui8, [@CMX_NN, 0]>) -> memref<1x1x1x1024xui8, [@CMX_NN, 0]>

    // CHECK:       [[VAR2:%.*]] = VPUIP.SubView [[VAR1]] [0, 0, 0, 0] [1, 1, 1, 256] :
    // CHECK-SAME:      memref<1x1x1x1024xui8, [@CMX_NN, 0]> to memref<1x1x1x256xui8,
    // CHECK-SAME:      {order = #NCHW, strides = [1024, 1024, 1024, 1]}, [@CMX_NN, 0]>

    // CHECK:       [[VAR3:%.*]] = VPUIP.ViewOp [[VAR2]] : memref<1x1x1x256xui8,
    // CHECK_SAME:      {order = #NCHW, strides = [1024, 1024, 1024, 1]}, [@CMX_NN, 0]> to memref<16x1x1x4xsi32, @CMX_NN>

    // CHECK:       [[VAR4:%.*]] = VPUIP.SubView [[VAR1]] [0, 0, 0, 256] [1, 1, 1, 512] :
    // CHECK-SAME:      memref<1x1x1x1024xui8, [@CMX_NN, 0]> to memref<1x1x1x512xui8,
    // CHECK-SAME:      {order = #NCHW, strides = [1024, 1024, 1024, 1]}, [@CMX_NN, 0]>

    // CHECK:       [[VAR5:%.*]] = VPUIP.ViewOp [[VAR4]] :
    // CHECK_SAME:      memref<1x1x1x512xui8, {order = #NCHW, strides = [1024, 1024, 1024, 1]}, [@CMX_NN, 0]> to
    // CHECK-SAME:      memref<16x16x1x1xf16, {compressionScheme = #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<16> : tensor<16xi64>, alignment = 16 : i64>, order = #NHWC}, @CMX_NN>

    // CHECK:       [[VAR6:%.*]] = VPUIP.SubView [[VAR1]] [0, 0, 0, 768] [1, 1, 1, 256] :
    // CHECK-SAME:      memref<1x1x1x1024xui8, [@CMX_NN, 0]> to memref<1x1x1x256xui8,
    // CHECK-SAME:      {order = #NCHW, strides = [1024, 1024, 1024, 1]}, [@CMX_NN, 0]>

    // CHECK:       [[VAR7:%.*]] = VPUIP.ViewOp [[VAR6]] : memref<1x1x1x256xui8,
    // CHECK_SAME:      {order = #NCHW, strides = [1024, 1024, 1024, 1]}, [@CMX_NN, 0]> to memref<16x1x1x128xi1, @CMX_NN>

    // CHECK:       [[VAR8:%.*]] = VPUIP.NCEClusterTask
    // CHECK-SAME:          activation_window_channel_length = 27 : i64,
    // CHECK-SAME:          constantsFused = true,
    // CHECK-SAME:          kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:          kernel_size = [1, 1],
    // CHECK-SAME:          kernel_strides = [1, 1],
    // CHECK-SAME:          task_type = "CONV"
    // CHECK-SAME:      input([[VAR_INPUT]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weights([[VAR5]] : memref<16x16x1x1xf16, {compressionScheme = #VPUIP.CompressionScheme<axis = 0 : i64, numElems = dense<16> : tensor<16xi64>, alignment = 16 : i64>, order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:      weights_sparsity_map([[VAR7]] : memref<16x1x1x128xi1, @CMX_NN>)
    // CHECK-SAME:      weight_table([[VAR3]] : memref<16x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      parent_input([[VAR_INPUT]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      parent_output([[OUT_BUF2]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[OUT_BUF2]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK:           DPUTask
    // CHECK-SAME:          mpe_mode = "VECTOR_FP16",
    // CHECK-SAME:          outEnd = [55, 10, 15],
    // CHECK-SAME:          outStart = [0, 0, 0],
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK:       return [[VAR8]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x16x56x56xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsDistributed = type !VPUIP.DistributedBuffer<
    32x32x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsTableDistributed = type !VPUIP.DistributedBuffer<
    32x1x1x4xsi32, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!OutputDistributed = type !VPUIP.DistributedBuffer<
    1x16x56x56xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = type memref<1x16x56x56xf16, #NHWC, @DDR>
!Weights_DDR = type memref<32x32x1x1xf16, #NHWC, @DDR>
!WeightsTable_DDR = type memref<32x1x1x4xsi32, @DDR>
!Output_DDR = type memref<1x16x56x56xf16, #NHWC, @DDR>

!InputStub_CMX = type memref<1x16x56x56xf16, #NHWC, [@CMX_NN, 0]>
!WeightsStub_CMX = type memref<32x32x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = type memref<32x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
!OutputStub_CMX = type memref<1x16x56x56xf16, #NHWC, [@CMX_NN, 0]>

// CHECK-LABEL: @FuseDuplicatedConstants
func @FuseDuplicatedConstants(%input : !Input_DDR) -> !Output_DDR 
{
    %weight_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<32x1x1x4xsi32>
    %weights = const.Declare !Weights_DDR = dense<1.0> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]

    %input_cmx = VPURT.AllocDistributed -> !InputDistributed
    %weights_table_cmx = VPURT.AllocDistributed -> !WeightsTableDistributed
    %weights_cmx = VPURT.AllocDistributed -> !WeightsDistributed
    %output_buff_1_cmx = VPURT.AllocDistributed -> !OutputDistributed
    %output_buff_2_cmx = VPURT.AllocDistributed -> !OutputDistributed
    %output = memref.alloc() : !Output_DDR

    %1 = VPUIP.NCEClusterTiling inputs(%input as %arg0: !Input_DDR) outputs(%input_cmx as %arg1: !InputStub_CMX) -> !InputDistributed {
        %0 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%arg0: !Input_DDR) outputs(%arg1: !InputStub_CMX) -> !InputStub_CMX
    }

    %2 = VPUIP.NCEClusterTiling inputs(%weight_table as %arg0: !WeightsTable_DDR) outputs(%weights_table_cmx as %arg1: !WeightsTableStub_CMX) -> !WeightsTableDistributed {
        %0 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%arg0: !WeightsTable_DDR) outputs(%arg1: !WeightsTableStub_CMX) -> !WeightsTableStub_CMX
    }

    %3 = VPUIP.NCEClusterTiling inputs(%weights as %arg0: !Weights_DDR) outputs(%weights_cmx as %arg1: !WeightsStub_CMX) -> !WeightsDistributed {
        %0 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%arg0: !Weights_DDR) outputs(%arg1: !WeightsStub_CMX) -> !WeightsStub_CMX
    }

    %4 = VPUIP.NCEClusterTiling
            inputs(%1 as %arg0: !InputStub_CMX,
                    %2 as %arg1: !WeightsTableStub_CMX,
                    %3 as %arg2: !WeightsStub_CMX)
            outputs(%output_buff_1_cmx as %arg3: !OutputStub_CMX) -> !OutputStub_CMX {
        %0 = VPUIP.NCEClusterTask
            {
                activation_window_channel_length = 27 : i64,
                kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = "CONV"
            }
            input(%arg0 : !InputStub_CMX)
            weights(%arg2 : !WeightsStub_CMX)
            weight_table(%arg1 : !WeightsTableStub_CMX)
            parent_input(%arg0 : !InputStub_CMX)
            parent_output(%arg3 : !OutputStub_CMX)
            outputs(%arg3 : !OutputStub_CMX) -> !OutputStub_CMX
            variants :
            {
                DPUTask
                {
                    outEnd = [55, 55, 15], mpe_mode = "VECTOR_FP16",
                    pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                    outStart = [0, 0, 0]
                }
            }
            PPE :
            {
            }
        }

    %5 = VPUIP.NCEClusterTiling inputs(%4 as %arg0: !OutputStub_CMX) outputs(%output as %arg1: !Output_DDR) -> !Output_DDR {
        %0 = VPUIP.Copy { out_mem_space = @DDR } inputs(%arg0: !OutputStub_CMX) outputs(%arg1: !Output_DDR) -> !Output_DDR
    }

    return %5 : !Output_DDR

    // Verify that constants are fused by checking no other cst is present other than fused constant

    // CHECK:   [[FUSED_CONSTANT:%.+]] = const.Declare memref<1x1x1x2560xui8>
    // CHECK-NOT:   [[WEIGHT_TABLE:%.+]] = const.Declare !WeightsTable_DDR
    // CHECK-NOT:   [[WEIGHTS:%.+]] = const.Declare !Weights_DDR

    // CHECK:   [[BUF_OUT_1_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:   [[BUF_OUT_2_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:   [[BUF_OUT_DDR:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @DDR>
    
    // CHECK:       [[COPY_INPUT:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs(%arg0 as %arg1: memref<1x16x56x56xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[BUF_OUT_1_CMX]]

    // CHECK:   [[FUSED_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x2560xui8, {order = #NCHW, strides = [2560, 2560, 2560, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[COPY_FUSED:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[FUSED_CONSTANT]] as %arg1: memref<1x1x1x2560xui8>)
    // CHECK-SAME:      outputs([[FUSED_CMX]]
}
