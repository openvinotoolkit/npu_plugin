//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-vpu-nce-to-vpuip --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX37XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @NCEPermuteQuantize
func.func @NCEPermuteQuantize(%arg0: memref<1x32x3x1568xf16, #NHWC, @CMX_NN>) -> memref<1x32x4x1568xf16, #NWCH, @CMX_NN> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x32x3x1568xf16, #NHWC, @CMX_NN>
        to tensor<1x32x3x1568xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %1 = VPU.NCE.PermuteQuantize(%0) {
        dstElemType = !qElemType,
        dstOrder = #NWCH,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPETask<
            clamp_high = 255 : i64,
            clamp_low = 0 : i64,
            fp_prelu_alpha = 1.000000e+00 : f64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = <NOOP>
        >
    } -> tensor<1x32x4x1568x!qElemType, {mem_space = @CMX_NN, order = #NWCH}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 3, 1568] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64> #VPU.mpe_mode<CUBOID_16x16>
    }

    %2 = builtin.unrealized_conversion_cast %1 : tensor<1x32x4x1568x!qElemType, {mem_space = @CMX_NN, order = #NWCH}>
        to memref<1x32x4x1568xf16, #NWCH, @CMX_NN>

    return %2 : memref<1x32x4x1568xf16, #NWCH, @CMX_NN>

    // CHECK-NOT:   VPU.NCE.PermuteQuantize

    // CHECK:       [[ALLOC:%.*]] = memref.alloc() : memref<1x32x4x1568x!qElemType, #NWCH, @CMX_NN>

    // CHECK:       VPUIP.NCEClusterTask {
    // CHECK-SAME:      activation_window_channel_length = 0 : i64,
    // CHECK-SAME:      is_superdense,
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<ELTWISE>
    // CHECK-SAME:  }
    // CHECK-SAME:  input(%arg0 : memref<1x32x3x1568xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  weights(%arg0 : memref<1x32x3x1568xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  parent_input(%arg0 : memref<1x32x3x1568xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  parent_output([[ALLOC]] : memref<1x32x4x1568x!qElemType, #NWCH, @CMX_NN>)
    // CHECK-SAME:  outputs([[ALLOC]] : memref<1x32x4x1568x!qElemType, #NWCH, @CMX_NN>)
    // CHECK-SAME:  -> memref<1x32x4x1568x!qElemType, #NWCH, @CMX_NN>
    // CHECK-SAME:  variants : {
    // CHECK:           DPUTask {
    // CHECK-SAME:          <CUBOID_16x16>,
    // CHECK-SAME:          outEnd = [1567, 2, 31],
    // CHECK-SAME:          outStart = [0, 0, 0],
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>
    // CHECK:           } PPE : {
    // CHECK:               PPETask <ADD> {
    // CHECK-SAME:              clamp_high = 255 : i64,
    // CHECK-SAME:              clamp_low = 0 : i64,
    // CHECK-SAME:              lrelu_mult = 1 : i64,
    // CHECK-SAME:              lrelu_shift = 0 : i64,
    // CHECK-SAME:              quant_scale = [5.000000e-01]
    // CHECK-SAME:          }
    // CHECK:           }
    // CHECK:       }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SuperdenseNCEConvolution
func.func @SuperdenseNCEConvolution(%arg0: memref<1x16x15x15xf16, #NHWC, @CMX_NN>,
                     %arg1: memref<16x16x1x1xf16, #NHWC, @CMX_NN>,
                     %arg2: memref<16x1x1x4xsi32, #NHWC, @CMX_NN>
                     ) -> memref<1x16x15x15xf16, #NCHW, @CMX_NN> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x16x15x15xf16, #NHWC, @CMX_NN>
        to tensor<1x16x15x15xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = builtin.unrealized_conversion_cast %arg1 : memref<16x16x1x1xf16, #NHWC, @CMX_NN>
        to tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %2 = builtin.unrealized_conversion_cast %arg2 : memref<16x1x1x4xsi32, #NHWC, @CMX_NN>
        to tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

    %3 = VPU.NCE.Convolution(%0, %1, %2) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPETask<
            clamp_high = 2147483647 : i64,
            clamp_low = -2147483648 : i64,
            fp_prelu_alpha = 1.000000e+00 : f64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = <NOOP>
        >,
        rawFilterShape = [16, 16, 1, 1],
        strides = [1, 1]
    } -> tensor<1x16x15x15xf16, {mem_space = @CMX_NN, order = #NCHW}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 15, 15] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 13 : i64> #VPU.mpe_mode<CUBOID_16x16>
    }

    %4 = builtin.unrealized_conversion_cast %3 : tensor<1x16x15x15xf16, {mem_space = @CMX_NN, order = #NCHW}>
        to memref<1x16x15x15xf16, #NCHW, @CMX_NN>

    return %4 : memref<1x16x15x15xf16, #NCHW, @CMX_NN>

    // CHECK:       VPUIP.NCEClusterTask {
    // CHECK-SAME:      is_superdense,
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME:  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SuperdenseNCEMaxPool
func.func @SuperdenseNCEMaxPool(%arg0: memref<1x16x15x15xf16, #NHWC, @CMX_NN>,
                 %arg1: memref<16x1x1x4xsi32, #NHWC, @CMX_NN>,
                 %arg2: memref<1x1x1x16xui8, #NHWC, @CMX_NN>
                 ) -> memref<1x16x15x15xf16, #NCHW, @CMX_NN> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x16x15x15xf16, #NHWC, @CMX_NN>
        to tensor<1x16x15x15xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = builtin.unrealized_conversion_cast %arg1 : memref<16x1x1x4xsi32, #NHWC, @CMX_NN>
        to tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
    %2 = builtin.unrealized_conversion_cast %arg2 : memref<1x1x1x16xui8, #NHWC, @CMX_NN>
        to tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NHWC}>

    %3 = VPU.NCE.MaxPool(%0, %1, %2) {
        activation_window_channel_length = 4 : i64,
        kernel_size = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPETask<
            clamp_high = 2147483647 : i64,
            clamp_low = -2147483648 : i64,
            fp_prelu_alpha = 1.000000e+00 : f64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = <NOOP>
        >,
        strides = [1, 1]
    } -> tensor<1x16x15x15xf16, {mem_space = @CMX_NN, order = #NCHW}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 15, 15] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 13 : i64> #VPU.mpe_mode<CUBOID_16x16>
    }

    %4 = builtin.unrealized_conversion_cast %3 : tensor<1x16x15x15xf16, {mem_space = @CMX_NN, order = #NCHW}>
        to memref<1x16x15x15xf16, #NCHW, @CMX_NN>

    return %4 : memref<1x16x15x15xf16, #NCHW, @CMX_NN>

    // CHECK:       VPUIP.NCEClusterTask {
    // CHECK-SAME:      is_superdense,
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<MAXPOOL>
    // CHECK-SAME:  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SuperdenseNCEAveragePool
func.func @SuperdenseNCEAveragePool(%arg0: memref<1x16x15x15xf16, #NHWC, @CMX_NN>) -> memref<1x16x15x15xf16, #NCHW, @CMX_NN> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x16x15x15xf16, #NHWC, @CMX_NN>
        to tensor<1x16x15x15xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %1 = VPU.NCE.AveragePool(%0) {
        kernel_size = [1, 1],
        minimumHardwareExecutionCost = 708 : i64,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPETask<
            clamp_high = 2147483647 : i64,
            clamp_low = -2147483648 : i64,
            fp_prelu_alpha = 1.000000e+00 : f64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = <NOOP>,
            quant_scale = [1.000000e+00]
        >,
        strides = [1, 1]
    } -> tensor<1x16x15x15xf16, {mem_space = @CMX_NN, order = #NCHW}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 15, 15] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 13 : i64> #VPU.mpe_mode<CUBOID_16x16>
    }

    %2 = builtin.unrealized_conversion_cast %1 : tensor<1x16x15x15xf16, {mem_space = @CMX_NN, order = #NCHW}>
        to memref<1x16x15x15xf16, #NCHW, @CMX_NN>

    return %2 : memref<1x16x15x15xf16, #NCHW, @CMX_NN>

    // CHECK:       VPUIP.NCEClusterTask {
    // CHECK-SAME:      is_superdense,
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<AVEPOOL>
    // CHECK-SAME:  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SuperdenseNCEEltwise
func.func @SuperdenseNCEEltwise(%arg0: memref<1x16x15x15xf16, #NHWC, @CMX_NN>,
                 %arg1: memref<1x16x15x15xf16, #NHWC, @CMX_NN>
                 ) -> memref<1x16x15x15xf16, #NCHW, @CMX_NN> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x16x15x15xf16, #NHWC, @CMX_NN>
        to tensor<1x16x15x15xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %1 = builtin.unrealized_conversion_cast %arg1 : memref<1x16x15x15xf16, #NHWC, @CMX_NN>
        to tensor<1x16x15x15xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %2 = VPU.NCE.Eltwise(%0, %1) {
        minimumHardwareExecutionCost = 585 : i64,
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPETask<
            clamp_high = 2147483647 : i64,
            clamp_low = -2147483648 : i64,
            fp_prelu_alpha = 1.000000e+00 : f64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = <NOOP>,
            quant_scale = [1.000000e+00]
        >
    } -> tensor<1x16x15x15xf16, {mem_space = @CMX_NN, order = #NCHW}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 15, 15] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 13 : i64> <CUBOID_16x16>
    }

    %3 = builtin.unrealized_conversion_cast %2 : tensor<1x16x15x15xf16, {mem_space = @CMX_NN, order = #NCHW}>
        to memref<1x16x15x15xf16, #NCHW, @CMX_NN>

    return %3 : memref<1x16x15x15xf16, #NCHW, @CMX_NN>

    // CHECK:       VPUIP.NCEClusterTask {
    // CHECK-SAME:      is_superdense,
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<ELTWISE>
    // CHECK-SAME:  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @InterpolateNearest(
    %arg0: memref<1x64x5x10xf16, #NHWC>,           // data
    %arg1: memref<64x64x1x1xf16, #NHWC, @CMX_NN>,  // weights
    %arg2: memref<64x1x1x4xsi32, @CMX_NN>          // weight table
) -> tensor<1x64x10x20xf16, {order = #NHWC, mem_space = @CMX_NN}> {
    %data = builtin.unrealized_conversion_cast %arg0 : memref<1x64x5x10xf16, #NHWC> to tensor<1x64x5x10xf16, {order = #NHWC}>
    %weights = builtin.unrealized_conversion_cast %arg1 : memref<64x64x1x1xf16, #NHWC, @CMX_NN> to tensor<64x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %weightsTable = builtin.unrealized_conversion_cast %arg2 : memref<64x1x1x4xsi32, @CMX_NN> to tensor<64x1x1x4xsi32, {mem_space = @CMX_NN}>

    %sparsityMap = const.Declare tensor<1x64x10x20xi1> = dense<1> : tensor<1x64x10x20xi1>

    %storageElement = VPU.StorageElementTable {
        dataElemType = i32,
        seDepth = 1,
        seSize = 64,
        dataShape = [1, 64, 5, 10],
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 10, 20]>
    } -> tensor<1x1x10x20xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%data, %sparsityMap, %storageElement) {
        seAttr = #VPU.SEInterpolate<
            nearest_mode = <FLOOR>,
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 10, 20]>
    } ->
        !VPU.SparseTensor<
            data=tensor<1x64x5x10xf16, {order = #NHWC}>,
            sparsity_map=tensor<1x64x10x20xi1>,
            storage_element_table=tensor<1x1x10x20xi32, {order = #NHWC}>,
            #VPU.SEInterpolate<
                mode = <NEAREST>,
                coordinate_transformation_mode = <ASYMMETRIC>,
                scale = [1.0, 1.0, 2.0, 2.0],
                nearest_mode = <FLOOR>,
                offsets = [0, 0, 0, 0],
                sizes = [1, 64, 10, 20]>>

    %input_cmx = VPU.Copy(%input) {out_mem_space = @CMX_NN} : !VPU.SparseTensor<
        data=tensor<1x64x5x10xf16, {order = #NHWC}>,
        sparsity_map=tensor<1x64x10x20xi1>,
        storage_element_table=tensor<1x1x10x20xi32, {order = #NHWC}>,
        #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 10, 20]>>
        -> !VPU.SparseTensor<
            data=tensor<1x64x5x10xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            sparsity_map=tensor<1x64x10x20xi1, {mem_space = @CMX_NN}>,
            storage_element_table=tensor<1x1x10x20xi32, {mem_space = @CMX_NN, order = #NHWC}>,
            #VPU.SEInterpolate<
                mode = <NEAREST>,
                coordinate_transformation_mode = <ASYMMETRIC>,
                scale = [1.0, 1.0, 2.0, 2.0],
                nearest_mode = <FLOOR>,
                offsets = [0, 0, 0, 0],
                sizes = [1, 64, 10, 20]>>

    %task = VPU.NCE.Interpolate(%input_cmx, %weights, %weightsTable) {
        rawFilterShape = [64, 64, 1, 1],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<NEAREST>,
        scales_attr = [2, 2],
        ppe = #VPU.PPETask<clamp_high = 2147483647, clamp_low = 0, lrelu_mult = 1, lrelu_shift = 0, mode = <AND>>
    } -> tensor<1x64x10x20xf16, {order = #NHWC, mem_space = @CMX_NN}>
    {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 10, 20] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<VECTOR_FP16>
    }

    return %task : tensor<1x64x10x20xf16, {order = #NHWC, mem_space = @CMX_NN}>

    // CHECK:       [[SPARSITY_MAP1:%.+]] = const.Declare tensor<1x64x10x20xi1> = dense<true> : tensor<1x64x10x20xi1>
    // CHECK:       [[DATA1:%.+]] = builtin.unrealized_conversion_cast %arg0 : memref<1x64x5x10xf16, #NHWC> to tensor<1x64x5x10xf16, {order = #NHWC}>
    // CHECK:       [[STORAGE_ELEMENT1:%.+]] = VPU.StorageElementTable {
    // CHECK-SAME:      dataElemType = i32,
    // CHECK-SAME:      dataShape = [1, 64, 5, 10],
    // CHECK-SAME:      seAttr = #VPU.SEInterpolate<
    // CHECK-SAME:          mode = <NEAREST>,
    // CHECK-SAME:          coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:          scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:          nearest_mode = <FLOOR>,
    // CHECK-SAME:          offsets = [0, 0, 0, 0],
    // CHECK-SAME:          sizes = [1, 64, 10, 20]>,
    // CHECK-SAME:      seDepth = 1 : i64,
    // CHECK-SAME:      seSize = 64 : i64
    // CHECK-SAME:  } -> tensor<1x1x10x20xi32, {order = #NHWC}>

    // CHECK:       [[SPARSE_TENSOR:%.+]] = VPU.GroupSparseTensor([[DATA1]], [[SPARSITY_MAP1]], [[STORAGE_ELEMENT1]]) {
    // CHECK-SAME:      seAttr = #VPU.SEInterpolate<
    // CHECK-SAME:          mode = <NEAREST>,
    // CHECK-SAME:          coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:          scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:          nearest_mode = <FLOOR>,
    // CHECK-SAME:          offsets = [0, 0, 0, 0],
    // CHECK-SAME:          sizes = [1, 64, 10, 20]
    // CHECK-SAME:      >} -> !VPU.SparseTensor<
    // CHECK-SAME:          data=tensor<1x64x5x10xf16, {order = #NHWC}>,
    // CHECK-SAME:          sparsity_map=tensor<1x64x10x20xi1>,
    // CHECK-SAME:          storage_element_table=tensor<1x1x10x20xi32, {order = #NHWC}>,
    // CHECK-SAME:          #VPU.SEInterpolate<
    // CHECK-SAME:              mode = <NEAREST>,
    // CHECK-SAME:              coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:              scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:              nearest_mode = <FLOOR>,
    // CHECK-SAME:              offsets = [0, 0, 0, 0],
    // CHECK-SAME:              sizes = [1, 64, 10, 20]>>

    // CHECK:       [[SPARSE_TENSOR_CMX:%.+]] = VPU.Copy([[SPARSE_TENSOR]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } :
    // CHECK-SAME:      !VPU.SparseTensor<
    // CHECK-SAME:          data=tensor<1x64x5x10xf16, {order = #NHWC}>,
    // CHECK-SAME:          sparsity_map=tensor<1x64x10x20xi1>,
    // CHECK-SAME:          storage_element_table=tensor<1x1x10x20xi32, {order = #NHWC}>,
    // CHECK-SAME:          #VPU.SEInterpolate<
    // CHECK-SAME:              mode = <NEAREST>,
    // CHECK-SAME:              coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:              scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:              nearest_mode = <FLOOR>,
    // CHECK-SAME:              offsets = [0, 0, 0, 0],
    // CHECK-SAME:                  sizes = [1, 64, 10, 20]>>
    // CHECK-SAME:      -> !VPU.SparseTensor<
    // CHECK-SAME:          data=tensor<1x64x5x10xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:          sparsity_map=tensor<1x64x10x20xi1, {mem_space = @CMX_NN}>,
    // CHECK-SAME:          storage_element_table=tensor<1x1x10x20xi32, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:          #VPU.SEInterpolate<
    // CHECK-SAME:              mode = <NEAREST>,
    // CHECK-SAME:              coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:              scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:              nearest_mode = <FLOOR>,
    // CHECK-SAME:              offsets = [0, 0, 0, 0],
    // CHECK-SAME:              sizes = [1, 64, 10, 20]>>

    // CHECK:       [[SPARSE_BUFFER:%.+]] = builtin.unrealized_conversion_cast [[SPARSE_TENSOR_CMX]] :
    // CHECK-SAME:      !VPU.SparseTensor<
    // CHECK-SAME:          data=tensor<1x64x5x10xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:          sparsity_map=tensor<1x64x10x20xi1, {mem_space = @CMX_NN}>,
    // CHECK-SAME:          storage_element_table=tensor<1x1x10x20xi32, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:          #VPU.SEInterpolate<
    // CHECK-SAME:              mode = <NEAREST>,
    // CHECK-SAME:              coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:              scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:              nearest_mode = <FLOOR>,
    // CHECK-SAME:              offsets = [0, 0, 0, 0],
    // CHECK-SAME:              sizes = [1, 64, 10, 20]>>
    // CHECK-SAME:      to !VPUIP.SparseBuffer<
    // CHECK-SAME:          data=memref<1x64x5x10xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:          sparsity_map=memref<1x64x10x20xi1, @CMX_NN>,
    // CHECK-SAME:          storage_element_table=memref<1x1x10x20xi32, #NHWC, @CMX_NN>,
    // CHECK-SAME:          #VPU.SEInterpolate<
    // CHECK-SAME:              mode = <NEAREST>,
    // CHECK-SAME:              coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:              scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:              nearest_mode = <FLOOR>,
    // CHECK-SAME:              offsets = [0, 0, 0, 0],
    // CHECK-SAME:              sizes = [1, 64, 10, 20]>>

    // CHECK:       [[ALLOC:%.+]] = memref.alloc() : memref<1x64x10x20xf16, #NHWC, @CMX_NN>

    // CHECK:       [[DATA2:%.+]], [[SPARSITY_MAP2:%.+]], [[STORAGE_ELEMENT2:%.+]] = VPUIP.UngroupSparseBuffer([[SPARSE_BUFFER]]) {
    // CHECK-SAME:      resultSegmentSizes = array<i32: 1, 1, 1>
    // CHECK-SAME:  } ->
    // CHECK-SAME:      memref<1x64x5x10xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:      memref<1x64x10x20xi1, @CMX_NN>,
    // CHECK-SAME:      memref<1x1x10x20xi32, #NHWC, @CMX_NN>

    // CHECK:       [[TASK_BUFFER:%.+]] = VPUIP.NCEClusterTask {
    // CHECK-SAME:      kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      kernel_strides = [1, 1],
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME:  }
    // CHECK-SAME:      input([[DATA2]] : memref<1x64x5x10xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      input_sparsity_map([[SPARSITY_MAP2]] : memref<1x64x10x20xi1, @CMX_NN>)
    // CHECK-SAME:      input_storage_element_table([[STORAGE_ELEMENT2]] : memref<1x1x10x20xi32, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weights(%arg1 : memref<64x64x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weight_table(%arg2 : memref<64x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      parent_input([[DATA2]] : memref<1x64x5x10xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      parent_input_sparsity_map([[SPARSITY_MAP2]] : memref<1x64x10x20xi1, @CMX_NN>)
    // CHECK-SAME:      parent_input_storage_element_table([[STORAGE_ELEMENT2]] : memref<1x1x10x20xi32, #NHWC, @CMX_NN>)
    // CHECK-SAME:      parent_output([[ALLOC]] : memref<1x64x10x20xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[ALLOC]] : memref<1x64x10x20xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  -> memref<1x64x10x20xf16, #NHWC, @CMX_NN> variants : {
    // CHECK:           DPUTask {mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [19, 9, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:       } PPE : {
    // CHECK:           PPETask <AND> {clamp_high = 2147483647 : i64, clamp_low = 0 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
    // CHECK:       }

    // CHECK:       [[TASK_TENSOR:%.+]] = builtin.unrealized_conversion_cast [[TASK_BUFFER]] : memref<1x64x10x20xf16, #NHWC, @CMX_NN> to tensor<1x64x10x20xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       return [[TASK_TENSOR]] : tensor<1x64x10x20xf16, {mem_space = @CMX_NN, order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @InterpolateBilinear(
    %arg0: memref<1x64x5x10xf16, #NHWC>,           // data
    %arg1: memref<64x64x1x1xf16, #NHWC, @CMX_NN>,  // weights
    %arg2: memref<64x1x1x4xsi32, @CMX_NN>          // weight table
) -> tensor<1x64x10x20xf16, {order = #NHWC, mem_space = @CMX_NN}> {
    %data = builtin.unrealized_conversion_cast %arg0 : memref<1x64x5x10xf16, #NHWC> to tensor<1x64x5x10xf16, {order = #NHWC}>
    %weights = builtin.unrealized_conversion_cast %arg1 : memref<64x64x1x1xf16, #NHWC, @CMX_NN> to tensor<64x64x1x1xf16, {order = #NHWC, mem_space = @CMX_NN}>
    %weightsTable = builtin.unrealized_conversion_cast %arg2 : memref<64x1x1x4xsi32, @CMX_NN> to tensor<64x1x1x4xsi32, {mem_space = @CMX_NN}>

    %sparsityMap = const.Declare tensor<1x64x11x21xi1> = dense<1> : tensor<1x64x11x21xi1>

    %storageElement = VPU.StorageElementTable {
        dataElemType = i32,
        seDepth = 1,
        seSize = 64,
        dataShape = [1, 64, 5, 10],
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 11, 21]>
    } -> tensor<1x1x11x21xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%data, %sparsityMap, %storageElement) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 11, 21]>
    } ->
        !VPU.SparseTensor<
            data=tensor<1x64x5x10xf16, {order = #NHWC}>,
            sparsity_map=tensor<1x64x11x21xi1>,
            storage_element_table=tensor<1x1x11x21xi32, {order = #NHWC}>,
            #VPU.SEInterpolate<
                mode = <BILINEAR>,
                coordinate_transformation_mode = <ASYMMETRIC>,
                scale = [1.0, 1.0, 2.0, 2.0],
                offsets = [0, 0, 0, 0],
                sizes = [1, 64, 11, 21]>>

    %input_cmx = VPU.Copy(%input) {out_mem_space = @CMX_NN} : !VPU.SparseTensor<
        data=tensor<1x64x5x10xf16, {order = #NHWC}>,
        sparsity_map=tensor<1x64x11x21xi1>,
        storage_element_table=tensor<1x1x11x21xi32, {order = #NHWC}>,
        #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 11, 21]>>
        -> !VPU.SparseTensor<
            data=tensor<1x64x5x10xf16, {order = #NHWC, mem_space = @CMX_NN}>,
            sparsity_map=tensor<1x64x11x21xi1, {mem_space = @CMX_NN}>,
            storage_element_table=tensor<1x1x11x21xi32, {order = #NHWC, mem_space = @CMX_NN}>,
            #VPU.SEInterpolate<
                mode = <BILINEAR>,
                coordinate_transformation_mode = <ASYMMETRIC>,
                scale = [1.0, 1.0, 2.0, 2.0],
                offsets = [0, 0, 0, 0],
                sizes = [1, 64, 11, 21]>>

    %task = VPU.NCE.Interpolate(%input_cmx, %weights, %weightsTable) {
        rawFilterShape = [64, 64, 2, 2],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        scales_attr = [2, 2],
        ppe = #VPU.PPETask<clamp_high = 2147483647, clamp_low = 0, lrelu_mult = 1, lrelu_shift = 0, mode = <AND>>
    } -> tensor<1x64x10x20xf16, {order = #NHWC, mem_space = @CMX_NN}>
    {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 10, 20] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<VECTOR_FP16>
    }

    return %task : tensor<1x64x10x20xf16, {order = #NHWC, mem_space = @CMX_NN}>

    // CHECK:       [[SPARSITY_MAP1:%.+]] = const.Declare tensor<1x64x11x21xi1> = dense<true> : tensor<1x64x11x21xi1>
    // CHECK:       [[DATA1:%.+]] = builtin.unrealized_conversion_cast %arg0 : memref<1x64x5x10xf16, #NHWC> to tensor<1x64x5x10xf16, {order = #NHWC}>
    // CHECK:       [[STORAGE_ELEMENT1:%.+]] = VPU.StorageElementTable {
    // CHECK-SAME:      dataElemType = i32,
    // CHECK-SAME:      dataShape = [1, 64, 5, 10],
    // CHECK-SAME:      seAttr = #VPU.SEInterpolate<
    // CHECK-SAME:          mode = <BILINEAR>,
    // CHECK-SAME:          coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:          scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:          offsets = [0, 0, 0, 0],
    // CHECK-SAME:          sizes = [1, 64, 11, 21]>,
    // CHECK-SAME:      seDepth = 1 : i64,
    // CHECK-SAME:      seSize = 64 : i64
    // CHECK-SAME:  } -> tensor<1x1x11x21xi32, {order = #NHWC}>

    // CHECK:       [[SPARSE_TENSOR:%.+]] = VPU.GroupSparseTensor([[DATA1]], [[SPARSITY_MAP1]], [[STORAGE_ELEMENT1]]) {
    // CHECK-SAME:      seAttr = #VPU.SEInterpolate<
    // CHECK-SAME:          mode = <BILINEAR>,
    // CHECK-SAME:          coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:          scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:          offsets = [0, 0, 0, 0],
    // CHECK-SAME:          sizes = [1, 64, 11, 21]
    // CHECK-SAME:      >} -> !VPU.SparseTensor<
    // CHECK-SAME:          data=tensor<1x64x5x10xf16, {order = #NHWC}>,
    // CHECK-SAME:          sparsity_map=tensor<1x64x11x21xi1>,
    // CHECK-SAME:          storage_element_table=tensor<1x1x11x21xi32, {order = #NHWC}>,
    // CHECK-SAME:          #VPU.SEInterpolate<
    // CHECK-SAME:              mode = <BILINEAR>,
    // CHECK-SAME:              coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:              scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:              offsets = [0, 0, 0, 0],
    // CHECK-SAME:              sizes = [1, 64, 11, 21]>>

    // CHECK:       [[SPARSE_TENSOR_CMX:%.+]] = VPU.Copy([[SPARSE_TENSOR]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } :
    // CHECK-SAME:      !VPU.SparseTensor<
    // CHECK-SAME:          data=tensor<1x64x5x10xf16, {order = #NHWC}>,
    // CHECK-SAME:          sparsity_map=tensor<1x64x11x21xi1>,
    // CHECK-SAME:          storage_element_table=tensor<1x1x11x21xi32, {order = #NHWC}>,
    // CHECK-SAME:          #VPU.SEInterpolate<
    // CHECK-SAME:              mode = <BILINEAR>,
    // CHECK-SAME:              coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:              scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:              offsets = [0, 0, 0, 0],
    // CHECK-SAME:                  sizes = [1, 64, 11, 21]>>
    // CHECK-SAME:      -> !VPU.SparseTensor<
    // CHECK-SAME:          data=tensor<1x64x5x10xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:          sparsity_map=tensor<1x64x11x21xi1, {mem_space = @CMX_NN}>,
    // CHECK-SAME:          storage_element_table=tensor<1x1x11x21xi32, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:          #VPU.SEInterpolate<
    // CHECK-SAME:              mode = <BILINEAR>,
    // CHECK-SAME:              coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:              scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:              offsets = [0, 0, 0, 0],
    // CHECK-SAME:              sizes = [1, 64, 11, 21]>>

    // CHECK:       [[SPARSE_BUFFER:%.+]] = builtin.unrealized_conversion_cast [[SPARSE_TENSOR_CMX]] :
    // CHECK-SAME:      !VPU.SparseTensor<
    // CHECK-SAME:          data=tensor<1x64x5x10xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:          sparsity_map=tensor<1x64x11x21xi1, {mem_space = @CMX_NN}>,
    // CHECK-SAME:          storage_element_table=tensor<1x1x11x21xi32, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:          #VPU.SEInterpolate<
    // CHECK-SAME:              mode = <BILINEAR>,
    // CHECK-SAME:              coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:              scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:              offsets = [0, 0, 0, 0],
    // CHECK-SAME:              sizes = [1, 64, 11, 21]>>
    // CHECK-SAME:      to !VPUIP.SparseBuffer<
    // CHECK-SAME:          data=memref<1x64x5x10xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:          sparsity_map=memref<1x64x11x21xi1, @CMX_NN>,
    // CHECK-SAME:          storage_element_table=memref<1x1x11x21xi32, #NHWC, @CMX_NN>,
    // CHECK-SAME:          #VPU.SEInterpolate<
    // CHECK-SAME:              mode = <BILINEAR>,
    // CHECK-SAME:              coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:              scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:              offsets = [0, 0, 0, 0],
    // CHECK-SAME:              sizes = [1, 64, 11, 21]>>

    // CHECK:       [[ALLOC:%.+]] = memref.alloc() : memref<1x64x10x20xf16, #NHWC, @CMX_NN>

    // CHECK:       [[DATA2:%.+]], [[SPARSITY_MAP2:%.+]], [[STORAGE_ELEMENT2:%.+]] = VPUIP.UngroupSparseBuffer([[SPARSE_BUFFER]]) {
    // CHECK-SAME:      resultSegmentSizes = array<i32: 1, 1, 1>
    // CHECK-SAME:  } ->
    // CHECK-SAME:      memref<1x64x5x10xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:      memref<1x64x11x21xi1, @CMX_NN>,
    // CHECK-SAME:      memref<1x1x11x21xi32, #NHWC, @CMX_NN>

    // CHECK:       [[TASK_BUFFER:%.+]] = VPUIP.NCEClusterTask {
    // CHECK-SAME:      kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      kernel_size = [2, 2],
    // CHECK-SAME:      kernel_strides = [1, 1],
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME:  }
    // CHECK-SAME:      input([[DATA2]] : memref<1x64x5x10xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      input_sparsity_map([[SPARSITY_MAP2]] : memref<1x64x11x21xi1, @CMX_NN>)
    // CHECK-SAME:      input_storage_element_table([[STORAGE_ELEMENT2]] : memref<1x1x11x21xi32, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weights(%arg1 : memref<64x64x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weight_table(%arg2 : memref<64x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      parent_input([[DATA2]] : memref<1x64x5x10xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      parent_input_sparsity_map([[SPARSITY_MAP2]] : memref<1x64x11x21xi1, @CMX_NN>)
    // CHECK-SAME:      parent_input_storage_element_table([[STORAGE_ELEMENT2]] : memref<1x1x11x21xi32, #NHWC, @CMX_NN>)
    // CHECK-SAME:      parent_output([[ALLOC]] : memref<1x64x10x20xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[ALLOC]] : memref<1x64x10x20xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  -> memref<1x64x10x20xf16, #NHWC, @CMX_NN> variants : {
    // CHECK:           DPUTask {mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [19, 9, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:       } PPE : {
    // CHECK:           PPETask <AND> {clamp_high = 2147483647 : i64, clamp_low = 0 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
    // CHECK:       }

    // CHECK:       [[TASK_TENSOR:%.+]] = builtin.unrealized_conversion_cast [[TASK_BUFFER]] : memref<1x64x10x20xf16, #NHWC, @CMX_NN> to tensor<1x64x10x20xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       return [[TASK_TENSOR]] : tensor<1x64x10x20xf16, {mem_space = @CMX_NN, order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @NCEPermuteOperationSingleTile
// CHECK-SAME:    ([[INPUT:%.+]]: memref<1x3x224x224xf16, @CMX_NN>)
func.func @NCEPermuteOperationSingleTile(%arg0: memref<1x3x224x224xf16, @CMX_NN>) -> memref<1x4x224x224x!qElemType, #NHWC, @CMX_NN> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x3x224x224xf16, @CMX_NN>
        to tensor<1x3x224x224xf16, {mem_space = @CMX_NN}>

    %1 = VPU.NCE.Permute(%0) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 4 : i64,
        minimumHardwareExecutionCost = 4294967300 : i64
    } -> tensor<1x4x224x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 3, 224, 224] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16>
    }

    %2 = builtin.unrealized_conversion_cast %1 : tensor<1x4x224x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>
        to memref<1x4x224x224x!qElemType, #NHWC, @CMX_NN>

    return %2 : memref<1x4x224x224x!qElemType, #NHWC, @CMX_NN>

    // CHECK-NOT:   VPU.NCE.Permute

    // CHECK:       [[VIEW_OP_IN:%.*]] = VPUIP.ViewOp [[INPUT]] : memref<1x3x224x224xf16, @CMX_NN>
    // CHECK-SAME:  to memref<1x224x3x224xf16, #NHWC, @CMX_NN>

    // CHECK:       [[ALLOC:%.*]] = memref.alloc() : memref<1x224x4x224x!qElemType, #NWCH, @CMX_NN>

    // CHECK:       [[NCE_CLUSTER_TASK:%.*]] = VPUIP.NCEClusterTask {
    // CHECK-SAME:      activation_window_channel_length = 0 : i64,
    // CHECK-SAME:      is_permute_quantize,
    // CHECK-SAME:      is_superdense,
    // CHECK-SAME:      minimumHardwareExecutionCost = 4294967300 : i64,
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<ELTWISE>
    // CHECK-SAME:  }
    // CHECK-SAME:  input([[VIEW_OP_IN]] : memref<1x224x3x224xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  weights([[VIEW_OP_IN]] : memref<1x224x3x224xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  parent_input([[VIEW_OP_IN]] : memref<1x224x3x224xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  parent_output([[ALLOC]] : memref<1x224x4x224x!qElemType, #NWCH, @CMX_NN>)
    // CHECK-SAME:  outputs([[ALLOC]] : memref<1x224x4x224x!qElemType, #NWCH, @CMX_NN>)
    // CHECK-SAME:  -> memref<1x224x4x224x!qElemType, #NWCH, @CMX_NN>
    // CHECK-SAME:  variants : {
    // CHECK:           DPUTask {
    // CHECK-SAME:          mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
    // CHECK-SAME:          outEnd = [223, 2, 223],
    // CHECK-SAME:          outStart = [0, 0, 0],
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:           } PPE : {
    // CHECK:           PPETask <ADD> {
    // CHECK-SAME:          clamp_high = 255 : i64,
    // CHECK-SAME:          clamp_low = 0 : i64,
    // CHECK-SAME:          lrelu_mult = 1 : i64,
    // CHECK-SAME:          lrelu_shift = 0 : i64,
    // CHECK-SAME:          quant_scale = [5.000000e-01]}
    // CHECK:       }

    // CHECK:       [[VIEW_OP_OUT:%.*]] = VPUIP.ViewOp [[NCE_CLUSTER_TASK]] : memref<1x224x4x224x!qElemType, #NWCH, @CMX_NN>
    // CHECK-SAME:  to memref<1x4x224x224x!qElemType, #NHWC, @CMX_NN>

    // CHECK:       return [[VIEW_OP_OUT]] : memref<1x4x224x224x!qElemType, #NHWC, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

!CopyOutDistrTensor = !VPU.DistributedTensor<
    1x3x256x224xf16, #NCHW, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [1, 1],
    pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    strides = [2, 1],
    num_clusters = 2 : i64
}>

!OTensorDistributed = !VPU.DistributedTensor<
    1x4x256x224x!qElemType, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [1, 1],
    pads = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
    strides = [2, 1],
    num_clusters = 2,
    equal_memory_and_compute_view
}>

!inType_DDR_tensor = tensor<1x3x256x224xf16>
!inType_DDR_memref = memref<1x3x256x224xf16>
!inType_CMX_tensor = tensor<1x3x256x224xf16, {mem_space = @CMX_NN, order = #NCHW}>
!inType_CMX_memref = memref<1x3x256x224xf16, @CMX_NN>

!outType_DDR_tensor = tensor<1x4x256x224x!qElemType, {order = #NHWC}>
!outType_DDR_memref = memref<1x4x256x224x!qElemType, #NHWC>
!outType_CMX_tensor = tensor<1x4x256x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>
!outType_CMX_memref = memref<1x4x256x224x!qElemType, #NHWC, @CMX_NN>

// CHECK-LABEL: @NCEPermuteOperationMultiTile
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x3x256x224xf16>)
func.func @NCEPermuteOperationMultiTile(%arg0: !inType_DDR_tensor) -> !outType_DDR_tensor {
    %0 = VPU.NCE.ClusterTiling (%arg0 as %arg1: !inType_DDR_tensor) -> !CopyOutDistrTensor {
      %3 = builtin.unrealized_conversion_cast %arg1 : !inType_DDR_tensor to !inType_DDR_memref
      %4 = memref.alloc() : !inType_CMX_memref
      %5 = VPUIP.Copy inputs(%3 : !inType_DDR_memref) outputs(%4 : !inType_CMX_memref) -> !inType_CMX_memref
      %6 = builtin.unrealized_conversion_cast %5 : !inType_CMX_memref to !inType_CMX_tensor
      VPU.Yield %6
    }

    %1 = VPU.NCE.ClusterTiling (%0 as %arg1: tensor<1x3x256x224xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !OTensorDistributed {
      %3 = VPU.NCE.Permute(%arg1) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 4 : i64,
        ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64,
            lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>
    } -> !outType_CMX_tensor {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 3, 256, 224] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16>
    }
      VPU.Yield %3
    }

    %2 = VPU.NCE.ClusterTiling (%1 as %arg1: !outType_CMX_tensor) -> !outType_DDR_tensor {
      %3 = builtin.unrealized_conversion_cast %arg1 : !outType_CMX_tensor to !outType_CMX_memref
      %4 = memref.alloc() : !outType_DDR_memref
      %5 = VPUIP.Copy inputs(%3 : !outType_CMX_memref) outputs(%4 : !outType_DDR_memref) -> !outType_DDR_memref
      %6 = builtin.unrealized_conversion_cast %5 : !outType_DDR_memref to !outType_DDR_tensor
      VPU.Yield %6
    }

    return %2 : tensor<1x4x256x224x!qElemType, {order = #NHWC}>

   // CHECK-NOT:   VPU.NCE.Permute

    // CHECK:       [[COPY_IN:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as %arg1: tensor<1x3x256x224xf16>)
    // CHECK-SAME:    -> !VPU.DistributedTensor<1x3x256x224xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:    {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [1, 1],
    // CHECK-SAME:    pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:    strides = [2, 1], num_clusters = 2 : i64}> {
    // CHECK:       }

    // CHECK:       [[UNREALIZE_CAST_INPUT_VIEW_OP_IN:%.*]] = builtin.unrealized_conversion_cast [[COPY_IN]] :
    // CHECK-SAME:    !VPU.DistributedTensor<1x3x256x224xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:    {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [1, 1],
    // CHECK-SAME:    pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:    strides = [2, 1], num_clusters = 2 : i64}>
    // CHECK-SAME:    to
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x3x256x224xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:    {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [1, 1],
    // CHECK-SAME:    pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:    strides = [2, 1], num_clusters = 2 : i64}>


    // CHECK:       [[VIEW_OP_IN:%.*]] = VPUIP.ViewOp [[UNREALIZE_CAST_INPUT_VIEW_OP_IN]] :
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x3x256x224xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:    {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [1, 1],
    // CHECK-SAME:    pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:    strides = [2, 1], num_clusters = 2 : i64}>
    // CHECK-SAME:    to
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x224x3x256xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:    {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [1, 1],
    // CHECK-SAME:    pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:    strides = [1, 2], num_clusters = 2 : i64}>

    // CHECK:       [[UNREALIZE_CAST_OUTPUT_VIEW_OP_IN:%.*]] = builtin.unrealized_conversion_cast [[VIEW_OP_IN]] :
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x224x3x256xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:    {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [1, 1],
    // CHECK-SAME:    pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:    strides = [1, 2], num_clusters = 2 : i64}>
    // CHECK-SAME:    to
    // CHECK-SAME:    !VPU.DistributedTensor<1x224x3x256xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:    {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [1, 1],
    // CHECK-SAME:    pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:    strides = [1, 2], num_clusters = 2 : i64}>

    // CHECK:       [[NCEPermute:%.*]] = VPU.NCE.ClusterTiling ([[UNREALIZE_CAST_OUTPUT_VIEW_OP_IN]] as [[INNER_ARG:[^:]+]]: tensor<1x224x3x256xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:    -> !VPU.DistributedTensor<1x224x4x256x!qElemType, #NWCH, @CMX_NN,
    // CHECK-SAME:    {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [1, 1],
    // CHECK-SAME:    pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:    strides = [1, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}> {

    // CHECK:             [[INNER_IN_CAST:%.*]] = builtin.unrealized_conversion_cast [[INNER_ARG]] :
    // CHECK-SAME:            tensor<1x224x3x256xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:            to memref<1x224x3x256xf16, #NHWC, @CMX_NN>

    // CHECK:             [[ALLOC:%.*]] = memref.alloc() : memref<1x224x4x256x!qElemType, #NWCH, @CMX_NN>

    // CHECK:             [[CLUSTER_TASK:%.*]] = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i64,
    // CHECK-SAME:            is_permute_quantize, is_superdense, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:            input([[INNER_IN_CAST]] : memref<1x224x3x256xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:            weights([[INNER_IN_CAST]] : memref<1x224x3x256xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:            parent_input([[INNER_IN_CAST]] : memref<1x224x3x256xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:            parent_output([[ALLOC]] : memref<1x224x4x256x!qElemType, #NWCH, @CMX_NN>)
    // CHECK-SAME:            outputs([[ALLOC]] : memref<1x224x4x256x!qElemType, #NWCH, @CMX_NN>)
    // CHECK-SAME:            -> memref<1x224x4x256x!qElemType, #NWCH, @CMX_NN> variants : {
    // CHECK:                   DPUTask {mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [255, 2, 223], outStart = [0, 0, 0],
    // CHECK-SAME:                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:                   } PPE : {
    // CHECK:                   PPETask <ADD> {clamp_high = 255 : i64, clamp_low = 0 : i64, lrelu_mult = 1 : i64,
    // CHECK-SAME:                lrelu_shift = 0 : i64, quant_scale = [5.000000e-01]}
    // CHECK:                 }

    // CHECK:             [[INNER_OUT_CAST:%.*]] = builtin.unrealized_conversion_cast [[CLUSTER_TASK]] :
    // CHECK-SAME:            memref<1x224x4x256x!qElemType, #NWCH, @CMX_NN>
    // CHECK-SAME:            to tensor<1x224x4x256x!qElemType, {mem_space = @CMX_NN, order = #NWCH}>

    // CHECK:             VPU.Yield [[INNER_OUT_CAST]]
    // CHECK:        }

    // CHECK:       [[UNREALIZE_CAST_INPUT_VIEW_OP_OUT:%.*]] = builtin.unrealized_conversion_cast [[NCEPermute]] :
    // CHECK-SAME:    !VPU.DistributedTensor<1x224x4x256x!qElemType, #NWCH, @CMX_NN,
    // CHECK-SAME:    {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [1, 1],
    // CHECK-SAME:    pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:    strides = [1, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>
    // CHECK-SAME:    to
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x224x4x256x!qElemType, #NWCH, @CMX_NN,
    // CHECK-SAME:    {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [1, 1],
    // CHECK-SAME:    pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:    strides = [1, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>

    // CHECK:       [[VIEW_OP_OUT:%.*]] = VPUIP.ViewOp [[UNREALIZE_CAST_INPUT_VIEW_OP_OUT]] :
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x224x4x256x!qElemType, #NWCH, @CMX_NN,
    // CHECK-SAME:    {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [1, 1],
    // CHECK-SAME:    pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:    strides = [1, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>
    // CHECK-SAME:    to
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x4x256x224x!qElemType, #NHWC, @CMX_NN,
    // CHECK-SAME:    {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [1, 1],
    // CHECK-SAME:    pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:    strides = [2, 1], num_clusters = 2 : i64, equal_memory_and_compute_view}>

    // CHECK:       [[UNREALIZE_CAST_OUTPUT_VIEW_OP_OUT:%.*]] = builtin.unrealized_conversion_cast [[VIEW_OP_OUT]] :
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x4x256x224x!qElemType, #NHWC, @CMX_NN,
    // CHECK-SAME:    {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [1, 1],
    // CHECK-SAME:    pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:    strides = [2, 1], num_clusters = 2 : i64, equal_memory_and_compute_view}>
    // CHECK-SAME:    to
    // CHECK-SAME:    !VPU.DistributedTensor<1x4x256x224x!qElemType, #NHWC, @CMX_NN,
    // CHECK-SAME:    {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [1, 1],
    // CHECK-SAME:    pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:    strides = [2, 1], num_clusters = 2 : i64, equal_memory_and_compute_view}>

    // CHECK:       [[COPY_OUT:%.*]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:    tensor<1x4x256x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:    -> tensor<1x4x256x224x!qElemType, {order = #NHWC}> {
    // CHECK:       }

    // CHECK:       return [[COPY_OUT]] : tensor<1x4x256x224x!qElemType, {order = #NHWC}>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

!CopyOutDistrTensor = !VPU.DistributedTensor<
    1x3x224x224xf16, #NCHW, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 3, 112, 224], [1, 3, 112, 224]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
    memory_shapes = [[1, 3, 112, 224], [1, 3, 112, 224]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]]
}>

!OTensorDistributed = !VPU.DistributedTensor<
    1x4x224x224x!qElemType, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2,
    uniform_distributed_segments,
    compute_shapes = [[1, 4, 112, 224], [1, 4, 112, 224]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
    memory_shapes = [[1, 4, 112, 224], [1, 4, 112, 224]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]]
}>

!inType_DDR_tensor = tensor<1x3x224x224xf16>
!inType_DDR_memref = memref<1x3x224x224xf16>
!inType_CMX_tensor = tensor<1x3x224x224xf16, {mem_space = @CMX_NN, order = #NCHW}>
!inType_CMX_memref = memref<1x3x224x224xf16, @CMX_NN>

!outType_DDR_tensor = tensor<1x4x224x224x!qElemType, {order = #NHWC}>
!outType_DDR_memref = memref<1x4x224x224x!qElemType, #NHWC>
!outType_CMX_tensor = tensor<1x4x224x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>
!outType_CMX_memref = memref<1x4x224x224x!qElemType, #NHWC, @CMX_NN>

// CHECK-LABEL: @NCEPermuteExplicitDistr
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x3x224x224xf16>)
func.func @NCEPermuteExplicitDistr(%arg0: !inType_DDR_tensor) -> !outType_DDR_tensor {
    %0 = VPU.NCE.ClusterTiling (%arg0 as %arg1: !inType_DDR_tensor) -> !CopyOutDistrTensor {
      %3 = builtin.unrealized_conversion_cast %arg1 : !inType_DDR_tensor to !inType_DDR_memref
      %4 = memref.alloc() : !inType_CMX_memref
      %5 = VPUIP.Copy inputs(%3 : !inType_DDR_memref) outputs(%4 : !inType_CMX_memref) -> !inType_CMX_memref
      %6 = builtin.unrealized_conversion_cast %5 : !inType_CMX_memref to !inType_CMX_tensor
      VPU.Yield %6
    }

    %1 = VPU.NCE.ClusterTiling (%0 as %arg1: !inType_CMX_tensor) -> !OTensorDistributed {
      %3 = VPU.NCE.Permute(%arg1) {
          dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4 : i64,
          ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64,
                clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
                fp_prelu_alpha = 1.000000e+00 : f64>
      } -> !outType_CMX_tensor {
          VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 3, 224, 224]
          <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16>
      }
      VPU.Yield %3
    }

    %2 = VPU.NCE.ClusterTiling (%1 as %arg1: !outType_CMX_tensor) -> !outType_DDR_tensor {
      %3 = builtin.unrealized_conversion_cast %arg1 : !outType_CMX_tensor to !outType_CMX_memref
      %4 = memref.alloc() : !outType_DDR_memref
      %5 = VPUIP.Copy inputs(%3 : !outType_CMX_memref) outputs(%4 : !outType_DDR_memref) -> !outType_DDR_memref
      %6 = builtin.unrealized_conversion_cast %5 : !outType_DDR_memref to !outType_DDR_tensor
      VPU.Yield %6
    }
    return %2 : !outType_DDR_tensor

   // CHECK-NOT:   VPU.NCE.Permute

    // CHECK:       [[COPY_IN:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as %arg1: tensor<1x3x224x224xf16>)
    // CHECK-SAME:    !VPU.DistributedTensor<1x3x224x224xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:    {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME:    uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 3, 112, 224], [1, 3, 112, 224]],
    // CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 3, 112, 224], [1, 3, 112, 224]],
    // CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]]}> {
    // CHECK:       }

    // CHECK:       [[UNREALIZE_CAST_INPUT_VIEW_OP_IN:%.*]] = builtin.unrealized_conversion_cast [[COPY_IN]] :
    // CHECK-SAME:    !VPU.DistributedTensor<1x3x224x224xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:    {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME:    uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 3, 112, 224], [1, 3, 112, 224]],
    // CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 3, 112, 224], [1, 3, 112, 224]],
    // CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]]}>
    // CHECK-SAME:    to
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x3x224x224xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:    {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME:    uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 3, 112, 224], [1, 3, 112, 224]],
    // CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 3, 112, 224], [1, 3, 112, 224]],
    // CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]]}>

    // CHECK:       [[VIEW_OP_IN:%.*]] = VPUIP.ViewOp [[UNREALIZE_CAST_INPUT_VIEW_OP_IN]] :
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x3x224x224xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:    {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME:    uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 3, 112, 224], [1, 3, 112, 224]],
    // CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 3, 112, 224], [1, 3, 112, 224]],
    // CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]]}>
    // CHECK-SAME:    to
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x224x3x224xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:    {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64,
    // CHECK-SAME:    uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 224, 3, 112], [1, 224, 3, 112]],
    // CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 112]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 224, 3, 112], [1, 224, 3, 112]],
    // CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 112]]}>

    // CHECK:       [[UNREALIZE_CAST_OUTPUT_VIEW_OP_IN:%.*]] = builtin.unrealized_conversion_cast [[VIEW_OP_IN]] :
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x224x3x224xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:    {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64,
    // CHECK-SAME:    uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 224, 3, 112], [1, 224, 3, 112]],
    // CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 112]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 224, 3, 112], [1, 224, 3, 112]],
    // CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 112]]}>
    // CHECK-SAME:    to
    // CHECK-SAME:    !VPU.DistributedTensor<1x224x3x224xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:    {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64,
    // CHECK-SAME:    uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 224, 3, 112], [1, 224, 3, 112]],
    // CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 112]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 224, 3, 112], [1, 224, 3, 112]],
    // CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 112]]}>

    // CHECK:       [[NCEPermute:%.*]] = VPU.NCE.ClusterTiling ([[UNREALIZE_CAST_OUTPUT_VIEW_OP_IN]] as [[INNER_ARG:[^:]+]]: tensor<1x224x3x224xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:    -> !VPU.DistributedTensor<1x224x4x224x!qElemType, #NWCH, @CMX_NN,
    // CHECK-SAME:    {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64,
    // CHECK-SAME:    uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 224, 4, 112], [1, 224, 4, 112]],
    // CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 112]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 224, 4, 112], [1, 224, 4, 112]],
    // CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 112]]}> {

    // CHECK:             [[INNER_IN_CAST:%.*]] = builtin.unrealized_conversion_cast [[INNER_ARG]] :
    // CHECK-SAME:            tensor<1x224x3x224xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:            to memref<1x224x3x224xf16, #NHWC, @CMX_NN>

    // CHECK:             [[ALLOC:%.*]] = memref.alloc() : memref<1x224x4x224x!qElemType, #NWCH, @CMX_NN>

    // CHECK:             [[CLUSTER_TASK:%.*]] = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i64,
    // CHECK-SAME:            is_permute_quantize, is_superdense, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:            input([[INNER_IN_CAST]] : memref<1x224x3x224xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:            weights([[INNER_IN_CAST]] : memref<1x224x3x224xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:            parent_input([[INNER_IN_CAST]] : memref<1x224x3x224xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:            parent_output([[ALLOC]] : memref<1x224x4x224x!qElemType, #NWCH, @CMX_NN>)
    // CHECK-SAME:            outputs([[ALLOC]] : memref<1x224x4x224x!qElemType, #NWCH, @CMX_NN>)
    // CHECK-SAME:            -> memref<1x224x4x224x!qElemType, #NWCH, @CMX_NN> variants : {
    // CHECK:                   DPUTask {mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [223, 2, 223], outStart = [0, 0, 0],
    // CHECK-SAME:                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:                   } PPE : {
    // CHECK:                   PPETask <ADD> {clamp_high = 255 : i64, clamp_low = 0 : i64, lrelu_mult = 1 : i64,
    // CHECK-SAME:                lrelu_shift = 0 : i64, quant_scale = [5.000000e-01]}
    // CHECK:                 }

    // CHECK:             [[INNER_OUT_CAST:%.*]] = builtin.unrealized_conversion_cast [[CLUSTER_TASK]] :
    // CHECK-SAME:            memref<1x224x4x224x!qElemType, #NWCH, @CMX_NN>
    // CHECK-SAME:            to tensor<1x224x4x224x!qElemType, {mem_space = @CMX_NN, order = #NWCH}>

    // CHECK:             VPU.Yield [[INNER_OUT_CAST]]
    // CHECK:        }

    // CHECK:       [[UNREALIZE_CAST_INPUT_VIEW_OP_OUT:%.*]] = builtin.unrealized_conversion_cast [[NCEPermute]] :
    // CHECK-SAME:    !VPU.DistributedTensor<1x224x4x224x!qElemType, #NWCH, @CMX_NN,
    // CHECK-SAME:    {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64,
    // CHECK-SAME:    uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 224, 4, 112], [1, 224, 4, 112]],
    // CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 112]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 224, 4, 112], [1, 224, 4, 112]],
    // CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 112]]}>
    // CHECK-SAME:    to
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x224x4x224x!qElemType, #NWCH, @CMX_NN,
    // CHECK-SAME:    {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64,
    // CHECK-SAME:    uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 224, 4, 112], [1, 224, 4, 112]],
    // CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 112]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 224, 4, 112], [1, 224, 4, 112]],
    // CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 112]]}>

    // CHECK:       [[VIEW_OP_OUT:%.*]] = VPUIP.ViewOp [[UNREALIZE_CAST_INPUT_VIEW_OP_OUT]] :
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x224x4x224x!qElemType, #NWCH, @CMX_NN,
    // CHECK-SAME:    {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64,
    // CHECK-SAME:    uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 224, 4, 112], [1, 224, 4, 112]],
    // CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 112]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 224, 4, 112], [1, 224, 4, 112]],
    // CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 112]]}>
    // CHECK-SAME:    to
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x4x224x224x!qElemType, #NHWC, @CMX_NN,
    // CHECK-SAME:    {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME:    uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 4, 112, 224], [1, 4, 112, 224]],
    // CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 4, 112, 224], [1, 4, 112, 224]],
    // CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]]}>

    // CHECK:       [[UNREALIZE_CAST_OUTPUT_VIEW_OP_OUT:%.*]] = builtin.unrealized_conversion_cast [[VIEW_OP_OUT]] :
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x4x224x224x!qElemType, #NHWC, @CMX_NN,
    // CHECK-SAME:    {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME:    uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 4, 112, 224], [1, 4, 112, 224]],
    // CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 4, 112, 224], [1, 4, 112, 224]],
    // CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]]}>
    // CHECK-SAME:    to
    // CHECK-SAME:    !VPU.DistributedTensor<1x4x224x224x!qElemType, #NHWC, @CMX_NN,
    // CHECK-SAME:    {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    // CHECK-SAME:    uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 4, 112, 224], [1, 4, 112, 224]],
    // CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 4, 112, 224], [1, 4, 112, 224]],
    // CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]]}>

    // CHECK:       [[COPY_OUT:%.*]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:    tensor<1x4x224x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:    -> tensor<1x4x224x224x!qElemType, {order = #NHWC}> {
    // CHECK:       }

    // CHECK:       return [[COPY_OUT]] : tensor<1x4x224x224x!qElemType, {order = #NHWC}>

}

