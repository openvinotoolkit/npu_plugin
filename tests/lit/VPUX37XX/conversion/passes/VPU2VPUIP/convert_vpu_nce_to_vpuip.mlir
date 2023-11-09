//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX allow-custom-values=true" --convert-vpu-nce-to-vpuip --canonicalize %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

module @PermuteQuantize attributes {VPU.arch = #VPU.arch_kind<VPUX37XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
  IE.ExecutorResource 2 of @NCE {
      IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
  }

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
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 3, 1568] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 13 : i64> #VPU.mpe_mode<CUBOID_16x16>
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
    // CHECK-SAME:  parent_output(%0 : memref<1x32x4x1568x!qElemType, #NWCH, @CMX_NN>)
    // CHECK-SAME:  outputs(%0 : memref<1x32x4x1568x!qElemType, #NWCH, @CMX_NN>)
    // CHECK-SAME:  -> memref<1x32x4x1568x!qElemType, #NWCH, @CMX_NN>
    // CHECK-SAME:  variants : {
    // CHECK:           DPUTask {
    // CHECK-SAME:          <CUBOID_16x16>,
    // CHECK-SAME:          outEnd = [1567, 2, 31],
    // CHECK-SAME:          outStart = [0, 0, 0],
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 13 : i64>
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

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @SuperdenseConv attributes {VPU.arch = #VPU.arch_kind<VPUX37XX>} {
// CHECK-LABEL: @NCEConvolution
func.func @NCEConvolution(%arg0: memref<1x16x15x15xf16, #NHWC, @CMX_NN>,
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

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @SuperdenseMaxPool attributes {VPU.arch = #VPU.arch_kind<VPUX37XX>} {
// CHECK-LABEL: @NCEMaxPool
func.func @NCEMaxPool(%arg0: memref<1x16x15x15xf16, #NHWC, @CMX_NN>,
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

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @SuperdenseAveragePool attributes {VPU.arch = #VPU.arch_kind<VPUX37XX>} {
// CHECK-LABEL: @NCEAveragePool
func.func @NCEAveragePool(%arg0: memref<1x16x15x15xf16, #NHWC, @CMX_NN>) -> memref<1x16x15x15xf16, #NCHW, @CMX_NN> {
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

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @SuperdenseEltwise attributes {VPU.arch = #VPU.arch_kind<VPUX37XX>} {
// CHECK-LABEL: @NCEEltwise
func.func @NCEEltwise(%arg0: memref<1x16x15x15xf16, #NHWC, @CMX_NN>,
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
    // CHECK-SAME:      result_segment_sizes = dense<1> : vector<3xi32>
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
    // CHECK-SAME:      result_segment_sizes = dense<1> : vector<3xi32>
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
