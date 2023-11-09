//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX allow-custom-values=true" --wrap-vpu-ops-in-ncecluster-tiling %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

module @PermuteQuantize attributes {VPU.arch = #VPU.arch_kind<VPUX37XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
  IE.ExecutorResource 2 of @NCE at 1.300000e+03 MHz {
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @SHAVE_NN
    IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 1 of @DPU
  }
  IE.ExecutorResource 2 of @DMA_NN
  IE.MemoryResource 524288000 bytes of @DDR {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64}

func.func @NCEPermuteQuantize3x224x224(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x4x224x224x!qElemType, {order = #NHWC}> {
    %0 = VPU.Reshape(%arg0) {
        shape_value = [1, 224, 3, 224]
    } : tensor<1x3x224x224xf16> -> tensor<1x224x3x224xf16>

    %1 = VPU.LayoutCast(%0) {
        dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    } : tensor<1x224x3x224xf16> -> tensor<1x224x3x224xf16, {order = #NHWC}>

    %2 = VPU.NCE.PermuteQuantize(%1) {
        dstElemType = !qElemType,
        dstOrder = #NWCH,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPETask<
            clamp_high = 255 : i64,
            clamp_low = 0 : i64,
            fp_prelu_alpha = 1.000000e+00 : f64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = <NOOP>
        >
    } -> tensor<1x224x4x224x!qElemType, {order = #NWCH}>

    %3 = VPU.LayoutCast(%2) {
        dst_order = #NHWC
    } : tensor<1x224x4x224x!qElemType, {order = #NWCH}> -> tensor<1x224x4x224x!qElemType, {order = #NHWC}>

    %4 = VPU.AffineReshape(%3) {
        dim_mapping = [[0], [1], [2], [3]],
        shape_value = [1, 4, 224, 224]
    } : tensor<1x224x4x224x!qElemType, {order = #NHWC}> -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    return %4 : tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    // CHECK:   [[COPY_INPUT:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x3x224x224xf16>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x3x224x224xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }

    // CHECK:   VPU.Copy(%arg1) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x3x224x224xf16> -> tensor<1x3x224x224xf16, {mem_space = @CMX_NN, order = #NCHW}>

    // CHECK: [[CAST_INPUT:%.*]] = VPU.WorkloadCast([[COPY_INPUT]]
    // CHECK-SAME:  : !VPU.DistributedTensor<1x3x224x224xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x224x3x224xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 1, 2],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }

    // CHECK:   [[WORKLOAD:%.*]] = VPU.NCE.ClusterTiling ([[CAST_INPUT]]
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x224x4x224x!qElemType, #NWCH, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 1, 2],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }

    // CHECK:   VPU.NCE.PermuteQuantize
    // CHECK-SAME:  tensor<1x224x4x224x!qElemType, {mem_space = @CMX_NN, order = #NWCH}>

    // CHECK: [[CAST_OUTPUT:%.*]] = VPU.WorkloadCast([[WORKLOAD]]
    // CHECK-SAME:  : !VPU.DistributedTensor<1x224x4x224x!qElemType, #NWCH, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 1, 2],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x4x224x224x!qElemType, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }

    // CHECK:   [[COPY_OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[CAST_OUTPUT]]
    // CHECK-SAME:  [[COPY_ARG:%.*]]: tensor<1x4x224x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:  -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    // CHECK:   VPU.Copy([[COPY_ARG]]) : tensor<1x4x224x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:  -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

module @DepthwiseConv attributes {VPU.arch = #VPU.arch_kind<VPUX37XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
  IE.ExecutorResource 2 of @NCE at 1.300000e+03 MHz {
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @SHAVE_NN
    IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 1 of @DPU
  }
  IE.ExecutorResource 2 of @DMA_NN
  IE.MemoryResource 524288000 bytes of @DDR {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64}

func.func @DWCONV3x224x224(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x16x224x224x!qElemType, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>
    %WEIGHTS = const.Declare tensor<16x16x1x1x!qElemType, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [
            #const.ConvertElemType<ui8>,
            #const.QuantCast<!qElemType>,
            #const.Reorder<#NHWC>
        ]

    %WEIGHT_TABLE = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %0 = VPU.Reshape(%arg0) {
        shape_value = [1, 224, 3, 224]
    } : tensor<1x3x224x224xf16> -> tensor<1x224x3x224xf16>

    %1 = VPU.LayoutCast(%0) {
        dst_order = #NHWC
    } : tensor<1x224x3x224xf16> -> tensor<1x224x3x224xf16, {order = #NHWC}>

    %2 = VPU.NCE.PermuteQuantize(%1) {
        dstElemType = !qElemType,
        dstOrder = #NWCH,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 13 : i64>,
        ppe = #VPU.PPETask<
            clamp_high = 255 : i64,
            clamp_low = 0 : i64,
            fp_prelu_alpha = 1.000000e+00 : f64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = <NOOP>
        >
    } -> tensor<1x224x16x224x!qElemType, {order = #NWCH}>

    %3 = VPU.LayoutCast(%2) {
        dst_order = #NHWC
    } : tensor<1x224x16x224x!qElemType, {order = #NWCH}> -> tensor<1x224x16x224x!qElemType, {order = #NHWC}>

    %4 = VPU.AffineReshape(%3) {
        dim_mapping = [[0], [1], [2], [3]],
        shape_value = [1, 16, 224, 224]
    } : tensor<1x224x16x224x!qElemType, {order = #NHWC}> -> tensor<1x16x224x224x!qElemType, {order = #NHWC}>

    %5 = VPU.NCE.DepthConvolution(%4, %WEIGHTS, %WEIGHT_TABLE, %cst) {
        activation_window_channel_length = 16 : i64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPETask<
            clamp_high = 255 : i64,
            clamp_low = 0 : i64,
            fp_prelu_alpha = 1.000000e+00 : f64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = <NOOP>
        >,
        rawFilterShape = [16, 1, 1, 1],
        strides = [1, 1]
    } -> tensor<1x16x224x224x!qElemType, {order = #NHWC}>

    return %5 : tensor<1x16x224x224x!qElemType, {order = #NHWC}>

    // CHECK:   [[COPY_INPUT:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x3x224x224xf16>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x3x224x224xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }

    // CHECK:   VPU.Copy(%arg1) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x3x224x224xf16> -> tensor<1x3x224x224xf16, {mem_space = @CMX_NN, order = #NCHW}>

    // CHECK: [[CAST_INPUT:%.*]] = VPU.WorkloadCast([[COPY_INPUT]]
    // CHECK-SAME:  : !VPU.DistributedTensor<1x3x224x224xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x224x3x224xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 1, 2],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }

    // CHECK:   [[WORKLOAD:%.*]] = VPU.NCE.ClusterTiling ([[CAST_INPUT]]
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x224x16x224x!qElemType, #NWCH, @CMX_NN,  {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 1, 2],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }

    // CHECK:   VPU.NCE.PermuteQuantize
    // CHECK-SAME:  tensor<1x224x16x224x!qElemType, {mem_space = @CMX_NN, order = #NWCH}>

    // CHECK: [[CAST_OUTPUT:%.*]] = VPU.WorkloadCast([[WORKLOAD]]
    // CHECK-SAME:  : !VPU.DistributedTensor<1x224x16x224x!qElemType, #NWCH, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 1, 2],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x16x224x224x!qElemType, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }

    // CHECK:   [[COPY_OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[CAST_OUTPUT]]
    // CHECK-SAME:  [[COPY_ARG:%.*]]: tensor<1x16x224x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:  -> tensor<1x16x224x224x!qElemType, {order = #NHWC}>

    // CHECK:   VPU.Copy([[COPY_ARG]]) : tensor<1x16x224x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:  -> tensor<1x16x224x224x!qElemType, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

module @CompressConv attributes {VPU.arch = #VPU.arch_kind<VPUX37XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
  IE.ExecutorResource 2 of @NCE at 1.300000e+03 MHz {
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @SHAVE_NN
    IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 1 of @DPU
  }
  IE.ExecutorResource 2 of @DMA_NN
  IE.MemoryResource 524288000 bytes of @DDR {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64}

func.func @CONV3x224x224(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x16x112x112xf16, {order = #NHWC}> {
    %WEIGHTS = const.Declare tensor<16x1x1x48x!qElemType, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x1x1x48xf16>, [
            #const.ConvertElemType<ui8>,
            #const.QuantCast<!qElemType>,
            #const.Reorder<#NHWC>
        ]

    %WEIGHT_TABLE = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %0 = VPU.Reshape(%arg0) {
        shape_value = [1, 224, 3, 224]
    } : tensor<1x3x224x224xf16> -> tensor<1x224x3x224xf16>

    %1 = VPU.LayoutCast(%0) {
        dst_order = #NHWC
    } : tensor<1x224x3x224xf16> -> tensor<1x224x3x224xf16, {order = #NHWC}>

    %2 = VPU.NCE.PermuteQuantize(%1) {
        dstElemType = !qElemType,
        dstOrder = #NWCH,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPETask<
            clamp_high = 255 : i64,
            clamp_low = 0 : i64,
            fp_prelu_alpha = 1.000000e+00 : f64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = <NOOP>
        >
    } -> tensor<1x224x4x224x!qElemType, {order = #NWCH}>

    %3 = VPU.LayoutCast(%2) {
        dst_order = #NHWC
    } : tensor<1x224x4x224x!qElemType, {order = #NWCH}> -> tensor<1x224x4x224x!qElemType, {order = #NHWC}>

    %4 = VPU.AffineReshape(%3) {
        dim_mapping = [[0], [1], [2], [3]],
        shape_value = [1, 4, 224, 224]
    } : tensor<1x224x4x224x!qElemType, {order = #NHWC}> -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    %5 = VPU.NCE.CompressConvolution(%4, %WEIGHTS, %WEIGHT_TABLE) {
        cm_sp_pattern = 7 : i64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPETask<
            clamp_high = 2147483647 : i64,
            clamp_low = -2147483648 : i64,
            fp_prelu_alpha = 1.000000e+00 : f64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = <NOOP>
        >,
        rawFilterShape = [16, 4, 3, 3],
        strides = [2, 2]
    } -> tensor<1x16x112x112xf16, {order = #NHWC}>

    return %5 : tensor<1x16x112x112xf16, {order = #NHWC}>

    // CHECK:   [[COPY_INPUT:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x3x224x224xf16>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x3x224x224xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:      mode = "OVERLAPPED",
    // CHECK-SAME:      num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }

    // CHECK:   VPU.Copy(%arg1) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x3x224x224xf16> -> tensor<1x3x224x224xf16, {mem_space = @CMX_NN, order = #NCHW}>

    // CHECK: [[CAST_INPUT:%.*]] = VPU.WorkloadCast([[COPY_INPUT]]
    // CHECK-SAME:  : !VPU.DistributedTensor<1x3x224x224xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:      mode = "OVERLAPPED",
    // CHECK-SAME:      num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x224x3x224xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "OVERLAPPED",
    // CHECK-SAME:      num_tiles = [1, 1, 1, 2],
    // CHECK-SAME:      kernel = [3, 3],
    // CHECK-SAME:      pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:      strides = [2, 2],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-NOT:      equal_memory_and_compute_view
    // CHECK-SAME:  }

    // CHECK:   [[WORKLOAD:%.*]] = VPU.NCE.ClusterTiling ([[CAST_INPUT]]
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x224x4x224x!qElemType, #NWCH, @CMX_NN,  {
    // CHECK-SAME:      mode = "OVERLAPPED",
    // CHECK-SAME:      num_tiles = [1, 1, 1, 2],
    // CHECK-SAME:      kernel = [3, 3],
    // CHECK-SAME:      pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:      strides = [2, 2],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:      equal_memory_and_compute_view
    // CHECK-SAME:  }

    // CHECK:   VPU.NCE.PermuteQuantize
    // CHECK-SAME:  tensor<1x224x4x224x!qElemType, {mem_space = @CMX_NN, order = #NWCH}>

    // CHECK: [[CAST_OUTPUT:%.*]] = VPU.WorkloadCast([[WORKLOAD]]
    // CHECK-SAME:  : !VPU.DistributedTensor<1x224x4x224x!qElemType, #NWCH, @CMX_NN, {
    // CHECK-SAME:      mode = "OVERLAPPED",
    // CHECK-SAME:      num_tiles = [1, 1, 1, 2],
    // CHECK-SAME:      kernel = [3, 3],
    // CHECK-SAME:      pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:      strides = [2, 2],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:      equal_memory_and_compute_view
    // CHECK-SAME:  }
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x4x224x224x!qElemType, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "OVERLAPPED",
    // CHECK-SAME:      num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:      kernel = [3, 3],
    // CHECK-SAME:      pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:      strides = [2, 2],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-NOT:       equal_memory_and_compute_view
    // CHECK-SAME:  }

    // CHECK:   [[COPY_OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[CAST_OUTPUT]]
    // CHECK-SAME:  [[COPY_ARG:%.*]]: tensor<1x4x224x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:  -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    // CHECK:   VPU.Copy([[COPY_ARG]]) : tensor<1x4x224x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:  -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

module @NoAffineReshape attributes {VPU.arch = #VPU.arch_kind<VPUX37XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
  IE.ExecutorResource 2 of @NCE at 1.300000e+03 MHz {
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @SHAVE_NN
    IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 1 of @DPU
  }
  IE.ExecutorResource 2 of @DMA_NN
  IE.MemoryResource 524288000 bytes of @DDR {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64}

func.func @NCEPermuteQuantize1x3x16x16(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x16x16x16x!qElemType, {order = #NHWC}> {
    %0 = VPU.Reshape(%arg0) {
        shape_value = [1, 16, 3, 16]
    } : tensor<1x3x16x16xf16> -> tensor<1x16x3x16xf16>

    %1 = VPU.LayoutCast(%0) {
        dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    } : tensor<1x16x3x16xf16> -> tensor<1x16x3x16xf16, {order = #NHWC}>

    %2 = VPU.NCE.PermuteQuantize(%1) {
        dstElemType = !qElemType,
        dstOrder = #NWCH,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 13 : i64>,
        ppe = #VPU.PPETask<
            clamp_high = 255 : i64,
            clamp_low = 0 : i64,
            fp_prelu_alpha = 1.000000e+00 : f64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = <NOOP>
        >
    } -> tensor<1x16x16x16x!qElemType, {order = #NWCH}>

    %3 = VPU.LayoutCast(%2) {
        dst_order = #NHWC
    } : tensor<1x16x16x16x!qElemType, {order = #NWCH}> -> tensor<1x16x16x16x!qElemType, {order = #NHWC}>

    return %3 : tensor<1x16x16x16x!qElemType, {order = #NHWC}>

    // CHECK:   [[COPY_INPUT:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x3x16x16xf16>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x3x16x16xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }

    // CHECK:   VPU.Copy(%arg1) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16, {mem_space = @CMX_NN, order = #NCHW}>

    // CHECK: [[CAST_INPUT:%.*]] = VPU.WorkloadCast([[COPY_INPUT]]
    // CHECK-SAME:  : !VPU.DistributedTensor<1x3x16x16xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x16x3x16xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 1, 2],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }

    // CHECK:   [[WORKLOAD:%.*]] = VPU.NCE.ClusterTiling ([[CAST_INPUT]]
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x16x16x16x!qElemType, #NWCH, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 1, 2],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }

    // CHECK:   VPU.NCE.PermuteQuantize
    // CHECK-SAME:  tensor<1x16x16x16x!qElemType, {mem_space = @CMX_NN, order = #NWCH}>

    // CHECK: [[CAST_OUTPUT:%.*]] = VPU.WorkloadCast([[WORKLOAD]]
    // CHECK-SAME:  : !VPU.DistributedTensor<1x16x16x16x!qElemType, #NWCH, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 1, 2],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x16x16x16x!qElemType, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:      num_clusters = 2 : i64
    // CHECK-SAME:  }

    // CHECK:   [[COPY_OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[CAST_OUTPUT]]
    // CHECK-SAME:  [[COPY_ARG:%.*]]: tensor<1x16x16x16x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:  -> tensor<1x16x16x16x!qElemType, {order = #NHWC}>

    // CHECK:   VPU.Copy([[COPY_ARG]]) : tensor<1x16x16x16x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:  -> tensor<1x16x16x16x!qElemType, {order = #NHWC}>
}

}
