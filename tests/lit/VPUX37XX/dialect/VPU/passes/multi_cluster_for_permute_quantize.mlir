// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX allow-custom-values=true" --multi-cluster-strategy-assignment %s | FileCheck %s

// This operation needs a separate lit-test file because vpux::VPU::NCEPermuteQuantizeOp::verify() fails to check arch.
// See E#60343 for details. Merge this file into multi_cluster_strategy_assignment.mlir when the ticket is resolved.

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: module @PermuteQuantizeAssignedSOW
module @PermuteQuantizeAssignedSOW attributes {VPU.arch = #VPU.arch_kind<VPUX37XX>} {
  IE.ExecutorResource 2 of @NCE at 1.300000e+03 MHz {
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @SHAVE_NN
    IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 1 of @DPU
  }
  IE.ExecutorResource 2 of @DMA_NN

func.func @main(%arg0: tensor<1x224x3x256xf16, {order = #NHWC}>) -> tensor<1x224x4x256x!qElemType, {order = #NWCH}> {
    %PERMUTE_QUANTIZE = VPU.NCE.PermuteQuantize(%arg0) {
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
    } -> tensor<1x224x4x256x!qElemType, {order = #NWCH}>

    // CHECK:   multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>

    return %PERMUTE_QUANTIZE : tensor<1x224x4x256x!qElemType, {order = #NWCH}>
}

}

// -----

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: module @PermuteQuantizeFP16AssignedSOW
module @PermuteQuantizeFP16AssignedSOW attributes {VPU.arch = #VPU.arch_kind<VPUX37XX>} {
  IE.ExecutorResource 2 of @NCE at 1.300000e+03 MHz {
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @SHAVE_NN
    IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 1 of @DPU
  }
  IE.ExecutorResource 2 of @DMA_NN

func.func @main(%arg0: tensor<1x224x3x256xf16, {order = #NHWC}>) -> tensor<1x224x4x256xf16, {order = #NWCH}> {
    %PERMUTE_QUANTIZE = VPU.NCE.PermuteQuantize(%arg0) {
        dstElemType = f16,
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
    } -> tensor<1x224x4x256xf16, {order = #NWCH}>

    // CHECK:   multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>

    return %PERMUTE_QUANTIZE : tensor<1x224x4x256xf16, {order = #NWCH}>
}

}

// -----

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: module @SkipTrivialWidth
module @SkipTrivialWidth attributes {VPU.arch = #VPU.arch_kind<VPUX37XX>} {
  IE.ExecutorResource 2 of @NCE at 1.300000e+03 MHz {
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @SHAVE_NN
    IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 1 of @DPU
  }
  IE.ExecutorResource 2 of @DMA_NN

func.func @main(%arg0: tensor<1x224x3x1xf16, {order = #NHWC}>) -> tensor<1x224x4x1x!qElemType, {order = #NWCH}> {
    %PERMUTE_QUANTIZE = VPU.NCE.PermuteQuantize(%arg0) {
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
    } -> tensor<1x224x4x1x!qElemType, {order = #NWCH}>

    // CHECK-NOT:   multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>

    return %PERMUTE_QUANTIZE : tensor<1x224x4x1x!qElemType, {order = #NWCH}>
}

}

// -----

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: module @SkipIsolatedTiling
module @SkipIsolatedTiling attributes {VPU.arch = #VPU.arch_kind<VPUX37XX>} {
  IE.ExecutorResource 2 of @NCE at 1.300000e+03 MHz {
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @SHAVE_NN
    IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 1 of @DPU
  }
  IE.ExecutorResource 2 of @DMA_NN

func.func @main(%arg0: tensor<1x224x3x2560xf16, {order = #NHWC}>) -> tensor<1x224x4x2560xf16, {order = #NWCH}> {
    %PERMUTE_QUANTIZE = VPU.NCE.PermuteQuantize(%arg0) {
        dstElemType = f16,
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
    } -> tensor<1x224x4x2560xf16, {order = #NWCH}>

    // CHECK:   multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>

    return %PERMUTE_QUANTIZE : tensor<1x224x4x2560xf16, {order = #NWCH}>
}

}

// -----

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: module @SkipPQForMCSubgraphOptimization
module @SkipPQForMCSubgraphOptimization attributes {VPU.arch = #VPU.arch_kind<VPUX37XX>} {
  IE.ExecutorResource 2 of @NCE at 1.300000e+03 MHz {
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @SHAVE_NN
    IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 1 of @DPU
  }
  IE.ExecutorResource 2 of @DMA_NN

func.func @main(%arg0: tensor<1x16x4x64xf16, {order = #NHWC}>, %arg1: tensor<1x16x4x64xf16, {order = #NHWC}>)
                -> tensor<1x16x4x128x!qElemType, {order = #NWCH}> {
    %PERMUTE_QUANTIZE_0 = VPU.NCE.PermuteQuantize(%arg0) {
      dstElemType = !qElemType,
      dstOrder = #NWCH,
      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPETask<clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = <NOOP>>
    } -> tensor<1x16x4x64x!qElemType, {order = #NWCH}>

    %PERMUTE_QUANTIZE_1 = VPU.NCE.PermuteQuantize(%arg1) {
      dstElemType = !qElemType,
      dstOrder = #NWCH,
      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPETask<clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = <NOOP>>
    } -> tensor<1x16x4x64x!qElemType, {order = #NWCH}>
 
    %CONCAT = VPU.Concat(%PERMUTE_QUANTIZE_0, %PERMUTE_QUANTIZE_1) {
      static_offsets = [[0, 0, 0, 0], [0, 0, 0, 64]]}
      : tensor<1x16x4x64x!qElemType, {order = #NWCH}>, tensor<1x16x4x64x!qElemType, {order = #NWCH}>
      -> tensor<1x16x4x128x!qElemType, {order = #NWCH}>

    // CHECK:   VPU.NCE.PermuteQuantize
    // CHECK:   multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>
    // CHECK:   VPU.NCE.PermuteQuantize
    // CHECK:   multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>

    return %CONCAT : tensor<1x16x4x128x!qElemType, {order = #NWCH}>
}

}
