//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: env IE_NPU_LOG_FILTER=dump-statistics-of-task-ops vpux-opt --compress-weights-btc --dump-statistics-of-task-ops -o /dev/null %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qtype = !quant.uniform<u8:f32, 1.000000e+00>

module @dual_tile attributes {VPU.arch = #VPU.arch_kind<VPUX37XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
  IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
      DataInfo "input_0" : tensor<1x16x16x16xui8>
    } outputsInfo : {
      DataInfo "output_0" : tensor<2x16x16x16xf16>
    }

  IE.MemoryResource 31457280 bytes of @DDR {VPU.bandwidth = 8, VPU.derateFactor = 6.000000e-01}
  IE.ExecutorResource 1 of @NCE  {
    IE.ExecutorResource 1 of @SHAVE_UPA
    IE.MemoryResource 2097152 bytes of @CMX_NN {VPU.bandwidth = 32, VPU.derateFactor = 1.000000e+00}
    IE.ExecutorResource 1 of @DPU
  }
  IE.ExecutorResource 2 of @DMA_NN

  func.func @main(
        %input_arg: memref<1x16x16x16x!qtype, #NHWC, @DDR>,
        %output_arg: memref<2x16x16x16xf16, #NHWC, @DDR>
      ) -> memref<2x16x16x16xf16, #NHWC, @DDR> {
    %weights_constant = const.Declare memref<16x4x8x16x!qtype, #NHWC, @DDR> =
      dense<1> : tensor<16x4x8x16xui8>, [#const.QuantCast<!qtype>, #const.Reorder<#NHWC>]
    %weights0 = VPURT.DeclareBuffer <CMX_NN> [0] <12544>
      -> memref<16x4x8x16x!qtype, #NHWC, [@CMX_NN, 0]>
    %weights1 = VPURT.DeclareBuffer <CMX_NN> [1] <12544>
      -> memref<16x4x8x16x!qtype, #NHWC, [@CMX_NN, 1]>

    %input_0 = VPURT.DeclareBuffer <CMX_NN> [0] <8192>
      -> memref<1x16x16x16x!qtype, #NHWC, [@CMX_NN, 0]>
    %output_0 = VPURT.DeclareBuffer <CMX_NN> [0] <0>
      -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %output_ddr_0 = VPURT.DeclareBuffer <DDR> <0>
      -> memref<1x16x16x16xf16, #NHWC, @DDR>
    %parent_input_0 = VPURT.DeclareBuffer <CMX_NN> [0] <8192>
      -> memref<1x16x16x16x!qtype, #NHWC, [@CMX_NN, 0]>
    %parent_output_0 = VPURT.DeclareBuffer <CMX_NN> [0] <0>
      -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>

    %input_1 = VPURT.DeclareBuffer <CMX_NN> [1] <8192>
      -> memref<1x16x16x16x!qtype, #NHWC, [@CMX_NN, 1]>
    %output_1 = VPURT.DeclareBuffer <CMX_NN> [1] <0>
      -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>
    %output_ddr_1 = VPURT.DeclareBuffer <DDR> <8192>
      -> memref<1x16x16x16xf16, #NHWC, @DDR>
    %parent_input_1 = VPURT.DeclareBuffer <CMX_NN> [1] <8192>
      -> memref<1x16x16x16x!qtype, #NHWC, [@CMX_NN, 1]>
    %parent_output_1 = VPURT.DeclareBuffer <CMX_NN> [1] <0>
      -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>

    %output_ddr = VPURT.DeclareBuffer <DDR> <0>
      -> memref<2x16x16x16xf16, #NHWC, @DDR>

    %weight_table_constant = const.Declare memref<16x1x1x4xsi32, #NHWC, @DDR> =
      dense<[[[[12544, 16777215, 1073761792, 0]]], [[[12560, 16777215, 1073761792, 0]]], [[[12576, 16777215, 1073761792, 0]]], [[[12592, 16777215, 1073761792, 0]]], [[[12608, 16777215, 1073761792, 0]]], [[[12624, 16777215, 1073761792, 0]]], [[[12640, 16777215, 1073761792, 0]]], [[[12656, 16777215, 1073761792, 0]]], [[[12672, 16777215, 1073761792, 0]]], [[[12688, 16777215, 1073761792, 0]]], [[[12704, 16777215, 1073761792, 0]]], [[[12720, 16777215, 1073761792, 0]]], [[[12736, 16777215, 1073761792, 0]]], [[[12752, 16777215, 1073761792, 0]]], [[[12768, 16777215, 1073761792, 0]]], [[[12784, 16777215, 1073761792, 0]]]]> : tensor<16x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %weight_table0 = VPURT.DeclareBuffer <CMX_NN> [0] <12288>
      -> memref<16x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
    %weight_table1 = VPURT.DeclareBuffer <CMX_NN> [1] <12288>
      -> memref<16x1x1x4xsi32, #NHWC, [@CMX_NN, 1]>

    %inputs_ready = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %conv_complete = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    %output_ready = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier

    VPURT.Task updates(%inputs_ready : !VPURT.Barrier) {
      VPUIP.NNDMA {port = 0}
        inputs(%input_arg : memref<1x16x16x16x!qtype, #NHWC, @DDR>)
        outputs(%input_0 : memref<1x16x16x16x!qtype, #NHWC, [@CMX_NN, 0]>)
        -> memref<1x16x16x16x!qtype, #NHWC, [@CMX_NN, 0]>
    }
    VPURT.Task updates(%inputs_ready : !VPURT.Barrier) {
      VPUIP.NNDMA {port = 0}
        inputs(%input_arg : memref<1x16x16x16x!qtype, #NHWC, @DDR>)
        outputs(%input_1 : memref<1x16x16x16x!qtype, #NHWC, [@CMX_NN, 1]>)
        -> memref<1x16x16x16x!qtype, #NHWC, [@CMX_NN, 1]>
    }

    VPURT.Task updates(%inputs_ready : !VPURT.Barrier) {
      VPUIP.NNDMA {port = 0}
        inputs(%weights_constant : memref<16x4x8x16x!qtype, #NHWC, @DDR>)
        outputs(%weights0 : memref<16x4x8x16x!qtype, #NHWC, [@CMX_NN, 0]>)
        -> memref<16x4x8x16x!qtype, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task updates(%inputs_ready : !VPURT.Barrier) {
      VPUIP.NNDMA {port = 0}
        inputs(%weights_constant : memref<16x4x8x16x!qtype, #NHWC, @DDR>)
        outputs(%weights1 : memref<16x4x8x16x!qtype, #NHWC, [@CMX_NN, 1]>)
        -> memref<16x4x8x16x!qtype, #NHWC, [@CMX_NN, 1]>
    }

    VPURT.Task updates(%inputs_ready : !VPURT.Barrier) {
      VPUIP.NNDMA {port = 0}
        inputs(%weight_table_constant : memref<16x1x1x4xsi32, #NHWC, @DDR>)
        outputs(%weight_table0 : memref<16x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>)
        -> memref<16x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task updates(%inputs_ready : !VPURT.Barrier) {
      VPUIP.NNDMA {port = 0}
        inputs(%weight_table_constant : memref<16x1x1x4xsi32, #NHWC, @DDR>)
        outputs(%weight_table1 : memref<16x1x1x4xsi32, #NHWC, [@CMX_NN, 1]>)
        -> memref<16x1x1x4xsi32, #NHWC, [@CMX_NN, 1]>
    }

    VPURT.Task waits(%inputs_ready : !VPURT.Barrier) updates(%conv_complete : !VPURT.Barrier) {
      VPUIP.NCEClusterTask {
          kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          kernel_size = [1, 8],
          kernel_strides = [1, 1],
          task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%input_0 : memref<1x16x16x16x!qtype, #NHWC, [@CMX_NN, 0]>)
        weights(%weights0 : memref<16x4x8x16x!qtype, #NHWC, [@CMX_NN, 0]>)
        weight_table(%weight_table0 : memref<16x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>)
        parent_input(%parent_input_0 : memref<1x16x16x16x!qtype, #NHWC, [@CMX_NN, 0]>)
        parent_output(%parent_output_0 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>)
        outputs(%output_0 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>)
        -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
        variants : {
          DPUTask {
            outEnd = [15, 15, 15],
            mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            outStart = [0, 0, 0]
          }
        }
        PPE : {
        }
    }

    VPURT.Task waits(%inputs_ready : !VPURT.Barrier) updates(%conv_complete : !VPURT.Barrier) {
      VPUIP.NCEClusterTask {
          kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          kernel_size = [1, 8],
          kernel_strides = [1, 1],
          task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%input_1 : memref<1x16x16x16x!qtype, #NHWC, [@CMX_NN, 1]>)
        weights(%weights1 : memref<16x4x8x16x!qtype, #NHWC, [@CMX_NN, 1]>)
        weight_table(%weight_table1 : memref<16x1x1x4xsi32, #NHWC, [@CMX_NN, 1]>)
        parent_input(%parent_input_1 : memref<1x16x16x16x!qtype, #NHWC, [@CMX_NN, 1]>)
        parent_output(%parent_output_1 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>)
        outputs(%output_1 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>)
        -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>
        variants : {
          DPUTask {
            outEnd = [15, 15, 15],
            mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            outStart = [0, 0, 0]
          }
        }
        PPE : {
        }
    }

    VPURT.Task waits(%conv_complete : !VPURT.Barrier) updates(%output_ready : !VPURT.Barrier) {
      VPUIP.NNDMA {port = 0}
        inputs(%output_0 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>)
        outputs(%output_ddr_0 : memref<1x16x16x16xf16, #NHWC, @DDR>)
        -> memref<1x16x16x16xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%conv_complete : !VPURT.Barrier) updates(%output_ready : !VPURT.Barrier) {
      VPUIP.NNDMA {port = 0}
        inputs(%output_1 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>)
        outputs(%output_ddr_1 : memref<1x16x16x16xf16, #NHWC, @DDR>)
        -> memref<1x16x16x16xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%output_ready : !VPURT.Barrier) {
      VPUIP.NNDMA {port = 0}
        inputs(%output_ddr : memref<2x16x16x16xf16, #NHWC, @DDR>)
        outputs(%output_arg : memref<2x16x16x16xf16, #NHWC, @DDR>)
        -> memref<2x16x16x16xf16, #NHWC, @DDR>
    }

    return %output_arg : memref<2x16x16x16xf16, #NHWC, @DDR>
  }
}

// CHECK: VPUIP tasks statistics:
// CHECK: VPUIP Tasks - 11 ops
// CHECK:   VPUIP.NNDMA - 7 ops
// CHECK:     CMX2DDR - 2 ops
// CHECK:     DDR2CMX - 4 ops
// CHECK:     DDR2DDR - 1 ops
// CHECK:   VPUIP.DecompressDMAOp - 2 ops
// CHECK:     DDR2CMX - 2 ops
// CHECK:   VPUIP.NCEClusterTask - 2 ops

// CHECK: Weights compression statistics:
// CHECK: Constants size before compression: 8448 bytes
// CHECK: Constants size after compression: 2752 bytes
// CHECK: Constants that were compressed: 2496 bytes (90.70% of total)
// CHECK: Constants that couldn't be compressed: 256 bytes (9.30% of total)
// CHECK: Compression rate of compressed constants: 30.47%
// CHECK: Total compression rate: 32.58%
