// RUN: vpux-opt %s

//
// This file generates a blob that runs convolutions on two tiles, used to
// demonstrate that the runtime can handle this.  It's also a lit test to help
// check for regressions in the VPUIP dialect.
//
// To generate a blob, use:
//
//    vpux-translate --export-VPUIP < dual_tile.mlir > dual_tile.blob
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map0 = affine_map<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 256 + d2 * 16 + d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 1024 + d2 * 64 + d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0 * 16 + d1 * 16 + d2 * 16 + d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0 * 4 + d1 * 4 + d2 + d3)>

!qtype = type !quant.uniform<ui8<0:1>:f32, 1.000000e+00>

module @mainModule attributes {VPUIP.arch = "MTL", VPUIP.compilationMode = "ReferenceHW"}  {
  VPUIP.Graph options : "NONE" version : {contextStr = "VPUX Compiler", hash = "custom_mtl-pss-1.0_683329294c304fd3c4609f8d0fc9301f67dc8321", majorV = 3 : i32, minorV = 26 : i32, patchV = 0 : i32}
  IERT.RunTimeResources availableMemory :  {
    IERT.MemoryResource 524288000 bytes of "DDR" {VPUIP.bandwidth = 8 : i64, VPUIP.derateFactor = 6.000000e-01}
    IERT.MemoryResource 1966080 bytes of "CMX_NN" {VPUIP.bandwidth = 32 : i64, VPUIP.derateFactor = 1.000000e+00}
  } usedMemory :  {
      IERT.MemoryResource 65536 bytes of "DDR" {VPUIP.bandwidth = 8 : i64, VPUIP.derateFactor = 6.000000e-01}
      IERT.MemoryResource 38912 bytes of "CMX_NN" {VPUIP.bandwidth = 32 : i64, VPUIP.derateFactor = 1.000000e+00}
  } executors :  {
    IERT.ExecutorResource 2 of "DMA_NN"
    IERT.ExecutorResource 2 of "NCE_Cluster"  {
      IERT.ExecutorResource 1 of "NCE_PerClusterDPU"
    }
  }
  func private @"dual_tile"(%arg0: memref<1x16x16x16x!qtype, #NHWC, #map0, "ProgrammableInput">, %arg1: memref<2x64x16x16xf16, #NHWC, #map1, "ProgrammableOutput">) -> memref<2x64x16x16xf16, #NHWC, #map1, "ProgrammableOutput"> {
    %weights_const = const.Declare memref<64x16x1x1x!qtype, #NHWC, #map2, "GraphFile"> = #const.Content<dense<"0x00000101000100010000010001010101000001010000010101000100010000010100000001010100010101010100010100010100010001000000000001010001000001000000000101010100000001000100010101010001000000000001000101000000000000010101000100010001010000000100010100010001010000010101010001000000000101000000010101000001010000010100000000010101010100000001000000010000000000000100000000010001000000000100000100000100010101010100010101010101000101000001010001000100000101010101000100010101010100000100010100000100000100000000000101000101000100000101010100000000000101010101000001010001010001000101010100000000000101000100010000000001000101010101000000000001000001010001010001010000010101010000000100010000010001010001010101010101000001000100010101000100010001000100000101010101000001010100000001010100010101010000010000000100000101000100010100010101010001000001010000010100010001010001000100000101010000010101000000000001010000010101000100010100010101010001010101000000010100000101000100010000010001000100010001000001000101010101000100010001000001010101010000010000000100000000010001000100010000010001000001010100000001000000000001000101000100010001010000010001000001000001010001010001010100000100010000010101000100000101010101010001010101000101000000010100010000000000010000010100000001010001010000000000000001010000000000010001000000010001000000000001000001000001000101010000010101010001000000000000010001010100010101000100000100010001000001000100010100010101010001000101000000010101000101000101010001010001000100000100000000000100000001000100000001000101010100000000010100000101010101000101000100010000000100010000000100010101010000000001010000000100010000010101010100000000000001010101000100000001000001000000000100010000010100010100000100010000010000000101010100010101000101000000010101010100010001000101010001000101000101010001010001000100000101000001010100010101000101000100000100010100000000010100000101010001000101000101000101000100010001000100010100010101000000010001010001010100000101010100010000010100010000010001010100000101010000010001010101000100010101000001"> : tensor<64x16x1x1xui8>, [#const.QuantCast<!qtype>, #const.Reorder<#NHWC>]>
    %weights_cmx_all = VPUIP.DeclareTensor "VPU_CMX_NN" [0, 1] <37888> -> memref<64x16x1x1x!qtype, #NHWC, #map2, "VPU_CMX_NN">
    %weights_cmx_0 = VPUIP.DeclareTensor "VPU_CMX_NN" [0] <37888> -> memref<64x16x1x1x!qtype, #NHWC, #map2, "VPU_CMX_NN">
    %weights_cmx_1 = VPUIP.DeclareTensor "VPU_CMX_NN" [1] <37888> -> memref<64x16x1x1x!qtype, #NHWC, #map2, "VPU_CMX_NN">
    %act_cmx_all = VPUIP.DeclareTensor "VPU_CMX_NN" [0, 1] <32768> -> memref<1x16x16x16x!qtype, #NHWC, #map0, "VPU_CMX_NN">
    %act_cmx_0 = VPUIP.DeclareTensor "VPU_CMX_NN" [0] <32768> -> memref<1x16x16x16x!qtype, #NHWC, #map0, "VPU_CMX_NN">
    %act_cmx_1 = VPUIP.DeclareTensor "VPU_CMX_NN" [1] <32768> -> memref<1x16x16x16x!qtype, #NHWC, #map0, "VPU_CMX_NN">
    %out_cmx_0 = VPUIP.DeclareTensor "VPU_CMX_NN" [0] <0> -> memref<1x64x16x16xf16, #NHWC, #map1, "VPU_CMX_NN">
    %out_cmx_1 = VPUIP.DeclareTensor "VPU_CMX_NN" [1] <0> -> memref<1x64x16x16xf16, #NHWC, #map1, "VPU_CMX_NN">
    %act_parent_cmx_0 = VPUIP.DeclareTensor "VPU_CMX_NN" [0] <32768> -> memref<1x16x16x16x!qtype, #NHWC, #map0, "VPU_CMX_NN">
    %act_parent_cmx_1 = VPUIP.DeclareTensor "VPU_CMX_NN" [1] <32768> -> memref<1x16x16x16x!qtype, #NHWC, #map0, "VPU_CMX_NN">
    %out_parent_cmx_0 = VPUIP.DeclareTensor "VPU_CMX_NN" [0] <0> -> memref<1x64x16x16xf16, #NHWC, #map1, "VPU_CMX_NN">
    %out_parent_cmx_1 = VPUIP.DeclareTensor "VPU_CMX_NN" [1] <0> -> memref<1x64x16x16xf16, #NHWC, #map1, "VPU_CMX_NN">
    %out_ddr_0 = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<1x64x16x16xf16, #NHWC, #map1, "VPU_DDR_Heap">
    %out_ddr_1 = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <32768> -> memref<1x64x16x16xf16, #NHWC, #map1, "VPU_DDR_Heap">
    %out_ddr_all = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<2x64x16x16xf16, #NHWC, #map1, "VPU_DDR_Heap">
    %weight_table_const = const.Declare memref<64x1x1x4xsi32, #NHWC, #map3, "GraphFile"> = #const.Content<dense<"0x00940000FFFFFF00004E00400000000010940000FFFFFF00004E00400000000020940000FFFFFF00004E00400000000030940000FFFFFF00004E00400000000040940000FFFFFF00004E00400000000050940000FFFFFF00004E00400000000060940000FFFFFF00004E00400000000070940000FFFFFF00004E00400000000080940000FFFFFF00004E00400000000090940000FFFFFF00004E004000000000A0940000FFFFFF00004E004000000000B0940000FFFFFF00004E004000000000C0940000FFFFFF00004E004000000000D0940000FFFFFF00004E004000000000E0940000FFFFFF00004E004000000000F0940000FFFFFF00004E00400000000000950000FFFFFF00004E00400000000010950000FFFFFF00004E00400000000020950000FFFFFF00004E00400000000030950000FFFFFF00004E00400000000040950000FFFFFF00004E00400000000050950000FFFFFF00004E00400000000060950000FFFFFF00004E00400000000070950000FFFFFF00004E00400000000080950000FFFFFF00004E00400000000090950000FFFFFF00004E004000000000A0950000FFFFFF00004E004000000000B0950000FFFFFF00004E004000000000C0950000FFFFFF00004E004000000000D0950000FFFFFF00004E004000000000E0950000FFFFFF00004E004000000000F0950000FFFFFF00004E00400000000000960000FFFFFF00004E00400000000010960000FFFFFF00004E00400000000020960000FFFFFF00004E00400000000030960000FFFFFF00004E00400000000040960000FFFFFF00004E00400000000050960000FFFFFF00004E00400000000060960000FFFFFF00004E00400000000070960000FFFFFF00004E00400000000080960000FFFFFF00004E00400000000090960000FFFFFF00004E004000000000A0960000FFFFFF00004E004000000000B0960000FFFFFF00004E004000000000C0960000FFFFFF00004E004000000000D0960000FFFFFF00004E004000000000E0960000FFFFFF00004E004000000000F0960000FFFFFF00004E00400000000000970000FFFFFF00004E00400000000010970000FFFFFF00004E00400000000020970000FFFFFF00004E00400000000030970000FFFFFF00004E00400000000040970000FFFFFF00004E00400000000050970000FFFFFF00004E00400000000060970000FFFFFF00004E00400000000070970000FFFFFF00004E00400000000080970000FFFFFF00004E00400000000090970000FFFFFF00004E004000000000A0970000FFFFFF00004E004000000000B0970000FFFFFF00004E004000000000C0970000FFFFFF00004E004000000000D0970000FFFFFF00004E004000000000E0970000FFFFFF00004E004000000000F0970000FFFFFF00004E004000000000"> : tensor<64x1x1x4xsi32>, [#const.Reorder<#NHWC>]>
    %weight_table_cmx_all = VPUIP.DeclareTensor "VPU_CMX_NN" [0, 1] <36864> -> memref<64x1x1x4xsi32, #NHWC, #map3, "VPU_CMX_NN">
    %weight_table_cmx_0 = VPUIP.DeclareTensor "VPU_CMX_NN" [0] <36864> -> memref<64x1x1x4xsi32, #NHWC, #map3, "VPU_CMX_NN">
    %weight_table_cmx_1 = VPUIP.DeclareTensor "VPU_CMX_NN" [1] <36864> -> memref<64x1x1x4xsi32, #NHWC, #map3, "VPU_CMX_NN">
    %inputs_done = VPUIP.ConfigureBarrier<0> -> !VPUIP.Barrier
    %convs_done = VPUIP.ConfigureBarrier<1> -> !VPUIP.Barrier
    %outputs_done = VPUIP.ConfigureBarrier<2> -> !VPUIP.Barrier

    // Load inputs to CMX
    VPUIP.NNDMA {port = 0} inputs(%arg0 : memref<1x16x16x16x!qtype, #NHWC, #map0, "ProgrammableInput">) outputs(%act_cmx_all : memref<1x16x16x16x!qtype, #NHWC, #map0, "VPU_CMX_NN">) updates(%inputs_done : !VPUIP.Barrier) -> memref<1x16x16x16x!qtype, #NHWC, #map0, "VPU_CMX_NN">
    VPUIP.NNDMA {port = 0} inputs(%weights_const : memref<64x16x1x1x!qtype, #NHWC, #map2, "GraphFile">) outputs(%weights_cmx_all : memref<64x16x1x1x!qtype, #NHWC, #map2, "VPU_CMX_NN">) updates(%inputs_done : !VPUIP.Barrier) -> memref<64x16x1x1x!qtype, #NHWC, #map2, "VPU_CMX_NN">
    VPUIP.NNDMA {port = 0} inputs(%weight_table_const : memref<64x1x1x4xsi32, #NHWC, #map3, "GraphFile">) outputs(%weight_table_cmx_all : memref<64x1x1x4xsi32, #NHWC, #map3, "VPU_CMX_NN">) updates(%inputs_done : !VPUIP.Barrier) -> memref<64x1x1x4xsi32, #NHWC, #map3, "VPU_CMX_NN">

    // Store outputs to DDR for concatenation
    VPUIP.NNDMA {port = 0} inputs(%out_cmx_0 : memref<1x64x16x16xf16, #NHWC, #map1, "VPU_CMX_NN">) outputs(%out_ddr_0 : memref<1x64x16x16xf16, #NHWC, #map1, "VPU_DDR_Heap">) waits(%convs_done : !VPUIP.Barrier) updates(%outputs_done : !VPUIP.Barrier) -> memref<1x64x16x16xf16, #NHWC, #map1, "VPU_DDR_Heap">
    VPUIP.NNDMA {port = 0} inputs(%out_cmx_0 : memref<1x64x16x16xf16, #NHWC, #map1, "VPU_CMX_NN">) outputs(%out_ddr_1 : memref<1x64x16x16xf16, #NHWC, #map1, "VPU_DDR_Heap">) waits(%convs_done : !VPUIP.Barrier) updates(%outputs_done : !VPUIP.Barrier) -> memref<1x64x16x16xf16, #NHWC, #map1, "VPU_DDR_Heap">

    // Store concatenated output to program output
    VPUIP.NNDMA {port = 0} inputs(%out_ddr_all : memref<2x64x16x16xf16, #NHWC, #map1, "VPU_DDR_Heap">) outputs(%arg1 : memref<2x64x16x16xf16, #NHWC, #map1, "ProgrammableOutput">) waits(%outputs_done : !VPUIP.Barrier) -> memref<2x64x16x16xf16, #NHWC, #map1, "ProgrammableOutput">

    // The convolution tasks
    VPUIP.NCEClusterTask {
        kernel_padding = [0, 0, 0, 0],
        kernel_size = [1, 1],
        kernel_strides = [1, 1],
        task_type = "CONV"
      }
      input(%act_cmx_0 : memref<1x16x16x16x!qtype, #NHWC, #map0, "VPU_CMX_NN">)
      weights(%weights_cmx_0 : memref<64x16x1x1x!qtype, #NHWC, #map2, "VPU_CMX_NN">)
      weight_table(%weight_table_cmx_0 : memref<64x1x1x4xsi32, #NHWC, #map3, "VPU_CMX_NN">)
      parent_input(%act_parent_cmx_0 : memref<1x16x16x16x!qtype, #NHWC, #map0, "VPU_CMX_NN">)
      parent_output(%out_parent_cmx_0 : memref<1x64x16x16xf16, #NHWC, #map1, "VPU_CMX_NN">)
      outputs(%out_cmx_0 : memref<1x64x16x16xf16, #NHWC, #map1, "VPU_CMX_NN">)
      waits(%inputs_done : !VPUIP.Barrier)
      updates(%convs_done : !VPUIP.Barrier)
      -> memref<1x64x16x16xf16, #NHWC, #map1, "VPU_CMX_NN">
      variants : {
        DPUTask {
          end = [15, 15, 63],
          mpe_mode = "CUBOID_16x16",
          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
          start = [0, 0, 0]}
      } PPE : {
        VPUIP.PPETask "NOOP" {
          clamp_high = 2147483647,
          clamp_low = -2147483648,
          lrelu_mult = 1,
          lrelu_shift = 0 : ui32
        }
      }

    VPUIP.NCEClusterTask {
        kernel_padding = [0, 0, 0, 0],
        kernel_size = [1, 1],
        kernel_strides = [1, 1],
        task_type = "CONV"}
        input(%act_cmx_1 : memref<1x16x16x16x!qtype, #NHWC, #map0, "VPU_CMX_NN">)
        weights(%weights_cmx_1 : memref<64x16x1x1x!qtype, #NHWC, #map2, "VPU_CMX_NN">)
        weight_table(%weight_table_cmx_1 : memref<64x1x1x4xsi32, #NHWC, #map3, "VPU_CMX_NN">)
        parent_input(%act_parent_cmx_1 : memref<1x16x16x16x!qtype, #NHWC, #map0, "VPU_CMX_NN">)
        parent_output(%out_parent_cmx_1 : memref<1x64x16x16xf16, #NHWC, #map1, "VPU_CMX_NN">)
        outputs(%out_cmx_1 : memref<1x64x16x16xf16, #NHWC, #map1, "VPU_CMX_NN">)
        waits(%inputs_done : !VPUIP.Barrier)
        updates(%convs_done : !VPUIP.Barrier)
        -> memref<1x64x16x16xf16, #NHWC, #map1, "VPU_CMX_NN">
        variants : {
          VPUIP.DPUTask {
            end = [15, 15, 63],
            mpe_mode = "CUBOID_16x16",
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            start = [0, 0, 0]
          }
        } PPE : {
          VPUIP.PPETask "NOOP" {
            clamp_high = 2147483647,
            clamp_low = -2147483648,
            lrelu_mult = 1,
            lrelu_shift = 0 : ui32
          }
        }
    return %arg1 : memref<2x64x16x16xf16, #NHWC, #map1, "ProgrammableOutput">
  }
  IE.CNNNetwork entryPoint : @"dual_tile" inputsInfo :  {
    IE.DataInfo "input_0" : tensor<1x16x16x16x!qtype, {order = #NHWC}>
  } outputsInfo :  {
    IE.DataInfo "output_0" : tensor<2x64x16x16xf16, {order = #NHWC}>
  }
}
