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

// logical shape = ?x16x16x16, mem shape(NHWC) = ?x16x16x16
#act_mem_strides = affine_map<(md0, md1, md2, md3) -> (md0 * 4096 + md1 * 256 + md2 * 16 + md3)>

// logical shape = 16x1x1x16, mem shape(NHWC) = 16x1x16x1
#filter_mem_strides = affine_map<(md0, md1, md2, md3) -> (md0 * 16 + md1 * 16 + md2 + md3)>

// logical shape = 16x1x1x4, mem shape(NHWC) = 16x1x4x1
#wt_mem_strides = affine_map<(md0, md1, md2, md3) -> (md0 * 4 + md1 * 4 + md2 + md3)>

!qtype = type !quant.uniform<u8:f32, 1.000000e+00>

module @dual_tile attributes {VPUIP.arch = "MTL", VPUIP.compilationMode = "ReferenceHW"} {
  VPUIP.Graph
    options : "NONE"
    version : {
      contextStr = "VPUX Compiler",
      hash = "",
      majorV = 3,
      minorV = 11,
      patchV = 0
    }

  IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
      IE.DataInfo "input_0" : tensor<1x16x16x16xui8, {order = #NHWC}>
    } outputsInfo : {
      IE.DataInfo "output_0" : tensor<2x16x16x16xf16, {order = #NHWC}>
    }

  IERT.RunTimeResources
    availableMemory :  {
      IERT.MemoryResource 1073741824 bytes
      IERT.MemoryResource 31457280 bytes of "DDR" {VPUIP.bandwidth = 8, VPUIP.derateFactor = 6.000000e-01}
      IERT.MemoryResource 2097152 bytes of "CMX_NN" {VPUIP.bandwidth = 32, VPUIP.derateFactor = 1.000000e+00}
    } usedMemory :  {
    } executors :  {
      IERT.ExecutorResource 1 of "Leon_RT"
      IERT.ExecutorResource 1 of "Leon_NN"
      IERT.ExecutorResource 1 of "DMA_UPA"
      IERT.ExecutorResource 1 of "SHAVE_NN"
      IERT.ExecutorResource 1 of "NCE_Cluster"  {
        IERT.ExecutorResource 1 of "NCE_PerClusterDPU"
      }
      IERT.ExecutorResource 2 of "DMA_NN"
    }

  func @main(
        %input_arg: memref<1x16x16x16x!qtype, #NHWC, #act_mem_strides, "ProgrammableInput">,
        %output_arg: memref<2x16x16x16xf16, #NHWC, #act_mem_strides, "ProgrammableOutput">
      ) -> memref<2x16x16x16xf16, #NHWC, #act_mem_strides, "ProgrammableOutput"> {
    %weights_constant = const.Declare memref<16x1x1x16x!qtype, #NHWC, #filter_mem_strides, "GraphFile"> =
      #const.Content<dense<1> : tensor<16x1x1x16xui8>, [#const.QuantCast<!qtype>, #const.Reorder<#NHWC>]>
    %weights = VPUIP.DeclareTensor "VPU_CMX_NN" [0] <12544>
      -> memref<16x1x1x16x!qtype, #NHWC, #filter_mem_strides, "VPU_CMX_NN">

    %input_broadcast = VPUIP.DeclareTensor "VPU_CMX_NN" [0, 1] <8192>
      -> memref<1x16x16x16x!qtype, #NHWC, #act_mem_strides, "VPU_CMX_NN">
    %input_0 = VPUIP.DeclareTensor "VPU_CMX_NN" [0] <8192>
      -> memref<1x16x16x16x!qtype, #NHWC, #act_mem_strides, "VPU_CMX_NN">

    %output_0 = VPUIP.DeclareTensor "VPU_CMX_NN" [0] <0>
      -> memref<1x16x16x16xf16, #NHWC, #act_mem_strides, "VPU_CMX_NN">
    %output_ddr_0 = VPUIP.DeclareTensor "VPU_DDR_Heap" <0>
      -> memref<1x16x16x16xf16, #NHWC, #act_mem_strides, "VPU_DDR_Heap">
    %parent_input_0 = VPUIP.DeclareTensor "VPU_CMX_NN" [0] <8192>
      -> memref<1x16x16x16x!qtype, #NHWC, #act_mem_strides, "VPU_CMX_NN">
    %parent_output_0 = VPUIP.DeclareTensor "VPU_CMX_NN" [0] <0>
      -> memref<1x16x16x16xf16, #NHWC, #act_mem_strides, "VPU_CMX_NN">

    %input_1 = VPUIP.DeclareTensor "VPU_CMX_NN" [1] <8192>
      -> memref<1x16x16x16x!qtype, #NHWC, #act_mem_strides, "VPU_CMX_NN">
    %output_1 = VPUIP.DeclareTensor "VPU_CMX_NN" [1] <0>
      -> memref<1x16x16x16xf16, #NHWC, #act_mem_strides, "VPU_CMX_NN">
    %output_ddr_1 = VPUIP.DeclareTensor "VPU_DDR_Heap" <8192>
      -> memref<1x16x16x16xf16, #NHWC, #act_mem_strides, "VPU_DDR_Heap">
    %parent_input_1 = VPUIP.DeclareTensor "VPU_CMX_NN" [1] <8192>
      -> memref<1x16x16x16x!qtype, #NHWC, #act_mem_strides, "VPU_CMX_NN">
    %parent_output_1 = VPUIP.DeclareTensor "VPU_CMX_NN" [1] <0>
      -> memref<1x16x16x16xf16, #NHWC, #act_mem_strides, "VPU_CMX_NN">

    %output_ddr = VPUIP.DeclareTensor "VPU_DDR_Heap" <0>
      -> memref<2x16x16x16xf16, #NHWC, #act_mem_strides, "VPU_DDR_Heap">

    %weight_table_constant = const.Declare memref<16x1x1x4xsi32, #NHWC, #wt_mem_strides, "GraphFile"> =
      #const.Content<dense<[[[[12544, 16777215, 1073761792, 0]]], [[[12560, 16777215, 1073761792, 0]]], [[[12576, 16777215, 1073761792, 0]]], [[[12592, 16777215, 1073761792, 0]]], [[[12608, 16777215, 1073761792, 0]]], [[[12624, 16777215, 1073761792, 0]]], [[[12640, 16777215, 1073761792, 0]]], [[[12656, 16777215, 1073761792, 0]]], [[[12672, 16777215, 1073761792, 0]]], [[[12688, 16777215, 1073761792, 0]]], [[[12704, 16777215, 1073761792, 0]]], [[[12720, 16777215, 1073761792, 0]]], [[[12736, 16777215, 1073761792, 0]]], [[[12752, 16777215, 1073761792, 0]]], [[[12768, 16777215, 1073761792, 0]]], [[[12784, 16777215, 1073761792, 0]]]]> : tensor<16x1x1x4xsi32>, [#const.Reorder<#NHWC>]>

    %weight_table = VPUIP.DeclareTensor "VPU_CMX_NN" [0] <12288>
      -> memref<16x1x1x4xsi32, #NHWC, #wt_mem_strides, "VPU_CMX_NN">

    %inputs_ready = VPUIP.ConfigureBarrier<0> -> !VPUIP.Barrier
    %conv_complete = VPUIP.ConfigureBarrier<1> -> !VPUIP.Barrier
    %output_ready = VPUIP.ConfigureBarrier<2> -> !VPUIP.Barrier

    VPUIP.NNDMA {port = 0}
      inputs(%input_arg : memref<1x16x16x16x!qtype, #NHWC, #act_mem_strides, "ProgrammableInput">)
      outputs(%input_broadcast : memref<1x16x16x16x!qtype, #NHWC, #act_mem_strides, "VPU_CMX_NN">)
      updates(%inputs_ready : !VPUIP.Barrier)
      -> memref<1x16x16x16x!qtype, #NHWC, #act_mem_strides, "VPU_CMX_NN">

    VPUIP.NNDMA {port = 0}
      inputs(%weights_constant : memref<16x1x1x16x!qtype, #NHWC, #filter_mem_strides, "GraphFile">)
      outputs(%weights : memref<16x1x1x16x!qtype, #NHWC, #filter_mem_strides, "VPU_CMX_NN">)
      updates(%inputs_ready : !VPUIP.Barrier)
      -> memref<16x1x1x16x!qtype, #NHWC, #filter_mem_strides, "VPU_CMX_NN">

    VPUIP.NNDMA {port = 0}
      inputs(%weight_table_constant : memref<16x1x1x4xsi32, #NHWC, #wt_mem_strides, "GraphFile">)
      outputs(%weight_table : memref<16x1x1x4xsi32, #NHWC, #wt_mem_strides, "VPU_CMX_NN">)
      updates(%inputs_ready : !VPUIP.Barrier)
      -> memref<16x1x1x4xsi32, #NHWC, #wt_mem_strides, "VPU_CMX_NN">

    VPUIP.NCEClusterTask {
        kernel_padding = [0, 0, 0, 0],
        kernel_size = [1, 8],
        kernel_strides = [1, 1],
        task_type = "CONV"
      }
      input(%input_0 : memref<1x16x16x16x!qtype, #NHWC, #act_mem_strides, "VPU_CMX_NN">)
      weights(%weights : memref<16x1x1x16x!qtype, #NHWC, #filter_mem_strides, "VPU_CMX_NN">)
      weight_table(%weight_table : memref<16x1x1x4xsi32, #NHWC, #wt_mem_strides, "VPU_CMX_NN">)
      parent_input(%parent_input_0 : memref<1x16x16x16x!qtype, #NHWC, #act_mem_strides, "VPU_CMX_NN">)
      parent_output(%parent_output_0 : memref<1x16x16x16xf16, #NHWC, #act_mem_strides, "VPU_CMX_NN">)
      outputs(%output_0 : memref<1x16x16x16xf16, #NHWC, #act_mem_strides, "VPU_CMX_NN">)
      waits(%inputs_ready : !VPUIP.Barrier)
      updates(%conv_complete : !VPUIP.Barrier)
      -> memref<1x16x16x16xf16, #NHWC, #act_mem_strides, "VPU_CMX_NN">
      variants : {
        VPUIP.DPUTask {
          end = [15, 15, 15],
          mpe_mode = "CUBOID_16x16",
          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
          start = [0, 0, 0]
        }
      }
      PPE : {
      }

    VPUIP.NCEClusterTask {
        kernel_padding = [0, 0, 0, 0],
        kernel_size = [1, 8],
        kernel_strides = [1, 1],
        task_type = "CONV"
      }
      input(%input_1 : memref<1x16x16x16x!qtype, #NHWC, #act_mem_strides, "VPU_CMX_NN">)
      weights(%weights : memref<16x1x1x16x!qtype, #NHWC, #filter_mem_strides, "VPU_CMX_NN">)
      weight_table(%weight_table : memref<16x1x1x4xsi32, #NHWC, #wt_mem_strides, "VPU_CMX_NN">)
      parent_input(%parent_input_1 : memref<1x16x16x16x!qtype, #NHWC, #act_mem_strides, "VPU_CMX_NN">)
      parent_output(%parent_output_1 : memref<1x16x16x16xf16, #NHWC, #act_mem_strides, "VPU_CMX_NN">)
      outputs(%output_1 : memref<1x16x16x16xf16, #NHWC, #act_mem_strides, "VPU_CMX_NN">)
      waits(%inputs_ready : !VPUIP.Barrier)
      updates(%conv_complete : !VPUIP.Barrier)
      -> memref<1x16x16x16xf16, #NHWC, #act_mem_strides, "VPU_CMX_NN">
      variants : {
        VPUIP.DPUTask {
          end = [15, 15, 15],
          mpe_mode = "CUBOID_16x16",
          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
          start = [0, 0, 0]
        }
      }
      PPE : {
      }

    VPUIP.NNDMA {port = 0}
      inputs(%output_0 : memref<1x16x16x16xf16, #NHWC, #act_mem_strides, "VPU_CMX_NN">)
      outputs(%output_ddr_0 : memref<1x16x16x16xf16, #NHWC, #act_mem_strides, "VPU_DDR_Heap">)
      waits(%conv_complete : !VPUIP.Barrier)
      updates(%output_ready : !VPUIP.Barrier)
      -> memref<1x16x16x16xf16, #NHWC, #act_mem_strides, "VPU_DDR_Heap">

    VPUIP.NNDMA {port = 0}
      inputs(%output_1 : memref<1x16x16x16xf16, #NHWC, #act_mem_strides, "VPU_CMX_NN">)
      outputs(%output_ddr_1 : memref<1x16x16x16xf16, #NHWC, #act_mem_strides, "VPU_DDR_Heap">)
      waits(%conv_complete : !VPUIP.Barrier)
      updates(%output_ready : !VPUIP.Barrier)
      -> memref<1x16x16x16xf16, #NHWC, #act_mem_strides, "VPU_DDR_Heap">

    VPUIP.NNDMA {port = 0}
      inputs(%output_ddr : memref<2x16x16x16xf16, #NHWC, #act_mem_strides, "VPU_DDR_Heap">)
      outputs(%output_arg : memref<2x16x16x16xf16, #NHWC, #act_mem_strides, "ProgrammableOutput">)
      waits(%output_ready : !VPUIP.Barrier)
      -> memref<2x16x16x16xf16, #NHWC, #act_mem_strides, "ProgrammableOutput">

    return %output_arg : memref<2x16x16x16xf16, #NHWC, #act_mem_strides, "ProgrammableOutput">
  }
}
