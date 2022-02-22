// RUN: vpux-translate --split-input-file --export-VPUIP -o %t %s && flatc --raw-binary --json %vpuip_schema_file% -- %t && FileCheck %s --input-file %basename_t.json

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ParentInputDistributed = type !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED"
}>

!ParentOutputDistributed = type !VPUIP.DistributedBuffer<
    1x32x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 4, 1, 1]
}>

!OutputDistributed = type !VPUIP.DistributedBuffer<
    1x16x32x32xf16, {
        order = #NHWC,
        strides = [32768, 1, 1024, 32]
    },
    @CMX_NN, {
    mode = "DUPLICATED"
}>

module @TestMultiClusterSOK attributes {VPU.arch = "KMB"} {

IE.MemoryResource 31457280 bytes of @DDR {VPU.bandwidth = 8, VPU.derateFactor = 6.000000e-01}
IE.MemoryResource 4194304 bytes of @CMX_UPA {VPU.bandwidth = 16, VPU.derateFactor = 8.500000e-01}
IE.MemoryResource 1048576 bytes of @CMX_NN {VPU.bandwidth = 32, VPU.derateFactor = 1.000000e+00}

module @UsedMemory {
    IE.MemoryResource 2048 bytes of @DDR
    IE.MemoryResource 1048576 bytes of @CMX_NN
}

IE.ExecutorResource 16 of @SHAVE_UPA
IE.ExecutorResource 4 of  @NCE {
    IE.ExecutorResource 5 of @DPU
}
IE.ExecutorResource 1 of @DMA_NN

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x16x32x32xf16>
    }
    outputsInfo : {
        DataInfo "conv" : tensor<1x32x32x32xf16>
    }

func @main(%arg0: memref<1x16x32x32xf16>, %arg1: memref<1x32x32x32xf16>) -> memref<1x32x32x32xf16> {

    %weights_cst1 = const.Declare memref<16x16x1x1xf16, #NHWC> =
        #const.Content<dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]>
    %weights_cst2 = const.Declare memref<16x16x1x1xf16, #NHWC> =
        #const.Content<dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]>
    %weights_table_cst1 = const.Declare memref<16x1x1x4xsi32> = #const.Content<dense<1> : tensor<16x1x1x4xsi32>>
    %weights_table_cst2 = const.Declare memref<16x1x1x4xsi32> = #const.Content<dense<1> : tensor<16x1x1x4xsi32>>

    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar10 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // DDR input buffer
    %zmajor_in = VPURT.DeclareBuffer "DDR" <0> -> memref<1x16x32x32xf16, #NHWC>

    // DDR output buffers for SOK tiling
    %parent_out_ddr = VPURT.DeclareBuffer "DDR" <32768> -> memref<1x32x32x32xf16, #NHWC, @DDR>

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer "CMX_NN" [0, 1] <0> -> !ParentInputDistributed
    %input_1 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>
    %input_2 = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 1]>

    %parent_out_cmx = VPURT.DeclareBuffer "CMX_NN" [0, 1] <32768> -> !ParentOutputDistributed
    %parent_out_cmx_compact = VPURT.DeclareBuffer "CMX_NN" [0] <32768> -> memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 0]>
    %output1 = VPURT.DeclareBuffer "CMX_NN" [0, 1] <32768> -> !OutputDistributed
    %output2 = VPURT.DeclareBuffer "CMX_NN" [0, 1] <32768> -> !OutputDistributed

    %weights1 = VPURT.DeclareBuffer "CMX_NN" [0] <98304> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %weights2 = VPURT.DeclareBuffer "CMX_NN" [1] <98304> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    %weight_table1 = VPURT.DeclareBuffer "CMX_NN" [0] <98816> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %weight_table2 = VPURT.DeclareBuffer "CMX_NN" [1] <98816> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>


    // Upload weights and weights table

    VPURT.Task updates(%bar10: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%weights_cst1: memref<16x16x1x1xf16, #NHWC>)
            outputs(%weights1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task updates(%bar10: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%weights_cst2: memref<16x16x1x1xf16, #NHWC>)
            outputs(%weights2: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
            -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    }

    VPURT.Task updates(%bar10: !VPURT.Barrier)  {
         VPUIP.NNDMA
            inputs(%weights_table_cst1: memref<16x1x1x4xsi32>)
            outputs(%weight_table1: memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    }

    VPURT.Task updates(%bar10: !VPURT.Barrier)  {
         VPUIP.NNDMA
            inputs(%weights_table_cst2: memref<16x1x1x4xsi32>)
            outputs(%weight_table2: memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
            -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    }

    // Reorder input

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.PermuteUPA {order_value = #NHWC}
            inputs(%arg0: memref<1x16x32x32xf16>)
            outputs(%zmajor_in: memref<1x16x32x32xf16, #NHWC>)
            -> memref<1x16x32x32xf16, #NHWC>
    }

    // Broadcast input

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%zmajor_in: memref<1x16x32x32xf16, #NHWC>)
            outputs(%parent_input_cmx: !ParentInputDistributed)
            -> !ParentInputDistributed
    }

    // 1st tile

    VPURT.Task waits(%bar1, %bar10: !VPURT.Barrier, !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
        VPUIP.NCEClusterTask {
                kernel_padding = {bottom = 0, left = 0, right = 0, top = 0},
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = "CONV"
            }
            input(%input_1: memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>)
            weights(%weights1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%weight_table1: memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%parent_input_cmx: !ParentInputDistributed)
            parent_output(%parent_out_cmx: !ParentOutputDistributed)
            outputs(%output1: !OutputDistributed)
            -> !OutputDistributed
            variants : {
                DPUTask {
                    start = [0, 0, 0],
                    end = [31, 31, 15],
                    pad = {bottom = 0, left = 0, right = 0, top = 0},
                    mpe_mode = "VECTOR_FP16"
                }
            } PPE : {
            }
    }

    // 2nd tile

    VPURT.Task waits(%bar1, %bar10: !VPURT.Barrier, !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
        VPUIP.NCEClusterTask {
                kernel_padding = {bottom = 0, left = 0, right = 0, top = 0},
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = "CONV"
            }
            input(%input_2: memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 1]>)
            weights(%weights2: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
            weight_table(%weight_table2: memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
            parent_input(%parent_input_cmx: !ParentInputDistributed)
            parent_output(%parent_out_cmx: !ParentOutputDistributed)
            outputs(%output2: !OutputDistributed)
            -> !OutputDistributed
            variants : {
                DPUTask {
                    start = [0, 0, 16],
                    end = [31, 31, 31],
                    pad = {bottom = 0, left = 0, right = 0, top = 0},
                    mpe_mode = "VECTOR_FP16"
                }
            } PPE : {
            }
    }

    // Copyback result

    VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%parent_out_cmx_compact: memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%parent_out_ddr: memref<1x32x32x32xf16, #NHWC, @DDR>)
            -> memref<1x32x32x32xf16, #NHWC, @DDR>
    }

    // Reorder output

    VPURT.Task waits(%bar3: !VPURT.Barrier) {
        VPUIP.PermuteUPA {order_value = #NCHW}
            inputs(%parent_out_ddr: memref<1x32x32x32xf16, #NHWC, @DDR>)
            outputs(%arg1: memref<1x32x32x32xf16>)
            -> memref<1x32x32x32xf16>
    }

    return %arg1: memref<1x32x32x32xf16>
}

}

// CHECK:   identifier: "TestMultiClusterSOK",
// CHECK:         task_type: "NCE2Task",
// CHECK:             mpe_frequent_mode: "VECTOR_FP16",
// CHECK:             parent_input_tensor: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 16,
// CHECK:                 32,
// CHECK:                 32
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 2.0,
// CHECK:                 32768.0,
// CHECK:                 2.0,
// CHECK:                 1024.0,
// CHECK:                 32.0
// CHECK:               ],
// CHECK:               data: {
// CHECK:                 data_index: 0
// CHECK:               },
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0,
// CHECK:                 1
// CHECK:               ],
// CHECK:               data_dtype: "FP16",
// CHECK:             parent_output_tensor: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 32,
// CHECK:                 32,
// CHECK:                 32
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 2.0,
// CHECK:                 65536.0,
// CHECK:                 2.0,
// CHECK:                 2048.0,
// CHECK:                 64.0
// CHECK:               ],
// CHECK:               data: {
// CHECK:                 data_index: 32768
// CHECK:               },
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0,
// CHECK:                 1
// CHECK:               ],
// CHECK:               data_dtype: "FP16",
// CHECK:             input_data: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 16,
// CHECK:                 32,
// CHECK:                 32
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 2.0,
// CHECK:                 32768.0,
// CHECK:                 2.0,
// CHECK:                 1024.0,
// CHECK:                 32.0
// CHECK:               ],
// CHECK:               data: {
// CHECK:                 data_index: 0
// CHECK:               },
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               data_dtype: "FP16",
// CHECK:             output_data: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 16,
// CHECK:                 32,
// CHECK:                 32
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 2.0,
// CHECK:                 65536.0,
// CHECK:                 2.0,
// CHECK:                 2048.0,
// CHECK:                 64.0
// CHECK:               ],
// CHECK:               data: {
// CHECK:                 data_index: 32768
// CHECK:               },
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0,
// CHECK:                 1
// CHECK:               ],
// CHECK:               data_dtype: "FP16",
// CHECK:             weights_data: {
// CHECK:               dimensions: [
// CHECK:                 16,
// CHECK:                 16,
// CHECK:                 1,
// CHECK:                 1
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 2.0,
// CHECK:                 32.0,
// CHECK:                 2.0,
// CHECK:                 32.0,
// CHECK:                 32.0
// CHECK:               ],
// CHECK:               data: {
// CHECK:                 data_index: 98304
// CHECK:               },
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               data_dtype: "FP16",
// CHECK:             weights_table: {
// CHECK:               dimensions: [
// CHECK:                 16,
// CHECK:                 1,
// CHECK:                 1,
// CHECK:                 4
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 4.0,
// CHECK:                 16.0,
// CHECK:                 16.0,
// CHECK:                 16.0,
// CHECK:                 4.0
// CHECK:               ],
// CHECK:               data: {
// CHECK:                 data_index: 98816
// CHECK:               },
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               data_dtype: "I32",
// CHECK:           variant: [
// CHECK:               mpe_mode: "VECTOR_FP16",
// CHECK:               workload_end_X: 31,
// CHECK:               workload_end_Y: 31,
// CHECK:               workload_end_Z: 15
// CHECK:         task_type: "NCE2Task",
// CHECK:             parent_input_tensor: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 16,
// CHECK:                 32,
// CHECK:                 32
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 2.0,
// CHECK:                 32768.0,
// CHECK:                 2.0,
// CHECK:                 1024.0,
// CHECK:                 32.0
// CHECK:               ],
// CHECK:               data: {
// CHECK:                 data_index: 0
// CHECK:               },
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0,
// CHECK:                 1
// CHECK:               ],
// CHECK:               data_dtype: "FP16",
// CHECK:             parent_output_tensor: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 32,
// CHECK:                 32,
// CHECK:                 32
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 2.0,
// CHECK:                 65536.0,
// CHECK:                 2.0,
// CHECK:                 2048.0,
// CHECK:                 64.0
// CHECK:               ],
// CHECK:               data: {
// CHECK:                 data_index: 32768
// CHECK:               },
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0,
// CHECK:                 1
// CHECK:               ],
// CHECK:               data_dtype: "FP16",
// CHECK:             input_data: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 16,
// CHECK:                 32,
// CHECK:                 32
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 2.0,
// CHECK:                 32768.0,
// CHECK:                 2.0,
// CHECK:                 1024.0,
// CHECK:                 32.0
// CHECK:               ],
// CHECK:               data: {
// CHECK:                 data_index: 0
// CHECK:               },
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 1
// CHECK:               ],
// CHECK:               data_dtype: "FP16",
// CHECK:             output_data: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 16,
// CHECK:                 32,
// CHECK:                 32
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 2.0,
// CHECK:                 65536.0,
// CHECK:                 2.0,
// CHECK:                 2048.0,
// CHECK:                 64.0
// CHECK:               ],
// CHECK:               data: {
// CHECK:                 data_index: 32768
// CHECK:               },
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0,
// CHECK:                 1
// CHECK:               ],
// CHECK:               data_dtype: "FP16",
// CHECK:             weights_data: {
// CHECK:               dimensions: [
// CHECK:                 16,
// CHECK:                 16,
// CHECK:                 1,
// CHECK:                 1
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 2.0,
// CHECK:                 32.0,
// CHECK:                 2.0,
// CHECK:                 32.0,
// CHECK:                 32.0
// CHECK:               ],
// CHECK:               data: {
// CHECK:                 data_index: 98304
// CHECK:               },
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 1
// CHECK:               ],
// CHECK:             weights_table: {
// CHECK:               dimensions: [
// CHECK:                 16,
// CHECK:                 1,
// CHECK:                 1,
// CHECK:                 4
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 4.0,
// CHECK:                 16.0,
// CHECK:                 16.0,
// CHECK:                 16.0,
// CHECK:                 4.0
// CHECK:               ],
// CHECK:               data: {
// CHECK:                 data_index: 98816
// CHECK:               },
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 1
// CHECK:               ],
// CHECK:               data_dtype: "I32",
// CHECK:           variant: [
// CHECK:               mpe_mode: "VECTOR_FP16",
// CHECK:               workload_start_Z: 16,
// CHECK:               workload_end_X: 31,
// CHECK:               workload_end_Y: 31,
// CHECK:               workload_end_Z: 31
// CHECK:         task_type: "NNDMATask",
// CHECK:         task: {
// CHECK:           src: {
// CHECK:             dimensions: [
// CHECK:               1,
// CHECK:               16,
// CHECK:               32,
// CHECK:               32
// CHECK:             ],
// CHECK:             strides: [
// CHECK:               2.0,
// CHECK:               32768.0,
// CHECK:               2.0,
// CHECK:               1024.0,
// CHECK:               32.0
// CHECK:             ],
// CHECK:             data: {
// CHECK:               data_index: 0
// CHECK:             },
// CHECK:             locale: "VPU_DDR_Heap",
// CHECK:             locale_index: [
// CHECK:               0
// CHECK:             ],
// CHECK:             data_dtype: "FP16",
// CHECK:           dst: {
// CHECK:             dimensions: [
// CHECK:               1,
// CHECK:               16,
// CHECK:               32,
// CHECK:               32
// CHECK:             ],
// CHECK:             strides: [
// CHECK:               2.0,
// CHECK:               32768.0,
// CHECK:               2.0,
// CHECK:               1024.0,
// CHECK:               32.0
// CHECK:             ],
// CHECK:             data: {
// CHECK:               data_index: 0
// CHECK:             },
// CHECK:             locale: "VPU_CMX_NN",
// CHECK:             locale_index: [
// CHECK:               0,
// CHECK:               1
// CHECK:             ],
// CHECK:             data_dtype: "FP16",
// CHECK:         task_type: "NNDMATask",
// CHECK:         task: {
// CHECK:           src: {
// CHECK:             dimensions: [
// CHECK:               1,
// CHECK:               32,
// CHECK:               32,
// CHECK:               32
// CHECK:             ],
// CHECK:             strides: [
// CHECK:               2.0,
// CHECK:               65536.0,
// CHECK:               2.0,
// CHECK:               2048.0,
// CHECK:               64.0
// CHECK:             ],
// CHECK:             data: {
// CHECK:               data_index: 32768
// CHECK:             },
// CHECK:             locale: "VPU_CMX_NN",
// CHECK:             locale_index: [
// CHECK:               0
// CHECK:             ],
// CHECK:             data_dtype: "FP16",
// CHECK:           },
// CHECK:           dst: {
// CHECK:             dimensions: [
// CHECK:               1,
// CHECK:               32,
// CHECK:               32,
// CHECK:               32
// CHECK:             ],
// CHECK:             strides: [
// CHECK:               2.0,
// CHECK:               65536.0,
// CHECK:               2.0,
// CHECK:               2048.0,
// CHECK:               64.0
// CHECK:             ],
// CHECK:             data: {
// CHECK:               data_index: 32768
// CHECK:             },
// CHECK:             locale: "VPU_DDR_Heap",
// CHECK:             locale_index: [
// CHECK:               0
// CHECK:             ],
// CHECK:             data_dtype: "FP16",
