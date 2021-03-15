// RUN: vpux-translate --export-VPUIP -o %t %s && flatc --raw-binary --json %vpuip_schema_file% -- %t && FileCheck %s --input-file %basename_t.json

#NHWC = affine_map<(n, c, h, w) -> (n, h, w, c)>
#OXYI = affine_map<(o, i, x, y) -> (o, x, y, i)>

module @Test attributes {VPUIP.arch = "MA2490"} {

IERT.RunTimeResources
    availableMemory : {
        IERT.MemoryResource 1073741824 bytes
        IERT.MemoryResource 31457280 bytes of "DDR" {VPUIP.bandwidth = 8 : i64, VPUIP.derateFactor = 6.000000e-01 : f64}
        IERT.MemoryResource 4194304 bytes of "CMX_UPA" {VPUIP.bandwidth = 16 : i64, VPUIP.derateFactor = 8.500000e-01 : f64}
        IERT.MemoryResource 1048576 bytes of "CMX_NN" {VPUIP.bandwidth = 32 : i64, VPUIP.derateFactor = 1.000000e+00 : f64}
    }
    usedMemory : {
        IERT.MemoryResource 4096 bytes of "DDR"
        IERT.MemoryResource 917504 bytes of "CMX_NN"
    }
    availableExecutors : {
        IERT.ExecutorResource 2 of "ARM" {}
        IERT.ExecutorResource 1 of "Leon_RT" {}
        IERT.ExecutorResource 1 of "Leon_NN" {}
        IERT.ExecutorResource 16 of "SHAVE_UPA" {}
        IERT.ExecutorResource 20 of "SHAVE_NN" {}
        IERT.ExecutorResource 4 of "NCE_Cluster" {
            IERT.ExecutorResource 5 of "NCE_PerClusterDPU" {}
        }
        IERT.ExecutorResource 1 of "DMA_UPA" {}
        IERT.ExecutorResource 1 of "DMA_NN" {}
    }
    usedExecutors : {
        IERT.ExecutorResource 2 of "ARM" {}
        IERT.ExecutorResource 1 of "Leon_NN" {}
        IERT.ExecutorResource 1 of "Leon_RT" {}
        IERT.ExecutorResource 0 of "SHAVE_NN" {}
        IERT.ExecutorResource 16 of "SHAVE_UPA" {}
        IERT.ExecutorResource 5 of "NCE_PerClusterDPU" {
            IERT.ExecutorResource 4 of "NCE_Cluster" {}
        }
    }

VPUIP.Graph
    options : "NONE"
    version : {
        majorV = 3 : i32,
        minorV = 11 : i32,
        patchV = 0 : i32, hash = "",
        contextStr = "VPUX Compiler"
    }

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "input" : memref<1x16x16x16xui8, #NHWC, "ProgrammableInput">
    }
    outputsInfo : {
        IE.DataInfo "output" : memref<1x16x16x16xf16, #NHWC, "ProgrammableOutput">
    }

func @main(%input_ddr : memref<1x16x16x16xui8, #NHWC, "ProgrammableInput">, %output : memref<1x16x16x16xf16, #NHWC, "ProgrammableOutput">) {

    //
    // DDR Tensors
    //

    // 'weight_data_ddr' is assumed to be folded into a dequantized ui8 type.
    %weight_data_ddr = VPUIP.DeclareConstantTensor memref<16x1x1x16xui8, #OXYI, "GraphFile"> =
        dense<[
                  [[[  0,  34, 193, 117, 136,  56,  12, 173, 173, 238,  98, 132, 212,   9,  14, 135]]],
                  [[[171,   2,  98,  17, 106, 175, 150, 237, 216, 134,  23, 167, 106, 179, 232, 194]]],
                  [[[ 67,  12, 188,  84, 161, 193, 253,  93,  63, 251, 184, 192, 166,  19, 161, 226]]],
                  [[[ 70, 111, 195, 122,  61,  70,  92,  42, 124, 229, 232,  15, 231, 129, 132,  81]]],
                  [[[252, 126,  68,  23, 242,  19, 128,  98,  71, 233, 135, 118, 240,  13, 194, 196]]],
                  [[[211,  32,   4, 176, 221, 161, 188, 185, 255, 227,  59,  78,  90, 131, 151, 216]]],
                  [[[105, 215,  69, 106, 137, 119,  73,  45,  39, 146, 205,   8, 136, 127, 244, 191]]],
                  [[[141, 227, 159, 215,  41,  54, 182,  33,  23,  70,   1, 106,   7, 181, 239,  61]]],
                  [[[ 46,  81, 226, 166,  38, 174,  98,  99, 127,  38, 150, 216, 150, 244, 142,  38]]],
                  [[[251, 104,  36, 144,  64, 125, 118, 245,  32,  51,  81, 160,  32, 166, 159, 205]]],
                  [[[ 63, 121,  99,  52,   7, 230, 109,  36, 242, 105,  33, 226,  24,  41,  18,  93]]],
                  [[[ 65,  34, 200, 116,  89, 115, 206, 238, 166,  55, 173, 232,  64, 220, 120, 129]]],
                  [[[153, 208, 193, 118, 243, 161, 112, 210, 176, 179, 252, 243, 217,  74, 137, 131]]],
                  [[[ 26, 106, 147, 224, 112, 186, 222, 182, 204, 180, 189,   5, 226, 134, 118,  17]]],
                  [[[182, 125, 170, 174,  51, 234, 221, 227, 139,  35, 115, 252,  55, 114,  81, 131]]],
                  [[[225, 112, 119, 206,  93,  54, 255,  39, 161, 157,   0,   0, 197, 185,  81, 107]]]
        ]> : tensor<16x1x1x16xui8>
    %weight_table_ddr = VPUIP.DeclareConstantTensor memref<16x1x1x4xsi32, "GraphFile"> =
        dense<[
                  [[[      8448,   16777215, 1077958144,          0]]],
                  [[[      8464,   16777215, 1077958144,          0]]],
                  [[[      8480,   16777215, 1077958144,          0]]],
                  [[[      8496,   16777215, 1077958144,          0]]],
                  [[[      8512,   16777215, 1077958144,          0]]],
                  [[[      8528,   16777215, 1077958144,          0]]],
                  [[[      8544,   16777215, 1077958144,          0]]],
                  [[[      8560,   16777215, 1077958144,          0]]],
                  [[[      8576,   16777215, 1077958144,          0]]],
                  [[[      8592,   16777215, 1077958144,          0]]],
                  [[[      8608,   16777215, 1077958144,          0]]],
                  [[[      8624,   16777215, 1077958144,          0]]],
                  [[[      8640,   16777215, 1077958144,          0]]],
                  [[[      8656,   16777215, 1077958144,          0]]],
                  [[[      8672,   16777215, 1077958144,          0]]],
                  [[[      8688,   16777215, 1077958144,          0]]]
        ]> : tensor<16x1x1x4xsi32>
    %output_ddr = VPUIP.DeclareTensor "VPU_DDR_Heap" <0> -> memref<1x16x16x16x!quant.uniform<u8:f16, 1.0>, #NHWC, "DDR">

    //
    // NN_CMX Tensors
    //

    %weight_data_cmx = VPUIP.DeclareTensor "VPU_CMX_NN" [0] <8448> -> memref<16x1x1x16xui8, #OXYI, "CMX_NN">
    %weight_table_cmx = VPUIP.DeclareTensor "VPU_CMX_NN" [0] <8192> -> memref<16x1x1x4xsi32, "CMX_NN">
    %input_cmx = VPUIP.DeclareTensor "VPU_CMX_NN" [0] <4096> -> memref<1x16x16x16xui8, #NHWC, "CMX_NN">
    %output_cmx = VPUIP.DeclareTensor "VPU_CMX_NN" [0] <0> -> memref<1x16x16x16x!quant.uniform<u8:f16, 1.0>, #NHWC, "CMX_NN">

    //
    // Barriers
    //

    %0 = VPUIP.ConfigureBarrier <0> -> !VPUIP.Barrier
    %1 = VPUIP.ConfigureBarrier <1> -> !VPUIP.Barrier

    //
    // DMAs
    //

    VPUIP.NNDMA
        inputs(%weight_data_ddr : memref<16x1x1x16xui8, #OXYI, "GraphFile">)
        outputs(%weight_data_cmx : memref<16x1x1x16xui8, #OXYI, "CMX_NN">)
    VPUIP.NNDMA
        inputs(%weight_table_ddr : memref<16x1x1x4xsi32, "GraphFile">)
        outputs(%weight_table_cmx : memref<16x1x1x4xsi32, "CMX_NN">)
        updates(%0 : !VPUIP.Barrier)
    VPUIP.NNDMA
        inputs(%input_ddr : memref<1x16x16x16xui8, #NHWC, "ProgrammableInput">)
        outputs(%input_cmx : memref<1x16x16x16xui8, #NHWC, "CMX_NN">)
    VPUIP.NNDMA
        inputs(%output_cmx : memref<1x16x16x16x!quant.uniform<u8:f16, 1.0>, #NHWC, "CMX_NN">)
        outputs(%output_ddr : memref<1x16x16x16x!quant.uniform<u8:f16, 1.0>, #NHWC, "DDR">)
        waits(%1: !VPUIP.Barrier)

    //
    // NCEClusterTask Parent Tensors
    //

    %parent_input_cmx = VPUIP.DeclareTensor "VPU_CMX_NN" [0] <4096> -> memref<1x16x16x16xui8, #NHWC, "CMX_NN">
    %parent_output_cmx = VPUIP.DeclareTensor "VPU_CMX_NN" [0] <0> -> memref<1x16x16x16x!quant.uniform<u8:f16, 0.0039215686274509803>, #NHWC, "CMX_NN">

    //
    // NCEClusterTask
    //

    // NCEClusterTasks using 'Clustering' split strategy
    VPUIP.NCEClusterTask { task_type = "CONV", strides = [1 : i32, 1 : i32], kernel_padding = [0 : i32, 0 : i32, 0 : i32, 0 : i32] }
        inputs(
            %input_cmx : memref<1x16x16x16xui8, #NHWC, "CMX_NN">,
            %weight_data_cmx : memref<16x1x1x16xui8, #OXYI, "CMX_NN">,
            %weight_table_cmx : memref<16x1x1x4xsi32, "CMX_NN">
        )
        outputs(%output_cmx : memref<1x16x16x16x!quant.uniform<u8:f16, 1.0>, #NHWC, "CMX_NN">)
        waits(%0 : !VPUIP.Barrier)
        updates(%1 : !VPUIP.Barrier)
        parent_input(%parent_input_cmx : memref<1x16x16x16xui8, #NHWC, "CMX_NN">)
        parent_output(%parent_output_cmx : memref<1x16x16x16x!quant.uniform<u8:f16, 0.0039215686274509803>, #NHWC, "CMX_NN">)
        variants : {
            VPUIP.DPUTask {
                start = [0 : i32, 0 : i32, 0 : i32],
                end = [15 : i32, 3 : i32, 15 : i32],
                pads_begin = [0 : i32, 0 : i32],
                pads_end = [0 : i32, 0 : i32],
                mpe_mode = "MATRIX"
            }
            VPUIP.DPUTask {
                start = [0 : i32, 4 : i32, 0 : i32],
                end = [15 : i32, 7 : i32, 15 : i32],
                pads_begin = [0 : i32, 0 : i32],
                pads_end = [0 : i32, 0 : i32],
                mpe_mode = "MATRIX"
            }
            VPUIP.DPUTask {
                start = [0 : i32, 8 : i32, 0 : i32],
                end = [15 : i32, 11 : i32, 15 : i32],
                pads_begin = [0 : i32, 0 : i32],
                pads_end = [0 : i32, 0 : i32],
                mpe_mode = "MATRIX"
            }
            VPUIP.DPUTask {
                start = [0 : i32, 12 : i32, 0 : i32],
                end = [15 : i32, 15 : i32, 15 : i32],
                pads_begin = [0 : i32, 0 : i32],
                pads_end = [0 : i32, 0 : i32],
                mpe_mode = "MATRIX"
            }
        }

    VPUIP.NCEClusterTask { task_type = "CONV", strides = [1 : i32, 1 : i32], kernel_padding = [0 : i32, 0 : i32, 0 : i32, 0 : i32] }
        inputs(
            %input_cmx : memref<1x16x16x16xui8, #NHWC, "CMX_NN">,
            %weight_data_cmx : memref<16x1x1x16xui8, #OXYI, "CMX_NN">,
            %weight_table_cmx : memref<16x1x1x4xsi32, "CMX_NN">
        )
        outputs(%output_cmx : memref<1x16x16x16x!quant.uniform<u8:f16, 1.0>, #NHWC, "CMX_NN">)
        waits(%0 : !VPUIP.Barrier)
        updates(%1 : !VPUIP.Barrier)
        parent_input(%parent_input_cmx : memref<1x16x16x16xui8, #NHWC, "CMX_NN">)
        parent_output(%parent_output_cmx : memref<1x16x16x16x!quant.uniform<u8:f16, 0.0039215686274509803>, #NHWC, "CMX_NN">)
        variants : {
            VPUIP.DPUTask {
                start = [0 : i32, 0 : i32, 0 : i32],
                end = [15 : i32, 3 : i32, 15 : i32],
                pads_begin = [0 : i32, 0 : i32],
                pads_end = [0 : i32, 0 : i32],
                mpe_mode = "MATRIX"
            }
            VPUIP.DPUTask {
                start = [0 : i32, 4 : i32, 0 : i32],
                end = [15 : i32, 7 : i32, 15 : i32],
                pads_begin = [0 : i32, 0 : i32],
                pads_end = [0 : i32, 0 : i32],
                mpe_mode = "MATRIX"
            }
            VPUIP.DPUTask {
                start = [0 : i32, 8 : i32, 0 : i32],
                end = [15 : i32, 11 : i32, 15 : i32],
                pads_begin = [0 : i32, 0 : i32],
                pads_end = [0 : i32, 0 : i32],
                mpe_mode = "MATRIX"
            }
            VPUIP.DPUTask {
                start = [0 : i32, 12 : i32, 0 : i32],
                end = [15 : i32, 15 : i32, 15 : i32],
                pads_begin = [0 : i32, 0 : i32],
                pads_end = [0 : i32, 0 : i32],
                mpe_mode = "MATRIX"
            }
        }

    VPUIP.NCEClusterTask { task_type = "CONV", strides = [1 : i32, 1 : i32], kernel_padding = [0 : i32, 0 : i32, 0 : i32, 0 : i32] }
        inputs(
            %input_cmx : memref<1x16x16x16xui8, #NHWC, "CMX_NN">,
            %weight_data_cmx : memref<16x1x1x16xui8, #OXYI, "CMX_NN">,
            %weight_table_cmx : memref<16x1x1x4xsi32, "CMX_NN">
        )
        outputs(%output_cmx : memref<1x16x16x16x!quant.uniform<u8:f16, 1.0>, #NHWC, "CMX_NN">)
        waits(%0 : !VPUIP.Barrier)
        updates(%1 : !VPUIP.Barrier)
        parent_input(%parent_input_cmx : memref<1x16x16x16xui8, #NHWC, "CMX_NN">)
        parent_output(%parent_output_cmx : memref<1x16x16x16x!quant.uniform<u8:f16, 0.0039215686274509803>, #NHWC, "CMX_NN">)
        variants : {
            VPUIP.DPUTask {
                start = [0 : i32, 0 : i32, 0 : i32],
                end = [15 : i32, 3 : i32, 15 : i32],
                pads_begin = [0 : i32, 0 : i32],
                pads_end = [0 : i32, 0 : i32],
                mpe_mode = "MATRIX"
            }
            VPUIP.DPUTask {
                start = [0 : i32, 4 : i32, 0 : i32],
                end = [15 : i32, 7 : i32, 15 : i32],
                pads_begin = [0 : i32, 0 : i32],
                pads_end = [0 : i32, 0 : i32],
                mpe_mode = "MATRIX"
            }
            VPUIP.DPUTask {
                start = [0 : i32, 8 : i32, 0 : i32],
                end = [15 : i32, 11 : i32, 15 : i32],
                pads_begin = [0 : i32, 0 : i32],
                pads_end = [0 : i32, 0 : i32],
                mpe_mode = "MATRIX"
            }
            VPUIP.DPUTask {
                start = [0 : i32, 12 : i32, 0 : i32],
                end = [15 : i32, 15 : i32, 15 : i32],
                pads_begin = [0 : i32, 0 : i32],
                pads_end = [0 : i32, 0 : i32],
                mpe_mode = "MATRIX"
            }
        }

    VPUIP.NCEClusterTask { task_type = "CONV", strides = [1 : i32, 1 : i32], kernel_padding = [0 : i32, 0 : i32, 0 : i32, 0 : i32] }
        inputs(
            %input_cmx : memref<1x16x16x16xui8, #NHWC, "CMX_NN">,
            %weight_data_cmx : memref<16x1x1x16xui8, #OXYI, "CMX_NN">,
            %weight_table_cmx : memref<16x1x1x4xsi32, "CMX_NN">
        )
        outputs(%output_cmx : memref<1x16x16x16x!quant.uniform<u8:f16, 1.0>, #NHWC, "CMX_NN">)
        waits(%0 : !VPUIP.Barrier)
        updates(%1 : !VPUIP.Barrier)
        parent_input(%parent_input_cmx : memref<1x16x16x16xui8, #NHWC, "CMX_NN">)
        parent_output(%parent_output_cmx : memref<1x16x16x16x!quant.uniform<u8:f16, 0.0039215686274509803>, #NHWC, "CMX_NN">)
        variants : {
            VPUIP.DPUTask {
                start = [0 : i32, 0 : i32, 0 : i32],
                end = [15 : i32, 3 : i32, 15 : i32],
                pads_begin = [0 : i32, 0 : i32],
                pads_end = [0 : i32, 0 : i32],
                mpe_mode = "MATRIX"
            }
            VPUIP.DPUTask {
                start = [0 : i32, 4 : i32, 0 : i32],
                end = [15 : i32, 7 : i32, 15 : i32],
                pads_begin = [0 : i32, 0 : i32],
                pads_end = [0 : i32, 0 : i32],
                mpe_mode = "MATRIX"
            }
            VPUIP.DPUTask {
                start = [0 : i32, 8 : i32, 0 : i32],
                end = [15 : i32, 11 : i32, 15 : i32],
                pads_begin = [0 : i32, 0 : i32],
                pads_end = [0 : i32, 0 : i32],
                mpe_mode = "MATRIX"
            }
            VPUIP.DPUTask {
                start = [0 : i32, 12 : i32, 0 : i32],
                end = [15 : i32, 15 : i32, 15 : i32],
                pads_begin = [0 : i32, 0 : i32],
                pads_end = [0 : i32, 0 : i32],
                mpe_mode = "MATRIX"
            }
        }

    //
    // UPATasks
    //

    VPUIP.QuantCastUPA { isTrailingSWLayer }
        inputs(%output_ddr : memref<1x16x16x16x!quant.uniform<u8:f16, 1.0>, #NHWC, "DDR">)
        outputs(%output : memref<1x16x16x16xf16, #NHWC, "ProgrammableOutput">)

    return
}

}

// CHECK:   identifier: "Test"

// CHECK:   net_input: [
// CHECK:     {
// CHECK:       name: "input",
// CHECK:       dimensions: [
// CHECK:           1,
// CHECK:           16,
// CHECK:           16,
// CHECK:           16
// CHECK:       ],
// CHECK:       strides: [
// CHECK:           1.0,
// CHECK:           4096.0,
// CHECK:           1.0,
// CHECK:           256.0,
// CHECK:           16.0
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "U8",
// CHECK:     }
// CHECK:   ],

// CHECK:   net_output: [
// CHECK:     {
// CHECK:       name: "output",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         16,
// CHECK:         16,
// CHECK:         16
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         2.0,
// CHECK:         8192.0,
// CHECK:         2.0,
// CHECK:         512.0,
// CHECK:         32.0
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP16",
// CHECK:     }
// CHECK:   ],

// CHECK:   task_count: 11

// CHECK:   options: [
// CHECK:   ],

// CHECK:   in_tensor_desc: [
// CHECK:     {
// CHECK:       name: "input",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         16,
// CHECK:         16,
// CHECK:         16
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         1.0,
// CHECK:         4096.0,
// CHECK:         1.0,
// CHECK:         256.0,
// CHECK:         16.0
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "U8"
// CHECK:     }
// CHECK:   ],

// CHECK:   out_tensor_desc: [
// CHECK:     {
// CHECK:       name: "output",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         16,
// CHECK:         16,
// CHECK:         16
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         2.0,
// CHECK:         8192.0,
// CHECK:         2.0,
// CHECK:         512.0,
// CHECK:         32.0
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP16"
// CHECK:     }
// CHECK:   ]

// CHECK:   task_lists: [

// CHECK:     {
// CHECK:       content: [
// CHECK:         {
// CHECK:           associated_barriers: {
// CHECK:             wait_barriers: [
// CHECK:             ],
// CHECK:             update_barriers: [
// CHECK:             ],
// CHECK:             virtual_wait_barriers: [
// CHECK:             ],
// CHECK:             virtual_update_barriers: [
// CHECK:             ]
// CHECK:           },
// CHECK:           task_type: "UPALayerTask",
// CHECK:           task: {
// CHECK:             softLayerParams_type: "QuantizeParams",
// CHECK:             softLayerParams: {
// CHECK:               scale: [
// CHECK:                 15360
// CHECK:               ],
// CHECK:               zero: [
// CHECK:                 0
// CHECK:               ]
// CHECK:             },
// CHECK:             inputs: [
// CHECK:               {
// CHECK:                 dimensions: [
// CHECK:                   1,
// CHECK:                   16,
// CHECK:                   16,
// CHECK:                   16
// CHECK:                 ],
// CHECK:                 strides: [
// CHECK:                   1.0,
// CHECK:                   4096.0,
// CHECK:                   1.0,
// CHECK:                   256.0,
// CHECK:                   16.0
// CHECK:                 ],
// CHECK:                 data: {
// CHECK:                   data_index: 0
// CHECK:                 }
// CHECK:                 locale: "VPU_DDR_Heap"
// CHECK:                 locale_index: [
// CHECK:                   0
// CHECK:                 ],
// CHECK:                 data_dtype: "U8"
// CHECK:                 quant_zero: [
// CHECK:                   0
// CHECK:                 ],
// CHECK:                 quant_mult: [
// CHECK:                   16384
// CHECK:                 ],
// CHECK:                 quant_shift: [
// CHECK:                   14
// CHECK:                 ],
// CHECK:                 order: 4930
// CHECK:               }
// CHECK:             ],
// CHECK:             outputs: [
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 16,
// CHECK:                 16,
// CHECK:                 16
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 2.0,
// CHECK:                 8192.0,
// CHECK:                 2.0,
// CHECK:                 512.0,
// CHECK:                 32.0
// CHECK:               ],
// CHECK:               data: {
// CHECK:                 data_index: 0
// CHECK:               }
// CHECK:               locale: "ProgrammableOutput"
// CHECK:               locale_index: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               data_dtype: "FP16"
// CHECK:               quant_zero: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               quant_mult: [
// CHECK:                 1
// CHECK:               ],
// CHECK:               quant_shift: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               order: 4930
// CHECK:             ],
// CHECK:             isTrailingSWLayer: true
// CHECK:           }
// CHECK:         }
// CHECK:       ]
// CHECK:     },

// CHECK:     {
// CHECK:       content: [
// CHECK:         {
// CHECK:           associated_barriers: {
// CHECK:             wait_barriers: [
// CHECK:               0
// CHECK:             ],
// CHECK:             update_barriers: [
// CHECK:               1
// CHECK:             ],
// CHECK:             virtual_wait_barriers: [
// CHECK:               0
// CHECK:             ],
// CHECK:             virtual_update_barriers: [
// CHECK:               1
// CHECK:             ]
// CHECK:           },
// CHECK:           task_type: "NCE2Task",
// CHECK:           task: {
// CHECK:             invariant: {
// CHECK:               ppe_task: {
// CHECK:                 fixed_function: {
// CHECK:                   Ops: [
// CHECK:                   ]
// CHECK:                 }
// CHECK:               },
// CHECK:               input_data: {
// CHECK:                 dimensions: [
// CHECK:                   1,
// CHECK:                   16,
// CHECK:                   16,
// CHECK:                   16
// CHECK:                 ],
// CHECK:                 strides: [
// CHECK:                   1.0,
// CHECK:                   4096.0,
// CHECK:                   1.0,
// CHECK:                   256.0,
// CHECK:                   16.0
// CHECK:                 ],
// CHECK:                 data: {
// CHECK:                   data_index: 4096
// CHECK:                 }
// CHECK:                 locale: "VPU_CMX_NN"
// CHECK:                 locale_index: [
// CHECK:                   0
// CHECK:                 ],
// CHECK:                 data_dtype: "U8"
// CHECK:                 quant_zero: [
// CHECK:                   0
// CHECK:                 ],
// CHECK:                 quant_mult: [
// CHECK:                   1
// CHECK:                 ],
// CHECK:                 quant_shift: [
// CHECK:                   0
// CHECK:                 ],
// CHECK:                 order: 4930
// CHECK:               },
// CHECK:               output_data: {
// CHECK:                 dimensions: [
// CHECK:                   1,
// CHECK:                   16,
// CHECK:                   16,
// CHECK:                   16
// CHECK:                 ],
// CHECK:                 strides: [
// CHECK:                   1.0,
// CHECK:                   4096.0,
// CHECK:                   1.0,
// CHECK:                   256.0,
// CHECK:                   16.0
// CHECK:                 ],
// CHECK:                 data: {
// CHECK:                   data_index: 4096
// CHECK:                 }
// CHECK:                 locale: "VPU_CMX_NN"
// CHECK:                 locale_index: [
// CHECK:                   0
// CHECK:                 ],
// CHECK:                 data_dtype: "U8"
// CHECK:                 quant_zero: [
// CHECK:                   0
// CHECK:                 ],
// CHECK:                 quant_mult: [
// CHECK:                   16384
// CHECK:                 ],
// CHECK:                 quant_shift: [
// CHECK:                   0
// CHECK:                 ],
// CHECK:                 order: 4930
// CHECK:               },
// CHECK:               weights_data: {
// CHECK:                 dimensions: [
// CHECK:                   16,
// CHECK:                   1,
// CHECK:                   1,
// CHECK:                   16
// CHECK:                 ],
// CHECK:                 strides: [
// CHECK:                   1.0,
// CHECK:                   16.0,
// CHECK:                   1.0,
// CHECK:                   16.0,
// CHECK:                   1.0
// CHECK:                 ],
// CHECK:                 data: {
// CHECK:                   data_index: 8448
// CHECK:                 }
// CHECK:                 locale: "VPU_CMX_NN"
// CHECK:                 locale_index: [
// CHECK:                   0
// CHECK:                 ],
// CHECK:                 data_dtype: "U8"
// CHECK:                 quant_zero: [
// CHECK:                   0
// CHECK:                 ],
// CHECK:                 quant_mult: [
// CHECK:                   1
// CHECK:                 ],
// CHECK:                 quant_shift: [
// CHECK:                   0
// CHECK:                 ],
// CHECK:                 order: 4930
// CHECK:               },
// CHECK:               weights_table: {
// CHECK:                 dimensions: [
// CHECK:                   16,
// CHECK:                   1,
// CHECK:                   1,
// CHECK:                   4 
// CHECK:                 ],
// CHECK:                 strides: [
// CHECK:                   4.0,
// CHECK:                   16.0,
// CHECK:                   16.0,
// CHECK:                   16.0,
// CHECK:                   4.0
// CHECK:                 ],
// CHECK:                 data: {
// CHECK:                   data_index: 8192
// CHECK:                 }
// CHECK:                 locale: "VPU_CMX_NN"
// CHECK:                 locale_index: [
// CHECK:                   0
// CHECK:                 ],
// CHECK:                 data_dtype: "I32"
// CHECK:                 quant_zero: [
// CHECK:                   0
// CHECK:                 ],
// CHECK:                 quant_mult: [
// CHECK:                   1
// CHECK:                 ],
// CHECK:                 quant_shift: [
// CHECK:                   0
// CHECK:                 ],
// CHECK:                 order: 4660
// CHECK:               }
// CHECK:             },
// CHECK:             variant: [
// CHECK:               {
// CHECK:                 mpe_mode: "MATRIX",
// CHECK:                 workload_end_X: 15,
// CHECK:                 workload_end_Y: 3,
// CHECK:                 workload_end_Z: 15
// CHECK:               },
// CHECK:               {
// CHECK:                 mpe_mode: "MATRIX",
// CHECK:                 workload_start_Y: 4,
// CHECK:                 workload_end_X: 15,
// CHECK:                 workload_end_Y: 7,
// CHECK:                 workload_end_Z: 15
// CHECK:               },
// CHECK:               {
// CHECK:                 mpe_mode: "MATRIX",
// CHECK:                 workload_start_Y: 8,
// CHECK:                 workload_end_X: 15,
// CHECK:                 workload_end_Y: 11,
// CHECK:                 workload_end_Z: 15
// CHECK:               },
// CHECK:               {
// CHECK:                 mpe_mode: "MATRIX",
// CHECK:                 workload_start_Y: 12,
// CHECK:                 workload_end_X: 15,
// CHECK:                 workload_end_Y: 15,
// CHECK:                 workload_end_Z: 15
// CHECK:               }
// CHECK:             ]
// CHECK:           }
// CHECK:         }
// CHECK:       ]
// CHECK:     },

// CHECK:     {
// CHECK:       content: [
// CHECK:         {
// CHECK:           associated_barriers: {
// CHECK:             wait_barriers: [
// CHECK:             ],
// CHECK:             update_barriers: [
// CHECK:             ],
// CHECK:             virtual_wait_barriers: [
// CHECK:             ],
// CHECK:             virtual_update_barriers: [
// CHECK:             ]
// CHECK:           },
// CHECK:           task_type: "ControllerTask",
// CHECK:           task: {
// CHECK:             task_type: "BarrierConfigurationTask",
// CHECK:             task: {
// CHECK:               target: {
// CHECK:                 consumer_count: 16,
// CHECK:                 producer_count: 1
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:         },
// CHECK:         {
// CHECK:           associated_barriers: {
// CHECK:             wait_barriers: [
// CHECK:             ],
// CHECK:             update_barriers: [
// CHECK:             ],
// CHECK:             virtual_wait_barriers: [
// CHECK:             ],
// CHECK:             virtual_update_barriers: [
// CHECK:             ]
// CHECK:           },
// CHECK:           task_type: "ControllerTask",
// CHECK:           task: {
// CHECK:             task_type: "BarrierConfigurationTask",
// CHECK:             task: {
// CHECK:               target: {
// CHECK:                 consumer_count: 1,
// CHECK:                 producer_count: 16
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK:       ]
// CHECK:     },

// CHECK:     {
// CHECK:       content: [
// CHECK:         {
// CHECK:           associated_barriers: {
// CHECK:             wait_barriers: [
// CHECK:             ],
// CHECK:             update_barriers: [
// CHECK:             ],
// CHECK:             virtual_wait_barriers: [
// CHECK:             ],
// CHECK:             virtual_update_barriers: [
// CHECK:             ]
// CHECK:           },
// CHECK:           task_type: "NNDMATask",
// CHECK:           task: {
// CHECK:             src: {
// CHECK:               dimensions: [
// CHECK:                 16,
// CHECK:                 1,
// CHECK:                 1,
// CHECK:                 16
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 1.0,
// CHECK:                 16.0,
// CHECK:                 1.0,
// CHECK:                 16.0,
// CHECK:                 16.0
// CHECK:               ],
// CHECK:               data: {
// CHECK:                 data_index: 0
// CHECK:               },
// CHECK:               locale: "GraphFile",
// CHECK:               locale_index: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               data_dtype: "U8",
// CHECK:               quant_zero: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               quant_mult: [
// CHECK:                 1
// CHECK:               ],
// CHECK:               quant_shift: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               order: 4930
// CHECK:             },
// CHECK:             dst: {
// CHECK:               dimensions: [
// CHECK:                 16,
// CHECK:                 1,
// CHECK:                 1,
// CHECK:                 16
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 1.0,
// CHECK:                 16.0,
// CHECK:                 1.0,
// CHECK:                 16.0,
// CHECK:                 16.0
// CHECK:               ],
// CHECK:               data: {
// CHECK:                 data_index: 8448
// CHECK:               },
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               data_dtype: "U8",
// CHECK:               quant_zero: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               quant_mult: [
// CHECK:                 1
// CHECK:               ],
// CHECK:               quant_shift: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               order: 4930
// CHECK:             }
// CHECK:           }
// CHECK:         },
// CHECK:         {
// CHECK:           associated_barriers: {
// CHECK:             wait_barriers: [
// CHECK:             ],
// CHECK:             update_barriers: [
// CHECK:               0
// CHECK:             ],
// CHECK:             virtual_wait_barriers: [
// CHECK:             ],
// CHECK:             virtual_update_barriers: [
// CHECK:               0
// CHECK:             ]
// CHECK:           },
// CHECK:           task_type: "NNDMATask",
// CHECK:           task: {
// CHECK:             src: {
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
// CHECK:                 data_index: 0
// CHECK:               },
// CHECK:               locale: "GraphFile",
// CHECK:               locale_index: [
// CHECK:                 1
// CHECK:               ],
// CHECK:               data_dtype: "I32",
// CHECK:               quant_zero: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               quant_mult: [
// CHECK:                 1
// CHECK:               ],
// CHECK:               quant_shift: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               order: 4660
// CHECK:             },
// CHECK:             dst: {
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
// CHECK:                 data_index: 8192
// CHECK:               },
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               data_dtype: "I32",
// CHECK:               quant_zero: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               quant_mult: [
// CHECK:                 1
// CHECK:               ],
// CHECK:               quant_shift: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               order: 4660
// CHECK:             }
// CHECK:           }
// CHECK:         },
// CHECK:         {
// CHECK:           associated_barriers: {
// CHECK:             wait_barriers: [
// CHECK:             ],
// CHECK:             update_barriers: [
// CHECK:             ],
// CHECK:             virtual_wait_barriers: [
// CHECK:             ],
// CHECK:             virtual_update_barriers: [
// CHECK:             ]
// CHECK:           },
// CHECK:           task_type: "NNDMATask",
// CHECK:           task: {
// CHECK:             src: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 16,
// CHECK:                 16,
// CHECK:                 16
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 1.0,
// CHECK:                 4096.0,
// CHECK:                 1.0,
// CHECK:                 256.0,
// CHECK:                 16.0
// CHECK:               ],
// CHECK:               data: {
// CHECK:                 data_index: 0
// CHECK:               },
// CHECK:               locale: "ProgrammableInput",
// CHECK:               locale_index: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               data_dtype: "U8",
// CHECK:               quant_zero: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               quant_mult: [
// CHECK:                 1
// CHECK:               ],
// CHECK:               quant_shift: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               order: 4930
// CHECK:             },
// CHECK:             dst: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 16,
// CHECK:                 16,
// CHECK:                 16
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 1.0,
// CHECK:                 4096.0,
// CHECK:                 1.0,
// CHECK:                 256.0,
// CHECK:                 16.0
// CHECK:               ],
// CHECK:               data: {
// CHECK:                 data_index: 4096
// CHECK:               },
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               data_dtype: "U8",
// CHECK:               quant_zero: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               quant_mult: [
// CHECK:                 1
// CHECK:               ],
// CHECK:               quant_shift: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               order: 4930
// CHECK:             }
// CHECK:           }
// CHECK:         },
// CHECK:         {
// CHECK:           associated_barriers: {
// CHECK:             wait_barriers: [
// CHECK:               1
// CHECK:             ],
// CHECK:             update_barriers: [
// CHECK:             ],
// CHECK:             virtual_wait_barriers: [
// CHECK:               1
// CHECK:             ],
// CHECK:             virtual_update_barriers: [
// CHECK:             ]
// CHECK:           },
// CHECK:           task_type: "NNDMATask",
// CHECK:           task: {
// CHECK:             src: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 16,
// CHECK:                 16,
// CHECK:                 16
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 1.0,
// CHECK:                 4096.0,
// CHECK:                 1.0,
// CHECK:                 256.0,
// CHECK:                 16.0
// CHECK:               ],
// CHECK:               data: {
// CHECK:                 data_index: 0
// CHECK:               },
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               data_dtype: "U8",
// CHECK:               quant_zero: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               quant_mult: [
// CHECK:                 16384
// CHECK:               ],
// CHECK:               quant_shift: [
// CHECK:                 14
// CHECK:               ],
// CHECK:               order: 4930
// CHECK:             },
// CHECK:             dst: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 16,
// CHECK:                 16,
// CHECK:                 16
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 1.0,
// CHECK:                 4096.0,
// CHECK:                 1.0,
// CHECK:                 256.0,
// CHECK:                 16.0
// CHECK:               ],
// CHECK:               data: {
// CHECK:                 data_index: 0
// CHECK:               },
// CHECK:               locale: "VPU_DDR_Heap",
// CHECK:               locale_index: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               data_dtype: "U8",
// CHECK:               quant_zero: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               quant_mult: [
// CHECK:                 16384
// CHECK:               ],
// CHECK:               quant_shift: [
// CHECK:                 14
// CHECK:               ],
// CHECK:               order: 4930
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK:       ]
// CHECK:     }
// CHECK:   ],
// CHECK:   binary_data: [
// CHECK:     {
// CHECK:       underlying_type: "U8",
// CHECK:       length: 256,
// CHECK:       data: [
// CHECK:         12469403627024359936,
// CHECK:         9731726653153013421,
// CHECK:         17120063903588549291,
// CHECK:         14044672708606133976,
// CHECK:         6772782313290599491,
// CHECK:         16330355330740976447,
// CHECK:         3052391877316931398,
// CHECK:         5873962643402319228,
// CHECK:         7097694943229279996,
// CHECK:         14177909903347673415,
// CHECK:         13383750166151176403,
// CHECK:         15607087457558586367,
// CHECK:         3263270837083821929,
// CHECK:         13831820577801605671,
// CHECK:         2429188602352100237,
// CHECK:         4462984799195317783,
// CHECK:         7161477838504349998,
// CHECK:         2778426948850165375,
// CHECK:         17687462302793885947,
// CHECK:         14816743958131061536,
// CHECK:         2625007076444961087,
// CHECK:         6706467981957884402,
// CHECK:         17207818054277800513,
// CHECK:         9329448799437666214,
// CHECK:         15163798012397998233,
// CHECK:         9478189190739702704,
// CHECK:         13177174553699510810,
// CHECK:         1258341451213681868,
// CHECK:         16419537324157337014,
// CHECK:         9462469876866294667,
// CHECK:         2882082063064199393,
// CHECK:         7733166290931588513
// CHECK:       ]
// CHECK:     },
// CHECK:     {
// CHECK:       underlying_type: "U8",
// CHECK:       length: 256,
// CHECK:       data: [
// CHECK:         72057589742969088,
// CHECK:         1077958144,
// CHECK:         72057589742969104,
// CHECK:         1077958144,
// CHECK:         72057589742969120,
// CHECK:         1077958144,
// CHECK:         72057589742969136,
// CHECK:         1077958144,
// CHECK:         72057589742969152,
// CHECK:         1077958144,
// CHECK:         72057589742969168,
// CHECK:         1077958144,
// CHECK:         72057589742969184,
// CHECK:         1077958144,
// CHECK:         72057589742969200,
// CHECK:         1077958144,
// CHECK:         72057589742969216,
// CHECK:         1077958144,
// CHECK:         72057589742969232,
// CHECK:         1077958144,
// CHECK:         72057589742969248,
// CHECK:         1077958144,
// CHECK:         72057589742969264,
// CHECK:         1077958144,
// CHECK:         72057589742969280,
// CHECK:         1077958144,
// CHECK:         72057589742969296,
// CHECK:         1077958144,
// CHECK:         72057589742969312,
// CHECK:         1077958144,
// CHECK:         72057589742969328,
// CHECK:         1077958144
// CHECK:       ]
// CHECK:     }
// CHECK:   ]
