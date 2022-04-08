//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --init-compiler="vpu-arch=VPUX37XX" %s | vpux-translate --export-VPUIP -o %t
// RUN: flatc --raw-binary --json %vpuip_schema_file% -- %t
// RUN: FileCheck %s --input-file %basename_t.json
// RUN: rm %basename_t.json
//
// This file generates a blob with i420_to_rgb activation shave
// demonstrate that the runtime cannot handle this.  It's also a lit test to help
// check for regressions in the VPUIP dialect.
//

module @Test {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "Parameter_143" : tensor<1x240x320x1xf16>
        IE.DataInfo "Parameter_144" : tensor<1x120x160x1xf16>
        IE.DataInfo "Parameter_145" : tensor<1x120x160x1xf16>
    }
    outputsInfo : {
        IE.DataInfo "I420toRGB_146" : tensor<1x240x320x3xf16>
    }

// Sub-module, which holds SW kernel declarations and optional implementations.
// Used to group those declarations for faster access.
module @VPU.SW {
    // The declaration should match C++ params structure in decomposed form.
    // `memref` will be translated to `MemRefData`, while raw scalars will be translated as is.
    func private @builtin_YuvToRgb(%input0 : memref<*xf16>, %input1 : memref<*xf16>, %input2 : memref<*xf16>, %output : memref<*xf16>, %rgbFormat : i64)
        attributes {
            VPU.kernel_code = "single_shave_convert_color_i420_to_rgb.cpp",
            VPU.kernel_entry = "single_shave_convert_color_i420_to_rgb"        
        }

    // management kernel definition
    func private @runtime()
        attributes {
            VPU.kernel_code = "nnActEntry"
        }
}

func @main(%1: memref<1x240x320x1xf16>, %2: memref<1x120x160x1xf16>, %3: memref<1x120x160x1xf16>, %4: memref<1x240x320x3xf16>) -> memref<1x240x320x3xf16> {

    %in0_tile0_cmx  = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x240x320x1xf16, [@CMX_NN, 0]>
    %in1_tile0_cmx  = VPURT.DeclareBuffer "CMX_NN" [0] <153600> -> memref<1x120x160x1xf16, [@CMX_NN, 0]>
    %in2_tile0_cmx  = VPURT.DeclareBuffer "CMX_NN" [0] <192000> -> memref<1x120x160x1xf16, [@CMX_NN, 0]>
    %out_tile0_cmx = VPURT.DeclareBuffer "CMX_NN" [0] <230400> -> memref<1x240x320x3xf16, [@CMX_NN, 0]>

    %b0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %b1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%1 : memref<1x240x320x1xf16>) outputs(%in0_tile0_cmx : memref<1x240x320x1xf16, [@CMX_NN, 0]>) -> memref<1x240x320x1xf16, [@CMX_NN, 0]>
    }

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%2 : memref<1x120x160x1xf16>) outputs(%in1_tile0_cmx : memref<1x120x160x1xf16, [@CMX_NN, 0]>) -> memref<1x120x160x1xf16, [@CMX_NN, 0]>
    }

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%3 : memref<1x120x160x1xf16>) outputs(%in2_tile0_cmx : memref<1x120x160x1xf16, [@CMX_NN, 0]>) -> memref<1x120x160x1xf16, [@CMX_NN, 0]>
    }

    // Genetic Kernel information for the scheduler.
    VPURT.Task waits(%b0 : !VPURT.Barrier) updates(%b1 : !VPURT.Barrier) {
        VPUIP.SW.Kernel
                    {result_segment_sizes = dense<[1, 0]> : vector<2xi32>}
                    @VPU.SW::@builtin_YuvToRgb           // The reference to the Kernel function.
                    inputs(%in0_tile0_cmx as %arg0: memref<1x240x320x1xf16, [@CMX_NN, 0]>, %in1_tile0_cmx as %arg1: memref<1x120x160x1xf16, [@CMX_NN, 0]>, %in2_tile0_cmx as %arg2: memref<1x120x160x1xf16, [@CMX_NN, 0]>)     // Inputs/outputs buffers for generic operation interface
                    outputs(%out_tile0_cmx as %arg3: memref<1x240x320x3xf16, [@CMX_NN, 0]>)   //
                    on tile 0                           // The tile index to execute on.
        -> memref<1x240x320x3xf16, [@CMX_NN, 0]> {

                // The arguments mapping, the order must match the kernel parameter structure.
                VPUIP.SW.Kernel.run {attrs = [0, 1]}(%arg0, %arg1, %arg2, %arg3)
                    : memref<1x240x320x1xf16, [@CMX_NN, 0]>
                    , memref<1x120x160x1xf16, [@CMX_NN, 0]>
                    , memref<1x120x160x1xf16, [@CMX_NN, 0]>
                    , memref<1x240x320x3xf16, [@CMX_NN, 0]>
        }
    }

    VPURT.Task waits(%b1 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%out_tile0_cmx : memref<1x240x320x3xf16, [@CMX_NN, 0]>) outputs(%4 : memref<1x240x320x3xf16>) -> memref<1x240x320x3xf16>
    }
    return %4: memref<1x240x320x3xf16>

}

}

// CHECK:   identifier: "Test"

// CHECK:   net_input: [
// CHECK:     {
// CHECK:       name: "Parameter_143",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         240,
// CHECK:         320,
// CHECK:         1
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         2.0,
// CHECK:         153600.0,
// CHECK:         640.0,
// CHECK:         2.0,
// CHECK:         2.0
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP16"
// CHECK:     },
// CHECK:     {
// CHECK:       name: "Parameter_144",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         120,
// CHECK:         160,
// CHECK:         1
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         2.0,
// CHECK:         38400.0,
// CHECK:         320.0,
// CHECK:         2.0,
// CHECK:         2.0
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP16"
// CHECK:     },
// CHECK:     {
// CHECK:       name: "Parameter_145",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         120,
// CHECK:         160,
// CHECK:         1
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         2.0,
// CHECK:         38400.0,
// CHECK:         320.0,
// CHECK:         2.0,
// CHECK:         2.0
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP16"
// CHECK:     }
// CHECK:   ],

// CHECK:   net_output: [
// CHECK:     {
// CHECK:       name: "I420toRGB_146",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         240,
// CHECK:         320,
// CHECK:         3
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         2.0,
// CHECK:         460800.0,
// CHECK:         1920.0,
// CHECK:         6.0,
// CHECK:         2.0
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP16"
// CHECK:     }
// CHECK:   ],

// CHECK:   task_count: 7,

// CHECK:   options: [
// CHECK:   ],

// CHECK:   in_tensor_desc: [
// CHECK:     {
// CHECK:       name: "Parameter_143",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         240,
// CHECK:         320,
// CHECK:         1
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         2.0,
// CHECK:         153600.0,
// CHECK:         640.0,
// CHECK:         2.0,
// CHECK:         2.0
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP16"
// CHECK:     },
// CHECK:     {
// CHECK:       name: "Parameter_144",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         120,
// CHECK:         160,
// CHECK:         1
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         2.0,
// CHECK:         38400.0,
// CHECK:         320.0,
// CHECK:         2.0,
// CHECK:         2.0
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP16"
// CHECK:     },
// CHECK:     {
// CHECK:       name: "Parameter_145",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         120,
// CHECK:         160,
// CHECK:         1
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         2.0,
// CHECK:         38400.0,
// CHECK:         320.0,
// CHECK:         2.0,
// CHECK:         2.0
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP16"
// CHECK:     }
// CHECK:   ],

// CHECK:   out_tensor_desc: [
// CHECK:     {
// CHECK:       name: "I420toRGB_146",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         240,
// CHECK:         320,
// CHECK:         3
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         2.0,
// CHECK:         460800.0,
// CHECK:         1920.0,
// CHECK:         6.0,
// CHECK:         2.0
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP16"
// CHECK:     }
// CHECK:   ]
