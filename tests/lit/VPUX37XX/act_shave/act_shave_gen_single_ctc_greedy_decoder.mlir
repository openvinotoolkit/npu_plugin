//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --init-compiler="vpu-arch=VPUX37XX" %s | vpux-translate --export-VPUIP -o %t
// RUN: flatc --raw-binary --json %vpuip_schema_file% -- %t
// RUN: FileCheck %s --input-file %basename_t.json
// RUN: rm %basename_t.json
//
// This file generates a blob with ctc_greedy_decoder activation shave
// demonstrate that the runtime cannot handle this.  It's also a lit test to help
// check for regressions in the VPUIP dialect.
//

module @Test {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "Parameter_143" : tensor<10x1x16xf16>
    }
    outputsInfo : {
        IE.DataInfo "CTCGreedyDecoder_145" : tensor<1x10x1x1xf16>
    }

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [
        4096,  // Size in bytes for the actSHAVE0 in the first tile.
        4096,  // Size in bytes for the actSHAVE1 in the first tile.
        4096,  // Size in bytes for the actSHAVE2 in the second tile.
        4096   // Size in bytes for the actSHAVE3 in the second tile.
    ]

// Sub-module, which holds SW kernel declarations and optional implementations.
// Used to group those declarations for faster access.
module @VPU.SW {
    // The declaration should match C++ params structure in decomposed form.
    // `memref` will be translated to `MemRefData`, while raw scalars will be translated as is.
    func private @builtin_CTCGreedyDecoder(%input0 : memref<*xf16>, %input1 : memref<*xf16>, %output : memref<*xf16>, %mergeRepeated : i64)
        attributes {
            VPU.kernel_code = "single_shave_ctc_greedy_decoder.cpp",
            VPU.kernel_entry = "single_shave_ctc_greedy_decoder"
        }
    // management kernel definition
    func private @runtime()
        attributes {
            VPU.kernel_code = "nnActEntry"
        }
}

func @main(%0: memref<10x1x16xf16>, %2: memref<1x10x1x1xf16>) -> memref<1x10x1x1xf16> {
    %cst = const.Declare memref<10x1xf16> = dense<[[1.000000e+00], [1.000000e+00], [1.000000e+00], [1.000000e+00], [1.000000e+00], [0.000000e+00], [0.000000e+00], [0.000000e+00], [0.000000e+00], [0.000000e+00]]> : tensor<10x1xf32>, [#const.ConvertElemType<f16>]
    %in0_tile0_cmx = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<10x1x16xf16, [@CMX_NN, 0]>
    %in1_tile0_cmx = VPURT.DeclareBuffer "CMX_NN" [0] <320> -> memref<10x1xf16, [@CMX_NN, 0]>
    %out_tile0_cmx = VPURT.DeclareBuffer "CMX_NN" [0] <340> -> memref<1x10x1x1xf16, [@CMX_NN, 0]>

    %b0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %b1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%0 : memref<10x1x16xf16>) outputs(%in0_tile0_cmx : memref<10x1x16xf16, [@CMX_NN, 0]>) -> memref<10x1x16xf16, [@CMX_NN, 0]>
    }

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%cst : memref<10x1xf16>) outputs(%in1_tile0_cmx : memref<10x1xf16, [@CMX_NN, 0]>) -> memref<10x1xf16, [@CMX_NN, 0]>
    }

    // Genetic Kernel information for the scheduler.
    VPURT.Task waits(%b0  : !VPURT.Barrier) updates(%b1  : !VPURT.Barrier) {
        VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>}
                    @VPU.SW::@builtin_CTCGreedyDecoder            // The reference to the Kernel function.
                    inputs(%in0_tile0_cmx as %arg0: memref<10x1x16xf16, [@CMX_NN, 0]>, %in1_tile0_cmx as %arg1: memref<10x1xf16, [@CMX_NN, 0]>)     // Inputs/outputs buffers for generic operation interface
                    outputs(%out_tile0_cmx as %arg2: memref<1x10x1x1xf16, [@CMX_NN, 0]>)   // and their mapping to inner region.
                    on tile 0                           // The tile index to execute on.

        -> memref<1x10x1x1xf16, [@CMX_NN, 0]> {

                // The arguments mapping, the order must match the kernel parameter structure.
                VPUIP.SW.Kernel.run {attrs = [1]}(%arg0, %arg1, %arg2)
                    : memref<10x1x16xf16, [@CMX_NN, 0]>
                    , memref<10x1xf16, [@CMX_NN, 0]>
                    , memref<1x10x1x1xf16, [@CMX_NN, 0]>
        }
    }

    VPURT.Task waits(%b1 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%out_tile0_cmx : memref<1x10x1x1xf16, [@CMX_NN, 0]>) outputs(%2 : memref<1x10x1x1xf16>) -> memref<1x10x1x1xf16>
    }
    return %2: memref<1x10x1x1xf16>

}


}

// CHECK:   identifier: "Test"

// CHECK:   net_input: [
// CHECK:     {
// CHECK:       name: "Parameter_143",
// CHECK:       dimensions: [
// CHECK:         10,
// CHECK:         1,
// CHECK:         16
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         2.0,
// CHECK:         32.0,
// CHECK:         32.0,
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
// CHECK:       name: "CTCGreedyDecoder_145",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         10,
// CHECK:         1,
// CHECK:         1
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         2.0,
// CHECK:         20.0,
// CHECK:         2.0,
// CHECK:         2.0,
// CHECK:         2.0
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP16"
// CHECK:     }
// CHECK:   ],

// CHECK:   task_count: 6,

// CHECK:   options: [
// CHECK:   ],

// CHECK:   in_tensor_desc: [
// CHECK:     {
// CHECK:       name: "Parameter_143",
// CHECK:       dimensions: [
// CHECK:         10,
// CHECK:         1,
// CHECK:         16
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         2.0,
// CHECK:         32.0,
// CHECK:         32.0,
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
// CHECK:       name: "CTCGreedyDecoder_145",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         10,
// CHECK:         1,
// CHECK:         1
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         2.0,
// CHECK:         20.0,
// CHECK:         2.0,
// CHECK:         2.0,
// CHECK:         2.0
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP16"
// CHECK:     }
// CHECK:   ]
