//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --init-compiler="vpu-arch=VPUX37XX" %s | vpux-translate --export-VPUIP -o %t
// RUN: flatc --raw-binary --json %vpuip_schema_file% -- %t
// RUN: FileCheck %s --input-file %basename_t.json
// RUN: rm %basename_t.json

//
// This file generates a blob with prelu activation shave
// demonstrate that the runtime cannot handle this.  It's also a lit test to help
// check for regressions in the VPUIP dialect.
//

module @Test {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "input" : tensor<1x1000x1x1xf16>
    }
    outputsInfo : {
        IE.DataInfo "prelu" : tensor<1x1000x1x1xf16>
    }


// Sub-module, which holds SW kernel declarations and optional implementations.
// Used to group those declarations for faster access.
module @VPU.SW {
    // The declaration should match C++ params structure in decomposed form.
    // `memref` will be translated to `MemRefData`, while raw scalars will be translated as is.
    func private @builtin_prelu(%input : memref<*xf16>, %output : memref<*xf16>)
        attributes {
            VPU.kernel_code = "prelu_fp16.cpp",
            VPU.kernel_entry = "prelu_fp16"
        }

    // management kernel definition
    func private @runtime()
        attributes {
            VPU.kernel_code = "nnActEntry"
        }
}

func @main(%1: memref<1x1000x1x1xf16>, %2: memref<1x1000x1x1xf16>) -> memref<1x1000x1x1xf16> {
    %cst = const.Declare memref<1000xf16> = dense<0.00999999977> : tensor<1000xf32>, [#const.ConvertElemType<f16>]
    %in0_tile0_cmx  = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x1000x1x1xf16, [@CMX_NN, 0]>
    %in1_tile0_cmx  = VPURT.DeclareBuffer "CMX_NN" [0] <2048> -> memref<1000xf16, [@CMX_NN, 0]>
    %out_tile0_cmx = VPURT.DeclareBuffer "CMX_NN" [0] <4096> -> memref<1x1000x1x1xf16, [@CMX_NN, 0]>

    %b0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %b1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    %b2 = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%1 : memref<1x1000x1x1xf16>) outputs(%in0_tile0_cmx : memref<1x1000x1x1xf16, [@CMX_NN, 0]>) -> memref<1x1000x1x1xf16, [@CMX_NN, 0]>
    }

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%cst : memref<1000xf16>) outputs(%in1_tile0_cmx : memref<1000xf16, [@CMX_NN, 0]>) -> memref<1000xf16, [@CMX_NN, 0]>
    }

    // Genetic Kernel information for the scheduler.
    VPURT.Task waits(%b1 : !VPURT.Barrier) updates(%b2 : !VPURT.Barrier) {
        VPUIP.SW.Kernel
                    {result_segment_sizes = dense<[1, 0]> : vector<2xi32>}
                    @VPU.SW::@builtin_prelu           // The reference to the Kernel function.
                    inputs(%in0_tile0_cmx as %arg0: memref<1x1000x1x1xf16, [@CMX_NN, 0]>, %in1_tile0_cmx as %arg1: memref<1000xf16, [@CMX_NN, 0]>)     // Inputs/outputs buffers for generic operation interface
                    outputs(%out_tile0_cmx as %arg2: memref<1x1000x1x1xf16, [@CMX_NN, 0]>)   //
                    on tile 0                           // The tile index to execute on.
        -> memref<1x1000x1x1xf16, [@CMX_NN, 0]> {

                // The arguments mapping, the order must match the kernel parameter structure.
                VPUIP.SW.Kernel.run(%arg0, %arg1, %arg2)
                    : memref<1x1000x1x1xf16, [@CMX_NN, 0]>
                    , memref<1000xf16, [@CMX_NN, 0]>
                    , memref<1x1000x1x1xf16, [@CMX_NN, 0]>
        }
    }

    VPURT.Task waits(%b2 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%out_tile0_cmx : memref<1x1000x1x1xf16, [@CMX_NN, 0]>) outputs(%2 : memref<1x1000x1x1xf16>) -> memref<1x1000x1x1xf16>
    }
    return %2: memref<1x1000x1x1xf16>

}


}

// CHECK:   identifier: "Test"

// CHECK:   net_input: [
// CHECK:     {
// CHECK:       name: "input",
// CHECK:       dimensions: [
// CHECK:           1,
// CHECK:           1000,
// CHECK:           1,
// CHECK:           1
// CHECK:       ],
// CHECK:       strides: [
// CHECK:           2.0,
// CHECK:           2000.0,
// CHECK:           2.0,
// CHECK:           2.0,
// CHECK:           2.0
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
// CHECK:       name: "prelu",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         1000,
// CHECK:         1,
// CHECK:         1
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         2.0,
// CHECK:         2000.0,
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

// CHECK:   task_count: 7,

// CHECK:   options: [
// CHECK:   ],

// CHECK:   in_tensor_desc: [
// CHECK:     {
// CHECK:       name: "input",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         1000,
// CHECK:         1,
// CHECK:         1
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         2.0,
// CHECK:         2000.0,
// CHECK:         2.0,
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
// CHECK:       name: "prelu",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         1000,
// CHECK:         1,
// CHECK:         1
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         2.0,
// CHECK:         2000.0,
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

// CHECK:   kernel_data: [
// CHECK:      ]
