//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=VPUX37XX" %s | vpux-translate --vpu-arch=VPUX37XX --export-VPUIP -o %t
// RUN: flatc --raw-binary --json %vpuip_schema_file% -- %t
// RUN: FileCheck %s --input-file %basename_t.json
// RUN: rm %basename_t.json
//
// This file generates a blob with vau_dp4m activation shave
// demonstrate that the runtime cannot handle this.  It's also a lit test to help
// check for regressions in the VPUIP dialect.
//

module @Test {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "input0" : tensor<1x1008x1x1xsi8>
        IE.DataInfo "input1" : tensor<1x1008x1x1xsi8>
    }
    outputsInfo : {
        IE.DataInfo "vau_dp4m" : tensor<1x252x1x1xsi32>
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
    func.func private @builtin_vau_dp4_m(%input0 : memref<*xsi8>, %input1 : memref<*xsi8>, %output : memref<*xsi32>)
        attributes {
            VPU.kernel_code = "vau_dp4m.cpp",
            VPU.kernel_entry = "vau_dp4m"        
        }

    // management kernel definition
    func.func private @runtime()
        attributes {
            VPU.kernel_code = "nnActEntry"
        }
}

func.func @main(%1: memref<1x1008x1x1xsi8>, %2: memref<1x1008x1x1xsi8>, %3: memref<1x252x1x1xsi32>) -> memref<1x252x1x1xsi32> {

    %in0_tile0_cmx  = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1008x1x1xsi8, [@CMX_NN, 0]>
    %in1_tile0_cmx  = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1008x1x1xsi8, [@CMX_NN, 0]>
    %out_tile0_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <4000> -> memref<1x252x1x1xsi32, [@CMX_NN, 0]>

    %b0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %b1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%1 : memref<1x1008x1x1xsi8>) outputs(%in0_tile0_cmx : memref<1x1008x1x1xsi8, [@CMX_NN, 0]>) -> memref<1x1008x1x1xsi8, [@CMX_NN, 0]>
    }

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%2 : memref<1x1008x1x1xsi8>) outputs(%in1_tile0_cmx : memref<1x1008x1x1xsi8, [@CMX_NN, 0]>) -> memref<1x1008x1x1xsi8, [@CMX_NN, 0]>
    }

    // Genetic Kernel information for the scheduler.
    VPURT.Task waits(%b0 : !VPURT.Barrier) updates(%b1 : !VPURT.Barrier) {
        VPUIP.SW.Kernel
                    {result_segment_sizes = dense<[1, 0]> : vector<2xi32>}
                    @VPU.SW::@builtin_vau_dp4_m           // The reference to the Kernel function.
                    inputs(%in0_tile0_cmx as %arg0: memref<1x1008x1x1xsi8, [@CMX_NN, 0]>, %in1_tile0_cmx as %arg1: memref<1x1008x1x1xsi8, [@CMX_NN, 0]>)     // Inputs/outputs buffers for generic operation interface
                    outputs(%out_tile0_cmx as %arg2: memref<1x252x1x1xsi32, [@CMX_NN, 0]>)   //
                    on tile 0                           // The tile index to execute on.
        -> memref<1x252x1x1xsi32, [@CMX_NN, 0]> {

                // The arguments mapping, the order must match the kernel parameter structure.
                VPUIP.SW.Kernel.run(%arg0, %arg1, %arg2)
                    : memref<1x1008x1x1xsi8, [@CMX_NN, 0]>
                    , memref<1x1008x1x1xsi8, [@CMX_NN, 0]>
                    , memref<1x252x1x1xsi32, [@CMX_NN, 0]>
        }
    }

    VPURT.Task waits(%b1 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%out_tile0_cmx : memref<1x252x1x1xsi32, [@CMX_NN, 0]>) outputs(%3 : memref<1x252x1x1xsi32>) -> memref<1x252x1x1xsi32>
    }
    return %3: memref<1x252x1x1xsi32>

}


}

// CHECK:   identifier: "Test"

// CHECK:   net_input: [
// CHECK:     {
// CHECK:       name: "input0",
// CHECK:       dimensions: [
// CHECK:           1,
// CHECK:           1008,
// CHECK:           1,
// CHECK:           1
// CHECK:       ],
// CHECK:       strides: [
// CHECK:           1.0,
// CHECK:           1008.0,
// CHECK:           1.0,
// CHECK:           1.0,
// CHECK:           1.0
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "I8"
// CHECK:     },
// CHECK:     {
// CHECK:       name: "input1",
// CHECK:       dimensions: [
// CHECK:           1,
// CHECK:           1008,
// CHECK:           1,
// CHECK:           1
// CHECK:       ],
// CHECK:       strides: [
// CHECK:           1.0,
// CHECK:           1008.0,
// CHECK:           1.0,
// CHECK:           1.0,
// CHECK:           1.0
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "I8"
// CHECK:     }
// CHECK:   ],

// CHECK:   net_output: [
// CHECK:     {
// CHECK:       name: "vau_dp4m",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         252,
// CHECK:         1,
// CHECK:         1
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         4.0,
// CHECK:         1008.0,
// CHECK:         4.0,
// CHECK:         4.0,
// CHECK:         4.0
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "I32"
// CHECK:     }
// CHECK:   ],

// CHECK:   task_count: 6,

// CHECK:   options: [
// CHECK:   ],

// CHECK:   in_tensor_desc: [
// CHECK:     {
// CHECK:       name: "input0",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         1008,
// CHECK:         1,
// CHECK:         1
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         1.0,
// CHECK:         1008.0,
// CHECK:         1.0,
// CHECK:         1.0,
// CHECK:         1.0
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "I8"
// CHECK:     },
// CHECK:     {
// CHECK:       name: "input1",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         1008,
// CHECK:         1,
// CHECK:         1
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         1.0,
// CHECK:         1008.0,
// CHECK:         1.0,
// CHECK:         1.0,
// CHECK:         1.0
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "I8"
// CHECK:     }
// CHECK:   ],

// CHECK:   out_tensor_desc: [
// CHECK:     {
// CHECK:       name: "vau_dp4m",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         252,
// CHECK:         1,
// CHECK:         1
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         4.0,
// CHECK:         1008.0,
// CHECK:         4.0,
// CHECK:         4.0,
// CHECK:         4.0
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "I32"
// CHECK:     }
// CHECK:   ]

// CHECK:   kernel_data: [
// CHECK:      ]
