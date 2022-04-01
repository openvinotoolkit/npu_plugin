// RUN: vpux-translate --export-VPUIP -o %t %s && flatc --raw-binary --json %vpuip_schema_file% -- %t && FileCheck %s --input-file %basename_t.json
//
// This file generates a blob with softmax activation shave
// demonstrate that the runtime cannot handle this.  It's also a lit test to help
// check for regressions in the VPUIP dialect.
//

module @Test attributes {VPU.arch = "VPUX37XX", VPU.compilationMode = "ReferenceHW"} {

IE.MemoryResource 31457280 bytes of @DDR {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64}
IE.MemoryResource 2097152 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}

IE.ExecutorResource 1 of @DMA_NN
IE.ExecutorResource 1 of @SHAVE_UPA
IE.ExecutorResource 1 of @SHAVE_ACT
IE.ExecutorResource 1 of @NCE {
    IE.ExecutorResource 1 of @DPU
}

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "input" : tensor<1x1000xf16>
    }
    outputsInfo : {
        IE.DataInfo "softmax" : tensor<1x1000xf16>
    }

// Sub-module, which holds SW kernel declarations and optional implementations.
// Used to group those declarations for faster access.
module @VPU.SW {
    // The declaration should match C++ params structure in decomposed form.
    // `memref` will be translated to `MemRefData`, while raw scalars will be translated as is.
    func private @builtin_softmax(%input : memref<*xf16>, %output : memref<*xf16>, %axis : i64)
        attributes {
            VPU.kernel_code = "single_shave_softmax.cpp",
            VPU.kernel_entry = "singleShaveSoftmax"
        }
}

func @main(%1: memref<1x1x1x1000xf16>, %2: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {

    %in_tile0_cmx  = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %out_tile0_cmx = VPURT.DeclareBuffer "CMX_NN" [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    %b0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %b1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%1 : memref<1x1x1x1000xf16>) outputs(%in_tile0_cmx : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }

    // Genetic Kernel information for the scheduler.
    VPURT.Task  waits(%b0 : !VPURT.Barrier) updates(%b1 : !VPURT.Barrier) {
        VPUIP.SW.Kernel
                    @VPU.SW::@builtin_softmax            // The reference to the Kernel function.
                    inputs(%in_tile0_cmx : memref<1x1x1x1000xf16, [@CMX_NN, 0]>)     // Inputs/outputs buffers for generic operation interface
                    outputs(%out_tile0_cmx : memref<1x1x1x1000xf16, [@CMX_NN, 0]>)   // and their mapping to inner region.
                    on tile 0                           // The tile index to execute on.
        -> memref<1x1x1x1000xf16, [@CMX_NN, 0]> {

            ^bb0(%arg0 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>, %arg1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>):
                // Inner region, isolated from above, which holds the information about arguments mapping.

                // The arguments mapping, the order must match the kernel parameter structure.
                VPUIP.SW.Kernel.run {attrs = [0]}(%arg0, %arg1)
                    : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
                    , memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        }
    }

    VPURT.Task waits(%b1 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%out_tile0_cmx : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%2 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
    }
    return %2: memref<1x1x1x1000xf16>

}


}

// CHECK:   identifier: "Test"

// CHECK:   net_input: [
// CHECK:     {
// CHECK:       name: "input",
// CHECK:       dimensions: [
// CHECK:           1,
// CHECK:           1,
// CHECK:           1,
// CHECK:           1000
// CHECK:       ],
// CHECK:       strides: [
// CHECK:           2.0,
// CHECK:           2000.0,
// CHECK:           2000.0,
// CHECK:           2000.0,
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
// CHECK:       name: "softmax",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         1,
// CHECK:         1,
// CHECK:         1000
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         2.0,
// CHECK:         2000.0,
// CHECK:         2000.0,
// CHECK:         2000.0,
// CHECK:         2.0
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP16"
// CHECK:     }
// CHECK:   ],

// CHECK:   task_count: 5,

// CHECK:   options: [
// CHECK:   ],

// CHECK:   in_tensor_desc: [
// CHECK:     {
// CHECK:       name: "input",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         1000
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         2.0,
// CHECK:         2000.0,
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
// CHECK:       name: "softmax",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         1000
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         2.0,
// CHECK:         2000.0,
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
