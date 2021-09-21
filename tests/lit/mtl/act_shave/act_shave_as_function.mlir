// RUN: vpux-translate --export-VPUIP -o %t %s && flatc --raw-binary --json %vpuip_schema_file% -- %t && FileCheck %s --input-file %basename_t.json
//
// This file generates a blob with sigmoid activation shave
// demonstrate that the runtime cannot handle this.  It's also a lit test to help
// check for regressions in the VPUIP dialect.
//

module @Test attributes {VPUIP.arch = "VPU3720", VPUIP.compilationMode = "ReferenceHW"} {

IERT.RunTimeResources
    availableMemory : {
        IERT.MemoryResource 1073741824 bytes
        IERT.MemoryResource 31457280 bytes of "DDR" {VPUIP.bandwidth = 8 : i64, VPUIP.derateFactor = 6.000000e-01 : f64}
        IERT.MemoryResource 2097152 bytes of "CMX_NN" {VPUIP.bandwidth = 32 : i64, VPUIP.derateFactor = 1.000000e+00 : f64}
    }
    usedMemory : {
    }
    executors : {
        IERT.ExecutorResource 1 of  "Leon_RT"
        IERT.ExecutorResource 1 of  "Leon_NN"
        IERT.ExecutorResource 1 of  "ACT_SHAVE"
        IERT.ExecutorResource 1 of  "SHAVE_NN"
        IERT.ExecutorResource 1 of  "NCE_Cluster" {
            IERT.ExecutorResource 1 of "NCE_PerClusterDPU"
        }
        IERT.ExecutorResource 1 of "DMA_UPA"
        IERT.ExecutorResource 1 of "DMA_NN"
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
        IE.DataInfo "input" : tensor<1x1000xf16>
    }
    outputsInfo : {
        IE.DataInfo "sigmoid" : tensor<1x1000xf16>
    }

// Sub-module, which holds SW kernel declarations and optional implementations.
// Used to group those declarations for faster access.
module @VPU.SW {
    // The declaration should match C++ params structure in decomposed form.
    // `memref` will be translated to `MemRefData`, while raw scalars will be translated as is.
    func private @builtin_softmax(%input : memref<*xf16,>, %output : memref<*xf16>, %axis : i32)
        attributes { VPU.kernel_code = "softmax_fp16.c" }
}

func @main(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {

    %in_tile0_cmx  = VPUIP.DeclareTensor "VPU_CMX_NN" [0] <0> -> memref<1x1x1x1000xf16, "VPU_CMX_NN">
    %out_tile0_cmx = VPUIP.DeclareTensor "VPU_CMX_NN" [0] <2000> -> memref<1x1x1x1000xf16, "VPU_CMX_NN">

    // Genetic Kernel information for the scheduler.
    %sigmoid_krn =
    VPURT.Task waits(%b0) updates(%b1) {
        VPUIP.SW.Kernel
                    @VPU.SW.builtin_softmax             // The reference to the Kernel function.
                    "GFEmbeddedKernel"                  // locale to pack kernel into
                    on tile 0                           // The tile index to execute on.
                    inputs(%in_tile_cmx_0 as %arg0)     // Inputs/outputs buffers for generic operation interface
                    outputs(%out_tile_cmx_0 as %arg1)   // and their mapping to inner region.
        {
            // Inner region, isolated from above, which holds the information about arguments mapping.
            // We can use constant scalars/arrays definitions here.
            %axis   = constant 0 : i32

            // The arguments mapping, the order must match the kernel parameter structure.
            VPUIP.SW.Kernel.run(%arg0, %arg1, %axis)
        }
    }


    %b0 = VPUIP.ConfigureBarrier<0> -> !VPUIP.Barrier
    %b1 = VPUIP.ConfigureBarrier<1> -> !VPUIP.Barrier

    %4 = VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16>) outputs(%0 : memref<1x1x1x1000xf16, "VPU_CMX_NN">) updates(%2 : !VPUIP.Barrier) -> memref<1x1x1x1000xf16, "VPU_CMX_NN">

#    %mk = VPUIP.ACTShaveTaskOp kernel(%sigmoid_krn : memref<1000xui8, "VPU_CMX_NN">)
#    inputs(%0 : memref<1x1x1x1000xf16, "VPU_CMX_NN">)
#    outputs(%1 : memref<1x1x1x1000xf16, "VPU_CMX_NN">)
#    waits(%2 : !VPUIP.Barrier)
#    updates(%3 : !VPUIP.Barrier) -> memref<1x1x1x1000xf16, "VPU_CMX_NN">
#    %5 = VPUIP.ACTShaveTaskOp kernel(%sigmoid_krn : memref<1000xui8, "VPU_CMX_NN">)  inputs(%0 : memref<1x1x1x1000xf16, "VPU_CMX_NN">) outputs(%1 : memref<1x1x1x1000xf16, "VPU_CMX_NN">) waits(%2 : !VPUIP.Barrier) updates(%3 : !VPUIP.Barrier) -> memref<1x1x1x1000xf16, "VPU_CMX_NN">

    %6 = VPUIP.NNDMA inputs(%1 : memref<1x1x1x1000xf16, "VPU_CMX_NN">) outputs(%arg1 : memref<1x1x1x1000xf16>) waits(%3 : !VPUIP.Barrier) -> memref<1x1x1x1000xf16>
    return %6: memref<1x1x1x1000xf16>

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
// CHECK:       name: "ACT-shave",
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

// CHECK:   task_count: 3,

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
// CHECK:         4.0,
// CHECK:         4000.0,
// CHECK:         4.0
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP32"
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
// CHECK:         4.0,
// CHECK:         4000.0,
// CHECK:         4.0
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP32"
// CHECK:     }
// CHECK:   ]
