// RUN: vpux-translate --export-VPUIP -o %t blob %s mlir && flatc --raw-binary --json %vpuip_schema_file% -- %t && FileCheck %s --input-file %basename_t.json
//
// This file generates a blob with topk activation shave
// demonstrate that the runtime cannot handle this.  It's also a lit test to help
// check for regressions in the VPUIP dialect.
//

module @Test attributes {VPU.arch = "MTL", VPU.compilationMode = "ReferenceHW"} {

IERT.RunTimeResources
    availableMemory : {
        IERT.MemoryResource 31457280 bytes of "DDR" {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64}
        IERT.MemoryResource 2097152 bytes of "CMX_NN" {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    }
    usedMemory : {
    }
    executors : {
        IERT.ExecutorResource 1 of "DMA_NN"
        IERT.ExecutorResource 1 of  "SHAVE_NN"
        IERT.ExecutorResource 1 of  "SHAVE_ACT"
        IERT.ExecutorResource 1 of  "NCE" {
            IERT.ExecutorResource 1 of "DPU"
        }
    }

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "inputValues" : tensor<1x1000xf16>
        IE.DataInfo "inputK" : tensor<1x1000xf16>
    }
    outputsInfo : {
        IE.DataInfo "outputValues" : tensor<1x1000xf16>
        IE.DataInfo "outputIndex" : tensor<1x1000xf16>
    }

// Sub-module, which holds SW kernel declarations and optional implementations.
// Used to group those declarations for faster access.
module @VPU.SW {
    // The declaration should match C++ params structure in decomposed form.
    // `memref` will be translated to `MemRefData`, while raw scalars will be translated as is.
    func private @builtin_topk(%input : memref<*xf16>, %output : memref<*xf16>)
        attributes {
            VPU.kernel_code = "single_shave_topk.cpp",
            VPU.kernel_entry = "single_shave_topk"
        }
}

func @main(%1: memref<1x1x1x1000xf16>, %2: memref<1x1x1x1000xf16>,%3: memref<1x1x1x1000xf16>, %4: memref<1x1x1x1000xf16>) -> (memref<1x1x1x1000xf16>, memref<1x1x1x1000xf16>) {

    %in_tile0_cmx  = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x1x1x1000xf16, "CMX_NN">
    %in_tile1_cmx  = VPURT.DeclareBuffer "CMX_NN" [0] <2000> -> memref<1x1x1x1000xf16, "CMX_NN">
    %out_tile0_cmx = VPURT.DeclareBuffer "CMX_NN" [0] <4000> -> memref<1x1x1x1000xf16, "CMX_NN">
    %out_tile1_cmx = VPURT.DeclareBuffer "CMX_NN" [0] <6000> -> memref<1x1x1x1000xf16, "CMX_NN">

    %b0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %b1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%1 : memref<1x1x1x1000xf16>) outputs(%in_tile0_cmx : memref<1x1x1x1000xf16, "CMX_NN">) -> memref<1x1x1x1000xf16, "CMX_NN">
        VPUIP.NNDMA inputs(%2 : memref<1x1x1x1000xf16>) outputs(%in_tile1_cmx : memref<1x1x1x1000xf16, "CMX_NN">) -> memref<1x1x1x1000xf16, "CMX_NN">
    }

    // Genetic Kernel information for the scheduler.
    VPURT.Task waits(%b0  : !VPURT.Barrier) updates(%b1  : !VPURT.Barrier) {
    %topk_krn =
        VPUIP.SW.Kernel
                    @VPU.SW::@builtin_topk            // The reference to the Kernel function.
                    inputs(%in_tile0_cmx : memref<1x1x1x1000xf16, "CMX_NN">)     // Inputs/outputs buffers for generic operation interface
                    inputs(%in_tile1_cmx : memref<1x1x1x1000xf16, "CMX_NN">) 
                    outputs(%out_tile0_cmx : memref<1x1x1x1000xf16, "CMX_NN">)   // and their mapping to inner region.
                    outputs(%out_tile1_cmx : memref<1x1x1x1000xf16, "CMX_NN">)
                    on tile 0                           // The tile index to execute on.

        -> memref<1x1x1x1000xf16, "CMX_NN"> {

            ^bb0(%arg0 : memref<1x1x1x1000xf16, "CMX_NN">, %arg1 : memref<1x1x1x1000xf16, "CMX_NN">, %arg2 : memref<1x1x1x1000xf16, "CMX_NN">, %arg3 : memref<1x1x1x1000xf16, "CMX_NN">):
                // Inner region, isolated from above, which holds the information about arguments mapping.
                // We can use constant scalars/arrays definitions here.
                
                %mode = arith.constant 0 : i64
                %sort = arith.constant 0 : i64
                %axis = arith.constant 0 : i64
                
                // The arguments mapping, the order must match the kernel parameter structure.
                VPUIP.SW.Kernel.run(%arg0, %arg1, %arg2, %arg3, %mode, %sort, %axis)
                    : memref<1x1x1x1000xf16, "CMX_NN">
                    , memref<1x1x1x1000xf16, "CMX_NN">
                    , memref<1x1x1x1000xf16, "CMX_NN">
                    , memref<1x1x1x1000xf16, "CMX_NN">
                    , i64
                    , i64
                    , i64
        }
    }

    VPURT.Task waits(%b1 : !VPURT.Barrier) {
        %5 = VPUIP.NNDMA inputs(%out_tile0_cmx : memref<1x1x1x1000xf16, "CMX_NN">) outputs(%3 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
        %6 = VPUIP.NNDMA inputs(%out_tile1_cmx : memref<1x1x1x1000xf16, "CMX_NN">) outputs(%4 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
    }
    return %5, %6: memref<1x1x1x1000xf16>, memref<1x1x1x1000xf16>

}


}