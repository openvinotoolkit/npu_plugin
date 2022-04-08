// RUN: vpux-opt --init-compiler="vpu-arch=VPUX37XX" %s
// this test can only be (correctly) run manually until E#48620 is solved

module @Test {
    
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
    // CHECK:       %[[VAL0:.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    %out_tile0_cmx = VPURT.DeclareBuffer "CMX_NN" [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    // CHECK:       %[[VAL1:.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    %b0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    // CHECK:       %[[VAL2:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPUIPRegMapped.Index<0>

    %b1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    // CHECK:       %[[VAL3:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPUIPRegMapped.Index<1>

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%1 : memref<1x1x1x1000xf16>) outputs(%in_tile0_cmx : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }
    // CHECK:       %[[VAL4:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%[[VAL5:.*]] : memref<1x1x1x1000xf16>) outputs(%[[VAL0]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) updates(%[[VAL2]] : !VPUIPRegMapped.Index<0>) start_after(0) -> !VPUIPRegMapped.Index<0>

    VPURT.Task  waits(%b0 : !VPURT.Barrier) updates(%b1 : !VPURT.Barrier) {
        VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>}
                    @VPU.SW::@builtin_softmax            // The reference to the Kernel function.
                    inputs(%in_tile0_cmx as %arg0: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)     // Inputs/outputs buffers for generic operation interface
                    outputs(%out_tile0_cmx as %arg1: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)   // and their mapping to inner region.
                    on tile 0                           // The tile index to execute on.
        -> memref<1x1x1x1000xf16, [@CMX_NN, 0]> {

                // The arguments mapping, the order must match the kernel parameter structure.
                VPUIP.SW.Kernel.run {attrs = [0]}(%arg0, %arg1)
                    : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
                    , memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        }
    }
    // CHECK:       %[[VAL6:.*]] = VPUIPRegMapped.DeclareKernelText kernel_path([[VAL7:.*]]) -> !VPUIPRegMapped.Index<0>
    // CHECK:       %[[VAL8:.*]] = VPUIPRegMapped.DeclareKernelArgs kernel_path([[VAL7]]) -> !VPUIPRegMapped.Index<0>
    // CHECK:       %[[VAL9:.*]] = VPUIPRegMapped.DeclareKernelEntry kernel_path([[VAL7]]) -> !VPUIPRegMapped.Index<0>

    // CHECK:       %[[VAL10:.*]] = VPUIPRegMapped.ActKernelRange kernel_text_index(%[[VAL6]] : <0>) kernel_args_index(%[[VAL8]] : <0>) kernel_entry_index(%[[VAL9]] : <0>) -> !VPUIPRegMapped.Index<0>
    // CHECK:       %[[VAL11:.*]] = VPUIPRegMapped.ActKernelInvocation range_index(%[[VAL10]] : <0>) waits(%[[VAL2]] : !VPUIPRegMapped.Index<0>) updates(%[[VAL3]] : !VPUIPRegMapped.Index<1>) tile(0) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<0>
    // CHECK:       %[[VAL12:.*]] = VPUIPRegMapped.KernelParams inputs(%[[VAL0]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%[[VAL1]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("{{.*}}") kernel_params({{.*}}) -> !VPUIPRegMapped.Index<0>


    VPURT.Task waits(%b1 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%out_tile0_cmx : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%2 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
    }
    // CHECK:       %[[VAL13:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%[[VAL1]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%[[VAL14:.*]] : memref<1x1x1x1000xf16>) previousDMA(%[[VAL4]] : !VPUIPRegMapped.Index<0>) waits(%[[VAL3]] : !VPUIPRegMapped.Index<1>) start_after(0) -> !VPUIPRegMapped.Index<1>

    return %2: memref<1x1x1x1000xf16>

}
}
