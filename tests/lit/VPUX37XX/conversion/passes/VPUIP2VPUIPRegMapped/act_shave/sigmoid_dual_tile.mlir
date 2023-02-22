//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --convert-VPUIP-to-VPUIPRegMapped %s | FileCheck %s
//

module @Test {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "input" : tensor<1x1000xf16>
    }
    outputsInfo : {
        IE.DataInfo "sigmoid" : tensor<1x1000xf16>
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
    func private @builtin_sigmoid(%input : memref<*xf16>, %output : memref<*xf16>)
        attributes {
            VPU.kernel_code = "sigmoid_fp16.c",
            VPU.kernel_entry = "sigmoid_fp16"
        }

    // management kernel definition
    func private @runtime()
        attributes {
            VPU.kernel_code = "nnActEntry"
        }
}



func @main(%1: memref<1x1x1x1000xf16>, %2: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {

    // using CMX memory from CMX Slice 1
    // while it is not mandatory to run shave on tile 1, it is needed  to access all 4mb of CMX
    %in_tile1_cmx  = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 1]>
    %out_tile1_cmx = VPURT.DeclareBuffer "CMX_NN" [1] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 1]>

    %b0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %b1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%1 : memref<1x1x1x1000xf16>) outputs(%in_tile1_cmx : memref<1x1x1x1000xf16, [@CMX_NN, 1]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 1]>
    }

    // Genetic Kernel information for the scheduler.
    VPURT.Task waits(%b0  : !VPURT.Barrier) updates(%b1  : !VPURT.Barrier) {
        VPUIP.SW.Kernel {
	            result_segment_sizes = dense<[1, 0]> : vector<2xi32>}
                    @VPU.SW::@builtin_sigmoid                                                // The reference to the Kernel function.
                    inputs(%in_tile1_cmx as %arg0: memref<1x1x1x1000xf16, [@CMX_NN, 1]>)     // Inputs/outputs buffers for generic operation interface
                    outputs(%out_tile1_cmx as %arg1: memref<1x1x1x1000xf16, [@CMX_NN, 1]>)   // and their mapping to inner region.
                    on tile 1                                                                // The tile index to execute act shaves on

        -> memref<1x1x1x1000xf16, [@CMX_NN, 1]> {

                // The arguments mapping, the order must match the kernel parameter structure.
                VPUIP.SW.Kernel.run(%arg0, %arg1)
                    : memref<1x1x1x1000xf16, [@CMX_NN, 1]>
                    , memref<1x1x1x1000xf16, [@CMX_NN, 1]>
        }
    }

    VPURT.Task waits(%b1 : !VPURT.Barrier) {
       %0 = VPUIP.NNDMA inputs(%out_tile1_cmx : memref<1x1x1x1000xf16, [@CMX_NN, 1]>) outputs(%2 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
    }

    return %2: memref<1x1x1x1000xf16>

}


}

//CHECK: %[[VAL0:.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 1]>
//CHECK-NEXT: %[[VAL1:.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 1]>
//CHECK-NEXT: %[[VAL2:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPUIPRegMapped.Index<0>
//CHECK-NEXT: %[[VAL3:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPUIPRegMapped.Index<1>
//CHECK-NOT: VPURT.Task
//CHECK-NEXT: %[[VAL4:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%[[VAL5:.*]] : memref<1x1x1x1000xf16>) outputs(%[[VAL0]] : memref<1x1x1x1000xf16, [@CMX_NN, 1]>) updates(%[[VAL2]] : !VPUIPRegMapped.Index<0>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<0>
//CHECK-NEXT: %[[VAL6:.*]] = VPUIPRegMapped.DeclareKernelText kernel_path([[VAL7:.*]]) -> !VPUIPRegMapped.Index<0>
//CHECK-NEXT: %[[VAL8:.*]] = VPUIPRegMapped.DeclareKernelArgs kernel_path([[VAL7]]) -> !VPUIPRegMapped.Index<0>
//CHECK-NEXT: %[[VAL9:.*]] = VPUIPRegMapped.DeclareKernelEntry kernel_path([[VAL7]]) -> !VPUIPRegMapped.Index<0>
//CHECK-NEXT: %[[VAL10:.*]] = VPUIPRegMapped.ActKernelRange kernel_text_index(%[[VAL6]] : <0>) kernel_args_index(%[[VAL8]] : <0>) kernel_entry_index(%[[VAL9]] : <0>) -> !VPUIPRegMapped.Index<0>
//CHECK-NEXT: %[[VAL11:.*]] = VPUIPRegMapped.ActKernelInvocation range_index(%[[VAL10]] : <0>) waits(%[[VAL2]] : !VPUIPRegMapped.Index<0>) updates(%[[VAL3]] : !VPUIPRegMapped.Index<1>) tile(1) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<0>
//CHECK-NEXT: %[[VAL12:.*]] = VPUIPRegMapped.KernelParams inputs(%[[VAL0]] : memref<1x1x1x1000xf16, [@CMX_NN, 1]>) outputs(%[[VAL1]] : memref<1x1x1x1000xf16, [@CMX_NN, 1]>) kernel_type([[VAL7]]) kernel_params({{.*}}) -> !VPUIPRegMapped.Index<0>
//CHECK-NOT: VPURT.Task
//CHECK-NEXT: %[[VAL13:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%[[VAL1]] : memref<1x1x1x1000xf16, [@CMX_NN, 1]>) outputs(%[[VAL14:.*]] : memref<1x1x1x1000xf16>) previousDMA(%[[VAL4]] : !VPUIPRegMapped.Index<0>) waits(%[[VAL3]] : !VPUIPRegMapped.Index<1>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<1>
