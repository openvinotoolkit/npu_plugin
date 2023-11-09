//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --convert-VPUIP-to-VPUMI37XX %s | FileCheck %s
//

module @Test {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "input" : tensor<1x1000xf16>
    }
    outputsInfo : {
        IE.DataInfo "hswish" : tensor<1x1000xf16>
    }

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [
        4096, // Size in bytes for the SHAVEs in the first tile.
        4096  // Size in bytes for the SHAVEs in the second tile.
    ]


// Sub-module, which holds SW kernel declarations and optional implementations.
// Used to group those declarations for faster access.
module @VPU.SW {
    // The declaration should match C++ params structure in decomposed form.
    // `memref` will be translated to `MemRefData`, while raw scalars will be translated as is.
    func.func private @builtin_hswish(%input : memref<*xf16>, %output : memref<*xf16>)
        attributes {
            VPU.kernel_code = "hswish_fp16.cpp",
            VPU.kernel_entry = "hswish_fp16"
        }

    // management kernel definition
    func.func private @runtime()
        attributes {
            VPU.kernel_code = "nnActEntry"
        }
}



func.func @main(%in0: memref<1x1x1x1000xf16>, %in1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {

    %in_tile0_cmx  = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %out_tile0_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    %b0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %b1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%in0 : memref<1x1x1x1000xf16>) outputs(%in_tile0_cmx : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%b0  : !VPURT.Barrier) updates(%b1  : !VPURT.Barrier) {
        VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>}
                    @VPU.SW::@builtin_hswish
                    inputs(%in_tile0_cmx as %arg0: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)
                    outputs(%out_tile0_cmx as %arg1: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)
                    on tile 0

        -> memref<1x1x1x1000xf16, [@CMX_NN, 0]> {

                VPUIP.SW.Kernel.run(%arg0, %arg1)
                    : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
                    , memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        }
    }

    VPURT.Task waits(%b1 : !VPURT.Barrier) {
        %0 = VPUIP.NNDMA inputs(%out_tile0_cmx : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%in1 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
    }

    // The second HSwish
    %in_tile1_cmx  = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %out_tile1_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <4000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    %b2 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %b3 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier

    VPURT.Task updates(%b2 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%in1 : memref<1x1x1x1000xf16>) outputs(%in_tile1_cmx : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%b2  : !VPURT.Barrier) updates(%b3  : !VPURT.Barrier) {
        VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>}
                    @VPU.SW::@builtin_hswish
                    inputs(%in_tile1_cmx as %arg0: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)
                    outputs(%out_tile1_cmx as %arg1: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)
                    on tile 0

        -> memref<1x1x1x1000xf16, [@CMX_NN, 0]> {

                VPUIP.SW.Kernel.run(%arg0, %arg1)
                    : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
                    , memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        }
    }

    VPURT.Task waits(%b3 : !VPURT.Barrier) {
        %0 = VPUIP.NNDMA inputs(%out_tile1_cmx : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%in1 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
    }

    // The third HSwish
    %in_tile2_cmx  = VPURT.DeclareBuffer <CMX_NN> [0] <6000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %out_tile2_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <8000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    %b4 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %b5 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier

    VPURT.Task updates(%b4 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%in1 : memref<1x1x1x1000xf16>) outputs(%in_tile2_cmx : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%b4  : !VPURT.Barrier) updates(%b5  : !VPURT.Barrier) {
        VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>}
                    @VPU.SW::@builtin_hswish
                    inputs(%in_tile2_cmx as %arg0: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)
                    outputs(%out_tile2_cmx as %arg1: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)
                    on tile 0

        -> memref<1x1x1x1000xf16, [@CMX_NN, 0]> {

                VPUIP.SW.Kernel.run(%arg0, %arg1)
                    : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
                    , memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        }
    }

    VPURT.Task waits(%b5 : !VPURT.Barrier) {
        %0 = VPUIP.NNDMA inputs(%out_tile2_cmx : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%in1 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
    }
    return %in1: memref<1x1x1x1000xf16>

}


}

//CHECK: %[[VAL0:.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
//CHECK-NEXT: %[[VAL1:.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
//CHECK-NEXT: %[[VAL2:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: %[[VAL3:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPURegMapped.Index<0:0:1>
//CHECK-NOT: VPURT.Task
//CHECK-NEXT: %[[VAL4:.*]] = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%[[VAL5:.*]] : memref<1x1x1x1000xf16>) outputs(%[[VAL0]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) updates(%[[VAL2]] : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: %[[VAL6:.*]] = VPUMI37XX.DeclareKernelText kernel_path([[VAL7:.*]]) -> !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: %[[VAL8:.*]] = VPUMI37XX.DeclareKernelEntry kernel_path([[VAL7]]) -> !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: %[[VAL9:.*]] = VPUMI37XX.DeclareKernelArgs kernel_path([[VAL7]]) -> !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: %[[VAL10:.*]] = VPUMI37XX.ActKernelRange kernel_text_index(%[[VAL6]] : <0:0:0>) kernel_args_index(%[[VAL9]] : <0:0:0>) kernel_entry_index(%[[VAL8]] : <0:0:0>) -> !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: %[[VAL11:.*]] = VPUMI37XX.ActKernelInvocation range_index(%[[VAL10]] : <0:0:0>) waits(%[[VAL2]] : !VPURegMapped.Index<0:0:0>) updates(%[[VAL3]] : !VPURegMapped.Index<0:0:1>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: %[[VAL12:.*]] = VPUMI37XX.KernelParams inputs(%[[VAL0]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%[[VAL1]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type([[VAL7]]) kernel_params({{.*}}) -> !VPURegMapped.Index<0:0:0>
//CHECK-NOT: VPURT.Task
//CHECK-NEXT: %[[VAL13:.*]] = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%[[VAL1]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%[[VAL14:.*]] : memref<1x1x1x1000xf16>) previousDMA(%[[VAL4]] : !VPURegMapped.Index<0:0:0>) waits(%[[VAL3]] : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>


//CHECK: %[[VAL15:.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
//CHECK-NEXT: %[[VAL16:.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
//CHECK-NEXT: %[[VAL17:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:2>
//CHECK-NEXT: %[[VAL18:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPURegMapped.Index<0:0:3>
//CHECK-NOT: VPURT.Task
//CHECK-NEXT: %[[VAL19:.*]] = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%[[VAL14]] : memref<1x1x1x1000xf16>) outputs(%[[VAL15]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) previousDMA(%[[VAL13]] : !VPURegMapped.Index<0:0:1>) updates(%[[VAL17]] : !VPURegMapped.Index<0:0:2>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:2>
//CHECK-NEXT: %[[VAL20:.*]] = VPUMI37XX.DeclareKernelArgs kernel_path([[VAL7]]) -> !VPURegMapped.Index<0:0:1>
//CHECK-NEXT: %[[VAL21:.*]] = VPUMI37XX.ActKernelRange kernel_text_index(%[[VAL6]] : <0:0:0>) kernel_args_index(%[[VAL20]] : <0:0:1>) kernel_entry_index(%[[VAL8]] : <0:0:0>) -> !VPURegMapped.Index<0:0:1>
//CHECK-NEXT: %[[VAL22:.*]] = VPUMI37XX.ActKernelInvocation range_index(%[[VAL21]] : <0:0:1>) waits(%[[VAL17]] : !VPURegMapped.Index<0:0:2>) updates(%[[VAL18]] : !VPURegMapped.Index<0:0:3>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>
//CHECK-NEXT: %[[VAL23:.*]] = VPUMI37XX.KernelParams inputs(%[[VAL15]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%[[VAL16]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type([[VAL7]]) kernel_params({{.*}}) -> !VPURegMapped.Index<0:0:1>
//CHECK-NOT: VPURT.Task
//CHECK-NEXT: %[[VAL24:.*]] = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%[[VAL16]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%[[VAL14]] : memref<1x1x1x1000xf16>) previousDMA(%[[VAL19]] : !VPURegMapped.Index<0:0:2>) waits(%[[VAL18]] : !VPURegMapped.Index<0:0:3>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:3>


//CHECK: %[[VAL25:.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <6000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
//CHECK-NEXT: %[[VAL26:.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <8000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
//CHECK-NEXT: %[[VAL27:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:4>
//CHECK-NEXT: %[[VAL28:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPURegMapped.Index<0:0:5>
//CHECK-NOT: VPURT.Task
//CHECK-NEXT: %[[VAL29:.*]] = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%[[VAL14]] : memref<1x1x1x1000xf16>) outputs(%[[VAL25]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) previousDMA(%[[VAL24]] : !VPURegMapped.Index<0:0:3>) updates(%[[VAL27]] : !VPURegMapped.Index<0:0:4>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:4>
//CHECK-NEXT: %[[VAL30:.*]] = VPUMI37XX.DeclareKernelArgs kernel_path([[VAL7]]) -> !VPURegMapped.Index<0:0:2>
//CHECK-NEXT: %[[VAL31:.*]] = VPUMI37XX.ActKernelRange kernel_text_index(%[[VAL6]] : <0:0:0>) kernel_args_index(%[[VAL30]] : <0:0:2>) kernel_entry_index(%[[VAL8]] : <0:0:0>) -> !VPURegMapped.Index<0:0:2>
//CHECK-NEXT: %[[VAL33:.*]] = VPUMI37XX.ActKernelInvocation range_index(%[[VAL31]] : <0:0:2>) waits(%[[VAL27]] : !VPURegMapped.Index<0:0:4>) updates(%[[VAL28]] : !VPURegMapped.Index<0:0:5>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:2>
//CHECK-NEXT: %[[VAL33:.*]] = VPUMI37XX.KernelParams inputs(%[[VAL25]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%[[VAL26]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type([[VAL7]]) kernel_params({{.*}}) -> !VPURegMapped.Index<0:0:2>
//CHECK-NOT: VPURT.Task
//CHECK-NEXT: %[[VAL34:.*]] = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%[[VAL26]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%[[VAL14]] : memref<1x1x1x1000xf16>) previousDMA(%[[VAL29]] : !VPURegMapped.Index<0:0:4>) waits(%[[VAL28]] : !VPURegMapped.Index<0:0:5>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:5>
