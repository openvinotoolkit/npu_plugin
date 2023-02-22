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
    func private @builtin_hswish(%input : memref<*xf16>, %output : memref<*xf16>)
        attributes {
            VPU.kernel_code = "hswish_fp16.cpp",
            VPU.kernel_entry = "hswish_fp16"
        }

    // management kernel definition
    func private @runtime()
        attributes {
            VPU.kernel_code = "nnActEntry"
        }
}



func @main(%in0: memref<1x1x1x1000xf16>, %in1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {

    %in_tile0_cmx  = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %out_tile0_cmx = VPURT.DeclareBuffer "CMX_NN" [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>

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
    %in_tile1_cmx  = VPURT.DeclareBuffer "CMX_NN" [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %out_tile1_cmx = VPURT.DeclareBuffer "CMX_NN" [0] <4000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>

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
    %in_tile2_cmx  = VPURT.DeclareBuffer "CMX_NN" [0] <6000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %out_tile2_cmx = VPURT.DeclareBuffer "CMX_NN" [0] <8000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>

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

//CHECK: %[[VAL0:.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
//CHECK-NEXT: %[[VAL1:.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
//CHECK-NEXT: %[[VAL2:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPUIPRegMapped.Index<0>
//CHECK-NEXT: %[[VAL3:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPUIPRegMapped.Index<1>
//CHECK-NOT: VPURT.Task
//CHECK-NEXT: %[[VAL4:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%[[VAL5:.*]] : memref<1x1x1x1000xf16>) outputs(%[[VAL0]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) updates(%[[VAL2]] : !VPUIPRegMapped.Index<0>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<0>
//CHECK-NEXT: %[[VAL6:.*]] = VPUIPRegMapped.DeclareKernelText kernel_path([[VAL7:.*]]) -> !VPUIPRegMapped.Index<0>
//CHECK-NEXT: %[[VAL8:.*]] = VPUIPRegMapped.DeclareKernelArgs kernel_path([[VAL7]]) -> !VPUIPRegMapped.Index<0>
//CHECK-NEXT: %[[VAL9:.*]] = VPUIPRegMapped.DeclareKernelEntry kernel_path([[VAL7]]) -> !VPUIPRegMapped.Index<0>
//CHECK-NEXT: %[[VAL10:.*]] = VPUIPRegMapped.ActKernelRange kernel_text_index(%[[VAL6]] : <0>) kernel_args_index(%[[VAL8]] : <0>) kernel_entry_index(%[[VAL9]] : <0>) -> !VPUIPRegMapped.Index<0>
//CHECK-NEXT: %[[VAL11:.*]] = VPUIPRegMapped.ActKernelInvocation range_index(%[[VAL10]] : <0>) waits(%[[VAL2]] : !VPUIPRegMapped.Index<0>) updates(%[[VAL3]] : !VPUIPRegMapped.Index<1>) tile(0) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<0>
//CHECK-NEXT: %[[VAL12:.*]] = VPUIPRegMapped.KernelParams inputs(%[[VAL0]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%[[VAL1]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type([[VAL7]]) kernel_params({{.*}}) -> !VPUIPRegMapped.Index<0>
//CHECK-NOT: VPURT.Task
//CHECK-NEXT: %[[VAL13:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%[[VAL1]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%[[VAL14:.*]] : memref<1x1x1x1000xf16>) previousDMA(%[[VAL4]] : !VPUIPRegMapped.Index<0>) waits(%[[VAL3]] : !VPUIPRegMapped.Index<1>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<1>


//CHECK: %[[VAL15:.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
//CHECK-NEXT: %[[VAL16:.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <4000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
//CHECK-NEXT: %[[VAL17:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPUIPRegMapped.Index<2>
//CHECK-NEXT: %[[VAL18:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPUIPRegMapped.Index<3>
//CHECK-NOT: VPURT.Task
//CHECK-NEXT: %[[VAL19:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%[[VAL14]] : memref<1x1x1x1000xf16>) outputs(%[[VAL15]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) previousDMA(%[[VAL13]] : !VPUIPRegMapped.Index<1>) updates(%[[VAL17]] : !VPUIPRegMapped.Index<2>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<2>
//CHECK-NEXT: %[[VAL20:.*]] = VPUIPRegMapped.DeclareKernelText kernel_path([[VAL7]]) -> !VPUIPRegMapped.Index<1>
//CHECK-NEXT: %[[VAL21:.*]] = VPUIPRegMapped.DeclareKernelArgs kernel_path([[VAL7]]) -> !VPUIPRegMapped.Index<1>
//CHECK-NEXT: %[[VAL22:.*]] = VPUIPRegMapped.DeclareKernelEntry kernel_path([[VAL7]]) -> !VPUIPRegMapped.Index<1>
//CHECK-NEXT: %[[VAL23:.*]] = VPUIPRegMapped.ActKernelRange kernel_text_index(%[[VAL20]] : <1>) kernel_args_index(%[[VAL21]] : <1>) kernel_entry_index(%[[VAL22]] : <1>) -> !VPUIPRegMapped.Index<1>
//CHECK-NEXT: %[[VAL24:.*]] = VPUIPRegMapped.ActKernelInvocation range_index(%[[VAL23]] : <1>) waits(%[[VAL17]] : !VPUIPRegMapped.Index<2>) updates(%[[VAL18]] : !VPUIPRegMapped.Index<3>) tile(0) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<1>
//CHECK-NEXT: %[[VAL25:.*]] = VPUIPRegMapped.KernelParams inputs(%[[VAL15]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%[[VAL16]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type([[VAL7]]) kernel_params({{.*}}) -> !VPUIPRegMapped.Index<1>
//CHECK-NOT: VPURT.Task
//CHECK-NEXT: %[[VAL26:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%[[VAL16]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%[[VAL14]] : memref<1x1x1x1000xf16>) previousDMA(%[[VAL19]] : !VPUIPRegMapped.Index<2>) waits(%[[VAL18]] : !VPUIPRegMapped.Index<3>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<3>


//CHECK: %[[VAL27:.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <6000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
//CHECK-NEXT: %[[VAL28:.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <8000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
//CHECK-NEXT: %[[VAL29:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPUIPRegMapped.Index<4>
//CHECK-NEXT: %[[VAL30:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPUIPRegMapped.Index<5>
//CHECK-NOT: VPURT.Task
//CHECK-NEXT: %[[VAL31:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%[[VAL14]] : memref<1x1x1x1000xf16>) outputs(%[[VAL27]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) previousDMA(%[[VAL26]] : !VPUIPRegMapped.Index<3>) updates(%[[VAL29]] : !VPUIPRegMapped.Index<4>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<4>
//CHECK-NEXT: %[[VAL32:.*]] = VPUIPRegMapped.DeclareKernelText kernel_path([[VAL7]]) -> !VPUIPRegMapped.Index<2>
//CHECK-NEXT: %[[VAL33:.*]] = VPUIPRegMapped.DeclareKernelArgs kernel_path([[VAL7]]) -> !VPUIPRegMapped.Index<2>
//CHECK-NEXT: %[[VAL34:.*]] = VPUIPRegMapped.DeclareKernelEntry kernel_path([[VAL7]]) -> !VPUIPRegMapped.Index<2>
//CHECK-NEXT: %[[VAL35:.*]] = VPUIPRegMapped.ActKernelRange kernel_text_index(%[[VAL32]] : <2>) kernel_args_index(%[[VAL33]] : <2>) kernel_entry_index(%[[VAL34]] : <2>) -> !VPUIPRegMapped.Index<2>
//CHECK-NEXT: %[[VAL36:.*]] = VPUIPRegMapped.ActKernelInvocation range_index(%[[VAL35]] : <2>) waits(%[[VAL29]] : !VPUIPRegMapped.Index<4>) updates(%[[VAL30]] : !VPUIPRegMapped.Index<5>) tile(0) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<2>
//CHECK-NEXT: %[[VAL37:.*]] = VPUIPRegMapped.KernelParams inputs(%[[VAL27]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%[[VAL28]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type([[VAL7]]) kernel_params({{.*}}) -> !VPUIPRegMapped.Index<2>
//CHECK-NOT: VPURT.Task
//CHECK-NEXT: %[[VAL38:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%[[VAL28]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%[[VAL14]] : memref<1x1x1x1000xf16>) previousDMA(%[[VAL31]] : !VPUIPRegMapped.Index<4>) waits(%[[VAL30]] : !VPUIPRegMapped.Index<5>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<5>
