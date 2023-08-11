//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=VPUX37XX" %s
// this test can only be (correctly) run manually until E#48620 is solved

module @Test {

  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1000xf16>
  } outputsInfo : {
    DataInfo "sigmoid" : tensor<1x1000xf16>
  }
  module @VPU.SW {
    func.func private @builtin_sigmoid(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "sigmoid_fp16.c", VPU.kernel_entry = "sigmoid_fp16"}
    func.func private @builtin_softmax(memref<*xf16>, memref<*xf16>, i64) attributes {VPU.kernel_code = "single_shave_softmax.cpp", VPU.kernel_entry = "singleShaveSoftmax"}
  }
  func.func @main(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    %0 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer "CMX_NN" [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer "CMX_NN" [0] <4000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer "CMX_NN" [0] <6000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %4 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:0>
    %5 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPURegMapped.Index<0:0:1>
    %6 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<2, -1> -> !VPURegMapped.Index<0:0:2>
    %7 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<3, -1> -> !VPURegMapped.Index<0:0:3>
    %8 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x1x1000xf16>) outputs(%0 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) updates(%4 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
    %9 = VPUMI37XX.DeclareKernelText kernel_path("sigmoid_fp16") -> !VPURegMapped.Index<0:0:0>
    %10 = VPUMI37XX.DeclareKernelArgs kernel_path("sigmoid_fp16") -> !VPURegMapped.Index<0:0:0>
    %11 = VPUMI37XX.DeclareKernelEntry kernel_path("sigmoid_fp16") -> !VPURegMapped.Index<0:0:0>
    %12 = VPUMI37XX.ActKernelRange kernel_text_index(%9 : <0:0:0>) kernel_args_index(%10 : <0:0:0>) kernel_entry_index(%11 : <0:0:0>) -> !VPURegMapped.Index<0:0:0>
    %13 = VPUMI37XX.ActKernelInvocation range_index(%12 : <0:0:0>) waits(%4 : !VPURegMapped.Index<0:0:0>) updates(%5 : !VPURegMapped.Index<0:0:1>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
    %14 = VPUMI37XX.KernelParams inputs(%0 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("sigmoid_fp16") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]> : vector<72xui8>) -> !VPURegMapped.Index<0:0:0>
    %15 = VPUMI37XX.DeclareKernelText kernel_path("singleShaveSoftmax") -> !VPURegMapped.Index<0:0:1>
    %16 = VPUMI37XX.DeclareKernelArgs kernel_path("singleShaveSoftmax") -> !VPURegMapped.Index<0:0:1>
    %17 = VPUMI37XX.DeclareKernelEntry kernel_path("singleShaveSoftmax") -> !VPURegMapped.Index<0:0:1>
    %18 = VPUMI37XX.ActKernelRange kernel_text_index(%15 : <0:0:1>) kernel_args_index(%16 : <0:0:1>) kernel_entry_index(%17 : <0:0:1>) -> !VPURegMapped.Index<0:0:1>
    %19 = VPUMI37XX.ActKernelInvocation range_index(%18 : <0:0:1>) waits(%5 : !VPURegMapped.Index<0:0:1>) updates(%6 : !VPURegMapped.Index<0:0:2>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>
    %20 = VPUMI37XX.KernelParams inputs(%1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%2 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("singleShaveSoftmax") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : vector<80xui8>) -> !VPURegMapped.Index<0:0:1>
    %21 = VPUMI37XX.DeclareKernelText kernel_path("sigmoid_fp16") -> !VPURegMapped.Index<0:0:2>
    %22 = VPUMI37XX.DeclareKernelArgs kernel_path("sigmoid_fp16") -> !VPURegMapped.Index<0:0:2>
    %23 = VPUMI37XX.DeclareKernelEntry kernel_path("sigmoid_fp16") -> !VPURegMapped.Index<0:0:2>
    %24 = VPUMI37XX.ActKernelRange kernel_text_index(%21 : <0:0:2>) kernel_args_index(%22 : <0:0:2>) kernel_entry_index(%23 : <0:0:2>) -> !VPURegMapped.Index<0:0:2>
    %25 = VPUMI37XX.ActKernelInvocation range_index(%24 : <0:0:2>) waits(%6 : !VPURegMapped.Index<0:0:2>) updates(%7 : !VPURegMapped.Index<0:0:3>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:2>
    %26 = VPUMI37XX.KernelParams inputs(%2 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("sigmoid_fp16") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]> : vector<72xui8>) -> !VPURegMapped.Index<0:0:2>
    %27 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%3 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x1x1x1000xf16>) previousDMA(%8 : !VPURegMapped.Index<0:0:0>) waits(%7 : !VPURegMapped.Index<0:0:3>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>
    return %arg1 : memref<1x1x1x1000xf16>

    // CHECK:      ELF.CreateSection {{.*}}dmaTasks
        // CHECK:      ELF.PutOpInSection
        // CHECK-NOT:      ELF.Pad
        // CHECK:      ELF.PutOpInSection
        // CHECK-NOT:      ELF.Pad

    // CHECK:      ELF.CreateSection {{.*}}BarrierConfigs
        // CHECK:      ELF.PutOpInSection
        // CHECK-NOT:      ELF.Pad
        // CHECK:      ELF.PutOpInSection
        // CHECK-NOT:      ELF.Pad
        // CHECK:      ELF.PutOpInSection
        // CHECK-NOT:      ELF.Pad
        // CHECK:      ELF.PutOpInSection
        // CHECK-NOT:      ELF.Pad

    // CHECK:      ELF.CreateSection {{.*}}KernelText
        // CHECK:      ELF.PutOpInSection
        // CHECK:      ELF.Pad
        // CHECK:      ELF.PutOpInSection
        // CHECK:      ELF.Pad
        // CHECK:      ELF.PutOpInSection
        // CHECK-NOT:      ELF.Pad

    // CHECK:      ELF.CreateSection {{.*}}KernelData
        // CHECK:      ELF.PutOpInSection
        // CHECK-NOT:      ELF.Pad
        // CHECK:      ELF.PutOpInSection
        // CHECK-NOT:      ELF.Pad
        // CHECK:      ELF.PutOpInSection
        // CHECK-NOT:      ELF.Pad

    // CHECK:      ELF.CreateSection {{.*}}KernelParams
        // CHECK:      ELF.PutOpInSection
        // CHECK:      ELF.Pad
        // CHECK:      ELF.PutOpInSection
        // CHECK:      ELF.Pad
        // CHECK:      ELF.PutOpInSection
        // CHECK-NOT:      ELF.Pad

    // CHECK:      ELF.CreateSection {{.*}}ActKernelRanges
        // CHECK:      ELF.PutOpInSection
        // CHECK-NOT:      ELF.Pad
        // CHECK:      ELF.PutOpInSection
        // CHECK-NOT:      ELF.Pad
        // CHECK:      ELF.PutOpInSection
        // CHECK-NOT:      ELF.Pad

    // CHECK:      ELF.CreateSection {{.*}}ActKernelInvocations
        // CHECK:      ELF.PutOpInSection
        // CHECK-NOT:      ELF.Pad
        // CHECK:      ELF.PutOpInSection
        // CHECK-NOT:      ELF.Pad
        // CHECK:      ELF.PutOpInSection
        // CHECK-NOT:      ELF.Pad

    // CHECK:      ELF.CreateSection {{.*}}MappedInference
        // CHECK:      ELF.PutOpInSection
        // CHECK-NOT:      ELF.Pad

    // CHECK:      ELF.CreateMetadataSection
        // CHECK:      VPUMI37XX.NetworkMetadata
        // CHECK-NOT:      ELF.Pad
  }
}
