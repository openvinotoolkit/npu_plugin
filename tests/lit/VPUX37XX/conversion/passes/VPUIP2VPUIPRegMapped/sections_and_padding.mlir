//
// Copyright (C) 2023 Intel Corporation
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
    func private @builtin_sigmoid(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "sigmoid_fp16.c", VPU.kernel_entry = "sigmoid_fp16"}
    func private @builtin_softmax(memref<*xf16>, memref<*xf16>, i64) attributes {VPU.kernel_code = "single_shave_softmax.cpp", VPU.kernel_entry = "singleShaveSoftmax"}
  }
  func @main(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    %0 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer "CMX_NN" [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer "CMX_NN" [0] <4000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer "CMX_NN" [0] <6000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %4 = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPUIPRegMapped.Index<0>
    %5 = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPUIPRegMapped.Index<1>
    %6 = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<2, -1> -> !VPUIPRegMapped.Index<2>
    %7 = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<3, -1> -> !VPUIPRegMapped.Index<3>
    %8 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x1x1000xf16>) outputs(%0 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) updates(%4 : !VPUIPRegMapped.Index<0>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<0>
    %9 = VPUIPRegMapped.DeclareKernelText kernel_path("sigmoid_fp16") -> !VPUIPRegMapped.Index<0>
    %10 = VPUIPRegMapped.DeclareKernelArgs kernel_path("sigmoid_fp16") -> !VPUIPRegMapped.Index<0>
    %11 = VPUIPRegMapped.DeclareKernelEntry kernel_path("sigmoid_fp16") -> !VPUIPRegMapped.Index<0>
    %12 = VPUIPRegMapped.ActKernelRange kernel_text_index(%9 : <0>) kernel_args_index(%10 : <0>) kernel_entry_index(%11 : <0>) -> !VPUIPRegMapped.Index<0>
    %13 = VPUIPRegMapped.ActKernelInvocation range_index(%12 : <0>) waits(%4 : !VPUIPRegMapped.Index<0>) updates(%5 : !VPUIPRegMapped.Index<1>) tile(0) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<0>
    %14 = VPUIPRegMapped.KernelParams inputs(%0 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("sigmoid_fp16") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]> : vector<72xui8>) -> !VPUIPRegMapped.Index<0>
    %15 = VPUIPRegMapped.DeclareKernelText kernel_path("singleShaveSoftmax") -> !VPUIPRegMapped.Index<1>
    %16 = VPUIPRegMapped.DeclareKernelArgs kernel_path("singleShaveSoftmax") -> !VPUIPRegMapped.Index<1>
    %17 = VPUIPRegMapped.DeclareKernelEntry kernel_path("singleShaveSoftmax") -> !VPUIPRegMapped.Index<1>
    %18 = VPUIPRegMapped.ActKernelRange kernel_text_index(%15 : <1>) kernel_args_index(%16 : <1>) kernel_entry_index(%17 : <1>) -> !VPUIPRegMapped.Index<1>
    %19 = VPUIPRegMapped.ActKernelInvocation range_index(%18 : <1>) waits(%5 : !VPUIPRegMapped.Index<1>) updates(%6 : !VPUIPRegMapped.Index<2>) tile(0) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<1>
    %20 = VPUIPRegMapped.KernelParams inputs(%1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%2 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("singleShaveSoftmax") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : vector<80xui8>) -> !VPUIPRegMapped.Index<1>
    %21 = VPUIPRegMapped.DeclareKernelText kernel_path("sigmoid_fp16") -> !VPUIPRegMapped.Index<2>
    %22 = VPUIPRegMapped.DeclareKernelArgs kernel_path("sigmoid_fp16") -> !VPUIPRegMapped.Index<2>
    %23 = VPUIPRegMapped.DeclareKernelEntry kernel_path("sigmoid_fp16") -> !VPUIPRegMapped.Index<2>
    %24 = VPUIPRegMapped.ActKernelRange kernel_text_index(%21 : <2>) kernel_args_index(%22 : <2>) kernel_entry_index(%23 : <2>) -> !VPUIPRegMapped.Index<2>
    %25 = VPUIPRegMapped.ActKernelInvocation range_index(%24 : <2>) waits(%6 : !VPUIPRegMapped.Index<2>) updates(%7 : !VPUIPRegMapped.Index<3>) tile(0) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<2>
    %26 = VPUIPRegMapped.KernelParams inputs(%2 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("sigmoid_fp16") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]> : vector<72xui8>) -> !VPUIPRegMapped.Index<2>
    %27 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%3 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x1x1x1000xf16>) previousDMA(%8 : !VPUIPRegMapped.Index<0>) waits(%7 : !VPUIPRegMapped.Index<3>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<1>
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
        // CHECK:      VPUIPRegMapped.NetworkMetadata
        // CHECK-NOT:      ELF.Pad
  }
}
