//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --convert-VPUIPRegMapped-to-ELF %s | FileCheck %s
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
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1000xf16>
  } outputsInfo : {
    DataInfo "hswish" : tensor<1x1000xf16>
  }
  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096]
  module @VPU.SW {
    func private @builtin_hswish(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "hswish_fp16.cpp", VPU.kernel_entry = "hswish_fp16"}
    func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }
  func @main(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    %0 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer "CMX_NN" [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %2 = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPUIPRegMapped.Index<0>
    %3 = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPUIPRegMapped.Index<1>
    %4 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x1x1000xf16>) outputs(%0 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) updates(%2 : !VPUIPRegMapped.Index<0>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<0>
    %5 = VPUIPRegMapped.DeclareKernelText kernel_path("hswish_fp16") -> !VPUIPRegMapped.Index<0>
    %6 = VPUIPRegMapped.DeclareKernelArgs kernel_path("hswish_fp16") -> !VPUIPRegMapped.Index<0>
    %7 = VPUIPRegMapped.DeclareKernelEntry kernel_path("hswish_fp16") -> !VPUIPRegMapped.Index<0>
    %8 = VPUIPRegMapped.ActKernelRange kernel_text_index(%5 : <0>) kernel_args_index(%6 : <0>) kernel_entry_index(%7 : <0>) -> !VPUIPRegMapped.Index<0>
    %9 = VPUIPRegMapped.ActKernelInvocation range_index(%8 : <0>) waits(%2 : !VPUIPRegMapped.Index<0>) updates(%3 : !VPUIPRegMapped.Index<1>) tile(0) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<0>
    %10 = VPUIPRegMapped.KernelParams inputs(%0 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("hswish_fp16") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]> : vector<72xui8>) -> !VPUIPRegMapped.Index<0>
    %11 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x1x1x1000xf16>) previousDMA(%4 : !VPUIPRegMapped.Index<0>) waits(%3 : !VPUIPRegMapped.Index<1>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<1>
    %12 = VPURT.DeclareBuffer "CMX_NN" [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %13 = VPURT.DeclareBuffer "CMX_NN" [0] <4000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %14 = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPUIPRegMapped.Index<2>
    %15 = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPUIPRegMapped.Index<3>
    %16 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg1 : memref<1x1x1x1000xf16>) outputs(%12 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) previousDMA(%11 : !VPUIPRegMapped.Index<1>) updates(%14 : !VPUIPRegMapped.Index<2>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<2>
    %17 = VPUIPRegMapped.DeclareKernelText kernel_path("hswish_fp16") -> !VPUIPRegMapped.Index<1>
    %18 = VPUIPRegMapped.DeclareKernelArgs kernel_path("hswish_fp16") -> !VPUIPRegMapped.Index<1>
    %19 = VPUIPRegMapped.DeclareKernelEntry kernel_path("hswish_fp16") -> !VPUIPRegMapped.Index<1>
    %20 = VPUIPRegMapped.ActKernelRange kernel_text_index(%17 : <1>) kernel_args_index(%18 : <1>) kernel_entry_index(%19 : <1>) -> !VPUIPRegMapped.Index<1>
    %21 = VPUIPRegMapped.ActKernelInvocation range_index(%20 : <1>) waits(%14 : !VPUIPRegMapped.Index<2>) updates(%15 : !VPUIPRegMapped.Index<3>) tile(0) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<1>
    %22 = VPUIPRegMapped.KernelParams inputs(%12 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%13 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("hswish_fp16") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]> : vector<72xui8>) -> !VPUIPRegMapped.Index<1>
    %23 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%13 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x1x1x1000xf16>) previousDMA(%16 : !VPUIPRegMapped.Index<2>) waits(%15 : !VPUIPRegMapped.Index<3>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<3>
    %24 = VPURT.DeclareBuffer "CMX_NN" [0] <6000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %25 = VPURT.DeclareBuffer "CMX_NN" [0] <8000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %26 = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPUIPRegMapped.Index<4>
    %27 = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPUIPRegMapped.Index<5>
    %28 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg1 : memref<1x1x1x1000xf16>) outputs(%24 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) previousDMA(%23 : !VPUIPRegMapped.Index<3>) updates(%26 : !VPUIPRegMapped.Index<4>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<4>
    %29 = VPUIPRegMapped.DeclareKernelText kernel_path("hswish_fp16") -> !VPUIPRegMapped.Index<2>
    %30 = VPUIPRegMapped.DeclareKernelArgs kernel_path("hswish_fp16") -> !VPUIPRegMapped.Index<2>
    %31 = VPUIPRegMapped.DeclareKernelEntry kernel_path("hswish_fp16") -> !VPUIPRegMapped.Index<2>
    %32 = VPUIPRegMapped.ActKernelRange kernel_text_index(%29 : <2>) kernel_args_index(%30 : <2>) kernel_entry_index(%31 : <2>) -> !VPUIPRegMapped.Index<2>
    %33 = VPUIPRegMapped.ActKernelInvocation range_index(%32 : <2>) waits(%26 : !VPUIPRegMapped.Index<4>) updates(%27 : !VPUIPRegMapped.Index<5>) tile(0) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<2>
    %34 = VPUIPRegMapped.KernelParams inputs(%24 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%25 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("hswish_fp16") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]> : vector<72xui8>) -> !VPUIPRegMapped.Index<2>
    %35 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%25 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x1x1x1000xf16>) previousDMA(%28 : !VPUIPRegMapped.Index<4>) waits(%27 : !VPUIPRegMapped.Index<5>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<5>
    %36 = VPUIPRegMapped.MappedInference dmas(%4 : !VPUIPRegMapped.Index<0>) actKernelRanges(%8 : !VPUIPRegMapped.Index<0>) actKernelInvocations(%9 : !VPUIPRegMapped.Index<0>) barriers(%2 : !VPUIPRegMapped.Index<0>) dmaCount([6]) invariantCount(0) variantCount(0) actKernelRangesCount(3) actKernelInvocationsCount(3) barrierCount(6) -> !VPUIPRegMapped.Index<0>
    return %arg1 : memref<1x1x1x1000xf16>
  }
}

//CHECK: %[[VAL0:.*]] = VPURT.DeclareBuffer
//CHECK: %[[VAL1:.*]] = VPURT.DeclareBuffer
//CHECK: %[[VAL2:.*]] = VPUIPRegMapped.ConfigureBarrier
//CHECK: %[[VAL3:.*]] = VPUIPRegMapped.ConfigureBarrier
//CHECK: %[[VAL4:.*]] = VPUIPRegMapped.NNDMA
//CHECK: %[[VAL5:.*]] = VPUIPRegMapped.DeclareKernelText
//CHECK: %[[VAL6:.*]] = VPUIPRegMapped.DeclareKernelArgs
//CHECK: %[[VAL7:.*]] = VPUIPRegMapped.DeclareKernelEntry
//CHECK: %[[VAL8:.*]] = VPUIPRegMapped.ActKernelRange
//CHECK: %[[VAL9:.*]] = VPUIPRegMapped.ActKernelInvocation
//CHECK: %[[VAL10:.*]] = VPUIPRegMapped.KernelParams
//CHECK: %[[VAL11:.*]] = VPUIPRegMapped.NNDMA

//CHECK: %[[VAL12:.*]] = VPURT.DeclareBuffer
//CHECK: %[[VAL13:.*]] = VPURT.DeclareBuffer
//CHECK: %[[VAL14:.*]] = VPUIPRegMapped.ConfigureBarrier
//CHECK: %[[VAL15:.*]] = VPUIPRegMapped.ConfigureBarrier
//CHECK: %[[VAL16:.*]] = VPUIPRegMapped.NNDMA
//CHECK: %[[VAL17:.*]] = VPUIPRegMapped.DeclareKernelText
//CHECK: %[[VAL18:.*]] = VPUIPRegMapped.DeclareKernelArgs
//CHECK: %[[VAL19:.*]] = VPUIPRegMapped.DeclareKernelEntry
//CHECK: %[[VAL20:.*]] = VPUIPRegMapped.ActKernelRange
//CHECK: %[[VAL21:.*]] = VPUIPRegMapped.ActKernelInvocation
//CHECK: %[[VAL22:.*]] = VPUIPRegMapped.KernelParams
//CHECK: %[[VAL23:.*]] = VPUIPRegMapped.NNDMA

//CHECK: %[[VAL24:.*]] = VPURT.DeclareBuffer
//CHECK: %[[VAL25:.*]] = VPURT.DeclareBuffer
//CHECK: %[[VAL26:.*]] = VPUIPRegMapped.ConfigureBarrier
//CHECK: %[[VAL27:.*]] = VPUIPRegMapped.ConfigureBarrier
//CHECK: %[[VAL28:.*]] = VPUIPRegMapped.NNDMA
//CHECK: %[[VAL29:.*]] = VPUIPRegMapped.DeclareKernelText
//CHECK: %[[VAL30:.*]] = VPUIPRegMapped.DeclareKernelArgs
//CHECK: %[[VAL31:.*]] = VPUIPRegMapped.DeclareKernelEntry
//CHECK: %[[VAL32:.*]] = VPUIPRegMapped.ActKernelRange
//CHECK: %[[VAL33:.*]] = VPUIPRegMapped.ActKernelInvocation
//CHECK: %[[VAL34:.*]] = VPUIPRegMapped.KernelParams
//CHECK: %[[VAL35:.*]] = VPUIPRegMapped.NNDMA

//CHECK-DAG: %[[VAL36:.*]] = VPURT.DeclareBuffer "DDR"
//CHECK-NEXT: %[[VAL37:.*]] = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags(VPU_SHF_PROC_SHAVE) {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".bss.actShaveStack_0"} -> !ELF.Section {
//CHECK-NEXT: ELF.PutOpInSection %[[VAL36]] : memref
//CHECK: %[[VAL38:.*]]  = ELF.Symbol %[[VAL37]] name("sym_actShaveStack_0") : !ELF.Section

//CHECK-DAG: %[[VAL39:.*]] = VPURT.DeclareBuffer "DDR"
//CHECK-NEXT: %[[VAL40:.*]] = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags(VPU_SHF_PROC_SHAVE) {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".bss.actShaveStack_1"} -> !ELF.Section {
//CHECK-NEXT: ELF.PutOpInSection %[[VAL39]] : memref
//CHECK: %[[VAL41:.*]]  = ELF.Symbol %[[VAL40]] name("sym_actShaveStack_1") : !ELF.Section

//CHECK-DAG: %[[VAL42:.*]] = VPURT.DeclareBuffer "DDR"
//CHECK-NEXT: %[[VAL43:.*]] = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags(VPU_SHF_PROC_SHAVE) {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".bss.actShaveStack_2"} -> !ELF.Section {
//CHECK-NEXT: ELF.PutOpInSection %[[VAL42]] : memref
//CHECK: %[[VAL44:.*]]  = ELF.Symbol %[[VAL43]] name("sym_actShaveStack_2") : !ELF.Section

//CHECK-DAG: %[[VAL45:.*]] = VPURT.DeclareBuffer "DDR"
//CHECK-NEXT: %[[VAL46:.*]] = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags(VPU_SHF_PROC_SHAVE) {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".bss.actShaveStack_3"} -> !ELF.Section {
//CHECK-NEXT: ELF.PutOpInSection %[[VAL45]] : memref
//CHECK: %[[VAL47:.*]]  = ELF.Symbol %[[VAL46]] name("sym_actShaveStack_3") : !ELF.Section

//CHECK-DAG: %[[VAL48:.*]] = VPUIPRegMapped.ActShaveRt kernel("nnActEntry") -> !VPUIPRegMapped.Index<0>
//CHECK-NEXT: %[[VAL49:.*]] = ELF.CreateSection {{.*}} {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".text.actKernelRtConfigSec"} -> !ELF.Section {
//CHECK-NEXT: ELF.PutOpInSection %[[VAL48]] : !VPUIPRegMapped.Index<0>
//CHECK: %[[VAL50:.*]]  = ELF.Symbol %[[VAL49]] name("sym_actKernelRtConfigsSec") : !ELF.Section

//CHECK-DAG: %[[VAL51:.*]] = ELF.CreateSymbolTableSection secName(".symtab.actKernelRtConfig")
//CHECK-NEXT: ELF.PutOpInSection %[[VAL50]] : !ELF.Symbol
//CHECK-NEXT: ELF.PutOpInSection %[[VAL38]] : !ELF.Symbol
//CHECK-NEXT: ELF.PutOpInSection %[[VAL41]] : !ELF.Symbol
//CHECK-NEXT: ELF.PutOpInSection %[[VAL44]] : !ELF.Symbol
//CHECK-NEXT: ELF.PutOpInSection %[[VAL47]] : !ELF.Symbol

//CHECK: %[[VAL52:.*]] = VPUIPRegMapped.MappedInference
//CHECK-SAME: dmas(%[[VAL4]] : !VPUIPRegMapped.Index<0>)
//CHECK-SAME: actKernelRanges(%[[VAL8]] : !VPUIPRegMapped.Index<0>)
//CHECK-SAME: actKernelInvocations(%[[VAL9]] : !VPUIPRegMapped.Index<0>)
//CHECK-SAME: barriers(%[[VAL2]] : !VPUIPRegMapped.Index<0>)
//CHECK-SAME: actShaveRt(%[[VAL48]] : !VPUIPRegMapped.Index<0>)
//CHECK-SAME: actShaveStacks(%[[VAL36]], %[[VAL39]], %[[VAL42]], %[[VAL45]] : {{.*}}>)
//CHECK-SAME: dmaCount([6]) invariantCount(0) variantCount(0) actKernelRangesCount(3) actKernelInvocationsCount(3) barrierCount(6)
//CHECK-SAME: -> !VPUIPRegMapped.Index<0>


//CHECK-DAG: %[[DMASEC:.*]] = ELF.CreateSection {{.*}} secName = ".text.dmaTasks0"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL4]] : !VPUIPRegMapped.Index<0>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL11]] : !VPUIPRegMapped.Index<1>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL16]] : !VPUIPRegMapped.Index<2>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL23]] : !VPUIPRegMapped.Index<3>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL28]] : !VPUIPRegMapped.Index<4>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL35]] : !VPUIPRegMapped.Index<5>

//CHECK-DAG: %[[BARSEC:.*]] = ELF.CreateSection {{.*}} secName = ".text.BarrierConfigs"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL2]] : !VPUIPRegMapped.Index<0>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL3]] : !VPUIPRegMapped.Index<1>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL14]] : !VPUIPRegMapped.Index<2>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL15]] : !VPUIPRegMapped.Index<3>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL26]] : !VPUIPRegMapped.Index<4>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL27]] : !VPUIPRegMapped.Index<5>

//CHECK-DAG: %[[KERNELTEXT:.*]] = ELF.CreateSection {{.*}} secName = ".text.KernelText"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL5]] : !VPUIPRegMapped.Index<0>
//CHECK-NEXT: ELF.Pad {{.*}}
//CHECK-NEXT: ELF.PutOpInSection %[[VAL17]] : !VPUIPRegMapped.Index<1>
//CHECK-NEXT: ELF.Pad {{.*}}
//CHECK-NEXT: ELF.PutOpInSection %[[VAL29]] : !VPUIPRegMapped.Index<2>

//CHECK-DAG: %[[KERNELDATA:.*]] = ELF.CreateSection {{.*}} secName = ".text.KernelData"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL6]] : !VPUIPRegMapped.Index<0>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL18]] : !VPUIPRegMapped.Index<1>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL30]] : !VPUIPRegMapped.Index<2>

//CHECK-DAG: %[[KERNELPARAMS:.*]] = ELF.CreateSection {{.*}} secName = ".text.KernelParams"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL10]] : !VPUIPRegMapped.Index<0>
//CHECK-NEXT: ELF.Pad {{.*}}
//CHECK-NEXT: ELF.PutOpInSection %[[VAL22]] : !VPUIPRegMapped.Index<1>
//CHECK-NEXT: ELF.Pad {{.*}}
//CHECK-NEXT: ELF.PutOpInSection %[[VAL34]] : !VPUIPRegMapped.Index<2>

//CHECK-DAG: %[[ACTKERNELR:.*]] = ELF.CreateSection {{.*}} secName = ".text.ActKernelRanges"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL8]] : !VPUIPRegMapped.Index<0>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL20]] : !VPUIPRegMapped.Index<1>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL32]] : !VPUIPRegMapped.Index<2>

//CHECK-DAG: %[[ACTKERNELI:.*]] = ELF.CreateSection {{.*}} secName = ".text.ActKernelInvocations"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL9]] : !VPUIPRegMapped.Index<0>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL21]] : !VPUIPRegMapped.Index<1>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL33]] : !VPUIPRegMapped.Index<2>

//CHECK-DAG: %[[MAPPEDINF:.*]] = ELF.CreateSection {{.*}} secName = ".text.MappedInference"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL52]] : !VPUIPRegMapped.Index<0>

//CHECK-DAG: %[[INVARSEC:.*]] = ELF.CreateSection {{.*}} secName = ".text.DPUInvariants"

//CHECK-DAG: %[[VARSEC:.*]] = ELF.CreateSection {{.*}} secName = ".text.DPUVariants"

//CHECK-DAG: ELF.CreateMetadataSection {{.*}} secName = ".metadata"
//CHECK-NEXT: VPUIPRegMapped.NetworkMetadata

//CHECK: %[[BUILTIN_SYMTABSEC:.*]] = ELF.CreateSymbolTableSection secName("VPU_RT_SYMTAB")
//CHECK: %[[SYMTABSEC:.*]] = ELF.CreateSymbolTableSection secName(".symtab.tasks")

//CHECK: ELF.CreateRelocationSection secName(".rlt.text.dmaTasks0") sourceSymbolTableSection(%[[BUILTIN_SYMTABSEC]])
//CHECK-DAG: ELF.Reloc baseOp(%[[VAL4]] : !VPUIPRegMapped.Index<0>) {{.*}} "R_VPU_64" {{.*}}
//CHECK-DAG: ELF.RelocImmOffset baseOp(%[[VAL4]] : !VPUIPRegMapped.Index<0>) {{.*}} "R_VPU_32_RTM" {{.*}}
//CHECK-DAG: ELF.Reloc baseOp(%[[VAL11]] : !VPUIPRegMapped.Index<1>) {{.*}} "R_VPU_64" {{.*}}
//CHECK-DAG: ELF.RelocImmOffset baseOp(%[[VAL11]] : !VPUIPRegMapped.Index<1>) {{.*}} "R_VPU_32_RTM" {{.*}}
//CHECK-DAG: ELF.Reloc baseOp(%[[VAL16]] : !VPUIPRegMapped.Index<2>) {{.*}} "R_VPU_64" {{.*}}
//CHECK-DAG: ELF.RelocImmOffset baseOp(%[[VAL16]] : !VPUIPRegMapped.Index<2>) {{.*}} "R_VPU_32_RTM" {{.*}}
//CHECK-DAG: ELF.Reloc baseOp(%[[VAL23]] : !VPUIPRegMapped.Index<3>) {{.*}} "R_VPU_64" {{.*}}
//CHECK-DAG: ELF.RelocImmOffset baseOp(%[[VAL23]] : !VPUIPRegMapped.Index<3>) {{.*}} "R_VPU_32_RTM" {{.*}}
//CHECK-DAG: ELF.Reloc baseOp(%[[VAL28]] : !VPUIPRegMapped.Index<4>) {{.*}} "R_VPU_64" {{.*}}
//CHECK-DAG: ELF.RelocImmOffset baseOp(%[[VAL28]] : !VPUIPRegMapped.Index<4>) {{.*}} "R_VPU_32_RTM" {{.*}}
//CHECK-DAG: ELF.Reloc baseOp(%[[VAL35]] : !VPUIPRegMapped.Index<5>) {{.*}} "R_VPU_64" {{.*}}

//CHECK: ELF.CreateRelocationSection secName(".rlt.text.KernelParams") sourceSymbolTableSection(%[[SYMTABSEC]])
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL10]] : !VPUIPRegMapped.Index<0>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL10]] : !VPUIPRegMapped.Index<0>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL10]] : !VPUIPRegMapped.Index<0>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL10]] : !VPUIPRegMapped.Index<0>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL22]] : !VPUIPRegMapped.Index<1>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL22]] : !VPUIPRegMapped.Index<1>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL22]] : !VPUIPRegMapped.Index<1>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL22]] : !VPUIPRegMapped.Index<1>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL34]] : !VPUIPRegMapped.Index<2>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL34]] : !VPUIPRegMapped.Index<2>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL34]] : !VPUIPRegMapped.Index<2>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL34]] : !VPUIPRegMapped.Index<2>) {{.*}} "R_VPU_32"

//CHECK: ELF.CreateRelocationSection secName(".rlt.text.KernelParams") sourceSymbolTableSection(%[[BUILTIN_SYMTABSEC]])
//CHECK-DAG ELF.Reloc baseOp(%[[VAL10]] : !VPUIPRegMapped.Index<0>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.Reloc baseOp(%[[VAL10]] : !VPUIPRegMapped.Index<0>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.Reloc baseOp(%[[VAL22]] : !VPUIPRegMapped.Index<1>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.Reloc baseOp(%[[VAL22]] : !VPUIPRegMapped.Index<1>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.Reloc baseOp(%[[VAL34]] : !VPUIPRegMapped.Index<2>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.Reloc baseOp(%[[VAL34]] : !VPUIPRegMapped.Index<2>) {{.*}} "R_VPU_32"

//CHECK: ELF.CreateRelocationSection secName(".rlt.text.ActKernelRanges") sourceSymbolTableSection(%[[SYMTABSEC]])
//CHECK-DAG ELF.Reloc baseOp(%[[VAL8]] : !VPUIPRegMapped.Index<0>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.Reloc baseOp(%[[VAL20]] : !VPUIPRegMapped.Index<1>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.Reloc baseOp(%[[VAL32]] : !VPUIPRegMapped.Index<2>) {{.*}} "R_VPU_32"

//CHECK: ELF.CreateRelocationSection secName(".rlt.text.ActKernelInvocations") sourceSymbolTableSection(%[[BUILTIN_SYMTABSEC]])
//CHECK-DAG ELF.Reloc baseOp(%[[VAL9]] : !VPUIPRegMapped.Index<0>) {{.*}} "R_VPU_32_RTM"
//CHECK-DAG ELF.Reloc baseOp(%[[VAL21]] : !VPUIPRegMapped.Index<1>) {{.*}} "R_VPU_32_RTM"
//CHECK-DAG ELF.Reloc baseOp(%[[VAL33]] : !VPUIPRegMapped.Index<2>) {{.*}} "R_VPU_32_RTM"

//CHECK: ELF.CreateRelocationSection secName(".rlt.text.ActKernelInvocations") sourceSymbolTableSection(%[[SYMTABSEC]])
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL9]] : !VPUIPRegMapped.Index<0>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL9]] : !VPUIPRegMapped.Index<0>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL21]] : !VPUIPRegMapped.Index<1>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL21]] : !VPUIPRegMapped.Index<1>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL33]] : !VPUIPRegMapped.Index<2>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL33]] : !VPUIPRegMapped.Index<2>) {{.*}} "R_VPU_32"

//CHECK: ELF.CreateRelocationSection secName(".rlt.text.MappedInference") sourceSymbolTableSection(%[[SYMTABSEC]])
//CHECK-DAG ELF.Reloc baseOp(%[[VAL12]] : !VPUIPRegMapped.Index<0>) {{.*}} "R_VPU_64"
//CHECK-DAG ELF.Reloc baseOp(%[[VAL12]] : !VPUIPRegMapped.Index<0>) {{.*}} "R_VPU_64"
//CHECK-DAG ELF.Reloc baseOp(%[[VAL12]] : !VPUIPRegMapped.Index<0>) {{.*}} "R_VPU_64"
//CHECK-DAG ELF.Reloc baseOp(%[[VAL12]] : !VPUIPRegMapped.Index<0>) {{.*}} "R_VPU_64"
