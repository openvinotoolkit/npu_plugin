//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=VPUX37XX allow-custom-values=true" --convert-VPUMI37XX-to-ELF %s | FileCheck %s
//

module @Test attributes {VPU.arch = "VPUX37XX", VPU.compilationMode = "ReferenceHW"} {
  IE.MemoryResource 31457280 bytes of @DDR {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64}
  IE.MemoryResource 2097152 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
  IE.ExecutorResource 1 of @DMA_NN
  IE.ExecutorResource 1 of @SHAVE_UPA
  IE.ExecutorResource 1 of @SHAVE_ACT
  IE.ExecutorResource 2 of @NCE {
    IE.ExecutorResource 1 of @DPU
  }
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1000xf16>
  } outputsInfo : {
    DataInfo "sigmoid" : tensor<1x1000xf16>
  }
  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_sigmoid(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "sigmoid_fp16.c", VPU.kernel_entry = "sigmoid_fp16"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }
  func.func @main(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    %0 = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 1]>
    %1 = VPURT.DeclareBuffer "CMX_NN" [1] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 1]>
    %2 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:0>
    %3 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPURegMapped.Index<0:0:1>
    %4 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x1x1000xf16>) outputs(%0 : memref<1x1x1x1000xf16, [@CMX_NN, 1]>) updates(%2 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
    %5 = VPUMI37XX.DeclareKernelText kernel_path("sigmoid_fp16") -> !VPURegMapped.Index<0:0:0>
    %6 = VPUMI37XX.DeclareKernelArgs kernel_path("sigmoid_fp16") -> !VPURegMapped.Index<0:0:0>
    %7 = VPUMI37XX.DeclareKernelEntry kernel_path("sigmoid_fp16") -> !VPURegMapped.Index<0:0:0>
    %8 = VPUMI37XX.ActKernelRange kernel_text_index(%5 : <0:0:0>) kernel_args_index(%6 : <0:0:0>) kernel_entry_index(%7 : <0:0:0>) -> !VPURegMapped.Index<0:0:0>
    %9 = VPUMI37XX.ActKernelInvocation range_index(%8 : <0:0:0>) waits(%2 : !VPURegMapped.Index<0:0:0>) updates(%3 : !VPURegMapped.Index<0:0:1>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
    %10 = VPUMI37XX.KernelParams inputs(%0 : memref<1x1x1x1000xf16, [@CMX_NN, 1]>) outputs(%1 : memref<1x1x1x1000xf16, [@CMX_NN, 1]>) kernel_type("sigmoid_fp16") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]> : vector<72xui8>) -> !VPURegMapped.Index<0:0:0>
    %11 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%1 : memref<1x1x1x1000xf16, [@CMX_NN, 1]>) outputs(%arg1 : memref<1x1x1x1000xf16>) previousDMA(%4 : !VPURegMapped.Index<0:0:0>) waits(%3 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>
    %12 = VPUMI37XX.MappedInference dmas(%4 : !VPURegMapped.Index<0:0:0>) actKernelRanges(%8 : !VPURegMapped.Index<0:0:0>) actKernelInvocations(%9 : !VPURegMapped.Index<0:0:0>) barriers(%2 : !VPURegMapped.Index<0:0:0>) dmaCount([2]) invariantCount(0) variantCount(0) actKernelRangesCount(1) actKernelInvocationsCount(1) barrierCount(2) -> !VPURegMapped.Index<0:0:0>
    return %arg1 : memref<1x1x1x1000xf16>
  }
}

//CHECK: %[[VAL0:.*]] = VPURT.DeclareBuffer
//CHECK: %[[VAL1:.*]] = VPURT.DeclareBuffer
//CHECK: %[[VAL2:.*]] = VPUMI37XX.ConfigureBarrier
//CHECK: %[[VAL3:.*]] = VPUMI37XX.ConfigureBarrier
//CHECK: %[[VAL4:.*]] = VPUMI37XX.NNDMA
//CHECK: %[[VAL5:.*]] = VPUMI37XX.DeclareKernelText
//CHECK: %[[VAL6:.*]] = VPUMI37XX.DeclareKernelArgs
//CHECK: %[[VAL7:.*]] = VPUMI37XX.DeclareKernelEntry
//CHECK: %[[VAL8:.*]] = VPUMI37XX.ActKernelRange
//CHECK: %[[VAL9:.*]] = VPUMI37XX.ActKernelInvocation
//CHECK: %[[VAL10:.*]] = VPUMI37XX.KernelParams
//CHECK: %[[VAL11:.*]] = VPUMI37XX.NNDMA

//CHECK-DAG: %[[VAL12:.*]] = VPURT.DeclareBuffer "DDR"
//CHECK-NEXT: %[[VAL13:.*]] = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags(VPU_SHF_PROC_SHAVE) {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".bss.actShaveStack_0"} -> !ELF.Section {
//CHECK-NEXT: ELF.PutOpInSection %[[VAL12]] : memref
//CHECK: %[[VAL14:.*]]  = ELF.Symbol %[[VAL13]] name("sym_actShaveStack_0") : !ELF.Section

//CHECK-DAG: %[[VAL15:.*]] = VPURT.DeclareBuffer "DDR"
//CHECK-NEXT: %[[VAL16:.*]] = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags(VPU_SHF_PROC_SHAVE) {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".bss.actShaveStack_1"} -> !ELF.Section {
//CHECK-NEXT: ELF.PutOpInSection %[[VAL15]] : memref
//CHECK: %[[VAL17:.*]]  = ELF.Symbol %[[VAL16]] name("sym_actShaveStack_1") : !ELF.Section

//CHECK-DAG: %[[VAL18:.*]] = VPURT.DeclareBuffer "DDR"
//CHECK-NEXT: %[[VAL19:.*]] = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags(VPU_SHF_PROC_SHAVE) {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".bss.actShaveStack_2"} -> !ELF.Section {
//CHECK-NEXT: ELF.PutOpInSection %[[VAL18]] : memref
//CHECK: %[[VAL20:.*]]  = ELF.Symbol %[[VAL19]] name("sym_actShaveStack_2") : !ELF.Section

//CHECK-DAG: %[[VAL21:.*]] = VPURT.DeclareBuffer "DDR"
//CHECK-NEXT: %[[VAL22:.*]] = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags(VPU_SHF_PROC_SHAVE) {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".bss.actShaveStack_3"} -> !ELF.Section {
//CHECK-NEXT: ELF.PutOpInSection %[[VAL21]] : memref
//CHECK: %[[VAL23:.*]]  = ELF.Symbol %[[VAL22]] name("sym_actShaveStack_3") : !ELF.Section

//CHECK-DAG: %[[VAL24:.*]] = VPUMI37XX.ActShaveRt kernel("nnActEntry") -> !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: %[[VAL25:.*]] = ELF.CreateSection {{.*}} {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".text.actKernelRtConfigSec"} -> !ELF.Section {
//CHECK-NEXT: ELF.PutOpInSection %[[VAL24]] : !VPURegMapped.Index<0:0:0>
//CHECK: %[[VAL26:.*]]  = ELF.Symbol %[[VAL25]] name("sym_actKernelRtConfigsSec") : !ELF.Section

//CHECK-DAG: %[[VAL27:.*]] = ELF.CreateSymbolTableSection secName(".symtab.actKernelRtConfig")
//CHECK-NEXT: ELF.PutOpInSection %[[VAL26]] : !ELF.Symbol
//CHECK-NEXT: ELF.PutOpInSection %[[VAL14]] : !ELF.Symbol
//CHECK-NEXT: ELF.PutOpInSection %[[VAL17]] : !ELF.Symbol
//CHECK-NEXT: ELF.PutOpInSection %[[VAL20]] : !ELF.Symbol
//CHECK-NEXT: ELF.PutOpInSection %[[VAL23]] : !ELF.Symbol

//CHECK: %[[VAL28:.*]] = VPUMI37XX.MappedInference
//CHECK-SAME: dmas(%[[VAL4]] : !VPURegMapped.Index<0:0:0>)
//CHECK-SAME: actKernelRanges(%[[VAL8]] : !VPURegMapped.Index<0:0:0>)
//CHECK-SAME: actKernelInvocations(%[[VAL9]] : !VPURegMapped.Index<0:0:0>)
//CHECK-SAME: barriers(%[[VAL2]] : !VPURegMapped.Index<0:0:0>)
//CHECK-SAME: actShaveRt(%[[VAL24]] : !VPURegMapped.Index<0:0:0>)
//CHECK-SAME: actShaveStacks(%[[VAL12]], %[[VAL15]], %[[VAL18]], %[[VAL21]] : {{.*}}>)
//CHECK-SAME: dmaCount([2]) invariantCount(0) variantCount(0) actKernelRangesCount(1) actKernelInvocationsCount(1) barrierCount(2)
//CHECK-SAME: -> !VPURegMapped.Index<0:0:0>

//CHECK-DAG: %[[DMASEC:.*]] = ELF.CreateSection {{.*}} secName = ".text.dmaTasks0"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL4]] : !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL11]] : !VPURegMapped.Index<0:0:1>

//CHECK-DAG: %[[BARSEC:.*]] = ELF.CreateSection {{.*}} secName = ".text.BarrierConfigs"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL2]] : !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL3]] : !VPURegMapped.Index<0:0:1>

//CHECK-DAG: %[[KERNELTEXT:.*]] = ELF.CreateSection {{.*}} secName = ".text.KernelText"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL5]] : !VPURegMapped.Index<0:0:0>

//CHECK-DAG: %[[KERNELDATA:.*]] = ELF.CreateSection {{.*}} secName = ".text.KernelData"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL6]] : !VPURegMapped.Index<0:0:0>

//CHECK-DAG: %[[KERNELPARAMS:.*]] = ELF.CreateSection {{.*}} secName = ".text.KernelParams"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL10]] : !VPURegMapped.Index<0:0:0>

//CHECK-DAG: %[[ACTKERNELR:.*]] = ELF.CreateSection {{.*}} secName = ".text.ActKernelRanges"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL8]] : !VPURegMapped.Index<0:0:0>

//CHECK-DAG: %[[ACTKERNELI:.*]] = ELF.CreateSection {{.*}} secName = ".text.ActKernelInvocations"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL9]] : !VPURegMapped.Index<0:0:0>

//CHECK-DAG: %[[MAPPEDINF:.*]] = ELF.CreateSection {{.*}} secName = ".text.MappedInference"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL28]] : !VPURegMapped.Index<0:0:0>

//CHECK-DAG: %[[INVARSEC:.*]] = ELF.CreateSection {{.*}} secName = ".text.DPUInvariants"

//CHECK-DAG: %[[VARSEC:.*]] = ELF.CreateSection {{.*}} secName = ".text.DPUVariants"

//CHECK-DAG: ELF.CreateMetadataSection {{.*}} secName = ".metadata"
//CHECK-NEXT: VPUMI37XX.NetworkMetadata

//CHECK: %[[BUILTIN_SYMTABSEC:.*]] = ELF.CreateSymbolTableSection secName("VPU_RT_SYMTAB")
//CHECK: %[[SYMTABSEC:.*]] = ELF.CreateSymbolTableSection secName(".symtab.tasks")

//CHECK: ELF.CreateRelocationSection secName(".rlt.text.dmaTasks0") sourceSymbolTableSection(%[[BUILTIN_SYMTABSEC]])
//CHECK-DAG: ELF.Reloc baseOp(%[[VAL4]] : !VPURegMapped.Index<0:0:0>) {{.*}} "R_VPU_64" {{.*}}
//CHECK-DAG: ELF.RelocImmOffset baseOp(%[[VAL4]] : !VPURegMapped.Index<0:0:0>) {{.*}} "R_VPU_32_RTM" {{.*}}
//CHECK-DAG: ELF.Reloc baseOp(%[[VAL11]] : !VPURegMapped.Index<0:0:1>) {{.*}} "R_VPU_64" {{.*}}

//CHECK: ELF.CreateRelocationSection secName(".rlt.text.KernelParams") sourceSymbolTableSection(%[[SYMTABSEC]])
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL10]] : !VPURegMapped.Index<0:0:0>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL10]] : !VPURegMapped.Index<0:0:0>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL10]] : !VPURegMapped.Index<0:0:0>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL10]] : !VPURegMapped.Index<0:0:0>) {{.*}} "R_VPU_32"

//CHECK: ELF.CreateRelocationSection secName(".rlt.text.KernelParams") sourceSymbolTableSection(%[[BUILTIN_SYMTABSEC]])
//CHECK-DAG ELF.Reloc baseOp(%[[VAL10]] : !VPURegMapped.Index<0:0:0>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.Reloc baseOp(%[[VAL10]] : !VPURegMapped.Index<0:0:0>) {{.*}} "R_VPU_32"

//CHECK: ELF.CreateRelocationSection secName(".rlt.text.ActKernelRanges") sourceSymbolTableSection(%[[SYMTABSEC]])
//CHECK-DAG ELF.Reloc baseOp(%[[VAL8]] : !VPURegMapped.Index<0:0:0>) {{.*}} "R_VPU_32"

//CHECK: ELF.CreateRelocationSection secName(".rlt.text.ActKernelInvocations") sourceSymbolTableSection(%[[BUILTIN_SYMTABSEC]])
//CHECK-DAG ELF.Reloc baseOp(%[[VAL9]] : !VPURegMapped.Index<0:0:0>) {{.*}} "R_VPU_32_RTM"

//CHECK: ELF.CreateRelocationSection secName(".rlt.text.ActKernelInvocations") sourceSymbolTableSection(%[[SYMTABSEC]])
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL9]] : !VPURegMapped.Index<0:0:0>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL9]] : !VPURegMapped.Index<0:0:0>) {{.*}} "R_VPU_32"

//CHECK: ELF.CreateRelocationSection secName(".rlt.text.MappedInference") sourceSymbolTableSection(%[[SYMTABSEC]])
//CHECK-DAG ELF.Reloc baseOp(%[[VAL28]] : !VPURegMapped.Index<0:0:0>) {{.*}} "R_VPU_64"
//CHECK-DAG ELF.Reloc baseOp(%[[VAL28]] : !VPURegMapped.Index<0:0:0>) {{.*}} "R_VPU_64"
//CHECK-DAG ELF.Reloc baseOp(%[[VAL28]] : !VPURegMapped.Index<0:0:0>) {{.*}} "R_VPU_64"
//CHECK-DAG ELF.Reloc baseOp(%[[VAL28]] : !VPURegMapped.Index<0:0:0>) {{.*}} "R_VPU_64"
