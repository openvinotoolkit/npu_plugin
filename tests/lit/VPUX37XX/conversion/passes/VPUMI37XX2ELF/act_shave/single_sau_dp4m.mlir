//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=VPUX37XX allow-custom-values=true" --convert-VPUMI37XX-to-ELF %s | FileCheck %s

module @Test {
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xsi32>
    DataInfo "input1" : tensor<1x1x1x1000xsi32>
  } outputsInfo : {
    DataInfo "sau_dp4m" : tensor<1x1x1x1000xsi32>
  }
  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096]
  module @VPU.SW {
    func.func private @builtin_sau_dp4_m(memref<*xsi32>, memref<*xsi32>, memref<*xsi32>) attributes {VPU.kernel_code = "sau_dp4m.cpp", VPU.kernel_entry = "sau_dp4m"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }
  func.func @main(%arg0: memref<1x1x1x1000xsi32>, %arg1: memref<1x1x1x1000xsi32>, %arg2: memref<1x1x1x1000xsi32>) -> memref<1x1x1x1000xsi32> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xsi32, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <4000> -> memref<1x1x1x1000xsi32, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <8000> -> memref<1x1x1x1000xsi32, [@CMX_NN, 0]>
    %3 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 2 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:0>
    %4 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPURegMapped.Index<0:0:1>
    %5 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x1x1000xsi32>) outputs(%0 : memref<1x1x1x1000xsi32, [@CMX_NN, 0]>) updates(%3 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
    %6 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg1 : memref<1x1x1x1000xsi32>) outputs(%1 : memref<1x1x1x1000xsi32, [@CMX_NN, 0]>) previousDMA(%5 : !VPURegMapped.Index<0:0:0>) updates(%3 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>
    %7 = VPUMI37XX.DeclareKernelText kernel_path("sau_dp4m") -> !VPURegMapped.Index<0:0:0>
    %8 = VPUMI37XX.DeclareKernelArgs kernel_path("sau_dp4m") -> !VPURegMapped.Index<0:0:0>
    %9 = VPUMI37XX.DeclareKernelEntry kernel_path("sau_dp4m") -> !VPURegMapped.Index<0:0:0>
    %10 = VPUMI37XX.ActKernelRange kernel_text_index(%7 : <0:0:0>) kernel_args_index(%8 : <0:0:0>) kernel_entry_index(%9 : <0:0:0>) -> !VPURegMapped.Index<0:0:0>
    %11 = VPUMI37XX.ActKernelInvocation range_index(%10 : <0:0:0>) waits(%3 : !VPURegMapped.Index<0:0:0>) updates(%4 : !VPURegMapped.Index<0:0:1>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
    %12 = VPUMI37XX.KernelParams inputs(%0, %1 : memref<1x1x1x1000xsi32, [@CMX_NN, 0]>, memref<1x1x1x1000xsi32, [@CMX_NN, 0]>) outputs(%2 : memref<1x1x1x1000xsi32, [@CMX_NN, 0]>) kernel_type("sau_dp4m") kernel_params(dense<"0x000000000000000004000000000000000000000000000000214300000000000002000000000000000000000004000000000000000000000000000000214300000000000002000000000000000000000004000000000000000000000000000000214300000000000002000000"> : vector<108xui8>) -> !VPURegMapped.Index<0:0:0>
    %13 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%2 : memref<1x1x1x1000xsi32, [@CMX_NN, 0]>) outputs(%arg2 : memref<1x1x1x1000xsi32>) previousDMA(%6 : !VPURegMapped.Index<0:0:1>) waits(%4 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:2>
    %14 = VPUMI37XX.MappedInference dmas(%5 : !VPURegMapped.Index<0:0:0>) actKernelRanges(%10 : !VPURegMapped.Index<0:0:0>) actKernelInvocations(%11 : !VPURegMapped.Index<0:0:0>) barriers(%3 : !VPURegMapped.Index<0:0:0>) dmaCount([3, 0]) invariantCount(0) variantCount(0) actKernelRangesCount(1) actKernelInvocationsCount(1) barrierCount(2) -> !VPURegMapped.Index<0:0:0>
    return %arg2 : memref<1x1x1x1000xsi32>
  }
}

//CHECK: %[[VAL0:.*]] = VPURT.DeclareBuffer
//CHECK: %[[VAL1:.*]] = VPURT.DeclareBuffer
//CHECK: %[[VAL2:.*]] = VPURT.DeclareBuffer
//CHECK: %[[VAL3:.*]] = VPUMI37XX.ConfigureBarrier
//CHECK: %[[VAL4:.*]] = VPUMI37XX.ConfigureBarrier
//CHECK: %[[VAL5:.*]] = VPUMI37XX.NNDMA
//CHECK: %[[VAL6:.*]] = VPUMI37XX.NNDMA
//CHECK: %[[VAL7:.*]] = VPUMI37XX.DeclareKernelText
//CHECK: %[[VAL8:.*]] = VPUMI37XX.DeclareKernelArgs
//CHECK: %[[VAL9:.*]] = VPUMI37XX.DeclareKernelEntry
//CHECK: %[[VAL10:.*]] = VPUMI37XX.ActKernelRange
//CHECK: %[[VAL11:.*]] = VPUMI37XX.ActKernelInvocation
//CHECK: %[[VAL12:.*]] = VPUMI37XX.KernelParams
//CHECK: %[[VAL13:.*]] = VPUMI37XX.NNDMA

//CHECK-DAG: %[[VAL14:.*]] = VPURT.DeclareBuffer <DDR>
//CHECK-NEXT: %[[VAL15:.*]] = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags(VPU_SHF_PROC_SHAVE) {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".bss.actShaveStack_0"} -> !ELF.Section {
//CHECK-NEXT: ELF.PutOpInSection %[[VAL14]] : memref
//CHECK: %[[VAL16:.*]]  = ELF.Symbol %[[VAL15]] name("sym_actShaveStack_0") : !ELF.Section

//CHECK-DAG: %[[VAL17:.*]] = VPURT.DeclareBuffer <DDR>
//CHECK-NEXT: %[[VAL18:.*]] = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags(VPU_SHF_PROC_SHAVE) {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".bss.actShaveStack_1"} -> !ELF.Section {
//CHECK-NEXT: ELF.PutOpInSection %[[VAL17]] : memref
//CHECK: %[[VAL19:.*]]  = ELF.Symbol %[[VAL18]] name("sym_actShaveStack_1") : !ELF.Section

//CHECK-DAG: %[[VAL20:.*]] = VPURT.DeclareBuffer <DDR>
//CHECK-NEXT: %[[VAL21:.*]] = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags(VPU_SHF_PROC_SHAVE) {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".bss.actShaveStack_2"} -> !ELF.Section {
//CHECK-NEXT: ELF.PutOpInSection %[[VAL20]] : memref
//CHECK: %[[VAL22:.*]]  = ELF.Symbol %[[VAL21]] name("sym_actShaveStack_2") : !ELF.Section

//CHECK-DAG: %[[VAL23:.*]] = VPURT.DeclareBuffer <DDR>
//CHECK-NEXT: %[[VAL24:.*]] = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags(VPU_SHF_PROC_SHAVE) {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".bss.actShaveStack_3"} -> !ELF.Section {
//CHECK-NEXT: ELF.PutOpInSection %[[VAL23]] : memref
//CHECK: %[[VAL25:.*]]  = ELF.Symbol %[[VAL24]] name("sym_actShaveStack_3") : !ELF.Section

//CHECK-DAG: %[[VAL26:.*]] = VPUMI37XX.ActShaveRt kernel("nnActEntry") -> !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: %[[VAL27:.*]] = ELF.CreateSection {{.*}} {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".text.actKernelRtConfigSec"} -> !ELF.Section {
//CHECK-NEXT: ELF.PutOpInSection %[[VAL26]] : !VPURegMapped.Index<0:0:0>
//CHECK: %[[VAL28:.*]]  = ELF.Symbol %[[VAL27]] name("sym_actKernelRtConfigsSec") : !ELF.Section

//CHECK-DAG: %[[VAL29:.*]] = ELF.CreateSymbolTableSection secName(".symtab.actKernelRtConfig")
//CHECK-NEXT: ELF.PutOpInSection %[[VAL28]] : !ELF.Symbol
//CHECK-NEXT: ELF.PutOpInSection %[[VAL16]] : !ELF.Symbol
//CHECK-NEXT: ELF.PutOpInSection %[[VAL19]] : !ELF.Symbol
//CHECK-NEXT: ELF.PutOpInSection %[[VAL22]] : !ELF.Symbol
//CHECK-NEXT: ELF.PutOpInSection %[[VAL25]] : !ELF.Symbol

//CHECK: %[[VAL30:.*]] = VPUMI37XX.MappedInference
//CHECK-SAME: dmas(%[[VAL5]] : !VPURegMapped.Index<0:0:0>)
//CHECK-SAME: actKernelRanges(%[[VAL10]] : !VPURegMapped.Index<0:0:0>)
//CHECK-SAME: actKernelInvocations(%[[VAL11]] : !VPURegMapped.Index<0:0:0>)
//CHECK-SAME: barriers(%[[VAL3]] : !VPURegMapped.Index<0:0:0>)
//CHECK-SAME: actShaveRt(%[[VAL26]] : !VPURegMapped.Index<0:0:0>)
//CHECK-SAME: actShaveStacks(%[[VAL14]], %[[VAL17]], %[[VAL20]], %[[VAL23]] : {{.*}}>)
//CHECK-SAME: dmaCount([3, 0]) invariantCount(0) variantCount(0) actKernelRangesCount(1) actKernelInvocationsCount(1) barrierCount(2)
//CHECK-SAME: -> !VPURegMapped.Index<0:0:0>


//CHECK-DAG: %[[DMASEC:.*]] = ELF.CreateSection {{.*}} secName = ".text.dmaTasks0"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL5]] : !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL6]] : !VPURegMapped.Index<0:0:1>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL13]] : !VPURegMapped.Index<0:0:2>

//CHECK-DAG: %[[BARSEC:.*]] = ELF.CreateSection {{.*}} secName = ".text.BarrierConfigs"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL3]] : !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL4]] : !VPURegMapped.Index<0:0:1>

//CHECK-DAG: %[[KERNELTEXT:.*]] = ELF.CreateSection {{.*}} secName = ".text.KernelText"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL7]] : !VPURegMapped.Index<0:0:0>

//CHECK-DAG: %[[KERNELDATA:.*]] = ELF.CreateSection {{.*}} secName = ".text.KernelData"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL8]] : !VPURegMapped.Index<0:0:0>

//CHECK-DAG: %[[KERNELPARAMS:.*]] = ELF.CreateSection {{.*}} secName = ".text.KernelParams"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL12]] : !VPURegMapped.Index<0:0:0>

//CHECK-DAG: %[[ACTKERNELR:.*]] = ELF.CreateSection {{.*}} secName = ".text.ActKernelRanges"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL10]] : !VPURegMapped.Index<0:0:0>

//CHECK-DAG: %[[ACTKERNELI:.*]] = ELF.CreateSection {{.*}} secName = ".text.ActKernelInvocations"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL11]] : !VPURegMapped.Index<0:0:0>

//CHECK-DAG: %[[MAPPEDINF:.*]] = ELF.CreateSection {{.*}} secName = ".text.MappedInference"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL30]] : !VPURegMapped.Index<0:0:0>

//CHECK-DAG: %[[INVARSEC:.*]] = ELF.CreateSection {{.*}} secName = ".text.DPUInvariants"

//CHECK-DAG: %[[VARSEC:.*]] = ELF.CreateSection {{.*}} secName = ".text.DPUVariants"

//CHECK-DAG: ELF.CreateMetadataSection {{.*}} secName = ".metadata"
//CHECK-NEXT: VPUMI37XX.NetworkMetadata

//CHECK: %[[BUILTIN_SYMTABSEC:.*]] = ELF.CreateSymbolTableSection secName("VPU_RT_SYMTAB")
//CHECK: %[[SYMTABSEC:.*]] = ELF.CreateSymbolTableSection secName(".symtab.tasks")

//CHECK: ELF.CreateRelocationSection secName(".rlt.text.dmaTasks0") sourceSymbolTableSection(%[[BUILTIN_SYMTABSEC]])
//CHECK-DAG: ELF.Reloc baseOp(%[[VAL5]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_64> {{.*}}
//CHECK-DAG: ELF.RelocImmOffset baseOp(%[[VAL5]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32_RTM> {{.*}}
//CHECK-DAG: ELF.Reloc baseOp(%[[VAL6]] : !VPURegMapped.Index<0:0:1>) {{.*}} <R_VPU_64> {{.*}}
//CHECK-DAG: ELF.RelocImmOffset baseOp(%[[VAL6]] : !VPURegMapped.Index<0:0:1>) {{.*}} <R_VPU_32_RTM> {{.*}}
//CHECK-DAG: ELF.Reloc baseOp(%[[VAL13]] : !VPURegMapped.Index<0:0:2>) {{.*}} <R_VPU_64> {{.*}}

//CHECK: ELF.CreateRelocationSection secName(".rlt.text.KernelParams") sourceSymbolTableSection(%[[SYMTABSEC]])
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL12]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32>
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL12]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32>
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL12]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32>
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL12]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32>
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL12]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32>
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL12]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32>

//CHECK: ELF.CreateRelocationSection secName(".rlt.text.KernelParams") sourceSymbolTableSection(%[[BUILTIN_SYMTABSEC]])
//CHECK-DAG ELF.Reloc baseOp(%[[VAL12]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32>
//CHECK-DAG ELF.Reloc baseOp(%[[VAL12]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32>
//CHECK-DAG ELF.Reloc baseOp(%[[VAL12]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32>

//CHECK: ELF.CreateRelocationSection secName(".rlt.text.ActKernelRanges") sourceSymbolTableSection(%[[SYMTABSEC]])
//CHECK-DAG ELF.Reloc baseOp(%[[VAL10]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32>

//CHECK: ELF.CreateRelocationSection secName(".rlt.text.ActKernelInvocations") sourceSymbolTableSection(%[[BUILTIN_SYMTABSEC]])
//CHECK-DAG ELF.Reloc baseOp(%[[VAL11]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32_RTM>

//CHECK: ELF.CreateRelocationSection secName(".rlt.text.ActKernelInvocations") sourceSymbolTableSection(%[[SYMTABSEC]])
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL11]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32>
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL11]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32>

//CHECK: ELF.CreateRelocationSection secName(".rlt.text.MappedInference") sourceSymbolTableSection(%[[SYMTABSEC]])
//CHECK-DAG ELF.Reloc baseOp(%[[VAL30]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_64>
//CHECK-DAG ELF.Reloc baseOp(%[[VAL30]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_64>
//CHECK-DAG ELF.Reloc baseOp(%[[VAL30]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_64>
//CHECK-DAG ELF.Reloc baseOp(%[[VAL30]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_64>
