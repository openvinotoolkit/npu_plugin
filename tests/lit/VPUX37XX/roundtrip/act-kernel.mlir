//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-translate --export-ELF %s -o t.elf 
// RUN: vpux-translate --import-ELF t.elf | FileCheck %s
// RUN: rm t.elf
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
    %12 = VPURT.DeclareBuffer "DDR" <0> -> memref<8192xi8, @DDR>
    %13 = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags(VPU_SHF_PROC_SHAVE) {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".bss.actShaveStack_0"} -> !ELF.Section {
      ELF.PutOpInSection %12 : memref<8192xi8, @DDR>
    }
    %14 = ELF.Symbol %13 name("sym_actShaveStack_0") : !ELF.Section
    %15 = VPURT.DeclareBuffer "DDR" <0> -> memref<8192xi8, @DDR>
    %16 = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags(VPU_SHF_PROC_SHAVE) {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".bss.actShaveStack_1"} -> !ELF.Section {
      ELF.PutOpInSection %15 : memref<8192xi8, @DDR>
    }
    %17 = ELF.Symbol %16 name("sym_actShaveStack_1") : !ELF.Section
    %18 = VPURT.DeclareBuffer "DDR" <0> -> memref<8192xi8, @DDR>
    %19 = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags(VPU_SHF_PROC_SHAVE) {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".bss.actShaveStack_2"} -> !ELF.Section {
      ELF.PutOpInSection %18 : memref<8192xi8, @DDR>
    }
    %20 = ELF.Symbol %19 name("sym_actShaveStack_2") : !ELF.Section
    %21 = VPURT.DeclareBuffer "DDR" <0> -> memref<8192xi8, @DDR>
    %22 = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags(VPU_SHF_PROC_SHAVE) {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".bss.actShaveStack_3"} -> !ELF.Section {
      ELF.PutOpInSection %21 : memref<8192xi8, @DDR>
    }
    %23 = ELF.Symbol %22 name("sym_actShaveStack_3") : !ELF.Section
    %24 = VPUIPRegMapped.ActShaveRt kernel("nnActEntry") -> !VPUIPRegMapped.Index<0>
    %25 = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".text.actKernelRtConfigSec"} -> !ELF.Section {
      ELF.PutOpInSection %24 : !VPUIPRegMapped.Index<0>
    }
    %26 = ELF.Symbol %25 name("sym_actKernelRtConfigsSec") : !ELF.Section
    %27 = ELF.CreateSymbolTableSection secName(".symtab.actKernelRtConfig") secFlags("SHF_NONE") -> !ELF.Section {
      ELF.PutOpInSection %26 : !ELF.Symbol
      ELF.PutOpInSection %14 : !ELF.Symbol
      ELF.PutOpInSection %17 : !ELF.Symbol
      ELF.PutOpInSection %20 : !ELF.Symbol
      ELF.PutOpInSection %23 : !ELF.Symbol
    }
    %28 = VPUIPRegMapped.MappedInference dmas(%4 : !VPUIPRegMapped.Index<0>) actKernelRanges(%8 : !VPUIPRegMapped.Index<0>) actKernelInvocations(%9 : !VPUIPRegMapped.Index<0>) barriers(%2 : !VPUIPRegMapped.Index<0>) actShaveRt(%24 : !VPUIPRegMapped.Index<0>) actShaveStacks(%12, %15, %18, %21 : memref<8192xi8, @DDR>, memref<8192xi8, @DDR>, memref<8192xi8, @DDR>, memref<8192xi8, @DDR>) dmaCount([2]) invariantCount(0) variantCount(0) actKernelRangesCount(1) actKernelInvocationsCount(1) barrierCount(2) -> !VPUIPRegMapped.Index<0>
    %29 = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks"} -> !ELF.Section {
      ELF.PutOpInSection %4 : !VPUIPRegMapped.Index<0>
      ELF.PutOpInSection %11 : !VPUIPRegMapped.Index<1>
    }
    %30 = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.BarrierConfigs"} -> !ELF.Section {
      ELF.PutOpInSection %2 : !VPUIPRegMapped.Index<0>
      ELF.PutOpInSection %3 : !VPUIPRegMapped.Index<1>
    }
    %31 = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".text.KernelText"} -> !ELF.Section {
      ELF.PutOpInSection %5 : !VPUIPRegMapped.Index<0>
    }
    %32 = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".text.KernelData"} -> !ELF.Section {
      ELF.PutOpInSection %6 : !VPUIPRegMapped.Index<0>
    }
    %33 = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".text.KernelParams"} -> !ELF.Section {
      ELF.PutOpInSection %10 : !VPUIPRegMapped.Index<0>
    }
    %34 = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.ActKernelRanges"} -> !ELF.Section {
      ELF.PutOpInSection %8 : !VPUIPRegMapped.Index<0>
    }
    %35 = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.ActKernelInvocations"} -> !ELF.Section {
      ELF.PutOpInSection %9 : !VPUIPRegMapped.Index<0>
    }
    %36 = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.MappedInference"} -> !ELF.Section {
      ELF.PutOpInSection %28 : !VPUIPRegMapped.Index<0>
    }
    %37 = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.DPUInvariants"} -> !ELF.Section {
    }
    %38 = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.DPUVariants"} -> !ELF.Section {
    }
    %39 = ELF.CreateMetadataSection secFlags("SHF_NONE") {secAddrAlign = 8 : i64, secInfo = 0 : i64, secName = ".metadata"} -> !ELF.Section {
      %78 = VPUIPRegMapped.NetworkMetadata -> !VPUIPRegMapped.Index<0>
    }
    %40 = ELF.Symbol %29 name("sym_dmaSection0") : !ELF.Section
    %41 = ELF.Symbol %30 name("sym_barrierSection") : !ELF.Section
    %42 = ELF.Symbol %34 name("sym_actKernelRangeSection") : !ELF.Section
    %43 = ELF.Symbol %35 name("sym_actKernelInvo") : !ELF.Section
    %44 = ELF.Symbol %31 name("sym_kernelTextSection") : !ELF.Section
    %45 = ELF.Symbol %32 name("sym_kernelDataSection") : !ELF.Section
    %46 = ELF.Symbol %33 name("sym_kernelParamsSection") : !ELF.Section
    %47 = ELF.Symbol %37 name("sym_inVariantsSection") : !ELF.Section
    %48 = ELF.Symbol %38 name("sym_variantsSection") : !ELF.Section
    %49 = ELF.Symbol %arg0 name("input") size(2000) : memref<1x1x1x1000xf16>
    %50 = ELF.Symbol %arg1 name("hswish") size(2000) : memref<1x1x1x1000xf16>
    %51 = ELF.CreateSymbolTableSection secName(".symtab.input") secFlags(VPU_SHF_USERINPUT) -> !ELF.Section {
      ELF.PutOpInSection %49 : !ELF.Symbol
    }
    %52 = ELF.CreateSymbolTableSection secName(".symtab.output") secFlags(VPU_SHF_USEROUTPUT) -> !ELF.Section {
      ELF.PutOpInSection %50 : !ELF.Symbol
    }
    %53 = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags(SHF_ALLOC) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".data.BuffersIO"} -> !ELF.Section {
    }
    %54 = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".data.ConstIO"} -> !ELF.Section {
    }
    %55 = ELF.Symbol %54 name("sym_constSection") : !ELF.Section
    %56 = ELF.Symbol %53 name("sym_bufferSection") : !ELF.Section
    %57 = ELF.CreateSymbolTableSection secName(".symtab.buffers") secFlags("SHF_NONE") -> !ELF.Section {
      ELF.PutOpInSection %56 : !ELF.Symbol
      ELF.PutOpInSection %55 : !ELF.Symbol
    }
    %c0_i8 = arith.constant 0 : i8
    %58 = ELF.Symbol %c0_i8 name("VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR") {isBuiltin} : i8
    %c1_i8 = arith.constant 1 : i8
    %59 = ELF.Symbol %c1_i8 name("VPU_NNRD_SYM_RTM_IVAR") {isBuiltin} : i8
    %c2_i8 = arith.constant 2 : i8
    %60 = ELF.Symbol %c2_i8 name("VPU_NNRD_SYM_RTM_ACT") {isBuiltin} : i8
    %c3_i8 = arith.constant 3 : i8
    %61 = ELF.Symbol %c3_i8 name("VPU_NNRD_SYM_RTM_DMA0") {isBuiltin} : i8
    %c4_i8 = arith.constant 4 : i8
    %62 = ELF.Symbol %c4_i8 name("VPU_NNRD_SYM_RTM_DMA1") {isBuiltin} : i8
    %c5_i8 = arith.constant 5 : i8
    %63 = ELF.Symbol %c5_i8 name("VPU_NNRD_SYM_FIFO_BASE") {isBuiltin} : i8
    %c6_i8 = arith.constant 6 : i8
    %64 = ELF.Symbol %c6_i8 name("VPU_NNRD_SYM_BARRIERS_START") {isBuiltin} : i8
    %65 = ELF.CreateSymbolTableSection secName("VPU_RT_SYMTAB") secFlags("SHF_NONE") {isBuiltin} -> !ELF.Section {
      ELF.PutOpInSection %58 : !ELF.Symbol
      ELF.PutOpInSection %59 : !ELF.Symbol
      ELF.PutOpInSection %60 : !ELF.Symbol
      ELF.PutOpInSection %61 : !ELF.Symbol
      ELF.PutOpInSection %62 : !ELF.Symbol
      ELF.PutOpInSection %63 : !ELF.Symbol
      ELF.PutOpInSection %64 : !ELF.Symbol
    }
    %66 = ELF.CreateSymbolTableSection secName(".symtab.tasks") secFlags("SHF_NONE") -> !ELF.Section {
      ELF.PutOpInSection %40 : !ELF.Symbol
      ELF.PutOpInSection %41 : !ELF.Symbol
      ELF.PutOpInSection %44 : !ELF.Symbol
      ELF.PutOpInSection %45 : !ELF.Symbol
      ELF.PutOpInSection %46 : !ELF.Symbol
      ELF.PutOpInSection %42 : !ELF.Symbol
      ELF.PutOpInSection %43 : !ELF.Symbol
      ELF.PutOpInSection %47 : !ELF.Symbol
      ELF.PutOpInSection %48 : !ELF.Symbol
      %77 = ELF.Symbol %28 name("MappedInference_entry") type("VPU_STT_ENTRY") : !VPUIPRegMapped.Index<0>
    }
    %67 = ELF.CreateRelocationSection secName(".rlt.DMA_NetInput") sourceSymbolTableSection(%51) targetSection(%29) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USERINPUT") -> !ELF.Section {
      ELF.RelocImmOffset baseOp(%4 : !VPUIPRegMapped.Index<0>) offset(16) "R_VPU_64" %49 0
    }
    %68 = ELF.CreateRelocationSection secName(".rlt.DMA_NetOutput") sourceSymbolTableSection(%52) targetSection(%29) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USEROUTPUT") -> !ELF.Section {
      ELF.RelocImmOffset baseOp(%11 : !VPUIPRegMapped.Index<1>) offset(24) "R_VPU_64" %50 0
    }
    %69 = ELF.CreateRelocationSection secName(".rlt.text.dmaTasks") sourceSymbolTableSection(%65) targetSection(%29) secFlags(SHF_INFO_LINK) -> !ELF.Section {
      ELF.Reloc baseOp(%4 : !VPUIPRegMapped.Index<0>) offsetOf(%0 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) "R_VPU_64" %58 0
      ELF.RelocImmOffset baseOp(%4 : !VPUIPRegMapped.Index<0>) offset(0) "R_VPU_32_RTM" %61 128
      ELF.Reloc baseOp(%11 : !VPUIPRegMapped.Index<1>) offsetOf(%1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) "R_VPU_64" %58 2000
    }
    %70 = ELF.CreateRelocationSection secName(".rlt.text.KernelParams") sourceSymbolTableSection(%66) targetSection(%33) secFlags(SHF_INFO_LINK) -> !ELF.Section {
      ELF.RelocImmOffset baseOp(%10 : !VPUIPRegMapped.Index<0>) offset(12) "R_VPU_32" %46 72
      ELF.RelocImmOffset baseOp(%10 : !VPUIPRegMapped.Index<0>) offset(16) "R_VPU_32" %46 88
      ELF.RelocImmOffset baseOp(%10 : !VPUIPRegMapped.Index<0>) offset(48) "R_VPU_32" %46 120
      ELF.RelocImmOffset baseOp(%10 : !VPUIPRegMapped.Index<0>) offset(52) "R_VPU_32" %46 136
    }
    %71 = ELF.CreateRelocationSection secName(".rlt.text.KernelParams") sourceSymbolTableSection(%65) targetSection(%33) secFlags(SHF_INFO_LINK) -> !ELF.Section {
      ELF.Reloc baseOp(%10 : !VPUIPRegMapped.Index<0>) offsetOf(%0 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) "R_VPU_32" %58 0
      ELF.Reloc baseOp(%10 : !VPUIPRegMapped.Index<0>) offsetOf(%1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) "R_VPU_32" %58 2000
    }
    %72 = ELF.CreateRelocationSection secName(".rlt.text.ActKernelRanges") sourceSymbolTableSection(%66) targetSection(%34) secFlags(SHF_INFO_LINK) -> !ELF.Section {
      ELF.Reloc baseOp(%8 : !VPUIPRegMapped.Index<0>) offsetOf(%5 : !VPUIPRegMapped.Index<0>) "R_VPU_32" %44 0
    }
    %73 = ELF.CreateRelocationSection secName(".rlt.text.ActKernelInvocations") sourceSymbolTableSection(%65) targetSection(%35) secFlags(SHF_INFO_LINK) -> !ELF.Section {
      ELF.Reloc baseOp(%9 : !VPUIPRegMapped.Index<0>) offsetOf(%8 : !VPUIPRegMapped.Index<0>) "R_VPU_32_RTM" %60 24
    }
    %74 = ELF.CreateRelocationSection secName(".rlt.text.ActKernelInvocations") sourceSymbolTableSection(%66) targetSection(%35) secFlags(SHF_INFO_LINK) -> !ELF.Section {
      ELF.RelocImmOffset baseOp(%9 : !VPUIPRegMapped.Index<0>) offset(8) "R_VPU_32" %45 0
      ELF.RelocImmOffset baseOp(%9 : !VPUIPRegMapped.Index<0>) offset(4) "R_VPU_32" %46 0
    }
    %75 = ELF.CreateRelocationSection secName(".rlt.text.MappedInference") sourceSymbolTableSection(%27) targetSection(%36) secFlags(SHF_INFO_LINK) -> !ELF.Section {
      ELF.RelocImmOffset baseOp(%28 : !VPUIPRegMapped.Index<0>) offset(908) "R_VPU_32" %26 0
      ELF.Reloc baseOp(%28 : !VPUIPRegMapped.Index<0>) offsetOf(%12 : memref<8192xi8, @DDR>) "R_VPU_32" %14 8192
      ELF.Reloc baseOp(%28 : !VPUIPRegMapped.Index<0>) offsetOf(%15 : memref<8192xi8, @DDR>) "R_VPU_32" %17 8192
      ELF.Reloc baseOp(%28 : !VPUIPRegMapped.Index<0>) offsetOf(%18 : memref<8192xi8, @DDR>) "R_VPU_32" %20 8192
      ELF.Reloc baseOp(%28 : !VPUIPRegMapped.Index<0>) offsetOf(%21 : memref<8192xi8, @DDR>) "R_VPU_32" %23 8192
    }
    %76 = ELF.CreateRelocationSection secName(".rlt.text.MappedInference") sourceSymbolTableSection(%66) targetSection(%36) secFlags(SHF_INFO_LINK) -> !ELF.Section {
      ELF.Reloc baseOp(%28 : !VPUIPRegMapped.Index<0>) offsetOf(%4 : !VPUIPRegMapped.Index<0>) "R_VPU_64" %40 0
      ELF.Reloc baseOp(%28 : !VPUIPRegMapped.Index<0>) offsetOf(%2 : !VPUIPRegMapped.Index<0>) "R_VPU_64" %41 0
      ELF.Reloc baseOp(%28 : !VPUIPRegMapped.Index<0>) offsetOf(%8 : !VPUIPRegMapped.Index<0>) "R_VPU_64" %42 0
      ELF.Reloc baseOp(%28 : !VPUIPRegMapped.Index<0>) offsetOf(%9 : !VPUIPRegMapped.Index<0>) "R_VPU_64" %43 0
    }
    return %arg1 : memref<1x1x1x1000xf16>
  }
}

//CHECK-DAG: %[[VAL0:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPUIPRegMapped.Index<0>
//CHECK-DAG: %[[VAL1:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPUIPRegMapped.Index<1>
//CHECK: %[[VAL2:.*]] = ELF.CreateSection {{.*}} secName = ".text.BarrierConfigs"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL0]] : !VPUIPRegMapped.Index<0>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL1]] : !VPUIPRegMapped.Index<1>

//CHECK-DAG: %[[VAL3:.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<2000xui8, [@CMX_NN, 0]>
//CHECK-DAG: %[[VAL4:.*]] = VPUIPRegMapped.NNDMA  {dma_descriptor = {dstPlaneStride = 0 : i64, dstStride = 2000 : i64, dstWidth = 2000 : i64, len = 2000 : i64, numPlanes = 0 : i64, srcPlaneStride = 0 : i64, srcStride = 2000 : i64, srcWidth = 2000 : i64}, is_critical, is_out_of_order, port = 0 : si64} inputs(%[[VALINPUT:.*]] : {{.*}}) outputs(%[[VAL3]] : memref<2000xui8, [@CMX_NN, 0]>) updates(%[[VAL0]] : !VPUIPRegMapped.Index<0>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<0>
//CHECK-DAG: %[[VAL5:.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <2000> -> memref<2000xui8, [@CMX_NN, 0]>
//CHECK-DAG: %[[VAL6:.*]] = VPUIPRegMapped.NNDMA  {dma_descriptor = {dstPlaneStride = 0 : i64, dstStride = 2000 : i64, dstWidth = 2000 : i64, len = 2000 : i64, numPlanes = 0 : i64, srcPlaneStride = 0 : i64, srcStride = 2000 : i64, srcWidth = 2000 : i64}, is_critical, is_out_of_order, port = 0 : si64} inputs(%[[VAL5]] : memref<2000xui8, [@CMX_NN, 0]>) outputs(%[[VALOUTPUT:.*]] : {{.*}}) previousDMA(%[[VAL4]] : !VPUIPRegMapped.Index<0>) waits(%[[VAL1]] : !VPUIPRegMapped.Index<1>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<1>
//CHECK: %[[VAL7:.*]] = ELF.CreateSection {{.*}} secName = ".text.dmaTasks"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL4]] : !VPUIPRegMapped.Index<0>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL6]] : !VPUIPRegMapped.Index<1>

//CHECK-DAG: %[[VAL8:.*]] = VPUIPRegMapped.DeclareKernelText
//CHECK-DAG: %[[VAL9:.*]] = VPUIPRegMapped.DeclareKernelArgs
//CHECK-DAG: %[[VAL10:.*]] = VPUIPRegMapped.DeclareKernelEntry
//CHECK-DAG: %[[VAL11:.*]] = VPUIPRegMapped.ActKernelRange kernel_text_index(%[[VAL8]] : <0>) kernel_args_index(%[[VAL9]] : <0>) kernel_entry_index(%[[VAL10]] : <0>) -> !VPUIPRegMapped.Index<0>
//CHECK-DAG: %[[VAL12:.*]] = ELF.CreateSection {{.*}} secName = ".text.ActKernelRanges"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL11]] : !VPUIPRegMapped.Index<0>
//CHECK-DAG: %[[VAL13:.*]] = ELF.CreateSection {{.*}} secName = ".text.KernelText"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL8]] : !VPUIPRegMapped.Index<0>
//CHECK-DAG: %[[VAL14:.*]] = ELF.CreateSection {{.*}} secName = ".text.KernelData"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL9]] : !VPUIPRegMapped.Index<0>

//CHECK-DAG: %[[VAL15:.*]] = VPUIPRegMapped.ActKernelInvocation  range_index(%[[VAL11]] : <0>) waits(%[[VAL0]] : !VPUIPRegMapped.Index<0>) updates(%[[VAL1]] : !VPUIPRegMapped.Index<1>) tile(0) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<0>
//CHECK-DAG: %[[VAL16:.*]] = ELF.CreateSection {{.*}} secName = ".text.ActKernelInvocations"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL15]] : !VPUIPRegMapped.Index<0>

//CHECK-DAG: %[[VAL17:.*]] = VPUIPRegMapped.KernelParams inputs(%[[VAL3]] : memref<2000xui8, [@CMX_NN, 0]>) outputs(%[[VAL5]] : memref<2000xui8, [@CMX_NN, 0]>) kernel_type("Softmax") kernel_params({{.*}}) -> !VPUIPRegMapped.Index<0>
//CHECK-DAG: %[[VAL18:.*]] = ELF.CreateSection {{.*}} secName = ".text.KernelParams"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL17]] : !VPUIPRegMapped.Index<0>

//CHECK-DAG: %[[VAL19:.*]] = VPURT.DeclareBuffer "DDR"
//CHECK-NEXT: %[[VAL20:.*]] = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags(VPU_SHF_PROC_SHAVE) {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".bss.actShaveStack_0"} -> !ELF.Section {
//CHECK-NEXT: ELF.PutOpInSection %[[VAL19]] : memref<8192xui8, @DDR>

//CHECK-DAG: %[[VAL21:.*]] = VPURT.DeclareBuffer "DDR"
//CHECK-NEXT: %[[VAL22:.*]] = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags(VPU_SHF_PROC_SHAVE) {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".bss.actShaveStack_1"} -> !ELF.Section {
//CHECK-NEXT: ELF.PutOpInSection %[[VAL21]] : memref

//CHECK-DAG: %[[VAL23:.*]] = VPURT.DeclareBuffer "DDR"
//CHECK-NEXT: %[[VAL24:.*]] = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags(VPU_SHF_PROC_SHAVE) {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".bss.actShaveStack_2"} -> !ELF.Section {
//CHECK-NEXT: ELF.PutOpInSection %[[VAL23]] : memref

//CHECK-DAG: %[[VAL25:.*]] = VPURT.DeclareBuffer "DDR"
//CHECK-NEXT: %[[VAL26:.*]] = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags(VPU_SHF_PROC_SHAVE) {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".bss.actShaveStack_3"} -> !ELF.Section {
//CHECK-NEXT: ELF.PutOpInSection %[[VAL25]] : memref

//CHECK-DAG: %[[VAL27:.*]] = VPUIPRegMapped.ActShaveRt kernel("nnActEntry") -> !VPUIPRegMapped.Index<0>
//CHECK-NEXT: %[[VAL28:.*]] = ELF.CreateSection {{.*}} {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".text.actKernelRtConfigSec"} -> !ELF.Section {
//CHECK-NEXT: ELF.PutOpInSection %[[VAL27]] : !VPUIPRegMapped.Index<0>

//CHECK-DAG: %[[VAL29:.*]] = VPUIPRegMapped.MappedInference
//CHECK-SAME: dmas(%[[VAL4]] : !VPUIPRegMapped.Index<0>)
//CHECK-SAME: actKernelRanges(%[[VAL11]] : !VPUIPRegMapped.Index<0>)
//CHECK-SAME: actKernelInvocations(%[[VAL15]] : !VPUIPRegMapped.Index<0>)
//CHECK-SAME: barriers(%[[VAL0]] : !VPUIPRegMapped.Index<0>)
//CHECK-SAME: actShaveRt(%[[VAL27]] : !VPUIPRegMapped.Index<0>)
//CHECK-SAME: actShaveStacks(%[[VAL19]], %[[VAL21]], %[[VAL23]], %[[VAL25]] : {{.*}}>)
//CHECK-SAME: dmaCount([2]) invariantCount(0) variantCount(0) actKernelRangesCount(1) actKernelInvocationsCount(1) barrierCount(2)
//CHECK-SAME: -> !VPUIPRegMapped.Index<0>

//CHECK-DAG: %[[VAL30:.*]] = ELF.CreateSection {{.*}} secName = ".text.MappedInference"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL29]] : !VPUIPRegMapped.Index<0>

//CHECK-DAG: %[[VAL31:.*]] = ELF.CreateMetadataSection {{.*}} secName = ".metadata"
//CHECK-NEXT: VPUIPRegMapped.NetworkMetadata

//CHECK-DAG: %[[VAL32:.*]] = ELF.CreateSection {{.*}} secName = ".text.DPUInvariants"

//CHECK-DAG: %[[VAL33:.*]] = ELF.CreateSection {{.*}} secName = ".text.DPUVariants"

//CHECK-DAG: %[[VAL34:.*]] = ELF.CreateSection {{.*}} secName = ".data.ConstIO"

//CHECK-DAG: %[[VAL35:.*]] = ELF.CreateLogicalSection {{.*}} secName = ".data.BuffersIO"

//CHECK-DAG: %[[VAL36:.*]] = ELF.Symbol %[[VAL28]] name("sym_actKernelRtConfigsSec") {{.*}} : !ELF.Section
//CHECK-DAG: %[[VAL37:.*]] = ELF.Symbol %[[VAL20]] name("sym_actShaveStack_0") {{.*}} : !ELF.Section
//CHECK-DAG: %[[VAL38:.*]] = ELF.Symbol %[[VAL22]] name("sym_actShaveStack_1") {{.*}} : !ELF.Section
//CHECK-DAG: %[[VAL39:.*]] = ELF.Symbol %[[VAL24]] name("sym_actShaveStack_2") {{.*}} : !ELF.Section
//CHECK-DAG: %[[VAL40:.*]] = ELF.Symbol %[[VAL26]] name("sym_actShaveStack_3") {{.*}} : !ELF.Section

//CHECK-DAG: %[[VAL41:.*]] = ELF.CreateSymbolTableSection secName(".symtab.actKernelRtConfig")
//CHECK-NEXT: ELF.PutOpInSection %[[VAL36]] : !ELF.Symbol
//CHECK-NEXT: ELF.PutOpInSection %[[VAL37]] : !ELF.Symbol
//CHECK-NEXT: ELF.PutOpInSection %[[VAL38]] : !ELF.Symbol
//CHECK-NEXT: ELF.PutOpInSection %[[VAL39]] : !ELF.Symbol
//CHECK-NEXT: ELF.PutOpInSection %[[VAL40]] : !ELF.Symbol

//CHECK-DAG: %[[VAL42:.*]] = ELF.Symbol %[[VALINPUT]] name("input") type("STT_NOTYPE") size(2000) {value = 0 : ui64} : {{.*}}
//CHECK-DAG: %[[VAL43:.*]] = ELF.CreateSymbolTableSection secName(".symtab.input") secFlags(VPU_SHF_USERINPUT) -> !ELF.Section
//CHECK-NEXT: ELF.PutOpInSection %[[VAL42]] : !ELF.Symbol

//CHECK-DAG: %[[VAL44:.*]] = ELF.Symbol %[[VALOUTPUT]] name("hswish") type("STT_NOTYPE") size(2000) {value = 0 : ui64} : {{.*}}
//CHECK-DAG: %[[VAL45:.*]] = ELF.CreateSymbolTableSection secName(".symtab.output") secFlags(VPU_SHF_USEROUTPUT) -> !ELF.Section
//CHECK-NEXT: ELF.PutOpInSection %[[VAL44]] : !ELF.Symbol

//CHECK-DAG: %[[VAL46:.*]] = ELF.Symbol %[[VAL35]] name("sym_bufferSection") type("STT_NOTYPE") size(0) {value = 0 : ui64} : !ELF.Section
//CHECK-DAG: %[[VAL47:.*]] = ELF.Symbol %[[VAL34]] name("sym_constSection") type("STT_NOTYPE") size(0) {value = 0 : ui64} : !ELF.Section
//CHECK-DAG: %[[VAL48:.*]] = ELF.CreateSymbolTableSection secName(".symtab.buffers") secFlags("SHF_NONE") -> !ELF.Section
//CHECK-NEXT: ELF.PutOpInSection %[[VAL46]] : !ELF.Symbol
//CHECK-NEXT: ELF.PutOpInSection %[[VAL47]] : !ELF.Symbol

//CHECK-DAG: %[[VAL49:.*]] = ELF.Symbol %[[VAL7]] name("sym_dmaSection0") type("STT_NOTYPE") size(0) {value = 0 : ui64} : !ELF.Section
//CHECK-DAG: %[[VAL50:.*]] = ELF.Symbol %[[VAL2]] name("sym_barrierSection") type("STT_NOTYPE") size(0) {value = 0 : ui64} : !ELF.Section
//CHECK-DAG: %[[VAL51:.*]] = ELF.Symbol %[[VAL13]] name("sym_kernelTextSection") type("STT_NOTYPE") size(0) {value = 0 : ui64} : !ELF.Section
//CHECK-DAG: %[[VAL52:.*]] = ELF.Symbol %[[VAL14]] name("sym_kernelDataSection") type("STT_NOTYPE") size(0) {value = 0 : ui64} : !ELF.Section
//CHECK-DAG: %[[VAL53:.*]] = ELF.Symbol %[[VAL18]] name("sym_kernelParamsSection") type("STT_NOTYPE") size(0) {value = 0 : ui64} : !ELF.Section
//CHECK-DAG: %[[VAL54:.*]] = ELF.Symbol %[[VAL12]] name("sym_actKernelRangeSection") type("STT_NOTYPE") size(0) {value = 0 : ui64} : !ELF.Section
//CHECK-DAG: %[[VAL55:.*]] = ELF.Symbol %[[VAL16]] name("sym_actKernelInvo") type("STT_NOTYPE") size(0) {value = 0 : ui64} : !ELF.Section
//CHECK-DAG: %[[VAL56:.*]] = ELF.Symbol %[[VAL32]] name("sym_inVariantsSection") type("STT_NOTYPE") size(0) {value = 0 : ui64} : !ELF.Section
//CHECK-DAG: %[[VAL57:.*]] = ELF.Symbol %[[VAL33]] name("sym_variantsSection") type("STT_NOTYPE") size(0) {value = 0 : ui64} : !ELF.Section
//CHECK-DAG: %[[VAL58:.*]] = ELF.Symbol %[[VAL30]] name("MappedInference_entry") type("VPU_STT_ENTRY") size(0) {value = 0 : ui64} : !ELF.Section
//CHECK-DAG: %[[VAL59:.*]] = ELF.CreateSymbolTableSection secName(".symtab.tasks") secFlags("SHF_NONE") -> !ELF.Section
//CHECK-NEXT: ELF.PutOpInSection %[[VAL49]] : !ELF.Symbol
//CHECK-NEXT: ELF.PutOpInSection %[[VAL50]] : !ELF.Symbol
//CHECK-NEXT: ELF.PutOpInSection %[[VAL51]] : !ELF.Symbol
//CHECK-NEXT: ELF.PutOpInSection %[[VAL52]] : !ELF.Symbol
//CHECK-NEXT: ELF.PutOpInSection %[[VAL53]] : !ELF.Symbol
//CHECK-NEXT: ELF.PutOpInSection %[[VAL54]] : !ELF.Symbol
//CHECK-NEXT: ELF.PutOpInSection %[[VAL55]] : !ELF.Symbol
//CHECK-NEXT: ELF.PutOpInSection %[[VAL56]] : !ELF.Symbol
//CHECK-NEXT: ELF.PutOpInSection %[[VAL57]] : !ELF.Symbol
//CHECK-NEXT: ELF.PutOpInSection %[[VAL58]] : !ELF.Symbol

//CHECK: %[[VALC0:.*]] = arith.constant 0 : i8
//CHECK: %[[VAL60:.*]] = ELF.Symbol %[[VALC0]] name("VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR") {isBuiltin} : i8
//CHECK: %[[VALC1:.*]] = arith.constant 1 : i8
//CHECK: %[[VAL61:.*]] = ELF.Symbol %[[VALC1]] name("VPU_NNRD_SYM_RTM_IVAR") {isBuiltin} : i8
//CHECK: %[[VALC2:.*]] = arith.constant 2 : i8
//CHECK: %[[VAL62:.*]] = ELF.Symbol %[[VALC2]] name("VPU_NNRD_SYM_RTM_ACT") {isBuiltin} : i8
//CHECK: %[[VALC3:.*]] = arith.constant 3 : i8
//CHECK: %[[VAL63:.*]] = ELF.Symbol %[[VALC3]] name("VPU_NNRD_SYM_RTM_DMA0") {isBuiltin} : i8
//CHECK: %[[VALC4:.*]] = arith.constant 4 : i8
//CHECK: %[[VAL64:.*]] = ELF.Symbol %[[VALC4]] name("VPU_NNRD_SYM_RTM_DMA1") {isBuiltin} : i8
//CHECK: %[[VALC5:.*]] = arith.constant 5 : i8
//CHECK: %[[VAL65:.*]] = ELF.Symbol %[[VALC5]] name("VPU_NNRD_SYM_FIFO_BASE") {isBuiltin} : i8
//CHECK: %[[VALC6:.*]] = arith.constant 6 : i8
//CHECK: %[[VAL66:.*]] = ELF.Symbol %[[VALC6]] name("VPU_NNRD_SYM_BARRIERS_START") {isBuiltin} : i8
//CHECK: %[[VAL67:.*]] = ELF.CreateSymbolTableSection secName("VPU_RT_SYMTAB") secFlags("SHF_NONE") {isBuiltin} -> !ELF.Section {
//CHECK-NEXT: ELF.PutOpInSection %[[VAL60]] : !ELF.Symbol
//CHECK-NEXT: ELF.PutOpInSection %[[VAL61]] : !ELF.Symbol
//CHECK-NEXT: ELF.PutOpInSection %[[VAL62]] : !ELF.Symbol
//CHECK-NEXT: ELF.PutOpInSection %[[VAL63]] : !ELF.Symbol
//CHECK-NEXT: ELF.PutOpInSection %[[VAL64]] : !ELF.Symbol
//CHECK-NEXT: ELF.PutOpInSection %[[VAL65]] : !ELF.Symbol
//CHECK-NEXT: ELF.PutOpInSection %[[VAL66]] : !ELF.Symbol

//CHECK: %[[VAL68:.*]] = ELF.CreateRelocationSection secName(".rlt.DMA_NetInput") sourceSymbolTableSection(%[[VAL43]]) targetSection(%[[VAL7]]) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USERINPUT") -> !ELF.Section
//CHECK-DAG: ELF.RelocImmOffset offset(16) "R_VPU_64" %[[VAL42]] 0
//CHECK: %[[VAL69:.*]] = ELF.CreateRelocationSection secName(".rlt.DMA_NetOutput") sourceSymbolTableSection(%[[VAL45]]) targetSection(%[[VAL7]]) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USEROUTPUT") -> !ELF.Section
//CHECK-DAG: ELF.RelocImmOffset offset(152) "R_VPU_64" %[[VAL44]] 0

//CHECK: %[[VAL70:.*]] = ELF.CreateRelocationSection secName(".rlt.text.dmaTasks") sourceSymbolTableSection(%[[VAL67]]) targetSection(%[[VAL7]]) secFlags(SHF_INFO_LINK) -> !ELF.Section
//CHECK-DAG: ELF.RelocImmOffset offset(24) "R_VPU_64" %[[VAL60]] 0
//CHECK-DAG: ELF.RelocImmOffset offset(0) "R_VPU_32_RTM" %[[VAL63]] 128
//CHECK-DAG: ELF.RelocImmOffset offset(144) "R_VPU_64" %[[VAL60]] 2000


//CHECK: %[[VAL71:.*]] = ELF.CreateRelocationSection secName(".rlt.text.KernelParams") sourceSymbolTableSection(%[[VAL59]]) targetSection(%[[VAL18]]) secFlags(SHF_INFO_LINK) -> !ELF.Section
//CHECK-DAG: ELF.RelocImmOffset offset(12) "R_VPU_32" %[[VAL53]] 72
//CHECK-DAG: ELF.RelocImmOffset offset(16) "R_VPU_32" %[[VAL53]] 88
//CHECK-DAG: ELF.RelocImmOffset offset(48) "R_VPU_32" %[[VAL53]] 120
//CHECK-DAG: ELF.RelocImmOffset offset(52) "R_VPU_32" %[[VAL53]] 136
//CHECK: %[[VAL72:.*]] = ELF.CreateRelocationSection secName(".rlt.text.KernelParams") sourceSymbolTableSection(%[[VAL67]]) targetSection(%[[VAL18]]) secFlags(SHF_INFO_LINK) -> !ELF.Section
//CHECK-DAG: ELF.RelocImmOffset offset(0) "R_VPU_32" %[[VAL60]] 0
//CHECK-DAG: ELF.RelocImmOffset offset(36) "R_VPU_32" %[[VAL60]] 2000

//CHECK: %[[VAL73:.*]] = ELF.CreateRelocationSection secName(".rlt.text.ActKernelRanges") sourceSymbolTableSection(%[[VAL59]]) targetSection(%[[VAL12]]) secFlags(SHF_INFO_LINK) -> !ELF.Section
//CHECK-DAG: ELF.RelocImmOffset offset(8) "R_VPU_32" %[[VAL51]] 0
//CHECK: %[[VAL74:.*]] = ELF.CreateRelocationSection secName(".rlt.text.ActKernelInvocations") sourceSymbolTableSection(%[[VAL67]]) targetSection(%[[VAL16]]) secFlags(SHF_INFO_LINK) -> !ELF.Section
//CHECK-DAG: ELF.RelocImmOffset offset(0) "R_VPU_32_RTM" %[[VAL62]] 24
//CHECK: %[[VAL75:.*]] = ELF.CreateRelocationSection secName(".rlt.text.ActKernelInvocations") sourceSymbolTableSection(%[[VAL59]]) targetSection(%[[VAL16]]) secFlags(SHF_INFO_LINK) -> !ELF.Section
//CHECK-DAG: ELF.RelocImmOffset offset(8) "R_VPU_32" %[[VAL52]] 0
//CHECK-DAG: ELF.RelocImmOffset offset(4) "R_VPU_32" %[[VAL53]] 0

//CHECK: %[[VAL76:.*]] = ELF.CreateRelocationSection secName(".rlt.text.MappedInference") sourceSymbolTableSection(%[[VAL41]]) targetSection(%[[VAL30]]) secFlags(SHF_INFO_LINK) -> !ELF.Section
//CHECK-DAG: ELF.RelocImmOffset offset(908) "R_VPU_32" %[[VAL36]] 0
//CHECK-DAG: ELF.RelocImmOffset offset(912) "R_VPU_32" %[[VAL37]] 8192
//CHECK-DAG: ELF.RelocImmOffset offset(916) "R_VPU_32" %[[VAL38]] 8192
//CHECK-DAG: ELF.RelocImmOffset offset(920) "R_VPU_32" %[[VAL39]] 8192
//CHECK-DAG: ELF.RelocImmOffset offset(924) "R_VPU_32" %[[VAL40]] 8192
//CHECK: %[[VAL77:.*]] = ELF.CreateRelocationSection secName(".rlt.text.MappedInference") sourceSymbolTableSection(%[[VAL59]]) targetSection(%[[VAL30]]) secFlags(SHF_INFO_LINK) -> !ELF.Section
//CHECK-DAG: ELF.RelocImmOffset offset(8) "R_VPU_64" %[[VAL49]] 0
//CHECK-DAG: ELF.RelocImmOffset offset(72) "R_VPU_64" %[[VAL50]] 0
//CHECK-DAG: ELF.RelocImmOffset offset(88) "R_VPU_64" %[[VAL54]] 0
//CHECK-DAG: ELF.RelocImmOffset offset(104) "R_VPU_64" %[[VAL55]] 0
