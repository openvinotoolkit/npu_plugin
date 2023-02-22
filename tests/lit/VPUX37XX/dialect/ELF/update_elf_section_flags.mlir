//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --update-ELF-section-flags %s | FileCheck %s
module @Test {
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1000xf16>
  } outputsInfo : {
    DataInfo "softmax" : tensor<1x1000xf16>
  }
  module @VPU.SW {
    func private @builtin_softmax(memref<*xf16>, memref<*xf16>, i64) attributes {VPU.kernel_code = "single_shave_softmax.cpp", VPU.kernel_entry = "singleShaveSoftmax"}
  }
  func @main(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    %0 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer "CMX_NN" [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %2 = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPUIPRegMapped.Index<0>
    %3 = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPUIPRegMapped.Index<1>
    %4 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x1x1000xf16>) outputs(%0 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) updates(%2 : !VPUIPRegMapped.Index<0>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<0>
    %5 = VPUIPRegMapped.DeclareKernelText kernel_path("singleShaveSoftmax") -> !VPUIPRegMapped.Index<0>
    %6 = VPUIPRegMapped.DeclareKernelArgs kernel_path("singleShaveSoftmax") -> !VPUIPRegMapped.Index<0>
    %7 = VPUIPRegMapped.DeclareKernelEntry kernel_path("singleShaveSoftmax") -> !VPUIPRegMapped.Index<0>
    %8 = VPUIPRegMapped.ActKernelRange kernel_text_index(%5 : <0>) kernel_args_index(%6 : <0>) kernel_entry_index(%7 : <0>) -> !VPUIPRegMapped.Index<0>
    %9 = VPUIPRegMapped.ActKernelInvocation range_index(%8 : <0>) waits(%2 : !VPUIPRegMapped.Index<0>) updates(%3 : !VPUIPRegMapped.Index<1>) tile(0) start_after(0) clean_after(1) -> !VPUIPRegMapped.Index<0>
    %10 = VPUIPRegMapped.KernelParams inputs(%0 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("singleShaveSoftmax") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : vector<80xui8>) -> !VPUIPRegMapped.Index<0>
    %11 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x1x1x1000xf16>) previousDMA(%4 : !VPUIPRegMapped.Index<0>) waits(%3 : !VPUIPRegMapped.Index<1>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<1>
    %12 = VPUIPRegMapped.MappedInference dmas(%4 : !VPUIPRegMapped.Index<0>) actKernelRanges(%8 : !VPUIPRegMapped.Index<0>) actKernelInvocations(%9 : !VPUIPRegMapped.Index<0>) barriers(%2 : !VPUIPRegMapped.Index<0>) dmaCount([2, 0]) invariantCount(0) variantCount(0) actKernelRangesCount(1) actKernelInvocationsCount(1) barrierCount(2) -> !VPUIPRegMapped.Index<0>
    %13 = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_ALLOC") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks"} -> !ELF.Section {
      ELF.PutOpInSection %4 : !VPUIPRegMapped.Index<0>
      ELF.PutOpInSection %11 : !VPUIPRegMapped.Index<1>
    }
    // CHECK:  %[[VAL1:.*]] = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR|VPU_SHF_PROC_DMA") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks"} -> !ELF.Section {
    %14 = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_NONE) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.BarrierConfigs"} -> !ELF.Section {
      ELF.PutOpInSection %2 : !VPUIPRegMapped.Index<0>
      ELF.PutOpInSection %3 : !VPUIPRegMapped.Index<1>
    }
    // CHECK:  %[[VAL2:.*]] = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.BarrierConfigs"} -> !ELF.Section {
    %15 = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_NONE) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.KernelText"} -> !ELF.Section {
      ELF.PutOpInSection %5 : !VPUIPRegMapped.Index<0>
    }
    // CHECK:  %[[VAL3:.*]] = ELF.CreateSection secType(SHT_PROGBITS) secFlags(VPU_SHF_PROC_SHAVE) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.KernelText"} -> !ELF.Section {
    %16 = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_NONE) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.KernelData"} -> !ELF.Section {
      ELF.PutOpInSection %6 : !VPUIPRegMapped.Index<0>
    }
    // CHECK:  %[[VAL4:.*]] = ELF.CreateSection secType(SHT_PROGBITS) secFlags(VPU_SHF_PROC_SHAVE) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.KernelData"} -> !ELF.Section {
    %17 = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_NONE) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.KernelParams"} -> !ELF.Section {
      ELF.PutOpInSection %10 : !VPUIPRegMapped.Index<0>
    }
    // CHECK:  %[[VAL5:.*]] = ELF.CreateSection secType(SHT_PROGBITS) secFlags(VPU_SHF_PROC_SHAVE) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.KernelParams"} -> !ELF.Section {
    %18 = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_NONE) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.ActKernelRanges"} -> !ELF.Section {
      ELF.PutOpInSection %8 : !VPUIPRegMapped.Index<0>
    }
    // CHECK:  %[[VAL6:.*]] = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_EXECINSTR|VPU_SHF_PROC_DMA") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.ActKernelRanges"} -> !ELF.Section {
    %19 = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_NONE) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.ActKernelInvocations"} -> !ELF.Section {
      ELF.PutOpInSection %9 : !VPUIPRegMapped.Index<0>
    }
    // CHECK:  %[[VAL7:.*]] = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_EXECINSTR|VPU_SHF_PROC_DMA") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.ActKernelInvocations"} -> !ELF.Section {
    %20 = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_NONE) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.MappedInference"} -> !ELF.Section {
      ELF.PutOpInSection %12 : !VPUIPRegMapped.Index<0>
    }
    // CHECK:  %[[VAL8:.*]] = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.MappedInference"} -> !ELF.Section {
    %21 = ELF.CreateMetadataSection secFlags("SHF_NONE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".metadata"} -> !ELF.Section {
      %55 = VPUIPRegMapped.NetworkMetadata -> !VPUIPRegMapped.Index<0>
    }
    %22 = ELF.Symbol %13 name("sym_dmaSection0") : !ELF.Section
    %23 = ELF.Symbol %14 name("sym_barrierSection") : !ELF.Section
    %24 = ELF.Symbol %18 name("sym_actKernelRangeSection") : !ELF.Section
    %25 = ELF.Symbol %19 name("sym_actKernelInvo") : !ELF.Section
    %26 = ELF.Symbol %15 name("sym_kernelTextSection") : !ELF.Section
    %27 = ELF.Symbol %16 name("sym_kernelDataSection") : !ELF.Section
    %28 = ELF.Symbol %17 name("sym_kernelParamsSection") : !ELF.Section
    %29 = ELF.Symbol %arg0 name("input") size(2000) : memref<1x1x1x1000xf16>
    %30 = ELF.Symbol %arg1 name("softmax") size(2000) : memref<1x1x1x1000xf16>
    %31 = ELF.CreateSymbolTableSection secName(".symtab.input") secFlags(VPU_SHF_USERINPUT) -> !ELF.Section {
      ELF.PutOpInSection %29 : !ELF.Symbol
    }
    %32 = ELF.CreateSymbolTableSection secName(".symtab.output") secFlags(VPU_SHF_USEROUTPUT) -> !ELF.Section {
      ELF.PutOpInSection %30 : !ELF.Symbol
    }
    %c0_i8 = arith.constant 0 : i8
    %33 = ELF.Symbol %c0_i8 name("VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR") {isBuiltin} : i8
    %c1_i8 = arith.constant 1 : i8
    %34 = ELF.Symbol %c1_i8 name("VPU_NNRD_SYM_RTM_IVAR") {isBuiltin} : i8
    %c2_i8 = arith.constant 2 : i8
    %35 = ELF.Symbol %c2_i8 name("VPU_NNRD_SYM_RTM_ACT") {isBuiltin} : i8
    %c3_i8 = arith.constant 3 : i8
    %36 = ELF.Symbol %c3_i8 name("VPU_NNRD_SYM_RTM_DMA0") {isBuiltin} : i8
    %c4_i8 = arith.constant 4 : i8
    %37 = ELF.Symbol %c4_i8 name("VPU_NNRD_SYM_RTM_DMA1") {isBuiltin} : i8
    %c5_i8 = arith.constant 5 : i8
    %38 = ELF.Symbol %c5_i8 name("VPU_NNRD_SYM_FIFO_BASE") {isBuiltin} : i8
    %c6_i8 = arith.constant 6 : i8
    %39 = ELF.Symbol %c6_i8 name("VPU_NNRD_SYM_BARRIERS_START") {isBuiltin} : i8
    %40 = ELF.CreateSymbolTableSection secName("VPU_RT_SYMTAB") secFlags("SHF_NONE") {isBuiltin} -> !ELF.Section {
      ELF.PutOpInSection %33 : !ELF.Symbol
      ELF.PutOpInSection %34 : !ELF.Symbol
      ELF.PutOpInSection %35 : !ELF.Symbol
      ELF.PutOpInSection %36 : !ELF.Symbol
      ELF.PutOpInSection %37 : !ELF.Symbol
      ELF.PutOpInSection %38 : !ELF.Symbol
      ELF.PutOpInSection %39 : !ELF.Symbol
    }
    %41 = ELF.CreateSymbolTableSection secName(".symtab.tasks") secFlags("SHF_NONE") -> !ELF.Section {
      ELF.PutOpInSection %22 : !ELF.Symbol
      ELF.PutOpInSection %23 : !ELF.Symbol
      ELF.PutOpInSection %26 : !ELF.Symbol
      ELF.PutOpInSection %27 : !ELF.Symbol
      ELF.PutOpInSection %28 : !ELF.Symbol
      ELF.PutOpInSection %24 : !ELF.Symbol
      ELF.PutOpInSection %25 : !ELF.Symbol
      %55 = ELF.Symbol %12 name("MappedInference_entry") type("VPU_STT_ENTRY") : !VPUIPRegMapped.Index<0>
    }
    %42 = ELF.CreateRelocationSection secName(".rlt.DMA_NetInput") sourceSymbolTableSection(%31) targetSection(%13) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USERINPUT") -> !ELF.Section {
      ELF.RelocImmOffset baseOp(%4 : !VPUIPRegMapped.Index<0>) offset(16) "R_VPU_64" %29 0
    }
    %43 = ELF.CreateRelocationSection secName(".rlt.DMA_NetOutput") sourceSymbolTableSection(%32) targetSection(%13) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USEROUTPUT") -> !ELF.Section {
      ELF.RelocImmOffset baseOp(%11 : !VPUIPRegMapped.Index<1>) offset(24) "R_VPU_64" %30 0
    }
    %44 = ELF.CreateRelocationSection secName(".rlt.dmaIO_CMX") sourceSymbolTableSection(%40) targetSection(%13) secFlags(SHF_INFO_LINK) -> !ELF.Section {
      ELF.RelocImmOffset baseOp(%4 : !VPUIPRegMapped.Index<0>) offset(24) "R_VPU_64" %33 0
      ELF.RelocImmOffset baseOp(%4 : !VPUIPRegMapped.Index<0>) offset(0) "R_VPU_32_RTM" %36 192
      ELF.RelocImmOffset baseOp(%11 : !VPUIPRegMapped.Index<1>) offset(16) "R_VPU_64" %33 2000
    }
    %45 = ELF.CreateRelocationSection secName(".rlt.KernelParams") sourceSymbolTableSection(%41) targetSection(%17) secFlags(SHF_INFO_LINK) -> !ELF.Section {
      ELF.RelocImmOffset baseOp(%10 : !VPUIPRegMapped.Index<0>) offset(12) "R_VPU_32" %28 80
      ELF.RelocImmOffset baseOp(%10 : !VPUIPRegMapped.Index<0>) offset(48) "R_VPU_32" %28 96
      ELF.RelocImmOffset baseOp(%10 : !VPUIPRegMapped.Index<0>) offset(16) "R_VPU_32" %28 112
      ELF.RelocImmOffset baseOp(%10 : !VPUIPRegMapped.Index<0>) offset(52) "R_VPU_32" %28 144
    }
    %46 = ELF.CreateRelocationSection secName(".rlt.KernelParamsIO_CMX") sourceSymbolTableSection(%40) targetSection(%17) secFlags(SHF_INFO_LINK) -> !ELF.Section {
      ELF.RelocImmOffset baseOp(%10 : !VPUIPRegMapped.Index<0>) offset(0) "R_VPU_32" %33 0
      ELF.RelocImmOffset baseOp(%10 : !VPUIPRegMapped.Index<0>) offset(36) "R_VPU_32" %33 2000
    }
    %47 = ELF.CreateRelocationSection secName(".rlt.ActKernelRange") sourceSymbolTableSection(%41) targetSection(%18) secFlags(SHF_INFO_LINK) -> !ELF.Section {
      ELF.RelocImmOffset baseOp(%8 : !VPUIPRegMapped.Index<0>) offset(8) "R_VPU_32" %26 0
    }
    %48 = ELF.CreateRelocationSection secName(".rlt.ActKernelInvo") sourceSymbolTableSection(%41) targetSection(%19) secFlags(SHF_INFO_LINK) -> !ELF.Section {
      ELF.RelocImmOffset baseOp(%9 : !VPUIPRegMapped.Index<0>) offset(0) "R_VPU_32" %24 0
      ELF.RelocImmOffset baseOp(%9 : !VPUIPRegMapped.Index<0>) offset(8) "R_VPU_32" %27 0
      ELF.RelocImmOffset baseOp(%9 : !VPUIPRegMapped.Index<0>) offset(4) "R_VPU_32" %28 0
    }
    %49 = VPURT.DeclareBuffer "DDR" <64> -> memref<262144xi32, @DDR>
    %50 = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags("SHF_NONE") {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".bss.actKernelRtConfigSec"} -> !ELF.Section {
      ELF.PutOpInSection %49 : memref<262144xi32, @DDR>
    }
    %51 = ELF.Symbol %50 name("sym_actKernelRtConfigsSec") : !ELF.Section
    %52 = ELF.CreateSymbolTableSection secName(".symtab.actKernelRtConfig") secFlags("SHF_NONE") -> !ELF.Section {
      ELF.PutOpInSection %51 : !ELF.Symbol
    }
    %53 = ELF.CreateRelocationSection secName(".rlt.MI_AKRtConfig") sourceSymbolTableSection(%52) targetSection(%20) secFlags(SHF_INFO_LINK) -> !ELF.Section {
      ELF.RelocImmOffset baseOp(%12 : !VPUIPRegMapped.Index<0>) offset(828) "R_VPU_32" %51 0
    }
    %54 = ELF.CreateRelocationSection secName(".rlt.MappedInference") sourceSymbolTableSection(%41) targetSection(%20) secFlags(SHF_INFO_LINK) -> !ELF.Section {
      ELF.RelocImmOffset baseOp(%12 : !VPUIPRegMapped.Index<0>) offset(0) "R_VPU_64" %22 0
      ELF.RelocImmOffset baseOp(%12 : !VPUIPRegMapped.Index<0>) offset(72) "R_VPU_64" %23 0
      ELF.RelocImmOffset baseOp(%12 : !VPUIPRegMapped.Index<0>) offset(768) "R_VPU_64" %24 0
      ELF.RelocImmOffset baseOp(%12 : !VPUIPRegMapped.Index<0>) offset(784) "R_VPU_64" %25 0
    }
    return %arg1 : memref<1x1x1x1000xf16>
  }
}

// -----

module @Test {
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "Parameter" : tensor<1x100xui32>
  } outputsInfo : {
    DataInfo "Convert" : tensor<1x100xui32>
  }
  func @main(%arg0: memref<1x100xui32, @DDR>, %arg1: memref<1x100xui32, @DDR>) -> memref<1x100xui32, @DDR> {
    %0 = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPUIPRegMapped.Index<0>
    %1 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x100xui32, @DDR>
    %cst = const.Declare memref<1x100xui32> = dense<1> : tensor<1x100xui32>
    %2 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%cst : memref<1x100xui32>) outputs(%1 : memref<1x100xui32, @DDR>) updates(%0 : !VPUIPRegMapped.Index<0>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<0>
    %3 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%1 : memref<1x100xui32, @DDR>) outputs(%arg1 : memref<1x100xui32, @DDR>) previousDMA(%2 : !VPUIPRegMapped.Index<0>) waits(%0 : !VPUIPRegMapped.Index<0>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<1>
    %4 = VPUIPRegMapped.MappedInference dmas(%2 : !VPUIPRegMapped.Index<0>) barriers(%0 : !VPUIPRegMapped.Index<0>) dmaCount([2, 0]) invariantCount(0) variantCount(0) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(1) -> !VPUIPRegMapped.Index<0>
    %5 = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_ALLOC") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks"} -> !ELF.Section {
      ELF.PutOpInSection %2 : !VPUIPRegMapped.Index<0>
      ELF.PutOpInSection %3 : !VPUIPRegMapped.Index<1>
    }
    // CHECK:  %[[VAL1:.*]] = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR|VPU_SHF_PROC_DMA") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks"} -> !ELF.Section {
    %6 = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_NONE) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.BarrierConfigs"} -> !ELF.Section {
      ELF.PutOpInSection %0 : !VPUIPRegMapped.Index<0>
    }
    // CHECK:  %[[VAL2:.*]] = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.BarrierConfigs"} -> !ELF.Section {
    %7 = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_NONE) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.MappedInference"} -> !ELF.Section {
      ELF.PutOpInSection %4 : !VPUIPRegMapped.Index<0>
    }
    // CHECK:  %[[VAL3:.*]] = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.MappedInference"} -> !ELF.Section {
    %8 = ELF.CreateMetadataSection secFlags("SHF_NONE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".metadata"} -> !ELF.Section {
      %38 = VPUIPRegMapped.NetworkMetadata -> !VPUIPRegMapped.Index<0>
    }
    %9 = ELF.Symbol %5 name("sym_dmaSection0") : !ELF.Section
    %10 = ELF.Symbol %6 name("sym_barrierSection") : !ELF.Section
    %11 = ELF.Symbol %arg0 name("Parameter") size(400) : memref<1x100xui32, @DDR>
    %12 = ELF.Symbol %arg1 name("Convert") size(400) : memref<1x100xui32, @DDR>
    %13 = ELF.CreateSymbolTableSection secName(".symtab.input") secFlags(VPU_SHF_USERINPUT) -> !ELF.Section {
      ELF.PutOpInSection %11 : !ELF.Symbol
    }
    %14 = ELF.CreateSymbolTableSection secName(".symtab.output") secFlags(VPU_SHF_USEROUTPUT) -> !ELF.Section {
      ELF.PutOpInSection %12 : !ELF.Symbol
    }
    %15 = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags(SHF_ALLOC) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".data.BuffersIO"} -> !ELF.Section {
      ELF.PutOpInSection %1 : memref<1x100xui32, @DDR>
    }
    %16 = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".data.ConstIO"} -> !ELF.Section {
      ELF.PutOpInSection %cst : memref<1x100xui32>
    }
    // CHECK:  %[[VAL4:.*]] = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_ALLOC|VPU_SHF_PROC_DMA") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".data.ConstIO"} -> !ELF.Section {
    %17 = ELF.Symbol %cst name("sym_const0") : memref<1x100xui32>
    %18 = ELF.Symbol %15 name("sym_bufferSection") : !ELF.Section
    %19 = ELF.CreateSymbolTableSection secName(".symtab.buffers") secFlags("SHF_NONE") -> !ELF.Section {
      ELF.PutOpInSection %18 : !ELF.Symbol
      ELF.PutOpInSection %17 : !ELF.Symbol
    }
    %c0_i8 = arith.constant 0 : i8
    %20 = ELF.Symbol %c0_i8 name("VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR") {isBuiltin} : i8
    %c1_i8 = arith.constant 1 : i8
    %21 = ELF.Symbol %c1_i8 name("VPU_NNRD_SYM_RTM_IVAR") {isBuiltin} : i8
    %c2_i8 = arith.constant 2 : i8
    %22 = ELF.Symbol %c2_i8 name("VPU_NNRD_SYM_RTM_ACT") {isBuiltin} : i8
    %c3_i8 = arith.constant 3 : i8
    %23 = ELF.Symbol %c3_i8 name("VPU_NNRD_SYM_RTM_DMA0") {isBuiltin} : i8
    %c4_i8 = arith.constant 4 : i8
    %24 = ELF.Symbol %c4_i8 name("VPU_NNRD_SYM_RTM_DMA1") {isBuiltin} : i8
    %c5_i8 = arith.constant 5 : i8
    %25 = ELF.Symbol %c5_i8 name("VPU_NNRD_SYM_FIFO_BASE") {isBuiltin} : i8
    %c6_i8 = arith.constant 6 : i8
    %26 = ELF.Symbol %c6_i8 name("VPU_NNRD_SYM_BARRIERS_START") {isBuiltin} : i8
    %27 = ELF.CreateSymbolTableSection secName("VPU_RT_SYMTAB") secFlags("SHF_NONE") {isBuiltin} -> !ELF.Section {
      ELF.PutOpInSection %20 : !ELF.Symbol
      ELF.PutOpInSection %21 : !ELF.Symbol
      ELF.PutOpInSection %22 : !ELF.Symbol
      ELF.PutOpInSection %23 : !ELF.Symbol
      ELF.PutOpInSection %24 : !ELF.Symbol
      ELF.PutOpInSection %25 : !ELF.Symbol
      ELF.PutOpInSection %26 : !ELF.Symbol
    }
    %28 = ELF.CreateSymbolTableSection secName(".symtab.tasks") secFlags("SHF_NONE") -> !ELF.Section {
      ELF.PutOpInSection %9 : !ELF.Symbol
      ELF.PutOpInSection %10 : !ELF.Symbol
      %38 = ELF.Symbol %4 name("MappedInference_entry") type("VPU_STT_ENTRY") : !VPUIPRegMapped.Index<0>
    }
    %29 = ELF.CreateRelocationSection secName(".rlt.DMA_NetOutput") sourceSymbolTableSection(%14) targetSection(%5) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USEROUTPUT") -> !ELF.Section {
      ELF.RelocImmOffset baseOp(%3 : !VPUIPRegMapped.Index<1>) offset(24) "R_VPU_64" %12 0
    }
    %30 = ELF.CreateRelocationSection secName(".rlt.dmaIO") sourceSymbolTableSection(%19) targetSection(%5) secFlags(SHF_INFO_LINK) -> !ELF.Section {
      ELF.RelocImmOffset baseOp(%2 : !VPUIPRegMapped.Index<0>) offset(16) "R_VPU_64" %17 0
      ELF.RelocImmOffset baseOp(%2 : !VPUIPRegMapped.Index<0>) offset(24) "R_VPU_64" %18 0
      ELF.RelocImmOffset baseOp(%3 : !VPUIPRegMapped.Index<1>) offset(16) "R_VPU_64" %18 0
    }
    %31 = ELF.CreateRelocationSection secName(".rlt.dmaIO_CMX") sourceSymbolTableSection(%27) targetSection(%5) secFlags(SHF_INFO_LINK) -> !ELF.Section {
      ELF.RelocImmOffset baseOp(%2 : !VPUIPRegMapped.Index<0>) offset(0) "R_VPU_32_RTM" %23 192
    }
    %32 = VPURT.DeclareBuffer "DDR" <64> -> memref<262144xi32, @DDR>
    %33 = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags("SHF_NONE") {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".bss.actKernelRtConfigSec"} -> !ELF.Section {
      ELF.PutOpInSection %32 : memref<262144xi32, @DDR>
    }
    %34 = ELF.Symbol %33 name("sym_actKernelRtConfigsSec") : !ELF.Section
    %35 = ELF.CreateSymbolTableSection secName(".symtab.actKernelRtConfig") secFlags("SHF_NONE") -> !ELF.Section {
      ELF.PutOpInSection %34 : !ELF.Symbol
    }
    %36 = ELF.CreateRelocationSection secName(".rlt.MI_AKRtConfig") sourceSymbolTableSection(%35) targetSection(%7) secFlags(SHF_INFO_LINK) -> !ELF.Section {
      ELF.RelocImmOffset baseOp(%4 : !VPUIPRegMapped.Index<0>) offset(828) "R_VPU_32" %34 0
    }
    %37 = ELF.CreateRelocationSection secName(".rlt.MappedInference") sourceSymbolTableSection(%28) targetSection(%7) secFlags(SHF_INFO_LINK) -> !ELF.Section {
      ELF.RelocImmOffset baseOp(%4 : !VPUIPRegMapped.Index<0>) offset(0) "R_VPU_64" %9 0
      ELF.RelocImmOffset baseOp(%4 : !VPUIPRegMapped.Index<0>) offset(72) "R_VPU_64" %10 0
    }
    return %arg1 : memref<1x100xui32, @DDR>
  }
}
