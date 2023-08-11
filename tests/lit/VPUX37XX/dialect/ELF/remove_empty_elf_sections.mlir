//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --remove-empty-elf-sections  %s | FileCheck %s

module @Test  {

// CHECK-LABEL: @oneDma
  func.func @oneDma(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    %0 = VPUMI37XX.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16>) outputs(%arg1 : memref<1x1x1x1000xf16>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
    %1 = VPUMI37XX.MappedInference dmas(%0 : !VPURegMapped.Index<0:0:0>) dmaCount([1, 0]) invariantCount(0) variantCount(0) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(0) -> !VPURegMapped.Index<0:0:0>
    %2 = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks"} -> !ELF.Section  {
      ELF.PutOpInSection %0 : !VPURegMapped.Index<0:0:0>
    }
    %3 = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.BarrierConfigs"} -> !ELF.Section  {
    }
    %4 = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.KernelText"} -> !ELF.Section  {
    }
    %5 = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.KernelData"} -> !ELF.Section  {
    }
    %6 = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.KernelParams"} -> !ELF.Section  {
    }
    %7 = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.ActKernelRanges"} -> !ELF.Section  {
    }
    %8 = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.ActKernelInvocations"} -> !ELF.Section  {
    }
    %9 = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.MappedInference"} -> !ELF.Section  {
      ELF.PutOpInSection %1 : !VPURegMapped.Index<0:0:0>
    }
    %10 = ELF.CreateMetadataSection secFlags("SHF_NONE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".metadata"} -> !ELF.Section  {
      %49 = VPUMI37XX.NetworkMetadata -> !VPURegMapped.Index<0:0:0>
    }
    %11 = ELF.Symbol %2 name("sym_dmaSection0") : !ELF.Section
    %12 = ELF.Symbol %3 name("sym_barrierSection") : !ELF.Section
    %13 = ELF.Symbol %7 name("sym_actKernelRangeSection") : !ELF.Section
    %14 = ELF.Symbol %8 name("sym_actKernelInvo") : !ELF.Section
    %15 = ELF.Symbol %4 name("sym_kernelTextSection") : !ELF.Section
    %16 = ELF.Symbol %5 name("sym_kernelDataSection") : !ELF.Section
    %17 = ELF.Symbol %6 name("sym_kernelParamsSection") : !ELF.Section
    %18 = ELF.Symbol %arg0 name("inputCNN") size(2000) : memref<1x1x1x1000xf16>
    %19 = ELF.Symbol %arg1 name("outputCNN") size(2000) : memref<1x1x1x1000xf16>
    %20 = ELF.CreateSymbolTableSection secName(".symtab.input") secFlags(VPU_SHF_USERINPUT) -> !ELF.Section  {
      ELF.PutOpInSection %18 : !ELF.Symbol
    }
    %21 = ELF.CreateSymbolTableSection secName(".symtab.output") secFlags(VPU_SHF_USEROUTPUT) -> !ELF.Section  {
      ELF.PutOpInSection %19 : !ELF.Symbol
    }
    %22 = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags(SHF_ALLOC) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".data.BuffersIO"} -> !ELF.Section  {
    }
    %23 = ELF.Symbol %22 name("sym_bufferSection") : !ELF.Section
    %24 = ELF.CreateSymbolTableSection secName(".symtab.buffers") secFlags("SHF_NONE") -> !ELF.Section  {
      ELF.PutOpInSection %23 : !ELF.Symbol
    }
    %c0_i8 = arith.constant 0 : i8
    %25 = ELF.Symbol %c0_i8 name("VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR") {isBuiltin} : i8
    %c1_i8 = arith.constant 1 : i8
    %26 = ELF.Symbol %c1_i8 name("VPU_NNRD_SYM_RTM_IVAR") {isBuiltin} : i8
    %c2_i8 = arith.constant 2 : i8
    %27 = ELF.Symbol %c2_i8 name("VPU_NNRD_SYM_RTM_ACT") {isBuiltin} : i8
    %c3_i8 = arith.constant 3 : i8
    %28 = ELF.Symbol %c3_i8 name("VPU_NNRD_SYM_RTM_DMA0") {isBuiltin} : i8
    %c4_i8 = arith.constant 4 : i8
    %29 = ELF.Symbol %c4_i8 name("VPU_NNRD_SYM_RTM_DMA1") {isBuiltin} : i8
    %c5_i8 = arith.constant 5 : i8
    %30 = ELF.Symbol %c5_i8 name("VPU_NNRD_SYM_FIFO_BASE") {isBuiltin} : i8
    %c6_i8 = arith.constant 6 : i8
    %31 = ELF.Symbol %c6_i8 name("VPU_NNRD_SYM_BARRIERS_START") {isBuiltin} : i8
    %32 = ELF.CreateSymbolTableSection secName("VPU_RT_SYMTAB") secFlags("SHF_NONE") {isBuiltin} -> !ELF.Section  {
      ELF.PutOpInSection %25 : !ELF.Symbol
      ELF.PutOpInSection %26 : !ELF.Symbol
      ELF.PutOpInSection %27 : !ELF.Symbol
      ELF.PutOpInSection %28 : !ELF.Symbol
      ELF.PutOpInSection %29 : !ELF.Symbol
      ELF.PutOpInSection %30 : !ELF.Symbol
      ELF.PutOpInSection %31 : !ELF.Symbol
    }
    %33 = ELF.CreateSymbolTableSection secName(".symtab.tasks") secFlags("SHF_NONE") -> !ELF.Section  {
      ELF.PutOpInSection %11 : !ELF.Symbol
      ELF.PutOpInSection %12 : !ELF.Symbol
      ELF.PutOpInSection %15 : !ELF.Symbol
      ELF.PutOpInSection %16 : !ELF.Symbol
      ELF.PutOpInSection %17 : !ELF.Symbol
      ELF.PutOpInSection %13 : !ELF.Symbol
      ELF.PutOpInSection %14 : !ELF.Symbol
      %49 = ELF.Symbol %1 name("MappedInference_entry") type("VPU_STT_ENTRY") : !VPURegMapped.Index<0:0:0>
    }
    %34 = ELF.CreateRelocationSection secName(".rlt.DMA_NetInput") sourceSymbolTableSection(%20) targetSection(%2) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USERINPUT") -> !ELF.Section  {
      ELF.RelocImmOffset baseOp(%0 : !VPURegMapped.Index<0:0:0>) offset(16) "R_VPU_64" %18 0
    }
    %35 = ELF.CreateRelocationSection secName(".rlt.DMA_NetOutput") sourceSymbolTableSection(%21) targetSection(%2) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USEROUTPUT") -> !ELF.Section  {
      ELF.RelocImmOffset baseOp(%0 : !VPURegMapped.Index<0:0:0>) offset(24) "R_VPU_64" %19 0
    }
    %36 = ELF.CreateRelocationSection secName(".rlt.dmaIO_DDR") sourceSymbolTableSection(%24) targetSection(%2) secFlags(SHF_INFO_LINK) -> !ELF.Section  {
    }
    %37 = ELF.CreateRelocationSection secName(".rlt.dmaIO_CMX") sourceSymbolTableSection(%32) targetSection(%2) secFlags(SHF_INFO_LINK) -> !ELF.Section  {
    }
    %38 = ELF.CreateRelocationSection secName(".rlt.KernelParams") sourceSymbolTableSection(%33) targetSection(%6) secFlags(SHF_INFO_LINK) -> !ELF.Section  {
    }
    %39 = ELF.CreateRelocationSection secName(".rlt.KernelParamsIO_DDR") sourceSymbolTableSection(%24) targetSection(%6) secFlags(SHF_INFO_LINK) -> !ELF.Section  {
    }
    %40 = ELF.CreateRelocationSection secName(".rlt.KernelParamsIO_CMX") sourceSymbolTableSection(%32) targetSection(%6) secFlags(SHF_INFO_LINK) -> !ELF.Section  {
    }
    %41 = ELF.CreateRelocationSection secName(".rlt.ActKernelRange") sourceSymbolTableSection(%33) targetSection(%7) secFlags(SHF_INFO_LINK) -> !ELF.Section  {
    }
    %42 = ELF.CreateRelocationSection secName(".rlt.ActKernelInvo") sourceSymbolTableSection(%33) targetSection(%8) secFlags(SHF_INFO_LINK) -> !ELF.Section  {
    }
    %43 = VPURT.DeclareBuffer "DDR" <64> -> memref<262144xi32, @DDR>
    %44 = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags("SHF_NONE") {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".bss.actKernelRtConfigSec"} -> !ELF.Section  {
      ELF.PutOpInSection %43 : memref<262144xi32, @DDR>
    }
    %45 = ELF.Symbol %44 name("sym_actKernelRtConfigsSec") : !ELF.Section
    %46 = ELF.CreateSymbolTableSection secName(".symtab.actKernelRtConfig") secFlags("SHF_NONE") -> !ELF.Section  {
      ELF.PutOpInSection %45 : !ELF.Symbol
    }
    %47 = ELF.CreateRelocationSection secName(".rlt.MI_AKRtConfig") sourceSymbolTableSection(%46) targetSection(%9) secFlags(SHF_INFO_LINK) -> !ELF.Section  {
      ELF.RelocImmOffset baseOp(%1 : !VPURegMapped.Index<0:0:0>) offset(828) "R_VPU_32" %45 0
    }
    %48 = ELF.CreateRelocationSection secName(".rlt.MappedInference") sourceSymbolTableSection(%33) targetSection(%9) secFlags(SHF_INFO_LINK) -> !ELF.Section  {
      ELF.RelocImmOffset baseOp(%1 : !VPURegMapped.Index<0:0:0>) offset(0) "R_VPU_64" %11 0
    }
    return %arg1 : memref<1x1x1x1000xf16>
  }

    // CHECK:      [[DMA:%.+]] =  VPUMI37XX.NNDMA
    // CHECK:      [[MPI:%.+]] =  VPUMI37XX.MappedInference
    // CHECK:      [[DMASEC:%.+]] = ELF.CreateSection
    // CHECK-NOT:  ELF.CreateSection
    // CHECK-NOT:  ELF.CreateSection
    // CHECK-NOT:  ELF.CreateSection
    // CHECK-NOT:  ELF.CreateSection
    // CHECK-NOT:  ELF.CreateSection
    // CHECK-NOT:  ELF.CreateSection
    // CHECK:      ELF.CreateSection
    // CHECK:      ELF.CreateMetadataSection
    // CHECK:      ELF.Symbol [[DMASEC]]
    // CHECK-NOT:  ELF.Symbol
    // CHECK-NOT:  ELF.Symbol
    // CHECK-NOT:  ELF.Symbol
    // CHECK-NOT:  ELF.Symbol
    // CHECK-NOT:  ELF.Symbol
    // CHECK-NOT:  ELF.Symbol
    // CHECK-NEXT:      [[SYMARG0:%.+]] = ELF.Symbol [[ARG0:%.+]]
    // CHECK-NEXT:      [[SYMARG1:%.+]] = ELF.Symbol [[ARG1:%.+]]
    // CHECK-NEXT:      ELF.CreateSymbolTableSection
    // CHECK-NEXT: ELF.PutOpInSection [[SYMARG0]] : !ELF.Symbol
    // CHECK:      ELF.CreateSymbolTableSection
    // CHECK-NEXT: ELF.PutOpInSection [[SYMARG1]] : !ELF.Symbol
    // CHECK-NOT:  ELF.CreateLogicalSection
    // CHECK-NOT:  ELF.Symbol
    // CHECK-NOT:  ELF.CreateSymbolTableSection
    // CHECK:      arith.constant 0 : i8
    // CHECK:      ELF.CreateRelocationSection
    // CHECK:      ELF.CreateRelocationSection
    // CHECK-NOT:  ELF.CreateRelocationSection
    // CHECK-NOT:  ELF.CreateRelocationSection
    // CHECK-NOT:  ELF.CreateRelocationSection
    // CHECK-NOT:  ELF.CreateRelocationSection
    // CHECK-NOT:  ELF.CreateRelocationSection
    // CHECK-NOT:  ELF.CreateRelocationSection
    // CHECK-NOT:  ELF.CreateRelocationSection
    // CHECK:      VPURT.DeclareBuffer

 }

// -----
