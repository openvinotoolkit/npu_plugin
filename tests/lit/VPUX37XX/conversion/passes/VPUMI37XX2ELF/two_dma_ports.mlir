//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --convert-VPUMI37XX-to-ELF %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {
  IE.CNNNetwork entryPoint : @race_condition_dma_f16_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x16x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x16x16xf16>
    DataInfo "output_1" : tensor<1x16x16x16xf16>
  }
  func.func private @race_condition_dma_f16_f16(%arg0: memref<1x16x16x16xf16, #NHWC, @DDR>, %arg1: memref<1x16x16x16xf16, #NHWC, @DDR>, %arg2: memref<1x16x16x16xf16, #NHWC, @DDR>) -> (memref<1x16x16x16xf16, #NHWC, @DDR>, memref<1x16x16x16xf16, #NHWC, @DDR>) {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>
    %2 = VPUMI37XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:0>
    %3 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%0 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) updates(%2 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
    %4 = VPUMI37XX.NNDMA {port = 1 : i64} inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%1 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) updates(%2 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
    %5 = VPUMI37XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8}<1, -1> -> !VPURegMapped.Index<0:0:1>
    %6 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%0 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%3 : !VPURegMapped.Index<0:0:0>) waits(%2 : !VPURegMapped.Index<0:0:0>) updates(%5 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>
    %7 = VPUMI37XX.NNDMA {port = 1 : i64} inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%1 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) previousDMA(%4 : !VPURegMapped.Index<0:0:0>) waits(%2 : !VPURegMapped.Index<0:0:0>) updates(%5 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>
    %8 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x16x16x16xf16, #NHWC, @DDR>) previousDMA(%6 : !VPURegMapped.Index<0:0:1>) waits(%5 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:2>
    %9 = VPUMI37XX.NNDMA {port = 1 : i64} inputs(%1 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) outputs(%arg2 : memref<1x16x16x16xf16, #NHWC, @DDR>) previousDMA(%7 : !VPURegMapped.Index<0:0:1>) waits(%5 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:2>
    %10 = VPUMI37XX.MappedInference dmas(%3, %4 : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>) barriers(%2 : !VPURegMapped.Index<0:0:0>) dmaCount([3, 3]) invariantCount(0) variantCount(0) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(2) -> !VPURegMapped.Index<0:0:0>

    // CHECK: %[[dmaSec0:.*]] = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks0"} -> !ELF.Section {
      // CHECK: ELF.PutOpInSection %3 : !VPURegMapped.Index<0:0:0>
      // CHECK: ELF.PutOpInSection %6 : !VPURegMapped.Index<0:0:1>
      // CHECK: ELF.PutOpInSection %8 : !VPURegMapped.Index<0:0:2>

    // CHECK: %[[dmaSec1:.*]] = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks1"} -> !ELF.Section {
      // CHECK: ELF.PutOpInSection %4 : !VPURegMapped.Index<0:0:0>
      // CHECK: ELF.PutOpInSection %7 : !VPURegMapped.Index<0:0:1>
      // CHECK: ELF.PutOpInSection %9 : !VPURegMapped.Index<0:0:2>

    // CHECK: %[[barSec:.*]] = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.BarrierConfigs"} -> !ELF.Section {
      // CHECK: ELF.PutOpInSection %2 : !VPURegMapped.Index<0:0:0>
      // CHECK: ELF.PutOpInSection %5 : !VPURegMapped.Index<0:0:1>

    // CHECK: %[[miSec:.*]] = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.MappedInference"} -> !ELF.Section {
      // CHECK: ELF.PutOpInSection %10 : !VPURegMapped.Index<0:0:0>

    // CHECK: %[[metaSec:.*]] = ELF.CreateMetadataSection secFlags("SHF_NONE") {secAddrAlign = 8 : i64, secInfo = 0 : i64, secName = ".metadata"} -> !ELF.Section {
      // CHECK: %[[netMeta:.*]] = VPUMI37XX.NetworkMetadata -> !VPURegMapped.Index<0:0:0>

    // CHECK: %[[symDmaSec0:.*]] = ELF.Symbol %[[dmaSec0]] name("sym_dmaSection0") : !ELF.Section
    // CHECK: %[[symDmaSec1:.*]] = ELF.Symbol %[[dmaSec1]] name("sym_dmaSection1") : !ELF.Section
    // CHECK: %[[symBarSec:.*]] = ELF.Symbol %[[barSec]] name("sym_barrierSection") : !ELF.Section

    // CHECK: %[[symIn0:.*]] = ELF.Symbol %arg0 name("input_0") size(8192) : memref<1x16x16x16xf16, #NHWC, @DDR>
    // CHECK: %[[symOut0:.*]] = ELF.Symbol %arg1 name("output_0") size(8192) : memref<1x16x16x16xf16, #NHWC, @DDR>
    // CHECK: %[[symOut1:.*]] = ELF.Symbol %arg2 name("output_1") size(8192) : memref<1x16x16x16xf16, #NHWC, @DDR>

    // CHECK: %[[symTabSecIn:.*]] = ELF.CreateSymbolTableSection secName(".symtab.input") secFlags(VPU_SHF_USERINPUT) -> !ELF.Section {
      // CHECK: ELF.PutOpInSection %[[symIn0]] : !ELF.Symbol

    // CHECK: %[[symTabSecOut:.*]] = ELF.CreateSymbolTableSection secName(".symtab.output") secFlags(VPU_SHF_USEROUTPUT) -> !ELF.Section {
      // CHECK: ELF.PutOpInSection %[[symOut0]] : !ELF.Symbol
      // CHECK: ELF.PutOpInSection %[[symOut1]] : !ELF.Symbol

    // CHECK: %[[c0:.*]] = arith.constant 0 : i8
    // CHECK: %[[SYM_NNCMX_SLICE_BASE:.*]] = ELF.Symbol %[[c0]] name("VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR") {isBuiltin} : i8
    // CHECK: %[[c1:.*]] = arith.constant 1 : i8
    // CHECK: %[[SYM_RTM_IVAR:.*]] = ELF.Symbol %[[c1]] name("VPU_NNRD_SYM_RTM_IVAR") {isBuiltin} : i8
    // CHECK: %[[c2:.*]] = arith.constant 2 : i8
    // CHECK: %[[SYM_RTM_ACT:.*]] = ELF.Symbol %[[c2]] name("VPU_NNRD_SYM_RTM_ACT") {isBuiltin} : i8
    // CHECK: %[[c3:.*]] = arith.constant 3 : i8
    // CHECK: %[[SYM_RTM_DMA0:.*]] = ELF.Symbol %[[c3]] name("VPU_NNRD_SYM_RTM_DMA0") {isBuiltin} : i8
    // CHECK: %[[c4:.*]] = arith.constant 4 : i8
    // CHECK: %[[SYM_RTM_DMA1:.*]] = ELF.Symbol %[[c4]] name("VPU_NNRD_SYM_RTM_DMA1") {isBuiltin} : i8
    // CHECK: %[[c5:.*]] = arith.constant 5 : i8
    // CHECK: %[[SYM_FIFO_BASE:.*]] = ELF.Symbol %[[c5]] name("VPU_NNRD_SYM_FIFO_BASE") {isBuiltin} : i8
    // CHECK: %[[c6:.*]] = arith.constant 6 : i8
    // CHECK: %[[SYM_BARRIERS_START:.*]] = ELF.Symbol %[[c6]] name("VPU_NNRD_SYM_BARRIERS_START") {isBuiltin} : i8

    // CHECK: %[[RT_SYMTAB:.*]] = ELF.CreateSymbolTableSection secName("VPU_RT_SYMTAB") secFlags("SHF_NONE") {isBuiltin} -> !ELF.Section {
      // CHECK: ELF.PutOpInSection %[[SYM_NNCMX_SLICE_BASE]] : !ELF.Symbol
      // CHECK: ELF.PutOpInSection %[[SYM_RTM_IVAR]] : !ELF.Symbol
      // CHECK: ELF.PutOpInSection %[[SYM_RTM_ACT]] : !ELF.Symbol
      // CHECK: ELF.PutOpInSection %[[SYM_RTM_DMA0]] : !ELF.Symbol
      // CHECK: ELF.PutOpInSection %[[SYM_RTM_DMA1]] : !ELF.Symbol
      // CHECK: ELF.PutOpInSection %[[SYM_FIFO_BASE]] : !ELF.Symbol
      // CHECK: ELF.PutOpInSection %[[SYM_BARRIERS_START]] : !ELF.Symbol

    // CHECK: %[[symTabTasks:.*]] = ELF.CreateSymbolTableSection secName(".symtab.tasks") secFlags("SHF_NONE") -> !ELF.Section {
      // CHECK: ELF.PutOpInSection %[[symDmaSec0]] : !ELF.Symbol
      // CHECK: ELF.PutOpInSection %[[symDmaSec1]] : !ELF.Symbol
      // CHECK: ELF.PutOpInSection %[[symBarSec]] : !ELF.Symbol
      // CHECK: %[[miSym:.*]] = ELF.Symbol %10 name("MappedInference_entry") type(<VPU_STT_ENTRY>) : !VPURegMapped.Index<0:0:0>

    // CHECK: %[[rltDmaNetIn0:.*]] = ELF.CreateRelocationSection secName(".rlt.DMA_NetInput0") sourceSymbolTableSection(%[[symTabSecIn]]) targetSection(%[[dmaSec0]]) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USERINPUT") -> !ELF.Section {
      // CHECK: ELF.RelocImmOffset baseOp(%3 : !VPURegMapped.Index<0:0:0>) offset(16) <R_VPU_64> %[[symIn0]] 0
      // CHECK: ELF.RelocImmOffset baseOp(%6 : !VPURegMapped.Index<0:0:1>) offset(16) <R_VPU_64> %[[symIn0]] 0

    // CHECK: %[[rltDmaNetOut0:.*]] = ELF.CreateRelocationSection secName(".rlt.DMA_NetOutput0") sourceSymbolTableSection(%[[symTabSecOut]]) targetSection(%[[dmaSec0]]) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USEROUTPUT") -> !ELF.Section {
      // CHECK: ELF.RelocImmOffset baseOp(%8 : !VPURegMapped.Index<0:0:2>) offset(24) <R_VPU_64> %[[symOut0]] 0

    // CHECK: %[[rltDmaTasks0:.*]] = ELF.CreateRelocationSection secName(".rlt.text.dmaTasks0") sourceSymbolTableSection(%[[RT_SYMTAB]]) targetSection(%[[dmaSec0]]) secFlags(SHF_INFO_LINK) -> !ELF.Section {
      // CHECK: ELF.Reloc baseOp(%3 : !VPURegMapped.Index<0:0:0>) offsetOf(%0 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) <R_VPU_64> %[[SYM_NNCMX_SLICE_BASE]] 0
      // CHECK: ELF.RelocImmOffset baseOp(%3 : !VPURegMapped.Index<0:0:0>) offset(0) <R_VPU_32_RTM> %[[SYM_RTM_DMA0]] 128
      // CHECK: ELF.Reloc baseOp(%6 : !VPURegMapped.Index<0:0:1>) offsetOf(%0 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) <R_VPU_64> %[[SYM_NNCMX_SLICE_BASE]] 0
      // CHECK: ELF.RelocImmOffset baseOp(%6 : !VPURegMapped.Index<0:0:1>) offset(0) <R_VPU_32_RTM> %[[SYM_RTM_DMA0]] 128
      // CHECK: ELF.Reloc baseOp(%8 : !VPURegMapped.Index<0:0:2>) offsetOf(%0 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) <R_VPU_64> %[[SYM_NNCMX_SLICE_BASE]] 0

    // CHECK: %[[rltDmaNetIn1:.*]] = ELF.CreateRelocationSection secName(".rlt.DMA_NetInput1") sourceSymbolTableSection(%[[symTabSecIn]]) targetSection(%[[dmaSec1]]) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USERINPUT") -> !ELF.Section {
      // CHECK: ELF.RelocImmOffset baseOp(%4 : !VPURegMapped.Index<0:0:0>) offset(16) <R_VPU_64> %[[symIn0]] 0
      // CHECK: ELF.RelocImmOffset baseOp(%7 : !VPURegMapped.Index<0:0:1>) offset(16) <R_VPU_64> %[[symIn0]] 0

    // CHECK: %[[rltDmaNetOut1:.*]] = ELF.CreateRelocationSection secName(".rlt.DMA_NetOutput1") sourceSymbolTableSection(%[[symTabSecOut]]) targetSection(%[[dmaSec1]]) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USEROUTPUT") -> !ELF.Section {
      // CHECK: ELF.RelocImmOffset baseOp(%9 : !VPURegMapped.Index<0:0:2>) offset(24) <R_VPU_64> %[[symOut1]] 0

    // CHECK: %[[rltDmaTasks1:.*]] = ELF.CreateRelocationSection secName(".rlt.text.dmaTasks1") sourceSymbolTableSection(%[[RT_SYMTAB]]) targetSection(%[[dmaSec1]]) secFlags(SHF_INFO_LINK) -> !ELF.Section {
      // CHECK: ELF.Reloc baseOp(%4 : !VPURegMapped.Index<0:0:0>) offsetOf(%1 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) <R_VPU_64> %[[SYM_NNCMX_SLICE_BASE]] 2097152
      // CHECK: ELF.RelocImmOffset baseOp(%4 : !VPURegMapped.Index<0:0:0>) offset(0) <R_VPU_32_RTM> %[[SYM_RTM_DMA1]] 128
      // CHECK: ELF.Reloc baseOp(%7 : !VPURegMapped.Index<0:0:1>) offsetOf(%1 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) <R_VPU_64> %[[SYM_NNCMX_SLICE_BASE]] 2097152
      // CHECK: ELF.RelocImmOffset baseOp(%7 : !VPURegMapped.Index<0:0:1>) offset(0) <R_VPU_32_RTM> %[[SYM_RTM_DMA1]] 128
      // CHECK: ELF.Reloc baseOp(%9 : !VPURegMapped.Index<0:0:2>) offsetOf(%1 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) <R_VPU_64> %[[SYM_NNCMX_SLICE_BASE]] 2097152

    // CHECK: %[[rltMi:.*]] = ELF.CreateRelocationSection secName(".rlt.text.MappedInference") sourceSymbolTableSection(%[[symTabTasks]]) targetSection(%[[miSec]]) secFlags(SHF_INFO_LINK) -> !ELF.Section {
      // CHECK: ELF.Reloc baseOp(%10 : !VPURegMapped.Index<0:0:0>) offsetOf(%3 : !VPURegMapped.Index<0:0:0>) <R_VPU_64> %[[symDmaSec0]] 0
      // CHECK: ELF.Reloc baseOp(%10 : !VPURegMapped.Index<0:0:0>) offsetOf(%4 : !VPURegMapped.Index<0:0:0>) <R_VPU_64> %[[symDmaSec1]] 0
      // CHECK: ELF.Reloc baseOp(%10 : !VPURegMapped.Index<0:0:0>) offsetOf(%2 : !VPURegMapped.Index<0:0:0>) <R_VPU_64> %[[symBarSec]] 0

    return %arg1, %arg2 : memref<1x16x16x16xf16, #NHWC, @DDR>, memref<1x16x16x16xf16, #NHWC, @DDR>
  }
}
