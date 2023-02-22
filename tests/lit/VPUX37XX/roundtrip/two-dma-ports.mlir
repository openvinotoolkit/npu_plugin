//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --init-compiler="vpu-arch=VPUX37XX" --convert-VPUIPRegMapped-to-ELF %s -o t_dma.mlir
// RUN: vpux-translate --export-ELF t_dma.mlir -o t_dma.elf 
// RUN: vpux-translate --import-ELF t_dma.elf | FileCheck %s
// RUN: rm t_dma.elf t_dma.mlir
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {
  IE.CNNNetwork entryPoint : @race_condition_dma_f16_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x16x16x16xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x16x16xf16, {order = #NHWC}>
    DataInfo "output_1" : tensor<1x16x16x16xf16, {order = #NHWC}>
  }
  func private @race_condition_dma_f16_f16(%arg0: memref<1x16x16x16xf16, #NHWC, @DDR>, %arg1: memref<1x16x16x16xf16, #NHWC, @DDR>, %arg2: memref<1x16x16x16xf16, #NHWC, @DDR>) -> (memref<1x16x16x16xf16, #NHWC, @DDR>, memref<1x16x16x16xf16, #NHWC, @DDR>) {
    %0 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>
    %2 = VPUIPRegMapped.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8}<0, -1> -> !VPUIPRegMapped.Index<0>
    %3 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%0 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) updates(%2 : !VPUIPRegMapped.Index<0>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<0>
    %4 = VPUIPRegMapped.NNDMA {port = 1 : i64} inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%1 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) updates(%2 : !VPUIPRegMapped.Index<0>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<0>
    %5 = VPUIPRegMapped.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8}<1, -1> -> !VPUIPRegMapped.Index<1>
    %6 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%0 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%3 : !VPUIPRegMapped.Index<0>) waits(%2 : !VPUIPRegMapped.Index<0>) updates(%5 : !VPUIPRegMapped.Index<1>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<1>
    %7 = VPUIPRegMapped.NNDMA {port = 1 : i64} inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%1 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) previousDMA(%4 : !VPUIPRegMapped.Index<0>) waits(%2 : !VPUIPRegMapped.Index<0>) updates(%5 : !VPUIPRegMapped.Index<1>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<1>
    %8 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%0 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x16x16x16xf16, #NHWC, @DDR>) previousDMA(%6 : !VPUIPRegMapped.Index<1>) waits(%5 : !VPUIPRegMapped.Index<1>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<2>
    %9 = VPUIPRegMapped.NNDMA {port = 1 : i64} inputs(%1 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) outputs(%arg2 : memref<1x16x16x16xf16, #NHWC, @DDR>) previousDMA(%7 : !VPUIPRegMapped.Index<1>) waits(%5 : !VPUIPRegMapped.Index<1>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<2>
    %10 = VPUIPRegMapped.MappedInference dmas(%3, %4 : !VPUIPRegMapped.Index<0>, !VPUIPRegMapped.Index<0>) barriers(%2 : !VPUIPRegMapped.Index<0>) dmaCount([3, 3]) invariantCount(0) variantCount(0) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(2) -> !VPUIPRegMapped.Index<0>

    return %arg1, %arg2 : memref<1x16x16x16xf16, #NHWC, @DDR>, memref<1x16x16x16xf16, #NHWC, @DDR>
  }
}

//CHECK: %[[VAL0:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8}<0, -1> -> !VPUIPRegMapped.Index<0>
//CHECK: %[[VAL1:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8}<1, -1> -> !VPUIPRegMapped.Index<1>
//CHECK-DAG: %[[VAL2:.*]] = ELF.CreateSection {{.*}} secName = ".text.BarrierConfigs"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL0]] : !VPUIPRegMapped.Index<0>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL1]] : !VPUIPRegMapped.Index<1>

//CHECK: %[[VAL3:.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<8192xui8, [@CMX_NN, 0]>
//CHECK: %[[VAL4:.*]] = VPUIPRegMapped.NNDMA {dma_descriptor = {dstPlaneStride = 0 : i64, dstStride = 8192 : i64, dstWidth = 8192 : i64, len = 8192 : i64, numPlanes = 0 : i64, srcPlaneStride = 0 : i64, srcStride = 8192 : i64, srcWidth = 8192 : i64}, is_critical, is_out_of_order, port = 0 : si64} inputs(%[[VALINPUT0:.*]] : {{.*}}) outputs(%[[VAL3]] : memref<8192xui8, [@CMX_NN, 0]>) updates(%[[VAL0]] : !VPUIPRegMapped.Index<0>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<0>
//CHECK: %[[VAL5:.*]] = VPUIPRegMapped.NNDMA {dma_descriptor = {dstPlaneStride = 0 : i64, dstStride = 8192 : i64, dstWidth = 8192 : i64, len = 8192 : i64, numPlanes = 0 : i64, srcPlaneStride = 0 : i64, srcStride = 8192 : i64, srcWidth = 8192 : i64}, is_critical, is_out_of_order, port = 0 : si64} inputs(%[[VALINPUT0]] : {{.*}}) outputs(%[[VAL3]] : memref<8192xui8, [@CMX_NN, 0]>) previousDMA(%[[VAL4]] : !VPUIPRegMapped.Index<0>) waits(%[[VAL0]] : !VPUIPRegMapped.Index<0>) updates(%[[VAL1]] : !VPUIPRegMapped.Index<1>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<1>
//CHECK: %[[VAL6:.*]] = VPUIPRegMapped.NNDMA {dma_descriptor = {dstPlaneStride = 0 : i64, dstStride = 8192 : i64, dstWidth = 8192 : i64, len = 8192 : i64, numPlanes = 0 : i64, srcPlaneStride = 0 : i64, srcStride = 8192 : i64, srcWidth = 8192 : i64}, is_critical, is_out_of_order, port = 0 : si64} inputs(%3 : memref<8192xui8, [@CMX_NN, 0]>) outputs(%[[VALOUTPUT0:.*]] : {{.*}}) previousDMA(%[[VAL5]] : !VPUIPRegMapped.Index<1>) waits(%1 : !VPUIPRegMapped.Index<1>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<2>

//CHECK: %[[VAL7:.*]] = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks0"} -> !ELF.Section {
//CHECK-NEXT: ELF.PutOpInSection %[[VAL4]] : !VPUIPRegMapped.Index<0>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL5]] : !VPUIPRegMapped.Index<1>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL6]] : !VPUIPRegMapped.Index<2>

//CHECK: %[[VAL8:.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <2097152> -> memref<8192xui8, [@CMX_NN, 0]>
//CHECK: %[[VAL9:.*]] = VPUIPRegMapped.NNDMA {dma_descriptor = {dstPlaneStride = 0 : i64, dstStride = 8192 : i64, dstWidth = 8192 : i64, len = 8192 : i64, numPlanes = 0 : i64, srcPlaneStride = 0 : i64, srcStride = 8192 : i64, srcWidth = 8192 : i64}, is_critical, is_out_of_order, port = 0 : si64} inputs(%[[VALOUTPUT0]] : {{.*}}) outputs(%[[VAL8]] : memref<8192xui8, [@CMX_NN, 0]>) updates(%[[VAL0]] : !VPUIPRegMapped.Index<0>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<0>
//CHECK: %[[VAL10:.*]] = VPUIPRegMapped.NNDMA {dma_descriptor = {dstPlaneStride = 0 : i64, dstStride = 8192 : i64, dstWidth = 8192 : i64, len = 8192 : i64, numPlanes = 0 : i64, srcPlaneStride = 0 : i64, srcStride = 8192 : i64, srcWidth = 8192 : i64}, is_critical, is_out_of_order, port = 0 : si64} inputs(%[[VALOUTPUT0]] : {{.*}}) outputs(%[[VAL8]] : memref<8192xui8, [@CMX_NN, 0]>) previousDMA(%[[VAL9]] : !VPUIPRegMapped.Index<0>) waits(%[[VAL0]] : !VPUIPRegMapped.Index<0>) updates(%[[VAL1]] : !VPUIPRegMapped.Index<1>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<1>
//CHECK: %[[VAL11:.*]] = VPUIPRegMapped.NNDMA {dma_descriptor = {dstPlaneStride = 0 : i64, dstStride = 8192 : i64, dstWidth = 8192 : i64, len = 8192 : i64, numPlanes = 0 : i64, srcPlaneStride = 0 : i64, srcStride = 8192 : i64, srcWidth = 8192 : i64}, is_critical, is_out_of_order, port = 0 : si64} inputs(%[[VAL8]] : memref<8192xui8, [@CMX_NN, 0]>) outputs(%[[VALOUTPUT1:.*]] : {{.*}}) previousDMA(%[[VAL10]] : !VPUIPRegMapped.Index<1>) waits(%[[VAL1]] : !VPUIPRegMapped.Index<1>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<2>

//CHECK: %[[VAL12:.*]] = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks1"} -> !ELF.Section {
//CHECK-NEXT: ELF.PutOpInSection %[[VAL9]] : !VPUIPRegMapped.Index<0>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL10]] : !VPUIPRegMapped.Index<1>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL11]] : !VPUIPRegMapped.Index<2>

//CHECK: %[[VAL13:.*]] = VPUIPRegMapped.MappedInference dmas(%[[VAL4]], %[[VAL9]] : !VPUIPRegMapped.Index<0>, !VPUIPRegMapped.Index<0>) barriers(%[[VAL0]] : !VPUIPRegMapped.Index<0>) dmaCount([3, 3]) invariantCount(0) variantCount(0) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(2) -> !VPUIPRegMapped.Index<0>
//CHECK: %[[VAL14:.*]] = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_NONE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.MappedInference"} -> !ELF.Section {
//CHECK: ELF.PutOpInSection  %[[VAL13]] : !VPUIPRegMapped.Index<0>

//CHECK-DAG: %[[symIn0:.*]] = ELF.Symbol %[[VALINPUT0]] name("input_0") {{.*}}
//CHECK-DAG: %[[symTabSecIn:.*]] = ELF.CreateSymbolTableSection secName(".symtab.input") secFlags(VPU_SHF_USERINPUT) -> !ELF.Section {
//CHECK-DAG: ELF.PutOpInSection %[[symIn0]] : !ELF.Symbol

//CHECK-DAG: %[[symOut0:.*]] = ELF.Symbol %[[VALOUTPUT0]] name("output_0") {{.*}}
//CHECK-DAG: %[[symOut1:.*]] = ELF.Symbol %[[VALOUTPUT1]] name("output_1") {{.*}}
//CHECK-DAG: %[[symTabSecOut:.*]] = ELF.CreateSymbolTableSection secName(".symtab.output") secFlags(VPU_SHF_USEROUTPUT) -> !ELF.Section {
//CHECK-DAG: ELF.PutOpInSection %[[symOut0]] : !ELF.Symbol
//CHECK-DAG: ELF.PutOpInSection %[[symOut1]] : !ELF.Symbol

//CHECK-DAG: %[[symDmaSec0:.*]] = ELF.Symbol %[[VAL7]] name("sym_dmaSection0") {{.*}} : !ELF.Section
//CHECK-DAG: %[[symDmaSec1:.*]] = ELF.Symbol %[[VAL12]] name("sym_dmaSection1") {{.*}} : !ELF.Section
//CHECK-DAG: %[[symBarSec:.*]] = ELF.Symbol %[[VAL2]] name("sym_barrierSection") {{.*}} : !ELF.Section

//CHECK-DAG: %[[miSym:.*]] = ELF.Symbol %[[VAL14]] name("MappedInference_entry") type("VPU_STT_ENTRY") {{.*}} : !ELF.Section
//CHECK-DAG: %[[symTabTasks:.*]] = ELF.CreateSymbolTableSection secName(".symtab.tasks") secFlags("SHF_NONE") -> !ELF.Section {
//CHECK-DAG: ELF.PutOpInSection %[[symDmaSec0]] : !ELF.Symbol
//CHECK-DAG: ELF.PutOpInSection %[[symDmaSec1]] : !ELF.Symbol
//CHECK-DAG: ELF.PutOpInSection %[[symBarSec]] : !ELF.Symbol

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

//CHECK: %[[rltDmaNetIn0:.*]] = ELF.CreateRelocationSection secName(".rlt.DMA_NetInput0") sourceSymbolTableSection(%[[symTabSecIn]]) targetSection(%[[VAL7]]) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USERINPUT") -> !ELF.Section {
//CHECK: ELF.RelocImmOffset offset(16) "R_VPU_64" %[[symIn0]] 0
//CHECK: ELF.RelocImmOffset offset(144) "R_VPU_64" %[[symIn0]] 0

//CHECK: %[[rltDmaNetOut0:.*]] = ELF.CreateRelocationSection secName(".rlt.DMA_NetOutput0") sourceSymbolTableSection(%[[symTabSecOut]]) targetSection(%[[VAL7]]) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USEROUTPUT") -> !ELF.Section {
//CHECK: ELF.RelocImmOffset offset(280) "R_VPU_64" %[[symOut0]] 0

//CHECK: %[[rltDmaTasks0:.*]] = ELF.CreateRelocationSection secName(".rlt.text.dmaTasks0") sourceSymbolTableSection(%[[VAL67]]) targetSection(%[[VAL7]]) secFlags(SHF_INFO_LINK) -> !ELF.Section {
//CHECK: ELF.RelocImmOffset offset(24) "R_VPU_64" %[[VAL60]] 0
//CHECK: ELF.RelocImmOffset offset(0) "R_VPU_32_RTM" %[[VAL63]] 128
//CHECK: ELF.RelocImmOffset offset(152) "R_VPU_64" %[[VAL60]] 0
//CHECK: ELF.RelocImmOffset offset(128) "R_VPU_32_RTM" %[[VAL63]] 128
//CHECK: ELF.RelocImmOffset offset(272) "R_VPU_64" %[[VAL60]] 0

//CHECK: %[[rltDmaNetIn1:.*]] = ELF.CreateRelocationSection secName(".rlt.DMA_NetInput1") sourceSymbolTableSection(%[[symTabSecIn]]) targetSection(%[[VAL12]]) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USERINPUT") -> !ELF.Section {
//CHECK: ELF.RelocImmOffset offset(16) "R_VPU_64" %[[symIn0]] 0
//CHECK: ELF.RelocImmOffset offset(144) "R_VPU_64" %[[symIn0]] 0

//CHECK: %[[rltDmaNetOut1:.*]] = ELF.CreateRelocationSection secName(".rlt.DMA_NetOutput1") sourceSymbolTableSection(%[[symTabSecOut]]) targetSection(%[[VAL12]]) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USEROUTPUT") -> !ELF.Section {
//CHECK: ELF.RelocImmOffset offset(280) "R_VPU_64" %[[symOut1]] 0

//CHECK: %[[rltDmaTasks1:.*]] = ELF.CreateRelocationSection secName(".rlt.text.dmaTasks1") sourceSymbolTableSection(%[[VAL67]]) targetSection(%[[VAL12]]) secFlags(SHF_INFO_LINK) -> !ELF.Section {
//CHECK: ELF.RelocImmOffset offset(24) "R_VPU_64" %[[VAL60]] 2097152
//CHECK: ELF.RelocImmOffset offset(0) "R_VPU_32_RTM" %[[VAL64]] 128
//CHECK: ELF.RelocImmOffset offset(152) "R_VPU_64" %[[VAL60]] 2097152
//CHECK: ELF.RelocImmOffset offset(128) "R_VPU_32_RTM" %[[VAL64]] 128
//CHECK: ELF.RelocImmOffset offset(272) "R_VPU_64" %[[VAL60]] 2097152

//CHECK: %[[rltMi:.*]] = ELF.CreateRelocationSection secName(".rlt.text.MappedInference") sourceSymbolTableSection(%[[symTabTasks]]) targetSection(%[[VAL14]]) secFlags(SHF_INFO_LINK) -> !ELF.Section {
//CHECK: ELF.RelocImmOffset offset(8) "R_VPU_64" %[[symDmaSec0]] 0
//CHECK: ELF.RelocImmOffset offset(24) "R_VPU_64" %[[symDmaSec1]] 0
//CHECK: ELF.RelocImmOffset offset(72) "R_VPU_64" %[[symBarSec]] 0
