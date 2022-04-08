//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
module @Test attributes {VPU.arch = "VPUX37XX", VPU.compilationMode = "ReferenceHW"}  {
  IE.MemoryResource 31457280 bytes of @DDR {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64}
  IE.MemoryResource 2097152 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
  IE.ExecutorResource 1 of @DMA_NN
  IE.ExecutorResource 1 of @SHAVE_UPA
  IE.ExecutorResource 1 of @SHAVE_ACT
  IE.ExecutorResource 1 of @NCE  {
    IE.ExecutorResource 1 of @DPU
  }

IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "inputCNN" : tensor<1x1x1x1000xf16>
} outputsInfo :  {
    DataInfo "outputCNN" : tensor<1x1x1x1000xf16>
}

func @main(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {

    %dma0 = VPUIPRegMapped.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16>) outputs(%arg1 : memref<1x1x1x1000xf16>) start_after(0) -> !VPUIPRegMapped.Index<0>
    %mappedInference = VPUIPRegMapped.MappedInference
                            dmas(%dma0 : !VPUIPRegMapped.Index<0>) // we can verify with the MappedInference Verifier: if we have dmaCount(>1) we need to have dmaTasks() (reverse applies also), else error; the same for the other ...
                            dmaCount(1)
                            invariantCount(0)
                            variantCount(0)
                            actKernelRangesCount(0)
                            actKernelInvocationsCount(0)
                            barrierCount(0)
                            -> !VPUIPRegMapped.Index<0>

    %dmaSection = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secName=".text.dmaTasks", secInfo = 0, secAddrAlign = 64 } -> !ELF.Section
    {
        ELF.PutOpInSection %dma0 : !VPUIPRegMapped.Index<0>
    }

    %mappedInfSec = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secName=".text.mappedInference", secInfo = 0, secAddrAlign = 64} -> !ELF.Section
    {
        ELF.PutOpInSection %mappedInference : !VPUIPRegMapped.Index<0>
    }

    %metadataSec = ELF.CreateMetadataSection secFlags(SHF_EXECINSTR) {secName=".text.networkMetadata", secInfo = 0, secAddrAlign = 64} -> !ELF.Section
    {
        %metadata = VPUIPRegMapped.NetworkMetadata -> !VPUIPRegMapped.Index<0>
    }

    %sym_for_dmaSection = ELF.Symbol %dmaSection name("symDmaSection") : !ELF.Section
    %sym_for_mappedInfSec = ELF.Symbol %mappedInfSec name("symMappedInfSec") : !ELF.Section

    %symArg0 = ELF.Symbol %arg0 name("inputCNN") size(2000) : memref<1x1x1x1000xf16>
    %symArg1 = ELF.Symbol %arg1 name("outputCNN") size(2000) : memref<1x1x1x1000xf16>

    %genericSymSection = ELF.CreateSymbolTableSection secName(".symtab") secFlags(SHF_NONE) -> !ELF.Section
    {
        ELF.PutOpInSection %sym_for_dmaSection : !ELF.Symbol
        ELF.PutOpInSection %sym_for_mappedInfSec : !ELF.Symbol
        ELF.Symbol %mappedInference name("MappedInference") type("VPU_STT_ENTRY") : !VPUIPRegMapped.Index<0>
    }

    %inputSymSection = ELF.CreateSymbolTableSection secName(".symtab.inputs") secFlags(VPU_SHF_USERINPUT) -> !ELF.Section
    {
        ELF.PutOpInSection %symArg0 : !ELF.Symbol
    }
    %outputSymSection = ELF.CreateSymbolTableSection secName(".symtab.outputs") secFlags(VPU_SHF_USEROUTPUT) -> !ELF.Section
    {
        ELF.PutOpInSection %symArg1 : !ELF.Symbol
    }

    %mappedInferenceRelocs = ELF.CreateRelocationSection secName(".rlt.mappedInference") sourceSymbolTableSection(%genericSymSection) targetSection(%mappedInfSec) secFlags(SHF_INFO_LINK) -> !ELF.Section
    {
        ELF.Reloc baseOp(%mappedInference : !VPUIPRegMapped.Index<0>) offsetOf(%dma0 : !VPUIPRegMapped.Index<0>) "R_VPU_64" %sym_for_dmaSection 0
    }

    %inputRelocs = ELF.CreateRelocationSection secName(".rlt.inputs") sourceSymbolTableSection(%inputSymSection) targetSection(%dmaSection) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USERINPUT") -> !ELF.Section
    {
        ELF.Reloc baseOp(%dma0 : !VPUIPRegMapped.Index<0>) offsetOf(%arg0 : memref<1x1x1x1000xf16>) "R_VPU_64" %symArg0 0
    }

    %outputRelocs = ELF.CreateRelocationSection secName(".rlt.outputs") sourceSymbolTableSection(%outputSymSection) targetSection(%dmaSection) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USEROUTPUT") -> !ELF.Section
    {
        ELF.Reloc baseOp(%dma0 : !VPUIPRegMapped.Index<0>) offsetOf(%arg1 : memref<1x1x1x1000xf16>) "R_VPU_64" %symArg1 0
    }

    return %arg1 : memref<1x1x1x1000xf16>
}
}
