module @Test {

IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "inputCNN" : tensor<1x1x2x1000xf16>
} outputsInfo :  {
    DataInfo "outputCNN" : tensor<1x1x2x1000xf16>
}

func @main(%arg0: memref<1x1x2x1000xf16>, %arg1: memref<1x1x2x1000xf16>) -> memref<1x1x2x1000xf16> {

    %buffer = VPUIPRegMapped.DeclareBuffer "VPU_DDR_Heap" [0] <0> -> memref<1x1x2x1000xf16, "DDR">
    %barrier0 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <0,-1> -> !VPURT.Barrier

    %dma0 = VPUIPRegMapped.NNDMA inputs(%arg0 : memref<1x1x2x1000xf16>)
                                    outputs(%buffer : memref<1x1x2x1000xf16, "DDR">)
                                    updates(%barrier0 : !VPURT.Barrier)
                                    start_after(0) -> memref<1x1x2x1000xf16, "DDR">
    %dma1 = VPUIPRegMapped.NNDMA inputs(%buffer : memref<1x1x2x1000xf16, "DDR">)
                                    outputs(%arg1 : memref<1x1x2x1000xf16>)
                                    waits(%barrier0 : !VPURT.Barrier)
                                    start_after(0) -> memref<1x1x2x1000xf16>

    %mappedInference = VPUIPRegMapped.MappedInference
                            dmas(%dma0 : memref<1x1x2x1000xf16, "DDR">)
                            barriers(%barrier0 : !VPURT.Barrier)
                            dmaCount(2)
                            invariantCount(0)
                            variantCount(0)
                            actInvocationsCount(0)
                            barrierCount(1)

    %dmaSection = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secName=".text.dmaTasks", secInfo = 1, secAddrAlign = 64 } -> !ELF.Section
    {
        ELF.PutOpInSection %dma0 : memref<1x1x2x1000xf16, "DDR">
        ELF.PutOpInSection %dma1 : memref<1x1x2x1000xf16>
    }

    %barriersSection = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secName = ".text.barriers", secInfo = 1, secAddrAlign = 64} -> !ELF.Section
    {
        ELF.PutOpInSection %barrier0 : !VPURT.Barrier
    }

    %mappedInfSec = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secName=".text.mappedInference", secInfo = 1, secAddrAlign = 64} -> !ELF.Section
    {
        ELF.PutOpInSection %mappedInference : index
    }

    %scratchSection = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags(SHF_NONE) {secName=".bss.ddrScratch", secInfo = 1, secAddrAlign = 64} -> !ELF.Section
    {
        ELF.PutOpInSection %buffer : memref<1x1x2x1000xf16, "DDR">
    }

    %sym_for_dmaSection = ELF.Symbol %dmaSection name("symDmaSection") : !ELF.Section
    %sym_for_barrierSection = ELF.Symbol %barriersSection name("symBarriersSection") : !ELF.Section
    %sym_for_mappedInfSec = ELF.Symbol %mappedInfSec name("symMappedInfSec") : !ELF.Section
    %sym_for_scratchSection = ELF.Symbol %scratchSection name("symScratchSection") : !ELF.Section

    %symArg0 = ELF.Symbol %arg0 name("inputCNN") size(2000) : memref<1x1x2x1000xf16>
    %symArg1 = ELF.Symbol %arg1 name("outputCNN") size(2000) : memref<1x1x2x1000xf16>

    %genericSymSection = ELF.CreateSymbolTableSection secName(".symtab") secFlags(SHF_NONE) -> !ELF.Section
    {
        ELF.PutOpInSection %sym_for_dmaSection : !ELF.Symbol
        ELF.PutOpInSection %sym_for_barrierSection : !ELF.Symbol
        ELF.PutOpInSection %sym_for_mappedInfSec : !ELF.Symbol
        ELF.PutOpInSection %sym_for_scratchSection : !ELF.Symbol

        ELF.Symbol %mappedInference name("MappedInference") type("VPU_STT_ENTRY") : index
    }

    %inputSymSection = ELF.CreateSymbolTableSection secName(".symtab.inputs") secFlags(VPU_SHF_USERINPUT) -> !ELF.Section
    {
        ELF.PutOpInSection %symArg0 : !ELF.Symbol
    }
    %outputSymSection = ELF.CreateSymbolTableSection secName(".symtab.outputs") secFlags(VPU_SHF_USEROUTPUT) -> !ELF.Section
    {
        ELF.PutOpInSection %symArg1 : !ELF.Symbol
    }

    %mappedInferenceRelocs = ELF.CreateRelocationSection secName(".rlt.mappedInference") sourceSymbolTableSection(%genericSymSection) targetSection(%mappedInfSec) secFlags(SHF_NONE) -> !ELF.Section
    {
        ELF.Reloc 0 "R_VPU_64" %sym_for_dmaSection 0
        ELF.Reloc 72 "R_VPU_64" %sym_for_barrierSection 0
    }

    %dmaRelocs = ELF.CreateRelocationSection secName(".rlt.dmaRelocations") sourceSymbolTableSection(%genericSymSection) targetSection(%dmaSection) secFlags(SHF_NONE) -> !ELF.Section
    {
        ELF.Reloc 24 "R_VPU_64" %sym_for_scratchSection 0
        ELF.Reloc 208 "R_VPU_64" %sym_for_scratchSection 0
    }

    %inputRelocs = ELF.CreateRelocationSection secName(".rlt.inputs") sourceSymbolTableSection(%inputSymSection) targetSection(%dmaSection) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USERINPUT") -> !ELF.Section
    {
        ELF.Reloc 16 "R_VPU_64" %symArg0 0
    }

    %outputRelocs = ELF.CreateRelocationSection secName(".rlt.outputs") sourceSymbolTableSection(%outputSymSection) targetSection(%dmaSection) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USEROUTPUT") -> !ELF.Section
    {
        ELF.Reloc 216 "R_VPU_64" %symArg1 0
    }

    return %arg1 : memref<1x1x2x1000xf16>
}
}
