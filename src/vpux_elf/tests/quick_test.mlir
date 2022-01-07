module @Test {

IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "inputCNN" : tensor<1x1x2x1000xf16>
} outputsInfo :  {
    DataInfo "outputCNN" : tensor<1x1x2x1000xf16>
}

func @main(%arg0: memref<1x1x2x1000xf16>, %arg1: memref<1x1x2x1000xf16>) -> memref<1x1x2x1000xf16> {

    %buffer = VPUIPRegMapped.DeclareBuffer "VPU_DDR_Heap" [0] <0> -> memref<1x1x2x1000xf16, "DDR">
    %dma0_buffer = VPUIPRegMapped.DeclareBuffer "VPU_DDR_Heap" [0] <0> -> memref<1x1x2x1000xf16, "DDR">
    %act_out_buffer = VPUIPRegMapped.DeclareBuffer "VPU_DDR_Heap" [0] <0> -> memref<1x1x2x1000xf16, "DDR">

    %barrier0 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <0,-1> -> !VPURT.Barrier 
    %barrier1 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <1,-1> -> !VPURT.Barrier

    %dma0 = VPUIPRegMapped.NNDMA inputs(%arg0 : memref<1x1x2x1000xf16>)
                                    outputs(%dma0_buffer : memref<1x1x2x1000xf16, "DDR">)
                                    updates(%barrier0 : !VPURT.Barrier)
                                    start_after(0) -> memref<1x1x2x1000xf16, "DDR">

    %kernel_text_index = VPUIPRegMapped.DeclareKernelText kernel_path("kernel_elf_path") -> !VPUIPRegMapped.Index<0> // returns a VPUIP Index

    // the serialization of this op will get relocated in ActKernelInvo
    %kernel_data_index = VPUIPRegMapped.DeclareKernelArgs kernel_path("kernel_elf_path") -> !VPUIPRegMapped.Index<0> // returns a VPUIP Index

    // ^^ these ops will search for the elf and use ELFReader32 to extract the sections

    %kernel_entry_index = VPUIPRegMapped.DeclareKernelEntry kernel_path("kernel_elf_path") -> !VPUIPRegMapped.Index<0> // returns a VPUIP Index


    // the serialization of this op will get relocated in ActKernelInvo
    %kernel_params_index = VPUIPRegMapped.KernelParams
                                                    input(%dma0_buffer : memref<1x1x2x1000xf16, "DDR">)
                                                    output(%act_out_buffer : memref<1x1x2x1000xf16, "DDR">)
                                                    kernel_type("Softmax") // type of kernel - will be known from lowering
                                                    kernel_params(dense<0> : tensor<2xui8>) // this will get serialized / unique structure for each software layer that will hold all the params for the kernel
                                                    -> !VPUIPRegMapped.Index<0>


    // range needs the size of the .text & .data sections + the kernel_entry - declareKernelTextOp & declareKernelDataOp have getBinarySize() method, which gets the size of those sections
    // kernel_entry will be taken at serialization, using ELFReader again
    %range_index = VPUIPRegMapped.ActKernelRange kernel_text_index(%kernel_text_index : !VPUIPRegMapped.Index<0>)
                                                 kernel_args_index(%kernel_data_index : !VPUIPRegMapped.Index<0>)
                                                 kernel_entry_index(%kernel_entry_index : !VPUIPRegMapped.Index<0>)
                                                 -> !VPUIPRegMapped.Index<0>

    %invo_index = VPUIPRegMapped.ActKernelInvocation range_index(%range_index : !VPUIPRegMapped.Index<0>)
                                                     waits(%barrier0 : !VPURT.Barrier)
                                                     updates(%barrier1 : !VPURT.Barrier)
                                                     tile(0) // attribute %tile ui64
                                                     start_after(1) // attribute %start_after ui64
                                                     clean_after(100) // attribute %clean_after ui64
                                                     -> !VPUIPRegMapped.Index<0>


    // KernelParamsWrapper kernel_type("Softmax") %input %output + any other args of the ^bb
    // {
    //     // ^bb0
    //     %axis   = arith.constant 0 : i64


    //     KernelParams(%input, %output, %axis, .. other args ..)  //Variadic <AnyType> args
    //     {

    //     }

    // }
                                           

    // %mappedInference = VPUIPRegMapped.MappedInference
    //                         dmaCount(2)
    //                         invariantCount(0)
    //                         variantCount(0)
    //                         actKernelRangesCount(1)
    //                         actKernelInvocationsCount(1)
    //                         barrierCount(2)



    // act shave relocs are not complete here
    %kernelTextSection = ELF.CreateSection secType("SHT_PROGBITS") secFlags(SHF_EXECINSTR) {secName=".text.kernelText", secInfo = 1, secAddrAlign = 64 } -> !ELF.Section
    {
        ELF.PutOpInSection %kernel_text_index : !VPUIPRegMapped.Index<0>
    }
    
    %kernelDataSection = ELF.CreateSection secType("SHT_PROGBITS") secFlags(SHF_EXECINSTR) {secName=".text.kernelData", secInfo = 1, secAddrAlign = 64 } -> !ELF.Section
    {
        ELF.PutOpInSection %kernel_data_index : !VPUIPRegMapped.Index<0>
    }

    // %kernelParamsSection = ELF.CreateSection secFlags(SHF_EXECINSTR) {secName=".text.kernelParams", secType="SHT_PROGBITS", secInfo = 1, secAddrAlign = 64 } -> !ELF.Section
    // {
    //     ELF.PutOpInSection %kernel_params_index : ui64
    // }
    
    %actKernelRangeSection = ELF.CreateSection secType("SHT_PROGBITS") secFlags(SHF_EXECINSTR) {secName=".text.kernelRange", secInfo = 1, secAddrAlign = 64 } -> !ELF.Section
    {
        ELF.PutOpInSection %range_index : !VPUIPRegMapped.Index<0>
    }

    %actKernelInvoSection = ELF.CreateSection secType("SHT_PROGBITS") secFlags(SHF_EXECINSTR) {secName=".text.kernelInvo", secInfo = 1, secAddrAlign = 64 } -> !ELF.Section
    {
        ELF.PutOpInSection %invo_index : !VPUIPRegMapped.Index<0>
    }
    

    // %sym_for_kernelTextSection = ELF.Symbol %kernelTextSection name("symKernelTextSection") : !ELF.Section
    // %sym_for_kernelDataSection = ELF.Symbol %kernelDataSection name("symKernelDataSection") : !ELF.Section
    // %sym_for_kernelParamsSection = ELF.Symbol %kernelParamsSection name("symKernelParamsSection") : !ELF.Section
    // %sym_for_actKernelRangeSection = ELF.Symbol %actKernelRangeSection name("symActKernelRangeSection") : !ELF.Section
    // %sym_for_actKernelInvoSection = ELF.Symbol %actKernelInvoSection name("symActKernelInvoSection") : !ELF.Section

    // %actKernelSymSection = ELF.CreateSymbolTableSection secName(".actKernelSymTab") secFlags(SHF_NONE) -> !ELF.Section
    // {
    //     ELF.PutOpInSection %sym_for_kernelTextSection : !ELF.Symbol
    //     ELF.PutOpInSection %sym_for_kernelDataSection : !ELF.Symbol
    //     ELF.PutOpInSection %sym_for_kernelParamsSection : !ELF.Symbol
    //     ELF.PutOpInSection %sym_for_actKernelRangeSection : !ELF.Symbol
    //     ELF.PutOpInSection %sym_for_actKernelInvoSection : !ELF.Symbol
    // }

    // %actKernelRangeRelocs = ELF.CreateRelocationSection secName(".RelA.actKernelRange") sourceSymbolTableSection(%actKernelSymSection) targetSection(%actKernelRangeSection) secFlags(SHF_NONE) -> !ELF.Section
    // {
    //     ELF.Reloc ? "R_VPU_64" %sym_for_kernelTextSection 0
    // }

    // %actKernelInvoRelocs = ELF.CreateRelocationSection secName(".RelA.actKernelInvo") sourceSymbolTableSection(%actKernelSymSection) targetSection(%actKernelInvoSection) secFlags(SHF_NONE) -> !ELF.Section
    // {
    //     ELF.Reloc ? "R_VPU_64" %sym_for_kernelDataSection 0
    //     ELF.Reloc ? "R_VPU_64" %sym_for_kernelParamsSection 0
    // }

    // %mappedInfSec = ELF.CreateSection secFlags(SHF_EXECINSTR) {secName=".text.mappedInference", secType="SHT_PROGBITS", secInfo = 1, secAddrAlign = 64} -> !ELF.Section
    // {
    //     ELF.PutOpInSection %mappedInference : index
    // }

    // %sym_for_mappedInfSec = ELF.Symbol %mappedInfSec name("symMappedInfSec") : !ELF.Section

    // %symArg0 = ELF.Symbol %arg0 name("inputCNN") size(2000) : memref<1x1x2x1000xf16>
    // %symArg1 = ELF.Symbol %arg1 name("outputCNN") size(2000) : memref<1x1x2x1000xf16>

    // %genericSymSection = ELF.CreateSymbolTableSection secName(".symTab") secFlags(SHF_NONE) -> !ELF.Section
    // {
    //     ELF.PutOpInSection %sym_for_dmaSection : !ELF.Symbol
    //     ELF.PutOpInSection %sym_for_barrierSection : !ELF.Symbol
    //     ELF.PutOpInSection %sym_for_mappedInfSec : !ELF.Symbol
    //     ELF.PutOpInSection %sym_for_scratchSection : !ELF.Symbol

    //     ELF.Symbol %mappedInference name("MappedInference") type("VPU_STT_ENTRY") : index
    // }

    // %inputSymSection = ELF.CreateSymbolTableSection secName(".symTab.inputs") secFlags(VPU_SHF_USERINPUT) -> !ELF.Section
    // {
    //     ELF.PutOpInSection %symArg0 : !ELF.Symbol
    // }
    // %outputSymSection = ELF.CreateSymbolTableSection secName(".symTab.outputs") secFlags(VPU_SHF_USEROUTPUT) -> !ELF.Section
    // {
    //     ELF.PutOpInSection %symArg1 : !ELF.Symbol
    // }

    // %mappedInferenceRelocs = ELF.CreateRelocationSection secName(".RelA.mappedInference") sourceSymbolTableSection(%genericSymSection) targetSection(%mappedInfSec) secFlags(SHF_NONE) -> !ELF.Section
    // {
    //     ELF.Reloc 0 "R_VPU_64" %sym_for_dmaSection 0
    //     ELF.Reloc 72 "R_VPU_64" %sym_for_barrierSection 0
    // }

    // %dmaRelocs = ELF.CreateRelocationSection secName(".RelA.dmaRelocations") sourceSymbolTableSection(%genericSymSection) targetSection(%dmaSection) secFlags(SHF_NONE) -> !ELF.Section
    // {
    //     ELF.Reloc 24 "R_VPU_64" %sym_for_scratchSection 0
    //     ELF.Reloc 208 "R_VPU_64" %sym_for_scratchSection 0
    // }

    // %inputRelocs = ELF.CreateRelocationSection secName(".RelA.inputs") sourceSymbolTableSection(%inputSymSection) targetSection(%dmaSection) secFlags("VPU_SHF_JIT|VPU_SHF_USERINPUT") -> !ELF.Section
    // {
    //     ELF.Reloc 16 "R_VPU_64" %symArg0 0
    // }

    // %outputRelocs = ELF.CreateRelocationSection secName(".RelA.outputs") sourceSymbolTableSection(%outputSymSection) targetSection(%dmaSection) secFlags("VPU_SHF_JIT|VPU_SHF_USEROUTPUT") -> !ELF.Section
    // {
    //     ELF.Reloc 216 "R_VPU_64" %symArg1 0
    // }

    return %arg1 : memref<1x1x2x1000xf16>
}
}
