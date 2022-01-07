module @Test {

IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "inputCNN" : tensor<2x1000xf16>
} outputsInfo :  {
    DataInfo "outputCNN" : tensor<2x1000xf16>
}

func @main(%arg0: memref<2x1000xf16>, %arg1: memref<2x1000xf16>) -> memref<2x1000xf16> {

    %buffer = VPURT.DeclareBuffer "DDR" <0> -> memref<262144xi32, @DDR>
    %dma0_buffer = VPURT.DeclareBuffer "CMX_NN" <4000> -> memref<2x1000xf16, @CMX_NN>
    %act_out_buffer = VPURT.DeclareBuffer "CMX_NN" <4000> -> memref<2x1000xf16, @CMX_NN>

    %barrier0 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <0,-1> -> !VPURT.Barrier 
    %barrier1 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <1,-1> -> !VPURT.Barrier

    %dims = const.Declare memref<2xsi32> = #const.Content<dense<[2, 1000]> : tensor<2xsi32>>
    %strides = const.Declare memref<3xsi64> = #const.Content<dense<[0, 0, 0]> : tensor<3xsi64>>

    %dma0 = VPUIPRegMapped.NNDMA inputs(%arg0 : memref<2x1000xf16>)
                                    outputs(%dma0_buffer : memref<2x1000xf16, @CMX_NN>)
                                    updates(%barrier0 : !VPURT.Barrier)
                                    start_after(0) -> memref<2x1000xf16, @CMX_NN>

    // edit the paths to reflect the path of the softmax kernel
    %kernel_text_index = VPUIPRegMapped.DeclareKernelText kernel_path("/home/pcarabas/work/vpux-plugin/sw_runtime_kernels/kernels/prebuild/singleShaveSoftmax_3010xx.elf") -> !VPUIPRegMapped.Index<0> // returns a VPUIP Index

    // the serialization of this op will get relocated in ActKernelInvo
    %kernel_data_index = VPUIPRegMapped.DeclareKernelArgs kernel_path("/home/pcarabas/work/vpux-plugin/sw_runtime_kernels/kernels/prebuild/singleShaveSoftmax_3010xx.elf") -> !VPUIPRegMapped.Index<0> // returns a VPUIP Index

    // ^^ the above ops will search for the elf and use ELFReader32 to extract the sections

    %kernel_entry_index = VPUIPRegMapped.DeclareKernelEntry kernel_path("/home/pcarabas/work/vpux-plugin/sw_runtime_kernels/kernels/prebuild/singleShaveSoftmax_3010xx.elf") -> !VPUIPRegMapped.Index<0> // returns a VPUIP Index


    // the serialization of this op will get relocated in ActKernelInvo
    %kernel_params_index = VPUIPRegMapped.KernelParams
                                                    input(%dma0_buffer : memref<2x1000xf16, @CMX_NN>)
                                                    output(%act_out_buffer : memref<2x1000xf16, @CMX_NN>)
                                                    kernel_type("Softmax") // type of kernel - will be known from lowering
                                                    kernel_params(dense<  // dense elements not used atm
                                                    [
                                                        // input
                                                        0x00, 0x00, 0x00, 0x00, // dataAddr
                                                        0x01, 0x00, 0x00, 0x00, // isStatic
                                                        0x04, 0x00, 0x00, 0x00, // numDims
                                                        0x00, 0x00, 0x00, 0x00, // dimsAddr
                                                        0x00, 0x00, 0x00, 0x00, // stridesAddr
                                                        0x00, 0x00, 0x00, 0x00, // dataType
                                                        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // dimsOrder
                                                        0x02, 0x00, 0x00, 0x00, // location

                                                        // output
                                                        0x00, 0x00, 0x00, 0x00, // dataAddr
                                                        0x01, 0x00, 0x00, 0x00, // isStatic
                                                        0x04, 0x00, 0x00, 0x00, // numDims
                                                        0x00, 0x00, 0x00, 0x00, // dimsAddr
                                                        0x00, 0x00, 0x00, 0x00, // stridesAddr
                                                        0x00, 0x00, 0x00, 0x00, // dataType
                                                        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // dimsOrder
                                                        0x02, 0x00, 0x00, 0x00, // location

                                                        0x01, 0x00, 0x00, 0x00 // axis
                                                    ]
                                                    > : vector<76xui8>) // unique structure for each software layer that will hold all the params for the kernel
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
                                                     start_after(0) // attribute %start_after ui64
                                                     clean_after(1) // attribute %clean_after ui64
                                                     -> !VPUIPRegMapped.Index<0>
                                           

    %dma1 = VPUIPRegMapped.NNDMA inputs(%act_out_buffer : memref<2x1000xf16, @CMX_NN>)
                                    outputs(%arg1 : memref<2x1000xf16>)
                                    waits(%barrier1 : !VPURT.Barrier)
                                    start_after(0) -> memref<2x1000xf16>
    
    %mappedInference = VPUIPRegMapped.MappedInference
                            dmaCount(2)
                            invariantCount(0)
                            variantCount(0)
                            actKernelRangesCount(1)
                            actKernelInvocationsCount(1)
                            barrierCount(2)

    // act shave relocs are not complete here
    %ioDimsSection = ELF.CreateSection secType("SHT_PROGBITS") secFlags(SHF_EXECINSTR) {secName=".text.kernelIOdims", secInfo = 1, secAddrAlign = 64 } -> !ELF.Section
    {
        ELF.PutOpInSection %dims : memref<2xsi32>
    }

    %ioStridesSection = ELF.CreateSection secType("SHT_PROGBITS") secFlags(SHF_EXECINSTR) {secName=".text.kernelIOstrides", secInfo = 1, secAddrAlign = 64 } -> !ELF.Section
    {
        ELF.PutOpInSection %strides : memref<3xsi64>
    }

    %kernelTextSection = ELF.CreateSection secType("SHT_PROGBITS") secFlags("SHF_EXECINSTR|SHF_ALLOC") {secName=".text.kernelText", secInfo = 1, secAddrAlign = 64 } -> !ELF.Section
    {
        ELF.PutOpInSection %kernel_text_index : !VPUIPRegMapped.Index<0>
    }
    
    %kernelDataSection = ELF.CreateSection secType("SHT_PROGBITS") secFlags("SHF_EXECINSTR|SHF_ALLOC") {secName=".text.kernelData", secInfo = 1, secAddrAlign = 64 } -> !ELF.Section
    {
        ELF.PutOpInSection %kernel_data_index : !VPUIPRegMapped.Index<0>
    }

    %kernelParamsSection = ELF.CreateSection secType("SHT_PROGBITS") secFlags("SHF_EXECINSTR|SHF_ALLOC") {secName=".text.kernelParams", secInfo = 1, secAddrAlign = 64 } -> !ELF.Section
    {
        ELF.PutOpInSection %kernel_params_index : !VPUIPRegMapped.Index<0>
    }
    
    %actKernelRangeSection = ELF.CreateSection secType("SHT_PROGBITS") secFlags(SHF_EXECINSTR) {secName=".text.kernelRange", secInfo = 1, secAddrAlign = 64 } -> !ELF.Section
    {
        ELF.PutOpInSection %range_index : !VPUIPRegMapped.Index<0>
    }

    %actKernelInvoSection = ELF.CreateSection secType("SHT_PROGBITS") secFlags(SHF_EXECINSTR) {secName=".text.kernelInvo", secInfo = 1, secAddrAlign = 64 } -> !ELF.Section
    {
        ELF.PutOpInSection %invo_index : !VPUIPRegMapped.Index<0>
    }

    %actRtCfgBufferSection = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags(SHF_NONE) {secName=".bss.actRtCfgSec", secInfo = 1, secAddrAlign = 1024} -> !ELF.Section
    {
        ELF.PutOpInSection %buffer : memref<262144xi32, @DDR>
    }
    
    %sym_for_kernelIOdimsSection = ELF.Symbol %ioDimsSection name("symKernelIOdimsSection") : !ELF.Section
    %sym_for_kernelIOstridesSection = ELF.Symbol %ioStridesSection name("symKernelIOstridesSection") : !ELF.Section
    %sym_for_kernelTextSection = ELF.Symbol %kernelTextSection name("symKernelTextSection") : !ELF.Section
    %sym_for_kernelDataSection = ELF.Symbol %kernelDataSection name("symKernelDataSection") : !ELF.Section
    %sym_for_kernelParamsSection = ELF.Symbol %kernelParamsSection name("symKernelParamsSection") : !ELF.Section
    %sym_for_actKernelRangeSection = ELF.Symbol %actKernelRangeSection name("symActKernelRangeSection") : !ELF.Section
    %sym_for_actKernelInvoSection = ELF.Symbol %actKernelInvoSection name("symActKernelInvoSection") : !ELF.Section
    %sym_for_actRtCfgBufferSection = ELF.Symbol %actRtCfgBufferSection name("symActRtCfgBufferSection") : !ELF.Section


    %actKernelSymSection = ELF.CreateSymbolTableSection secName(".actKernelSymTab") secFlags(SHF_NONE) -> !ELF.Section
    {
        ELF.PutOpInSection %sym_for_kernelIOdimsSection : !ELF.Symbol
        ELF.PutOpInSection %sym_for_kernelIOstridesSection : !ELF.Symbol
        ELF.PutOpInSection %sym_for_kernelTextSection : !ELF.Symbol
        ELF.PutOpInSection %sym_for_kernelDataSection : !ELF.Symbol
        ELF.PutOpInSection %sym_for_kernelParamsSection : !ELF.Symbol
        ELF.PutOpInSection %sym_for_actKernelRangeSection : !ELF.Symbol
        ELF.PutOpInSection %sym_for_actKernelInvoSection : !ELF.Symbol
        ELF.PutOpInSection %sym_for_actRtCfgBufferSection : !ELF.Symbol
    }

    %dmaSection = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secName=".text.dmaTasks", secInfo = 1, secAddrAlign = 64 } -> !ELF.Section
    {
        ELF.PutOpInSection %dma0 : memref<2x1000xf16, @CMX_NN>
        ELF.PutOpInSection %dma1 : memref<2x1000xf16>
    }

    %barriersSection = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secName = ".text.barriers", secInfo = 1, secAddrAlign = 64} -> !ELF.Section
    {
        ELF.PutOpInSection %barrier0 : !VPURT.Barrier
        ELF.PutOpInSection %barrier1 : !VPURT.Barrier
    }

    %mappedInfSec = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secName=".text.mappedInference", secInfo = 1, secAddrAlign = 64} -> !ELF.Section
    {
        ELF.PutOpInSection %mappedInference : index
    }

    %scratchSection = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags(SHF_NONE) {secName=".bss.ddrScratch", secInfo = 1, secAddrAlign = 64} -> !ELF.Section
    {
        ELF.PutOpInSection %dma0_buffer : memref<2x1000xf16, @CMX_NN>
        ELF.PutOpInSection %act_out_buffer : memref<2x1000xf16, @CMX_NN>
    }

    %sym_for_dmaSection = ELF.Symbol %dmaSection name("symDmaSection") : !ELF.Section
    %sym_for_barrierSection = ELF.Symbol %barriersSection name("symBarriersSection") : !ELF.Section
    %sym_for_mappedInfSec = ELF.Symbol %mappedInfSec name("symMappedInfSec") : !ELF.Section
    %sym_for_scratchSection = ELF.Symbol %scratchSection name("symScratchSection") : !ELF.Section

    %symArg0 = ELF.Symbol %arg0 name("inputCNN") size(4000) : memref<2x1000xf16>
    %symArg1 = ELF.Symbol %arg1 name("outputCNN") size(4000) : memref<2x1000xf16>

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

    %mappedInferenceRelocs = ELF.CreateRelocationSection secName(".rlt.mappedInference") sourceSymbolTableSection(%genericSymSection) targetSection(%mappedInfSec) secFlags(SHF_INFO_LINK) -> !ELF.Section
    {
        ELF.Reloc 0 "R_VPU_64" %sym_for_dmaSection 0
        ELF.Reloc 72 "R_VPU_64" %sym_for_barrierSection 0
    }

    %mappedInferenceActKernelRelocs = ELF.CreateRelocationSection secName(".rlt.mappedInferenceActKernels") sourceSymbolTableSection(%actKernelSymSection) targetSection(%mappedInfSec) secFlags(SHF_INFO_LINK) -> !ELF.Section
    {
        ELF.Reloc 768 "R_VPU_64" %sym_for_actKernelRangeSection 0
        ELF.Reloc 784 "R_VPU_64" %sym_for_actKernelInvoSection 0
        ELF.Reloc 828 "R_VPU_32" %sym_for_actRtCfgBufferSection 0
    }


    %const_0 = arith.constant 0 : i32
    %const_1 = arith.constant 1 : i32 
    %const_2 = arith.constant 2 : i32 
    %const_3 = arith.constant 3 : i32 
    %const_4 = arith.constant 4 : i32 
    %const_5 = arith.constant 5 : i32 
    %const_6 = arith.constant 6 : i32 

    %sym_0 = ELF.Symbol %const_0 name("VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR") { isBuiltin = true } : i32
    %sym_1 = ELF.Symbol %const_1 name("VPU_NNRD_SYM_RTM_IVAR") { isBuiltin = true } : i32
    %sym_2 = ELF.Symbol %const_2 name("VPU_NNRD_SYM_RTM_ACT") { isBuiltin = true } : i32
    %sym_3 = ELF.Symbol %const_3 name("VPU_NNRD_SYM_RTM_DMA0") { isBuiltin = true } : i32
    %sym_4 = ELF.Symbol %const_4 name("VPU_NNRD_SYM_RTM_DMA1") { isBuiltin = true } : i32
    %sym_5 = ELF.Symbol %const_5 name("VPU_NNRD_SYM_FIFO_BASE") { isBuiltin = true } : i32
    %sym_6 = ELF.Symbol %const_6 name("VPU_NNRD_SYM_BARRIERS_START") { isBuiltin = true } : i32

    %vpu_symtab = ELF.CreateSymbolTableSection secName("VPU_RT_SYMTAB") secFlags(SHF_NONE) { isBuiltin = true } -> !ELF.Section
    {
        ELF.PutOpInSection %sym_0 : !ELF.Symbol
        ELF.PutOpInSection %sym_1 : !ELF.Symbol
        ELF.PutOpInSection %sym_2 : !ELF.Symbol
        ELF.PutOpInSection %sym_3 : !ELF.Symbol
        ELF.PutOpInSection %sym_4 : !ELF.Symbol
        ELF.PutOpInSection %sym_5 : !ELF.Symbol
        ELF.PutOpInSection %sym_6 : !ELF.Symbol
    }

    // %dmaRelocs = ELF.CreateRelocationSection secName(".rlt.dmaRelocations") sourceSymbolTableSection(%genericSymSection) targetSection(%dmaSection) secFlags(SHF_INFO_LINK) -> !ELF.Section
    // {
    //     ELF.Reloc 24 "R_VPU_64" %sym_for_scratchSection 0
    //     ELF.Reloc 208 "R_VPU_64" %sym_for_scratchSection 4000
    // }

    %dmaSpecialRelocs = ELF.CreateRelocationSection secName(".rlt.dmaSpecialRelocations") sourceSymbolTableSection(%vpu_symtab) targetSection(%dmaSection) secFlags(SHF_INFO_LINK) -> !ELF.Section
    {
        ELF.Reloc 0 "R_VPU_32_RTM" %sym_3 192  // link address for first dma task
        ELF.Reloc 24 "R_VPU_32" %sym_0 0 // dst for first dma task
        ELF.Reloc 208 "R_VPU_32" %sym_0 4000 // src for second dma task
    }   

    %actKernelParamsRelocs = ELF.CreateRelocationSection secName(".RelA.kernelParams") sourceSymbolTableSection(%actKernelSymSection) targetSection(%kernelParamsSection) secFlags(SHF_INFO_LINK) -> !ELF.Section
    {
        ELF.Reloc 12 "R_VPU_32" %sym_for_kernelIOdimsSection 0
        ELF.Reloc 48 "R_VPU_32" %sym_for_kernelIOdimsSection 0
        ELF.Reloc 16 "R_VPU_32" %sym_for_kernelIOstridesSection 0
        ELF.Reloc 52 "R_VPU_32" %sym_for_kernelIOstridesSection 0
    }

    %actKernelParamsSpecialRelocs = ELF.CreateRelocationSection secName(".RelA.SpecialKernelParams") sourceSymbolTableSection(%vpu_symtab) targetSection(%kernelParamsSection) secFlags(SHF_INFO_LINK) -> !ELF.Section
    {
        ELF.Reloc 0 "R_VPU_32" %sym_0 0  // src
        ELF.Reloc 36 "R_VPU_32" %sym_0 4000  // dst
    }

    %actKernelRangeRelocs = ELF.CreateRelocationSection secName(".RelA.actKernelRange") sourceSymbolTableSection(%actKernelSymSection) targetSection(%actKernelRangeSection) secFlags(SHF_INFO_LINK) -> !ELF.Section
    {
        ELF.Reloc 8 "R_VPU_32" %sym_for_kernelTextSection 0
    }

    %actKernelInvoRelocs = ELF.CreateRelocationSection secName(".RelA.actKernelInvo") sourceSymbolTableSection(%actKernelSymSection) targetSection(%actKernelInvoSection) secFlags(SHF_INFO_LINK) -> !ELF.Section
    {
        ELF.Reloc 0 "R_VPU_32" %sym_for_actKernelRangeSection 0
        ELF.Reloc 4 "R_VPU_32" %sym_for_kernelParamsSection 0
        ELF.Reloc 8 "R_VPU_32" %sym_for_kernelDataSection 0
    }

    %inputRelocs = ELF.CreateRelocationSection secName(".rlt.inputs") sourceSymbolTableSection(%inputSymSection) targetSection(%dmaSection) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USERINPUT") -> !ELF.Section
    {
        ELF.Reloc 16 "R_VPU_64" %symArg0 0
    }

    %outputRelocs = ELF.CreateRelocationSection secName(".rlt.outputs") sourceSymbolTableSection(%outputSymSection) targetSection(%dmaSection) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USEROUTPUT") -> !ELF.Section
    {
        ELF.Reloc 216 "R_VPU_64" %symArg1 0
    }

    return %arg1 : memref<2x1000xf16>
}
}
