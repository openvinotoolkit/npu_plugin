module @Test {

IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "inputCNN" : tensor<1x1x1x1000xf16>
} outputsInfo :  {
    DataInfo "outputCNN" : tensor<1x1x1x1000xf16>
}

func @main(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    %0 = VPUIPRegMapped.DeclareBuffer "VPU_CMX_NN" [0, 1, 2, 3] <0> -> memref<1x1x1x1000xf16, "CMX_NN">
    %1 = VPUIPRegMapped.DeclareBuffer "VPU_CMX_NN" [0, 1, 2, 3] <2000> -> memref<1x1x1x1000xf16, "CMX_NN">

    %3 = VPUIPRegMapped.ConfigureBarrier<0, -1> -> !VPURT.Barrier
    %5 = VPUIPRegMapped.ConfigureBarrier<1, -1> -> !VPURT.Barrier


    // Note: this section has type PROGBITS.
    // See Table from slide 26 of Andrew Bakalin's presentation ELF PoC_new.pptx
    %secDW = ELF.CreateSection secFlags(SHF_ALLOC) {secName = ".data.Weights", secType = "SHT_PROGBITS", secInfo = 1, secAddrAlign = 4} -> !ELF.Section
      {
        ELF.PutOpInSection %0 : memref<1x1x1x1000xf16, "CMX_NN">
        ELF.PutOpInSection %1 : memref<1x1x1x1000xf16, "CMX_NN">
      }


    %ctDcl = const.Declare memref<1x3x1x1xf16> = #const.Content<dense<[[[[-4.077150e-01]], [[-4.580080e-01]], [[-4.851070e-01]]]]> : tensor<1x3x1x1xf16>>

    %secDWCT = ELF.CreateSection secFlags(SHF_ALLOC) {secName = ".data.Weights_ct", secType = "SHT_PROGBITS", secInfo = 1, secAddrAlign = 4} -> !ELF.Section
      {
        ELF.PutOpInSection %ctDcl : memref<1x3x1x1xf16>
      }


    %sec0 = ELF.CreateSection secFlags(SHF_EXECINSTR) {secName = ".text.BarrierConfigs", secType = "SHT_PROGBITS", secInfo = 1, secAddrAlign = 4} -> !ELF.Section
      {
        ELF.PutOpInSection %3 : !VPURT.Barrier
        ELF.PutOpInSection %5 : !VPURT.Barrier
     }


    %2 = VPUIPRegMapped.NNDMA inputs(%arg0 : memref<1x1x1x1000xf16>) outputs(%0 : memref<1x1x1x1000xf16, "CMX_NN">) start_after(0) -> memref<1x1x1x1000xf16, "CMX_NN">
    %4 = VPUIPRegMapped.NNDMA inputs(%2 : memref<1x1x1x1000xf16, "CMX_NN">) outputs(%1 : memref<1x1x1x1000xf16, "CMX_NN">) waits(%3 : !VPURT.Barrier) start_after(0) -> memref<1x1x1x1000xf16, "CMX_NN">
    %6 = VPUIPRegMapped.NNDMA inputs(%4 : memref<1x1x1x1000xf16, "CMX_NN">) outputs(%arg1 : memref<1x1x1x1000xf16>) waits(%5 : !VPURT.Barrier) start_after(0) -> memref<1x1x1x1000xf16>

    %sec1 = ELF.CreateSection secFlags("SHF_ALLOC|SHF_EXECINSTR") {secName = ".text.dmaTasks", secType = "SHT_PROGBITS", secInfo = 1, secAddrAlign = 4} -> !ELF.Section
      {
          ELF.PutOpInSection %2 : memref<1x1x1x1000xf16, "CMX_NN">
          ELF.PutOpInSection %4 : memref<1x1x1x1000xf16, "CMX_NN">
          ELF.PutOpInSection %6 : memref<1x1x1x1000xf16>
      }



    %symArg0 = ELF.Symbol %arg0 name("inputCNN") : memref<1x1x1x1000xf16>
    %symArg1 = ELF.Symbol %arg1 name("outputCNN") : memref<1x1x1x1000xf16>
    //
    %sym0 = ELF.Symbol %0 name("nndmaOp0_output") : memref<1x1x1x1000xf16, "CMX_NN">
    %sym1 = ELF.Symbol %1 name("nndmaOp1_output") : memref<1x1x1x1000xf16, "CMX_NN">
    %sym2 = ELF.Symbol %2 name("nndmaOp1_input") : memref<1x1x1x1000xf16, "CMX_NN">
    %sym4 = ELF.Symbol %4 name("nndmaOp2_input") : memref<1x1x1x1000xf16, "CMX_NN">


    // These symbols are used for 1 relocation per inference run (JIT relocation)
    %InputSymTableSection = ELF.CreateSymbolTableSection secName(".input.symbolTableSection") secFlags(VPU_SHF_USERINPUT) -> !ELF.Section
    {
        ELF.PutOpInSection %symArg0 : !ELF.Symbol
    }

    %OutputSymTableSection = ELF.CreateSymbolTableSection secName(".output.symbolTableSection") secFlags(VPU_SHF_USEROUTPUT) -> !ELF.Section
    {
        ELF.PutOpInSection %symArg1 : !ELF.Symbol
    }


    // These symbols are used for 1 relocation per executive run
    %restSymSection = ELF.CreateSymbolTableSection secName(".rest.symbolTableSection") secFlags(SHF_NONE) -> !ELF.Section
    {
        ELF.PutOpInSection %sym0 : !ELF.Symbol
        ELF.PutOpInSection %sym1 : !ELF.Symbol
        ELF.PutOpInSection %sym2 : !ELF.Symbol
        ELF.PutOpInSection %sym4 : !ELF.Symbol
    }





    %nndmaRelocSection = ELF.CreateRelocationSection secName(".rela.dma") sourceSymbolTableSection(%restSymSection) targetSection(%sec1) secFlags(SHF_NONE) -> !ELF.Section
    {
     ELF.Reloc 24 // relocating/patching the dst field of NNDMAOp %2, i.e. %0 (24 is the offset w.r.t. the beginning of %sec1)
               "R_VPU_64" // relocation type
               %sym0 // symbol from source symbol table
               0 // 12 // addend - offset of %0 = VPUIPRegMapped.DeclareBuffer... w.r.t. the beginning of its containing section secDW. So the final value patched is beginning_addr_secDW + addend.

     ELF.Reloc 104 // relocating/patching the dst field of NNDMAOp %4, i.e. %1 (80 + 24 = 104 is the offset w.r.t. the beginning of %sec1)
               "R_VPU_64" // relocation type
               %sym1 // symbol from source symbol table
               0 // 20 // addend - offset of %1 = VPUIPRegMapped.DeclareBuffer... w.r.t. the beginning of its containing section secDW. So the final value patched is beginning_addr_secDW + addend.

     ELF.Reloc 96 // relocating/patching the dst field of NNDMAOp %4, i.e. %2 (96 is the offset w.r.t. the beginning of %sec1)
               "R_VPU_64" // relocation type
               %sym2 // symbol from source symbol table
               0 // addend

     ELF.Reloc 176 // relocating/patching the dst field of NNDMAOp %6, i.e. %4 (176 is the offset w.r.t. the beginning of %sec1)
               "R_VPU_64" // relocation type
               %sym4 // symbol from source symbol table
               0 // addend - offset of %0 = VPUIPRegMapped.DeclareBuffer... w.r.t. the beginning of its containing section secDW. So the final value patched is beginning_addr_secDW + addend.
    }



    %InputRelocSection = ELF.CreateRelocationSection secName(".rela.input") sourceSymbolTableSection(%InputSymTableSection) targetSection(%sec1) secFlags(VPU_SHF_USERINPUT) -> !ELF.Section
    {
     ELF.Reloc 16 // relocating/patching the src field of NNDMAOp %2, i.e. %arg0 (16 is the offset w.r.t. the beginning of %sec1)
               "R_VPU_64" // relocation type
               %symArg0 // symbol from source symbol table
               0 // addend is 0 since the input argument of the NN graph, %arg0, is not contained AFAIK in any other section
    }

    %OutputRelocSection = ELF.CreateRelocationSection secName(".rela.output") sourceSymbolTableSection(%OutputSymTableSection) targetSection(%sec1) secFlags(VPU_SHF_USEROUTPUT) -> !ELF.Section
    {
     ELF.Reloc 184 // relocating/patching the dst field of NNDMAOp %6, i.e. %1 (2*80 + 24 = 184 is the offset w.r.t. the beginning of %sec1)
               "R_VPU_64" // relocation type
               %symArg1 // symbol from source symbol table
               0 // addend is 0 since the input argument of the NN graph, %arg0, is not contained AFAIK in any other section
    }


    return %arg1 : memref<1x1x1x1000xf16>
}
}
