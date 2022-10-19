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
    DataInfo "input" : tensor<1x1000xf16>
  } outputsInfo :  {
    DataInfo "softmax" : tensor<1x1000xf16>
  }
  module @VPU.SW  {
    func private @builtin_softmax(memref<*xf16>, memref<*xf16>, i64) attributes {VPU.kernel_code = "single_shave_softmax.cpp", VPU.kernel_entry = "singleShaveSoftmax"}
  }
  func @main(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {

    // Kernel IO buffers (located in CMX)
    %0 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer "CMX_NN" [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    // barriers
    %2 = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPUIPRegMapped.Index<0>
    %3 = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPUIPRegMapped.Index<1>

    // DMA task 0
    %4 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x1x1000xf16>) outputs(%0 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) updates(%2 : !VPUIPRegMapped.Index<0>) start_after(0) -> !VPUIPRegMapped.Index<0>

    // Declaration of Software Layer .text, .data and EntryPoint, for serialization purposes
    %5 = VPUIPRegMapped.DeclareKernelText kernel_path("singleShaveSoftmax.3720xx.elf") -> !VPUIPRegMapped.Index<0>
    %6 = VPUIPRegMapped.DeclareKernelArgs kernel_path("singleShaveSoftmax.3720xx.elf") -> !VPUIPRegMapped.Index<0>
    %7 = VPUIPRegMapped.DeclareKernelEntry kernel_path("singleShaveSoftmax.3720xx.elf") -> !VPUIPRegMapped.Index<0>

    // Act Kernel Range and Invocation Structures
    %8 = VPUIPRegMapped.ActKernelRange kernel_text_index(%5 : !VPUIPRegMapped.Index<0>) kernel_args_index(%6 : !VPUIPRegMapped.Index<0>) kernel_entry_index(%7 : !VPUIPRegMapped.Index<0>) -> !VPUIPRegMapped.Index<0>
    %9 = VPUIPRegMapped.ActKernelInvocation range_index(%8 : !VPUIPRegMapped.Index<0>) waits(%2 : !VPUIPRegMapped.Index<0>) updates(%3 : !VPUIPRegMapped.Index<1>) tile(0) start_after(0) clean_after(1) -> !VPUIPRegMapped.Index<0>

    // Kernel Params structure that holds the binary serialization of the kernel parameters (as in the params C/C++ structure defined for each layer - see sw_runtime_kernels/kernels/inc/*)
    %10 = VPUIPRegMapped.KernelParams input(%0 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) output(%1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("singleShaveSoftmax") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : vector<80xui8>) -> !VPUIPRegMapped.Index<0>

    // DMA task 1
    %11 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x1x1x1000xf16>) previousDMA(%4 : !VPUIPRegMapped.Index<0>) waits(%3 : !VPUIPRegMapped.Index<1>) start_after(0) -> !VPUIPRegMapped.Index<1>

    // Mapped Inference Op
    %12 = VPUIPRegMapped.MappedInference dmas(%4 : !VPUIPRegMapped.Index<0>) actKernelRanges(%8 : !VPUIPRegMapped.Index<0>) actKernelInvocations(%9 : !VPUIPRegMapped.Index<0>) barriers(%2 : !VPUIPRegMapped.Index<0>) dmaCount(2) invariantCount(0) variantCount(0) actKernelRangesCount(1) actKernelInvocationsCount(1) barrierCount(2) -> !VPUIPRegMapped.Index<0>

    // Sections for all the serializable structures
    %13 = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR|VPU_SHF_PROC_DMA") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks"} -> !ELF.Section  {
      ELF.PutOpInSection %4 : !VPUIPRegMapped.Index<0>
      ELF.PutOpInSection %11 : !VPUIPRegMapped.Index<1>
    }
    %14 = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_EXECINSTR|VPU_SHF_PROC_SHAVE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.BarrierConfigs"} -> !ELF.Section  {
      ELF.PutOpInSection %2 : !VPUIPRegMapped.Index<0>
      ELF.PutOpInSection %3 : !VPUIPRegMapped.Index<1>
    }
    %15 = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_EXECINSTR|VPU_SHF_PROC_SHAVE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.KernelText"} -> !ELF.Section  {
      ELF.PutOpInSection %5 : !VPUIPRegMapped.Index<0>
    }
    %16 = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_EXECINSTR|VPU_SHF_PROC_SHAVE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.KernelData"} -> !ELF.Section  {
      ELF.PutOpInSection %6 : !VPUIPRegMapped.Index<0>
    }
    %17 = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_EXECINSTR|VPU_SHF_PROC_SHAVE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.KernelParams"} -> !ELF.Section  {
      ELF.PutOpInSection %10 : !VPUIPRegMapped.Index<0>
    }
    %18 = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_EXECINSTR|VPU_SHF_PROC_DMA") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.ActKernelRanges"} -> !ELF.Section  {
      ELF.PutOpInSection %8 : !VPUIPRegMapped.Index<0>
    }
    %19 = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_EXECINSTR|VPU_SHF_PROC_DMA") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.ActKernelInvocations"} -> !ELF.Section  {
      ELF.PutOpInSection %9 : !VPUIPRegMapped.Index<0>
    }
    %20 = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.MappedInference"} -> !ELF.Section  {
      ELF.PutOpInSection %12 : !VPUIPRegMapped.Index<0>
    }

    // ELF Metadata Section
    %21 = ELF.CreateMetadataSection secFlags("SHF_NONE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".metadata"} -> !ELF.Section  {
      %60 = VPUIPRegMapped.NetworkMetadata -> !VPUIPRegMapped.Index<0>
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
    %31 = ELF.CreateSymbolTableSection secName(".symtab.input") secFlags(VPU_SHF_USERINPUT) -> !ELF.Section  {
      ELF.PutOpInSection %29 : !ELF.Symbol
    }
    %32 = ELF.CreateSymbolTableSection secName(".symtab.output") secFlags(VPU_SHF_USEROUTPUT) -> !ELF.Section  {
      ELF.PutOpInSection %30 : !ELF.Symbol
    }
    %33 = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags(SHF_ALLOC) {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".data.BuffersIO"} -> !ELF.Section  {
    }
    %34 = ELF.Symbol %33 name("sym_bufferSection") : !ELF.Section
    %35 = ELF.CreateSymbolTableSection secName(".symtab.buffers") secFlags("SHF_NONE") -> !ELF.Section  {
      ELF.PutOpInSection %34 : !ELF.Symbol
    }

    // Special SymTab
    %c0_i8 = arith.constant 0 : i8
    %36 = ELF.Symbol %c0_i8 name("VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR") {isBuiltin} : i8
    %c1_i8 = arith.constant 1 : i8
    %37 = ELF.Symbol %c1_i8 name("VPU_NNRD_SYM_RTM_IVAR") {isBuiltin} : i8
    %c2_i8 = arith.constant 2 : i8
    %38 = ELF.Symbol %c2_i8 name("VPU_NNRD_SYM_RTM_ACT") {isBuiltin} : i8
    %c3_i8 = arith.constant 3 : i8
    %39 = ELF.Symbol %c3_i8 name("VPU_NNRD_SYM_RTM_DMA0") {isBuiltin} : i8
    %c4_i8 = arith.constant 4 : i8
    %40 = ELF.Symbol %c4_i8 name("VPU_NNRD_SYM_RTM_DMA1") {isBuiltin} : i8
    %c5_i8 = arith.constant 5 : i8
    %41 = ELF.Symbol %c5_i8 name("VPU_NNRD_SYM_FIFO_BASE") {isBuiltin} : i8
    %c6_i8 = arith.constant 6 : i8
    %42 = ELF.Symbol %c6_i8 name("VPU_NNRD_SYM_BARRIERS_START") {isBuiltin} : i8
    %43 = ELF.CreateSymbolTableSection secName("VPU_RT_SYMTAB") secFlags("SHF_NONE") {isBuiltin} -> !ELF.Section  {
      ELF.PutOpInSection %36 : !ELF.Symbol
      ELF.PutOpInSection %37 : !ELF.Symbol
      ELF.PutOpInSection %38 : !ELF.Symbol
      ELF.PutOpInSection %39 : !ELF.Symbol
      ELF.PutOpInSection %40 : !ELF.Symbol
      ELF.PutOpInSection %41 : !ELF.Symbol
      ELF.PutOpInSection %42 : !ELF.Symbol
    }

    // Main tasks symtab
    %44 = ELF.CreateSymbolTableSection secName(".symtab.tasks") secFlags("SHF_NONE") -> !ELF.Section  {
      ELF.PutOpInSection %22 : !ELF.Symbol
      ELF.PutOpInSection %23 : !ELF.Symbol
      ELF.PutOpInSection %26 : !ELF.Symbol
      ELF.PutOpInSection %27 : !ELF.Symbol
      ELF.PutOpInSection %28 : !ELF.Symbol
      ELF.PutOpInSection %24 : !ELF.Symbol
      ELF.PutOpInSection %25 : !ELF.Symbol
      %60 = ELF.Symbol %12 name("MappedInference_entry") type("VPU_STT_ENTRY") : !VPUIPRegMapped.Index<0>
    }

    // DMA Relocs
    %45 = ELF.CreateRelocationSection secName(".rlt.DMA_NetInput") sourceSymbolTableSection(%31) targetSection(%13) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USERINPUT") -> !ELF.Section  {
      ELF.RelocImmOffset baseOp(%4 : !VPUIPRegMapped.Index<0>) offset(16) "R_VPU_64" %29 0
    }
    %46 = ELF.CreateRelocationSection secName(".rlt.DMA_NetOutput") sourceSymbolTableSection(%32) targetSection(%13) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USEROUTPUT") -> !ELF.Section  {
      ELF.RelocImmOffset baseOp(%11 : !VPUIPRegMapped.Index<1>) offset(24) "R_VPU_64" %30 0
    }
    %47 = ELF.CreateRelocationSection secName(".rlt.dmaIO_DDR") sourceSymbolTableSection(%35) targetSection(%13) secFlags(SHF_INFO_LINK) -> !ELF.Section  {
    }
    %48 = ELF.CreateRelocationSection secName(".rlt.dmaIO_CMX") sourceSymbolTableSection(%43) targetSection(%13) secFlags(SHF_INFO_LINK) -> !ELF.Section  {
      ELF.RelocImmOffset baseOp(%4 : !VPUIPRegMapped.Index<0>) offset(24) "R_VPU_64" %36 0
      ELF.RelocImmOffset baseOp(%4 : !VPUIPRegMapped.Index<0>) offset(0) "R_VPU_32_RTM" %39 192
      ELF.RelocImmOffset baseOp(%11 : !VPUIPRegMapped.Index<1>) offset(16) "R_VPU_64" %36 2000
    }

    // ActKernel Relocs
    %49 = ELF.CreateRelocationSection secName(".rlt.KernelParams") sourceSymbolTableSection(%44) targetSection(%17) secFlags(SHF_INFO_LINK) -> !ELF.Section  {
      ELF.RelocImmOffset baseOp(%10 : !VPUIPRegMapped.Index<0>) offset(12) "R_VPU_32" %28 80
      ELF.RelocImmOffset baseOp(%10 : !VPUIPRegMapped.Index<0>) offset(48) "R_VPU_32" %28 96
      ELF.RelocImmOffset baseOp(%10 : !VPUIPRegMapped.Index<0>) offset(16) "R_VPU_32" %28 112
      ELF.RelocImmOffset baseOp(%10 : !VPUIPRegMapped.Index<0>) offset(52) "R_VPU_32" %28 144
    }
    %50 = ELF.CreateRelocationSection secName(".rlt.KernelParamsIO_DDR") sourceSymbolTableSection(%35) targetSection(%17) secFlags(SHF_INFO_LINK) -> !ELF.Section  {
    }
    %51 = ELF.CreateRelocationSection secName(".rlt.KernelParamsIO_CMX") sourceSymbolTableSection(%43) targetSection(%17) secFlags(SHF_INFO_LINK) -> !ELF.Section  {
      ELF.RelocImmOffset baseOp(%10 : !VPUIPRegMapped.Index<0>) offset(0) "R_VPU_32" %36 0
      ELF.RelocImmOffset baseOp(%10 : !VPUIPRegMapped.Index<0>) offset(36) "R_VPU_32" %36 2000
    }
    %52 = ELF.CreateRelocationSection secName(".rlt.ActKernelRange") sourceSymbolTableSection(%44) targetSection(%18) secFlags(SHF_INFO_LINK) -> !ELF.Section  {
      ELF.RelocImmOffset baseOp(%8 : !VPUIPRegMapped.Index<0>) offset(8) "R_VPU_32" %26 0
    }
    %53 = ELF.CreateRelocationSection secName(".rlt.ActKernelInvo") sourceSymbolTableSection(%44) targetSection(%19) secFlags(SHF_INFO_LINK) -> !ELF.Section  {
      ELF.RelocImmOffset baseOp(%9 : !VPUIPRegMapped.Index<0>) offset(0) "R_VPU_32" %24 0
      ELF.RelocImmOffset baseOp(%9 : !VPUIPRegMapped.Index<0>) offset(8) "R_VPU_32" %27 0
      ELF.RelocImmOffset baseOp(%9 : !VPUIPRegMapped.Index<0>) offset(4) "R_VPU_32" %28 0
    }

    // Declaration of special 1MB section for the ActKernel Runtime Configs
    %54 = VPURT.DeclareBuffer "DDR" <64> -> memref<262144xi32, @DDR>
    %55 = ELF.CreateLogicalSection secType(SHT_NOBITS) secFlags("SHF_NONE") {secAddrAlign = 1024 : i64, secInfo = 0 : i64, secName = ".bss.actKernelRtConfigSec"} -> !ELF.Section  {
      ELF.PutOpInSection %54 : memref<262144xi32, @DDR>
    }
    %56 = ELF.Symbol %55 name("sym_actKernelRtConfigsSec") : !ELF.Section
    %57 = ELF.CreateSymbolTableSection secName(".symtab.actKernelRtConfig") secFlags("SHF_NONE") -> !ELF.Section  {
      ELF.PutOpInSection %56 : !ELF.Symbol
    }

    // Final Mapped Inference Relocs
    %58 = ELF.CreateRelocationSection secName(".rlt.MI_AKRtConfig") sourceSymbolTableSection(%57) targetSection(%20) secFlags(SHF_INFO_LINK) -> !ELF.Section  {
      ELF.RelocImmOffset baseOp(%12 : !VPUIPRegMapped.Index<0>) offset(828) "R_VPU_32" %56 0
    }
    %59 = ELF.CreateRelocationSection secName(".rlt.MappedInference") sourceSymbolTableSection(%44) targetSection(%20) secFlags(SHF_INFO_LINK) -> !ELF.Section  {
      ELF.RelocImmOffset baseOp(%12 : !VPUIPRegMapped.Index<0>) offset(0) "R_VPU_64" %22 0
      ELF.RelocImmOffset baseOp(%12 : !VPUIPRegMapped.Index<0>) offset(72) "R_VPU_64" %23 0
      ELF.RelocImmOffset baseOp(%12 : !VPUIPRegMapped.Index<0>) offset(768) "R_VPU_64" %24 0
      ELF.RelocImmOffset baseOp(%12 : !VPUIPRegMapped.Index<0>) offset(784) "R_VPU_64" %25 0
    }
    return %arg1 : memref<1x1x1x1000xf16>
  }
}

