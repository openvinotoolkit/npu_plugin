//
// Copyright 2021 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/act_kernels/act_kernel_gen.h"

#include <algorithm>
#include <string>

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/ToolOutputFile.h>
#include <mlir/Support/FileUtilities.h>

#include "vpux/compiler/dialect/VPUIP/ops.hpp"


using namespace llvm;  // NOLINT

namespace vpux {

flatbuffers::Offset<MVCNN::BinaryData> buildBinaryData(flatbuffers::FlatBufferBuilder& fbb,
                                                       const std::vector<uint8_t>& data) {
    const auto pack = [](const std::vector<uint8_t>& src) {
      std::vector<uint64_t> packed(llvm::divideCeil(src.size(), sizeof(uint64_t)));
      auto ptr = reinterpret_cast<uint8_t*>(packed.data());
      for (size_t i = 0; i < src.size(); i++) {
          ptr[i] = src[i];
      }
      return packed;
    };
    auto packedData = fbb.CreateVector(pack(data));
    MVCNN::BinaryDataBuilder builder(fbb);
    builder.add_data(packedData);
    builder.add_csram_cacheable(true);
    builder.add_length(data.size());
    builder.add_underlying_type(MVCNN::DType::DType_U8);
    return builder.Finish();
}

static void makeLinkerScript(StringRef searchDir, StringRef input, StringRef output, StringRef ldsPath,
                             const movitools::MoviCompileParams& params, StringRef name) {
    std::string str;
    llvm::raw_string_ostream os(str);
    os << "SEARCH_DIR (" << searchDir << ")\n";
    os << "INPUT (" << input << ")\n";
    os << "GROUP (";
    for (const auto& lib : params.mdkLibs) {
        os << lib << " ";
    }
    os << ")\n";
    os << "OUTPUT (" << output << ")\n";
    os << "ENTRY (" << name << ")\n";
    const std::string section =
            R"(SECTIONS {
  . = 0x1E000000;
  .dyn.text : {
        *(.text*)
  }
  . = 0x1F000000;
  .dyn.data : {
        *(.data*)
        *(.rodata*)
  }
}
    )";
    os << section << "\n";
    //IVLOG(1, "linker script:\n" << os.str());

    std::string err;
    auto ldsFile = mlir::openOutputFile(ldsPath, &err);
    if (!err.empty()) {
        VPUX_THROW("Failed to create linker script: {0}", err);
    }

    ldsFile->os() << os.str();
    ldsFile->keep();
}


static void compileAndLinkSHAVE(const movitools::MoviCompileParams& params, StringRef name,
                                std::vector<uint8_t>& elfBinary) {
   // IVLOG(1, "Kernel C source code:\n" << cSourceCode.str());

    std::string mvToolsDir = movitools::getMoviToolsDir();
    //IVLOG(1, "MV_TOOLS_DIR = " << mvToolsDir);

    std::string genDir = ".";
    auto KERNEL_GENDIR = llvm::sys::Process::GetEnv("KERNEL_DIR");
    if (KERNEL_GENDIR) {
        genDir = *KERNEL_GENDIR;
    }

    // Generate linker script
    std::string srcName = name.str();
    SmallString<128> srcPath(genDir);
    sys::path::append(srcPath, srcName);
    sys::path::append(srcPath, "src");
    sys::path::append(srcPath, srcName + "_fp16.c");

    SmallString<128> incPath(genDir);
    sys::path::append(incPath, "common");
    sys::path::append(incPath, "inc");

    SmallString<128> objPath(genDir);
    sys::path::append(objPath, "common");
    sys::path::append(objPath, "build");
    sys::path::append(objPath, srcName + ".o");

    SmallString<128> searchDir(mvToolsDir);
    sys::path::append(searchDir, params.mdkLibDir);
    SmallString<128> elfPath(genDir);
    sys::path::append(elfPath, srcName + ".elf");
    SmallString<128> ldsPath(genDir);
    sys::path::append(ldsPath, srcName + ".lds");
    makeLinkerScript(searchDir, objPath, elfPath, ldsPath, params, name);

    SmallString<128> moviCompile(mvToolsDir);
    sys::path::append(moviCompile, params.moviCompile);

    auto compileCmd = formatv("cd {6}; {0} -mcpu={1} -O3 -c {2} -o {3} -I {4} -I {5} --save-temps", moviCompile, params.cpu,
                              srcPath, objPath, mvToolsDir, incPath, genDir)
            .str();
    //IVLOG(1, compileCmd);
    if (std::system(compileCmd.c_str())) {
        VPUX_THROW((std::string("moviCompile failed: ") + compileCmd).c_str());
    }

    SmallString<128> linker(mvToolsDir);
    sys::path::append(linker, params.mdkLinker);
    auto linkCmd = formatv("{0} -EL -S -O9 --demangle --gc-sections -T {1}", linker, ldsPath).str();
  //  IVLOG(1, linkCmd);
    if (std::system(linkCmd.c_str())) {
        VPUX_THROW("linker failed");
    }

    {
        std::string err;
        auto elfFile = mlir::openInputFile(elfPath, &err);
        if (!elfFile) {
            VPUX_THROW("Could not open ELF binary: {0}", err);
        }

        auto elfBuffer = elfFile->getBuffer();
        std::copy(elfBuffer.begin(), elfBuffer.end(), std::back_inserter(elfBinary));
    }
}

flatbuffers::Offset<MVCNN::BinaryData> generateKernelForACTShave(mlir::StringRef funcName, const movitools::MoviCompileParams& params,
                                                              flatbuffers::FlatBufferBuilder& fbb) {
    //auto module = func->getParentOfType<mlir::ModuleOp>();
    //if (translateSCFToC(module, func, os, func.getName()).failed()) {
      //  VPUX_THROW("Cannot generate C code from SCF.");
    //}

    // Use moviCompile to compile and link C source code into an ELF binary.
    std::vector<uint8_t> elfBinary;
    compileAndLinkSHAVE(params, funcName, elfBinary);
    return buildBinaryData(fbb, elfBinary);
}


}  // namespace vpux
