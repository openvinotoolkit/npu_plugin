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

#include "vpux/compiler/edsl/kernel_gen.hpp"

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
#include "vpux/compiler/edsl/emit_c.hpp"

#ifdef ENABLE_PLAIDML
#include "pmlc/util/logging.h"
#endif

using namespace llvm;  // NOLINT

namespace vpux {
namespace edsl {

#ifdef ENABLE_PLAIDML

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
                             const MoviCompileParams& params, StringRef name) {
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
    IVLOG(1, "linker script:\n" << os.str());

    std::string err;
    auto ldsFile = mlir::openOutputFile(ldsPath, &err);
    if (!err.empty()) {
        VPUX_THROW("Failed to create linker script: {0}", err);
    }

    ldsFile->os() << os.str();
    ldsFile->keep();
}

static std::string getMoviToolsDir() {
    auto maybeToolsDir = llvm::sys::Process::GetEnv("MV_TOOLS_DIR");
    if (!maybeToolsDir) {
        VPUX_THROW("MV_TOOLS_DIR env var must be set");
    }
    SmallString<128> moviToolsDir{*maybeToolsDir};
    if (!sys::fs::is_directory(moviToolsDir)) {
        VPUX_THROW("MV_TOOLS_DIR must be a directory");
    }
    return moviToolsDir.str().str();
}

static void compileAndLinkSHAVE(StringRef cSourceCode, const MoviCompileParams& params, StringRef name,
                                std::vector<uint8_t>& elfBinary) {
    IVLOG(1, "Kernel C source code:\n" << cSourceCode.str());

    std::string mvToolsDir = getMoviToolsDir();
    IVLOG(1, "MV_TOOLS_DIR = " << mvToolsDir);

    std::string genDir = ".";
    auto KERNEL_GENDIR = llvm::sys::Process::GetEnv("KERNEL_GENDIR");
    if (KERNEL_GENDIR) {
        genDir = *KERNEL_GENDIR;
    }

    // Generate linker script
    std::string srcName = name.str();
    SmallString<128> srcPath(genDir);
    sys::path::append(srcPath, srcName + ".c");
    SmallString<128> objPath(genDir);
    sys::path::append(objPath, srcName + ".o");

    auto MANUAL_SRC = llvm::sys::Process::GetEnv("MANUAL_SRC");
    if (!MANUAL_SRC) {
        std::string err;
        auto srcFile = mlir::openOutputFile(srcPath, &err);
        if (!err.empty()) {
            VPUX_THROW("Failed to create C source file: {0}", err);
        }
        srcFile->os() << cSourceCode;
        srcFile->keep();
    }

    SmallString<128> searchDir(mvToolsDir);
    sys::path::append(searchDir, params.mdkLibDir);
    SmallString<128> elfPath(genDir);
    sys::path::append(elfPath, srcName + ".elf");
    SmallString<128> ldsPath(genDir);
    sys::path::append(ldsPath, srcName + ".lds");
    makeLinkerScript(searchDir, objPath, elfPath, ldsPath, params, name);

    SmallString<128> moviCompile(mvToolsDir);
    sys::path::append(moviCompile, params.moviCompile);
    auto compileCmd = formatv("cd {5}; {0} -mcpu={1} -O3 -c {2} -o {3} -I {4} --save-temps", moviCompile, params.cpu,
                              srcPath, objPath, mvToolsDir, genDir)
                              .str();
    IVLOG(1, compileCmd);
    if (std::system(compileCmd.c_str())) {
        VPUX_THROW("moviCompile failed");
    }

    SmallString<128> linker(mvToolsDir);
    sys::path::append(linker, params.mdkLinker);
    auto linkCmd = formatv("{0} -EL -S -O9 --demangle --gc-sections -T {1}", linker, ldsPath).str();
    IVLOG(1, linkCmd);
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

flatbuffers::Offset<MVCNN::BinaryData> generateKernelForSHAVE(mlir::FuncOp func, const MoviCompileParams& params,
                                                              flatbuffers::FlatBufferBuilder& fbb) {
    // Translate the module into C source code.
    std::string src;
    llvm::raw_string_ostream os(src);
    auto module = func->getParentOfType<mlir::ModuleOp>();
    if (translateSCFToC(module, func, os, func.getName()).failed()) {
        VPUX_THROW("Cannot generate C code from SCF.");
    }

    // Use moviCompile to compile and link C source code into an ELF binary.
    std::vector<uint8_t> elfBinary;
    compileAndLinkSHAVE(os.str(), params, func.getName(), elfBinary);
    return buildBinaryData(fbb, elfBinary);
}

#else

flatbuffers::Offset<MVCNN::BinaryData> generateKernelForSHAVE(mlir::FuncOp, const MoviCompileParams&,
                                                              flatbuffers::FlatBufferBuilder&) {
    VPUX_THROW("generateKernelForSHAVE is only supported when ENABLE_PLAIDML=ON");
}

#endif

}  // namespace edsl
}  // namespace vpux
