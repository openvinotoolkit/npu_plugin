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
#include "vpux/compiler/act_kernels/nn_act_args.h"

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

flatbuffers::Offset<MVCNN::KernelData> buildKernelData(flatbuffers::FlatBufferBuilder& fbb,
                                                       llvm::ArrayRef<uint8_t> content) {
    auto packedData = fbb.CreateVector(content.data(), content.size());
    MVCNN::KernelDataBuilder builder(fbb);
    builder.add_data(packedData);
    builder.add_length(content.size());
    return builder.Finish();
}
/*
flatbuffers::Offset<MVCNN::BinaryData> buildKernelArgsData(flatbuffers::FlatBufferBuilder& fbb,
                                                           cfg_dpu_description &) {

    //serialize structure raw view
    const auto pack = [](const std::vector<uint8_t>& src) {
      std::vector<uint64_t> packed(llvm::divideCeil(src.size(), sizeof(uint64_t)));
      auto ptr = reinterpret_cast<uint8_t*>(packed.data());
      for (size_t i = 0; i < src.size(); i++) {
          ptr[i] = src[i];
      }
      return packed;
    };
    //auto packedData = fbb.CreateVector(pack(data));
    MVCNN::BinaryDataBuilder builder(fbb);
    //builder.add_data(packedData);
    builder.add_csram_cacheable(true);
    //builder.add_length(data.size());
    builder.add_underlying_type(MVCNN::DType::DType_U8);
    return builder.Finish();
}
*/
//static void makeLinkerScript(StringRef searchDir, StringRef input, StringRef output, StringRef ldsPath,
//                             const movitools::MoviCompileParams& params, StringRef name) {
//    std::string str;
//    llvm::raw_string_ostream os(str);
//    os << "SEARCH_DIR (" << searchDir << ")\n";
//    os << "INPUT (" << input << ")\n";
//    os << "GROUP (";
//    for (const auto& lib : params.mdkLibs) {
//        os << lib << " ";
//    }
//    os << ")\n";
//    os << "OUTPUT (" << output << ")\n";
//    os << "ENTRY (" << name << ")\n";
//    const std::string section =
//            R"(SECTIONS {
//  . = 0x1E000000;
//  .dyn.text : {
//        *(.text*)
//  }
//  . = 0x1F000000;
//  .dyn.data : {
//        *(.data*)
//        *(.rodata*)
//  }
//}
//    )";
//    os << section << "\n";
//    //IVLOG(1, "linker script:\n" << os.str());
//
//    std::string err;
//    auto ldsFile = mlir::openOutputFile(ldsPath, &err);
//    if (!err.empty()) {
//        VPUX_THROW("Failed to create linker script: {0}", err);
//    }
//
//    ldsFile->os() << os.str();
//    ldsFile->keep();
//}
//

static void compileAndLinkSHAVE(const movitools::MoviCompileParams& params, StringRef name,
                                std::vector<uint8_t>& textBinary, std::vector<uint8_t>& dataBinary) {
    // IVLOG(1, "Kernel C source code:\n" << cSourceCode.str());

    std::string mvToolsDir = movitools::getMoviToolsDir();
    //IVLOG(1, "MV_TOOLS_DIR = " << mvToolsDir);

    std::string genDir = ".";
    auto KERNEL_GENDIR = llvm::sys::Process::GetEnv("KERNEL_DIR");
    if (KERNEL_GENDIR) {
        genDir = *KERNEL_GENDIR;
    }

    std::string srcName = name.str();

    // Generate linker script name
    SmallString<128> linkerScriptPath(genDir);
    sys::path::append(linkerScriptPath, "build");
    sys::path::append(linkerScriptPath, "shave_kernel.ld");

    SmallString<128> srcPath(genDir);
    sys::path::append(srcPath, srcName);
    sys::path::append(srcPath, "src");
    sys::path::append(srcPath, srcName + "_fp16.c");

    SmallString<128> incPath(genDir);
    sys::path::append(incPath, "common");
    sys::path::append(incPath, "inc");

    SmallString<128> incPath2(genDir);
    sys::path::append(incPath2, "asds");

    SmallString<128> singleLib(mvToolsDir);
    sys::path::append(singleLib, params.mdkLibDir);
    sys::path::append(singleLib, params.mdkLibs[0]);

    SmallString<128> objPath(genDir);
    sys::path::append(objPath, "build");
    sys::path::append(objPath, srcName + ".o");

    SmallString<128> objDir(genDir);
    sys::path::append(objDir, "build");

    SmallString<128> elfPath(genDir);
    sys::path::append(elfPath, "build");
    sys::path::append(elfPath, srcName + ".elf");

//    SmallString<128> ldsPath(genDir);
//    sys::path::append(ldsPath, srcName + ".lds");
//   // makeLinkerScript(searchDir, objPath, elfPath, ldsPath, params, name);

    SmallString<128> moviCompile(mvToolsDir);
    sys::path::append(moviCompile, params.moviCompile);

    {
        auto compileCmd = formatv("cd {6}; {1} -mcpu={2} -c {3} -o {4} -I {5} -I {6} -I{7} ", genDir, moviCompile, params.cpu,
                                  srcPath, objPath, mvToolsDir, incPath, incPath2).str();
        // IVLOG(1, compileCmd);
        if (std::system(compileCmd.c_str())) {
            VPUX_THROW((std::string("moviCompile failed: ") + compileCmd).c_str());
        }
    }

#ifdef GEN_SYM_FILE
    SmallString<128> symPath(genDir);
    sys::path::append(symPath, "build");
    sys::path::append(symPath, srcName + ".s");

    {
        auto compileCmd = formatv("cd {0}; {1} -mcpu={2} -S {3} -o {4} -I {5} -I {6} -I {7} ", genDir, moviCompile, params.cpu,
                                  srcPath, symPath, mvToolsDir, incPath, incPath2).str();
        // IVLOG(1, compileCmd);
        if (std::system(compileCmd.c_str())) {
            VPUX_THROW((std::string("moviCompile failed: ") + compileCmd).c_str());
        }
    }
#endif  // GEN_SYM_FILE

    //$sparc/sparc-myriad-rtems-ld  -zmax-page-size=16 --script ../../build/$LINKER_SCRIPT -entry sigmoid_fp16 --gc-sections --strip-debug --discard-all $KERNEL_SRC.o -EL ../../lib/$MCPU_LIB/mlibm.a --output $KERNEL_SRC.elf

    SmallString<128> linker(mvToolsDir);
    sys::path::append(linker, params.mdkLinker);
    auto linkCmd = formatv("{0} -zmax-page-size=16 --script {1}"
                           " -entry {2}_fp16 --gc-sections --strip-debug --discard-all  {3}"
                            " -EL {4} --output {5}",
                            linker, linkerScriptPath, srcName.c_str(), objPath, singleLib, elfPath).str();
    // IVLOG(1, linkCmd);
    if (std::system(linkCmd.c_str())) {
        VPUX_THROW((std::string("linker failed: ") + linkCmd).c_str());
    }

    //$sparc/sparc-myriad-rtems-objcopy -O binary --only-section=.text $KERNEL_SRC.elf sk.sigmoid_fp16_"$MCPU".text
    //$sparc/sparc-myriad-rtems-objcopy -O binary --only-section=.arg.data $KERNEL_SRC.elf sk.sigmoid_fp16_"$MCPU".data

    SmallString<128> objcopy(mvToolsDir);
    sys::path::append(objcopy, params.mdkObjCopy);

    SmallString<128> textPath(genDir);
    sys::path::append(textPath, "build");
    sys::path::append(textPath, "sk." + srcName + "_fp16_" + params.cpu + ".text");

    {
        auto objCopyCmd = formatv("{0} -O binary --only-section=.text {1} {2}",
                                  objcopy, elfPath, textPath).str();
        // IVLOG(1, compileCmd);
        if (std::system(objCopyCmd.c_str())) {
            VPUX_THROW((std::string("objcopy failed: ") + objCopyCmd).c_str());
        }
    }

    SmallString<128> dataPath(genDir);
    sys::path::append(dataPath, "build");
    sys::path::append(dataPath, "sk." + srcName + "_fp16_" + params.cpu + ".data");

    {
        auto objCopyCmd = formatv("{0} -O binary --only-section=.arg.data {1} {2}",
                                  objcopy, elfPath, dataPath).str();

        if (std::system(objCopyCmd.c_str())) {
            VPUX_THROW((std::string("objcopy failed: ") + objCopyCmd).c_str());
        }
    }


    auto readBinary = [](SmallString<128> & path, std::vector<uint8_t>& buffer, uint32_t alignment = 1) {
          std::string err;
          auto elfFile = mlir::openInputFile(path, &err);
          if (!elfFile) {
              VPUX_THROW("Could not open {0} binary, err:{1}", path.c_str(), err);
          }

          auto elfBuffer = elfFile->getBuffer();
          std::copy(elfBuffer.begin(), elfBuffer.end(), std::back_inserter(buffer));

          if (alignment  & (alignment - 1)) {
              VPUX_THROW("Could not align to now power of 2:{1}", alignment);
          }
          auto totalBytes = std::distance(elfBuffer.begin(), elfBuffer.end());
          auto padBytes = -totalBytes & (alignment - 1);
          if (padBytes) {
              std::fill_n(back_inserter(buffer), padBytes, 0);
          }
    };

    readBinary(textPath, textBinary, 0x10);
    readBinary(dataPath, dataBinary, 0x10);
}

ActKernelDesc compileKernelForACTShave(mlir::StringRef kernelName, const movitools::MoviCompileParams& params,
                                                              flatbuffers::FlatBufferBuilder& fbb) {

    // Use moviCompile to compile and link C source code into an ELF binary.
    // and then using objcopy teardown elf into text and data sections
    std::vector<uint8_t> textBinary;
    std::vector<uint8_t> dataBinary;
    compileAndLinkSHAVE(params, kernelName, textBinary, dataBinary);

    ActKernelDesc result;

    result.text = {kernelName.data(), buildKernelData(fbb, textBinary), textBinary.size()};

    auto dataName = std::string(kernelName) + ".data";
    result.data = {dataName, buildKernelData(fbb, dataBinary), dataBinary.size()};

    return result;
}


// todo provide some arguments for kernel
/*flatbuffers::Offset<flatbuffers::Vector<uint64_t>> packKernelArgs(flatbuffers::FlatBufferBuilder& fbb) {
    act_kernel_args dummyArgs;
    auto packedData = buildKernelArgsData(fbb, dummyArgs);

    throw std::runtime_error("dont know what next");
}
*/
}  // namespace vpux
