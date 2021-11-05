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

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4267)  // size_t to integer conversion
#endif

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Process.h>
#include <mlir/Support/FileUtilities.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif

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

static void compileAndLinkSHAVE(const movitools::MoviCompileParams& params, const CompilationUnitDesc& unitDesc,
                                SmallVector<uint8_t, 128>& textBinary, SmallVector<uint8_t, 128>& dataBinary) {
    std::string mvToolsDir = movitools::getMoviToolsDir();
    SmallString<128> genDir;
    SmallString<128> tmpDir;
    // TODO: weak assumption on tools dir better switch to DEVELOPER BUILD style
    if (sys::fs::exists(KERNEL_DIRECTORY) && sys::fs::exists(LIBRARY_OUTPUT_DIRECTORY)) {
        genDir = KERNEL_DIRECTORY;
        tmpDir = LIBRARY_OUTPUT_DIRECTORY;
    } else {
        // probe for OV_BUILD_DIR
        const auto ovBuildDir = std::getenv("OV_BUILD_DIR");
        VPUX_THROW_UNLESS(ovBuildDir,
                          "OV_BUILD_DIR env directory must be specified, in order to reach act-shave kernels");
        VPUX_THROW_UNLESS(sys::fs::exists(ovBuildDir),
                          "OpenVino build directory {0}, taken from OV_BUILD_DIR env variable is not exist", genDir);

        genDir = ovBuildDir;
        sys::path::append(genDir, "act-kernels");

        VPUX_THROW_UNLESS(sys::fs::exists(genDir),
                          "act-kernels directory {0}, taken from OV_BUILD_DIR env variable is not exist", genDir);

        tmpDir = genDir;
        sys::path::append(tmpDir, "output");
    }

    SmallString<128> srcNamePath = unitDesc.codePath;

    SmallString<128> srcNameNoExt = sys::path::filename(srcNamePath);
    sys::path::replace_extension(srcNameNoExt, "");

    std::string entryPoint = unitDesc.entry.str();

    SmallString<128> buildDirPath;
    {
        SmallString<128> tmpPath(tmpDir);
        sys::path::append(tmpPath, "act-kernels-build");
        sys::path::append(tmpPath, srcNamePath);
        buildDirPath = sys::path::parent_path(tmpPath);
        sys::fs::create_directories(buildDirPath);
    }

    // Generate linker script name - and copy it from
    SmallString<128> linkerScriptPath(genDir);
    sys::path::append(linkerScriptPath, "build");
    sys::path::append(linkerScriptPath, "shave_kernel.ld");

    SmallString<128> srcPath(genDir);
    sys::path::append(srcPath, srcNamePath);

    SmallString<128> incPath(genDir);
    sys::path::append(incPath, "inc");

    SmallString<128> singleLib(mvToolsDir);
    sys::path::append(singleLib, params.mdkLibDir);
    sys::path::append(singleLib, params.mdkLibs[0]);

    SmallString<128> objPath(buildDirPath);
    sys::path::append(objPath, srcNameNoExt + ".o");

    SmallString<128> objDir(buildDirPath);

    SmallString<128> elfPath(buildDirPath);
    sys::path::append(elfPath, srcNameNoExt + ".elf");

    SmallString<128> moviCompile(mvToolsDir);
    sys::path::append(moviCompile, params.moviCompile);

    {
        auto compileCmd = formatv("{0} -mcpu={1} -c {2} -o {3} -I {4} -I{5} ", moviCompile, params.cpu, srcPath,
                                  objPath, mvToolsDir, incPath)
                                  .str();
        if (std::system(compileCmd.c_str())) {
            VPUX_THROW("moviCompile failed: {0}", compileCmd);
        }
    }

    SmallString<128> linker(mvToolsDir);
    sys::path::append(linker, params.mdkLinker);
    auto linkCmd = formatv("{0} -zmax-page-size=16 --script {1}"
                           " -entry {2} --gc-sections --strip-debug --discard-all  {3}"
                           " -EL {4} --output {5}",
                           linker, linkerScriptPath, entryPoint.c_str(), objPath, singleLib, elfPath)
                           .str();
    std::cout << linkCmd << std::endl;
    if (std::system(linkCmd.c_str())) {
        VPUX_THROW("linker failed: {0}", linkCmd);
    }

    SmallString<128> objcopy(mvToolsDir);
    sys::path::append(objcopy, params.mdkObjCopy);

    SmallString<128> textPath(buildDirPath);
    sys::path::append(textPath, "sk." + srcNameNoExt + "." + params.cpu + ".text");

    {
        auto objCopyCmd = formatv("{0} -O binary --only-section=.text {1} {2}", objcopy, elfPath, textPath).str();
        std::cout << objCopyCmd << std::endl;
        if (std::system(objCopyCmd.c_str())) {
            VPUX_THROW("objcopy failed: {0}", objCopyCmd);
        }
    }

    SmallString<128> dataPath(buildDirPath);
    sys::path::append(dataPath, "sk." + srcNameNoExt + "." + params.cpu + ".data");

    {
        auto objCopyCmd = formatv("{0} -O binary --only-section=.arg.data {1} {2}", objcopy, elfPath, dataPath).str();

        std::cout << objCopyCmd << std::endl;
        if (std::system(objCopyCmd.c_str())) {
            VPUX_THROW("objcopy failed: {0}", objCopyCmd);
        }
    }

    auto readBinary = [](SmallString<128>& path, SmallVector<uint8_t, 128>& buffer, uint32_t alignment = 1) {
        std::string err;
        auto elfFile = mlir::openInputFile(path, &err);
        if (!elfFile) {
            VPUX_THROW("Could not open {0} binary, err:{1}", path.c_str(), err);
        }

        auto elfBuffer = elfFile->getBuffer();
        std::copy(elfBuffer.begin(), elfBuffer.end(), std::back_inserter(buffer));

        if (alignment & (alignment - 1)) {
            VPUX_THROW("Could not align to now power of 2:{1}", alignment);
        }
        auto totalBytes = std::distance(elfBuffer.begin(), elfBuffer.end());
        auto padBytes = -totalBytes & (alignment - 1);
        if (padBytes) {
            std::fill_n(std::back_inserter(buffer), padBytes, 0);
        }
    };

    readBinary(textPath, textBinary, 0x10);
    readBinary(dataPath, dataBinary, 0x10);
}

ActKernelDesc compileKernelForACTShave(const CompilationUnitDesc& unitDesc,
                                       const movitools::MoviCompileParams& params) {
    // Use moviCompile to compile and link C source code into an ELF binary.
    // and then using objcopy teardown elf into text and data sections
    SmallVector<uint8_t, 128> textBinary;
    SmallVector<uint8_t, 128> dataBinary;
    compileAndLinkSHAVE(params, unitDesc, textBinary, dataBinary);

    // lets pad textBinary by 1K array at the end with FC CC FC CC
    for (int i = 0; i != 512; i++) {
        textBinary.push_back(0xFC);
        textBinary.push_back(0xCC);
    }

    ActKernelDesc result;
    result.text = {unitDesc.name.data(), textBinary, textBinary.size() - 1024};

    auto dataName = std::string(unitDesc.name) + ".data";
    result.data = {dataName, dataBinary, dataBinary.size()};

    return result;
}

}  // namespace vpux
