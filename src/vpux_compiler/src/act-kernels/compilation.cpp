//
// Copyright Intel Corporation.
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

#include "vpux/compiler/act_kernels/compilation.h"

#include "vpux/utils/core/small_string.hpp"

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

static void getActShaveBinaries(const ActShaveCompileParams& params, const CompilationUnitDesc& unitDesc,
                                SmallVector<uint8_t>& textBinary, SmallVector<uint8_t>& dataBinary) {
    SmallString genDir;
    std::string ovBuildDir;
    if (sys::fs::exists(LIBRARY_OUTPUT_DIRECTORY)) {
        ovBuildDir = std::string(sys::path::parent_path(LIBRARY_OUTPUT_DIRECTORY));
        std::cout << "LIBRARY_OUTPUT_DIRECTORY" << std::endl;
    } else {
        // probe for OV_BUILD_DIR
        ovBuildDir = std::getenv("OV_BUILD_DIR");
        std::cout << "OV_BUILD_DIR " << ovBuildDir << "exists: " << sys::fs::exists(ovBuildDir) << std::endl;
        VPUX_THROW_UNLESS(ovBuildDir.c_str(),
                          "OV_BUILD_DIR env directory must be specified, in order to reach act-shave kernels");
        VPUX_THROW_UNLESS(sys::fs::exists(ovBuildDir),
                          "OpenVino build directory {0}, taken from OV_BUILD_DIR env variable is not exist", genDir);
    }
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
    std::replace(ovBuildDir.begin(), ovBuildDir.end(), '/', '\\');
#endif

    genDir = ovBuildDir;
    auto libDir = genDir;
    sys::path::append(libDir, "lib");
    auto inLibDir = libDir;
    sys::path::append(inLibDir, "act-kernels-build");
    std::cout << "lib/act-kernels-build " << inLibDir.c_str() << " exists: " << sys::fs::exists(inLibDir) << std::endl;

    inLibDir = libDir;
    sys::path::append(inLibDir, "kmb_custom_ocl_kernels");
    std::cout << "lib/kmb_custom_ocl_kernels " << inLibDir.c_str() << " exists: " << sys::fs::exists(inLibDir)
              << std::endl;

    inLibDir = libDir;
    sys::path::append(inLibDir, "mcm_config");
    std::cout << "lib/mcm_config " << inLibDir.c_str() << " exists: " << sys::fs::exists(inLibDir) << std::endl;

    genDir = ovBuildDir;
    sys::path::append(genDir, "act-kernels");

    std::cout << "act-kernels dir " << genDir.c_str() << " exists: " << sys::fs::exists(genDir) << std::endl;
    VPUX_THROW_UNLESS(sys::fs::exists(genDir), "{0}} directory is not exist", genDir);
    std::string ls = "ls ";
    ls = ls + genDir.c_str();
    system(ls.c_str());
    ls = "dir ";
    ls = ls + genDir.c_str();
    system(ls.c_str());
    std::string entryPoint = unitDesc.entry.str();

    SmallString prebuiltKernelBinariesPath(genDir);

    SmallString prebuiltKernelText(prebuiltKernelBinariesPath);
    sys::path::append(prebuiltKernelText, "sk." + entryPoint + "." + params.cpu + ".text");
    SmallString prebuiltKernelData(prebuiltKernelBinariesPath);
    sys::path::append(prebuiltKernelData, "sk." + entryPoint + "." + params.cpu + ".data");

    auto readBinary = [](SmallString& path, SmallVector<uint8_t>& buffer, uint32_t alignment = 1) {
        std::string err;
        auto elfFile = mlir::openInputFile(path, &err);
        if (!elfFile) {
            std::cout << "Could not open " << path.c_str() << " err: " << err.c_str() << std::endl;
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

    // Use moviCompile to compile and link C source code into an ELF binary.
    // and then using objcopy teardown elf into text and data sections
    readBinary(prebuiltKernelText, textBinary, 0x10);
    readBinary(prebuiltKernelData, dataBinary, 0x10);
}

ActKernelDesc compileKernelForACTShave(const CompilationUnitDesc& unitDesc, const ActShaveCompileParams& params) {
    SmallVector<uint8_t> textBinary;
    SmallVector<uint8_t> dataBinary;
    getActShaveBinaries(params, unitDesc, textBinary, dataBinary);

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

const CompilationUnitDesc& managementKernelCompilationDesc() {
    static const CompilationUnitDesc unitDesc{
            "nnActEntry",
            "nnActEntry",
    };

    return unitDesc;
}

ActKernelDesc compileManagementKernelForACTShave(const ActShaveCompileParams& params) {
    const auto& unitDesc = managementKernelCompilationDesc();

    return compileKernelForACTShave(unitDesc, params);
}

}  // namespace vpux
