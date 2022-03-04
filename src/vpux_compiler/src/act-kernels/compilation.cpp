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
    if (sys::fs::exists(LIBRARY_OUTPUT_DIRECTORY)) {
        genDir = sys::path::parent_path(LIBRARY_OUTPUT_DIRECTORY);
    } else {
        // probe for OV_BUILD_DIR
        const auto ovBuildDir = std::getenv("OV_BUILD_DIR");
        VPUX_THROW_UNLESS(ovBuildDir,
                          "OV_BUILD_DIR env directory must be specified, in order to reach act-shave kernels");
        VPUX_THROW_UNLESS(sys::fs::exists(ovBuildDir),
                          "OpenVino build directory {0}, taken from OV_BUILD_DIR env variable is not exist", genDir);

        genDir = ovBuildDir;
    }
    sys::path::append(genDir, "act-kernels");

    VPUX_THROW_UNLESS(sys::fs::exists(genDir), "{0}} directory is not exist", genDir);

    std::string entryPoint = unitDesc.entry.str();

    SmallString prebuiltKernelBinariesPath(genDir);

    SmallString prebuiltKernelText;
    SmallString prebuiltKernelData;
    for (const auto cpu : params.cpu) {
        prebuiltKernelText = SmallString(prebuiltKernelBinariesPath);
        prebuiltKernelData = SmallString(prebuiltKernelBinariesPath);
        sys::path::append(prebuiltKernelText, "sk." + entryPoint + "." + cpu + ".text");
        sys::path::append(prebuiltKernelData, "sk." + entryPoint + "." + cpu + ".data");
        if (sys::fs::exists(prebuiltKernelText) && sys::fs::exists(prebuiltKernelData)) {
            break;
        }
    }

    auto readBinary = [](SmallString& path, SmallVector<uint8_t>& buffer, uint32_t alignment = 1) {
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
