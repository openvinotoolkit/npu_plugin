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

static void getActShaveBinaries(const movitools::MoviCompileParams& params, const CompilationUnitDesc& unitDesc,
                                SmallVector<uint8_t, 128>& textBinary, SmallVector<uint8_t, 128>& dataBinary) {
    SmallString<128> genDir;
    genDir = LIBRARY_OUTPUT_DIRECTORY;
    sys::path::append(genDir, "act-kernels");

    VPUX_THROW_UNLESS(sys::fs::exists(genDir), "act-kernels directory is not exist in {0}", LIBRARY_OUTPUT_DIRECTORY);

    std::string entryPoint = unitDesc.entry.str();

    SmallString<128> prebuiltKernelBinariesPath(genDir);

    SmallString<128> prebuiltKernelText(prebuiltKernelBinariesPath);
    sys::path::append(prebuiltKernelText, "sk." + entryPoint + "." + params.cpu + ".text");
    SmallString<128> prebuiltKernelData(prebuiltKernelBinariesPath);
    sys::path::append(prebuiltKernelData, "sk." + entryPoint + "." + params.cpu + ".data");

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

    readBinary(prebuiltKernelText, textBinary, 0x10);
    readBinary(prebuiltKernelData, dataBinary, 0x10);
}

ActKernelDesc compileKernelForACTShave(const CompilationUnitDesc& unitDesc,
                                       const movitools::MoviCompileParams& params) {
    // Use moviCompile to compile and link C source code into an ELF binary.
    // and then using objcopy teardown elf into text and data sections
    SmallVector<uint8_t, 128> textBinary;
    SmallVector<uint8_t, 128> dataBinary;
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

}  // namespace vpux
