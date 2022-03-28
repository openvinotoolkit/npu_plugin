//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/act_kernels/compilation.h"

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <file_utils.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4267)  // size_t to integer conversion
#endif

#include <llvm/Support/FileSystem.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <algorithm>
#include <fstream>
#include <string>

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
    const auto prebuiltKernelBinariesPath =
            printToString("{0}/vpux/act_shave_bin", InferenceEngine::getIELibraryPath());
    VPUX_THROW_UNLESS(llvm::sys::fs::exists(prebuiltKernelBinariesPath), "'{0}' directory is not exist",
                      prebuiltKernelBinariesPath);

    std::string prebuiltKernelText;
    std::string prebuiltKernelData;

    for (const auto& cpu : params.cpu) {
        prebuiltKernelText = printToString("{0}/sk.{1}.{2}.text", prebuiltKernelBinariesPath, unitDesc.entry, cpu);
        prebuiltKernelData = printToString("{0}/sk.{1}.{2}.data", prebuiltKernelBinariesPath, unitDesc.entry, cpu);

        if (llvm::sys::fs::exists(prebuiltKernelText) && llvm::sys::fs::exists(prebuiltKernelData)) {
            break;
        }
    }

    VPUX_THROW_UNLESS(llvm::sys::fs::exists(prebuiltKernelText), "Can't find '.text' part for kernel '{0}'",
                      unitDesc.entry);
    VPUX_THROW_UNLESS(llvm::sys::fs::exists(prebuiltKernelData), "Can't find '.data' part for kernel '{0}'",
                      unitDesc.entry);

    const auto readBinary = [](StringRef filePath, SmallVector<uint8_t>& buffer, uint64_t alignment = 1) {
        uint64_t fileSize = 0;
        const auto err = llvm::sys::fs::file_size(filePath, fileSize);
        VPUX_THROW_WHEN(err, "Can't get file '{0}' size : {1}", filePath, err.message());

        const auto bufSize = checked_cast<size_t>(alignVal(fileSize, alignment));

        buffer.clear();
        buffer.resize(bufSize, 0);

        std::ifstream elfFile(filePath.data(), std::ios_base::binary);
        VPUX_THROW_UNLESS(elfFile.is_open(), "Can't open file '{0}'", filePath);

        elfFile.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    };

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
