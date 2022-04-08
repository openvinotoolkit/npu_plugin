//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/act_kernels/compilation.h"

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/numeric.hpp"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4267)  // size_t to integer conversion
#endif

#include "vpux/compiler/act_kernels/shave_binary_resources.h"

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

ActKernelDesc compileKernelForACTShave(const CompilationUnitDesc& unitDesc) {
    auto& kernelInfo = ShaveBinaryResources::getInstance();

    const std::string cpu = "3720xx";

    auto textBinary = kernelInfo.getText(unitDesc.entry, cpu);
    auto dataBinary = kernelInfo.getData(unitDesc.entry, cpu);

    ActKernelDesc result;
    auto dataName = std::string(unitDesc.name) + ".data";

    // A copy is made for each vector in order not to modify their original content when padding is added
    result.text = {unitDesc.name.data(), to_small_vector(textBinary), textBinary.size()};
    result.data = {dataName, to_small_vector(dataBinary), dataBinary.size()};

    // lets pad textBinary by 1K array at the end with FC CC FC CC
    for (int i = 0; i != 512; i++) {
        result.text.data.push_back(0xFC);
        result.text.data.push_back(0xCC);
    }

    return result;
}

const CompilationUnitDesc& managementKernelCompilationDesc() {
    static const CompilationUnitDesc unitDesc{
            "nnActEntry",
            "nnActEntry",
    };

    return unitDesc;
}

ActKernelDesc compileManagementKernelForACTShave() {
    const auto& unitDesc = managementKernelCompilationDesc();

    return compileKernelForACTShave(unitDesc);
}

}  // namespace vpux
