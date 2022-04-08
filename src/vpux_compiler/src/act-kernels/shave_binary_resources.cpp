//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/act_kernels/shave_binary_resources.h"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"

#include <string>
#include <unordered_map>
#include <vector>

using namespace vpux;

extern std::unordered_map<std::string, const std::vector<uint8_t>> shaveBinaryResourcesMap;

const ShaveBinaryResources& ShaveBinaryResources::getInstance() {
    static ShaveBinaryResources instance;
    return instance;
}

llvm::ArrayRef<uint8_t> ShaveBinaryResources::getText(llvm::StringRef entry, llvm::StringRef cpu) const {
    auto symbolName = printToString("sk_{0}_{1}_text", entry, cpu);
    auto data = shaveBinaryResourcesMap.find(symbolName);

    VPUX_THROW_UNLESS(data != shaveBinaryResourcesMap.end(),
                      "Can't find '.text' section for kernel '{0}' and cpu '{1}'", entry, cpu);

    return data->second;
}

llvm::ArrayRef<uint8_t> ShaveBinaryResources::getData(llvm::StringRef entry, llvm::StringRef cpu) const {
    auto symbolName = printToString("sk_{0}_{1}_data", entry, cpu);
    auto data = shaveBinaryResourcesMap.find(symbolName);

    VPUX_THROW_UNLESS(data != shaveBinaryResourcesMap.end(),
                      "Can't find '.data' section for kernel '{0}' and cpu '{1}'", entry, cpu);

    return data->second;
}

llvm::ArrayRef<uint8_t> ShaveBinaryResources::getElf(llvm::StringRef entry, llvm::StringRef cpu) const {
    auto symbolName = printToString("{0}_{1}_elf", entry, cpu);
    auto data = shaveBinaryResourcesMap.find(symbolName);

    VPUX_THROW_UNLESS(data != shaveBinaryResourcesMap.end(), "Can't find 'elf' for kernel '{0}' and cpu '{1}'", entry,
                      cpu);

    return data->second;
}
