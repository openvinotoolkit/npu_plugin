//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>

#include <cstdint>

namespace vpux {

class ShaveBinaryResources {
public:
    static const ShaveBinaryResources& getInstance();

private:
    ShaveBinaryResources() = default;

public:
    ShaveBinaryResources(ShaveBinaryResources const&) = delete;
    void operator=(ShaveBinaryResources const&) = delete;

    llvm::ArrayRef<uint8_t> getData(llvm::StringRef entryPoint, llvm::StringRef cpu) const;
    llvm::ArrayRef<uint8_t> getText(llvm::StringRef entryPoint, llvm::StringRef cpu) const;
    llvm::ArrayRef<uint8_t> getElf(llvm::StringRef entryPoint, llvm::StringRef cpu) const;
};

}  // namespace vpux
