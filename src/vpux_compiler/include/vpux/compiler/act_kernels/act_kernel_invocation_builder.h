//
// Copyright 2020 Intel Corporation.
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

#pragma once

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Value.h>
#include <vpux/utils/core/logger.hpp>
#include "Nce2p7.h"

namespace vpux {

/**
 * Helper builder for creation activation-shaves invocation in memory arguments
 */
class InvocationBuilder {

    llvm::SmallVector<char,  128> _storage;         // keeps basic elements
    llvm::SmallVector<char , 128> _arrayStorage;    // keeps arrays elements

    Logger _log;

    // keeps offset to patchable field within _storage structure that need to be
    // updated after _storage and _arrayStorage gets concatenated
    struct PatchPoint {
        std::function<void(MutableArrayRef<uint8_t>, size_t)> patchCallback;
        size_t offset;

        void patch(MutableArrayRef<uint8_t> resialStorage, size_t patchBase) const {
            patchCallback(resialStorage, patchBase + offset);
        }
    };
    llvm::SmallVector<PatchPoint , 128> _deferredPointers;

public:

    InvocationBuilder(Logger log) : _log(log){}

    /**
     * register serialisation for given invocation argument, might be MemRefType or any other supported types
     * @param operand
     */
    void addArg(mlir::Value operand);

    /*
     * actual serialising routine
     */
    llvm::SmallVector<uint8_t> store() const;

protected:

    template <class T>
    static void storeSimple(llvm::SmallVectorImpl<char> & storage, const T& anyValue) {
        ArrayRef<char> valueAsArray(reinterpret_cast<const char*>(&anyValue), sizeof(anyValue));
        storage.insert(storage.end(), valueAsArray.begin(), valueAsArray.end());
    }

    /**
     * create a patch entry, that can be further updated
     * @tparam U structure type
     * @tparam T field type
     */
    template <class U, class T>
    void createPatchPoint(const T& patcher) {
        auto fieldPatcher = [offset = _storage.size(), this, patcher] (MutableArrayRef<uint8_t> serialStorage, size_t updateTo) {
            auto& base = reinterpret_cast<U&>(*(serialStorage.begin() + offset));
            patcher(base, updateTo);
        };
        _deferredPointers.push_back({fieldPatcher, _arrayStorage.size()});
    }

    // memref serialisation
    void addMemrefArg(const mlir::Value & value);
};
}  // namespace vpux
