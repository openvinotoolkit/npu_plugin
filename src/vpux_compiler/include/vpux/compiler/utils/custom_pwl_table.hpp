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

#pragma once

#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>

#include <string>
#include <unordered_map>

namespace vpux {

struct PWLTableType {
    StringRef activation;
    mlir::Type dtype;
};

struct PWLTableEntry {
    SmallVector<int32_t> range;
    SmallVector<int32_t> shift;
    SmallVector<int32_t> bias;
    std::pair<double, double> floatRange;
    int32_t postShift;
};

struct PWLTableHash {
    std::size_t operator()(const PWLTableType& key) const {
        const auto h1 = std::hash<std::string>()(key.activation.str());
        const auto h2 = mlir::hash_value(key.dtype);

        return h1 ^ h2;
    }
};

struct PWLTableEq {
    bool operator()(const PWLTableType& key1, const PWLTableType& key2) const {
        return key1.activation == key2.activation && key1.dtype == key2.dtype;
    }
};

using PWLTableMap = std::unordered_map<PWLTableType, SmallVector<PWLTableEntry>, PWLTableHash, PWLTableEq>;

vpux::PWLTableEntry getLeakyReluPWLEntry();
Optional<vpux::PWLTableEntry> findCustomPWLTable(StringRef activationName, mlir::Type outElemType);
}  // namespace vpux
