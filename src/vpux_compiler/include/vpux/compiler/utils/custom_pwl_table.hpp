#pragma once

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <string>
#include <unordered_map>
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/utils/types.hpp"

namespace vpux {

struct PWLTableType {
    llvm::StringRef activation;
    mlir::Type dtype;
};

struct PWLTableEntry {
    SmallVector<int> range;
    SmallVector<int> shift;
    SmallVector<int> bias;
    std::pair<double, double> floatRange;
    int postShift;
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

vpux::PWLTableEntry getLeakyReluPWLEntry(mlir::Type outElemType);
Optional<vpux::PWLTableEntry> findCustomPWLTable(const StringRef activationName, mlir::Type outElemType);
}  // namespace vpux
