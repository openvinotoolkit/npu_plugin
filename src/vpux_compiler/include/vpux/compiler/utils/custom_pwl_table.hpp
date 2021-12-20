#pragma once

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <string>
#include <unordered_map>
#include <vector>
#include "vpux/compiler/utils/types.hpp"

namespace vpux {

struct PWLTableType {
    std::string activation;
    mlir::Type dtype;
};

struct PWLTableEntry {
    std::vector<int> range;
    std::vector<int> shift;
    std::vector<int> bias;
    std::pair<double, double> floatRange;
    int postShift;
};

struct PWLTableHash {
    std::size_t operator()(const PWLTableType& key) const {
        const auto h1 = std::hash<std::string>()(key.activation);
        const auto h2 = mlir::hash_value(key.dtype);

        return h1 ^ h2;
    }
};

struct PWLTableEq {
    bool operator()(const PWLTableType& key1, const PWLTableType& key2) const {
        return key1.activation == key2.activation && key1.dtype == key2.dtype;
    }
};

typedef std::unordered_map<PWLTableType, std::vector<PWLTableEntry>, PWLTableHash, PWLTableEq> PWLTableMap;

PWLTableMap* customPWLTable_leakyRelu();
}  // namespace vpux