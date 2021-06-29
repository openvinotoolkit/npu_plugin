#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include "include/mcm/computation/model/iterator/tensor.hpp"

namespace mv {
struct PWLTableType {
    std::string activation;
    mv::DType dtype;
};

struct PWLTableEntry {
    std::vector<int> range;
    std::vector<int> shift;
    std::vector<int> bias;
    std::pair<double, double> float_range;
    int post_shift;
};

struct PWLTableHash {
    std::size_t operator()(const PWLTableType& key) const {
        const auto h1 = std::hash<std::string>()(key.activation);
        const auto h2 = std::hash<std::string>()(key.dtype.toString());

        return h1 ^ h2;
    }
};

struct PWLTableEq {
    bool operator()(const PWLTableType& key1, const PWLTableType& key2) const {
        return key1.activation == key2.activation && key1.dtype == key2.dtype;
    }
};

typedef std::unordered_map<PWLTableType, std::vector<PWLTableEntry>, PWLTableHash, PWLTableEq> PWLTableMap;
}  // namespace mv
