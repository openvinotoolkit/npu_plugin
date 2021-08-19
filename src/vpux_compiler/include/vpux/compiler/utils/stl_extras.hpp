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

#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include <map>
#include <set>

namespace vpux {

struct OpOrderCmp final {
    bool operator()(mlir::Operation* lhs, mlir::Operation* rhs) const;
};

struct ValueOrderCmp final {
    bool operator()(mlir::Value lhs, mlir::Value rhs) const;
};

using OpOrderedSet = std::set<mlir::Operation*, OpOrderCmp>;
using ValueOrderedSet = std::set<mlir::Value, ValueOrderCmp>;

template <typename T>
using OpOrderedMap = std::map<mlir::Operation*, T, OpOrderCmp>;

template <typename T>
using ValueOrderedMap = std::map<mlir::Value, T, ValueOrderCmp>;

}  // namespace vpux
