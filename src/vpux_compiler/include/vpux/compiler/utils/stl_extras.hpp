//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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
