//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <llvm/ADT/STLExtras.h>
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

template <typename R, typename Comparator>
auto min_element(R&& Range, Comparator Comp) {
    return std::min_element(llvm::adl_begin(Range), llvm::adl_end(Range), Comp);
}

template <typename R, typename Comparator>
auto max_element(R&& Range, Comparator Comp) {
    return std::max_element(llvm::adl_begin(Range), llvm::adl_end(Range), Comp);
}

}  // namespace vpux
