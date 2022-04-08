//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/utils/core/dense_map.hpp"
#include "vpux/utils/core/logger.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>

#include <llvm/ADT/SmallPtrSet.h>

namespace vpux {

class AliasesInfo {
public:
    using ValuesSet = llvm::SmallPtrSet<mlir::Value, 16>;
    using ValuesMap = DenseMap<mlir::Value, ValuesSet>;
    using OpRange = llvm::iterator_range<mlir::Region::OpIterator>;

public:
    explicit AliasesInfo(mlir::FuncOp func);

    // Returns the sources of a value.
    // Will return an empty set if `val` is a root value.
    const ValuesSet& getSources(mlir::Value val) const;

    // Returns the source of a value. The value is expected to have only one source, otherwise will throw an error.
    // Will return NULL if `val` is a root value.
    mlir::Value getSource(mlir::Value val) const;

    // Returns the roots of a value.
    // The set will contain `val` if it is a root value.
    const ValuesSet& getRoots(mlir::Value val) const;

    // The `val` must be a root value.
    const ValuesSet& getAllAliases(mlir::Value val) const;
    void addAlias(mlir::Value source, mlir::Value alias);

private:
    void traverse(OpRange ops);

private:
    ValuesMap _sources;  // closest source of the alias
    ValuesMap _roots;    // top-root of the alias

    ValuesMap _allAliases;  // all aliases, direct and indirect

    Logger _log;
};

}  // namespace vpux
