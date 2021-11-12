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

#include "vpux/utils/core/logger.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>

#include <llvm/ADT/SmallPtrSet.h>

namespace vpux {

class AliasesInfo {
public:
    using ValuesSet = llvm::SmallPtrSet<mlir::Value, 16>;
    using AliasesMap = llvm::DenseMap<mlir::Value, ValuesSet>;
    using ValuesMap = llvm::DenseMap<mlir::Value, mlir::Value>;
    using OpRange = llvm::iterator_range<mlir::Region::OpIterator>;

public:
    explicit AliasesInfo(mlir::FuncOp func);

    // Will return NULL if the `val` is a root value.
    mlir::Value getSource(mlir::Value val) const;

    // Will return `val` itlsef if the `val` is a root value.
    mlir::Value getRoot(mlir::Value val) const;

    // The `val` must be a root value.
    const ValuesSet& getAllAliases(mlir::Value val) const;
    void addAlias(mlir::Value source, mlir::Value alias);

private:
    void traverse(OpRange ops);

private:
    ValuesMap _sources;  // closest source of the alias
    ValuesMap _roots;    // top-root of the alias

    AliasesMap _allAliases;  // all aliases, direct and indirect

    Logger _log;
};

}  // namespace vpux
