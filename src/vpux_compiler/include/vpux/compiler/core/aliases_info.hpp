//
// Copyright Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
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

    mlir::Value getRoot(mlir::Value val) const;
    const ValuesSet& getAliases(mlir::Value val) const;

private:
    void addAlias(mlir::Value root, mlir::Value alias);
    void traverse(OpRange ops);

private:
    AliasesMap _aliases;
    ValuesMap _roots;
    Logger _log;
};

}  // namespace vpux
