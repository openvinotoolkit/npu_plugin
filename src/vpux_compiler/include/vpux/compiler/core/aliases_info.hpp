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

#include <llvm/ADT/SmallPtrSet.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>

namespace vpux {

class AliasesInfo {
public:
    using ValuesSet = llvm::SmallPtrSet<mlir::Value, 16>;
    using AliasesMap = llvm::DenseMap<mlir::Value, ValuesSet>;
    using ValuesMap = llvm::DenseMap<mlir::Value, mlir::Value>;

public:
    explicit AliasesInfo(mlir::FuncOp func);

    const ValuesSet& getAliases(mlir::Value val) const;
    mlir::Value getRoot(mlir::Value val) const;

private:
    AliasesMap _aliases;
    ValuesMap _roots;
};

}  // namespace vpux
