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

#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

#include <llvm/ADT/BitVector.h>

namespace vpux {

class AsyncDepsInfo final {
public:
    explicit AsyncDepsInfo(mlir::FuncOp func);

public:
    void addDependency(mlir::async::ExecuteOp from, mlir::async::ExecuteOp to);
    void optimizeDepsMap();
    void updateTokenDependencies();

private:
    void setIndex(mlir::async::ExecuteOp execOp, uint64_t index);
    uint32_t getIndex(mlir::async::ExecuteOp execOp) const;

private:
    void buildDepsMap(mlir::FuncOp func);
    void addExecOp(mlir::async::ExecuteOp execOp);

private:
    Logger _log;

    mlir::Identifier _indexAttrName;

    SmallVector<mlir::async::ExecuteOp> _allExecOps;

    // indexOf(mlir::async::ExecuteOp) 'depends on' [ indexOf(mlir::async::ExecuteOp)... ].
    SmallVector<llvm::BitVector> _depsMap;
};

}  // namespace vpux
