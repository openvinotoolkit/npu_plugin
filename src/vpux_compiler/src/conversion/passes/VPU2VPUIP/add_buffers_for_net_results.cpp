//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

// Updates the func op and entry block.
//
// Any args appended to the entry block are added to `appendedEntryArgs`.
void updateFuncOp(mlir::func::FuncOp func, SmallVectorImpl<mlir::BlockArgument>& appendedEntryArgs) {
    auto functionType = func.getFunctionType();

    // Add the new arguments to the function type.
    auto newArgTypes =
            to_small_vector(llvm::concat<const mlir::Type>(functionType.getInputs(), functionType.getResults()));
    auto newFunctionType = mlir::FunctionType::get(func.getContext(), newArgTypes, functionType.getResults());
    func.setType(newFunctionType);

    const auto numInputs = functionType.getNumInputs();
    for (auto resultType : functionType.getResults() | indexed) {
        // Transfer the result attributes to arg attributes.
        const auto idx = checked_cast<unsigned>(resultType.index());
        func.setArgAttrs(numInputs + idx, func.getResultAttrs(idx));

        // Add the new arguments to the function type.
        auto newArg = func.front().addArgument(resultType.value(), func.getLoc());
        appendedEntryArgs.push_back(newArg);
    }
}

// Updates all ReturnOps in the scope of the given FuncOp by  copying the associated buffer contents into the given
// out-params.
void updateReturnOps(mlir::func::FuncOp func, ArrayRef<mlir::BlockArgument> appendedEntryArgs) {
    func.walk([&](mlir::func::ReturnOp op) {
        mlir::OpBuilder builder(op);
        for (auto i : irange(op.getNumOperands())) {
            auto copyOp = builder.create<VPUIP::CopyOp>(op.getLoc(), op.getOperand(i), appendedEntryArgs[i]);
            op.setOperand(i, copyOp.getOutput());
        }
    });
}

// Updates call op
void updateCallOp(mlir::ModuleOp module) {
    module.walk([&](mlir::func::CallOp callOp) {
        mlir::OpBuilder builder(callOp);

        SmallVector<mlir::Value> outParams;
        SmallVector<mlir::Value> currentResults;
        SmallVector<mlir::Type> resultTypes;
        for (auto result : callOp.getResults()) {
            auto resType = result.getType().dyn_cast<mlir::MemRefType>();
            VPUX_THROW_WHEN(resType == nullptr, "Only MemRefType is supported for now");

            auto outParam = builder.create<mlir::memref::AllocOp>(callOp.getLoc(), resType);
            outParams.push_back(outParam);

            currentResults.push_back(result);
            resultTypes.push_back(resType);
        }

        auto newOperands = to_vector(callOp.getOperands());
        newOperands.append(outParams.begin(), outParams.end());

        auto newCallOp =
                builder.create<mlir::func::CallOp>(callOp.getLoc(), callOp.getCalleeAttr(), resultTypes, newOperands);

        for (const auto& [result, newResult] : zip(currentResults, newCallOp.getResults())) {
            result.replaceAllUsesWith(newResult);
        }

        callOp.erase();
    });
}

//
// AddBuffersForNetResults
//

class AddBuffersForNetResults final : public AddBuffersForNetResultsBase<AddBuffersForNetResults> {
public:
    explicit AddBuffersForNetResults(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

//
// safeRunOnFunc
//

void AddBuffersForNetResults::safeRunOnModule() {
    auto module = getOperation();

    for (auto func : module.getOps<mlir::func::FuncOp>()) {
        if (func.isExternal()) {
            _log.trace("Can't convert external Function '@{0}'", func.getSymName());
            signalPassFailure();
        }

        SmallVector<mlir::BlockArgument> appendedEntryArgs;
        updateFuncOp(func, appendedEntryArgs);
        updateReturnOps(func, appendedEntryArgs);
    }

    updateCallOp(module);
}

}  // namespace

//
// createAddBuffersForNetResults
//

std::unique_ptr<mlir::Pass> vpux::createAddBuffersForNetResults(Logger log) {
    return std::make_unique<AddBuffersForNetResults>(log);
}
