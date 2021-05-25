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

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/dialect/IERT/passes.hpp"

#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

// Updates the func op and entry block.
//
// Any args appended to the entry block are added to `appendedEntryArgs`.
void updateFuncOp(mlir::FuncOp func, SmallVectorImpl<mlir::BlockArgument>& appendedEntryArgs) {
    auto functionType = func.getType();

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
        auto newArg = func.front().addArgument(resultType.value());
        appendedEntryArgs.push_back(newArg);
    }
}

// Updates all ReturnOps in the scope of the given FuncOp by  copying the associated buffer contents into the given
// out-params.
void updateReturnOps(mlir::FuncOp func, ArrayRef<mlir::BlockArgument> appendedEntryArgs) {
    func.walk([&](mlir::ReturnOp op) {
        mlir::OpBuilder builder(op);
        for (auto i : irange(op.getNumOperands())) {
            auto copyOp = builder.create<IERT::CopyOp>(op.getLoc(), op.getOperand(i), appendedEntryArgs[i]);
            op.setOperand(i, copyOp.output());
        }
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

    for (auto func : module.getOps<mlir::FuncOp>()) {
        if (func.isExternal()) {
            _log.trace("Can't convert external Function '@{0}'", func.sym_name());
            signalPassFailure();
        }

        SmallVector<mlir::BlockArgument> appendedEntryArgs;
        updateFuncOp(func, appendedEntryArgs);
        updateReturnOps(func, appendedEntryArgs);
    }
}

}  // namespace

//
// createAddBuffersForNetResults
//

std::unique_ptr<mlir::Pass> vpux::createAddBuffersForNetResults(Logger log) {
    return std::make_unique<AddBuffersForNetResults>(log);
}
