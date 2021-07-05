//
// Copyright 2021 Intel Corporation.
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

#include <llvm/ADT/SetVector.h>

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/RegionUtils.h>

#include "vpux/compiler/edsl/passes.hpp"
#include "vpux/compiler/edsl/utils.hpp"

#ifdef ENABLE_PLAIDML
#include "pmlc/dialect/pxa/ir/ops.h"
#endif

using namespace mlir;          // NOLINT
using namespace mlir::memref;  // NOLINT

#ifdef ENABLE_PLAIDML
namespace pxa = pmlc::dialect::pxa;
#endif

namespace vpux {
namespace edsl {

struct SinkScalarPass : public SinkScalarBase<SinkScalarPass> {
    void runOnFunction() final {
#ifdef ENABLE_PLAIDML
        auto func = getFunction();
        // If the return value is not defined by an AffineParallelOp, we need to
        // wrap it.
        for (auto returnOp : func.getOps<ReturnOp>()) {
            wrapReturnValue(returnOp);
        }
        for (AffineParallelOp band : func.getOps<AffineParallelOp>()) {
            llvm::SetVector<Value> sinkCandidates;
            getUsedValuesDefinedAbove(band.getLoopBody(), sinkCandidates);

            llvm::SetVector<Operation*> sunkOperations;
            for (Value candidate : sinkCandidates) {
                Operation* definingOp = candidate.getDefiningOp();
                collectScalarOps(definingOp, &sunkOperations);
            }

            BlockAndValueMapping map;
            OpBuilder builder(band.getLoopBody());
            for (Operation* op : sunkOperations) {
                Operation* clonedOp = builder.clone(*op, map);
                for (auto pair : llvm::zip(op->getResults(), clonedOp->getResults())) {
                    replaceAllUsesInRegionWith(std::get<0>(pair), std::get<1>(pair), band.getLoopBody());
                }
            }
        }
#else
        VPUX_THROW("SinkScalarPass is only supported when ENABLE_PLAIDML=ON");
#endif
    }

#ifdef ENABLE_PLAIDML
    void wrapReturnValue(ReturnOp returnOp) {
        FuncOp program = getFunction();
        for (auto operand : returnOp.getOperands()) {
            Operation* def = operand.getDefiningOp();
            if (isa<AffineParallelOp>(def)) {
                continue;
            }
            if (auto reduceOp = dyn_cast<pxa::PxaReduceOp>(def)) {
                // Wrap the reduce op with an AffineParallelOp.
                OpBuilder builder(returnOp);
                auto loop = builder.create<AffineParallelOp>(reduceOp.getLoc(), reduceOp.getResult().getType(),
                                                             AtomicRMWKind::assign, ArrayRef<int64_t>{});
                reduceOp.getResult().replaceAllUsesWith(loop.getResult(0));
                builder.setInsertionPointToStart(loop.getBody());
                auto yieldOp =
                        builder.create<AffineYieldOp>(builder.getUnknownLoc(), ArrayRef<Value>{reduceOp.getResult()});
                reduceOp.getOperation()->moveBefore(yieldOp);
            } else {
                returnOp.emitOpError("Unexpected definition of return operand.");
                return;
            }
        }
    }

    void collectScalarOps(Operation* op, llvm::SetVector<Operation*>* into) {
        if (!op || isa<AllocOp, pxa::PxaReduceOp, AffineParallelOp>(op) || into->count(op)) {
            return;
        }

        for (Value operand : op->getOperands()) {
            Operation* definingOp = operand.getDefiningOp();
            collectScalarOps(definingOp, into);
        }
        into->insert(op);
    }
#endif
};

std::unique_ptr<mlir::Pass> createSinkScalarPass() {
    return std::make_unique<SinkScalarPass>();
}

}  // namespace edsl
}  // namespace vpux
