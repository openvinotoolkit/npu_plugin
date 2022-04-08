//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/IERT/passes.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

using namespace vpux;
namespace {

//
// OptimizeParallelCopiesPass
//

class OptimizeParallelCopiesPass final : public IERT::OptimizeParallelCopiesBase<OptimizeParallelCopiesPass> {
public:
    explicit OptimizeParallelCopiesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

bool isCopyFusable(IERT::CopyOp copyOp) {
    // Check 1: copy DDR->CMX
    const auto srcMemory = copyOp.input().getType().cast<vpux::NDTypeInterface>().getMemoryKind();
    const auto dstMemory = copyOp.output().getType().cast<vpux::NDTypeInterface>().getMemoryKind();
    if (srcMemory == dstMemory || srcMemory == VPU::MemoryKind::CMX_NN) {
        return false;
    }

    // Check 2: parallel
    // All the consumers of the parent op should be copies
    // At least one more copy except for the current one
    auto parentOp = copyOp.input().getDefiningOp();
    if (parentOp == nullptr) {
        return false;
    }

    // Temporally disable the optimization for constant nodes, like weights, activation_windows, etc.
    // [Track number: E#28833]
    if (mlir::isa<Const::DeclareOp>(parentOp)) {
        return false;
    }
    bool hasSubView = false;
    auto subViewOp = mlir::dyn_cast<IERT::SubViewOp>(copyOp.input().getDefiningOp());
    if (mlir::isa<IERT::SubViewOp>(parentOp)) {
        hasSubView = true;
        parentOp = mlir::dyn_cast<IERT::SubViewOp>(parentOp).source().getDefiningOp();
    }
    bool hasSiblingCopy = false;
    if (parentOp == nullptr || parentOp->getNumResults() <= 0) {
        return false;
    }
    for (auto siblingOp : parentOp->getResult(0).getUsers()) {
        if (!mlir::isa<IERT::CopyOp>(*siblingOp)) {
            if (!mlir::isa<IERT::SubViewOp>(*siblingOp)) {
                continue;
            } else {
                auto childOfSiblingOp = *siblingOp->getResult(0).getUsers().begin();
                if (!mlir::isa<IERT::CopyOp>(childOfSiblingOp)) {
                    continue;
                }
                // match SubView->Copy
                if (!hasSubView) {
                    continue;
                }
                auto siblingSubViewOp = mlir::dyn_cast<IERT::SubViewOp>(siblingOp);
                if (parseIntArrayAttr<int64_t>(subViewOp.static_offsets()) !=
                            parseIntArrayAttr<int64_t>(siblingSubViewOp.static_offsets()) ||
                    parseIntArrayAttr<int64_t>(subViewOp.static_sizes()) !=
                            parseIntArrayAttr<int64_t>(siblingSubViewOp.static_sizes())) {
                    continue;
                }
                siblingOp = childOfSiblingOp;
            }
        }

        // Check 3: current op's consumers are copied to DDR immediately after execution
        for (const auto childOfSiblingOp : siblingOp->getResult(0).getUsers()) {
            for (const auto grandChildOfSiblingOp : childOfSiblingOp->getResult(0).getUsers()) {
                auto childCopyOfSiblingOp = mlir::dyn_cast<IERT::CopyOp>(grandChildOfSiblingOp);
                if (childCopyOfSiblingOp == nullptr) {
                    return false;
                }
                const auto input = childCopyOfSiblingOp.input().getType().cast<vpux::NDTypeInterface>();
                const auto output = childCopyOfSiblingOp.output().getType().cast<vpux::NDTypeInterface>();
                if (input.getMemoryKind() != VPU::MemoryKind::CMX_NN ||
                    output.getMemoryKind() != VPU::MemoryKind::DDR) {
                    return false;
                }
            }
        }

        if (siblingOp != copyOp) {
            hasSiblingCopy = true;
        }
    }

    return hasSiblingCopy;
}

mlir::LogicalResult fuseParallelCopyOp(IERT::CopyOp copyOp, Logger log) {
    auto parentOp = copyOp.input().getDefiningOp();
    if (parentOp == nullptr) {
        return mlir::failure();
    }

    if (mlir::isa<IERT::SubViewOp>(parentOp)) {
        auto subViewOp = mlir::dyn_cast<IERT::SubViewOp>(parentOp);
        parentOp = subViewOp.source().getDefiningOp();
        auto getSiblingSubViews = [&]() -> SmallVector<IERT::SubViewOp> {
            SmallVector<IERT::SubViewOp> res;
            for (auto siblingOp : parentOp->getResult(0).getUsers()) {
                if (mlir::isa<IERT::SubViewOp>(*siblingOp) && siblingOp != subViewOp) {
                    auto siblingSubViewOp = mlir::dyn_cast<IERT::SubViewOp>(*siblingOp);
                    if (siblingSubViewOp.static_offsets() == subViewOp.static_offsets() &&
                        siblingSubViewOp.static_sizes() == subViewOp.static_sizes()) {
                        res.push_back(mlir::dyn_cast<IERT::SubViewOp>(*siblingOp));
                    }
                }
            }
            return res;
        };
        auto siblingSubViews = getSiblingSubViews();
        for (auto siblingSubView : siblingSubViews) {
            log.trace("Fuse SubView op {0} to {1}", subViewOp->getLoc(), siblingSubView->getLoc());
            auto siblingCopy = mlir::dyn_cast<IERT::CopyOp>(*siblingSubView.result().getUsers().begin());

            siblingSubView.getOperation()->replaceAllUsesWith(subViewOp.getOperation());
            siblingCopy.getOperation()->replaceAllUsesWith(copyOp);
            siblingCopy->erase();
            siblingSubView->erase();
        }

        return mlir::success();
    }

    auto getSiblingCopies = [&]() -> SmallVector<IERT::CopyOp> {
        SmallVector<IERT::CopyOp> res;
        for (auto siblingOp : parentOp->getResult(0).getUsers()) {
            if (mlir::isa<IERT::CopyOp>(*siblingOp) && siblingOp != copyOp) {
                res.push_back(mlir::dyn_cast<IERT::CopyOp>(*siblingOp));
            }
        }
        return res;
    };
    auto siblingCopies = getSiblingCopies();
    for (auto siblingCopy : siblingCopies) {
        log.trace("Fuse copy op {0} to {1}", copyOp->getLoc(), siblingCopy->getLoc());
        siblingCopy.getOperation()->replaceAllUsesWith(copyOp.getOperation());
        siblingCopy->erase();
    }

    return mlir::success();
}

// safeRunOnFunc

void OptimizeParallelCopiesPass::safeRunOnFunc() {
    getFunction()->walk([&](IERT::CopyOp copyOp) {
        if (isCopyFusable(copyOp)) {
            _log.nest(1).trace("Fuse parallel copy op '{0}' at '{1}'", copyOp->getName(), copyOp->getLoc());
            if (mlir::failed(fuseParallelCopyOp(copyOp, _log))) {
                _log.nest(1).trace("Failed copy fusion of {0} at {1}", copyOp->getName(), copyOp->getLoc());
            }
        }
    });
}
}  // namespace

//
// createOptimizeParallelCopiesPass
//

std::unique_ptr<mlir::Pass> vpux::IERT::createOptimizeParallelCopiesPass(Logger log) {
    return std::make_unique<OptimizeParallelCopiesPass>(log);
}
