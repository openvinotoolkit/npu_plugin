//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
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

class OptimizeParallelCopiesPass final : public VPUIP::OptimizeParallelCopiesBase<OptimizeParallelCopiesPass> {
public:
    explicit OptimizeParallelCopiesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

// Utility function to extract VPUIP::CopyOp from NCEClusterTilingOp. If op isnt NCEClusterTilingOp or didnt contain
// CopyOp return nullptr
mlir::Operation* getCopyOpFromClusterTilingOp(mlir::Operation* op) {
    if (auto nceClusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(op)) {
        return mlir::dyn_cast<VPUIP::CopyOp>(nceClusterTilingOp.getInnerTaskOp());
    }
    return nullptr;
}

mlir::Operation* getParentOfCopyOp(VPUIP::CopyOp copyOp) {
    // Check is CopyOp wrapper by NCECluster task
    if (auto wrapperOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(copyOp->getParentOp())) {
        return wrapperOp.inputs()[0].getDefiningOp();
    }
    return copyOp.input().getDefiningOp();
}

// Utility function to get DistributedBufferType from NCEClusterTilingOp. If op isnt NCEClusterTilingOp or return type
// isnt DistributedBufferType return empty value
llvm::Optional<VPUIP::DistributedBufferType> getResultType(mlir::Operation* op) {
    if (auto nceClusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(op)) {
        const auto resType = nceClusterTilingOp->getResult(0).getType();
        if (resType.isa<VPUIP::DistributedBufferType>()) {
            return resType.cast<VPUIP::DistributedBufferType>();
        }
    }
    return {};
}

bool isCopyFusable(VPUIP::CopyOp copyOp) {
    // Check 1: copy DDR->CMX
    const auto srcMemory = copyOp.input().getType().cast<vpux::NDTypeInterface>().getMemoryKind();
    const auto dstMemory = copyOp.output().getType().cast<vpux::NDTypeInterface>().getMemoryKind();
    if (srcMemory == dstMemory || srcMemory == VPU::MemoryKind::CMX_NN) {
        return false;
    }

    // Check 2: parallel
    // All the consumers of the parent op should be copies
    // At least one more copy except for the current one
    auto parentOp = getParentOfCopyOp(copyOp);
    if (parentOp == nullptr) {
        return false;
    }

    // Temporally disable the optimization for constant nodes, like weights, activation_windows, etc.
    // [Track number: E#28833]
    if (mlir::isa<Const::DeclareOp>(parentOp)) {
        return false;
    }
    bool hasSubView = false;
    auto subViewOp = mlir::dyn_cast<VPUIP::SubViewOp>(parentOp);
    if (mlir::isa<VPUIP::SubViewOp>(parentOp)) {
        hasSubView = true;
        parentOp = mlir::dyn_cast<VPUIP::SubViewOp>(parentOp).source().getDefiningOp();
    }
    bool hasSiblingCopy = false;
    if (parentOp == nullptr || parentOp->getNumResults() <= 0) {
        return false;
    }
    for (auto siblingOp : parentOp->getResult(0).getUsers()) {
        if (auto wrappedCopyOp = getCopyOpFromClusterTilingOp(siblingOp)) {
            siblingOp = wrappedCopyOp;
        } else {
            if (!mlir::isa<VPUIP::CopyOp>(*siblingOp)) {
                if (!mlir::isa<VPUIP::SubViewOp>(*siblingOp)) {
                    continue;
                } else {
                    auto childOfSiblingOp = *siblingOp->getResult(0).getUsers().begin();
                    if (auto wrappedCopyOp = getCopyOpFromClusterTilingOp(childOfSiblingOp)) {
                        childOfSiblingOp = wrappedCopyOp;
                    }
                    if (!mlir::isa<VPUIP::CopyOp>(childOfSiblingOp)) {
                        continue;
                    }
                    // match SubView->Copy
                    if (!hasSubView) {
                        continue;
                    }
                    auto siblingSubViewOp = mlir::dyn_cast<VPUIP::SubViewOp>(siblingOp);
                    if (parseIntArrayAttr<int64_t>(subViewOp.static_offsets()) !=
                                parseIntArrayAttr<int64_t>(siblingSubViewOp.static_offsets()) ||
                        parseIntArrayAttr<int64_t>(subViewOp.static_sizes()) !=
                                parseIntArrayAttr<int64_t>(siblingSubViewOp.static_sizes())) {
                        continue;
                    }
                    siblingOp = childOfSiblingOp;
                }
            }
        }

        // Check 3: current op's consumers are copied to DDR immediately after execution
        for (const auto childOfSiblingOp : siblingOp->getResult(0).getUsers()) {
            for (const auto grandChildOfSiblingOp : childOfSiblingOp->getResult(0).getUsers()) {
                auto childCopyOfSiblingOp = mlir::dyn_cast<VPUIP::CopyOp>(grandChildOfSiblingOp);
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

mlir::LogicalResult fuseParallelCopyOp(VPUIP::CopyOp copyOp, Logger log) {
    auto parentOp = getParentOfCopyOp(copyOp);
    if (parentOp == nullptr) {
        return mlir::failure();
    }
    auto copyReplaceTarget = copyOp.getOperation();
    llvm::Optional<VPUIP::DistributedBufferType> srcOpResultType;
    if (auto nceClusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(copyOp->getParentOp())) {
        copyReplaceTarget = nceClusterTilingOp.getOperation();
        srcOpResultType = getResultType(nceClusterTilingOp);
    }

    // Optimize pattern parentOp -> SubView -> CopyOp
    if (mlir::isa<VPUIP::SubViewOp>(parentOp)) {
        auto subViewOp = mlir::dyn_cast<VPUIP::SubViewOp>(parentOp);
        parentOp = subViewOp.source().getDefiningOp();

        for (auto siblingOp : llvm::make_early_inc_range(parentOp->getResult(0).getUsers())) {
            auto siblingSubViewOp = mlir::dyn_cast<VPUIP::SubViewOp>(*siblingOp);
            if (siblingSubViewOp != nullptr && siblingOp != subViewOp) {
                if (siblingSubViewOp.static_offsets() == subViewOp.static_offsets() &&
                    siblingSubViewOp.static_sizes() == subViewOp.static_sizes()) {
                    log.trace("Fuse SubView op {0} to {1}", subViewOp->getLoc(), siblingSubViewOp->getLoc());
                    auto siblingCopyOp = *siblingSubViewOp.result().getUsers().begin();
                    if (!mlir::isa<VPUIP::CopyOp>(siblingCopyOp) &&
                        getCopyOpFromClusterTilingOp(siblingCopyOp) == nullptr) {
                        log.trace("The child of SubViewOp is not Copy Op.");
                        return mlir::failure();
                    }

                    siblingSubViewOp.getOperation()->replaceAllUsesWith(subViewOp.getOperation());
                    siblingCopyOp->replaceAllUsesWith(copyReplaceTarget);
                    siblingCopyOp->erase();
                    siblingSubViewOp->erase();
                }
            }
        }

        return mlir::success();
    }

    // Optimize pattern parentOp -> CopyOp
    if (!srcOpResultType.hasValue()) {
        for (auto siblingOp : llvm::make_early_inc_range(parentOp->getResult(0).getUsers())) {
            auto siblingCopy = mlir::dyn_cast<VPUIP::CopyOp>(*siblingOp);
            if (siblingCopy != nullptr && siblingOp != copyOp) {
                log.trace("Fuse copy op {0} to {1}", copyOp->getLoc(), siblingCopy->getLoc());

                siblingCopy.getOperation()->replaceAllUsesWith(copyReplaceTarget);
                siblingCopy->erase();
            }
        }
    }

    // Optimize pattern parentOp -> NCEClusterTiling{CopyOp}
    for (auto siblingOp : llvm::make_early_inc_range(parentOp->getResult(0).getUsers())) {
        const auto nestedCopyOp = getCopyOpFromClusterTilingOp(siblingOp);
        const auto siblingResultType = getResultType(siblingOp);
        if (nestedCopyOp != nullptr && nestedCopyOp != copyOp && srcOpResultType == siblingResultType) {
            auto siblingCluster = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(*siblingOp);
            log.trace("Fuse NCEClusterTiling wrapped op {0} to {1}", copyOp->getLoc(), siblingCluster->getLoc());

            siblingCluster->replaceAllUsesWith(copyReplaceTarget);
            siblingCluster.getInnerTaskOp()->erase();
            siblingCluster->erase();
        }
    }
    return mlir::success();
}

// safeRunOnFunc

void OptimizeParallelCopiesPass::safeRunOnFunc() {
    getFunction()->walk([&](VPUIP::CopyOp copyOp) {
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

std::unique_ptr<mlir::Pass> vpux::VPUIP::createOptimizeParallelCopiesPass(Logger log) {
    return std::make_unique<OptimizeParallelCopiesPass>(log);
}
