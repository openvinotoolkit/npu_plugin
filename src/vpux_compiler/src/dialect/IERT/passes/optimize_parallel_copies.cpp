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
    // Check 1: copy DDR->DMA
    const auto srcMemory = VPU::getMemoryKind(copyOp.input().getType().cast<mlir::MemRefType>());
    const auto dstMemory = VPU::getMemoryKind(copyOp.output().getType().cast<mlir::MemRefType>());
    if (srcMemory == dstMemory || srcMemory == VPU::MemoryKind::CMX_NN) {
        return false;
    }

    // Check 2: parallel
    // All the consumer of the parent op should be copies
    // At least one more copy except for the current one
    auto parentOp = copyOp.input().getDefiningOp();
    if (parentOp == nullptr) {
        return false;
    }
    bool hasSiblingCopy = false;
    for (auto siblingOp : parentOp->getResult(0).getUsers()) {
        if (!mlir::isa<IERT::CopyOp>(*siblingOp)) {
            return false;
        }

        // Check 3: current op's consumer is copied to DDR immediately after execution
        auto childOfSiblingOp = siblingOp->getResult(0).getUsers().begin();
        auto childCopyOfSiblingOp = mlir::dyn_cast<IERT::CopyOp>(*childOfSiblingOp->getResult(0).getUsers().begin());
        if (childCopyOfSiblingOp == nullptr) {
            return false;
        }
        if (VPU::getMemoryKind(childCopyOfSiblingOp.input().getType().cast<mlir::MemRefType>()) !=
                    VPU::MemoryKind::CMX_NN ||
            VPU::getMemoryKind(childCopyOfSiblingOp.output().getType().cast<mlir::MemRefType>()) !=
                    VPU::MemoryKind::DDR) {
            return false;
        }

        if (siblingOp != copyOp) {
            hasSiblingCopy = true;
        }
    }
    if (!hasSiblingCopy) {
        return false;
    }

    return true;
}

mlir::LogicalResult fuseParallelCopyOp(IERT::CopyOp copyOp, Logger log) {
    auto parentOp = copyOp.input().getDefiningOp();
    if (parentOp == nullptr) {
        return mlir::failure();
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
        SmallVector<mlir::Operation*> wtUsingOutBuf;
        for (auto use : siblingCopy.output_buff().getUsers()) {
            if (mlir::isa<VPUIP::WeightsTableOp>(*use)) {
                wtUsingOutBuf.push_back(use);
            }
        }
        for (auto wt : wtUsingOutBuf) {
            size_t index = 0;
            for (const auto operand : wt->getOperands()) {
                if (operand == siblingCopy.output_buff()) {
                    break;
                }
                index++;
            }
            wt->setOperand(index, copyOp.output_buff());
        }

        siblingCopy.getOperation()->replaceAllUsesWith(copyOp.getOperation());
        siblingCopy->erase();
    }

    // DEBUG LOG
    for (auto t : copyOp->getResult(0).getUsers()) {
        std::cout << llvm::formatv("\t\t~~~~ now the children: {0}, {1}", t->getLoc(), t->getName()).str() << std::endl;
    }

    return mlir::success();
}

// safeRunOnFunc

void OptimizeParallelCopiesPass::safeRunOnFunc() {
    getFunction()->walk([&](IERT::CopyOp copyOp) {
        if (isCopyFusable(copyOp)) {
            _log.nest(1).trace("Fuse parallel copy op '{0}' at '{1}'", copyOp->getName(), copyOp->getLoc());
            std::cout
                    << llvm::formatv("Fuse parallel copy op '{0}' at '{1}'", copyOp->getName(), copyOp->getLoc()).str()
                    << std::endl;
            if (mlir::failed(fuseParallelCopyOp(copyOp, _log))) {
                _log.nest(1).trace("Failed copy fusion of {0} at {1}", copyOp->getName(), copyOp->getLoc());
                std::cout
                        << llvm::formatv("Failed copy fusion of {0} at {1}", copyOp->getName(), copyOp->getLoc()).str()
                        << std::endl;
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
