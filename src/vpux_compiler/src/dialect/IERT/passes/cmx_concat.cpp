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

#include "vpux/compiler/dialect/IERT/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "vpux/compiler/dialect/IERT/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops_interfaces.hpp"

#include "vpux/compiler/utils/logging.hpp"

#include "vpux/utils/core/range.hpp"

using namespace vpux;

namespace {

//
// ConcatSequence
//

class ConcatSequence final : public mlir::OpRewritePattern<IERT::ConcatViewOp> {
public:
    ConcatSequence(mlir::MLIRContext* ctx, size_t cmxSize, Logger log)
            : mlir::OpRewritePattern<IERT::ConcatViewOp>(ctx), _cmxSize(cmxSize), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::ConcatViewOp concat, mlir::PatternRewriter& rewriter) const final;

private:
    size_t _cmxSize;
    Logger _log;
};

size_t getSize(mlir::Value val) {
    // if input doesn't exist return 0
    if (val == nullptr) {
        return 0UL;
    }
    // assert the value is a memref
    const auto type = val.getType().dyn_cast<mlir::MemRefType>();
    VPUX_THROW_UNLESS(type != nullptr, "StaticAllocation can work only with MemRef Type, got '{0}'", val.getType());
    // return the size of the memref
    const Byte totalSize = getTotalSize(type);
    return checked_cast<size_t>(totalSize.count());
}

bool canConcatFitInCMX(IERT::ConcatViewOp concat, SmallVector<VPUIP::NCEClusterTaskOp> nceTiles, size_t cmxSize) {
    auto output = concat.getResult();
    size_t concatSize = getSize(output);
    // std::cout << "output size: " << getSize(output) << std::endl;
    // std::cout << "number of users: " << nceTiles.size() << std::endl;
    size_t maxUserSize = 0;
    // from all users find the one with the biggest size
    // TODO: a better way to find size
    for (auto user : nceTiles) {
        size_t currUserSize = 0;
        currUserSize += getSize(user.input());
        currUserSize += getSize(user.weights());
        currUserSize += getSize(user.weight_table());
        currUserSize += getSize(user.activation_window());
        if (maxUserSize < currUserSize) {
            maxUserSize = currUserSize;
        }
    }
    // std::cout << "max input size: " << maxUserSize << std::endl;
    concatSize += maxUserSize;
    // std::cout << "total concat size: " << concatSize << std::endl;
    // return concat size smaller than CMX size
    return concatSize < cmxSize;
}

mlir::LogicalResult ConcatSequence::matchAndRewrite(IERT::ConcatViewOp concat, mlir::PatternRewriter& rewriter) const {
    if (concat.output_buff().isa<mlir::BlockArgument>()) {
        return mlir::failure();
    }

    /*      ### Concat Input Pattern ###

    (Operation)  (SubView)  (Operation)  (SubView)
              \      /           \      /
               (Copy)             (Copy)
                     \           /
                       (Concat)
    */

    // store incoming operations to the Concat
    SmallVector<VPUIP::NCEClusterTaskOp> nceTiles;
    SmallVector<mlir::Value> nceTileValues;
    SmallVector<mlir::Value> copyInSubViews;
    SmallVector<IERT::CopyOp> copyInOps;

    for (const auto& input : concat.inputs()) {
        auto inputCopyOp = input.getDefiningOp<IERT::CopyOp>();
        if (inputCopyOp == nullptr) {
            return mlir::failure();
        }
        copyInOps.push_back(inputCopyOp);
        auto nceParent = inputCopyOp.input().getDefiningOp<VPUIP::NCEClusterTaskOp>();
        if (nceParent == nullptr) {
            return mlir::failure();
        }
        copyInSubViews.push_back(inputCopyOp.getOperand(1));
        nceTiles.push_back(nceParent);
        nceTileValues.push_back(inputCopyOp.getOperand(0));
    }

    // no NCE operations found
    if (nceTiles.empty()) {
        return mlir::failure();
    }

    /*  ### Concat Output Pattern ###

        1. With child streaming

                    (Concat)
                    /      \
             (SubView)    (SubView)
               /               \
            (Copy)            (Copy)
             /                   \
        (Operation)           (Operation)

        2. Without child streaming

             (Concat)
                |
              (Copy)
                |
            (Operation)
    */

    // store outgoing operations from the concat
    SmallVector<IERT::CopyOp> copyOutOps;
    SmallVector<IERT::SubViewOp> copyOutSubViews;
    SmallVector<IERT::CopyOp> copyOutOpsWithSubView;

    for (auto user : concat.output().getUsers()) {
        if (mlir::isa<IERT::SubViewOp>(user)) {
            for (auto subUser : user->getUsers()) {
                if (!mlir::isa<IERT::CopyOp>(subUser)) {
                    return mlir::failure();
                }
                auto copyOut = mlir::dyn_cast<IERT::CopyOp>(subUser);
                const auto dstMemory = VPU::getMemoryKind(copyOut.output().getType().cast<mlir::MemRefType>());
                // If failed to obtain, or the destination is not CMX
                if (dstMemory != VPU::MemoryKind::CMX_NN) {
                    return mlir::failure();
                }
                copyOutOpsWithSubView.push_back(copyOut);
            }
            copyOutSubViews.push_back(mlir::dyn_cast<IERT::SubViewOp>(user));
        } else if (mlir::isa<IERT::CopyOp>(*user)) {
            auto copyOut = mlir::dyn_cast<IERT::CopyOp>(user);
            const auto dstMemory = VPU::getMemoryKind(copyOut.output().getType().cast<mlir::MemRefType>());
            // If failed to obtain, or the destination is not CMX
            if (dstMemory != VPU::MemoryKind::CMX_NN) {
                return mlir::failure();
            }
            copyOutOps.push_back(copyOut);
        } else {
            // unsupported type
            return mlir::failure();
        }
    }

    // assert that the concat will fit in CMX
    if (!canConcatFitInCMX(concat, nceTiles, _cmxSize)) {
        return mlir::failure();
    }

    // create a new CMX memref and AllocOp
    auto masterBufferOutput = concat.getResult();
    auto masterBufferOutputMemRefType = masterBufferOutput.getType().cast<mlir::MemRefType>();

    // retrieve attributes of the DDR memref
    const auto cmxMemSpaceAttr = VPU::MemoryKindAttr::get(rewriter.getContext(), VPU::MemoryKind::CMX_NN);
    const auto shape = ShapeRef(masterBufferOutputMemRefType.getShape());
    const auto elemType = masterBufferOutputMemRefType.getElementType();
    const auto order = DimsOrder::fromType(masterBufferOutputMemRefType);

    // create new in CMX
    rewriter.setInsertionPointAfter(nceTiles[0].input().getDefiningOp());
    auto newBufferMemType = getMemRefType(shape, elemType, order, cmxMemSpaceAttr);
    auto newBuffer = rewriter.create<mlir::memref::AllocOp>(masterBufferOutput.getLoc(), newBufferMemType);

    // create new SubViewOps
    SmallVector<mlir::Value> newInSubViews;
    for (size_t idx = 0; idx < copyInSubViews.size(); idx++) {
        auto copyInSubView = mlir::dyn_cast<IERT::SubViewOp>(copyInSubViews[idx].getDefiningOp());
        auto newInSubView =
                rewriter.create<IERT::SubViewOp>(copyInSubView.getLoc(), newBuffer, copyInSubView.static_offsetsAttr(),
                                                 copyInSubView.static_sizesAttr());
        newInSubViews.push_back(newInSubView);
    }

    // create a new CMX Concat
    rewriter.setInsertionPointAfter(concat);
    auto newConcat = rewriter.replaceOpWithNewOp<IERT::ConcatViewOp>(concat, nceTileValues, newBuffer);
    newConcat->setAttr("CMXConcat", rewriter.getBoolAttr(true));

    // update NCE Task output buffer with new master CMX buffer
    for (size_t idx = 0; idx < nceTiles.size(); idx++) {
        // update output buffer
        nceTiles[idx].output_buff().replaceAllUsesWith(newInSubViews[idx]);
        // update type of result
        copyInOps[idx].input().setType(nceTiles[idx].output_buff().getType());
    }

    // create new CMX-out SubViewOps
    rewriter.setInsertionPointAfter(newConcat);
    for (size_t idx = 0; idx < copyOutOpsWithSubView.size(); idx++) {
        // Case 1. With child streaming
        auto newOutSubView = rewriter.create<IERT::SubViewOp>(copyOutSubViews[idx].getLoc(), newConcat.output(),
                                                              copyOutSubViews[idx].static_offsetsAttr(),
                                                              copyOutSubViews[idx].static_sizesAttr());
        copyOutOpsWithSubView[idx].output().replaceAllUsesWith(newOutSubView);
        copyOutOpsWithSubView[idx].output_buff().replaceAllUsesWith(newOutSubView);
    }
    for (size_t idx = 0; idx < copyOutOps.size(); idx++) {
        // Case 2. Without child streaming
        copyOutOps[idx].output().replaceAllUsesWith(newConcat);
        copyOutOps[idx].output_buff().replaceAllUsesWith(newBuffer);
    }

    std::cout << "concat CMX-ed" << std::endl;
    return mlir::success();
}

//
// CMXConcatPass
//

class CMXConcatPass final : public IERT::CMXConcatBase<CMXConcatPass> {
public:
    explicit CMXConcatPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void CMXConcatPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    auto resOp = IERT::RunTimeResourcesOp::getFromModule(module);
    const auto cmxAttr = VPU::MemoryKindAttr::get(module->getContext(), VPU::MemoryKind::CMX_NN);
    auto cmxRes = resOp.getAvailableMemory(cmxAttr);
    const auto cmxSize = checked_cast<size_t>(cmxRes.size().count());

    // This pass will do the following:
    // 1. Locate concat subgraphs
    // 2. Check if concat fit in CMX
    // 3. Move concats to CMX

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<ConcatSequence>(&ctx, cmxSize, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createCMXConcatPass
//

std::unique_ptr<mlir::Pass> vpux::IERT::createCMXConcatPass(Logger log) {
    return std::make_unique<CMXConcatPass>(log);
}
