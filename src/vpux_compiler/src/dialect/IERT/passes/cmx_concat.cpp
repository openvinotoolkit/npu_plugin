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

#include "vpux/compiler/core/attributes/dims_order.hpp"

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/IERT/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/stl_extras.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "vpux/compiler/dialect/IERT/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops_interfaces.hpp"

#include "vpux/compiler/utils/logging.hpp"

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
    const auto type = val.getType().dyn_cast<vpux::NDTypeInterface>();
    VPUX_THROW_UNLESS(type != nullptr, "CMX Concat can work only with MemRef Type, got '{0}'", val.getType());
    const Byte totalSize = type.getTotalAllocSize();
    return checked_cast<size_t>(totalSize.count());
}

bool isThisAComplexConcat(SmallVector<VPUIP::NCEClusterTaskOp> nceTiles, SmallVector<IERT::CopyOp> copyInOps) {
    // avoid concats which are complex, where the inputs to the concat are used
    // by other operations
    for (size_t idx = 0; idx < nceTiles.size(); idx++) {
        for (auto user : nceTiles[idx].output().getUsers()) {
            if (user != copyInOps[idx].getOperation()) {
                // the NCE contains a different user
                return true;
            }
        }
    }

    return false;
}

bool concatOperationDoesNotFitInCMX(IERT::ConcatViewOp concat, SmallVector<VPUIP::NCEClusterTaskOp> nceTiles,
                                    size_t cmxSize) {
    // check if the concat can fit in CMX
    // in order to CMX a concat the entire output buffer + inputs for the
    // largest tile must fit in CMX at the same time
    auto output = concat.getResult();
    size_t concatSize = getSize(output);
    size_t maxUserSize = 0;
    size_t currUserSize = 0;
    ValueOrderedSet inputs;

    // from all users find the one with the largest size
    for (auto user : nceTiles) {
        currUserSize = 0;
        inputs.clear();
        // add all inputs
        for (auto input : user.getInputs()) {
            if (inputs.find(input) == inputs.end()) {
                currUserSize += getSize(input);
                inputs.insert(input);
            }
        }
        // subtract output as reference in inputs
        for (auto output : user.getOutputs()) {
            currUserSize -= getSize(output);
        }
        maxUserSize = std::max<size_t>(maxUserSize, currUserSize);
    }

    // return concat size smaller than CMX size
    return (concatSize + maxUserSize) > cmxSize;
}

bool childOperationsDoNotFitInCMX(IERT::ConcatViewOp concat, SmallVector<IERT::CopyOp> copyOutOps, size_t cmxSize) {
    // check if the child operations - operations using the concat output buffer
    // will fit in CMX along with their inputs and output
    auto output = concat.getResult();
    size_t concatSize = getSize(output);
    size_t maxConsumerSize = 0;

    // from all users find the one with the biggest size
    for (auto& copyOut : copyOutOps) {
        for (auto user : copyOut.output().getUsers()) {
            size_t currentConsumerSize = 0;
            auto userOp = mlir::dyn_cast<IERT::LayerOpInterface>(user);
            for (auto input : userOp.getInputs()) {
                if (input.getDefiningOp() == copyOut.getOperation()) {
                    continue;
                }
                currentConsumerSize += getSize(input);
            }
            for (auto output : userOp.getOutputs()) {
                currentConsumerSize += getSize(output);
            }
            // input has reference to output, no need to loop through outputs
            maxConsumerSize = std::max<size_t>(maxConsumerSize, currentConsumerSize);
        }
    }

    // return concat size greater than CMX size
    return (maxConsumerSize + concatSize) > cmxSize;
}

bool isSplitSupportedOnDPU(IERT::SubViewOp subViewOp) {
    // Check if SubView performs a split along major dimension taking into account order in memory
    // For NCHW that would be split along C
    // For NHWC that would be split along H
    // Only such cases are supported by DPU IDU because only then input to DPU is a contiguous
    // block in memory. Otherwise this behavior needs to be performed by DMA
    const auto inputTypeShape = getShape(subViewOp.source()).raw();
    const auto outputType = subViewOp.result().getType().cast<vpux::NDTypeInterface>();
    const auto outputTypeShape = outputType.getShape().raw();

    if (inputTypeShape.size() != outputTypeShape.size() || inputTypeShape.size() != 4) {
        return false;
    }

    size_t dimsDifference = 0;
    size_t dimsDifferenceCount = 0;
    const auto order = outputType.getDimsOrder();

    for (size_t i = 0; i < 4; i++) {
        if (inputTypeShape[i] != outputTypeShape[i]) {
            dimsDifference = i;
            dimsDifferenceCount++;
        }
    }

    if (dimsDifferenceCount > 1) {
        return false;
    }

    if (dimsDifference == 1 && order == DimsOrder::NCHW) {
        return true;
    }

    if (dimsDifference == 2 && order == DimsOrder::NHWC) {
        return true;
    }

    return false;
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
            auto subViewOp = mlir::dyn_cast<IERT::SubViewOp>(user);
            if (!isSplitSupportedOnDPU(subViewOp)) {
                return mlir::failure();
            }
            for (auto subUser : user->getUsers()) {
                if (!mlir::isa<IERT::CopyOp>(subUser)) {
                    return mlir::failure();
                }
                auto copyOut = mlir::dyn_cast<IERT::CopyOp>(subUser);
                const auto dstMemory = copyOut.output().getType().cast<vpux::NDTypeInterface>().getMemoryKind();
                // If failed to obtain, or the destination is not CMX
                if (dstMemory != VPU::MemoryKind::CMX_NN) {
                    return mlir::failure();
                }
                // In cases where multiple copy out ops use one sub view
                // add in parallel
                copyOutOpsWithSubView.push_back(copyOut);
                copyOutSubViews.push_back(subViewOp);
            }
        } else if (mlir::isa<IERT::CopyOp>(*user)) {
            auto copyOut = mlir::dyn_cast<IERT::CopyOp>(user);
            const auto dstMemory = copyOut.output().getType().cast<vpux::NDTypeInterface>().getMemoryKind();
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
    if (concatOperationDoesNotFitInCMX(concat, nceTiles, _cmxSize)) {
        return mlir::failure();
    }

    // verify the following operation can fit in CMX
    if (childOperationsDoNotFitInCMX(concat, copyOutOpsWithSubView, _cmxSize) ||
        childOperationsDoNotFitInCMX(concat, copyOutOps, _cmxSize)) {
        return mlir::failure();
    }

    if (isThisAComplexConcat(nceTiles, copyInOps)) {
        // TODO implement complex concat
        // where part of the concatenated buffer is also used by another operation
        // visible in yolo-v4-tiny concatenated 4
        return mlir::failure();
    }

    // create a new CMX memref and AllocOp
    auto masterBufferOutput = concat.getResult();
    auto masterBufferOutputMemRefType = masterBufferOutput.getType().cast<vpux::NDTypeInterface>();

    // retrieve attributes of the DDR memref
    const auto cmxMemSpaceAttr = VPU::MemoryKind::CMX_NN;
    const auto shape = ShapeRef(masterBufferOutputMemRefType.getShape());
    const auto elemType = masterBufferOutputMemRefType.getElementType();
    const auto order = masterBufferOutputMemRefType.getDimsOrder();

    // find IR location that is dominated by all NCE Tiles
    auto firstNce = nceTiles.front();
    for (auto& nceTile : nceTiles) {
        if (nceTile->isBeforeInBlock(firstNce.getOperation())) {
            firstNce = nceTile;
        }
    }
    // create new memref in CMX
    rewriter.setInsertionPointAfterValue(firstNce.input());
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
        // update old subView use
        auto copyInSubView = mlir::dyn_cast<IERT::SubViewOp>(copyInSubViews[idx].getDefiningOp());
        copyInSubView.result().replaceAllUsesWith(newInSubViews[idx]);
        // remove usused
        rewriter.eraseOp(copyInOps[idx]);
        rewriter.eraseOp(copyInSubView);
    }

    // create new CMX-out SubViewOps
    rewriter.setInsertionPointAfter(newConcat);
    for (size_t idx = 0; idx < copyOutOpsWithSubView.size(); idx++) {
        // Case 1. With child streaming
        auto newOutSubView = rewriter.create<IERT::SubViewOp>(copyOutSubViews[idx].getLoc(), newConcat.output(),
                                                              copyOutSubViews[idx].static_offsetsAttr(),
                                                              copyOutSubViews[idx].static_sizesAttr());
        copyOutSubViews[idx].result().replaceAllUsesWith(newOutSubView);
        copyOutOpsWithSubView[idx].output().replaceAllUsesWith(newOutSubView);
        copyOutOpsWithSubView[idx].output_buff().replaceAllUsesWith(newOutSubView);
        rewriter.eraseOp(copyOutOpsWithSubView[idx]);
    }
    for (size_t idx = 0; idx < copyOutOps.size(); idx++) {
        // Case 2. Without child streaming
        copyOutOps[idx].output().replaceAllUsesWith(newConcat.output());
        copyOutOps[idx].output_buff().replaceAllUsesWith(newConcat.output_buff());
    }

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

    auto availableMem = IE::getAvailableMemory(module, VPU::MemoryKind::CMX_NN);
    const auto cmxSize = checked_cast<size_t>(availableMem.size().count());

    // This pass will do the following:
    // 1. Locate concat subgraphs
    // 2. Check if concat fit in CMX
    // 3. Move concats to CMX

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<ConcatSequence>(&ctx, cmxSize, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }

    // 4. Remove left over operations with no users
    func->walk([](IERT::LayerOpInterface op) {
        // Skip for operations wrapped in NCEClusterTiling as their result
        // is not used directly
        if (mlir::isa_and_nonnull<VPUIP::NCEClusterTilingOp>(op->getParentOp())) {
            return;
        }
        bool canRemove = true;
        for (const auto res : op->getResults()) {
            if (!res.use_empty()) {
                canRemove = false;
                break;
            }
        }

        if (canRemove) {
            op->erase();
        }
    });
}

}  // namespace

//
// createCMXConcatPass
//

std::unique_ptr<mlir::Pass> vpux::IERT::createCMXConcatPass(Logger log) {
    return std::make_unique<CMXConcatPass>(log);
}
