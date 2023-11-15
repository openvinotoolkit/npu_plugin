//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion_utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/BlockAndValueMapping.h>

#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>

using namespace vpux;
using namespace VPU;

namespace {

//
// VerticalFusionTilingRewriter
//

class VerticalFusionTilingRewriter final : public mlir::OpRewritePattern<VPU::VerticalFusionOp> {
public:
    VerticalFusionTilingRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::VerticalFusionOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(VPU::VerticalFusionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    void adjustInputShape(mlir::PatternRewriter& rewriter, mlir::Operation* operation, InputTiling& inputTiling,
                          mlir::BlockAndValueMapping& mapper, TilingStorage& tilingStorage, int64_t tilingIndex) const;

    Logger _log;
};

/*
 This function slice to original tile shape in case bigger tile size was chosen
 during backpropagation process.
 In this case adjust shapes to original one by slicing
*/
void VerticalFusionTilingRewriter::adjustInputShape(mlir::PatternRewriter& rewriter, mlir::Operation* operation,
                                                    InputTiling& inputTiling, mlir::BlockAndValueMapping& mapper,
                                                    TilingStorage& tilingStorage, int64_t tilingIndex) const {
    VPUX_THROW_WHEN(inputTiling.tiles.size() < operation->getOperands().size(),
                    "Number of operands {0} is more than number of operand tiles {1}", operation->getOperands().size(),
                    inputTiling.tiles.size());
    for (auto& op : operation->getOperands() | indexed) {
        auto operand = op.value();
        auto opIndex = op.index();

        auto expectedOp = mapper.lookupOrNull(operand);
        if (expectedOp == nullptr) {
            continue;
        }

        auto originalTiling = inputTiling.tiles[opIndex];
        const auto expectedShape = getShape(expectedOp);
        const auto expectedOpSize = expectedShape.totalSize();
        const auto originalOpSize = originalTiling.shape.totalSize();
        if (expectedOpSize == originalOpSize) {
            continue;
        }

        VPUX_THROW_WHEN(
                expectedOpSize < originalOpSize,
                "Original shape size for operand {0} is bigger than current one. Current size {1}, original size {2}",
                operand, expectedOpSize, originalOpSize);

        VPUX_THROW_WHEN(expectedShape.size() != originalTiling.shape.size(),
                        "Expected shape {0} and original one {1} must have same rank", expectedShape,
                        originalTiling.shape);

        // correct offset of operations based on offsets of block argument
        // In case the output of previous operation is bigger than expected
        // which might happen when bigger tile was chosen for same block argument
        // slice operation is needed after the output with correct offsets
        // calculated based on tiling information of current operation and previous one
        _log.trace("Offset before {0}, shape {1}", originalTiling.offsets, expectedShape);

        for (auto& item : originalTiling.offsets | indexed) {
            auto& offset = item.value();
            if (offset != 0) {
                const auto dim = item.index();
                const auto origShape = originalTiling.shape.raw()[dim];

                if (auto blockArg = operand.dyn_cast<mlir::BlockArgument>()) {
                    // in case previous operation is outside the block and
                    // operand is block argument, correct offset on its offset from tiling info
                    auto tileInfo = tilingStorage.get(blockArg.getArgNumber(), tilingIndex).value();

                    VPUX_THROW_UNLESS(dim < tileInfo.shape.size(), "Got invalid dim index {0}", dim);
                    const auto inputOffset = tileInfo.offsets.raw()[dim];
                    const auto inputShape = tileInfo.shape.raw()[dim];

                    _log.trace("Input Offset {0}, shape {1} ==> offset: {2}, shape: {3} ", inputOffset, inputShape,
                               offset, origShape);

                    VPUX_THROW_WHEN((inputOffset > offset) || ((inputOffset + inputShape) < (offset + origShape)),
                                    "Got invalid offsets");
                    offset -= inputOffset;
                } else if (auto parentTilingOp = operand.getDefiningOp<VPU::TilingBuilderOpInterface>()) {
                    // in case there is parent operation which has tiling info
                    // restore original tiling of that op based on original tiling info
                    // and correct offset on it
                    auto inputOldTiling = parentTilingOp.backInferTileInfo(originalTiling, _log);

                    VPUX_THROW_WHEN(inputOldTiling.tiles.empty() || dim >= inputOldTiling.tiles[0].offsets.size(),
                                    "Got invalid offsets");

                    offset -= inputOldTiling.tiles[0].offsets.raw()[dim];

                } else {
                    // by default just correct it by original shape
                    offset = expectedShape.raw()[dim] - origShape;
                }
            }
        }
        _log.trace("Offset after {0}", originalTiling.offsets);

        const auto valName = printToString("input {0}", opIndex);
        auto opSlice = makeTile(rewriter, operation->getLoc(), expectedOp, originalTiling, valName);

        mapper.map(operand, opSlice);
    }
}

mlir::LogicalResult VerticalFusionTilingRewriter::matchAndRewrite(VPU::VerticalFusionOp vfOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    const auto tilingStrategy = parseIntArrayAttr<int64_t>(vfOp.tilingStrategy().cast<mlir::ArrayAttr>());

    const auto numTiledAxis = llvm::count_if(tilingStrategy, [](auto num) {
        return num > 1;
    });

    VPUX_THROW_WHEN(numTiledAxis != 1, "VF tiling is supported only for one axis");

    auto maxTiledLen = std::max_element(tilingStrategy.begin(), tilingStrategy.end());

    if (maxTiledLen == tilingStrategy.end()) {
        return mlir::failure();
    }

    VPUX_THROW_WHEN(*maxTiledLen <= 1, "There is no tiling for VF");

    auto operationStorage = std::make_unique<TilingOperationStorage>();
    auto tilingStorage = restoreTilingRegions(vfOp, _log, operationStorage);

    SmallVector<mlir::Value> resultTileVals;
    resultTileVals.reserve(*maxTiledLen);
    SmallVector<Shape> resultTileOffsets;
    mlir::BlockAndValueMapping mapper;
    for (auto index : irange(*maxTiledLen)) {
        mlir::Value currentResult;
        Shape currentTile;
        for (auto& op : vfOp.getBody()->without_terminator()) {
            for (auto operand : op.getOperands()) {
                if (auto blockArg = operand.dyn_cast<mlir::BlockArgument>()) {
                    const auto valName = printToString("input {0}", index);
                    auto origInput = vfOp.getOperand(blockArg.getArgNumber());
                    auto tileInfo = tilingStorage.get(blockArg.getArgNumber(), index);

                    VPUX_THROW_WHEN(!tileInfo.has_value(),
                                    "Couldn't find tile information for argument {0} and tile {1}",
                                    blockArg.getArgNumber(), index);

                    auto operandTile = VPU::makeTile(rewriter, op.getLoc(), origInput, tileInfo.value(), valName);
                    mapper.map(operand, operandTile);
                }
            }

            auto inputTiling = operationStorage->get(&op, index);

            VPUX_THROW_WHEN(!inputTiling.has_value(), "Couldn't find tile information for operation {0} and tile {1}",
                            op, index);

            const auto inputTilingPair = inputTiling.value();
            auto inputTilingInfo = inputTilingPair.first;
            adjustInputShape(rewriter, &op, inputTilingInfo, mapper, tilingStorage, index);

            auto* copiedOp = rewriter.clone(op, mapper);
            currentResult = copiedOp->getResult(0);

            currentTile = inputTilingPair.second.offsets;
            if (auto tiledBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(copiedOp)) {
                tiledBuilderOp.adjustAttrs(inputTilingInfo, inputTilingPair.second);
            }

            const auto baseResType = op.getResult(0).getType().cast<vpux::NDTypeInterface>();
            const auto tiledResType =
                    baseResType.extractDenseTile(inputTilingPair.second.offsets, inputTilingPair.second.shape);

            currentResult.setType(tiledResType);

            mapper.map(op.getResult(0), currentResult);
        }

        resultTileVals.push_back(currentResult);
        resultTileOffsets.push_back(currentTile);
    }

    rewriter.replaceOpWithNewOp<VPU::ConcatOp>(vfOp, vfOp->getResult(0).getType(), mlir::ValueRange(resultTileVals),
                                               makeArrayRef(resultTileOffsets));

    return mlir::success();
}

//
// VfTilingPass
//

class VfTilingPass final : public VfTilingBase<VfTilingPass> {
public:
    explicit VfTilingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnModule
//

void VfTilingPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<VPU::VerticalFusionOp>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<VPU::VPUDialect>();
    target.addLegalOp<mlir::func::FuncOp, mlir::func::ReturnOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<VerticalFusionTilingRewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createVfTilingPass
//

std::unique_ptr<mlir::Pass> VPU::createVfTilingPass(Logger log) {
    return std::make_unique<VfTilingPass>(log);
}
