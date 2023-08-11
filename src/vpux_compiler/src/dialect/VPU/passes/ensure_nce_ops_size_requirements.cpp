//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/BlockAndValueMapping.h>
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

SmallVector<Dim> getDimsOverKHWLimit(ShapeRef shape) {
    SmallVector<Dim> wrongDims = {};
    for (size_t i = 0; i < shape.size(); i++) {
        const auto dim = Dim(i);
        if (shape[dim] > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
            wrongDims.push_back(dim);
        }
    }
    return wrongDims;
}

class EnsureNCEOpSizeRequirements final : public mlir::OpInterfaceRewritePattern<VPU::TilingBuilderOpInterface> {
public:
    EnsureNCEOpSizeRequirements(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<VPU::TilingBuilderOpInterface>(ctx), _log(log) {
        this->setDebugName("EnsureNCEOpSizeRequirements");
    }
    mlir::LogicalResult matchAndRewrite(VPU::TilingBuilderOpInterface origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult EnsureNCEOpSizeRequirements::matchAndRewrite(VPU::TilingBuilderOpInterface origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    auto op = origOp.getOperation();
    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface", op->getName());
    rewriter.setInsertionPoint(op);

    const auto outputType = op->getResult(0).getType().cast<NDTypeInterface>();
    const auto outputShape = outputType.getShape();
    Shape nTilesOnDim(outputShape.size(), 1);
    const auto log = _log.nest();
    const auto tilingMode = TilingMode::ISOLATED;
    const auto tileDimOrder = getTileDimOrder(op, tilingMode, log);
    _log.nest(4).trace("Tile Dim order is {0}", tileDimOrder);

    const auto isSupportedTileSize = [&](ShapeRef nTilesOnDim, int32_t dimToTile) -> bool {
        const auto tiles = fillDividedTiles(op, nTilesOnDim, outputShape);
        for (auto tile : tiles) {
            if (tile.shape.raw()[dimToTile] > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
                return false;
            }
            auto inputTiling = origOp.backInferTileInfo(tile, log);
            auto& inTiles = inputTiling.tiles;
            if ((dimToTile != Dims4D::Act::C.ind()) &&
                (inTiles.begin()->shape.raw()[dimToTile] > VPU::NCEInvariant::VPU_DIMENSION_LIMIT)) {
                return false;
            }
        }
        return true;
    };

    for (auto tileDimIter = tileDimOrder.begin(); tileDimIter < tileDimOrder.end(); ++tileDimIter) {
        auto dimToTile = *tileDimIter;
        while (!isSupportedTileSize(nTilesOnDim, dimToTile.ind())) {
            ++nTilesOnDim[dimToTile];
        }
    }

    // In case of single tile scheduled there is no need for tiling
    if (llvm::none_of(nTilesOnDim, [](int64_t tiles) {
            return tiles > 1;
        })) {
        return mlir::failure();
    }

    const auto tilesNew = fillDividedTiles(op, nTilesOnDim, outputShape);

    return VPU::applyTileStrategy(origOp, tilesNew, rewriter, log.nest());
}

//
//  EnsureConvICRequirements
//

class EnsureConvICRequirements final : public mlir::OpRewritePattern<VPU::NCEConvolutionOp> {
public:
    EnsureConvICRequirements(mlir::MLIRContext* ctx, VPU::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<VPU::NCEConvolutionOp>(ctx), _arch(arch), _log(log) {
        this->setDebugName("EnsureConvICRequirements");
    }
    mlir::LogicalResult matchAndRewrite(VPU::NCEConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    VPU::ArchKind _arch;
    Logger _log;
};

mlir::LogicalResult EnsureConvICRequirements::matchAndRewrite(VPU::NCEConvolutionOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    // Split over IC supported only for NCEConvolutionOp
    // TODO: E#70421

    // Get the NCEConvolutionOp's input and kernel sizes
    const auto inputShape = getShape(origOp.input());
    auto inputW = inputShape[Dims4D::Act::W];
    auto inputH = inputShape[Dims4D::Act::H];
    auto inputC = inputShape[Dims4D::Act::C];
    auto inputN = inputShape[Dims4D::Act::N];

    if (inputC <= VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
        return mlir::failure();
    }

    const auto kernelShape = getShape(origOp.filter());
    auto kernelW = kernelShape[Dims4D::Filter::KX];
    auto kernelH = kernelShape[Dims4D::Filter::KY];
    auto kernelC = kernelShape[Dims4D::Filter::IC];
    auto kernelN = kernelShape[Dims4D::Filter::OC];

    SmallVector<VPU::NCEConvolutionOp> convOps;
    auto maxTiles = vpux::divUp(inputC, VPU::NCEInvariant::VPU_DIMENSION_LIMIT);

    if (maxTiles == 1) {
        return mlir::failure();
    }

    auto weightsTable = origOp.weightsTable();
    auto weightsTableConst = weightsTable.getDefiningOp<Const::DeclareOp>();
    if (weightsTableConst == nullptr) {
        _log.trace("Could not extract constant from weights table.");
        return mlir::failure();
    }
    auto weightsTableContent = weightsTableConst.content();
    auto weightsTableValues = weightsTableContent.getValues<int32_t>();
    auto weightsTableVecSize = weightsTableValues.size();
    std::vector<int32_t> weightsTableVec(weightsTableVecSize);
    std::copy(weightsTableValues.begin(), weightsTableValues.end(), weightsTableVec.begin());

    auto inType = origOp.input().getType().cast<vpux::NDTypeInterface>();
    auto inElemType = inType.getElementType();

    // TODO: E#70371 - Remaining opens for InputChannels 8K size
    for (auto tile = 0; tile < maxTiles; tile++) {
        const auto offsetIC = tile * VPU::NCEInvariant::VPU_DIMENSION_LIMIT;
        const auto sizeIC = std::min(VPU::NCEInvariant::VPU_DIMENSION_LIMIT, inputC - offsetIC);
        _log.nest().trace("Slicing channels {0} - {1}", offsetIC, sizeIC);

        // Slice inputs
        const Shape inSliceOffsets{0, offsetIC, 0, 0};
        const Shape inSliceShape{inputN, sizeIC, inputH, inputW};
        auto convInput = rewriter.create<VPU::SliceOp>(origOp->getLoc(), origOp.input(),
                                                       getIntArrayAttr(rewriter, inSliceOffsets.raw()),
                                                       getIntArrayAttr(rewriter, inSliceShape.raw()));

        // Slice kernels
        const Shape kernelSliceOffsets{0, offsetIC, 0, 0};
        const Shape kernelSliceShape{kernelN, std::min(VPU::NCEInvariant::VPU_DIMENSION_LIMIT, kernelC - offsetIC),
                                     kernelH, kernelW};
        const auto rawKernelSliceShape = getIntArrayAttr(rewriter, kernelSliceShape);
        auto convFilter = rewriter.create<VPU::SliceOp>(origOp.getLoc(), origOp.filter(),
                                                        getIntArrayAttr(rewriter, kernelSliceOffsets.raw()),
                                                        getIntArrayAttr(rewriter, kernelSliceShape.raw()));

        // Adjust the weights table pointers to correspond to the new offsets of the slices
        const auto noOfBytes = vpux::getElemTypeSize(inElemType).to<Byte>().count();

        // Apply bias for the first convolution only
        if (tile != 0) {
            // Set the bias values to 0
            for (size_t i = 3; i < weightsTableVecSize; i += VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC) {
                weightsTableVec[i] = checked_cast<int32_t>(0);
            }
        }

        // Adjust the weight pointers
        for (size_t i = 0; i < weightsTableVecSize; i += VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC) {
            weightsTableVec[i] = checked_cast<int32_t>((i / VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC) *
                                                       kernelH * kernelW * sizeIC * noOfBytes);
        }

        // Adjust the sparsity pointers
        for (size_t i = 1; i < weightsTableVecSize; i += VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC) {
            weightsTableVec[i] = checked_cast<int32_t>((i / VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC) *
                                                       kernelH * kernelW * sizeIC / 8);
        }

        auto weightsTable = VPU::createWeightsTableTensor(rewriter, origOp->getLoc(), weightsTableVec);
        auto convOp = rewriter.create<VPU::NCEConvolutionOp>(
                origOp.getLoc(), origOp.getType(), convInput.result(), convFilter.result(), weightsTable,
                origOp.activationWindow(), origOp.instructionListTable(), origOp.strides(), origOp.pad(), nullptr,
                rawKernelSliceShape, origOp.activation_window_channel_lengthAttr(), origOp.multiClusterStrategyAttr());

        convOps.push_back(convOp);
    }

    // Add the outputs of the convolutions with NCEEltwise Add operations. This is needed because NCEConvolutionOp
    // accumulates all its input channels into 1 output channel. Splitting the Convolutions into smaller Convolutions,
    // the outputs have to be added together.
    auto output = origOp->getResult(0);
    auto targetEltwiseOutputType = output.getType().cast<vpux::NDTypeInterface>();
    const auto opType = VPU::EltwiseType::ADD;
    SmallVector<VPU::NCEEltwiseOp> addOps;
    VPU::NCEEltwiseOp addResult;

    for (size_t index = 0; index < convOps.size() - 1; index++) {
        auto addOperand = index == 0 ? convOps[index].output() : addResult.output();

        // Construct ppeTaskAttr for NCEEltwise (the last NCEEltwiseAdd will get the PPE from the original Conv)
        auto ppeTaskAttr = VPU::getNCEEltwisePPETaskAttr(addOperand.getType(), convOps[index + 1].output().getType(),
                                                         addOperand.getType(), nullptr, addOperand.getLoc(), opType,
                                                         addOperand.getContext(), _arch);

        addResult = rewriter.create<VPU::NCEEltwiseOp>(
                origOp->getLoc(), targetEltwiseOutputType, addOperand, convOps[index + 1].output(), opType,
                ((index == (convOps.size() - 2) && origOp.ppe().hasValue()) ? origOp.ppeAttr() : ppeTaskAttr), nullptr,
                nullptr);
        addOps.push_back(addResult);
    }

    rewriter.replaceOp(origOp, addResult.output());

    return mlir::success();
}

//
// EnsureNCEOpsSizeRequirementsPass
//

class EnsureNCEOpsSizeRequirementsPass final :
        public VPU::EnsureNCEOpsSizeRequirementsBase<EnsureNCEOpsSizeRequirementsPass> {
public:
    explicit EnsureNCEOpsSizeRequirementsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void EnsureNCEOpsSizeRequirementsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    mlir::ConversionTarget target(ctx);
    mlir::RewritePatternSet patterns(&ctx);
    target.addLegalOp<VPU::SliceOp, VPU::ConcatOp>();

    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (!mlir::isa<VPU::NCEConvolutionOp>(op)) {
            return true;
        }

        const auto inputShape = getShape(op->getOperand(0));
        return inputShape[Dims4D::Act::C] <= VPU::NCEInvariant::VPU_DIMENSION_LIMIT;
    });

    patterns.add<EnsureConvICRequirements>(&ctx, arch, _log);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }

    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (!mlir::isa<VPU::NCEOpInterface>(op)) {
            return true;
        }

        if (mlir::isa<VPU::TilingInfoOpInterface>(op)) {
            const auto inputShape = getShape(op->getOperand(0));
            const auto outputShape = getShape(op->getResult(0));

            auto inSizeWrongDims = getDimsOverKHWLimit(inputShape);
            if (!inSizeWrongDims.empty()) {
                _log.nest(2).info("Input size has dims greater than HW requirements: {0}", inSizeWrongDims);
            }
            const auto outSizeWrongDims = getDimsOverKHWLimit(outputShape);
            if (!outSizeWrongDims.empty()) {
                _log.nest(2).info("Output size has dims greater than HW requirements: {0}", outSizeWrongDims);
            }
            return !(!inSizeWrongDims.empty() || !outSizeWrongDims.empty());
        }

        return true;
    });

    patterns.clear();
    patterns.add<EnsureNCEOpSizeRequirements>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createEnsureNCEOpsSizeRequirementsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createEnsureNCEOpsSizeRequirementsPass(Logger log) {
    return std::make_unique<EnsureNCEOpsSizeRequirementsPass>(log);
}
