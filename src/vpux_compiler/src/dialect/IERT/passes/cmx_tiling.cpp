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

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/tiling.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <functional>
#include <numeric>

using namespace vpux;
using namespace VPUIP;

namespace {

Dim largestTileDim(ShapeRef shape, ShapeRef nTilesOnDim) {
    int64_t maxSize = 0;
    Dim maxDim(0);

    for (auto d : irange(shape.size())) {
        const auto dim = Dim(d);

        VPUX_THROW_UNLESS(nTilesOnDim[dim] != 0, "Cannot divide by 0 tiles");

        const auto tiledDimSize = shape[dim] / nTilesOnDim[dim];

        if (tiledDimSize > maxSize) {
            maxSize = tiledDimSize;
            maxDim = dim;
        }
    }

    return maxDim;
}

class CMXTilingPass final : public IERT::CXMTilingBase<CMXTilingPass> {
public:
    explicit CMXTilingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    // tilers
    class SimpleFillHalfCMXTiler;

    // rewriters
    class ConvolutionTiling;

private:
    void safeRunOnFunc() final;
};

class CMXTilingPass::SimpleFillHalfCMXTiler {
public:
    explicit SimpleFillHalfCMXTiler(Byte cmxSize, mlir::MLIRContext* ctx, Logger log)
            : _memSize(cmxSize.count() / FillFactor), _ctx(ctx), _log(log) {
    }

    SmallVector<Tile> convolutionTiler(IERT::ConvolutionOp op) const;

    void buildTilingPatterns(mlir::RewritePatternSet& patterns);

private:
    Byte _memSize;
    mlir::MLIRContext* _ctx;
    Logger _log;

    static constexpr unsigned FillFactor = 2;
};

void CMXTilingPass::SimpleFillHalfCMXTiler::buildTilingPatterns(mlir::RewritePatternSet& patterns) {
    auto convTilerFunc = std::bind(&SimpleFillHalfCMXTiler::convolutionTiler, this, std::placeholders::_1);

    patterns.insert<CMXTilingPass::ConvolutionTiling>(_ctx, convTilerFunc, _log);
}

SmallVector<Tile> CMXTilingPass::SimpleFillHalfCMXTiler::convolutionTiler(IERT::ConvolutionOp op) const {
    const auto inputType = op.input().getType().cast<mlir::MemRefType>();
    const auto outputType = op.output().getType().cast<mlir::MemRefType>();
    const auto weightType = op.filter().getType().cast<mlir::MemRefType>();
    const auto biasType = op.bias() != nullptr ? op.bias().getType().cast<mlir::MemRefType>() : nullptr;

    const auto inputShape = getShape(inputType);
    const auto outputShape = getShape(outputType);

    auto convTileSize = [&](ShapeRef nTilesOnDim) -> Byte {
        Byte iSize = getElemTypeSize(inputType);
        Byte oSize = getElemTypeSize(outputType);

        for (auto d : irange(inputShape.size())) {
            const auto dim = Dim(d);

            VPUX_THROW_UNLESS(nTilesOnDim[dim] != 0, "Cannot divide by 0 tiles");

            if (dim == op.act_channel_dim()) {
                iSize *= inputShape[dim];
            } else {
                iSize *= inputShape[dim] / nTilesOnDim[dim];
            }
        }

        for (auto d : irange(outputShape.size())) {
            const auto dim = Dim(d);

            VPUX_THROW_UNLESS(nTilesOnDim[dim] != 0, "Cannot divide by 0 tiles");

            oSize *= outputShape[dim] / nTilesOnDim[dim];
        }

        VPUX_THROW_UNLESS(nTilesOnDim[op.act_channel_dim()] != 0, "Cannot divide by 0 tiles");

        Byte wSize = getTypeTotalSize(weightType) / nTilesOnDim[op.act_channel_dim()];
        Byte bSize = (biasType != nullptr) ? getTypeTotalSize(biasType) / nTilesOnDim[op.act_channel_dim()] : Byte(0);

        return iSize + oSize + wSize + bSize;
    };

    Shape nTilesOnDim(outputShape.size(), 1);

    while (convTileSize(nTilesOnDim) > _memSize) {
        const auto dim = largestTileDim(outputShape, nTilesOnDim);
        nTilesOnDim[dim]++;
    }

    return Tiling::fillDividedTiles(nTilesOnDim, outputShape);
}

class CMXTilingPass::ConvolutionTiling final : public mlir::OpRewritePattern<IERT::ConvolutionOp> {
public:
    using TilerFunc = std::function<SmallVector<Tile>(IERT::ConvolutionOp)>;

    ConvolutionTiling(mlir::MLIRContext* ctx, TilerFunc tiler, Logger log)
            : mlir::OpRewritePattern<IERT::ConvolutionOp>(ctx), _tiler(tiler), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(IERT::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    TilerFunc _tiler;
    Logger _log;
};

mlir::Value makeTile(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value origVal, const Tile& tile) {
    const auto origType = origVal.getType().cast<mlir::MemRefType>();

    if (tile.shape == getShape(origType)) {
        return origVal;
    }

    SmallVector<int64_t> viewStrides(tile.shape.size(), 1);
    auto viewOp =
            builder.create<mlir::memref::SubViewOp>(loc, origVal, tile.offsets.raw(), tile.shape.raw(), viewStrides);

    const auto tileType = changeShape(origType, tile.shape);
    auto allocOp = builder.create<mlir::memref::AllocOp>(loc, tileType);

    auto copyOp = builder.create<IERT::CopyOp>(loc, viewOp.result(), allocOp.memref());
    return copyOp.output();
}

mlir::LogicalResult CMXTilingPass::ConvolutionTiling::matchAndRewrite(IERT::ConvolutionOp origOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    auto* ctx = rewriter.getContext();

    auto tilings = _tiler(origOp);

    if (tilings.size() < 2) {
        return matchFailed(rewriter, origOp, "No tiling required");
    }

    _log.trace("For ConvOp '{0}' have '{1}' tiles: ", origOp->getLoc(), tilings.size());
    for (auto i : irange(tilings.size())) {
        _log.nest().trace("shape '{0}'", i, tilings[i].shape);
    }
    for (auto i : irange(tilings.size())) {
        _log.nest().trace("offsets '{0}'", i, tilings[i].offsets);
    }

    SmallVector<mlir::Value> finalResults;

    for (auto i : irange(tilings.size())) {
        const auto inputConfig = Tiling::backInferConvTile(origOp, tilings[i]);

        const auto inputTile = inputConfig.inputTile;
        const auto weightsTile = inputConfig.filterTile;
        const auto biasTile = inputConfig.biasTile;

        const auto padsBegin = inputConfig.pads.begin;
        const auto padsEnd = inputConfig.pads.end;

        const auto actInput = makeTile(rewriter, origOp->getLoc(), origOp.input(), inputTile);
        const auto weightInput = makeTile(rewriter, origOp->getLoc(), origOp.filter(), weightsTile);
        const auto biasInput =
                origOp.bias() != nullptr ? makeTile(rewriter, origOp->getLoc(), origOp.bias(), biasTile) : nullptr;

        const auto tileTypeOut = changeShape(origOp.output_buff().getType().cast<mlir::MemRefType>(), tilings[i].shape);
        auto allocOutOp = rewriter.create<mlir::memref::AllocOp>(origOp.getLoc(), tileTypeOut);

        auto convOp = rewriter.create<IERT::ConvolutionOp>(
                origOp.getLoc(), actInput, weightInput, biasInput, allocOutOp.memref(), origOp.strides(),
                getInt32ArrayAttr(ctx, padsBegin), getInt32ArrayAttr(ctx, padsEnd), origOp.dilations());

        SmallVector<int64_t> viewStrides(tilings[i].shape.size(), 1);
        auto subViewOut = rewriter.create<mlir::memref::SubViewOp>(
                origOp->getLoc(), origOp.output_buff(), tilings[i].offsets.raw(), tilings[i].shape.raw(), viewStrides);

        auto copyOut = rewriter.create<IERT::CopyOp>(origOp.getLoc(), convOp.output(), subViewOut.result());
        finalResults.push_back(copyOut);
    }

    rewriter.replaceOpWithNewOp<IERT::ConcatViewOp>(origOp, finalResults, origOp.output_buff());
    return mlir::success();
}

void CMXTilingPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    // TODO: should we add the "known HW ops" as dinamically legal by some estimated condition that they would
    // fit into CMX??

    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto resOp = IERT::RunTimeResourcesOp::getFromModule(module);

    auto cmxSize = resOp.getAvailableMemory(VPUIP::PhysicalMemoryAttr::get(&ctx, VPUIP::PhysicalMemory::CMX_NN));
    VPUX_THROW_UNLESS(cmxSize != nullptr, "Can't get information about {0} memory", VPUIP::PhysicalMemory::CMX_NN);

    SimpleFillHalfCMXTiler simpleTiler(cmxSize.size(), &ctx, _log);

    mlir::RewritePatternSet tilingPatterns(&ctx);
    simpleTiler.buildTilingPatterns(tilingPatterns);

    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(func, std::move(tilingPatterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createCMXTilingPass
//

std::unique_ptr<mlir::Pass> vpux::IERT::createCMXTilingPass(Logger log) {
    return std::make_unique<CMXTilingPass>(log);
}
