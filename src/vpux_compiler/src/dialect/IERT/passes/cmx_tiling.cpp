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

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IERT/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

#include <functional>
#include <numeric>

using namespace vpux;

namespace {

using OutputTiling = SmallVector<Tile>;

//
// makeTile
//

mlir::Value makeTile(mlir::OpBuilder& builder, mlir::Location baseLoc, mlir::Value origVal, const Tile& tile,
                     StringRef valName) {
    const auto origType = origVal.getType().cast<mlir::MemRefType>();

    if (tile.shape == getShape(origType)) {
        return origVal;
    }

    const SmallVector<int64_t> viewStrides(tile.shape.size(), 1);

    const auto tileName = llvm::formatv("{0} tile {1}", valName, tile.offsets).str();
    const auto loc = appendLoc(baseLoc, tileName);

    auto viewOp =
            builder.create<mlir::memref::SubViewOp>(loc, origVal, tile.offsets.raw(), tile.shape.raw(), viewStrides);

    const auto tileType = changeShape(origType, tile.shape);
    auto allocOp = builder.create<mlir::memref::AllocOp>(loc, tileType);

    auto copyOp = builder.create<IERT::CopyOp>(loc, viewOp.result(), allocOp.memref());
    return copyOp.output();
}

mlir::Value makeFilterTile(mlir::OpBuilder& builder, mlir::Location baseLoc, mlir::Value origVal, const Tile& tile,
                           StringRef valName) {
    const auto origType = origVal.getType().cast<mlir::MemRefType>();

    if (tile.shape == getShape(origType)) {
        return origVal;
    }

    const auto tileName = llvm::formatv("{0} tile {1}", valName, tile.offsets).str();
    const auto loc = appendLoc(baseLoc, tileName);

    const auto tileShape = tile.shape.raw();
    const auto tileOffsets = tile.offsets.raw();

    const auto attrOffsets = getIntArrayAttr(builder.getContext(), tileOffsets);
    const auto attrShape = getIntArrayAttr(builder.getContext(), tileShape);
    auto viewOp = builder.create<IERT::SubViewOp>(loc, origVal, attrOffsets, attrShape);

    const auto tileType = changeShape(origType, tile.shape);
    auto allocOp = builder.create<mlir::memref::AllocOp>(loc, tileType);

    auto copyOp = builder.create<IERT::CopyOp>(loc, viewOp.result(), allocOp.memref());
    return copyOp.output();
}

//
// ConvolutionTiling
//

class ConvolutionTiling final : public mlir::OpRewritePattern<IERT::ConvolutionOp> {
    using TilerFunc = std::function<OutputTiling(IERT::ConvolutionOp)>;

public:
    ConvolutionTiling(mlir::MLIRContext* ctx, TilerFunc tiler, Logger log)
            : mlir::OpRewritePattern<IERT::ConvolutionOp>(ctx), _tiler(std::move(tiler)), _log(log) {
        setDebugName("ConvolutionTiling");
    }

    mlir::LogicalResult matchAndRewrite(IERT::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    TilerFunc _tiler;
    Logger _log;
};

mlir::LogicalResult ConvolutionTiling::matchAndRewrite(IERT::ConvolutionOp origOp,
                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Convolution at '{1}'", getDebugName(), origOp->getLoc());

    const auto tilings = _tiler(origOp);

    _log.nest(1).trace("Create {0} tiles:", tilings.size());
    for (const auto& outputTile : tilings) {
        _log.nest(2).trace("Output tile shape '{0}' offsets '{1}'", outputTile.shape, outputTile.offsets);
    }

    SmallVector<mlir::Value> finalResults;
    finalResults.reserve(tilings.size());

    for (const auto& outputTile : tilings) {
        const auto tileConf = backInferConvTile(origOp, outputTile);

        const auto& inputTile = tileConf.inputTile;
        const auto& filterTile = tileConf.filterTile;
        const auto& biasTile = tileConf.biasTile;

        SmallVector<int64_t> padsBegin = {tileConf.pads.padTop, tileConf.pads.padLeft};
        SmallVector<int64_t> padsEnd = {tileConf.pads.padBottom, tileConf.pads.padRight};

        const auto actInput = makeTile(rewriter, origOp->getLoc(), origOp.input(), inputTile, "input");
        const auto filterInput = makeFilterTile(rewriter, origOp->getLoc(), origOp.filter(), filterTile, "filter");
        const auto biasInput = origOp.bias() != nullptr
                                       ? makeTile(rewriter, origOp->getLoc(), origOp.bias(), biasTile, "bias")
                                       : nullptr;

        const auto tileName = llvm::formatv("output tile {0}", outputTile.offsets).str();
        const auto loc = appendLoc(origOp->getLoc(), tileName);

        const auto tileTypeOut = changeShape(origOp.output().getType().cast<mlir::MemRefType>(), outputTile.shape);
        auto allocOutOp = rewriter.create<mlir::memref::AllocOp>(loc, tileTypeOut);

        auto tiledOp = rewriter.create<IERT::ConvolutionOp>(loc, actInput, filterInput, biasInput, allocOutOp.memref(),
                                                            origOp.strides(), getIntArrayAttr(getContext(), padsBegin),
                                                            getIntArrayAttr(getContext(), padsEnd), origOp.dilations(),
                                                            origOp.post_opAttr());

        SmallVector<int64_t> viewStrides(outputTile.shape.size(), 1);
        auto subViewOut = rewriter.create<mlir::memref::SubViewOp>(loc, origOp.output_buff(), outputTile.offsets.raw(),
                                                                   outputTile.shape.raw(), viewStrides);

        auto copyOut = rewriter.create<IERT::CopyOp>(loc, tiledOp.output(), subViewOut.result());
        finalResults.push_back(copyOut);
    }

    rewriter.replaceOpWithNewOp<IERT::ConcatViewOp>(origOp, finalResults, origOp.output_buff());
    return mlir::success();
}

//
// EltwiseAddTiling
//

class EltwiseAddTiling final : public mlir::OpRewritePattern<IERT::AddOp> {
    using TilerFunc = std::function<OutputTiling(IERT::AddOp)>;

public:
    EltwiseAddTiling(mlir::MLIRContext* ctx, TilerFunc tiler, Logger log)
            : mlir::OpRewritePattern<IERT::AddOp>(ctx), _tiler(std::move(tiler)), _log(log) {
        setDebugName("EltwiseAddTiling");
    }

    mlir::LogicalResult matchAndRewrite(IERT::AddOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    TilerFunc _tiler;
    Logger _log;
};

mlir::LogicalResult EltwiseAddTiling::matchAndRewrite(IERT::AddOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Eltwise Add at '{1}'", getDebugName(), origOp->getLoc());

    const OutputTiling tilings = _tiler(origOp);

    _log.nest(1).trace("Create {0} tiles:", tilings.size());
    for (const auto& outputTile : tilings) {
        _log.nest(2).trace("Output tile shape '{0}' offsets '{1}'", outputTile.shape, outputTile.offsets);
    }

    SmallVector<mlir::Value> finalResults;
    finalResults.reserve(tilings.size());

    for (const Tile& outputTile : tilings) {
        const EltwiseTileConfig tileConf = backInferEltwiseAddTile(outputTile);

        const Tile& inputTile = tileConf.inputTile;

        const mlir::Value actInput1 = makeTile(rewriter, origOp->getLoc(), origOp.input1(), inputTile, "input1");
        const mlir::Value actInput2 = makeTile(rewriter, origOp->getLoc(), origOp.input2(), inputTile, "input2");

        const std::string tileName = llvm::formatv("output tile {0}", outputTile.offsets).str();
        const mlir::Location loc = appendLoc(origOp->getLoc(), tileName);

        const mlir::MemRefType tileTypeOut =
                changeShape(origOp.output().getType().cast<mlir::MemRefType>(), outputTile.shape);
        auto allocOutOp = rewriter.create<mlir::memref::AllocOp>(loc, tileTypeOut);

        auto tiledOp = rewriter.create<IERT::AddOp>(loc, actInput1, actInput2, allocOutOp.memref());

        SmallVector<int64_t> viewStrides(outputTile.shape.size(), 1);
        auto subViewOut = rewriter.create<mlir::memref::SubViewOp>(loc, origOp.output_buff(), outputTile.offsets.raw(),
                                                                   outputTile.shape.raw(), viewStrides);

        auto copyOut = rewriter.create<IERT::CopyOp>(loc, tiledOp.output(), subViewOut.result());
        finalResults.push_back(copyOut);
    }

    rewriter.replaceOpWithNewOp<IERT::ConcatViewOp>(origOp, finalResults, origOp.output_buff());
    return mlir::success();
}

//
// MaxPoolTiling
//

class MaxPoolTiling final : public mlir::OpRewritePattern<IERT::MaxPoolOp> {
    using TilerFunc = std::function<OutputTiling(IERT::MaxPoolOp)>;

public:
    MaxPoolTiling(mlir::MLIRContext* ctx, TilerFunc tiler, Logger log)
            : mlir::OpRewritePattern<IERT::MaxPoolOp>(ctx), _tiler(std::move(tiler)), _log(log) {
        setDebugName("MaxPoolTiling");
    }

    mlir::LogicalResult matchAndRewrite(IERT::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    TilerFunc _tiler;
    Logger _log;
};

mlir::LogicalResult MaxPoolTiling::matchAndRewrite(IERT::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got MaxPool at '{1}'", getDebugName(), origOp->getLoc());

    const auto tilings = _tiler(origOp);

    _log.nest(1).trace("Create {0} tiles:", tilings.size());
    for (const auto& outputTile : tilings) {
        _log.nest(2).trace("Output tile shape '{0}' offsets '{1}'", outputTile.shape, outputTile.offsets);
    }

    SmallVector<mlir::Value> finalResults;
    finalResults.reserve(tilings.size());

    for (const auto& outputTile : tilings) {
        const auto tileConf = backInferPoolTile(origOp, outputTile);

        const auto& inputTile = tileConf.inputTile;

        SmallVector<int64_t> padsBegin = {tileConf.pads.padTop, tileConf.pads.padLeft};
        SmallVector<int64_t> padsEnd = {tileConf.pads.padBottom, tileConf.pads.padRight};

        const auto actInput = makeTile(rewriter, origOp->getLoc(), origOp.input(), inputTile, "input");

        const auto tileName = llvm::formatv("output tile {0}", outputTile.offsets).str();
        const auto loc = appendLoc(origOp->getLoc(), tileName);

        const auto tileTypeOut = changeShape(origOp.output().getType().cast<mlir::MemRefType>(), outputTile.shape);
        auto allocOutOp = rewriter.create<mlir::memref::AllocOp>(loc, tileTypeOut);

        auto tiledOp = rewriter.create<IERT::MaxPoolOp>(loc, actInput, allocOutOp.memref(), origOp.kernel_size(),
                                                        origOp.strides(), getIntArrayAttr(getContext(), padsBegin),
                                                        getIntArrayAttr(getContext(), padsEnd), origOp.post_opAttr());

        SmallVector<int64_t> viewStrides(outputTile.shape.size(), 1);
        auto subViewOut = rewriter.create<mlir::memref::SubViewOp>(loc, origOp.output_buff(), outputTile.offsets.raw(),
                                                                   outputTile.shape.raw(), viewStrides);

        auto copyOut = rewriter.create<IERT::CopyOp>(loc, tiledOp.output(), subViewOut.result());
        finalResults.push_back(copyOut);
    }

    rewriter.replaceOpWithNewOp<IERT::ConcatViewOp>(origOp, finalResults, origOp.output_buff());
    return mlir::success();
}

//
// GroupConvolutionTiling
//

class GroupConvolutionTiling final : public mlir::OpRewritePattern<IERT::GroupConvolutionOp> {
    using TilerFunc = std::function<OutputTiling(IERT::GroupConvolutionOp)>;

public:
    GroupConvolutionTiling(mlir::MLIRContext* ctx, TilerFunc tiler, Logger log)
            : mlir::OpRewritePattern<IERT::GroupConvolutionOp>(ctx), _tiler(std::move(tiler)), _log(log) {
        setDebugName("GroupConvolutionTiling");
    }

    mlir::LogicalResult matchAndRewrite(IERT::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    TilerFunc _tiler;
    Logger _log;
};

mlir::LogicalResult GroupConvolutionTiling::matchAndRewrite(IERT::GroupConvolutionOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got GroupConvolution at '{1}'", getDebugName(), origOp->getLoc());

    const auto tilings = _tiler(origOp);

    _log.nest(1).trace("Create {0} tiles:", tilings.size());
    for (const auto& outputTile : tilings) {
        _log.nest(2).trace("Output tile shape '{0}' offsets '{1}'", outputTile.shape, outputTile.offsets);
    }

    SmallVector<mlir::Value> finalResults;
    finalResults.reserve(tilings.size());

    for (const auto& outputTile : tilings) {
        const auto tileConf = backInferGroupConvTile(origOp, outputTile);

        const auto& inputTile = tileConf.inputTile;
        const auto& filterTile = tileConf.filterTile;
        const auto& biasTile = tileConf.biasTile;

        SmallVector<int64_t> padsBegin = {tileConf.pads.padTop, tileConf.pads.padLeft};
        SmallVector<int64_t> padsEnd = {tileConf.pads.padBottom, tileConf.pads.padRight};

        const auto actInput = makeTile(rewriter, origOp->getLoc(), origOp.input(), inputTile, "input");
        const auto filterInput = makeFilterTile(rewriter, origOp->getLoc(), origOp.filter(), filterTile, "filter");
        const auto biasInput = origOp.bias() != nullptr
                                       ? makeTile(rewriter, origOp->getLoc(), origOp.bias(), biasTile, "bias")
                                       : nullptr;

        const auto tileName = llvm::formatv("output tile {0}", outputTile.offsets).str();
        const auto loc = appendLoc(origOp->getLoc(), tileName);

        const auto tileTypeOut = changeShape(origOp.output().getType().cast<mlir::MemRefType>(), outputTile.shape);
        auto allocOutOp = rewriter.create<mlir::memref::AllocOp>(loc, tileTypeOut);

        const auto groups = filterTile.shape[IE::Dims4D::Filter::OC];
        const auto groupsAttr = getIntAttr(getContext(), groups);

        auto tiledOp = rewriter.create<IERT::GroupConvolutionOp>(
                loc, actInput, filterInput, biasInput, allocOutOp.memref(), origOp.strides(),
                getIntArrayAttr(getContext(), padsBegin), getIntArrayAttr(getContext(), padsEnd), origOp.dilations(),
                groupsAttr, origOp.post_opAttr());

        SmallVector<int64_t> viewStrides(outputTile.shape.size(), 1);
        auto subViewOut = rewriter.create<mlir::memref::SubViewOp>(loc, origOp.output_buff(), outputTile.offsets.raw(),
                                                                   outputTile.shape.raw(), viewStrides);

        auto copyOut = rewriter.create<IERT::CopyOp>(loc, tiledOp.output(), subViewOut.result());
        finalResults.push_back(copyOut);
    }

    rewriter.replaceOpWithNewOp<IERT::ConcatViewOp>(origOp, finalResults, origOp.output_buff());
    return mlir::success();
}

//
// SimpleTiler
//

class SimpleTiler {
public:
    explicit SimpleTiler(Logger log): _log(log) {
    }

    void buildTilingPatterns(mlir::RewritePatternSet& patterns);

private:
    OutputTiling convolutionTiler(IERT::ConvolutionOp op) const;
    OutputTiling eltwiseAddTiler(IERT::AddOp op) const;
    OutputTiling maxPoolTiler(IERT::MaxPoolOp op) const;
    OutputTiling groupConvolutionTiler(IERT::GroupConvolutionOp op) const;

    OutputTiling genericTiler(mlir::Operation* op, mlir::MemRefType outputType,
                              FuncRef<bool(ShapeRef)> isSupportedTileSize) const;
    OutputTiling groupConvTiler(mlir::Operation* op, mlir::MemRefType outputType,
                                FuncRef<bool(ShapeRef)> isSupportedTileSize) const;

private:
    Logger _log;
};

void SimpleTiler::buildTilingPatterns(mlir::RewritePatternSet& patterns) {
    const auto convTilerFunc = std::bind(&SimpleTiler::convolutionTiler, this, std::placeholders::_1);
    patterns.add<ConvolutionTiling>(patterns.getContext(), convTilerFunc, _log);

    const auto eltwiseTilerFunc = std::bind(&SimpleTiler::eltwiseAddTiler, this, std::placeholders::_1);
    patterns.add<EltwiseAddTiling>(patterns.getContext(), eltwiseTilerFunc, _log);

    const auto maxPoolTilerFunc = std::bind(&SimpleTiler::maxPoolTiler, this, std::placeholders::_1);
    patterns.add<MaxPoolTiling>(patterns.getContext(), maxPoolTilerFunc, _log);

    const auto groupConvTilerFunc = std::bind(&SimpleTiler::groupConvolutionTiler, this, std::placeholders::_1);
    patterns.add<GroupConvolutionTiling>(patterns.getContext(), groupConvTilerFunc, _log);
    // TODO Replace std::bind calls with corresponding anonymous functions.
}

OutputTiling SimpleTiler::genericTiler(mlir::Operation* op, mlir::MemRefType outputType,
                                       FuncRef<bool(ShapeRef)> isSupportedTileSize) const {
    const auto outputShape = getShape(outputType);

    const auto minChannelSize = VPUIP::NCEInvariant::getChannelAlignment(outputType.getElementType());
    const auto maxChannelTiles = outputShape[IE::Dims4D::Act::C] / minChannelSize;

    Shape nTilesOnDim(outputShape.size(), 1);

    const auto isSupportedChannelDivision = [&]() {
        if ((outputShape[IE::Dims4D::Act::C] % nTilesOnDim[IE::Dims4D::Act::C]) != 0) {
            return false;
        }

        const auto tileChannels = outputShape[IE::Dims4D::Act::C] / nTilesOnDim[IE::Dims4D::Act::C];
        return (tileChannels % minChannelSize) == 0;
    };

    while (!isSupportedTileSize(nTilesOnDim)) {
        // First try tiling over output channels

        if (nTilesOnDim[IE::Dims4D::Act::C] < maxChannelTiles) {
            do {
                ++nTilesOnDim[IE::Dims4D::Act::C];
            } while (!isSupportedChannelDivision());

            continue;
        }

        // Then try tiling over spatial dimensions (prefer height first)

        Optional<Dim> dimToTile;

        for (auto ind : irange(IE::Dims4D::Act::numSpatialDims)) {
            const auto spatialDim = IE::Dims4D::Act::getSpatialDim(ind);

            const auto origSize = outputShape[spatialDim];
            const auto prevDivisor = nTilesOnDim[spatialDim];

            if (origSize / prevDivisor > 1) {
                dimToTile = spatialDim;
                break;
            }
        }

        VPUX_THROW_UNLESS(dimToTile.hasValue(), "Failed to tile {0} at '{1}'", op->getName(), op->getLoc());
        nTilesOnDim[dimToTile.getValue()]++;
    }

    return fillDividedTiles(nTilesOnDim, outputShape);
}

OutputTiling SimpleTiler::groupConvTiler(mlir::Operation* op, mlir::MemRefType outputType,
                                         FuncRef<bool(ShapeRef)> isSupportedTileSize) const {
    const auto outputShape = getShape(outputType);

    Shape nTilesOnDim(outputShape.size(), 1);

    // FIXME tiling over channels has to leave 16 channels in each tile.
    // Otherwise, depthwise convolutions produce worse accuracy.
    const auto depthwiseOutChanCount = VPUIP::NCEInvariant::getChannelAlignment(outputType.getElementType());
    VPUX_THROW_UNLESS(outputShape[IE::Dims4D::Act::C] % depthwiseOutChanCount == 0,
                      "Depthwise convolution output channels must be a multiple of {0}, got {1}", depthwiseOutChanCount,
                      outputShape[IE::Dims4D::Act::C]);
    nTilesOnDim[IE::Dims4D::Act::C] = outputShape[IE::Dims4D::Act::C] / depthwiseOutChanCount;

    while (!isSupportedTileSize(nTilesOnDim)) {
        Optional<Dim> dimToTile;

        for (auto ind : irange(IE::Dims4D::Act::numSpatialDims)) {
            const auto spatialDim = IE::Dims4D::Act::getSpatialDim(ind);

            const auto origSize = outputShape[spatialDim];
            const auto prevDivisor = nTilesOnDim[spatialDim];

            if (origSize / prevDivisor > 1) {
                dimToTile = spatialDim;
                break;
            }
        }

        VPUX_THROW_UNLESS(dimToTile.hasValue(), "Failed to tile {0} at '{1}'", op->getName(), op->getLoc());
        nTilesOnDim[dimToTile.getValue()]++;
    }

    return fillDividedTiles(nTilesOnDim, outputShape);
}

OutputTiling SimpleTiler::convolutionTiler(IERT::ConvolutionOp op) const {
    const auto inputType = op.input().getType().cast<mlir::MemRefType>();
    const auto filterType = op.filter().getType().cast<mlir::MemRefType>();
    const auto outputType = op.output().getType().cast<mlir::MemRefType>();

    const auto outputShape = getShape(outputType);

    const auto isSupportedTileSize = [&](ShapeRef nTilesOnDim) -> bool {
        const auto outputTiles = fillDividedTiles(nTilesOnDim, outputShape);

        return llvm::all_of(outputTiles, [&](const auto& outputTile) {
            const auto tileConf = backInferConvTile(op, outputTile);

            const auto inputTileType = changeShape(inputType, tileConf.inputTile.shape);
            const auto filterTileType = changeShape(filterType, tileConf.filterTile.shape);
            const auto outputTileType = changeShape(outputType, outputTile.shape);

            return mlir::succeeded(
                    VPUIP::NCEInvariant::verifyConvCMX(op->getLoc(), op->getParentOfType<mlir::ModuleOp>(),
                                                       inputTileType, filterTileType, outputTileType, _log));
        });
    };

    return genericTiler(op, outputType, isSupportedTileSize);
}

OutputTiling SimpleTiler::eltwiseAddTiler(IERT::AddOp op) const {
    const auto input1Type = op.input1().getType().cast<mlir::MemRefType>();
    const auto input2Type = op.input2().getType().cast<mlir::MemRefType>();
    const auto outputType = op.output().getType().cast<mlir::MemRefType>();

    const ShapeRef outputShape = getShape(outputType);

    const auto isSupportedTileSize = [&](ShapeRef nTilesOnDim) -> bool {
        const auto outputTiles = fillDividedTiles(nTilesOnDim, outputShape);

        return llvm::all_of(outputTiles, [&](const auto& outputTile) {
            const EltwiseTileConfig tileConf = backInferEltwiseAddTile(outputTile);

            const mlir::MemRefType input1TileType = changeShape(input1Type, tileConf.inputTile.shape);
            const mlir::MemRefType input2TileType = changeShape(input2Type, tileConf.inputTile.shape);
            const mlir::MemRefType outputTileType = changeShape(outputType, outputTile.shape);

            return mlir::succeeded(
                    VPUIP::NCEInvariant::verifyEltwiseCMX(op->getLoc(), op->getParentOfType<mlir::ModuleOp>(),
                                                          input1TileType, input2TileType, outputTileType, _log));
        });
    };

    return genericTiler(op, outputType, isSupportedTileSize);
}

OutputTiling SimpleTiler::maxPoolTiler(IERT::MaxPoolOp op) const {
    const auto inputType = op.input().getType().cast<mlir::MemRefType>();
    const auto outputType = op.output().getType().cast<mlir::MemRefType>();

    const auto outputShape = getShape(outputType);

    const auto isSupportedTileSize = [&](ShapeRef nTilesOnDim) -> bool {
        const auto outputTiles = fillDividedTiles(nTilesOnDim, outputShape);

        return llvm::all_of(outputTiles, [&](const auto& outputTile) {
            const auto tileConf = backInferPoolTile(op, outputTile);

            const auto inputTileType = changeShape(inputType, tileConf.inputTile.shape);
            const auto outputTileType = changeShape(outputType, outputTile.shape);

            return mlir::succeeded(VPUIP::NCEInvariant::verifyPoolCMX(
                    op->getLoc(), op->getParentOfType<mlir::ModuleOp>(), inputTileType, outputTileType,
                    op.kernel_size(), op.strides(), _log));
        });
    };

    return genericTiler(op, outputType, isSupportedTileSize);
}

OutputTiling SimpleTiler::groupConvolutionTiler(IERT::GroupConvolutionOp op) const {
    const auto inputType = op.input().getType().cast<mlir::MemRefType>();
    const auto filterType = op.filter().getType().cast<mlir::MemRefType>();
    const auto outputType = op.output().getType().cast<mlir::MemRefType>();

    const auto outputShape = getShape(outputType);

    const auto isSupportedTileSize = [&](ShapeRef nTilesOnDim) -> bool {
        const auto outputTiles = fillDividedTiles(nTilesOnDim, outputShape);

        return llvm::all_of(outputTiles, [&](const auto& outputTile) {
            const auto tileConf = backInferGroupConvTile(op, outputTile);

            const auto inputTileType = changeShape(inputType, tileConf.inputTile.shape);
            const auto filterTileType = changeShape(filterType, tileConf.filterTile.shape);
            const auto outputTileType = changeShape(outputType, outputTile.shape);

            return mlir::succeeded(
                    VPUIP::NCEInvariant::verifyConvCMX(op->getLoc(), op->getParentOfType<mlir::ModuleOp>(),
                                                       inputTileType, filterTileType, outputTileType, _log));
        });
    };

    return groupConvTiler(op, outputType, isSupportedTileSize);
}

//
// CMXTilingPass
//

class CMXTilingPass final : public IERT::CXMTilingBase<CMXTilingPass> {
public:
    explicit CMXTilingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

template <class ConcreteOp>
bool isSupportedByNCE(ConcreteOp op, Logger log) {
    return VPUIP::NCEInvariant::verifyKernel(op, log).succeeded() &&
           VPUIP::NCEInvariant::verifyChannels(op, log).succeeded();
}

void CMXTilingPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<IERT::IERTDialect>();
    target.addLegalDialect<mlir::memref::MemRefDialect>();
    target.addLegalOp<mlir::FuncOp, mlir::ReturnOp>();
    target.addDynamicallyLegalOp<IERT::ConvolutionOp>([&](IERT::ConvolutionOp op) {
        if (!isSupportedByNCE(op, _log.nest())) {
            // It will be computed on SHAVEs
            return true;
        }

        return VPUIP::NCEInvariant::verifyCMX(op, _log.nest()).succeeded();
    });
    target.addDynamicallyLegalOp<IERT::AddOp>([&](IERT::AddOp op) {
        if (!isSupportedByNCE(op, _log.nest())) {
            // It will be computed on SHAVEs
            return true;
        }

        return VPUIP::NCEInvariant::verifyCMX(op, _log.nest()).succeeded();
    });
    target.addDynamicallyLegalOp<IERT::MaxPoolOp>([&](IERT::MaxPoolOp op) {
        if (!isSupportedByNCE(op, _log.nest())) {
            // It will be computed on SHAVEs
            return true;
        }

        return VPUIP::NCEInvariant::verifyCMX(op, _log.nest()).succeeded();
    });
    target.addDynamicallyLegalOp<IERT::GroupConvolutionOp>([&](IERT::GroupConvolutionOp op) {
        if (!isSupportedByNCE(op, _log.nest())) {
            // Falls back to Conv2dUPA
            return true;
        }

        return VPUIP::NCEInvariant::verifyCMX(op, _log.nest()).succeeded();
    });

    mlir::RewritePatternSet patterns(&ctx);

    SimpleTiler simpleTiler(_log);
    simpleTiler.buildTilingPatterns(patterns);

    if (mlir::failed(mlir::applyPartialConversion(getFunction(), target, std::move(patterns)))) {
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
