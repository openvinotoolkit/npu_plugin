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

        const auto& padsBegin = tileConf.pads.begin;
        const auto& padsEnd = tileConf.pads.end;

        const auto actInput = makeTile(rewriter, origOp->getLoc(), origOp.input(), inputTile, "input");
        const auto filterInput = makeTile(rewriter, origOp->getLoc(), origOp.filter(), filterTile, "filter");
        const auto biasInput = origOp.bias() != nullptr
                                       ? makeTile(rewriter, origOp->getLoc(), origOp.bias(), biasTile, "bias")
                                       : nullptr;

        const auto tileName = llvm::formatv("output tile {0}", outputTile.offsets).str();
        const auto loc = appendLoc(origOp->getLoc(), tileName);

        const auto tileTypeOut = changeShape(origOp.output().getType().cast<mlir::MemRefType>(), outputTile.shape);
        auto allocOutOp = rewriter.create<mlir::memref::AllocOp>(loc, tileTypeOut);

        auto tiledOp = rewriter.create<IERT::ConvolutionOp>(
                loc, actInput, filterInput, biasInput, allocOutOp.memref(), origOp.strides(),
                getInt32ArrayAttr(getContext(), padsBegin), getInt32ArrayAttr(getContext(), padsEnd),
                origOp.dilations(), origOp.post_opAttr());

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

        const auto& padsBegin = tileConf.pads.begin;
        const auto& padsEnd = tileConf.pads.end;

        const auto actInput = makeTile(rewriter, origOp->getLoc(), origOp.input(), inputTile, "input");

        const auto tileName = llvm::formatv("output tile {0}", outputTile.offsets).str();
        const auto loc = appendLoc(origOp->getLoc(), tileName);

        const auto tileTypeOut = changeShape(origOp.output().getType().cast<mlir::MemRefType>(), outputTile.shape);
        auto allocOutOp = rewriter.create<mlir::memref::AllocOp>(loc, tileTypeOut);

        auto tiledOp = rewriter.create<IERT::MaxPoolOp>(loc, actInput, allocOutOp.memref(), origOp.kernel_size(),
                                                        origOp.strides(), getInt32ArrayAttr(getContext(), padsBegin),
                                                        getInt32ArrayAttr(getContext(), padsEnd), origOp.post_opAttr());

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

        const auto& padsBegin = tileConf.pads.begin;
        const auto& padsEnd = tileConf.pads.end;

        const auto actInput = makeTile(rewriter, origOp->getLoc(), origOp.input(), inputTile, "input");
        const auto filterInput = makeTile(rewriter, origOp->getLoc(), origOp.filter(), filterTile, "filter");
        const auto biasInput = origOp.bias() != nullptr
                                       ? makeTile(rewriter, origOp->getLoc(), origOp.bias(), biasTile, "bias")
                                       : nullptr;

        const auto tileName = llvm::formatv("output tile {0}", outputTile.offsets).str();
        const auto loc = appendLoc(origOp->getLoc(), tileName);

        const auto tileTypeOut = changeShape(origOp.output().getType().cast<mlir::MemRefType>(), outputTile.shape);
        auto allocOutOp = rewriter.create<mlir::memref::AllocOp>(loc, tileTypeOut);

        const auto filter_out_channel_dim = IERT::ConvolutionOp::filter_out_channel_dim();
        const auto groups = filterTile.shape[filter_out_channel_dim];
        const auto groupsAttr = getInt32Attr(getContext(), checked_cast<uint32_t>(groups));

        auto tiledOp = rewriter.create<IERT::GroupConvolutionOp>(
                loc, actInput, filterInput, biasInput, allocOutOp.memref(), origOp.strides(),
                getInt32ArrayAttr(getContext(), padsBegin), getInt32ArrayAttr(getContext(), padsEnd),
                origOp.dilations(), groupsAttr, origOp.post_opAttr());

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

    const auto maxPoolTilerFunc = std::bind(&SimpleTiler::maxPoolTiler, this, std::placeholders::_1);
    patterns.add<MaxPoolTiling>(patterns.getContext(), maxPoolTilerFunc, _log);

    const auto groupConvTilerFunc = std::bind(&SimpleTiler::groupConvolutionTiler, this, std::placeholders::_1);
    patterns.add<GroupConvolutionTiling>(patterns.getContext(), groupConvTilerFunc, _log);
}

OutputTiling SimpleTiler::genericTiler(mlir::Operation* op, mlir::MemRefType outputType,
                                       FuncRef<bool(ShapeRef)> isSupportedTileSize) const {
    const auto act_channel_dim = IERT::ConvolutionOp::act_channel_dim();

    const auto outputShape = getShape(outputType);

    const auto minChannelSize = VPUIP::NCEInvariant::getChannelAlignment(outputType.getElementType());
    const auto maxChannelTiles = outputShape[act_channel_dim] / minChannelSize;

    Shape nTilesOnDim(outputShape.size(), 1);

    const auto isSupportedChannelDivision = [&]() {
        if ((outputShape[act_channel_dim] % nTilesOnDim[act_channel_dim]) != 0) {
            return false;
        }

        const auto tileChannels = outputShape[act_channel_dim] / nTilesOnDim[act_channel_dim];
        return (tileChannels % minChannelSize) == 0;
    };

    while (!isSupportedTileSize(nTilesOnDim)) {
        // First try tiling over output channels

        if (nTilesOnDim[act_channel_dim] < maxChannelTiles) {
            do {
                ++nTilesOnDim[act_channel_dim];
            } while (!isSupportedChannelDivision());

            continue;
        }

        // Then try tiling over spatial dimensions (prefer height first)

        Optional<Dim> dimToTile;

        for (auto spatialDim : irange(IERT::ConvolutionOp::act_spatial_dims())) {
            const auto act_spatial_dim = IERT::ConvolutionOp::act_spatial_dim(spatialDim);

            const auto origSize = outputShape[act_spatial_dim];
            const auto prevDivisor = nTilesOnDim[act_spatial_dim];

            if (origSize / prevDivisor > 1) {
                dimToTile = act_spatial_dim;
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

    while (!isSupportedTileSize(nTilesOnDim)) {
        // Try tiling over spatial dimensions only.
        // FIXME Split over channels does not seem to work properly with depthwise convolution.

        Optional<Dim> dimToTile;

        for (auto spatialDim : irange(IERT::ConvolutionOp::act_spatial_dims())) {
            const auto act_spatial_dim = IERT::ConvolutionOp::act_spatial_dim(spatialDim);

            const auto origSize = outputShape[act_spatial_dim];
            const auto prevDivisor = nTilesOnDim[act_spatial_dim];

            if (origSize / prevDivisor > 1) {
                dimToTile = act_spatial_dim;
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
