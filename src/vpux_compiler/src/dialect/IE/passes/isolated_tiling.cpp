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

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/FunctionExtras.h>

#include <numeric>

using namespace vpux;

namespace {

//
// makeTile
//

mlir::Value makeTile(mlir::OpBuilder& builder, mlir::Location baseLoc, mlir::Value origVal, const TileInfo& tile,
                     StringRef valName) {
    if (tile.shape == getShape(origVal)) {
        return origVal;
    }

    const auto tileName = llvm::formatv("{0} tile {1}", valName, tile.offsets).str();
    const auto loc = appendLoc(baseLoc, tileName);

    auto sliceOp = builder.create<IE::SliceOp>(loc, origVal, tile.offsets, tile.shape);
    return sliceOp.result();
}

//
// needTiling
//

// TODO: convert this to operation interface

template <class ConcreteOp>
bool isSupportedByNCE(ConcreteOp op, Logger log) {
    return VPUIP::NCEInvariant::verifyKernel(op, log).succeeded() &&
           VPUIP::NCEInvariant::verifyChannels(op, log).succeeded();
}

template <class ConcreteOp>
bool needTiling(ConcreteOp origOp, Logger log) {
    if (!isSupportedByNCE(origOp, log.nest())) {
        // It will be computed on SHAVEs
        return false;
    }

    return VPUIP::NCEInvariant::verifyCMX(origOp, log.nest()).failed();
}

//
// isSupportedTiling
//

// TODO: convert this to operation interface

bool isSupportedEltwiseTiling(mlir::Operation* origOp, const OutputTiling& tiles, Logger log) {
    const auto input1Type = origOp->getOperand(0).getType().cast<mlir::ShapedType>();
    const auto input2Type = origOp->getOperand(1).getType().cast<mlir::ShapedType>();
    const auto outputType = origOp->getResult(0).getType().cast<mlir::ShapedType>();

    return llvm::all_of(tiles, [&](const TileInfo& tile) {
        const auto input1TileType = getDenseTileType(input1Type, tile.offsets, tile.shape);
        const auto input2TileType = getDenseTileType(input2Type, tile.offsets, tile.shape);
        const auto outputTileType = getDenseTileType(outputType, tile.offsets, tile.shape);

        return mlir::succeeded(
                VPUIP::NCEInvariant::verifyEltwiseCMX(origOp->getLoc(), origOp->getParentOfType<mlir::ModuleOp>(),
                                                      input1TileType, input2TileType, outputTileType, log));
    });
};

bool isSupportedTiling(IE::ConvolutionOp origOp, const OutputTiling& tiles, Logger log) {
    const auto inputType = origOp.input().getType().cast<mlir::ShapedType>();
    const auto filterType = origOp.filter().getType().cast<mlir::ShapedType>();
    const auto outputType = origOp.output().getType().cast<mlir::ShapedType>();

    return llvm::all_of(tiles, [&](const TileInfo& outputTile) {
        const auto origInputShape = getShape(origOp.input());
        const auto origFilterShape = getShape(origOp.filter());
        const auto origBiasShape = origOp.bias() != nullptr ? getShape(origOp.bias()) : ShapeRef();

        const auto tileConf = backInferConvTile(outputTile, origInputShape, origFilterShape, origBiasShape,
                                                origOp.strides(), origOp.pads_begin(), origOp.pads_end());

        const auto inputTileType = getDenseTileType(inputType, tileConf.inputTile.offsets, tileConf.inputTile.shape);
        const auto filterTileType =
                getDenseTileType(filterType, tileConf.filterTile.offsets, tileConf.filterTile.shape);
        const auto outputTileType = getDenseTileType(outputType, outputTile.offsets, outputTile.shape);

        return mlir::succeeded(VPUIP::NCEInvariant::verifyConvCMX(origOp->getLoc(),
                                                                  origOp->getParentOfType<mlir::ModuleOp>(),
                                                                  inputTileType, filterTileType, outputTileType, log));
    });
}

bool isSupportedTiling(IE::GroupConvolutionOp origOp, const OutputTiling& tiles, Logger log) {
    const auto inputType = origOp.input().getType().cast<mlir::ShapedType>();
    const auto filterType = origOp.filter().getType().cast<mlir::ShapedType>();
    const auto outputType = origOp.output().getType().cast<mlir::ShapedType>();

    return llvm::all_of(tiles, [&](const TileInfo& outputTile) {
        const auto origInputShape = getShape(origOp.input());
        const auto origFilterShape = getShape(origOp.filter());
        const auto origBiasShape = origOp.bias() != nullptr ? getShape(origOp.bias()) : ShapeRef();

        const auto tileConf = backInferGroupConvTile(outputTile, origInputShape, origFilterShape, origBiasShape,
                                                     origOp.strides(), origOp.pads_begin(), origOp.pads_end());

        const auto inputTileType = getDenseTileType(inputType, tileConf.inputTile.offsets, tileConf.inputTile.shape);
        const auto filterTileType =
                getDenseTileType(filterType, tileConf.filterTile.offsets, tileConf.filterTile.shape);
        const auto outputTileType = getDenseTileType(outputType, outputTile.offsets, outputTile.shape);

        return mlir::succeeded(VPUIP::NCEInvariant::verifyConvCMX(origOp->getLoc(),
                                                                  origOp->getParentOfType<mlir::ModuleOp>(),
                                                                  inputTileType, filterTileType, outputTileType, log));
    });
}

bool isSupportedTiling(IE::MaxPoolOp origOp, const OutputTiling& tiles, Logger log) {
    const auto inputType = origOp.input().getType().cast<mlir::ShapedType>();
    const auto outputType = origOp.output().getType().cast<mlir::ShapedType>();

    return llvm::all_of(tiles, [&](const TileInfo& outputTile) {
        const auto origInputShape = getShape(origOp.input());

        const auto tileConf = backInferPoolTile(outputTile, origInputShape, origOp.kernel_size(), origOp.strides(),
                                                origOp.pads_begin(), origOp.pads_end());

        const auto inputTileType = getDenseTileType(inputType, tileConf.inputTile.offsets, tileConf.inputTile.shape);
        const auto outputTileType = getDenseTileType(outputType, outputTile.offsets, outputTile.shape);

        return mlir::succeeded(VPUIP::NCEInvariant::verifyPoolCMX(
                origOp->getLoc(), origOp->getParentOfType<mlir::ModuleOp>(), inputTileType, outputTileType,
                origOp.kernel_size(), origOp.strides(), log));
    });
}

//
// reifyTile
//

// TODO: convert this to operation interface

mlir::Value reifyTile(IE::ConvolutionOp origOp, const TileInfo& outputTile, mlir::OpBuilder& builder) {
    const auto origInputShape = getShape(origOp.input());
    const auto origFilterShape = getShape(origOp.filter());
    const auto origBiasShape = origOp.bias() != nullptr ? getShape(origOp.bias()) : ShapeRef();

    const auto tileConf = backInferConvTile(outputTile, origInputShape, origFilterShape, origBiasShape,
                                            origOp.strides(), origOp.pads_begin(), origOp.pads_end());

    const std::array<int64_t, 2> padsBegin = {tileConf.pads.top, tileConf.pads.left};
    const std::array<int64_t, 2> padsEnd = {tileConf.pads.bottom, tileConf.pads.right};

    const auto inputTileVal = makeTile(builder, origOp->getLoc(), origOp.input(), tileConf.inputTile, "input");
    const auto filterTileVal = makeTile(builder, origOp->getLoc(), origOp.filter(), tileConf.filterTile, "filter");
    const auto biasTileVal = origOp.bias() != nullptr
                                     ? makeTile(builder, origOp->getLoc(), origOp.bias(), tileConf.biasTile, "bias")
                                     : nullptr;

    const auto tileName = llvm::formatv("output tile {0}", outputTile.offsets).str();
    const auto tileLoc = appendLoc(origOp->getLoc(), tileName);

    const auto tiledResType = getDenseTileType(origOp.getType(), outputTile.offsets, outputTile.shape);

    auto tiledOp = builder.create<IE::ConvolutionOp>(tileLoc, tiledResType, inputTileVal, filterTileVal, biasTileVal,
                                                     origOp.stridesAttr(), getIntArrayAttr(builder, padsBegin),
                                                     getIntArrayAttr(builder, padsEnd), origOp.dilationsAttr(),
                                                     origOp.post_opAttr());

    return tiledOp.output();
}

mlir::Value reifyTile(IE::GroupConvolutionOp origOp, const TileInfo& outputTile, mlir::OpBuilder& builder) {
    const auto origInputShape = getShape(origOp.input());
    const auto origFilterShape = getShape(origOp.filter());
    const auto origBiasShape = origOp.bias() != nullptr ? getShape(origOp.bias()) : ShapeRef();

    const auto tileConf = backInferGroupConvTile(outputTile, origInputShape, origFilterShape, origBiasShape,
                                                 origOp.strides(), origOp.pads_begin(), origOp.pads_end());

    const std::array<int64_t, 2> padsBegin = {tileConf.pads.top, tileConf.pads.left};
    const std::array<int64_t, 2> padsEnd = {tileConf.pads.bottom, tileConf.pads.right};

    const auto inputTileVal = makeTile(builder, origOp->getLoc(), origOp.input(), tileConf.inputTile, "input");
    const auto filterTileVal = makeTile(builder, origOp->getLoc(), origOp.filter(), tileConf.filterTile, "filter");
    const auto biasTileVal = origOp.bias() != nullptr
                                     ? makeTile(builder, origOp->getLoc(), origOp.bias(), tileConf.biasTile, "bias")
                                     : nullptr;

    const auto tileName = llvm::formatv("output tile {0}", outputTile.offsets).str();
    const auto tileLoc = appendLoc(origOp->getLoc(), tileName);

    const auto groups = tileConf.filterTile.shape[Dims4D::Filter::OC];
    const auto groupsAttr = getIntAttr(builder, groups);

    const auto tiledResType = getDenseTileType(origOp.getType(), outputTile.offsets, outputTile.shape);

    auto tiledOp = builder.create<IE::GroupConvolutionOp>(
            tileLoc, tiledResType, inputTileVal, filterTileVal, biasTileVal, origOp.stridesAttr(),
            getIntArrayAttr(builder, padsBegin), getIntArrayAttr(builder, padsEnd), origOp.dilationsAttr(), groupsAttr,
            origOp.post_opAttr());

    return tiledOp.output();
}

mlir::Value reifyTile(IE::MaxPoolOp origOp, const TileInfo& outputTile, mlir::OpBuilder& builder) {
    const auto origInputShape = getShape(origOp.input());

    const auto tileConf = backInferPoolTile(outputTile, origInputShape, origOp.kernel_size(), origOp.strides(),
                                            origOp.pads_begin(), origOp.pads_end());

    const std::array<int64_t, 2> padsBegin = {tileConf.pads.top, tileConf.pads.left};
    const std::array<int64_t, 2> padsEnd = {tileConf.pads.bottom, tileConf.pads.right};

    const auto inputTileVal = makeTile(builder, origOp->getLoc(), origOp.input(), tileConf.inputTile, "input");

    const auto tileName = llvm::formatv("output tile {0}", outputTile.offsets).str();
    const auto tileLoc = appendLoc(origOp->getLoc(), tileName);

    const auto tiledResType = getDenseTileType(origOp.getType(), outputTile.offsets, outputTile.shape);

    auto tiledOp = builder.create<IE::MaxPoolOp>(tileLoc, tiledResType, inputTileVal, origOp.kernel_sizeAttr(),
                                                 origOp.stridesAttr(), getIntArrayAttr(builder, padsBegin),
                                                 getIntArrayAttr(builder, padsEnd), origOp.rounding_typeAttr(),
                                                 origOp.post_opAttr());

    return tiledOp.output();
}

//
// EltwiseTiling
//

class EltwiseTiling final : public mlir::OpTraitRewritePattern<IE::EltwiseOp> {
public:
    EltwiseTiling(mlir::MLIRContext* ctx, TilingGenerator<mlir::Operation*>&& generator, Logger log)
            : mlir::OpTraitRewritePattern<IE::EltwiseOp>(ctx), _generator(std::move(generator)), _log(log) {
        setDebugName("EltwiseTiling");
    }

    mlir::LogicalResult matchAndRewrite(mlir::Operation* origOp, mlir::PatternRewriter& rewriter) const final;

private:
    TilingGenerator<mlir::Operation*> _generator;
    Logger _log;
};

mlir::LogicalResult EltwiseTiling::matchAndRewrite(mlir::Operation* origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' Eltwise operation at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->getNumResults() != 1) {
        return matchFailed(rewriter, origOp, "Eltwise operations with multiple results are not supported in tiling");
    }

    for (const auto input : origOp->getOperands()) {
        const auto inShape = getShape(input);
        const auto outShape = getShape(origOp->getResult(0));

        if (inShape != outShape) {
            return matchFailed(rewriter, origOp, "Input shapes broadcasting is not supported in tiling");
        }
    }

    const auto tiles = _generator(origOp, _log.nest());

    _log.nest(1).trace("Create {0} tiles:", tiles.size());
    for (const auto& outputTile : tiles) {
        _log.nest(2).trace("{0}", outputTile);
    }

    const auto baseResType = origOp->getResult(0).getType().cast<mlir::ShapedType>();

    SmallVector<mlir::Value> resultTileVals;
    SmallVector<ShapeRef> resultTileOffsets;

    resultTileVals.reserve(tiles.size());
    resultTileOffsets.reserve(tiles.size());

    for (const auto& tile : tiles) {
        mlir::BlockAndValueMapping mapper;

        for (auto& origInput : origOp->getOpOperands()) {
            const auto valName = llvm::formatv("input {0}", origInput.getOperandNumber()).str();
            const auto tiledInput = makeTile(rewriter, origOp->getLoc(), origInput.get(), tile, valName);
            mapper.map(origInput.get(), tiledInput);
        }

        const auto tileName = llvm::formatv("output tile {0}", tile.offsets).str();
        const auto tileLoc = appendLoc(origOp->getLoc(), tileName);

        auto* tiledOp = rewriter.clone(*origOp, mapper);
        tiledOp->setLoc(tileLoc);

        auto tiledRes = tiledOp->getResult(0);

        const auto tiledResType = getDenseTileType(baseResType, tile.offsets, tile.shape);
        tiledRes.setType(tiledResType);

        resultTileVals.push_back(tiledRes);
        resultTileOffsets.push_back(tile.offsets);
    }

    rewriter.replaceOpWithNewOp<IE::ConcatOp>(origOp, origOp->getResult(0).getType(), mlir::ValueRange(resultTileVals),
                                              makeArrayRef(resultTileOffsets));
    return mlir::success();
}

//
// GenericTiling
//

template <class ConcreteOp>
class GenericTiling final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    GenericTiling(mlir::MLIRContext* ctx, TilingGenerator<ConcreteOp>&& generator, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _generator(std::move(generator)), _log(log) {
        this->setDebugName("GenericTiling");
    }

    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    TilingGenerator<ConcreteOp> _generator;
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult GenericTiling<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                               mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    const auto tiles = _generator(origOp, _log.nest());

    _log.nest(1).trace("Create {0} tiles:", tiles.size());
    for (const auto& outputTile : tiles) {
        _log.nest(2).trace("{0}", outputTile);
    }

    SmallVector<mlir::Value> resultTileVals;
    SmallVector<ShapeRef> resultTileOffsets;

    resultTileVals.reserve(tiles.size());
    resultTileOffsets.reserve(tiles.size());

    for (const auto& outputTile : tiles) {
        const auto tiledRes = reifyTile(origOp, outputTile, rewriter);

        const auto tiledShape = getShape(tiledRes);
        VPUX_THROW_UNLESS(tiledShape == outputTile.shape,
                          "Inferred tiled output shape '{0}' doesn't match with generated '{1}'", tiledShape,
                          outputTile.shape);

        resultTileVals.push_back(tiledRes);
        resultTileOffsets.push_back(outputTile.offsets);
    }

    rewriter.replaceOpWithNewOp<IE::ConcatOp>(origOp, origOp.getType(), mlir::ValueRange(resultTileVals),
                                              makeArrayRef(resultTileOffsets));
    return mlir::success();
}

//
// generateTiles
//

OutputTiling generateTiles(mlir::Operation* op, mlir::ShapedType outputType,
                           FuncRef<bool(ShapeRef)> isSupportedTileSize) {
    const auto outputShape = getShape(outputType);

    int64_t minChannelSize = 1;
    if (auto iface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op)) {
        minChannelSize = iface.getChannelAlignment();
    }

    const auto maxChannelTiles = outputShape[Dims4D::Act::C] / minChannelSize;

    Shape nTilesOnDim(outputShape.size(), 1);

    // Try to tile the largest dim (C or H) first, then proceed with other dims
    SmallVector<Dim> tileDimOrder = {Dims4D::Act::C, Dims4D::Act::H, Dims4D::Act::W};
    if (outputShape[Dims4D::Act::C] < outputShape[Dims4D::Act::H]) {
        tileDimOrder = {Dims4D::Act::H, Dims4D::Act::C, Dims4D::Act::W};
    }

    auto tileDimIter = tileDimOrder.begin();
    auto dimToTile = *tileDimIter;

    const auto isSupportedChannelDivision = [&]() {
        if ((outputShape[Dims4D::Act::C] % nTilesOnDim[Dims4D::Act::C]) != 0) {
            return false;
        }

        const auto tileChannels = outputShape[Dims4D::Act::C] / nTilesOnDim[Dims4D::Act::C];
        return (tileChannels % minChannelSize) == 0;
    };

    const auto isDimLeftToTile = [&]() {
        if (dimToTile == Dims4D::Act::C) {
            return nTilesOnDim[Dims4D::Act::C] < maxChannelTiles;
        }

        // Spatial dims
        const auto origSize = static_cast<double>(outputShape[dimToTile]);
        const auto prevDivisor = static_cast<double>(nTilesOnDim[dimToTile]);
        return (origSize / prevDivisor) > 1.0;
    };

    while (!isSupportedTileSize(nTilesOnDim)) {
        VPUX_THROW_WHEN(tileDimIter == tileDimOrder.end(), "Failed to tile {0} at '{1}'", op->getName(), op->getLoc());

        if (!isDimLeftToTile()) {
            dimToTile = *(++tileDimIter);
        }

        if (dimToTile == Dims4D::Act::C) {
            do {
                ++nTilesOnDim[Dims4D::Act::C];
            } while (!isSupportedChannelDivision());
        } else if (dimToTile == Dims4D::Act::H || dimToTile == Dims4D::Act::W) {
            nTilesOnDim[dimToTile]++;
        } else {
            // Trying to tile in unsupported dimension, tiling in supported dimensions not sufficient
            VPUX_THROW("Failed to tile {0} at '{1}'", op->getName(), op->getLoc());
        }
    }

    return fillDividedTiles(nTilesOnDim, outputShape);
}

//
// eltwiseTilingGenerator
//

OutputTiling eltwiseTilingGenerator(mlir::Operation* origOp, Logger log) {
    VPUX_THROW_UNLESS(origOp->getNumOperands() == 2, "Unsupported Eltwise operation '{0}'", origOp->getName());
    VPUX_THROW_UNLESS(origOp->getNumResults() == 1, "Unsupported Eltwise operation '{0}'", origOp->getName());

    const auto outputType = origOp->getResult(0).getType().cast<mlir::ShapedType>();
    const auto outputShape = getShape(outputType);

    const auto isSupportedTileSize = [&](ShapeRef nTilesOnDim) -> bool {
        const auto tiles = fillDividedTiles(nTilesOnDim, outputShape);
        return isSupportedEltwiseTiling(origOp, tiles, log);
    };

    return generateTiles(origOp, outputType, isSupportedTileSize);
}

//
// convTilingGenerator
//

OutputTiling convTilingGenerator(IE::ConvolutionOp origOp, Logger log) {
    const auto outputType = origOp.output().getType().cast<mlir::ShapedType>();
    const auto outputShape = getShape(outputType);

    const auto isSupportedTileSize = [&](ShapeRef nTilesOnDim) -> bool {
        const auto tiles = fillDividedTiles(nTilesOnDim, outputShape);
        return isSupportedTiling(origOp, tiles, log);
    };

    return generateTiles(origOp, outputType, isSupportedTileSize);
}

//
// groupConvTilingGenerator
//

OutputTiling generateGroupConvTiles(mlir::Operation* op, mlir::ShapedType outputType,
                                    FuncRef<bool(ShapeRef)> isSupportedTileSize) {
    const auto outputShape = getShape(outputType);

    Shape nTilesOnDim(outputShape.size(), 1);

    if (auto iface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op)) {
        const auto chanAlignment = iface.getChannelAlignment();

        VPUX_THROW_UNLESS(outputShape[Dims4D::Act::C] % chanAlignment == 0,
                          "Depthwise convolution output channels must be a multiple of {0}, got {1}", chanAlignment,
                          outputShape[Dims4D::Act::C]);

        nTilesOnDim[Dims4D::Act::C] = outputShape[Dims4D::Act::C] / chanAlignment;
    }

    while (!isSupportedTileSize(nTilesOnDim)) {
        Optional<Dim> dimToTile;

        for (auto ind : irange(Dims4D::Act::numSpatialDims)) {
            const auto spatialDim = Dims4D::Act::getSpatialDim(ind);

            const auto origSize = static_cast<double>(outputShape[spatialDim]);
            const auto prevDivisor = static_cast<double>(nTilesOnDim[spatialDim]);

            if ((origSize / prevDivisor) > 1.0) {
                dimToTile = spatialDim;
                break;
            }
        }

        VPUX_THROW_UNLESS(dimToTile.hasValue(), "Failed to tile {0} at '{1}'", op->getName(), op->getLoc());
        nTilesOnDim[dimToTile.getValue()]++;
    }

    return fillDividedTiles(nTilesOnDim, outputShape);
}

OutputTiling groupConvTilingGenerator(IE::GroupConvolutionOp origOp, Logger log) {
    const auto outputType = origOp.output().getType().cast<mlir::ShapedType>();
    const auto outputShape = getShape(outputType);

    const auto isSupportedTileSize = [&](ShapeRef nTilesOnDim) -> bool {
        const auto tiles = fillDividedTiles(nTilesOnDim, outputShape);
        return isSupportedTiling(origOp, tiles, log);
    };

    return generateGroupConvTiles(origOp, outputType, isSupportedTileSize);
}

//
// maxPoolTilingGenerator
//

OutputTiling maxPoolTilingGenerator(IE::MaxPoolOp origOp, Logger log) {
    const auto outputType = origOp.output().getType().cast<mlir::ShapedType>();
    const auto outputShape = getShape(outputType);

    const auto isSupportedTileSize = [&](ShapeRef nTilesOnDim) -> bool {
        const auto tiles = fillDividedTiles(nTilesOnDim, outputShape);
        return isSupportedTiling(origOp, tiles, log);
    };

    return generateTiles(origOp, outputType, isSupportedTileSize);
}

//
// IsolatedTilingPass
//

class IsolatedTilingPass final : public IE::IsolatedTilingBase<IsolatedTilingPass> {
public:
    explicit IsolatedTilingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

template <class ConcreteOp>
void addLayerToTarget(mlir::ConversionTarget& target, Logger log) {
    target.addDynamicallyLegalOp<ConcreteOp>([log](ConcreteOp op) {
        return !needTiling(op, log);
    });
}

void IsolatedTilingPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    addLayerToTarget<IE::AddOp>(target, _log);
    addLayerToTarget<IE::MultiplyOp>(target, _log);
    addLayerToTarget<IE::SubtractOp>(target, _log);
    addLayerToTarget<IE::AndOp>(target, _log);
    addLayerToTarget<IE::ConvolutionOp>(target, _log);
    addLayerToTarget<IE::GroupConvolutionOp>(target, _log);
    addLayerToTarget<IE::MaxPoolOp>(target, _log);
    target.addLegalOp<IE::SliceOp, IE::ConcatOp>();
    target.markUnknownOpDynamicallyLegal([](mlir::Operation*) {
        return true;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<EltwiseTiling>(&ctx, eltwiseTilingGenerator, _log);
    patterns.add<GenericTiling<IE::ConvolutionOp>>(&ctx, convTilingGenerator, _log);
    patterns.add<GenericTiling<IE::GroupConvolutionOp>>(&ctx, groupConvTilingGenerator, _log);
    patterns.add<GenericTiling<IE::MaxPoolOp>>(&ctx, maxPoolTilingGenerator, _log);

    if (mlir::failed(mlir::applyPartialConversion(getFunction(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createIsolatedTilingPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createIsolatedTilingPass(Logger log) {
    return std::make_unique<IsolatedTilingPass>(log);
}
