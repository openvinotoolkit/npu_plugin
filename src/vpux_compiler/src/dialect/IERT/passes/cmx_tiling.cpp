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
#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <functional>
#include <numeric>

using namespace vpux;

namespace {

struct Tile final {
    Tile() = delete;

    explicit Tile(size_t rank): shape(rank), offsets(rank) {
    }

    explicit Tile(ShapeRef shape): shape(shape.raw()), offsets(shape.size(), 0) {
    }

    Shape shape;
    Shape offsets;
};

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

// helper function to generate a set of tiles from dividing a shape. A shape divided across multiple dimensions will
// generate a set of tiles, each having its own size and offsets
void fillDividedTiles(MutableArrayRef<Tile> dividedTiles, ShapeRef divisors, ShapeRef orig) {
    int64_t repeatCtr = 1;

    for (auto d : irange(divisors.size())) {
        const auto dim = Dim(d);

        const auto origSize = orig[dim];
        const auto divisor = divisors[dim];
        VPUX_THROW_UNLESS(divisor != 0, "Cannot divide by 0 tiles");

        if (divisor == 1) {
            for (auto i : irange(dividedTiles.size())) {
                dividedTiles[i].shape[dim] = origSize;
                dividedTiles[i].offsets[dim] = 0;
            }

            continue;
        }

        const auto tileSize = origSize / divisor;

        int64_t offset = 0;
        for (int64_t i : irange(dividedTiles.size())) {
            const bool remainderTile = !(((i / repeatCtr) + 1) % (divisor));

            if (remainderTile) {
                dividedTiles[i].shape[dim] = origSize - (tileSize * (divisor - 1));
            } else {
                dividedTiles[i].shape[dim] = tileSize;
            }

            dividedTiles[i].offsets[dim] = offset;

            const bool incrementOffset = !((i + 1) % repeatCtr);
            if (incrementOffset) {
                offset += tileSize;
            }

            const bool resetOffset = (remainderTile && incrementOffset);
            if (resetOffset) {
                offset = 0;
            }
        }

        repeatCtr *= divisor;
    }
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

    SmallVector<Tile> finalTiles(nTilesOnDim.totalSize(), Tile(nTilesOnDim.size()));
    fillDividedTiles(finalTiles, nTilesOnDim, outputShape);

    return finalTiles;
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

mlir::MemRefType reduceShape(mlir::MemRefType origType, ShapeRef newShape) {
    const auto order = DimsOrder::fromType(origType);
    const auto orderAffineMap = order.toAffineMap(origType.getContext());
    return mlir::MemRefType::get(newShape.raw(), origType.getElementType(), makeArrayRef(orderAffineMap),
                                 origType.getMemorySpace());
}

mlir::Value makeTile(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value origVal, const Tile& tile) {
    const auto origType = origVal.getType().cast<mlir::MemRefType>();

    if (tile.shape == getShape(origType)) {
        return origVal;
    }

    SmallVector<int64_t> viewStrides(tile.shape.size(), 1);
    auto viewOp =
            builder.create<mlir::memref::SubViewOp>(loc, origVal, tile.offsets.raw(), tile.shape.raw(), viewStrides);

    const auto tileType = reduceShape(origType, tile.shape);
    auto allocOp = builder.create<mlir::memref::AllocOp>(loc, tileType);

    auto copyOp = builder.create<IERT::CopyOp>(loc, viewOp.result(), allocOp.memref());
    return copyOp.output();
}

mlir::LogicalResult CMXTilingPass::ConvolutionTiling::matchAndRewrite(IERT::ConvolutionOp origOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    using InputTileConfig = std::tuple<Tile, Tile, Tile, SmallVector<int64_t>, SmallVector<int64_t>>;

    auto* ctx = rewriter.getContext();

    auto tilings = _tiler(origOp);

    if (tilings.size() < 2) {
        return matchFailed(rewriter, origOp, "No tiling required");
    }

    const auto origOutput = getShape(origOp.output());
    const auto origInput = getShape(origOp.input());
    const auto origWeights = getShape(origOp.filter());
    const auto origBias = origOp.bias() != nullptr ? getShape(origOp.bias()) : ShapeRef();

    auto backInferConvInput = [&](Tile& outputTile) -> InputTileConfig {
        Tile inputTile(origInput);
        Tile weightsTile(origWeights);
        Tile biasTile(origBias);

        SmallVector<int64_t> padsBegin(origOp.filter_spatial_dims());
        SmallVector<int64_t> padsEnd(origOp.filter_spatial_dims());

        for (auto spatialDim : irange(origOp.filter_spatial_dims())) {
            const auto act_spatial_dim = origOp.act_spatial_dim(spatialDim);
            const auto filter_spatial_dim = origOp.filter_spatial_dim(spatialDim);

            const auto outSize = outputTile.shape[act_spatial_dim];
            const auto outOffset = outputTile.offsets[act_spatial_dim];

            const auto opPadBegin = origOp.pads_begin()[spatialDim].cast<mlir::IntegerAttr>().getInt();
            const auto opPadEnd = origOp.pads_begin()[spatialDim].cast<mlir::IntegerAttr>().getInt();

            const auto kSize = origWeights[filter_spatial_dim];
            const auto kStride = origOp.strides()[spatialDim].cast<mlir::IntegerAttr>().getInt();

            const int64_t tilePadStart = outOffset == 0 ? opPadBegin : 0;
            const int64_t tilePadEnd = (outOffset + outSize) == origOutput[act_spatial_dim] ? opPadEnd : 0;

            const int64_t inputSize = ((outSize - 1) * kStride) - tilePadStart - tilePadEnd + kSize;
            const int64_t inputOffset = outOffset != 0 ? outOffset * kStride - opPadBegin - ((kSize - 1) / 2) : 0;

            inputTile.shape[act_spatial_dim] = inputSize;
            inputTile.offsets[act_spatial_dim] = inputOffset;

            padsBegin[spatialDim] = tilePadStart;
            padsEnd[spatialDim] = tilePadEnd;
        }

        // do the kernel tile
        inputTile.shape[origOp.act_channel_dim()] =
                origInput[origOp.act_channel_dim()];  // will not tile on InputChannels
        inputTile.offsets[origOp.act_channel_dim()] = 0;

        weightsTile.shape[origOp.filter_out_channel_dim()] = outputTile.shape[origOp.act_channel_dim()];
        weightsTile.offsets[origOp.filter_out_channel_dim()] = outputTile.offsets[origOp.act_channel_dim()];

        if (!biasTile.shape.empty()) {
            biasTile.shape[origOp.act_channel_dim()] = outputTile.shape[origOp.act_channel_dim()];
            biasTile.offsets[origOp.act_channel_dim()] = outputTile.offsets[origOp.act_channel_dim()];
        }

        // do the batch dim
        inputTile.shape[origOp.act_batch_dim()] = outputTile.shape[origOp.act_batch_dim()];
        inputTile.offsets[origOp.act_batch_dim()] = outputTile.offsets[origOp.act_batch_dim()];

        return {inputTile, weightsTile, biasTile, padsBegin, padsEnd};
    };

    _log.trace("For ConvOp '{0}' have '{1}' tiles: ", origOp->getLoc(), tilings.size());
    for (auto i : irange(tilings.size())) {
        _log.nest().trace("shape '{0}'", i, tilings[i].shape);
    }
    for (auto i : irange(tilings.size())) {
        _log.nest().trace("offsets '{0}'", i, tilings[i].offsets);
    }

    SmallVector<mlir::Value> finalResults;

    for (auto i : irange(tilings.size())) {
        const auto inputConfig = backInferConvInput(tilings[i]);

        const auto inputTile = std::get<0>(inputConfig);
        const auto weightsTile = std::get<1>(inputConfig);
        const auto biasTile = std::get<2>(inputConfig);

        const auto padsBegin = std::get<3>(inputConfig);
        const auto padsEnd = std::get<4>(inputConfig);

        const auto actInput = makeTile(rewriter, origOp->getLoc(), origOp.input(), inputTile);
        const auto weightInput = makeTile(rewriter, origOp->getLoc(), origOp.filter(), weightsTile);
        const auto biasInput =
                origOp.bias() != nullptr ? makeTile(rewriter, origOp->getLoc(), origOp.bias(), biasTile) : nullptr;

        const auto tileTypeOut = reduceShape(origOp.output_buff().getType().cast<mlir::MemRefType>(), tilings[i].shape);
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

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(tilingPatterns)))) {
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
