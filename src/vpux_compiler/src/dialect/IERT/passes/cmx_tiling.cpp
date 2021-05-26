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

struct Tile {
    Tile() = delete;
    explicit Tile(int64_t rank): sizes(rank), offsets(rank){};
    explicit Tile(const mlir::ShapedType& shape)
            : sizes(shape.getShape().begin(), shape.getShape().end()), offsets(shape.getRank(), 0){};

    SmallVector<int64_t> sizes;
    SmallVector<int64_t> offsets;
};

template <class InputIt, class T>
inline T product(InputIt start, InputIt end, T init) {
    return std::accumulate(start, end, init, std::multiplies<T>());
}

vpux::Dim largestTileDim(const mlir::ShapedType& shape, llvm::ArrayRef<int64_t> nTilesOnDim) {
    int64_t maxSize = 0;
    unsigned maxDim = 0;

    for (auto i : irange(shape.getRank())) {
        auto dim = static_cast<unsigned int>(i);
        VPUX_THROW_UNLESS(nTilesOnDim[dim] != 0, "Cannot divide by 0 tiles");

        int64_t tiledDimSize = shape.getDimSize(dim) / nTilesOnDim[dim];
        if (tiledDimSize > maxSize) {
            maxSize = tiledDimSize;
            maxDim = dim;
        }
    }

    return vpux::Dim(maxDim);
}

// helper function to generate a set of tiles from dividing a shape. A shape divided across multiple dimensions will
// generate a set of tiles , each having its own size and offsets
void fillDividedTiles(llvm::MutableArrayRef<Tile> dividedTiles, llvm::ArrayRef<int64_t> divisors,
                      const mlir::ShapedType& orig) {
    int64_t repeatCtr = 1;

    for (auto dim : irange(divisors.size())) {
        auto origSize = orig.getDimSize(static_cast<unsigned int>(dim));
        auto divisor = divisors[dim];

        if (divisor == 1) {
            for (auto i : irange(dividedTiles.size())) {
                dividedTiles[i].sizes[dim] = origSize;
                dividedTiles[i].offsets[dim] = 0;
            }
            continue;
        }

        int64_t offset = 0;
        int64_t tileSize = origSize / divisor;
        for (int64_t i : irange(dividedTiles.size())) {
            bool remainderTile = !(((i / repeatCtr) + 1) % (divisor));

            if (remainderTile)
                dividedTiles[i].sizes[dim] = origSize - (tileSize * (divisor - 1));
            else
                dividedTiles[i].sizes[dim] = tileSize;

            dividedTiles[i].offsets[dim] = offset;

            bool incrementOffset = !((i + 1) % repeatCtr);
            if (incrementOffset) {
                offset += tileSize;
            }
            bool resetOffset = (remainderTile && incrementOffset);
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
    void safeRunOnModule() final;
};

class CMXTilingPass::SimpleFillHalfCMXTiler {
public:
    explicit SimpleFillHalfCMXTiler(Byte cmxSize, mlir::MLIRContext* ctx, Logger log)
            : _memSize(cmxSize.count() / FillFactor), _ctx(ctx), _log(log) {
    }

    llvm::SmallVector<Tile> convolutionTiler(IERT::ConvolutionOp op) const;

    void buildTilingPatterns(mlir::RewritePatternSet& patterns);

private:
    Byte _memSize;
    mlir::MLIRContext* _ctx;
    Logger _log;
    static const unsigned FillFactor = 2;
};

void CMXTilingPass::SimpleFillHalfCMXTiler::buildTilingPatterns(mlir::RewritePatternSet& patterns) {
    auto convTilerFunc = std::bind(&SimpleFillHalfCMXTiler::convolutionTiler, this, std::placeholders::_1);

    patterns.insert<CMXTilingPass::ConvolutionTiling>(_ctx, convTilerFunc, _log);
}

llvm::SmallVector<Tile> CMXTilingPass::SimpleFillHalfCMXTiler::convolutionTiler(IERT::ConvolutionOp op) const {
    const auto inputType = op.input().getType().cast<mlir::MemRefType>();
    const auto outputType = op.output().getType().cast<mlir::MemRefType>();
    const auto weightType = op.filter().getType().cast<mlir::MemRefType>();
    const auto bias = op.bias();
    const auto biasType = (bias != nullptr) ? bias.getType().cast<mlir::MemRefType>() : nullptr;
    const auto actChannelDim = static_cast<int64_t>(op.act_channel_dim().ind());

    auto convTileSize = [&](llvm::ArrayRef<int64_t> nTilesOnDim) -> Byte {
        Byte iSize = getElemTypeSize(inputType);
        Byte oSize = getElemTypeSize(outputType);

        for (auto i : irange(inputType.getRank())) {
            auto dim = static_cast<unsigned int>(i);
            VPUX_THROW_UNLESS(nTilesOnDim[dim] != 0, "Cannot divide by 0 tiles");

            if (i == actChannelDim)
                iSize *= inputType.getDimSize(dim);
            else
                iSize *= inputType.getDimSize(dim) / nTilesOnDim[dim];
        }

        for (auto i : irange(outputType.getRank())) {
            auto dim = static_cast<unsigned int>(i);
            VPUX_THROW_UNLESS(nTilesOnDim[dim] != 0, "Cannot divide by 0 tiles");

            oSize *= outputType.getDimSize(dim) / nTilesOnDim[dim];
        }

        VPUX_THROW_UNLESS(nTilesOnDim[actChannelDim] != 0, "Cannot divide by 0 tiles");

        Byte wSize = getTypeTotalSize(weightType) / nTilesOnDim[actChannelDim];
        Byte bSize = (biasType != nullptr) ? getTypeTotalSize(biasType) / nTilesOnDim[actChannelDim] : Byte(0);

        return iSize + oSize + wSize + bSize;
    };

    llvm::SmallVector<int64_t> nTilesOnDim(outputType.getRank(), 1);

    while (convTileSize(nTilesOnDim) > _memSize) {
        auto dim = largestTileDim(outputType, nTilesOnDim);
        nTilesOnDim[dim.ind()]++;
    }

    auto totalTiles = product(nTilesOnDim.begin(), nTilesOnDim.end(), static_cast<int64_t>(1));
    llvm::SmallVector<Tile> finalTiles(totalTiles, Tile(nTilesOnDim.size()));

    fillDividedTiles(finalTiles, nTilesOnDim, outputType);

    return finalTiles;
}

class CMXTilingPass::ConvolutionTiling final : public mlir::OpRewritePattern<IERT::ConvolutionOp> {
public:
    using TilerFunc = std::function<llvm::SmallVector<Tile>(IERT::ConvolutionOp)>;

    ConvolutionTiling(mlir::MLIRContext* ctx, TilerFunc tiler, Logger log)
            : mlir::OpRewritePattern<IERT::ConvolutionOp>(ctx), _tiler(tiler), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(IERT::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    TilerFunc _tiler;
    Logger _log;
};

mlir::LogicalResult CMXTilingPass::ConvolutionTiling::matchAndRewrite(IERT::ConvolutionOp origOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    using InputTileConfig = std::tuple<Tile, Tile, Tile, SmallVector<int64_t>, SmallVector<int64_t>>;

    llvm::SmallVector<Tile> tilings = _tiler(origOp);
    mlir::MLIRContext* ctx = rewriter.getContext();
    // const unsigned convWeightsSpatialFilterOffset = 2;  // Based on IRv10 spec TODO(review) : are there macros for
    // this?

    if (tilings.size() < 2) {
        _log.trace("No tiling required for ConvolutionOp '{0}'", origOp->getLoc());
        return mlir::failure();
    }

    const auto bias = origOp.bias();

    auto backInferConvInput = [&](Tile& outputTile, IERT::ConvolutionOp& op) -> InputTileConfig {
        const auto origOutput = op.output().getType().cast<mlir::ShapedType>();
        const auto origInput = op.input().getType().cast<mlir::ShapedType>();
        const auto weights = op.filter().getType().cast<mlir::ShapedType>();
        const auto rank = outputTile.sizes.size();
        const auto spatialDims = op.filter_spatial_dims();

        Tile inputTile(rank), weightsTile(weights), biasTile(1);
        llvm::SmallVector<int64_t> padsBegin(spatialDims), padsEnd(spatialDims);

        for (unsigned spatialDim = 0; spatialDim < spatialDims; spatialDim++) {
            const auto tensorDim = static_cast<unsigned int>(op.act_spatial_dim(spatialDim).ind());
            const auto spatialKernelDim = static_cast<unsigned int>(op.filter_spatial_dim(spatialDim).ind());

            const auto outSize = outputTile.sizes[tensorDim];
            const auto outOffset = outputTile.offsets[tensorDim];

            const auto opPadBegin = op.pads_begin()[spatialDim].cast<mlir::IntegerAttr>().getInt();
            const auto opPadEnd = op.pads_begin()[spatialDim].cast<mlir::IntegerAttr>().getInt();
            const auto kSize = weights.getDimSize(spatialKernelDim);
            const auto kStride = op.strides()[spatialDim].cast<mlir::IntegerAttr>().getInt();

            const int64_t tilePadStart = outOffset == 0 ? opPadBegin : 0;
            const int64_t tilePadEnd = (outOffset + outSize) == origOutput.getDimSize(tensorDim) ? opPadEnd : 0;

            const int64_t inputSize = ((outSize - 1) * kStride) - tilePadStart - tilePadEnd + kSize;
            const int64_t inputOffset = outOffset != 0 ? outOffset * kStride - opPadBegin - ((kSize - 1) / 2) : 0;

            inputTile.sizes[tensorDim] = inputSize;
            inputTile.offsets[tensorDim] = inputOffset;
            padsBegin[spatialDim] = tilePadStart;
            padsEnd[spatialDim] = tilePadEnd;
        }

        // do the kernel tile
        auto actChannelDim = static_cast<unsigned int>(op.act_channel_dim().ind());
        auto weightsOutChannelDim = static_cast<unsigned int>(op.filter_out_channel_dim().ind());
        inputTile.sizes[actChannelDim] = origInput.getDimSize(actChannelDim);  // will not tile on InputChannels
        inputTile.offsets[actChannelDim] = 0;

        weightsTile.sizes[weightsOutChannelDim] = outputTile.sizes[actChannelDim];
        weightsTile.offsets[weightsOutChannelDim] = outputTile.offsets[actChannelDim];
        biasTile.sizes[0] = outputTile.sizes[actChannelDim];
        biasTile.offsets[0] = outputTile.offsets[actChannelDim];

        // do the batch dim
        auto batchDim = op.act_batch_dim().ind();
        inputTile.sizes[batchDim] = outputTile.sizes[batchDim];
        inputTile.offsets[batchDim] = outputTile.offsets[batchDim];

        return {inputTile, weightsTile, biasTile, padsBegin, padsEnd};
    };

    _log.trace("For ConvOp '{0}' have '{1}' tiles: ", origOp->getLoc(), tilings.size());
    for (auto i : irange(tilings.size()))
        _log.trace("\tdims '{0}' '{1}' '{2}' '{3}' '{4}'", i, tilings[i].sizes[0], tilings[i].sizes[1],
                   tilings[i].sizes[2], tilings[i].sizes[3]);
    for (auto i : irange(tilings.size()))
        _log.trace("\toffs '{0}' '{1}' '{2}' '{3}' '{4}'", i, tilings[i].offsets[0], tilings[i].offsets[1],
                   tilings[i].offsets[2], tilings[i].offsets[3]);

    llvm::SmallVector<int64_t> actSvStrides(tilings[0].sizes.size(), 1);
    llvm::SmallVector<int64_t> weightSvStrides(origOp.filter().getType().cast<mlir::ShapedType>().getRank(), 1);
    llvm::SmallVector<int64_t> biasSvStrides(1, 1);
    SmallVector<mlir::Value> finalResults;

    for (auto i : irange(tilings.size())) {
        const auto inputConfig = backInferConvInput(tilings[i], origOp);
        const auto inputTile = std::get<0>(inputConfig);
        const auto weightsTile = std::get<1>(inputConfig);
        const auto biasTile = std::get<2>(inputConfig);
        const auto padsBegin = std::get<3>(inputConfig);
        const auto padsEnd = std::get<4>(inputConfig);

        const auto inputShape = origOp.input().getType().cast<mlir::ShapedType>();
        const auto weightsShape = origOp.filter().getType().cast<mlir::ShapedType>();

        mlir::Value actInput, weightInput, biasInput;
        ;
        if (llvm::ArrayRef<int64_t>(inputTile.sizes) != inputShape.getShape()) {
            auto subViewIn = rewriter.create<mlir::memref::SubViewOp>(origOp->getLoc(), origOp.input(),
                                                                      inputTile.offsets, inputTile.sizes, actSvStrides);
            actInput = subViewIn;
        } else {
            actInput = origOp.input();
        }

        if (llvm::ArrayRef<int64_t>(weightsTile.sizes) != weightsShape.getShape()) {
            auto subViewWeights = rewriter.create<mlir::memref::SubViewOp>(
                    origOp->getLoc(), origOp.filter(), weightsTile.offsets, weightsTile.sizes, weightSvStrides);

            if (bias) {
                biasInput = rewriter.create<mlir::memref::SubViewOp>(origOp->getLoc(), origOp.bias(), biasTile.offsets,
                                                                     biasTile.sizes, biasSvStrides);
            } else {
                biasInput = nullptr;
            }

            weightInput = subViewWeights;

        } else {
            weightInput = origOp.filter();
            biasInput = origOp.bias();
        }

        auto subViewOut = rewriter.create<mlir::memref::SubViewOp>(origOp->getLoc(), origOp.output_buff(),
                                                                   tilings[i].offsets, tilings[i].sizes, actSvStrides);

        auto tileTypeIn = actInput.getType().cast<mlir::MemRefType>();
        auto tileTypeWeights = weightInput.getType().cast<mlir::MemRefType>();
        auto tileTypeBias = (bias != nullptr) ? biasInput.getType().cast<mlir::MemRefType>() : nullptr;
        auto tileTypeOut = subViewOut.getType();

        auto allocInput = rewriter.create<mlir::memref::AllocOp>(origOp.getLoc(), tileTypeIn);
        auto allocWeights = rewriter.create<mlir::memref::AllocOp>(origOp.getLoc(), tileTypeWeights);
        auto allocBias =
                (bias != nullptr) ? rewriter.create<mlir::memref::AllocOp>(origOp.getLoc(), tileTypeBias) : nullptr;
        auto allocOut = rewriter.create<mlir::memref::AllocOp>(origOp.getLoc(), tileTypeOut);

        auto copyInput = rewriter.create<IERT::CopyOp>(origOp.getLoc(), actInput, allocInput);
        auto copyWeights = rewriter.create<IERT::CopyOp>(origOp.getLoc(), weightInput, allocWeights);
        mlir::Value copyBias =
                (bias != nullptr) ? rewriter.create<IERT::CopyOp>(origOp.getLoc(), biasInput, allocBias).getResult()
                                  : nullptr;

        auto conv = rewriter.create<IERT::ConvolutionOp>(origOp.getLoc(), copyInput, copyWeights, copyBias, allocOut,
                                                         origOp.strides(), getInt32ArrayAttr(ctx, padsBegin),
                                                         getInt32ArrayAttr(ctx, padsEnd), origOp.dilations());
        auto copyOut = rewriter.create<IERT::CopyOp>(origOp.getLoc(), conv, subViewOut);
        finalResults.push_back(copyOut);
    }

    rewriter.replaceOpWithNewOp<IERT::ConcatViewOp>(origOp, finalResults, origOp.output_buff());

    return mlir::success();
}

void CMXTilingPass::safeRunOnModule() {
    mlir::MLIRContext& ctx = getContext();

    // TODO(review): should we add the "known HW ops" as dinamically legal by some estimated condition that they would
    // fit into CMX??

    auto module = getOperation();
    mlir::RewritePatternSet tilingPatterns(&ctx);

    auto rtOp = IERT::RunTimeResourcesOp::getFromModule(module);
    if (!rtOp) {
        _log.trace("Cannot get RunTimeResourcesOp");
        signalPassFailure();
        return;
    }

    auto cmxSize = rtOp.getAvailableMemory(VPUIP::PhysicalMemoryAttr::get(&ctx, VPUIP::PhysicalMemory::CMX_NN));
    SimpleFillHalfCMXTiler simpleTiler(cmxSize.size(), &ctx, _log);
    simpleTiler.buildTilingPatterns(tilingPatterns);

    mlir::FrozenRewritePatternSet frozenPattenrs(std::move(tilingPatterns));

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(module, frozenPattenrs, 2))) {
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
