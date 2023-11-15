//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/VPU37XX/dialect/VPU/passes.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

int64_t getAxisValue(VPU::GatherOp op) {
    VPUX_THROW_UNLESS(op.axis_valueAttr() != nullptr, "Axis value is required for GatherOp");
    int64_t axisValue = op.axis_valueAttr().dyn_cast_or_null<mlir::IntegerAttr>().getValue().getSExtValue();

    return axisValue;
}

vpux::InputTiling backInferTileInfoDecomposedGather(VPU::GatherOp op, const vpux::TileInfo& outputTile, vpux::Logger) {
    const auto origInputShape = getShape(op.input());
    const auto origIndicesShape = getShape(op.indices());

    int64_t axisValue = getAxisValue(op);

    VPUX_THROW_UNLESS(op.batch_dimsAttr() != nullptr, "BatchDims is required for GatherOp");
    int64_t batchDims = op.batch_dimsAttr().dyn_cast_or_null<mlir::IntegerAttr>().getValue().getSExtValue();

    TileInfo inputTile(origInputShape);
    TileInfo indicesTile(origIndicesShape);

    auto inputRank = origInputShape.size();
    auto indicesRank = origIndicesShape.size();

    for (int64_t i = 0; i < static_cast<int64_t>(inputRank); ++i) {
        if (i < axisValue) {
            inputTile.shape[Dim(i)] = outputTile.shape[Dim(i)];
            inputTile.offsets[Dim(i)] = outputTile.offsets[Dim(i)];
        } else if (i == axisValue) {
            continue;
        } else {
            inputTile.shape[Dim(i)] = outputTile.shape[Dim(i + indicesRank - batchDims - 1)];
            inputTile.offsets[Dim(i)] = outputTile.offsets[Dim(i + indicesRank - batchDims - 1)];
        }
    }

    for (int64_t i = 0; i < static_cast<int64_t>(indicesRank); ++i) {
        if (i < batchDims) {
            indicesTile.shape[Dim(i)] = outputTile.shape[Dim(i)];
            indicesTile.offsets[Dim(i)] = outputTile.offsets[Dim(i)];
        } else {
            indicesTile.shape[Dim(i)] = outputTile.shape[Dim(i + axisValue - batchDims)];
            indicesTile.offsets[Dim(i)] = outputTile.offsets[Dim(i + axisValue - batchDims)];
        }
    }

    return InputTiling{{std::move(inputTile), std::move(indicesTile)}};
}

bool isSupportedTilingDecomposedGather(VPU::GatherOp op, const OutputTiling& tiles, int64_t slices, Logger log) {
    const auto origOp = op.getOperation();

    const auto operands = origOp->getOperands();
    const auto results = origOp->getResults();

    auto tilingOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(origOp);
    VPUX_THROW_UNLESS(tilingOp != nullptr, "Not a tileable operation {0}", origOp->getName());
    const auto cmxAvailableBytes = vpux::VPU::getTotalCMXSize(origOp).to<Byte>().count();

    auto outputType = results[0].getType().cast<vpux::NDTypeInterface>();
    auto outputByteSize = outputType.getElemTypeSize().to<Byte>().count();

    return llvm::all_of(tiles, [&](const TileInfo& outputTile) {
        auto inputTiles = backInferTileInfoDecomposedGather(op, outputTile, log);
        if (inputTiles.tiles.size() < 1) {
            log.trace("No input tiles for {0}", origOp->getLoc());
            return false;
        }
        const auto& inTiles = inputTiles.tiles;
        // Use bool type as flag element type.
        const auto flagByteSize = 1;
        // Calculate required CMX of GatherSlice Op.
        const auto outputTileSizeBytesGatherSlice =
                outputTile.shape.totalSize() * outputByteSize + inTiles[1].shape.totalSize() * flagByteSize;
        log.trace("outputTileSizeBytesGatherSlice: {0}", outputTileSizeBytesGatherSlice);
        const auto inputByteSize =
                operands[0].getType().cast<vpux::NDTypeInterface>().getElemTypeSize().to<Byte>().count();
        const auto indicesByteSize =
                operands[1].getType().cast<vpux::NDTypeInterface>().getElemTypeSize().to<Byte>().count();
        auto requiredCMX = outputTileSizeBytesGatherSlice;
        requiredCMX = requiredCMX + inTiles[0].shape.totalSize() * inputByteSize / slices +
                      inTiles[1].shape.totalSize() * indicesByteSize;

        if (requiredCMX > cmxAvailableBytes) {
            log.trace("Tile does not fit into CMX. required CMX "
                      "size {0}, "
                      "max available CMX: {1}",
                      requiredCMX, cmxAvailableBytes);
            return false;
        }
        // Calculate required CMX of ExtractValue Op.
        const auto outputTileSizeBytesExtractValue = outputTile.shape.totalSize() * outputByteSize;
        log.trace("outputTileSizeBytesExtractValue: {0}", outputTileSizeBytesExtractValue);
        requiredCMX = outputTileSizeBytesExtractValue;
        requiredCMX = requiredCMX + slices * outputTile.shape.totalSize() * inputByteSize +
                      slices * outputTile.shape.totalSize() * flagByteSize;
        if (requiredCMX > cmxAvailableBytes) {
            log.trace("Tile does not fit into CMX. required CMX "
                      "size {0}, "
                      "max available CMX: {1}",
                      requiredCMX, cmxAvailableBytes);
            return false;
        }
        log.trace("Op {0} out tiling probe valid", origOp->getLoc());
        return true;
    });
}

//
// getTilingStrategyDecomposedGather
//

mlir::FailureOr<OutputTiling> getTilingStrategyDecomposedGather(TilingMode tilingMode, VPU::GatherOp gatherOp,
                                                                int64_t& slices, Logger log) {
    auto baseOp = gatherOp.getOperation();
    VPUX_THROW_WHEN(tilingMode != TilingMode::ISOLATED,
                    "Only supporting isolated tiling currently, for op {0} at '{1}'", baseOp->getName(),
                    baseOp->getLoc());

    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(baseOp);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface",
                    baseOp->getName());

    const auto outputType = baseOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputShape = outputType.getShape();
    Shape nTilesOnDimforGather(outputShape.size(), 1);
    const auto isSupportedTileSize = [gatherOp, baseOp, outputShape, log](ShapeRef nTilesOnDim,
                                                                          int64_t slices) -> bool {
        const auto tiles = fillDividedTiles(baseOp, nTilesOnDim, outputShape);
        if (mlir::failed(tiles)) {
            return false;
        }
        return isSupportedTilingDecomposedGather(gatherOp, tiles.value(), slices, log);
    };

    // Slice means the number of tiles at axis dimension of input tensor.
    int64_t tiles = 1;
    int64_t tileDivSlice = 1;
    const auto outputRank = static_cast<int64_t>(outputShape.size());
    int64_t dimToTile = 0;
    while (dimToTile < outputRank && !isSupportedTileSize(nTilesOnDimforGather, slices)) {
        if (slices == tileDivSlice) {
            tileDivSlice++;
        } else if (slices < tileDivSlice) {
            slices++;
            tileDivSlice--;
        } else {
            tileDivSlice++;
        }
        tiles = slices * tileDivSlice;
        while (nTilesOnDimforGather.totalSize() < tiles) {
            if (nTilesOnDimforGather[Dim(dimToTile)] >= outputShape[Dim(dimToTile)]) {
                dimToTile++;
            } else {
                ++nTilesOnDimforGather[Dim(dimToTile)];
            }
        }
    }

    log.trace("Isolated tiling strategy: {0}", nTilesOnDimforGather);
    return fillDividedTiles(baseOp, nTilesOnDimforGather, outputShape);
}

mlir::Value reifyTileDecomposedGather(VPU::GatherOp gatherOp, const TileInfo& outputTile, int64_t slices,
                                      mlir::OpBuilder& builder, Logger log) {
    log = log.nest(2);
    log.trace("{0}", outputTile);
    auto inputTiling = backInferTileInfoDecomposedGather(gatherOp, outputTile, log);
    auto& inTiles = inputTiling.tiles;
    VPUX_THROW_UNLESS(!inTiles.empty(), "Got empty tile information");

    int64_t axisValue = getAxisValue(gatherOp);
    const auto loc = gatherOp->getLoc();
    SmallVector<mlir::Value> outputSliceVals, flagSliceVals;
    const auto valIndicesName = printToString("indices");
    const auto tiledIndices = vpux::VPU::makeTile(builder, loc, gatherOp.indices(), inTiles[1], valIndicesName);
    auto sliceDimLen = inTiles[0].shape[Dim(axisValue)] / slices;
    auto sliceRemainder = inTiles[0].shape[Dim(axisValue)] - inTiles[0].shape[Dim(axisValue)] / slices * slices;

    const auto ctx = gatherOp.getContext();
    const auto axisAttr = getIntAttr(ctx, axisValue);
    const auto maxAxisDimensionAttr = getIntAttr(ctx, inTiles[0].shape[Dim(axisValue)]);

    for (int64_t slice = 0; slice < slices; ++slice) {
        const auto valInputName = printToString("input");
        auto sliceInputTiles = inTiles[0];
        if (slice < sliceRemainder) {
            sliceInputTiles.shape[Dim(axisValue)] = sliceDimLen + 1;
            sliceInputTiles.offsets[Dim(axisValue)] = (sliceDimLen + 1) * slice;
        } else {
            sliceInputTiles.shape[Dim(axisValue)] = sliceDimLen;
            sliceInputTiles.offsets[Dim(axisValue)] =
                    (sliceDimLen + 1) * sliceRemainder + sliceDimLen * (slice - sliceRemainder);
        }
        const auto tiledSliceInput = vpux::VPU::makeTile(builder, loc, gatherOp.input(), sliceInputTiles, valInputName);

        // Create GatherSliceOp.
        const auto sliceHeadAttr = getIntAttr(ctx, sliceInputTiles.offsets[Dim(axisValue)]);
        const auto sliceTailAttr =
                getIntAttr(ctx, sliceInputTiles.offsets[Dim(axisValue)] + sliceInputTiles.shape[Dim(axisValue)]);
        auto gatherSliceOp = builder.create<VPU::GatherSliceOp>(loc, tiledSliceInput, tiledIndices, axisAttr,
                                                                gatherOp.batch_dimsAttr(), maxAxisDimensionAttr,
                                                                sliceHeadAttr, sliceTailAttr);

        const auto unsqueezeParamsAttr = getIntArrayAttr(ctx, SmallVector<int64_t>({0}));
        auto unsqueezedOutput =
                builder.create<VPU::UnsqueezeOp>(loc, gatherSliceOp.output(), nullptr, unsqueezeParamsAttr);
        auto unsqueezedFlag = builder.create<VPU::UnsqueezeOp>(loc, gatherSliceOp.flag(), nullptr, unsqueezeParamsAttr);
        outputSliceVals.push_back(unsqueezedOutput.output());
        flagSliceVals.push_back(unsqueezedFlag.output());
    }
    auto ExtractValueInput = builder.create<VPU::ConcatOp>(loc, outputSliceVals, 0);
    auto ExtractValueFlag = builder.create<VPU::ConcatOp>(loc, flagSliceVals, 0);

    auto ExtractValueOutput = builder.create<VPU::ExtractValueOp>(
            loc, ExtractValueInput.output(), ExtractValueFlag.output(), axisAttr, gatherOp.batch_dimsAttr());

    return ExtractValueOutput.output();
}

//
// applyTileStrategyDecomposedGather
//

mlir::LogicalResult applyTileStrategyDecomposedGather(VPU::GatherOp gatherOp, const OutputTiling& tiles,
                                                      mlir::PatternRewriter& rewriter, int64_t slices, Logger log) {
    // apply the generated fake tiling strategy and convert gather to gatherSlice and extractValue.
    SmallVector<mlir::Value> resultTileVals;
    SmallVector<ShapeRef> resultTileOffsets;

    for (const auto& outputTile : tiles) {
        const auto tiledRes = reifyTileDecomposedGather(gatherOp, outputTile, slices, rewriter, log);
        const auto tiledShape = getShape(tiledRes);
        VPUX_THROW_UNLESS(tiledShape == outputTile.shape,
                          "Inferred tiled output shape '{0}' doesn't match with generated '{1}'", tiledShape,
                          outputTile.shape);
        resultTileVals.push_back(tiledRes);
        resultTileOffsets.push_back(outputTile.offsets);
    }

    auto opConcat = rewriter.create<VPU::ConcatOp>(gatherOp->getLoc(), gatherOp.output().getType(),
                                                   mlir::ValueRange(resultTileVals), makeArrayRef(resultTileOffsets));
    rewriter.replaceOp(gatherOp, opConcat.output());

    return mlir::success();
}

//
// DecomposeGatherPass
//

class DecomposeGatherPass final : public VPU::arch37xx::DecomposeGatherBase<DecomposeGatherPass> {
public:
    explicit DecomposeGatherPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class GatherConverter;

private:
    void safeRunOnFunc() final;
};

//
// GatherConverter
//

class DecomposeGatherPass::GatherConverter final : public mlir::OpRewritePattern<VPU::GatherOp> {
public:
    GatherConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::GatherOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::GatherOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Decompose GatherOp into GatherSliceOp and ExtractValueOp, GatherSliceOp can tile axis dimension to split GatherOp
// into smaller size. ExractValueOp can extract actual value from multi-outputs of GatherSliceOp. And slice is used
// to determine how many GatherSliceOps and ExtractValueOps a tiled GatherOP will be decomposed into.
//
//               GatherSlice0 GatherSlice1 GatherSlice2 GatherSlice3
//                        \      /               \     /
//                         \    /                 \   /
//    Gather -->          ExtractValue0    ExtractValue1
//                                   \       /
//                                    \     /
//                                   ConcatOp
mlir::LogicalResult DecomposeGatherPass::GatherConverter::matchAndRewrite(VPU::GatherOp origOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    int64_t slices = 1;
    const auto tilingStrategy = getTilingStrategyDecomposedGather(TilingMode::ISOLATED, origOp, slices, _log);
    if (mlir::failed(tilingStrategy)) {
        VPUX_THROW("Can't get feasible tiling strategy for decomposed gather.");
    }
    return applyTileStrategyDecomposedGather(origOp, tilingStrategy.value(), rewriter, slices, _log);
}

//
// safeRunOnFunc
//

void DecomposeGatherPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);

    target.addDynamicallyLegalOp<VPU::GatherOp>([&](VPU::GatherOp op) {
        // TODO Refactor when E#79282 is closed.
        const auto origOp = op.getOperation();
        const auto cmxAvailableBytes = vpux::VPU::getTotalCMXSize(origOp).to<Byte>().count();
        const auto inputType = op.input().getType().cast<vpux::NDTypeInterface>();
        const auto inputShape = inputType.getShape().raw();
        const auto inputByteSize = inputType.getElemTypeSize().to<Byte>().count();
        int64_t axisValue = getAxisValue(op);
        const auto axisDimSizeBytes = inputShape[axisValue] * inputByteSize;

        // Can't get feasible tiling strategy because axis dimension of gatherOp can't be tiled.
        if (axisDimSizeBytes > cmxAvailableBytes) {
            _log.nest(1).trace("Can't still fit into CMX after tiling. The pass is used to decompose gatherOp to "
                               "gatheSliceOp and ExtractValueOp to meet the requirement of CMX.");
            return false;
        }
        return true;
    });
    target.addLegalOp<VPU::GatherSliceOp>();
    target.addLegalOp<VPU::ExtractValueOp>();
    target.addLegalOp<VPU::SliceOp>();
    target.addLegalOp<VPU::UnsqueezeOp>();
    target.addLegalOp<VPU::ConcatOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<GatherConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        _log.debug("Failed to decompose GatherOp into GatherSliceOp and ExtractValueOp.");
        signalPassFailure();
    }
}

}  // namespace

//
// createDecomposeGatherPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::arch37xx::createDecomposeGatherPass(Logger log) {
    return std::make_unique<DecomposeGatherPass>(log);
}
