//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/IRMapping.h>

using namespace vpux;

mlir::LogicalResult vpux::VPU::GRUSequenceOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::GRUSequenceOpAdaptor gru(operands, attrs);
    if (mlir::failed(gru.verify(loc))) {
        return mlir::failure();
    }

    const auto initialStateType = gru.getInitialHiddenState().getType().cast<vpux::NDTypeInterface>();
    const auto outputStateType = initialStateType;
    const auto outputStateShape = outputStateType.getShape().raw();
    const auto seqLength = gru.getSeqLength();
    SmallVector<int64_t> middleStateShape = {outputStateShape[0], outputStateShape[1], seqLength, outputStateShape[2]};
    const auto middleStateType = initialStateType.changeShape(Shape(middleStateShape));

    inferredReturnShapes.push_back(middleStateType);
    inferredReturnShapes.push_back(outputStateType);

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::GRUSequenceOp::backInferTileInfo(const vpux::TileInfo& outputTileY, vpux::Logger) {
    const auto origInputShape = getShape(getInputData());
    const auto origInitialHiddenStateShape = getShape(getInitialHiddenState());
    const auto origWShape = getShape(getWeights());
    const auto origRShape = getShape(getRecurrenceWeights());
    const auto origBShape = getShape(getBiases());

    TileInfo inputTile(origInputShape);
    TileInfo initialHiddenStateTile(origInitialHiddenStateShape);
    TileInfo wTile(origWShape);
    TileInfo rTile(origRShape);
    TileInfo bTile(origBShape);

    inputTile.shape[Dim(0)] = outputTileY.shape[Dim(0)];
    inputTile.offsets[Dim(0)] = outputTileY.offsets[Dim(0)];
    inputTile.shape[Dim(1)] = outputTileY.shape[Dim(2)];
    inputTile.offsets[Dim(1)] = outputTileY.offsets[Dim(2)];

    initialHiddenStateTile.shape[Dim(0)] = outputTileY.shape[Dim(0)];
    initialHiddenStateTile.offsets[Dim(0)] = outputTileY.offsets[Dim(0)];
    initialHiddenStateTile.shape[Dim(1)] = outputTileY.shape[Dim(1)];
    initialHiddenStateTile.offsets[Dim(1)] = outputTileY.offsets[Dim(1)];

    wTile.shape[Dim(0)] = outputTileY.shape[Dim(1)];
    wTile.offsets[Dim(0)] = outputTileY.offsets[Dim(1)];

    rTile.shape[Dim(0)] = outputTileY.shape[Dim(1)];
    rTile.offsets[Dim(0)] = outputTileY.offsets[Dim(1)];

    bTile.shape[Dim(0)] = outputTileY.shape[Dim(1)];
    bTile.offsets[Dim(0)] = outputTileY.offsets[Dim(1)];

    return InputTiling{{inputTile, initialHiddenStateTile, wTile, std::move(rTile), bTile}};
}

void vpux::VPU::GRUSequenceOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& outputYTile) {
    auto* ctx = this->getContext();
    auto inputTileInfo = inputTiling.tiles[0];
    VPUX_THROW_UNLESS(inputTileInfo.shape[Dim(1)] == outputYTile.shape[Dim(2)],
                      "seq_length dimension in input tile is incompatible with output tile, seq_length dimension of "
                      "input tile is {0}, but it's {1} in output tile",
                      inputTileInfo.shape[Dim(1)], outputYTile.shape[Dim(2)]);
    auto origSeqLength = getSeqLengthAttr().dyn_cast_or_null<mlir::IntegerAttr>().getValue().getSExtValue();
    auto tiledSeqLength = inputTileInfo.shape[Dim(1)];
    if (origSeqLength != tiledSeqLength) {
        const auto newSeqLength = getIntAttr(ctx, tiledSeqLength);
        this->setSeqLengthAttr(newSeqLength);
    }
    // The num_direction dimension of output can be tiled when direction is BIDIRECTONAL.
    // GRUSequence with BIDIRECTIONAL attribute will be split into two GRUSequence, GRUSequence with FORWARD attribute
    // and GRUSequence with REVERSE attribute when num_direction dimension was tiled.
    auto origDirection = getDirectionAttr().getValue();
    if (origDirection == IE::RNNSequenceDirection::BIDIRECTIONAL && outputYTile.shape[Dim(1)] == 1 &&
        outputYTile.offsets[Dim(1)] == 0) {
        const auto newDirectionAttr = IE::RNNSequenceDirectionAttr::get(ctx, IE::RNNSequenceDirection::FORWARD);
        this->setDirectionAttr(newDirectionAttr);
    }
    if (origDirection == IE::RNNSequenceDirection::BIDIRECTIONAL && outputYTile.shape[Dim(1)] == 1 &&
        outputYTile.offsets[Dim(1)] == 1) {
        const auto newDirectionAttr = IE::RNNSequenceDirectionAttr::get(ctx, IE::RNNSequenceDirection::REVERSE);
        this->setDirectionAttr(newDirectionAttr);
    }
}

mlir::FailureOr<OutputTiling> vpux::VPU::GRUSequenceOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto baseOp = this->getOperation();
    VPUX_THROW_WHEN(tilingMode != TilingMode::ISOLATED,
                    "Only supporting isolated tiling for GRUSequence currently, for op {0} at '{1}'", baseOp->getName(),
                    getLoc());

    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(baseOp);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface",
                    baseOp->getName());

    // There are two outputs of GRUSequence, the namse Y and Ho are from OpenVINO doc.
    // The shape of Y is [batch_size, num_directions, seq_len, hidden_size],
    // and the shape of Ho is [batch_size, num_directions, hidden_size].
    // Ho-tiles can be inferred by Y-tiles.
    const auto outputYType = baseOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputYShape = outputYType.getShape();
    Shape nTilesOnDimForOutputY(outputYShape.size(), 1);

    const auto isSupportedTileSize = [baseOp, &tilingInfo, outputYShape, log](ShapeRef nTilesOnDim,
                                                                              TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(baseOp, nTilesOnDim, outputYShape);
        if (mlir::failed(tiles)) {
            return false;
        }
        return tilingInfo.isSupportedTiling(tiles.value(), tilingMode, log);
    };

    int64_t tileDim = 0;
    while (!isSupportedTileSize(nTilesOnDimForOutputY, tilingMode)) {
        // Dimension of hidden_size can't be tiled.
        VPUX_THROW_UNLESS(tileDim < 3, "can't get feasible tiling strategy for GRUSequence.");
        if (nTilesOnDimForOutputY[Dim(tileDim)] >= outputYShape[Dim(tileDim)]) {
            ++tileDim;
        } else {
            ++nTilesOnDimForOutputY[Dim(tileDim)];
        }
    }
    log.trace("Isolated tiling strategy: {0}", nTilesOnDimForOutputY);

    auto origTilesY = fillDividedTiles(baseOp, nTilesOnDimForOutputY, outputYShape);
    return origTilesY;
}

//
// reifyTileGRUSequence
//

// There are some reasons for custom applyTileStrategy and reifyTile :
// 1, There are two outputs of GRUSequence, names Y and Ho from OpenVINO doc, it's a little
// different from TopK because the shapes of two TopK outputs are same. The shape of Y is
// [batch_size, num_directions, seq_len, hidden_size], and the shape of Ho is [batch_size,
// num_directions, hidden_size]. And Ho-tiles can be inferred by Y-tiles. Besides, a
// inferoutputHoTile logic is needed.
// 2, These tiles GRUSequence aren't independent of each other when seq_length dimension is tiled.
// So, output of previous tile GRUSequence is needed to create current tile GRUSequence.
// And the logic is different according to direction attribute.
// 3, The function to reverse tiles order for REVERSE mode is also necessary.
OutputTiling inferOutputHoTile(const OutputTiling& tilesY) {
    // The rank of outputHo equals 3.
    OutputTiling tilesHo;
    for (const auto& outputYTile : tilesY) {
        TileInfo outputHoTile(3);
        outputHoTile.shape[Dim(0)] = outputYTile.shape[Dim(0)];
        outputHoTile.shape[Dim(1)] = outputYTile.shape[Dim(1)];
        outputHoTile.shape[Dim(2)] = outputYTile.shape[Dim(3)];
        outputHoTile.offsets[Dim(0)] = outputYTile.offsets[Dim(0)];
        outputHoTile.offsets[Dim(1)] = outputYTile.offsets[Dim(1)];
        outputHoTile.offsets[Dim(2)] = outputYTile.offsets[Dim(3)];
        tilesHo.push_back(outputHoTile);
    }
    return tilesHo;
}

void reverseTilesOrderForReverseMode(OutputTiling& tilesY, int64_t seqLengthTile,
                                     IE::RNNSequenceDirection origDirection) {
    const auto reverse = [seqLengthTile, &tilesY](size_t i) {
        for (size_t j = 0; j < size_t(seqLengthTile / 2); ++j) {
            std::swap(tilesY[i + j], tilesY[i + seqLengthTile - 1 - j]);
        }
    };
    if (origDirection == IE::RNNSequenceDirection::BIDIRECTIONAL) {
        for (size_t i = 0; i < tilesY.size(); i += seqLengthTile) {
            if (tilesY[i].offsets[Dim(1)] == 1) {
                reverse(i);
            }
        }
    }
    if (origDirection == IE::RNNSequenceDirection::REVERSE) {
        for (size_t i = 0; i < tilesY.size(); i += seqLengthTile) {
            reverse(i);
        }
    }
}

SmallVector<mlir::Value> vpux::VPU::GRUSequenceOp::reifyTileGRUSequence(VPU::TilingBuilderOpInterface origOp,
                                                                        const TileInfo& outputYTile,
                                                                        const TileInfo& outputHoTile,
                                                                        mlir::OpBuilder& builder, Logger log,
                                                                        bool isInitialState, mlir::Value prevHo) {
    log = log.nest(2);
    log.trace("{0}", outputYTile);

    auto inputTiling = origOp.backInferTileInfo(outputYTile, log);
    auto& inTiles = inputTiling.tiles;
    VPUX_THROW_UNLESS(!inTiles.empty(), "Got empty tile information");

    mlir::IRMapping mapper;
    for (auto p : origOp->getOperands() | indexed) {
        auto origInput = p.value();
        auto inputIdx = p.index();
        const auto valName = printToString("input {0}", inputIdx);
        if (inputIdx == 1 && !isInitialState) {
            mapper.map(origInput, prevHo);
        } else {
            const auto tiledInput =
                    vpux::VPU::makeTile(builder, origOp->getLoc(), origInput, inTiles[inputIdx], valName);
            mapper.map(origInput, tiledInput);
        }
    }

    const auto tileLoc = appendLoc(origOp->getLoc(), "output tile {0}", outputYTile.offsets, outputHoTile.offsets);
    auto* tiledOp = builder.clone(*origOp, mapper);
    tiledOp->setLoc(tileLoc);

    auto tiledBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(tiledOp);
    VPUX_THROW_WHEN(tiledBuilderOp == nullptr, "Operation '{0}' doesn't implement TilingBuilderOpInterface",
                    tiledBuilderOp->getName());
    tiledBuilderOp.adjustAttrs(inputTiling, outputYTile);

    SmallVector<mlir::Value> ret;
    for (auto p : origOp->getResults() | indexed) {
        auto idx = static_cast<uint32_t>(p.index());
        const auto baseResType = origOp->getResult(idx).getType().cast<vpux::NDTypeInterface>();
        const TileInfo& tileInfo = idx == 1 ? outputHoTile : outputYTile;
        auto tiledResType = baseResType.extractDenseTile(tileInfo.offsets, tileInfo.shape);
        auto tiledRes = tiledOp->getResult(idx);
        tiledRes.setType(tiledResType);
        ret.push_back(tiledRes);
    }
    return ret;
}

//
// applyTileStrategyGRUSequence
//

mlir::LogicalResult vpux::VPU::GRUSequenceOp::applyTileStrategyGRUSequence(VPU::TilingBuilderOpInterface origOp,
                                                                           OutputTiling tilesY,
                                                                           mlir::PatternRewriter& rewriter,
                                                                           Logger log) {
    // apply the generated tiling strategy and create tiled operations
    // insert the tiled pattern with a concat to the IR
    auto gruSequenceOp = mlir::dyn_cast<VPU::GRUSequenceOp>(origOp.getOperation());
    VPUX_THROW_UNLESS(gruSequenceOp != nullptr, "The function just for GRUSequence");
    SmallVector<mlir::Value> resultTileYVals, resultTileHoVals;
    SmallVector<ShapeRef> resultTileYOffsets, resultTileHoOffsets;
    auto origOutputYShape = getShape(origOp->getResult(0));
    auto origSeqLength = origOutputYShape[Dim(2)];
    auto origDirection = gruSequenceOp.getDirectionAttr().getValue();

    mlir::Value prevHo;
    bool isInitialState = false;
    const auto& seqLengthTile = (origSeqLength + tilesY[0].shape[Dim(2)] - 1) / tilesY[0].shape[Dim(2)];
    // The order of tilesY for GRUSequence with FORWARD is same as the order of tilesY for GRUSequence with REVERSE.
    // For example, if the shape of output equals [1, 1, 100000, 1], the tiling strategy equals [1, 1, 2, 1].
    // So, the tilesY will be
    // Tile 0: shape: [1, 1, 50000, 1],  offsets: [0, 0, 0, 0]
    // Tile 1: shape: [1, 1, 50000, 1],  offsets: [0, 0, 50000, 0]
    // The order of tilesY is correct order for FORWARD mode to generate two GRUSequence. But fail for REVERSE mode,
    // because the initial_hidden_state of first tile GRUSequence is the outputHo of second tile GRUSequence
    // when direction is REVERSE mode. The function can reverse order of tilesY for REVERSE mode to correct order:
    // Tile 0: shape: [1, 1, 50000, 1],  offsets: [0, 0, 50000, 0]
    // Tile 1: shape: [1, 1, 50000, 1],  offsets: [0, 0, 0, 0]
    reverseTilesOrderForReverseMode(tilesY, seqLengthTile, origDirection);
    const OutputTiling& tilesHo = inferOutputHoTile(tilesY);
    for (auto p : tilesY | indexed) {
        auto idx = p.index();
        const auto& outputYTile = p.value();
        const auto& outputHoTile = tilesHo[idx];
        if (origDirection == IE::RNNSequenceDirection::BIDIRECTIONAL) {
            isInitialState = (outputYTile.offsets[Dim(1)] == 0 && outputYTile.offsets[Dim(2)] == 0) ||
                             (outputYTile.offsets[Dim(1)] == 1 &&
                              origSeqLength - outputYTile.shape[Dim(2)] == outputYTile.offsets[Dim(2)]);
        } else if (origDirection == IE::RNNSequenceDirection::FORWARD) {
            isInitialState = outputYTile.offsets[Dim(2)] == 0;
        } else if (origDirection == IE::RNNSequenceDirection::REVERSE) {
            isInitialState = origSeqLength - outputYTile.shape[Dim(2)] == outputYTile.offsets[Dim(2)];
        } else {
            VPUX_THROW("Unsupported direction mode {0}.", origDirection);
        }
        const auto tiledRes = gruSequenceOp.reifyTileGRUSequence(origOp, outputYTile, outputHoTile, rewriter, log,
                                                                 isInitialState, prevHo);
        prevHo = tiledRes[1];

        const auto tiledYShape = getShape(tiledRes[0]);
        const auto tiledHoShape = getShape(tiledRes[1]);
        VPUX_THROW_UNLESS(tiledYShape == outputYTile.shape,
                          "Inferred tiled Y output shape '{0}' doesn't match with generated '{1}'", tiledYShape,
                          outputYTile.shape);
        VPUX_THROW_UNLESS(tiledHoShape == outputHoTile.shape,
                          "Inferred tiled Ho output shape '{0}' doesn't match with generated '{1}'", tiledHoShape,
                          outputHoTile.shape);
        resultTileYVals.push_back(tiledRes[0]);
        resultTileYOffsets.push_back(outputYTile.offsets);
        if ((idx + 1) % seqLengthTile == 0) {
            resultTileHoVals.push_back(tiledRes[1]);
            resultTileHoOffsets.push_back(outputHoTile.offsets);
        }
    }

    SmallVector<mlir::Value> opsConcat;
    auto opConcatY = rewriter.create<VPU::ConcatOp>(origOp->getLoc(), origOp->getResult(0).getType(),
                                                    mlir::ValueRange(resultTileYVals), ArrayRef(resultTileYOffsets));
    opsConcat.push_back(opConcatY.getOutput());
    auto opConcatHo = rewriter.create<VPU::ConcatOp>(origOp->getLoc(), origOp->getResult(1).getType(),
                                                     mlir::ValueRange(resultTileHoVals), ArrayRef(resultTileHoOffsets));
    opsConcat.push_back(opConcatHo.getOutput());

    rewriter.replaceOp(origOp, opsConcat);

    return mlir::success();
}
