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

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"

#pragma once

namespace vpux {

struct Tile final {
    Tile() = delete;

    explicit Tile(size_t rank): shape(rank), offsets(rank) {
    }

    explicit Tile(ShapeRef shape): shape(shape.raw()), offsets(shape.size(), 0) {
    }

    Shape shape;
    Shape offsets;
};

struct PadsTileConfig final {
    SmallVector<int64_t> begin;
    SmallVector<int64_t> end;
};

struct ConvTileConfig final {
    Tile inputTile;
    Tile filterTile;
    Tile biasTile;
    PadsTileConfig pads;
};

struct PoolTileConfig final {
    Tile inputTile;
    PadsTileConfig pads;
};

// helper function to generate a set of tiles from dividing a shape. A shape divided across multiple dimensions will
// generate a set of tiles, each having its own size and offsets
SmallVector<Tile> fillDividedTiles(ShapeRef divisors, ShapeRef orig);

PadsTileConfig backInferPadsTile(const Tile& outputTile, ShapeRef outShape, ArrayRef<int64_t> opPadsBegin,
                                 ArrayRef<int64_t> opPadsEnd);

// TODO: Replace IERT::ConvolutionOp with Operation Interface
ConvTileConfig backInferConvTile(IERT::ConvolutionOp origOp, const Tile& outputTile);

// TODO: Replace IERT::MaxPoolOp with Operation Interface
PoolTileConfig backInferPoolTile(IERT::MaxPoolOp origOp, const Tile& outputTile);

// TODO: Replace IERT::GroupConvolutionOp with Operation Interface
ConvTileConfig backInferGroupConvTile(IERT::GroupConvolutionOp origOp, const Tile& outputTile);

}  // namespace vpux
