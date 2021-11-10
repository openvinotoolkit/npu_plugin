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
#include "vpux/compiler/core/layers.hpp"

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/logger.hpp"

#include <mlir/IR/BuiltinAttributes.h>

#include <llvm/Support/raw_ostream.h>

#include <functional>

#pragma once

namespace vpux {

//
// TileInfo
//

struct TileInfo final {
    Shape shape;
    Shape offsets;

    TileInfo() = delete;

    explicit TileInfo(size_t rank): shape(rank), offsets(rank) {
    }

    explicit TileInfo(ShapeRef shape): shape(shape.raw()), offsets(shape.size(), 0) {
    }

    void printFormat(llvm::raw_ostream& stream) const {
        printTo(stream, "Tile [shape = {0}, offsets = {1}]", shape, offsets);
    }
};

using OutputTiling = SmallVector<TileInfo>;

// helper function to generate a set of tiles from dividing a shape. A shape divided across multiple dimensions will
// generate a set of tiles, each having its own size and offsets
OutputTiling fillDividedTiles(ShapeRef divisors, ShapeRef orig);

//
// PadInfo
//

struct PadInfo final {
    int64_t left = 0;
    int64_t right = 0;
    int64_t top = 0;
    int64_t bottom = 0;

    PadInfo() = default;

    PadInfo(int64_t left, int64_t right, int64_t top, int64_t bottom)
            : left(left), right(right), top(top), bottom(bottom) {
    }

    PadInfo(mlir::ArrayAttr pads_begin, mlir::ArrayAttr pads_end) {
        top = pads_begin[Dims4D::PadsBegin::Top.ind()].cast<mlir::IntegerAttr>().getValue().getSExtValue();
        bottom = pads_end[Dims4D::PadsEnd::Bottom.ind()].cast<mlir::IntegerAttr>().getValue().getSExtValue();
        left = pads_begin[Dims4D::PadsBegin::Left.ind()].cast<mlir::IntegerAttr>().getValue().getSExtValue();
        right = pads_end[Dims4D::PadsEnd::Right.ind()].cast<mlir::IntegerAttr>().getValue().getSExtValue();
    }

    bool enabled() const {
        return left != 0 || right != 0 || top != 0 || bottom != 0;
    }

    bool operator==(const PadInfo& other) const {
        return left == other.left && right == other.right && top == other.top && bottom == other.bottom;
    }
    bool operator!=(const PadInfo& other) const {
        return !(*this == other);
    }

    void printFormat(llvm::raw_ostream& stream) const {
        printTo(stream, "PadInfo [left = {0}, right = {1}, top = {2}, bottom = {3}]", left, right, top, bottom);
    }
};

PadInfo backInferPadsTile(const TileInfo& outputTile, ShapeRef outShape, const PadInfo& origPads);

//
// Convolution tiling
//

struct ConvTileConfig final {
    TileInfo inputTile;
    TileInfo filterTile;
    TileInfo biasTile;
    PadInfo pads;
};

ConvTileConfig backInferConvTile(const TileInfo& outputTile, ShapeRef origInputShape, ShapeRef origFilterShape,
                                 ShapeRef origBiasShape, mlir::ArrayAttr strides, mlir::ArrayAttr pads_begin,
                                 mlir::ArrayAttr pads_end);

ConvTileConfig backInferGroupConvTile(const TileInfo& outputTile, ShapeRef origInputShape, ShapeRef origFilterShape,
                                      ShapeRef origBiasShape, mlir::ArrayAttr strides, mlir::ArrayAttr pads_begin,
                                      mlir::ArrayAttr pads_end);

//
// Pooling tiling
//

struct PoolTileConfig final {
    TileInfo inputTile;
    PadInfo pads;
};

PoolTileConfig backInferPoolTile(const TileInfo& outputTile, ShapeRef origInputShape, mlir::ArrayAttr kernel_size,
                                 mlir::ArrayAttr strides, mlir::ArrayAttr pads_begin, mlir::ArrayAttr pads_end);

}  // namespace vpux
