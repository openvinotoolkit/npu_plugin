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
#include "vpux/compiler/utils/attributes.hpp"

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
    Shape axis;

    TileInfo() = delete;

    explicit TileInfo(size_t rank): shape(rank), offsets(rank), axis(rank) {
    }

    explicit TileInfo(ShapeRef shape): shape(shape.raw()), offsets(shape.size(), 0), axis(shape.size(), 1) {
    }

    void printFormat(llvm::raw_ostream& stream) const {
        printTo(stream, "Tile [shape = {0}, offsets = {1}, axis = {2}]", shape, offsets, axis);
    }

    bool operator==(const TileInfo& other) const {
        return shape == other.shape && offsets == other.offsets && axis == other.axis;
    }

    bool operator!=(const TileInfo& other) const {
        return !(*this == other);
    }

    bool operator<(const TileInfo& other) const {
        if (shape != other.shape) {
            return shape < other.shape;
        } else if (offsets != other.offsets) {
            return offsets < other.offsets;
        }
        return axis < other.axis;
    }
};

// Operation output tiles information
using OutputTiling = SmallVector<TileInfo>;

// helper function to generate a set of tiles from dividing a shape. A shape divided across multiple dimensions will
// generate a set of tiles, each having its own size and offsets
OutputTiling fillDividedTiles(ShapeRef divisors, ShapeRef orig, mlir::IntegerAttr clusterId = nullptr);

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
// TilingInfo
//

struct TilingInfo final {
    SmallVector<TileInfo> tiles;
    Optional<PadInfo> pads;

    explicit TilingInfo(ArrayRef<TileInfo> tiles): tiles(tiles.begin(), tiles.end()) {
    }

    explicit TilingInfo(ArrayRef<TileInfo> tiles, PadInfo pads): tiles(tiles.begin(), tiles.end()), pads(pads) {
    }
};

// Operation inputs tiling information
using InputTiling = TilingInfo;

//
// Convolution tiling
//

InputTiling backInferConvTile(const TileInfo& outputTile, ShapeRef origInputShape, ShapeRef origFilterShape,
                              ShapeRef origBiasShape, mlir::ArrayAttr strides, const PadInfo& origPadding);

InputTiling backInferGroupConvTile(const TileInfo& outputTile, ShapeRef origInputShape, ShapeRef origFilterShape,
                                   ShapeRef origBiasShape, mlir::ArrayAttr strides, const PadInfo& origPadding);

//
// Pooling tiling
//

InputTiling backInferPoolTile(const TileInfo& outputTile, ShapeRef origInputShape, mlir::ArrayAttr kernel_size,
                              mlir::ArrayAttr strides, const PadInfo& origPadding);

//
// DimRange
//

struct DimRange final {
    int64_t begin = 0;
    int64_t end = 0;

    DimRange() = default;
    DimRange(int64_t begin, int64_t end): begin(begin), end(end) {
        VPUX_THROW_UNLESS(end >= begin, "Got wrong dimension range [{0}, {1})", begin, end);
    }

    int64_t length() const {
        return end - begin;
    }

    bool intersects(const DimRange& other) const {
        return (begin < other.end) && (other.begin < end);
    }

    bool contains(const DimRange& other) const {
        return (begin <= other.begin) && (end >= other.end);
    }

    // Represents `other` range to ROI of the current one.
    DimRange asROI(const DimRange& other) const {
        VPUX_THROW_UNLESS(contains(other), "DimRange '{0}' is not contained in '{1}'", other, *this);
        return {other.begin - begin, other.end - begin};
    }

    bool operator==(const DimRange& other) const {
        return begin == other.begin && end == other.end;
    }
    bool operator!=(const DimRange& other) const {
        return !(*this == other);
    }

    void printFormat(llvm::raw_ostream& stream) const {
        printTo(stream, "DimRange [{0}, {1})", begin, end);
    }
};

//
// Tiling utils
//

std::tuple<DimRange, int64_t, int64_t> inputForOutputDim(const DimRange& output, int64_t kernel, int64_t stride,
                                                         const DimRange& initialInputRange, int64_t padBefore,
                                                         int64_t padAfter);

SmallVector<int64_t> alignShape(ArrayRef<int64_t> shape, Optional<ArrayRef<int64_t>> alignment);

}  // namespace vpux
