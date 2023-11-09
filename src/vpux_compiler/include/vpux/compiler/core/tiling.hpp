//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/attributes.hpp"

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/logger.hpp"

#include <mlir/IR/BuiltinAttributes.h>

#include <llvm/Support/raw_ostream.h>

#include <functional>

namespace vpux {

// Experimental number to avoid memory fragmentation when generating tiling.
// This one is also used in memory check of long-term spilling.
static constexpr double FRAGMENTATION_AVOID_RATIO = 0.9;

// Experimental number to avoid memory fragmentation when pipelining
static constexpr double FRAGMENTATION_AVOID_RATIO_PIPELINING = 0.85;

// Experimental number to define large constant size
// The constant filter is considered as large constant value
// when its size is bigger than CMXSize*LARGE_CONST_THRESHOLD_RATIO
static constexpr double LARGE_CONST_THRESHOLD_RATIO = 0.25;

// An experimental number from activation prefetch pass.
// The purpose is to avoid excessive tiling.
static constexpr int MAX_PREFETCH_TILING_TIME = 3;

// Track [E#87286]
// Experimental number to avoid spilling in vertical fusion
static constexpr double VF_WEIGHTS_RATIO = 0.5;

//
// Tiling Mode
//

enum class TilingMode {
    ISOLATED,    // (default) Split each original layer isolated with no heuristics or tweaks
    PIPELINING,  // Create more tiles to enable DMA/DPU overlapping between sub-tiles of one operation
    PREFETCHING  // Create more tiles to enable DMA/DPU overlapping between child and parent operations
};

inline StringRef getTilingModeStr(TilingMode mode) {
    switch (mode) {
    case TilingMode::ISOLATED:
        return StringRef("ISOLATED");
    case TilingMode::PIPELINING:
        return StringRef("PIPELINING");
    case TilingMode::PREFETCHING:
        return StringRef("PREFETCHING");
    default:
        VPUX_THROW("Tiling mode name is not defined");
    }
}

//
// TileInfo
//

struct TileInfo final {
    Shape shape;
    Shape offsets;
    Shape axis;
    // This flag represents a real tile by a tiling process and offsets & axis are meaningful
    bool isCompletedTile = false;

    TileInfo() = delete;

    explicit TileInfo(size_t rank): shape(rank), offsets(rank), axis(rank) {
    }

    explicit TileInfo(ShapeRef shape): shape(shape.raw()), offsets(shape.size(), 0), axis(shape.size(), 1) {
    }

    explicit TileInfo(ShapeRef shape, ShapeRef offsets, ShapeRef axis)
            : shape(shape.raw()), offsets(offsets.raw()), axis(axis.raw()) {
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

// helper function to generate a set of tiles from dividing a shape. A shape divided across multiple
// dimensions will generate a set of tiles, each having its own size and offsets. Additionally an alignment
// can be specified per each dimension.
mlir::FailureOr<OutputTiling> fillDividedTiles(ShapeRef divisors, ShapeRef orig,
                                               Optional<ArrayRef<int64_t>> alignment = None);
mlir::FailureOr<OutputTiling> fillDividedTiles(mlir::Operation* op, ShapeRef divisors, ShapeRef shape);

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

PadInfo backInferPadsTile(const TileInfo& outputTile, ShapeRef inShape, const PadInfo& origPads,
                          ArrayRef<int64_t> kernel, ArrayRef<int64_t> strides);

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
                                   ShapeRef origBiasShape, mlir::ArrayAttr strides, const PadInfo& origPadding,
                                   int64_t groups);

//
// Pooling tiling
//

InputTiling backInferPoolTile(const TileInfo& outputTile, ShapeRef origInputShape, mlir::ArrayAttr kernel_size,
                              mlir::ArrayAttr strides, const PadInfo& origPadding);

//
// Interpolate tiling
//

InputTiling backInferInterpolateTile(const vpux::TileInfo& outputTile, ArrayRef<int64_t> initialInputDims,
                                     ArrayRef<int64_t> initialOutputDims, ArrayRef<int64_t> initialInputOffsets,
                                     ArrayRef<int64_t> initialOutputOffsets, vpux::IE::InterpolateCoordMode coordMode,
                                     vpux::Logger log);

//
// Gather tiling
//

InputTiling backInferGatherTile(const vpux::TileInfo& outputTile, const ShapeRef& origInputShape,
                                const ShapeRef& origIndicesShape, int64_t axisValue, int64_t batchDims,
                                bool hasAxisTensor, vpux::Logger log);

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

std::pair<int64_t, int64_t> spatialOutputForInputWindowSize(const std::pair<int64_t, int64_t>& inputHW,
                                                            const mlir::ArrayAttr& kernel,
                                                            const mlir::ArrayAttr& strides, const PadInfo& pads);

//
// Tiling utils
//

std::tuple<DimRange, int64_t, int64_t> inputForOutputDim(const DimRange& output, int64_t kernel, int64_t stride,
                                                         const DimRange& initialInputRange, int64_t padBefore,
                                                         int64_t padAfter);

template <typename AlignFunc>
SmallVector<int64_t> alignShape(ArrayRef<int64_t> shape, Optional<ArrayRef<int64_t>> alignment, AlignFunc alignFunc) {
    auto alignedShape = to_small_vector(shape);
    if (!alignment.has_value()) {
        return alignedShape;
    }
    std::transform(shape.begin(), shape.end(), alignment.value().begin(), alignedShape.begin(), alignFunc);
    return alignedShape;
}
SmallVector<Strides> adaptStrides(ShapeRef origShape, StridesRef origStrides, ArrayRef<Shape> adaptedShapes,
                                  DimsOrder dimsOrder);

//
// EltwiseOp
//

SmallVector<int64_t> getMaxNumTiles(mlir::Operation* op);
InputTiling backInferEltwiseTile(mlir::Operation* op, const vpux::TileInfo& outputTile);

// SWLayer

mlir::FailureOr<OutputTiling> getSWLayerTilingStrategyWithTileDimOrder(mlir::Operation* op, TilingMode tilingMode,
                                                                       DimArrRef tileDimOrder, Logger log,
                                                                       ArrayRef<int64_t> maxTilesPerDim = {});
mlir::FailureOr<OutputTiling> getSWLayerTilingStrategy(mlir::Operation* op, TilingMode tilingMode, Logger log,
                                                       ArrayRef<int64_t> maxTilesPerDim = {});

InputTiling getSWLayerInputTiles(mlir::Operation* op, const vpux::TileInfo& outputTile);
SmallVector<int64_t> getMaxNumTilesWithAxesExclusion(mlir::Operation* op, ArrayRef<int64_t> axes);

// HWLayer

mlir::FailureOr<OutputTiling> getHWLayerTilingStrategyWithTileDimOrder(mlir::Operation* op, TilingMode tilingMode,
                                                                       DimArrRef tileDimOrder, Logger log);
mlir::FailureOr<OutputTiling> getHWLayerTilingStrategy(mlir::Operation* op, TilingMode tilingMode, Logger log);

DimArr getTileDimOrder(mlir::Operation* op, TilingMode tilingMode, Logger log);
DimArr getTileDimOrderND(MemShape memShape, DimsOrder dimOrder);

}  // namespace vpux
