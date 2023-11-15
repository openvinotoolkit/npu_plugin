//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <llvm/ADT/TypeSwitch.h>

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/utils/core/numeric.hpp"

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"

using namespace vpux;

//
// TileInfo
//

// Imagine shape [8, 8, 9] and divisor [2, 3, 2].
// We'll end up with the following shapes and offsets.

// Shapes   {[4, 3, 5], [4, 3, 5], [4, 3, 5], [4, 3, 5], [4, 2, 5], [4, 2, 5],
//           [4, 3, 4], [4, 3, 4], [4, 3, 4], [4, 3, 4], [4, 2, 4], [4, 2, 4]}
// Offsets  {[0, 0, 0], [4, 0, 0], [0, 3, 0], [4, 3, 0], [0, 6, 0], [4, 6, 0],
//           [0, 0, 5], [4, 0, 5], [0, 3, 5], [4, 3, 5], [0, 6, 0], [4, 6, 5]}

// Arguments:
// dividedTiles - final array of computed tiles
// divisors - array with the tile divisors for each dimension
// shape - original shape to tile
// alignment - array with alignments for each dimension
// dimensionIndex - current dimension index to be processed
// ongoingTile - individual tile solution which we construct and push to dividedTiles array
mlir::LogicalResult divideTiles(OutputTiling& dividedTiles, ShapeRef divisors, ShapeRef shape,
                                ArrayRef<int64_t> alignment, size_t dimensionIndex, vpux::TileInfo& ongoingTile,
                                Logger log = Logger::global()) {
    const auto dimension = Dim(dimensionIndex);

    auto dividedShapes = std::vector<int64_t>(divisors[dimension]);
    auto dividedOffsets = std::vector<int64_t>(divisors[dimension]);
    const auto dividedAxis = divisors[dimension];

    const auto shapeVal = shape[dimension];
    const auto divisorVal = divisors[dimension];
    const auto alignmentVal = alignment[dimensionIndex];

    // Compute tiling scheme per current dimension
    int64_t tileSize, remainderTileSize;
    size_t tileSizeInterval, remainderTileSizeInterval;

    if (alignmentVal > 1) {
        // Whenever there is alignment, all N-1 tiles need to be multiple
        // of said align value.
        // The remainder shape is admitted to not be a mutiple of align value,
        // since this code is tasked to simply tile the original shape, not also align it.
        tileSize = alignValUp(divUp(shapeVal, divisorVal), alignmentVal);
        remainderTileSize = shapeVal - tileSize * (divisorVal - 1);

        if (remainderTileSize <= 0) {
            log.trace("DivideTiles can't meet the request: ShapeVal = {0}, divisorVal = {1}, alignmentTileSize = {2}",
                      shapeVal, divisorVal, tileSize);
            return mlir::failure();
        }

        tileSizeInterval = divisorVal - 1;
        remainderTileSizeInterval = 1;
    } else {
        // When there is no alignment needed, we prefer to distribute the remainder in an
        // equal way across the first tiles.
        // For example 17 tiled 4 ways can be done as:
        // A) [5, 5, 5, 2] when we take the ceil value of the division
        // and leave the remainder as the last tile.
        // B) [5, 4, 4, 4] when we take the floor of the division and distribute
        // the remainder across the first tiles.
        // In any of the two cases, we'll have just 2 distinct values in the shape array.
        tileSize = shapeVal / divisorVal;
        remainderTileSize = shapeVal % divisorVal;

        if (remainderTileSize) {
            tileSizeInterval = remainderTileSize;
            remainderTileSizeInterval = divisorVal - remainderTileSize;
            remainderTileSize = tileSize;
            tileSize++;
        } else {
            tileSizeInterval = divisorVal;
            remainderTileSizeInterval = 0;
        }
    }

    // Compose arrays for shapes and offsets which will be the basis of our backtracking algorithm
    std::fill_n(dividedShapes.begin(), tileSizeInterval, tileSize);
    std::fill_n(dividedShapes.begin() + tileSizeInterval, remainderTileSizeInterval, remainderTileSize);
    int64_t offset = 0;
    for (const auto shapeIndex : irange(dividedShapes.size())) {
        dividedOffsets[shapeIndex] = offset;
        offset += dividedShapes[shapeIndex];
    }

    // Iterate and backtrack on the current list of shapes and offsets
    for (const auto& shapeOffsetPair : zip(dividedShapes, dividedOffsets)) {
        ongoingTile.shape[dimension] = std::get<0>(shapeOffsetPair);
        ongoingTile.offsets[dimension] = std::get<1>(shapeOffsetPair);
        ongoingTile.axis[dimension] = dividedAxis;

        // Full dividedTile is created so need to register the solution
        if (dimensionIndex == (divisors.size() - 1)) {
            dividedTiles.push_back(ongoingTile);
        } else {
            auto isSuccessful = divideTiles(dividedTiles, divisors, shape, alignment, dimensionIndex + 1, ongoingTile);
            if (mlir::failed(isSuccessful)) {
                return mlir::failure();
            }
        }
    }

    return mlir::success();
}

mlir::LogicalResult divideTilesYuvToRgbOp(OutputTiling& dividedTiles, ShapeRef divisors, ShapeRef shape,
                                          vpux::TileInfo& ongoingTile) {
    // N C H W. Tile on C and H dimensions, minimum granularity is 2
    const auto dimC = Dim(Dims4D::Act::C);
    const auto dimH = Dim(Dims4D::Act::H);
    ongoingTile.shape[Dim(Dims4D::Act::N)] = shape[Dim(Dims4D::Act::N)];
    ongoingTile.shape[Dim(Dims4D::Act::W)] = shape[Dim(Dims4D::Act::W)];

    ongoingTile.axis[Dim(Dims4D::Act::N)] = divisors[Dim(Dims4D::Act::N)];
    ongoingTile.axis[Dim(Dims4D::Act::C)] = divisors[Dim(Dims4D::Act::C)];
    ongoingTile.axis[Dim(Dims4D::Act::H)] = divisors[Dim(Dims4D::Act::H)];
    ongoingTile.axis[Dim(Dims4D::Act::W)] = divisors[Dim(Dims4D::Act::W)];

    const auto shapeValC = shape[dimC];
    auto divisorValC = divisors[dimC];

    size_t tileSizeInitC, tileSizeC, remainderTileSizeC;

    tileSizeInitC = shapeValC / divisorValC;
    tileSizeC = tileSizeInitC + (tileSizeInitC % 2);
    divisorValC = shapeValC / tileSizeC;
    remainderTileSizeC = shapeValC % tileSizeC;

    ongoingTile.shape[dimC] = tileSizeC;
    for (int i = 0; i < divisorValC; ++i) {
        ongoingTile.offsets[dimC] = tileSizeC * i;

        const auto shapeValH = shape[dimH];
        auto divisorValH = divisors[dimH];
        size_t tileSizeInitH, tileSizeH, remainderTileSizeH;

        tileSizeInitH = shapeValH / divisorValH;
        tileSizeH = tileSizeInitH + (tileSizeInitH % 2);
        divisorValH = shapeValH / tileSizeH;
        remainderTileSizeH = shapeValH % tileSizeH;
        ongoingTile.shape[dimH] = tileSizeH;

        for (int j = 0; j < divisorValH; ++j) {
            ongoingTile.offsets[dimH] = tileSizeH * j;
            dividedTiles.push_back(ongoingTile);
        }

        if (remainderTileSizeH) {
            ongoingTile.shape[dimH] = remainderTileSizeH;
            ongoingTile.offsets[dimH] = tileSizeH * divisorValH;
            dividedTiles.push_back(ongoingTile);
        }
    }

    if (remainderTileSizeC) {
        ongoingTile.shape[dimC] = remainderTileSizeC;
        ongoingTile.offsets[dimC] = tileSizeC * divisorValC;

        const auto shapeValH = shape[dimH];
        auto divisorValH = divisors[dimH];
        size_t tileSizeInitH, tileSizeH, remainderTileSizeH;

        tileSizeInitH = shapeValH / divisorValH;
        tileSizeH = tileSizeInitH + (tileSizeInitH % 2);
        divisorValH = shapeValH / tileSizeH;
        remainderTileSizeH = shapeValH % tileSizeH;
        ongoingTile.shape[dimH] = tileSizeH;

        for (int j = 0; j < divisorValH; ++j) {
            ongoingTile.offsets[dimH] = tileSizeH * j;
            dividedTiles.push_back(ongoingTile);
        }

        if (remainderTileSizeH) {
            ongoingTile.shape[dimH] = remainderTileSizeH;
            ongoingTile.offsets[dimH] = tileSizeH * divisorValH;
            dividedTiles.push_back(ongoingTile);
        }
    }

    return mlir::success();
}

mlir::FailureOr<OutputTiling> fillDividedTilesYuvToRgbOp(ShapeRef divisors, ShapeRef shape) {
    OutputTiling dividedTiles;

    auto ongoingTile = vpux::TileInfo(divisors.size());
    ongoingTile.isCompletedTile = true;

    auto isSuccessful = divideTilesYuvToRgbOp(dividedTiles, divisors, shape, ongoingTile);
    if (mlir::failed(isSuccessful)) {
        return mlir::failure();
    }

    return dividedTiles;
}

mlir::FailureOr<OutputTiling> vpux::fillDividedTiles(ShapeRef divisors, ShapeRef shape,
                                                     Optional<ArrayRef<int64_t>> alignment) {
    OutputTiling dividedTiles;

    auto ongoingTile = vpux::TileInfo(divisors.size());
    ongoingTile.isCompletedTile = true;

    auto alignmentShape = SmallVector<int64_t>(shape.size(), 1);
    auto alignmentShapeRef = makeArrayRef(alignmentShape);
    if (alignment.has_value()) {
        alignmentShapeRef = alignment.value();
    }

    auto isSuccessful = divideTiles(dividedTiles, divisors, shape, alignmentShapeRef, 0, ongoingTile);
    if (mlir::failed(isSuccessful)) {
        return mlir::failure();
    }

    return dividedTiles;
}

mlir::FailureOr<OutputTiling> vpux::fillDividedTiles(mlir::Operation* op, ShapeRef divisors, ShapeRef shape) {
    OutputTiling dividedTiles;
    if (mlir::isa<VPU::YuvToRgbOp>(op)) {
        return fillDividedTilesYuvToRgbOp(divisors, shape);
    }

    Optional<ArrayRef<int64_t>> optionalAlignment = None;
    auto alignment = SmallVector<int64_t>(shape.size(), 1);
    if (auto tilingIface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op)) {
        alignment[vpux::Dims4D::Act::C.ind()] = tilingIface.getOutputChannelAlignment();
        optionalAlignment = Optional<ArrayRef<int64_t>>(alignment);
    }

    return vpux::fillDividedTiles(divisors, shape, optionalAlignment);
}

//
// PadInfo
//

PadInfo vpux::backInferPadsTile(const TileInfo& outputTile, ShapeRef inShape, const PadInfo& origPads,
                                ArrayRef<int64_t> kernel, ArrayRef<int64_t> strides) {
    const std::array<int64_t, Dims4D::Act::numSpatialDims> origPadsBegin = {origPads.top, origPads.left};
    const std::array<int64_t, Dims4D::Act::numSpatialDims> origPadsEnd = {origPads.bottom, origPads.right};

    SmallVector<int64_t> tilePadsBegin(Dims4D::Act::numSpatialDims);
    SmallVector<int64_t> tilePadsEnd(Dims4D::Act::numSpatialDims);

    for (auto ind : irange(Dims4D::Act::numSpatialDims)) {
        const auto spatialDim = Dims4D::Act::getSpatialDim(ind);

        const auto outSize = outputTile.shape[spatialDim];
        const auto outOffset = outputTile.offsets[spatialDim];

        const DimRange inputRange(0, inShape[spatialDim]);
        const DimRange tileRange(outOffset, outOffset + outSize);

        std::tie(std::ignore, tilePadsBegin[ind], tilePadsEnd[ind]) = inputForOutputDim(
                tileRange, kernel[ind], strides[ind], inputRange, origPadsBegin[ind], origPadsEnd[ind]);
    }

    return PadInfo(tilePadsBegin[1], tilePadsEnd[1], tilePadsBegin[0], tilePadsEnd[0]);
}

//
// Common tiling utilities
//

namespace {

struct PlaneTile final {
    DimRange width;
    DimRange height;

    int64_t area() const {
        return width.length() * height.length();
    }

    // Checks if rhs located completely in this.
    bool contains(const PlaneTile& other) const {
        return width.contains(other.width) && height.contains(other.height);
    }

    // Returns new `PlaneTile` which represents `other` as ROI of this.
    PlaneTile asROI(const PlaneTile& other) const {
        return {width.asROI(other.width), height.asROI(other.height)};
    }

    bool operator==(const PlaneTile& other) const {
        return width == other.width && height == other.height;
    }
    bool operator!=(const PlaneTile& other) const {
        return !(*this == other);
    }

    void printFormat(llvm::raw_ostream& stream) const {
        printTo(stream, "PlaneTile [width tile = {0}, height tile = {1}]", width, height);
    }
};

struct PlaneTileSolution final {
    // Input tile which meets HW requirements in terms of alignment.
    PlaneTile inputTile;

    // Padding which should be applied to input tile in order to calculate output tile.
    // Meets HW requirements in terms of size and symmetry.
    PadInfo inputPad;

    void printFormat(llvm::raw_ostream& stream) const {
        printTo(stream, "PlaneTileSolution [inputTile = {0}, inputPad = {1}]", inputTile, inputPad);
    }
};

// Return input tile and padding required to calculate the output tile.
// Padding should be applied to the input tile. It could be asymmetric, or doesn't meet HW requirements in terms of its
// size.
// * initialInputDims - Dims of the whole input tensor (not of specific tile).
// * initialPad - padding which should be applied to the whole input tensor (not to specific tile).
std::tuple<PlaneTile, PadInfo> inputForOutputTile(const PlaneTile& output, int64_t kernelX, int64_t kernelY,
                                                  int64_t strideX, int64_t strideY, ShapeRef initialInputDims,
                                                  const PadInfo& initialPad) {
    PlaneTile inputTile = {{0, 0}, {0, 0}};
    PadInfo pad = {0, 0, 0, 0};

    std::tie(inputTile.height, pad.top, pad.bottom) = inputForOutputDim(
            output.height, kernelY, strideY, {0, initialInputDims[Dims4D::Act::H]}, initialPad.top, initialPad.bottom);

    std::tie(inputTile.width, pad.left, pad.right) = inputForOutputDim(
            output.width, kernelX, strideX, {0, initialInputDims[Dims4D::Act::W]}, initialPad.left, initialPad.right);

    return std::make_tuple(inputTile, pad);
}

PlaneTileSolution solutionForOutputTile(const PlaneTile& output, int64_t kernelX, int64_t kernelY, int64_t strideX,
                                        int64_t strideY, ShapeRef initialInputDims, const PadInfo& initialPad) {
    PlaneTileSolution solution;
    std::tie(solution.inputTile, solution.inputPad) =
            inputForOutputTile(output, kernelX, kernelY, strideX, strideY, initialInputDims, initialPad);

    return solution;
}

}  // namespace

// inputTile planar H/W size should keep the same with original input H/W when no tiling over those axis.
// However the back inferring size may become smaller, e.g., OutputTile 7x7, Kernel 1x1, Stride 2x2.
// The inferring inputTile planar shape is 13x13 however original planar input shape may be 14x14, which will cause
// a redundant data slice from input. Here is to restore original input planar shape to avoid extra copies.
void restorePlanarShapeForInputTile(TileInfo& inputTile, ShapeRef origInputShape, vpux::Dim planarDim) {
    if (planarDim != Dims4D::Act::H && planarDim != Dims4D::Act::W) {
        VPUX_THROW("Invalid planar dim {0}", planarDim);
    }
    if (inputTile.shape[planarDim] > origInputShape[planarDim]) {
        VPUX_THROW("Invalid back inferring size {0} over dim {1}", inputTile.shape[planarDim], planarDim);
    }

    inputTile.shape[planarDim] = origInputShape[planarDim];
}

//
// Convolution tiling
//

InputTiling vpux::backInferConvTile(const TileInfo& outputTile, ShapeRef origInputShape, ShapeRef origFilterShape,
                                    ShapeRef origBiasShape, mlir::ArrayAttr strides, const PadInfo& origPadding) {
    PlaneTile output;
    output.height.begin = outputTile.offsets[Dims4D::Act::H];
    output.height.end = outputTile.offsets[Dims4D::Act::H] + outputTile.shape[Dims4D::Act::H];
    output.width.begin = outputTile.offsets[Dims4D::Act::W];
    output.width.end = outputTile.offsets[Dims4D::Act::W] + outputTile.shape[Dims4D::Act::W];

    const auto strideY = strides[Dims4D::Strides::Y.ind()].cast<mlir::IntegerAttr>().getValue().getSExtValue();
    const auto strideX = strides[Dims4D::Strides::X.ind()].cast<mlir::IntegerAttr>().getValue().getSExtValue();

    const auto solution =
            solutionForOutputTile(output, origFilterShape[Dims4D::Filter::KX], origFilterShape[Dims4D::Filter::KY],
                                  strideX, strideY, origInputShape, origPadding);

    TileInfo inputTile(origInputShape);
    TileInfo filterTile(origFilterShape);
    TileInfo biasTile(origBiasShape);

    inputTile.shape[Dims4D::Act::N] = outputTile.shape[Dims4D::Act::N];
    inputTile.offsets[Dims4D::Act::N] = outputTile.offsets[Dims4D::Act::N];

    inputTile.offsets[Dims4D::Act::H] = solution.inputTile.height.begin;
    inputTile.shape[Dims4D::Act::H] = solution.inputTile.height.length();

    inputTile.offsets[Dims4D::Act::W] = solution.inputTile.width.begin;
    inputTile.shape[Dims4D::Act::W] = solution.inputTile.width.length();

    if (outputTile.isCompletedTile && outputTile.axis[Dims4D::Act::H] == 1) {
        restorePlanarShapeForInputTile(inputTile, origInputShape, Dims4D::Act::H);
    }
    if (outputTile.isCompletedTile && outputTile.axis[Dims4D::Act::W] == 1) {
        restorePlanarShapeForInputTile(inputTile, origInputShape, Dims4D::Act::W);
    }

    filterTile.shape[Dims4D::Filter::OC] = outputTile.shape[Dims4D::Act::C];
    filterTile.offsets[Dims4D::Filter::OC] = outputTile.offsets[Dims4D::Act::C];

    if (!biasTile.shape.empty()) {
        biasTile.shape[Dims4D::Act::C] = outputTile.shape[Dims4D::Act::C];
        biasTile.offsets[Dims4D::Act::C] = outputTile.offsets[Dims4D::Act::C];
        return TilingInfo{{inputTile, filterTile, biasTile}, solution.inputPad};
    }
    return TilingInfo{{inputTile, filterTile}, solution.inputPad};
}

InputTiling vpux::backInferGroupConvTile(const TileInfo& outputTile, ShapeRef origInputShape, ShapeRef origFilterShape,
                                         ShapeRef origBiasShape, mlir::ArrayAttr strides, const PadInfo& origPadding,
                                         int64_t groups) {
    auto res = backInferConvTile(outputTile, origInputShape, origFilterShape, origBiasShape, strides, origPadding);

    const auto inputTileIdx = 0;
    auto& inputTiles = res.tiles[inputTileIdx];

    // For GroupConv, the weights' OC dim is the product of num_group * num_channels_per_group
    const auto numOutChannelsPerGroup = origFilterShape[Dims4D::Filter::OC] / groups;

    // To correctly compute input tile when tiling is done over out channels, we need to determine
    // the start group for the tile and the number of groups it spans.
    // Based on them, we can back-infer the necessary input tile.
    // E.g. GroupConv groups = 6; in channels = 12; out channels = 18; filter = (groups * 3 out ch) x 2 in ch
    //      w/ tiling = [1, 3, 1, 1]
    // The resulting tiled GroupConvs are:
    //      Tile 0: GC w/ groups = 2 (group 0 & 1 of orig GC): out channels 0 - 5, in channels 0 - 3
    //      Tile 1: GC w/ groups = 2 (group 2 & 3 of orig GC): out channels 6 - 11, in channels 4 - 7
    //      Tile 2: GC w/ groups = 2 (group 4 & 5 of orig GC): out channels 12 - 17, in channels 8 - 11
    const auto startGroupForTile = outputTile.offsets[Dims4D::Act::C] / numOutChannelsPerGroup;
    const auto numGroupsForTile = divUp(outputTile.shape[Dims4D::Act::C], numOutChannelsPerGroup);

    inputTiles.offsets[Dims4D::Act::C] = startGroupForTile * origFilterShape[Dims4D::Filter::IC];
    inputTiles.shape[Dims4D::Act::C] = numGroupsForTile * origFilterShape[Dims4D::Filter::IC];

    return res;
}

//
// Pooling tiling
//

InputTiling vpux::backInferPoolTile(const TileInfo& outputTile, ShapeRef origInputShape, mlir::ArrayAttr kernel_size,
                                    mlir::ArrayAttr strides, const PadInfo& origPadding) {
    PlaneTile output;
    output.height.begin = outputTile.offsets[Dims4D::Act::H];
    output.height.end = outputTile.offsets[Dims4D::Act::H] + outputTile.shape[Dims4D::Act::H];
    output.width.begin = outputTile.offsets[Dims4D::Act::W];
    output.width.end = outputTile.offsets[Dims4D::Act::W] + outputTile.shape[Dims4D::Act::W];

    const auto kernelY = kernel_size[Dims4D::Kernel::Y.ind()].cast<mlir::IntegerAttr>().getValue().getSExtValue();
    const auto kernelX = kernel_size[Dims4D::Kernel::X.ind()].cast<mlir::IntegerAttr>().getValue().getSExtValue();

    const auto strideY = strides[Dims4D::Strides::Y.ind()].cast<mlir::IntegerAttr>().getValue().getSExtValue();
    const auto strideX = strides[Dims4D::Strides::X.ind()].cast<mlir::IntegerAttr>().getValue().getSExtValue();

    const auto solution =
            solutionForOutputTile(output, kernelX, kernelY, strideX, strideY, origInputShape, origPadding);

    TileInfo inputTile(origInputShape);

    inputTile.shape[Dims4D::Act::N] = outputTile.shape[Dims4D::Act::N];
    inputTile.offsets[Dims4D::Act::N] = outputTile.offsets[Dims4D::Act::N];

    inputTile.shape[Dims4D::Act::C] = outputTile.shape[Dims4D::Act::C];
    inputTile.offsets[Dims4D::Act::C] = outputTile.offsets[Dims4D::Act::C];

    inputTile.offsets[Dims4D::Act::H] = solution.inputTile.height.begin;
    inputTile.shape[Dims4D::Act::H] = solution.inputTile.height.length();

    inputTile.offsets[Dims4D::Act::W] = solution.inputTile.width.begin;
    inputTile.shape[Dims4D::Act::W] = solution.inputTile.width.length();

    if (outputTile.isCompletedTile && outputTile.axis[Dims4D::Act::H] == 1) {
        restorePlanarShapeForInputTile(inputTile, origInputShape, Dims4D::Act::H);
    }
    if (outputTile.isCompletedTile && outputTile.axis[Dims4D::Act::W] == 1) {
        restorePlanarShapeForInputTile(inputTile, origInputShape, Dims4D::Act::W);
    }

    return TilingInfo{{inputTile}, solution.inputPad};
}

namespace {

SmallVector<int64_t> propagateOffsetForInterpolate(ArrayRef<int64_t> axes, ArrayRef<int64_t> offset,
                                                   ArrayRef<int64_t> initialInputDims,
                                                   ArrayRef<int64_t> initialOutputDims,
                                                   vpux::IE::InterpolateCalcMode calcMode,
                                                   vpux::IE::InterpolateCoordMode coordMode, ArrayRef<int64_t> sizes,
                                                   ArrayRef<double> scales, bool roundUp, vpux::Logger log) {
    log.trace("Interp propagate offset: input = {0}", offset);

    std::function<double(double)> func =
            roundUp ? static_cast<double (*)(double)>(ceil) : static_cast<double (*)(double)>(floor);

    // Transform the coordinate in the resized tensor to the coordinate in the original tensor.
    // It is from Interpolate-4 document at OpenVINO.
    // scale = input_shape / output_shape
    auto inferInCoord = [&](int64_t outCoord, int64_t origInSize, int64_t origOutSize, double scale) {
        int64_t inCoord = 0;
        if (coordMode == IE::InterpolateCoordMode::HALF_PIXEL) {
            inCoord = static_cast<int64_t>(func(scale * (outCoord + 0.5) - 0.5));
        } else if (coordMode == IE::InterpolateCoordMode::PYTORCH_HALF_PIXEL) {
            inCoord = origOutSize == 1 ? static_cast<int64_t>(0)
                                       : static_cast<int64_t>(func(scale * (outCoord + 0.5) - 0.5));
        } else if (coordMode == IE::InterpolateCoordMode::ASYMMETRIC) {
            inCoord = static_cast<int64_t>(func(outCoord * scale));
        } else if (coordMode == IE::InterpolateCoordMode::TF_HALF_PIXEL_FOR_NN) {
            inCoord = static_cast<int64_t>(func((outCoord + 0.5) * scale));
        } else if (coordMode == IE::InterpolateCoordMode::ALIGN_CORNERS) {
            inCoord = origOutSize == 1
                              ? static_cast<int64_t>(0)
                              : static_cast<int64_t>(func(outCoord * (origInSize - 1.0) / (origOutSize - 1.0)));
        } else {
            VPUX_THROW("Doesn't support coordMode: {0}", coordMode);
        }
        return std::min(std::max(static_cast<int64_t>(0), inCoord), origInSize - 1);
    };

    SmallVector<int64_t> inferedOffset(offset.begin(), offset.end());
    if (calcMode == IE::InterpolateCalcMode::SIZES) {
        VPUX_THROW_WHEN(sizes.size() != axes.size(),
                        "Num of elements in sizes tensor: {0} should be equal to number of indices in axes: {1}",
                        sizes.size(), axes.size());
        auto sizesIter = sizes.begin();
        for (const auto& i : axes) {
            log.trace("Interp sizes - axis: {0}", i);
            inferedOffset[i] = *sizesIter++;
        }
    } else if (calcMode == IE::InterpolateCalcMode::SCALES) {
        VPUX_THROW_WHEN(scales.size() != axes.size(),
                        "Num of elements in scales tensor: {0} should be equal to number of indices in axes: {1}",
                        scales.size(), axes.size());
        auto scalesIter = scales.begin();
        for (const auto& i : axes) {
            log.trace("Interp scales - axis: {0}", i);
            inferedOffset[i] = inferInCoord(offset[i], initialInputDims[i], initialOutputDims[i], (*scalesIter++));
        }
    } else {
        VPUX_THROW("Doesn't support shape_calculation_mode: {0}", calcMode);
    }

    log.trace("Interp propagate offset: output = {0}", inferedOffset);
    return inferedOffset;
}

SmallVector<int64_t> backInferOffsetForInterpolate(ArrayRef<int64_t> offset, IE::InterpolateCoordMode coordMode,
                                                   ArrayRef<int64_t> initialInputDims,
                                                   ArrayRef<int64_t> initialOutputDims, bool roundUp, Logger log) {
    SmallVector<int64_t> axes;
    for (auto i : irange(initialInputDims.size())) {
        if (initialInputDims[i] != initialOutputDims[i]) {
            axes.push_back(i);
        }
    }

    // Compute scale-factors based on full I/O resolution ratio
    SmallVector<int64_t> fullOutSize;
    SmallVector<double> backwardScale;
    for (size_t i = 0; i < axes.size(); i++) {
        backwardScale.push_back(static_cast<double>(initialInputDims[axes[i]]) / initialOutputDims[axes[i]]);
        fullOutSize.push_back(initialOutputDims[axes[i]]);
    }

    // TODO: E#36318 how to deal with calc-mode = size if scales missed - recalc them somewhere:
    auto shapeCalcMode = IE::InterpolateCalcMode::SCALES;
    return propagateOffsetForInterpolate(axes, offset, initialInputDims, initialOutputDims, shapeCalcMode, coordMode,
                                         fullOutSize, backwardScale, roundUp, log);
}
}  // namespace

//
// Interpolate tiling
//

InputTiling vpux::backInferInterpolateTile(const vpux::TileInfo& outputTile, ArrayRef<int64_t> initialInputDims,
                                           ArrayRef<int64_t> initialOutputDims, ArrayRef<int64_t> initialInputOffsets,
                                           ArrayRef<int64_t> initialOutputOffsets,
                                           vpux::IE::InterpolateCoordMode coordMode, vpux::Logger log) {
    log.trace("Try to back infer input tiling for Interpolate, output tile: {0}", outputTile);
    // Before get the input tiles,  need update the output tile's offset against to the initial output. Otherwise the
    // input tiles may be invalid
    auto outputOffsetBegin = to_small_vector(outputTile.offsets);
    auto outputOffsetEnd = to_small_vector(outputTile.offsets);
    for (size_t ind = 0; ind < outputOffsetEnd.size(); ind++) {
        outputOffsetBegin[ind] += initialOutputOffsets[ind];
        outputOffsetEnd[ind] = outputOffsetBegin[ind] + outputTile.shape[Dim(ind)] - 1;
    }

    auto inferedInputOffsetBegin = backInferOffsetForInterpolate(outputOffsetBegin, coordMode, initialInputDims,
                                                                 initialOutputDims, false, log);
    auto inferedInputOffsetEnd =
            backInferOffsetForInterpolate(outputOffsetEnd, coordMode, initialInputDims, initialOutputDims, true, log);

    SmallVector<int64_t> inferedInputShape(inferedInputOffsetEnd.size(), 0);
    for (size_t ind = 0; ind < inferedInputOffsetEnd.size(); ind++) {
        inferedInputShape[ind] = inferedInputOffsetEnd[ind] - inferedInputOffsetBegin[ind] + 1;
        inferedInputOffsetBegin[ind] -= initialInputOffsets[ind];
    }

    TileInfo inputTile{inferedInputShape.size()};
    inputTile.shape = Shape(inferedInputShape);
    inputTile.offsets = Shape(inferedInputOffsetBegin);
    inputTile.axis = outputTile.axis;
    SmallVector<TileInfo> tiles(1, inputTile);
    auto iTiling = InputTiling{tiles};
    return iTiling;
}

//
// Gather tiling
//

InputTiling vpux::backInferGatherTile(const vpux::TileInfo& outputTile, const ShapeRef& origInputShape,
                                      const ShapeRef& origIndicesShape, int64_t axisValue, int64_t batchDims,
                                      bool hasAxisTensor, vpux::Logger log) {
    log.trace("Try to back infer input tiling for Gather, output tile: {0}", outputTile);
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

    if (hasAxisTensor) {
        return InputTiling{{std::move(inputTile), std::move(indicesTile), TileInfo(ShapeRef({1}))}};
    } else {
        return InputTiling{{std::move(inputTile), std::move(indicesTile)}};
    }
}

/// @brief Infer output window size OH X OW from input window size IH X IW
std::pair<int64_t, int64_t> vpux::spatialOutputForInputWindowSize(const std::pair<int64_t, int64_t>& inputHW,
                                                                  const mlir::ArrayAttr& kernelAttr,
                                                                  const mlir::ArrayAttr& stridesAttr,
                                                                  const PadInfo& pads) {
    VPUX_THROW_WHEN(kernelAttr == nullptr, "Kernel shouldn't be nullptr");
    VPUX_THROW_WHEN(stridesAttr == nullptr, "Strides shouldn't be nullptr");

    const auto kernel = parseIntArrayAttr<int64_t>(kernelAttr);
    VPUX_THROW_WHEN(kernel.size() != 2, "Expected kernel size to be 2. Got '{0}'", kernel.size());
    const auto KY = kernel[Dims4D::Kernel::Y.ind()];
    const auto KX = kernel[Dims4D::Kernel::X.ind()];

    const auto strides = parseIntArrayAttr<int64_t>(stridesAttr);
    VPUX_THROW_WHEN(strides.size() != 2, "Expected strides size to be 2. Got '{0}'", strides.size());
    const auto SY = strides[Dims4D::Strides::Y.ind()];
    const auto SX = strides[Dims4D::Strides::X.ind()];

    const auto padTop = pads.top;
    const auto padBottom = pads.bottom;
    const auto padLeft = pads.left;
    const auto padRight = pads.right;
    if (padTop < 0 || padBottom < 0 || padLeft < 0 || padRight < 0) {
        VPUX_THROW("Invalid pads: top '{0}', bottom '{1}', left '{2}', right '{3}'", padTop, padBottom, padLeft,
                   padRight);
    }

    const auto outputHeight = (inputHW.first - KY + padTop + padBottom) / SY + 1;
    const auto outputWidth = (inputHW.second - KX + padLeft + padRight) / SX + 1;

    VPUX_THROW_UNLESS(outputHeight > 0 && outputWidth > 0,
                      "Inferred output height and width should be larger than zero");
    return {outputHeight, outputWidth};
}

//
// Tiling utils
//

std::tuple<DimRange, int64_t, int64_t> vpux::inputForOutputDim(const DimRange& output, int64_t kernel, int64_t stride,
                                                               const DimRange& initialInputRange, int64_t padBefore,
                                                               int64_t padAfter) {
    VPUX_THROW_UNLESS(output.length() > 0, "Wrong output tile '{0}'", output);
    VPUX_THROW_UNLESS(initialInputRange.length() > 0, "Wrong initial input range '{0}'", initialInputRange);
    VPUX_THROW_UNLESS(kernel > 0, "Wrong kernel '{0}'", kernel);
    VPUX_THROW_UNLESS(stride > 0, "Wrong stride '{0}'", stride);
    VPUX_THROW_UNLESS(padBefore >= 0, "Wrong padBefore '{0}'", padBefore);
    VPUX_THROW_UNLESS(padAfter >= 0, "Wrong padAfter '{0}'", padAfter);

    DimRange input = {0, 0};
    int64_t before = 0;
    int64_t after = 0;

    input.begin = output.begin * stride - padBefore;

    if (input.begin < initialInputRange.begin) {
        VPUX_THROW_UNLESS(initialInputRange.begin - input.begin <= padBefore,
                          "Input tile '{0}' and padBefore '{1}' doesn't match to initial range '{2}'", input, padBefore,
                          initialInputRange);

        before = std::min(initialInputRange.begin - input.begin, padBefore);
        input.begin = initialInputRange.begin;
    }

    VPUX_THROW_UNLESS(input.begin < initialInputRange.end, "Input tile '{0}' doesn't match to initial range '{1}'",
                      input, initialInputRange);

    input.end = (output.end - 1) * stride + kernel - padBefore;

    if (input.end > initialInputRange.end) {
        VPUX_THROW_UNLESS(input.end - initialInputRange.end <= padAfter,
                          "Input tile '{0}' and padAfter '{1}' doesn't match to initial range '{2}'", input, padAfter,
                          initialInputRange);

        after = std::min(input.end - initialInputRange.end, padAfter);
        input.end = initialInputRange.end;
    }

    VPUX_THROW_UNLESS(input.end > initialInputRange.begin, "Input tile '{0}' doesn't match to initial range '{1}'",
                      input, initialInputRange);
    VPUX_THROW_UNLESS(input.length() > 0, "Input tile '{0}' doesn't match to initial range '{1}'", input,
                      initialInputRange);

    return std::make_tuple(input, before, after);
}

// @brief Following function computes new strides based on the new tensor shape.
// @warning The new shape can be a result of tiling or aligning or something else.
SmallVector<Strides> vpux::adaptStrides(ShapeRef origShape, StridesRef origStrides, ArrayRef<Shape> adaptedShapes,
                                        DimsOrder dimsOrder) {
    auto adaptedStrides = SmallVector<Strides>();
    const auto memShape = dimsOrder.toMemoryOrder(origShape);
    const auto memStrides = dimsOrder.toMemoryOrder(origStrides);

    for (const auto& adaptedShape : adaptedShapes) {
        const auto adaptedMemShape = dimsOrder.toMemoryOrder(Shape(adaptedShape));

        SmallVector<Bit> adaptedMemStrides(memStrides.raw());
        // Automatically adaptedMemStrides.back() is equal to the element type size
        for (int i = static_cast<int>(memStrides.size()) - 2; i >= 0; --i) {
            // Compute the ration between consecutive strides.
            // This tells us how many elements were accounted for in the original
            // strides and by using this, we incrementally construct the new adapted strides.
            const auto currStride = memStrides[MemDim(i)].count();
            const auto prevStride = memStrides[MemDim(i + 1)].count();
            const auto prevAdaptedStride = adaptedMemStrides[i + 1].count();

            auto adaptedStride = prevAdaptedStride * currStride / prevStride;

            if (adaptedStride != (int)adaptedStride) {
                vpux::Logger log("VPUX Adapt Strides Tiling method", vpux::LogLevel::Error);
                log.error("Adapted strides has decimals and may cause problems");
            }

            const auto strideRatio = currStride / prevStride;
            // If there is a change between the original and the new shape,
            // we favor striding with the new shape size instead of the previous stride ratio.
            if (memShape[MemDim(i + 1)] != adaptedMemShape[MemDim(i + 1)]) {
                // In the case of multiclustering, all such scenarios like H|K cluster tiling
                // with H|K prefetch tiling should be concatenated in DDR as simple tensors.

                // Long story to why we don't allow strides and tiling on same axis:
                // Mostly it's unclear how we should handle correctly such a case, because the nature
                // of strides can be very multifaceted, and we don't have explicit knowledge of the
                // scope for that stride.
                //
                // Let's take a case like 24 dimension strided to 32.
                // You may do this to either stride to 32 to fit in a concat over the specific axis
                // Or you may do this for alignment reasons, such that each pixel starts at a 16 byte
                // aligned address.
                //
                // So if we tile 24 by 2, and have 12. How should the strides be adapted?
                // Should we keep them as 32 to satisfy the concat or should we readjust them and align
                // to next value multiple of 16, which will be 16.
                // It's this lack of information and very context dependent reason why we avoid to
                // tackle this case.
                //
                // Without having a solid and functional infrastructure, to do everything in full knowledge
                // of context it can easily lead to a lot of problems and instabilities in the future.

                VPUX_THROW_WHEN(strideRatio != memShape[MemDim(i + 1)],
                                "Can't have both stride ratio '{0}' != shape '{1}' and also adapted shape '{2}' on "
                                "same axis '{3}'.",
                                strideRatio, memShape[MemDim(i + 1)], adaptedMemShape[MemDim(i + 1)], i + 1);
                adaptedStride = adaptedMemShape[MemDim(i + 1)] * prevAdaptedStride;
            }

            adaptedMemStrides[i] = Bit(adaptedStride);
        }
        adaptedStrides.emplace_back(dimsOrder.toLogicalOrder(MemStrides(adaptedMemStrides)));
    }

    return adaptedStrides;
}

DimArr vpux::getTileDimOrderND(MemShape memShape, DimsOrder dimOrder) {
    // Function calculates tile dim order from memory shape and dimOrder
    // It prioritize dim order depending on dim size and dimsOrder
    // Ex: MemShape: 3x80x80x40x80  DimOrder: NCDHW (0x12345)
    //      the return will be {1, 2, 4, 3, 0}
    //            equivalent:  {C, D, W, H, N}
    auto outputMemShape = memShape.raw();
    auto outputSortShape = memShape.raw();
    const auto outputDimOrderVec = dimOrder.toPermutation();

    std::sort(outputSortShape.begin(), outputSortShape.end(), std::greater<int64_t>());

    DimArr returntileDimOrder;

    for (auto it : outputSortShape) {
        if (it > 1) {
            // find the first value that match
            auto dimIt = std::find(outputMemShape.begin(), outputMemShape.end(), it);
            // extract the DimOrder
            returntileDimOrder.push_back(outputDimOrderVec[dimIt - outputMemShape.begin()]);
            // set the value to 0 to avoid geting the same index if more values are equals
            *dimIt = 0;
        }
    }

    return returntileDimOrder;
}

DimArr vpux::getTileDimOrder(mlir::Operation* op, TilingMode tilingMode, Logger log) {
    // Compare the Activation and Filter size
    // if activation size > filter size
    //      First tile at H
    // if activation size <= filter size
    //      First tile at C
    // Result in less tiles being required to fit in CMX.
    auto tileDimOrder =
            llvm::TypeSwitch<mlir::Operation*, DimArr>(op)
                    .Case<VPU::NCEConvolutionOp, VPU::NCEDepthConvolutionOp, VPU::NCECompressConvolutionOp>(
                            [&](mlir::Operation* op) {
                                log.nest(2).trace("Check tile Dim order for Op at {0}", op->getLoc());
                                const auto activationType = op->getOperand(0).getType().cast<vpux::NDTypeInterface>();
                                const auto filterType = op->getOperand(1).getType().cast<vpux::NDTypeInterface>();
                                return filterType.getTotalAllocSize() < activationType.getTotalAllocSize()
                                               ? DimArr{Dims4D::Act::H, Dims4D::Act::C, Dims4D::Act::W}
                                               : DimArr{Dims4D::Act::C, Dims4D::Act::H, Dims4D::Act::W};
                            })
                    .Case<VPU::NCEMaxPoolOp, VPU::NCEAveragePoolOp>([&](mlir::Operation* op) {
                        log.nest(2).trace("Check tile Dim order for Op at {0}", op->getLoc());
                        return DimArr{Dims4D::Act::H, Dims4D::Act::C, Dims4D::Act::W};
                    })
                    .Case<VPU::MVNOp>([&](mlir::Operation* op) {
                        log.nest(2).trace("Check tile Dim order for Op at {0}", op->getLoc());
                        auto mvn1 = mlir::dyn_cast<VPU::MVNOp>(op);
                        auto dims = mvn1.getNonNormDims();
                        VPUX_THROW_UNLESS(dims.size(), "Could not find non-norm axes");
                        return dims;
                    })
                    .Case<VPU::MVN6Op>([&](mlir::Operation* op) {
                        auto mvn6 = mlir::dyn_cast<VPU::MVN6Op>(op);
                        auto dims = mvn6.getNonNormDims();
                        VPUX_THROW_UNLESS(dims.size(), "Could not find non-norm axes");
                        return dims;
                    })
                    .Case<VPU::QuantizeOp>([&](mlir::Operation*) {
                        // Not splitting over C, to keep aligned with number of Scales in qType
                        // and so avoid 'validateQuantElemType' fail
                        return DimArr{Dims4D::Act::H, Dims4D::Act::W};
                    })
                    .Case<VPU::DequantizeOp>([&](mlir::Operation*) {
                        return DimArr{Dims4D::Act::H, Dims4D::Act::W};
                    })
                    .Case<VPU::DetectionOutputDecodeBoxesOp>([&](mlir::Operation*) {
                        return DimArr{Dims4D::Act::C, Dims4D::Act::H};
                    })
                    .Case<VPU::DetectionOutputSelectBoxesOp>([&](mlir::Operation*) {
                        return DimArr{Dims4D::Act::C};
                    })
                    .Case<VPU::DetectionOutputSortTopKOp>([&](mlir::Operation*) {
                        return DimArr{Dim(1)};  // [N, numClasses, numBoxes]
                    })
                    .Case<VPU::NCEEltwiseOp>([&](mlir::Operation* op) {
                        const auto outputShape = getShape(op->getResult(0));
                        return outputShape[Dims4D::Act::C] / VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT <
                                               outputShape[Dims4D::Act::H]
                                       ? DimArr{Dims4D::Act::H, Dims4D::Act::C, Dims4D::Act::W}
                                       : DimArr{Dims4D::Act::C, Dims4D::Act::H, Dims4D::Act::W};
                    })
                    .Case<VPU::NCEPermuteQuantizeOp>([&](mlir::Operation* op) {
                        log.nest(2).trace("Check tile Dim order for Op at {0}", op->getLoc());
                        // Prefer splitting PermuteQuantize over tensor width.
                        // Input tensors have 1x32xCxW shape.
                        // W here is the total size of original tensor divided by 32 * C.
                        // Split over width produces more even chunks than split over height in that case.
                        // If channel is bigger than VPU_DIMENSION_LIMIT like 1x343984x16x1, should split
                        // over C.
                        return DimArr{Dims4D::Act::W, Dims4D::Act::H, Dims4D::Act::C};
                    })
                    .Case<VPU::SoftMaxOp>([&](mlir::Operation* op) {
                        const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
                        auto tileDimOrder = getTileDimOrderND(outputType.getMemShape(), outputType.getDimsOrder());
                        auto softMaxOp = mlir::cast<VPU::SoftMaxOp>(op);
                        auto axis = softMaxOp.axisIndAttr().getValue().getSExtValue();
                        auto dimIt = std::find(tileDimOrder.begin(), tileDimOrder.end(), Dim(axis));
                        if (dimIt != tileDimOrder.end()) {
                            // Tiling along SoftMax operation axis is not supported
                            log.nest(2).trace("Removing axis dim {0} for SoftMax {1}", *dimIt, tileDimOrder);
                            tileDimOrder.erase(dimIt);
                        }
                        return tileDimOrder;
                    })
                    .Case<VPU::NCEPermuteOp>([&](mlir::Operation* op) {
                        log.nest(2).trace("Check tile Dim order for Op at {0}", op->getLoc());
                        return DimArr{Dims4D::Act::H, Dims4D::Act::W, Dims4D::Act::C};
                    })
                    .Default([&](mlir::Operation* op) -> DimArr {
                        log.nest(2).trace("Check tile Dim order for Op at {0}", op->getLoc());
                        const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();

                        return getTileDimOrderND(outputType.getMemShape(), outputType.getDimsOrder());
                    });

    // For prefetching mode, only weights can be pre-fetched to the parent op
    if (tilingMode == TilingMode::PREFETCHING) {
        tileDimOrder = SmallVector<Dim>({Dims4D::Act::C});
    }
    return tileDimOrder;
}

bool isMultiClusterCompatibleForTiling(mlir::Operation* op, const OutputTiling& tiles, Logger log) {
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(op);
    VPUX_THROW_WHEN(op == nullptr, "Operation '{0}' doesn't implement ClusteredOpInterface", op->getName());
    if (!clusteredOp->hasAttr(VPU::multiClusterStrategy)) {
        return true;
    }
    const auto outputType = clusteredOp->getResult(0).getType().cast<vpux::NDTypeInterface>();

    return llvm::all_of(tiles, [&](const TileInfo& outputTile) {
        const auto outputTileType = outputType.extractDenseTile(outputTile.offsets, outputTile.shape);
        return VPU::isStrategyCompatibleShape(clusteredOp, outputTileType.getShape(),
                                              clusteredOp.getMultiClusterStrategy().getValue(), log);
    });
}

//
// EltwiseOp
//

// Compute the maximum of tile number for each dimension to make sure:
// the tiling numbers are compatible for each dimension
// (Height) DPUs are fully used - at least one line for each DPU
// (Channel) No extra channel alignment - output channel for each cluster should be larger than minChannelSize
SmallVector<int64_t> vpux::getMaxNumTiles(mlir::Operation* op) {
    const auto outputShape = getShape(op->getResult(0));
    auto maxNumTiles = SmallVector<int64_t>(outputShape.begin(), outputShape.end());

    if (!mlir::isa<VPU::MemPermuteOp>(op)) {
        VPUX_THROW_UNLESS(outputShape.size() == 4, "Unsupported shape rank: {0}", outputShape.size());
    }

    int64_t minChannelSize = 1;
    if (auto channelsInfo = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op)) {
        VPUX_THROW_UNLESS(outputShape.size() == 4, "Unsupported shape rank: {0}", outputShape.size());
        minChannelSize = channelsInfo.getOutputChannelAlignment();
        const auto maxChannelTiles = outputShape[Dims4D::Act::C] / minChannelSize;
        maxNumTiles[Dims4D::Act::C.ind()] = maxChannelTiles;
    }

    if (op->hasAttr(VPU::multiClusterStrategy)) {
        VPUX_THROW_UNLESS(outputShape.size() == 4, "Unsupported shape rank: {0}", outputShape.size());

        auto strategy = op->getAttrOfType<VPU::MultiClusterStrategyAttr>(VPU::multiClusterStrategy).getValue();
        auto module = op->getParentOfType<mlir::ModuleOp>();
        auto nceOp = IE::getAvailableExecutor(module, VPU::ExecutorKind::NCE);
        if (strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
            strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped) {
            // To make sure the SOH MultiCluster strategy still compatible after tiling,
            // Each cluster should compute at least one output line
            // e.g., 4 cluster compilation, when tiling a layer with output height = 16
            // the tile number for height should be <= 16/4 = 4
            maxNumTiles[Dims4D::Act::H.ind()] = outputShape[Dims4D::Act::H] / nceOp.count();
        } else if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
            // To make sure the SOK MultiCluster strategy still compatible after tiling,
            // each cluster should compute at least minChannelSize(=16) output channels.
            // For SOK, we can use less than the specified number of clusters, to avoid the requirement to align output
            // tensor per cluster, the minimum of clusters that can be used is 2
            constexpr int64_t minNumClustersForSOK = 2;
            maxNumTiles[Dims4D::Act::C.ind()] = outputShape[Dims4D::Act::C] / (minChannelSize * minNumClustersForSOK);
        }
    }

    return maxNumTiles;
}

InputTiling vpux::backInferEltwiseTile(mlir::Operation* op, const vpux::TileInfo& outputTile) {
    SmallVector<TileInfo> inputTiles;
    for (auto& origInput : op->getOpOperands()) {
        const auto curShape = getShape(origInput.get());
        VPUX_THROW_UNLESS(curShape.size() == outputTile.shape.size(),
                          "Can't tile eltwise operation '{0}' at '{1}', which has operands with different rank",
                          op->getName(), op->getLoc());

        // Handle broadcasted inputs
        auto curTile = outputTile;
        for (auto ind : irange(curShape.size())) {
            const auto d = Dim(ind);
            if (curShape[d] == 1) {
                curTile.shape[d] = 1;
                curTile.offsets[d] = 0;
            }
        }

        inputTiles.push_back(curTile);
    }
    return TilingInfo{inputTiles};
}

// SWLayer

mlir::FailureOr<OutputTiling> vpux::getSWLayerTilingStrategyWithTileDimOrder(mlir::Operation* op, TilingMode tilingMode,
                                                                             DimArrRef tileDimOrder, Logger log,
                                                                             ArrayRef<int64_t> maxTilesPerDim) {
    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface", op->getName());
    auto tilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op);
    VPUX_THROW_WHEN(tilingBuilder == nullptr, "Operation '{0}' doesn't implement TilingBuilderOpInterface",
                    op->getName());
    VPUX_THROW_WHEN(tilingMode != TilingMode::ISOLATED,
                    "Only supporting isolated tiling for SW currently, for op {0} at '{1}'", op->getName(),
                    op->getLoc());

    const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputShape = outputType.getShape();

    Shape nTilesOnDim(outputShape.size(), 1);
    auto tileDimIter = tileDimOrder.begin();
    auto dimToTile = *tileDimIter;

    const auto isSupportedTileSize = [op, &tilingInfo, outputShape, log](ShapeRef nTilesOnDim,
                                                                         TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(op, nTilesOnDim, outputShape);
        if (mlir::failed(tiles)) {
            return false;
        }
        return tilingInfo.isSupportedTiling(tiles.value(), tilingMode, log);
    };

    SmallVector<int64_t> maxNumTiles(maxTilesPerDim.begin(), maxTilesPerDim.end());
    if (maxTilesPerDim.empty()) {
        maxNumTiles = tilingBuilder.getMaxNumTiles();
    }
    const auto isDimLeftToTile = [&](ShapeRef tileShape) -> bool {
        return tileShape[dimToTile] < maxNumTiles[dimToTile.ind()];
    };

    // Get an feasible isolated tiling strategy
    while (!isSupportedTileSize(nTilesOnDim, tilingMode)) {
        while ((tileDimIter < tileDimOrder.end()) && (!isDimLeftToTile(nTilesOnDim))) {
            dimToTile = *(++tileDimIter);
            if (tileDimIter == tileDimOrder.end()) {
                VPUX_THROW_WHEN(tilingMode == TilingMode::ISOLATED, "Failed to tile {0} at '{1}'", op->getName(),
                                op->getLoc());
            }
        }
        ++nTilesOnDim[dimToTile];
    }

    log.trace("Isolated tiling strategy: {0}", nTilesOnDim);
    return fillDividedTiles(op, nTilesOnDim, outputShape);
}

mlir::FailureOr<OutputTiling> vpux::getSWLayerTilingStrategy(mlir::Operation* op, TilingMode tilingMode, Logger log,
                                                             ArrayRef<int64_t> maxTilesPerDim) {
    const auto tileDimOrder = getTileDimOrder(op, tilingMode, log);
    log.nest(2).trace("Tile Dim order is {0}", tileDimOrder);
    return getSWLayerTilingStrategyWithTileDimOrder(op, tilingMode, tileDimOrder, log, maxTilesPerDim);
}

// Compute the maximum of tile number for each dimension with respect of not tile specified axes. No other restriction
// apply, rest of maximum tiles reflect output shape.
SmallVector<int64_t> vpux::getMaxNumTilesWithAxesExclusion(mlir::Operation* op, ArrayRef<int64_t> axes) {
    const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputShape = outputType.getShape();
    SmallVector<int64_t> maxNumTiles(outputShape.begin(), outputShape.end());
    const auto tileDimOrder = getTileDimOrderND(outputType.getMemShape(), outputType.getDimsOrder());
    for (const auto dimVal : tileDimOrder) {
        if (std::find(axes.begin(), axes.end(), dimVal.ind()) != axes.end()) {
            // not tile over axis, not alowed
            maxNumTiles[dimVal.ind()] = 1;
        }
    }
    return maxNumTiles;
}

// HWLayer

mlir::FailureOr<OutputTiling> vpux::getHWLayerTilingStrategyWithTileDimOrder(mlir::Operation* op, TilingMode tilingMode,
                                                                             DimArrRef tileDimOrder, Logger log) {
    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface", op->getName());
    auto tilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op);
    VPUX_THROW_WHEN(tilingBuilder == nullptr, "Operation '{0}' doesn't implement TilingBuilderOpInterface",
                    op->getName());

    const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputShape = outputType.getShape();

    VPUX_THROW_UNLESS(outputShape.size() == 4, "Unsupported operation '{0}' at '{1}', it has non 4D result",
                      op->getName(), op->getLoc());

    int64_t dimAlignment = 1;
    const auto dimToAlign = Dims4D::Act::C;
    if (auto channelsInfo = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op)) {
        dimAlignment = channelsInfo.getOutputChannelAlignment();
    }

    Shape nTilesOnDim(outputShape.size(), 1);

    auto tileDimIter = tileDimOrder.begin();
    auto dimToTile = *tileDimIter;

    const auto isSupportedTileSize = [op, &tilingInfo, outputShape, log](ShapeRef nTilesOnDim,
                                                                         TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(op, nTilesOnDim, outputShape);
        if (mlir::failed(tiles)) {
            return false;
        }
        return isMultiClusterCompatibleForTiling(op, tiles.value(), log) &&
               tilingInfo.isSupportedTiling(tiles.value(), tilingMode, log);
    };

    // Allow uneven tiling over OC, such as OC = 80 can be tiled as three tiles [32, 32, 16]
    const auto isSupportedAlignedDivision = [](int64_t dimSize, int64_t tiles, int64_t alignment) {
        auto base = vpux::divUp(dimSize, tiles);
        auto alignedBase = alignValUp(base, alignment);
        auto remainder = dimSize - alignedBase * (tiles - 1);
        return remainder > 0;
    };

    const auto& maxNumTiles = tilingBuilder.getMaxNumTiles();
    const auto isDimLeftToTile = [&](ShapeRef tileShape, Dim testTileDim) -> bool {
        return tileShape[testTileDim] < maxNumTiles[testTileDim.ind()];
    };

    // In case of pipelining, an isolated tiling strategy is first created
    // Then the tiling number would be increased to get a pipelining tiling strategy
    // If no feasible pipelining tiling could be found, fallback to isolated tiling strategy
    const auto tilingModeToCheck = tilingMode == TilingMode::PIPELINING ? TilingMode::ISOLATED : tilingMode;

    // Step1. get an feasible isolated tiling strategy or prefetching strategy
    while (!isSupportedTileSize(nTilesOnDim, tilingModeToCheck)) {
        while ((tileDimIter < tileDimOrder.end()) && (!isDimLeftToTile(nTilesOnDim, dimToTile))) {
            // If the current tiling dimension is not supported because of multicluster strategy
            // decrease the current dimension tiling size until the multicluster strategy is compatible again
            auto tiles = fillDividedTiles(op, nTilesOnDim, outputShape);
            while (nTilesOnDim[dimToTile] > 1) {
                if (!mlir::failed(tiles)) {
                    if (isMultiClusterCompatibleForTiling(op, tiles.value(), log)) {
                        break;
                    }
                }
                nTilesOnDim[dimToTile]--;
                // Skip the tiling numbers which are not aligned
                while ((dimToTile == dimToAlign && dimAlignment != 1 &&
                        !isSupportedAlignedDivision(outputShape[dimToTile], nTilesOnDim[dimToTile], dimAlignment)) &&
                       nTilesOnDim[dimToTile] > 1) {
                    nTilesOnDim[dimToTile]--;
                }
                tiles = fillDividedTiles(op, nTilesOnDim, outputShape);
            }
            dimToTile = *(++tileDimIter);
            if (tileDimIter == tileDimOrder.end()) {
                VPUX_THROW_WHEN(tilingModeToCheck == TilingMode::ISOLATED, "Failed to tile {0} at '{1}'", op->getName(),
                                op->getLoc());
                // If still not find the tiling strategy in PREFETCHING, fall back to neutral tiling
                auto neutralTiling = Shape(outputShape.size(), 1);
                log.nest(1).trace("Fallback to neutral tiling while attempting prefetching: {0}", neutralTiling);
                return fillDividedTiles(op, neutralTiling, outputShape);
            }
        }
        if (dimToTile == dimToAlign && dimAlignment != 1) {
            do {
                ++nTilesOnDim[dimToTile];
            } while (!isSupportedAlignedDivision(outputShape[dimToTile], nTilesOnDim[dimToTile], dimAlignment));
        } else {
            ++nTilesOnDim[dimToTile];
        }
    }

    // Step 1.1 reduce tiling scheme
    tileDimIter = tileDimOrder.begin();
    auto reduceDim = *tileDimIter;

    while (tileDimIter < tileDimOrder.end()) {
        while (isSupportedTileSize(nTilesOnDim, tilingModeToCheck) && nTilesOnDim[reduceDim] > 1) {
            --nTilesOnDim[reduceDim];

            // Skip the tiling numbers which are not aligned
            while ((reduceDim == dimToAlign && dimAlignment != 1 &&
                    !isSupportedAlignedDivision(outputShape[reduceDim], nTilesOnDim[reduceDim], dimAlignment)) &&
                   nTilesOnDim[reduceDim] > 1) {
                --nTilesOnDim[reduceDim];
            }
        }

        while (!isSupportedTileSize(nTilesOnDim, tilingModeToCheck)) {
            ++nTilesOnDim[reduceDim];

            // Skip the tiling numbers which are not aligned
            while ((reduceDim == dimToAlign && dimAlignment != 1 &&
                    !isSupportedAlignedDivision(outputShape[reduceDim], nTilesOnDim[reduceDim], dimAlignment))) {
                ++nTilesOnDim[reduceDim];
            }
        }

        reduceDim = *(++tileDimIter);
    }

    auto getDimsToTile = [](const Shape& nTilesOnDim) -> SmallVector<Dim> {
        SmallVector<Dim> res;
        for (unsigned i = 0; i < nTilesOnDim.size(); i++) {
            if (nTilesOnDim[Dim(i)] > 1)
                res.emplace_back(Dim(i));
        }
        return res;
    };
    auto dimsToTile = getDimsToTile(nTilesOnDim);
    auto isolatedTiles = fillDividedTiles(op, nTilesOnDim, outputShape);

    if (tilingMode != TilingMode::PIPELINING) {
        // return isolated tiling when getting nested tiles.
        log.nest(1).trace("Return isolated strategy: {0}", nTilesOnDim);
        return isolatedTiles;
    }

    if (dimsToTile.size() > 1) {
        log.nest(1).trace("Fallback to isolated strategy due to nested tiling: {0}", nTilesOnDim);
        return isolatedTiles;
    }

    // Step2. For pipelining, continue to increase on the dimension of isolated tiling
    //        or on the channel dimension in case of neutral tiling to cover cases with large constants
    const auto targetDim = dimsToTile.size() == 0 ? Dims4D::Act::C : dimsToTile[0];
    Shape prefetchableTilesOnDim = nTilesOnDim;
    log.trace("Attempting to generate tiling strategy for pipelining");
    while (!isSupportedTileSize(prefetchableTilesOnDim, TilingMode::PIPELINING)) {
        if (prefetchableTilesOnDim[targetDim] >= MAX_PREFETCH_TILING_TIME * nTilesOnDim[targetDim] ||
            !isDimLeftToTile(prefetchableTilesOnDim, targetDim)) {
            log.nest(3).trace("Fallback to isolated strategy: {0}", nTilesOnDim);
            return isolatedTiles;
        }
        if (targetDim == dimToAlign && dimAlignment != 1) {
            do {
                ++prefetchableTilesOnDim[targetDim];
                if (!isDimLeftToTile(prefetchableTilesOnDim, targetDim)) {
                    return isolatedTiles;
                }
            } while (!isSupportedAlignedDivision(outputShape[targetDim], prefetchableTilesOnDim[targetDim],
                                                 dimAlignment));
        } else {
            ++prefetchableTilesOnDim[dimToTile];
        }
    }

    log.trace("Pipelining tiling strategy: {0}", prefetchableTilesOnDim);
    return fillDividedTiles(op, prefetchableTilesOnDim, outputShape);
}

mlir::FailureOr<OutputTiling> vpux::getHWLayerTilingStrategy(mlir::Operation* op, TilingMode tilingMode, Logger log) {
    const auto tileDimOrder = getTileDimOrder(op, tilingMode, log);
    log.nest(2).trace("Tile Dim order is {0}", tileDimOrder);
    return getHWLayerTilingStrategyWithTileDimOrder(op, tilingMode, tileDimOrder, log);
}
