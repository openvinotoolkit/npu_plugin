//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/utils/core/numeric.hpp"

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"

#include <mlir/Parser.h>

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
void divideTiles(OutputTiling& dividedTiles, ShapeRef divisors, ShapeRef shape, ArrayRef<int64_t> alignment,
                 size_t dimensionIndex, vpux::TileInfo& ongoingTile) {
    const auto dimension = Dim(dimensionIndex);

    auto dividedShapes = std::vector<int64_t>(divisors[dimension]);
    auto dividedOffsets = std::vector<int64_t>(divisors[dimension]);
    const auto dividedAxis = divisors[dimension];

    const auto shapeVal = shape[dimension];
    const auto divisorVal = divisors[dimension];
    const auto alignmentVal = alignment[dimensionIndex];

    // Compute tiling scheme per current dimension
    size_t tileSize, remainderTileSize;
    size_t tileSizeInterval, remainderTileSizeInterval;

    if (alignmentVal > 1) {
        // Whenever there is alignment, all N-1 tiles need to be multiple
        // of said align value.
        // The remainder shape is admitted to not be a mutiple of align value,
        // since this code is tasked to simply tile the original shape, not also align it.
        tileSize = alignVal(divUp(shapeVal, divisorVal), alignmentVal);
        remainderTileSize = shapeVal - tileSize * (divisorVal - 1);

        VPUX_THROW_UNLESS(remainderTileSize > 0,
                          "Incompatible tiling of shape {0} times {1} and alignment of {2}. Remainder is {3}", shapeVal,
                          divisorVal, alignmentVal, remainderTileSize);

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
    for (const auto shapeOffsetPair : zip(dividedShapes, dividedOffsets)) {
        ongoingTile.shape[dimension] = std::get<0>(shapeOffsetPair);
        ongoingTile.offsets[dimension] = std::get<1>(shapeOffsetPair);
        ongoingTile.axis[dimension] = dividedAxis;

        // Full dividedTile is created so need to register the solution
        if (dimensionIndex == (divisors.size() - 1)) {
            dividedTiles.push_back(ongoingTile);
        } else {
            divideTiles(dividedTiles, divisors, shape, alignment, dimensionIndex + 1, ongoingTile);
        }
    }
}

OutputTiling vpux::fillDividedTiles(ShapeRef divisors, ShapeRef shape, Optional<ArrayRef<int64_t>> alignment) {
    OutputTiling dividedTiles;

    auto ongoingTile = vpux::TileInfo(divisors.size());

    auto alignmentShape = makeArrayRef(SmallVector<int64_t>(shape.size(), 1));
    if (alignment.hasValue()) {
        alignmentShape = alignment.getValue();
    }

    divideTiles(dividedTiles, divisors, shape, alignmentShape, 0, ongoingTile);
    return dividedTiles;
}

OutputTiling vpux::fillDividedTiles(mlir::Operation* op, ShapeRef divisors, ShapeRef shape) {
    OutputTiling dividedTiles;

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

    filterTile.shape[Dims4D::Filter::OC] = outputTile.shape[Dims4D::Act::C];
    filterTile.offsets[Dims4D::Filter::OC] = outputTile.offsets[Dims4D::Act::C];

    if (!biasTile.shape.empty()) {
        biasTile.shape[Dims4D::Act::C] = outputTile.shape[Dims4D::Act::C];
        biasTile.offsets[Dims4D::Act::C] = outputTile.offsets[Dims4D::Act::C];
    }

    return TilingInfo{{inputTile, filterTile, biasTile}, solution.inputPad};
}

InputTiling vpux::backInferGroupConvTile(const TileInfo& outputTile, ShapeRef origInputShape, ShapeRef origFilterShape,
                                         ShapeRef origBiasShape, mlir::ArrayAttr strides, const PadInfo& origPadding) {
    auto res = backInferConvTile(outputTile, origInputShape, origFilterShape, origBiasShape, strides, origPadding);

    auto& inputTiles = res.tiles;
    const auto inputTileIdx = 0;
    inputTiles[inputTileIdx].shape[Dims4D::Act::C] = outputTile.shape[Dims4D::Act::C];
    inputTiles[inputTileIdx].offsets[Dims4D::Act::C] = outputTile.offsets[Dims4D::Act::C];

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

    return TilingInfo{{inputTile}, solution.inputPad};
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

SmallVector<int64_t> vpux::alignShape(ArrayRef<int64_t> shape, Optional<ArrayRef<int64_t>> alignment) {
    auto alignedShape = to_small_vector(shape);
    if (!alignment.hasValue()) {
        return alignedShape;
    }
    std::transform(shape.begin(), shape.end(), alignment.getValue().begin(), alignedShape.begin(), alignVal<int64_t>);
    return alignedShape;
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
            const auto strideRatio = currStride / prevStride;

            const auto prevAdaptedStride = adaptedMemStrides[i + 1].count();

            auto adaptedStride = strideRatio * prevAdaptedStride;
            // If there is a change between the original and the new shape,
            // we favor striding with the new shape size instead of the previous stride ratio.
            if (memShape[MemDim(i + 1)] != adaptedMemShape[MemDim(i + 1)]) {
                // In the case of multiclustering, all such scenarios like H|K cluster tiling
                // with H|K prefetch tiling should be concatenated in DDR as simple tensors.
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

//
// EltwiseOp
//

// Compute the maximum of tile number for each dimension to make sure:
// the tiling numbers are compatible for each dimension
// (Height) DPUs are fully used - at least one line for each DPU
// (Channel) No extra channel alignment - output channel for each cluster should be larger than minChannelSize
SmallVector<int64_t> vpux::getMaxNumTiles(mlir::Operation* op) {
    const auto& outputShape = getShape(op->getResult(0));

    VPUX_THROW_UNLESS(outputShape.size() == 4, "Unsupported shape rank: {0}", outputShape.size());

    int64_t minChannelSize = 1;
    if (auto channelsInfo = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op)) {
        minChannelSize = channelsInfo.getOutputChannelAlignment();
    }

    const auto maxChannelTiles = outputShape[Dims4D::Act::C] / minChannelSize;

    SmallVector<int64_t> maxNumTiles(outputShape.begin(), outputShape.end());
    maxNumTiles[Dims4D::Act::C.ind()] = maxChannelTiles;

    if (op->hasAttr(VPU::multiClusterStrategy)) {
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
