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

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Parser.h>

using namespace vpux;

//
// TileInfo
//

OutputTiling vpux::fillDividedTiles(ShapeRef divisors, ShapeRef orig, mlir::IntegerAttr clusterIdAttr) {
    OutputTiling dividedTiles(divisors.totalSize(), TileInfo(divisors.size()));

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
                dividedTiles[i].axis[dim] = 1;
            }

            continue;
        }

        const auto tileSize = origSize / divisor;
        VPUX_THROW_UNLESS(tileSize > 0, "Got too many tiles request '{0}'", divisor);

        int64_t offset = 0;
        for (int64_t i : irange(dividedTiles.size())) {
            const bool remainderTile = !(((i / repeatCtr) + 1) % (divisor));

            const bool firstClusterFirstTile = clusterIdAttr && (clusterIdAttr.getValue().getSExtValue() == 0);

            if (firstClusterFirstTile) {
                if (i == 0) {
                    dividedTiles[i].shape[dim] = origSize - (tileSize * (divisor - 1));
                } else {
                    dividedTiles[i].shape[dim] = tileSize;
                }
            } else {
                if (remainderTile) {
                    dividedTiles[i].shape[dim] = origSize - (tileSize * (divisor - 1));
                } else {
                    dividedTiles[i].shape[dim] = tileSize;
                }
            }

            dividedTiles[i].offsets[dim] = offset;
            dividedTiles[i].axis[dim] = divisors[dim];

            const bool incrementOffset = !((i + 1) % repeatCtr);
            if (incrementOffset) {
                offset += dividedTiles[i].shape[dim];
            }

            const bool resetOffset = (remainderTile && incrementOffset);
            if (resetOffset) {
                offset = 0;
            }
        }

        repeatCtr *= divisor;
    }

    return dividedTiles;
}

//
// PadInfo
//

PadInfo vpux::backInferPadsTile(const TileInfo& outputTile, ShapeRef outShape, const PadInfo& origPads) {
    const std::array<int64_t, 2> origPadsBegin = {origPads.top, origPads.left};
    const std::array<int64_t, 2> origPadsEnd = {origPads.bottom, origPads.right};

    SmallVector<int64_t> tilePadsBegin(Dims4D::Act::numSpatialDims);
    SmallVector<int64_t> tilePadsEnd(Dims4D::Act::numSpatialDims);

    for (auto ind : irange(Dims4D::Act::numSpatialDims)) {
        const auto spatialDim = Dims4D::Act::getSpatialDim(ind);

        const auto outSize = outputTile.shape[spatialDim];
        const auto outOffset = outputTile.offsets[spatialDim];

        tilePadsBegin[ind] = outOffset == 0 ? origPadsBegin[ind] : 0;
        tilePadsEnd[ind] = (outOffset + outSize) == outShape[spatialDim] ? origPadsEnd[ind] : 0;
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

    // Checks if rhs located completly in this.
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
// Padding should be applied to the input tile. It could be assymetric, or doesn't meet HW requirements in terms of its
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
