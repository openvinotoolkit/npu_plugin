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

using namespace vpux;

SmallVector<Tile> vpux::fillDividedTiles(ShapeRef divisors, ShapeRef orig) {
    SmallVector<Tile> dividedTiles(divisors.totalSize(), Tile(divisors.size()));

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
            }

            continue;
        }

        const auto tileSize = origSize / divisor;
        VPUX_THROW_UNLESS(tileSize > 0, "Got too many tiles request '{0}'", divisor);

        int64_t offset = 0;
        for (int64_t i : irange(dividedTiles.size())) {
            const bool remainderTile = !(((i / repeatCtr) + 1) % (divisor));

            if (remainderTile) {
                dividedTiles[i].shape[dim] = origSize - (tileSize * (divisor - 1));
            } else {
                dividedTiles[i].shape[dim] = tileSize;
            }

            dividedTiles[i].offsets[dim] = offset;

            const bool incrementOffset = !((i + 1) % repeatCtr);
            if (incrementOffset) {
                offset += tileSize;
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

PadsTileConfig vpux::backInferPadsTile(const Tile& outputTile, ShapeRef outShape, ArrayRef<int64_t> opPadsBegin,
                                       ArrayRef<int64_t> opPadsEnd) {
    SmallVector<int64_t> padsBegin(IERT::ConvolutionOp::filter_spatial_dims());
    SmallVector<int64_t> padsEnd(IERT::ConvolutionOp::filter_spatial_dims());

    for (auto spatialDim : irange(IERT::ConvolutionOp::filter_spatial_dims())) {
        const auto act_spatial_dim = IERT::ConvolutionOp::act_spatial_dim(spatialDim);

        const auto outSize = outputTile.shape[act_spatial_dim];
        const auto outOffset = outputTile.offsets[act_spatial_dim];

        const int64_t tilePadStart = outOffset == 0 ? opPadsBegin[spatialDim] : 0;
        const int64_t tilePadEnd = (outOffset + outSize) == outShape[act_spatial_dim] ? opPadsEnd[spatialDim] : 0;

        padsBegin[spatialDim] = tilePadStart;
        padsEnd[spatialDim] = tilePadEnd;
    }

    return {padsBegin, padsEnd};
}

//
// Tiling utilities
//

namespace {

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

struct PadInfo final {
    int64_t left = 0;
    int64_t right = 0;
    int64_t top = 0;
    int64_t bottom = 0;

    PadInfo() = default;
    PadInfo(int64_t left, int64_t right, int64_t top, int64_t bottom)
            : left(left), right(right), top(top), bottom(bottom) {
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
        printTo(stream, "PadInfo [left = {0}, right = {1}, top = {1}, bottom = {1}]", left, right, top, bottom);
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

std::tuple<DimRange, int64_t, int64_t> inputForOutputDim(const DimRange& output, int64_t kernel, int64_t stride,
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

// Return input tile and padding required to calculate the output tile.
// Padding should be applied to the input tile. It could be assymetric, or doesn't meet HW requirements in terms of its
// size.
// * initialInputDims - Dims of the whole input tensor (not of specific tile).
// * initialPad - padding which should be applied to the whole input tensor (not to specific tile).
std::tuple<PlaneTile, PadInfo> inputForOutputTile(const PlaneTile& output, int64_t kernelX, int64_t kernelY,
                                                  int64_t strideX, int64_t strideY, ShapeRef initialInputDims,
                                                  const PadInfo& initialPad) {
    const auto act_height_dim = IERT::ConvolutionOp::act_height_dim();
    const auto act_width_dim = IERT::ConvolutionOp::act_width_dim();

    PlaneTile inputTile = {{0, 0}, {0, 0}};
    PadInfo pad = {0, 0, 0, 0};

    std::tie(inputTile.height, pad.top, pad.bottom) = inputForOutputDim(
            output.height, kernelY, strideY, {0, initialInputDims[act_height_dim]}, initialPad.top, initialPad.bottom);

    std::tie(inputTile.width, pad.left, pad.right) = inputForOutputDim(
            output.width, kernelX, strideX, {0, initialInputDims[act_width_dim]}, initialPad.left, initialPad.right);

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

ConvTileConfig vpux::backInferConvTile(IERT::ConvolutionOp origOp, const Tile& outputTile) {
    const auto act_batch_dim = IERT::ConvolutionOp::act_batch_dim();
    const auto act_channel_dim = IERT::ConvolutionOp::act_channel_dim();
    const auto act_height_dim = IERT::ConvolutionOp::act_height_dim();
    const auto act_width_dim = IERT::ConvolutionOp::act_width_dim();

    const auto filter_out_channel_dim = IERT::ConvolutionOp::filter_out_channel_dim();
    const auto filter_spatial_height_dim = IERT::ConvolutionOp::filter_spatial_height_dim();
    const auto filter_spatial_width_dim = IERT::ConvolutionOp::filter_spatial_width_dim();

    const auto origInputShape = getShape(origOp.input());
    const auto origFilterShape = getShape(origOp.filter());
    const auto origBiasShape = origOp.bias() != nullptr ? getShape(origOp.bias()) : ShapeRef();

    PlaneTile output;
    output.height.begin = outputTile.offsets[act_height_dim];
    output.height.end = outputTile.offsets[act_height_dim] + outputTile.shape[act_height_dim];
    output.width.begin = outputTile.offsets[act_width_dim];
    output.width.end = outputTile.offsets[act_width_dim] + outputTile.shape[act_width_dim];

    PadInfo initialPad;
    initialPad.top = origOp.pads_begin()[0].cast<mlir::IntegerAttr>().getInt();
    initialPad.bottom = origOp.pads_end()[0].cast<mlir::IntegerAttr>().getInt();
    initialPad.left = origOp.pads_begin()[1].cast<mlir::IntegerAttr>().getInt();
    initialPad.right = origOp.pads_end()[1].cast<mlir::IntegerAttr>().getInt();

    const auto solution = solutionForOutputTile(
            output, origFilterShape[filter_spatial_width_dim], origFilterShape[filter_spatial_height_dim],
            origOp.strides()[1].cast<mlir::IntegerAttr>().getInt(),
            origOp.strides()[0].cast<mlir::IntegerAttr>().getInt(), origInputShape, initialPad);

    Tile inputTile(origInputShape);
    Tile filterTile(origFilterShape);
    Tile biasTile(origBiasShape);

    inputTile.shape[act_batch_dim] = outputTile.shape[act_batch_dim];
    inputTile.offsets[act_batch_dim] = outputTile.offsets[act_batch_dim];

    inputTile.offsets[act_height_dim] = solution.inputTile.height.begin;
    inputTile.shape[act_height_dim] = solution.inputTile.height.length();

    inputTile.offsets[act_width_dim] = solution.inputTile.width.begin;
    inputTile.shape[act_width_dim] = solution.inputTile.width.length();

    filterTile.shape[filter_out_channel_dim] = outputTile.shape[act_channel_dim];
    filterTile.offsets[filter_out_channel_dim] = outputTile.offsets[act_channel_dim];

    if (!biasTile.shape.empty()) {
        biasTile.shape[act_channel_dim] = outputTile.shape[act_channel_dim];
        biasTile.offsets[act_channel_dim] = outputTile.offsets[act_channel_dim];
    }

    SmallVector<int64_t> padsBegin(IERT::ConvolutionOp::filter_spatial_dims());
    SmallVector<int64_t> padsEnd(IERT::ConvolutionOp::filter_spatial_dims());

    padsBegin[0] = solution.inputPad.top;
    padsEnd[0] = solution.inputPad.bottom;
    padsBegin[1] = solution.inputPad.left;
    padsEnd[1] = solution.inputPad.right;

    return {inputTile, filterTile, biasTile, {padsBegin, padsEnd}};
}

PoolTileConfig vpux::backInferPoolTile(IERT::MaxPoolOp origOp, const Tile& outputTile) {
    const auto act_batch_dim = IERT::ConvolutionOp::act_batch_dim();
    const auto act_channel_dim = IERT::ConvolutionOp::act_channel_dim();
    const auto act_height_dim = IERT::ConvolutionOp::act_height_dim();
    const auto act_width_dim = IERT::ConvolutionOp::act_width_dim();

    const auto origInputShape = getShape(origOp.input());

    PlaneTile output;
    output.height.begin = outputTile.offsets[act_height_dim];
    output.height.end = outputTile.offsets[act_height_dim] + outputTile.shape[act_height_dim];
    output.width.begin = outputTile.offsets[act_width_dim];
    output.width.end = outputTile.offsets[act_width_dim] + outputTile.shape[act_width_dim];

    PadInfo initialPad;
    initialPad.top = origOp.pads_begin()[0].cast<mlir::IntegerAttr>().getInt();
    initialPad.bottom = origOp.pads_end()[0].cast<mlir::IntegerAttr>().getInt();
    initialPad.left = origOp.pads_begin()[1].cast<mlir::IntegerAttr>().getInt();
    initialPad.right = origOp.pads_end()[1].cast<mlir::IntegerAttr>().getInt();

    const auto solution =
            solutionForOutputTile(output, origOp.kernel_size()[1].cast<mlir::IntegerAttr>().getInt(),
                                  origOp.kernel_size()[0].cast<mlir::IntegerAttr>().getInt(),
                                  origOp.strides()[1].cast<mlir::IntegerAttr>().getInt(),
                                  origOp.strides()[0].cast<mlir::IntegerAttr>().getInt(), origInputShape, initialPad);

    Tile inputTile(origInputShape);

    inputTile.shape[act_batch_dim] = outputTile.shape[act_batch_dim];
    inputTile.offsets[act_batch_dim] = outputTile.offsets[act_batch_dim];

    inputTile.shape[act_channel_dim] = outputTile.shape[act_channel_dim];
    inputTile.offsets[act_channel_dim] = outputTile.offsets[act_channel_dim];

    inputTile.offsets[act_height_dim] = solution.inputTile.height.begin;
    inputTile.shape[act_height_dim] = solution.inputTile.height.length();

    inputTile.offsets[act_width_dim] = solution.inputTile.width.begin;
    inputTile.shape[act_width_dim] = solution.inputTile.width.length();

    SmallVector<int64_t> padsBegin(IERT::MaxPoolOp::act_spatial_dims());
    SmallVector<int64_t> padsEnd(IERT::MaxPoolOp::act_spatial_dims());

    padsBegin[0] = solution.inputPad.top;
    padsEnd[0] = solution.inputPad.bottom;
    padsBegin[1] = solution.inputPad.left;
    padsEnd[1] = solution.inputPad.right;

    return {inputTile, {padsBegin, padsEnd}};
}

ConvTileConfig vpux::backInferGroupConvTile(IERT::GroupConvolutionOp origOp, const Tile& outputTile) {
    const auto actChannelDim = IERT::ConvolutionOp::act_channel_dim();
    const auto actHeightDim = IERT::ConvolutionOp::act_height_dim();
    const auto actWidthDim = IERT::ConvolutionOp::act_width_dim();

    const auto filterOutChannelDim = IERT::ConvolutionOp::filter_out_channel_dim();
    const auto filterHeightDim = IERT::ConvolutionOp::filter_spatial_height_dim();
    const auto filterWidthDim = IERT::ConvolutionOp::filter_spatial_width_dim();

    const auto origInputShape = getShape(origOp.input());
    const auto origFilterShape = getShape(origOp.filter());
    const auto origBiasShape = origOp.bias() != nullptr ? getShape(origOp.bias()) : ShapeRef();

    PlaneTile output;
    output.height.begin = outputTile.offsets[actHeightDim];
    output.height.end = outputTile.offsets[actHeightDim] + outputTile.shape[actHeightDim];
    output.width.begin = outputTile.offsets[actWidthDim];
    output.width.end = outputTile.offsets[actWidthDim] + outputTile.shape[actWidthDim];

    PadInfo initialPad;
    initialPad.top = origOp.pads_begin()[0].cast<mlir::IntegerAttr>().getInt();
    initialPad.bottom = origOp.pads_end()[0].cast<mlir::IntegerAttr>().getInt();
    initialPad.left = origOp.pads_begin()[1].cast<mlir::IntegerAttr>().getInt();
    initialPad.right = origOp.pads_end()[1].cast<mlir::IntegerAttr>().getInt();

    const auto solution =
            solutionForOutputTile(output, origFilterShape[filterWidthDim], origFilterShape[filterHeightDim],
                                  origOp.strides()[1].cast<mlir::IntegerAttr>().getInt(),
                                  origOp.strides()[0].cast<mlir::IntegerAttr>().getInt(), origInputShape, initialPad);

    Tile inputTile(origInputShape);
    Tile filterTile(origFilterShape);
    Tile biasTile(origBiasShape);

    inputTile.shape[actChannelDim] = outputTile.shape[actChannelDim];
    inputTile.offsets[actChannelDim] = outputTile.offsets[actChannelDim];

    inputTile.offsets[actHeightDim] = solution.inputTile.height.begin;
    inputTile.shape[actHeightDim] = solution.inputTile.height.length();

    inputTile.offsets[actWidthDim] = solution.inputTile.width.begin;
    inputTile.shape[actWidthDim] = solution.inputTile.width.length();

    filterTile.shape[filterOutChannelDim] = outputTile.shape[actChannelDim];
    filterTile.offsets[filterOutChannelDim] = outputTile.offsets[actChannelDim];

    if (!biasTile.shape.empty()) {
        biasTile.shape[actChannelDim] = outputTile.shape[actChannelDim];
        biasTile.offsets[actChannelDim] = outputTile.offsets[actChannelDim];
    }

    SmallVector<int64_t> padsBegin(IERT::ConvolutionOp::filter_spatial_dims());
    SmallVector<int64_t> padsEnd(IERT::ConvolutionOp::filter_spatial_dims());

    padsBegin[0] = solution.inputPad.top;
    padsEnd[0] = solution.inputPad.bottom;
    padsBegin[1] = solution.inputPad.left;
    padsEnd[1] = solution.inputPad.right;

    return {inputTile, filterTile, biasTile, {padsBegin, padsEnd}};
}
