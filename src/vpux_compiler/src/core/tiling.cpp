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

PadsTileConfig vpux::backInferPadsTile(const Tile& outputTile, ShapeRef outShape, int64_t padLeft, int64_t padRight,
                                       int64_t padTop, int64_t padBottom) {
    SmallVector<int64_t> padsBegin(IE::Dims4D::Act::numSpatialDims);
    SmallVector<int64_t> padsEnd(IE::Dims4D::Act::numSpatialDims);
    SmallVector<int64_t> opPadsBegin = {padTop, padLeft};
    SmallVector<int64_t> opPadsEnd = {padBottom, padRight};

    for (auto ind : irange(IE::Dims4D::Act::numSpatialDims)) {
        const auto spatialDim = IE::Dims4D::Act::getSpatialDim(ind);

        const auto outSize = outputTile.shape[spatialDim];
        const auto outOffset = outputTile.offsets[spatialDim];

        const int64_t tilePadStart = outOffset == 0 ? opPadsBegin[ind] : 0;
        const int64_t tilePadEnd = (outOffset + outSize) == outShape[spatialDim] ? opPadsEnd[ind] : 0;

        padsBegin[ind] = tilePadStart;
        padsEnd[ind] = tilePadEnd;
    }

    return {padsBegin[1], padsEnd[1], padsBegin[0], padsEnd[0]};
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
    PlaneTile inputTile = {{0, 0}, {0, 0}};
    PadInfo pad = {0, 0, 0, 0};

    std::tie(inputTile.height, pad.top, pad.bottom) =
            inputForOutputDim(output.height, kernelY, strideY, {0, initialInputDims[IE::Dims4D::Act::H]},
                              initialPad.top, initialPad.bottom);

    std::tie(inputTile.width, pad.left, pad.right) =
            inputForOutputDim(output.width, kernelX, strideX, {0, initialInputDims[IE::Dims4D::Act::W]},
                              initialPad.left, initialPad.right);

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
    const auto origInputShape = getShape(origOp.input());
    const auto origFilterShape = getShape(origOp.filter());
    const auto origBiasShape = origOp.bias() != nullptr ? getShape(origOp.bias()) : ShapeRef();

    PlaneTile output;
    output.height.begin = outputTile.offsets[IE::Dims4D::Act::H];
    output.height.end = outputTile.offsets[IE::Dims4D::Act::H] + outputTile.shape[IE::Dims4D::Act::H];
    output.width.begin = outputTile.offsets[IE::Dims4D::Act::W];
    output.width.end = outputTile.offsets[IE::Dims4D::Act::W] + outputTile.shape[IE::Dims4D::Act::W];

    PadInfo initialPad;
    initialPad.top = origOp.pads_begin()[0].cast<mlir::IntegerAttr>().getInt();
    initialPad.bottom = origOp.pads_end()[0].cast<mlir::IntegerAttr>().getInt();
    initialPad.left = origOp.pads_begin()[1].cast<mlir::IntegerAttr>().getInt();
    initialPad.right = origOp.pads_end()[1].cast<mlir::IntegerAttr>().getInt();

    const auto solution = solutionForOutputTile(
            output, origFilterShape[IE::Dims4D::Filter::KX], origFilterShape[IE::Dims4D::Filter::KY],
            origOp.strides()[1].cast<mlir::IntegerAttr>().getInt(),
            origOp.strides()[0].cast<mlir::IntegerAttr>().getInt(), origInputShape, initialPad);

    Tile inputTile(origInputShape);
    Tile filterTile(origFilterShape);
    Tile biasTile(origBiasShape);

    inputTile.shape[IE::Dims4D::Act::N] = outputTile.shape[IE::Dims4D::Act::N];
    inputTile.offsets[IE::Dims4D::Act::N] = outputTile.offsets[IE::Dims4D::Act::N];

    inputTile.offsets[IE::Dims4D::Act::H] = solution.inputTile.height.begin;
    inputTile.shape[IE::Dims4D::Act::H] = solution.inputTile.height.length();

    inputTile.offsets[IE::Dims4D::Act::W] = solution.inputTile.width.begin;
    inputTile.shape[IE::Dims4D::Act::W] = solution.inputTile.width.length();

    filterTile.shape[IE::Dims4D::Filter::OC] = outputTile.shape[IE::Dims4D::Act::C];
    filterTile.offsets[IE::Dims4D::Filter::OC] = outputTile.offsets[IE::Dims4D::Act::C];

    if (!biasTile.shape.empty()) {
        biasTile.shape[IE::Dims4D::Act::C] = outputTile.shape[IE::Dims4D::Act::C];
        biasTile.offsets[IE::Dims4D::Act::C] = outputTile.offsets[IE::Dims4D::Act::C];
    }

    return {inputTile,
            filterTile,
            biasTile,
            {solution.inputPad.left, solution.inputPad.right, solution.inputPad.top, solution.inputPad.bottom}};
}

PoolTileConfig vpux::backInferPoolTile(IERT::MaxPoolOp origOp, const Tile& outputTile) {
    const auto origInputShape = getShape(origOp.input());

    PlaneTile output;
    output.height.begin = outputTile.offsets[IE::Dims4D::Act::H];
    output.height.end = outputTile.offsets[IE::Dims4D::Act::H] + outputTile.shape[IE::Dims4D::Act::H];
    output.width.begin = outputTile.offsets[IE::Dims4D::Act::W];
    output.width.end = outputTile.offsets[IE::Dims4D::Act::W] + outputTile.shape[IE::Dims4D::Act::W];

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

    inputTile.shape[IE::Dims4D::Act::N] = outputTile.shape[IE::Dims4D::Act::N];
    inputTile.offsets[IE::Dims4D::Act::N] = outputTile.offsets[IE::Dims4D::Act::N];

    inputTile.shape[IE::Dims4D::Act::C] = outputTile.shape[IE::Dims4D::Act::C];
    inputTile.offsets[IE::Dims4D::Act::C] = outputTile.offsets[IE::Dims4D::Act::C];

    inputTile.offsets[IE::Dims4D::Act::H] = solution.inputTile.height.begin;
    inputTile.shape[IE::Dims4D::Act::H] = solution.inputTile.height.length();

    inputTile.offsets[IE::Dims4D::Act::W] = solution.inputTile.width.begin;
    inputTile.shape[IE::Dims4D::Act::W] = solution.inputTile.width.length();

    return {inputTile,
            {solution.inputPad.left, solution.inputPad.right, solution.inputPad.top, solution.inputPad.bottom}};
}

EltwiseTileConfig vpux::backInferEltwiseAddTile(const Tile& outputTile) {
    return EltwiseTileConfig{outputTile};
}

ConvTileConfig vpux::backInferGroupConvTile(IERT::GroupConvolutionOp origOp, const Tile& outputTile) {
    const auto origInputShape = getShape(origOp.input());
    const auto origFilterShape = getShape(origOp.filter());
    const auto origBiasShape = origOp.bias() != nullptr ? getShape(origOp.bias()) : ShapeRef();

    PlaneTile output;
    output.height.begin = outputTile.offsets[IE::Dims4D::Act::H];
    output.height.end = outputTile.offsets[IE::Dims4D::Act::H] + outputTile.shape[IE::Dims4D::Act::H];
    output.width.begin = outputTile.offsets[IE::Dims4D::Act::W];
    output.width.end = outputTile.offsets[IE::Dims4D::Act::W] + outputTile.shape[IE::Dims4D::Act::W];

    PadInfo initialPad;
    initialPad.top = origOp.pads_begin()[0].cast<mlir::IntegerAttr>().getInt();
    initialPad.bottom = origOp.pads_end()[0].cast<mlir::IntegerAttr>().getInt();
    initialPad.left = origOp.pads_begin()[1].cast<mlir::IntegerAttr>().getInt();
    initialPad.right = origOp.pads_end()[1].cast<mlir::IntegerAttr>().getInt();

    const auto solution = solutionForOutputTile(
            output, origFilterShape[IE::Dims4D::Filter::KX], origFilterShape[IE::Dims4D::Filter::KY],
            origOp.strides()[1].cast<mlir::IntegerAttr>().getInt(),
            origOp.strides()[0].cast<mlir::IntegerAttr>().getInt(), origInputShape, initialPad);

    Tile inputTile(origInputShape);
    Tile filterTile(origFilterShape);
    Tile biasTile(origBiasShape);

    inputTile.shape[IE::Dims4D::Act::C] = outputTile.shape[IE::Dims4D::Act::C];
    inputTile.offsets[IE::Dims4D::Act::C] = outputTile.offsets[IE::Dims4D::Act::C];

    inputTile.offsets[IE::Dims4D::Act::H] = solution.inputTile.height.begin;
    inputTile.shape[IE::Dims4D::Act::H] = solution.inputTile.height.length();

    inputTile.offsets[IE::Dims4D::Act::W] = solution.inputTile.width.begin;
    inputTile.shape[IE::Dims4D::Act::W] = solution.inputTile.width.length();

    filterTile.shape[IE::Dims4D::Filter::OC] = outputTile.shape[IE::Dims4D::Act::C];
    filterTile.offsets[IE::Dims4D::Filter::OC] = outputTile.offsets[IE::Dims4D::Act::C];

    if (!biasTile.shape.empty()) {
        biasTile.shape[IE::Dims4D::Act::C] = outputTile.shape[IE::Dims4D::Act::C];
        biasTile.offsets[IE::Dims4D::Act::C] = outputTile.offsets[IE::Dims4D::Act::C];
    }

    return {inputTile,
            filterTile,
            biasTile,
            {solution.inputPad.left, solution.inputPad.right, solution.inputPad.top, solution.inputPad.bottom}};
}
