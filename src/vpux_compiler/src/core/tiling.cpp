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

ConvTileConfig vpux::backInferConvTile(IERT::ConvolutionOp origOp, const Tile& outputTile) {
    const auto origInputShape = getShape(origOp.input());
    const auto origFilterShape = getShape(origOp.filter());
    const auto origBiasShape = origOp.bias() != nullptr ? getShape(origOp.bias()) : ShapeRef();
    const auto origOutputShape = getShape(origOp.output());

    Tile inputTile(origInputShape);
    Tile filterTile(origFilterShape);
    Tile biasTile(origBiasShape);

    SmallVector<int64_t> padsBegin(IERT::ConvolutionOp::filter_spatial_dims());
    SmallVector<int64_t> padsEnd(IERT::ConvolutionOp::filter_spatial_dims());

    for (auto spatialDim : irange(IERT::ConvolutionOp::filter_spatial_dims())) {
        const auto act_spatial_dim = IERT::ConvolutionOp::act_spatial_dim(spatialDim);
        const auto filter_spatial_dim = IERT::ConvolutionOp::filter_spatial_dim(spatialDim);

        const auto outSize = outputTile.shape[act_spatial_dim];
        const auto outOffset = outputTile.offsets[act_spatial_dim];

        const auto opPadBegin = origOp.pads_begin()[spatialDim].cast<mlir::IntegerAttr>().getInt();
        const auto opPadEnd = origOp.pads_end()[spatialDim].cast<mlir::IntegerAttr>().getInt();

        const auto kSize = origFilterShape[filter_spatial_dim];
        const auto kStride = origOp.strides()[spatialDim].cast<mlir::IntegerAttr>().getInt();

        const int64_t tilePadStart = outOffset == 0 ? opPadBegin : 0;
        const int64_t tilePadEnd = (outOffset + outSize) == origOutputShape[act_spatial_dim] ? opPadEnd : 0;

        const int64_t inputSize = ((outSize - 1) * kStride) - tilePadStart - tilePadEnd + kSize;
        const int64_t inputOffset = outOffset != 0 ? outOffset * kStride - opPadBegin - ((kSize - 1) / 2) : 0;

        inputTile.shape[act_spatial_dim] = inputSize;
        inputTile.offsets[act_spatial_dim] = inputOffset;

        padsBegin[spatialDim] = tilePadStart;
        padsEnd[spatialDim] = tilePadEnd;
    }

    const auto act_batch_dim = IERT::ConvolutionOp::act_batch_dim();
    const auto act_channel_dim = IERT::ConvolutionOp::act_channel_dim();
    const auto filter_out_channel_dim = IERT::ConvolutionOp::filter_out_channel_dim();

    // will not tile on InputChannels
    inputTile.shape[act_channel_dim] = origInputShape[act_channel_dim];
    inputTile.offsets[act_channel_dim] = 0;

    filterTile.shape[filter_out_channel_dim] = outputTile.shape[act_channel_dim];
    filterTile.offsets[filter_out_channel_dim] = outputTile.offsets[act_channel_dim];

    if (!biasTile.shape.empty()) {
        biasTile.shape[act_channel_dim] = outputTile.shape[act_channel_dim];
        biasTile.offsets[act_channel_dim] = outputTile.offsets[act_channel_dim];
    }

    // do the batch dim
    inputTile.shape[act_batch_dim] = outputTile.shape[act_batch_dim];
    inputTile.offsets[act_batch_dim] = outputTile.offsets[act_batch_dim];

    return {inputTile, filterTile, biasTile, {padsBegin, padsEnd}};
};
