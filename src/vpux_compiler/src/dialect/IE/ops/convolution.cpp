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

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/IR/PatternMatch.h>

#include <ngraph/coordinate.hpp>
#include <ngraph/op/max_pool.hpp>
#include <ngraph/validation_util.hpp>

using namespace vpux;

//
// FuseConvAndBias
//

namespace {

class FuseConvAndBias final : public mlir::OpRewritePattern<IE::ScaleShiftOp> {
public:
    using mlir::OpRewritePattern<IE::ScaleShiftOp>::OpRewritePattern;

    void initialize() {
        setDebugName("FuseConvAndBias");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ScaleShiftOp biasOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseConvAndBias::matchAndRewrite(IE::ScaleShiftOp biasOp, mlir::PatternRewriter& rewriter) const {
    if (biasOp.weights() != nullptr) {
        return matchFailed(rewriter, biasOp, "ScaleShift has scales operand");
    }
    if (!biasOp.input().hasOneUse()) {
        return matchFailed(rewriter, biasOp, "ScaleShift is not the only user of its operand");
    }

    auto* convOp = biasOp.input().getDefiningOp();
    if (convOp == nullptr || !mlir::isa<IE::ConvolutionOp, IE::GroupConvolutionOp>(convOp)) {
        return matchFailed(rewriter, biasOp, "ScaleShift producer is not a Convolution layer");
    }
    if (convOp->getNumOperands() != 2) {
        return matchFailed(rewriter, biasOp, "ScaleShift producer already has fused biases");
    }

    const auto convOutShape = getShape(convOp->getOpResult(0));
    const auto biasShape = getShape(biasOp.biases());

    if (biasShape.size() != 4) {
        return matchFailed(rewriter, biasOp, "ScaleShift 'shift' operand shape doesn't match bias restrictions");
    }
    if (biasShape[Dims4D::Act::N] != 1) {
        return matchFailed(rewriter, biasOp, "ScaleShift 'shift' operand shape doesn't match bias restrictions");
    }
    if (biasShape[Dims4D::Act::C] != convOutShape[Dims4D::Act::C]) {
        return matchFailed(rewriter, biasOp, "ScaleShift 'shift' operand shape doesn't match bias restrictions");
    }
    if (biasShape[Dims4D::Act::H] != 1) {
        return matchFailed(rewriter, biasOp, "ScaleShift 'shift' operand shape doesn't match bias restrictions");
    }
    if (biasShape[Dims4D::Act::W] != 1) {
        return matchFailed(rewriter, biasOp, "ScaleShift 'shift' operand shape doesn't match bias restrictions");
    }

    auto mainOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(convOp);
    if (mainOp == nullptr) {
        return matchFailed(rewriter, biasOp, "Convolution implementation doesn't support PostOp fusing");
    }
    if (!mainOp.isSupportedPostOp(biasOp)) {
        return matchFailed(rewriter, biasOp, "Convolution implementation doesn't support Bias fusing");
    }

    auto* newConv = rewriter.clone(*convOp);
    newConv->insertOperands(newConv->getNumOperands(), biasOp.biases());

    rewriter.replaceOp(biasOp, newConv->getOpResults());

    return mlir::success();
}

}  // namespace

//
// Convolution
//

mlir::LogicalResult vpux::IE::ConvolutionOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::ConvolutionOpAdaptor conv(operands, attrs);
    if (mlir::failed(conv.verify(loc))) {
        return mlir::failure();
    }

    const auto inShape = conv.input().getType().cast<mlir::ShapedType>().getShape();
    const auto inType = conv.input().getType().cast<mlir::ShapedType>().getElementType();
    const auto filterShape = conv.filter().getType().cast<mlir::ShapedType>().getShape();

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(conv.pads_end());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(conv.pads_begin());
    const auto windowStrides = parseIntArrayAttr<int64_t>(conv.strides());
    const auto windowDilations = parseIntArrayAttr<int64_t>(conv.dilations());

    static const auto ChanDim = Dim(1);
    if (inShape[ChanDim.ind()] != filterShape[ChanDim.ind()]) {
        return errorAt(loc, "Channels count of input tensor shape and filter shape must be the same: {0} != {1}",
                       inShape[ChanDim.ind()], filterShape[ChanDim.ind()]);
    }

    const auto outputShape =
            ngraph::infer_convolution_forward(nullptr, ngraph::Shape(inShape.begin(), inShape.end()),
                                              ngraph::Strides(windowStrides.size(), 1),  // dummy data dilations
                                              ngraph::CoordinateDiff(dataPaddingBelow.begin(), dataPaddingBelow.end()),
                                              ngraph::CoordinateDiff(dataPaddingAbove.begin(), dataPaddingAbove.end()),
                                              ngraph::Shape(filterShape.begin(), filterShape.end()),
                                              ngraph::Strides(windowStrides.begin(), windowStrides.end()),
                                              ngraph::Strides(windowDilations.begin(), windowDilations.end()));

    const auto shapeI64 = to_small_vector(outputShape.get_shape() | transformed([](size_t val) {
                                              return checked_cast<int64_t>(val);
                                          }));
    inferredReturnShapes.emplace_back(shapeI64, inType);

    return mlir::success();
}

void vpux::IE::ConvolutionOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                          mlir::MLIRContext* context) {
    patterns.insert<FuseConvAndBias>(context);
}

mlir::Value vpux::IE::ConvolutionOp::reifyTile(const TileInfo& outputTile, mlir::OpBuilder& builder) {
    const auto origInputShape = getShape(input());
    const auto origFilterShape = getShape(filter());
    const auto origBiasShape = bias() != nullptr ? getShape(bias()) : ShapeRef();

    const auto tileConf = backInferConvTile(outputTile, origInputShape, origFilterShape, origBiasShape, strides(),
                                            pads_begin(), pads_end());

    const std::array<int64_t, 2> padsBegin = {tileConf.pads.top, tileConf.pads.left};
    const std::array<int64_t, 2> padsEnd = {tileConf.pads.bottom, tileConf.pads.right};

    const auto inputTileVal = makeTile(builder, getLoc(), input(), tileConf.inputTile, "input");
    const auto filterTileVal = makeTile(builder, getLoc(), filter(), tileConf.filterTile, "filter");
    const auto biasTileVal =
            bias() != nullptr ? makeTile(builder, getLoc(), bias(), tileConf.biasTile, "bias") : nullptr;

    const auto tileName = llvm::formatv("output tile {0}", outputTile.offsets).str();
    const auto tileLoc = appendLoc(getLoc(), tileName);

    const auto tiledResType = getDenseTileType(getType(), outputTile.offsets, outputTile.shape);

    auto tiledOp = builder.create<IE::ConvolutionOp>(tileLoc, tiledResType, inputTileVal, filterTileVal, biasTileVal,
                                                     stridesAttr(), getIntArrayAttr(builder, padsBegin),
                                                     getIntArrayAttr(builder, padsEnd), dilationsAttr(), post_opAttr());

    return tiledOp.output();
}

//
// GroupConvolution
//

mlir::LogicalResult vpux::IE::GroupConvolutionOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::GroupConvolutionOpAdaptor conv(operands, attrs);
    if (mlir::failed(conv.verify(loc))) {
        return mlir::failure();
    }

    auto inShape = to_small_vector(conv.input().getType().cast<mlir::ShapedType>().getShape());
    const auto inType = conv.input().getType().cast<mlir::ShapedType>().getElementType();
    auto filterShape = to_small_vector(conv.filter().getType().cast<mlir::ShapedType>().getShape());

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(conv.pads_end());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(conv.pads_begin());
    const auto windowStrides = parseIntArrayAttr<int64_t>(conv.strides());
    const auto windowDilations = parseIntArrayAttr<int64_t>(conv.dilations());

    int64_t groups = 0;
    if (conv.groups() != 0) {
        if (filterShape.size() != inShape.size()) {
            return errorAt(loc, "Wrong filter shape '{0}'", filterShape);
        }

        groups = conv.groups().getInt();
    } else {
        if (filterShape.size() != inShape.size() + 1) {
            return errorAt(loc, "Wrong filter shape '{0}'", filterShape);
        }

        groups = filterShape[0];

        // we need to adjust filters_shape to reuse helpers for normal convolution
        filterShape[1] *= groups;
        filterShape.erase(filterShape.begin());
    }

    inShape[1] /= groups;

    const auto outputShape =
            ngraph::infer_convolution_forward(nullptr, ngraph::Shape(inShape.begin(), inShape.end()),
                                              ngraph::Strides(windowStrides.size(), 1),  // dummy data dilations
                                              ngraph::CoordinateDiff(dataPaddingBelow.begin(), dataPaddingBelow.end()),
                                              ngraph::CoordinateDiff(dataPaddingAbove.begin(), dataPaddingAbove.end()),
                                              ngraph::Shape(filterShape.begin(), filterShape.end()),
                                              ngraph::Strides(windowStrides.begin(), windowStrides.end()),
                                              ngraph::Strides(windowDilations.begin(), windowDilations.end()));

    const auto shapeI64 = to_small_vector(outputShape.get_shape() | transformed([](size_t val) {
                                              return checked_cast<int64_t>(val);
                                          }));
    inferredReturnShapes.emplace_back(shapeI64, inType);

    return mlir::success();
}

namespace {

class GroupsToAttr final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    using mlir::OpRewritePattern<IE::GroupConvolutionOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp convOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult GroupsToAttr::matchAndRewrite(IE::GroupConvolutionOp convOp,
                                                  mlir::PatternRewriter& rewriter) const {
    if (convOp.groups().hasValue()) {
        return mlir::failure();
    }

    auto filterShape = to_small_vector(convOp.filter().getType().cast<mlir::ShapedType>().getShape());

    const auto groups = filterShape[0];

    // Adjust filters_shape
    filterShape[1] *= groups;
    filterShape.erase(filterShape.begin());

    const auto filterShapeAttr = getIntArrayAttr(getContext(), filterShape);
    auto newFilter = rewriter.create<IE::ReshapeOp>(convOp->getLoc(), convOp.filter(), nullptr, false, filterShapeAttr);

    rewriter.replaceOpWithNewOp<IE::GroupConvolutionOp>(convOp, convOp.input(), newFilter.output(), convOp.bias(),
                                                        convOp.stridesAttr(), convOp.pads_beginAttr(),
                                                        convOp.pads_endAttr(), convOp.dilationsAttr(),
                                                        getIntAttr(convOp.getContext(), groups), convOp.post_opAttr());

    return mlir::success();
}

}  // namespace

void vpux::IE::GroupConvolutionOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                               mlir::MLIRContext* context) {
    patterns.insert<FuseConvAndBias>(context);
    patterns.insert<GroupsToAttr>(context);
}

OutputTiling vpux::IE::GroupConvolutionOp::generateTiling(Logger log) {
    auto tilingInfo = mlir::dyn_cast<IE::TilingInfoOpInterface>(getOperation());
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface",
                    getOperationName());

    const auto outputShape = getShape(output());

    const auto isSupportedTileSize = [&tilingInfo, outputShape, log](ShapeRef nTilesOnDim) -> bool {
        const auto tiles = fillDividedTiles(nTilesOnDim, outputShape);
        return tilingInfo.isSupportedTiling(tiles, log);
    };

    Shape nTilesOnDim(outputShape.size(), 1);

    if (auto channelsInfo = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(getOperation())) {
        const auto chanAlignment = channelsInfo.getChannelAlignment();

        VPUX_THROW_UNLESS(outputShape[Dims4D::Act::C] % chanAlignment == 0,
                          "Depthwise convolution output channels must be a multiple of {0}, got {1}", chanAlignment,
                          outputShape[Dims4D::Act::C]);

        nTilesOnDim[Dims4D::Act::C] = outputShape[Dims4D::Act::C] / chanAlignment;
    }

    while (!isSupportedTileSize(nTilesOnDim)) {
        Optional<Dim> dimToTile;

        for (auto ind : irange(Dims4D::Act::numSpatialDims)) {
            const auto spatialDim = Dims4D::Act::getSpatialDim(ind);

            const auto origSize = static_cast<double>(outputShape[spatialDim]);
            const auto prevDivisor = static_cast<double>(nTilesOnDim[spatialDim]);

            if ((origSize / prevDivisor) > 1.0) {
                dimToTile = spatialDim;
                break;
            }
        }

        VPUX_THROW_UNLESS(dimToTile.hasValue(), "Failed to tile {0} at '{1}'", getOperationName(), getLoc());
        nTilesOnDim[dimToTile.getValue()]++;
    }

    return fillDividedTiles(nTilesOnDim, outputShape);
}

mlir::Value vpux::IE::GroupConvolutionOp::reifyTile(const TileInfo& outputTile, mlir::OpBuilder& builder) {
    const auto origInputShape = getShape(input());
    const auto origFilterShape = getShape(filter());
    const auto origBiasShape = bias() != nullptr ? getShape(bias()) : ShapeRef();

    const auto tileConf = backInferGroupConvTile(outputTile, origInputShape, origFilterShape, origBiasShape, strides(),
                                                 pads_begin(), pads_end());

    const std::array<int64_t, 2> padsBegin = {tileConf.pads.top, tileConf.pads.left};
    const std::array<int64_t, 2> padsEnd = {tileConf.pads.bottom, tileConf.pads.right};

    const auto inputTileVal = makeTile(builder, getLoc(), input(), tileConf.inputTile, "input");
    const auto filterTileVal = makeTile(builder, getLoc(), filter(), tileConf.filterTile, "filter");
    const auto biasTileVal =
            bias() != nullptr ? makeTile(builder, getLoc(), bias(), tileConf.biasTile, "bias") : nullptr;

    const auto tileName = llvm::formatv("output tile {0}", outputTile.offsets).str();
    const auto tileLoc = appendLoc(getLoc(), tileName);

    const auto groups = tileConf.filterTile.shape[Dims4D::Filter::OC];
    const auto groupsAttr = getIntAttr(builder, groups);

    const auto tiledResType = getDenseTileType(getType(), outputTile.offsets, outputTile.shape);

    auto tiledOp = builder.create<IE::GroupConvolutionOp>(
            tileLoc, tiledResType, inputTileVal, filterTileVal, biasTileVal, stridesAttr(),
            getIntArrayAttr(builder, padsBegin), getIntArrayAttr(builder, padsEnd), dilationsAttr(), groupsAttr,
            post_opAttr());

    return tiledOp.output();
}
