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

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

namespace {

mlir::FailureOr<SmallVector<int64_t>> extractIntVector(mlir::Location loc, const mlir::Value& value,
                                                       const mlir::ArrayAttr& attr) {
    if (attr != nullptr) {
        return parseIntArrayAttr<int64_t>(attr);
    } else if (value != nullptr) {
        auto valueConst = value.getDefiningOp<Const::DeclareOp>();
        if (valueConst == nullptr) {
            return errorAt(loc, "Only constant input is supported for interpolate attribute");
        }

        const auto valueContent = valueConst.content();
        return to_small_vector(valueContent.getValues<int64_t>());
    }
    return errorAt(loc, "Parameter were not provided");
}

mlir::FailureOr<SmallVector<double>> extractFPVector(mlir::Location loc, const mlir::Value& value,
                                                     const mlir::ArrayAttr& attr) {
    if (attr != nullptr) {
        return parseFPArrayAttr<double>(attr);
    } else if (value != nullptr) {
        auto valueConst = value.getDefiningOp<Const::DeclareOp>();

        if (valueConst == nullptr) {
            return errorAt(loc, "Only constant input is supported for interpolate attribute");
        }

        const auto valueContent = valueConst.content();
        return to_small_vector(valueContent.getValues<double>());
    }
    return errorAt(loc, "Parameter were not provided");
}

void applyInterpPads(MutableArrayRef<int64_t> outShape, ArrayRef<int64_t> padsBegin, ArrayRef<int64_t> padsEnd) {
    // pads might be zero initialized
    if (padsBegin.size() != padsEnd.size() || padsBegin.size() != outShape.size()) {
        return;
    }
    // TODO: naive implementation only apply pads to calculated output shape
    for (auto d : outShape | indexed) {
        d.value() += padsBegin[d.index()] + padsEnd[d.index()];
    }
}

mlir::FailureOr<SmallVector<int64_t>> propagateShape(mlir::Location loc, mlir::FailureOr<SmallVector<int64_t>> axes,
                                                     ArrayRef<int64_t> inputShape,
                                                     mlir::FailureOr<ArrayRef<int64_t>> padsBegin,
                                                     mlir::FailureOr<ArrayRef<int64_t>> padsEnd,
                                                     vpux::IE::InterpolateCalcMode calcMode,
                                                     mlir::FailureOr<ArrayRef<int64_t>> sizes,
                                                     mlir::FailureOr<ArrayRef<double>> scales, vpux::Logger log) {
    log.trace("Interp propagate shape: input = {0}", inputShape);
    const auto axes_val = axes.getValue();

    SmallVector<int64_t> outShape;

    for (size_t i = 0; i < inputShape.size(); ++i) {
        outShape.emplace_back(inputShape[i]);
    }

    if (calcMode == IE::InterpolateCalcMode::sizes) {
        const auto sizes_val = sizes.getValue();

        if (sizes_val.size() != axes_val.size()) {
            return errorAt(loc,
                           "Num of elements in sizes tensor: {0} should be equal to number of indices in axes: {1}",
                           sizes_val.size(), axes_val.size());
        }
        auto sizesIter = sizes_val.begin();

        for (const auto& i : axes_val) {
            log.trace("Interp sizes - axis: {0}", i);
            outShape[i] = *sizesIter++;
        }

    } else if (calcMode == IE::InterpolateCalcMode::scales) {
        const auto scales_val = scales.getValue();

        if (scales_val.size() != axes_val.size()) {
            return errorAt(loc,
                           "Num of elements in scales tensor: {0} should be equal to number of indices in axes: {1}",
                           scales_val.size(), axes_val.size());
        }

        auto scalesIter = scales_val.begin();

        for (const auto& i : axes_val) {
            log.trace("Interp scales - axis: {0}", i);
            outShape[i] = static_cast<int64_t>(floor((*scalesIter++) * inputShape[i]));
        }

    } else
        return errorAt(loc, "Doesn't support shape_calculation_mode: {0}", calcMode);

    // meaning pads provided in attributes
    if (mlir::succeeded(padsBegin) && succeeded(padsEnd)) {
        applyInterpPads(outShape, padsBegin.getValue(), padsEnd.getValue());
    }

    log.trace("Interp propagate shape: output = {0}", outShape);

    return outShape;
}

mlir::FailureOr<SmallVector<int64_t>> calcOutputShapes(mlir::Location loc, IE::InterpolateOpAdaptor interpolate,
                                                       vpux::Logger log) {
    const auto axes = extractIntVector(loc, interpolate.axes(), interpolate.axes_attr());
    const auto beginPads = extractIntVector(loc, {}, interpolate.attr().pads_begin());
    const auto endPads = extractIntVector(loc, {}, interpolate.attr().pads_end());

    const auto inType = interpolate.input().getType().cast<mlir::ShapedType>();
    const auto inputShape = inType.getShape();

    return propagateShape(loc, axes, inputShape, beginPads, endPads, interpolate.attr().shape_calc_mode().getValue(),
                          extractIntVector(loc, interpolate.sizes(), interpolate.sizes_attr()),
                          extractFPVector(loc, interpolate.scales(), interpolate.scales_attr()), log);
}

}  // namespace

mlir::LogicalResult vpux::IE::InterpolateOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::InterpolateOpAdaptor interpolate(operands, attrs);
    if (mlir::failed(interpolate.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = interpolate.input().getType().cast<mlir::ShapedType>();
    // todo: is there better way to send log object from here
    vpux::Logger log("none", LogLevel::None);

    auto outShape = calcOutputShapes(loc, interpolate, log);

    if (mlir::failed(outShape)) {
        return mlir::failure();
    }

    inferredReturnShapes.emplace_back(outShape.getValue(), inType.getElementType());
    return mlir::success();
}

namespace {

//
// ConvertInputsToAttr
//

class ConvertInputsToAttr final : public mlir::OpRewritePattern<IE::InterpolateOp> {
public:
    using mlir::OpRewritePattern<IE::InterpolateOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::InterpolateOp InterpolateOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertInputsToAttr::matchAndRewrite(IE::InterpolateOp InterpolateOp,
                                                         mlir::PatternRewriter& rewriter) const {
    if (InterpolateOp.sizes_attr().hasValue() || InterpolateOp.scales_attr().hasValue() ||
        InterpolateOp.axes_attr().hasValue()) {
        return mlir::failure();
    }

    // convert sizes
    auto sizes = extractIntVector(InterpolateOp.getLoc(), InterpolateOp.sizes(), nullptr);

    if (mlir::failed(sizes)) {
        return mlir::failure();
    }
    const auto sizesAttr = getIntArrayAttr(InterpolateOp.getContext(), sizes.getValue());

    // convert scales
    auto scales = extractFPVector(InterpolateOp.getLoc(), InterpolateOp.scales(), nullptr);

    if (mlir::failed(scales)) {
        return mlir::failure();
    }
    const auto scalesAttr = getFPArrayAttr(InterpolateOp.getContext(), scales.getValue());

    // convert axes
    auto axes = extractIntVector(InterpolateOp.getLoc(), InterpolateOp.axes(), nullptr);

    if (mlir::failed(axes)) {
        return mlir::failure();
    }
    const auto axesAttr = getIntArrayAttr(InterpolateOp.getContext(), axes.getValue());

    // rewrite layer
    rewriter.replaceOpWithNewOp<IE::InterpolateOp>(InterpolateOp, InterpolateOp.input(), nullptr, nullptr, nullptr,
                                                   sizesAttr, scalesAttr, axesAttr, InterpolateOp.attr());

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::InterpolateOp::getCanonicalizationPatterns(mlir::OwningRewritePatternList& patterns,
                                                          mlir::MLIRContext* context) {
    patterns.insert<ConvertInputsToAttr>(context);
}

InputTiling vpux::IE::InterpolateOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger log) {
    const auto origAxes = extractIntVector(getLoc(), axes(), axes_attrAttr());

    auto forwardScaleOrError = extractFPVector(getLoc(), scales(), scales_attr().getValueOr<mlir::ArrayAttr>({}));
    VPUX_THROW_UNLESS(mlir::succeeded(forwardScaleOrError),
                      "InterpolateOp::backInferTileInfo failed to extract scales");

    SmallVector<double> backwardScale;
    for (auto fwScale : forwardScaleOrError.getValue()) {
        backwardScale.push_back(1. / fwScale);
    }

    auto inferShape = [&](ShapeRef inputShapeFromTile, MutableArrayRef<int64_t> beginPads,
                          MutableArrayRef<int64_t> endPads) {
        SmallVector<int64_t> inputShape;
        for (auto dim : inputShapeFromTile) {
            inputShape.push_back(dim);
        }
        // TODO: how to deal with calc-mode = size if scales missed - recalc them somewhere: attr().shape_calc_mode()
        auto shape_calc_mode = IE::InterpolateCalcMode::scales;

        auto outputShape =
                propagateShape(getLoc(), origAxes, inputShape, {beginPads}, {endPads}, shape_calc_mode,
                               extractIntVector(getLoc(), sizes(), sizes_attr().getValueOr<mlir::ArrayAttr>({})),
                               {backwardScale}, log);
        VPUX_THROW_UNLESS(mlir::succeeded(outputShape),
                          "InterpolateOp::backInferTileInfo failed to propagate Shape-back");

        // need to run forward propagate in order to adjust pads
        auto inputShapeAfterProp =
                propagateShape(getLoc(), origAxes, outputShape.getValue(), {beginPads}, {endPads}, shape_calc_mode,
                               extractIntVector(getLoc(), sizes(), sizes_attr().getValueOr<mlir::ArrayAttr>({})),
                               {forwardScaleOrError}, log);

        VPUX_THROW_UNLESS(mlir::succeeded(inputShapeAfterProp),
                          "InterpolateOp::backInferTileInfo failed to propagate Shape-forward");

        // TODO: we counting only endpads - begin pad might matter for offsets not for dims
        if (endPads.size() == inputShape.size()) {
            for (auto shapeOrig : inputShape | indexed) {
                endPads[shapeOrig.index()] = shapeOrig.value() - inputShapeAfterProp.getValue()[shapeOrig.index()];
            }
        }

        return outputShape;
    };

    SmallVector<int64_t> beginPads(4, 0);
    SmallVector<int64_t> endPads(4, 0);

    auto inputShapeForTile = inferShape(outputTile.shape, beginPads, endPads);
    if (mlir::failed(inputShapeForTile)) {
        return TilingInfo(ArrayRef<TileInfo>());
    }

    auto inputOffsetForTile = inferShape(outputTile.offsets, {}, {});
    if (mlir::failed(inputOffsetForTile)) {
        return TilingInfo(ArrayRef<TileInfo>());
    }

    TileInfo inputTile{inputShapeForTile.getValue().size()};

    inputTile.shape = Shape(inputShapeForTile.getValue());
    inputTile.offsets = Shape(inputOffsetForTile.getValue());

    auto convertShape = [](::mlir::Value plainShape) {
        auto origShape = getShape(plainShape);
        TileInfo tileShape{origShape.size()};
        tileShape.shape = getShape(plainShape).toValues();
        return tileShape;
    };

    auto sizeTiling = convertShape(sizes());
    auto scaleTiling = convertShape(scales());
    auto axisTiling = convertShape(axes());

    auto iTiling = InputTiling{{inputTile, sizeTiling, scaleTiling, axisTiling}};

    iTiling.pads = {0, endPads[2], 0, endPads[3]};

    return iTiling;
}

void vpux::IE::InterpolateOp::adjustAttrs(const TilingInfo& inputTiling) {
    if (!inputTiling.pads.hasValue()) {
        return;
    }
    mlir::Builder builder(*this);

    SmallVector<int64_t> endPads = {0, 0, 0, 0};
    SmallVector<int64_t> beginPads = {0, 0, 0, 0};

    endPads[2] = inputTiling.pads.getValue().right;
    endPads[3] = inputTiling.pads.getValue().bottom;

    auto newEndPads = builder.getI64ArrayAttr(endPads);
    auto newBeginPads = builder.getI64ArrayAttr(beginPads);

    auto newAttrs =
            InterpolateAttr::get(attr().mode(), attr().shape_calc_mode(), attr().coord_mode(), attr().nearest_mode(),
                                 attr().antialias(), newBeginPads, newEndPads, attr().cube_coeff(), getContext());

    // set pads begin + end attrs
    attrAttr(newAttrs);
}

bool vpux::IE::InterpolateOp::isSupportedTiling(const vpux::OutputTiling& tiles, vpux::Logger log) {
    const auto arch = VPU::getArch(*this);
    if (arch != VPU::ArchKind::MTL) {
        // TODO: E**W-35009 UPA shaves has no need to use tiling yet
        // when switching to shave-nn, or another runtime implementation support might need to be added based on
        // restrictions
        return true;
    }

    const auto origInputShape = getShape(input());

    auto cmxAvailable = vpux::VPU::getTotalCMXSize(*this);

    for (auto&& outputTile : tiles) {
        auto inputTiles = backInferTileInfo(outputTile, log);
        VPUX_THROW_UNLESS(inputTiles.tiles.size() == 4,
                          "Only single input tile and params tiles expected in tile backprop for interpolate");

        auto inputTile = inputTiles.tiles.front();
        if (inputTile.shape.totalSize() + outputTile.shape.totalSize() > cmxAvailable.to<Byte>().count()) {
            log.trace("Interp tiling probe invalid: from {0} -> {1}, total CMX size: {2}, max: {3}", inputTile.shape,
                      outputTile, inputTile.shape.totalSize() + outputTile.shape.totalSize(),
                      cmxAvailable.to<Byte>().count());
            return false;
        }
    }

    log.trace("Interp tiling probe: from {0} -> {1}", origInputShape, (tiles.empty() ? Shape() : tiles.front().shape));

    for (auto&& tile : tiles) {
        tile.printFormat(log.getLevelStream(LogLevel::Trace));
        log.trace("\n");
    }

    return true;
}

bool vpux::IE::InterpolateOp::isSupportedPrefetchTiling(ShapeRef /*tileAxis*/, Logger /*log*/) {
    const auto arch = VPU::getArch(*this);

    if (arch != VPU::ArchKind::MTL) {
        // TODO: The prefetch tiling fo act Shave OP need to be investigated
    }
    return false;
}

bool vpux::IE::InterpolateOp::isSupportedPrefetchPattern(ShapeRef /*tileAxis*/, mlir::Operation* /*parentOp*/,
                                                         ShapeRef /*parentTileAxis*/, vpux::Logger /*log*/) {
    const auto arch = VPU::getArch(*this);
    if (arch != VPU::ArchKind::MTL) {
        // TODO: The prefetch tiling fo act Shave OP need to be investigated
    }

    return true;
}
