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

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/enums.hpp"

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

mlir::FailureOr<SmallVector<int64_t>> calcOutputShapes(mlir::Location loc, IE::InterpolateOpAdaptor interpolate) {
    const auto axes = extractIntVector(loc, interpolate.axes(), interpolate.axes_attr());
    const auto axes_val = axes.getValue();

    const auto inType = interpolate.input().getType().cast<mlir::ShapedType>();
    const auto inputShape = inType.getShape();

    SmallVector<int64_t> outShape;

    for (size_t i = 0; i < inputShape.size(); ++i) {
        outShape.emplace_back(inputShape[i]);
    }

    auto calcMode = interpolate.attr().shape_calc_mode().getValue();

    if (calcMode == IE::InterpolateCalcMode::sizes) {
        const auto sizes = extractIntVector(loc, interpolate.sizes(), interpolate.sizes_attr());
        const auto sizes_val = sizes.getValue();

        if (sizes_val.size() != axes_val.size()) {
            return errorAt(loc,
                           "Num of elements in sizes tensor: {0} should be equal to number of indices in axes: {1}",
                           sizes_val.size(), axes_val.size());
        }
        auto sizesIter = sizes_val.begin();

        for (const auto& i : axes_val)
            outShape[i] = *sizesIter++;

        return outShape;

    } else if (calcMode == IE::InterpolateCalcMode::scales) {
        const auto scales = extractFPVector(loc, interpolate.scales(), interpolate.scales_attr());
        const auto scales_val = scales.getValue();

        if (scales_val.size() != axes_val.size()) {
            return errorAt(loc,
                           "Num of elements in scales tensor: {0} should be equal to number of indices in axes: {1}",
                           scales_val.size(), axes_val.size());
        }

        auto scalesIter = scales_val.begin();

        for (const auto& i : axes_val) {
            outShape[i] = static_cast<int64_t>(floor((*scalesIter++) * inputShape[i]));
        }

        return outShape;

    } else
        return errorAt(loc, "Doesn't support shape_calculation_mode: {0}", calcMode);
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

    auto outShape = calcOutputShapes(loc, interpolate);

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

namespace {

const EnumMap<IE::InterpolateMode, int> supportedInterpModeMap = {
        {IE::InterpolateMode::nearest, 0},      //
        {IE::InterpolateMode::linear, 1},       //
        {IE::InterpolateMode::linear_onnx, 3},  //
};

const EnumMap<IE::InterpolateNearestMode, int> nearestModeMap = {
        {IE::InterpolateNearestMode::round_prefer_floor, 0},  //
        {IE::InterpolateNearestMode::round_prefer_ceil, 1},   //
        {IE::InterpolateNearestMode::floor, 2},               //
        {IE::InterpolateNearestMode::ceil, 3},                //
        {IE::InterpolateNearestMode::simple, 4},              //
};

const EnumMap<IE::InterpolateCoordMode, int> coordTransformModeMap = {
        {IE::InterpolateCoordMode::half_pixel, 0},            //
        {IE::InterpolateCoordMode::pytorch_half_pixel, 1},    //
        {IE::InterpolateCoordMode::asymmetric, 2},            //
        {IE::InterpolateCoordMode::tf_half_pixel_for_nn, 3},  //
        {IE::InterpolateCoordMode::align_corners, 4},         //
};

}  // namespace

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::IE::InterpolateOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::InterpolateParamsBuilder builder(writer);

    const auto mode = attr().mode().getValue();
    const auto coord_mode = attr().coord_mode().getValue();
    const auto nearest_mode = attr().nearest_mode().getValue();
    const auto antialias = attr().antialias().getValue();

    const auto interpolateModeIter = supportedInterpModeMap.find(mode);
    VPUX_THROW_UNLESS(interpolateModeIter != supportedInterpModeMap.end(), "Unsupported interpolate mode {0}", mode);
    builder.add_interpolationMode(MVCNN::InterpolationMethod(interpolateModeIter->second));

    const auto coordModeIter = coordTransformModeMap.find(coord_mode);
    VPUX_THROW_UNLESS(coordModeIter != coordTransformModeMap.end(), "Unsupported coordinate transformation mode {0}",
                      coord_mode);
    builder.add_coordTransformMode(MVCNN::InterpolationCoordTransMode(coordModeIter->second));

    const auto nearestModeIter = nearestModeMap.find(nearest_mode);
    VPUX_THROW_UNLESS(nearestModeIter != nearestModeMap.end(), "Unsupported nearest mode {0}", nearest_mode);
    builder.add_nearestMode(MVCNN::InterpolationNearestMode(nearestModeIter->second));

    builder.add_align_corners(coord_mode == IE::InterpolateCoordMode::align_corners);
    builder.add_antialias(antialias);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_InterpolateParams});
}
