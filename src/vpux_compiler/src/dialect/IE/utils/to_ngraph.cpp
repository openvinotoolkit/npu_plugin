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

#include "vpux/compiler/dialect/IE/utils/to_ngraph.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/enums.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>

using namespace vpux;

ngraph::op::AutoBroadcastType IE::exportBroadcastType(IE::AutoBroadcastType bType) {
    switch (bType) {
    case IE::AutoBroadcastType::NONE_OR_EXPLICIT:
        return ngraph::op::AutoBroadcastType::NONE;
    case IE::AutoBroadcastType::NUMPY:
        return ngraph::op::AutoBroadcastType::NUMPY;
    case IE::AutoBroadcastType::PDPD:
        return ngraph::op::AutoBroadcastType::PDPD;
    default:
        VPUX_THROW("Unknown AutoBroadcastType");
    }
}

ngraph::op::BroadcastType IE::exportBroadcastMode(IE::BroadcastType bType) {
    switch (bType) {
    case IE::BroadcastType::NUMPY:
        return ngraph::op::BroadcastType::NUMPY;
    case IE::BroadcastType::EXPLICIT:
        return ngraph::op::BroadcastType::EXPLICIT;
    case IE::BroadcastType::BIDIRECTIONAL:
        return ngraph::op::BroadcastType::BIDIRECTIONAL;
    default:
        VPUX_THROW("Unknown BroadcastType");
    }
}

ngraph::op::RoundingType IE::exportRoundingType(IE::RoundingType roundingType) {
    switch (roundingType) {
    case IE::RoundingType::FLOOR:
        return ngraph::op::RoundingType::FLOOR;
    case IE::RoundingType::CEIL:
        return ngraph::op::RoundingType::CEIL;
    default:
        VPUX_THROW("Unknown RoundingType");
    }
}

ngraph::element::Type IE::exportElemType(mlir::MLIRContext* ctx, mlir::Type type) {
    if (type == mlir::Float32Type::get(ctx)) {
        return ngraph::element::f32;
    } else if (type == mlir::Float16Type::get(ctx)) {
        return ngraph::element::f16;
    } else if (type == getSInt64Type(ctx) || type == getInt64Type(ctx)) {
        return ngraph::element::i64;
    } else if (type == getUInt64Type(ctx)) {
        return ngraph::element::u64;
    } else if (type == getSInt32Type(ctx)) {
        return ngraph::element::i32;
    } else if (type == getUInt32Type(ctx)) {
        return ngraph::element::u32;
    } else if (type == getSInt16Type(ctx)) {
        return ngraph::element::i16;
    } else if (type == getUInt16Type(ctx)) {
        return ngraph::element::u16;
    } else if (type == getSInt8Type(ctx)) {
        return ngraph::element::i8;
    } else if (type == getUInt8Type(ctx)) {
        return ngraph::element::u8;
    } else {
        VPUX_THROW("Unsupported precision : '{0}'", type);
    }
}

ngraph::op::DetectionOutputAttrs IE::exportDetectionOutputAttrs(IE::DetectionOutputAttr val) {
    ngraph::op::DetectionOutputAttrs attrs;
    attrs.num_classes = val.num_classes().getInt();
    attrs.background_label_id = val.background_label_id().getInt();
    attrs.top_k = val.top_k().getInt();

    attrs.variance_encoded_in_target = val.variance_encoded_in_target().getValue();

    const auto keep_top_k = parseIntArrayAttr<int32_t>(val.keep_top_k());
    attrs.keep_top_k = std::vector<int32_t>{keep_top_k.begin(), keep_top_k.end()};
    attrs.code_type = val.code_type().getValue().str();

    attrs.share_location = val.share_location().getValue();

    attrs.nms_threshold = val.nms_threshold().getValueAsDouble();
    attrs.confidence_threshold = val.confidence_threshold().getValueAsDouble();

    attrs.clip_after_nms = val.clip_after_nms().getValue();
    attrs.clip_before_nms = val.clip_before_nms().getValue();
    attrs.decrease_label_id = val.decrease_label_id().getValue();
    attrs.normalized = val.normalized().getValue();

    attrs.input_height = val.input_height().getInt();
    attrs.input_width = val.input_width().getInt();

    attrs.objectness_score = val.objectness_score().getValueAsDouble();

    return attrs;
}

ngraph::opset7::Interpolate::InterpolateAttrs IE::exportInterpolateAttrs(IE::InterpolateAttr val) {
    ngraph::opset7::Interpolate::InterpolateAttrs attrs;
    // mode
    switch (val.mode().getValue()) {
    case IE::InterpolateMode::nearest:
        attrs.mode = opset_latest::Interpolate::InterpolateMode::NEAREST;
        break;
    case IE::InterpolateMode::linear:
        attrs.mode = opset_latest::Interpolate::InterpolateMode::LINEAR;
        break;
    case IE::InterpolateMode::linear_onnx:
        attrs.mode = opset_latest::Interpolate::InterpolateMode::LINEAR_ONNX;
        break;
    case IE::InterpolateMode::cubic:
        attrs.mode = opset_latest::Interpolate::InterpolateMode::CUBIC;
        break;
    default:
        VPUX_THROW("Unsupported interpolate mode");
    }

    // shape calculation mode
    switch (val.shape_calc_mode().getValue()) {
    case IE::InterpolateCalcMode::sizes:
        attrs.shape_calculation_mode = opset_latest::Interpolate::ShapeCalcMode::SIZES;
        break;
    case IE::InterpolateCalcMode::scales:
        attrs.shape_calculation_mode = opset_latest::Interpolate::ShapeCalcMode::SCALES;
        break;
    default:
        VPUX_THROW("Unsupported interpolate shape calculation mode");
    }

    // coordinate transformation mode
    switch (val.coord_mode().getValue()) {
    case IE::InterpolateCoordMode::half_pixel:
        attrs.coordinate_transformation_mode = opset_latest::Interpolate::CoordinateTransformMode::HALF_PIXEL;
        break;
    case IE::InterpolateCoordMode::pytorch_half_pixel:
        attrs.coordinate_transformation_mode = opset_latest::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL;
        break;
    case IE::InterpolateCoordMode::asymmetric:
        attrs.coordinate_transformation_mode = opset_latest::Interpolate::CoordinateTransformMode::ASYMMETRIC;
        break;
    case IE::InterpolateCoordMode::tf_half_pixel_for_nn:
        attrs.coordinate_transformation_mode = opset_latest::Interpolate::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN;
        break;
    case IE::InterpolateCoordMode::align_corners:
        attrs.coordinate_transformation_mode = opset_latest::Interpolate::CoordinateTransformMode::ALIGN_CORNERS;
        break;
    default:
        VPUX_THROW("Unsupported interpolate coordinate transformation mode");
    }

    // coordinate transformation mode
    switch (val.nearest_mode().getValue()) {
    case IE::InterpolateNearestMode::round_prefer_floor:
        attrs.nearest_mode = opset_latest::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
        break;
    case IE::InterpolateNearestMode::round_prefer_ceil:
        attrs.nearest_mode = opset_latest::Interpolate::NearestMode::ROUND_PREFER_CEIL;
        break;
    case IE::InterpolateNearestMode::floor:
        attrs.nearest_mode = opset_latest::Interpolate::NearestMode::FLOOR;
        break;
    case IE::InterpolateNearestMode::ceil:
        attrs.nearest_mode = opset_latest::Interpolate::NearestMode::CEIL;
        break;
    case IE::InterpolateNearestMode::simple:
        attrs.nearest_mode = opset_latest::Interpolate::NearestMode::SIMPLE;
        break;
    default:
        VPUX_THROW("Unsupported interpolate nearest mode");
    }

    attrs.antialias = val.antialias().getValue();
    auto pads_begin = parseIntArrayAttr<size_t>(val.pads_begin());
    attrs.pads_begin = InferenceEngine::SizeVector{pads_begin.begin(), pads_begin.end()};
    auto pads_end = parseIntArrayAttr<size_t>(val.pads_end());
    attrs.pads_end = InferenceEngine::SizeVector{pads_end.begin(), pads_end.end()};
    attrs.cube_coeff = val.cube_coeff().getValueAsDouble();

    return attrs;
}

std::string IE::exportLRN_IERegion(IE::LRN_IERegion region) {
    switch (region) {
    case IE::LRN_IERegion::same:
        return "same";
    case IE::LRN_IERegion::across:
        return "across";
    default:
        VPUX_THROW("Unknown LRN_IERegion");
    }
}

ngraph::op::RecurrentSequenceDirection IE::exportRNNSequenceDirection(IE::RNNSequenceDirection val) {
    if (val == IE::RNNSequenceDirection::FORWARD) {
        return ngraph::op::RecurrentSequenceDirection::FORWARD;
    } else if (val == IE::RNNSequenceDirection::REVERSE) {
        return ngraph::op::RecurrentSequenceDirection::REVERSE;
    } else {
        VPUX_THROW("Unknown RNNSequence direction");
    }
}

ngraph::op::TopKSortType IE::exportTopKSortType(IE::TopKSortType val) {
    switch (val) {
    case IE::TopKSortType::NONE:
        return ngraph::op::TopKSortType::NONE;
    case IE::TopKSortType::SORT_INDICES:
        return ngraph::op::TopKSortType::SORT_INDICES;
    case IE::TopKSortType::SORT_VALUES:
        return ngraph::op::TopKSortType::SORT_VALUES;
    default:
        VPUX_THROW("Unknown TopKSortType");
    }
}

ngraph::op::PadMode IE::exportPadMode(IE::PadMode mode) {
    switch (mode) {
    case IE::PadMode::CONSTANT:
        return ngraph::op::PadMode::CONSTANT;
    case IE::PadMode::EDGE:
        return ngraph::op::PadMode::EDGE;
    case IE::PadMode::REFLECT:
        return ngraph::op::PadMode::REFLECT;
    case IE::PadMode::SYMMETRIC:
        return ngraph::op::PadMode::SYMMETRIC;
    default:
        VPUX_THROW("Unknown PadMode");
    }
}

ngraph::op::ProposalAttrs IE::exportProposalAttrs(IE::ProposalAttr val) {
    ngraph::op::ProposalAttrs attrs;
    attrs.base_size = val.baseSize().getInt();
    attrs.pre_nms_topn = val.preNmsTopN().getInt();
    attrs.post_nms_topn = val.postNmsTopN().getInt();
    attrs.nms_thresh = val.nmsThresh().getValueAsDouble();
    attrs.feat_stride = val.featStride().getInt();
    attrs.min_size = val.minSize().getInt();
    const auto ratio = parseFPArrayAttr<float>(val.ratio());
    attrs.ratio = std::vector<float>{ratio.begin(), ratio.end()};
    const auto scale = parseFPArrayAttr<float>(val.scale());
    attrs.scale = std::vector<float>{scale.begin(), scale.end()};
    attrs.clip_before_nms = val.clipBeforeNms().getValue();
    attrs.clip_after_nms = val.clipAfterNms().getValue();
    attrs.normalize = val.normalize().getValue();
    attrs.box_size_scale = val.boxSizeScale().getValueAsDouble();
    attrs.box_coordinate_scale = val.boxCoordinateScale().getValueAsDouble();
    attrs.framework = val.framework().getValue().str();
    attrs.infer_probs = val.inferProbs().getValue();

    return attrs;
}

ngraph::op::v5::Round::RoundMode IE::exportRoundMode(IE::RoundMode val) {
    switch (val) {
    case IE::RoundMode::HALF_TO_EVEN:
        return ngraph::op::v5::Round::RoundMode::HALF_TO_EVEN;
    case IE::RoundMode::HALF_AWAY_FROM_ZERO:
        return ngraph::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO;
    default:
        VPUX_THROW("Unknown RoundMode");
    }
}

std::string IE::exportROIPoolingMethod(IE::ROIPoolingMethod method) {
    switch (method) {
    case IE::ROIPoolingMethod::max:
        return "max";
    case IE::ROIPoolingMethod::bilinear:
        return "bilinear";
    default:
        VPUX_THROW("Unknown ROIPoolingMethod");
    }
}

ngraph::op::TopKMode IE::exportTopKMode(IE::TopKMode val) {
    switch (val) {
    case IE::TopKMode::MAX:
        return ngraph::op::TopKMode::MAX;
    case IE::TopKMode::MIN:
        return ngraph::op::TopKMode::MIN;
    default:
        VPUX_THROW("Unknown TopKMode");
    }
}

InferenceEngine::TensorDesc IE::exportUserTensor(mlir::RankedTensorType tensor) {
    const Shape shape = tensor.getShape();
    InferenceEngine::SizeVector dims;
    for (auto ddim : shape)
        dims.push_back(ddim);
    const mlir::Type elementType = tensor.getElementType();
    const InferenceEngine::Precision precision = IE::exportPrecision(elementType.getContext(), elementType);
    auto dimsOrder = DimsOrder::fromAffineMap(IE::getOrder(tensor));
    InferenceEngine::Layout layout = dimsOrder.toIE();
    return InferenceEngine::TensorDesc{precision, dims, layout};
}

ngraph::element::Type IE::toNGraphType(InferenceEngine::Precision precision) {
    if (precision == InferenceEngine::Precision::FP32) {
        return ngraph::element::f32;
    } else if (precision == InferenceEngine::Precision::FP16) {
        return ngraph::element::f16;
    } else if (precision == InferenceEngine::Precision::I64) {
        return ngraph::element::i64;
    } else if (precision == InferenceEngine::Precision::U64) {
        return ngraph::element::u64;
    } else if (precision == InferenceEngine::Precision::I32) {
        return ngraph::element::i32;
    } else if (precision == InferenceEngine::Precision::U32) {
        return ngraph::element::u32;
    } else if (precision == InferenceEngine::Precision::I16) {
        return ngraph::element::i16;
    } else if (precision == InferenceEngine::Precision::U16) {
        return ngraph::element::u16;
    } else if (precision == InferenceEngine::Precision::I8) {
        return ngraph::element::i8;
    } else if (precision == InferenceEngine::Precision::U8) {
        return ngraph::element::u8;
    } else if (precision == InferenceEngine::Precision::BOOL) {
        return ngraph::element::boolean;
    } else {
        VPUX_THROW("Unsupported precision : '{0}'", precision);
    }
}

InferenceEngine::Precision IE::exportPrecision(mlir::MLIRContext* ctx, mlir::Type type) {
    if (type == mlir::Float32Type::get(ctx)) {
        return InferenceEngine::Precision::FP32;
    } else if (type == mlir::Float16Type::get(ctx)) {
        return InferenceEngine::Precision::FP16;
    } else if (type == getSInt64Type(ctx)) {
        return InferenceEngine::Precision::I64;
    } else if (type == getUInt64Type(ctx)) {
        return InferenceEngine::Precision::U64;
    } else if (type == getSInt32Type(ctx)) {
        return InferenceEngine::Precision::I32;
    } else if (type == getUInt32Type(ctx)) {
        return InferenceEngine::Precision::U32;
    } else if (type == getSInt16Type(ctx)) {
        return InferenceEngine::Precision::I16;
    } else if (type == getUInt16Type(ctx)) {
        return InferenceEngine::Precision::U16;
    } else if (type == getSInt8Type(ctx)) {
        return InferenceEngine::Precision::I8;
    } else if (type == getUInt8Type(ctx)) {
        return InferenceEngine::Precision::U8;
    } else if (type == getBool8Type(ctx)) {
        return InferenceEngine::Precision::BOOL;
    } else {
        VPUX_THROW("Unsupported precision : '{0}'", type);
    }
}
