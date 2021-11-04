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

#include "vpux/compiler/backend/IE.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes/arch.hpp"
#include "vpux/compiler/dialect/VPUIP/blob_writer.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/schema.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/IE/format.hpp"
#include "vpux/utils/IE/hash.hpp"

#include "vpux/utils/IE/loop.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/string_ref.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>

#include <ie_common.h>
#include <ie_layouts.h>
#include <ie_precision.hpp>

#include <cpp/ie_cnn_network.h>

#include <precision_utils.h>

#include <unordered_map>

//#include <ngraph/opsets/opset7.hpp>
#include <ngraph/op/abs.hpp>
#include <legacy/ngraph_ops/lrn_ie.hpp>
#include <ngraph/op/parameter.hpp>
#include "ngraph/function.hpp"
#include "ngraph/shape.hpp"
#include <ngraph/node.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/type/element_type.hpp>
#include <ngraph/type.hpp>
#include <fstream>

using namespace vpux;

namespace {

namespace opset_latest = ngraph::opset7;

using NodeMap = std::unordered_map<mlir::Operation *, std::shared_ptr<ngraph::Node>>;


ngraph::op::AutoBroadcastType exportBroadcastType(IE::AutoBroadcastType bType) {
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

InferenceEngine::Precision exportPrecision(mlir::MLIRContext* ctx, mlir::Type type) {
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
    } else {
        VPUX_THROW("Unsupported precision : '{0}'", type);
    }
}

ngraph::element::Type exportElemType(mlir::MLIRContext* ctx, mlir::Type type) {
    if (type == mlir::Float32Type::get(ctx)) {
        return ngraph::element::f32;
    } else if (type == mlir::Float16Type::get(ctx)) {
        return ngraph::element::f16;
    } else if (type == getSInt64Type(ctx)) {
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

ngraph::element::Type toNGraphType(InferenceEngine::Precision precision)
{
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
    } else {
        VPUX_THROW("Unsupported precision : '{0}'", precision);
    }
}

ngraph::op::RoundingType exportRoundingType(IE::RoundingType roundingType) {
    switch (roundingType) {
    case IE::RoundingType::FLOOR:
        return ngraph::op::RoundingType::FLOOR;
    case IE::RoundingType::CEIL:
        return ngraph::op::RoundingType::CEIL;
    default:
        VPUX_THROW("Unknown RoundingType");
    }
}

ngraph::op::RecurrentSequenceDirection exportRNNSequenceDirection(
        const IE::RNNSequenceDirection val) {
    if (val == IE::RNNSequenceDirection::FORWARD) {
        return ngraph::op::RecurrentSequenceDirection::FORWARD;
    } else if (val == IE::RNNSequenceDirection::REVERSE) {
        return ngraph::op::RecurrentSequenceDirection::REVERSE;
    } else {
        VPUX_THROW("Unknown RNNSequence direction");
    }
}

ngraph::op::PadMode exportPadMode(IE::PadMode mode) {
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

ngraph::op::ProposalAttrs exportProposalAttrs(const IE::ProposalAttr& val) {
    ngraph::op::ProposalAttrs attrs;
    attrs.base_size = val.baseSize().getUInt();
    attrs.pre_nms_topn = val.preNmsTopN().getUInt();
    attrs.post_nms_topn = val.postNmsTopN().getUInt();
    attrs.feat_stride = val.featStride().getUInt();
    attrs.min_size = val.minSize().getUInt();
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

ngraph::op::v5::Round::RoundMode exportRoundMode(IE::RoundMode val) {
    switch (val) {
    case IE::RoundMode::HALF_TO_EVEN:
        return ngraph::op::v5::Round::RoundMode::HALF_TO_EVEN;
    case IE::RoundMode::HALF_AWAY_FROM_ZERO:
        return ngraph::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO;
    default:
        VPUX_THROW("Unknown RoundMode");
    }
}

opset_latest::Interpolate::InterpolateAttrs exportInterpolateAttrs(const IE::InterpolateAttr& val) {
    opset_latest::Interpolate::InterpolateAttrs attrs;
    // mode
    switch (val.mode().getValue()) {
    case IE::InterpolateMode::nearest:
        attrs.mode = opset_latest::Interpolate::InterpolateMode::nearest;
        break;
    case IE::InterpolateMode::linear:
        attrs.mode = opset_latest::Interpolate::InterpolateMode::linear;
        break;
    case IE::InterpolateMode::linear_onnx:
        attrs.mode = opset_latest::Interpolate::InterpolateMode::linear_onnx;
        break;
    case IE::InterpolateMode::cubic:
        attrs.mode = opset_latest::Interpolate::InterpolateMode::cubic;
        break;
    default:
        VPUX_THROW("Unsupported interpolate mode");
    }

    // shape calculation mode
    switch (val.shape_calc_mode().getValue()) {
    case IE::InterpolateCalcMode::sizes:
        attrs.shape_calculation_mode = opset_latest::Interpolate::ShapeCalcMode::sizes;
        break;
    case IE::InterpolateCalcMode::scales:
        attrs.shape_calculation_mode = opset_latest::Interpolate::ShapeCalcMode::scales;
        break;
    default:
        VPUX_THROW("Unsupported interpolate shape calculation mode");
    }

    // coordinate transformation mode
    switch (val.coord_mode().getValue()) {
    case IE::InterpolateCoordMode::half_pixel:
        attrs.coordinate_transformation_mode = opset_latest::Interpolate::CoordinateTransformMode::half_pixel;
        break;
    case IE::InterpolateCoordMode::pytorch_half_pixel:
        attrs.coordinate_transformation_mode = opset_latest::Interpolate::CoordinateTransformMode::pytorch_half_pixel;
        break;
    case IE::InterpolateCoordMode::asymmetric:
        attrs.coordinate_transformation_mode = opset_latest::Interpolate::CoordinateTransformMode::asymmetric;
        break;
    case IE::InterpolateCoordMode::tf_half_pixel_for_nn:
        attrs.coordinate_transformation_mode = opset_latest::Interpolate::CoordinateTransformMode::tf_half_pixel_for_nn;
        break;
    case IE::InterpolateCoordMode::align_corners:
        attrs.coordinate_transformation_mode = opset_latest::Interpolate::CoordinateTransformMode::align_corners;
        break;
    default:
        VPUX_THROW("Unsupported interpolate coordinate transformation mode");
    }

    // coordinate transformation mode
    switch (val.nearest_mode().getValue()) {
    case IE::InterpolateNearestMode::round_prefer_floor:
        attrs.nearest_mode = opset_latest::Interpolate::NearestMode::round_prefer_floor;
        break;
    case IE::InterpolateNearestMode::round_prefer_ceil:
        attrs.nearest_mode = opset_latest::Interpolate::NearestMode::round_prefer_ceil;
        break;
    case IE::InterpolateNearestMode::floor:
        attrs.nearest_mode = opset_latest::Interpolate::NearestMode::floor;
        break;
    case IE::InterpolateNearestMode::ceil:
        attrs.nearest_mode = opset_latest::Interpolate::NearestMode::ceil;
        break;
    case IE::InterpolateNearestMode::simple:
        attrs.nearest_mode = opset_latest::Interpolate::NearestMode::simple;
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

ngraph::op::TopKMode exportTopKMode(IE::TopKMode val) {
    switch (val) {
    case IE::TopKMode::MAX:
        return ngraph::op::TopKMode::MAX;
    case IE::TopKMode::MIN:
        return ngraph::op::TopKMode::MIN;
    default:
        VPUX_THROW("Unknown TopKMode");
    }
}

ngraph::op::TopKSortType exportTopKSortType(IE::TopKSortType val) {
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

ngraph::op::DetectionOutputAttrs exportDetectionOutputAttrs(const IE::DetectionOutputAttr& val) {
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

ngraph::op::EpsMode exportEpsMode(IE::EpsMode val) {
    switch (val) {
    case IE::EpsMode::ADD:
        return ngraph::op::EpsMode::ADD;
    case IE::EpsMode::MAX:
        return ngraph::op::EpsMode::MAX;
    default:
        VPUX_THROW("Unknown EpsMode");
    }
}

std::string exportROIPoolingMethod(const IE::ROIPoolingMethod& method) {
    switch (method) {
    case IE::ROIPoolingMethod::max:
        return "max";
    case IE::ROIPoolingMethod::bilinear:
        return "bilinear";
    default:
        VPUX_THROW("Unknown ROIPoolingMethod");
    }
}

InferenceEngine::TensorDesc exportUserTensor(const mlir::RankedTensorType &tensor) {
    const Shape shape = tensor.getShape();
    InferenceEngine::SizeVector dims;
    for (auto ddim : shape)
        dims.push_back(ddim);
    const mlir::Type elementType = tensor.getElementType();
    const InferenceEngine::Precision precision = exportPrecision(elementType.getContext(), elementType);
    DimsOrder dimsOrder = DimsOrder::fromType(tensor);
    InferenceEngine::Layout layout = dimsOrder.toIE();
    return InferenceEngine::TensorDesc{precision, dims, layout};
}

std::string provide_bin_path(const std::string &xmlPath) {
    assert(xmlPath.size() > 4); // should be check by valid_xml_path
    std::string bestPath = xmlPath;
    const char *const extension = "bin";
    const auto ext_size = std::strlen(extension);
    bestPath.replace(bestPath.size() - ext_size, ext_size, extension);
    return bestPath;
}

//
// Parsers
//
std::shared_ptr<ngraph::Node> parseNode(Const::DeclareOp origOp, ngraph::OutputVector &)
{
    auto cont = origOp.content();
    mlir::Type elType = cont.getElementType();
    mlir::MLIRContext* ctx = elType.getContext();
    auto valsRange = cont.getValues<double>();
    auto elShape = cont.getShape();
    ngraph::Shape sh(elShape.begin(), elShape.end());
    return std::make_shared<ngraph::opset7::Constant>(exportElemType(ctx, elType), sh, std::vector<double>(valsRange.begin(), valsRange.end()));
}

std::shared_ptr<ngraph::Node> parseNode(IE::ConvertOp origOp, ngraph::OutputVector &inputs)
{
    mlir::Type dstElemType = origOp.dstElemType();
    return std::make_shared<ngraph::opset7::Convert>(inputs.at(0), exportElemType(dstElemType.getContext(), dstElemType));
}

std::shared_ptr<ngraph::Node> parseNode(IE::SoftMaxOp origOp, ngraph::OutputVector &inputs)
{
    auto axisIndVal = origOp.axisInd();
    return std::make_shared<ngraph::opset7::Softmax>(inputs.at(0), axisIndVal);
}

std::shared_ptr<ngraph::Node> parseNode(IE::TileOp, ngraph::OutputVector &inputs)
{
    return std::make_shared<ngraph::opset7::Tile>(inputs.at(0), inputs.at(1));
}

std::shared_ptr<ngraph::Node> parseNode(IE::SplitOp origOp, ngraph::OutputVector &inputs)
{
    auto numSplits = origOp.num_splits();
    return std::make_shared<ngraph::opset7::Split>(inputs.at(0), inputs.at(1), numSplits);
}

std::shared_ptr<ngraph::Node> parseNode(IE::PowerOp origOp, ngraph::OutputVector &inputs)
{
    ngraph::op::AutoBroadcastType autoBroadCastType = exportBroadcastType(origOp.auto_broadcast());
    return std::make_shared<ngraph::opset7::Power>(inputs.at(0), inputs.at(1), ngraph::op::AutoBroadcastSpec(autoBroadCastType));
}

std::shared_ptr<ngraph::Node> parseNode(IE::MultiplyOp origOp, ngraph::OutputVector &inputs)
{
    ngraph::op::AutoBroadcastType autoBroadCastType = exportBroadcastType(origOp.auto_broadcast());
    return std::make_shared<ngraph::opset7::Multiply>(inputs.at(0), inputs.at(1), ngraph::op::AutoBroadcastSpec(autoBroadCastType));
}

std::shared_ptr<ngraph::Node> parseNode(IE::ReLUOp, ngraph::OutputVector &inputs)
{
    return std::make_shared<ngraph::opset7::Relu>(inputs.at(0));
}

std::shared_ptr<ngraph::Node> parseNode(IE::ConvolutionOp origOp, ngraph::OutputVector &inputs)
{
    const auto strides = parseIntArrayAttr<size_t>(origOp.stridesAttr());
    const auto pads_begin = parseIntArrayAttr<std::ptrdiff_t>(origOp.pads_begin());
    const auto pads_end = parseIntArrayAttr<std::ptrdiff_t>(origOp.pads_end());
    const auto dilations = parseIntArrayAttr<size_t>(origOp.dilations());
    return std::make_shared<ngraph::opset7::Convolution>(inputs.at(0), inputs.at(1), ngraph::Strides(strides.begin(),strides.end()),
        ngraph::CoordinateDiff(pads_begin.begin(), pads_begin.end()), ngraph::CoordinateDiff(pads_end.begin(), pads_end.end()),
        ngraph::Strides(dilations.begin(), dilations.end()), ngraph::op::PadType::SAME_UPPER);
}

std::shared_ptr<ngraph::Node> parseNode(IE::GroupConvolutionOp origOp, ngraph::OutputVector &inputs)
{
    const auto strides = parseIntArrayAttr<size_t>(origOp.stridesAttr());
    const auto pads_begin = parseIntArrayAttr<std::ptrdiff_t>(origOp.pads_begin());
    const auto pads_end = parseIntArrayAttr<std::ptrdiff_t>(origOp.pads_end());
    const auto dilations = parseIntArrayAttr<size_t>(origOp.dilations());
    return std::make_shared<ngraph::opset7::GroupConvolution>(inputs.at(0), inputs.at(1), ngraph::Strides(strides.begin(),strides.end()),
        ngraph::CoordinateDiff(pads_begin.begin(), pads_begin.end()), ngraph::CoordinateDiff(pads_end.begin(), pads_end.end()),
        ngraph::Strides(dilations.begin(), dilations.end()), ngraph::op::PadType::SAME_UPPER);
}

std::shared_ptr<ngraph::Node> parseNode(IE::AvgPoolOp origOp, ngraph::OutputVector &inputs)
{
    const auto strides = parseIntArrayAttr<size_t>(origOp.stridesAttr());
    const auto pads_begin = parseIntArrayAttr<size_t>(origOp.pads_begin());
    const auto pads_end = parseIntArrayAttr<size_t>(origOp.pads_end());
    const auto kernel = parseIntArrayAttr<size_t>(origOp.kernel_size());
    const auto exclude_pads = origOp.exclude_pads();
    const auto rounding_type = exportRoundingType(origOp.rounding_type());
    return std::make_shared<ngraph::opset7::AvgPool>(inputs.at(0), ngraph::Strides(strides.begin(),strides.end()),
        ngraph::Shape(pads_begin.begin(), pads_begin.end()), ngraph::Shape(pads_end.begin(), pads_end.end()),
        ngraph::Shape(kernel.begin(), kernel.end()), exclude_pads, rounding_type);
}

std::shared_ptr<ngraph::Node> parseNode(IE::MaxPoolOp origOp, ngraph::OutputVector &inputs)
{
    const auto strides = parseIntArrayAttr<size_t>(origOp.stridesAttr());
    const auto pads_begin = parseIntArrayAttr<size_t>(origOp.pads_begin());
    const auto pads_end = parseIntArrayAttr<size_t>(origOp.pads_end());
    const auto kernel = parseIntArrayAttr<size_t>(origOp.kernel_size());
    const auto rounding_type = exportRoundingType(origOp.rounding_type());
    return std::make_shared<ngraph::opset7::MaxPool>(inputs.at(0), ngraph::Strides(strides.begin(), strides.end()),
        ngraph::Shape(pads_begin.begin(), pads_begin.end()), ngraph::Shape(pads_end.begin(), pads_end.end()),
        ngraph::Shape(kernel.begin(), kernel.end()), rounding_type);
}

std::shared_ptr<ngraph::Node> parseNode(IE::GatherOp, ngraph::OutputVector &inputs)
{
    return std::make_shared<ngraph::opset7::Gather>(inputs.at(0), inputs.at(1), inputs.at(2));
}

std::shared_ptr<ngraph::Node> parseNode(IE::ClampOp origOp, ngraph::OutputVector &inputs)
{
    auto min = origOp.min();
    auto max = origOp.max();
    return std::make_shared<ngraph::opset7::Clamp>(inputs.at(0), min.convertToDouble(), max.convertToDouble());
}

std::shared_ptr<ngraph::Node> parseNode(IE::EluOp origOp, ngraph::OutputVector &inputs)
{
    auto x = origOp.x();
    return std::make_shared<ngraph::opset7::Elu>(inputs.at(0), x.convertToDouble());
}

std::shared_ptr<ngraph::Node> parseNode(IE::ReshapeOp origOp, ngraph::OutputVector &inputs)
{
    auto special_zero = origOp.special_zero();
    return std::make_shared<ngraph::opset7::Reshape>(inputs.at(0), inputs.at(1), special_zero);
}

std::shared_ptr<ngraph::Node> parseNode(IE::SqueezeOp, ngraph::OutputVector &inputs)
{
    return std::make_shared<ngraph::opset7::Squeeze>(inputs.at(0), inputs.at(1));
}

std::shared_ptr<ngraph::Node> parseNode(IE::SigmoidOp, ngraph::OutputVector &inputs)
{
    return std::make_shared<ngraph::opset7::Sigmoid>(inputs.at(0));
}

std::shared_ptr<ngraph::Node> parseNode(IE::LRNOp origOp, ngraph::OutputVector &inputs)
{
    auto alpha = origOp.alpha().convertToDouble();
    auto beta = origOp.beta().convertToDouble();
    auto bias = origOp.bias().convertToDouble();
    auto size = origOp.size();
    return std::make_shared<ngraph::opset7::LRN>(inputs.at(0), inputs.at(1), alpha, beta, bias, size);
}

std::shared_ptr<ngraph::Node> parseNode(IE::UnsqueezeOp, ngraph::OutputVector &inputs)
{
    return std::make_shared<ngraph::opset7::Unsqueeze>(inputs.at(0), inputs.at(1));
}

std::shared_ptr<ngraph::Node> parseNode(IE::MinimumOp origOp, ngraph::OutputVector &inputs)
{
    ngraph::op::AutoBroadcastType autoBroadCastType = exportBroadcastType(origOp.auto_broadcast());

    return std::make_shared<ngraph::opset7::Minimum>(inputs.at(0), inputs.at(1), ngraph::op::AutoBroadcastSpec(autoBroadCastType));
}

std::shared_ptr<ngraph::Node> parseNode(IE::MaximumOp origOp, ngraph::OutputVector &inputs)
{
    ngraph::op::AutoBroadcastType autoBroadCastType = exportBroadcastType(origOp.auto_broadcast());

    return std::make_shared<ngraph::opset7::Maximum>(inputs.at(0), inputs.at(1), ngraph::op::AutoBroadcastSpec(autoBroadCastType));
}

std::shared_ptr<ngraph::Node> parseNode(IE::AddOp origOp, ngraph::OutputVector &inputs)
{
    ngraph::op::AutoBroadcastType autoBroadCastType = exportBroadcastType(origOp.auto_broadcast());
    return std::make_shared<ngraph::opset7::Add>(inputs.at(0), inputs.at(1), ngraph::op::AutoBroadcastSpec(autoBroadCastType));
}

std::shared_ptr<ngraph::Node> parseNode(IE::DivideOp origOp, ngraph::OutputVector &inputs)
{
    ngraph::op::AutoBroadcastType autoBroadCastType = exportBroadcastType(origOp.auto_broadcast());
    return std::make_shared<ngraph::opset7::Divide>(inputs.at(0), inputs.at(1), ngraph::op::AutoBroadcastSpec(autoBroadCastType));
}

std::shared_ptr<ngraph::Node> parseNode(IE::SquaredDifferenceOp origOp, ngraph::OutputVector &inputs)
{
    ngraph::op::AutoBroadcastType autoBroadCastType = exportBroadcastType(origOp.auto_broadcast());
    return std::make_shared<ngraph::opset7::SquaredDifference>(inputs.at(0), inputs.at(1), ngraph::op::AutoBroadcastSpec(autoBroadCastType));
}

std::shared_ptr<ngraph::Node> parseNode(IE::FloorModOp origOp, ngraph::OutputVector &inputs)
{
    ngraph::op::AutoBroadcastType autoBroadCastType = exportBroadcastType(origOp.auto_broadcast());
    return std::make_shared<ngraph::opset7::FloorMod>(inputs.at(0), inputs.at(1), ngraph::op::AutoBroadcastSpec(autoBroadCastType));
}

std::shared_ptr<ngraph::Node> parseNode(IE::ProposalOp origOp, ngraph::OutputVector &inputs)
{
    const auto attrs = origOp.proposal_attrs();
    return std::make_shared<ngraph::opset7::Proposal>(inputs.at(0), inputs.at(1), inputs.at(2), exportProposalAttrs(attrs));
}

std::shared_ptr<ngraph::Node> parseNode(IE::FakeQuantizeOp origOp, ngraph::OutputVector &inputs)
{
    auto levels = origOp.levels();
    ngraph::op::AutoBroadcastType autoBroadCastType = exportBroadcastType(origOp.auto_broadcast());

    return std::make_shared<ngraph::opset7::FakeQuantize>(inputs.at(0), inputs.at(1), inputs.at(2), inputs.at(3),
        inputs.at(4), levels, ngraph::op::AutoBroadcastSpec(autoBroadCastType));
}

std::shared_ptr<ngraph::Node> parseNode(IE::MatMulOp origOp, ngraph::OutputVector &inputs)
{
    const auto transpose_a = origOp.transpose_a();
    const auto transpose_b = origOp.transpose_b();

    return std::make_shared<ngraph::opset7::MatMul>(inputs.at(0), inputs.at(1), transpose_a, transpose_b);
}

std::shared_ptr<ngraph::Node> parseNode(IE::TanhOp, ngraph::OutputVector &inputs)
{
    return std::make_shared<ngraph::opset7::Tanh>(inputs.at(0));
}

std::shared_ptr<ngraph::Node> parseNode(IE::ExpOp, ngraph::OutputVector &inputs)
{
    return std::make_shared<ngraph::opset7::Exp>(inputs.at(0));
}

std::shared_ptr<ngraph::Node> parseNode(IE::HSwishOp, ngraph::OutputVector &inputs)
{
    return std::make_shared<ngraph::opset7::HSwish>(inputs.at(0));
}

std::shared_ptr<ngraph::Node> parseNode(IE::FloorOp, ngraph::OutputVector &inputs)
{
    return std::make_shared<ngraph::opset7::Floor>(inputs.at(0));
}

std::shared_ptr<ngraph::Node> parseNode(IE::RoundOp origOp, ngraph::OutputVector &inputs)
{
    const auto mode = origOp.mode();
    return std::make_shared<ngraph::opset7::Round>(inputs.at(0), exportRoundMode(mode));
}

std::shared_ptr<ngraph::Node> parseNode(IE::MishOp, ngraph::OutputVector &inputs)
{
    return std::make_shared<ngraph::opset7::Mish>(inputs.at(0));
}

std::shared_ptr<ngraph::Node> parseNode(IE::ErfOp, ngraph::OutputVector &inputs)
{
    return std::make_shared<ngraph::opset7::Erf>(inputs.at(0));
}

std::shared_ptr<ngraph::Node> parseNode(IE::TransposeOp, ngraph::OutputVector &inputs)
{
    return std::make_shared<ngraph::opset7::Transpose>(inputs.at(0), inputs.at(1));
}

std::shared_ptr<ngraph::Node> parseNode(IE::InterpolateOp origOp, ngraph::OutputVector &inputs)
{
    ngraph::opset7::Interpolate::InterpolateAttrs attrs = exportInterpolateAttrs(origOp.attr());
    return std::make_shared<ngraph::opset7::Interpolate>(inputs.at(0), inputs.at(1), inputs.at(2), inputs.at(3), attrs);
}

std::shared_ptr<ngraph::Node> parseNode(IE::TopKOp origOp, ngraph::OutputVector &inputs)
{
    //ngraph::opset7::Interpolate::InterpolateAttrs attrs = exportInterpolateAttrs(origOp.attr());
    const auto axis = origOp.axis();
    mlir::Type elType = origOp.element_type();
    mlir::MLIRContext* ctx = elType.getContext();

    return std::make_shared<ngraph::opset7::TopK>(inputs.at(0), inputs.at(1), axis, exportTopKMode(origOp.mode()),
        exportTopKSortType(origOp.sort()), exportElemType(ctx, elType));
}

std::shared_ptr<ngraph::Node> parseNode(IE::RegionYoloOp origOp, ngraph::OutputVector &inputs)
{
    const auto coords = origOp.coords();
    const auto classes = origOp.classes();
    const auto regions = origOp.regions();
    const auto do_softmax = origOp.do_softmax();
    const auto mask = parseIntArrayAttr<int64_t>(origOp.mask());
    const auto axis = origOp.axis();
    const auto end_axis = origOp.end_axis();
    const auto anchors = parseFPArrayAttr<float>(origOp.anchors());

    return std::make_shared<ngraph::opset7::RegionYolo>(inputs.at(0), coords, classes, regions, do_softmax,
        std::vector<int64_t>{mask.begin(),mask.end()}, axis, end_axis, std::vector<float>{anchors.begin(),anchors.end()});
}

std::shared_ptr<ngraph::Node> parseNode(IE::ReorgYoloOp origOp, ngraph::OutputVector &inputs)
{
    return std::make_shared<ngraph::opset7::ReorgYolo>(inputs.at(0), origOp.stride());
}

std::shared_ptr<ngraph::Node> parseNode(IE::DetectionOutputOp origOp, ngraph::OutputVector &inputs)
{
    if (inputs.size() == 3)
        return std::make_shared<ngraph::opset7::DetectionOutput>(inputs.at(0), inputs.at(1), inputs.at(2),
            exportDetectionOutputAttrs(origOp.attr()));
    else if (inputs.size() == 5)
        return std::make_shared<ngraph::opset7::DetectionOutput>(inputs.at(0), inputs.at(1), inputs.at(2),
            inputs.at(3), inputs.at(4), exportDetectionOutputAttrs(origOp.attr()));
    else
        VPUX_THROW("IE::DetectionOutputOp has unsupported number of inputs '{0}'", inputs.size());
}

std::shared_ptr<ngraph::Node> parseNode(IE::NormalizeL2Op origOp, ngraph::OutputVector &inputs)
{
    const auto eps = origOp.eps().convertToDouble();
    const auto eps_mode = exportEpsMode(origOp.eps_mod());

    return std::make_shared<ngraph::opset7::NormalizeL2>(inputs.at(0), inputs.at(1), eps, eps_mode);
}

std::shared_ptr<ngraph::Node> parseNode(IE::MVNOp origOp, ngraph::OutputVector &inputs)
{
    const auto across_channels = origOp.across_channels();
    const auto normalize_variance = origOp.normalize_variance();
    const auto eps = origOp.eps().convertToDouble();

    return std::make_shared<ngraph::opset4::MVN>(inputs.at(0), across_channels, normalize_variance, eps);
}

std::shared_ptr<ngraph::Node> parseNode(IE::ConcatOp origOp, ngraph::OutputVector &inputs)
{
    auto axis = origOp.axis();
    return std::make_shared<ngraph::opset7::Concat>(ngraph::OutputVector{inputs.at(0), inputs.at(1)}, axis);
}

std::shared_ptr<ngraph::Node> parseNode(IE::ROIPoolingOp origOp, ngraph::OutputVector &inputs)
{
    const auto output_size = parseIntArrayAttr<size_t>(origOp.output_size());
    const auto spatial_scale = origOp.spatial_scale().convertToFloat();
    const auto method = exportROIPoolingMethod(origOp.method());
    return std::make_shared<ngraph::opset7::ROIPooling>(inputs.at(0), inputs.at(1),
        ngraph::Shape(output_size.begin(), output_size.end()), spatial_scale, method);
}

std::shared_ptr<ngraph::Node> parseNode(IE::StridedSliceOp origOp, ngraph::OutputVector &inputs)
{
    const auto begin_mask = parseIntArrayAttr<int64_t>(origOp.begin_mask());
    const auto end_mask = parseIntArrayAttr<int64_t>(origOp.end_mask());
    const auto new_axis_mask = parseIntArrayAttr<int64_t>(origOp.new_axis_mask());
    const auto shrink_axis_mask = parseIntArrayAttr<int64_t>(origOp.shrink_axis_mask());
    const auto ellipsis_mask = parseIntArrayAttr<int64_t>(origOp.ellipsis_mask());

    return std::make_shared<ngraph::opset7::StridedSlice>(inputs.at(0), inputs.at(1), inputs.at(2), inputs.at(3),
        std::vector<int64_t>{begin_mask.begin(), begin_mask.end()},
        std::vector<int64_t>{end_mask.begin(), end_mask.end()},
        std::vector<int64_t>{new_axis_mask.begin(), new_axis_mask.end()},
        std::vector<int64_t>{shrink_axis_mask.begin(), shrink_axis_mask.end()},
        std::vector<int64_t>{ellipsis_mask.begin(), ellipsis_mask.end()});
}

std::shared_ptr<ngraph::Node> parseNode(IE::PReluOp, ngraph::OutputVector &inputs)
{
    return std::make_shared<ngraph::opset7::PRelu>(inputs.at(0), inputs.at(1));
}

std::shared_ptr<ngraph::Node> parseNode(IE::SwishOp, ngraph::OutputVector &inputs)
{
    return std::make_shared<ngraph::opset7::Swish>(inputs.at(0), inputs.at(1));
}

std::shared_ptr<ngraph::Node> parseNode(IE::NegativeOp, ngraph::OutputVector &inputs)
{
    return std::make_shared<ngraph::opset7::Negative>(inputs.at(0));
}

std::shared_ptr<ngraph::Node> parseNode(IE::CTCGreedyDecoderOp origOp, ngraph::OutputVector &inputs)
{
    const auto mergeRepeated = origOp.mergeRepeated();

    return std::make_shared<ngraph::opset7::CTCGreedyDecoder>(inputs.at(0), inputs.at(1), mergeRepeated);
}

std::shared_ptr<ngraph::Node> parseNode(IE::CTCGreedyDecoderSeqLenOp origOp, ngraph::OutputVector &inputs)
{
    const auto mergeRepeated = origOp.mergeRepeated();

    return std::make_shared<ngraph::opset7::CTCGreedyDecoderSeqLen>(inputs.at(0), inputs.at(1), inputs.at(2), mergeRepeated);
}

std::shared_ptr<ngraph::Node> parseNode(IE::PadOp origOp, ngraph::OutputVector &inputs)
{
    const auto pad_mode = origOp.mode();
    if (inputs.size() == 4)
        return std::make_shared<ngraph::opset7::Pad>(inputs.at(0), inputs.at(1), inputs.at(2), inputs.at(3),
        exportPadMode(pad_mode));
    else if (inputs.size() == 3)
        return std::make_shared<ngraph::opset7::Pad>(inputs.at(0), inputs.at(1), inputs.at(2),
        exportPadMode(pad_mode));
    else
        VPUX_THROW("IE::PadOp has unsupported number of inputs '{0}'", inputs.size());
}

std::shared_ptr<ngraph::Node> parseNode(IE::LSTMCellOp origOp, ngraph::OutputVector &inputs)
{
    const auto hidden_size = origOp.hiddenSize();
    return std::make_shared<ngraph::opset7::LSTMCell>(inputs.at(0), inputs.at(1), inputs.at(2), inputs.at(3),
        inputs.at(4), inputs.at(5), hidden_size);
}

std::shared_ptr<ngraph::Node> parseNode(IE::SubtractOp origOp, ngraph::OutputVector &inputs)
{
    ngraph::op::AutoBroadcastType autoBroadCastType = exportBroadcastType(origOp.auto_broadcast());

    return std::make_shared<ngraph::opset7::Subtract>(inputs.at(0), inputs.at(1),
        ngraph::op::AutoBroadcastSpec(autoBroadCastType));
}

std::shared_ptr<ngraph::Node> parseNode(IE::LSTMSequenceOp origOp, ngraph::OutputVector &inputs)
{
    const auto hidden_size = origOp.sequenceLength();
    const auto lstm_direction = origOp.direction();

    return std::make_shared<ngraph::opset7::LSTMSequence>(inputs.at(0), inputs.at(1), inputs.at(2), inputs.at(3),
        inputs.at(4), inputs.at(5), inputs.at(6), hidden_size, exportRNNSequenceDirection(lstm_direction));
}

std::shared_ptr<ngraph::Node> parseNode(IE::AndOp origOp, ngraph::OutputVector &inputs)
{
    ngraph::op::AutoBroadcastType autoBroadCastType = exportBroadcastType(origOp.auto_broadcast());

    return std::make_shared<ngraph::opset7::LogicalAnd>(inputs.at(0), inputs.at(1),
        ngraph::op::AutoBroadcastSpec(autoBroadCastType));
}

std::shared_ptr<ngraph::Node> parseNode(mlir::ReturnOp, ngraph::OutputVector &inputs)
{
    std::shared_ptr<ngraph::Node> ngNode = std::make_shared<ngraph::opset7::Result>(inputs.at(0));
    return ngNode;
}

template <class NodeType>
std::shared_ptr<ngraph::Node> parseDispatch(mlir::Operation *origOp, ngraph::OutputVector &inputs) {
    return parseNode(llvm::dyn_cast<NodeType>(*origOp), inputs);
}

std::shared_ptr<ngraph::Function> exportToNgraph(IE::CNNNetworkOp, mlir::FuncOp netFunc)
{
    using Callback = std::shared_ptr<ngraph::Node> (*)(mlir::Operation *origOp, ngraph::OutputVector &inputs);
    using DispatchMap = std::map<std::string, Callback>;

#define MAP_ENTRY(_OpName_, _OpType_) \
    { _OpName_, &parseDispatch<_OpType_> }

    static DispatchMap dispatchMap {
            MAP_ENTRY("const.Declare", Const::DeclareOp),
            MAP_ENTRY("IE.Convert", IE::ConvertOp),
            MAP_ENTRY("IE.SoftMax", IE::SoftMaxOp),
            MAP_ENTRY("IE.Tile", IE::TileOp),
            MAP_ENTRY("IE.Split", IE::SplitOp),
            MAP_ENTRY("IE.Power", IE::PowerOp),
            MAP_ENTRY("IE.Multiply", IE::MultiplyOp),
            MAP_ENTRY("IE.ReLU", IE::ReLUOp),
            MAP_ENTRY("IE.Convolution", IE::ConvolutionOp),
            MAP_ENTRY("IE.GroupConvolution", IE::GroupConvolutionOp),
            MAP_ENTRY("IE.AvgPool", IE::AvgPoolOp),
            MAP_ENTRY("IE.MaxPool", IE::MaxPoolOp),
            MAP_ENTRY("IE.Gather", IE::GatherOp),
            MAP_ENTRY("IE.Clamp", IE::ClampOp),
            MAP_ENTRY("IE.Elu", IE::EluOp),
            MAP_ENTRY("IE.Reshape", IE::ReshapeOp),
            MAP_ENTRY("IE.Squeeze", IE::SqueezeOp),
            MAP_ENTRY("IE.Sigmoid", IE::SigmoidOp),
            MAP_ENTRY("IE.LRN", IE::LRNOp),
            MAP_ENTRY("IE.Unsqueeze", IE::UnsqueezeOp),
            MAP_ENTRY("IE.Minimum", IE::MinimumOp),
            MAP_ENTRY("IE.Maximum", IE::MaximumOp),
            MAP_ENTRY("IE.Add", IE::AddOp),
            MAP_ENTRY("IE.Divide", IE::DivideOp),
            MAP_ENTRY("IE.SquaredDiff", IE::SquaredDifferenceOp),
            MAP_ENTRY("IE.FloorMod", IE::FloorModOp),
            MAP_ENTRY("IE.Proposal", IE::ProposalOp),
            MAP_ENTRY("IE.FakeQuantize", IE::FakeQuantizeOp),
            MAP_ENTRY("IE.MatMul", IE::MatMulOp),
            MAP_ENTRY("IE.Tanh", IE::TanhOp),
            MAP_ENTRY("IE.Exp", IE::ExpOp),
            MAP_ENTRY("IE.HSwish", IE::HSwishOp),
            MAP_ENTRY("IE.Floor", IE::FloorOp),
            MAP_ENTRY("IE.Round", IE::RoundOp),
            MAP_ENTRY("IE.Mish", IE::MishOp),
            MAP_ENTRY("IE.Erf", IE::ErfOp),
            MAP_ENTRY("IE.Transpose", IE::TransposeOp),
            MAP_ENTRY("IE.Interpolate", IE::InterpolateOp),
            MAP_ENTRY("IE.TopK", IE::TopKOp),
            MAP_ENTRY("IE.RegionYolo", IE::RegionYoloOp),
            MAP_ENTRY("IE.ReorgYolo", IE::ReorgYoloOp),
            MAP_ENTRY("IE.DetectionOutput", IE::DetectionOutputOp),
            MAP_ENTRY("IE.NormalizeL2", IE::NormalizeL2Op),
            MAP_ENTRY("IE.MVN", IE::MVNOp),
            MAP_ENTRY("IE.Concat", IE::ConcatOp),
            MAP_ENTRY("IE.ROIPooling", IE::ROIPoolingOp),
            MAP_ENTRY("IE.StridedSlice", IE::StridedSliceOp),
            MAP_ENTRY("IE.PRelu", IE::PReluOp),
            MAP_ENTRY("IE.Swish", IE::SwishOp),
            MAP_ENTRY("IE.Negative", IE::NegativeOp),
            MAP_ENTRY("IE.CTCGreedyDecoder", IE::CTCGreedyDecoderOp),
            MAP_ENTRY("IE.CTCGreedyDecoderSeqLen", IE::CTCGreedyDecoderSeqLenOp),
            MAP_ENTRY("IE.Pad", IE::PadOp),
            MAP_ENTRY("IE.LSTMCell", IE::LSTMCellOp),
            MAP_ENTRY("IE.Subtract", IE::SubtractOp),
            MAP_ENTRY("IE.And", IE::AndOp),
            MAP_ENTRY("IE.LSTMSequence", IE::LSTMSequenceOp),
            MAP_ENTRY("std.return", mlir::ReturnOp),
    };
#undef MAP_ENTRY

    llvm::raw_os_ostream os(std::cout);
    mlir::Block &block = *(netFunc.body().getBlocks().begin());
    std::shared_ptr<ngraph::Function> netGraph;
    ngraph::ParameterVector inputnodes;
    block.walk([&](mlir::Operation *op) {
        os << "visiting op: '" << op->getName() << "' with "
            << op->getNumOperands() << " operands and "
            << op->getNumResults() << " results\n";
        os.flush();
        static NodeMap _exportedNodes;
        std::shared_ptr<ngraph::Node> ngNode;
        ngraph::OutputVector inputs;
        for (auto val : op->getOperands())
        {
            mlir::Operation *sourceOp = val.getDefiningOp();
            if (sourceOp == nullptr)
            {
                auto type = val.getType();
                const mlir::RankedTensorType& rankedTensorType = type.dyn_cast<mlir::RankedTensorType>();
                InferenceEngine::TensorDesc tensor = exportUserTensor(rankedTensorType);
                ngraph::Shape ngShape{tensor.getDims().begin(), tensor.getDims().end()};
                std::shared_ptr<ngraph::opset7::Parameter> ngParam = 
                    std::make_shared<ngraph::opset7::Parameter>(toNGraphType(tensor.getPrecision()), ngShape);
                _exportedNodes.insert({nullptr, ngParam});
                inputs.push_back(ngParam);
                inputnodes.push_back(ngParam);
            }
            else
            {
                inputs.push_back(ngraph::Output<ngraph::Node>(_exportedNodes.at(sourceOp)));
            }
        }

        const auto dispatchIt = dispatchMap.find(op->getName().getStringRef().str());
        VPUX_THROW_UNLESS(dispatchIt != dispatchMap.end(), "Unsupported operation {0}", op->getName().getStringRef().str());

        const auto parser = dispatchIt->second;
        ngNode = (*parser)(op, inputs);
        _exportedNodes.insert({op, ngNode});
        if (ngraph::is_type<ngraph::opset7::Result>(ngNode))
            netGraph = std::make_shared<ngraph::Function>(ngraph::OutputVector{ngNode}, inputnodes);

    });
    return netGraph;
}

}  // namespace

//
// exportToIRv10
//

mlir::LogicalResult vpux::IE::exportToIRv10(mlir::ModuleOp module, llvm::raw_ostream& output, const std::string &filePath) {
    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);

    std::shared_ptr<ngraph::Function> netGraph = exportToNgraph(netOp, netFunc);
    InferenceEngine::CNNNetwork ieNet(netGraph);

    std::ofstream binFile(provide_bin_path(filePath), std::ios::out | std::ios::binary);
    if (!binFile)
        VPUX_THROW("Unable to open weights file for writing");
    std::ostringstream ostr;
    ieNet.serialize(ostr, binFile);
    output << ostr.str();
    return mlir::success();
}
