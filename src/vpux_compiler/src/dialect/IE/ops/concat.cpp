//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/dialect/IE/utils/propagate_quantize_dequantize_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include <map>
#include <unordered_set>

using namespace vpux;

//
// build
//

void vpux::IE::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange inputs,
                               ConcatAttr per_axis) {
    build(builder, state, inputs, per_axis, nullptr);
}

void vpux::IE::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange inputs,
                               mlir::IntegerAttr axis, mlir::IntegerAttr offset, mlir::IntegerAttr stride) {
    build(builder, state, inputs, ConcatAttr::get(builder.getContext(), axis, offset, stride));
}

void vpux::IE::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange inputs,
                               int64_t axis, int64_t offset, int64_t stride) {
    build(builder, state, inputs, getIntAttr(builder, axis), offset != 0 ? getIntAttr(builder, offset) : nullptr,
          stride != 1 ? getIntAttr(builder, stride) : nullptr);
}

void vpux::IE::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange inputs, Dim axis,
                               int64_t offset, int64_t stride) {
    build(builder, state, inputs, axis.ind(), offset, stride);
}

void vpux::IE::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type outType,
                               mlir::ValueRange inputs, mlir::ArrayAttr static_offsets) {
    build(builder, state, outType, inputs, nullptr, static_offsets);
}

void vpux::IE::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type outType,
                               mlir::ValueRange inputs, ArrayRef<Shape> static_offsets) {
    const auto attrArr = to_small_vector(static_offsets | transformed([&](ShapeRef arr) -> mlir::Attribute {
                                             return getIntArrayAttr(builder, arr);
                                         }));

    build(builder, state, outType, inputs, builder.getArrayAttr(attrArr));
}

void vpux::IE::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type outType,
                               mlir::ValueRange inputs, ArrayRef<ShapeRef> static_offsets) {
    const auto attrArr = to_small_vector(static_offsets | transformed([&](ShapeRef arr) -> mlir::Attribute {
                                             return getIntArrayAttr(builder, arr);
                                         }));

    build(builder, state, outType, inputs, builder.getArrayAttr(attrArr));
}

//
// inferReturnTypeComponents
//

namespace {

Dim normalizeAxis(IE::ConcatOpAdaptor concat) {
    const auto inType = concat.inputs().front().getType().cast<mlir::ShapedType>();
    const auto inRank = inType.getRank();

    auto axisInd = concat.per_axis().value().getAxis().getValue().getSExtValue();

    // Negative value means counting dimension from the end
    if (axisInd < 0) {
        axisInd += inRank;
    }

    VPUX_THROW_UNLESS(axisInd >= 0 && axisInd < inRank, "Got wrong Concat axis '{0}', out of range '{1}'", axisInd,
                      inRank);

    return Dim(axisInd);
}

mlir::FailureOr<Shape> inferOutShapeWithAxis(IE::ConcatOpAdaptor concat, mlir::Location loc) {
    const auto inType = concat.inputs().front().getType().cast<vpux::NDTypeInterface>();
    const auto axis = normalizeAxis(concat);

    auto outShape = inType.getShape().toValues();

    for (const auto val : concat.inputs().drop_front()) {
        const auto curShape = getShape(val);

        if (curShape.size() != outShape.size()) {
            return errorAt(loc, "Concat inputs have mismatched ranks: '{0}' vs '{1}'", curShape.size(),
                           outShape.size());
        }

        outShape[axis] += curShape[axis];
    }

    const auto perAxis = concat.per_axis().value();
    const auto offset = perAxis.getOffset() ? perAxis.getOffset().getValue().getSExtValue() : 0;
    const auto stride = perAxis.getStride() ? perAxis.getStride().getValue().getSExtValue() : 1;

    int64_t maxLatestIdx = -1;
    for (const auto idx : irange(concat.inputs().size())) {
        const auto curShape = getShape(concat.inputs()[idx]);
        const int64_t sizeByAxis = curShape[axis];
        const int64_t latestElemIdx = offset * idx + (sizeByAxis > 0 ? stride * (sizeByAxis - 1) : 0);
        maxLatestIdx = std::max(maxLatestIdx, latestElemIdx);
    }

    if (maxLatestIdx >= outShape[axis]) {
        return errorAt(loc, "Concat with offset '{0}' and stride '{1}' doesn't fit to output dimension '{2}'", offset,
                       stride, outShape[axis]);
    }

    return outShape;
}

mlir::FailureOr<Shape> inferOutShapeWithOffsets(IE::ConcatOpAdaptor concat, mlir::Location loc) {
    if (!concat.static_offsets().has_value()) {
        return errorAt(loc, "Missing static_offsets attribute");
    }

    const auto staticOffsets = concat.static_offsets().value();
    if (staticOffsets.size() != concat.inputs().size()) {
        return errorAt(loc, "Concat 'static_offsets' count '{0}' doesn't match inputs count '{1}'",
                       staticOffsets.size(), concat.inputs().size());
    }

    const auto inType = concat.inputs().front().getType().cast<vpux::NDTypeInterface>();
    const auto allOffsets = staticOffsets.getAsRange<mlir::ArrayAttr>();

    Shape outShape(checked_cast<size_t>(inType.getRank()), 0);

    for (const auto p : zip(concat.inputs(), allOffsets)) {
        const auto curVal = std::get<0>(p);
        const auto curShape = getShape(curVal);

        if (curShape.size() != outShape.size()) {
            return errorAt(loc, "Concat inputs have mismatched ranks: '{0}' vs '{1}'", curShape.size(),
                           outShape.size());
        }

        const auto curOffsets = Shape(parseIntArrayAttr<int64_t>(std::get<1>(p)));

        if (curOffsets.size() != curShape.size()) {
            return errorAt(loc, "Concat 'static_offsets' rank doesn't match its input");
        }

        for (const auto ind : irange(outShape.size())) {
            const auto d = Dim(ind);

            outShape[d] = std::max(outShape[d], curOffsets[d] + curShape[d]);
        }
    }

    // TODO: validate that inputs+static_offsets fully covers the output without intersections

    return outShape;
}

mlir::FailureOr<mlir::Type> inferOutElemTypeWithAxis(ArrayRef<mlir::Type> elemTypes, IE::ConcatOpAdaptor concat,
                                                     LogCb logCb = emptyLogCb) {
    const auto inElemType = elemTypes[0];

    const auto perAxisQType = inElemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>();
    SmallVector<mlir::quant::UniformQuantizedPerAxisType> inPerAxisQTypes;

    if (perAxisQType != nullptr) {
        const auto axis = normalizeAxis(concat);

        if (perAxisQType.getQuantizedDimension() == axis.ind()) {
            inPerAxisQTypes.push_back(perAxisQType);
        }
    }

    for (const auto curElemType : elemTypes.drop_front()) {
        if (inPerAxisQTypes.empty()) {
            if (curElemType != inElemType) {
                logCb(formatv("Misaligned element types : '{0}' vs '{1}'", curElemType, inElemType));
                return mlir::failure();
            }
        } else {
            const auto curPerAxisQType = curElemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>();

            if (curPerAxisQType == nullptr) {
                logCb(formatv("Misaligned element types : not all of them are per-axis quantized : '{0}' vs '{1}'",
                              curElemType, inElemType));
                return mlir::failure();
            }

            if (curPerAxisQType.getQuantizedDimension() != perAxisQType.getQuantizedDimension()) {
                logCb(formatv(
                        "Misaligned element types : per-axis quantization is done on different axis : '{0}' vs '{1}'",
                        curPerAxisQType.getQuantizedDimension(), perAxisQType.getQuantizedDimension()));
                return mlir::failure();
            }

            if (!canBeMerged(curPerAxisQType, perAxisQType)) {
                logCb(formatv("Misaligned element types : per-axis quantization parameters can't be merged"));
                return mlir::failure();
            }

            inPerAxisQTypes.push_back(curPerAxisQType);
        }
    }

    return inPerAxisQTypes.empty() ? inElemType : concatScalesAndZP(inPerAxisQTypes);
}

mlir::FailureOr<mlir::Type> inferOutElemTypeWithAxis(IE::ConcatOpAdaptor concat, mlir::Location loc) {
    SmallVector<mlir::Type> types;
    const auto getElemTypeFromValue = [](mlir::Value operand) {
        return operand.getType().cast<vpux::NDTypeInterface>().getElementType();
    };
    std::transform(concat.getOperands().begin(), concat.getOperands().end(), std::back_inserter(types),
                   getElemTypeFromValue);

    const auto logCb = [loc](const formatv_object_base& msg) {
        std::ignore = errorAt(loc, "{0}", msg.str());
    };

    return inferOutElemTypeWithAxis(types, concat, logCb);
}

std::unordered_set<Dim> getConcatAxesFromOffsets(IE::ConcatOpAdaptor concat, ShapeRef outShape) {
    std::unordered_set<Dim> res;

    for (const auto inVal : concat.inputs()) {
        const auto curShape = getShape(inVal);

        for (const auto ind : irange(outShape.size())) {
            const auto d = Dim(ind);

            if (curShape[d] != outShape[d]) {
                res.insert(d);
            }
        }
    }

    return res;
}

mlir::FailureOr<mlir::Type> inferOutElemTypeWithOffsets(ArrayRef<mlir::Type> elemTypes, IE::ConcatOpAdaptor concat,
                                                        ShapeRef outShape, LogCb logCb = emptyLogCb) {
    const auto inElemType = elemTypes[0];

    const auto perAxisQType = inElemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>();

    const auto isConcatOverPerAxisQuantization = [&]() {
        if (perAxisQType == nullptr) {
            return false;
        }

        const auto qDim = Dim(perAxisQType.getQuantizedDimension());
        const auto concatAxes = getConcatAxesFromOffsets(concat, outShape);

        return concatAxes.count(qDim) != 0;
    }();

    if (!isConcatOverPerAxisQuantization) {
        for (const auto curElemType : elemTypes.drop_front()) {
            if (curElemType != inElemType) {
                logCb(formatv("Misaligned element types : '{0}' vs '{1}'", curElemType, inElemType));
                return mlir::failure();
            }
        }

        return inElemType;
    }

    const auto qDim = perAxisQType.getQuantizedDimension();
    const auto allOffsets = concat.static_offsets().value().getAsRange<mlir::ArrayAttr>();

    std::map<int64_t, mlir::quant::UniformQuantizedPerAxisType> perSliceQuantTypes;

    for (const auto p : zip(elemTypes, allOffsets)) {
        const auto curElemType = std::get<0>(p);
        const auto curPerAxisQType = curElemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>();

        if (curPerAxisQType == nullptr) {
            logCb(formatv("Misaligned element types : not all of them are per-axis quantized : '{0}' vs '{1}'",
                          curElemType, inElemType));
            return mlir::failure();
        }

        if (curPerAxisQType.getQuantizedDimension() != qDim) {
            logCb(formatv("Misaligned element types : per-axis quantization is done on different axis : '{0}' vs '{1}'",
                          curPerAxisQType.getQuantizedDimension(), qDim));
            return mlir::failure();
        }

        const auto curOffsets = parseIntArrayAttr<int64_t>(std::get<1>(p));
        const auto sliceOffset = curOffsets[checked_cast<size_t>(qDim)];

        const auto it = perSliceQuantTypes.find(sliceOffset);
        if (it == perSliceQuantTypes.end()) {
            perSliceQuantTypes.insert({sliceOffset, curPerAxisQType});
        } else {
            if (curPerAxisQType != it->second) {
                logCb(formatv("Per-axis quantization is not aligned over non quantized axis : '{0}' vs '{1}'",
                              curPerAxisQType, it->second));
                return mlir::failure();
            }
        }
    }

    const auto inPerAxisQTypes = to_small_vector(perSliceQuantTypes | map_values);
    return concatScalesAndZP(inPerAxisQTypes);
}

mlir::FailureOr<mlir::Type> inferOutElemTypeWithOffsets(IE::ConcatOpAdaptor concat, ShapeRef outShape,
                                                        mlir::Location loc) {
    SmallVector<mlir::Type> types;
    const auto getElemTypeFromValue = [](mlir::Value operand) {
        return operand.getType().cast<vpux::NDTypeInterface>().getElementType();
    };
    std::transform(concat.getOperands().begin(), concat.getOperands().end(), std::back_inserter(types),
                   getElemTypeFromValue);

    const auto logCb = [loc](const formatv_object_base& msg) {
        std::ignore = errorAt(loc, "{0}", msg.str());
    };

    return inferOutElemTypeWithOffsets(types, concat, outShape, logCb);
}

}  // namespace

mlir::LogicalResult vpux::IE::ConcatOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ConcatOpAdaptor concat(operands, attrs);
    if (mlir::failed(concat.verify(loc))) {
        return mlir::failure();
    }

    if (concat.inputs().empty()) {
        return errorAt(loc, "Missing inputs for '{0}'", IE::ConcatOp::getOperationName());
    }

    if (!concat.per_axis().has_value() && !concat.static_offsets().has_value()) {
        return errorAt(loc, "Missing either 'per_axis' or 'static_offsets' attribute");
    }
    if (concat.per_axis().has_value() && concat.static_offsets().has_value()) {
        return errorAt(loc, "Only one attribute ('per_axis' or 'static_offsets') should be provided");
    }

    const auto inType = concat.inputs().front().getType().cast<mlir::RankedTensorType>();

    // Check consistent tensor attributes

    const auto inDesc = vpux::getTensorAttr(inType);

    for (const auto val : concat.inputs().drop_front()) {
        const auto curType = val.getType().cast<mlir::RankedTensorType>();
        const auto curDesc = vpux::getTensorAttr(curType);

        if (curDesc != inDesc) {
            return errorAt(loc, "Misaligned TensorType attributes for '{0}' inputs", IE::ConcatOp::getOperationName());
        }
    }

    // Infer output shape

    const auto outShape =
            concat.per_axis() ? inferOutShapeWithAxis(concat, loc) : inferOutShapeWithOffsets(concat, loc);
    if (mlir::failed(outShape)) {
        return mlir::failure();
    }

    // Infer output element type

    const auto outElemType = concat.per_axis() ? inferOutElemTypeWithAxis(concat, loc)
                                               : inferOutElemTypeWithOffsets(concat, outShape.value(), loc);
    if (mlir::failed(outElemType)) {
        return mlir::failure();
    }

    // Return inferred components

    inferredReturnShapes.emplace_back(outShape.value().raw(), outElemType.value(), inDesc);
    return mlir::success();
}

//
// inferElemTypeInfo
//

void vpux::IE::ConcatOp::inferElemTypeInfo(vpux::IE::LayerDataInfo<mlir::Type>& info) {
    auto loc = getLoc();
    IE::ConcatOpAdaptor concat(getOperands(), getOperation()->getAttrDictionary());

    mlir::FailureOr<mlir::Type> outElemType;
    if (!concat.per_axis()) {
        const auto outShape = inferOutShapeWithOffsets(concat, loc);
        if (mlir::failed(outShape)) {
            return;
        }

        outElemType = inferOutElemTypeWithOffsets(info.getInputs(), concat, outShape.value());
    } else {
        outElemType = inferOutElemTypeWithAxis(info.getInputs(), concat);
    }

    if (mlir::failed(outElemType)) {
        return;
    }

    info.setOutput(0, outElemType.value());
}

void vpux::IE::ConcatOp::inferElemTypeInfoUp(vpux::IE::LayerDataInfo<mlir::Type>& info) {
    // E#84659: implement propagate type up for per channel, currently it leads to failures in later passes.
    propagateElementTypeUp(info);
}

//
// ConvertPerAxisToOffsets
//

namespace {

class ConvertPerAxisToOffsets final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    using mlir::OpRewritePattern<IE::ConcatOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;
};

const mlir::ArrayAttr inferOffsetsAttrWithAxis(IE::ConcatOp origOp, int64_t& axis) {
    auto rank = origOp.output().getType().cast<vpux::NDTypeInterface>().getRank();

    SmallVector<SmallVector<int64_t>> finalOffsets;
    finalOffsets.push_back(SmallVector<int64_t>(rank, 0));
    int64_t correctAxis;
    if (axis < 0) {
        correctAxis = axis + rank;
    } else {
        correctAxis = axis;
    }
    for (auto input : origOp.inputs() | indexed) {
        auto inputShape = getShape(input.value());
        auto offsets = SmallVector<int64_t>(rank, 0);
        offsets[correctAxis] = inputShape[Dim(correctAxis)] + finalOffsets.back()[correctAxis];
        finalOffsets.push_back(offsets);
    }
    finalOffsets.pop_back();

    return getIntArrayOfArray(origOp.getContext(), finalOffsets);
}

mlir::LogicalResult ConvertPerAxisToOffsets::matchAndRewrite(IE::ConcatOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    if (origOp.static_offsetsAttr()) {
        return mlir::failure();
    }

    if (origOp.per_axisAttr().getStride() || origOp.per_axisAttr().getOffset()) {
        return mlir::failure();
    }

    const auto outType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    auto axis = origOp.per_axisAttr().getAxis().getValue().getSExtValue();
    auto rank = origOp.inputs().front().getType().cast<vpux::NDTypeInterface>().getRank();
    // Negative value means counting dimension from the end
    if (axis < 0) {
        axis += rank;
    }
    const auto finalOffsetsAttr = inferOffsetsAttrWithAxis(origOp, axis);

    rewriter.replaceOpWithNewOp<IE::ConcatOp>(origOp, outType, origOp.inputs(), finalOffsetsAttr);
    return mlir::success();
}

}  // namespace

//
// FuseConcat
//

namespace {

class FuseConcat final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(IE::ConcatOp op, mlir::PatternRewriter& rewriter) const final;
};

SmallVector<mlir::Value> getAllInputOp(IE::ConcatOp origOp, const std::unordered_set<Dim>& axis) {
    SmallVector<mlir::Value> inputOps;
    for (auto preOps : origOp.inputs()) {
        auto producerConcatOp = preOps.getDefiningOp<IE::ConcatOp>();

        if (producerConcatOp != nullptr && producerConcatOp.static_offsetsAttr()) {
            const auto subAxis = getConcatAxesFromOffsets(producerConcatOp, getShape(producerConcatOp.output()));
            if (subAxis == axis) {
                for (auto inputTensor : producerConcatOp.getInputs()) {
                    inputOps.emplace_back(inputTensor);
                }
                continue;
            }
        }

        inputOps.emplace_back(preOps);
    }
    return inputOps;
}

mlir::LogicalResult FuseConcat::matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const {
    if (origOp.per_axisAttr()) {
        return mlir::failure();
    }

    const auto axis = getConcatAxesFromOffsets(origOp, getShape(origOp.output()));
    if (axis.size() != 1) {
        return mlir::failure();
    }

    const auto fuseInputs = getAllInputOp(origOp, axis);
    if (fuseInputs.size() <= origOp.inputs().size()) {
        return mlir::failure();
    }

    const auto axisValue = *axis.begin();
    rewriter.replaceOpWithNewOp<IE::ConcatOp>(origOp, fuseInputs, axisValue.ind());

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::ConcatOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results, mlir::MLIRContext* ctx) {
    results.add<ConvertPerAxisToOffsets>(ctx);
    results.add<FuseConcat>(ctx);
}

//
// fold
//

mlir::OpFoldResult vpux::IE::ConcatOp::fold(ArrayRef<mlir::Attribute>) {
    if (inputs().size() == 1) {
        return inputs().front();
    }

    return nullptr;
}
