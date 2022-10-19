//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"

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

void vpux::VPU::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange inputs,
                                IE::ConcatAttrs per_axis) {
    build(builder, state, inputs, per_axis, nullptr);
}

void vpux::VPU::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange inputs,
                                mlir::IntegerAttr axis, mlir::IntegerAttr offset, mlir::IntegerAttr stride) {
    build(builder, state, inputs, IE::ConcatAttrs::get(axis, offset, stride, builder.getContext()));
}

void vpux::VPU::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange inputs,
                                int64_t axis, int64_t offset, int64_t stride) {
    build(builder, state, inputs, getIntAttr(builder, axis), offset != 0 ? getIntAttr(builder, offset) : nullptr,
          stride != 1 ? getIntAttr(builder, stride) : nullptr);
}

void vpux::VPU::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange inputs,
                                Dim axis, int64_t offset, int64_t stride) {
    build(builder, state, inputs, axis.ind(), offset, stride);
}

void vpux::VPU::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type outType,
                                mlir::ValueRange inputs, mlir::ArrayAttr static_offsets) {
    build(builder, state, outType, inputs, nullptr, static_offsets);
}

void vpux::VPU::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type outType,
                                mlir::ValueRange inputs, ArrayRef<Shape> static_offsets) {
    const auto attrArr = to_small_vector(static_offsets | transformed([&](ShapeRef arr) -> mlir::Attribute {
                                             return getIntArrayAttr(builder, arr);
                                         }));

    build(builder, state, outType, inputs, builder.getArrayAttr(attrArr));
}

void vpux::VPU::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type outType,
                                mlir::ValueRange inputs, ArrayRef<ShapeRef> static_offsets) {
    const auto attrArr = to_small_vector(static_offsets | transformed([&](ShapeRef arr) -> mlir::Attribute {
                                             return getIntArrayAttr(builder, arr);
                                         }));

    build(builder, state, outType, inputs, builder.getArrayAttr(attrArr));
}

//
// InferTypeOpInterface
//

namespace {

Dim normalizeAxis(VPU::ConcatOpAdaptor concat) {
    const auto inType = concat.inputs().front().getType().cast<vpux::NDTypeInterface>();
    const auto inRank = inType.getRank();

    auto axisInd = concat.per_axis().axis().getValue().getSExtValue();

    // Negative value means counting dimension from the end
    if (axisInd < 0) {
        axisInd += inRank;
    }

    VPUX_THROW_UNLESS(axisInd >= 0 && axisInd < inRank, "Got wrong Concat axis '{0}', out of range '{1}'", axisInd,
                      inRank);

    return Dim(axisInd);
}

mlir::FailureOr<Shape> inferOutShapeWithAxis(VPU::ConcatOpAdaptor concat, mlir::Location loc) {
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

    const auto offset = concat.per_axis().offset() ? concat.per_axis().offset().getValue().getSExtValue() : 0;
    const auto stride = concat.per_axis().stride() ? concat.per_axis().stride().getValue().getSExtValue() : 1;

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

mlir::FailureOr<Shape> inferOutShapeWithOffsets(VPU::ConcatOpAdaptor concat, mlir::Location loc) {
    if (concat.static_offsets().size() != concat.inputs().size()) {
        return errorAt(loc, "Concat 'static_offsets' count '{0}' doesn't match inputs count '{1}'",
                       concat.static_offsets().size(), concat.inputs().size());
    }

    const auto inType = concat.inputs().front().getType().cast<vpux::NDTypeInterface>();
    const auto allOffsets = concat.static_offsets().getAsRange<mlir::ArrayAttr>();

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

mlir::FailureOr<mlir::Type> inferOutElemTypeWithAxis(VPU::ConcatOpAdaptor concat, mlir::Location loc) {
    const auto inType = concat.inputs().front().getType().cast<vpux::NDTypeInterface>();
    const auto inElemType = inType.getElementType();

    const auto perAxisQType = inElemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>();
    SmallVector<mlir::quant::UniformQuantizedPerAxisType> inPerAxisQTypes;

    if (perAxisQType != nullptr) {
        const auto axis = normalizeAxis(concat);

        if (perAxisQType.getQuantizedDimension() == axis.ind()) {
            inPerAxisQTypes.push_back(perAxisQType);
        }
    }

    for (const auto val : concat.inputs().drop_front()) {
        const auto curType = val.getType().cast<vpux::NDTypeInterface>();
        const auto curElemType = curType.getElementType();

        if (inPerAxisQTypes.empty()) {
            if (curElemType != inElemType) {
                return errorAt(loc, "Misaligned element types : '{0}' vs '{1}'", curElemType, inElemType);
            }
        } else {
            const auto curPerAxisQType = curElemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>();

            if (curPerAxisQType == nullptr) {
                return errorAt(loc,
                               "Misaligned element types : not all of them are per-axis quantized : '{0}' vs '{1}'",
                               curElemType, inElemType);
            }

            if (curPerAxisQType.getQuantizedDimension() != perAxisQType.getQuantizedDimension()) {
                return errorAt(
                        loc,
                        "Misaligned element types : per-axis quantization is done on different axis : '{0}' vs '{1}'",
                        curPerAxisQType.getQuantizedDimension(), perAxisQType.getQuantizedDimension());
            }

            if (!canBeMerged(curPerAxisQType, perAxisQType)) {
                return errorAt(loc, "Misaligned element types : per-axis quantization parameters can't be merged");
            }

            inPerAxisQTypes.push_back(curPerAxisQType);
        }
    }

    return inPerAxisQTypes.empty() ? inElemType : concatScalesAndZP(inPerAxisQTypes);
}

std::unordered_set<Dim> getConcatAxesFromOffsets(VPU::ConcatOpAdaptor concat, ShapeRef outShape) {
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

mlir::FailureOr<mlir::Type> inferOutElemTypeWithOffsets(VPU::ConcatOpAdaptor concat, ShapeRef outShape,
                                                        mlir::Location loc) {
    const auto inType = concat.inputs().front().getType().cast<vpux::NDTypeInterface>();
    const auto inElemType = inType.getElementType();

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
        for (const auto val : concat.inputs().drop_front()) {
            const auto curType = val.getType().cast<vpux::NDTypeInterface>();
            const auto curElemType = curType.getElementType();

            if (curElemType != inElemType) {
                return errorAt(loc, "Misaligned element types : '{0}' vs '{1}'", curElemType, inElemType);
            }
        }

        return inElemType;
    }

    const auto qDim = perAxisQType.getQuantizedDimension();
    const auto allOffsets = concat.static_offsets().getAsRange<mlir::ArrayAttr>();

    std::map<int64_t, mlir::quant::UniformQuantizedPerAxisType> perSliceQuantTypes;

    for (const auto p : zip(concat.inputs(), allOffsets)) {
        const auto curVal = std::get<0>(p);

        const auto curType = curVal.getType().cast<vpux::NDTypeInterface>();
        const auto curElemType = curType.getElementType();
        const auto curPerAxisQType = curElemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>();

        if (curPerAxisQType == nullptr) {
            return errorAt(loc, "Misaligned element types : not all of them are per-axis quantized : '{0}' vs '{1}'",
                           curElemType, inElemType);
        }

        if (curPerAxisQType.getQuantizedDimension() != qDim) {
            return errorAt(
                    loc, "Misaligned element types : per-axis quantization is done on different axis : '{0}' vs '{1}'",
                    curPerAxisQType.getQuantizedDimension(), qDim);
        }

        const auto curOffsets = parseIntArrayAttr<int64_t>(std::get<1>(p));
        const auto sliceOffset = curOffsets[checked_cast<size_t>(qDim)];

        const auto it = perSliceQuantTypes.find(sliceOffset);
        if (it == perSliceQuantTypes.end()) {
            perSliceQuantTypes.insert({sliceOffset, curPerAxisQType});
        } else {
            if (curPerAxisQType != it->second) {
                return errorAt(loc, "Per-axis quantization is not aligned over non quantized axis : '{0}' vs '{1}'",
                               curPerAxisQType, it->second);
            }
        }
    }

    const auto inPerAxisQTypes = to_small_vector(perSliceQuantTypes | map_values);
    return concatScalesAndZP(inPerAxisQTypes);
}

}  // namespace

mlir::LogicalResult vpux::VPU::ConcatOp::inferReturnTypes(mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc,
                                                          mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                          mlir::RegionRange /*regions*/,
                                                          mlir::SmallVectorImpl<mlir::Type>& inferredTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::ConcatOpAdaptor concat(operands, attrs);
    if (mlir::failed(concat.verify(loc))) {
        return mlir::failure();
    }

    if (concat.inputs().empty()) {
        return errorAt(loc, "Missing inputs for '{0}'", VPU::ConcatOp::getOperationName());
    }

    if (concat.per_axis() == nullptr && concat.static_offsets() == nullptr) {
        return errorAt(loc, "Missing either 'per_axis' or 'static_offsets' attribute");
    }
    if (concat.per_axis() != nullptr && concat.static_offsets() != nullptr) {
        return errorAt(loc, "Only one attribute ('per_axis' or 'static_offsets') should be provided");
    }

    auto input1Type = concat.inputs().front().getType();
    if (const auto inTypeRanked = input1Type.dyn_cast<mlir::RankedTensorType>()) {
        // Check consistent tensor attributes

        const auto inDesc = IE::getTensorAttr(inTypeRanked);

        for (const auto val : concat.inputs().drop_front()) {
            if (!val.getType().isa<mlir::RankedTensorType>()) {
                return errorAt(loc, "Misaligned tensor type for '{0}' inputs", VPU::ConcatOp::getOperationName());
            }

            const auto curType = val.getType().cast<mlir::RankedTensorType>();
            const auto curDesc = IE::getTensorAttr(curType);

            if (curDesc != inDesc) {
                return errorAt(loc, "Misaligned TensorType attributes for '{0}' inputs",
                               VPU::ConcatOp::getOperationName());
            }
        }
    } else if (const auto inTypeDistributed = input1Type.dyn_cast<VPU::DistributedTensorType>()) {
        const auto inOrder = inTypeDistributed.getOrder();
        const auto inMemSpace = inTypeDistributed.getMemSpace();
        const auto inDistribution = inTypeDistributed.getDistribution();

        // Check consistent distributed tensor attributes

        for (const auto val : concat.inputs().drop_front()) {
            if (!val.getType().isa<VPU::DistributedTensorType>()) {
                return errorAt(loc, "Misaligned tensor type for '{0}' inputs", VPU::ConcatOp::getOperationName());
            }

            const auto curType = val.getType().cast<VPU::DistributedTensorType>();

            if (curType.getOrder() != inOrder || curType.getMemSpace() != inMemSpace ||
                curType.getDistribution() != inDistribution) {
                return errorAt(loc, "Misaligned DistributedTensorType attributes for '{0}' inputs",
                               VPU::ConcatOp::getOperationName());
            }
        }
    } else {
        VPUX_THROW("Unsupported VPU::Concat type on input - `{0}`", input1Type);
    }

    // Infer output shape
    const auto outShape =
            concat.per_axis() ? inferOutShapeWithAxis(concat, loc) : inferOutShapeWithOffsets(concat, loc);
    if (mlir::failed(outShape)) {
        return mlir::failure();
    }

    // Infer output element type

    const auto outElemType = concat.per_axis() ? inferOutElemTypeWithAxis(concat, loc)
                                               : inferOutElemTypeWithOffsets(concat, outShape.getValue(), loc);
    if (mlir::failed(outElemType)) {
        return mlir::failure();
    }

    mlir::Type outputType;
    if (const auto inTypeRanked = input1Type.dyn_cast<mlir::RankedTensorType>()) {
        const auto tensorDesc = IE::getTensorAttr(inTypeRanked);

        outputType = mlir::RankedTensorType::get(outShape.getValue().raw(), outElemType.getValue(), tensorDesc);
    } else if (const auto inTypeDistributed = input1Type.dyn_cast<VPU::DistributedTensorType>()) {
        outputType = VPU::DistributedTensorType::get(ctx, outShape.getValue().raw(), outElemType.getValue(),
                                                     inTypeDistributed.getOrder(), inTypeDistributed.getMemSpace(),
                                                     inTypeDistributed.getDistribution());
    } else {
        VPUX_THROW("Unsupported VPU::Concat type on input - `{0}`", input1Type);
    }

    // Return inferred type

    inferredTypes.emplace_back(outputType);

    return mlir::success();
}

//
// ConvertPerAxisToOffsets
//

namespace {

class ConvertPerAxisToOffsets final : public mlir::OpRewritePattern<VPU::ConcatOp> {
public:
    using mlir::OpRewritePattern<VPU::ConcatOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(VPU::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;
};

const mlir::ArrayAttr inferOffsetsAttrWithAxis(VPU::ConcatOp origOp, const int64_t& axis) {
    auto rank = origOp.output().getType().cast<vpux::NDTypeInterface>().getRank();

    SmallVector<SmallVector<int64_t>> finalOffsets;
    finalOffsets.push_back(SmallVector<int64_t>(rank, 0));

    for (auto input : origOp.inputs() | indexed) {
        auto inputShape = getShape(input.value());
        auto offsets = SmallVector<int64_t>(rank, 0);
        offsets[axis] = inputShape[Dim(axis)] + finalOffsets.back()[axis];
        finalOffsets.push_back(offsets);
    }
    finalOffsets.pop_back();

    return getIntArrayOfArray(origOp.getContext(), finalOffsets);
}

mlir::LogicalResult ConvertPerAxisToOffsets::matchAndRewrite(VPU::ConcatOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    if (origOp.static_offsetsAttr()) {
        return mlir::failure();
    }

    if (origOp.per_axisAttr().stride() || origOp.per_axisAttr().offset()) {
        return mlir::failure();
    }

    const auto outType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    const auto axis = origOp.per_axisAttr().axis().getValue().getSExtValue();
    const auto finalOffsetsAttr = inferOffsetsAttrWithAxis(origOp, axis);

    rewriter.replaceOpWithNewOp<VPU::ConcatOp>(origOp, outType, origOp.inputs(), finalOffsetsAttr);
    return mlir::success();
}

}  // namespace

//
// FuseConcat
//

namespace {

class FuseConcat final : public mlir::OpRewritePattern<VPU::ConcatOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPU::ConcatOp op, mlir::PatternRewriter& rewriter) const final;
};

SmallVector<mlir::Value> getAllInputOp(VPU::ConcatOp origOp, const std::unordered_set<Dim>& axis) {
    SmallVector<mlir::Value> inputOps;
    for (auto preOps : origOp.inputs()) {
        auto producerConcatOp = preOps.getDefiningOp<VPU::ConcatOp>();

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

mlir::LogicalResult FuseConcat::matchAndRewrite(VPU::ConcatOp origOp, mlir::PatternRewriter& rewriter) const {
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
    rewriter.replaceOpWithNewOp<VPU::ConcatOp>(origOp, fuseInputs, axisValue.ind());

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::VPU::ConcatOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results, mlir::MLIRContext* ctx) {
    results.add<ConvertPerAxisToOffsets>(ctx);
    results.add<FuseConcat>(ctx);
}

//
// fold
//

mlir::OpFoldResult VPU::ConcatOp::fold(ArrayRef<mlir::Attribute>) {
    if (inputs().size() == 1) {
        return inputs().front();
    }

    return nullptr;
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::ConcatOp::serialize(EMU::BlobWriter& writer) {
    uint32_t axis = 0;
    uint32_t offset = 0;
    uint32_t stride = 1;

    if (per_axis().hasValue()) {
        const auto perAxis = per_axis().getValue();
        axis = static_cast<uint32_t>(perAxis.axis().getValue().getSExtValue());
        offset = perAxis.offset() ? static_cast<uint32_t>(perAxis.offset().getValue().getSExtValue()) : 0;
        stride = perAxis.stride() ? static_cast<uint32_t>(perAxis.stride().getValue().getSExtValue()) : 1;
    }

    MVCNN::ConcatAttrs perAxisAttrs(axis, offset, stride);

    SmallVector<flatbuffers::Offset<MVCNN::ConcatOffsets>> offsetsVec;
    if (static_offsets().hasValue()) {
        const auto staticOffsetsArr = parseIntArrayOfArrayAttr<int64_t>(static_offsetsAttr());

        for (const auto offsets : staticOffsetsArr) {
            const auto offsetsFb = writer.createVector(offsets | transformed([](int64_t val) {
                                                           return checked_cast<int32_t>(val);
                                                       }));
            MVCNN::ConcatOffsetsBuilder offsetsBuilder(writer);
            offsetsBuilder.add_offsets(offsetsFb);
            offsetsVec.push_back(offsetsBuilder.Finish());
        }
    }

    const auto staticOffsetsFbVec = writer.createVector(offsetsVec);
    MVCNN::ConcatParamsBuilder builder(writer);
    builder.add_per_axis(&perAxisAttrs);
    builder.add_static_offsets(staticOffsetsFbVec);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ConcatParams});
}
