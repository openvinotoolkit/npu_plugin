//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"

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
                                IE::ConcatAttr per_axis) {
    build(builder, state, inputs, per_axis, nullptr, nullptr);
}

void vpux::VPU::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange inputs,
                                IE::ConcatAttr per_axis, mlir::ArrayAttr static_offsets) {
    build(builder, state, inputs, per_axis, static_offsets, nullptr);
}

void vpux::VPU::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange inputs,
                                mlir::IntegerAttr axis, mlir::IntegerAttr offset, mlir::IntegerAttr stride) {
    build(builder, state, inputs, IE::ConcatAttr::get(builder.getContext(), axis, offset, stride));
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
    build(builder, state, outType, inputs, nullptr, static_offsets, nullptr);
}

void vpux::VPU::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type outType,
                                mlir::ValueRange inputs, IE::ConcatAttr per_axis, mlir::ArrayAttr static_offsets) {
    build(builder, state, outType, inputs, per_axis, static_offsets, nullptr);
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

    auto axisInd = concat.per_axis().value().getAxis().getValue().getSExtValue();

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

mlir::FailureOr<Shape> inferOutShapeWithOffsets(VPU::ConcatOpAdaptor concat, mlir::Location loc) {
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
    const auto allOffsets = concat.static_offsets().value().getAsRange<mlir::ArrayAttr>();

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
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ConcatOpAdaptor concat(operands, attrs);
    if (mlir::failed(concat.verify(loc))) {
        return mlir::failure();
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

    const auto inType = concat.inputs().front().getType();
    const auto distributedIn = inType.dyn_cast<VPU::DistributedTensorType>();
    if (distributedIn != nullptr &&
        VPU::isDistributedAttrWithExplicitShapesAndOffsets(distributedIn.getDistribution())) {
        const auto outputDistribution =
                VPU::getConcatExplicitDistributedAttrForNewShape(distributedIn, Shape(outShape.value()));
        const auto outputType =
                distributedIn.changeShapeForExplicitDistribution(Shape(outShape.value()), outputDistribution);
        inferredTypes.emplace_back(outputType);
    } else {
        const auto typeComponents =
                TypeComponents().setShape(Shape(outShape.value())).setElementType(outElemType.value());

        const auto outputType = inType.cast<NDTypeInterface>().changeTypeComponents(typeComponents);
        inferredTypes.emplace_back(outputType);
    }

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

mlir::LogicalResult ConvertPerAxisToOffsets::matchAndRewrite(VPU::ConcatOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    if (origOp.static_offsetsAttr()) {
        return mlir::failure();
    }

    if (origOp.per_axisAttr().getStride() || origOp.per_axisAttr().getOffset()) {
        return mlir::failure();
    }

    const auto outType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    const auto axis = origOp.per_axisAttr().getAxis().getValue().getSExtValue();
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

//
// FuseConcatsWithDifferentAxes
//

class FuseConcatsWithDifferentAxes final : public mlir::OpRewritePattern<VPU::ConcatOp> {
public:
    FuseConcatsWithDifferentAxes(mlir::MLIRContext* ctx): mlir::OpRewritePattern<VPU::ConcatOp>(ctx) {
        setDebugName("FuseConcatsWithDifferentAxes");
    }

    mlir::LogicalResult matchAndRewrite(VPU::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    static bool hasConcatProducer(const mlir::Value input);
    SmallVector<mlir::Value> flattenInputs(const mlir::ValueRange concatViewInputs) const;
    SmallVector<SmallVector<int64_t>> recalculateConcatOffsets(const mlir::Value concatInput,
                                                               ArrayRef<int64_t> origOpOffsets) const;
    SmallVector<SmallVector<int64_t>> recalculateOffsets(VPU::ConcatOp origOp) const;
};

bool FuseConcatsWithDifferentAxes::hasConcatProducer(const mlir::Value input) {
    auto concatOp = mlir::dyn_cast_or_null<VPU::ConcatOp>(input.getDefiningOp());
    // The producer must have static offsets defined.
    // Otherwise it's hard to recalculate the offsets properly.
    if (concatOp == nullptr || concatOp.static_offsetsAttr() == nullptr || !concatOp.static_offsets().has_value()) {
        return false;
    }
    // Fusion of VPU.Concat operations with multiple consumers results in scheduling errors.
    // It is safe to fuse VPU.Concat producer when all its consumers are VPU.Concat operations.
    // See [E#75133]
    const auto isConcat = [](const mlir::Operation* consumer) -> bool {
        return mlir::isa<VPU::ConcatOp>(consumer);
    };
    const auto consumers = concatOp->getUsers();
    return std::all_of(consumers.begin(), consumers.end(), isConcat);
}

// Propagate inputs from producer concat to consumer concat:
// %concat = VPU.Concat(%val0, %val1)
// VPU.Concat(%val2, %concat)
// Results in VPU.Concat(%val2, %val0, %val1)
SmallVector<mlir::Value> FuseConcatsWithDifferentAxes::flattenInputs(const mlir::ValueRange concatViewInputs) const {
    SmallVector<mlir::Value> newInputs;
    for (const auto& input : concatViewInputs) {
        if (hasConcatProducer(input)) {
            auto producerConcatOp = mlir::cast<VPU::ConcatOp>(input.getDefiningOp());
            const auto producerConcatInputs = producerConcatOp.inputs();
            newInputs.append(producerConcatInputs.begin(), producerConcatInputs.end());
        } else {
            newInputs.push_back(input);
        }
    }
    return newInputs;
}

SmallVector<SmallVector<int64_t>> FuseConcatsWithDifferentAxes::recalculateConcatOffsets(
        const mlir::Value concatInput, ArrayRef<int64_t> origOpOffsets) const {
    SmallVector<SmallVector<int64_t>> recalculatedOffsets;
    auto producerConcatOp = concatInput.getDefiningOp<VPU::ConcatOp>();
    const auto oldOffsetsArr = producerConcatOp.static_offsets().value().getAsRange<mlir::ArrayAttr>();
    for (const auto& oldOffsets : oldOffsetsArr) {
        auto offsets = parseIntArrayAttr<int64_t>(oldOffsets);
        VPUX_THROW_WHEN(offsets.size() != origOpOffsets.size(), "Rank of offsets mismatch: {0} vs {1}", offsets,
                        origOpOffsets);
        for (const auto& axis : irange(offsets.size())) {
            offsets[axis] += origOpOffsets[axis];
        }
        recalculatedOffsets.push_back(offsets);
    }
    return recalculatedOffsets;
}

// Consider VPU.Concat with concat axis C:
// %concat = VPU.Concat(%val0, %val1) { static_offsets = [[0, 0, 0, 0], [0, 0, 125, 0]] }
// VPU.Concat(%val2, %concat) { static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]] }
// In order to fuse the first VPU.Concat into the last, %val0 and %val1 must have offsets 64 by axis C:
// VPU.Concat(%val2, %val0, %val1) { static_offsets = [[0, 0, 0, 0], [[0, 64, 0, 0], [0, 64, 125, 0]] }
SmallVector<SmallVector<int64_t>> FuseConcatsWithDifferentAxes::recalculateOffsets(VPU::ConcatOp origOp) const {
    const auto concatViewInputs = origOp.inputs();
    const auto origOpOffsets = origOp.static_offsets().value().getAsRange<mlir::ArrayAttr>();
    SmallVector<SmallVector<int64_t>> newOffsets;
    for (const auto inputWithOffset : zip(concatViewInputs, origOpOffsets)) {
        const auto& input = std::get<0>(inputWithOffset);
        const auto origOpOffsets = parseIntArrayAttr<int64_t>(std::get<1>(inputWithOffset));
        if (hasConcatProducer(input)) {
            const auto offsets = recalculateConcatOffsets(input, makeArrayRef(origOpOffsets));
            newOffsets.append(offsets.begin(), offsets.end());
        } else {
            newOffsets.push_back(origOpOffsets);
        }
    }

    return newOffsets;
}

mlir::LogicalResult FuseConcatsWithDifferentAxes::matchAndRewrite(VPU::ConcatOp origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    if (origOp.static_offsetsAttr() == nullptr || !origOp.static_offsets().has_value()) {
        return mlir::failure();
    }

    const auto concatViewInputs = origOp.inputs();
    if (std::none_of(concatViewInputs.begin(), concatViewInputs.end(), hasConcatProducer)) {
        return mlir::failure();
    }
    // Fusion of VPU.Concat operations when origOp has const.Declare producers breaks inference.
    // See [E#75967]
    const auto hasConstantProducer = [](const mlir::Value input) -> bool {
        auto declOp = input.getDefiningOp<Const::DeclareOp>();
        return declOp != nullptr;
    };
    if (std::any_of(concatViewInputs.begin(), concatViewInputs.end(), hasConstantProducer)) {
        return mlir::failure();
    }

    // Fuse the inputs.
    const auto newInputs = flattenInputs(concatViewInputs);

    // Recalculate the offsets.
    const auto newOffsets = recalculateOffsets(origOp);

    auto opConcat = rewriter.create<VPU::ConcatOp>(origOp->getLoc(), origOp->getResult(0).getType(), newInputs,
                                                   getIntArrayOfArray(rewriter.getContext(), newOffsets));
    rewriter.replaceOp(origOp, opConcat->getResult(0));

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::VPU::ConcatOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results, mlir::MLIRContext* ctx) {
    results.add<ConvertPerAxisToOffsets>(ctx);
    results.add<FuseConcat>(ctx);
    results.add<FuseConcatsWithDifferentAxes>(ctx);
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

    if (per_axis().has_value()) {
        const auto perAxis = per_axis().value();
        axis = static_cast<uint32_t>(perAxis.getAxis().getValue().getSExtValue());
        offset = perAxis.getOffset() ? static_cast<uint32_t>(perAxis.getOffset().getValue().getSExtValue()) : 0;
        stride = perAxis.getStride() ? static_cast<uint32_t>(perAxis.getStride().getValue().getSExtValue()) : 1;
    }

    MVCNN::ConcatAttrs perAxisAttrs(axis, offset, stride);

    SmallVector<flatbuffers::Offset<MVCNN::ConcatOffsets>> offsetsVec;
    if (static_offsets().has_value()) {
        const auto staticOffsetsArr = parseIntArrayOfArrayAttr<int64_t>(static_offsetsAttr());

        for (const auto& offsets : staticOffsetsArr) {
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

namespace {
bool isSOHWithHeightConcatAxis(ShapeRef inputShape, ShapeRef outputShape, VPU::DistributedTensorAttr distributedAttr) {
    if (inputShape[Dims4D::Act::H] == outputShape[Dims4D::Act::H]) {
        // inputs are not concatenated over H
        return false;
    }

    return VPU::isSegmentedOverH(distributedAttr) || VPU::isOverlappedOverH(distributedAttr);
}
}  // namespace

//
// verify
//

mlir::LogicalResult vpux::VPU::ConcatOp::verify() {
    const auto loc = getLoc();

    if (inputs().empty()) {
        return errorAt(loc, "Missing inputs for '{0}'", VPU::ConcatOp::getOperationName());
    }

    if (!per_axis().has_value() && !static_offsets().has_value()) {
        return errorAt(loc, "Missing either 'per_axis' or 'static_offsets' attribute");
    }
    if (per_axis().has_value() && static_offsets().has_value()) {
        return errorAt(loc, "Only one attribute ('per_axis' or 'static_offsets') should be provided");
    }

    auto input1DataType = inputs().front().getType();
    auto input1SparseType = input1DataType.dyn_cast_or_null<VPU::SparseTensorType>();
    if (input1SparseType != nullptr) {
        input1DataType = input1SparseType.getData();
    }

    if (const auto inTypeRanked = input1DataType.dyn_cast<mlir::RankedTensorType>()) {
        // Check consistent tensor attributes

        const auto inDesc = vpux::getTensorAttr(inTypeRanked);

        for (const auto val : inputs().drop_front()) {
            if (!val.getType().isa<mlir::RankedTensorType, VPU::SparseTensorType>()) {
                return errorAt(loc, "Misaligned tensor type for '{0}' inputs", getOperationName());
            }

            const auto curType =
                    (input1SparseType != nullptr)
                            ? val.getType().cast<VPU::SparseTensorType>().getData().cast<mlir::RankedTensorType>()
                            : val.getType().cast<mlir::RankedTensorType>();
            const auto curDesc = vpux::getTensorAttr(curType);

            if (curDesc != inDesc) {
                return errorAt(loc, "Misaligned TensorType attributes for '{0}' inputs", getOperationName());
            }
        }
    } else if (const auto inTypeDistributed = input1DataType.dyn_cast<VPU::DistributedTensorType>()) {
        const auto inOrder = inTypeDistributed.getOrder();
        const auto inMemSpace = inTypeDistributed.getMemSpace();
        const auto inDistribution = inTypeDistributed.getDistribution();
        const auto inShape = inTypeDistributed.getShape();
        const auto outShape = getShape(output());

        // Check consistent distributed tensor attributes

        if (isSOHWithHeightConcatAxis(inShape, outShape, inDistribution)) {
            return errorAt(loc,
                           "Input is concatenated over H, but clustering mode is SEGMENTED or OVERLAPPED on H dim "
                           "for op {0}",
                           getOperationName());
        }

        for (const auto val : inputs().drop_front()) {
            if (!val.getType().isa<VPU::DistributedTensorType, VPU::SparseTensorType>()) {
                return errorAt(loc, "Misaligned tensor type for '{0}' inputs", getOperationName());
            }

            const auto curType =
                    (input1SparseType != nullptr)
                            ? val.getType().cast<VPU::SparseTensorType>().getData().cast<VPU::DistributedTensorType>()
                            : val.getType().cast<VPU::DistributedTensorType>();

            if (curType.getOrder() != inOrder || curType.getMemSpace() != inMemSpace) {
                return errorAt(loc, "Misaligned DistributedTensorType attributes for '{0}' inputs", getOperationName());
            }

            if (isSOHWithHeightConcatAxis(curType.getShape(), outShape, curType.getDistribution())) {
                return errorAt(loc,
                               "Input is concatenated over H, but clustering mode is SEGMENTED or OVERLAPPED on H dim "
                               "for op {0}",
                               getOperationName());
            }

            if (curType.getDistribution() != inDistribution) {
                if ((!VPU::isDistributedAttrWithExplicitShapesAndOffsets(curType.getDistribution()) ||
                     !VPU::isDistributedAttrWithExplicitShapesAndOffsets(inDistribution))) {
                    return errorAt(loc, "Misaligned DistributedTensorAttr for '{0}' inputs", getOperationName());
                }

                const auto curTypeShape = curType.getShape();

                const auto checkShapesOffsets =
                        [&](const SmallVector<SmallVector<int64_t>>& lhs,
                            const SmallVector<SmallVector<int64_t>>& rhs) -> mlir::LogicalResult {
                    for (const auto& pair : zip(lhs, rhs)) {
                        const auto shapesOffsetsLhs = std::get<0>(pair);
                        const auto shapesOffsetsRhs = std::get<1>(pair);

                        if (shapesOffsetsLhs.size() != shapesOffsetsRhs.size()) {
                            return errorAt(loc, "Per cluster shapes/offsets don't have the same rank for '{0}' inputs",
                                           getOperationName());
                        }

                        for (size_t dim = 0; dim < shapesOffsetsLhs.size(); dim++) {
                            // if dim is not a concatenation axis, check that shapes/offsets are the same for
                            // the two inputs
                            if (inShape[Dim(dim)] == curTypeShape[Dim(dim)] &&
                                inShape[Dim(dim)] == outShape[Dim(dim)]) {
                                if (shapesOffsetsLhs[dim] != shapesOffsetsRhs[dim]) {
                                    return errorAt(loc, "Misaligned per cluster shapes/offsets for '{0}' inputs",
                                                   getOperationName());
                                }
                            }
                        }
                    }

                    return mlir::success();
                };

                const auto input0PerClusterOffsets =
                        vpux::parseIntArrayOfArrayAttr<int64_t>(curType.getDistribution().getMemoryOffsets());
                const auto input1PerClusterOffsets =
                        vpux::parseIntArrayOfArrayAttr<int64_t>(inDistribution.getMemoryOffsets());

                const auto verifyMemoryOffsets = checkShapesOffsets(input0PerClusterOffsets, input1PerClusterOffsets);
                if (verifyMemoryOffsets.failed()) {
                    return verifyMemoryOffsets;
                }

                const auto input0PerClusterShapes =
                        vpux::parseIntArrayOfArrayAttr<int64_t>(curType.getDistribution().getMemoryShapes());
                const auto input1PerClusterShapes =
                        vpux::parseIntArrayOfArrayAttr<int64_t>(inDistribution.getMemoryShapes());

                const auto verifyMemoryShapes = checkShapesOffsets(input0PerClusterShapes, input1PerClusterShapes);
                if (verifyMemoryShapes.failed()) {
                    return verifyMemoryShapes;
                }
            }
        }
    } else {
        VPUX_THROW("Unsupported VPU::Concat type on input - `{0}`", input1DataType);
    }

    return mlir::success();
}

//
// NCEOpInterface
//

bool vpux::VPU::ConcatOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy) {
    // cmx_concat has accuracy issue with SOH inputs and Height concatenate. Add a workaround here to avoid
    // it happens in strategy manager pass.
    auto outputType = output().getType().cast<NDTypeInterface>();
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeight) {
        auto outputShape = outputType.getShape();
        auto inputDataType = inputs().front().getType().cast<NDTypeInterface>();
        auto inputShape = inputDataType.getShape();

        if (inputShape[Dims4D::Act::H] != outputShape[Dims4D::Act::H]) {
            return false;
        }
    }

    return strategy == VPU::MultiClusterStrategy::SplitOverKernel ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight || strategy == VPU::MultiClusterStrategy::Clustering;
}

vpux::VPU::DistributedTensorAttr vpux::VPU::ConcatOp::getExplicitDistributedTensorAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, mlir::ArrayAttr numTiles,
        mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment, mlir::ArrayAttr kernel, vpux::VPU::PaddingAttr pad,
        mlir::ArrayAttr stride, mlir::UnitAttr uniformDistributedSegments) {
    return vpux::VPU::getConcatExplicitDistributedAttr(shape, distributionMode, numTiles, numClusters, alignment,
                                                       kernel, pad, stride, uniformDistributedSegments, getContext());
}

bool vpux::VPU::ConcatOp::fitIntoCMX(vpux::NDTypeInterface output, Byte reservedMem) {
    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();
    SmallVector<Byte> buffers{output.getTotalAllocSize()};
    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(getArch(getOperation()), buffers).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::ConcatOp::fitIntoCMX(vpux::NDTypeInterface output) {
    return fitIntoCMX(output, Byte(0));
}
