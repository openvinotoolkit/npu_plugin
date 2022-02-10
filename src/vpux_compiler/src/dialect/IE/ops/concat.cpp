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
#include "vpux/compiler/dialect/IE/utils/to_ngraph.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include <ngraph/opsets/opset7.hpp>

#include <map>
#include <unordered_set>

using namespace vpux;

//
// build
//

void vpux::IE::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange inputs,
                               ConcatAttrs per_axis) {
    build(builder, state, inputs, per_axis, nullptr);
}

void vpux::IE::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange inputs,
                               mlir::IntegerAttr axis, mlir::IntegerAttr offset, mlir::IntegerAttr stride) {
    build(builder, state, inputs, ConcatAttrs::get(axis, offset, stride, builder.getContext()));
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

    auto axisInd = concat.per_axis().axis().getValue().getSExtValue();

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

mlir::FailureOr<Shape> inferOutShapeWithOffsets(IE::ConcatOpAdaptor concat, mlir::Location loc) {
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

mlir::FailureOr<mlir::Type> inferOutElemTypeWithAxis(IE::ConcatOpAdaptor concat, mlir::Location loc) {
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

mlir::FailureOr<mlir::Type> inferOutElemTypeWithOffsets(IE::ConcatOpAdaptor concat, ShapeRef outShape,
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

mlir::LogicalResult vpux::IE::ConcatOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::ConcatOpAdaptor concat(operands, attrs);
    if (mlir::failed(concat.verify(loc))) {
        return mlir::failure();
    }

    if (concat.inputs().empty()) {
        return errorAt(loc, "Missing inputs for '{0}'", IE::ConcatOp::getOperationName());
    }

    if (concat.per_axis() == nullptr && concat.static_offsets() == nullptr) {
        return errorAt(loc, "Missing either 'per_axis' or 'static_offsets' attribute");
    }
    if (concat.per_axis() != nullptr && concat.static_offsets() != nullptr) {
        return errorAt(loc, "Only one attribute ('per_axis' or 'static_offsets') should be provided");
    }

    const auto inType = concat.inputs().front().getType().cast<mlir::RankedTensorType>();

    // Check consistent tensor attributes

    const auto inDesc = IE::getTensorAttr(inType);

    for (const auto val : concat.inputs().drop_front()) {
        const auto curType = val.getType().cast<mlir::RankedTensorType>();
        const auto curDesc = IE::getTensorAttr(curType);

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
                                               : inferOutElemTypeWithOffsets(concat, outShape.getValue(), loc);
    if (mlir::failed(outElemType)) {
        return mlir::failure();
    }

    // Return inferred components

    inferredReturnShapes.emplace_back(outShape.getValue().raw(), outElemType.getValue(), inDesc);
    return mlir::success();
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

const mlir::ArrayAttr inferOffsetsAttrWithAxis(IE::ConcatOp origOp, const int64_t& axis) {
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

mlir::LogicalResult ConvertPerAxisToOffsets::matchAndRewrite(IE::ConcatOp origOp,
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

std::unique_ptr<ngraph::Node> vpux::IE::ConcatOp::toNgraph(ngraph::OutputVector &outputs)
{
    VPUX_THROW_WHEN(per_axisAttr() == nullptr, "per_axis attribute for '{0}' missing", IE::ConcatOp::getOperationName());
    const auto axis = per_axisAttr().axis().getInt();

    return std::make_unique<opset_latest::Concat>(outputs, axis);
}
