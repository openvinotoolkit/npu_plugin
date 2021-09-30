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

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

//
// build
//

void vpux::IE::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange inputs,
                               mlir::IntegerAttr axis) {
    build(builder, state, inputs, axis, getIntAttr(axis.getContext(), 0), getIntAttr(axis.getContext(), 1));
}

void vpux::IE::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type type,
                               mlir::ValueRange inputs, mlir::IntegerAttr axis) {
    build(builder, state, type, inputs, axis, getIntAttr(axis.getContext(), 0), getIntAttr(axis.getContext(), 1));
}

//
// inferReturnTypeComponents
//

namespace {

Dim normalizeAxis(IE::ConcatOpAdaptor concat) {
    const auto inType = concat.inputs().front().getType().cast<mlir::ShapedType>();
    const auto inRank = inType.getRank();

    auto axisInd = concat.axis().getValue().getSExtValue();

    // Negative value means counting dimension from the end
    if (axisInd < 0) {
        axisInd += inRank;
    }

    VPUX_THROW_UNLESS(axisInd >= 0 && axisInd < inRank, "Got wrong Concat axis '{0}', out of range '{1}'", axisInd,
                      inRank);

    return Dim(axisInd);
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

    const auto axis = normalizeAxis(concat);

    auto outShape = getShape(inType).toValues();

    for (const auto val : concat.inputs().drop_front()) {
        const auto curShape = getShape(val);
        outShape[axis] += curShape[axis];
    }

    if (concat.offset().getInt() > outShape[axis] ||
        concat.offset().getInt() + concat.stride().getInt() > outShape[axis]) {
        return errorAt(loc, "Concat offset '{0}' and stride '{1}' are larger than output dimension '{2}'",
                       concat.offset(), concat.stride(), outShape[axis]);
    }

    // Infer output element type

    const auto inElemType = inType.getElementType();

    SmallVector<mlir::quant::UniformQuantizedPerAxisType> inPerAxisQTypes;

    const auto perAxisQType = inElemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>();
    if (perAxisQType != nullptr) {
        inPerAxisQTypes.push_back(perAxisQType);
    }

    for (const auto val : concat.inputs().drop_front()) {
        const auto curType = val.getType().cast<mlir::RankedTensorType>();
        const auto curElemType = curType.getElementType();

        if (perAxisQType != nullptr) {
            const auto curPerAxisQType = curType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>();

            if (curPerAxisQType == nullptr) {
                return errorAt(loc, "Misaligned element types for '{0}' inputs", IE::ConcatOp::getOperationName());
            }
            if (!canBeMerged(curPerAxisQType, perAxisQType)) {
                return errorAt(loc, "Misaligned element types for '{0}' inputs", IE::ConcatOp::getOperationName());
            }

            inPerAxisQTypes.push_back(curPerAxisQType);
        } else {
            if (curElemType != inElemType) {
                return errorAt(loc, "Misaligned element types for '{0}' inputs", IE::ConcatOp::getOperationName());
            }
        }
    }

    const auto outElemType = inPerAxisQTypes.empty() ? inElemType : concatScalesAndZP(inPerAxisQTypes);

    // Return inferred components

    inferredReturnShapes.emplace_back(outShape.raw(), outElemType, inDesc);
    return mlir::success();
}
