//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

void vpux::IE::ExpandOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                               Optional<ShapeRef> pads_begin, Optional<ShapeRef> pads_end) {
    VPUX_THROW_UNLESS(pads_begin.hasValue() || pads_end.hasValue(),
                      "pads_begin and/or pads_end must be provided for IE::ExpandOp");

    const auto origShape = getShape(input);

    const auto getPadsAttr = [&](Optional<ShapeRef> pads) {
        if (pads.hasValue()) {
            return getIntArrayAttr(builder.getContext(), pads.getValue());
        }

        const SmallVector<int64_t> zero(origShape.size(), 0);
        return getIntArrayAttr(builder.getContext(), zero);
    };

    build(builder, state, input, getPadsAttr(pads_begin), getPadsAttr(pads_end));
}

mlir::LogicalResult vpux::IE::ExpandOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::ExpandOpAdaptor expand(operands, attrs);
    if (mlir::failed(expand.verify(loc))) {
        return mlir::failure();
    }

    const auto padBegin = parseIntArrayAttr<int64_t>(expand.pads_begin());
    const auto padEnd = parseIntArrayAttr<int64_t>(expand.pads_end());

    const auto inType = expand.input().getType().cast<vpux::NDTypeInterface>();
    if (!inType) {
        return mlir::failure();
    }

    const auto newType = inType.pad(ShapeRef(padBegin), ShapeRef(padEnd));
    const auto newTensorType = newType.cast<mlir::RankedTensorType>();
    inferredReturnShapes.emplace_back(newTensorType.getShape(), newTensorType.getElementType(),
                                      newTensorType.getEncoding());

    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult vpux::IE::ExpandOp::fold(ArrayRef<mlir::Attribute> operands) {
    if (input().getType() == output().getType()) {
        return input();
    }

    // Check for Slice->Expand pair which can be optimized if ExpandOp
    // output is same as SliceOp input
    if (auto sliceOp = input().getDefiningOp<IE::SliceOp>()) {
        if (sliceOp.source().getType() == output().getType()) {
            return sliceOp.source();
        }
    }

    if (const auto attr = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        const auto padsBefore = Shape(parseIntArrayAttr<int64_t>(pads_begin()));
        const auto padsAfter = Shape(parseIntArrayAttr<int64_t>(pads_end()));
        return attr.padWithZero(padsBefore, padsAfter);
    }

    return nullptr;
}
