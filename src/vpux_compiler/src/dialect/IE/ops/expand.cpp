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
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::ExpandOpAdaptor expand(operands, attrs);
    if (mlir::failed(expand.verify(loc))) {
        return mlir::failure();
    }

    const auto padBegin = parseIntArrayAttr<int64_t>(expand.pads_begin());
    const auto padEnd = parseIntArrayAttr<int64_t>(expand.pads_end());

    const auto inType = expand.input().getType().cast<mlir::ShapedType>();
    if (!inType) {
        return mlir::failure();
    }

    const auto inputShape = inType.getShape();
    SmallVector<int64_t> outShape(inputShape.size());
    for (size_t i = 0; i < inputShape.size(); ++i) {
        outShape[i] = padBegin[i] + inputShape[i] + padEnd[i];
    }

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());

    return mlir::success();
}
