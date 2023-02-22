//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
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

//
// verify
//

mlir::LogicalResult vpux::IE::verifyOp(IE::ExpandOp op) {
    if (op.input().getDefiningOp() != nullptr && mlir::isa<Const::DeclareOp>(op.input().getDefiningOp())) {
        // Limitations below are not applicable to constants.
        return mlir::success();
    }
    const auto nonZeroPadPredicate = [](const int64_t dim) -> bool {
        return dim > 0;
    };
    const auto padsEnd = parseIntArrayAttr<int64_t>(op.pads_end());
    const auto nonZeroPadsEnd = std::count_if(padsEnd.begin(), padsEnd.end(), nonZeroPadPredicate);
    const auto padsBegin = parseIntArrayAttr<int64_t>(op.pads_begin());
    const auto nonZeroPadsBegin = std::count_if(padsBegin.begin(), padsBegin.end(), nonZeroPadPredicate);
    if (nonZeroPadsEnd == 0 && nonZeroPadsBegin == 0) {
        // Such pad configuration is foldable.
        return mlir::success();
    }

    if (nonZeroPadsBegin > 1) {
        return errorAt(op, "pads_begin must contain at most one non-zero value. Got: {0}", padsEnd);
    }

    if (nonZeroPadsEnd > 1) {
        return errorAt(op, "pads_end must contain at most one non-zero value. Got: {0}", padsEnd);
    }

    const auto padBeginAxisIter = std::find_if(padsBegin.begin(), padsBegin.end(), nonZeroPadPredicate);
    if (padBeginAxisIter != padsBegin.end()) {
        const auto padAxis = std::distance(padsBegin.begin(), padBeginAxisIter);
        const auto inShape = getShape(op.input());
        if (padAxis >= checked_cast<int64_t>(inShape.size())) {
            return errorAt(op, "pads_begin axis {0} exceeds input rank {1}", padAxis, inShape.size());
        }
        const auto outShape = getShape(op.output());
        if (padAxis >= checked_cast<int64_t>(outShape.size())) {
            return errorAt(op, "pads_begin axis {0} exceeds output rank {1}", padAxis, inShape.size());
        }
    }

    const auto padEndAxisIter = std::find_if(padsEnd.begin(), padsEnd.end(), nonZeroPadPredicate);
    if (padEndAxisIter != padsEnd.end()) {
        const auto padAxis = std::distance(padsEnd.begin(), padEndAxisIter);
        const auto inShape = getShape(op.input());
        if (padAxis >= checked_cast<int64_t>(inShape.size())) {
            return errorAt(op, "pads_end axis {0} exceeds input rank {1}", padAxis, inShape.size());
        }
        const auto outShape = getShape(op.output());
        if (padAxis >= checked_cast<int64_t>(outShape.size())) {
            return errorAt(op, "pads_end axis {0} exceeds output rank {1}", padAxis, inShape.size());
        }
    }

    if (padBeginAxisIter != padsBegin.end() && padEndAxisIter != padsEnd.end()) {
        const auto padBeginAxis = std::distance(padsBegin.begin(), padBeginAxisIter);
        const auto padEndAxis = std::distance(padsEnd.begin(), padEndAxisIter);
        if (padBeginAxis != padEndAxis) {
            return errorAt(op, "pads_begin axis {0} does not match pads_end {1}", padBeginAxis, padEndAxis);
        }
    }

    return mlir::success();
}
