//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ProposalOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::ProposalOpAdaptor proposal(operands, attrs);
    if (mlir::failed(proposal.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = proposal.class_probs().getType().cast<mlir::ShapedType>();

    // out shape must be [batch_size * post_nms_topn, 5]
    const SmallVector<int64_t> outShape{inType.getShape().front() * proposal.proposal_attrs().postNmsTopN().getInt(),
                                        5};
    inferredReturnShapes.emplace_back(outShape, inType.getElementType());

    return mlir::success();
}
