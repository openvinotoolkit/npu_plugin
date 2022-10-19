//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::TopKOp::inferReturnTypes(mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc,
                                                        mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                        mlir::RegionRange /*regions*/,
                                                        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::TopKOpAdaptor topK(operands, attrs);
    if (mlir::failed(topK.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = topK.input().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inType.getShape().raw();

    auto reshapeK = topK.k().getDefiningOp<VPU::ReshapeOp>();
    auto kConst = (reshapeK != nullptr) ? reshapeK.input().getDefiningOp<Const::DeclareOp>()
                                        : topK.k().getDefiningOp<Const::DeclareOp>();
    if (kConst == nullptr) {
        return errorAt(loc, "Only constant input is supported for k");
    }

    const auto kContent = kConst.content();
    if (!kContent.isSplat()) {
        return errorAt(loc, "K input must be scalar");
    }

    SmallVector<int64_t> outShape;
    for (size_t i = 0; i < inputShape.size(); ++i) {
        outShape.push_back(inputShape[i]);
    }
    int64_t axis = topK.axis().getInt();
    const auto inRank = inType.getRank();
    if (axis < 0) {
        axis += inRank;
    }
    outShape[axis] = kContent.getSplatValue<int64_t>();

    const auto outType = inType.changeShape(Shape(outShape));

    inferredReturnTypes.push_back(outType);

    const auto outType1 = outType.changeElemType(topK.element_type().getValue());
    inferredReturnTypes.push_back(outType1);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::TopKOp::serialize(EMU::BlobWriter& /*writer*/) {
    VPUX_THROW("TopK op without regions is not implemented in low level dialects.");
}
