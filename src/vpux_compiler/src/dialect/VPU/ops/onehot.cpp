//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::OneHotOp::inferReturnTypes(mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc,
                                                          mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                          mlir::RegionRange,
                                                          mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::OneHotOpAdaptor oneHot(operands, attrs);
    if (mlir::failed(oneHot.verify(loc))) {
        return mlir::failure();
    }

    auto inType = oneHot.input().getType().cast<vpux::NDTypeInterface>();
    mlir::Type outType = oneHot.outElemType();
    auto anyRankedTensorType = inType.changeElemType(outType);

    SmallVector<int64_t> outShape = to_small_vector(oneHot.input().getType().cast<vpux::NDTypeInterface>().getShape());
    const auto axis = oneHot.axis();
    int64_t depth = oneHot.depthAttr().getInt();
    if (axis < 0) {
        outShape.insert(outShape.end() + 1 + axis, depth);
    } else {
        outShape.insert(outShape.begin() + axis, depth);
    }

    inferredReturnTypes.push_back(anyRankedTensorType.changeShape(Shape(outShape)));

    return mlir::success();
}

void vpux::VPU::OneHotOp::inferLayoutInfo(mlir::Operation*, IE::LayerLayoutInfo& info) {
    IE::fillDefaultLayoutInfo(info);
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::OneHotOp::serialize(EMU::BlobWriter& /*writer*/) {
    VPUX_THROW("OneHot op without regions is not implemented in low level dialects.");
}
