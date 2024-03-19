//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::LayoutCastOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                              std::optional<mlir::Location> optLoc,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::OpaqueProperties, mlir::RegionRange /*regions*/,
                                                              mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::LayoutCastOpAdaptor overrideLayout(operands, attrs);
    if (mlir::failed(overrideLayout.verify(loc))) {
        return mlir::failure();
    }

    const auto outAffineMap = overrideLayout.getDstOrder();
    const auto inType = overrideLayout.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto outType = inType.changeDimsOrder(DimsOrder::fromAffineMap(outAffineMap));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// verify
//

mlir::LogicalResult vpux::VPU::LayoutCastOp::verify() {
    const auto outAffineMap = getDstOrder();
    const auto inType = getInput().getType().cast<vpux::NDTypeInterface>();
    if (inType.getRank() != outAffineMap.getNumDims()) {
        return errorAt(*this, "Cannot apply {0} map to {1}.", outAffineMap, inType.getShape());
    }

    return mlir::success();
}
