//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::ScatterElementsUpdateOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ScatterElementsUpdateOpAdaptor scatter(operands, attrs);
    if (mlir::failed(scatter.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = scatter.input().getType();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::ScatterElementsUpdateOp::serialize(EMU::BlobWriter&) {
    VPUX_THROW("ScatterElementsUpdate is not implemented in UPA Tasks.");
}
