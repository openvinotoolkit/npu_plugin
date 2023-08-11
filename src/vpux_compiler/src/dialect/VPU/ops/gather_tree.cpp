//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::GatherTreeOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                              mlir::Optional<mlir::Location> optLoc,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::RegionRange /*regions*/,
                                                              mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::GatherTreeOpAdaptor gatherTree(operands, attrs);
    if (mlir::failed(gatherTree.verify(loc))) {
        return mlir::failure();
    }

    const auto stepIdsType = gatherTree.stepIds().getType();
    inferredReturnTypes.push_back(stepIdsType);

    return mlir::success();
}

void vpux::VPU::GatherTreeOp::inferLayoutInfo(mlir::Operation*, IE::LayerLayoutInfo& info) {
    IE::fillDefaultLayoutInfo(info);
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::GatherTreeOp::serialize(EMU::BlobWriter&) {
    VPUX_THROW("GatherTree is not implemented in UPA Tasks.");
}
