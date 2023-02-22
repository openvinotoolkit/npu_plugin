//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
using namespace vpux;

mlir::LogicalResult vpux::VPU::SparsifyOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                            mlir::Optional<mlir::Location> optLoc,
                                                            mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                            mlir::RegionRange /*regions*/,
                                                            mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::SparsifyOpAdaptor sparsify(operands, attrs);
    if (mlir::failed(sparsify.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = sparsify.input().getType().cast<mlir::RankedTensorType>();

    const auto outType = VPU::SparseTensorType::get(inType);
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::SparsifyOp::serialize(EMU::BlobWriter&) {
    VPUX_THROW("VPU::SparsifyOp is not supported by EMU");
}
