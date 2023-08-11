//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/utils/quantization.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::DesparsifyOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                              mlir::Optional<mlir::Location> optLoc,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::RegionRange /*regions*/,
                                                              mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::DesparsifyOpAdaptor desparsify(operands, attrs);
    if (mlir::failed(desparsify.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = desparsify.input().getType().cast<vpux::VPU::SparseTensorType>();
    const auto dataType = inType.getData();

    inferredReturnTypes.push_back(dataType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::DesparsifyOp::serialize(EMU::BlobWriter&) {
    VPUX_THROW("VPU::DesparsifyOp is not supported by EMU");
}
