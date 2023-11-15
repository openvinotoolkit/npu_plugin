//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::RandomUniformOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::RandomUniformOpAdaptor rand(operands, attrs);
    if (mlir::failed(rand.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = rand.min().getType().cast<vpux::NDTypeInterface>();
    const auto shape = parseIntArrayAttr<int64_t>(rand.output_shape());
    auto outShapeType = inType.changeShape(Shape(shape));
    auto outType = outShapeType.changeElemType(rand.output_type());
    inferredReturnTypes.push_back(outType);
    return mlir::success();
}

//
// EMU serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::RandomUniformOp::serialize(EMU::BlobWriter& /*writer*/) {
    VPUX_THROW("VPU::RandomUniformOp not supported");
}
