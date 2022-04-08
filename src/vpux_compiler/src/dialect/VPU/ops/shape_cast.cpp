//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;
using namespace VPU;

mlir::LogicalResult vpux::VPU::ShapeCastOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                             mlir::Optional<mlir::Location> optLoc,
                                                             mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                             mlir::RegionRange /*regions*/,
                                                             mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::ShapeCastOpAdaptor shapeCast(operands, attrs);
    if (mlir::failed(shapeCast.verify(loc))) {
        return mlir::failure();
    }

    const auto outShape = parseIntArrayAttr<int64_t>(shapeCast.shape());
    const auto inType = shapeCast.source().getType().cast<vpux::NDTypeInterface>();

    auto outType = inType.changeShape(Shape(outShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::ShapeCastOp::serialize(EMU::BlobWriter& /*writer*/) {
    VPUX_THROW("ShapeCastOp shouldn't have a serializer");
}
