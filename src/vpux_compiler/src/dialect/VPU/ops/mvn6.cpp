//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::MVN6Op::inferReturnTypes(mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc,
                                                        mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                        mlir::RegionRange /*regions*/,
                                                        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::MVN6OpAdaptor mvn(operands, attrs);
    if (mlir::failed(mvn.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mvn.input().getType().cast<vpux::NDTypeInterface>();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}

// Return a list with all dims that are not in 'axes' list.
// (useful for tiling)
DimArr vpux::VPU::MVN6Op::getNonNormDims() {
    const auto rank = input().getType().cast<vpux::NDTypeInterface>().getRank();
    VPUX_THROW_UNLESS(rank == 4, "Function valid only for 4D shape, got {0}D", rank);

    DimArr dims;
    const auto axes = parseIntArrayAttr<int64_t>(axesAttr());
    DimArr allDims = {Dims4D::Act::N, Dims4D::Act::C, Dims4D::Act::H, Dims4D::Act::W};
    for (const auto dim : allDims) {
        if (std::find(axes.begin(), axes.end(), dim.ind()) != axes.end()) {
            continue;
        } else {
            dims.push_back(dim);
        }
    }
    return dims;
}

void vpux::VPU::MVN6Op::inferLayoutInfo(mlir::Operation*, IE::LayerLayoutInfo& info) {
    VPU::inferLayoutInfoSameInOutSpecificDimsOrder(
            info, {DimsOrder::NCHW, DimsOrder::NCWH, DimsOrder::NHWC, DimsOrder::NWHC});
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::MVN6Op::serialize(EMU::BlobWriter&) {
    VPUX_THROW("VPU::MVN6Op not supported");
}
