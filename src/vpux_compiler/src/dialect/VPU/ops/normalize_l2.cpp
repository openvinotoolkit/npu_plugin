//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::NormalizeL2Op::inferReturnTypes(mlir::MLIRContext* ctx,
                                                               mlir::Optional<mlir::Location> optLoc,
                                                               mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                               mlir::RegionRange /*regions*/,
                                                               mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::NormalizeL2OpAdaptor normalizeL2(operands, attrs);
    if (mlir::failed(normalizeL2.verify(loc))) {
        return mlir::failure();
    }

    auto axes = IE::constInputToData(loc, normalizeL2.axes());
    if (mlir::failed(axes)) {
        return mlir::failure();
    }

    const auto inType = normalizeL2.data().getType().cast<vpux::NDTypeInterface>();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::NormalizeL2Op::serialize(EMU::BlobWriter&) {
    VPUX_THROW("NormalizeL2Op implemented just on 37xx.");
}

//
// verify
//

mlir::LogicalResult vpux::VPU::NormalizeL2Op::verify() {
    const auto inRank = data().getType().cast<vpux::NDTypeInterface>().getRank();
    auto axesVec = parseIntArrayAttr<int64_t>(vpux::IE::getIntArrayAttrValue(axes()));

    for (auto& axis : axesVec) {
        if (axis < 0) {
            axis += inRank;
        }
    }

    bool isAllUnique = std::unique(axesVec.begin(), axesVec.end()) == axesVec.end();
    if (!isAllUnique) {
        return errorAt(*this, "Axes values should be unique");
    }

    return mlir::success();
}
