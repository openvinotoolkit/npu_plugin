//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/dilated_utils.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult vpux::VPU::ExpandDilatedOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::ExpandDilatedOpAdaptor expandDilated(operands, attrs);
    if (mlir::failed(expandDilated.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = expandDilated.input().getType().dyn_cast<vpux::NDTypeInterface>();
    if (!inType) {
        return mlir::failure();
    }

    const auto dilations = parseIntArrayAttr<int64_t>(expandDilated.dilations());
    const auto outType = getDilatedType(inType, ShapeRef(dilations));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// inferElemTypeInfo
//

void vpux::VPU::ExpandDilatedOp::inferElemTypeInfo(vpux::IE::LayerDataInfo<mlir::Type>& info) {
    const auto inputElemType = info.getInput(0);

    for (size_t outputInd = 0; outputInd < info.getNumOutputs(); ++outputInd) {
        info.setOutput(outputInd, inputElemType);
    }
}

void vpux::VPU::ExpandDilatedOp::inferElemTypeInfoUp(vpux::IE::LayerDataInfo<mlir::Type>& info) {
    const auto outputElemType = info.getOutput(0);

    for (size_t inputInd = 0; inputInd < info.getNumInputs(); ++inputInd) {
        info.setInput(inputInd, outputElemType);
    }
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::ExpandDilatedOp::serialize(EMU::BlobWriter& /*writer*/) {
    VPUX_THROW("Unreacheable code, since all ExpandDilatedOp ops should be folded");
}
