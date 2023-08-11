//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

namespace {

Dim normalizeAxis(VPU::SplitOpAdaptor split) {
    VPUX_THROW_UNLESS(split.axis_value().hasValue(), "Got non constant axis");

    const auto inType = split.input().getType().cast<vpux::NDTypeInterface>();
    const auto inRank = inType.getRank();

    auto axisInd = split.axis_value().getValue();

    // Negative value means counting dimension from the end
    if (axisInd < 0) {
        axisInd += inRank;
    }

    VPUX_THROW_UNLESS(axisInd >= 0 && axisInd < inRank, "Got wrong Split axis '{0}', out of range '{1}'", axisInd,
                      inRank);

    return Dim(axisInd);
}

mlir::FailureOr<Dim> extractAxis(mlir::Location loc, VPU::SplitOpAdaptor split) {
    if (split.axis() != nullptr) {
        auto axisConst = split.axis().getDefiningOp<Const::DeclareOp>();
        if (axisConst == nullptr) {
            return errorAt(loc, "Only constant input is supported for axis");
        }

        const auto axisContent = axisConst.content();
        if (!axisContent.isSplat()) {
            return errorAt(loc, "Axis value must be a scalar");
        }

        const auto inType = split.input().getType().cast<vpux::NDTypeInterface>();
        const auto inRank = inType.getRank();

        auto axisInd = axisContent.getSplatValue<int64_t>();

        // Negative value means counting dimension from the end
        if (axisInd < 0) {
            axisInd += inRank;
        }

        VPUX_THROW_UNLESS(axisInd >= 0 && axisInd < inRank, "Got wrong Split axis '{0}', out of range '{1}'", axisInd,
                          inRank);

        return Dim(axisInd);
    } else if (split.axis_value().hasValue()) {
        return normalizeAxis(split);
    } else {
        return errorAt(loc, "Axis was not provided");
    }
}

}  // namespace

mlir::LogicalResult vpux::VPU::SplitOp::inferReturnTypes(mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc,
                                                         mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                         mlir::RegionRange /*regions*/,
                                                         mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::SplitOpAdaptor split(operands, attrs);
    if (mlir::failed(split.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = split.input().getType().cast<vpux::NDTypeInterface>();

    const auto axis = extractAxis(loc, split);
    if (mlir::failed(axis)) {
        return mlir::failure();
    }

    const auto num_splits = split.num_splits();

    auto outShape = inType.cast<vpux::NDTypeInterface>().getShape().toValues();
    if ((outShape[*axis] < num_splits) || (outShape[*axis] % num_splits != 0)) {
        return errorAt(loc, "Unsupported num_splits parameter");
    }
    outShape[*axis] /= num_splits;

    for (int i = 0; i < num_splits; ++i) {
        const auto outType = inType.changeShape(Shape(outShape.raw()));
        inferredReturnTypes.push_back(outType);
    }

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::SplitOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::SplitParamsBuilder builder(writer);
    const auto paramsOff = builder.Finish();
    builder.add_axis(checked_cast<uint32_t>(axis_valueAttr().getValue().getSExtValue()));
    builder.add_num_splits(checked_cast<uint32_t>(num_splitsAttr().getValue().getSExtValue()));

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_SplitParams});
}
