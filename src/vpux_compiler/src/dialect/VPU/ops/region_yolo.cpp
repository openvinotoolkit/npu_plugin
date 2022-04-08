//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::RegionYoloOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                              mlir::Optional<mlir::Location> optLoc,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::RegionRange /*regions*/,
                                                              mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::RegionYoloOpAdaptor regionYolo(operands, attrs);
    if (mlir::failed(regionYolo.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = regionYolo.input().getType().cast<vpux::NDTypeInterface>();

    SmallVector<int64_t> outputShape;
    if (regionYolo.do_softmax()) {
        for (int64_t i = 0; i < regionYolo.axis(); i++) {
            outputShape.push_back(inType.getShape().raw()[i]);
        }

        size_t flat_dim = 1;
        for (int64_t i = regionYolo.axis(); i < regionYolo.end_axis() + 1; i++) {
            flat_dim *= inType.getShape().raw()[i];
        }
        outputShape.push_back(flat_dim);

        for (size_t i = regionYolo.end_axis() + 1; i < inType.getShape().size(); i++) {
            outputShape.push_back(inType.getShape().raw()[i]);
        }
    } else {
        outputShape.push_back(inType.getShape().raw()[0]);
        outputShape.push_back((regionYolo.classes() + regionYolo.coords() + 1) *
                              checked_cast<int64_t>(regionYolo.mask().size()));
        outputShape.push_back(inType.getShape().raw()[2]);
        outputShape.push_back(inType.getShape().raw()[3]);
    }

    const auto outType = inType.changeShape(Shape(outputShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

void vpux::VPU::RegionYoloOp::inferLayoutInfo(mlir::Operation* origOp, IE::LayerLayoutInfo& info) {
    auto regionYoloOp = mlir::dyn_cast<IE::RegionYoloOp>(origOp);
    VPUX_THROW_UNLESS(regionYoloOp != nullptr, "Operation '{0}' is not a RegionYolo", origOp->getName());

    if (regionYoloOp.do_softmax()) {
        IE::fillDefaultLayoutInfo(info);
    } else {
        VPU::inferLayoutInfoSameInOutSpecificDimsOrder(info, {DimsOrder::NCHW});
    }
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::RegionYoloOp::serialize(EMU::BlobWriter& writer) {
    EMU::BlobWriter::Vector<int32_t> serializedMask;
    serializedMask = writer.createVector(parseIntArrayAttr<int32_t>(mask()));

    MVCNN::RegionYOLOParamsBuilder builder(writer);
    builder.add_coords(checked_cast<int32_t>(coords()));
    builder.add_classes(checked_cast<int32_t>(classes()));
    builder.add_num(checked_cast<int32_t>(regions()));
    builder.add_do_softmax(do_softmax());
    builder.add_mask(serializedMask);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_RegionYOLOParams});
}
