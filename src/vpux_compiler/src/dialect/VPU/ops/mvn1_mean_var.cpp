//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::MVN1MeanVarOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                               mlir::Optional<mlir::Location> optLoc,
                                                               mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                               mlir::RegionRange /*regions*/,
                                                               mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::MVN1MeanVarOpAdaptor op(operands, attrs);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto iType = op.sum().getType().cast<vpux::NDTypeInterface>();
    const auto iShape = iType.getShape().raw();
    const auto iOrder = iType.getDimsOrder();
    const auto inN = iShape[0];

    // For default order NxCxW shape, expecting data in memory as NxWxC (i.e C-minor)
    // for alignment with original MvnOp main tensor NHWC layout.
    // The (0,1,2) -> (0,2,1) permutation is available via 'DimsOrder::CWH'
    VPUX_THROW_UNLESS(iOrder == DimsOrder::CWH, "Expecting CWH layout, got {0}", iOrder);

    const auto fullShape = parseIntArrayAttr<int64_t>(op.orig_shape());
    const auto fullC = fullShape[Dims4D::Act::C.ind()];
    const auto fullN = fullShape[Dims4D::Act::N.ind()];

    VPUX_THROW_UNLESS(inN == fullN, "Mismatch N: {0} != {1}", inN, fullN);

    const auto outC = (op.across_channels() ? 1 : fullC);
    const auto outW = op.normalize_variance() ? 2 : 1;  // {mean, var} or {mean}

    SmallVector<int64_t> oShape{inN, outC, outW};
    auto oShapeType = iType.changeShape(Shape(oShape));
    auto oType = oShapeType.changeElemType(op.output_type());
    inferredReturnTypes.push_back(oType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::MVN1MeanVarOp::serialize(EMU::BlobWriter&) {
    VPUX_THROW("VPU::MVN1MeanVarOp not supported");
}
