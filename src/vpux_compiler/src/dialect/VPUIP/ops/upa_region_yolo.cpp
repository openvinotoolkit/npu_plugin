//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/blob_reader.hpp"
#include "vpux/compiler/utils/subspaces.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

void vpux::VPUIP::RegionYoloUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                         mlir::Value output, mlir::IntegerAttr coords, mlir::IntegerAttr classes,
                                         mlir::IntegerAttr regions, mlir::BoolAttr do_softmax, mlir::ArrayAttr mask) {
    build(builder, state, input, output, coords, classes, regions, do_softmax, mask, mlir::ValueRange{},
          mlir::ValueRange{}, nullptr, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::RegionYoloUPAOp::serialize(VPUIP::BlobWriter& writer) {
    VPUIP::BlobWriter::Vector<int32_t> serializedMask;
    if (mask().hasValue()) {
        serializedMask = writer.createVector(parseIntArrayAttr<int32_t>(mask().getValue()));
    }

    MVCNN::RegionYOLOParamsBuilder builder(writer);
    builder.add_coords(checked_cast<int32_t>(coords()));
    builder.add_classes(checked_cast<int32_t>(classes()));
    builder.add_num(checked_cast<int32_t>(regions()));
    builder.add_do_softmax(do_softmax().getValueOr(false));
    if (mask().hasValue()) {
        builder.add_mask(serializedMask);
    }

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_RegionYOLOParams});
}

bool vpux::VPUIP::RegionYoloUPAOp::isSupportedLayout(mlir::Operation* op, IE::DataOrderInfo& info) {
    VPUX_THROW_UNLESS(mlir::isa<IE::RegionYoloOp>(op), "Operation {0} is not RegionYolo", op->getName());

    auto regionYoloOp = mlir::dyn_cast<IE::RegionYoloOp>(op);
    if (regionYoloOp.do_softmax() == false) {
        return IERT::isSupportedLayoutSameInOutSpecificDimsOrder(op, info, IERT::CHW_HWC_NCHW_NHWC);
    } else {
        if (info.hasInput(0)) {
            const auto order = info.getInput(0);
            if (!std::count(IERT::CHW_HWC_NCHW_NHWC.cbegin(), IERT::CHW_HWC_NCHW_NHWC.cend(), order)) {
                if (order.numDims() == 3)
                    info.setInput(0, DimsOrder::CHW);
                else
                    info.setInput(0, DimsOrder::NCHW);
                return false;
            }
        }
        if (info.hasOutput(0)) {
            if (info.getOutput(0) != DimsOrder::NC) {
                info.setOutput(0, DimsOrder::NC);
                return false;
            }
        }
    }

    return true;
}
