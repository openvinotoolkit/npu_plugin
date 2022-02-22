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

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/subspaces.hpp"

using namespace vpux;

void vpux::VPUIP::ReorgYoloUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                        mlir::Value output, mlir::IntegerAttr stride) {
    build(builder, state, input, output, stride, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ReorgYoloUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::ReorgYOLOParamsBuilder builder(writer);
    builder.add_stride(checked_cast<uint32_t>(stride()));
    const auto paramsOff = builder.Finish();
    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ReorgYOLOParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseReorgYolo(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                         ArrayRef<mlir::Value> outputs,
                                                         const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 1, "ReorgYoloUPA supports only 1 input, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "ReorgYoloUPA supports only 1 output, got {0}", outputs.size());

    const auto params = task->softLayerParams_as_ReorgYOLOParams();
    const auto stride = getIntAttr(_ctx, params->stride());
    return builder.create<VPUIP::ReorgYoloUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0], stride);
}
