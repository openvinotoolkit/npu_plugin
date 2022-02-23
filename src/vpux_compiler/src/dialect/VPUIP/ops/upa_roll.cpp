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

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::RollUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::RollParamsBuilder builder(writer);
    builder.add_axis(checked_cast<int32_t>(axis()));
    const auto paramsOff = builder.Finish();
    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_RollParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseRoll(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                              ArrayRef<mlir::Value> outputs,
                                                              const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 3, "RollUPA supports only 3 inputs", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "RollUPA supports only 1 output", outputs.size());

    return builder.create<VPUIP::RollUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], inputs[3], outputs[0],
                                                      axis);
}
