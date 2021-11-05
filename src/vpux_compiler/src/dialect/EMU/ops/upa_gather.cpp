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

#include "vpux/compiler/dialect/EMU/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/subspaces.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

mlir::LogicalResult vpux::EMU::verifyOp(GatherUPAOp op) {
    // Axis should not exceed input rank
    const auto axisNo = op.axis();
    const auto inShape = getShape(op.input());
    if (checked_cast<size_t>(axisNo) >= inShape.size()) {
        return errorAt(op, "Gather axis '{0}' is out of range [0,{1}]", axisNo, inShape.size() - 1);
    }

    return mlir::success();
}

EMU::BlobWriter::SpecificTask vpux::EMU::GatherUPAOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::GatherParamsBuilder builder(writer);
    builder.add_axis(checked_cast<uint32_t>(axis()));
    const auto paramsOff = builder.Finish();
    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_GatherParams});
}
