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

#include "vpux/compiler/dialect/VPUIP/blob_reader.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/subspaces.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

void vpux::VPUIP::ScatterUpdateUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                            mlir::Value indices, mlir::Value updates, mlir::Value axis,
                                            mlir::Value output) {
    build(builder, state, input, indices, updates, axis, output, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ScatterUpdateUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::ScatterUpdateParamsBuilder builder(writer);
    const auto paramsOff = builder.Finish();
    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ScatterUpdateParams});
}
