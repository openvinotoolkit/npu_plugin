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
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/subspaces.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

void vpux::VPUIP::ReduceUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                     mlir::Value axes, mlir::Value output, mlir::BoolAttr keep_dims,
                                     VPUIP::ReduceLayerTypeAttr type) {
    build(builder, state, input, axes, output, mlir::ValueRange{}, mlir::ValueRange{}, keep_dims, type, nullptr,
          nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ReduceUPAOp::serialize(VPUIP::BlobWriter& writer) {
    VPUIP::BlobWriter::String type;
    switch (this->type()) {
    case VPUIP::ReduceLayerType::MEAN:
        type = writer.createString("mean");
        break;
    default:
        VPUX_THROW("Unsupported ReduceLayerType {0}", this->type());
    }

    const auto keepDims = keep_dimsAttr().getValue();

    MVCNN::ReduceParamsBuilder builder(writer);
    builder.add_operation(type);
    builder.add_keep_dims(keepDims);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ReduceParams});
}
