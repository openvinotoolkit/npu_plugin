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

void vpux::VPUIP::NormalizeIEUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value data,
                                          mlir::Value axes, mlir::Value output, mlir::FloatAttr eps,
                                          mlir::BoolAttr across_spatial, mlir::BoolAttr channel_shared) {
    build(builder, state, data, axes, output, mlir::ValueRange{}, mlir::ValueRange{}, eps, across_spatial,
          channel_shared, nullptr, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::NormalizeIEUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::NormalizeParamsBuilder builder(writer);
    builder.add_eps(static_cast<float>(eps().convertToDouble()));
    builder.add_across_spatial(static_cast<int32_t>(across_spatial()));
    builder.add_channel_shared(static_cast<int32_t>(channel_shared()));

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_NormalizeParams});
}
