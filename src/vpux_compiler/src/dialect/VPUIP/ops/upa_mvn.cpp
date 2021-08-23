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


void vpux::VPUIP::MVNUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                               mlir::Value output, mlir::BoolAttr across_channels,
                                               mlir::BoolAttr normalize_variance,  mlir::FloatAttr eps) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, across_channels, normalize_variance, eps,
          nullptr, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::MVNUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::MVNParamsBuilder builder(writer);
    builder.add_across_channels(across_channels().getValueOr(false));
    builder.add_normalize_variance(normalize_variance().getValueOr(false));
    builder.add_eps(static_cast<float>(eps().convertToDouble()));
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_MVNParams});
}

mlir::LogicalResult vpux::VPUIP::verifyOp(MVNUPAOp op) {
    const auto inShape = getShape(op.input());
 
    if (inShape.size() != 3 && inShape.size() != 4 && inShape.size() != 5) {
        return errorAt(op, "Input shape should have 3, 4 or 5 dimensions");
    }
    
    return mlir::success();
 
}
