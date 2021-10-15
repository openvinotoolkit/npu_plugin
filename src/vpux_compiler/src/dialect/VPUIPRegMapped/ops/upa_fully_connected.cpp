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

#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"

#include "vpux/compiler/utils/analysis.hpp"

using namespace vpux;

void vpux::VPUIPRegMapped::FullyConnectedUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                                                      mlir::Value input, mlir::Value weights, mlir::Value bias,
                                                      mlir::Value output) {
    build(builder, state, input, weights, bias, output, mlir::ValueRange{}, mlir::ValueRange{}, nullptr, false);
}

void vpux::VPUIPRegMapped::FullyConnectedUPAOp::inferLayoutInfo(mlir::Operation*, IE::LayerLayoutInfo& info) {
    IE::fillDefaultLayoutInfo(info);
}

// VPUIPRegMapped::BlobWriter::SpecificTask
// vpux::VPUIPRegMapped::FullyConnectedUPAOp::serialize(VPUIPRegMapped::BlobWriter& writer) {
void vpux::VPUIPRegMapped::FullyConnectedUPAOp::serialize(std::vector<char>& buffer) {
    /*
    MVCNN::FullyConnectedParamsBuilder builder(writer);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_FullyConnectedParams});
    */

    (void)buffer;
}
