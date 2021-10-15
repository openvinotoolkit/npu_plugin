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
#include "vpux/compiler/utils/subspaces.hpp"

#include <mlir/IR/BuiltinTypes.h>
using namespace vpux;

void vpux::VPUIPRegMapped::StridedSliceUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                                                    mlir::Value input, mlir::Value output, mlir::ArrayAttr begins,
                                                    mlir::ArrayAttr ends, mlir::ArrayAttr strides) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, begins, ends, strides, nullptr,
          nullptr);
}

void vpux::VPUIPRegMapped::StridedSliceUPAOp::serialize(std::vector<char>& buffer) {
    /*
    auto attrToVector = [&](mlir::ArrayAttr attr) {
        return to_std_vector(parseIntArrayAttr<uint32_t>(attr));
    };

    const auto beginsVec = attrToVector(begins());
    const auto endsVec = attrToVector(ends());
    const auto stridesVec = attrToVector(strides());

    const auto paramsOff = MVCNN::CreateStridedSliceParamsDirect(writer, &beginsVec, &endsVec, &stridesVec);

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_StridedSliceParams});
    */

    (void)buffer;
}
