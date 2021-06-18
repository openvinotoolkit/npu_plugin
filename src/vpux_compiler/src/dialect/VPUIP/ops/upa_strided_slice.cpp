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
#include "vpux/compiler/utils/subspaces.hpp"

#include <mlir/IR/BuiltinTypes.h>
using namespace vpux;

void vpux::VPUIP::StridedSliceUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                           mlir::Value output, mlir::ArrayAttr begins, mlir::ArrayAttr ends,
                                           mlir::ArrayAttr strides) {
    auto ctx = builder.getContext();
    auto arrayI64toUI32 = [&](mlir::ArrayAttr attr) {
        auto values = parseIntArrayAttr(attr) | transformed([](auto value) {
                          return checked_cast<uint32_t>(value);
                      });
        return getInt32ArrayAttr(ctx, values);
    };

    auto beginsUI32 = arrayI64toUI32(begins);
    auto endsUI32 = arrayI64toUI32(ends);
    auto stridesUI32 = arrayI64toUI32(strides);

    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, beginsUI32, endsUI32, stridesUI32,
          nullptr, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::StridedSliceUPAOp::serialize(VPUIP::BlobWriter& writer) {
    auto attrToVector = [&](mlir::ArrayAttr attr) {
        auto values = parseIntArrayAttr(attr) | transformed([](auto value) {
                          return checked_cast<uint32_t>(value);
                      });
        return to_std_vector(values);
    };

    const auto beginsVec = attrToVector(begins());
    const auto endsVec = attrToVector(ends());
    const auto stridesVec = attrToVector(strides());

    const auto paramsOff = MVCNN::CreateStridedSliceParamsDirect(writer, &beginsVec, &endsVec, &stridesVec);

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_StridedSliceParams});
}
