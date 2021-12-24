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

void vpux::VPUIP::UpsamplingUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                         mlir::ArrayAttr upsampling_factor, mlir::ArrayAttr pad_l,
                                         mlir::ArrayAttr pad_r, mlir::Value output) {
    build(builder, state, input, output, upsampling_factor, pad_l, pad_r, nullptr, false);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::UpsamplingUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::UpsamplingParamsBuilder builder(writer);
    builder.add_upsampling_factor_x((int32_t)upsampling_factor()[0].cast<mlir::IntegerAttr>().getInt());
    builder.add_upsampling_factor_y((int32_t)upsampling_factor()[1].cast<mlir::IntegerAttr>().getInt());
    builder.add_upsampling_factor_z((int32_t)upsampling_factor()[2].cast<mlir::IntegerAttr>().getInt());

    SmallVector<int32_t> pad_x = {(int32_t)pad_l()[0].cast<mlir::IntegerAttr>().getInt(),
                                  (int32_t)pad_r()[0].cast<mlir::IntegerAttr>().getInt()};
    builder.add_pad_x(writer.createVector(pad_x));

    SmallVector<int32_t> pad_y = {(int32_t)pad_l()[1].cast<mlir::IntegerAttr>().getInt(),
                                  (int32_t)pad_r()[1].cast<mlir::IntegerAttr>().getInt()};
    builder.add_pad_y(writer.createVector(pad_y));

    SmallVector<int32_t> pad_z = {(int32_t)pad_l()[2].cast<mlir::IntegerAttr>().getInt(),
                                  (int32_t)pad_r()[2].cast<mlir::IntegerAttr>().getInt()};
    builder.add_pad_z(writer.createVector(pad_z));

    const auto paramsOff = builder.Finish();
    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_UpsamplingParams});
}
