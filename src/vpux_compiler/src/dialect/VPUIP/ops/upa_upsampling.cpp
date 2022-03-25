//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/subspaces.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::UpsamplingUPAOp::serialize(VPUIP::BlobWriter& writer) {
    SmallVector<int32_t> pad_x = {(int32_t)pad_l()[0].cast<mlir::IntegerAttr>().getInt(),
                                  (int32_t)pad_r()[0].cast<mlir::IntegerAttr>().getInt()};
    auto pad_x_vector = writer.createVector(pad_x);

    SmallVector<int32_t> pad_y = {(int32_t)pad_l()[1].cast<mlir::IntegerAttr>().getInt(),
                                  (int32_t)pad_r()[1].cast<mlir::IntegerAttr>().getInt()};
    auto pad_y_vector = writer.createVector(pad_y);

    SmallVector<int32_t> pad_z = {(int32_t)pad_l()[2].cast<mlir::IntegerAttr>().getInt(),
                                  (int32_t)pad_r()[2].cast<mlir::IntegerAttr>().getInt()};
    auto pad_z_vector = writer.createVector(pad_z);

    MVCNN::UpsamplingParamsBuilder builder(writer);
    builder.add_upsampling_factor_x((int32_t)upsampling_factor()[0].cast<mlir::IntegerAttr>().getInt());
    builder.add_upsampling_factor_y((int32_t)upsampling_factor()[1].cast<mlir::IntegerAttr>().getInt());
    builder.add_upsampling_factor_z((int32_t)upsampling_factor()[2].cast<mlir::IntegerAttr>().getInt());
    builder.add_pad_x(pad_x_vector);
    builder.add_pad_y(pad_y_vector);
    builder.add_pad_z(pad_z_vector);

    const auto paramsOff = builder.Finish();
    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_UpsamplingParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseUpsampling(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                          ArrayRef<mlir::Value> outputs,
                                                          const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 1, "UPAUpsampling supports only 1 input, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPAUpsampling supports only 1 output, got {0}", outputs.size());

    const auto* params = task->softLayerParams_as_UpsamplingParams();
    const SmallVector<int32_t, 3> upsampling_factor = {params->upsampling_factor_x(), params->upsampling_factor_y(),
                                                       params->upsampling_factor_z()};
    const SmallVector<int32_t, 3> pad_l = {params->pad_x()->Get(0), params->pad_y()->Get(0), params->pad_z()->Get(0)};
    const SmallVector<int32_t, 3> pad_r = {params->pad_x()->Get(1), params->pad_y()->Get(1), params->pad_z()->Get(1)};

    return builder.create<VPUIP::UpsamplingUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0],
                                                  getIntArrayAttr(_ctx, upsampling_factor),
                                                  getIntArrayAttr(_ctx, pad_l), getIntArrayAttr(_ctx, pad_r));
}
