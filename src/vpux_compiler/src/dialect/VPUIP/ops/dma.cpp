//
// Copyright 2020 Intel Corporation.
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

#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

//
// UPADMAOp
//

void vpux::VPUIP::UPADMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value src,
                                  mlir::Value dst) {
    build(builder, state, src, dst, mlir::ValueRange{}, mlir::ValueRange{});
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::UPADMAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto inputOff = writer.getTensor(input());
    const auto outputOff = writer.getTensor(output());

    MVCNN::UPADMATaskBuilder builder(writer);
    builder.add_src(inputOff);
    builder.add_dst(outputOff);
    return {builder.Finish().Union(), MVCNN::SpecificTask_UPADMATask};
}

//
// NNDMAOp
//

void vpux::VPUIP::NNDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value src,
                                 mlir::Value dst) {
    build(builder, state, src, dst, mlir::ValueRange{}, mlir::ValueRange{}, false);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::NNDMAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto srcOff = writer.getTensor(input());
    const auto dstOff = writer.getTensor(output_buff());

    MVCNN::NNDMATaskBuilder builder(writer);
    builder.add_src(srcOff);
    builder.add_dst(dstOff);
    builder.add_compression(compression());
    builder.add_port(checked_cast<uint8_t>(port()));
    return {builder.Finish().Union(), MVCNN::SpecificTask_NNDMATask};
}
