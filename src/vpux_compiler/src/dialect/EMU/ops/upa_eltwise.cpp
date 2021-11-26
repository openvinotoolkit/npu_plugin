//
// Copyright 2021 Intel Corporation.
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

#include "vpux/compiler/dialect/EMU/ops.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

EMU::BlobWriter::SpecificTask vpux::EMU::EltwiseUPAOp::serialize(EMU::BlobWriter& writer) {
    EMU::BlobWriter::String type;
    switch (this->type()) {
    case EMU::EltwiseLayerType::ADD:
        type = writer.createString("sum");
        break;
    case EMU::EltwiseLayerType::MULTIPLY:
        type = writer.createString("prod");
        break;
    case EMU::EltwiseLayerType::DIVIDE:
        type = writer.createString("div");
        break;
    case EMU::EltwiseLayerType::SQUARED_DIFF:
        type = writer.createString("sqdiff");
        break;
    case EMU::EltwiseLayerType::POWER:
        type = writer.createString("pow");
        break;
    case EMU::EltwiseLayerType::FLOOR_MOD:
        type = writer.createString("floormod");
        break;
    case EMU::EltwiseLayerType::MIN:
        type = writer.createString("min");
        break;
    case EMU::EltwiseLayerType::MAX:
        type = writer.createString("max");
        break;
    case EMU::EltwiseLayerType::AND:
        type = writer.createString("logicaland");
        break;
    case EMU::EltwiseLayerType::EQUAL:
        type = writer.createString("compareeq");
        break;
    default:
        VPUX_THROW("Unsupported EltwiseLayerType {0}", this->type());
    }

    MVCNN::EltwiseParamsBuilder builder(writer);
    builder.add_operation(type);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_EltwiseParams});
}
