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

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ReduceUPAOp::serialize(VPUIP::BlobWriter& writer) {
    VPUIP::BlobWriter::String type;

    switch (this->type()) {
    case VPUIP::ReduceLayerType::MAX:
        type = writer.createString("max");
        break;
    case VPUIP::ReduceLayerType::MEAN:
        type = writer.createString("mean");
        break;
    case VPUIP::ReduceLayerType::SUM:
        type = writer.createString("sum");
        break;
    case VPUIP::ReduceLayerType::LOGICALAND:
        type = writer.createString("logicalAnd");
        break;
    default:
        VPUX_THROW("Unsupported ReduceLayerType {0}", this->type());
    }

    MVCNN::ReduceParamsBuilder builder(writer);
    builder.add_keep_dims(checked_cast<bool>(keep_dims()));
    builder.add_operation(type);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ReduceParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseReduce(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                      ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 2, "UPAReduce supports only 2 input, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPAReduce supports only 1 output, got {0}", outputs.size());
    const auto params = task->softLayerParams_as_ReduceParams();
    const auto typeStr = params->operation()->str();

    VPUIP::ReduceLayerType type;
    if (typeStr == std::string("max")) {
        type = VPUIP::ReduceLayerType::MAX;
    } else if (typeStr == std::string("mean")) {
        type = VPUIP::ReduceLayerType::MEAN;
    } else if (typeStr == std::string("sum")) {
        type = VPUIP::ReduceLayerType::SUM;
    } else if (typeStr == std::string("logicalAnd")) {
        type = VPUIP::ReduceLayerType::LOGICALAND;
    } else {
        VPUX_THROW("Unsupported ReduceLayerType {0}", typeStr);
    }

    const auto keep_dims = params->keep_dims() ? mlir::UnitAttr::get(_ctx) : nullptr;

    return builder.create<VPUIP::ReduceUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], outputs[0], keep_dims,
                                              VPUIP::ReduceLayerTypeAttr::get(_ctx, type));
}
