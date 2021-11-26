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

#include "vpux/compiler/dialect/EMU/ops.hpp"

#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::EMU::verifyOp(NormUPAOp op) {
    const auto inShape = getShape(op.input());

    if (inShape.size() == 4 && inShape[Dim(0)] != 1) {
        return errorAt(op, "Only input tensor batch = 1 is supported, got '{0}'", inShape[Dim(0)]);
    }

    const auto bias = op.bias().convertToDouble();
    if (bias != 1.0) {
        return errorAt(op, "Only bias = 1.0 is supported, got '{0}'", bias);
    }

    return mlir::success();
}

EMU::BlobWriter::SpecificTask vpux::EMU::NormUPAOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::NormParamsBuilder builder(writer);

    EMU::BlobWriter::String region;
    switch (this->region()) {
    case IE::LRN_IERegion::across:
        region = writer.createString("across");
        break;
    case IE::LRN_IERegion::same:
        region = writer.createString("same");
        break;
    default:
        VPUX_THROW("Unsupported LRN_IERegion {0}", this->region());
    }

    builder.add_alpha(static_cast<float>(alpha().convertToDouble()));
    builder.add_beta(static_cast<float>(beta().convertToDouble()));
    builder.add_local_size(checked_cast<int32_t>(local_size()));
    builder.add_region(region);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_NormParams});
}
