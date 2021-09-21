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

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::verifyOp(NormUPAOp op) {
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

void vpux::VPUIP::NormUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                   mlir::Value output, mlir::FloatAttr alpha, mlir::FloatAttr beta,
                                   mlir::FloatAttr bias, mlir::IntegerAttr local_size, IE::LRN_IERegionAttr region) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, alpha, beta, bias, local_size, region,
          nullptr, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::NormUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::NormParamsBuilder builder(writer);

    VPUIP::BlobWriter::String region;
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

mlir::Operation* vpux::VPUIP::BlobReader::parseNorm(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                    ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 1, "UPANorm supports only 1 input, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPANorm supports only 1 output, got {0}", outputs.size());
    const auto params = task->softLayerParams_as_NormParams();
    const auto regionStr = params->region()->str();

    IE::LRN_IERegion region;
    if (regionStr == std::string("across")) {
        region = IE::LRN_IERegion::across;
    } else if (regionStr == std::string("same")) {
        region = IE::LRN_IERegion::same;
    } else {
        VPUX_THROW("Unsupported LRN_IERegion {0}", regionStr);
    }

    const auto alpha = getFPAttr(_ctx, params->alpha());
    const auto beta = getFPAttr(_ctx, params->beta());
    const auto bias = getFPAttr(_ctx, 1.0);
    const auto local_size = getIntAttr(_ctx, params->local_size());

    return builder.create<VPUIP::NormUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0], alpha, beta, bias,
                                            local_size, IE::LRN_IERegionAttr::get(_ctx, region));
}
