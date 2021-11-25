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

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

namespace {

mlir::LogicalResult poolSizesCheck(EMU::PoolingUPAOp op, int64_t sizeI, int64_t sizeO, int64_t kernel, int64_t stride,
                                   int64_t lPad, int64_t rPad) {
    if (sizeI <= 0 || sizeO <= 0 || kernel <= 0 || stride <= 0) {
        return errorAt(op, "sizeI, sizeO kernel, stride negative values do not make sense");
    }

    if (lPad < 0) {
        return errorAt(op, "Left/top side padding can not be negative '{0}'", lPad);
    }
    if (rPad < 0) {
        return errorAt(op, "Right/bottom side padding can not be negative '{0}'", rPad);
    }

    if (lPad > (kernel - 1)) {
        return errorAt(op, "Left/top padding is too big '{0}'", lPad);
    }
    if ((sizeO - 1) * stride - lPad > sizeI - 1) {
        return errorAt(op, "Output size is too big, the last kernel application is out of real data '{0}'", sizeO);
    }
    if ((sizeO - 1) * stride - lPad + (kernel - 1) > sizeI - 1 + rPad) {
        return errorAt(op, "The last kernel application is out of input size + rPad range");
    }

    return mlir::success();
}

}  // namespace

mlir::LogicalResult vpux::EMU::verifyOp(PoolingUPAOp op) {
    static const auto N = Dim(0);
    static const auto C = Dim(1);
    static const auto H = Dim(2);
    static const auto W = Dim(3);

    const auto inShape = getShape(op.input());
    const auto outShape = getShape(op.output());

    if (inShape[N] != outShape[N]) {
        return errorAt(op, "Input batch '{0}' doesn't match with output '{1}'", inShape[N], outShape[N]);
    }
    if (inShape[C] != outShape[C]) {
        return errorAt(op, "Input number of channels '{0}' doesn't match with output '{1}'", inShape[C], outShape[C]);
    }

    const auto kernel = parseIntArrayAttr<int64_t>(op.kernel());
    const auto strides = parseIntArrayAttr<int64_t>(op.strides());
    const auto padsBegin = parseIntArrayAttr<int64_t>(op.padsBegin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(op.padsEnd());

    if (kernel.size() != 2) {
        return errorAt(op, "Got unsupported kernel '{0}', only 2D is supported", kernel);
    }
    if (strides.size() != 2) {
        return errorAt(op, "Got unsupported strides '{0}', only 2D is supported", strides);
    }
    if (padsBegin.size() != 2) {
        return errorAt(op, "Got unsupported padsBegin '{0}', only 2D is supported", padsBegin);
    }
    if (padsEnd.size() != 2) {
        return errorAt(op, "Got unsupported padsEnd '{0}', only 2D is supported", padsEnd);
    }

    const auto kernelY = kernel[0];
    const auto kernelX = kernel[1];

    if (kernelY < 1 || kernelY > 64 || kernelX < 1 || kernelX > 64) {
        return errorAt(op, "Got unsupported kernel '{0}', only up to 64 is supported", kernel);
    }

    const auto strideY = strides[0];
    const auto strideX = strides[1];

    const auto padY = padsBegin[0];
    const auto padX = padsBegin[1];

    const auto bpadY = padsEnd[0];
    const auto rpadX = padsEnd[1];

    if (mlir::failed(poolSizesCheck(op, inShape[W], outShape[W], kernelX, strideX, padX, rpadX))) {
        return mlir::failure();
    }
    if (mlir::failed(poolSizesCheck(op, inShape[H], outShape[H], kernelY, strideY, padY, bpadY))) {
        return mlir::failure();
    }

    return mlir::success();
}

EMU::BlobWriter::SpecificTask vpux::EMU::PoolingUPAOp::serialize(EMU::BlobWriter& writer) {
    const auto kernel = EMU::BlobWriter::createOrder3(this->kernel());
    const auto strides = EMU::BlobWriter::createOrder3(this->strides());
    const auto padsBegin = EMU::BlobWriter::createOrder3(this->padsBegin());
    const auto padsEnd = EMU::BlobWriter::createOrder3(this->padsEnd());

    EMU::BlobWriter::String type;
    switch (this->type()) {
    case EMU::PoolLayerType::MAX:
        type = writer.createString("max");
        break;
    case EMU::PoolLayerType::AVG:
        type = writer.createString("avg");
        break;
    default:
        VPUX_THROW("Unsupported PoolLayerType {0}", this->type());
    }

    const auto excludePad = writer.createString(this->excludePad() ? "true" : "false");

    MVCNN::PoolingParamsBuilder builder(writer);
    builder.add_pool_method(type);
    builder.add_kernel(&kernel);
    builder.add_strides(&strides);
    builder.add_pads_begin(&padsBegin);
    builder.add_pads_end(&padsEnd);
    builder.add_exclude_pad(excludePad);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PoolingParams});
}