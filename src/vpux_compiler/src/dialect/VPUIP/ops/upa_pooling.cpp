//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

namespace {

mlir::LogicalResult poolSizesCheck(VPUIP::PoolingUPAOp op, int64_t sizeI, int64_t sizeO, int64_t kernel, int64_t stride,
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

mlir::LogicalResult vpux::VPUIP::verifyOp(PoolingUPAOp op) {
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

    const auto kernel = parseIntArrayAttr(op.kernel());
    const auto strides = parseIntArrayAttr(op.strides());
    const auto padsBegin = parseIntArrayAttr(op.padsBegin());
    const auto padsEnd = parseIntArrayAttr(op.padsEnd());

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

    if (kernelY < 2 || kernelY > 64 || kernelX < 2 || kernelX > 64) {
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

void vpux::VPUIP::PoolingUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                      mlir::Value output, VPUIP::PoolLayerTypeAttr type, mlir::ArrayAttr kernel,
                                      mlir::ArrayAttr strides, mlir::ArrayAttr padsBegin, mlir::ArrayAttr padsEnd,
                                      mlir::UnitAttr excludePad) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, type, kernel, strides, padsBegin,
          padsEnd, excludePad, nullptr, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::PoolingUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto kernel = VPUIP::BlobWriter::createOrder3(this->kernel());
    const auto strides = VPUIP::BlobWriter::createOrder3(this->strides());
    const auto padsBegin = VPUIP::BlobWriter::createOrder3(this->padsBegin());
    const auto padsEnd = VPUIP::BlobWriter::createOrder3(this->padsEnd());

    VPUIP::BlobWriter::String type;
    switch (this->type()) {
    case VPUIP::PoolLayerType::MAX:
        type = writer.createString("max");
        break;
    case VPUIP::PoolLayerType::AVG:
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
