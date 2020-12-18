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

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"

using namespace vpux;

void vpux::VPUIP::ConvolutionUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                          mlir::Value filter, mlir::Value output, mlir::ArrayAttr strides,
                                          mlir::ArrayAttr dilations, mlir::ArrayAttr padsBegin,
                                          mlir::ArrayAttr padsEnd) {
    build(builder, state, input, filter, nullptr, output, mlir::ValueRange{}, mlir::ValueRange{}, strides, dilations,
          padsBegin, padsEnd, 1, nullptr, false);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ConvolutionUPAOp::serialize(VPUIP::BlobWriter& writer) {
    static const auto dY = Dim(2);
    static const auto dX = Dim(3);

    const auto strides = VPUIP::BlobWriter::createOrder3(this->strides());
    const auto dilations = VPUIP::BlobWriter::createOrder3(this->dilations());
    const auto padsBegin = VPUIP::BlobWriter::createOrder3(this->padsBegin());
    const auto padsEnd = VPUIP::BlobWriter::createOrder3(this->padsEnd());

    const auto filterType = filter().getType().cast<mlir::ShapedType>();
    const auto filterShape = getShape(filterType);
    const auto kernel =
            MVCNN::order3(checked_cast<uint8_t>(filterShape[dX]), checked_cast<uint8_t>(filterShape[dY]), 0);

    MVCNN::ConvolutionParamsBuilder builder(writer);
    builder.add_kernel(&kernel);
    builder.add_strides(&strides);
    builder.add_dilations(&dilations);
    builder.add_pads_begin(&padsBegin);
    builder.add_pads_end(&padsEnd);
    builder.add_group(checked_cast<int32_t>(group()));
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(getOperation(), {paramsOff.Union(), MVCNN::SoftwareLayerParams_ConvolutionParams},
                                     maxShaves(), isTrailingSWLayer());
}
