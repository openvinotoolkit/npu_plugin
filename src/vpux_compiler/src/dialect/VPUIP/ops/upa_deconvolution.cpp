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

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPUIP/blob_reader.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::verifyOp(DeconvolutionUPAOp op) {
    const auto featureShape = getShape(op.feature());
    const auto outputShape = getShape(op.output());

    if (featureShape.size() != 4)
        return errorAt(op, "Input shape should have 4 dimensions");

    if (featureShape[Dim(0)] != 1)
        return errorAt(op, "Input tensor [N C H W] = [{0} {1} {2} {3}] has unsupported dimension N != 1",
                       featureShape[Dim(0)], featureShape[Dim(1)], featureShape[Dim(2)], featureShape[Dim(3)]);

    if (featureShape[Dim(1)] != outputShape[Dim(1)] || featureShape[Dim(1)] != op.groups())
        return errorAt(op, "Only depth-wise deconvolution are supported");

    auto vec = parseIntArrayAttr<int64_t>(op.dilations());
    if (vec.size() != 2)
        return errorAt(op, "Supported dilations only [1, 1], got {0}", op.dilationsAttr());

    if (vec[0] != 1 || vec[1] != 1)
        return errorAt(op, "Supported dilations only [1, 1], got {0}", op.dilationsAttr());

    return mlir::success();
}

void vpux::VPUIP::DeconvolutionUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value feature,
                                            mlir::Value filter, mlir::Value output_shape, mlir::Value output,
                                            mlir::ArrayAttr strides, mlir::ArrayAttr pads_begin,
                                            mlir::ArrayAttr pads_end, mlir::ArrayAttr dilations,
                                            mlir::ArrayAttr output_padding, uint32_t groups) {
    build(builder, state, feature, filter, output_shape, output, mlir::ValueRange{}, mlir::ValueRange{}, strides,
          pads_begin, pads_end, dilations, output_padding, groups, nullptr, false);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::DeconvolutionUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::DeconvolutionParamsBuilder builder(writer);

    const auto strides = VPUIP::BlobWriter::createOrder3(this->strides());
    const auto padsBegin = VPUIP::BlobWriter::createOrder3(this->pads_begin());
    const auto padsEnd = VPUIP::BlobWriter::createOrder3(this->pads_end());
    const auto dilations = VPUIP::BlobWriter::createOrder3(this->dilations());
    const auto outputPadding = VPUIP::BlobWriter::createOrder3(this->output_padding());

    static const auto dY = Dim(3);
    static const auto dX = Dim(4);
    const auto filterShape = getShape(filter());
    const auto kernel =
            MVCNN::order3(checked_cast<uint8_t>(filterShape[dX]), checked_cast<uint8_t>(filterShape[dY]), 0);

    builder.add_kernel(&kernel);
    builder.add_strides(&strides);
    builder.add_pads_begin(&padsBegin);
    builder.add_pads_end(&padsEnd);
    builder.add_dilations(&dilations);
    builder.add_output_padding(&outputPadding);
    builder.add_is_depthwise(true);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_DeconvolutionParams});
}
