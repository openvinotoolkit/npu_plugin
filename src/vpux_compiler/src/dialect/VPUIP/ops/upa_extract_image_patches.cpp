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

//#include "vpux/compiler/core/attributes/dim.hpp"
//#include "vpux/compiler/core/attributes/shape.hpp"
//#include "vpux/compiler/utils/analysis.hpp"
//#include "vpux/compiler/utils/subspaces.hpp"
// #include "vpux/compiler/utils/subspaces.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

void vpux::VPUIP::ExtractImagePatchesUPAOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                      mlir::Value data, mlir::Value output,
                                      mlir::ArrayAttr sizes, mlir::ArrayAttr strides, mlir::ArrayAttr rates,
                                      IE::PadTypeAttr auto_pad) {
   build(odsBuilder, odsState, data, output, sizes, strides, rates, auto_pad);
}

mlir::LogicalResult vpux::VPUIP::verifyOp(ExtractImagePatchesUPAOp op) {
    const auto inShape = getShape(op.data());

    if (inShape.size() != 4 ) {
        return errorAt(op, "Dimension of the input data should be 4D tensor. Got {0} D tensor", inShape.size());
    }

    const auto sizes = parseIntArrayAttr<int64_t>(op.sizes());
    if (sizes.size() != 2) {
        return errorAt(op, "Dimension of sizes attributes is expected to be equal to 2. Got {0}", sizes.size());
    }

    if (sizes[0] <= 0 || sizes[1] <= 0) {
        return errorAt(op, "Sizes attributes sizeRows and sizeCols should be positive.");
    }

    const auto strides = parseIntArrayAttr<int64_t>(op.strides());

    if (strides.size() != 2) {
        return errorAt(op, "Dimension of strides attributes is expected to be equal to 2. Got {0}", strides.size());
    }

    if (strides[0] <= 0 || strides[1] <= 0) {
        return errorAt(op, "strides attributes stridesRows and stridesCols should be positive.");
    }

    const auto rates = parseIntArrayAttr<int64_t>(op.rates());

    if (rates.size() != 2) {
        return errorAt(op, "Dimension of rates attributes is expected to be equal to 2. Got {0}", rates.size());
    }

    if (rates[0] <= 0 || rates[1] <= 0) {
        return errorAt(op, "rates attributes ratesRows and ratesCols should be positive.");
    }

    return mlir::success();
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ExtractImagePatchesUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::ExtractImagePatchesParamsBuilder builder(writer);

    MVCNN::ExtractImagePatchesPadMode vpux_padding;

     if (this->auto_pad() == IE::PadType::SAME_UPPER) {
        vpux_padding = MVCNN::ExtractImagePatchesPadMode::ExtractImagePatchesPadMode_SAME_UPPER;
    } else if (this->auto_pad() == IE::PadType::SAME_LOWER) {
        vpux_padding = MVCNN::ExtractImagePatchesPadMode::ExtractImagePatchesPadMode_SAME_LOWER;
    } else if (this->auto_pad() == IE::PadType::VALID) {
        vpux_padding = MVCNN::ExtractImagePatchesPadMode::ExtractImagePatchesPadMode_VALID;
    } else {
        VPUX_THROW("Unsupported pad type {0}", this->auto_pad());
    }

    const auto sizes = parseIntArrayAttr<int64_t>(sizesAttr());
    const auto strides = parseIntArrayAttr<int64_t>(stridesAttr());
    const auto rates = parseIntArrayAttr<int64_t>(ratesAttr());

    //sizes is a size [size_rows, size_cols]
    builder.add_sizeRows(checked_cast<int32_t>(sizes[0]));
    builder.add_sizeCols(checked_cast<int32_t>(sizes[1]));

    //strides is a distance [stride_rows, stride_cols]
    builder.add_strideRows(checked_cast<int32_t>(strides[0]));
    builder.add_strideCols(checked_cast<int32_t>(strides[1]));

    // rates is the input stride [rate_rows, rate_cols],
    builder.add_rateRows(checked_cast<int32_t>(rates[0]));
    builder.add_rateCols(checked_cast<int32_t>(rates[1]));

    builder.add_autoPad(vpux_padding);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ExtractImagePatchesParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseExtractImagePatches(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                       ArrayRef<mlir::Value> outputs,
                                                       const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 1, "UPAExtractImagePatches supports only 1 input, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPAExtractImagePatches supports only 1 output, got {0}", outputs.size());

    const auto params = task->softLayerParams_as_ExtractImagePatchesParams();

    const auto sizes = getIntArrayAttr(_ctx, SmallVector<int32_t>{params->sizeRows(), params->sizeCols()});
    const auto strides = getIntArrayAttr(_ctx, SmallVector<int32_t>{params->strideRows(), params->strideCols()});
    const auto rates = getIntArrayAttr(_ctx, SmallVector<int32_t>{params->rateRows(), params->rateCols()});

    IE::PadType padding;
    switch (params->autoPad()) {
    case 0:
        padding = IE::PadType::SAME_LOWER;
        break;
    case 1:
        padding = IE::PadType::SAME_UPPER;
        break;
    case 2:
        padding = IE::PadType::VALID;
        break;
    default:
        VPUX_THROW("Unknown PadType. same upper (same lower) and valid types are supported only");
    }

    return builder.create<VPUIP::ExtractImagePatchesUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0],
                                               outputs[0], sizes, strides, rates,
                                               IE::PadTypeAttr::get(_ctx, padding));
}
