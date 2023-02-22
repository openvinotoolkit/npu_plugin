//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::ExtractImagePatchesOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::ExtractImagePatchesOpAdaptor extractImagePatches(operands, attrs);
    if (mlir::failed(extractImagePatches.verify(loc))) {
        return mlir::failure();
    }

    const auto paddingType = extractImagePatches.autoPad();
    const auto inputType = extractImagePatches.data().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inputType.getShape();

    const auto sizes = parseIntArrayAttr<int64_t>(extractImagePatches.sizes());
    const auto strides = parseIntArrayAttr<int64_t>(extractImagePatches.strides());
    const auto rates = parseIntArrayAttr<int64_t>(extractImagePatches.rates());

    const auto checkAttributes = [loc](ArrayRef<int64_t> values, StringRef name) {
        if (values.size() != 2) {
            return errorAt(loc, "Dimension of {0} attributes is expected to be equal to 2. Got {1}", name,
                           values.size());
        }

        if (values[0] < 0 || values[1] < 0) {
            return errorAt(loc, "{0} attributes Rows and Cols should be non-negative integer number.", name);
        }

        return mlir::success();
    };

    if (mlir::failed(checkAttributes(sizes, "sizes"))) {
        return mlir::failure();
    }

    if (mlir::failed(checkAttributes(strides, "strides"))) {
        return mlir::failure();
    }

    if (mlir::failed(checkAttributes(rates, "rates"))) {
        return mlir::failure();
    }

    SmallVector<int64_t> outputShape;

    int64_t inputRows = inputShape[Dims4D::Act::H];
    int64_t inputCols = inputShape[Dims4D::Act::W];

    int64_t outputRows(0);
    int64_t outputCols(0);

    if (paddingType == IE::PadType::SAME_UPPER || paddingType == IE::PadType::SAME_LOWER) {
        outputRows = 1 + (inputRows - 1) / strides[0];
        outputCols = 1 + (inputCols - 1) / strides[1];
    } else if (paddingType == IE::PadType::VALID) {
        outputRows = (inputRows - rates[0] * (sizes[0] - 1) - 1) / strides[0] + 1;
        outputCols = (inputCols - rates[1] * (sizes[1] - 1) - 1) / strides[1] + 1;
    }

    VPUX_THROW_UNLESS(outputRows >= 1 && outputCols >= 1, "Invalid inferred output spatial size: H={0}, W={1}",
                      outputRows, outputCols);
    outputShape.push_back(inputShape[Dims4D::Act::N]);
    outputShape.push_back(inputShape[Dims4D::Act::C] * sizes[0] * sizes[1]);
    outputShape.push_back(outputRows);
    outputShape.push_back(outputCols);

    const auto outType = inputType.changeShape(Shape(outputShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::ExtractImagePatchesOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::ExtractImagePatchesParamsBuilder builder(writer);

    MVCNN::ExtractImagePatchesPadMode vpux_padding;

    if (this->autoPad() == IE::PadType::SAME_UPPER) {
        vpux_padding = MVCNN::ExtractImagePatchesPadMode::ExtractImagePatchesPadMode_SAME_UPPER;
    } else if (this->autoPad() == IE::PadType::SAME_LOWER) {
        vpux_padding = MVCNN::ExtractImagePatchesPadMode::ExtractImagePatchesPadMode_SAME_LOWER;
    } else if (this->autoPad() == IE::PadType::VALID) {
        vpux_padding = MVCNN::ExtractImagePatchesPadMode::ExtractImagePatchesPadMode_VALID;
    } else {
        VPUX_THROW("Unsupported pad type {0}", this->autoPad());
    }

    const auto sizes = parseIntArrayAttr<int64_t>(sizesAttr());
    const auto strides = parseIntArrayAttr<int64_t>(stridesAttr());
    const auto rates = parseIntArrayAttr<int64_t>(ratesAttr());

    builder.add_sizeRows(checked_cast<int32_t>(sizes[0]));
    builder.add_sizeCols(checked_cast<int32_t>(sizes[1]));

    builder.add_strideRows(checked_cast<int32_t>(strides[0]));
    builder.add_strideCols(checked_cast<int32_t>(strides[1]));

    builder.add_rateRows(checked_cast<int32_t>(rates[0]));
    builder.add_rateCols(checked_cast<int32_t>(rates[1]));

    builder.add_autoPad(vpux_padding);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ExtractImagePatchesParams});
}
