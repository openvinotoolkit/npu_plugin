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

//#include "vpux/compiler/core/attributes/dim.hpp"
//#include "vpux/compiler/core/attributes/shape.hpp"

//#include "vpux/compiler/utils/analysis.hpp"
//#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//TODO

//namespace {
//MVCNN::ExtractImagePatchesAutoPadType ExtractImagePatchesAutoPadType2MVCNN(IE::ExtractImagePatchesAutoPadType padding) {
//    MVCNN::ExtractImagePatchesAutoPadType mvcnn_padding;
//    switch (padding) {
//    case IE::ExtractImagePatchesAutoPadType::same_upper:
//        mvcnn_padding = MVCNN::ExtractImagePatchesAutoPadType_padding_same_upper;
//        break;
//    case IE::ExtractImagePatchesAutoPadType::same_lower:
//        mvcnn_padding = MVCNN::ExtractImagePatchesAutoPadType_padding_same_lower;
//        break;
//    case IE::ExtractImagePatchesAutoPadType::valid:
//        mvcnn_padding = MVCNN::ExtractImagePatchesAutoPadType_padding_valid;
//        break;
//    default:
//        VPUX_THROW("Unknown ExtractImagePatchesAutoPadType. same upper (same lower) and valid types are supported only");
//    }
//    return mvcnn_padding;
//}
//}  // namespace

//TODO

//void vpux::VPUIP::ExtractImagePatchesUPAOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
//                                       mlir::Value data, mlir::Value output,
//                                       mlir::ArrayAttr sizes, mlir::ArrayAttr strides, mlir::ArrayAttr rates
//                                       IE::ExtractImagePatchesAutoPadTypeAttr paddingType) {
//    build(odsBuilder, odsState, data, output, mlir::ValueRange{}, mlir::ValueRange{}, sizes, strides, rates, paddingType, nullptr, nullptr);
//}

//VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ExtractImagePatchesUPAOp::serialize(VPUIP::BlobWriter& writer) {
//      MVCNN::ExtractImagePatchesParamsBuilder builder(writer);
//      builder.add_padding(ExtractImagePatchesAutoPadType2MVCNN(paddingType()));
//      builder.add_sizes(checked_cast<uint64_t>(sizes()));
//      builder.add_strides(checked_cast<uint64_t>(strides()));
//      builder.add_rates(checked_cast<uint64_t>(rates()));
//
//      const auto paramsOff = builder.Finish();

//    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ExtractImagePatchesParams});
//}

//IE::ExtractImagePatchesAutoPadType softLayerParam2IEType(size_t padding) {
//    IE::ExtractImagePatchesAutoPadType ieType;
//    switch (padding) {
//    case 0:
//        ieType = IE::ExtractImagePatchesAutoPadType_padding_same_upper;
//        break;
//    case 1:
//        ieType = IE::ExtractImagePatchesAutoPadType_padding_same_lower;
//        break;
//    case 2:
//        ieType = IE::ExtractImagePatchesAutoPadType_padding_valid;
//    default:
//        VPUX_THROW("Unknown ExtractImagePatchesAutoPadType. same upper (same lower) and valid types are supported only");
//    }

//    return ieType;
//}

//mlir::Operation* vpux::VPUIP::BlobReader::parseExtractImagePatches(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
//                                                        ArrayRef<mlir::Value> outputs,
//                                                        const MVCNN::UPALayerTask* task) {
//    VPUX_THROW_UNLESS(inputs.size() == 1, "UPAExtractImagePatches supports only 1 input, got {0}", inputs.size());
//    VPUX_THROW_UNLESS(outputs.size() == 1, "UPAExtractImagePatches supports only 1 output, got {0}", outputs.size());
//    const auto params = task->softLayerParams_as_ExtractImagePatchesParams();
//    //
//    IE::ExtractImagePatchesAutoPadType padding = softLayerParam2IEType(params->padding());

//    return builder.create<VPUIP::ExtractImagePatchesUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0],
//                                                outputs[0], sizes, strides, rates,
//                                                IE::ExtractImagePatchesAutoPadTypeAttr::get(_ctx, padding));
//}
