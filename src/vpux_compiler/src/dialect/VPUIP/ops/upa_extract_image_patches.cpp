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

//namespace {
//MVCNN::PadType PadType2MVCNN(IE::PadType padding) {
//    MVCNN::PadType mvcnn_padding;
//    switch (padding) {
//    case IE::PadType::SAME_UPPER:
//        mvcnn_padding = MVCNN::PadType::PadType_SAME_UPPER;
//        break;
//    case IE::PadType::SAME_LOWER:
//        mvcnn_padding = MVCNN::PadType::PadType_SAME_LOWER;
//        break;
//    case IE::PadType::VALID:
//        mvcnn_padding = MVCNN::PadType::PadType_VALID;
//        break;
//    default:
//        VPUX_THROW("Unknown PadType. same upper (same lower) and valid types are supported only");
//    }
//    return mvcnn_padding;
//}
//}  // namespace

//void vpux::VPUIP::ExtractImagePatchesUPAOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
//                                       mlir::Value data, mlir::Value output,
//                                       mlir::ArrayAttr sizes, mlir::ArrayAttr strides, mlir::ArrayAttr rates,
//                                       IE::PadTypeAttr paddingType) {
//    build(odsBuilder, odsState, data, output, mlir::ValueRange{}, mlir::ValueRange{}, sizes, strides, rates, paddingType, nullptr, nullptr);
//}

//VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ExtractImagePatchesUPAOp::serialize(VPUIP::BlobWriter& writer) {
//
//      MVCNN::ExtractImagePatchesParamsBuilder builder(writer);
//
//      auto attrToVector = [&](mlir::ArrayAttr attr) {
//      auto values = parseIntArrayAttr(attr) | transformed([](auto value) {
//                    return checked_cast<uint32_t>(value);  //uint32_t only??? uint_64 not ???
//                      });
//      return to_std_vector(values);
//      };
//
//      builder.add_padding(PadType2MVCNN(paddingType()));
//      builder.add_sizes(builder.fbb_.CreateVector(attrToVector(sizes().getValue())));
//      builder.add_strides(builder.fbb_.CreateVector(attrToVector(strides().getValue())));
//      builder.add_rates(builder.fbb_.CreateVector(attrToVector(rates().getValue())));
//
//      const auto paramsOff = builder.Finish();

//      return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ExtractImagePatchesParams});
//}

//IE::PadType softLayerParam2IEType(size_t padding) {
//    IE::PadType ieType;
//    switch (padding) {
//    case 1:
//        ieType = IE::PadType::SAME_LOWER;
//        break;
//    case 2:
//        ieType = IE::PadType::SAME_UPPER;
//        break;
//    case 3:
//        ieType = IE::PadType::VALID;
//    default:
//        VPUX_THROW("Unknown PadType. same upper (same lower) and valid types are supported only");
//    }

//    return ieType;
//}

//mlir::Operation* vpux::VPUIP::BlobReader::parseExtractImagePatches(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
//                                                        ArrayRef<mlir::Value> outputs,
//                                                        const MVCNN::UPALayerTask* task) {
//    VPUX_THROW_UNLESS(inputs.size() == 1, "UPAExtractImagePatches supports only 1 input, got {0}", inputs.size());
//    VPUX_THROW_UNLESS(outputs.size() == 1, "UPAExtractImagePatches supports only 1 output, got {0}", outputs.size());
//
//    const auto params = task->softLayerParams_as_ExtractImagePatchesParams();
//    /* ??? ArrayAttr !
//    const auto sizes = getIntAttr(_ctx, params->sizes());
//    const auto strides = getIntAttr(_ctx, params->strides());
//    const auto rates = getIntAttr(_ctx, params->rates());
//    */
//    IE::PadType padding = softLayerParam2IEType(params->padding());
//
//    return builder.create<VPUIP::ExtractImagePatchesUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0],
//                                                outputs[0], sizes, strides, rates,
//                                                IE::PadTypeAttr::get(_ctx, padding));
//}
