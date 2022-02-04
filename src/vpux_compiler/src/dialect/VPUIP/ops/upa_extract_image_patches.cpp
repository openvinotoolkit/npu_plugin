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
                                      IE::PadTypeAttr paddingType) {

   build(odsBuilder, odsState, data, output, sizes, strides, rates, paddingType, nullptr);

}

// /home/dpapgher/WORK/Intel_work/applications.ai.vpu-accelerators.vpux-plugin/src/vpux_compiler/src/dialect/VPUIP/ops/upa_extract_image_patches.cpp:39:13: error: 
// ‘PadType’ is not a member of ‘MVCNN’; did you mean ‘vpux::IE::PadType’?
//    39 |      MVCNN::PadType mvcnn_padding;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ExtractImagePatchesUPAOp::serialize(VPUIP::BlobWriter& writer) {

     MVCNN::ExtractImagePatchesParamsBuilder builder(writer);

     vpux::IE::PadType vpux_padding;

     if (this->paddingType() == IE::PadType::SAME_UPPER) {
        vpux_padding = vpux::IE::PadType::SAME_UPPER;
    } else if (this->paddingType() == IE::PadType::SAME_LOWER) {
        vpux_padding = vpux::IE::PadType::SAME_LOWER;
    } else if (this->paddingType() == IE::PadType::VALID) {
        vpux_padding = vpux::IE::PadType::VALID;
    } else {
        VPUX_THROW("Unsupported pad type {0}", this->paddingType());
    }

    builder.add_autoPad(checked_cast<vpux::IE::PadType>(paddingType()));

     const auto paramsOff = builder.Finish();

     return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ExtractImagePatchesParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseExtractImagePatches(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                       ArrayRef<mlir::Value> outputs,
                                                       const MVCNN::UPALayerTask* task) {
   VPUX_THROW_UNLESS(inputs.size() == 1, "UPAExtractImagePatches supports only 1 input, got {0}", inputs.size());
   VPUX_THROW_UNLESS(outputs.size() == 1, "UPAExtractImagePatches supports only 1 output, got {0}", outputs.size());

   const auto params = task->softLayerParams_as_ExtractImagePatchesParams();

   const auto vpux_paddingType = params->paddingType();

   IE::PadType padding;

    if (vpux_paddingType == vpux::IE::PadType::SAME_UPPER) {
        padding = IE::PadType::SAME_UPPER;
    } else if (vpux_paddingType == vpux::IE::PadType::SAME_LOWER) {
        padding = IE::PadType::SAME_LOWER;
    } else if (vpux_paddingType == vpux::IE::PadType::VALID) {
        padding = IE::PadType::VALID;
    } else {
        VPUX_THROW("Unsupported pad type {0}", vpux_paddingType);
    }

   const auto sizes = params->sizes();
   const auto strides = params->strides();
   const auto rates = params->rates();

   return builder.create<VPUIP::ExtractImagePatchesUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0],
                                               outputs[0], sizes, strides, rates,
                                               IE::PadTypeAttr::get(_ctx, padding));
}


// VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ExtractImagePatchesUPAOp::serialize(VPUIP::BlobWriter& writer) {

//      MVCNN::ExtractImagePatchesParamsBuilder builder(writer);

//      MVCNN::PadType mvcnn_padding;

//      if (this->paddingType() == IE::PadType::SAME_UPPER) {
//         mvcnn_padding = MVCNN::PadType::PadType_SAME_UPPER;
//     } else if (this->paddingType() == IE::PadType::SAME_LOWER) {
//         mvcnn_padding = MVCNN::PadType::PadType_SAME_LOWER;
//     } else if (this->paddingType() == IE::PadType::VALID) {
//         mvcnn_padding = MVCNN::PadType::PadType_VALID;
//     } else {
//         VPUX_THROW("Unsupported pad type {0}", this->paddingType());
//     }

//     builder.add_paddingType(mvcnn_padding);


//      // auto attrToVector = [&](mlir::ArrayAttr attr) {
//      // auto values = parseIntArrayAttr(attr) | transformed([](auto value) {
//      //               return checked_cast<uint32_t>(value);  //uint32_t only??? uint_64 not ???
//      //                 });
//      // return to_std_vector(values);
//      // };

//      // builder.add_padding(padType());
//      // builder.add_sizes(sizes());
//      // builder.add_strides(strides());
//      // builder.add_rates(rates());
//     //  builder.add_sizes(builder.fbb_.CreateVector(attrToVector(sizes().getValue())));
//     //  builder.add_strides(builder.fbb_.CreateVector(attrToVector(strides().getValue())));
//     //  builder.add_rates(builder.fbb_.CreateVector(attrToVector(rates().getValue())));

//      const auto paramsOff = builder.Finish();

//      return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ExtractImagePatchesParams});
// }

// IE::PadType softLayerParam2IEType(size_t vpux_padding) {
//    IE::PadType ieType;
//    switch (vpux_padding) {
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
// }

// mlir::Operation* vpux::VPUIP::BlobReader::parseExtractImagePatches(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
//                                                        ArrayRef<mlir::Value> outputs,
//                                                        const MVCNN::UPALayerTask* task) {
//    VPUX_THROW_UNLESS(inputs.size() == 1, "UPAExtractImagePatches supports only 1 input, got {0}", inputs.size());
//    VPUX_THROW_UNLESS(outputs.size() == 1, "UPAExtractImagePatches supports only 1 output, got {0}", outputs.size());

//    const auto params = task->softLayerParams_as_ExtractImagePatchesParams();

//    const auto mvcnn_padding = params->paddyngType();
    

//    IE::PadType padding;

//     if (mvcnn_padding == MVCNN::PadType::PadType_SAME_UPPER) {
//         padding = IE::PadType::SAME_UPPER;
//     } else if (mvcnn_padding == MVCNN::PadType::PadType_SAME_LOWER) {
//         padding = IE::PadType::SAME_LOWER;
//     } else if (mvcnn_padding == MVCNN::PadType::PadType_VALID) {
//         padding = IE::PadType::VALID;
//     } else {
//         VPUX_THROW("Unsupported pad type {0}", mvcnn_padding);
//     }

//    const auto sizes = getIntArrayAttr(_ctx, params->sizes());
//    const auto strides = getIntArrayAttr(_ctx, params->strides());
//    const auto rates = getIntArrayAttr(_ctx, params->rates());

//    return builder.create<VPUIP::ExtractImagePatchesUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0],
//                                                outputs[0], sizes, strides, rates,
//                                                IE::PadTypeAttr::get(_ctx, vpux_padding));
// }
