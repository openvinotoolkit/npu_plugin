
// Copyright 2020 Intel Corporation.
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

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

//
// inferReturnTypeComponents
//

mlir::LogicalResult vpux::IE::NormalizeIEOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::NormalizeIEOpAdaptor normalize(operands, attrs);
    if (mlir::failed(normalize.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = normalize.data().getType().cast<mlir::ShapedType>();
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::IE::NormalizeIEOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::NormalizeParamsBuilder builder(writer);
    builder.add_eps(static_cast<float>(eps().convertToDouble()));
    builder.add_across_spatial(static_cast<int32_t>(across_spatial()));
    builder.add_channel_shared(static_cast<int32_t>(channel_shared()));

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_NormalizeParams});
}
