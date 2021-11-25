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

mlir::LogicalResult vpux::EMU::verifyOp(PerAxisTileUPAOp op) {
    const auto inShape = getShape(op.input());
    if (checked_cast<size_t>(op.axis()) > inShape.size()) {
        return errorAt(op, "Tile axis '{0}' is out of range [1,{1}]", op.axis(), inShape.size());
    }
    return mlir::success();
}

EMU::BlobWriter::SpecificTask vpux::EMU::PerAxisTileUPAOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::TileParamsBuilder builder(writer);
    builder.add_axis(checked_cast<uint32_t>(axis()));
    builder.add_tiles(checked_cast<uint32_t>(tiles()));

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_TileParams});
}
