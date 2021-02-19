//
// Copyright 2021 Intel Corporation.
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

mlir::LogicalResult vpux::VPUIP::verifyOp(PerAxisTileUPAOp op) {
    const auto inShape = getShape(op.input());
    if (op.axis() > inShape.size()) {
        return errorAt(op, "Tile axis '{0}' is out of range [1,{1}]", op.axis(), inShape.size());
    }
    return mlir::success();
}

void vpux::VPUIP::PerAxisTileUPAOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                          mlir::Value input, mlir::Value output, mlir::IntegerAttr axis,
                                          mlir::IntegerAttr tiles) {
    build(odsBuilder, odsState, input, output, mlir::ValueRange{}, mlir::ValueRange{}, axis, tiles, nullptr, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::PerAxisTileUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::TileParamsBuilder builder(writer);
    builder.add_axis(axis());
    builder.add_tiles(tiles());

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_TileParams});
}
