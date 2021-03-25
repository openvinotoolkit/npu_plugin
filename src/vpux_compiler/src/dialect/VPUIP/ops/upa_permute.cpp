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

mlir::LogicalResult vpux::VPUIP::verifyOp(PermuteUPAOp op) {
    const auto order = DimsOrder::fromAffineMap(op.order_value());
    const auto inShape = getShape(op.input());

    if (order.numDims() > inShape.size()) {
        return errorAt(op, "Order vector size {0} doesn't match input rank {1}", order.numDims(), inShape.size());
    }

    const auto outRank = static_cast<int64_t>(inShape.size());
    for (auto i = 0; i < outRank; i++) {
        if (order.dimAt(i).ind() >= outRank) {
            return errorAt(op, "Order index {0} is out of range [0, {1}]", order.dimAt(i).ind(), outRank - 1);
        }
    }

    // TODO Support custom input/output layout [Track number: W#6148]
    const auto& inAffineMaps = op.input().getType().cast<mlir::MemRefType>().getAffineMaps();
    const auto& outAffineMaps = op.output().getType().cast<mlir::MemRefType>().getAffineMaps();

    const auto hasSpecificLayout = [](ArrayRef<mlir::AffineMap> maps) {
        return !maps.empty() && maps[0].isPermutation();
    };

    if (hasSpecificLayout(inAffineMaps) || hasSpecificLayout(outAffineMaps)) {
        return errorAt(op, "Supported only default input/output layout");
    }

    return mlir::success();
}

void vpux::VPUIP::PermuteUPAOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                      mlir::Value input, mlir::Value output, mlir::AffineMapAttr order) {
    build(odsBuilder, odsState, input, output, mlir::ValueRange{}, mlir::ValueRange{}, order, nullptr, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::PermuteUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto order = DimsOrder::fromAffineMap(this->order_value());
    const auto orderUPA = writer.createVector(irange(order.numDims()) | reversed | transformed([&](int64_t idx) {
                                                  return checked_cast<int32_t>(order.dimAt(idx).ind());
                                              }));

    MVCNN::PermuteNDParamsBuilder builder(writer);
    builder.add_permute_nd_order(orderUPA);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PermuteNDParams});
}
