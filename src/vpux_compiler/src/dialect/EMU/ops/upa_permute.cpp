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

//
// verifyOp
//

mlir::LogicalResult vpux::EMU::verifyOp(PermuteUPAOp op) {
    const auto inType = op.input().getType().dyn_cast<mlir::ShapedType>();
    const auto outType = op.output().getType().dyn_cast<mlir::ShapedType>();

    if (inType.getRank() > outType.getRank()) {
        return errorAt(op, "Input rank {0} doesn't match output rank {1}", inType.getRank(), outType.getRank());
    }

    const auto order = DimsOrder::fromPermutationAffineMap(op.order_value().getValue());
    const auto inShape = getShape(inType);

    if (order.numDims() > inShape.size()) {
        return errorAt(op, "Order vector size {0} doesn't match input rank {1}", order.numDims(), inShape.size());
    }

    const auto outRank = static_cast<int64_t>(inShape.size());
    for (auto i = 0; i < outRank; i++) {
        if (order.dimAt(i).ind() >= outRank) {
            return errorAt(op, "Order index {0} is out of range [0, {1}]", order.dimAt(i).ind(), outRank - 1);
        }
    }

    return mlir::success();
}

//
// TaskOpInterface
//

EMU::BlobWriter::SpecificTask vpux::EMU::PermuteUPAOp::serialize(EMU::BlobWriter& writer) {
    const mlir::AffineMap inputOrderMap = DimsOrder::fromValue(input()).toPermutationAffineMap(this->getContext());
    const mlir::AffineMap permMem = order_value().getValue();
    const mlir::AffineMap outputOrderMapInv =
            inversePermutation(DimsOrder::fromValue(output()).toPermutationAffineMap(this->getContext()));

    const mlir::AffineMap permLog = outputOrderMapInv.compose(permMem.compose(inputOrderMap));

    const auto permLogOrder = DimsOrder::fromPermutationAffineMap(permLog);
    const auto orderUPA = writer.createVector(irange(permLogOrder.numDims()) | transformed([&](int64_t idx) {
                                                  return checked_cast<int32_t>(permLogOrder.dimAt(idx).ind());
                                              }));

    MVCNN::PermuteNDParamsBuilder builder(writer);
    builder.add_permute_nd_order(orderUPA);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PermuteNDParams});
}
