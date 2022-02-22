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

#include "vpux/compiler/dialect/IERT/attributes/structs.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/Identifier.h>

using namespace vpux;

//
// Generated
//

#include <vpux/compiler/dialect/IERT/generated/attributes/structs.cpp.inc>

//
// MemRefAttrLayout
//

mlir::AffineMap IERT::MemRefAttrLayout::getAffineMap(mlir::Attribute attr) const {
    const auto desc = attr.dyn_cast<IERT::MemRefAttr>();
    VPUX_THROW_WHEN(desc == nullptr, "Unsupported MemRef layout '{0}'", attr);

    const auto orderMap = desc.order().getValue();

    const auto elemStrides = parseIntArrayAttr<int64_t>(desc.strides());
    const auto stridesMap = mlir::makeStridedLinearLayoutMap(elemStrides, 0, attr.getContext());

    return stridesMap.compose(orderMap);
}

bool IERT::MemRefAttrLayout::isIdentity(mlir::Attribute) const {
    return false;
}

mlir::LogicalResult IERT::MemRefAttrLayout::verifyLayout(mlir::Attribute attr, ArrayRef<int64_t> shape,
                                                         FuncRef<mlir::InFlightDiagnostic()> emitError) const {
    const auto desc = attr.dyn_cast<IERT::MemRefAttr>();
    if (desc == nullptr) {
        return printTo(emitError(), "Unsupported MemRef layout '{0}'", attr);
    }

    if (!desc.order().getValue().isPermutation()) {
        return printTo(emitError(), "Dims order '{0}' is not a permutation affine map", desc.order());
    }

    const auto order = DimsOrder::fromAffineMap(desc.order().getValue());
    const auto elemStrides = parseIntArrayAttr<int64_t>(desc.strides());

    const auto memShape = order.toMemoryOrder(ShapeRef(shape));

    const auto elemSize = 1_Bit;
    const auto strides = Strides(to_small_vector(elemStrides | transformed([&](int64_t stride) {
                                                     return stride * elemSize;
                                                 })));
    const auto memStrides = order.toMemoryOrder(strides);

    const auto reqs = StrideReqs::simple(shape.size());

    if (!reqs.checkStrides(memStrides, elemSize, memShape)) {
        return printTo(emitError(), "Strides '{0}' do not match with shape '{1}' and order '{2}'", desc.strides(),
                       shape, order);
    }

    return mlir::success();
}
