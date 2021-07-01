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

#include "vpux/compiler/dialect/IE/ops_interfaces.hpp"

#include "vpux/compiler/conversion.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

//
// IELayer
//

mlir::LogicalResult vpux::IE::verifyIELayerOp(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in verifyRTLayerOp");

    auto isTensorized = std::all_of(op->getOperands().begin(), op->getOperands().end(), [](mlir::Value type) {
        return type.getType().isa<mlir::RankedTensorType>();
    });

    isTensorized &= std::all_of(op->getResults().begin(), op->getResults().end(), [](mlir::Value type) {
        return type.getType().isa<mlir::RankedTensorType>();
    });

    if (!isTensorized) {
        return errorAt(op, "Operation '{0}' is not a IE Layer, it operates with non Tensor types", op->getName());
    }

    return mlir::success();
}

mlir::LogicalResult vpux::IE::inferTensorTypes(InferTypeComponentsCb componentsCb, mlir::MLIRContext* ctx,
                                               Optional<mlir::Location> loc, mlir::ValueRange operands,
                                               mlir::DictionaryAttr attrs, mlir::RegionRange regions,
                                               SmallVectorImpl<mlir::Type>& inferredTypes) {
    SmallVector<mlir::ShapedTypeComponents> components;
    if (mlir::failed(componentsCb(ctx, loc, operands, attrs, regions, components))) {
        return mlir::failure();
    }

    for (const auto& shapeAndType : components) {
        mlir::Type resType;

        if (shapeAndType.hasRank()) {
            resType = mlir::RankedTensorType::get(shapeAndType.getDims(), shapeAndType.getElementType(),
                                                  shapeAndType.getAttribute());
        } else {
            resType = mlir::UnrankedTensorType::get(shapeAndType.getElementType());
        }

        inferredTypes.push_back(resType);
    }

    return mlir::success();
}

//
// Generated
//

#include <vpux/compiler/dialect/IE/generated/ops_interfaces.cpp.inc>
