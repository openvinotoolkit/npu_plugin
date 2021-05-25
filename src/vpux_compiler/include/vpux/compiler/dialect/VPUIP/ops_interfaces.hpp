//
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

#pragma once

#include "vpux/compiler/dialect/IERT/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes/enums.hpp"
#include "vpux/compiler/dialect/VPUIP/blob_writer.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace vpux {
namespace VPUIP {

//
// Forward declarations
//

class BlobWriter;

//
// verifyUPATask
//

mlir::LogicalResult verifyUPATask(mlir::Operation* op);

//
// verifyNCETask
//

mlir::LogicalResult verifyNCETask(mlir::Operation* op);

//
// getTaskEffects
//

using MemoryEffect = mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>;

void getTaskEffects(mlir::Operation* op, SmallVectorImpl<MemoryEffect>& effects);

//
// SameShape
//

mlir::LogicalResult verifySameShape(mlir::Operation* op);

template <typename ConcreteOp>
class SameShape : public mlir::OpTrait::TraitBase<ConcreteOp, SameShape> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySameShape(op);
    }
};

//
// SameElementType
//

mlir::LogicalResult verifySameElementType(mlir::Operation* op);

template <typename ConcreteOp>
class SameElementType : public mlir::OpTrait::TraitBase<ConcreteOp, SameElementType> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySameElementType(op);
    }
};

//
// SameDimsOrder
//

mlir::LogicalResult verifySameDimsOrder(mlir::Operation* op);
mlir::LogicalResult isSupportedLayoutSameDimsOrder(mlir::Operation* op, DataOrderInfo& info);

template <typename ConcreteOp>
class SameDimsOrder : public mlir::OpTrait::TraitBase<ConcreteOp, SameDimsOrder> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySameDimsOrder(op);
    }

    static mlir::LogicalResult isSupportedLayout(mlir::Operation* op, vpux::DataOrderInfo& info) {
        return isSupportedLayoutSameDimsOrder(op, info);
    }
};

//
// SameInOutDimsOrder
//

mlir::LogicalResult verifySameInOutDimsOrder(mlir::Operation* op);
mlir::LogicalResult isSupportedLayoutSameInOutDimsOrder(mlir::Operation* op, DataOrderInfo& info);

template <typename ConcreteOp>
class SameInOutDimsOrder : public mlir::OpTrait::TraitBase<ConcreteOp, SameInOutDimsOrder> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySameInOutDimsOrder(op);
    }

    static mlir::LogicalResult isSupportedLayout(mlir::Operation* op, vpux::DataOrderInfo& info) {
        return isSupportedLayoutSameInOutDimsOrder(op, info);
    }
};

//
// SameInOutSpecificDimsOrder
//

mlir::LogicalResult verifySameInOutSpecificDimsOrder(mlir::Operation* op, ArrayRef<DimsOrder> supportedLayouts);
mlir::LogicalResult isSupportedLayoutSameInOutSpecificDimsOrder(mlir::Operation* op, DataOrderInfo& info,
                                                                ArrayRef<DimsOrder> supportedLayouts);

//
// SameInOutDimsOrder_NCHW_NHWC
//

extern const std::array<DimsOrder, 2> NCHW_NHWC;

template <typename ConcreteOp>
class SameInOutDimsOrder_NCHW_NHWC : public mlir::OpTrait::TraitBase<ConcreteOp, SameInOutDimsOrder_NCHW_NHWC> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySameInOutSpecificDimsOrder(op, NCHW_NHWC);
    }

    static mlir::LogicalResult isSupportedLayout(mlir::Operation* op, vpux::DataOrderInfo& info) {
        return isSupportedLayoutSameInOutSpecificDimsOrder(op, info, NCHW_NHWC);
    }
};

//
// SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
//

extern const std::array<DimsOrder, 4> CHW_HWC_NCHW_NHWC;

template <typename ConcreteOp>
class SameInOutDimsOrder_CHW_HWC_NCHW_NHWC :
        public mlir::OpTrait::TraitBase<ConcreteOp, SameInOutDimsOrder_CHW_HWC_NCHW_NHWC> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySameInOutSpecificDimsOrder(op, CHW_HWC_NCHW_NHWC);
    }

    static mlir::LogicalResult isSupportedLayout(mlir::Operation* op, vpux::DataOrderInfo& info) {
        return isSupportedLayoutSameInOutSpecificDimsOrder(op, info, CHW_HWC_NCHW_NHWC);
    }
};

//
// AnyDimsOrder
//

template <typename ConcreteOp>
class AnyDimsOrder : public mlir::OpTrait::TraitBase<ConcreteOp, AnyDimsOrder> {
public:
    static mlir::LogicalResult isSupportedLayout(mlir::Operation*, vpux::DataOrderInfo&) {
        return mlir::success();
    }
};

//
// Legacy4D
//

mlir::LogicalResult verifyLegacy4D(mlir::Operation* op);

template <typename ConcreteOp>
class Legacy4D : public mlir::OpTrait::TraitBase<ConcreteOp, SameDimsOrder> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifyLegacy4D(op);
    }
};

}  // namespace VPUIP
}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/dialect/VPUIP/generated/ops_interfaces.hpp.inc>
