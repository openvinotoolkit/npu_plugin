//
// Copyright 2020 Intel Corporation.
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
