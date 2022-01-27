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

#pragma once

#include "vpux/compiler/dialect/IERT/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/attributes/enums.hpp"

//#include "vpux/compiler/dialect/VPUIPRegMapped/blob_writer.hpp"
// Alex: added manually these files from the beginning of blob_writer.hpp
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/attributes/enums.hpp"
// Alex: #include "vpux/compiler/dialect/VPUIPRegMapped/schema.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
// END Alex

#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace vpux {
namespace VPUIPRegMapped {

// Alex: class BlobWriter;

//
// TaskOpInterface
//

using MemoryEffect = mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>;
void getTaskEffects(mlir::Operation* op, SmallVectorImpl<MemoryEffect>& effects);

mlir::SymbolRefAttr getDMAEngine(uint32_t& numUnits, mlir::MLIRContext* ctx,
                                 VPUIPRegMapped::DMAEngine engine);  // 2022_01_11
/*
// 2022_01_11
mlir::SymbolRefAttr getPhysicalProcessor(uint32_t& numUnits, mlir::Operation* op, VPUIPRegMapped::PhysicalProcessor
proc, Optional<int64_t> opUnits = None);
*/

// 2022_01_11
mlir::SymbolRefAttr getTaskOpExecutor(mlir::Operation* op, uint32_t& numUnits);

//
// Legacy4D
//

mlir::LogicalResult verifyLegacy4D(mlir::Operation* op);

template <typename ConcreteOp>
class Legacy4D : public mlir::OpTrait::TraitBase<ConcreteOp, Legacy4D> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifyLegacy4D(op);
    }
};

}  // namespace VPUIPRegMapped
}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/dialect/VPUIPRegMapped/generated/ops_interfaces.hpp.inc>
