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

//#include "vpux/compiler/dialect/IERT/ops_interfaces.hpp"
//#include "vpux/compiler/dialect/VPUIP/attributes/enums.hpp"
//#include "vpux/compiler/dialect/VPUIP/blob_writer.hpp"
//#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

/*
namespace vpux {
namespace VPUIP {

class BlobWriter;

//
// TaskOpInterface
//

using MemoryEffect = mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>;
void getTaskEffects(mlir::Operation* op, SmallVectorImpl<MemoryEffect>& effects);

mlir::Attribute getDMAEngine(uint32_t& numUnits, mlir::MLIRContext* ctx, VPUIP::DMAEngine engine);
mlir::Attribute getPhysicalProcessor(uint32_t& numUnits, mlir::Operation* op, VPUIP::PhysicalProcessor proc,
                                     Optional<int64_t> opUnits = None);

mlir::Attribute getTaskOpExecutor(mlir::Operation* op, uint32_t& numUnits);

//
// UPATaskOpInterface
//

mlir::LogicalResult verifyUPATask(mlir::Operation* op);

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

}  // namespace VPUIP
}  // namespace vpux
*/

//
// Generated
//

#include <vpux/compiler/dialect/const/generated/ops_interfaces.hpp.inc>
