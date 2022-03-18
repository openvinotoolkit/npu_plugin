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

#include "vpux/compiler/utils/analysis.hpp"

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/utils/core/error.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <algorithm>

using namespace vpux;

//
// getFirstUser
//

mlir::Operation* vpux::getFirstUser(mlir::Value output) {
    VPUX_THROW_UNLESS(output != nullptr, "Got NULL pointer in getFirstUser");

    const auto users = output.getUsers();
    const auto firstUser = std::min_element(users.begin(), users.end(), [](mlir::Operation* lhs, mlir::Operation* rhs) {
        return lhs->getBlock() == rhs->getBlock() && lhs->isBeforeInBlock(rhs);
    });

    return firstUser == users.end() ? nullptr : *firstUser;
}

//
// isBufAllocOp
//

bool vpux::isBufAllocOp(mlir::Operation* op) {
    if (!op) {
        return false;
    }

    if (op->getNumOperands() != 0 || op->getNumResults() != 1) {
        return false;
    }

    if (!op->getResult(0).getType().isa<mlir::MemRefType, VPUIP::DistributedBufferType>()) {
        return false;
    }

    if (auto iface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
        return iface.onlyHasEffect<mlir::MemoryEffects::Allocate>();
    }

    return false;
}

//
// getTopLevelModule
//

mlir::ModuleOp vpux::getTopLevelModule(mlir::Operation* op) {
    auto module =
            op->getParentOp() == nullptr ? mlir::dyn_cast<mlir::ModuleOp>(op) : op->getParentOfType<mlir::ModuleOp>();
    VPUX_THROW_UNLESS(module != nullptr, "Can't get parent Module from Operation '{0}' at '{1}'", op->getName(),
                      op->getLoc());
    return module;
}
