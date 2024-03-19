//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/ELFNPU37XX/passes.hpp"
#include "vpux/compiler/utils/adjust_layout_utils.hpp"
#include "vpux/compiler/utils/logging.hpp"

using namespace vpux;

namespace {

// An ELF Section OP is basically a container for other OPS. An empty one represents no real usage.
// Furthermore, by current ELF design, op-operand relationships are direct usage-based.
// AKA: if a section can be removed, all it's users are safe to remove (aka a section's symbol or relocation sections
// that target the particular section)

void recursivelyErase(mlir::Operation* operation) {
    auto users = operation->getUsers();

    for (auto user : llvm::make_early_inc_range(users)) {
        recursivelyErase(user);
    }

    operation->erase();
    return;
}

class RemoveEmptyELFSectionsPass : public ELFNPU37XX::RemoveEmptyELFSectionsBase<RemoveEmptyELFSectionsPass> {
public:
    explicit RemoveEmptyELFSectionsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void RemoveEmptyELFSectionsPass::safeRunOnFunc() {
    auto funcOp = getOperation();

    auto sections = funcOp.getOps<ELFNPU37XX::ElfSectionInterface>();

    for (auto section : llvm::make_early_inc_range(sections)) {
        if (section.getBlock()->empty()) {
            auto operation = section.getOperation();

            recursivelyErase(operation);
        };
    }
}

}  // namespace

//
// createRemoveEmptyELFSectionsPass
//

std::unique_ptr<mlir::Pass> vpux::ELFNPU37XX::createRemoveEmptyELFSectionsPass(Logger log) {
    return std::make_unique<RemoveEmptyELFSectionsPass>(log);
}
