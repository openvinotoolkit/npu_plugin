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

#include "vpux/compiler/core/control_dependencies_save_restore.hpp"



using namespace vpux;

//
// Constructor
//

ControlDependenciesSaveRestore::ControlDependenciesSaveRestore(mlir::MLIRContext* ctx, mlir::FuncOp func)
        : _ctx(ctx), _func(func) {
}

void ControlDependenciesSaveRestore::saveInitialControlFlow() {
    // Get all producers and consumers of barriers (NCE,UPA, DMA) only
    _allBarrierOps = to_small_vector(_func.getOps<VPURT::DeclareVirtualBarrierOp>());

    for (auto& barrierOp : _allBarrierOps) {
        SmallVector<mlir::Operation*> producers;
        SmallVector<mlir::Operation*> consumers;

        for (auto* userOp : barrierOp->getUsers()) {
            auto opEffects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(userOp);

            VPUX_THROW_WHEN(opEffects == nullptr,
                            "Barrier '{0}' is used by Operation '{1}' without MemoryEffects interface",
                            barrierOp->getLoc(), userOp->getName());

            using MemEffect = mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>;

            SmallVector<MemEffect> valEffects;

            opEffects.getEffectsOnValue(barrierOp.barrier(), valEffects);

            VPUX_THROW_WHEN(
                    valEffects.size() != 1,
                    "Barrier '{0}' must have exactly 1 MemoryEffect per Operation, got '{1}' for Operation '{2}'",
                    barrierOp->getLoc(), valEffects.size(), userOp->getLoc());

            const auto& effect = valEffects.front();

            if (effect.getEffect() == mlir::MemoryEffects::Write::get()) {
                auto task = mlir::dyn_cast<VPURT::TaskOp>(userOp);
                if (task.getTaskType() == VPUIP::TaskType::NCE2) {
                    producers.push_back(userOp);
                } else if (task.getTaskType() == VPUIP::TaskType::NNDMA) {
                    producers.push_back(userOp);
                } else if (task.getTaskType() == VPUIP::TaskType::UPA) {
                    producers.push_back(userOp);
                }
            } else if (effect.getEffect() == mlir::MemoryEffects::Read::get()) {
                auto task = mlir::dyn_cast<VPURT::TaskOp>(userOp);
                if (task.getTaskType() == VPUIP::TaskType::NCE2) {
                    consumers.push_back(userOp);
                } else if (task.getTaskType() == VPUIP::TaskType::NNDMA) {
                    consumers.push_back(userOp);
                } else if (task.getTaskType() == VPUIP::TaskType::UPA) {
                    consumers.push_back(userOp);
                }
            } else {
                VPUX_THROW("Barrier '{0}' has unsupported Effect in Operation '{1}'", barrierOp->getLoc(),
                           userOp->getLoc());
            }
        }
        barrierProducersMap.insert(std::make_pair(barrierOp, producers));
        barrierConsumersMap.insert(std::make_pair(barrierOp, consumers));
    }
}

void ControlDependenciesSaveRestore::restore() {

    //Clear all barriers and control flows
       _func->walk([](VPURT::TaskOp op) {
        op.updateBarriersMutable().clear();
        op.waitBarriersMutable().clear();
    });

    _func->walk([&](VPURT::DeclareVirtualBarrierOp op) {
        op->dropAllUses();
        op.erase();
    });

    //Restore the model
    for (const auto& p : barrierProducersMap) {
        auto barrierOp = mlir::dyn_cast_or_null<VPURT::DeclareVirtualBarrierOp>(p.first);
        for (auto* user : p.second) {
            auto taskOp = mlir::dyn_cast_or_null<VPURT::TaskOp>(user);
            assert(taskOp != NULL);
            assert(barrierOp.barrier() != NULL);
            Logger::global().error("Adding Barrier ID {0} as an update barrier for operation {1}", barrierOp->getAttr("id"),FeasibleScheduleGenerator::getUniqueID(user));
            taskOp.updateBarriersMutable().append(barrierOp.barrier());
        }
    }

    for (const auto& p : barrierConsumersMap) {
        auto barrierOp = mlir::dyn_cast_or_null<VPURT::DeclareVirtualBarrierOp>(p.first);
        for (auto* user : p.second) {
            auto taskOp = mlir::dyn_cast_or_null<VPURT::TaskOp>(user);
            assert(taskOp != NULL);
            assert(barrierOp.barrier() != NULL);
            Logger::global().error("Adding Barrier ID {0} as an wait barrier for operation {1}", barrierOp->getAttr("id"),FeasibleScheduleGenerator::getUniqueID(user));
            taskOp.waitBarriersMutable().append(barrierOp.barrier());
        }
    }
}