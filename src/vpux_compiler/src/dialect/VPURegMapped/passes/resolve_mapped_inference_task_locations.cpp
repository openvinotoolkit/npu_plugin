//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <limits>

#include "vpux/compiler/dialect/VPURegMapped/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPURegMapped/passes.hpp"

using namespace vpux;

namespace {

class ResolveTaskLocationsPass final : public VPURegMapped::ResolveTaskLocationsBase<ResolveTaskLocationsPass> {
public:
    ResolveTaskLocationsPass(VPURegMapped::UpperBoundsCallable&& upperBoundsCallable, Logger log)
            : upperBounds(std::move(upperBoundsCallable)) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    using TileIndex = uint32_t;
    using ListIndex = uint32_t;

    void safeRunOnFunc() final override {
        auto function = getOperation();
        auto builder = mlir::OpBuilder::atBlockBegin(&function.getBody().front());

        for (auto task : function.getOps<VPURegMapped::TaskOpInterface>()) {
            const auto type = task.getTaskType();
            const auto index = task.getIndex();
            const auto upperBound = upperBounds(type, index);
            auto& taskBuffers = taskLocations[type][index.getTileIdx()][index.getListIdx()];

            VPUX_THROW_WHEN((index.getValue() < upperBound && taskBuffers.size() != index.getValue()) ||
                                    (index.getValue() >= upperBound && taskBuffers.size() != upperBound),
                            "Expected to encounter tasks only once in their internal list order");

            if (index.getValue() < upperBound) {
                taskBuffers.push_back(builder.create<VPURegMapped::DeclareTaskBufferOp>(
                        function.getLoc(), index, VPURegMapped::TaskTypeAttr::get(function.getContext(), type)));
            }

            task.setTaskLocation(taskBuffers[index.getValue() % upperBound]);
        }
    };

    llvm::DenseMap<VPURegMapped::TaskType,
                   llvm::DenseMap<TileIndex, llvm::DenseMap<ListIndex, llvm::SmallVector<mlir::Value>>>>
            taskLocations;

    VPURegMapped::UpperBoundsCallable upperBounds;
};

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPURegMapped::resolveMappedInferenceTaskLocationsPass(
        VPURegMapped::UpperBoundsCallable upperBounds, Logger log) {
    return std::make_unique<ResolveTaskLocationsPass>(std::move(upperBounds), log);
}

std::unique_ptr<mlir::Pass> vpux::VPURegMapped::resolveMappedInferenceTaskLocationsPass(Logger log) {
    return std::make_unique<ResolveTaskLocationsPass>(
            [](auto, auto) {
                return std::numeric_limits<size_t>::max();
            },
            log);
}
