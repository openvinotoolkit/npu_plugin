//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/strings.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include "vpux/utils/core/profiling.hpp"
#include "vpux/utils/plugin/profiling_parser.hpp"

namespace vpux {

VPUIP::DpuProfilingMetadataAttr getDpuProfilingMetaAttr(mlir::MLIRContext* ctx, unsigned bufferId, unsigned taskId,
                                                        unsigned maxVariants, llvm::Optional<unsigned> numVariants,
                                                        llvm::Optional<unsigned> clusterId);

vpux::profiling::ExecutorType convertDataInfoNameToExecType(StringRef name);

// Post processing of profiling is relay on uniqueness of locations, but this may be violated. To ensure that all names
// are unique this class is used
class NameUniqifier {
public:
    NameUniqifier(vpux::Logger& log): _log(log) {
    }

    mlir::Location getUniqueLoc(mlir::Location baseLoc) {
        VPUX_THROW_WHEN(baseLoc.isa<mlir::UnknownLoc>(), "Unknown location");
        std::string strLoc = stringifyLocation(baseLoc);
        if (_counter.count(strLoc) == 0) {
            _counter[strLoc] = 1;
            return baseLoc;
        }
        _counter[strLoc]++;
        _log.warning("Duplicate location attribute: '{0}'", baseLoc);
        return appendLoc(baseLoc, formatv("Duplicated_{0}", _counter[strLoc]));
    }

private:
    vpux::Logger& _log;
    std::map<std::string, size_t> _counter;
};

// Utility structure to save information of individual task that is profiled
template <class InnerTask>
struct TaskSignature {
    InnerTask _task;
    unsigned _maxSubTasks;
    SmallVector<unsigned> _subTasksAtCluster;

    VPUIP::DpuProfilingMetadataAttr dpuSignature(mlir::MLIRContext* ctx, unsigned bufferId, unsigned taskId) const {
        llvm::Optional<unsigned> numVariants;
        llvm::Optional<unsigned> clusterId;
        // Setting clusterId for single-cluster case. Unroll-Nce-Tiling pass won't handle this case, so doing this now
        if (_subTasksAtCluster.size() == 1) {
            numVariants = _subTasksAtCluster.front();
            clusterId = 0;
        }
        return getDpuProfilingMetaAttr(ctx, bufferId, taskId, _maxSubTasks, numVariants, clusterId);
    }
};

mlir::BlockArgument addNewProfilingOutput(mlir::MLIRContext* ctx, mlir::func::FuncOp& netFunc, IE::CNNNetworkOp& netOp,
                                          mlir::MemRefType outputType, profiling::ExecutorType execType);

/**
 * @brief utility function to check if a given task belongs to a set of profiled DMA operations,
 * i.e. has DMATypeOpInterface defined
 */
bool isProfiledDmaTask(VPURT::TaskOp taskOp);

/**
 * @brief convenience function to define dma_hwp_id HWP attribute for a given operation
 *
 * @param ctx - mlir context
 * @param op - profiled operation
 * @param dmaHwpId - HWP id value. Set to 0 to ignore the profiling entry.
 */
void setDmaHwpIdAttribute(mlir::MLIRContext* ctx, VPUIP::DMATypeOpInterface& op, int32_t dmaHwpId);
/**
 * @brief check whether HWP argument was used in any profiled DMA operation
 * dma_hwp_id attribute set
 *
 * @return true if at least one profiled DMA operation has dma_hwp_id argument set, false otherwise.
 */
bool isDmaHwpUsedInVPURT(mlir::func::FuncOp& func);
/**
 * @brief check whether HWP argument was used in any profiled DMA operation
 * dma_hwp_id attribute set
 *
 * @return false for architectures <= VPUX37XX.
 * For other architectures true if at least one profiled DMA operation has dma_hwp_id argument set,
 * false otherwise.
 *
 */
bool isDmaHwpUsedInVPURT(mlir::ModuleOp& module);
}  // namespace vpux
