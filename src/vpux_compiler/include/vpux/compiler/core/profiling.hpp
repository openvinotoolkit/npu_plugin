//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/strings.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include "vpux/utils/core/profiling.hpp"

namespace vpux {

// Post processing of profiling is relay on uniqueness of locations, but this may be violated. To ensure that all names
// are unique this class used
class NameUniqifier {
public:
    NameUniqifier(vpux::Logger& log): _log(log) {
    }

    mlir::Location getUniqueLoc(mlir::Location baseLoc) {
        std::string strLoc;
        if (baseLoc.isa<mlir::UnknownLoc>()) {
            strLoc = "UNKNOWN";
        } else {
            strLoc = stringifyLocation(baseLoc);
        }
        if (_counter.count(strLoc) == 0) {
            _counter[strLoc] = 1;
            return baseLoc;
        }
        _counter[strLoc]++;
        _log.warning("Duplicated '{0}' location attribute.", baseLoc);
        return appendLoc(baseLoc, formatv("Duplicated_{0}", _counter[strLoc]));
    }

private:
    vpux::Logger& _log;
    std::map<std::string, size_t> _counter{};
};

// Utility structure to save information of individual task that is profiled
template <class InnerTask>
struct TaskSignature {
    InnerTask _task;
    unsigned _maxSubTasks;
    SmallVector<unsigned> _subTasksAtCluster;

    // Save signature to string, which will be parsed by prof_parser
    // Format is PROF_{TASK_ID}_{BUFFER_ID}_{NUM_CLUSTERS}_{ALIGNMENT}-{TASK_CNT1},{TASK_CNT2}...
    std::string dpuSignature(int taskId, int bufferId) const {
        const auto clustersNum = _subTasksAtCluster.size();
        const auto clusterAlignment = _maxSubTasks;
        std::stringstream formatter;
        formatter << PROFILING_PREFIX << "_" << taskId << "_" << bufferId << "_" << clustersNum << "_"
                  << clusterAlignment << "-";
        for (const unsigned variantsAmount : _subTasksAtCluster) {
            formatter << variantsAmount << ",";
        }
        return formatter.str();
    }
};

mlir::BlockArgument addNewProfilingOutput(mlir::MLIRContext* ctx, mlir::func::FuncOp& netFunc, IE::CNNNetworkOp& netOp,
                                          mlir::MemRefType outputType, StringRef name);

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
}  // namespace vpux
