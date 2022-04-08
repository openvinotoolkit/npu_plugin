//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/strings.hpp"
#include "vpux/utils/core/func_ref.hpp"

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
    // Format is _PROF_{TASK_ID}_{BUFFER_ID}_{NUM_CLUSTERS}_{ALIGNMENT}-{TASK_CNT1},{TASK_CNT2}...
    std::string signature(int taskId, int bufferId) const {
        const auto clustersAmount = _subTasksAtCluster.size();
        const auto clusterAlignment = _maxSubTasks;
        std::stringstream formatter;
        formatter << "_PROF_" << taskId << "_" << bufferId << "_" << clustersAmount << "_" << clusterAlignment << "-";
        for (const unsigned variantsAmount : _subTasksAtCluster) {
            formatter << variantsAmount << ",";
        }
        return formatter.str();
    }
};

constexpr char PROFILING_CMX_2_DDR_OP_NAME[] = "ProfilingCMX2DDR";
constexpr char PROFILING_DMA_BEGIN_SUFFIX[] = "_PROFBEGIN";
constexpr char PROFILING_DMA_TASK_BEGIN_SUFFIX[] = "_PROFTASKBEGIN";
constexpr char PROFILING_DMA_TASK_END_SUFFIX[] = "_PROFTASKEND_";

class ChunkWalker {
private:
    unsigned chunkId = 0;       // Chunk id
    unsigned chunkItemId = 0;   // id inside current chunk
    const unsigned chunks;      // Total number of chunks including last one
    const unsigned opsInChunk;  // Number of items in the usual chunk
    const unsigned lastChunk;   // Number of items in the last chunk

public:
    ChunkWalker(unsigned totalSizeBytes, unsigned chunkSize, unsigned elemSize, Logger _log)
            : chunks(static_cast<unsigned>(ceil(static_cast<double>(totalSizeBytes) / chunkSize))),
              opsInChunk(chunkSize / elemSize),
              lastChunk((totalSizeBytes % chunkSize) / elemSize) {
        _log.trace("totalSizeBytes='{0}'\nchunks='{1}'\nops_in_chunk='{2}'\nlast_chunk='{3}'\n", totalSizeBytes, chunks,
                   opsInChunk, lastChunk);
    }

    template <typename T>
    void run(T items, FuncRef<void(unsigned, unsigned, bool)> chunkSwitchCallback,
             FuncRef<void(std::remove_reference_t<decltype(*std::begin(std::declval<T&>()))>, unsigned&)>
                     chunkItemCallback) {
        for (auto& item : llvm::make_early_inc_range(items)) {
            // Start new chunk once we reached the end of the previous one
            if (chunkItemId && ((chunkItemId % opsInChunk) == 0)) {
                chunkItemId = 0;
                chunkId++;
            }
            // Beginning of the chunk
            if (chunkItemId == 0) {
                chunkSwitchCallback(chunkId, opsInChunk, (chunkId == chunks - 1));
            }
            chunkItemCallback(item, chunkItemId);
        }
    }

    void increment() {
        chunkItemId++;
    }

    const auto getChunks() {
        return chunks;
    }

    const auto getOpsInChunk() {
        return opsInChunk;
    }

    const auto getOpsInLastChunk() {
        return (lastChunk) ? lastChunk : opsInChunk;
    }
};

mlir::BlockArgument addNewProfilingOutput(mlir::MLIRContext* ctx, mlir::FuncOp& netFunc, IE::CNNNetworkOp& netOp,
                                          mlir::MemRefType outputType, StringRef name);

}  // namespace vpux
