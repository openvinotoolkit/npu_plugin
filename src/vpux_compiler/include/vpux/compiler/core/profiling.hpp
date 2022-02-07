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

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/utils/core/func_ref.hpp"

namespace vpux {

constexpr char PROFILING_CMX_2_DDR_OP_NAME[] = "ProfilingCMX2DDR";

class ChunkWalker {
private:
    unsigned chunkId = 0;       // Chunk id
    unsigned chunkItemId = 0;   // id inside current chunk
    const unsigned chunks;      // Total number of chunks including last one
    const unsigned opsInChunk;  // Number of items in the usual chunk
    const unsigned lastChunk;   // Number of items in the last chunk

public:
    ChunkWalker(unsigned totalSizeBytes, unsigned chunkSize, unsigned elemSize, Logger _log)
            : chunks(static_cast<unsigned>(ceil((double)totalSizeBytes / chunkSize))),
              opsInChunk(chunkSize / elemSize),
              lastChunk((totalSizeBytes % chunkSize) / elemSize) {
        _log.trace("totalSizeBytes='{0}'\nchunks='{1}'\nops_in_chunk='{2}'\nlast_chunk='{3}'\n", totalSizeBytes, chunks,
                   opsInChunk, lastChunk);
    }

    template <typename T>
    void run(T items, FuncRef<void(unsigned, unsigned, bool)> chunkSwitchCallback,
             FuncRef<void(std::remove_reference_t<decltype(*std::begin(std::declval<T&>()))>, unsigned&)>
                     chunkItemCallback) {
        for (auto& item : items) {
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
