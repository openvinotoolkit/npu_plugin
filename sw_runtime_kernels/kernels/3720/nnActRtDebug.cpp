/*
 * {% copyright %}
 */

#include "nnActRtDebug.h"
#include "nn_fifo.h"
#include "nn_barrier.h"

using namespace nn::util;

#ifdef NN_ENABLE_CONTEXT_DEBUGGING
namespace nn {
namespace act_runtime {

void execDebug(ActKernelRange *wl, unsigned int shaveIndex, SHVFifoConfig cfg) {
    constexpr uint32_t rtError = 0xDEADDEAD;
    constexpr uint32_t dbgToken = 0xDEAD0000;
    constexpr uint32_t dbgToken2 = 0xCAFEBABE;
    const unsigned int shaveOtherTile = (shaveIndex >= AS_PER_TILE) ? 0 : AS_PER_TILE;
    const auto txShvId = reinterpret_cast<void *>(dbgToken | (shaveIndex & 0xFFFF));
    const FifoConfig rx = cfg.work;
    const FifoConfig tx = cfg.perf;
    const FifoConfig rxWrongFifo = unpackSHVConfig(acts_cfgs[shaveOtherTile]).work;
    const FifoConfig txWrongFifo = unpackSHVConfig(acts_cfgs[shaveOtherTile]).perf;

    switch (wl->dbg_type_) {
        case ActDebug::INVALID: {
            wl->dbg0_ = rtError;
            fifoDynamicSend(tx.fifo, tx.index, txShvId);
            break;
        }
        case ActDebug::DEBUG_ACK: {
            wl->dbg0_ = dbgToken;
            fifoDynamicSend(tx.fifo, tx.index, txShvId);
            break;
        }
        case ActDebug::DEBUG_ACK_WAIT: {
            // Wait here until LNN changes the debug token in dbg1_
            while (wl->dbg1_ == dbgToken2)
                ;

            wl->dbg0_ = dbgToken;
            // Send a debug token
            fifoDynamicSend(tx.fifo, tx.index, txShvId);

            break;
        }
        case ActDebug::DEBUG_FIFO_CLEAR: {
            void *dbgVal = nullptr;
            unsigned int fifo = (unsigned int)wl->dbg2_;
            unsigned int index = (unsigned int)wl->dbg3_;
            wl->dbg0_ = dbgToken;

            // Wait here until LNN changes the debug token in dbg1_
            while (wl->dbg1_ == dbgToken2)
                ;

            dbgVal = fifoDynamicReceive(fifo, index);

            // Send a debug token to signal test completion
            fifoDynamicSend(tx.fifo, tx.index, dbgVal);
            break;
        }
        case ActDebug::DEBUG_CACHE_INVALIDATE: {
            // dbg0 stores a DDR pointer where ACTSHV must read from (and fill a cache line)
            uint32_t *dbgRead = (uint32_t *)wl->dbg0_;

            // Store the value that we read from ACTSHV L2C for LNN to verify later
            wl->dbg1_ = *dbgRead;

            // Send a debug token
            fifoDynamicSend(tx.fifo, tx.index, txShvId);

            break;
        }
        case ActDebug::DEBUG_CACHE_FLUSH: {
            // dbg0 stores a DDR pointer where ACTSHV must write to (and fill a cache line)
            uint32_t *dbgWrite = (uint32_t *)wl->dbg0_;

            // Write to DDR via the cache mechanism
            *dbgWrite = dbgToken2;

            // Store the value that we wrote for comparison on LNN
            wl->dbg1_ = *dbgWrite;

            // Send a debug token
            fifoDynamicSend(tx.fifo, tx.index, txShvId);

            break;
        }
        case ActDebug::DEBUG_FIFO_CONFIG_GLOBAL: {
            // this is not allowed from a user context
            fifo3MonitorSelect(wl->dbg1_,wl->dbg2_);


            wl->dbg0_ = dbgToken;
            fifoDynamicSend(tx.fifo, tx.index, txShvId);
            break;
        }
        case ActDebug::DEBUG_FIFO_WRITE: {
            fifoDynamicSend(tx.fifo, tx.index, (void *)dbgToken);

            break;
        }
        case ActDebug::DEBUG_FIFO_WRONG_READ: {
            fifoDynamicReceive(rxWrongFifo.fifo, rxWrongFifo.index);
            fifoDynamicSend(tx.fifo, tx.index, txShvId);

            break;
        }
        case ActDebug::DEBUG_FIFO_WRONG_WRITE: {
            fifoDynamicSend(txWrongFifo.fifo, txWrongFifo.index, (void *)dbgToken);
            fifoDynamicSend(tx.fifo, tx.index, txShvId);

            break;
        }
        case ActDebug::DEBUG_VALID_BARRIER: {
            unsigned long long barrier_mask = rx.index == 0 ? 1 << 0 : (unsigned long long)1 << (NUM_BARRIERS / 2);
            wl->dbg0_ = dbgToken2;

            HglBarrierProduce(barrier_mask);
            HglBarrierConsume(barrier_mask);

            fifoDynamicSend(tx.fifo, tx.index, txShvId);

            break;
        }
        case ActDebug::DEBUG_INVALID_BARRIER_P: {
            unsigned long long barrier_mask = rx.index == 1 ? 1 << 0 : (unsigned long long)1 << (NUM_BARRIERS / 2);
            wl->dbg0_ = dbgToken2;

            HglBarrierProduce(barrier_mask);

            fifoDynamicSend(tx.fifo, tx.index, txShvId);

            break;
        }
        case ActDebug::DEBUG_INVALID_BARRIER_C: {
            unsigned long long barrier_mask = rx.index == 1 ? 1 << 0 : (unsigned long long)1 << (NUM_BARRIERS / 2);
            wl->dbg0_ = dbgToken2;

            HglBarrierConsume(barrier_mask);

            fifoDynamicSend(tx.fifo, tx.index, txShvId);

            break;
        }
        default: {
            break;
        }
    }
}
} // namespace act_runtime
} // namespace nn
#endif
