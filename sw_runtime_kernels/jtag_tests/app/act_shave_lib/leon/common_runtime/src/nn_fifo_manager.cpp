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

#include "nn_fifo_manager.h"
#include "nn_fifo.h"
#include "nn_log.h"

namespace nn {
namespace common_runtime {
namespace fifo {

SHVFifoConfig unpackSHVConfig(uint32_t packedCfg) {
    SHVFifoConfig ret{};

    ret.work.fifo = (uint8_t)(0b1111 & (packedCfg >> (32 - 4 * 1)));
    ret.work.index = (uint8_t)(0b1111 & (packedCfg >> (32 - 4 * 2)));
    ret.ctrx.fifo = (uint8_t)(0b1111 & (packedCfg >> (32 - 4 * 3)));
    ret.ctrx.index = (uint8_t)(0b1111 & (packedCfg >> (32 - 4 * 4)));
    ret.cttx.fifo = (uint8_t)(0b1111 & (packedCfg >> (32 - 4 * 5)));
    ret.cttx.index = (uint8_t)(0b1111 & (packedCfg >> (32 - 4 * 6)));
    ret.perf.fifo = (uint8_t)(0b1111 & (packedCfg >> (32 - 4 * 7)));
    ret.perf.index = (uint8_t)(0b1111 & (packedCfg >> (32 - 4 * 8)));

    return ret;
}

#if defined(__leon__) || defined(__leon_nn__)

uint32_t packSNNCtrlMessage(SNNCtrlMessage cm) {
    if (cm.payload & CTRL_MESSAGE_MASK)
        return 0;

    uint32_t out{cm.payload};
    out |= cm.message << MAX_CTRL_PAYLOAD_BITS;
    return out;
}

uint32_t packASCtrlMessage(ASCtrlMessage cm) {
    if (cm.payload & CTRL_MESSAGE_MASK)
        return 0;

    uint32_t out{cm.payload};
    out |= cm.message << MAX_CTRL_PAYLOAD_BITS;
    return out;
}

SNNCtrlResponse unpackSNNCtrlResponse(uint32_t cr) {
    UNUSED(cr);
    return SNNCtrlResponse();
}

ASCtrlResponse unpackASCtrlResponse(uint32_t cr) {
    UNUSED(cr);
    return ASCtrlResponse();
}

bool isSNNWorkFifoFull(uint8_t tile) {
    switch (tile) {
        case 0:
            return util::isFifoFullDynamic(TILE0_SNN0_WORK_FIFO_NUM, TILE0_SNN0_WORK_FIFO_INDEX);
        case 1:
            return util::isFifoFullDynamic(TILE1_SNN0_WORK_FIFO_NUM, TILE1_SNN0_WORK_FIFO_INDEX);
        default:
            nnLog(MVLOG_ERROR, "SNN Work fifo send fail: Tile number too high");
            return true;
    }
}

bool isASWorkFifoFull(uint8_t tile) {
    UNUSED(tile);
    nnLog(MVLOG_ERROR, "Error: function is unimplemented");
    return {};
}

bool isSNNCtrlFifoFull(uint8_t tile, uint8_t snn) {
    UNUSED(tile);
    UNUSED(snn);
    nnLog(MVLOG_ERROR, "Error: function is unimplemented");
    return {};
}

bool isASCtrlFifoFull(uint8_t tile, uint8_t as) {
    UNUSED(tile);
    UNUSED(as);
    nnLog(MVLOG_ERROR, "Error: function is unimplemented");
    return {};
}

void printFifoConfig(SHVFifoConfig config) {
    nnLog(MVLOG_DEBUG, "Work: FIFO%d_%d\tPerf: FIFO%d_%d", config.work.fifo, config.work.index, config.perf.fifo,
          config.perf.index);
    nnLog(MVLOG_DEBUG, "CtTX: FIFO%d_%d\tCtRX: FIFO%d_%d", config.cttx.fifo, config.cttx.index, config.ctrx.fifo,
          config.ctrx.index);
}

void sendWorkToSNNs(uint8_t tile, void *p) {
    switch (tile) {
        case 0:
            util::fifoDynamicSend(TILE0_SNN0_WORK_FIFO_NUM, TILE0_SNN0_WORK_FIFO_INDEX, p);
            break;
        case 1:
            util::fifoDynamicSend(TILE1_SNN0_WORK_FIFO_NUM, TILE1_SNN0_WORK_FIFO_INDEX, p);
            break;
        default:
            nnLog(MVLOG_ERROR, "SNN Work fifo send fail: Tile number too high");
            break;
    }
}

void sendWorkToASs(uint8_t tile, void *p) {
    switch (tile) {
        case 0:
            util::fifoDynamicSend(TILE0_AS0_WORK_FIFO_NUM, TILE0_AS0_WORK_FIFO_INDEX, p);
            break;
        case 1:
            util::fifoDynamicSend(TILE1_AS0_WORK_FIFO_NUM, TILE1_AS0_WORK_FIFO_INDEX, p);
            break;
        default:
            nnLog(MVLOG_ERROR, "ActSHV Work fifo send fail: Tile number too high");
            break;
    }
}

void sendCtrlToSNNs(uint8_t tile, uint8_t snn, uint32_t packedMessage) {
    UNUSED(tile);
    UNUSED(snn);
    UNUSED(packedMessage);
}

void sendCtrlToASs(uint8_t tile, uint8_t as, uint32_t packedMessage) {
    uint8_t fifoNum{0};
    uint8_t fifoIdx{0};

    static_assert(AS_PER_TILE == 2, "Rewrite this to support different number of act-shaves per tile");

    switch (tile) {
        case 0:
            switch (as) {
                case 0:
                    // TODO: check out the compilation and make sure these constexpr calls collapse into immidiate
                    // values
                    fifoNum = getActCtrlRxFifo(0, 0).fifo;
                    fifoIdx = getActCtrlRxFifo(0, 0).index;
                    break;
                case 1:
                    fifoNum = getActCtrlRxFifo(0, 1).fifo;
                    fifoIdx = getActCtrlRxFifo(0, 1).index;
                    break;
            }
            break;
        case 1:
            switch (as) {
                case 0:
                    fifoNum = getActCtrlRxFifo(1, 0).fifo;
                    fifoIdx = getActCtrlRxFifo(1, 0).index;
                    break;
                case 1:
                    fifoNum = getActCtrlRxFifo(1, 1).fifo;
                    fifoIdx = getActCtrlRxFifo(1, 1).index;
                    break;
            }
        default:
            nnLog(MVLOG_ERROR, "ActSHV Work fifo send fail: Tile number too high");
            break;
    }

    util::fifoDynamicSend(fifoNum, fifoIdx, reinterpret_cast<void *>(packedMessage));
}

// FIXME: we could further limit compiled code per shave type if there was a `defined(__act_shave__)`
#elif defined(__shave__) || defined(__shave_nn__) || defined(__act_shave__)

uint32_t packSNNCtrlResponse(SNNCtrlResponse cr) {
    UNUSED(cr);
    return 0;
}

uint32_t packASCtrlResponse(ASCtrlResponse cr) {
    UNUSED(cr);
    return 0;
}

//uint64_t packSNNPerfReport(SNNPerfReport pr) {
//    UNUSED(pr);
//    return 0;
//}
//
//uint64_t packASPerfReport(ASPerfReport pr) {
//    UNUSED(pr);
//    return 0;
//}

SNNCtrlMessage unpackSNNCtrlMessage(uint32_t cm) {
    return {((SHVCtrlMessage)((cm & CTRL_MESSAGE_MASK) >> MAX_CTRL_MESSAGE_BITS)), (cm & ~CTRL_MESSAGE_MASK)};
}

ASCtrlMessage unpackASCtrlMessage(uint32_t cm) {
    return {((SHVCtrlMessage)((cm & CTRL_MESSAGE_MASK) >> MAX_CTRL_MESSAGE_BITS)), (cm & ~CTRL_MESSAGE_MASK)};
}

#endif

} // namespace fifo
} // namespace common_runtime
} // namespace nn
