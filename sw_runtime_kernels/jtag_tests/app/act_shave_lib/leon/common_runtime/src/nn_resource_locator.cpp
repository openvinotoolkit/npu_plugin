/*
 * {% copyright %}
 */
#if defined(__leon__) || defined(__leon_nn__)

#include "nn_resource_locator.h"
#include <nn_hw_resources.h>
#include <nn_math.h>
#include <nn_log.h>
#include <assert.h>

namespace nn {
namespace common_runtime {
StaticMapping::StaticMapping(NNCmxMemoryMap *cmx)
    : globalData_{Buffer(&cmx->slice_[0].global_, sizeof(cmx->slice_[0].global_)),
                  Buffer(&cmx->slice_[1].global_, sizeof(cmx->slice_[1].global_))}
    , snnStack_{Buffer(&cmx->slice_[0].global_.snn_stack_, sizeof(cmx->slice_[0].global_.snn_stack_)),
                Buffer(&cmx->slice_[1].global_.snn_stack_, sizeof(cmx->slice_[1].global_.snn_stack_))}
    , workareas_{Buffer(&cmx->slice_[0].user_.workspace_, sizeof(cmx->slice_[0].user_.workspace_)),
                 Buffer(&cmx->slice_[1].user_.workspace_, sizeof(cmx->slice_[1].user_.workspace_))}
    , actShvStack_{Buffer(&cmx->slice_[0].user_.actshv_stack_, sizeof(cmx->slice_[0].user_.actshv_stack_)),
                   Buffer(&cmx->slice_[1].user_.actshv_stack_, sizeof(cmx->slice_[1].user_.actshv_stack_))}
    , metadataStorage_{Buffer(&cmx->slice_[0].user_.metadata_, sizeof(cmx->slice_[0].user_.metadata_)),
                       Buffer(&cmx->slice_[1].user_.metadata_, sizeof(cmx->slice_[1].user_.metadata_))}
    , dmaStorage_{Buffer(&cmx->slice_[0].user_.dma_storage_, sizeof(cmx->slice_[0].user_.dma_storage_)),
                  Buffer(&cmx->slice_[1].user_.dma_storage_, sizeof(cmx->slice_[1].user_.dma_storage_))} {}

RuntimeMapping::RuntimeMapping()
    : dma_()
    , akr_()
    , aki_()
    , config_(0)
    , fifos_() {}

RuntimeMapping::RuntimeMapping(const StaticMapping &global, ClusterMapper::Config config)
    : dma_()
    , akr_()
    , aki_()
    , config_(config)
    , fifos_() {
    if (config == 0)
        return;

    for (unsigned int i = 0; i < MAX_DMA_ENGINES; ++i) {
        uint32_t tileIndex = config.index();
        uint32_t size = global.dmaStorage_[tileIndex].size() / MAX_DMA_ENGINES;
        dma_[i] = DMALocator(global.dmaStorage_[tileIndex].addr32() + size * i, size);
    }

    {
        auto first = math::firstBitIndex(static_cast<unsigned int>(config));
        auto second = math::firstBitIndex(static_cast<unsigned int>(config) ^ (1u << first));

        // FIXME: Why is storage allocation so lopsided in the 2 tile case? i.e. vars_size >> inv_size

        if (second > 0) {
            akr_ = AKRangeLocator(global.metadataStorage_[static_cast<unsigned int>(first)].addr32(),
                                  KERNAL_RANGE_COUNT * sizeof(backend::ActKernelRangeWrapper));

            if (reinterpret_cast<uint32_t>(akr_.tasks() + akr_.count()) >=
                global.metadataStorage_[static_cast<unsigned int>(first)].addr32() +
                    global.metadataStorage_[static_cast<unsigned int>(first)].size()) {
                nnLog(MVLOG_ERROR, "Metadata storage exceeded for first tile");
            }

            aki_ = AKInvocationLocator(global.metadataStorage_[static_cast<unsigned int>(second)].addr32(),
                                       KERNAL_INVO_COUNT * sizeof(backend::ActKernelInvocationWrapper));

            if (reinterpret_cast<uint32_t>(aki_.tasks() + aki_.count()) >=
                global.metadataStorage_[static_cast<unsigned int>(second)].addr32() +
                    global.metadataStorage_[static_cast<unsigned int>(second)].size()) {
                nnLog(MVLOG_ERROR, "Metadata storage exceeded for second tile");
            }

        } else {
            akr_ = AKRangeLocator(global.metadataStorage_[static_cast<unsigned int>(first)].addr32(),
                                  KERNAL_RANGE_COUNT * sizeof(backend::ActKernelRangeWrapper));

            aki_ = AKInvocationLocator(akr_.tasks() + akr_.count(),
                                       KERNAL_INVO_COUNT * sizeof(backend::ActKernelInvocationWrapper));

            if (reinterpret_cast<uint32_t>(aki_.tasks() + aki_.count()) >=
                global.metadataStorage_[static_cast<unsigned int>(first)].addr32() +
                    global.metadataStorage_[static_cast<unsigned int>(first)].size()) {
                nnLog(MVLOG_ERROR, "Metadata storage exceeded");
            }
        }

        unsigned int active_fifos = 0;
        for (unsigned int i = 0, mask = config; i < fifos_.size(); ++i, mask >>= 1)
            if (mask & 1)
                fifos_[active_fifos++] = static_cast<unsigned char>(i);

        // Extend the resulting mapping(s) to all FIFO indirections
        for (unsigned int i = active_fifos; i < fifos_.size(); ++i)
            fifos_[i] = fifos_[i % active_fifos];
    }
}
} // namespace common_runtime
} // namespace nn

#endif
