/*
* {% copyright %}
*/
#include "nn_resource_locator.h"
#include <nn_resources.h>
#include <nn_math.h>
#include <nn_log.h>
#include <assert.h>

namespace nn
{
    namespace inference_runtime
    {
        StaticMapping::StaticMapping(NNCmxMemoryMap *cmx) :
            workareas_
            {
                Buffer(&cmx->slice0_.workspace_, sizeof(cmx->slice0_.workspace_)),
                Buffer(&cmx->slice1_.workspace_, sizeof(cmx->slice1_.workspace_)),
//FIXME: move to seperate impl
#if !defined(CONFIG_TARGET_SOC_3600) && !defined(CONFIG_TARGET_SOC_3710) && !defined(CONFIG_TARGET_SOC_3720)
                Buffer(&cmx->slice2_.workspace_, sizeof(cmx->slice2_.workspace_)),
                Buffer(&cmx->slice3_.workspace_, sizeof(cmx->slice3_.workspace_)),
#endif
            },
            metadataStorage_
            {
                Buffer(&cmx->slice0_.metadata_, sizeof(cmx->slice0_.metadata_)),
                Buffer(&cmx->slice1_.metadata_, sizeof(cmx->slice1_.metadata_)),
#if !defined(CONFIG_TARGET_SOC_3600) && !defined(CONFIG_TARGET_SOC_3710) && !defined(CONFIG_TARGET_SOC_3720)
                Buffer(&cmx->slice2_.metadata_, sizeof(cmx->slice2_.metadata_)),
                Buffer(&cmx->slice3_.metadata_, sizeof(cmx->slice3_.metadata_)),
#endif
            },
#if !defined(CONFIG_TARGET_SOC_3600 ) && !defined(CONFIG_TARGET_SOC_3710) && !defined(CONFIG_TARGET_SOC_3720)
            dmaStorage_
            {
                Buffer(&cmx->slice2_.lnn_data_, sizeof(cmx->slice2_.lnn_data_)),
                Buffer(&cmx->slice3_.lnn_data_, sizeof(cmx->slice3_.lnn_data_)),
            }
#else
    // TODO FIXME
            dmaStorage_
            {
                Buffer(&cmx->slice0_.lnn_data_, sizeof(cmx->slice0_.lnn_data_)),
                Buffer(&cmx->slice1_.lnn_data_, sizeof(cmx->slice1_.lnn_data_)),
            }
#endif
        {
        }

        RuntimeMapping::RuntimeMapping() :
            dma_(),
            inv_(),
            var_(),
            config_(0),
            fifos_()
        {
        }

        RuntimeMapping::RuntimeMapping(const StaticMapping &global, ClusterMapper::Config config) :
            dma_(),
            inv_(),
            var_(),
            config_(config),
            fifos_()
        {
            if (config == 0)
                return;

            for (unsigned int i = 0; i < MAX_DMA_ENGINES; ++i)
                dma_[i] = DMALocator(global.dmaStorage_[i].addr32() + global.dmaStorage_[i].size() / MAX_CLUSTERS * config.range().first,
                                     global.dmaStorage_[i].size() / MAX_CLUSTERS * config.range().second);

            {
                auto first = math::firstBitIndex(static_cast<unsigned int>(config));
                auto second = math::firstBitIndex(static_cast<unsigned int>(config) ^ (1u << first));

                if (second > 0)
                {
                    inv_ = InvariantLocator(global.metadataStorage_[static_cast<unsigned int>(first)].addr32(), global.metadataStorage_[static_cast<unsigned int>(first)].size());
                    var_ = VariantLocator(global.metadataStorage_[static_cast<unsigned int>(second)].addr32(), global.metadataStorage_[static_cast<unsigned int>(second)].size());
                }
                else
                {
                    inv_ = InvariantLocator(global.metadataStorage_[static_cast<unsigned int>(first)].addr32(), INVARIANT_COUNT * sizeof(backend::DPUInvariantWrapper));
                    var_ = VariantLocator(inv_.tasks() + inv_.count(), VARIANT_COUNT * sizeof(backend::DPUVariantWrapper));
                    assert(reinterpret_cast<uint32_t>(var_.tasks() + var_.count()) <= global.metadataStorage_[static_cast<unsigned int>(first)].addr32() + global.metadataStorage_[static_cast<unsigned int>(first)].size() && "Metadata storage exceeded");
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
}
