// {% copyright %}

#ifndef NN_INFERENCE_RUNTIME_TYPES_2490_H_
#define NN_INFERENCE_RUNTIME_TYPES_2490_H_

#include <sw_nn_runtime_types.h>
#include <nn_runtime_types.h>
#include <nn_relocation.h>
#include <nn_memory.h>
#include <array>
#include <algorithm>
#include <OsDrvCmxDma.h>
#include <nn_perf_measurement.h>

namespace nn
{
    namespace shave_lib
    {

        // The SLT context used by Player on LNN
        struct alignas(64) SoftLayerRuntime
        {
            SoftLayerExec exec_;
            dpu_runtime::BarrierUserConfig barriers_;

            SoftLayerRuntime() :
                exec_(),
                barriers_()
            {
            }
        };
    }

    namespace inference_runtime
    {
        namespace frontend
        {
            struct DPUInvariantWrapper
            {
                dpu_runtime::DPULayerTypes layerOpDPU_;
                dpu_runtime::DPUInvariant invariant_;
                unsigned short variant_count_;

                DPUInvariantWrapper() :
                    layerOpDPU_(dpu_runtime::DPULayerTypes::NO_OP),
                    invariant_(),
                    variant_count_(0)
                {
                }
            };

            struct DPUInvariantExtension
            {
                dpu_runtime::DPUAddresses addresses_;

                DPUInvariantExtension() :
                    addresses_()
                {
                }
            };

            struct DPUVariantWrapper
            {
                dpu_runtime::DPUVariant variant_;
                unsigned int invariant_index_;

                DPUVariantWrapper() :
                    variant_(),
                    invariant_index_(0)
                {
                }
            };

            struct DMATask
            {
                OsDrvCmxDmaTransaction transaction_;
                dpu_runtime::BarrierUserConfig barriers_;

                DMATask() :
                    transaction_(),
                    barriers_()
                {
                }
            };

            struct DMAExtension
            {
                RelativeAddress src_;
                RelativeAddress dst_;

                DMAExtension() :
                    src_(),
                    dst_()
                {
                }
            };

            struct SoftLayerTask
            {
                shave_lib::Layer layer_;
                dpu_runtime::BarrierUserConfig barriers_;

                inference_runtime::RelativeAddress rel_inputs_[shave_lib::MAX_INPUT_TENSORS];
                inference_runtime::RelativeAddress rel_outputs_[shave_lib::MAX_OUTPUT_TENSORS];

                uint8_t num_inputs_;
                uint8_t num_outputs_;

                bool is_trailing_layer;

                SoftLayerTask() :
                    layer_(),
                    barriers_(),
                    num_inputs_(0),
                    num_outputs_(0),
                    is_trailing_layer(false)
                {
                }

                SoftLayerTask(const SoftLayerTask &) = delete;
                SoftLayerTask &operator=(const SoftLayerTask &) = delete;

                SoftLayerTask(SoftLayerTask &&) = default;
                SoftLayerTask &operator=(SoftLayerTask &&) = default;
            };
        }

        namespace backend
        {
            typedef frontend::DPUInvariantWrapper DPUInvariantWrapper;
            typedef frontend::DPUVariantWrapper DPUVariantWrapper;
            typedef frontend::DMATask DMATask;
            typedef shave_lib::AbsoluteAddresses AbsoluteAddresses;
            typedef shave_lib::SoftLayerRuntime SoftLayerRuntime;
        }

        struct BarrierCfg
        {
            unsigned char real_id_;
            short next_same_id_;
            unsigned short producer_count_;
            unsigned short consumer_count_;

            BarrierCfg() :
                real_id_(255),
                next_same_id_(-1),
                producer_count_(0),
                consumer_count_(0)
            {
            }
        };

        struct NN_CACHE_ALIGNED ResourceRequirements
        {
            unsigned char upa_shaves_;
            unsigned char nn_slice_count_;
            unsigned char nn_barriers_;
            unsigned int nn_slice_length_;
            unsigned int ddr_scratch_length_;

            ResourceRequirements() :
                upa_shaves_(0),
                nn_slice_count_(0),
                nn_barriers_(0),
                nn_slice_length_(0),
                ddr_scratch_length_(0)
            {
            }
        };

        struct NN_CACHE_ALIGNED Inference
        {
            ResourceRequirements resource_requirements_;
            std::array<nn::memory::shared_vector<frontend::DMATask>, MAX_DMA_ENGINES> dmaTasks_;
            std::array<nn::memory::shared_vector<frontend::DMAExtension>, MAX_DMA_ENGINES> dmaExtensions_;
            nn::memory::shared_vector<frontend::DPUInvariantWrapper> invariants_;
            nn::memory::shared_vector<frontend::DPUInvariantExtension> extensions_;
            nn::memory::shared_vector<frontend::DPUVariantWrapper> variants_;

            // SoftLayerTasks need to be kept in protected storage because their
            // Layer.SoftParams member will polymorphically delete its LayerParams member which,
            // if corrupted, could call a random function based on a corrupt virtual table.
            nn::memory::cache_aligned_vector<frontend::SoftLayerTask> softLayers_;

            nn::memory::shared_vector<BarrierCfg> barrierConfigs_;
        };

        struct NN_CACHE_ALIGNED MappedInference
        {
            std::array<unsigned int, MAX_DMA_ENGINES> leadingDmaTasks_;
            std::array<memory::FixedVector<backend::DMATask>, MAX_DMA_ENGINES> dmaTasks_;
            memory::FixedVector<backend::DPUInvariantWrapper> invariants_;
            memory::FixedVector<backend::DPUVariantWrapper> variants_;
            memory::FixedVector<backend::SoftLayerRuntime> endSoftLayerRTs_;
            memory::FixedVector<backend::SoftLayerRuntime> softLayerRTs_;
            memory::FixedVector<BarrierCfg> barrierConfigs_;
        };

        struct NN_CACHE_ALIGNED ParsedInference
        {
            ResourceRequirements resource_requirements_;
            memory::FixedVector<MappedInference> mapped_;

        };

        struct NN_CACHE_ALIGNED InferenceRequest
        {
            enum Code
            {
                IDLE,
                PENDING,
                RUNNING,
                COMPLETE,
                OUT_OF_MEMORY,
                OUT_OF_RESOURCES,
                UNDEFINED
            };

            struct NN_CACHE_ALIGNED Request
            {
                shave_lib::ShavePerfCounters *perfCounters_ { nullptr };
                unsigned int *lnnTicksPerSoftLayer_ { nullptr };
                unsigned int *barrier_lift_times_ { nullptr };
                unsigned int *barrier_free_times_ { nullptr };
            } request_;

            struct NN_CACHE_ALIGNED Response
            {
                Code code_ { IDLE };
                unsigned int lnn_ticks_ { 0 };
                const unsigned int *timestamp_ { nullptr };
            } response_;

            InferenceRequest() :
                request_(),
                response_()
            {
            }
        };

        struct UserContextInfo
        {
            // Used in mvnci.h for parameter passing
            UserContextInfo() {}
            UserContextInfo(uint32_t, uint32_t) {}
        };

        struct alignas(64) WorkRequest
        {
            enum Phase
            {
                PREFETCH,
                PREPARE,
                RUN,
            };

            void *user_data_;
            Phase phase_;

            union
            {
                struct
                {
                    const ResourceRequirements &resources_;
                    const MappedInference *mapped_;
                    InferenceRequest *inference_request_;
                    unsigned char upa_ctrl_shave_;
                };
                struct
                {
                    unsigned int count_;
                    const ConstBuffer *from_;
                    const ConstBuffer *to_;
                };
            };

             WorkRequest(const ResourceRequirements &res, const MappedInference *mapped, InferenceRequest *ir, const UserContextInfo&) :
                user_data_(nullptr),
                phase_(ir ? RUN : PREPARE),
                resources_(res),
                mapped_(mapped),
                inference_request_(ir),
                upa_ctrl_shave_(INVALID_SHAVE_ID)
            {
            }

            WorkRequest(unsigned int count, const ConstBuffer *from, const ConstBuffer *to) :
                user_data_(nullptr),
                phase_(PREFETCH),
                count_(count),
                from_(from),
                to_(to)
            {
            }
        };
    }
}

#endif // NN_INFERENCE_RUNTIME_TYPES_2490_H_
