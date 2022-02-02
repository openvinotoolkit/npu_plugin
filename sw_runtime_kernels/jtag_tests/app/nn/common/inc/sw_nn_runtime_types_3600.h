/*
* {% copyright %}
*/
#ifndef SW_NN_RUNTIME_TYPES_H_3600__
#define SW_NN_RUNTIME_TYPES_H_3600__

#include <cstdint>
#include <sw_layer.h>

#define INVALID_SHAVE_ID 4 //
// MTL stubs
namespace nn {
  namespace util {
    struct TaskContext;
  }

  namespace shave_lib {
    constexpr uint8_t MAX_INPUT_TENSORS = 8;
    constexpr uint8_t MAX_OUTPUT_TENSORS = 4;

    struct AbsoluteAddresses {
      const unsigned char *inputs_[MAX_INPUT_TENSORS] = {nullptr};
      unsigned char *outputs_[MAX_OUTPUT_TENSORS] = {nullptr};
    };

    struct Layer {
        // Windowed address of the kernel function
        void setParams(unsigned int paramID, LayerParams *lp) {
            this->params.paramsID = paramID;
            this->params.layerParams = lp;
        }
        SoftParams params;
        preamble pre { nullptr };
        shaveKernelEntry kernelEntry { nullptr };
        unsigned char maxShaves { 1 };
        void setKernelEntry(shaveKernelEntry kernelEntry) {
            this->kernelEntry = kernelEntry;
        }

        void setPreamble(preamble newpre = nullptr) {
            this->pre = newpre;
        }
        memory::cache_aligned_vector<TensorRef> &getInputs() {
            return params.inputs;
        }
        memory::cache_aligned_vector<TensorRef> &getOutputs() {
            return params.outputs;
        }
    };

    struct alignas(64) SoftLayerExec
    {
      AbsoluteAddresses abs_addr_;
      const Layer *layer_;
      void *counters_;
      bool completed_;

      SoftLayerExec()
          : abs_addr_{}, layer_{nullptr}, counters_{nullptr},
            completed_{false} {}
    };
  } // namespace shave_lib
} // namespace nn

#endif //SW_NN_RUNTIME_TYPES_H_3600__
