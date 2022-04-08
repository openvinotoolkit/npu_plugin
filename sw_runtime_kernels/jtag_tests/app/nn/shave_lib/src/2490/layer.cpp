/*
* {% copyright %}
*/
#include "sw_layer.h"

#include "sw_layer_params.h"
#include "sw_tensor_ref.h"

#include <assert.h>
#include <nn_memory.h>
#include <nn_math.h>
#include <nn_cache.h>
#include <nn_relocation.h>

namespace nn {
namespace shave_lib {

using namespace nn::inference_runtime;

void SoftKernel::allocCodeSpace(uint32_t size) {
    assert(!unalignedCodeBuffer);

    // MMU alignment requirement on dKMB to 4K block size
    constexpr auto codeAlignment = 4096;
    codeSize = math::round_up<codeAlignment>(size);

    // DDRMemMgr serving the shared area does not support custom alignment
    unalignedCodeBuffer.reset(reinterpret_cast<uint8_t *>(memory::shared_alloc(codeAlignment + codeSize)));
    codeBaseAddress = math::ptr_align_up<codeAlignment>(unalignedCodeBuffer.get());

    // clear the padding to avoid leaking sensitive data
    memset(unalignedCodeBuffer.get() + size, 0, codeSize - size);
    cache::flush(math::ptr_align_down<NN_CACHE_LINE_LENGTH>(unalignedCodeBuffer.get() + size), math::round_up<NN_CACHE_LINE_LENGTH>(codeSize - size));
}

SoftParams::~SoftParams() { delete layerParams; }

Layer::~Layer() {
    if (lyrClean)
        (lyrClean)(params.layerParams);
}

memory::cache_aligned_vector<TensorRef> &Layer::getInputs() { return params.inputs; }

memory::cache_aligned_vector<TensorRef> &Layer::getOutputs() { return params.outputs; }

void Layer::setParams(unsigned int paramID, LayerParams *lp) {
    this->params.paramsID = paramID;
    this->params.layerParams = lp;
    // TODO: compute paramCRC
}

void Layer::setPreamble(preamble pre) {
    this->pre = pre;
}

void Layer::setKernelEntry(shaveKernelEntry kernelEntry) {
    this->kernelEntry = kernelEntry;
}

void Layer::setExecCleanup(execCleanup cleanup) { this->exeClean = cleanup; }

void Layer::setLayerCleanup(layerCleanup cleanup) { this->lyrClean = cleanup; }

} // namespace shave_lib
} // namespace nn
