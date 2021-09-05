/*
* {% copyright %}
*/
#pragma once

#include "sw_layer_params.h"
#include "sw_tensor_ref.h"

namespace nn
{
namespace shave_lib
{

struct NN_CACHE_ALIGNED SoftParams
{
    SoftParams() = default;
    ~SoftParams();

    SoftParams(const SoftParams &) = delete;
    SoftParams &operator=(const SoftParams &) = delete;

    SoftParams(SoftParams &&) = default;
    SoftParams &operator=(SoftParams &&) = default;

    memory::cache_aligned_vector<TensorRef> inputs;
    memory::cache_aligned_vector<TensorRef> outputs;

    const LayerParams *layerParams{};
    uint32_t kParamBuffSize{};
    uint32_t paramsID{};
};

class ShaveResourceManager;

typedef void (*preamble)(const LayerParams *, ShaveResourceManager *);
typedef void (*shaveKernelEntry)(void *);
typedef void (*execCleanup)(const LayerParams *, ShaveResourceManager *);
typedef void (*layerCleanup)(const LayerParams *);

typedef void preambleImpl(const LayerParams *, ShaveResourceManager *);
typedef void shaveKernelEntryImpl(void *);
typedef void execCleanupImpl(const LayerParams *, ShaveResourceManager *);
typedef void layerCleanupImpl(const LayerParams *);

struct SoftKernel
{
    SoftKernel() = default;

    SoftKernel(const SoftKernel &) = delete;
    SoftKernel &operator=(const SoftKernel &) = delete;

    SoftKernel(SoftKernel &&other) = default;
    SoftKernel &operator=(SoftKernel &&other) = default;

    void allocCodeSpace(uint32_t codeSize);

    // Base of .text section
    // There may be multiple functions/entry points inside this region
    // This area will be in DDR and 1KB aligned
    void *codeBaseAddress{};
    // Size of all .text sections
    uint32_t codeSize{};

    // Base of .data section. This section should be copied into CMX
    void *dataBaseAddress{};
    // Size of all .data sections
    uint32_t dataSize{};

    // Windowed address that will be executed
    shaveKernelEntry kernelEntry{};

private:
    memory::shared_unique_ptr<uint8_t> unalignedCodeBuffer;
};

#if defined(CONFIG_TARGET_SOC_MA2490) || defined(CONFIG_TARGET_SOC_MA2490_B0) || defined(CONFIG_TARGET_SOC_3100)

/**
 * This class encapsulates all data necessary to execute a NN shave software
 * layer from an Inference Runtime context. This includes:
 *   - The leon preamble
 *   - All tensor IO descriptors
 *   - The target shave kernel binary
 *   - (hopefully) The runtime negotiated shave resource allocation
 */
struct alignas(64) Layer
{
    // Windowed address of the preamble
    preamble pre { nullptr };

    // Windowed address of the kernel function
    shaveKernelEntry kernelEntry { nullptr };

    SoftParams params;

    unsigned char maxShaves { 1 };
    bool isMultistage = false;

    execCleanup exeClean {nullptr};
    layerCleanup lyrClean {nullptr};

    Layer() = default;
    ~Layer();

    Layer(const Layer &) = delete;
    Layer &operator=(const Layer &) = delete;

    Layer(Layer &&) = default;
    Layer &operator=(Layer &&) = default;

    memory::cache_aligned_vector<TensorRef> &getInputs();
    memory::cache_aligned_vector<TensorRef> &getOutputs();

    void setParams(unsigned int paramID, LayerParams *lp);
    void setPreamble(preamble pre = nullptr);
    void setKernelEntry(shaveKernelEntry kernelEntry);
    void setExecCleanup(execCleanup cleanup = nullptr);
    void setLayerCleanup(layerCleanup cleanup = nullptr);
};
#else
struct Layer;
#endif  //  defined(CONFIG_TARGET_SOC_MA2490) || defined(CONFIG_TARGET_SOC_MA2490_B0) || defined(CONFIG_TARGET_SOC_3100)

} // namespace shave_lib
} // namespace nn
