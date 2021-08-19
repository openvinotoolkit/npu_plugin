/*
* {% copyright %}
*/
#include "layers/parser_custom_cpp.h"
#include "ShaveElfMetadata/ShaveElfMetadataParser.h"
#include "layers/param_custom_cpp.h"
#include "layers/pre_custom_cpp.h"
#include "tensor_gf_util.h"

#define MODULE_NAME "[PARSER CUSTOM_CPP] "
#include "custom_common.h"

#include <mvMacros.h>
#include <nn_cache.h>
#include <nn_memory.h>
#include <sw_layer.h>

#include <cstring>
#include <stdio.h>

#include <sw_nn_runtime_types_3600.h>

extern void*  (shvNN0_preSingleSoftmax);
extern void*  (shvNN0_preCustomLayerCpp);
extern void*  (shvNN0_custom_cpp);
extern void*  (shvNN0_singleSoftmaxKernel);

namespace nn {
namespace shave_lib {

static bool parseCustomElf(const uint8_t* ElfBuffer, KernelCppDescriptor& descriptor, uint32_t argsNum)
{
    descriptor.kernelEntry = 0x1e000000; // default entry point
    descriptor.argumentsSize = argsNum * sizeof(u32);

    if (ElfBuffer != nullptr) {
        // Retrieve the .data section to get the size of local mem to reserve
        const Elf32_Shdr* dataSection = get_elf_section_with_name(ElfBuffer, ".dyn.data");
        const uint32_t dataSectionSize = dataSection ? dataSection->sh_size : 0;
        logI("Local data section size %u\n", dataSectionSize);

        descriptor.sec_mem_total = dataSectionSize;

        // Retrieve the .rodata section to get the size of local mem to reserve
        const Elf32_Shdr* rodataSection = get_elf_section_with_name(ElfBuffer, ".dyn.rodata");
        const uint32_t rodataSectionSize = rodataSection ? rodataSection->sh_size : 0;
        logI("Read-only data section size %u\n", rodataSectionSize);
        UNUSED(rodataSectionSize);
    }

    return true;
}

static void layerCleanupCustomCppLayer(const LayerParams *params) {
    auto custom_params = static_cast<const CustomLayerCppParams *>(params);

    nn::memory::shared_free(custom_params->argBuffer);
    auto kernelBuffer = reinterpret_cast<void *>(custom_params->kernelBuffer);
    if (kernelBuffer != nullptr) {
        nn::memory::shared_free(kernelBuffer);
    }
}

bool CustomLayerCppParser::parse(const MVCNN::UPALayerTask *task, Layer *layer) {
    if (task->softLayerParams_type() != MVCNN::SoftwareLayerParams::SoftwareLayerParams_CustomLayerCppParams) {
        return false;
    }

    // serialized params from blob
    const MVCNN::CustomLayerCppParams *gfParams = task->softLayerParams_as_CustomLayerCppParams();

    const uint8_t *elf = gfParams->kernelData()->data()->Data();
    uint32_t elfSize = gfParams->kernelData()->length();
    nnLog(MVLOG_DEBUG, "elf %p, elfSize %u\n", elf, elfSize);
    if (elfSize == 0) {
        elf = nullptr;
    }

    const uint32_t* argCountPtr = (uint32_t*)gfParams->paramData()->data()->Data();
    const uint32_t *arguments = (uint32_t *)(argCountPtr + 1);

    // parse kernel binary
    KernelCppDescriptor descriptor = {};
    RETURN_FALSE_UNLESS(parseCustomElf(elf, descriptor, *argCountPtr),
                        "Failed to parse kernel elf");

    logI("KernelDescriptor: descriptor.entry=%p descriptor.sec_mem_total=%u\n",
                (void*)descriptor.kernelEntry, descriptor.sec_mem_total);

    void * code = nullptr;
    if (elfSize > 0) {
        code = nn::memory::shared_alloc(1024, elfSize);
        nn::cache::invalidate(code, elfSize);

        RETURN_FALSE_UNLESS(loadElf(elf, code), "Unable to relocate kernel elf");
        nn::cache::flush(code, elfSize);
    }

    // ToDo: store argument info for future pointer relocation in preamble
    uint32_t* copyArgs = (uint32_t*)nn::memory::shared_alloc(descriptor.argumentsSize);
    memcpy_s(copyArgs, descriptor.argumentsSize, arguments, descriptor.argumentsSize);
    nn::cache::flush(copyArgs, descriptor.argumentsSize);

    // fill in programmed for execution config
    CustomLayerCppParams *params = new CustomLayerCppParams();

    params->kernelBuffer = (uint32_t)code;
    params->kernelOffset = descriptor.kernelEntry;

    params->argBufferSize = descriptor.argumentsSize;
    params->argBuffer = copyArgs;

    params->localSecMemTotal = descriptor.sec_mem_total;

    params->layerRequiresCacheFlushOnCompletion = true;

    auto inRefs = parseInputs(task->inputs(), layer);
    auto outRefs = parseOutputs(task->outputs(), layer);

    logI("inputs %lu outputs %lu kernel selected %x",
         inRefs.size(), outRefs.size(), params->kernelOffset);

    params->inputsSize = inRefs.size();
    params->outputsSize = outRefs.size();
    unsigned int id = MVCNN::SoftwareLayerParams::SoftwareLayerParams_CustomLayerCppParams;
//    layer->setParams(id, static_cast<LayerParams *>(postOpsParams.release()));

    cache::flush(params, sizeof(CustomLayerCppParams));
    layer->setParams(id/*getParamID(MVCNN::SoftwareLayerParams::SoftwareLayerParams_CustomLayerCppParams)*/,
                     static_cast<LayerParams *>(params));
    switch (copyArgs[1]) {
    case SOFTMAX:
        // in the future should be set to point to the code from blob
        // or to special preamble wrapper which will call the code from blob
        layer->setPreamble(reinterpret_cast<preamble>(&shvNN0_preSingleSoftmax));
        layer->setKernelEntry(reinterpret_cast<void (*)(void*)>(&shvNN0_singleSoftmaxKernel));
//        layer->setPreamble(PREAMBLE_FUNC(preSingleSoftmax));
        break;
    default:
//        layer->setPreamble(PREAMBLE_FUNC(preCustomLayerCpp));
        layer->setPreamble(reinterpret_cast<preamble>(&shvNN0_preCustomLayerCpp));
        layer->setKernelEntry(reinterpret_cast<void (*)(void*)>(&shvNN0_custom_cpp));
//        layer->setKernelEntry(KERNEL_FUNC(custom_cpp));
//        convParams->layerRequiresCacheFlushOnCompletion = true;
//        layer->requireCacheFlushOnCompletion();
        break;
    }
//    layer->setExecCleanup(PREAMBLE_FUNC(execCleanupCustomLayerCpp));
//    layer->setLayerCleanup(&layerCleanupCustomCppLayer);

    return true;
}
} // namespace shave_lib
} // namespace nn
