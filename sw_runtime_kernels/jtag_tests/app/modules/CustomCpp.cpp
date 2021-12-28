// {% copyright %}

#include "CustomCpp.h"
#include <mv_types.h>
#include <nn_log.h>

#include "upa_task_runner.hpp"

#include "layers/param_custom_cpp.h"
#include "layers/pre_custom_cpp.h"

#include "custom_common.h"

#include <mvMacros.h>
#include <nn_cache.h>
#include <nn_memory.h>
#include <sw_layer.h>

#include <cstring>
#include <stdio.h>

using namespace mv::tensor;

CustomCpp::~CustomCpp() = default;
using namespace nn;
using namespace nn::shave_lib;

#include "ShaveElfMetadata/ShaveElfMetadataParser.h"
#ifdef CONFIG_TARGET_SOC_3720
#include <sw_nn_runtime_types_3600.h>
extern void*  (shvNN0_preCustomLayerCpp);
extern void*  (shvNN0_custom_cpp);
#else
#include <sw_nn_runtime_types_2490.h>
#include "svuSLKernels_EP.h"
#endif

namespace {
static bool parseCustomElf(const uint8_t* ElfBuffer, KernelCppDescriptor& descriptor/*, uint32_t argsNum*/)
{
    descriptor.kernelEntry = 0x1e000000; // default entry point
//    descriptor.argumentsSize = argsNum * sizeof(u32);

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

//    nn::memory::shared_free(custom_params->argBuffer);
    auto kernelBuffer = reinterpret_cast<void *>(custom_params->kernelBuffer);
    if (kernelBuffer != nullptr) {
        nn::memory::shared_free(kernelBuffer);
    }
}
}

bool CustomCpp::parse(Layer * layer) {
    std::vector<uint64_t> kernelData((uint64_t*)ops.kernelData, (uint64_t*)ops.kernelData+(ops.kernelDataLen+7)/8);

    sw_params::BaseKernelParams * kernelParams = &(ops.baseParamData);

    printf("!!!!!!!!!!!! %s:%d: ops.paramData: %p\n", __FILE__, __LINE__, ops.paramData);
    sw_params::MemRefData* inTensors =
            reinterpret_cast<sw_params::MemRefData*>(reinterpret_cast<uint8_t*>(ops.paramData) + kernelParams->inputsOffset);
    for (unsigned i = 0; i < inputVec.size(); i++) {
        inTensors[i] = inputVec[i].toMemRefData(inputLocations[i]);
        inTensors[i].location = inputLocations[i];
    }
    sw_params::MemRefData* outTensors =
            reinterpret_cast<sw_params::MemRefData*>(reinterpret_cast<uint8_t*>(ops.paramData) + kernelParams->outputsOffset);
    for (unsigned i = 0; i < outputVec.size(); i++) {
        outTensors[i] = outputVec[i].toMemRefData(outputLocations[i]);
        outTensors[i].location = outputLocations[i];
    }

    const uint8_t *elf = reinterpret_cast<const uint8_t *>(kernelData.data());
    uint32_t elfSize = ops.kernelDataLen;
    nnLog(MVLOG_DEBUG, "elf %p, elfSize %u\n", elf, elfSize);
    if (elfSize == 0) {
        elf = nullptr;
    }

    // parse kernel binary
    KernelCppDescriptor descriptor = {};
    RETURN_FALSE_UNLESS(parseCustomElf(elf, descriptor/*, *argCountPtr*/),
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
    nn::cache::flush(ops.paramData, ops.paramDataLen);
    nn::cache::flush(kernelParams, sizeof(sw_params::BaseKernelParams));

    // fill in programmed for execution config
    CustomLayerCppParams *params = new CustomLayerCppParams();

    params->kernelBuffer = (uint32_t)code;
    params->kernelOffset = descriptor.kernelEntry;
    params->moveToCmxIfNecessary = true;
    params->argBufferSize = ops.paramDataLen;
    params->argBuffer = ops.paramData;
    params->baseParamData = *kernelParams;

    params->localSecMemTotal = descriptor.sec_mem_total;

    params->layerRequiresCacheFlushOnCompletion = true;

    auto inRefs = layer->getInputs();
    auto outRefs = layer->getOutputs();

    logI("inputs %lu outputs %lu kernel selected %x",
         inRefs.size(), outRefs.size(), params->kernelOffset);

    params->kernel = ops.kernel;

    cache::flush(params, sizeof(CustomLayerCppParams));
    unsigned int id = opType;

    layer->setParams(id,
                     static_cast<LayerParams *>(params));

#ifdef CONFIG_TARGET_SOC_3720
    layer->setPreamble(reinterpret_cast<preamble>(&shvNN0_preCustomLayerCpp));
//    layer->setKernelEntry(reinterpret_cast<void (*)(void*)>(&shvNN0_custom_cpp));
#else
    layer->setPreamble(PREAMBLE_FUNC(preCustomLayerCpp));
//    layer->setKernelEntry(KERNEL_FUNC(custom_cpp));
#endif
//        convParams->layerRequiresCacheFlushOnCompletion = true;
//        layer->requireCacheFlushOnCompletion();
//    layer->setExecCleanup(PREAMBLE_FUNC(execCleanupCustomLayerCpp));
//    layer->setLayerCleanup(&layerCleanupCustomCppLayer);

    return true;
}

void CustomCpp::run(mv::tensor::Processor& ,
                  t_MvTensorMyriadResources& myriadRes,
                  t_MvTensorDebugInfo&) {
    for(const OpTensor& input: inputVec) {
        nnLog(MVLOG_DEBUG, "ndims  = %ld\n", input.ndims);
        nnLog(MVLOG_DEBUG, "order  = 0x%x\n", input.order);
        nnLog(MVLOG_DEBUG, "dims.0 = %ld(%ld)\n", input.dims[0], input.strides[0]);
        nnLog(MVLOG_DEBUG, "dims.1 = %ld(%ld)\n", input.dims[1], input.strides[1]);
        nnLog(MVLOG_DEBUG, "dims.2 = %ld(%ld)\n", input.dims[2], input.strides[2]);
        nnLog(MVLOG_DEBUG, "dims.3 = %ld(%ld)\n", input.dims[3], input.strides[3]);
        nnLog(MVLOG_DEBUG, "Input buffer %p.\n", input.addr);
    }
    for(const OpTensor& output: outputVec) {
        nnLog(MVLOG_DEBUG, "Output buffer %p.\n", output.addr);
    }
    nnLog(MVLOG_DEBUG, "leonPreambleID %ld\n", ops.leonPreambleID);
    nnLog(MVLOG_DEBUG, "KernelData %p with length %ld.\n", ops.kernelData, ops.kernelDataLen);
    nnLog(MVLOG_DEBUG, "ParamData  %p with length %ld.\n", ops.paramData,  ops.paramDataLen);

    UPATaskRunner runner;
    mvTensorAssert(runner.enqueTask(this, std::move(inputVec), std::move(outputVec), myriadRes.lastShave - myriadRes.firstShave + 1, &perfData), "custom OpenCPP layer run failed");
    mvTensorAssert(runner.dequeResult(), "custom Cpp layer run failed");
}
