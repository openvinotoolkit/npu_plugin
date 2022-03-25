//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "CustomCpp.h"
#include <mv_types.h>
#include <nn_log.h>

#include "shave_task_runner.hpp"

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
#include <dma_shave_nn.h>
void preCustomLayerCpp(const LayerParams *params, ShaveResourceManager *resMgr);
#else
#include <sw_nn_runtime_types_2490.h>
#include "svuSLKernels_EP.h"
#endif

namespace {
//  FIXME: Temporarily are located on CMX due to problem of ACT_SHAVE cache invalidation
OpTensor cmxInputs[shave_lib::MAX_INPUT_TENSORS] __attribute__((section(".nncmx0.shared.data")));
OpTensor cmxOutputs[shave_lib::MAX_OUTPUT_TENSORS] __attribute__((section(".nncmx0.shared.data")));

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

    sw_params::MemRefData* inTensors =
            reinterpret_cast<sw_params::MemRefData*>(reinterpret_cast<uint8_t*>(ops.paramData) + kernelParams->inputsOffset);
    for (unsigned i = 0; i < inputVec.size(); i++) {
        cmxInputs[i] = inputVec[i];
        inTensors[i] = cmxInputs[i].toMemRefData(inputLocations[i], true);
        nn::cache::flush(inTensors[i]);
        uint32_t * dims = reinterpret_cast<uint32_t*>(inTensors[i].dimsAddr);
        nn::cache::flush(dims, inTensors[i].numDims * sizeof(uint32_t));
        nn::cache::flush(cmxInputs, shave_lib::MAX_INPUT_TENSORS * sizeof(OpTensor));
    }
    sw_params::MemRefData* outTensors =
            reinterpret_cast<sw_params::MemRefData*>(reinterpret_cast<uint8_t*>(ops.paramData) + kernelParams->outputsOffset);
    for (unsigned i = 0; i < outputVec.size(); i++) {
        cmxOutputs[i] = outputVec[i];
        outTensors[i] = cmxOutputs[i].toMemRefData(outputLocations[i], false);
        nn::cache::flush(outTensors[i]);
        uint32_t * dims = reinterpret_cast<uint32_t*>(outTensors[i].dimsAddr);
        nn::cache::flush(dims, outTensors[i].numDims * sizeof(uint32_t));
        nn::cache::flush(cmxOutputs, shave_lib::MAX_OUTPUT_TENSORS * sizeof(OpTensor));
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

    logD("KernelDescriptor: descriptor.entry=%p descriptor.sec_mem_total=%u\n",
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

    logD("inputs %lu outputs %lu kernel selected %x",
         inRefs.size(), outRefs.size(), params->kernelOffset);

    params->kernel = ops.kernel;

    cache::flush(params, sizeof(CustomLayerCppParams));
    unsigned int id = opType;

    layer->setParams(id,
                     static_cast<LayerParams *>(params));

#ifdef CONFIG_TARGET_SOC_MA2490
    layer->setPreamble(PREAMBLE_FUNC(preCustomLayerCpp));
//    layer->setKernelEntry(KERNEL_FUNC(custom_cpp));
#endif

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

    ShaveTaskRunner runner;
    mvTensorAssert(runner.enqueTask(this, std::move(inputVec), std::move(outputVec), myriadRes.lastShave - myriadRes.firstShave + 1, &perfData), "custom OpenCPP layer run failed");
    mvTensorAssert(runner.dequeResult(), "custom Cpp layer run failed");

#ifdef CONFIG_TARGET_SOC_3720
    sw_params::BaseKernelParams * kernelParams = &(ops.baseParamData);
    sw_params::MemRefData* outTensors =
            reinterpret_cast<sw_params::MemRefData*>(reinterpret_cast<uint8_t*>(ops.paramData) + kernelParams->outputsOffset);
    for (unsigned i = 0; i < outputVec.size(); i++) {
        if (outTensors[i].location == sw_params::Location::NN_CMX || outTensors[i].location == sw_params::Location::UPA_CMX) {
            DmaAlShave dmaTask;
            auto totalBytes = (outputVec[i].ndims > 0) ? outputVec[i].dims[outputVec[i].ndims - 1] * outputVec[i].strides[outputVec[i].ndims - 1] : 0;
            dmaTask.start(reinterpret_cast<uint8_t*>(outTensors[i].dataAddr), reinterpret_cast<uint8_t*>(outputVec[i].addr),
                    totalBytes);
            dmaTask.wait();
        }
    }
#endif
}
