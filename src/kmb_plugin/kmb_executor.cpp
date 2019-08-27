//
// Copyright 2019 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <mutex>
#include <map>
#include <algorithm>
#include <utility>
#include <cstring>

#include <fcntl.h>
#include <sys/stat.h>
#include <chrono>
#include <stdio.h>
#include <unistd.h>

#include <ie_common.h>
#include <thread>

#include <vpu/kmb_plugin_config.hpp>
#include <vpu/utils/extra.hpp>
#include <vpu/utils/logger.hpp>

#include "kmb_executor.h"
#include "kmb_config.h"

#include "kmb_vpusmm_allocator.h"
#include "kmb_udma_allocator.h"
#include "kmb_native_allocator.h"

#ifndef _WIN32
# include <libgen.h>
# include <dlfcn.h>
#endif

using namespace vpu::KmbPlugin;
using namespace InferenceEngine;
using namespace InferenceEngine::VPUConfigParams;
using namespace std;

const uint32_t POOL_SIZE = 30 * 1024 * 1024;
// XLink channel number to start allocation from
const uint32_t IE_VPU_KMB_XC_DEFAULT = 3;


// Get free XLink channel
static uint32_t getXlinkChannel(void) {
    static std::mutex mutex_;
    static int XlinkChannel = -1;

    uint32_t ret;
    std::unique_lock<std::mutex> lock(mutex_);

    if (XlinkChannel <= 0) {
        const char * pxc = getenv("IE_VPU_KMB_XC");
        XlinkChannel = pxc ? atoi(pxc):IE_VPU_KMB_XC_DEFAULT;
    }
    // In this simplified implementation we never reuse the cannel
    ret = XlinkChannel++;
    // Skipping "0xA: IP control channel (standard channel)"
    if (ret == 10) {
        ret = XlinkChannel++;
    }
    std::cout << "Allocated channel = " << ret << std::endl;
    return ret;
}

KmbExecutor::KmbExecutor(const Logger::Ptr& log, const std::shared_ptr<KmbConfig>& config)
            : _log(log), _config(config)  {
    auto parsedConfig = _config->getParsedConfig();
    if (parsedConfig[VPU_KMB_CONFIG_KEY(KMB_EXECUTOR)] == "NO") {
        return;
    }
    const char *allocatorEnvPtr = std::getenv("IE_VPU_KMB_MEMORY_ALLOCATOR_TYPE");
    std::string allocatorType = "";
    if (allocatorEnvPtr) {
        allocatorType = allocatorEnvPtr;
    }

    if (allocatorType == "UDMA") {
        allocator = make_shared<KmbUdmaAllocator>();
    } else if (allocatorType == "NATIVE") {
        allocator = make_shared<KmbNativeAllocator>();
    } else {
        allocator = make_shared<KmbVpusmmAllocator>();
    }
#ifdef ENABLE_VPUAL
    blob_file = nullptr;
    rgnAllocatorBuffer = nullptr;
#endif
}

void KmbExecutor::initVpualObjects() {
#ifdef ENABLE_VPUAL
    if (!RgnAlloc) {
        RgnAlloc  = make_shared<RgnAllocator>();
    }
    if (!nnPl) {
        nnPl = make_shared<NNFlicPlg>();
    }
    if (!gg) {
        gg = make_shared<GraphManagerPlg>();
    }
    if (!plgTensorInput_) {
        plgTensorInput_ = make_shared<PlgTensorSource>();
    }
    if (!plgTensorOutput_) {
        plgTensorOutput_ = make_shared<PlgStreamResult>();
    }
    if (!plgPoolOutputs) {
        plgPoolOutputs = make_shared<PlgPool<TensorMsg>>();
    }
    if (!BHandle) {
        BHandle = make_shared<BlobHandle_t>();
    }
    if (!pipe) {
        pipe = make_shared<Pipeline>();
    }
#endif
}

void KmbExecutor::allocateGraph(const std::vector<char> &graphFileContent, const char* networkName) {
    UNUSED(networkName);
    auto parsedConfig = _config->getParsedConfig();
    if (parsedConfig[VPU_KMB_CONFIG_KEY(KMB_EXECUTOR)] == "NO") {
        return;
    }

#ifdef ENABLE_VPUAL
    initVpualObjects();
    static int graphId_main = 1;
    int nThreads = 4;
    int nShaves = 16;

    std::cout << "Initiating verification of use case 1" << std::endl;

    BHandle->graphid = graphId_main++;
    BHandle->graphBuff = 0x00000000;
    BHandle->graphLen = graphFileContent.size();
    BHandle->refCount = 0;

    // ########################################################################
    // Try and get some CMA allocations.
    // ########################################################################
    blob_file = allocator->alloc(graphFileContent.size());

    if (!blob_file) {
        std::cout << "KmbExecutor::allocateGraph: Error getting CMA for graph" << std::endl;
        return;
    }

    // ########################################################################
    // Load the input files
    // ########################################################################

    std::memcpy(blob_file, graphFileContent.data(), graphFileContent.size());
    // Point Blob Handle to the newly loaded graph file. Only allow 32-bit

    // Assigning physical address of Blob file

    BHandle->graphBuff = allocator->getPhysicalAddress(blob_file);  // Only lower 32-bits

    gg->Create();

    GraphStatus status = gg->NNGraphCheckAvailable(BHandle->graphid);
    if (Success == status) {
        std::cout << "Blob available!" << std::endl;
        status = gg->NNGraphAllocateExistingBlob(BHandle.get());
        std::cout << "Allocated existing blob with status: " << status << std::endl;
    } else if (No_GraphId_Found == status) {
        std::cout << "Blob not found." << std::endl;
        status = gg->NNGraphAllocate(BHandle.get());
        std::cout << "Allocated new blob with id " << BHandle->graphid << "  with status: " << status << std::endl;
    } else {
        std::cerr << "Error checking graph availability: " << status << std::endl;
        // TODO: error
    }

    // Plugins:


    // Pool plugins (to allocate memory for the plugins which require some):


    std::cout << "Instantiated Plugins..." << std::endl;

    // FLIC Pipeline:

    // Setting number of threads for NNPlugin

    nnPl->SetNumberOfThreads(nThreads);
    nnPl->SetNumberOfShaves(nShaves);

    nnPl->Create(BHandle.get());

    std::cout << "NN Plugin Create finished..." << std::endl;

    NNPlgState state = nnPl->GetLatestState();
    if (SUCCESS != state) {
        std::cerr << "Error, bad NN Plugin state: " << state << std::endl;
        return;
    }

    auto tensor_deserializer = [](const flicTensorDescriptor_t & descriptor)->void {
        std::cout << "{";
        std::cout << "n: " << descriptor.n << ", ";
        std::cout << "c: " << descriptor.c << ", ";
        std::cout << "h: " << descriptor.h << ", ";
        std::cout << "w: " << descriptor.w << ", ";
        std::cout << "totalSize: " << descriptor.totalSize << ", ";
        std::cout << "widthStride: " << descriptor.widthStride << ", ";
        std::cout << "heightStride: " << descriptor.heightStride << ", ";
        std::cout << "channelsStride: " << descriptor.channelsStride << "}" << std::endl;
    };

    flicTensorDescriptor_t descOut = nnPl->GetOutputTensorDescriptor(0);
    flicTensorDescriptor_t  descIn = nnPl->GetInputTensorDescriptor(0);
    std::cout << "Deserializing descriptors:" << std::endl;
    std::cout << "Input: ";
    tensor_deserializer(descIn);
    std::cout << "Output: ";
    tensor_deserializer(descOut);

    InferenceEngine::SizeVector inputDims({descIn.n, descIn.c, descIn.h, descIn.w});
    InferenceEngine::Layout inputLayout = InferenceEngine::Layout::NCHW;
    // TODO: add proper precision handling
    InferenceEngine::Precision inputPrecision = InferenceEngine::Precision::U8;
    InferenceEngine::TensorDesc inputDesc(inputPrecision, inputDims, inputLayout);
    InferenceEngine::Data inputData("input", inputDesc);

    InferenceEngine::InputInfo inputInfo;
    inputInfo.setInputData(std::make_shared<InferenceEngine::Data>(inputData));
    m_networkInputs[inputInfo.name()] = std::make_shared<InferenceEngine::InputInfo>(inputInfo);

    InferenceEngine::SizeVector outputDims({descOut.n, descOut.c, descOut.h, descOut.w});
    InferenceEngine::Layout outputLayout = InferenceEngine::Layout::NCHW;
    InferenceEngine::Precision outputPrecision = InferenceEngine::Precision::U8;
    InferenceEngine::TensorDesc outputDesc(outputPrecision, outputDims, outputLayout);
    InferenceEngine::Data outputData("output", outputDesc);

    m_networkOutputs[outputData.getName()] = std::make_shared<InferenceEngine::Data>(outputData);

    rgnAllocatorBuffer = allocator->alloc(POOL_SIZE);
    if (!rgnAllocatorBuffer) {
        std::cout << "KmbExecutor::allocateGraph: Cannot allocate buffer for RgnAlloc" << std::endl;
        return;
    }
    RgnAlloc->Create(allocator->getPhysicalAddress(rgnAllocatorBuffer), POOL_SIZE);
    std::cout << "KmbExecutor::allocateGraph: Created RgnAlloc" << std::endl;

    // TODO - These
    const unsigned int shavel2CacheLineSize = 64;
    unsigned int outputTensorSize = ROUND_UP(descOut.totalSize, shavel2CacheLineSize);

    // TODO - These
    std::cout << "read memory pool finished..." << std::endl;
    plgPoolOutputs->Create(RgnAlloc.get(), 1, 3 * outputTensorSize);
    std::cout << "Created plgPoolOutputs" << std::endl;

    xlinkChannelIn = getXlinkChannel();
    xlinkChannelOut = getXlinkChannel();
    plgTensorInput_->Create(descIn.totalSize, xlinkChannelIn, descIn);
    std::cout << "Created plgTensorInput" << std::endl;

    plgTensorOutput_->Create(descOut.totalSize, xlinkChannelOut, descOut);
    std::cout << "reated plgTensorOutput" << std::endl;

    std::cout << "Created all Plugins" << std::endl;

    // Add the plugins to the pipeline:
    pipe->Add(plgPoolOutputs.get());
    pipe->Add(plgTensorInput_.get());
    pipe->Add(plgTensorOutput_.get());
    pipe->Add(nnPl.get());

    std::cout << "Added Plugins to Pipeline" << std::endl;

    // Link the plugins' messages:
    plgPoolOutputs->out.Link(&nnPl->resultInput);
    plgTensorInput_->tensorOut.Link(&nnPl->tensorInput);
    nnPl->output.Link(&plgTensorOutput_->dataIn);

    std::cout << "Linked Plugins..." << std::endl;

    pipe->Start();
    std::cout << "Started FLIC pipeline..." << std::endl;
#else
    UNUSED(graphFileContent);
#endif
}


void KmbExecutor::queueInference(void *input_data, size_t input_bytes,
                    void *result_data, size_t result_bytes) {
    UNUSED(result_data);
    UNUSED(result_bytes);
    auto parsedConfig = _config->getParsedConfig();
    if (parsedConfig[VPU_KMB_CONFIG_KEY(KMB_EXECUTOR)] == "NO") {
        return;
    }

#ifdef ENABLE_VPUAL
    auto physAddr = allocator->getPhysicalAddress(input_data);
    plgTensorInput_->Push(physAddr, input_bytes);
    std::cout << "Pushed input, size " << input_bytes << std::endl;
#else
    UNUSED(input_data);
    UNUSED(input_bytes);
#endif
}

void KmbExecutor::getResult(void *result_data, unsigned int result_bytes) {
    UNUSED(result_data);
    UNUSED(result_bytes);
    auto parsedConfig = _config->getParsedConfig();
    if (parsedConfig[VPU_KMB_CONFIG_KEY(KMB_EXECUTOR)] == "NO") {
        return;
    }

#ifdef ENABLE_VPUAL
    uint32_t len = 0;
    uint32_t pAddr = 0;
    plgTensorOutput_->Pull(&pAddr, &len);

    std::cout << "Output tensor returned of length: " << std::dec << len << std::endl;

    // Convert the physical address we received back to a virtual address we can use.
    uint32_t offset = pAddr - allocator->getPhysicalAddress(rgnAllocatorBuffer);
    unsigned char *data = static_cast<unsigned char *>(rgnAllocatorBuffer) + offset;

    uint32_t checksum = 0;
    for (uint32_t k = 0; k < len; k++) checksum += data[k];

    std::cout << "KmbExecutor::getResult memcpy started @" << offset
              << " checksum=" << checksum
              << " xlinkChannel=" << xlinkChannelIn
              << "," << xlinkChannelOut << std::endl;

    std::memcpy(result_data, data, len);
    std::memset(data, 0, len);
    std::cout << "KmbExecutor::getResult memcpy finished" << std::endl;
#endif
}

void KmbExecutor::deallocateGraph() {
    auto parsedConfig = _config->getParsedConfig();
    if (parsedConfig[VPU_KMB_CONFIG_KEY(KMB_EXECUTOR)] == "NO") {
        return;
    }
#ifdef ENABLE_VPUAL
    if (pipe) {
        pipe->Stop();
        pipe->Delete();
    }
    if (nnPl) {
        nnPl->Delete();
    }
    if (gg) {
        gg->NNDeallocateGraph(BHandle->graphid);
    }
    if (plgTensorInput_) {
        plgTensorInput_->Delete();
    }
    if (plgTensorOutput_) {
        plgTensorOutput_->Delete();
    }
    if (plgPoolOutputs) {
        plgPoolOutputs->Delete();
    }
    if (RgnAlloc) {
        RgnAlloc->Delete();
    }
    if (blob_file) {
        allocator->free(blob_file);
    }
    if (rgnAllocatorBuffer) {
        allocator->free(rgnAllocatorBuffer);
    }
#endif
}

std::shared_ptr<KmbAllocator> KmbExecutor::getAllocator() {
    return allocator;
}

