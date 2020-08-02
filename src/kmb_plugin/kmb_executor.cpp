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

#include "kmb_executor.h"

#include <fcntl.h>
#include <ie_common.h>
#include <ie_memcpy.h>
#include <stdio.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <utility>
#include <vector>
#include <vpu/utils/extra.hpp>
#include <vpu/utils/logger.hpp>

#include "ie_macro.hpp"
#include "kmb_allocator.h"
#include "kmb_config.h"

#ifndef _WIN32
#include <dlfcn.h>
#include <libgen.h>

#include <ie_profiling.hpp>

#endif

using namespace vpu::KmbPlugin;
using namespace InferenceEngine;
using namespace InferenceEngine::VPUConfigParams;
using namespace std;

#if defined(__arm__) || defined(__aarch64__)
const uint32_t POOL_SIZE = 30 * 1024 * 1024;
#endif

KmbExecutor::KmbExecutor(const vpux::NetworkDescription::Ptr& networkDescription, const KmbAllocator::Ptr& allocator,
    const KmbConfig& config)
    : _networkDescription(networkDescription),
      _allocator(allocator),
      _config(config),
      _logger(std::make_shared<Logger>("KmbExecutor", config.logLevel(), consoleOutput())) {
    if (!_config.useKmbExecutor()) {
        return;
    }

#if defined(__arm__) || defined(__aarch64__)
    blob_file = nullptr;
    rgnAllocatorBuffer = nullptr;
    _inferenceVirtAddr = nullptr;
    initVpualObjects();
    allocateGraph(_networkDescription->getCompiledNetwork());
#endif
}

KmbExecutor::~KmbExecutor() { deallocateGraph(); }

void KmbExecutor::initVpualObjects() {
#if defined(__arm__) || defined(__aarch64__)
    IE_PROFILING_AUTO_SCOPE(initVpualObjects);
    if (!RgnAlloc) {
        RgnAlloc = make_shared<RgnAllocator>();
    }
    if (!HeapAlloc) {
        HeapAlloc = make_shared<HeapAllocator>();
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
    if (!plgInferenceInput_) {
        plgInferenceInput_ = make_shared<PlgInferenceInput>();
    }
    if (!plgInferenceOutput_) {
        plgInferenceOutput_ = make_shared<PlgInferenceOutput>();
    }
    if (!plgPoolOutputs) {
        plgPoolOutputs = make_shared<PlgPool<TensorMsg>>();
    }
    if (!plgPoolInferenceMsg) {
        plgPoolInferenceMsg = make_shared<PlgPool<InferenceMsg>>();
    }
    if (!BHandle) {
        BHandle = make_shared<BlobHandle_t>();
    }
    if (!pipe) {
        pipe = make_shared<Pipeline>();
    }
    if (!_inferenceVirtAddr) {
        _inferenceVirtAddr = reinterpret_cast<uint32_t*>(_allocator->alloc(sizeof(uint32_t)));
    }
#endif
}

#if defined(__arm__) || defined(__aarch64__)
namespace {

/*
 * Wrapper to SetScratchBuffer
 * 1. Get required memory amount
 * 2. Make sure it's not less than 1 MB (mentioned in [Track number: h#18011677038])
 * 3. Allocate buffer for each NN thread and append physical addresses to collection
 * 4. Give result to SetScratchBuffer
 * 5. Track allocated chunks by virtual addresses to free them properly
 */
static std::vector<void*> setScratchHelper(const std::shared_ptr<NNFlicPlg>& nnFlicPtr, const unsigned int threadCount,
    const std::shared_ptr<KmbAllocator>& allocatorPtr, const std::shared_ptr<vpu::Logger>& logger) {
    if (threadCount > 1) {
        logger->warning("scratchHelper: trying to set scratch buffer to %u threads.", threadCount);
    }
    uint32_t memoryReqs = nnFlicPtr->GetMemoryRequirements();
    logger->info("scratchHelper: GetMemoryRequirements returned %u", memoryReqs);
    constexpr uint32_t minimalScratchSize = 1024 * 1024;
    if (memoryReqs < minimalScratchSize) {
        memoryReqs = minimalScratchSize;
    }

    std::vector<void*> virtAddrVec;
    virtAddrVec.reserve(threadCount);
    std::vector<uint32_t> physAddrVec;
    physAddrVec.reserve(threadCount);
    for (unsigned int threadIdx = 0; threadIdx < threadCount; threadIdx++) {
        uint8_t* scratchVirtAddr = reinterpret_cast<uint8_t*>(allocatorPtr->alloc(memoryReqs));
        if (scratchVirtAddr == nullptr) {
            THROW_IE_EXCEPTION << "scratchHelper: failed to allocate " << memoryReqs << " bytes of memory";
        }
        unsigned long scratchPhysAddr = allocatorPtr->getPhysicalAddress(scratchVirtAddr);
        if (scratchPhysAddr == 0) {
            THROW_IE_EXCEPTION << "scratchHelper: failed to get physical address";
        }
        // NB: narrowing unsigned long (uint64_t on 64-bit Yocto) to uint32_t here
        physAddrVec.push_back(scratchPhysAddr);
        virtAddrVec.push_back(scratchVirtAddr);
    }

    nnFlicPtr->SetScratchBuffer(physAddrVec);
    return virtAddrVec;
}
}  // namespace
#endif

void KmbExecutor::allocateGraph(const std::vector<char>& graphFileContent) {
    if (!_config.useKmbExecutor()) {
        return;
    }

#if defined(__arm__) || defined(__aarch64__)
    IE_PROFILING_AUTO_SCOPE(allocateGraph);
    initVpualObjects();
    static int graphId_main = 1;
    int nThreads = _config.throghputStreams();
    int nShaves = 16;

    _logger->info("KmbExecutor::allocateGraph begins");

    BHandle->graphid = graphId_main++;
    BHandle->graphBuff = 0x00000000;
    BHandle->graphLen = graphFileContent.size();
    BHandle->refCount = 0;

    // ########################################################################
    // Try and get some CMA allocations.
    // ########################################################################
    blob_file = _allocator->alloc(BHandle->graphLen);

    if (!blob_file) {
        _logger->error("KmbExecutor::allocateGraph: Error getting CMA for graph");
        THROW_IE_EXCEPTION << "allocateGraph: allocation failed for graph";
    }

    // ########################################################################
    // Load the input files
    // ########################################################################

    std::memcpy(blob_file, graphFileContent.data(), graphFileContent.size());
    std::memset(
        static_cast<uint8_t*>(blob_file) + graphFileContent.size(), 0, BHandle->graphLen - graphFileContent.size());
    // Point Blob Handle to the newly loaded graph file. Only allow 32-bit

    // Assigning physical address of Blob file

    BHandle->graphBuff = _allocator->getPhysicalAddress(blob_file);  // Only lower 32-bits

    gg->Create();

    GraphStatus status = gg->NNGraphCheckAvailable(BHandle->graphid);
    if (Success == status) {
        _logger->info("Blob available!");
        status = gg->NNGraphAllocateExistingBlob(BHandle.get());
        _logger->info("Allocated existing blob with status: %d", status);
    } else if (No_GraphId_Found == status) {
        _logger->info("Blob not found.");
        status = gg->NNGraphAllocate(BHandle.get());
        _logger->info("Allocated new blob with id: %d; with status: %d", BHandle->graphid, status);
    } else {
        _logger->error("Error checking graph availability: %d", status);
        // TODO: error
    }

    // Plugins:

    // Pool plugins (to allocate memory for the plugins which require some):

    _logger->info("Instantiated Plugins...");

    // FLIC Pipeline:

    // Setting number of threads for NNPlugin

    nnPl->SetNumberOfThreads(nThreads);
    nnPl->SetNumberOfShaves(nShaves);

    nnPl->Create(BHandle.get());

    _scratchBuffers = setScratchHelper(nnPl, nThreads, getKmbAllocator(), _logger);

    _logger->info("NN Plugin Create finished...");

    NNPlgState state = nnPl->GetLatestState();
    if (SUCCESS != state) {
        _logger->error("Error, bad NN Plugin state: %d", state);
        THROW_IE_EXCEPTION << "allocateGraph: flic NN is in unexpected state: " << state;
    }

    auto tensor_deserializer = [&](const flicTensorDescriptor_t& descriptor) -> void {
        _logger->info(
            "{ n: %d, c: %d, h: %d, w: %d, totalSize: %d, widthStride: %d, heightStride: %d, channelsStride: %d}",
            descriptor.n, descriptor.c, descriptor.h, descriptor.w, descriptor.totalSize, descriptor.widthStride,
            descriptor.heightStride, descriptor.channelsStride);
    };

    _logger->info("Deserializing descriptors:");
    size_t inputsSize = nnPl->GetNumberOfInputs();
    flicTensorDescriptor_t sumSizeTensorDescIn;
    // tensor batch is not a proper 4D tensor anymore, but a 1D tensor with concatenated reshaped inputs
    // use width and total size to determine the size of the blob. other dimensions are just 1
    sumSizeTensorDescIn.n = 1;
    sumSizeTensorDescIn.c = 1;
    sumSizeTensorDescIn.h = 1;
    sumSizeTensorDescIn.w = 0;
    sumSizeTensorDescIn.totalSize = 0;
    sumSizeTensorDescIn.widthStride = 1;
    for (size_t inputIdx = 0; inputIdx < inputsSize; inputIdx++) {
        flicTensorDescriptor_t descIn = nnPl->GetInputTensorDescriptor(inputIdx);
        _logger->info("Input: %d", inputIdx);
        tensor_deserializer(descIn);

        sumSizeTensorDescIn.totalSize += descIn.totalSize;
    }
    sumSizeTensorDescIn.w = sumSizeTensorDescIn.totalSize;
    sumSizeTensorDescIn.heightStride = sumSizeTensorDescIn.totalSize;
    sumSizeTensorDescIn.channelsStride = sumSizeTensorDescIn.totalSize;

    size_t outputsSize = nnPl->GetNumberOfOutputs();
    flicTensorDescriptor_t sumSizeTensorDescOut;
    sumSizeTensorDescOut.n = 1;
    sumSizeTensorDescOut.c = 1;
    sumSizeTensorDescOut.h = 1;
    sumSizeTensorDescOut.w = 0;
    sumSizeTensorDescOut.totalSize = 0;
    sumSizeTensorDescOut.widthStride = 1;
    for (size_t outputIdx = 0; outputIdx < outputsSize; outputIdx++) {
        flicTensorDescriptor_t descOut = nnPl->GetOutputTensorDescriptor(outputIdx);
        _logger->info("Output: %d", outputIdx);
        tensor_deserializer(descOut);

        sumSizeTensorDescOut.totalSize += descOut.totalSize;
    }
    sumSizeTensorDescOut.w = sumSizeTensorDescOut.totalSize;
    sumSizeTensorDescOut.heightStride = sumSizeTensorDescOut.totalSize;
    sumSizeTensorDescOut.channelsStride = sumSizeTensorDescOut.totalSize;

    rgnAllocatorBuffer = _allocator->alloc(POOL_SIZE);
    if (!rgnAllocatorBuffer) {
        _logger->error("KmbExecutor::allocateGraph: Cannot allocate buffer for RgnAlloc");
        THROW_IE_EXCEPTION << "allocateGraph: allocation failed for region allocator";
    }
    RgnAlloc->Create(_allocator->getPhysicalAddress(rgnAllocatorBuffer), POOL_SIZE);
    _logger->info("KmbExecutor::allocateGraph: Created RgnAlloc");

    const unsigned int shavel2CacheLineSize = 64;
    unsigned int outputTensorSize = ROUND_UP(sumSizeTensorDescOut.totalSize, shavel2CacheLineSize);

    _logger->info("read memory pool finished...");
    plgPoolOutputs->Create(RgnAlloc.get(), 1, 3 * outputTensorSize);
    _logger->info("Created plgPoolOutputs");

    unsigned int inferenceIDSize = ROUND_UP(sizeof(uint32_t), shavel2CacheLineSize);
    plgPoolInferenceMsg->Create(HeapAlloc.get(), 1, 3 * inferenceIDSize);
    _logger->info("Created plgPoolInferenceMsg");

    plgTensorInput_->Create(sumSizeTensorDescIn.totalSize, xlinkChannel, sumSizeTensorDescIn);
    _logger->info("Created plgTensorInput");

    plgTensorOutput_->Create(sumSizeTensorDescOut.totalSize, xlinkChannel, sumSizeTensorDescOut);
    _logger->info("Created plgTensorOutput");

    plgInferenceInput_->Create(3 * inferenceIDSize, xlinkChannel);
    _logger->info("Created plgInferenceInput_");

    plgInferenceOutput_->Create(3 * inferenceIDSize, xlinkChannel);
    _logger->info("Created plgInferenceOutput_");

    _logger->info("Created all Plugins");

    // Add the plugins to the pipeline:
    pipe->Add(plgPoolOutputs.get());
    pipe->Add(plgTensorInput_.get());
    pipe->Add(plgTensorOutput_.get());
    pipe->Add(plgPoolInferenceMsg.get());
    pipe->Add(plgInferenceInput_.get());
    pipe->Add(plgInferenceOutput_.get());
    pipe->Add(nnPl.get());

    _logger->info("Added Plugins to Pipeline");

    // Link the plugins' messages:
    plgPoolOutputs->out.Link(&nnPl->resultInput);
    plgTensorInput_->tensorOut.Link(&nnPl->tensorInput);
    nnPl->output.Link(&plgTensorOutput_->dataIn);

    plgPoolInferenceMsg->out.Link(&nnPl->inferenceResult);
    plgInferenceInput_->inferenceOut.Link(&nnPl->inferenceInput);
    nnPl->inferenceOutput.Link(&plgInferenceOutput_->inferenceIn);

    _logger->info("Linked Plugins...");
    pipe->Start();
    _logger->info("Started FLIC pipeline...");
#else
    UNUSED(graphFileContent);
#endif
}

void KmbExecutor::queueInference(void* input_data, size_t input_bytes) {
    if (!_config.useKmbExecutor()) {
        return;
    }

#if defined(__arm__) || defined(__aarch64__)
    IE_PROFILING_AUTO_SCOPE(queueInference);
    auto physAddr = _allocator->getPhysicalAddress(input_data);
    plgTensorInput_->Push(physAddr, input_bytes);
    _logger->info("Pushed input, size %d", input_bytes);

    uint32_t inferenceInputID = 1;
    _inferenceVirtAddr[0] = inferenceInputID;
    auto inferencePhysAddr = _allocator->getPhysicalAddress(_inferenceVirtAddr);
    plgInferenceInput_->PushInferenceID(inferencePhysAddr, sizeof(inferenceInputID));
#else
    UNUSED(input_data);
    UNUSED(input_bytes);
#endif
}

void KmbExecutor::getResult(void* result_data, unsigned int result_bytes) {
    if (!_config.useKmbExecutor()) {
        return;
    }

#if defined(__arm__) || defined(__aarch64__)
    IE_PROFILING_AUTO_SCOPE(getResult);
    uint32_t len_inferenceId = 0;
    uint32_t pAddr_inferenceId = 0;
    plgInferenceOutput_->PullInferenceID(&pAddr_inferenceId, &len_inferenceId);

    uint32_t len = 0;
    uint32_t pAddr = 0;
    plgTensorOutput_->Pull(&pAddr, &len);

    _logger->info("Output tensor returned of length: %d", len);

    // Convert the physical address we received back to a virtual address we can use.
    uint32_t offset = pAddr - _allocator->getPhysicalAddress(rgnAllocatorBuffer);
    unsigned char* data = static_cast<unsigned char*>(rgnAllocatorBuffer) + offset;

    _logger->info("KmbExecutor::getResult memcpy started @%d", offset);
    IE_ASSERT(result_bytes >= len);
    // FIXME output->Pull gives only the length of the first tensor
    // result_bytes size has to be used here in order to copy data from subsequent tensors
    std::memcpy(result_data, data, result_bytes);
    _logger->info("KmbExecutor::getResult memcpy finished");
#else
    UNUSED(result_data);
    UNUSED(result_bytes);
#endif
}

void KmbExecutor::deallocateGraph() {
    if (!_config.useKmbExecutor()) {
        return;
    }

#if defined(__arm__) || defined(__aarch64__)
    IE_PROFILING_AUTO_SCOPE(deallocateGraph);
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
        _allocator->free(blob_file);
    }
    if (rgnAllocatorBuffer) {
        _allocator->free(rgnAllocatorBuffer);
    }
    if (plgInferenceInput_) {
        plgInferenceInput_->Delete();
    }
    if (plgInferenceOutput_) {
        plgInferenceOutput_->Delete();
    }
    if (plgPoolInferenceMsg) {
        plgPoolInferenceMsg->Delete();
    }
    if (_inferenceVirtAddr) {
        _allocator->free(_inferenceVirtAddr);
    }

    for (const auto& scratchPtr : _scratchBuffers) {
        getKmbAllocator()->free(scratchPtr);
    }
#endif
}

#if defined(__arm__) || defined(__aarch64__)
std::shared_ptr<xlink_handle> getHandleById(const uint32_t& devId) {
    auto xlinkHandlePtr = std::make_shared<xlink_handle>();
    xlinkHandlePtr->sw_device_id = devId;
    xlinkHandlePtr->dev_type = VPUIP_DEVICE;
    return xlinkHandlePtr;
}

bool isDeviceFree(const std::shared_ptr<xlink_handle>& devHandle) {
    uint32_t devStatus = XLINK_DEV_ERROR;
    xlink_error getStatusResult = xlink_get_device_status(devHandle.get(), &devStatus);
    return (getStatusResult == X_LINK_SUCCESS && devStatus == XLINK_DEV_OFF);
}

std::string getNameByHandle(const std::shared_ptr<xlink_handle>& devHandle) {
    constexpr size_t maxDeviceNameSize = 128;
    std::vector<char> devNameData(maxDeviceNameSize, 0x0);
    xlink_error getNameResult = xlink_get_device_name(devHandle.get(), devNameData.data(), devNameData.size());
    if (getNameResult != X_LINK_SUCCESS) {
        THROW_IE_EXCEPTION << "getNameByDeviceId: xlink_get_device_name failed with error: " << getNameResult;
    }
    std::string devName = devNameData.data();
    static const std::map<std::string, std::string> xlinkNameMapping = {
        {"vpu-slice-0", "VPU-0"},
        {"vpu-slice-1", "VPU-1"},
        {"vpu-slice-2", "VPU-2"},
        {"vpu-slice-3", "VPU-3"},
    };
    return xlinkNameMapping.at(devName);
}
#endif

std::vector<std::string> KmbExecutor::getAvailableDevices() {
    std::vector<std::string> deviceNameList;
#if defined(__arm__) || defined(__aarch64__)
    xlink_error initResult = xlink_initialize();
    if (initResult != X_LINK_SUCCESS) {
        THROW_IE_EXCEPTION << "KmbExecutor::getDeviceList: xlink_inititalize failed with error: " << initResult;
    }

    // get all devices
    constexpr size_t maxDeviceListSize = 8;
    std::vector<uint32_t> deviceIdList(maxDeviceListSize, 0x0);
    uint32_t availableDevicesCount = 0;
    xlink_error getDevResult = xlink_get_device_list(deviceIdList.data(), &availableDevicesCount);
    if (getDevResult != X_LINK_SUCCESS) {
        THROW_IE_EXCEPTION << "KmbExecutor::getDeviceList: xlink_get_device_list failed with error: " << getDevResult;
    }
    deviceIdList.resize(availableDevicesCount);

    std::vector<std::shared_ptr<xlink_handle>> devHandleList;
    std::transform(deviceIdList.begin(), deviceIdList.end(), std::back_inserter(devHandleList), getHandleById);

    // filter devices by status
    std::vector<std::shared_ptr<xlink_handle>> freeDevIdList;
    std::copy_if(devHandleList.begin(), devHandleList.end(), std::back_inserter(freeDevIdList), isDeviceFree);

    // get names of free devices
    std::transform(freeDevIdList.begin(), freeDevIdList.end(), std::back_inserter(deviceNameList), getNameByHandle);
#endif
    return deviceNameList;
}
