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
#include <stdio.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <utility>
#include <vector>
#include <vpu/utils/extra.hpp>
#include <vpu/utils/logger.hpp>

#include "kmb_config.h"
#include "kmb_native_allocator.h"
#include "kmb_udma_allocator.h"
#include "kmb_vpusmm_allocator.h"

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

KmbExecutor::KmbExecutor(const KmbConfig& config)
    : _config(config),
      _logger(std::make_shared<Logger>("KmbExecutor", config.logLevel(), consoleOutput())),
      _outTensorLen(0),
      _outTensorAddr(0),
      _inferenceVirtAddr(nullptr) {
    if (!_config.useKmbExecutor()) {
        return;
    }

#if defined(__arm__) || defined(__aarch64__)
    blob_file = nullptr;
    rgnAllocatorBuffer = nullptr;
    _inferenceVirtAddr = nullptr;
#endif
}

void KmbExecutor::initVpualObjects() {
#if defined(__arm__) || defined(__aarch64__)
    IE_PROFILING_AUTO_SCOPE(initVpualObjects);
    if (!RgnAlloc) {
        RgnAlloc = make_shared<RgnAllocator>(_config.VPUSMMSliceIdx());
    }
    if (!HeapAlloc) {
        constexpr size_t heapAllocAlignment = 64;
        HeapAlloc = make_shared<HeapAllocator>(heapAllocAlignment, _config.VPUSMMSliceIdx());
    }
    if (!nnPl) {
        nnPl = make_shared<NNFlicPlg>(_config.VPUSMMSliceIdx());
    }
    if (!gg) {
        gg = make_shared<GraphManagerPlg>(_config.VPUSMMSliceIdx());
    }
    if (!plgTensorInput_) {
        plgTensorInput_ = make_shared<PlgTensorSource>(_config.VPUSMMSliceIdx());
    }
    if (!plgTensorOutput_) {
        plgTensorOutput_ = make_shared<PlgStreamResult>(_config.VPUSMMSliceIdx());
    }
    if (!plgInferenceInput_) {
        plgInferenceInput_ = make_shared<PlgInferenceInput>(_config.VPUSMMSliceIdx());
    }
    if (!plgInferenceOutput_) {
        plgInferenceOutput_ = make_shared<PlgInferenceOutput>(_config.VPUSMMSliceIdx());
    }
    if (!plgPoolOutputs) {
        plgPoolOutputs = make_shared<PlgPool<TensorMsg>>(_config.VPUSMMSliceIdx());
    }
    if (!plgPoolInferenceMsg) {
        plgPoolInferenceMsg = make_shared<PlgPool<InferenceMsg>>(_config.VPUSMMSliceIdx());
    }
    if (!BHandle) {
        BHandle = make_shared<BlobHandle_t>();
    }
    if (!pipe) {
        constexpr size_t maxPluginsPerPipeline = 32;
        pipe = make_shared<Pipeline>(maxPluginsPerPipeline, _config.VPUSMMSliceIdx());
    }
    if (!_inferenceVirtAddr) {
        _inferenceVirtAddr =
            reinterpret_cast<uint32_t*>(getKmbAllocator(_config.VPUSMMSliceIdx())->alloc(sizeof(uint32_t)));
    }
#endif
}

#if defined(__arm__) || defined(__aarch64__)
namespace {

InferenceEngine::Layout getIOLayout(const flicTensorDescriptor_t& descTemp) {
    InferenceEngine::Layout tensorLayout = InferenceEngine::Layout::NCHW;
    std::vector<uint32_t> strides {descTemp.heightStride, descTemp.widthStride, descTemp.channelsStride};
    std::vector<uint32_t>::iterator maxStrideIter = std::max_element(strides.begin(), strides.end());
    uint32_t maxStrideVal = *maxStrideIter;
    if (maxStrideVal == descTemp.heightStride) {
        if (std::max(descTemp.widthStride, descTemp.channelsStride) == descTemp.widthStride) {
            tensorLayout = InferenceEngine::Layout::NHWC;
        } else {
            // NHCW
            THROW_IE_EXCEPTION << "getIOLayout: NHCW layout is not supported";
        }
    } else if (maxStrideVal == descTemp.channelsStride) {
        if (std::max(descTemp.widthStride, descTemp.heightStride) == descTemp.heightStride) {
            tensorLayout = InferenceEngine::Layout::NCHW;
        } else {
            // NCWH
            THROW_IE_EXCEPTION << "getIOLayout: NCWH layout is not supported";
        }
    } else {
        // width-major
        THROW_IE_EXCEPTION << "getIOLayout: W-major layout is not supported";
    }

    return tensorLayout;
}

const std::map<precision_t, InferenceEngine::Precision> precisionMap = {
    std::pair<precision_t, InferenceEngine::Precision>(precision_t::FP32, InferenceEngine::Precision::FP32),
    std::pair<precision_t, InferenceEngine::Precision>(precision_t::FP16, InferenceEngine::Precision::FP16),
    std::pair<precision_t, InferenceEngine::Precision>(precision_t::U8, InferenceEngine::Precision::U8),
};

InferenceEngine::Precision getIOPrecision(const flicTensorDescriptor_t& descTemp) {
    precision_t flicPrecision = static_cast<precision_t>(descTemp.dtype);
    std::map<precision_t, InferenceEngine::Precision>::const_iterator found = precisionMap.find(flicPrecision);
    if (found == precisionMap.end()) {
        THROW_IE_EXCEPTION << "getIOPrecision: failed to convert FLIC precision " << flicPrecision;
    }
    return found->second;
}

// DataMapType expected to be either InputsDataMap or OutputsDataMap
// useNewFormat basically means that graph header doesn't have meta-data
// defaultNamePrefix is used only with old format
// dataIdx is the position of input/output in FLIC pipeline internal data structure
template <typename DataMapType>
InferenceEngine::Data flicTensorDescToIEData(const flicTensorDescriptor_t& flicTensorDesc, bool useNewFormat,
    const DataMapType& dataMap, const size_t& dataIdx, const std::string& defaultNamePrefix) {
    InferenceEngine::SizeVector ieDims({flicTensorDesc.n, flicTensorDesc.c, flicTensorDesc.h, flicTensorDesc.w});
    InferenceEngine::Layout ieLayout = getIOLayout(flicTensorDesc);
    InferenceEngine::Precision iePrecision = getIOPrecision(flicTensorDesc);
    InferenceEngine::TensorDesc ieDesc(iePrecision, ieDims, ieLayout);
    std::string ieDataName = "";
    if (useNewFormat) {
        // FIXME data maps have lexicographical order. however, runtime inputs and outputs are not ordered by name
        // find a better way to map names
        typename DataMapType::const_iterator dataMapIter = dataMap.begin();
        std::advance(dataMapIter, dataIdx);
        ieDataName = dataMapIter->first;
    } else {
        ieDataName = defaultNamePrefix + std::to_string(dataIdx);
    }
    InferenceEngine::Data ieData(ieDataName, ieDesc);
    return ieData;
}

/*
 * Wrapper to SetScratchBuffer
 * 1. Get required memory amount
 * 2. Make sure it's not less than 1 MB (mentioned in [Track number: h#18011677038])
 * 3. Allocate buffer for each NN thread and append physical addresses to collection
 * 4. Give result to SetScratchBuffer
 * 5. Track allocated chunks by virtual addresses to free them properly
 */
static std::vector<void*> setScratchHelper(const std::shared_ptr<NNFlicPlg>& nnFlicPtr, const unsigned int& threadCount,
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
    std::vector<void*> physAddrVec;
    for (unsigned int threadIdx = 0; threadIdx < threadCount; threadIdx++) {
        uint8_t* scratchVirtAddr = reinterpret_cast<uint8_t*>(allocatorPtr->alloc(memoryReqs));
        if (scratchVirtAddr == nullptr) {
            THROW_IE_EXCEPTION << "scratchHelper: failed to allocate " << memoryReqs << " bytes of memory";
        }
        unsigned long scratchPhysAddr = allocatorPtr->getPhysicalAddress(scratchVirtAddr);
        if (scratchPhysAddr == 0) {
            THROW_IE_EXCEPTION << "scratchHelper: failed to get physical address";
        }
        // NB: casting unsigned long to void may cause problems on some systems
        // vpualHost API has to be updated to accept integers
        physAddrVec.push_back(reinterpret_cast<void*>(scratchPhysAddr));
        virtAddrVec.push_back(scratchVirtAddr);
    }

    nnFlicPtr->SetScratchBuffer(physAddrVec);
    return virtAddrVec;
}
}  // namespace
#endif

void KmbExecutor::allocateGraph(const std::vector<char>& graphFileContent, const InputsDataMap& networkInputs,
    const OutputsDataMap& networkOutputs, bool newFormat) {
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
    blob_file = getKmbAllocator(_config.VPUSMMSliceIdx())->alloc(BHandle->graphLen);

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

    BHandle->graphBuff =
        getKmbAllocator(_config.VPUSMMSliceIdx())->getPhysicalAddress(blob_file);  // Only lower 32-bits

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

    _scratchBuffers = setScratchHelper(nnPl, nThreads, getKmbAllocator(_config.VPUSMMSliceIdx()), _logger);

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
    if (newFormat) {
        IE_ASSERT(networkInputs.size() == inputsSize);
    }
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

        InferenceEngine::Data inputData = flicTensorDescToIEData(descIn, newFormat, networkInputs, inputIdx, "input");
        InferenceEngine::InputInfo inputInfo;
        inputInfo.setInputData(std::make_shared<InferenceEngine::Data>(inputData));
        _runtimeInputs[inputInfo.name()] = std::make_shared<InferenceEngine::InputInfo>(inputInfo);

        sumSizeTensorDescIn.totalSize += descIn.totalSize;
    }
    sumSizeTensorDescIn.w = sumSizeTensorDescIn.totalSize;
    sumSizeTensorDescIn.heightStride = sumSizeTensorDescIn.totalSize;
    sumSizeTensorDescIn.channelsStride = sumSizeTensorDescIn.totalSize;

    size_t outputsSize = nnPl->GetNumberOfOutputs();
    if (newFormat) {
        IE_ASSERT(networkOutputs.size() == outputsSize);
    }
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

        InferenceEngine::Data outputData =
            flicTensorDescToIEData(descOut, newFormat, networkOutputs, outputIdx, "output");
        _runtimeOutputs[outputData.getName()] = std::make_shared<InferenceEngine::Data>(outputData);

        sumSizeTensorDescOut.totalSize += descOut.totalSize;
    }
    sumSizeTensorDescOut.w = sumSizeTensorDescOut.totalSize;
    sumSizeTensorDescOut.heightStride = sumSizeTensorDescOut.totalSize;
    sumSizeTensorDescOut.channelsStride = sumSizeTensorDescOut.totalSize;

    rgnAllocatorBuffer = getKmbAllocator(_config.VPUSMMSliceIdx())->alloc(POOL_SIZE);
    if (!rgnAllocatorBuffer) {
        _logger->error("KmbExecutor::allocateGraph: Cannot allocate buffer for RgnAlloc");
        THROW_IE_EXCEPTION << "allocateGraph: allocation failed for region allocator";
    }
    RgnAlloc->Create(getKmbAllocator(_config.VPUSMMSliceIdx())->getPhysicalAddress(rgnAllocatorBuffer), POOL_SIZE);
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
    UNUSED(networkInputs);
    UNUSED(networkOutputs);
    UNUSED(newFormat);
#endif
}

void KmbExecutor::queueInference(void* input_data, size_t input_bytes) {
    if (!_config.useKmbExecutor()) {
        return;
    }

#if defined(__arm__) || defined(__aarch64__)
    IE_PROFILING_AUTO_SCOPE(queueInference);
    auto physAddr = getKmbAllocator(_config.VPUSMMSliceIdx())->getPhysicalAddress(input_data);
    plgTensorInput_->Push(physAddr, input_bytes);
    _logger->info("Pushed input, size %d", input_bytes);

    uint32_t inferenceInputID = 1;
    _inferenceVirtAddr[0] = inferenceInputID;
    auto inferencePhysAddr = getKmbAllocator(_config.VPUSMMSliceIdx())->getPhysicalAddress(_inferenceVirtAddr);
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
    uint32_t offset = pAddr - getKmbAllocator(_config.VPUSMMSliceIdx())->getPhysicalAddress(rgnAllocatorBuffer);
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
        getKmbAllocator(_config.VPUSMMSliceIdx())->free(blob_file);
    }
    if (rgnAllocatorBuffer) {
        getKmbAllocator(_config.VPUSMMSliceIdx())->free(rgnAllocatorBuffer);
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
        getKmbAllocator(_config.VPUSMMSliceIdx())->free(_inferenceVirtAddr);
    }

    for (const auto& scratchPtr : _scratchBuffers) {
        getKmbAllocator(_config.VPUSMMSliceIdx())->free(scratchPtr);
    }
#endif
}
