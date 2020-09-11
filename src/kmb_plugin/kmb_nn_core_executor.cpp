//
// Copyright 2020 Intel Corporation.
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

#include "kmb_nn_core_executor.h"

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
const uint32_t PIPELINE_DEPTH = 4;
#endif

KmbNNCoreExecutor::KmbNNCoreExecutor(const KmbConfig& config)
    : _config(config),
      _logger(std::make_shared<Logger>("KmbNNCoreExecutor", config.logLevel(), consoleOutput())),
      _outputBuffer(nullptr, [](uint8_t* buffer) {
          getKmbAllocator()->free(buffer);
      }) {
    if (!_config.useKmbExecutor()) {
        return;
    }

#if defined(__arm__) || defined(__aarch64__)
    blob_file = nullptr;
    _outputBuffer.reset(reinterpret_cast<uint8_t*>(getKmbAllocator()->alloc(POOL_SIZE)));
#else
    UNUSED(_inferenceVirtAddr);
#endif
}

void KmbNNCoreExecutor::initVpualObjects() {
#if defined(__arm__) || defined(__aarch64__)
    IE_PROFILING_AUTO_SCOPE(initVpualObjects);
    if (!_nnCorePlg) {
        _nnCorePlg = make_shared<NnCorePlg>();
    }
    if (!_nnXlinkPlg) {
        _nnXlinkPlg = make_shared<NnXlinkPlg>();
    }
    if (!_blobHandle) {
        _blobHandle = make_shared<BlobHandle_t>();
    }
    if (!_pipe) {
        _pipe = make_shared<Pipeline>();
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
static std::vector<void*> setScratchHelper(const std::shared_ptr<NnCorePlg>& nnCorePtr, const unsigned int threadCount,
    const std::shared_ptr<KmbAllocator>& allocatorPtr, const std::shared_ptr<vpu::Logger>& logger) {
    if (threadCount > 1) {
        logger->warning("scratchHelper: trying to set scratch buffer to %u threads.", threadCount);
    }
    uint32_t memoryReqs = nnCorePtr->GetScratchBufferSize();
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

    nnCorePtr->SetScratchBuffers(physAddrVec);
    return virtAddrVec;
}
}  // namespace
#endif

void KmbNNCoreExecutor::allocateGraph(const std::vector<char>& graphFileContent, const InputsDataMap& networkInputs,
    const OutputsDataMap& networkOutputs, bool newFormat) {
    if (!_config.useKmbExecutor()) {
        return;
    }

#if defined(__arm__) || defined(__aarch64__)
    IE_PROFILING_AUTO_SCOPE(allocateGraph);
    initVpualObjects();
    static int graphId_main = 1;
    int nThreads = _config.throughputStreams();

    _logger->info("KmbNNCoreExecutor::allocateGraph begins");

    _blobHandle->graphid = graphId_main++;
    _blobHandle->graphBuff = 0x00000000;
    _blobHandle->graphLen = graphFileContent.size();
    _blobHandle->refCount = 0;

    // ########################################################################
    // Try and get some CMA allocations.
    // ########################################################################
    blob_file = getKmbAllocator()->alloc(_blobHandle->graphLen);

    if (!blob_file) {
        _logger->error("KmbNNCoreExecutor::allocateGraph: Error getting CMA for graph");
        THROW_IE_EXCEPTION << "allocateGraph: allocation failed for graph";
    }

    // ########################################################################
    // Load the input files
    // ########################################################################

    std::memcpy(blob_file, graphFileContent.data(), graphFileContent.size());
    std::memset(
        static_cast<uint8_t*>(blob_file) + graphFileContent.size(), 0, _blobHandle->graphLen - graphFileContent.size());
    // Point Blob Handle to the newly loaded graph file. Only allow 32-bit

    // Assigning physical address of Blob file

    _blobHandle->graphBuff = getKmbAllocator()->getPhysicalAddress(blob_file);  // Only lower 32-bits

    auto status = _nnCorePlg->Create(_blobHandle.get(), nThreads);
    if (MVNCI_SUCCESS != status) {
        _logger->error("KmbNNExecutor::allocateGraph: failed to create NnCorePlg");
        THROW_IE_EXCEPTION << "KmbNNExecutor::allocateGraph: failed to create NnCorePlg: " << status;
    }

    auto xlinkStatus = _nnXlinkPlg->Create(PIPELINE_DEPTH * 2);
    if (xlinkStatus) {
        _logger->error("KmbNNExecutor::allocateGraph: failed to create NnXlinkPlg");
        THROW_IE_EXCEPTION << "KmbNNExecutor::allocateGraph: failed to create NnXlinkPlg: " << xlinkStatus;
    }

    MvNCIVersion blobVersion;
    status = _nnCorePlg->GetBlobVersion(&blobVersion);
    if (MVNCI_SUCCESS != status) {
        _logger->error("KmbNNExecutor::allocateGraph: failed to get blob version");
        THROW_IE_EXCEPTION << "KmbNNExecutor::allocateGraph: failed to get blob version: " << status;
    }

    _logger->info("Blob Version: %d %d %d", static_cast<int>(blobVersion.major), static_cast<int>(blobVersion.minor),
        static_cast<int>(blobVersion.patch));
    _scratchBuffers = setScratchHelper(_nnCorePlg, nThreads, getKmbAllocator(), _logger);

    auto tensor_deserializer = [&](const flicTensorDescriptor_t& descriptor) -> void {
        _logger->info(
            "{ n: %d, c: %d, h: %d, w: %d, totalSize: %d, widthStride: %d, heightStride: %d, channelsStride: %d}",
            descriptor.n, descriptor.c, descriptor.h, descriptor.w, descriptor.totalSize, descriptor.widthStride,
            descriptor.heightStride, descriptor.channelsStride);
    };

    _logger->info("Deserializing descriptors:");
    size_t inputsSize = _nnCorePlg->GetNumberOfInputs();
    if (newFormat) {
        IE_ASSERT(networkInputs.size() == inputsSize);
    }

    for (size_t inputIdx = 0; inputIdx < inputsSize; inputIdx++) {
        flicTensorDescriptor_t descIn = _nnCorePlg->GetInputTensorDescriptor(inputIdx);
        _logger->info("Input: %d", inputIdx);
        tensor_deserializer(descIn);

        InferenceEngine::Data inputData = flicTensorDescToIEData(descIn, newFormat, networkInputs, inputIdx, "input");
        InferenceEngine::InputInfo inputInfo;
        inputInfo.setInputData(std::make_shared<InferenceEngine::Data>(inputData));
        _runtimeInputs[inputInfo.name()] = std::make_shared<InferenceEngine::InputInfo>(inputInfo);

        _inputSizes.push_back(descIn.totalSize);
    }

    size_t outputsSize = _nnCorePlg->GetNumberOfOutputs();
    if (newFormat) {
        IE_ASSERT(networkOutputs.size() == outputsSize);
    }

    size_t outputTotalSize = 0;
    for (size_t outputIdx = 0; outputIdx < outputsSize; outputIdx++) {
        flicTensorDescriptor_t descOut = _nnCorePlg->GetOutputTensorDescriptor(outputIdx);
        _logger->info("Output: %d", outputIdx);
        tensor_deserializer(descOut);

        InferenceEngine::Data outputData =
            flicTensorDescToIEData(descOut, newFormat, networkOutputs, outputIdx, "output");
        _runtimeOutputs[outputData.getName()] = std::make_shared<InferenceEngine::Data>(outputData);

        auto outPhysAddr = getKmbAllocator()->getPhysicalAddress(_outputBuffer.get()) + outputTotalSize;
        _outputPhysAddrs.push_back(outPhysAddr);
        outputTotalSize += descOut.totalSize;
    }

    _nnCorePlg->PrepareNetwork();

    _pipe->Add(_nnCorePlg.get());
    _pipe->Add(_nnXlinkPlg.get());
    _nnXlinkPlg->requestOut.Link(&_nnCorePlg->requestInput);
    _nnCorePlg->resultOut.Link(&_nnXlinkPlg->resultIn);

    // Start the pipeline.
    _pipe->Start();

    _logger->info("Started FLIC pipeline...");
#else
    UNUSED(xlinkChannel);
    UNUSED(graphFileContent);
    UNUSED(networkInputs);
    UNUSED(networkOutputs);
    UNUSED(newFormat);
#endif
}

void KmbNNCoreExecutor::queueInference(void* input_data, size_t input_bytes) {
    if (!_config.useKmbExecutor()) {
        return;
    }

#if defined(__arm__) || defined(__aarch64__)
    IE_PROFILING_AUTO_SCOPE(queueInference);
    _logger->info("KmbNNExecutor::push started");
    NnExecMsg request;
    request.inferenceID = 1;
    auto inPhysAddr = getKmbAllocator()->getPhysicalAddress(input_data);
    for (const auto& inputByteSize : _inputSizes) {
        request.inputTensors.push_back(inPhysAddr);
        inPhysAddr += inputByteSize;
    }

    // FIXME
    // this is how request.outputTensors are supposed to work
    /*
        for (const auto& inferOutput : _outputPhysAddrs) {
            request.outputTensors.push_back(inferOutput);
        }
    */
    // however, for some reason they actually work when only the first output is passed
    request.outputTensors.push_back(_outputPhysAddrs.at(0));

    auto status = _nnXlinkPlg->RequestInference(request);
    if (MVNCI_SUCCESS != status) {
        _logger->error("KmbNNExecutor::push: RequestInference failed");
        THROW_IE_EXCEPTION << "KmbNNExecutor::push: RequestInference failed" << status;
    }

    _logger->info("KmbNNExecutor::push finished");
    UNUSED(input_bytes);
#else
    UNUSED(input_data);
    UNUSED(input_bytes);
#endif
}

void KmbNNCoreExecutor::getResult(void* result_data, unsigned int result_bytes) {
    if (!_config.useKmbExecutor()) {
        return;
    }

#if defined(__arm__) || defined(__aarch64__)
    IE_PROFILING_AUTO_SCOPE(getResult);
    _logger->info("KmbNNExecutor::pull started");
    NnExecResponseMsg response;
    auto status = _nnXlinkPlg->WaitForResponse(response);
    if (MVNCI_SUCCESS != status) {
        _logger->error("KmbNNExecutor::pull: WaitForResponse failed");
        THROW_IE_EXCEPTION << "KmbNNExecutor::pull: WaitForResponse failed" << status;
    }

    const void* data = _outputBuffer.get();
    std::memcpy(result_data, data, result_bytes);
    _logger->info("KmbNNExecutor::pull finished");
#else
    UNUSED(result_data);
    UNUSED(result_bytes);
#endif
}

void KmbNNCoreExecutor::deallocateGraph() {
    if (!_config.useKmbExecutor()) {
        return;
    }

#if defined(__arm__) || defined(__aarch64__)
    IE_PROFILING_AUTO_SCOPE(deallocateGraph);
    if (_pipe) {
        _pipe->Stop();
        _pipe->Wait();
        _pipe->Delete();
    }
    if (_nnCorePlg) {
        _nnCorePlg->Delete();
    }
    if (blob_file) {
        getKmbAllocator()->free(blob_file);
    }
    for (const auto& scratchPtr : _scratchBuffers) {
        getKmbAllocator()->free(scratchPtr);
    }
#endif
}
