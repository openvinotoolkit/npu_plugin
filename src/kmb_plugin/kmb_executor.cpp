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
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>
#include <vpu/utils/extra.hpp>
#include <vpu/utils/logger.hpp>

#include "kmb_config.h"
#include "kmb_native_allocator.h"
#include "kmb_udma_allocator.h"
#include "kmb_vpusmm_allocator.h"
#include "ie_macro.hpp"

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
// XLink channel number to start allocation from
const uint32_t IE_VPU_KMB_XC_DEFAULT = 3;

// Get free XLink channel
static uint32_t getXlinkChannel(const vpu::Logger::Ptr& _logger) {
    static std::mutex mutex_;
    static int XlinkChannel = -1;

    std::unique_lock<std::mutex> lock(mutex_);

    if (XlinkChannel <= 0) {
        const char* pxc = getenv("IE_VPU_KMB_XC");
        XlinkChannel = pxc ? atoi(pxc) : IE_VPU_KMB_XC_DEFAULT;
    }
    // In this simplified implementation we never reuse the channel
    uint32_t ret = XlinkChannel++;

    // Skipping "0xA: IP control channel (standard channel)"
    if (ret == 10) {
        ret = XlinkChannel++;
    }
    _logger->info("Allocated channel = %d", ret);
    return ret;
}
#endif

KmbExecutor::KmbExecutor(const KmbConfig& config)
    : _config(config),
      _logger(std::make_shared<Logger>("KmbExecutor", config.logLevel(), consoleOutput())),
      xlinkChannelIn(0),
      xlinkChannelOut(0),
      _xlinkChannelInferenceInput(0),
      _xlinkChannelInferenceOutput(0),
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
        _inferenceVirtAddr = reinterpret_cast<uint32_t*>(getKmbAllocator()->alloc(sizeof(uint32_t)));
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

std::vector<size_t> filterDimsWithSizeOne(const InferenceEngine::SizeVector& dims) {
    std::vector<size_t> out;
    if (dims.size() == 0) {
        return out;
    }

    out.push_back(dims.at(0));  // always push batch size
    for (size_t idx = 1; idx < dims.size(); idx++) {
        if (dims.at(idx) > 1) {
            out.push_back(dims.at(idx));
        }
    }

    return out;
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
    blob_file = getKmbAllocator()->alloc(graphFileContent.size());

    if (!blob_file) {
        _logger->error("KmbExecutor::allocateGraph: Error getting CMA for graph");
        return;
    }

    // ########################################################################
    // Load the input files
    // ########################################################################

    std::memcpy(blob_file, graphFileContent.data(), graphFileContent.size());
    // Point Blob Handle to the newly loaded graph file. Only allow 32-bit

    // Assigning physical address of Blob file

    BHandle->graphBuff = getKmbAllocator()->getPhysicalAddress(blob_file);  // Only lower 32-bits

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

    _logger->info("NN Plugin Create finished...");

    NNPlgState state = nnPl->GetLatestState();
    if (SUCCESS != state) {
        _logger->error("Error, bad NN Plugin state: ");
        return;
    }

    auto tensor_deserializer = [&](const flicTensorDescriptor_t& descriptor) -> void {
        _logger->info(
            "{ n: %d, c: %d, h: %d, w: %d, totalSize: %d, widthStride: %d, heightStride: %d, channelsStride: %d}",
            descriptor.n, descriptor.c, descriptor.h, descriptor.w, descriptor.totalSize, descriptor.widthStride,
            descriptor.heightStride, descriptor.channelsStride);
    };

    flicTensorDescriptor_t descOut = nnPl->GetOutputTensorDescriptor(0);
    flicTensorDescriptor_t descIn = nnPl->GetInputTensorDescriptor(0);
    _logger->info("Deserializing descriptors:\nInput: ");
    tensor_deserializer(descIn);
    _logger->info("Output: ");
    tensor_deserializer(descOut);

    InferenceEngine::SizeVector inputDims({descIn.n, descIn.c, descIn.h, descIn.w});
    InferenceEngine::Layout inputLayout = getIOLayout(descIn);
    InferenceEngine::Precision inputPrecision = getIOPrecision(descIn);
    InferenceEngine::TensorDesc inputDesc(inputPrecision, inputDims, inputLayout);
    InferenceEngine::Data inputData("input", inputDesc);

    InferenceEngine::InputInfo inputInfo;
    inputInfo.setInputData(std::make_shared<InferenceEngine::Data>(inputData));
    m_networkInputs[inputInfo.name()] = std::make_shared<InferenceEngine::InputInfo>(inputInfo);
    _inputNetworkLayout = inputLayout;

    InferenceEngine::SizeVector outputDims({descOut.n, descOut.c, descOut.h, descOut.w});
    InferenceEngine::Layout outputLayout = getIOLayout(descOut);
    if (_config.force2DToNC()) {
        InferenceEngine::SizeVector nonTrivialDims = filterDimsWithSizeOne(outputDims);
        if (nonTrivialDims.size() == 2) {
            outputDims = nonTrivialDims;
            outputLayout = InferenceEngine::Layout::NC;
        }
    }
    InferenceEngine::Precision outputPrecision = getIOPrecision(descOut);
    InferenceEngine::TensorDesc outputDesc(outputPrecision, outputDims, outputLayout);
    InferenceEngine::Data outputData("output", outputDesc);

    m_networkOutputs[outputData.getName()] = std::make_shared<InferenceEngine::Data>(outputData);

    rgnAllocatorBuffer = getKmbAllocator()->alloc(POOL_SIZE);
    if (!rgnAllocatorBuffer) {
        _logger->error("KmbExecutor::allocateGraph: Cannot allocate buffer for RgnAlloc");
        return;
    }
    RgnAlloc->Create(getKmbAllocator()->getPhysicalAddress(rgnAllocatorBuffer), POOL_SIZE);
    _logger->info("KmbExecutor::allocateGraph: Created RgnAlloc");

    // TODO - These
    const unsigned int shavel2CacheLineSize = 64;
    unsigned int outputTensorSize = ROUND_UP(descOut.totalSize, shavel2CacheLineSize);

    // TODO - These
    _logger->info("read memory pool finished...");
    plgPoolOutputs->Create(RgnAlloc.get(), 1, 3 * outputTensorSize);
    _logger->info("Created plgPoolOutputs");

    unsigned int inferenceIDSize = ROUND_UP(sizeof(uint32_t), shavel2CacheLineSize);
    plgPoolInferenceMsg->Create(HeapAlloc.get(), 1, 3 * inferenceIDSize);
    _logger->info("Created plgPoolInferenceMsg");

    xlinkChannelIn = getXlinkChannel(_logger);
    xlinkChannelOut = getXlinkChannel(_logger);
    plgTensorInput_->Create(descIn.totalSize, xlinkChannelIn, descIn);
    _logger->info("Created plgTensorInput");

    plgTensorOutput_->Create(descOut.totalSize, xlinkChannelOut, descOut);
    _logger->info("Created plgTensorOutput");

    _xlinkChannelInferenceInput = getXlinkChannel(_logger);
    _xlinkChannelInferenceOutput = getXlinkChannel(_logger);
    plgInferenceInput_->Create(3 * inferenceIDSize, _xlinkChannelInferenceInput);
    _logger->info("Created plgInferenceInput_");

    plgInferenceOutput_->Create(3 * inferenceIDSize, _xlinkChannelInferenceOutput);
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
    auto physAddr = getKmbAllocator()->getPhysicalAddress(input_data);
    plgTensorInput_->Push(physAddr, input_bytes);
    _logger->info("Pushed input, size %d", input_bytes);

    uint32_t inferenceInputID = 1;
    _inferenceVirtAddr[0] = inferenceInputID;
    auto inferencePhysAddr = getKmbAllocator()->getPhysicalAddress(_inferenceVirtAddr);
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
    uint32_t offset = pAddr - getKmbAllocator()->getPhysicalAddress(rgnAllocatorBuffer);
    unsigned char* data = static_cast<unsigned char*>(rgnAllocatorBuffer) + offset;

    _logger->info(
        "KmbExecutor::getResult memcpy started @%d xlinkChannel=%d, %d ", offset, xlinkChannelIn, xlinkChannelOut);

    _logger->info("KmbExecutor::getResult memcpy started");
    IE_ASSERT(result_bytes >= len);
    std::memcpy(result_data, data, len);
    std::memset(data, 0, len);
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
        getKmbAllocator()->free(blob_file);
    }
    if (rgnAllocatorBuffer) {
        getKmbAllocator()->free(rgnAllocatorBuffer);
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
        getKmbAllocator()->free(_inferenceVirtAddr);
    }
#endif
}
