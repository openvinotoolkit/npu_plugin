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

#include "kmb_nn_core_executor.h"

#include <ie_common.h>

#include <algorithm>
#include <blob_factory.hpp>
#include <cpp_interfaces/exception2status.hpp>
#include <dims_parser.hpp>
#include <ie_itt.hpp>
#include <ie_macro.hpp>
#include <ie_utils.hpp>
#include <map>
#include <memory>
#include <utility>
#include <vector>
#include <vpu/utils/extra.hpp>
#include <vpu/utils/ie_helpers.hpp>
#include <vpu/utils/logger.hpp>

#include "kmb_config.h"

using namespace vpu::KmbPlugin;
using namespace InferenceEngine;
using namespace InferenceEngine::VPUConfigParams;
using namespace std;

#if defined(__arm__) || defined(__aarch64__)
const uint32_t POOL_SIZE = 30 * 1024 * 1024;
const uint32_t PIPELINE_DEPTH = 4;
#endif

KmbNNCoreExecutor::KmbNNCoreExecutor(const vpux::NetworkDescription::Ptr& networkDescription,
    const std::shared_ptr<vpux::Allocator>& allocator, const KmbConfig& config)
    : _networkDescription(networkDescription),
      _allocator(allocator),
      _config(config),
      _logger(std::make_shared<Logger>("KmbNNCoreExecutor", config.logLevel(), consoleOutput())),
      _inputBuffer(nullptr,
          [this](uint8_t* buffer) {
              _allocator->free(buffer);
          }),
      _outputBuffer(nullptr, [this](uint8_t* buffer) {
          _allocator->free(buffer);
      }) {
    if (!_config.useKmbExecutor()) {
        return;
    }

#if defined(__arm__) || defined(__aarch64__)
    blob_file = nullptr;

    std::size_t inputsTotalSize = 0;
    for (auto&& in : _networkDescription->getDeviceInputsInfo()) {
        const auto& tensorDesc = in.second->getTensorDesc();
        inputsTotalSize += utils::getByteSize(tensorDesc);
    }
    _inputBuffer.reset(reinterpret_cast<uint8_t*>(allocator->alloc(inputsTotalSize)));
    _logger->debug("Allocated buffer for input with the size: %d", inputsTotalSize);

    // FIXME: allocate real size instead of POOL_SIZE
    _outputBuffer.reset(reinterpret_cast<uint8_t*>(allocator->alloc(POOL_SIZE)));
    _logger->debug("Allocated buffer for output with the size: %d", POOL_SIZE);

    initVpualObjects();
    allocateGraph(_networkDescription->getCompiledNetwork());
#endif
}

KmbNNCoreExecutor::~KmbNNCoreExecutor() { deallocateGraph(); }

void KmbNNCoreExecutor::initVpualObjects() {
#if defined(__arm__) || defined(__aarch64__)
    OV_ITT_SCOPED_TASK(itt::domains::KmbPlugin, "initVpualObjects");
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
/*
 * Wrapper to SetScratchBuffer
 * 1. Get required memory amount
 * 2. Make sure it's not less than 1 MB (mentioned in [Track number: h#18011677038])
 * 3. Allocate buffer for each NN thread and append physical addresses to collection
 * 4. Give result to SetScratchBuffer
 * 5. Track allocated chunks by virtual addresses to free them properly
 */
static std::vector<void*> setScratchHelper(const std::shared_ptr<NnCorePlg>& nnCorePtr, const unsigned int threadCount,
    const std::shared_ptr<vpux::Allocator>& allocatorPtr, const std::shared_ptr<vpu::Logger>& logger) {
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

void KmbNNCoreExecutor::allocateGraph(const std::vector<char>& graphFileContent) {
    if (!_config.useKmbExecutor()) {
        return;
    }

#if defined(__arm__) || defined(__aarch64__)
    OV_ITT_SCOPED_TASK(itt::domains::KmbPlugin, "allocateGraph");
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
    blob_file = _allocator->alloc(_blobHandle->graphLen);

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

    _blobHandle->graphBuff = _allocator->getPhysicalAddress(blob_file);  // Only lower 32-bits

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
    _scratchBuffers = setScratchHelper(_nnCorePlg, nThreads, _allocator, _logger);

    auto tensor_deserializer = [&](const flicTensorDescriptor_t& descriptor) -> void {
        _logger->info(
            "{ n: %d, c: %d, h: %d, w: %d, totalSize: %d, widthStride: %d, heightStride: %d, channelsStride: %d}",
            descriptor.n, descriptor.c, descriptor.h, descriptor.w, descriptor.totalSize, descriptor.widthStride,
            descriptor.heightStride, descriptor.channelsStride);
    };

    _logger->info("Deserializing descriptors:");
    size_t inputsSize = _nnCorePlg->GetNumberOfInputs();
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
        flicTensorDescriptor_t descIn = _nnCorePlg->GetInputTensorDescriptor(inputIdx);
        _logger->info("Input: %d", inputIdx);
        tensor_deserializer(descIn);

        sumSizeTensorDescIn.totalSize += descIn.totalSize;
    }
    sumSizeTensorDescIn.w = sumSizeTensorDescIn.totalSize;
    sumSizeTensorDescIn.heightStride = sumSizeTensorDescIn.totalSize;
    sumSizeTensorDescIn.channelsStride = sumSizeTensorDescIn.totalSize;

    size_t outputsSize = _nnCorePlg->GetNumberOfOutputs();
    flicTensorDescriptor_t sumSizeTensorDescOut;
    sumSizeTensorDescOut.n = 1;
    sumSizeTensorDescOut.c = 1;
    sumSizeTensorDescOut.h = 1;
    sumSizeTensorDescOut.w = 0;
    sumSizeTensorDescOut.totalSize = 0;
    sumSizeTensorDescOut.widthStride = 1;

    size_t outputTotalSize = 0;
    for (size_t outputIdx = 0; outputIdx < outputsSize; outputIdx++) {
        flicTensorDescriptor_t descOut = _nnCorePlg->GetOutputTensorDescriptor(outputIdx);
        _logger->info("Output: %d", outputIdx);
        tensor_deserializer(descOut);

        auto outPhysAddr = _allocator->getPhysicalAddress(_outputBuffer.get()) + outputTotalSize;
        _outputPhysAddrs.push_back(outPhysAddr);
        outputTotalSize += descOut.totalSize;
    }
    sumSizeTensorDescOut.w = sumSizeTensorDescOut.totalSize;
    sumSizeTensorDescOut.heightStride = sumSizeTensorDescOut.totalSize;
    sumSizeTensorDescOut.channelsStride = sumSizeTensorDescOut.totalSize;

    _nnCorePlg->PrepareNetwork();

    _pipe->Add(_nnCorePlg.get());
    _pipe->Add(_nnXlinkPlg.get());
    _nnXlinkPlg->requestOut.Link(&_nnCorePlg->requestInput);
    _nnCorePlg->resultOut.Link(&_nnXlinkPlg->resultIn);

    // Start the pipeline.
    _pipe->Start();

    _logger->info("Started FLIC pipeline...");
#else
    UNUSED(graphFileContent);
#endif
}

static Blob::Ptr reallocateBlobToLayoutIgnoringOriginalLayout(const Blob::Ptr& blob, const Layout& srcLayout,
    const Layout& dstLayout, const std::shared_ptr<InferenceEngine::IAllocator>& allocator) {
    if (blob->getTensorDesc().getDims()[1] != 3) {
        THROW_IE_EXCEPTION << "reallocateBlobToLayoutIgnoringOriginalLayout works only with channels == 3";
    }

    // it would be nicer to construct srcTensorDesc from tensorDesc of blob
    // and then call srcTensorDesc.setLayout(srcLayout) but copyBlob does work in that case
    TensorDesc srcTensorDesc = {blob->getTensorDesc().getPrecision(), blob->getTensorDesc().getDims(), srcLayout};
    Blob::Ptr srcBlob = make_blob_with_precision(srcTensorDesc, blob->buffer());

    TensorDesc dstTensorDesc = {blob->getTensorDesc().getPrecision(), blob->getTensorDesc().getDims(), dstLayout};
    Blob::Ptr dstBlob = make_blob_with_precision(dstTensorDesc, allocator);
    dstBlob->allocate();

    vpu::copyBlob(srcBlob, dstBlob);
    return dstBlob;
}

static Blob::Ptr reallocateBlobToLayout(
    const Blob::Ptr& blob, const Layout& layout, const std::shared_ptr<InferenceEngine::IAllocator>& allocator) {
    TensorDesc dstTensorDesc = {blob->getTensorDesc().getPrecision(), blob->getTensorDesc().getDims(), layout};
    Blob::Ptr kmbBlob = make_blob_with_precision(dstTensorDesc, allocator);
    kmbBlob->allocate();

    vpu::copyBlob(blob, kmbBlob);

    return kmbBlob;
}

static bool needRepackForNHWC(const TensorDesc& actualDesc) {
    /* NB: Brief overview:
     * Runtime works only with NHWC layout, but actual input layout can be different
     * therefore it should be repacked, let's to observe cases:
         1) NC & C there isn't necessary to do repacking,
            because these layouts has the same representation in NCHW & NHWC
         2) NHWC isn't necessary to do repacking obviously
         3) NCHW in general case it should be repacked, however if it is 11HW it isn't necessary
         4) CHW the same as for NCHW case, it isn't necessary to do repacking in 1HW case
     */
    const auto actualLayout = actualDesc.getLayout();
    const auto& actualDims = actualDesc.getDims();
    switch (actualLayout) {
    case Layout::NHWC:
    case Layout::NC:
    case Layout::C:
        return false;
    case Layout::NCHW:
        return (actualDims[0] != 1) || (actualDims[1] != 1);
    case Layout::CHW:
        return actualDims[0] != 1;
    default:
        THROW_IE_EXCEPTION << "Unsupported layout for actual blob: " << actualLayout;
    }
}

Blob::Ptr KmbNNCoreExecutor::prepareInputForInference(
    const ie::Blob::Ptr& actualInput, const InferenceEngine::TensorDesc& deviceDesc) {
    OV_ITT_SCOPED_TASK(itt::domains::KmbPlugin, "prepareInputForInference");

    // HACK: to overcome inability python API to pass a blob of NHWC layout
    if (_config.forceNCHWToNHWC()) {
        _logger->warning("VPU_KMB_FORCE_NCHW_TO_NHWC is enabled. Need to do re-layout.");
        return reallocateBlobToLayoutIgnoringOriginalLayout(actualInput, Layout::NCHW, Layout::NHWC, _allocator);
    }

    Blob::Ptr inputForInference;
    if (!utils::isBlobAllocatedByAllocator(actualInput, _allocator)) {
        _logger->warning("Input blob is located in non-shareable memory. Need to do re-allocation.");
        inputForInference = utils::reallocateBlob(actualInput, _allocator);
    } else {
        inputForInference = actualInput;
    }

    const auto& actualDesc = actualInput->getTensorDesc();
    const auto& deviceLayout = deviceDesc.getLayout();

    if (needRepackForNHWC(actualDesc) && deviceLayout == Layout::NHWC) {
        _logger->warning("Input blob is inconsistent with network input. Need to do re-layout.");
        // NB: It's possible to make repack data only with the same number of dimensions
        // So just make a view without any copy
        const auto outputMemoryBlob = as<MemoryBlob>(actualInput);
        IE_ASSERT(outputMemoryBlob != nullptr);
        const auto outputMemory = outputMemoryBlob->rmap();
        IE_ASSERT(outputMemory != nullptr);
        const auto outputPtr = outputMemory.as<void*>();
        IE_ASSERT(outputPtr != nullptr);
        Blob::Ptr actualView4D = make_blob_with_precision(getNCHW(actualInput->getTensorDesc()), outputPtr);
        inputForInference = reallocateBlobToLayout(actualView4D, deviceLayout, _allocator);
    }

    return inputForInference;
}

void KmbNNCoreExecutor::push(const InferenceEngine::BlobMap& /*inputs*/, const vpux::PreprocMap& /*preProcMap*/) {
    THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
}

void KmbNNCoreExecutor::push(const InferenceEngine::BlobMap& inputs) {
    if (!_config.useKmbExecutor()) {
        return;
    }

#if defined(__arm__) || defined(__aarch64__)
    OV_ITT_SCOPED_TASK(itt::domains::KmbPlugin, "push");
    _logger->info("KmbNNCoreExecutor::push started");

    ie::BlobMap updatedInputs;
    const auto& deviceInputs = _networkDescription->getDeviceInputsInfo();
    int inputsByteSize = 0;
    for (const auto& inferInput : inputs) {
        const auto& name = inferInput.first;
        const auto& deviceInputDesc = deviceInputs.at(name)->getTensorDesc();
        const auto& input = inferInput.second;

        auto updatedInput = prepareInputForInference(input, deviceInputDesc);

        updatedInputs.insert({inferInput.first, updatedInput});
        inputsByteSize += updatedInput->byteSize();
    }

    NnExecMsg request;
    request.inferenceID = 1;
    for (const auto& input : updatedInputs) {
        auto blob = as<MemoryBlob>(input.second);
        auto memoryHolder = blob->rmap();
        auto inputBufferPhysAddr = _allocator->getPhysicalAddress(memoryHolder.as<uint8_t*>());
        request.inputTensors.push_back(inputBufferPhysAddr);
    }

    for (const auto& inferOutput : _outputPhysAddrs) {
        request.outputTensors.push_back(inferOutput);
    }

    auto status = _nnXlinkPlg->RequestInference(request);
    if (MVNCI_SUCCESS != status) {
        _logger->error("KmbNNExecutor::push: RequestInference failed");
        THROW_IE_EXCEPTION << "KmbNNExecutor::push: RequestInference failed" << status;
    }

    _logger->info("KmbNNCoreExecutor::push finished");
#else
    UNUSED(inputs);
#endif
}

uint32_t KmbNNCoreExecutor::extractPhysAddrForInference(const BlobMap& inputs) {
    uint32_t physAddr = 0;
    if (inputs.size() == 1) {
        auto blob = as<MemoryBlob>(inputs.begin()->second);
        auto memoryHolder = blob->rmap();
        physAddr = _allocator->getPhysicalAddress(memoryHolder.as<uint8_t*>());
        if (!physAddr) {
            THROW_IE_EXCEPTION << "Memory of input is not valid";
        }
    } else {
        _logger->warning("There are multiple blobs. Need to combine them into single buffer.");
        std::size_t offset = 0;
        for (const auto& input : inputs) {
            auto name = input.first;
            auto blob = as<MemoryBlob>(input.second);

            if (!blob) {
                THROW_IE_EXCEPTION << "Cannot cast to MemoryBlob";
            }
            auto memoryHolder = blob->rmap();

            ie_memcpy(_inputBuffer.get() + offset, blob->byteSize(), memoryHolder.as<uint8_t*>(), blob->byteSize());
            offset += blob->byteSize();
        }

        physAddr = _allocator->getPhysicalAddress(_inputBuffer.get());
        if (!physAddr) {
            THROW_IE_EXCEPTION << "Memory of input is not valid";
        }
    }

    return physAddr;
}

void KmbNNCoreExecutor::pull(InferenceEngine::BlobMap& outputs) {
    if (!_config.useKmbExecutor()) {
        return;
    }

#if defined(__arm__) || defined(__aarch64__)
    OV_ITT_SCOPED_TASK(itt::domains::KmbPlugin, "pull");
    _logger->info("KmbNNCoreExecutor::pull started");
    NnExecResponseMsg response;
    auto status = _nnXlinkPlg->WaitForResponse(response);
    if (MVNCI_SUCCESS != status) {
        _logger->error("KmbNNExecutor::pull: WaitForResponse failed");
        THROW_IE_EXCEPTION << "KmbNNExecutor::pull: WaitForResponse failed" << status;
    }

    BlobMap deviceOutputs = extractOutputsFromPhysAddr(_outputPhysAddrs.at(0));
    repackDeviceOutputsToNetworkOutputs(deviceOutputs, outputs);
    _logger->info("KmbNNCoreExecutor::pull finished");
#else
    UNUSED(outputs);
#endif
}

BlobMap KmbNNCoreExecutor::extractOutputsFromPhysAddr(uint32_t physAddr) {
    BlobMap deviceOutputs;
    std::size_t offset = physAddr - _allocator->getPhysicalAddress(_outputBuffer.get());
    for (auto&& out : _networkDescription->getDeviceOutputsInfo()) {
        auto desc = out.second->getTensorDesc();
        auto blob = make_blob_with_precision(desc, _outputBuffer.get() + offset);
        deviceOutputs.insert({out.first, blob});
        offset += utils::getByteSize(desc);
    }

    return deviceOutputs;
}

void KmbNNCoreExecutor::repackDeviceOutputsToNetworkOutputs(
    const ie::BlobMap& deviceOutputs, ie::BlobMap& networkOutputs) {
    for (const auto& item : deviceOutputs) {
        const auto& name = item.first;
        const auto& deviceBlob = item.second;
        const auto& deviceDesc = deviceBlob->getTensorDesc();
        const auto& outputBlob = networkOutputs[name];
        const auto& networkDesc = outputBlob->getTensorDesc();

        Blob::Ptr deviceBlobWithNetworkPrecision = nullptr;
        if (deviceDesc.getPrecision() != networkDesc.getPrecision()) {
            _logger->warning("Output blob is inconsistent with network output. "
                             "Need to do convert precision from %d to %d.",
                deviceDesc.getPrecision(), networkDesc.getPrecision());
            deviceBlobWithNetworkPrecision = utils::convertPrecision(deviceBlob, networkDesc.getPrecision());
        } else {
            deviceBlobWithNetworkPrecision = deviceBlob;
        }

        const auto& outputMemoryBlob = as<MemoryBlob>(outputBlob);
        IE_ASSERT(outputMemoryBlob != nullptr);
        const auto outputMemory = outputMemoryBlob->rmap();
        IE_ASSERT(outputMemory != nullptr);
        const auto outputPtr = outputMemory.as<void*>();
        IE_ASSERT(outputPtr != nullptr);
        if (needRepackForNHWC(networkDesc) && deviceDesc.getLayout() == ie::Layout::NHWC) {
            _logger->warning("Output blob is inconsistent with network output."
                             "Need to do re-layout from %d to %d.",
                networkDesc.getLayout(), deviceDesc.getLayout());
            // NB: It's possible to make repack data only with the same number of dimensions
            // So just make a view without any copy
            const auto actualView4D = make_blob_with_precision(getNCHW(networkDesc), outputPtr);
            vpu::copyBlob(deviceBlobWithNetworkPrecision, actualView4D);
        } else {
            vpu::copyBlob(deviceBlobWithNetworkPrecision, deviceDesc.getLayout(), outputPtr);
        }
    }
}

void KmbNNCoreExecutor::setup(const InferenceEngine::ParamMap&) { THROW_IE_EXCEPTION << "Not implemented"; }

bool KmbNNCoreExecutor::isPreProcessingSupported(const InferenceEngine::PreProcessInfo&) const { return false; }

std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> KmbNNCoreExecutor::getLayerStatistics() {
    THROW_IE_EXCEPTION << "Not implemented";
    return std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>();
}

InferenceEngine::Parameter KmbNNCoreExecutor::getParameter(const std::string&) const {
    return InferenceEngine::Parameter();
}

void KmbNNCoreExecutor::deallocateGraph() {
    if (!_config.useKmbExecutor()) {
        return;
    }

#if defined(__arm__) || defined(__aarch64__)
    OV_ITT_SCOPED_TASK(itt::domains::KmbPlugin, "deallocateGraph");
    if (_pipe) {
        _pipe->Stop();
        _pipe->Wait();
        _pipe->Delete();
    }
    if (_nnCorePlg) {
        _nnCorePlg->Delete();
    }
    if (blob_file) {
        _allocator->free(blob_file);
    }

    for (const auto& scratchPtr : _scratchBuffers) {
        _allocator->free(scratchPtr);
    }
#endif
}
