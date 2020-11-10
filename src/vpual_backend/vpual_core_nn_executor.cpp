//
// Copyright 2019-2020 Intel Corporation.
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

#include "vpual_core_nn_executor.hpp"

#include <ie_common.h>

#include <algorithm>
#include <blob_factory.hpp>
#include <dims_parser.hpp>
#include <ie_itt.hpp>
#include <ie_macro.hpp>
#include <ie_utils.hpp>
#include <map>
#include <utility>
#include <vector>
#include <vpu/utils/extra.hpp>
#include <vpu/utils/ie_helpers.hpp>
#include <vpu/utils/logger.hpp>

#include "vpual_config.hpp"
#include "vpusmm_allocator.hpp"

namespace ie = InferenceEngine;

namespace vpux {
#if defined(__arm__) || defined(__aarch64__)
constexpr uint32_t POOL_SIZE = 30 * 1024 * 1024;
#endif
constexpr int VPU_CSRAM_DEVICE_ID = 32;

VpualCoreNNExecutor::VpualCoreNNExecutor(const vpux::NetworkDescription::Ptr& networkDescription,
    const VpusmmAllocator::Ptr& allocator, const uint32_t deviceId, const VpualConfig& config)
    : _networkDescription(networkDescription),
      _allocator(allocator),
      _csramAllocator(std::make_shared<VpusmmAllocator>(VPU_CSRAM_DEVICE_ID)),
      _config(config),
      _logger(std::make_shared<vpu::Logger>("VpualCoreNNExecutor", _config.logLevel(), vpu::consoleOutput())),
#if defined(__arm__) || defined(__aarch64__)
      _nnXlinkPlg(new NnXlinkPlg(deviceId)),
      _nnCorePlg(new NnCorePlg(deviceId),
          [](NnCorePlg* nnCorePlgPtr) {
              if (nnCorePlgPtr != nullptr) {
                  nnCorePlgPtr->Delete();
              }
          }),
      _pipe(new Pipeline(MAX_PLUGS_PER_PIPE, deviceId),
          [](Pipeline* pipePtr) {
              if (pipePtr != nullptr) {
                  pipePtr->Stop();
                  pipePtr->Wait();
                  pipePtr->Delete();
              }
          }),
      blob_file(nullptr,
          [this](void* blobFilePtr) {
              _allocator->free(blobFilePtr);
          }),
      _blobHandle(new BlobHandle_t()),
#endif
      _preFetchBuffer(nullptr,
          [this](uint8_t* buffer) {
              _csramAllocator->free(buffer);
          }),
      _inputBuffer(nullptr,
          [this](uint8_t* buffer) {
              _allocator->free(buffer);
          }),
      _outputBuffer(nullptr, [this](uint8_t* buffer) {
          _allocator->free(buffer);
      }) {
#if defined(__arm__) || defined(__aarch64__)
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

    allocateGraph(_networkDescription->getCompiledNetwork());
#else
    UNUSED(deviceId);
#endif
}

VpualCoreNNExecutor::~VpualCoreNNExecutor() {
#if defined(__arm__) || defined(__aarch64__)
    for (const auto& scratchPtr : _scratchBuffers) {
        _allocator->free(scratchPtr);
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
static std::vector<void*> setScratchHelper(const std::unique_ptr<NnCorePlg, std::function<void(NnCorePlg*)>>& nnCorePtr,
    const unsigned int threadCount, const std::shared_ptr<vpux::Allocator>& allocatorPtr,
    const std::shared_ptr<vpu::Logger>& logger) {
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

static uint8_t* setPrefetchHelper(const std::unique_ptr<NnCorePlg, std::function<void(NnCorePlg*)>>& nnCorePtr,
    const uint32_t preFetchSize, const std::shared_ptr<vpux::Allocator>& allocatorPtr,
    const std::shared_ptr<vpu::Logger>& logger) {
    uint8_t* preFetchVirtAddr = nullptr;
    if (preFetchSize > 0) {
        if (allocatorPtr == nullptr) {
            THROW_IE_EXCEPTION << "prefetchHelper: allocator points to null";
        }
        preFetchVirtAddr = reinterpret_cast<uint8_t*>(allocatorPtr->alloc(preFetchSize));
        if (preFetchVirtAddr == nullptr) {
            THROW_IE_EXCEPTION << "prefetchHelper: failed to allocate " << preFetchSize << " bytes of memory";
        }
        unsigned long preFetchPhysAddr = allocatorPtr->getPhysicalAddress(preFetchVirtAddr);
        uint32_t preFetchAddrLower32Bits = preFetchPhysAddr & 0xffffffff;
        nnCorePtr->SetPrefetchBuffer(preFetchAddrLower32Bits);
    } else {
        logger->info("prefetchHelper: trying to set prefeth buffer with zero size. Skip.");
    }

    return preFetchVirtAddr;
}
}  // namespace
#endif

void VpualCoreNNExecutor::allocateGraph(const std::vector<char>& graphFileContent) {
#if defined(__arm__) || defined(__aarch64__)
    OV_ITT_SCOPED_TASK(vpu::itt::domains::KmbPlugin, "allocateGraph");
    static int graphId_main = 1;
    int nThreads = _config.throughputStreams();

    _logger->info("VpualCoreNNExecutor::allocateGraph begins");

    _blobHandle->graphid = graphId_main++;
    _blobHandle->graphBuff = 0x00000000;
    _blobHandle->graphLen = graphFileContent.size();
    _blobHandle->refCount = 0;

    // allocate memory for graph file
    blob_file.reset(_allocator->alloc(_blobHandle->graphLen));

    if (blob_file == nullptr) {
        _logger->error("VpualCoreNNExecutor::allocateGraph: Error getting CMA for graph");
        THROW_IE_EXCEPTION << "allocateGraph: allocation failed for graph";
    }

    std::memcpy(blob_file.get(), graphFileContent.data(), graphFileContent.size());
    std::memset(static_cast<uint8_t*>(blob_file.get()) + graphFileContent.size(), 0,
        _blobHandle->graphLen - graphFileContent.size());

    // only lower 32 bits have to be used
    // inference runtime cannot address more than that
    _blobHandle->graphBuff = _allocator->getPhysicalAddress(blob_file.get()) & 0xffffffff;

    auto status = _nnCorePlg->Create(_blobHandle.get(), nThreads);
    if (MVNCI_SUCCESS != status) {
        _logger->error("VpualCoreNNExecutor::allocateGraph: failed to create NnCorePlg");
        THROW_IE_EXCEPTION << "VpualCoreNNExecutor::allocateGraph: failed to create NnCorePlg: " << status;
    }

    // pipeline depth means the size of NnExec messages queue
    // when the message is sent and there's some free space in the queue, message is accepted
    // if there isn't, request must wait for vacant spaces
    // number of threads is multiplied by 2 in order to allow requests to be queued up
    // for example, number of executor threads equals to 2 which makes pipeline depth 4
    // two requests can be queued up while other two requests are being processed
    const uint32_t pipelineDepth = nThreads * 2;
    auto xlinkStatus = _nnXlinkPlg->Create(pipelineDepth);
    if (xlinkStatus) {
        _logger->error("VpualCoreNNExecutor::allocateGraph: failed to create NnXlinkPlg");
        THROW_IE_EXCEPTION << "VpualCoreNNExecutor::allocateGraph: failed to create NnXlinkPlg: " << xlinkStatus;
    }

    MvNCIVersion blobVersion;
    status = _nnCorePlg->GetBlobVersion(&blobVersion);
    if (MVNCI_SUCCESS != status) {
        _logger->error("VpualCoreNNExecutor::allocateGraph: failed to get blob version");
        THROW_IE_EXCEPTION << "VpualCoreNNExecutor::allocateGraph: failed to get blob version: " << status;
    }

    const uint32_t upaShaves = _config.numberOfNnCoreShaves();
    if (upaShaves > 0) {
        _logger->debug("VpualCoreNNExecutor::allocateGraph: SetNumUpaShaves to %d", upaShaves);
        _nnCorePlg->SetNumUpaShaves(upaShaves);
    }

    _logger->info("Blob Version: %d %d %d", static_cast<int>(blobVersion.major), static_cast<int>(blobVersion.minor),
        static_cast<int>(blobVersion.patch));
    _scratchBuffers = setScratchHelper(_nnCorePlg, nThreads, _allocator, _logger);
    const uint32_t csramUserSize = _config.CSRAMSize();
    if (csramUserSize != 0) {
        // if user set the size manually, use that amount
        _preFetchBuffer.reset(setPrefetchHelper(_nnCorePlg, csramUserSize, _csramAllocator, _logger));
    } else {
        // otherwise, get the size from NN Core plug-in
        const uint32_t preFetchSize = _nnCorePlg->GetPrefetchBufferSize();
        _preFetchBuffer.reset(setPrefetchHelper(_nnCorePlg, preFetchSize, _csramAllocator, _logger));
    }

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

    _pipe->Start();

    _logger->info("Started FLIC pipeline...");
#else
    UNUSED(graphFileContent);
#endif
}

static ie::Blob::Ptr reallocateBlobToLayoutIgnoringOriginalLayout(const ie::Blob::Ptr& blob,
    const ie::Layout& srcLayout, const ie::Layout& dstLayout, const VpusmmAllocator::Ptr& allocator) {
    if (blob->getTensorDesc().getDims()[1] != 3) {
        THROW_IE_EXCEPTION << "reallocateBlobToLayoutIgnoringOriginalLayout works only with channels == 3";
    }

    // it would be nicer to construct srcTensorDesc from tensorDesc of blob
    // and then call srcTensorDesc.setLayout(srcLayout) but copyBlob does work in that case
    ie::TensorDesc srcTensorDesc = {blob->getTensorDesc().getPrecision(), blob->getTensorDesc().getDims(), srcLayout};
    ie::Blob::Ptr srcBlob = make_blob_with_precision(srcTensorDesc, blob->buffer());

    ie::TensorDesc dstTensorDesc = {blob->getTensorDesc().getPrecision(), blob->getTensorDesc().getDims(), dstLayout};
    ie::Blob::Ptr dstBlob = make_blob_with_precision(dstTensorDesc, allocator);
    if (dstBlob == nullptr) {
        THROW_IE_EXCEPTION
            << "reallocateBlobToLayoutIgnoringOriginalLayout: can't make_blob_with_precision with given params";
    }
    dstBlob->allocate();

    vpu::copyBlob(srcBlob, dstBlob);
    return dstBlob;
}

static ie::Blob::Ptr reallocateBlobToLayout(
    const ie::Blob::Ptr& blob, const ie::Layout& layout, const VpusmmAllocator::Ptr& allocator) {
    ie::TensorDesc dstTensorDesc = {blob->getTensorDesc().getPrecision(), blob->getTensorDesc().getDims(), layout};
    ie::Blob::Ptr kmbBlob = make_blob_with_precision(dstTensorDesc, allocator);
    if (kmbBlob == nullptr) {
        THROW_IE_EXCEPTION << "reallocateBlobToLayout: can't make_blob_with_precision with given params";
    }
    kmbBlob->allocate();

    vpu::copyBlob(blob, kmbBlob);

    return kmbBlob;
}

static bool needRepackForNHWC(const ie::TensorDesc& actualDesc) {
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
    case ie::Layout::NHWC:
    case ie::Layout::NC:
    case ie::Layout::C:
        return false;
    case ie::Layout::NCHW:
        return (actualDims[0] != 1) || (actualDims[1] != 1);
    case ie::Layout::CHW:
        return actualDims[0] != 1;
    default:
        THROW_IE_EXCEPTION << "Unsupported layout for actual blob: " << actualLayout;
    }
}

ie::Blob::Ptr VpualCoreNNExecutor::prepareInputForInference(
    const ie::Blob::Ptr& actualInput, const ie::TensorDesc& deviceDesc) {
    OV_ITT_SCOPED_TASK(vpu::itt::domains::KmbPlugin, "prepareInputForInference");

    // HACK: to overcome inability python API to pass a blob of NHWC layout
    if (_config.repackInputLayout()) {
        _logger->warning("VPUX_VPUAL_REPACK_INPUT_LAYOUT is enabled. Need to do re-layout.");
        return reallocateBlobToLayoutIgnoringOriginalLayout(
            actualInput, ie::Layout::NCHW, ie::Layout::NHWC, _allocator);
    }

    ie::Blob::Ptr inputForInference;
    if (!utils::isBlobAllocatedByAllocator(actualInput, _allocator)) {
        _logger->warning("Input blob is located in non-shareable memory. Need to do re-allocation.");
        inputForInference = utils::reallocateBlob(actualInput, _allocator);
    } else {
        inputForInference = actualInput;
    }

    const auto& actualDesc = actualInput->getTensorDesc();
    const auto& deviceLayout = deviceDesc.getLayout();

    if (needRepackForNHWC(actualDesc) && deviceLayout == ie::Layout::NHWC) {
        _logger->warning("Input blob is inconsistent with network input. Need to do re-layout.");
        // NB: It's possible to make repack data only with the same number of dimensions
        // So just make a view without any copy
        const auto outputMemoryBlob = ie::as<ie::MemoryBlob>(actualInput);
        IE_ASSERT(outputMemoryBlob != nullptr);
        const auto outputMemory = outputMemoryBlob->rmap();
        IE_ASSERT(outputMemory != nullptr);
        const auto outputPtr = outputMemory.as<void*>();
        IE_ASSERT(outputPtr != nullptr);
        ie::Blob::Ptr actualView4D = make_blob_with_precision(vpu::getNCHW(actualInput->getTensorDesc()), outputPtr);
        inputForInference = reallocateBlobToLayout(actualView4D, deviceLayout, _allocator);
    }

    return inputForInference;
}
void VpualCoreNNExecutor::push(const ie::BlobMap& inputs) {
#if defined(__arm__) || defined(__aarch64__)
    OV_ITT_SCOPED_TASK(vpu::itt::domains::KmbPlugin, "push");
    _logger->info("VpualCoreNNExecutor::push started");

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
        auto blob = ie::as<ie::MemoryBlob>(input.second);
        auto memoryHolder = blob->rmap();
        auto inputBufferPhysAddr = _allocator->getPhysicalAddress(memoryHolder.as<uint8_t*>());
        request.inputTensors.push_back(inputBufferPhysAddr);
    }

    for (const auto& inferOutput : _outputPhysAddrs) {
        request.outputTensors.push_back(inferOutput);
    }

    auto status = _nnXlinkPlg->RequestInference(request);
    if (MVNCI_SUCCESS != status) {
        _logger->error("VpualCoreNNExecutor::push: RequestInference failed");
        THROW_IE_EXCEPTION << "VpualCoreNNExecutor::push: RequestInference failed" << status;
    }

    _logger->info("VpualCoreNNExecutor::push finished");
#else
    UNUSED(inputs);
#endif
}

void VpualCoreNNExecutor::push(const InferenceEngine::BlobMap&, const PreprocMap&) {
    THROW_IE_EXCEPTION << "Not implemented";
}

uint32_t VpualCoreNNExecutor::extractPhysAddrForInference(const ie::BlobMap& inputs) {
    uint32_t physAddr = 0;
    if (inputs.size() == 1) {
        auto blob = ie::as<ie::MemoryBlob>(inputs.begin()->second);
        if (blob == nullptr) {
            THROW_IE_EXCEPTION << "Input cannot be cast to memory blob";
        }
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
            auto blob = ie::as<ie::MemoryBlob>(input.second);

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

void VpualCoreNNExecutor::pull(ie::BlobMap& outputs) {
#if defined(__arm__) || defined(__aarch64__)
    OV_ITT_SCOPED_TASK(vpu::itt::domains::KmbPlugin, "pull");
    _logger->info("VpualCoreNNExecutor::pull started");
    NnExecResponseMsg response;
    auto status = _nnXlinkPlg->WaitForResponse(response);
    if (MVNCI_SUCCESS != status) {
        _logger->error("VpualCoreNNExecutor::pull: WaitForResponse failed");
        THROW_IE_EXCEPTION << "VpualCoreNNExecutor::pull: WaitForResponse failed" << status;
    }

    ie::BlobMap deviceOutputs = extractOutputsFromPhysAddr(_outputPhysAddrs.at(0));
    repackDeviceOutputsToNetworkOutputs(deviceOutputs, outputs);
    _logger->info("VpualCoreNNExecutor::pull finished");
#else
    UNUSED(outputs);
#endif
}

ie::BlobMap VpualCoreNNExecutor::extractOutputsFromPhysAddr(uint32_t physAddr) {
    ie::BlobMap deviceOutputs;
    std::size_t offset = physAddr - _allocator->getPhysicalAddress(_outputBuffer.get());
    for (auto&& out : _networkDescription->getDeviceOutputsInfo()) {
        auto desc = out.second->getTensorDesc();
        auto blob = make_blob_with_precision(desc, _outputBuffer.get() + offset);
        deviceOutputs.insert({out.first, blob});
        offset += utils::getByteSize(desc);
    }

    return deviceOutputs;
}

void VpualCoreNNExecutor::repackDeviceOutputsToNetworkOutputs(
    const ie::BlobMap& deviceOutputs, ie::BlobMap& networkOutputs) {
    for (const auto& item : deviceOutputs) {
        const auto& name = item.first;
        const auto& deviceBlob = item.second;
        const auto& deviceDesc = deviceBlob->getTensorDesc();
        const auto& outputBlob = networkOutputs[name];
        const auto& networkDesc = outputBlob->getTensorDesc();

        ie::Blob::Ptr deviceBlobWithNetworkPrecision = nullptr;
        if (deviceDesc.getPrecision() != networkDesc.getPrecision()) {
            _logger->warning("Output blob is inconsistent with network output. "
                             "Need to do convert precision from %d to %d.",
                deviceDesc.getPrecision(), networkDesc.getPrecision());
            deviceBlobWithNetworkPrecision = utils::convertPrecision(deviceBlob, networkDesc.getPrecision());
        } else {
            deviceBlobWithNetworkPrecision = deviceBlob;
        }

        const auto& outputMemoryBlob = ie::as<ie::MemoryBlob>(outputBlob);
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
            const auto actualView4D = make_blob_with_precision(vpu::getNCHW(networkDesc), outputPtr);
            vpu::copyBlob(deviceBlobWithNetworkPrecision, actualView4D);
        } else {
            vpu::copyBlob(deviceBlobWithNetworkPrecision, deviceDesc.getLayout(), outputPtr);
        }
    }
}

void VpualCoreNNExecutor::setup(const ie::ParamMap&) { THROW_IE_EXCEPTION << "Not implemented"; }

bool VpualCoreNNExecutor::isPreProcessingSupported(const InferenceEngine::PreProcessInfo&) const { return false; }

std::map<std::string, ie::InferenceEngineProfileInfo> VpualCoreNNExecutor::getLayerStatistics() {
    THROW_IE_EXCEPTION << "Not implemented";
    return std::map<std::string, ie::InferenceEngineProfileInfo>();
}

InferenceEngine::Parameter VpualCoreNNExecutor::getParameter(const std::string&) const {
    return InferenceEngine::Parameter();
}

}  // namespace vpux
