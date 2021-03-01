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

#include "vpual_flic_nn_executor.hpp"

#include "vpux/utils/IE/blob.hpp"
#include "vpux/utils/IE/itt.hpp"
#include "vpux/utils/core/helper_macros.hpp"

#include <ie_common.h>

#include <algorithm>
#include <blob_factory.hpp>
#include <dims_parser.hpp>
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
const uint32_t POOL_SIZE = 30 * 1024 * 1024;
#endif

VpualFlicNNExecutor::VpualFlicNNExecutor(const vpux::NetworkDescription::Ptr& networkDescription,
    const VpusmmAllocator::Ptr& allocator, const uint32_t deviceId, const VpualConfig& config)
    : _networkDescription(networkDescription),
      _allocator(allocator),
      _config(config),
      _logger(std::make_shared<vpu::Logger>("VpualFlicNNExecutor", _config.logLevel(), vpu::consoleOutput())),
      _inputBuffer(nullptr,
          [this](uint8_t* buffer) {
              _allocator->free(buffer);
          }),
      _outputBuffer(nullptr,
          [this](uint8_t* buffer) {
              _allocator->free(buffer);
          }),
      _inferenceId(nullptr, [this](uint32_t* buffer) {
          _allocator->free(buffer);
      }) {
#if defined(__arm__) || defined(__aarch64__)
    blob_file = nullptr;
    _inferenceId = nullptr;

    Byte inputsTotalSize(0);
    for (auto&& in : _networkDescription->getDeviceInputsInfo()) {
        const auto& tensorDesc = in.second->getTensorDesc();
        inputsTotalSize += getMemorySize(tensorDesc);
    }
    _inputBuffer.reset(reinterpret_cast<uint8_t*>(allocator->alloc(inputsTotalSize.count())));
    _logger->debug("Allocated buffer for input with the size: %d", inputsTotalSize);

    // FIXME: allocate real size instead of POOL_SIZE
    _outputBuffer.reset(reinterpret_cast<uint8_t*>(allocator->alloc(POOL_SIZE)));
    _logger->debug("Allocated buffer for output with the size: %d", POOL_SIZE);

    initVpualObjects(deviceId);
    allocateGraph(_networkDescription->getCompiledNetwork());
#else
    VPUX_UNUSED(deviceId);
#endif
}

VpualFlicNNExecutor::~VpualFlicNNExecutor() { deallocateGraph(); }

void VpualFlicNNExecutor::initVpualObjects(const uint32_t deviceId) {
#if defined(__arm__) || defined(__aarch64__)
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "initVpualObjects");
    if (!RgnAlloc) {
        RgnAlloc = std::make_shared<RgnAllocator>(deviceId);
    }
    if (!HeapAlloc) {
        constexpr uint32_t HEAP_ALIGNMENT = 64;
        HeapAlloc = std::make_shared<HeapAllocator>(HEAP_ALIGNMENT, deviceId);
    }
    if (!nnPl) {
        nnPl = std::make_shared<NNFlicPlg>(deviceId);
    }
    if (!gg) {
        gg = std::make_shared<GraphManagerPlg>(deviceId);
    }
    if (!plgTensorInput_) {
        plgTensorInput_ = std::make_shared<PlgTensorSource>(deviceId);
    }
    if (!plgTensorOutput_) {
        plgTensorOutput_ = std::make_shared<PlgStreamResult>(deviceId);
    }
    if (!plgInferenceInput_) {
        plgInferenceInput_ = std::make_shared<PlgInferenceInput>(deviceId);
    }
    if (!plgInferenceOutput_) {
        plgInferenceOutput_ = std::make_shared<PlgInferenceOutput>(deviceId);
    }
    if (!plgPoolOutputs) {
        plgPoolOutputs = std::make_shared<PlgPool<TensorMsg>>(deviceId);
    }
    if (!plgPoolInferenceMsg) {
        plgPoolInferenceMsg = std::make_shared<PlgPool<InferenceMsg>>(deviceId);
    }
    if (!BHandle) {
        BHandle = std::make_shared<BlobHandle_t>();
    }
    if (!pipe) {
        pipe = std::make_shared<Pipeline>(MAX_PLUGS_PER_PIPE, deviceId);
    }
    if (!_inferenceId) {
        _inferenceId.reset(reinterpret_cast<uint32_t*>(_allocator->alloc(sizeof(uint32_t))));
    }
#else
    VPUX_UNUSED(deviceId);
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
    const std::shared_ptr<VpusmmAllocator>& allocatorPtr, const std::shared_ptr<vpu::Logger>& logger) {
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

void VpualFlicNNExecutor::allocateGraph(const std::vector<char>& graphFileContent) {
#if defined(__arm__) || defined(__aarch64__)
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "allocateGraph");
    static int graphId_main = 1;
    int nThreads = _config.throughputStreams();
    int nShaves = 16;

    _logger->info("VpualFlicNNExecutor::allocateGraph begins");

    BHandle->graphid = graphId_main++;
    BHandle->graphBuff = 0x00000000;
    BHandle->graphLen = graphFileContent.size();
    BHandle->refCount = 0;

    // ########################################################################
    // Try and get some CMA allocations.
    // ########################################################################
    blob_file = _allocator->alloc(BHandle->graphLen);

    if (!blob_file) {
        _logger->error("VpualFlicNNExecutor::allocateGraph: Error getting CMA for graph");
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

    _scratchBuffers = setScratchHelper(nnPl, nThreads, _allocator, _logger);

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

    RgnAlloc->Create(_allocator->getPhysicalAddress(_outputBuffer.get()), POOL_SIZE);
    _logger->info("VpualFlicNNExecutor::allocateGraph: Created RgnAlloc");

    const unsigned int shavel2CacheLineSize = 64;
    unsigned int outputTensorSize = ROUND_UP(sumSizeTensorDescOut.totalSize, shavel2CacheLineSize);

    _logger->info("read memory pool finished...");
    plgPoolOutputs->Create(RgnAlloc.get(), 1, 3 * outputTensorSize);
    _logger->info("Created plgPoolOutputs");

    unsigned int inferenceIDSize = ROUND_UP(sizeof(uint32_t), shavel2CacheLineSize);
    plgPoolInferenceMsg->Create(HeapAlloc.get(), 1, 3 * inferenceIDSize);
    _logger->info("Created plgPoolInferenceMsg");

    plgTensorInput_->Create(sumSizeTensorDescIn.totalSize, 0 /*ignored*/, sumSizeTensorDescIn);
    _logger->info("Created plgTensorInput");

    plgTensorOutput_->Create(sumSizeTensorDescOut.totalSize, 0 /*ignored*/, sumSizeTensorDescOut);
    _logger->info("Created plgTensorOutput");

    plgInferenceInput_->Create(3 * inferenceIDSize, 0 /*ignored*/);
    _logger->info("Created plgInferenceInput_");

    plgInferenceOutput_->Create(3 * inferenceIDSize, 0 /*ignored*/);
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
    VPUX_UNUSED(graphFileContent);
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

ie::Blob::Ptr VpualFlicNNExecutor::prepareInputForInference(
    const ie::Blob::Ptr& actualInput, const ie::TensorDesc& deviceDesc) {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "prepareInputForInference");

    // HACK: to overcome inability python API to pass a blob of NHWC layout
    if (_config.repackInputLayout()) {
        _logger->warning("VPUX_VPUAL_REPACK_INPUT_LAYOUT is enabled. Need to do re-layout.");
        return reallocateBlobToLayoutIgnoringOriginalLayout(
            actualInput, ie::Layout::NCHW, ie::Layout::NHWC, _allocator);
    }

    ie::Blob::Ptr inputForInference;
    if (!utils::isBlobAllocatedByAllocator(actualInput, _allocator)) {
        _logger->warning("Input blob is located in non-shareable memory. Need to do re-allocation.");
        inputForInference = reallocateBlob(ie::as<ie::MemoryBlob>(actualInput), _allocator);
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
void VpualFlicNNExecutor::push(const ie::BlobMap& inputs) {
#if defined(__arm__) || defined(__aarch64__)
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "push");
    _logger->info("VpualFlicNNExecutor::push started");

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

    auto inputBufferPhysAddr = extractPhysAddrForInference(updatedInputs);
    plgTensorInput_->Push(inputBufferPhysAddr, inputsByteSize);

    *_inferenceId = 1;
    plgInferenceInput_->PushInferenceID(_allocator->getPhysicalAddress(_inferenceId.get()), sizeof(uint32_t));
    _logger->info("VpualFlicNNExecutor::push finished");
#else
    VPUX_UNUSED(inputs);
#endif
}

void VpualFlicNNExecutor::push(const InferenceEngine::BlobMap&, const PreprocMap&) {
    THROW_IE_EXCEPTION << "Not implemented";
}

uint32_t VpualFlicNNExecutor::extractPhysAddrForInference(const ie::BlobMap& inputs) {
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

void VpualFlicNNExecutor::pull(ie::BlobMap& outputs) {
#if defined(__arm__) || defined(__aarch64__)
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "pull");
    _logger->info("VpualFlicNNExecutor::pull started");
    uint32_t idPhysAddr = 0;
    uint32_t idLength = 0;
    // TODO: pass inference timeout from config
    auto result = plgInferenceOutput_->PullInferenceID(&idPhysAddr, &idLength/*,  _config->inferenceTimeoutMs()*/);
    if (0 != result) {
        _logger->error("VpualFlicNNExecutor::pull: failed");
        THROW_IE_EXCEPTION << "VpualFlicNNExecutor::pull: PullInferenceID failed" << result;
    }

    uint32_t outputBufferPhysAddr = 0;
    uint32_t outputBufferLength = 0;
    result = plgTensorOutput_->Pull(&outputBufferPhysAddr, &outputBufferLength/*,  _config->inferenceTimeoutMs()*/);

    if (0 != result) {
        _logger->error("VpualFlicNNExecutor::pull: failed");
        THROW_IE_EXCEPTION << "VpualFlicNNExecutor::pull: WaitForResponse failed" << 0;
    }

    // FIXME output->Pull gives only the length of the first tensor
    // need to check if we get buffer of expected size
    ie::BlobMap deviceOutputs = extractOutputsFromPhysAddr(outputBufferPhysAddr);
    repackDeviceOutputsToNetworkOutputs(deviceOutputs, outputs);
    _logger->info("VpualFlicNNExecutor::pull finished");
#else
    VPUX_UNUSED(outputs);
#endif
}

ie::BlobMap VpualFlicNNExecutor::extractOutputsFromPhysAddr(uint32_t physAddr) {
    ie::BlobMap deviceOutputs;
    Byte offset(physAddr - _allocator->getPhysicalAddress(_outputBuffer.get()));
    for (auto&& out : _networkDescription->getDeviceOutputsInfo()) {
        auto desc = out.second->getTensorDesc();
        auto blob = make_blob_with_precision(desc, _outputBuffer.get() + offset.count());
        deviceOutputs.insert({out.first, blob});
        offset += getMemorySize(desc);
    }

    return deviceOutputs;
}

void VpualFlicNNExecutor::repackDeviceOutputsToNetworkOutputs(
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
            deviceBlobWithNetworkPrecision = toPrecision(ie::as<ie::MemoryBlob>(deviceBlob), networkDesc.getPrecision());
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

void VpualFlicNNExecutor::setup(const ie::ParamMap&) { THROW_IE_EXCEPTION << "Not implemented"; }

bool VpualFlicNNExecutor::isPreProcessingSupported(const PreprocMap&) const { return false; }

std::map<std::string, ie::InferenceEngineProfileInfo> VpualFlicNNExecutor::getLayerStatistics() {
    THROW_IE_EXCEPTION << "Not implemented";
    return std::map<std::string, ie::InferenceEngineProfileInfo>();
}

InferenceEngine::Parameter VpualFlicNNExecutor::getParameter(const std::string&) const {
    return InferenceEngine::Parameter();
}

void VpualFlicNNExecutor::deallocateGraph() {
#if defined(__arm__) || defined(__aarch64__)
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "deallocateGraph");
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
    if (plgInferenceInput_) {
        plgInferenceInput_->Delete();
    }
    if (plgInferenceOutput_) {
        plgInferenceOutput_->Delete();
    }
    if (plgPoolInferenceMsg) {
        plgPoolInferenceMsg->Delete();
    }

    for (const auto& scratchPtr : _scratchBuffers) {
        _allocator->free(scratchPtr);
    }
#endif
}

}  // namespace vpux
