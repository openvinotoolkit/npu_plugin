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

#include <ie_common.h>

#include <algorithm>
#include <blob_factory.hpp>
#include <cpp_interfaces/exception2status.hpp>
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

#include "kmb_allocator.h"
#include "kmb_config.h"

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
      _logger(std::make_shared<Logger>("KmbExecutor", config.logLevel(), consoleOutput())),
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
    if (!_config.useKmbExecutor()) {
        return;
    }

#if defined(__arm__) || defined(__aarch64__)
    blob_file = nullptr;
    _inferenceId = nullptr;

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

KmbExecutor::~KmbExecutor() { deallocateGraph(); }

void KmbExecutor::initVpualObjects() {
#if defined(__arm__) || defined(__aarch64__)
    OV_ITT_SCOPED_TASK(itt::domains::KmbPlugin, "initVpualObjects");
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
    if (!_inferenceId) {
        _inferenceId.reset(reinterpret_cast<uint32_t*>(_allocator->alloc(sizeof(uint32_t))));
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
    std::vector<void*> physAddrVec;
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
        physAddrVec.push_back(reinterpret_cast<void*>(scratchPhysAddr));
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
    OV_ITT_SCOPED_TASK(itt::domains::KmbPlugin, "allocateGraph");
    initVpualObjects();
    static int graphId_main = 1;
    int nThreads = _config.throughputStreams();
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
    _logger->info("KmbExecutor::allocateGraph: Created RgnAlloc");

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
    UNUSED(graphFileContent);
#endif
}

static Blob::Ptr reallocateBlobToLayoutIgnoringOriginalLayout(
    const Blob::Ptr& blob, const Layout& srcLayout, const Layout& dstLayout, const KmbAllocator::Ptr& allocator) {
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
    const Blob::Ptr& blob, const Layout& layout, const KmbAllocator::Ptr& allocator) {
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

Blob::Ptr KmbExecutor::prepareInputForInference(
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

void KmbExecutor::push(const InferenceEngine::BlobMap& /*inputs*/, const vpux::PreprocMap& /*preProcMap*/) {
    THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
}

void KmbExecutor::push(const InferenceEngine::BlobMap& inputs) {
    if (!_config.useKmbExecutor()) {
        return;
    }

#if defined(__arm__) || defined(__aarch64__)
    OV_ITT_SCOPED_TASK(itt::domains::KmbPlugin, "push");
    _logger->info("KmbExecutor::push started");

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
    _logger->info("KmbExecutor::push finished");
#else
    UNUSED(inputs);
#endif
}

uint32_t KmbExecutor::extractPhysAddrForInference(const BlobMap& inputs) {
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

void KmbExecutor::pull(InferenceEngine::BlobMap& outputs) {
    if (!_config.useKmbExecutor()) {
        return;
    }

#if defined(__arm__) || defined(__aarch64__)
    OV_ITT_SCOPED_TASK(itt::domains::KmbPlugin, "pull");
    _logger->info("KmbExecutor::pull started");
    uint32_t idPhysAddr = 0;
    uint32_t idLength = 0;
    plgInferenceOutput_->PullInferenceID(&idPhysAddr, &idLength);

    uint32_t outputBufferPhysAddr = 0;
    uint32_t outputBufferLength = 0;
    plgTensorOutput_->Pull(&outputBufferPhysAddr, &outputBufferLength);
    // FIXME output->Pull gives only the length of the first tensor
    // need to check if we get buffer of expected size
    BlobMap deviceOutputs = extractOutputsFromPhysAddr(outputBufferPhysAddr);
    repackDeviceOutputsToNetworkOutputs(deviceOutputs, outputs);
    _logger->info("KmbExecutor::pull finished");
#else
    UNUSED(outputs);
#endif
}

BlobMap KmbExecutor::extractOutputsFromPhysAddr(uint32_t physAddr) {
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

void KmbExecutor::repackDeviceOutputsToNetworkOutputs(const ie::BlobMap& deviceOutputs, ie::BlobMap& networkOutputs) {
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

void KmbExecutor::setup(const InferenceEngine::ParamMap&) { THROW_IE_EXCEPTION << "Not implemented"; }

bool KmbExecutor::isPreProcessingSupported(const InferenceEngine::PreProcessInfo&) const { return false; }

std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> KmbExecutor::getLayerStatistics() {
    THROW_IE_EXCEPTION << "Not implemented";
    return std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>();
}

InferenceEngine::Parameter KmbExecutor::getParameter(const std::string&) const { return InferenceEngine::Parameter(); }

void KmbExecutor::deallocateGraph() {
    if (!_config.useKmbExecutor()) {
        return;
    }

#if defined(__arm__) || defined(__aarch64__)
    OV_ITT_SCOPED_TASK(itt::domains::KmbPlugin, "deallocateGraph");
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
