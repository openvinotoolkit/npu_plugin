//
// Copyright 2019-2021 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpual_core_nn_executor.hpp"

#include "vpux/utils/IE/blob.hpp"
#include "vpux/utils/IE/format.hpp"
#include "vpux/utils/IE/itt.hpp"
#include "vpux/utils/core/helper_macros.hpp"

#include <ie_common.h>

#include <algorithm>
#include <blob_factory.hpp>
#include <dims_parser.hpp>
#include <map>
#include <tensor_ref_helpers.hpp>
#include <utility>
#include <vector>

#include "vpux/utils/plugin/profiling_parser.hpp"
#include "vpux/utils/core/enums.hpp"

#if (defined(__arm__) || defined(__aarch64__)) && defined(VPUX_DEVELOPER_BUILD)
#include "mmapped_pointer.hpp"

#include <errno.h>
#include <mvLog.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <atomic>
#include <cstdio>
#include <mutex>
#include <thread>

#if __has_include("consoleTxQueue.h")
#include "consoleTxQueue.h"
#else
#define MV_CONSOLE_TX_QUEUE 0x0000000094400040
#endif

#endif

#include "vpual_config.hpp"
#include "vpux/al/config/common.hpp"
#include "vpux/al/config/runtime.hpp"

#include "vpusmm_allocator.hpp"

namespace ie = InferenceEngine;

using ie::VPUXConfigParams::ProfilingOutputTypeArg;
using vpux::profiling::OutputType;

namespace vpux {

constexpr int VPU_CSRAM_DEVICE_ID = 32;

#if defined(__arm__) || defined(__aarch64__)

constexpr uint16_t XLINK_IPC_CHANNELS = 1024;

void VpualCoreNNExecutor::initWatchDog() {
    _wd.reset(new WatchDog(_config.get<INFERENCE_TIMEOUT_MS>(), _logger, [this]() {
        _logger.error("{0} milliseconds have passed, closing xlink channels.", _config.get<INFERENCE_TIMEOUT_MS>());
        auto xhndl{getXlinkDeviceHandle(_nnXlinkPlg->getDeviceId())};
        for (uint16_t i{0}; i < XLINK_IPC_CHANNELS; ++i) {
            xlink_close_channel(&xhndl, i);
        }
    }));
}

#ifdef VPUX_DEVELOPER_BUILD

class VpualCoreNNExecutor::PipePrintHandler final {
public:
    static std::shared_ptr<PipePrintHandler> get();

public:
    ~PipePrintHandler();

private:
    static constexpr size_t VPU_CACHE_LINE_SIZE = 64;
    static constexpr size_t PAGE_SIZE = 4096;
    static constexpr uint64_t PIPEPRINT_CANARY_START = 0x22334455;
    static constexpr uint64_t PIPEPRINT_CANARY_END = 0xBBCCDDEE;

    struct tyMvConsoleQueue {
        volatile uint32_t canaryStart;
        volatile uint32_t in;
        volatile uint32_t out;
        volatile uint32_t queueSize;
        volatile uint32_t queuePtr;
        volatile uint32_t canaryEnd;
    };

private:
    PipePrintHandler();

    static void threadBody(PipePrintHandler* obj);

    static constexpr size_t alignDown(size_t val, size_t size) {
        return ((val) & (~(size - 1)));
    }

private:
    std::atomic<bool> _enabled{false};
    std::thread _thread;

    static std::weak_ptr<PipePrintHandler> _globalObj;
    static std::mutex _globalMtx;
};

std::weak_ptr<VpualCoreNNExecutor::PipePrintHandler> VpualCoreNNExecutor::PipePrintHandler::_globalObj;
std::mutex VpualCoreNNExecutor::PipePrintHandler::_globalMtx;

std::shared_ptr<VpualCoreNNExecutor::PipePrintHandler> VpualCoreNNExecutor::PipePrintHandler::get() {
    if (const auto* env = std::getenv("IE_VPUX_ENABLE_PIPEPRINT")) {
        if (std::stoi(env) != 0) {
            std::unique_lock<std::mutex> lock(_globalMtx);

            auto obj = _globalObj.lock();
            if (obj != nullptr) {
                return obj;
            }

            obj.reset(new PipePrintHandler);
            _globalObj = obj;
            return obj;
        }
    }

    return nullptr;
}

VpualCoreNNExecutor::PipePrintHandler::PipePrintHandler() {
    _enabled = true;
    _thread = std::thread(threadBody, this);
}

VpualCoreNNExecutor::PipePrintHandler::~PipePrintHandler() {
    try {
        if (_thread.joinable()) {
            _enabled = false;
            _thread.join();
        }
    } catch (...) {
        std::cerr << "Got an error during pipeprint thread join" << std::endl;
    }
}

void VpualCoreNNExecutor::PipePrintHandler::threadBody(PipePrintHandler* obj) {
    uint64_t phyAddr = MV_CONSOLE_TX_QUEUE;

    if (const auto* env = std::getenv("IE_VPUX_PIPEPRINT_PHY_ADDR")) {
        phyAddr = std::stoull(env);
    }
    MMappedPtr<tyMvConsoleQueue> header(phyAddr);
    MMappedPtr<uint8_t> queBuffer(static_cast<uint64_t>(header->queuePtr), header->queueSize);
    MMappedPtr<volatile uint32_t> inputCounterPtr(static_cast<uint64_t>(header->in));

    if (PIPEPRINT_CANARY_START != header->canaryStart) {
        std::cerr << "Invalid start Canary at given address: expected to be " + std::to_string(PIPEPRINT_CANARY_START);
        return;
    }
    if (PIPEPRINT_CANARY_END != header->canaryEnd) {
        std::cerr << "Invalid end Canary at given address: expected to be " + std::to_string(PIPEPRINT_CANARY_END);
        return;
    }

    uint32_t no_data_ticks = 0;

    auto in = 0;
    auto queueSize = header->queueSize;
    auto queuePtr = header->queuePtr;

    while (obj->_enabled) {
        const auto nextOffset = *(*inputCounterPtr);
        auto cnt = (nextOffset - in) % queueSize;

        if (cnt > 0) {
            // only 64bit aligned part is flushed from cache to RAM in time
            // the rest part will be flushed later by subsequent logs or forcefully with timeout
            const auto count_caligned =
                    (alignDown((queuePtr + nextOffset), VPU_CACHE_LINE_SIZE) - ((queuePtr + in))) % queueSize;

            if (count_caligned != 0)
                cnt = count_caligned;
            else if (no_data_ticks < 10000)
                cnt = 0;

            if (cnt != 0) {
                const auto res = write(STDOUT_FILENO, *queBuffer + in, cnt);
                (void)res;

                std::fputs(ANSI_COLOR_RESET, stdout);
                std::fflush(stdout);

                in = (in + cnt) % queueSize;
                no_data_ticks = 0;
                continue;
            }
        }

        // 1ms sleep when no logs are presented.
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        no_data_ticks++;
    }
    std::cout << std::endl;
}

#endif  // VPUX_DEVELOPER_BUILD

#endif

static vpux::profiling::OutputType argToProfType(const ie::VPUXConfigParams::ProfilingOutputTypeArg argument) {
    switch (argument) {
    case ProfilingOutputTypeArg::NONE:
        return OutputType::NONE;
    case ProfilingOutputTypeArg::JSON:
        return OutputType::JSON;
    case ProfilingOutputTypeArg::TEXT:
        return OutputType::TEXT;
    }
    IE_THROW() << "Unknown profiling output type.";
}

VpualCoreNNExecutor::VpualCoreNNExecutor(const vpux::NetworkDescription::Ptr& networkDescription,
                                         const VpusmmAllocator::Ptr& allocator, const uint32_t deviceId,
                                         const InferenceEngine::VPUXConfigParams::VPUXPlatform& platform,
                                         const Config& config)
        : _networkDescription(networkDescription),
          _allocator(allocator),
          _csramAllocator(std::make_shared<VpusmmAllocator>(VPU_CSRAM_DEVICE_ID)),
          _deallocator([this](uint8_t* buffer) {
              if (_allocator != nullptr) {
                  _allocator->free(buffer);
              }
          }),
          _csramDeallocator([this](uint8_t* buffer) {
              if (_csramAllocator != nullptr) {
                  _csramAllocator->free(buffer);
              }
          }),
          _config(config),
          _logger("VpualCoreNNExecutor", _config.get<LOG_LEVEL>()),
          _profilingType(argToProfType(_config.get<PRINT_PROFILING>())),
          _profilingOutputFile(_config.get<PROFILING_OUTPUT_FILE>()),
#if defined(__arm__) || defined(__aarch64__)
          _nnXlinkPlg(new NnXlinkPlg(deviceId)),
          _nnCorePlg(new NnCorePlg(deviceId),
                     [](NnCorePlg* nnCorePlgPtr) {
                         if (nnCorePlgPtr != nullptr) {
                             nnCorePlgPtr->Delete();
                             delete nnCorePlgPtr;
                         }
                     }),
          _vpualSyncImpl(VpualSyncXLinkImpl(_nnXlinkPlg)),
          _nnSync(new VpualCoreNNSynchronizer<VpualSyncXLinkImpl>(_vpualSyncImpl, _logger)),
          _pipe(new Pipeline(MAX_PLUGS_PER_PIPE, deviceId),
                [](Pipeline* pipePtr) {
                    if (pipePtr != nullptr) {
                        try {
                            pipePtr->Stop();
                            pipePtr->Wait();
                            pipePtr->Delete();
                        } catch (const std::exception& ex) {
                            std::cerr << "Pipeline object destruction failed: " << ex.what() << std::endl;
                        } catch (...) {
                            std::cerr << "Pipeline object destruction failed: unexpected exception." << std::endl;
                        }
                        delete pipePtr;
                    }
                }),
          blob_file(nullptr,
                    [this](void* blobFilePtr) {
                        if (_allocator != nullptr) {
                            _allocator->free(blobFilePtr);
                        }
                    }),
          _blobHandle(new BlobHandle_t()),
#endif
          _preFetchBuffer(nullptr, _csramDeallocator),
          _inputBuffer(nullptr, _deallocator),
          _outputBuffer(nullptr, _deallocator),
          _profilingOutputBuffer(nullptr, _deallocator),
          _platform(platform) {
#if defined(__arm__) || defined(__aarch64__)
#ifdef VPUX_DEVELOPER_BUILD
    _pipePrint = PipePrintHandler::get();
#endif

    Byte inputsTotalSize(0);
    for (auto&& in : _networkDescription->getDeviceInputsInfo()) {
        const auto& tensorDesc = in.second->getTensorDesc();
        inputsTotalSize += getMemorySize(tensorDesc);
    }
    _inputBuffer.reset(reinterpret_cast<uint8_t*>(_allocator->alloc(inputsTotalSize.count())));
    _logger.debug("Allocated buffer for input with the size: {0}", inputsTotalSize);

    allocateGraph(_networkDescription->getCompiledNetwork());
    initWatchDog();
#else
    VPUX_UNUSED(deviceId);
#endif
}

#if defined(__arm__) || defined(__aarch64__)
VpualCoreNNExecutor::VpualCoreNNExecutor(
        const vpux::NetworkDescription::Ptr& networkDescription, const VpusmmAllocator::Ptr& allocator,
        const std::shared_ptr<NnXlinkPlg>& other_nnXlinkPlg, const std::shared_ptr<NnCorePlg>& other_nnCorePlg,
        const std::shared_ptr<VpualCoreNNSynchronizer<VpualSyncXLinkImpl>>& other_nnSync,
        const std::shared_ptr<Pipeline>& other_pipe, const std::shared_ptr<WatchDog>& wd, const Config& config)
        : _networkDescription(networkDescription),
          _allocator(allocator),
          _csramAllocator(std::make_shared<VpusmmAllocator>(VPU_CSRAM_DEVICE_ID)),
          _deallocator([this](uint8_t* buffer) {
              if (_allocator != nullptr) {
                  _allocator->free(buffer);
              }
          }),
          _csramDeallocator([this](uint8_t* buffer) {
              if (_csramAllocator != nullptr) {
                  _csramAllocator->free(buffer);
              }
          }),
          _config(config),
          _logger("VpualCoreNNExecutor", _config.get<LOG_LEVEL>()),
          _profilingType(argToProfType(_config.get<PRINT_PROFILING>())),
          _profilingOutputFile(_config.get<PROFILING_OUTPUT_FILE>()),
          _nnXlinkPlg(other_nnXlinkPlg),
          _nnCorePlg(other_nnCorePlg),
          _vpualSyncImpl(other_nnXlinkPlg),
          _nnSync(other_nnSync),
          _pipe(other_pipe),
          blob_file(nullptr,
                    [this](void* blobFilePtr) {
                        if (_allocator != nullptr) {
                            _allocator->free(blobFilePtr);
                        }
                    }),
          _blobHandle(new BlobHandle_t()),
          _preFetchBuffer(nullptr, _csramDeallocator),
          _inputBuffer(nullptr, _deallocator),
          _outputBuffer(nullptr, _deallocator),
          _profilingOutputBuffer(nullptr, _deallocator) {
#ifdef VPUX_DEVELOPER_BUILD
    _pipePrint = PipePrintHandler::get();
#endif

    // need _execInferId to be a unique-id, but only across the current set of
    // executor clones..
    static unsigned int gExecInferId = 1;
    _execInferId = gExecInferId++;

    Byte inputsTotalSize(0);
    for (auto&& in : _networkDescription->getDeviceInputsInfo()) {
        const auto& tensorDesc = in.second->getTensorDesc();
        inputsTotalSize += getMemorySize(tensorDesc);
    }
    _inputBuffer.reset(reinterpret_cast<uint8_t*>(_allocator->alloc(inputsTotalSize.count())));
    _logger.debug("Allocated buffer for input with the size: {0}", inputsTotalSize);

    const size_t outputsSize = _nnCorePlg->GetNumberOfOutputs();
    size_t outputTotalSize = 0;
    for (size_t outputIdx = 0; outputIdx < outputsSize; outputIdx++) {
        TensorRefNDData descOut = _nnCorePlg->GetOutputTensorRef(outputIdx);
        outputTotalSize += vpu::utils::getTotalSize(descOut);
    }

    _outputBuffer.reset(reinterpret_cast<uint8_t*>(_allocator->alloc(outputTotalSize)));
    _logger.debug("Allocated buffer for output with the size: {0}", outputTotalSize);

    off_t outputOffset = 0;
    auto outputAddress = _allocator->getPhysicalAddress(_outputBuffer.get());
    for (size_t outputIdx = 0; outputIdx < outputsSize; outputIdx++) {
        TensorRefNDData descOut = _nnCorePlg->GetOutputTensorRef(outputIdx);

        _outputPhysAddrs.push_back(outputAddress + outputOffset);
        outputOffset += vpu::utils::getTotalSize(descOut);
    }

    const size_t profilingOutputsSize = _nnCorePlg->GetNumberOfProfilingOutputs();
    size_t profilingOutputTotalSize = 0;
    for (size_t profilingOutputIdx = 0; profilingOutputIdx < profilingOutputsSize; profilingOutputIdx++) {
        const flicTensorDescriptor_t descOut = _nnCorePlg->GetProfilingOutputTensorDescriptor(profilingOutputIdx);
        profilingOutputTotalSize += descOut.totalSize;
    }

    _profilingOutputBuffer.reset(reinterpret_cast<uint8_t*>(_allocator->alloc(profilingOutputTotalSize)));
    _logger.debug("Allocated buffer for profiling output with the size: {0}", profilingOutputTotalSize);

    off_t profilingOutputOffset = 0;
    auto profilingOutputAddress = _allocator->getPhysicalAddress(_profilingOutputBuffer.get());
    for (size_t profilingOutputIdx = 0; profilingOutputIdx < profilingOutputsSize; profilingOutputIdx++) {
        const flicTensorDescriptor_t descOut = _nnCorePlg->GetProfilingOutputTensorDescriptor(profilingOutputIdx);

        _profilingOutputPhysAddrs.push_back(profilingOutputAddress + profilingOutputOffset);
        profilingOutputOffset += descOut.totalSize;
    }
    _wd = wd;
}  // namespace vpux
#endif

VpualCoreNNExecutor::~VpualCoreNNExecutor() {
#if defined(__arm__) || defined(__aarch64__)
    if (_allocator != nullptr) {
        for (const auto& scratchPtr : _scratchBuffers) {
            _allocator->free(scratchPtr);
        }
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
                                           const std::shared_ptr<vpux::Allocator>& allocatorPtr, Logger logger) {
    if (threadCount > 1) {
        logger.warning("scratchHelper: trying to set scratch buffer to {0} threads.", threadCount);
    }
    uint32_t memoryReqs = nnCorePtr->GetScratchBufferSize();
    logger.info("scratchHelper: GetMemoryRequirements returned {0}", memoryReqs);
    constexpr uint32_t minimalScratchSize = 1024 * 1024;
    if (memoryReqs < minimalScratchSize) {
        memoryReqs = minimalScratchSize;
    }

    std::vector<void*> virtAddrVec;
    virtAddrVec.reserve(threadCount);
    std::vector<NnCorePlg::Buffer> physAddrVec;
    physAddrVec.reserve(threadCount);
    for (unsigned int threadIdx = 0; threadIdx < threadCount; threadIdx++) {
        uint8_t* scratchVirtAddr = reinterpret_cast<uint8_t*>(allocatorPtr->alloc(memoryReqs));
        if (scratchVirtAddr == nullptr) {
            IE_THROW() << "scratchHelper: failed to allocate " << memoryReqs << " bytes of memory";
        }
        unsigned long scratchPhysAddr = allocatorPtr->getPhysicalAddress(scratchVirtAddr);
        if (scratchPhysAddr == 0) {
            IE_THROW() << "scratchHelper: failed to get physical address";
        }
        // NB: narrowing unsigned long (uint64_t on 64-bit Yocto) to uint32_t here
        physAddrVec.push_back({scratchPhysAddr, memoryReqs});
        virtAddrVec.push_back(scratchVirtAddr);
    }

    nnCorePtr->SetScratchBuffers(physAddrVec);
    return virtAddrVec;
}

static uint8_t* setPrefetchHelper(const std::shared_ptr<NnCorePlg>& nnCorePtr, const uint32_t preFetchSize,
                                  const std::shared_ptr<vpux::Allocator>& allocatorPtr, Logger logger) {
    uint8_t* preFetchVirtAddr = nullptr;
    if (preFetchSize > 0) {
        if (allocatorPtr == nullptr) {
            IE_THROW() << "prefetchHelper: allocator points to null";
        }
        preFetchVirtAddr = reinterpret_cast<uint8_t*>(allocatorPtr->alloc(preFetchSize));
        if (preFetchVirtAddr == nullptr) {
            IE_THROW() << "prefetchHelper: failed to allocate " << preFetchSize << " bytes of memory";
        }
        unsigned long preFetchPhysAddr = allocatorPtr->getPhysicalAddress(preFetchVirtAddr);
        uint32_t preFetchAddrLower32Bits = preFetchPhysAddr & 0xffffffff;
        nnCorePtr->SetPrefetchBuffer({preFetchAddrLower32Bits, preFetchSize});
    } else {
        logger.info("prefetchHelper: trying to set prefeth buffer with zero size. Skip.");
    }

    return preFetchVirtAddr;
}

const static vpux::EnumSet<InferenceEngine::VPUXConfigParams::VPUXPlatform> platformsWithCSRAM = {
        InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3800,
        InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3900,
};
}  // namespace
#endif

void VpualCoreNNExecutor::allocateGraph(const std::vector<char>& graphFileContent) {
#if defined(__arm__) || defined(__aarch64__)
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "allocateGraph");
    static int graphId_main = 1;
    int nThreads = _config.get<THROUGHPUT_STREAMS>();
    if (nThreads < 0) {
        if (_config.has<PERFORMANCE_HINT>()) {
            switch (_config.get<PERFORMANCE_HINT>()) {
            case PerformanceHint::Latency:
                nThreads = 3;
                break;
            case PerformanceHint::Throughput:
            default:
                nThreads = 6;
                break;
            }
        } else {
            nThreads = 6;  // TODO: consider updating once multi-clustering is enabled in compiler
        }
    }

    _logger.info("allocateGraph begins");

    _blobHandle->graphid = graphId_main++;
    _blobHandle->graphBuff = 0x00000000;
    _blobHandle->graphLen = graphFileContent.size();
    _blobHandle->refCount = 0;

    // allocate memory for graph file
    blob_file.reset(_allocator->alloc(_blobHandle->graphLen));

    if (blob_file == nullptr) {
        _logger.error("allocateGraph: Error getting CMA for graph");
        IE_THROW() << "allocateGraph: allocation failed for graph";
    }

    std::memcpy(blob_file.get(), graphFileContent.data(), graphFileContent.size());
    std::memset(static_cast<uint8_t*>(blob_file.get()) + graphFileContent.size(), 0,
                _blobHandle->graphLen - graphFileContent.size());

    // only lower 32 bits have to be used
    // inference runtime cannot address more than that
    _blobHandle->graphBuff = _allocator->getPhysicalAddress(blob_file.get()) & 0xffffffff;

    auto status = _nnCorePlg->Create(*_blobHandle, nThreads);
    if (MVNCI_SUCCESS != status) {
        _logger.error("allocateGraph: failed to create NnCorePlg");
        IE_THROW() << "VpualCoreNNExecutor::allocateGraph: failed to create NnCorePlg: " << status;
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
        _logger.error("VpualCoreNNExecutor::allocateGraph: failed to create NnXlinkPlg");
        IE_THROW() << "VpualCoreNNExecutor::allocateGraph: failed to create NnXlinkPlg: " << xlinkStatus;
    }

    MvNCIVersion blobVersion;
    status = _nnCorePlg->GetBlobVersion(blobVersion);
    if (MVNCI_SUCCESS != status) {
        _logger.error("allocateGraph: failed to get blob version");
        IE_THROW() << "VpualCoreNNExecutor::allocateGraph: failed to get blob version: " << status;
    }

    const uint32_t upaShaves = _config.get<INFERENCE_SHAVES>();
    if (upaShaves > 0) {
        _logger.debug("::allocateGraph: SetNumUpaShaves to {0}", upaShaves);
        _nnCorePlg->SetNumUpaShaves(upaShaves);
    }

    _logger.info("Blob Version: {0} {1} {2}", blobVersion.major, blobVersion.minor, blobVersion.patch);
    _scratchBuffers = setScratchHelper(_nnCorePlg, nThreads, _allocator, _logger);
    auto detectedPlatform = _platform;
    auto configPlatform = _config.get<PLATFORM>();
    auto targetPlatform = InferenceEngine::VPUXConfigParams::VPUXPlatform::AUTO;
    if (configPlatform == InferenceEngine::VPUXConfigParams::VPUXPlatform::AUTO) {
        // use detected platfrom when auto detect is set
        targetPlatform = detectedPlatform;
    } else {
        // alternatively, use platform from user config
        targetPlatform = configPlatform;
    }

    const auto csramUserSize = _config.get<CSRAM_SIZE>();
    const bool platformHasCSRAM =
            std::any_of(platformsWithCSRAM.begin(), platformsWithCSRAM.end(),
                        [targetPlatform](const InferenceEngine::VPUXConfigParams::VPUXPlatform& platform) -> bool {
                            return targetPlatform == platform;
                        });
    uint32_t preFetchSize = 0;
    if (csramUserSize > 0) {
        if (platformHasCSRAM) {
            // if user set the size manually, use that amount
            preFetchSize = static_cast<uint32_t>(csramUserSize);
        } else {
            _logger.warning("VPUX_CSRAM_SIZE is not equal to zero, but the platform cannot allocate CSRAM");
        }
    } else if (csramUserSize < 0) {
        // otherwise, get the size from NN Core plug-in
        preFetchSize = _nnCorePlg->GetPrefetchBufferSize();
    }

    if (platformHasCSRAM && preFetchSize) {
        _preFetchBuffer.reset(setPrefetchHelper(_nnCorePlg, preFetchSize, _csramAllocator, _logger));
    }

    auto tensorDeserializer = [&](const TensorRefNDData& descriptor) -> void {
        _logger.info("{ Shape: {0}, Strides: {1}, DType: {2}, Order: {3} }", vpu::utils::serializeShape(descriptor),
                     vpu::utils::serializeStrides(descriptor), vpu::utils::serializeDType(descriptor),
                     vpu::utils::serializeOrder(descriptor));
    };
    auto tensorDeserializerFlic = [&](const flicTensorDescriptor_t& descriptor) -> void {
        _logger.info("{ n: {0}, c: {1}, h: {2}, w: {3}, totalSize: {4}, widthStride: {5}, heightStride: {6}, "
                     "channelsStride: {7}}",
                     descriptor.n, descriptor.c, descriptor.h, descriptor.w, descriptor.totalSize,
                     descriptor.widthStride, descriptor.heightStride, descriptor.channelsStride);
    };

    _logger.info("Deserializing descriptors:");
    size_t inputsSize = _nnCorePlg->GetNumberOfInputs();
    for (size_t inputIdx = 0; inputIdx < inputsSize; inputIdx++) {
        TensorRefNDData descIn = _nnCorePlg->GetInputTensorRef(inputIdx);
        _logger.info("Input: {0}", inputIdx);
        tensorDeserializer(descIn);
    }

    size_t outputsSize = _nnCorePlg->GetNumberOfOutputs();
    size_t outputTotalSize = 0;
    for (size_t outputIdx = 0; outputIdx < outputsSize; outputIdx++) {
        TensorRefNDData descOut = _nnCorePlg->GetOutputTensorRef(outputIdx);
        _logger.info("Output: {0}", outputIdx);
        tensorDeserializer(descOut);

        outputTotalSize += vpu::utils::getTotalSize(descOut);
    }

    _outputBuffer.reset(reinterpret_cast<uint8_t*>(_allocator->alloc(outputTotalSize)));
    _logger.debug("Allocated buffer for output with the size: {0}", outputTotalSize);

    off_t outputOffset = 0;
    for (size_t outputIdx = 0; outputIdx < outputsSize; outputIdx++) {
        TensorRefNDData descOut = _nnCorePlg->GetOutputTensorRef(outputIdx);

        auto outPhysAddr = _allocator->getPhysicalAddress(_outputBuffer.get()) + outputOffset;
        _outputPhysAddrs.push_back(outPhysAddr);
        outputOffset += vpu::utils::getTotalSize(descOut);
    }

    size_t profilingOutputsSize = _nnCorePlg->GetNumberOfProfilingOutputs();
    size_t profilingOutputTotalSize = 0;
    for (size_t profilingOutputIdx = 0; profilingOutputIdx < profilingOutputsSize; profilingOutputIdx++) {
        flicTensorDescriptor_t descOut = _nnCorePlg->GetProfilingOutputTensorDescriptor(profilingOutputIdx);
        _logger.info("Output: {0}", profilingOutputIdx);
        tensorDeserializerFlic(descOut);

        profilingOutputTotalSize += descOut.totalSize;
    }

    _profilingOutputBuffer.reset(reinterpret_cast<uint8_t*>(_allocator->alloc(profilingOutputTotalSize)));
    _logger.debug("Allocated buffer for profiling output with the size: {0}", profilingOutputTotalSize);

    off_t profilingOutputOffset = 0;
    for (size_t profilingOutputIdx = 0; profilingOutputIdx < profilingOutputsSize; profilingOutputIdx++) {
        flicTensorDescriptor_t descOut = _nnCorePlg->GetProfilingOutputTensorDescriptor(profilingOutputIdx);

        auto outPhysAddr = _allocator->getPhysicalAddress(_profilingOutputBuffer.get()) + profilingOutputOffset;
        _profilingOutputPhysAddrs.push_back(outPhysAddr);
        profilingOutputOffset += descOut.totalSize;
    }

    _nnCorePlg->PrepareNetwork();

    _pipe->Add(_nnCorePlg.get());
    _pipe->Add(_nnXlinkPlg.get());
    _nnXlinkPlg->requestOut.Link(&_nnCorePlg->requestInput);
    _nnCorePlg->resultOut.Link(&_nnXlinkPlg->resultIn);

    _pipe->Start();

    _logger.info("Started FLIC pipeline...");
#else
    VPUX_UNUSED(graphFileContent);
#endif
}

static ie::MemoryBlob::Ptr reallocateBlobToLayoutIgnoringOriginalLayout(const ie::MemoryBlob::Ptr& blob,
                                                                        ie::Layout srcLayout, ie::Layout dstLayout,
                                                                        const VpusmmAllocator::Ptr& allocator) {
    if (blob->getTensorDesc().getDims()[1] != 3) {
        IE_THROW() << "reallocateBlobToLayoutIgnoringOriginalLayout works only with channels == 3";
    }

    const auto blobMem = blob->rmap();

    // it would be nicer to construct srcTensorDesc from tensorDesc of blob
    // and then call srcTensorDesc.setLayout(srcLayout) but copyBlob does work in that case
    ie::TensorDesc srcTensorDesc = {blob->getTensorDesc().getPrecision(), blob->getTensorDesc().getDims(), srcLayout};
    const auto srcBlob = makeBlob(srcTensorDesc, nullptr, blobMem.as<void*>());

    return toLayout(srcBlob, dstLayout, allocator);
}

static bool needRepackForNHWC(const ie::TensorDesc& actualDesc) {
    /* NB: Brief overview:
     * Runtime works only with NHWC layout, but actual input layout can be different
     * therefore it should be repacked, let's to observe cases:
         1) NC & C there isn't necessary to do repacking,
            because these layouts has the same representation in NCHW & NHWC
         2) NHWC isn't necessary to do repacking obviously
         3) NDHWC isn't necessary to do repacking
         4) NCHW in general case it should be repacked, however if it is 11HW it isn't necessary
         5) CHW the same as for NCHW case, it isn't necessary to do repacking in 1HW case
         6) NCDHW in general case it should be repacked, however if it is 111HW it isn't necessary
     */
    const auto actualLayout = actualDesc.getLayout();
    const auto& actualDims = actualDesc.getDims();
    switch (actualLayout) {
    case ie::Layout::NDHWC:
    case ie::Layout::NHWC:
    case ie::Layout::NC:
    case ie::Layout::C:
        return false;
    case ie::Layout::NCDHW:
        return (actualDims[0] != 1) || (actualDims[1] != 1) || (actualDims[2] != 1);
    case ie::Layout::NCHW:
        return (actualDims[0] != 1) || (actualDims[1] != 1);
    case ie::Layout::CHW:
        return actualDims[0] != 1;
    default:
        IE_THROW() << "Unsupported layout for actual blob: " << actualLayout;
    }
}

ie::Blob::Ptr VpualCoreNNExecutor::prepareInputForInference(const ie::Blob::Ptr& actualInput,
                                                            const ie::TensorDesc& deviceDesc) {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "prepareInputForInference");

    const auto& actualDesc = actualInput->getTensorDesc();
    const auto& actualInputPrecision = actualDesc.getPrecision();

    const auto& devicePrecision = deviceDesc.getPrecision();
    const auto& deviceLayout = deviceDesc.getLayout();

    auto inputForInference = ie::as<ie::MemoryBlob>(actualInput);

    if (actualInputPrecision != devicePrecision) {
        _logger.warning("Input blob is inconsistent with network input. "
                        "Need to do convert precision from {0} to {1}.",
                        actualInputPrecision, devicePrecision);
        inputForInference = toPrecision(inputForInference, devicePrecision, vpux::None, _allocator);
    }

    // HACK: to overcome inability python API to pass a blob of NHWC layout
    if (_config.get<VPUAL_REPACK_INPUT_LAYOUT>()) {
        _logger.warning("VPUX_VPUAL_REPACK_INPUT_LAYOUT is enabled. Need to do re-layout.");
        return reallocateBlobToLayoutIgnoringOriginalLayout(inputForInference, ie::Layout::NCHW, ie::Layout::NHWC,
                                                            _allocator);
    }

    if (!isBlobAllocatedByAllocator(inputForInference, _allocator)) {
        _logger.warning("Input blob is located in non-shareable memory. Need to do re-allocation.");
        inputForInference = copyBlob(inputForInference, _allocator);
    }

    if (needRepackForNHWC(actualDesc) && deviceLayout == ie::Layout::NHWC) {
        _logger.warning("Input blob is inconsistent with network input. Need to do re-layout.");

        // NB: It's possible to make repack data only with the same number of dimensions
        // So just make a view without any copy
        const auto inMem = inputForInference->rmap();
        const auto actualView4D =
                makeBlob(vpu::getNCHW(inputForInference->getTensorDesc()), nullptr, inMem.as<void*>());
        inputForInference = toLayout(actualView4D, deviceLayout, _allocator);
    }

    return inputForInference;
}

void VpualCoreNNExecutor::push(const ie::BlobMap& inputs) {
#if defined(__arm__) || defined(__aarch64__)
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "push");
    _logger.info("::push started");

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

    NnExecWithProfilingMsg request;
    request.inferenceID = 1;
    for (const auto& input : updatedInputs) {
        auto blob = ie::as<ie::MemoryBlob>(input.second);
        auto memoryHolder = blob->rmap();
        auto inputBufferPhysAddr = _allocator->getPhysicalAddress(memoryHolder.as<uint8_t*>());
        request.inputTensors.push_back(inputBufferPhysAddr);
    }

    request.outputTensors = _outputPhysAddrs;
    request.profilingOutputTensors = _profilingOutputPhysAddrs;

    const auto status = _nnSync->RequestInference(request, _execInferId);
    if (X_LINK_SUCCESS != status) {
        _logger.error("push: RequestInference failed");
        IE_THROW() << "VpualCoreNNExecutor::push: RequestInference failed" << status;
    }

    _logger.info("::push finished");
#else
    VPUX_UNUSED(inputs);
#endif
}

void VpualCoreNNExecutor::push(const InferenceEngine::BlobMap&, const PreprocMap&) {
    IE_THROW() << "Not implemented";
}

uint32_t VpualCoreNNExecutor::extractPhysAddrForInference(const ie::BlobMap& inputs) {
    uint32_t physAddr = 0;
    if (inputs.size() == 1) {
        auto blob = ie::as<ie::MemoryBlob>(inputs.begin()->second);
        if (blob == nullptr) {
            IE_THROW() << "Input cannot be cast to memory blob";
        }
        auto memoryHolder = blob->rmap();
        physAddr = _allocator->getPhysicalAddress(memoryHolder.as<uint8_t*>());
        if (!physAddr) {
            IE_THROW() << "Memory of input is not valid";
        }
    } else {
        _logger.warning("There are multiple blobs. Need to combine them into single buffer.");
        std::size_t offset = 0;
        for (const auto& input : inputs) {
            auto name = input.first;
            auto blob = ie::as<ie::MemoryBlob>(input.second);

            if (!blob) {
                IE_THROW() << "Cannot cast to MemoryBlob";
            }
            auto memoryHolder = blob->rmap();

            ie_memcpy(_inputBuffer.get() + offset, blob->byteSize(), memoryHolder.as<uint8_t*>(), blob->byteSize());
            offset += blob->byteSize();
        }

        physAddr = _allocator->getPhysicalAddress(_inputBuffer.get());
        if (!physAddr) {
            IE_THROW() << "Memory of input is not valid";
        }
    }

    return physAddr;
}

void VpualCoreNNExecutor::pull(ie::BlobMap& outputs) {
#if defined(__arm__) || defined(__aarch64__)
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "pull");
    _logger.info("pull started");
    _wd->Start(this);
    const auto status = _nnSync->WaitForResponse(_execInferId);
    _wd->Pause(this);
    if (MVNCI_SUCCESS != status) {
        _logger.error("pull: for inference: {0}, received error response: {1}", _execInferId, status);
        IE_THROW() << "VpualCoreNNExecutor::pull: "
                   << ", for inference: " << _execInferId << " received error response: " << status;
    }
    if (_outputPhysAddrs.empty()) {
        _logger.error("_outputPhysAddrs.size() == 0");
        IE_THROW() << "Bad device output phys address";
    }
    ie::BlobMap deviceOutputs = extractOutputsFromPhysAddr(_outputPhysAddrs.at(0));
    repackDeviceOutputsToNetworkOutputs(deviceOutputs, outputs);

    handleProfiling();

    _logger.info("pull finished");
#else
    VPUX_UNUSED(outputs);
#endif
}

void VpualCoreNNExecutor::handleProfiling() {
    if (_profilingType == OutputType::NONE) {
        return;
    }

    if (_profilingOutputPhysAddrs.empty()) {
        _logger.error("Profiling printing is enabled but no profiling output detected.");
        return;
    }

    const auto& profilingOutputs = extractProfilingOutputsFromPhysAddr(_profilingOutputPhysAddrs.at(0));
    if (profilingOutputs.empty()) {
        IE_THROW() << "Can't extract profiling output.";
    }

    auto profilingOutputBlob = profilingOutputs.begin();
    const auto profilingMemoryBlob = ie::as<ie::MemoryBlob>(profilingOutputBlob->second);
    if (profilingMemoryBlob == nullptr) {
        IE_THROW() << "VPUX Plugin profiling blob is null: " << profilingOutputBlob->first;
    }
    const auto& blob = _networkDescription->getCompiledNetwork();
    const auto& profilingData =
            std::make_pair(profilingMemoryBlob->rmap().as<const void*>(), profilingMemoryBlob->byteSize());
    profiling::outputWriter(_profilingType, blob, profilingData, _profilingOutputFile);
}

ie::BlobMap VpualCoreNNExecutor::extractOutputsFromPhysAddr(uint32_t physAddr) {
    ie::BlobMap deviceOutputs;
    Byte offset(physAddr - _allocator->getPhysicalAddress(_outputBuffer.get()));
    for (auto&& out : _networkDescription->getDeviceOutputsInfo()) {
        const auto desc = out.second->getTensorDesc();
        auto blob = make_blob_with_precision(desc, _outputBuffer.get() + offset.count());
        deviceOutputs.insert({out.first, blob});
        offset += getMemorySize(desc);
    }

    return deviceOutputs;
}

ie::BlobMap VpualCoreNNExecutor::extractProfilingOutputsFromPhysAddr(uint32_t physAddr) {
    ie::BlobMap deviceProfilingOutputs;
    Byte offset(physAddr - _allocator->getPhysicalAddress(_profilingOutputBuffer.get()));
    for (auto&& out : _networkDescription->getDeviceProfilingOutputsInfo()) {
        const auto desc = out.second->getTensorDesc();
        auto blob = make_blob_with_precision(desc, _profilingOutputBuffer.get() + offset.count());
        deviceProfilingOutputs.insert({out.first, blob});
        offset += getMemorySize(desc);
    }

    return deviceProfilingOutputs;
}

void VpualCoreNNExecutor::repackDeviceOutputsToNetworkOutputs(const ie::BlobMap& deviceOutputs,
                                                              ie::BlobMap& networkOutputs) {
    for (const auto& item : deviceOutputs) {
        const auto& name = item.first;

        if (name == "profilingOutput")
            continue;

        const auto deviceBlob = ie::as<ie::MemoryBlob>(item.second);
        const auto& deviceDesc = deviceBlob->getTensorDesc();

        const auto& output = networkOutputs.find(name);
        if (networkOutputs.end() == output) {
            IE_THROW() << "VPUX Plugin cannot find output: " << name;
        }
        const auto outputBlob = ie::as<ie::MemoryBlob>(output->second);
        if (outputBlob == nullptr) {
            IE_THROW() << "VPUX Plugin output blob is null: " << name;
        }
        const auto& networkDesc = outputBlob->getTensorDesc();

        if (deviceDesc.getPrecision() != networkDesc.getPrecision()) {
            _logger.warning(
                    "Output blob is inconsistent with network output. Need to do convert precision from {0} to {1}.",
                    deviceDesc.getPrecision(), networkDesc.getPrecision());
        }

        const auto deviceBlobWithNetworkPrecision = toPrecision(deviceBlob, networkDesc.getPrecision());

        const auto outputMemory = outputBlob->wmap();
        if (needRepackForNHWC(networkDesc) && deviceDesc.getLayout() == ie::Layout::NHWC) {
            _logger.warning("Output blob is inconsistent with network output. Need to do re-layout from {0} to {1}.",
                            deviceDesc.getLayout(), networkDesc.getLayout());

            const auto actualView4D = makeBlob(vpu::getNCHW(networkDesc), nullptr, outputMemory.as<void*>());
            cvtBlobLayout(deviceBlobWithNetworkPrecision, actualView4D);
        } else {
            toLayout(deviceBlobWithNetworkPrecision, deviceDesc.getLayout(), nullptr, outputMemory.as<void*>());
        }
    }
}

void VpualCoreNNExecutor::setup(const ie::ParamMap&) {
    IE_THROW() << "Not implemented";
}

bool VpualCoreNNExecutor::isPreProcessingSupported(const PreprocMap&) const {
    return false;
}

std::map<std::string, ie::InferenceEngineProfileInfo> VpualCoreNNExecutor::getLayerStatistics() {
    const auto blob = _networkDescription->getCompiledNetwork();
    ie::BlobMap deviceOutputs;
    ie::BlobMap::iterator profilingOutputBlob;
    if (_profilingOutputPhysAddrs.size()) {
        deviceOutputs = extractProfilingOutputsFromPhysAddr(_profilingOutputPhysAddrs.at(0));
    }
    if (deviceOutputs.empty()) {
        deviceOutputs = extractOutputsFromPhysAddr(_outputPhysAddrs.at(0));
        profilingOutputBlob = deviceOutputs.find("profilingOutput");
        if (profilingOutputBlob == deviceOutputs.end()) {
            _logger.warning("No profiling output. Blob was compiled without profiling enabled or does not contain "
                            "profiling info.");
            return std::map<std::string, ie::InferenceEngineProfileInfo>();
        }
    } else {
        profilingOutputBlob = deviceOutputs.begin();
    }

    const auto profilingMemoryBlob = ie::as<ie::MemoryBlob>(profilingOutputBlob->second);
    if (profilingMemoryBlob == nullptr) {
        IE_THROW() << "VPUX Plugin profiling blob is null: " << profilingOutputBlob->first;
    }
    const auto& profilingOutput = profilingMemoryBlob->rmap().as<const void*>();

    std::vector<vpux::profiling::LayerInfo> layerProfiling;
    vpux::profiling::getLayerInfo(blob.data(), blob.size(), profilingOutput, profilingMemoryBlob->byteSize(),
                                  layerProfiling);

    return convertProfilingLayersToIEInfo(layerProfiling);
}

InferenceEngine::Parameter VpualCoreNNExecutor::getParameter(const std::string&) const {
    return InferenceEngine::Parameter();
}

Executor::Ptr VpualCoreNNExecutor::clone() const {
#if defined(__arm__) || defined(__aarch64__)
    return std::make_shared<VpualCoreNNExecutor>(_networkDescription, _allocator, _nnXlinkPlg, _nnCorePlg, _nnSync,
                                                 _pipe, _wd, _config);
#else
    IE_THROW() << "VpualCoreNNExecutor::clone not implemented for x86_64";
#endif
}

}  // namespace vpux
