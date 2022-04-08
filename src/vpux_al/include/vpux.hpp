//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include <ie_blob.h>
#include <ie_common.h>
#include <cpp_interfaces/interface/ie_iinfer_request_internal.hpp>
#include <ie_icnn_network.hpp>
#include <ie_remote_context.hpp>

#include "vpux/utils/IE/config.hpp"

#include "vpux_compiler.hpp"

namespace vpux {

bool isBlobAllocatedByAllocator(const InferenceEngine::Blob::Ptr& blob,
                                const std::shared_ptr<InferenceEngine::IAllocator>& allocator);

std::string getLibFilePath(const std::string& baseName);

//------------------------------------------------------------------------------
class IDevice;
class Device;

class IEngineBackend : public std::enable_shared_from_this<IEngineBackend> {
public:
    /** @brief Get device, which can be used for inference. Backend responsible for selection. */
    virtual const std::shared_ptr<IDevice> getDevice() const;
    /** @brief Search for a specific device by name */
    virtual const std::shared_ptr<IDevice> getDevice(const std::string& specificDeviceName) const;
    /** @brief Get device, which is configured/suitable for provided params */
    virtual const std::shared_ptr<IDevice> getDevice(const InferenceEngine::ParamMap& paramMap) const;
    /** @brief Provide a list of names of all devices, with which user can work directly */
    virtual const std::vector<std::string> getDeviceNames() const;
    /** @brief Get name of backend */
    virtual const std::string getName() const = 0;
    /** @brief Register backend-specific options */
    virtual void registerOptions(OptionsDesc& options) const;

#ifndef OPENVINO_STATIC_LIBRARY
protected:
#endif
    ~IEngineBackend() = default;
};

class EngineBackend final {
public:
    EngineBackend() = default;

#ifdef OPENVINO_STATIC_LIBRARY
    EngineBackend(std::shared_ptr<IEngineBackend> impl);
#endif

#ifndef OPENVINO_STATIC_LIBRARY
    EngineBackend(const std::string& pathToLib);
#endif

    // Destructor preserves unload order of implementation object and reference to library.
    // To preserve destruction order inside default generated assignment operator we store `_impl` before `_so`.
    // And use destructor to remove implementation object before reference to library explicitly.
    ~EngineBackend() {
        _impl = {};
    }

    const std::shared_ptr<Device> getDevice() const;
    const std::shared_ptr<Device> getDevice(const std::string& specificDeviceName) const;
    const std::shared_ptr<Device> getDevice(const InferenceEngine::ParamMap& paramMap) const;
    const std::vector<std::string> getDeviceNames() const {
        return _impl->getDeviceNames();
    }
    const std::string getName() const {
        return _impl->getName();
    }
    void registerOptions(OptionsDesc& options) const {
        _impl->registerOptions(options);
        options.addSharedObject(_so);
    }

private:
    std::shared_ptr<IEngineBackend> _impl;

    // Keep pointer to `_so` to avoid shared library unloading prior destruction of the `_impl` object.
    std::shared_ptr<void> _so;
};

//------------------------------------------------------------------------------
class Allocator : public InferenceEngine::IAllocator {
public:
    using Ptr = std::shared_ptr<Allocator>;
    using CPtr = std::shared_ptr<const Allocator>;

    /** @brief Wrap remote memory. Backend should get all required data from paramMap */
    virtual void* wrapRemoteMemory(const InferenceEngine::ParamMap& paramMap) noexcept;
    // TODO remove these methods
    // [Track number: E#23679]
    // TODO: need update methods to remove Kmb from parameters
    /** @deprecated These functions below should not be used */
    virtual void* wrapRemoteMemoryHandle(const int& remoteMemoryFd, const size_t size, void* memHandle) noexcept = 0;
    virtual void* wrapRemoteMemoryOffset(const int& remoteMemoryFd, const size_t size,
                                         const size_t& memOffset) noexcept = 0;

    // FIXME: temporary exposed to allow executor to use vpux::Allocator
    virtual unsigned long getPhysicalAddress(void* handle) noexcept = 0;
};

//------------------------------------------------------------------------------

class AllocatorWrapper : public Allocator {
public:
    AllocatorWrapper(const std::shared_ptr<Allocator>& impl, const std::shared_ptr<void>& so): _impl(impl), _so(so) {
    }

    // Destructor preserves unload order of implementation object and reference to library.
    // To preserve destruction order inside default generated assignment operator we store `_impl` before `_so`.
    // And use destructor to remove implementation object before reference to library explicitly.
    ~AllocatorWrapper() {
        _impl = {};
    }

    virtual void* lock(void* handle, InferenceEngine::LockOp op = InferenceEngine::LOCK_FOR_WRITE) noexcept override {
        return _impl->lock(handle, op);
    }
    virtual void unlock(void* handle) noexcept override {
        return _impl->unlock(handle);
    }
    virtual void* alloc(size_t size) noexcept override {
        return _impl->alloc(size);
    }
    virtual bool free(void* handle) noexcept override {
        return _impl->free(handle);
    }

    virtual void* wrapRemoteMemory(const InferenceEngine::ParamMap& paramMap) noexcept override {
        return _impl->wrapRemoteMemory(paramMap);
    }
    virtual void* wrapRemoteMemoryHandle(const int& remoteMemoryFd, const size_t size,
                                         void* memHandle) noexcept override {
        return _impl->wrapRemoteMemoryHandle(remoteMemoryFd, size, memHandle);
    }
    virtual void* wrapRemoteMemoryOffset(const int& remoteMemoryFd, const size_t size,
                                         const size_t& memOffset) noexcept override {
        return _impl->wrapRemoteMemoryOffset(remoteMemoryFd, size, memOffset);
    }
    virtual unsigned long getPhysicalAddress(void* handle) noexcept override {
        return _impl->getPhysicalAddress(handle);
    }

private:
    std::shared_ptr<Allocator> _impl;

    // Keep pointer to `_so` to avoid shared library unloading prior destruction of the `_impl` object.
    std::shared_ptr<void> _so;
};

//------------------------------------------------------------------------------

using PreprocMap = std::map<std::string, const InferenceEngine::PreProcessInfo>;

class Executor {
public:
    using Ptr = std::shared_ptr<Executor>;
    using CPtr = std::shared_ptr<const Executor>;

    virtual void setup(const InferenceEngine::ParamMap& params) = 0;
    virtual Executor::Ptr clone() const {
        IE_THROW() << "Not implemented";
    }

    virtual void push(const InferenceEngine::BlobMap& inputs) = 0;
    virtual void push(const InferenceEngine::BlobMap& inputs, const PreprocMap& preProcMap) = 0;

    virtual void pull(InferenceEngine::BlobMap& outputs) = 0;

    virtual bool isPreProcessingSupported(const PreprocMap& preProcMap) const = 0;
    virtual std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> getLayerStatistics() = 0;
    virtual InferenceEngine::Parameter getParameter(const std::string& paramName) const = 0;

    virtual ~Executor() = default;
};

//------------------------------------------------------------------------------
class IInferRequest : public InferenceEngine::IInferRequestInternal {
public:
    using Ptr = std::shared_ptr<IInferRequest>;
    explicit IInferRequest(const InferenceEngine::InputsDataMap& networkInputs,
                           const InferenceEngine::OutputsDataMap& networkOutputs)
            : IInferRequestInternal(networkInputs, networkOutputs) {
    }
    virtual void InferAsync() = 0;
    virtual void GetResult() = 0;
};

// TODO: extract to a separate header
// E#-34780
class InferRequest : public IInferRequest {
public:
    explicit InferRequest(const InferenceEngine::InputsDataMap& networkInputs,
                          const InferenceEngine::OutputsDataMap& networkOutputs, const Executor::Ptr& executor,
                          const Config& config, const std::string& netName,
                          const std::vector<std::shared_ptr<const ov::Node>>& parameters,
                          const std::vector<std::shared_ptr<const ov::Node>>& results,
                          const std::shared_ptr<InferenceEngine::IAllocator>& allocator = nullptr);

    void Infer() override;
    void InferImpl() override;
    void InferAsync() override;
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const override;

    void GetResult() override;

    using InferenceEngine::IInferRequestInternal::SetBlob;
    void SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr& data) override;

protected:
    void checkBlobs() override;

    /**
     * @brief Create map with preProcessing info
     * @param[in] networkInputs Contains information of pre-processing, which should be done
     * @param[in] preProcData Container with blobs, which should be preprocessed
     * @return Map with preprocess information
     */
    PreprocMap preparePreProcessing(const InferenceEngine::InputsDataMap& networkInputs,
                                    const std::map<std::string, InferenceEngine::PreProcessDataPtr>& preProcData);

    /**
     * @brief Move all preProcessing blobs to inputs BlobMap
     * @param[in/out] inputs Map with NN blobs. PP blobs should be placed instead for some inputs.
     * @details This should be done as separate step, if device cannot handle such preprocessing, input should not be
     * replaced
     */
    void moveBlobsForPreprocessingToInputs(
            InferenceEngine::BlobMap& inputs, const InferenceEngine::InputsDataMap& networkInputs,
            const std::map<std::string, InferenceEngine::PreProcessDataPtr>& preProcData);

    void updateRemoteBlobs(InferenceEngine::BlobMap& inputs, const PreprocMap& preProcMap);
    void updateRemoteBlobColorFormat(InferenceEngine::Blob::Ptr& blob, const InferenceEngine::ColorFormat colorFormat);

    // TODO Preprocessing should be moved into backend [Track number: S#43193]
#ifdef __aarch64__
    void execPreprocessing(InferenceEngine::BlobMap& inputs);
    void relocationAndExecKmbDataPreprocessing(InferenceEngine::BlobMap& inputs,
                                               InferenceEngine::InputsDataMap& networkInputs,
                                               InferenceEngine::ColorFormat out_format, unsigned int numShaves,
                                               unsigned int lpi, unsigned int numPipes);
    virtual void execKmbDataPreprocessing(InferenceEngine::BlobMap& inputs,
                                          std::map<std::string, InferenceEngine::PreProcessDataPtr>& preprocData,
                                          InferenceEngine::InputsDataMap& networkInputs,
                                          InferenceEngine::ColorFormat out_format, unsigned int numShaves,
                                          unsigned int lpi, unsigned int numPipes);
#endif

protected:
    const Executor::Ptr _executorPtr;
    const Config _config;
    Logger _logger;
    std::shared_ptr<InferenceEngine::IAllocator> _allocator;
    const int _deviceId;
    const std::string _netUniqueId;

    // TODO Specific details for KMB-standalone preprocessing [Track number: S#43193]
    // the buffer is used when non-shareable memory passed for preprocessing
    std::unique_ptr<uint8_t, std::function<void(uint8_t*)>> _preprocBuffer;
};

//------------------------------------------------------------------------------

class IDevice : public std::enable_shared_from_this<IDevice> {
public:
    virtual std::shared_ptr<Allocator> getAllocator() const = 0;
    /** @brief Get allocator, which is configured/suitable for provided params
     * @example Each backend may have many allocators, each of which suitable for different RemoteMemory param */
    virtual std::shared_ptr<Allocator> getAllocator(const InferenceEngine::ParamMap& paramMap) const;

    virtual std::shared_ptr<Executor> createExecutor(const NetworkDescription::Ptr& networkDescription,
                                                     const Config& config) = 0;

    virtual std::string getName() const = 0;

    // TODO: options:
    // * common implementation of infer request
    // * force each to implement its own
    virtual InferRequest::Ptr createInferRequest(const InferenceEngine::InputsDataMap& networkInputs,
                                                 const InferenceEngine::OutputsDataMap& networkOutputs,
                                                 const Executor::Ptr& executor, const Config& config,
                                                 const std::string& networkName,
                                                 const std::vector<std::shared_ptr<const ov::Node>>& parameters,
                                                 const std::vector<std::shared_ptr<const ov::Node>>& results,
                                                 const std::shared_ptr<InferenceEngine::IAllocator>& allocator) = 0;

protected:
    virtual ~IDevice() = default;
};

class Device final {
public:
    using Ptr = std::shared_ptr<Device>;
    using CPtr = std::shared_ptr<const Device>;

    Device(const std::shared_ptr<IDevice>& device, const std::shared_ptr<void>& so): _impl(device), _so(so) {
        if (_impl->getAllocator()) {
            _allocatorWrapper = std::make_shared<AllocatorWrapper>(_impl->getAllocator(), _so);
        }
    }

    // Destructor preserves unload order of implementation object and reference to library.
    // To preserve destruction order inside default generated assignment operator we store `_impl` before `_so`.
    // And use destructor to remove implementation object before reference to library explicitly.
    ~Device() {
        _impl = {};
        _allocatorWrapper = {};
    }

    std::shared_ptr<Allocator> getAllocator() const {
        return _allocatorWrapper;
    }
    std::shared_ptr<Allocator> getAllocator(const InferenceEngine::ParamMap& paramMap) {
        return std::make_shared<AllocatorWrapper>(_impl->getAllocator(paramMap), _so);
    }

    std::shared_ptr<Executor> createExecutor(const NetworkDescription::Ptr& networkDescription, const Config& config) {
        return _impl->createExecutor(networkDescription, config);
    }

    std::string getName() const {
        return _impl->getName();
    }

    // TODO: is it the correct place for the method?
    // probably, we need to wrap infer request to store pointer to so (need to check)
    // can we provide default implementation for infer requests?
    InferRequest::Ptr createInferRequest(const InferenceEngine::InputsDataMap& networkInputs,
                                         const InferenceEngine::OutputsDataMap& networkOutputs,
                                         const Executor::Ptr& executor, const Config& config,
                                         const std::string& netName,
                                         const std::vector<std::shared_ptr<const ov::Node>>& parameters,
                                         const std::vector<std::shared_ptr<const ov::Node>>& results,
                                         const std::shared_ptr<InferenceEngine::IAllocator>& allocator) {
        InferRequest::Ptr request = _impl->createInferRequest(networkInputs, networkOutputs, executor, config, netName,
                                                              parameters, results, allocator);
        if (!request) {
            request = std::make_shared<InferRequest>(networkInputs, networkOutputs, executor, config, netName,
                                                     parameters, results, allocator);
        }
        return request;
    }

private:
    std::shared_ptr<IDevice> _impl;
    std::shared_ptr<AllocatorWrapper> _allocatorWrapper;

    // Keep pointer to `_so` to avoid shared library unloading prior destruction of the `_impl` object.
    std::shared_ptr<void> _so;
};

}  // namespace vpux
