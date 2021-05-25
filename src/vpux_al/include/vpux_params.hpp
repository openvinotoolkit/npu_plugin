//
// Copyright 2020 Intel Corporation.
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
#pragma once

// IE
#include "ie_remote_context.hpp"
// Plugin
#include "vpu/utils/logger.hpp"
#include "vpux/vpux_plugin_params.hpp"
#include "vpux_params_private_options.hpp"

namespace vpux {

//------------------------------------------------------------------------------
/** @brief Represent container for all RemoteBlob related objects.
 * @details Backends may inherit this class, if some specific parameters parsing required in addition.
 */
class ParsedRemoteBlobParams {
public:
    using Ptr = std::shared_ptr<ParsedRemoteBlobParams>;
    using CPtr = std::shared_ptr<const ParsedRemoteBlobParams>;

    InferenceEngine::ParamMap getParamMap() const {
        return _paramMap;
    }
    std::shared_ptr<const InferenceEngine::ROI> getROIPtr() const {
        return _roiPtr;
    }
    std::shared_ptr<const InferenceEngine::TensorDesc> getOriginalTensorDesc() const {
        return _originalTensorDesc;
    }
    size_t getMemoryOffset() const {
        return _memoryOffset;
    }
    void* getMemoryHandle() const {
        return _memoryHandle;
    }
    VpuxRemoteMemoryFD getRemoteMemoryFD() const {
        return _remoteMemoryFD;
    }
    InferenceEngine::ColorFormat getBlobColorFormat() const {
        return _blobColorFormat;
    }

public:
    /** @brief Override current parameters with new options, not specified keep the same */
    virtual void update(const InferenceEngine::ParamMap& updateParams);

    /** @brief Override current parameters with new or existing options, not specified keep the same */
    virtual void updateFull(const InferenceEngine::ParamMap& updateParams);

protected:
    virtual void parse();
    // TODO On default, paramMap should keep ROI=nullptr?
    InferenceEngine::ParamMap _paramMap = {};

private:
    std::shared_ptr<const InferenceEngine::ROI> _roiPtr = nullptr;
    std::shared_ptr<const InferenceEngine::TensorDesc> _originalTensorDesc = nullptr;
    size_t _memoryOffset = 0;
    void* _memoryHandle = nullptr;
    VpuxRemoteMemoryFD _remoteMemoryFD = -1;
    InferenceEngine::ColorFormat _blobColorFormat = InferenceEngine::ColorFormat::BGR;
};

}  // namespace vpux
