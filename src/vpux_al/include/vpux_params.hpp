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
#pragma once

// IE
#include "ie_remote_context.hpp"
// Plugin
#include "vpu/utils/logger.hpp"

namespace vpux {

//------------------------------------------------------------------------------
/** @brief Represent container for all RemoteBlob related objects.
 * @details Backends may inherit this class, if some specific parameters parsing required in addition.
 */
class ParsedRemoteBlobParams {
public:
    InferenceEngine::ParamMap getParamMap() const { return _paramMap; }
    InferenceEngine::ColorFormat getColorFormat() const { return _colorFormat; }
    std::shared_ptr<const InferenceEngine::ROI> getROIPtr() const { return _roiPtr; }
    std::shared_ptr<const InferenceEngine::TensorDesc> getOriginalTensorDesc() const { return _originalTensorDesc; }

public:
    /** @brief Override current parameters with new options, not specified keep the same */
    virtual void update(const InferenceEngine::ParamMap& updateParams);

protected:
    virtual void parse();
    // TODO On default, paramMap should keep colorFormat=BGR and ROI=nullptr?
    InferenceEngine::ParamMap _paramMap = {};

private:
    /** @brief Since RemoteMemory represent black box, we need some way to understand that it's NV12 blob*/
    InferenceEngine::ColorFormat _colorFormat = InferenceEngine::ColorFormat::BGR;
    std::shared_ptr<const InferenceEngine::ROI> _roiPtr = nullptr;
    std::shared_ptr<const InferenceEngine::TensorDesc> _originalTensorDesc = nullptr;
};

}  // namespace vpux
