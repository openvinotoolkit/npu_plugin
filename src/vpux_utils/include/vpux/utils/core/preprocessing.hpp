//
// Copyright 2022 Intel Corporation.
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

#include <ie_preprocess.hpp>
#include <utility>

namespace vpux {
/**
 * @brief A helper class describing preprocess info for inputs
 */

class VPUXPreProcessInfo final : public InferenceEngine::PreProcessInfo {
public:
    using Ptr = std::shared_ptr<VPUXPreProcessInfo>;
    using CPtr = std::shared_ptr<const VPUXPreProcessInfo>;

    VPUXPreProcessInfo() = default;
    explicit VPUXPreProcessInfo(std::string inputName, InferenceEngine::ColorFormat inputFormat,
                                InferenceEngine::ColorFormat outputFormat, InferenceEngine::ResizeAlgorithm algorithm,
                                InferenceEngine::TensorDesc tensorDesc = {})
            : _inputName(std::move(inputName)), _outputFormat(outputFormat), _originTensorDesc(std::move(tensorDesc)) {
        setColorFormat(inputFormat);
        setResizeAlgorithm(algorithm);
    };

    auto getInputName() const {
        return _inputName;
    }

    auto getInputColorFormat() const {
        return getColorFormat();
    }

    auto getOutputColorFormat() const {
        return _outputFormat;
    }

    void setInputColorFormat(const InferenceEngine::ColorFormat inputFormat) {
        setColorFormat(inputFormat);
    }

    void setOutputColorFormat(const InferenceEngine::ColorFormat outputFormat) {
        _outputFormat = outputFormat;
    }

    InferenceEngine::TensorDesc getOriginTensorDesc() const {
        return _originTensorDesc;
    }

    void setOriginTensorDesc(InferenceEngine::TensorDesc originTensorDesc) {
        _originTensorDesc = originTensorDesc;
    }

    VPUXPreProcessInfo(const VPUXPreProcessInfo& p)
            : _inputName(p.getInputName()),
              _outputFormat(p.getOutputColorFormat()),
              _originTensorDesc(p.getOriginTensorDesc()) {
        setColorFormat(p.getColorFormat());
        setResizeAlgorithm(p.getResizeAlgorithm());
    };

    VPUXPreProcessInfo& operator=(const VPUXPreProcessInfo& p) {
        _inputName = p.getInputName();
        setColorFormat(p.getColorFormat());
        setOutputColorFormat(p.getOutputColorFormat());
        setResizeAlgorithm(p.getResizeAlgorithm());
        setOriginTensorDesc(p.getOriginTensorDesc());
        return *this;
    }

protected:
    std::string _inputName;
    InferenceEngine::ColorFormat _outputFormat = InferenceEngine::ColorFormat::RAW;
    InferenceEngine::TensorDesc _originTensorDesc;
};

}  // namespace vpux
