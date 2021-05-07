// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <ie_iextension.h>
#include <ie_api.h>
#include <ngraph/ngraph.hpp>
#include "add_with_offset_op.hpp"

namespace SampleExtension {

class INFERENCE_EXTENSION_API_CLASS(SampleExt) : public InferenceEngine::IExtension {
public:
    SampleExt() = default;
    void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept override;
    void Unload() noexcept override {};
    void Release() noexcept override {delete this;};
    std::map<std::string, ngraph::OpSet> getOpSets() override;
};
}

INFERENCE_EXTENSION_API(InferenceEngine::StatusCode) InferenceEngine::CreateExtension(InferenceEngine::IExtension *&ext,
                                                             InferenceEngine::ResponseDesc *resp) noexcept;
