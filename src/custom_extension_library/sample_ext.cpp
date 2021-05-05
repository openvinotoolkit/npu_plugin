// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "sample_ext.hpp"

using namespace SampleExtension;

InferenceEngine::StatusCode InferenceEngine::CreateExtension(InferenceEngine::IExtension *&ext,
                                                             InferenceEngine::ResponseDesc *resp) noexcept {
    try {
        ext = new SampleExt();
        return OK;
    } catch (std::exception &ex) {
        if (resp) {
            std::string err = ((std::string) "Couldn't create extension: ") + ex.what();
            err.copy(resp->msg, 255);
        }
        return InferenceEngine::GENERAL_ERROR;
    }
}

void SampleExt::GetVersion(const InferenceEngine::Version *&versionInfo) const noexcept {
    static InferenceEngine::Version ExtensionDescription = {
        {1, 0},           // extension API version
        "1.0",
        "Sample extension library"    // extension description message
    };

    versionInfo = &ExtensionDescription;
}

std::map<std::string, ngraph::OpSet> SampleExt::getOpSets() {
    ngraph::OpSet opset;
    opset.insert<AddWOffsetOp>();
    std::map<std::string, ngraph::OpSet> opsets{{"SampleExtension-Opset1", opset}};
    return opsets;
}

