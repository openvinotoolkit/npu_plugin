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

#include "ie_core.hpp"

namespace IE = InferenceEngine;

namespace ReferenceHelper {
    IE::Blob::Ptr CalcCpuReferenceSingleOutput(const std::string &model_path, const IE::Blob::Ptr& input_blob,
        const IE::PreProcessInfo* preproc_info = nullptr);
    IE::BlobMap CalcCpuReferenceMultipleOutput(const std::string& model_path, const IE::Blob::Ptr& input_blob,
        const IE::PreProcessInfo* preproc_info = nullptr);
}
