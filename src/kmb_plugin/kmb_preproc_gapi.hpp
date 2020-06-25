// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>

#include <ie_preprocess.hpp>
#include <memory>

// clang-format off
namespace InferenceEngine {

class SIPPPreprocEngine {
    class Priv;
    std::unique_ptr<Priv> _priv;

public:
    SIPPPreprocEngine(unsigned int shaveFirst, unsigned int shaveLast, unsigned int lpi);
    ~SIPPPreprocEngine();

    void preprocWithSIPP(const Blob::Ptr &inBlob, Blob::Ptr &outBlob,
                         const ResizeAlgorithm& algorithm,
                         ColorFormat in_fmt, ColorFormat out_fmt);
};

}  // namespace InferenceEngine
// clang-format on
