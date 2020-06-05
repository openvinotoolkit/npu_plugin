// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>

#include <ie_preprocess.hpp>
#include <memory>

#include "kmb_preproc.hpp"  // SippPreproc::Path

// clang-format off
namespace InferenceEngine {

class SIPPPreprocEngine {
public:
    SIPPPreprocEngine(unsigned int shaveFirst, unsigned int shaveLast, unsigned int lpi, SippPreproc::Path ppPath);
    ~SIPPPreprocEngine();

    // TODO: Drop SIPP from the name
    void preprocWithSIPP(const Blob::Ptr &inBlob, Blob::Ptr &outBlob,
                         const ResizeAlgorithm& algorithm,
                         ColorFormat in_fmt, ColorFormat out_fmt);

    class Priv;

private:
    std::unique_ptr<Priv> _priv;
};

}  // namespace InferenceEngine
// clang-format on
