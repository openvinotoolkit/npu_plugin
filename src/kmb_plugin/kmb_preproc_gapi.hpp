// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>
#include <ie_preprocess.hpp>

#include <memory>

namespace InferenceEngine {

class SIPPPreprocEngine {
    class Priv;
    std::unique_ptr<Priv> _priv;
public:
    SIPPPreprocEngine(unsigned int shaveFirst, unsigned int shaveLast, unsigned int lpi);
    ~SIPPPreprocEngine();

    void preprocWithSIPP(const Blob::Ptr &inBlob, Blob::Ptr &outBlob,
                         const ResizeAlgorithm& algorithm, ColorFormat in_fmt,
                         bool omp_serial, int batch_size);
};

}  // namespace InferenceEngine
