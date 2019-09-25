// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>
#include <ie_preprocess.hpp>

#include <opencv2/gapi.hpp>

namespace InferenceEngine {

class SIPPPreprocEngine {
    cv::GCompiled _lastCompiled;
    SizeVector    _lastInYDims;
    unsigned int  _shaveFirst;
    unsigned int  _shaveLast;

public:
    SIPPPreprocEngine(unsigned int shaveFirst, unsigned int shaveLast)
        : _shaveFirst(shaveFirst), _shaveLast(shaveLast) {}

    void preprocWithSIPP(const Blob::Ptr &inBlob, Blob::Ptr &outBlob,
                         const ResizeAlgorithm& algorithm, ColorFormat in_fmt,
                         bool omp_serial, int batch_size);
};

}  // namespace InferenceEngine
