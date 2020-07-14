// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>

#include <ie_preprocess.hpp>
#include <memory>

#include "kmb_preproc.hpp"  // KmbPreproc::Path

// clang-format off
namespace InferenceEngine {
namespace KmbPreproc {

class PreprocEngine {
public:
    PreprocEngine(unsigned int shaveFirst, unsigned int shaveLast, unsigned int lpi, Path ppPath);
    ~PreprocEngine();

    void preproc(const Blob::Ptr &inBlob, Blob::Ptr &outBlob,
                 const ResizeAlgorithm& algorithm,
                 ColorFormat in_fmt, ColorFormat out_fmt,
                 const int& deviceId);
    class Priv;

private:
    std::unique_ptr<Priv> _priv;
};

}  // namespace KmbPreproc
}  // namespace InferenceEngine
// clang-format on
