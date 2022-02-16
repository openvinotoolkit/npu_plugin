// Copyright (C) 2019 Intel Corporation
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
                 const int deviceId);
    class Priv;

private:
    std::unique_ptr<Priv> _priv;
};

}  // namespace KmbPreproc
}  // namespace InferenceEngine
// clang-format on
