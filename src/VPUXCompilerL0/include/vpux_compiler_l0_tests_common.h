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


#include <gtest/gtest.h>
#include <cpp/ie_cnn_network.h>

#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/serialize.hpp>

#if defined(_WIN32)
#include "Shlwapi.h"
// These two undefs are to avoid min/max macro interfering introduced by Shlwapi.h.
#undef min
#undef max
#else
#include <sys/stat.h>
#endif

namespace VpuxCompilerL0TestsUtils {

class VpuxCompilerL0TestsCommon {
public:
    VpuxCompilerL0TestsCommon() = default;
    virtual ~VpuxCompilerL0TestsCommon() = default;
    std::shared_ptr<ngraph::Function> create_simple_function();
    std::string getTestModelsBasePath();
};
}  // namespace VpuxCompilerL0TestsUtils
