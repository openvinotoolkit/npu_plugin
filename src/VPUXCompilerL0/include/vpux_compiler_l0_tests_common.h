//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <cpp/ie_cnn_network.h>
#include <gtest/gtest.h>
#include <fstream>

#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/serialize.hpp>
#include <nlohmann/json.hpp>

#if defined(_WIN32)
#include "Shlwapi.h"
// These two undefs are to avoid min/max macro interfering introduced by Shlwapi.h.
#undef min
#undef max
#else
#include <sys/stat.h>
#endif

namespace VpuxCompilerL0TestsUtils {

using Json = nlohmann::ordered_json;
using IRInfoTestType = std::vector<std::unordered_map<std::string, std::string>>;

const std::string SMOKE_TEST_CONFIG = "/test_smoke.json";
const std::string TEST_CONFIG = "/test.json";

class VpuxCompilerL0TestsCommon {
public:
    VpuxCompilerL0TestsCommon() = default;
    virtual ~VpuxCompilerL0TestsCommon() = default;
    std::shared_ptr<ngraph::Function> create_simple_function();
    std::string getTestModelsBasePath();
    static std::string getCidToolPath();
    static IRInfoTestType readJson2Vec(std::string fileName);
    void postProcessNetOptions(const std::string& device);
    std::string getNetOptions();

    std::string netOptions;
};
}  // namespace VpuxCompilerL0TestsUtils
