//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux_driver_compiler.h"

#include <cpp/ie_cnn_network.h>
#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

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

namespace VCLTestsUtils {

using Json = nlohmann::ordered_json;
using IRInfoTestType = std::vector<std::unordered_map<std::string, std::string>>;
using VCLTestsParams = std::tuple<std::unordered_map<std::string, std::string>>;

/// The file contains the models and its configuration for somke test
const std::string SMOKE_TEST_CONFIG = "/test_smoke.json";
/// The file contains the modesl and its configuration for noraml test
const std::string TEST_CONFIG = "/test.json";

/**
 * @brief Base class to parse config file to get test cases and provide helper functions
 */
class VCLTestsCommon : public testing::WithParamInterface<VCLTestsParams>, public testing::Test {
public:
    VCLTestsCommon(): modelIR(), modelIRSize(0) {
    }
    virtual ~VCLTestsCommon() = default;

    /**
     * @brief Create the test name suffixes with content of test params
     */
    static std::string getTestCaseName(const testing::TestParamInfo<VCLTestsParams>& obj) {
        auto param = obj.param;
        auto netInfo = std::get<0>(param);
        const std::string netName = netInfo.at("network");
        const std::string deviceID = netInfo.at("device");
        const std::string testCaseName = netName + std::string("_VPUX.") + deviceID;
        return testCaseName;
    }

    /**
     * @brief Prepare modelIRData of one test case for compiler
     */
    vcl_result_t initModelData(const char* netName, const char* weightName);

    /**
     * @brief Create a simple model to test compiler
     */
    std::shared_ptr<ngraph::Function> createSimpleFunction();

    /**
     * @brief Get the location of model package defined by POR_PATH
     */
    std::string getTestModelsBasePath();

    /**
     * @brief Get the location of config files for tests
     */
    static std::string getCidToolPath();

    /**
     * @brief Parse config file and detect test cases
     */
    static IRInfoTestType readJson2Vec(std::string fileName);

    /**
     * @brief Add platform info to model build options
     */
    void postProcessNetOptions(const std::string& device);

    /**
     * @brief Return the build options of model
     */
    std::string getNetOptions() {
        return netOptions;
    };

    /**
     * @brief Prepare modelIRData of all tests for compiler
     */
    void SetUp() override;

    std::vector<uint8_t>& getModelIR() {
        return modelIR;
    }

    size_t getModelIRSize() {
        return modelIRSize;
    }

private:
    /// Model build flags
    std::string netOptions;
    /// The data of modelIR to create executable
    std::vector<uint8_t> modelIR;
    size_t modelIRSize;
};
}  // namespace VCLTestsUtils
